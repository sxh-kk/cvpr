"""Train-only linear readouts of frozen priors; never updates MotionExpert.

The original validation split is used for reporting only. Ridge strengths are
selected on an original-train internal holdout of connected split groups.
DINO PCA sees original train features only, never validation or labels.
"""
from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch

from .prior_probe_math import probe_metric_rows, reduce_probe_metric_rows

ALPHAS = (.01, .1, 1., 10., 100.)
ERROR_METRICS = ("wrist_displacement_cm", "reference_displacement_xy_cm", "reference_yaw_deg")
BINARY_METRICS = ("contact_accuracy", "contact_f1", "contact_balanced_accuracy",
                  "transition_accuracy", "transition_f1", "transition_balanced_accuracy")
METRICS = ERROR_METRICS + BINARY_METRICS


def connected_group_ids(entries):
    """Join episodes sharing *any* original participant/capture split group."""
    parent = list(range(len(entries)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owners = {}
    for index, entry in enumerate(entries):
        groups = entry.get("split_group_ids")
        if not groups:
            raise ValueError("Probe entries require original split_group_ids")
        for group in groups:
            if group in owners:
                parent[find(index)] = find(owners[group])
            else:
                owners[group] = index
    roots = [find(index) for index in range(len(entries))]
    unique = {root: index for index, root in enumerate(sorted(set(roots)))}
    labels = np.asarray([unique[root] for root in roots], dtype=np.int64)
    for group in np.unique(labels):
        splits = {entries[index]["split"] for index in np.flatnonzero(labels == group)}
        if len(splits) != 1:
            raise ValueError("Original split groups leak between train and validation")
    return labels


def internal_group_split(entries, groups, seed=314159):
    """Deterministic source-stratified original-train group holdout."""
    split = np.asarray([entry["split"] for entry in entries])
    source = np.asarray([entry["source"] for entry in entries])
    train = split == "train"
    holdout_groups = set()
    rng = np.random.default_rng(seed)
    for name in sorted(set(source[train])):
        available = np.unique(groups[train & (source == name)])
        if len(available) < 2:
            raise ValueError(f"{name} needs at least two independent train groups")
        count = min(len(available) - 1, max(1, round(.2 * len(available))))
        holdout_groups.update(rng.permutation(available)[:count].tolist())
    holdout = train & np.isin(groups, sorted(holdout_groups))
    fitting = train & ~holdout
    for name in set(source[train]):
        if not (fitting & (source == name)).any() or not (holdout & (source == name)).any():
            raise ValueError(f"Connected groups cannot produce a source-balanced holdout: {name}")
    return fitting, holdout


def encode_targets(target):
    """Five horizons of XYZ/XY/sin(yaw)/cos(yaw), then contact/change."""
    continuous = np.asarray(target["continuous"]).reshape(-1, 5, 9)
    valid = np.asarray(target["continuous_valid"], dtype=bool).reshape(-1, 5, 9)
    encoded = np.concatenate((continuous[..., :8], np.sin(continuous[..., 8:9]),
                              np.cos(continuous[..., 8:9])), axis=-1).reshape(-1, 50)
    encoded_valid = np.concatenate((valid[..., :8], valid[..., 8:9], valid[..., 8:9]), axis=-1).reshape(-1, 50)
    values = np.concatenate((encoded, target["contact"], target["transition"]), axis=1).astype(np.float32)
    masks = np.concatenate((encoded_valid, target["contact_valid"], target["transition_valid"]), axis=1).astype(bool)
    if not np.isfinite(values[masks]).all():
        raise ValueError("Probe labels contain non-finite valid targets")
    return np.where(masks, values, 0), masks


def decode_targets(prediction):
    prediction = np.asarray(prediction)
    encoded = prediction[:, :50].reshape(-1, 5, 10)
    continuous = np.concatenate((encoded[..., :8], np.arctan2(encoded[..., 8:9], encoded[..., 9:10])), axis=-1)
    return {"continuous": continuous.reshape(-1, 45), "contact": prediction[:, 50:60],
            "transition": prediction[:, 60:70]}


def _statistics(values, valid=None):
    valid = torch.ones_like(values, dtype=torch.bool) if valid is None else valid
    count = valid.sum(0)
    clean = torch.where(valid, values, 0)
    mean = clean.double().sum(0) / count.clamp_min(1)
    variance = torch.where(valid, values.double() - mean, 0).square().sum(0) / count.clamp_min(1)
    std = variance.sqrt()
    std = torch.where((count > 0) & (std >= 1e-6), std, 1.)
    return mean.float(), std.float(), count


def _ridge_weights(x, y, valid, alphas):
    """Unpenalized intercept; objective mean squared error + alpha ||W||².

    Targets with identical validity masks share a single Gram factorization.
    Unobserved targets are explicitly reported by callers and predict zero in
    standardized space rather than solving an unsupported equation.
    """
    x = torch.cat((x.double(), torch.ones(len(x), 1, device=x.device, dtype=torch.float64)), axis=1)
    y = y.double()
    weights = torch.zeros(len(alphas), x.shape[1], y.shape[1], device=x.device, dtype=torch.float64)
    grouped = {}
    masks_cpu = valid.cpu().numpy()
    for coordinate in range(y.shape[1]):
        key = np.packbits(masks_cpu[:, coordinate]).tobytes()
        grouped.setdefault(key, []).append(coordinate)
    penalty = torch.eye(x.shape[1], device=x.device, dtype=torch.float64)
    penalty[-1, -1] = 0
    alpha = torch.as_tensor(alphas, device=x.device, dtype=torch.float64)
    for coordinates in grouped.values():
        selected = valid[:, coordinates[0]]
        count = int(selected.sum())
        if not count:
            continue
        observed_x = x[selected]
        gram = observed_x.T @ observed_x / count
        rhs = observed_x.T @ y[selected][:, coordinates] / count
        systems = gram[None] + alpha[:, None, None] * penalty
        solution = torch.linalg.solve(systems, rhs[None].expand(len(alpha), -1, -1))
        weights[:, :, coordinates] = solution
    if not bool(torch.isfinite(weights).all()):
        raise ValueError("Non-finite linear-probe weights")
    return weights


def _predict(x, weights):
    return (x.double() @ weights[:-1] + weights[-1]).float()


def _balanced_score(prediction, target, valid, sources):
    """Equal source contribution; within each source, average usable scalars."""
    scores = []
    for source in sorted(set(sources.tolist())):
        rows = torch.as_tensor(sources == source, device=target.device)
        usable = valid[rows]
        if bool(usable.any()):
            scores.append((prediction[rows] - target[rows]).square()[usable].mean())
    if not scores:
        raise ValueError("Internal holdout has no usable target")
    return torch.stack(scores).mean()


def _reports(prediction, target, entries):
    rows = probe_metric_rows(prediction, target)
    split = np.asarray([entry["split"] for entry in entries])
    source = np.asarray([entry["source"] for entry in entries])
    episode = np.asarray([entry["episode_index"] for entry in entries])
    report = {}
    for part in ("train", "val"):
        report[part] = {}
        for name in ["overall"] + sorted(set(source)):
            selected = (split == part) & ((source == name) if name != "overall" else True)
            pooled = reduce_probe_metric_rows(rows, selected)
            episode_results = [reduce_probe_metric_rows(rows, selected & (episode == index))
                               for index in np.unique(episode[selected])]
            macro = {metric: float(np.mean([result[metric] for result in episode_results if metric in result]))
                     for metric in METRICS if any(metric in result for result in episode_results)}
            report[part][name] = {"window_weighted": pooled, "episode_macro": macro,
                                 "windows": int(selected.sum()), "episodes": len(episode_results)}
    return report, rows


def _bootstrap(rows_by_model, entries, groups, repetitions=1000, seed=271828, contrasts=None):
    """Paired source-wise connected-group bootstrap of validation metrics."""
    contrasts = (("epoch100", "epoch20"), ("epoch20", "dino"), ("epoch100", "dino")) if contrasts is None else contrasts
    if any(left not in rows_by_model or right not in rows_by_model for left, right in contrasts):
        raise ValueError("Paired bootstrap contrast references a missing model")
    source = np.asarray([entry["source"] for entry in entries])
    split = np.asarray([entry["split"] for entry in entries])
    rng = np.random.default_rng(seed)
    result = {}
    for name in sorted(set(source)):
        selected = (source == name) & (split == "val")
        independent = np.unique(groups[selected])
        if not len(independent):
            continue
        if len(independent) < 2:
            result[name] = {"independent_connected_groups": len(independent),
                            "repetitions": 0, "contrasts": {},
                            "limitation": "只有一个独立连通组，无法估计组间不确定性，不报告置信区间。"}
            continue
        draws = rng.integers(0, len(independent), size=(repetitions, len(independent)))
        totals = {}
        for model, rows in rows_by_model.items():
            summed = {key: np.asarray([value[selected & (groups == group)].sum() for group in independent])
                      for key, value in rows.items()}
            samples = {key: value[draws].sum(1) for key, value in summed.items()}
            metrics = {}
            for metric in ERROR_METRICS:
                numerator, denominator = samples[metric + "_sum"], samples[metric + "_count"]
                metrics[metric] = np.divide(numerator, denominator, out=np.full(repetitions, np.nan), where=denominator > 0)
            for binary in ("contact", "transition"):
                tp, fp, fn, tn = (samples[binary + "_" + key] for key in ("tp", "fp", "fn", "tn"))
                denominator = 2 * tp + fp + fn
                metrics[binary + "_f1"] = np.divide(2 * tp, denominator, out=np.full(repetitions, np.nan), where=denominator > 0)
                sensitivity = np.divide(tp, tp + fn, out=np.full(repetitions, np.nan), where=tp + fn > 0)
                specificity = np.divide(tn, tn + fp, out=np.full(repetitions, np.nan), where=tn + fp > 0)
                metrics[binary + "_balanced_accuracy"] = .5 * (sensitivity + specificity)
            totals[model] = metrics
        contrast_results = {}
        for left, right in contrasts:
            contrast_results[left + "_minus_" + right] = {}
            for metric in totals[left]:
                delta = totals[left][metric] - totals[right][metric]
                delta = delta[np.isfinite(delta)]
                if len(delta):
                    observed_left = reduce_probe_metric_rows(rows_by_model[left], selected)
                    observed_right = reduce_probe_metric_rows(rows_by_model[right], selected)
                    contrast_results[left + "_minus_" + right][metric] = {
                        "mean_delta": float(delta.mean()), "ci95": np.quantile(delta, [.025, .975]).tolist(),
                        "defined_repetitions": len(delta), "lower_is_better": metric in ERROR_METRICS}
                    if metric in observed_left and metric in observed_right:
                        contrast_results[left + "_minus_" + right][metric]["observed_delta"] = observed_left[metric] - observed_right[metric]
        result[name] = {"independent_connected_groups": len(independent), "repetitions": repetitions,
                        "contrasts": contrast_results,
                        "limitation": "组数较少时置信区间不稳定；同一连通组内窗口不作为独立样本。"}
    return result


def _load_targets(path):
    with np.load(path) as arrays:
        result = {}
        for name, width in (("continuous", 45), ("contact", 10), ("transition", 10)):
            key = name if name in arrays else name + str(width)
            result[name] = np.asarray(arrays[key], dtype=np.float32)
            result[name + "_valid"] = np.asarray(arrays[name + "_valid"], dtype=bool)
        result["current_contact"] = np.asarray(arrays["current_contact"], dtype=np.float32)
        result["current_contact_valid"] = np.asarray(arrays["current_contact_valid"], dtype=bool)
    return result


def restore_frozen_readout(job, name, entries, targets, device, target_mean, target_std, target_count):
    """Reconstruct paired per-window predictions without fitting old heads."""
    artifact = torch.load(job / f"{name}-linear-probe.pt", map_location="cpu", weights_only=True)
    if artifact.get("model") != name:
        raise ValueError("Inherited linear readout model identity differs")
    for key, expected in (("target_mean", target_mean), ("target_std", target_std), ("target_count", target_count)):
        actual = artifact[key].to(device=expected.device)
        if actual.shape != expected.shape:
            raise ValueError(f"Inherited linear readout target statistics differ: {key}")
        equal = torch.equal(actual, expected) if key == "target_count" else torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
        if not equal:
            raise ValueError(f"Inherited linear readout target statistics differ: {key}")
    array = np.load(job / f"{name}.npy", mmap_mode="r", allow_pickle=False)
    if array.shape != (len(entries), 768) or array.dtype != np.float32:
        raise ValueError("Inherited linear readout features differ from selection")
    mean, std, weights = (artifact[key].to(device) for key in ("feature_mean", "feature_std", "weights"))
    if (mean.shape != (768,) or std.shape != (768,) or weights.shape != (769, 70)
            or not bool(torch.isfinite(mean).all() & torch.isfinite(std).all() & torch.isfinite(weights).all())
            or not bool((std > 0).all())):
        raise ValueError("Inherited linear readout parameters are invalid")
    features = torch.as_tensor(np.array(array), device=device)
    if not bool(torch.isfinite(features).all()):
        raise ValueError("Inherited linear readout features contain non-finite values")
    prediction = (_predict((features - mean) / std, weights) * artifact["target_std"].to(device)
                  + artifact["target_mean"].to(device)).cpu().numpy()
    decoded = decode_targets(prediction)
    report, rows = _reports(decoded, targets, entries)
    return decoded, report, rows


def _check_replayed_metrics(actual, expected):
    """Allow GPU reduction rounding; publish the original baseline unchanged."""
    largest = 0.
    if actual.keys() != expected.keys():
        raise ValueError("Replayed baseline metric membership differs")
    for key, old in expected.items():
        new = actual[key]
        if isinstance(old, dict):
            largest = max(largest, _check_replayed_metrics(new, old))
        elif isinstance(old, int):
            if type(new) is not int or new != old:
                raise ValueError(f"Replayed baseline observation count differs: {key}={new} vs {old}")
        elif isinstance(old, float):
            if not np.isclose(new, old, rtol=2e-6, atol=2e-5):
                raise ValueError(f"Replayed baseline metric differs: {key}={new} vs {old}")
            largest = max(largest, abs(new - old))
        elif new != old:
            raise ValueError(f"Replayed baseline metadata differs: {key}")
    return largest


def noise_stability_statistics(noise_features, feature_mean, feature_std, weights,
                               target_mean, target_std, targets, entries, noise_rows):
    """Diagnose fixed readout stability; no fitting, reselection or best-of-K.

    Feature RMS is per standardized coordinate. Readout jitter is Euclidean
    RMS around each window's three-draw prediction mean, in physical units.
    Yaw uses a circular mean and wrapped angle differences. Original target
    masks only select diagnostic observations, never model/probe inputs.
    """
    noise = np.asarray(noise_features, dtype=np.float64)
    indices = np.asarray(noise_rows, dtype=np.int64)
    mean, std = np.asarray(feature_mean), np.asarray(feature_std)
    if (noise.ndim != 3 or noise.shape[1:] != (3, len(mean))
            or len(noise) != len(indices) or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(entries))):
        raise ValueError("Noise features require unique selected rows and K x 3 x D features")
    if not np.isfinite(noise).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Noise diagnostics require finite features and positive train scales")
    standardized = (noise - mean) / std
    encoded = (standardized @ np.asarray(weights)[:-1] + np.asarray(weights)[-1])
    encoded = encoded * np.asarray(target_std) + np.asarray(target_mean)
    flattened = decode_targets(encoded.reshape(-1, 70))
    continuous = flattened["continuous"].reshape(len(indices), 3, 5, 9)
    selected_targets = {key: np.asarray(value)[indices] for key, value in targets.items()}
    target_mask = selected_targets["continuous_valid"].reshape(-1, 5, 9)
    sources = np.asarray([entries[index]["source"] for index in indices])
    prediction_by_draw = [{name: value.reshape(len(indices), 3, -1)[:, draw]
                           for name, value in flattened.items()} for draw in range(3)]
    rows_by_draw = [probe_metric_rows(prediction, selected_targets) for prediction in prediction_by_draw]
    report = {}

    def rms_jitter(value, usable, scale):
        # value: windows, draws, horizon[, hand], coordinate.
        centered = value - value.mean(axis=1, keepdims=True)
        squared = np.square(centered).sum(-1)
        usable = np.broadcast_to(usable[:, None], squared.shape)
        return float(np.sqrt(squared[usable].mean()) * scale) if usable.any() else None

    for source in sorted(set(sources)):
        selected = sources == source
        feature = standardized[selected]
        window_mean = feature.mean(1)
        within = float(np.sqrt(np.square(feature - window_mean[:, None]).mean()))
        between = float(np.sqrt(np.square(window_mean - window_mean.mean(0)).mean()))
        cosine = {}
        for draw in (1, 2):
            denominator = np.linalg.norm(feature[:, 0], axis=-1) * np.linalg.norm(feature[:, draw], axis=-1)
            usable = denominator > 1e-12
            value = np.sum(feature[:, 0] * feature[:, draw], axis=-1)
            cosine[str(draw)] = {"mean": float(np.mean(np.clip(value[usable] / denominator[usable], -1, 1))) if usable.any() else None,
                                 "count": int(usable.sum())}
        value = continuous[selected]
        usable = target_mask[selected]
        wrist = value[..., :6].reshape(-1, 3, 5, 2, 3)
        wrist_valid = usable[..., :6].reshape(-1, 5, 2, 3).all(-1)
        yaw = value[..., 8]
        circular_mean = np.arctan2(np.sin(yaw).mean(1), np.cos(yaw).mean(1))
        difference = yaw - circular_mean[:, None]
        wrapped = np.arctan2(np.sin(difference), np.cos(difference))
        yaw_valid = np.broadcast_to(usable[:, None, :, 8], wrapped.shape)
        yaw_jitter = float(np.sqrt(np.square(wrapped[yaw_valid]).mean()) * 180 / np.pi) if yaw_valid.any() else None
        report[source] = {
            "windows": int(selected.sum()), "draws": 3,
            "feature_within_noise_rms": within, "feature_between_window_rms": between,
            "feature_within_to_between_ratio": within / between if between > 1e-12 else None,
            "train_centered_cosine_draw0_vs": cosine,
            "wrist_readout_jitter_cm": rms_jitter(wrist, wrist_valid, 100.),
            "reference_xy_readout_jitter_cm": rms_jitter(value[..., 6:8], usable[..., 6:8].all(-1), 100.),
            "reference_yaw_readout_jitter_deg": yaw_jitter,
            "draw_metrics": {str(draw): reduce_probe_metric_rows(rows, selected) for draw, rows in enumerate(rows_by_draw)},
            "selection_rule": "同一固定ridge逐draw评分；不重选超参、不平均预测、不取best-of-K。"}
    return report


def _write_report(path, results):
    lines = ["# 冻结运动先验验证", "", "MotionExpert 全程冻结；只训练线性 ridge 读出头。主验证集不参与调参。",
             "DINO保留6帧×5token顺序后在原train拟合PCA768；该压缩可能损失信息。",
             "接触/切换采用最小二乘线性读出、0.5阈值，输出不是校准概率。", "",
             "| 来源 | 特征 | 腕位移误差cm | 参考位移误差cm | 转向误差° | 接触F1 | 切换F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for source in sorted(results["population"]["val"]):
        for model, report in results["models"].items():
            values = report["val"][source]["window_weighted"]
            formatted = [f"{values[key]:.4f}" if key in values else "无有效标签" for key in
                         ("wrist_displacement_cm", "reference_displacement_xy_cm", "reference_yaw_deg", "contact_f1", "transition_f1")]
            lines.append("| " + " | ".join([source, model] + formatted) + " |")
    if results.get("noise_stability"):
        lines += ["", "同一历史窗口更换三份噪声；使用最终train均值/尺度和固定ridge，不重选、不取最优噪声。",
                  "cosine先减train均值并按train尺度标准化，避免共同特征均值造成虚高。",
                  "| 来源 | 特征 | 特征噪声/窗口RMS比 | 腕读出抖动cm | 参考XY抖动cm | 转向抖动° |",
                  "|---|---|---:|---:|---:|---:|"]
        for model, sources in results["noise_stability"].items():
            for source, values in sources.items():
                numbers = [values[key] for key in ("feature_within_to_between_ratio", "wrist_readout_jitter_cm",
                                                   "reference_xy_readout_jitter_cm", "reference_yaw_readout_jitter_deg")]
                lines.append("| " + " | ".join([source, model] + [f"{number:.5f}" if number is not None else "无有效观测" for number in numbers]) + " |")
    lines += ["", "各组train、val、逐来源episode宏平均、覆盖数量、选择的ridge强度和配对组bootstrap见 metrics.json。",
              "误差越小越好，F1/平衡准确率越大越好。bootstrap差值为前者减后者。",
              "延续接触参照使用真实当前接触标签，不是可部署纯图像模型；仅在当前接触有效的共同子集评分。",
              "不同来源的接触切换比例不同，不直接比较两来源的accuracy或F1高低。",
              "", "这项实验检验冻结特征的运动信息可读性，不能替代真实下游有/无先验对照，也不能直接证明下游过拟合。", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def fit_job(job_dir, device="cuda:0"):
    """Fit frozen readouts, or only the newly extracted incremental readout."""
    job = Path(job_dir).resolve()
    started = time.monotonic()
    torch.set_num_threads(2)
    if torch.device(device).type == "cuda":
        torch.cuda.set_device(torch.device(device))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    raw = json.loads((job / "selection.json").read_text())
    entries = raw["entries"] if isinstance(raw, dict) else raw
    if not entries or any(entry["row"] != index for index, entry in enumerate(entries)):
        raise ValueError("Selection rows must exactly match contiguous feature rows")
    groups = connected_group_ids(entries)
    fitting, holdout = internal_group_split(entries, groups)
    split = np.asarray([entry["split"] for entry in entries])
    source = np.asarray([entry["source"] for entry in entries])
    if set(split) != {"train", "val"}:
        raise ValueError("Frozen probe requires train/val only; test must remain untouched")
    train = split == "train"
    targets = _load_targets(job / "targets.npz")
    plan_path = job / "plan.json"
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    reuse = plan.get("reuse_job")
    baseline = None
    models_to_fit = plan.get("fit_models", ["epoch20", "epoch100", "dino"])
    if reuse:
        from .probe_prior import assert_reuse_current
        assert_reuse_current(plan)
        if models_to_fit != ["epoch160"] or plan.get("extract_epochs") != [160]:
            raise ValueError("Incremental fitting may update only the new epoch160 readout")
        baseline = json.loads((Path(reuse["job_dir"]) / "metrics.json").read_text())
    elif models_to_fit != ["epoch20", "epoch100", "dino"]:
        raise ValueError("Original frozen probe must fit its complete baseline set")
    noise_rows = plan.get("noise_rows", [])
    if any((job / f"noise-{name}.npy").exists() for name in models_to_fit if name != "dino") and not noise_rows:
        raise ValueError("Noise arrays require plan.json noise_rows provenance")
    values, masks = encode_targets(targets)
    if len(values) != len(entries):
        raise ValueError("Label population differs from feature selection")
    y = torch.as_tensor(values, device=device)
    valid = torch.as_tensor(masks, device=device)
    fit_index = torch.as_tensor(fitting, device=device)
    hold_index = torch.as_tensor(holdout, device=device)
    train_index = torch.as_tensor(train, device=device)
    y_inner_mean, y_inner_std, inner_count = _statistics(y[fit_index], valid[fit_index])
    y_inner = torch.where(valid, (y - y_inner_mean) / y_inner_std, 0)
    y_mean, y_std, full_count = _statistics(y[train_index], valid[train_index])
    y_scaled = torch.where(valid, (y - y_mean) / y_std, 0)
    output = {"protocol": {"feature_dim": 768, "ridge_alphas": list(ALPHAS),
                          "device": str(device),
                          "gpu_name": torch.cuda.get_device_name(torch.device(device)) if torch.device(device).type == "cuda" else None,
                          "solve_dtype": "float64", "feature_dtype": "float32", "tf32": False,
                          "fit_source_weighting": "mean over valid observations; approximately balanced source window quotas",
                          "persistence_baseline": "uses true current contact labels; label-assisted reference, not deployable image-only model",
                          "objective": "mean masked standardized MSE + alpha squared weights; intercept unpenalized",
                          "alpha_selection": "original train internal connected-group holdout; equal source weighting",
                          "pca_scope": "original train features only, including internal holdout; no labels/val features",
                          "binary_readout": "linear least squares; uncalibrated; threshold 0.5",
                          "normalization": "feature/target selection statistics fit on internal fitting group; final refit on all train",
                          "no_target_count_inner": torch.nonzero(inner_count == 0).flatten().cpu().tolist(),
                          "no_target_count_train": torch.nonzero(full_count == 0).flatten().cpu().tolist()},
              "population": {}, "models": {}, "hyperparameters": {}, "noise_stability": {}}
    for part in ("train", "val"):
        output["population"][part] = {name: {
            "windows": int(((split == part) & (source == name)).sum()),
            "episodes": len({entry["episode_index"] for entry in entries if entry["split"] == part and entry["source"] == name}),
            "connected_groups": len(np.unique(groups[(split == part) & (source == name)]))}
            for name in sorted(set(source))}
    output["internal_split"] = {"fitting_rows": np.flatnonzero(fitting).tolist(),
                                "holdout_rows": np.flatnonzero(holdout).tolist()}
    rows_by_model, decoded_by_model = {}, {}
    if baseline is not None:
        if baseline["population"] != output["population"] or baseline["internal_split"] != output["internal_split"]:
            raise ValueError("Inherited readouts use a different population/internal group holdout")
        if (baseline["protocol"]["ridge_alphas"] != list(ALPHAS)
                or baseline["protocol"]["objective"] != output["protocol"]["objective"]
                or baseline["protocol"]["alpha_selection"] != output["protocol"]["alpha_selection"]):
            raise ValueError("Inherited ridge selection/training protocol differs")
        output["reused_baseline"] = {"job_dir": reuse["job_dir"], "readouts_refitted": False,
                                    "dino_pca_refitted": False, "metric_replay_max_abs_error": {}}
        for key in ("models", "hyperparameters", "noise_stability"):
            output[key].update(baseline[key])
        if "pca" in baseline:
            output["pca"] = baseline["pca"]
        for name in ("epoch20", "epoch100"):
            decoded, report, rows = restore_frozen_readout(job, name, entries, targets, device, y_mean, y_std, full_count)
            output["reused_baseline"]["metric_replay_max_abs_error"][name] = _check_replayed_metrics(report, baseline["models"][name])
            decoded_by_model[name], rows_by_model[name] = decoded, rows
    for name in models_to_fit:
        print(f"[prior-probe] fitting {name}", flush=True)
        array = np.load(job / (name + ".npy"), mmap_mode="r")
        expected = 30720 if name == "dino" else 768
        if array.shape != (len(entries), expected):
            raise ValueError(f"{name} feature shape differs from protocol")
        transform = {}
        if name == "dino":
            train_values = torch.as_tensor(np.array(array[train]), device=device)
            if not bool(torch.isfinite(train_values).all()):
                raise ValueError("DINO contains non-finite train values")
            pca_mean = train_values.mean(0)
            train_values -= pca_mean
            q = min(800, min(train_values.shape))
            if q < 768:
                raise ValueError("PCA768 requires at least 768 train samples")
            with torch.random.fork_rng(devices=[torch.device(device).index or 0] if torch.device(device).type == "cuda" else []):
                torch.manual_seed(161803)
                _, singular, basis = torch.pca_lowrank(train_values, q=q, center=False, niter=2)
            basis = basis[:, :768].contiguous()
            total_variance = train_values.double().square().sum()
            explained = float(singular[:768].double().square().sum() / total_variance.clamp_min(1e-20))
            output["pca"] = {"dimensions": 768, "q": q, "niter": 2,
                             "explained_variance_fraction": explained, "fit_rows": int(train.sum())}
            del train_values
            features = torch.empty(len(entries), 768, device=device)
            for first in range(0, len(entries), 1024):
                chunk = torch.as_tensor(np.array(array[first:first + 1024]), device=device)
                features[first:first + len(chunk)] = (chunk - pca_mean) @ basis
            transform = {"pca_mean": pca_mean.cpu(), "pca_basis": basis.cpu()}
        else:
            features = torch.as_tensor(np.array(array), device=device)
        if not bool(torch.isfinite(features).all()):
            raise ValueError(f"{name} contains non-finite features")
        mean, std, _ = _statistics(features[fit_index])
        standardized = (features - mean) / std
        candidates = _ridge_weights(standardized[fit_index], y_inner[fit_index], valid[fit_index], ALPHAS)
        scores = []
        for weights in candidates:
            prediction = _predict(standardized[hold_index], weights)
            scores.append(float(_balanced_score(prediction, y_inner[hold_index], valid[hold_index], source[holdout])))
        chosen = int(np.argmin(scores))
        alpha = ALPHAS[chosen]
        print(f"[prior-probe] {name} alpha={alpha} holdout={scores[chosen]:.6f}", flush=True)
        mean, std, _ = _statistics(features[train_index])
        standardized = (features - mean) / std
        weights = _ridge_weights(standardized[train_index], y_scaled[train_index], valid[train_index], [alpha])[0]
        prediction = (_predict(standardized, weights) * y_std + y_mean).cpu().numpy()
        decoded_by_model[name] = decode_targets(prediction)
        output["models"][name], rows_by_model[name] = _reports(decoded_by_model[name], targets, entries)
        output["hyperparameters"][name] = {"alpha": alpha, "holdout_scores": dict(zip(map(str, ALPHAS), scores))}
        artifact = {"model": name, "feature_mean": mean.cpu(), "feature_std": std.cpu(),
                    "weights": weights.cpu(), "target_mean": y_mean.cpu(), "target_std": y_std.cpu(),
                    "target_count": full_count.cpu(), "alpha": alpha, **transform}
        torch.save(artifact, job / (name + "-linear-probe.pt"))
        if name != "dino" and noise_rows:
            noise_path = job / f"noise-{name}.npy"
            if not noise_path.is_file():
                raise FileNotFoundError(f"Declared noise sensitivity arrays missing: {noise_path}")
            output["noise_stability"][name] = noise_stability_statistics(
                np.load(noise_path, mmap_mode="r"), mean.cpu().numpy(), std.cpu().numpy(), weights.cpu().numpy(),
                y_mean.cpu().numpy(), y_std.cpu().numpy(), targets, entries, noise_rows)
        del features, standardized, candidates, weights
    persistence = {"continuous": np.zeros_like(targets["continuous"]),
                   "contact": np.tile(targets["current_contact"], (1, 5)),
                   "transition": np.zeros_like(targets["transition"])}
    baseline_targets = dict(targets)
    baseline_targets["contact_valid"] = targets["contact_valid"] & np.tile(targets["current_contact_valid"], (1, 5))
    baseline_targets["transition_valid"] = targets["transition_valid"] & np.tile(targets["current_contact_valid"], (1, 5))
    persistence_report, _ = _reports(persistence, baseline_targets, entries)
    if baseline is not None:
        _check_replayed_metrics(persistence_report, baseline["models"]["zero_motion_persistent_contact"])
        persistence_report = baseline["models"]["zero_motion_persistent_contact"]
    output["models"]["zero_motion_persistent_contact"] = persistence_report
    output["persistent_contact_common_mask_comparison"] = {
        name: _reports(prediction, baseline_targets, entries)[0]
        for name, prediction in decoded_by_model.items()}
    if baseline is not None:
        output["persistent_contact_common_mask_comparison"].update(baseline["persistent_contact_common_mask_comparison"])
        output["paired_group_bootstrap"] = baseline["paired_group_bootstrap"]
        added = _bootstrap(rows_by_model, entries, groups,
                           contrasts=(("epoch160", "epoch20"), ("epoch160", "epoch100")))
        for source_name, result in added.items():
            previous = output["paired_group_bootstrap"].setdefault(source_name, result)
            if (previous["independent_connected_groups"] != result["independent_connected_groups"]
                    or previous["repetitions"] != result["repetitions"]):
                raise ValueError("Incremental group bootstrap population differs")
            previous["contrasts"].update(result["contrasts"])
        assert_reuse_current(plan)
    else:
        output["paired_group_bootstrap"] = _bootstrap(rows_by_model, entries, groups)
    output["runtime"] = {"elapsed_seconds": time.monotonic() - started,
                         "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(torch.device(device)) if torch.device(device).type == "cuda" else None,
                         "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                           for path in (Path(__file__), Path(__file__).with_name("prior_probe_math.py"))}}
    (job / "metrics.json").write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_report(job / "report.md", output)
    print(f"[prior-probe] complete: {job / 'report.md'}", flush=True)
    return output
