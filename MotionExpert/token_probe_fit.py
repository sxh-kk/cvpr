"""Equal-capacity readouts of frozen, temporally ordered prior tokens.

This module never loads MotionExpert or an optimizer belonging to the expert.
The four readouts have exactly the same trainable parameters.  Motion hidden
features are embedded isometrically into DINO's native 1024 dimensions; no PCA,
temporal mean pooling, or learned input adapters precede the readout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .prior_probe_fit import (
    ERROR_METRICS, _balanced_score, _bootstrap, _load_targets, _reports,
    connected_group_ids, decode_targets, encode_targets, internal_group_split,
)
from .prior_probe_math import masked_train_statistics, probe_metric_rows

ARMS = ("history6", "future5", "all11", "dino")
TOKEN_INDICES = {"history6": tuple(range(6)), "future5": tuple(range(6, 11)),
                 "all11": tuple(range(11)), "dino": tuple(i for i in range(6) for _ in range(5))}
PARAMETER_COUNT = 290886


def validate_extraction(job_dir, *, require_unwritten=True):
    """Reject foreign, incomplete, changed, or previously fitted job inputs.

    This guard performs metadata/header checks only. It does not open the
    source archive, repeat numeric dataset acceptance, or initialize CUDA.
    """
    from .probe_prior import file_stamp, make_shards
    from .probe_tokens import FORMAT, assert_bindings
    job = Path(job_dir).resolve(strict=True)
    repository = Path(__file__).resolve().parents[1]
    if job.is_relative_to(repository):
        raise ValueError("Token readout artifacts must be outside Git")
    plan = json.loads((job / "plan.json").read_text())
    if (plan.get("format") != FORMAT or Path(plan.get("job_dir", "")).resolve() != job
            or plan.get("epoch") != 100 or plan.get("precision") != "float32"
            or plan.get("weights") != "ema" or plan.get("token_shape") != [11, 768]):
        raise ValueError("Token readout requires its own epoch100 FP32 EMA extraction plan")
    base = Path(plan["reuse_job"]["job_dir"]).resolve()
    if job == base or job.is_relative_to(base) or base.is_relative_to(job):
        raise ValueError("Token readout job must be independent of its immutable baseline")
    if require_unwritten:
        patterns = ("token-*-head.pt", "token-*-predictions.npy", "token-fixed-isometry.pt",
                    "token-metrics*.json", "token-report.md", "token-fit.lock", "fit-plan.json", "fit-progress.json")
        if any(any(job.glob(pattern)) for pattern in patterns):
            raise FileExistsError("Refuse to overwrite an existing token readout run")
    assert_bindings(plan)
    raw = json.loads((job / "selection.json").read_text())
    entries = raw["entries"] if isinstance(raw, dict) else raw
    count = plan["windows"]
    if (not isinstance(entries, list) or len(entries) != count
            or any(entry.get("row") != index for index, entry in enumerate(entries))):
        raise ValueError("Token extraction selection is incomplete or reordered")
    completed = json.loads((job / "extraction-complete.json").read_text())
    if (completed.get("passed") is not True or completed.get("epoch") != 100
            or completed.get("windows") != count or completed.get("token_shape") != [count, 11, 768]
            or completed.get("historical_image_valid_shape") != [count, 6]
            or completed.get("all_tokens_finite") is not True
            or completed.get("expert_gradients") is not False
            or completed.get("expert_trainable_parameters") != 0):
        raise ValueError("Full-token extraction has not passed its complete GPU acceptance")
    logical = plan.get("logical_shards", [])
    if len(logical) != 8 or len(set(logical)) != 8 or completed.get("logical_shards") != 8:
        raise ValueError("Full-token extraction requires all eight paired logical shards")
    shards = make_shards(entries, logical)
    input_names = ["plan.json", "selection.json", "targets.npz", "dino.npy", "tokens.npy",
                   "image-valid.npy", "extraction-complete.json"]
    for index, shard in zip(logical, shards):
        name = f"done-shard{index}.json"
        result = json.loads((job / name).read_text())
        rows_sha256 = hashlib.sha256(np.asarray([row["row"] for row in shard], dtype=np.int64).tobytes()).hexdigest()
        if (result.get("passed") is not True or result.get("epoch") != 100
                or result.get("logical_shard") != index or result.get("windows") != len(shard)
                or result.get("hidden_shape") != [len(shard), 11, 768]
                or result.get("rows_sha256") != rows_sha256
                or result.get("expert_gradients") is not False
                or result.get("expert_trainable_parameters") != 0
                or result.get("acceptance", {}).get("passed") is not True):
            raise ValueError(f"Logical token shard lacks paired frozen-GPU acceptance: {index}")
        old = json.loads((base / f"done-100-gpu{index}.json").read_text())
        if any(result.get(key) != old.get(key) for key in ("rows_sha256", "noise_sha256", "dino_sha256", "windows")):
            raise ValueError(f"Logical token shard no longer matches the accepted baseline: {index}")
        input_names.append(name)
    specs = {"tokens.npy": ((count, 11, 768), np.float32),
             "dino.npy": ((count, 30720), np.float32), "image-valid.npy": ((count, 6), np.bool_)}
    for name, (shape, dtype) in specs.items():
        array = np.load(job / name, mmap_mode="r", allow_pickle=False)
        if array.shape != shape or array.dtype != dtype:
            raise ValueError(f"Accepted token input header changed: {name}")
        del array
    with np.load(job / "targets.npz", allow_pickle=False) as labels:
        if set(labels.files) != set(plan["label_specs"]):
            raise ValueError("Token target fields differ from accepted baseline")
        for name, spec in plan["label_specs"].items():
            if list(labels[name].shape) != spec["shape"] or str(labels[name].dtype) != spec["dtype"]:
                raise ValueError(f"Token target header changed: {name}")
    stamps = {name: file_stamp(job / name) for name in input_names}
    assert_bindings(plan)
    return plan, stamps


def assert_extraction_unchanged(job_dir, plan, stamps):
    from .probe_prior import file_stamp
    from .probe_tokens import assert_bindings
    job = Path(job_dir)
    assert_bindings(plan)
    for name, expected in stamps.items():
        if file_stamp(job / name) != expected:
            raise ValueError(f"Read-only token input changed during fitting: {name}")


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _state_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fixed_isometry(input_dim=768, output_dim=1024, seed=20261019):
    """An immutable, label-independent row-orthonormal embedding (P P^T=I)."""
    if input_dim > output_dim or input_dim < 1:
        raise ValueError("Information-preserving embedding requires 0 < input <= output")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    matrix = torch.randn(output_dim, input_dim, generator=generator, dtype=torch.float64)
    basis, triangular = torch.linalg.qr(matrix, mode="reduced")
    # Fix QR sign ambiguity before converting to the runtime feature dtype.
    signs = torch.where(torch.diag(triangular) < 0, -1., 1.)
    return (basis * signs).T.contiguous().float()


def fixed_time_encoding(indices, width=1024):
    """Shared sinusoidal keys at true 5 Hz times, current index=5."""
    indices = torch.as_tensor(indices, dtype=torch.float32)
    seconds = (indices - 5.) / 5.
    frequencies = torch.exp(torch.arange(0, width, 2, dtype=torch.float32)
                            * (-math.log(10000.) / width))
    phase = seconds[:, None] * 10. * frequencies[None]
    output = torch.empty(len(indices), width)
    output[:, 0::2], output[:, 1::2] = phase.sin(), phase.cos()
    return output


def key_encoding(arm, width=1024):
    if arm not in ARMS:
        raise ValueError(f"Unknown token arm: {arm}")
    encoding = fixed_time_encoding(TOKEN_INDICES[arm], width)
    if arm == "dino":
        # Distinguishes CLS and four register tokens without trainable adapters.
        type_ids = torch.arange(5).repeat(6)
        encoding = encoding + .25 * fixed_time_encoding(type_ids + 5, width)
    return encoding


class TokenQueryReadout(nn.Module):
    """Four learned queries, content values, and one 4096 -> 70 linear head.

    No Q/K/V projections, FFN, normalization parameters, or branch-dependent
    learned positional embeddings are used. Invalid DINO tokens are excluded
    from both attention normalization and values, including all-invalid rows.
    """
    def __init__(self, width=1024, queries=4, outputs=70):
        super().__init__()
        self.width = int(width)
        self.queries = nn.Parameter(torch.empty(queries, width))
        nn.init.normal_(self.queries, std=1.)
        self.output = nn.Linear(queries * width, outputs)

    def forward(self, tokens, position, token_valid=None, *, return_attention=False):
        if tokens.ndim != 3 or tokens.shape[-1] != self.width:
            raise ValueError("Readout input requires B x T x common width")
        if position.shape != tokens.shape[1:]:
            raise ValueError("Fixed key positions must retain the selected token time indices")
        if token_valid is None:
            token_valid = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        if token_valid.dtype != torch.bool or token_valid.shape != tokens.shape[:2]:
            raise ValueError("Input token mask must be boolean B x T")
        clean = torch.where(token_valid[..., None], tokens, 0.)
        scores = torch.einsum("btd,qd->bqt", clean + position[None], self.queries) / math.sqrt(self.width)
        scores = scores.masked_fill(~token_valid[:, None], -torch.inf)
        any_valid = token_valid.any(-1)
        scores = torch.where(any_valid[:, None, None], scores, 0.)
        attention = scores.softmax(-1) * token_valid[:, None]
        pooled = torch.einsum("bqt,btd->bqd", attention, clean)
        prediction = self.output(pooled.flatten(1))
        return (prediction, attention) if return_attention else prediction


def _native_statistics(array, fitting_rows, token_valid=None, *, chunk_size=512):
    """Per native channel moments; all eleven motion tokens share one scale."""
    rows = np.asarray(fitting_rows, dtype=np.int64)
    if not len(rows):
        raise ValueError("Feature statistics require non-empty fitting rows")
    width = array.shape[-1]
    total = np.zeros(width, dtype=np.float64)
    squared = np.zeros(width, dtype=np.float64)
    count = 0
    for first in range(0, len(rows), chunk_size):
        selected = rows[first:first + chunk_size]
        value = np.asarray(array[selected], dtype=np.float64)
        eligible = np.ones(value.shape[:2], dtype=bool) if token_valid is None else np.asarray(token_valid[selected], dtype=bool)
        clean = value[eligible]
        if not np.isfinite(clean).all():
            raise ValueError("A valid frozen input token contains non-finite features")
        total += clean.sum(0)
        squared += np.square(clean).sum(0)
        count += len(clean)
    if not count:
        raise ValueError("No valid input token exists in the fitting partition")
    mean = total / count
    deviation = np.sqrt(np.maximum(squared / count - mean * mean, 0.))
    std = np.where(deviation >= 1e-6, deviation, 1.)
    return {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "tokens": count,
            "rows": len(rows), "constant_channels": np.flatnonzero(deviation < 1e-6).tolist()}


def _materialize(array, statistics, device, projection=None, token_valid=None, *, batch_size=256):
    output_width = projection.shape[1] if projection is not None else array.shape[-1]
    features = torch.empty((len(array), array.shape[1], output_width), dtype=torch.float32, device=device)
    mean = torch.as_tensor(statistics["mean"], device=device)
    std = torch.as_tensor(statistics["std"], device=device)
    for first in range(0, len(array), batch_size):
        part = slice(first, first + batch_size)
        value = torch.as_tensor(np.array(array[part]), device=device)
        if token_valid is not None:
            usable = torch.as_tensor(np.array(token_valid[part]), dtype=torch.bool, device=device)
            if not bool(torch.isfinite(value[usable]).all()):
                raise ValueError("A valid frozen input token contains non-finite features")
            value = torch.where(usable[..., None], (value - mean) / std, 0.)
        else:
            if not bool(torch.isfinite(value).all()):
                raise ValueError("Frozen motion hidden contains non-finite features")
            value = (value - mean) / std
        if projection is not None:
            value = value @ projection
        features[part] = value
    return features


def _control(job, sleep_seconds):
    """Optional operational throttle; does not change numerical settings."""
    path = job / "fit-control.json"
    delay = sleep_seconds
    while True:
        control = json.loads(path.read_text()) if path.exists() else {}
        paused = control.get("paused", False)
        raw_delay = control.get("sleep_seconds", sleep_seconds)
        if type(paused) is not bool or isinstance(raw_delay, bool) or not isinstance(raw_delay, (int, float)):
            raise ValueError("fit-control requires boolean paused and a numeric sleep_seconds")
        delay = float(raw_delay)
        if not math.isfinite(delay) or delay < 0 or delay > 5:
            raise ValueError("fit-control sleep_seconds must be between zero and five")
        if not paused:
            break
        time.sleep(.5)
    if delay:
        time.sleep(delay)


def _new_head(seed, device):
    torch.manual_seed(seed)
    if torch.device(device).type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = TokenQueryReadout().to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != PARAMETER_COUNT:
        raise AssertionError("Readout capacity no longer matches the declared protocol")
    return model


def _masked_loss(prediction, values, masks):
    if not bool(masks.any()):
        raise ValueError("A readout batch has no usable target")
    # Labels are sanitized by encode_targets; masking also prevents an invalid
    # placeholder from affecting any scalar loss or gradient.
    return torch.where(masks, (prediction - values).square(), 0.).sum() / masks.sum()


def _first_batch_review(model, features, position, token_mask, loss):
    query_gradient = model.queries.grad
    if query_gradient is None or not bool(torch.isfinite(query_gradient).all()) or float(query_gradient.norm()) <= 0:
        raise ValueError("Query gradient must be finite and nonzero on the real first batch")
    with torch.no_grad():
        prediction, attention = model(features, position, token_mask, return_attention=True)
        if not bool(torch.isfinite(prediction).all() & torch.isfinite(attention).all()):
            raise ValueError("First-batch predictions/attention are not finite")
        mask = token_mask.clone() if token_mask is not None else torch.ones(features.shape[:2], dtype=torch.bool, device=features.device)
        mask[:, 0] = False
        ordinary = model(features, position, mask)
        corrupted = features.clone()
        corrupted[:, 0] = torch.nan
        masked = model(corrupted, position, mask)
        difference = float((ordinary - masked).abs().max())
        if difference != 0:
            raise ValueError("Invalid token contents changed a readout prediction")
        empty = model(corrupted, position, torch.zeros_like(mask))
        if not bool(torch.isfinite(empty).all()):
            raise ValueError("All-invalid image inputs must yield a finite bias-only prediction")
        valid_count = (token_mask.sum(-1) if token_mask is not None
                       else torch.full((len(features),), features.shape[1], device=features.device))
        eligible = valid_count > 1
        uniform = 1. / valid_count.clamp_min(1)
        attention_variance = float((attention[eligible] - uniform[eligible, None, None]).square().mean()) if bool(eligible.any()) else 0.
        entropy = -(attention * attention.clamp_min(1e-20).log()).sum(-1)
        if bool(eligible.any()) and attention_variance <= 1e-12:
            raise ValueError("Query attention collapsed to exact uniform averaging")
    return {"passed": True, "batch": len(features), "loss": float(loss),
            "query_gradient_norm": float(query_gradient.norm()),
            "prediction_variance": float(prediction.var(unbiased=False)),
            "attention_entropy_mean": float(entropy.mean()), "attention_nonuniform_variance": attention_variance,
            "masked_nan_max_abs_difference": difference, "all_invalid_finite": True}


@torch.no_grad()
def _predict(model, features, position, token_valid, indices, batch_size, job, sleep_seconds):
    prediction = []
    model.eval()
    for first in range(0, len(indices), batch_size):
        selected = indices[first:first + batch_size]
        mask = None if token_valid is None else token_valid[selected]
        value = model(features[selected], position, mask)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("Readout prediction contains a non-finite value")
        prediction.append(value)
        _control(job, sleep_seconds)
    return torch.cat(prediction)


def _train(model, features, position, token_valid, y, valid, fitting, *, seed,
           epochs, batch_size, job, sleep_seconds, holdout=None, sources=None,
           patience=6, progress=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    indices = np.asarray(fitting, dtype=np.int64)
    curve, best_score, best_epoch, stale = [], float("inf"), 0, 0
    generator = np.random.default_rng(seed + 1000003)
    first_review, initial_digest = None, _state_digest(model)
    for epoch in range(1, epochs + 1):
        model.train()
        permutation = generator.permutation(indices)
        squared_sum, observed = 0., 0
        for first in range(0, len(permutation), batch_size):
            selected = permutation[first:first + batch_size]
            mask = None if token_valid is None else token_valid[selected]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(features[selected], position, mask)
            loss = _masked_loss(prediction, y[selected], valid[selected])
            if not bool(torch.isfinite(loss)):
                raise ValueError("Readout training loss is non-finite")
            loss.backward()
            if first_review is None:
                first_review = _first_batch_review(model, features[selected], position, mask, loss.detach())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if not all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()):
                raise ValueError("Readout parameter became non-finite after the update")
            count = int(valid[selected].sum())
            squared_sum += float(loss.detach()) * count
            observed += count
            _control(job, sleep_seconds)
        record = {"epoch": epoch, "train_masked_standardized_mse": squared_sum / observed}
        if holdout is not None:
            predictions = _predict(model, features, position, token_valid, holdout,
                                   batch_size, job, sleep_seconds)
            score = float(_balanced_score(predictions, y[holdout], valid[holdout], sources[holdout]))
            if not math.isfinite(score):
                raise ValueError("Internal holdout score is non-finite")
            record["internal_source_balanced_mse"] = score
            if score < best_score - 1e-7:
                best_score, best_epoch, stale = score, epoch, 0
            else:
                stale += 1
        curve.append(record)
        if progress is not None:
            progress(record)
        if holdout is not None and stale >= patience:
            break
    final_digest = _state_digest(model)
    if final_digest == initial_digest:
        raise ValueError("Readout parameters did not update")
    first_review.update({"parameter_update": True, "initial_state_sha256": initial_digest,
                         "final_state_sha256": final_digest})
    return {"curve": curve, "chosen_epoch": best_epoch if holdout is not None else epochs,
            "best_internal_source_balanced_mse": best_score if holdout is not None else None,
            "gpu_review": first_review, "last_gradient_norm_before_clip": float(norm)}


def _extra_metrics(prediction, values, masks, targets, entries):
    """Squared errors supplement mean Euclidean errors, not an ensemble score."""
    split = np.asarray([entry["split"] for entry in entries])
    source = np.asarray([entry["source"] for entry in entries])
    decoded = decode_targets(prediction)
    difference = (decoded["continuous"] - targets["continuous"]).reshape(-1, 5, 9)
    continuous_valid = targets["continuous_valid"].reshape(-1, 5, 9)
    train = split == "train"
    statistics = masked_train_statistics(values[train], masks[train])
    standardized = np.square((prediction - values) / statistics["std"])
    output = {}
    for part in ("train", "val"):
        output[part] = {}
        for name in ("overall", *sorted(set(source))):
            selected = (split == part) & ((source == name) if name != "overall" else True)
            usable = masks[selected]
            result = {"masked_standardized_mse": float(standardized[selected][usable].mean())}
            wrist = difference[selected, :, :6].reshape(-1, 5, 2, 3)
            wrist_valid = continuous_valid[selected, :, :6].reshape(-1, 5, 2, 3).all(-1)
            ref_valid = continuous_valid[selected, :, 6:8].all(-1)
            if wrist_valid.any():
                result["wrist_vector_rmse_cm"] = float(np.sqrt(np.square(wrist).sum(-1)[wrist_valid].mean()) * 100.)
            if ref_valid.any():
                result["reference_xy_vector_rmse_cm"] = float(np.sqrt(np.square(difference[selected, :, 6:8]).sum(-1)[ref_valid].mean()) * 100.)
            output[part][name] = result
    return output


def _population(entries, groups):
    output = {}
    for part in ("train", "val"):
        output[part] = {}
        for source in sorted({entry["source"] for entry in entries}):
            selected = np.asarray([entry["split"] == part and entry["source"] == source for entry in entries])
            output[part][source] = {"windows": int(selected.sum()),
                "episodes": len({entry["episode_index"] for entry, keep in zip(entries, selected) if keep}),
                "connected_groups": len(np.unique(groups[selected]))}
    return output


def _mean_std_reports(reports):
    """Report the distribution across separately trained heads, never ensembles."""
    def reduce(values):
        if isinstance(values[0], dict):
            keys = set.intersection(*(set(value) for value in values))
            return {key: reduce([value[key] for value in values]) for key in sorted(keys)}
        if isinstance(values[0], (int, float)):
            return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.,
                    "seeds": len(values)}
        return values[0]
    return reduce(reports)


def _write_report(job, results):
    lines = ["# 第100轮完整 token 冻结读出对照", "",
        "MotionExpert冻结；比较历史6个、未来5个、全部11个hidden以及原始DINO 30个token。",
        "四路读出均为4个可学习query＋同一形式线性输出，共290,886个可训练参数；没有先mean pooling。",
        "hidden768→1024使用固定半正交等距升维，DINO保留1024维，不做PCA或降维。",
        "历史真实时间索引保留；DINO额外固定CLS/register编码，无效图像token独立屏蔽。",
        "仅在原train内部连通组holdout选轮数，再从相同初始化用全train重训；主val不参与调参。",
        "以下为独立head seed的均值±样本标准差，不是多个head的预测集成。", "",
        "| 来源 | 输入 | 腕位移cm↓ | 参考XY位移cm↓ | 转向°↓ | 接触F1↑ | 切换F1↑ |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for source in sorted(results["population"]["val"]):
        for arm in ARMS:
            metrics = results["summary"][arm]["val"][source]["window_weighted"]
            formatted = []
            for key in ("wrist_displacement_cm", "reference_displacement_xy_cm", "reference_yaw_deg", "contact_f1", "transition_f1"):
                item = metrics.get(key)
                formatted.append(f"{item['mean']:.4f} ± {item['std']:.4f}" if item else "无有效标签")
            lines.append("| " + " | ".join([source, arm, *formatted]) + " |")
    lines += ["", "逐seed指标、RMSE、masked MSE、训练曲线、GPU检查与逐seed配对连通组bootstrap见token-metrics.json。",
              "置信区间按原始独立连通组重采样；不同seed不当作独立数据组。接触输出为最小二乘读出，0.5阈值，非校准概率。",
              "本次DINO是新的等容量token读出，不能与旧PCA＋ridge分数混为同一基线。",
              "该实验只检验冻结表征可读性，不能证明历史重建监督的因果收益，也不替代真实下游有/无先验对照。", ""]
    (job / "token-report.md").write_text("\n".join(lines), encoding="utf-8")


def fit_tokens(job_dir, device="cuda:0", batch_size=256, max_epochs=40,
               seeds=(17, 29, 43), patience=6, sleep_seconds=.02,
               bootstrap_repetitions=1000):
    """Fit matched lightweight heads from an already accepted extraction job."""
    job = Path(job_dir).resolve()
    if batch_size < 1 or max_epochs < 1 or patience < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Readout budget and unique initialization seeds must be positive")
    if (any(type(seed) is not int or seed < 0 for seed in seeds)
            or not math.isfinite(sleep_seconds) or not 0 <= sleep_seconds <= 5
            or bootstrap_repetitions < 1):
        raise ValueError("Seeds, throttle, and bootstrap count are invalid")
    plan, input_stamps = validate_extraction(job)
    # Exclusive creation prevents simultaneous callers from sharing outputs.
    with (job / "token-fit.lock").open("x") as handle:
        json.dump({"pid": os.getpid(), "format": "motionexpert-full-token-fit-lock-v1"}, handle)
    fit_plan = {"format": "motionexpert-full-token-fit-v1", "job_dir": str(job),
        "extraction_plan_sha256": hashlib.sha256((job / "plan.json").read_bytes()).hexdigest(),
        "extraction_complete_sha256": hashlib.sha256((job / "extraction-complete.json").read_bytes()).hexdigest(),
        "fit_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                              for name in ("prior_probe_fit.py", "prior_probe_math.py")},
        "input_stamps": input_stamps, "seeds": list(seeds), "device": str(device),
        "batch_size": batch_size, "max_epochs": max_epochs, "patience": patience,
        "sleep_seconds": sleep_seconds, "bootstrap_repetitions": bootstrap_repetitions,
        "trainable_parameters": PARAMETER_COUNT}
    _write_json(job / "fit-plan.json", fit_plan)
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    raw = json.loads((job / "selection.json").read_text())
    entries = raw["entries"] if isinstance(raw, dict) else raw
    if not entries or any(entry["row"] != index for index, entry in enumerate(entries)):
        raise ValueError("Selection rows must be contiguous and match extraction order")
    splits = np.asarray([entry["split"] for entry in entries])
    sources = np.asarray([entry["source"] for entry in entries])
    if set(splits) != {"train", "val"}:
        raise ValueError("Only original train and val may enter this diagnostic")
    groups = connected_group_ids(entries)
    fitting, holdout = internal_group_split(entries, groups)
    train = splits == "train"
    targets = _load_targets(job / "targets.npz")
    values, masks = encode_targets(targets)
    if len(values) != len(entries):
        raise ValueError("Target population differs from token extraction")
    hidden = np.load(job / "tokens.npy", mmap_mode="r", allow_pickle=False)
    dino = np.load(job / "dino.npy", mmap_mode="r", allow_pickle=False)
    images = np.load(job / "image-valid.npy", mmap_mode="r", allow_pickle=False)
    count = len(entries)
    if hidden.shape != (count, 11, 768) or hidden.dtype != np.float32:
        raise ValueError("Frozen hidden must be float32 N x 11 x 768")
    if dino.shape not in ((count, 30720), (count, 6, 5, 1024)) or dino.dtype != np.float32:
        raise ValueError("DINO must preserve float32 N x 6 x 5 x 1024 tokens")
    if images.shape != (count, 6) or images.dtype != np.bool_:
        raise ValueError("DINO input eligibility requires boolean image-valid.npy N x 6")
    dino = dino.reshape(count, 30, 1024)
    image_token_valid = np.repeat(np.asarray(images), 5, axis=1)
    projection_cpu = fixed_isometry()
    isometry_error = float((projection_cpu.double() @ projection_cpu.double().T - torch.eye(768, dtype=torch.float64)).abs().max())
    if isometry_error > 1e-6:
        raise ValueError("Fixed motion projection failed isometry check")
    projection = projection_cpu.to(device)
    torch.save({"projection": projection_cpu, "seed": 20261019, "max_abs_isometry_error": isometry_error},
               job / "token-fixed-isometry.pt")
    results = {"fit_plan": fit_plan, "protocol": {"format": "motionexpert-full-token-probe-v1", "checkpoint_epoch": 100,
        "trainable_parameters_per_head": PARAMETER_COUNT, "queries": 4, "common_width": 1024,
        "native_motion_width": 768, "native_dino_width": 1024, "seeds": list(seeds),
        "device": str(device), "precision": "float32", "tf32": False, "batch_size": batch_size,
        "max_inner_epochs": max_epochs, "patience": patience, "optimizer": "AdamW", "lr": .001,
        "weight_decay": .01, "gradient_clip": 1., "sleep_seconds_per_batch": sleep_seconds,
        "projection": "motion-only fixed row-orthonormal 768-to-1024 embedding; DINO identity; no information reduction",
        "motion_normalization": "shared all-eleven-token native per-channel statistics; fitting rows only per phase",
        "dino_normalization": "native per-channel fitting statistics from valid image tokens only",
        "input_mask": "DINO image-valid expanded over CLS+four registers; labels and future availability never enter head inputs",
        "head": "four learned queries, scaled dot attention, content-only values, flattened linear70; no trainable QKV/FFN",
        "key_encoding": "fixed true-5Hz-time sinusoid; DINO fixed CLS/register type sinusoid",
        "selection": "equal-source normalized masked MSE on original-train internal connected-group holdout only",
        "refit": "same matched initialization and batch permutations; full original train for selected number of epochs",
        "summary": "mean and sample std of independently trained seed metrics; not prediction ensembling",
        "bootstrap": "each matched seed separately; paired original validation connected-group bootstrap",
        "binary_readout": "least-squares uncalibrated scores, physical 0.5 threshold"},
        "population": _population(entries, groups), "internal_split": {
            "fitting_rows": np.flatnonzero(fitting).tolist(), "holdout_rows": np.flatnonzero(holdout).tolist()},
        "isometry_max_abs_error": isometry_error, "runs": {}, "summary": {}, "squared_error_summary": {},
        "paired_group_bootstrap_by_seed": {}, "gpu_reviews": {}}
    stats = {}
    for phase, selected in (("internal", fitting), ("full_train", train)):
        rows = np.flatnonzero(selected)
        stats[phase] = {"motion": _native_statistics(hidden, rows),
                        "dino": _native_statistics(dino, rows, image_token_valid),
                        "target": masked_train_statistics(values[selected], masks[selected])}
    y = torch.as_tensor(values, device=device)
    valid = torch.as_tensor(masks, device=device)
    seed_rows = {int(seed): {} for seed in seeds}
    init_digests = {int(seed): {} for seed in seeds}
    all_indices = np.arange(count, dtype=np.int64)
    for arm in ARMS:
        results["runs"][arm] = {}
        position = key_encoding(arm).to(device)
        native = dino if arm == "dino" else hidden
        token_valid_np = image_token_valid if arm == "dino" else None
        token_valid = torch.as_tensor(token_valid_np, device=device) if token_valid_np is not None else None
        selected_tokens = None if arm == "dino" else list(TOKEN_INDICES[arm])
        chosen = {}
        for phase in ("internal", "full_train"):
            feature_stats = stats[phase]["dino" if arm == "dino" else "motion"]
            features = _materialize(native, feature_stats, device, projection if arm != "dino" else None,
                                    token_valid_np, batch_size=batch_size)
            if selected_tokens is not None:
                features = features[:, selected_tokens].contiguous()
            target_stats = stats[phase]["target"]
            target_mean = torch.as_tensor(target_stats["mean"], device=device)
            target_std = torch.as_tensor(target_stats["std"], device=device)
            scaled_y = torch.where(valid, (y - target_mean) / target_std, 0.)
            for seed in seeds:
                model = _new_head(int(seed), device)
                initial_digest = _state_digest(model)
                previous = init_digests[int(seed)]
                if previous and initial_digest != next(iter(previous.values())):
                    raise ValueError("Head initialization differs across matched arms/phases")
                previous[arm + ":" + phase] = initial_digest
                def progress(record):
                    _write_json(job / "fit-progress.json", {"state": "FITTING", "arm": arm, "phase": phase,
                        "seed": int(seed), **record, "elapsed_seconds": time.monotonic() - started})
                    print(f"[token-probe] {arm} {phase} seed={seed} epoch={record['epoch']} "
                          f"mse={record['train_masked_standardized_mse']:.6f} "
                          f"holdout={record.get('internal_source_balanced_mse', 'not-used')}", flush=True)
                fitted = _train(model, features, position, token_valid, scaled_y, valid,
                    np.flatnonzero(fitting if phase == "internal" else train), seed=int(seed),
                    epochs=max_epochs if phase == "internal" else chosen[int(seed)], batch_size=batch_size,
                    job=job, sleep_seconds=sleep_seconds,
                    holdout=np.flatnonzero(holdout) if phase == "internal" else None,
                    sources=sources, patience=patience, progress=progress)
                results["gpu_reviews"][f"{arm}-{seed}-{phase}"] = fitted["gpu_review"]
                if phase == "internal":
                    chosen[int(seed)] = fitted["chosen_epoch"]
                    fitted["convergence_limited"] = fitted["chosen_epoch"] == max_epochs
                    results["runs"][arm][str(seed)] = {"internal_selection": fitted}
                else:
                    predicted = _predict(model, features, position, token_valid, all_indices,
                                         batch_size, job, sleep_seconds)
                    encoded = (predicted * target_std + target_mean).cpu().numpy()
                    decoded = decode_targets(encoded)
                    report, rows = _reports(decoded, targets, entries)
                    seed_rows[int(seed)][arm] = rows
                    run = results["runs"][arm][str(seed)]
                    run.update({"full_train_fit": fitted, "metrics": report,
                                "squared_error": _extra_metrics(encoded, values, masks, targets, entries),
                                "trainable_parameters": sum(parameter.numel() for parameter in model.parameters())})
                    np.save(job / f"token-{arm}-seed{seed}-predictions.npy", encoded)
                    feature_artifact = {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                                        for key, value in feature_stats.items()}
                    target_artifact = {key: torch.from_numpy(value.copy()) if isinstance(value, np.ndarray) else value
                                       for key, value in target_stats.items()}
                    torch.save({"arm": arm, "seed": int(seed), "head_state": model.cpu().state_dict(),
                        "trainable_parameters": PARAMETER_COUNT, "native_feature_statistics": feature_artifact,
                        "target_statistics": target_artifact, "selected_epochs": chosen[int(seed)],
                        "initial_state_sha256": initial_digest, "fixed_key_encoding": position.cpu(),
                        "motion_projection_file": "token-fixed-isometry.pt" if arm != "dino" else None},
                        job / f"token-{arm}-seed{seed}-head.pt")
                del model
            del features, scaled_y
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results["summary"][arm] = _mean_std_reports([results["runs"][arm][str(seed)]["metrics"] for seed in seeds])
        results["squared_error_summary"][arm] = _mean_std_reports([results["runs"][arm][str(seed)]["squared_error"] for seed in seeds])
        assert_extraction_unchanged(job, plan, input_stamps)
        _write_json(job / "token-metrics.partial.json", results)
    contrasts = (("all11", "history6"), ("all11", "future5"), ("history6", "future5"),
                 ("history6", "dino"), ("future5", "dino"), ("all11", "dino"))
    for seed in seeds:
        results["paired_group_bootstrap_by_seed"][str(seed)] = _bootstrap(
            seed_rows[int(seed)], entries, groups, repetitions=bootstrap_repetitions, contrasts=contrasts)
    results["initialization_sha256_by_seed_arm_phase"] = {str(seed): value for seed, value in init_digests.items()}
    results["elapsed_seconds"] = time.monotonic() - started
    results["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    assert_extraction_unchanged(job, plan, input_stamps)
    _write_json(job / "token-metrics.json", results)
    _write_report(job, results)
    _write_json(job / "fit-progress.json", {"state": "COMPLETE", "elapsed_seconds": results["elapsed_seconds"],
                "metrics": str(job / "token-metrics.json"), "report": str(job / "token-report.md")})
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 29, 43])
    parser.add_argument("--sleep-seconds", type=float, default=.02)
    args = parser.parse_args(argv)
    owned, validated = False, False
    try:
        validate_extraction(args.job)
        validated = True
        fit_tokens(args.job, args.device, args.batch_size, args.max_epochs,
                   tuple(args.seeds), args.patience, args.sleep_seconds)
    except Exception as error:
        # Never write into a foreign directory or overwrite another run's
        # status merely because CLI validation rejected its output path.
        try:
            job = args.job.resolve(strict=True)
            lock = json.loads((job / "token-fit.lock").read_text())
            plan = json.loads((job / "plan.json").read_text())
            from .probe_tokens import FORMAT
            owned = (not job.is_relative_to(Path(__file__).resolve().parents[1])
                     and plan.get("format") == FORMAT
                     and Path(plan.get("job_dir", "")).resolve() == job
                     and lock.get("pid") == os.getpid())
            if owned and validated:
                _write_json(job / "fit-progress.json", {"state": "FAILED", "error": str(error), "type": type(error).__name__})
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
