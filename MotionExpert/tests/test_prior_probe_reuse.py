"""Incremental frozen-prior checks: fixed inputs and immutable old readouts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from MotionExpert import prior_probe_fit as fitting
from MotionExpert.probe_prior import assert_reuse_current, file_stamp, prepare_reuse


def write_json(path, value):
    path.write_text(json.dumps(value))


def baseline_job(tmp_path):
    torch.set_num_threads(2)
    base = tmp_path / "baseline"
    base.mkdir()
    records, indices, entries = {"train": [], "val": []}, {}, []
    for split, groups in (("train", 4), ("val", 2)):
        for source in ("egosuite", "egoexo4d_hm"):
            for group in range(groups):
                episode = len(entries)
                record = {"source": source, "episode_index": episode,
                          "source_key": str(episode), "split_group_ids": [f"{source}:{split}:{group}"]}
                entries.append({"row": episode, "split": split, "index": len(records[split]),
                                "start_frame": 0, **record})
                records[split].append(record)
        indices[split] = np.stack((np.arange(len(records[split])), np.zeros(len(records[split]), dtype=int)), 1)
    count = len(entries)
    rng = np.random.default_rng(17)
    target = {"continuous": rng.normal(0, .01, (count, 45)).astype(np.float32),
              "continuous_valid": np.ones((count, 45), bool),
              "contact": rng.integers(0, 2, (count, 10)).astype(np.float32),
              "contact_valid": np.ones((count, 10), bool),
              "transition": rng.integers(0, 2, (count, 10)).astype(np.float32),
              "transition_valid": np.ones((count, 10), bool),
              "current_contact": np.zeros((count, 2), np.float32),
              "current_contact_valid": np.ones((count, 2), bool)}
    np.savez(base / "targets.npz", **target)
    values, masks = fitting.encode_targets(target)
    train = np.asarray([entry["split"] == "train" for entry in entries])
    y_mean, y_std, y_count = fitting._statistics(torch.tensor(values[train]), torch.tensor(masks[train]))
    prediction = fitting.decode_targets(np.broadcast_to(y_mean.numpy(), values.shape))
    report, rows = fitting._reports(prediction, target, entries)
    groups = fitting.connected_group_ids(entries)
    inner, holdout = fitting.internal_group_split(entries, groups)
    noise_rows = np.flatnonzero(~train).tolist()
    for epoch in (20, 100):
        array = rng.normal(size=(count, 768)).astype(np.float32)
        np.save(base / f"epoch{epoch}.npy", array)
        mean, std, _ = fitting._statistics(torch.tensor(array[train]))
        torch.save({"model": f"epoch{epoch}", "feature_mean": mean, "feature_std": std,
                    "weights": torch.zeros(769, 70, dtype=torch.float64),
                    "target_mean": y_mean, "target_std": y_std, "target_count": y_count,
                    "alpha": .1}, base / f"epoch{epoch}-linear-probe.pt")
        np.save(base / f"noise-epoch{epoch}.npy", np.zeros((len(noise_rows), 3, 768), np.float32))
        write_json(base / f"done-{epoch}-gpu0.json", {"passed": True})
    np.save(base / "dino.npy", np.zeros((count, 30720), np.float32))
    torch.save({}, base / "dino-linear-probe.pt")
    checkpoints = {}
    for epoch in (20, 100):
        path = tmp_path / f"epoch{epoch}.ckpt"
        path.write_bytes(b"frozen")
        checkpoints[str(epoch)] = str(path)
    repo = Path(__file__).resolve().parents[2]
    plan = {"format": "motionexpert-frozen-prior-probe-v1", "windows": count,
            "seed": 620017, "gpus": [0], "batch_size": 128, "precision": "float32", "weights": "ema",
            "checkpoints": checkpoints, "checkpoint_stamps": {epoch: file_stamp(path) for epoch, path in checkpoints.items()},
            "cache_root": str(tmp_path / "cache"), "vision_root": str(tmp_path / "vision"),
            "cache_manifest_sha256": "manifest", "archive_fingerprint": "archive", "split_fingerprint": "split",
            "noise_rows": noise_rows, "noise_draws": 3,
            "label_specs": {name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in target.items()},
            "source_files": {name: hashlib.sha256((repo / name).read_bytes()).hexdigest() for name in
                             ("MotionExpert/dataset/probe_cache.py", "MotionExpert/prior_probe_math.py")}}
    write_json(base / "plan.json", plan)
    write_json(base / "selection.json", entries)
    write_json(base / "review.json", {"passed": True})
    write_json(base / "extraction-complete.json", {"passed": True})
    population = {split: {source: {
        "windows": sum(entry["split"] == split and entry["source"] == source for entry in entries),
        "episodes": sum(entry["split"] == split and entry["source"] == source for entry in entries),
        "connected_groups": sum(entry["split"] == split and entry["source"] == source for entry in entries)}
        for source in ("egoexo4d_hm", "egosuite")} for split in ("train", "val")}
    persistence = {"continuous": np.zeros_like(target["continuous"]),
                   "contact": np.tile(target["current_contact"], (1, 5)), "transition": np.zeros_like(target["transition"])}
    persistent_report, _ = fitting._reports(persistence, target, entries)
    metrics = {"protocol": {"ridge_alphas": list(fitting.ALPHAS),
               "objective": "mean masked standardized MSE + alpha squared weights; intercept unpenalized",
               "alpha_selection": "original train internal connected-group holdout; equal source weighting"},
               "population": population, "internal_split": {"fitting_rows": np.flatnonzero(inner).tolist(),
                                                            "holdout_rows": np.flatnonzero(holdout).tolist()},
               "models": {name: report for name in ("epoch20", "epoch100", "dino")},
               "hyperparameters": {name: {"alpha": .1} for name in ("epoch20", "epoch100", "dino")},
               "noise_stability": {}, "pca": {"dimensions": 768, "explained_variance_fraction": .8},
               "persistent_contact_common_mask_comparison": {name: report for name in ("epoch20", "epoch100", "dino")},
               "paired_group_bootstrap": fitting._bootstrap({name: rows for name in ("epoch20", "epoch100", "dino")}, entries, groups)}
    metrics["models"]["zero_motion_persistent_contact"] = persistent_report
    write_json(base / "metrics.json", metrics)
    cache = SimpleNamespace(root=Path(plan["cache_root"]), vision_root=Path(plan["vision_root"]),
                            manifest_sha256="manifest", binding={"archive_fingerprint": "archive", "split_fingerprint": "split"},
                            records=records, indices=lambda split: indices[split])
    checkpoint160 = tmp_path / "epoch160.ckpt"
    checkpoint160.write_bytes(b"new frozen weights")
    job = tmp_path / "increment"
    job.mkdir()
    return base, cache, checkpoint160, job, metrics


def test_reuse_inherits_rows_targets_and_artifacts_without_touching_baseline(tmp_path):
    base, cache, checkpoint, job, _ = baseline_job(tmp_path)
    before = {path.name: file_stamp(path) for path in base.iterdir()}
    plan, entries = prepare_reuse(base, cache, checkpoint, job, [0], 4)
    assert plan["extract_epochs"] == [160] and plan["fit_models"] == ["epoch160"]
    assert plan["seed"] == 620017 and len(entries) == plan["windows"]
    assert (job / "targets.npz").resolve() == base / "targets.npz"
    assert not (job / "plan.json").exists()
    assert before == {path.name: file_stamp(path) for path in base.iterdir()}
    assert_reuse_current(plan)
    (base / "epoch20.npy").touch()
    with pytest.raises(ValueError, match="artifact changed"):
        assert_reuse_current(plan)


@pytest.mark.parametrize("mismatch", ("cache", "selection", "acceptance", "layout"))
def test_reuse_rejects_mismatched_identity_before_linking(tmp_path, mismatch):
    base, cache, checkpoint, job, _ = baseline_job(tmp_path)
    gpus = [0]
    if mismatch == "cache":
        cache.manifest_sha256 = "different"
    elif mismatch == "selection":
        entries = json.loads((base / "selection.json").read_text())
        entries[0]["episode_index"] = 999
        write_json(base / "selection.json", entries)
    elif mismatch == "acceptance":
        write_json(base / "review.json", {"passed": False})
    else:
        gpus = [1]
    with pytest.raises(ValueError):
        prepare_reuse(base, cache, checkpoint, job, gpus, 4)
    assert not list(job.iterdir())


def test_increment_fits_only160_and_preserves_old_baselines_exactly(tmp_path, monkeypatch):
    base, cache, checkpoint, job, baseline = baseline_job(tmp_path)
    plan, entries = prepare_reuse(base, cache, checkpoint, job, [0], 4)
    write_json(job / "plan.json", plan)
    np.save(job / "epoch160.npy", np.random.default_rng(9).normal(size=(len(entries), 768)).astype(np.float32))
    np.save(job / "noise-epoch160.npy", np.zeros((len(plan["noise_rows"]), 3, 768), np.float32))
    before = {path.name: file_stamp(path) for path in base.iterdir()}
    calls = []

    def counted_ridge(x, y, valid, alphas):
        calls.append(1)
        # Numerical ridge behavior is covered separately on a small matrix;
        # this integration check verifies which heads are allowed to be fit.
        return torch.zeros(len(alphas), x.shape[1] + 1, y.shape[1], dtype=torch.float64, device=x.device)

    monkeypatch.setattr(fitting, "_ridge_weights", counted_ridge)
    monkeypatch.setattr(torch, "pca_lowrank", lambda *args, **kwargs: pytest.fail("Reused DINO PCA must not be fitted"))
    output = fitting.fit_job(job, device="cpu")
    assert len(calls) == 2  # New160 internal selection and final refit only.
    assert before == {path.name: file_stamp(path) for path in base.iterdir()}
    for name, report in baseline["models"].items():
        assert output["models"][name] == report
    assert output["pca"] == baseline["pca"]
    assert output["reused_baseline"]["readouts_refitted"] is False
    assert (job / "epoch160-linear-probe.pt").is_file()
    for source, report in output["paired_group_bootstrap"].items():
        assert report["contrasts"].keys() >= {"epoch160_minus_epoch20", "epoch160_minus_epoch100"}
        for name, contrast in baseline["paired_group_bootstrap"][source]["contrasts"].items():
            assert report["contrasts"][name] == contrast


def test_bootstrap_accepts_explicit_incremental_pairs_and_rejects_missing_model():
    rows = {"wrist_displacement_cm_sum": np.ones(4), "wrist_displacement_cm_count": np.ones(4)}
    entries = [{"source": "egosuite", "split": "val"}] * 4
    # A missing comparator must fail before reading any metric arrays.
    with pytest.raises(ValueError, match="missing model"):
        fitting._bootstrap({"epoch160": rows}, entries, np.arange(4), contrasts=(("epoch160", "epoch100"),))


def test_replayed_observation_counts_are_exact_even_for_large_populations():
    with pytest.raises(ValueError, match="observation count"):
        fitting._check_replayed_metrics({"count": 1_000_001}, {"count": 1_000_000})
    assert fitting._check_replayed_metrics({"error": .1 + 1e-8}, {"error": .1}) < 1e-7
