"""Capacity, temporal alignment, masking, and train-only token probe checks."""
import hashlib
import json

import numpy as np
import pytest
import torch

from MotionExpert import token_probe_fit as probe


def test_equal_capacity_heads_match_initialization_for_all_token_counts():
    torch.set_num_threads(2)
    digests = []
    for arm in probe.ARMS:
        model = probe._new_head(17, "cpu")
        assert sum(parameter.numel() for parameter in model.parameters()) == 290886
        count = len(probe.TOKEN_INDICES[arm])
        output = model(torch.randn(2, count, 1024), probe.key_encoding(arm))
        assert output.shape == (2, 70)
        digests.append(probe._state_digest(model))
    assert len(set(digests)) == 1


def test_fixed_isometry_preserves_information_and_norm():
    projection = probe.fixed_isometry(7, 11, seed=9)
    values = torch.randn(5, 7)
    transformed = values @ projection
    assert torch.allclose(transformed @ projection.T, values, atol=1e-6)
    assert torch.allclose(values.norm(dim=-1), transformed.norm(dim=-1), atol=1e-6)
    assert torch.equal(projection, probe.fixed_isometry(7, 11, seed=9))
    with pytest.raises(ValueError, match="Information-preserving"):
        probe.fixed_isometry(11, 7)


def test_future_time_indices_and_dino_register_identity_are_preserved():
    all_positions = probe.key_encoding("all11")
    assert torch.equal(probe.key_encoding("history6"), all_positions[:6])
    assert torch.equal(probe.key_encoding("future5"), all_positions[6:])
    dino = probe.key_encoding("dino").reshape(6, 5, 1024)
    assert not torch.equal(dino[0, 0], dino[0, 1])
    # Type offsets are shared across time, not a different trainable adapter.
    assert torch.allclose(dino[0, 1] - dino[0, 0], dino[5, 1] - dino[5, 0], atol=2e-7)


def test_masked_extreme_and_nan_tokens_cannot_change_attention_or_output():
    model = probe.TokenQueryReadout(width=8, queries=2, outputs=3)
    values = torch.randn(3, 4, 8)
    position = torch.randn(4, 8)
    valid = torch.tensor([[1, 0, 1, 0], [0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    expected, attention = model(values, position, valid, return_attention=True)
    corrupted = values.clone()
    corrupted[~valid] = torch.nan
    actual = model(corrupted, position, valid)
    assert torch.equal(expected, actual)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[1], model.output.bias)
    assert torch.equal(attention.masked_select(~valid[:, None].expand_as(attention)),
                       torch.zeros_like(attention.masked_select(~valid[:, None].expand_as(attention))))
    assert torch.allclose(attention[[0, 2]].sum(-1), torch.ones(2, 2))


def test_native_statistics_use_fitting_rows_and_valid_images_only():
    values = np.array([[[1., 2.], [3., 4.]], [[100., 200.], [np.nan, np.nan]],
                       [[10000., 10000.], [10000., 10000.]]], dtype=np.float32)
    valid = np.array([[1, 1], [1, 0], [1, 1]], dtype=bool)
    result = probe._native_statistics(values, np.array([0]), valid)
    np.testing.assert_array_equal(result["mean"], [2., 3.])
    np.testing.assert_array_equal(result["std"], [1., 1.])
    assert result["tokens"] == 2
    # Poisoned validation and masked tokens do not enter fitting moments.
    valid[0, 1] = False
    values[0, 1] = np.nan
    result = probe._native_statistics(values, np.array([0]), valid)
    np.testing.assert_array_equal(result["mean"], [1., 2.])
    np.testing.assert_array_equal(result["std"], [1., 1.])


def test_materialize_masks_missing_before_normalization_and_projection():
    values = np.array([[[1., 3.], [np.nan, np.nan]]], dtype=np.float32)
    valid = np.array([[True, False]])
    statistics = {"mean": np.array([1., 2.], np.float32), "std": np.array([2., 2.], np.float32)}
    projection = probe.fixed_isometry(2, 3)
    transformed = probe._materialize(values, statistics, "cpu", projection, valid)
    assert torch.isfinite(transformed).all()
    assert torch.equal(transformed[0, 1], torch.zeros(3))
    assert torch.allclose(transformed[0, 0], torch.tensor([0., .5]) @ projection)


def test_first_real_training_batch_updates_queries_and_is_nonuniform(tmp_path):
    model = probe._new_head(17, "cpu")
    feature = torch.randn(12, 6, 1024)
    target = torch.randn(12, 70)
    valid = torch.ones_like(target, dtype=torch.bool)
    fitted = probe._train(model, feature, probe.key_encoding("history6"), None,
                         target, valid, np.arange(8), seed=17, epochs=2, batch_size=4,
                         job=tmp_path, sleep_seconds=0., holdout=np.arange(8, 12),
                         sources=np.array(["a"] * 6 + ["b"] * 6), patience=6)
    review = fitted["gpu_review"]
    assert review["passed"] and review["parameter_update"]
    assert review["query_gradient_norm"] > 0
    assert review["attention_nonuniform_variance"] > 0
    assert review["masked_nan_max_abs_difference"] == 0
    assert fitted["chosen_epoch"] in (1, 2)
    assert len(fitted["curve"]) == 2


def test_invalid_target_does_not_change_loss_or_gradient():
    prediction = torch.tensor([[1., 2.]], requires_grad=True)
    valid = torch.tensor([[True, False]])
    loss = probe._masked_loss(prediction, torch.tensor([[3., 999999.]]), valid)
    loss.backward()
    assert loss.item() == 4.
    assert torch.equal(prediction.grad, torch.tensor([[-4., 0.]]))


def test_seed_summary_is_metric_average_not_prediction_ensemble():
    summary = probe._mean_std_reports([{"val": {"error": 1.}}, {"val": {"error": 3.}}])
    assert summary["val"]["error"]["mean"] == 2.
    assert summary["val"]["error"]["std"] == pytest.approx(np.sqrt(2.))
    assert summary["val"]["error"]["seeds"] == 2


def accepted_job(tmp_path, monkeypatch):
    from MotionExpert import probe_tokens
    from MotionExpert.probe_prior import make_shards
    base, job = tmp_path / "base", tmp_path / "tokens"
    base.mkdir()
    job.mkdir()
    rng = np.random.default_rng(23)
    entries = []
    for index in range(16):
        split = "train" if index < 12 else "val"
        source = "a" if index % 2 else "b"
        entries.append({"row": index, "episode_index": index, "index": index,
                        "split": split, "source": source,
                        "split_group_ids": [f"{split}-{source}-{index}"]})
    targets = {"continuous": rng.normal(0, .01, (16, 45)).astype(np.float32),
               "continuous_valid": np.ones((16, 45), bool),
               "current_contact": np.zeros((16, 2), np.float32),
               "current_contact_valid": np.ones((16, 2), bool)}
    for name in ("contact", "transition"):
        targets[name] = rng.integers(0, 2, (16, 10)).astype(np.float32)
        targets[name + "_valid"] = np.ones((16, 10), bool)
    np.savez(job / "targets.npz", **targets)
    np.save(job / "tokens.npy", rng.normal(size=(16, 11, 768)).astype(np.float32))
    np.save(job / "dino.npy", rng.normal(size=(16, 30720)).astype(np.float32))
    np.save(job / "image-valid.npy", np.ones((16, 6), bool))
    (job / "selection.json").write_text(json.dumps(entries))
    plan = {"format": probe_tokens.FORMAT, "job_dir": str(job), "epoch": 100,
            "precision": "float32", "weights": "ema", "windows": 16,
            "token_shape": [11, 768], "logical_shards": list(range(8)),
            "reuse_job": {"job_dir": str(base)}, "label_specs": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)} for name, value in targets.items()}}
    (job / "plan.json").write_text(json.dumps(plan))
    completed = {"passed": True, "epoch": 100, "windows": 16, "token_shape": [16, 11, 768],
                 "historical_image_valid_shape": [16, 6], "all_tokens_finite": True,
                 "expert_gradients": False, "expert_trainable_parameters": 0, "logical_shards": 8}
    (job / "extraction-complete.json").write_text(json.dumps(completed))
    for index, shard in enumerate(make_shards(entries, list(range(8)))):
        done = {"passed": True, "epoch": 100, "logical_shard": index, "windows": len(shard),
                "hidden_shape": [len(shard), 11, 768], "expert_gradients": False,
                "expert_trainable_parameters": 0, "acceptance": {"passed": True},
                "rows_sha256": hashlib.sha256(np.asarray([entry["row"] for entry in shard], dtype=np.int64).tobytes()).hexdigest(),
                "noise_sha256": str(index), "dino_sha256": str(index)}
        (job / f"done-shard{index}.json").write_text(json.dumps(done))
        (base / f"done-100-gpu{index}.json").write_text(json.dumps(done))
    monkeypatch.setattr(probe_tokens, "assert_bindings", lambda plan: None)
    return job


def test_validation_refuses_incomplete_extraction_without_writes(tmp_path, monkeypatch):
    job = accepted_job(tmp_path, monkeypatch)
    completed = json.loads((job / "extraction-complete.json").read_text())
    completed["passed"] = False
    (job / "extraction-complete.json").write_text(json.dumps(completed))
    before = set(job.iterdir())
    with pytest.raises(ValueError, match="complete GPU acceptance"):
        probe.fit_tokens(job, device="cpu", seeds=(17,), max_epochs=1)
    assert set(job.iterdir()) == before


def test_foreign_job_failure_never_writes_status(tmp_path, monkeypatch):
    job = accepted_job(tmp_path, monkeypatch)
    plan = json.loads((job / "plan.json").read_text())
    plan["job_dir"] = str(tmp_path / "another-job")
    (job / "plan.json").write_text(json.dumps(plan))
    before = {path.name: path.read_bytes() for path in job.iterdir()}
    with pytest.raises(ValueError, match="own epoch100"):
        probe.main(["--job", str(job), "--device", "cpu"])
    assert before == {path.name: path.read_bytes() for path in job.iterdir()}


def test_validation_refuses_existing_artifacts_and_changed_inputs(tmp_path, monkeypatch):
    job = accepted_job(tmp_path, monkeypatch)
    plan, stamps = probe.validate_extraction(job)
    probe.assert_extraction_unchanged(job, plan, stamps)
    (job / "tokens.npy").touch()
    with pytest.raises(ValueError, match="changed during fitting"):
        probe.assert_extraction_unchanged(job, plan, stamps)
    (job / "token-metrics.json").write_text('{"old":true}')
    with pytest.raises(FileExistsError, match="overwrite"):
        probe.validate_extraction(job)
    assert (job / "token-metrics.json").read_text() == '{"old":true}'


def test_small_complete_fit_matches_capacity_and_loads_safe_heads(tmp_path, monkeypatch):
    job = accepted_job(tmp_path, monkeypatch)
    result = probe.fit_tokens(job, "cpu", batch_size=4, max_epochs=1, seeds=(17,),
                              sleep_seconds=0, bootstrap_repetitions=3)
    assert set(result["runs"]) == set(probe.ARMS)
    assert result["protocol"]["trainable_parameters_per_head"] == 290886
    digests = result["initialization_sha256_by_seed_arm_phase"]["17"]
    assert len(digests) == 8 and len(set(digests.values())) == 1
    for arm in probe.ARMS:
        artifact = torch.load(job / f"token-{arm}-seed17-head.pt", weights_only=True)
        assert artifact["trainable_parameters"] == 290886
        assert result["runs"][arm]["17"]["internal_selection"]["convergence_limited"]
        assert np.load(job / f"token-{arm}-seed17-predictions.npy").shape == (16, 70)
    before = (job / "token-metrics.json").read_bytes()
    with pytest.raises(FileExistsError):
        probe.main(["--job", str(job), "--device", "cpu"])
    assert (job / "token-metrics.json").read_bytes() == before
    assert json.loads((job / "fit-progress.json").read_text())["state"] == "COMPLETE"
