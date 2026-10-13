"""5Hz joint motion windows and the explicit future-only comparison profile."""
from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from MotionExpert.compute_joint_motion_stats import compute_stats, _accumulate
from MotionExpert.dataset.dense_motion_dataset import DenseMotionDataset
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.dataset.dense_representation import quaternion_matrix
from MotionExpert.merge_joint_motion_stats import _load
from MotionExpert.mydiffusion.flow_matching import FlowMatching
from MotionExpert.tests.test_dense_dataset import synthetic_arrays, synthetic_metadata, write_archive
from MotionExpert.tests.test_dense_statistics_shards import make_partials, run_merge
from MotionExpert.tests.test_dense_v5_dataset import v5_arrays, rewrite_v5


def forecast_dataset(root, manifest, **kwargs):
    options = dict(archive_root=root, split_manifest=manifest, do_normalization=False,
                   fps=5, window=11, forecast_prefix=6, segment_stride=5,
                   train_task="fore", require_full_window=True,
                   motion_supervision="future_only",
                   rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1")
    options.update(kwargs)
    return DenseMotionDataset(**options)


@pytest.mark.parametrize("v5", [False, True])
def test_real_two_second_window_reads_every_sixth_frame_and_recomputes_motion(tmp_path, v5):
    root = tmp_path/"archive"; root.mkdir()
    original = synthetic_arrays(61)
    manifest, records = write_archive(root, original)
    if v5:
        rewrite_v5(root, v5_arrays(original), records, manifest)
    dataset = forecast_dataset(root, manifest)
    assert len(dataset) == 1
    sample = dataset[0]
    indices = list(range(0, 61, 6))
    assert json.loads(sample["misc"]["metadata_json"])["training_window"]["source_frame_indices"] == indices
    assert sample["misc"]["num_real_frames"] == 11
    assert sample["misc"]["motion"].shape == (11, 245)
    assert sample["y"]["img_embs"].shape == (11, 5, 1024)
    np.testing.assert_array_equal(sample["y"]["img_embs"][:, 0, 0], indices[:6]+[0]*5)
    assert sample["y"]["valid_img_embs"].tolist() == [True]*6+[False]*5
    assert sample["y"]["loss_mask"].tolist() == [False]*6+[True]*5
    assert sample["y"]["img_mask"].tolist() == [False]*6+[True]*5
    assert sample["misc"]["feature_valid_mask"][1:, 243:].all()
    expected = quaternion_matrix(np.array([[np.cos(.18/2), 0, 0, np.sin(.18/2)]]))[0]
    np.testing.assert_allclose(sample["misc"]["motion"][1, 108:114], expected[:2].reshape(-1), atol=1e-6)
    assert dataset.rejected_windows["incomplete_window"] == 1
    with pytest.raises(ValueError, match="divisible by 6"):
        dataset.get_from_seq_and_st(dataset.seq_names[0], 2)
    with pytest.raises(ValueError, match="complete window"):
        dataset.get_from_seq_and_st(dataset.seq_names[0], 6)


def test_index_checks_only_observed_images_and_motion_stats_never_read_embeddings(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    arrays = synthetic_arrays(61)
    arrays["observation.valid.dino"][30:] = False
    arrays["observation.features.dino"][36:] = np.nan
    manifest, _ = write_archive(root, arrays)
    dataset = forecast_dataset(root, manifest, min_valid_image_fraction=.8)
    assert len(dataset) == 1  #5/6 observed images;5/11 must not be used.
    motion_only = forecast_dataset(root, manifest, cond_img_feat=False, min_valid_image_fraction=.8)
    stats = compute_stats(motion_only)
    assert stats["min_valid_image_fraction"] == .8
    assert all("observation.features.dino" not in columns for _, _, columns in motion_only.reader.read_log)
    with pytest.raises(ValueError, match="No usable"):
        forecast_dataset(root, manifest, min_valid_image_fraction=.9)


def test_fore_floor_is_fixed_for_aux_tasks_and_statistics_use_only_future_targets(tmp_path):
    root = tmp_path/"archive"; root.mkdir()
    arrays = synthetic_arrays(61)
    arrays["observation.state.body_world"].reshape(-1, 22, 3)[36:, [10, 11], 2] += 9
    metadata = synthetic_metadata(); metadata["ground"]["valid"] = False
    manifest, _ = write_archive(root, arrays, metadata)
    dataset = forecast_dataset(root, manifest, cond_img_feat=False)
    fore = dataset[0]
    recon = dataset.get_sample(0, task="recon")
    assert fore["misc"]["floor_height"] == pytest.approx(-.7, abs=1e-6)
    assert recon["misc"]["floor_height"] == fore["misc"]["floor_height"]
    torch.testing.assert_close(recon["misc"]["motion"], fore["misc"]["motion"])
    assert json.loads(recon["misc"]["metadata_json"])["ground"]["estimation_scope"] == "observed_prefix"
    stats = compute_stats(dataset)
    assert stats["stats_schema_version"] == 4
    assert stats["statistics_scope"] == "future_only"
    assert stats["source_valid_frames"] == {"egosuite": 5}
    assert stats["motion_count"].eq(5).all()
    assert stats["traj_count"].eq(5).all()
    torch.testing.assert_close(stats["motion_mean"], fore["misc"]["motion"][6:].mean(0))
    assert stats["traj_mean"][8] == pytest.approx(1.8, abs=1e-6)
    MotionNormalizer(stats)


@pytest.mark.parametrize("case", ["short", "prefix_feet_missing", "future_body_missing", "future_camera_missing"])
def test_unusable_windows_are_filtered_before_training(tmp_path, case):
    root = tmp_path/"archive"; root.mkdir()
    arrays = synthetic_arrays(60 if case == "short" else 61)
    metadata = synthetic_metadata()
    if case == "prefix_feet_missing":
        metadata["ground"]["valid"] = False
        arrays["observation.valid.body_position"][:36, [10, 11]] = False
    elif case == "future_body_missing":
        arrays["observation.valid.body_position"][36:] = False
        arrays["observation.valid.body_rotation"][36:] = False
    elif case == "future_camera_missing":
        arrays["observation.valid.head_reference_rotation"][36:] = False
    manifest, _ = write_archive(root, arrays, metadata)
    with pytest.raises(ValueError, match="No usable"):
        forecast_dataset(root, manifest, cond_img_feat=False)


@pytest.mark.parametrize("field,value", [
    ("sampling_step", 2), ("label_task", "recon"), ("train_task", "recon"),
    ("ground_estimation_scope", "full_window"), ("statistics_scope", "all_real_frames"),
    ("complete_windows_only", False), ("image_validity_scope", "full_window"),
    ("min_valid_image_fraction", float("nan")), ("stats_schema_version", 3),
    ("motion_supervision", "joint"), ("motion_supervision", True),
])
def test_normalizer_rejects_mislabeled_forecast_statistics(tmp_path, field, value):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(61))
    dataset = forecast_dataset(root, manifest, cond_img_feat=False)
    stats = compute_stats(dataset)
    stats[field] = value
    with pytest.raises(ValueError):
        MotionNormalizer(stats)


def test_new_forecast_shards_merge_exactly_and_cannot_mix_profiles(tmp_path, monkeypatch):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(91))
    dataset = forecast_dataset(root, manifest, cond_img_feat=False)
    expected = compute_stats(dataset)
    partials = make_partials(dataset, 3, "contiguous")
    actual = run_merge(tmp_path, monkeypatch, partials)
    assert actual["statistics_scope"] == "future_only"
    assert actual["sampling_step"] == 6
    for key in ("motion_mean", "motion_std", "traj_mean", "traj_std", "motion_count", "traj_count"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-6, rtol=0)
    changed = deepcopy(partials)
    changed[1]["min_valid_image_fraction"] = .9
    with pytest.raises(ValueError, match="provenance differs"):
        run_merge(tmp_path, monkeypatch, changed)


def test_legacy15hz_statistics_remain_readable_but_not_as5hz_assets(tmp_path, monkeypatch):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(17))
    legacy = DenseMotionDataset(archive_root=root, split_manifest=manifest,
        window=4, forecast_prefix=2, segment_stride=3, cond_img_feat=False,
        do_normalization=False, rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1",
        train_task=None)
    assert legacy.train_task == "recon"
    assert not legacy[0]["y"].get("task_conditioning", False)
    raw = _accumulate(legacy, range(len(legacy)), progress=False)
    raw["stats_schema_version"] = 3
    for key in ("sampling_step", "train_task", "label_task", "complete_windows_only", "ground_estimation_scope",
                "image_validity_scope", "min_valid_image_fraction", "statistics_scope"):
        raw.pop(key)
    raw.update(kind="joint_masked_partial", num_shards=1, shard_index=0)
    path = tmp_path/"legacy.pt"; torch.save(raw, path)
    assert _load(path)["fps"] == 15
    merged = run_merge(tmp_path, monkeypatch, [raw])
    assert merged["stats_schema_version"] == 3
    assert "label_task" not in merged
    MotionNormalizer(merged)
    stats_path = tmp_path/"merged.pt"
    DenseMotionDataset(archive_root=root, split_manifest=manifest, stats_path=stats_path,
        window=4, forecast_prefix=2, segment_stride=3, cond_img_feat=False,
        rotation_basis_mode="source_native", target_joint_basis="egosuite_native_v1")
    stats = compute_stats(legacy)
    stats["stats_schema_version"] = 3
    MotionNormalizer(stats)
    stats["fps"] = 5
    with pytest.raises(ValueError, match="schema4"):
        MotionNormalizer(stats)


@pytest.mark.parametrize("kwargs", [{"fps": 7}, {"fps": True}, {"train_task": "gen"},
    {"require_full_window": 1}, {"train_task": "recon"}, {"require_full_window": False},
    {"motion_supervision": "unsupported"}])
def test_sampling_protocol_rejects_invalid_options(tmp_path, kwargs):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(61))
    with pytest.raises(ValueError):
        forecast_dataset(root, manifest, **kwargs)


@pytest.mark.parametrize("legacy", [False, True])
def test_real_sampled_forecast_decode_ignores_gaussian_history_and_uses_predictions(tmp_path, legacy):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(61))
    overrides = dict(fps=15, train_task=None, require_full_window=False) if legacy else {}
    dataset = forecast_dataset(root, manifest, cond_img_feat=False, **overrides)
    ret = dataset.get_sample(0, task="fore")
    ground_truth = dataset.ret_to_full_sequence(ret)
    assert ground_truth["body_positions"].shape == (11, 12, 3)

    class FutureModel(torch.nn.Module):
        def __init__(self, target):
            super().__init__()
            self.register_buffer("prediction", target.clone().unsqueeze(0))
            # Give the prediction a real future path distinct from the GT.
            self.prediction[:, 6:, 114] += .05

        def forward(self, state, timestep, y):
            return self.prediction

    model = FutureModel(ret["misc"]["motion"])
    generation_mask = (torch.arange(11) >= 6).unsqueeze(0)
    first_noise = torch.randn((1, 11, 245), generator=torch.Generator().manual_seed(7))
    second_noise = first_noise.clone()
    second_noise[:, :6] = torch.randn((1, 6, 245), generator=torch.Generator().manual_seed(99))*100
    predictions = []
    for noise in (first_noise, second_noise):
        prediction = FlowMatching().sample_loop(model, noise.shape, noise=noise,
            model_kwargs={"y": {"motion_generation_mask": generation_mask}}, num_steps=2)
        torch.testing.assert_close(prediction[:, :6], noise[:, :6], atol=0, rtol=0)
        sampled_ret = dict(ret, pred={"motion": prediction[0]})
        predictions.append(dataset.ret_to_full_sequence(sampled_ret))
    assert predictions[0]["body_positions"].shape == (5, 12, 3)
    for key in predictions[0]:
        torch.testing.assert_close(predictions[0][key], predictions[1][key], atol=0, rtol=0)
    assert (predictions[0]["floor_positions"]-ground_truth["floor_positions"][6:]).norm(dim=-1).max() > .1

    # Changing predicted history did not alter the source-label reconstruction.
    torch.testing.assert_close(dataset.ret_to_full_sequence(ret)["body_positions"], ground_truth["body_positions"], atol=0, rtol=0)


@pytest.mark.parametrize("v5", [False, True])
def test_joint_default_supervises_all11_and_statistics_include_valid_history(tmp_path, v5):
    root = tmp_path/"archive"; root.mkdir()
    arrays = synthetic_arrays(61)
    arrays["observation.valid.body_rotation"][6, 5] = False
    arrays["observation.features.dino"][36:] = np.nan
    arrays["observation.state.body_world"].reshape(-1, 22, 3)[36:, [10, 11], 2] += 9
    metadata = synthetic_metadata(); metadata["ground"]["valid"] = False
    manifest, records = write_archive(root, arrays, metadata)
    if v5:
        rewrite_v5(root, v5_arrays(arrays), records, manifest)
    dataset = forecast_dataset(root, manifest, motion_supervision=None)
    assert dataset.motion_supervision == "joint"
    assert dataset.statistics_scope == "all_valid_frames"
    sample = dataset[0]
    assert sample["y"]["loss_mask"].bool().all()
    assert "motion_generation_mask" not in sample["y"]
    assert sample["y"]["img_mask"].tolist() == [0]*6+[1]*5
    assert sample["y"]["valid_img_embs"].tolist() == [True]*6+[False]*5
    assert sample["misc"]["floor_height"] == pytest.approx(-.7, abs=1e-6)

    motion_only = forecast_dataset(root, manifest, motion_supervision="joint", cond_img_feat=False)
    labels = motion_only[0]
    stats = compute_stats(motion_only)
    expected_count = labels["misc"]["feature_valid_mask"].sum(0)
    assert stats["motion_supervision"] == "joint"
    assert stats["statistics_scope"] == "all_valid_frames"
    assert stats["source_valid_frames"] == {"egosuite": 11}
    torch.testing.assert_close(stats["motion_count"], expected_count)
    mask = labels["misc"]["feature_valid_mask"]
    expected_mean = labels["misc"]["motion"].masked_fill(~mask, 0).sum(0)/expected_count.clamp_min(1)
    torch.testing.assert_close(stats["motion_mean"], expected_mean, atol=1e-6, rtol=1e-6)
    assert stats["traj_count"].eq(11).all()
    assert all("observation.features.dino" not in columns for _, _, columns in motion_only.reader.read_log)
    MotionNormalizer(stats)

    comparison = forecast_dataset(root, manifest, motion_supervision="future_only", cond_img_feat=False)
    future = compute_stats(comparison)
    torch.testing.assert_close(labels["misc"]["motion"], comparison[0]["misc"]["motion"])
    assert dataset.idx_to_sidx_fidx == comparison.idx_to_sidx_fidx
    assert future["source_valid_frames"] == {"egosuite": 5}
    assert not torch.allclose(stats["motion_mean"][108:117], future["motion_mean"][108:117])


def test_joint_shards_cannot_mix_with_future_only_and_legacy_future_metadata_is_readable(tmp_path, monkeypatch):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(91))
    joint = forecast_dataset(root, manifest, motion_supervision="joint", cond_img_feat=False)
    future = forecast_dataset(root, manifest, motion_supervision="future_only", cond_img_feat=False)
    joint_partials = make_partials(joint, 3, "contiguous")
    future_partials = make_partials(future, 3, "contiguous")
    mixed = deepcopy(joint_partials); mixed[1] = future_partials[1]
    with pytest.raises(ValueError, match="provenance differs"):
        run_merge(tmp_path, monkeypatch, mixed)
    merged = run_merge(tmp_path, monkeypatch, joint_partials)
    assert merged["motion_supervision"] == "joint"
    assert merged["statistics_scope"] == "all_valid_frames"
    MotionNormalizer(merged)
    for raw in future_partials:
        raw.pop("motion_supervision")
    merged_future = run_merge(tmp_path, monkeypatch, future_partials)
    assert merged_future["motion_supervision"] == "future_only"
    assert merged_future["statistics_scope"] == "future_only"
    MotionNormalizer(merged_future)

    old_stats = compute_stats(future)
    old_stats.pop("motion_supervision")
    path = tmp_path/"old_future_stats.pt"; torch.save(old_stats, path)
    forecast_dataset(root, manifest, cond_img_feat=False, do_normalization=True, stats_path=path,
                     motion_supervision="future_only")
    with pytest.raises(ValueError, match="statistics .* mismatch"):
        forecast_dataset(root, manifest, cond_img_feat=False, do_normalization=True, stats_path=path,
                         motion_supervision="joint")


@pytest.mark.parametrize("field,value", [
    ("motion_supervision", "future_only"), ("motion_supervision", None),
    ("statistics_scope", "future_only"),
])
def test_joint_statistics_cannot_be_reinterpreted_as_old_future_profile(tmp_path, field, value):
    root = tmp_path/"archive"; root.mkdir()
    manifest, _ = write_archive(root, synthetic_arrays(61))
    joint = forecast_dataset(root, manifest, motion_supervision="joint", cond_img_feat=False)
    stats = compute_stats(joint)
    stats[field] = value
    with pytest.raises(ValueError):
        MotionNormalizer(stats)
