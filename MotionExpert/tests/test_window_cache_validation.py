"""Regression checks for cache acceptance and diagnostic validation boundaries."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.tests.fixtures import owned_archive
from MotionExpert.tests.test_window_cache_dataset import cached_archive
from MotionExpert.validate_window_cache import (
    OriginalWindowPath, compare_sample, dataset_from_config, gpu_equivalence, representative_indices,
    validate_config_change, validate_selected_indices, wait_for_cache,
)


def test_cache_transition_config_rejects_training_or_visual_changes():
    source = smplh_motion_expert_config()
    cached = deepcopy(source)
    cached.DATA.WINDOW_CACHE_ROOT = "/external/window-cache"
    validate_config_change(source, cached)
    for section, name, value in (("TRAIN", "LR", 2e-4), ("DATA", "BATCH_SIZE", 512),
                                 ("DATA", "IMAGE_PROFILE", "256x320")):
        changed = deepcopy(cached)
        setattr(getattr(changed, section), name, value)
        with pytest.raises(ValueError, match="only change"):
            validate_config_change(source, changed)


def test_validation_compares_new_cached_values_and_decoding_anchors(cached_archive):
    cfg = cached_archive
    kwargs = dict(archive_root=cfg.DATA.ARCHIVE_ROOT, do_normalization=False,
                  vision_profile_root=cfg.DATA.VISION_PROFILE_ROOT,
                  min_valid_image_fraction=cfg.DATA.MIN_VALID_IMAGE_FRACTION)
    original = SMPLHMotionDataset(**kwargs)
    cached = SMPLHMotionDataset(**kwargs, window_cache_root=cfg.DATA.WINDOW_CACHE_ROOT)
    expected, actual = original[0], cached[0]
    compare_sample(actual, expected)
    validate_selected_indices(cached, {0})
    compare_sample(cached[0], OriginalWindowPath(cached)[0])
    frozen = list(cached.idx_to_sidx_fidx)
    cached.idx_to_sidx_fidx = frozen[:-1]
    with pytest.raises(AssertionError, match="window index"):
        validate_selected_indices(cached, {item[0] for item in frozen})
    cached.idx_to_sidx_fidx = frozen
    changed = deepcopy(actual)
    changed["misc"]["motion"][0, 0] += .1
    with pytest.raises(AssertionError):
        compare_sample(changed, expected)
    changed = deepcopy(actual)
    changed["misc"]["floor_transform"][0, 0, 3] += .01
    with pytest.raises(AssertionError):
        compare_sample(changed, expected)


def test_future_visuals_cannot_pass_acceptance_even_if_both_paths_match(cached_archive):
    cfg = cached_archive
    data = SMPLHMotionDataset(archive_root=cfg.DATA.ARCHIVE_ROOT, do_normalization=False,
                             vision_profile_root=cfg.DATA.VISION_PROFILE_ROOT,
                             min_valid_image_fraction=cfg.DATA.MIN_VALID_IMAGE_FRACTION,
                             window_cache_root=cfg.DATA.WINDOW_CACHE_ROOT)
    value = data[0]
    value["y"]["img_embs"][6:] = 1
    with pytest.raises(AssertionError, match="Future images"):
        compare_sample(value, deepcopy(value))


def test_dead_builder_and_failed_manifest_cannot_wait_forever(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"state": "building"}))
    with pytest.raises(RuntimeError, match="exited"):
        wait_for_cache(tmp_path, builder_pid=999999999)
    path.write_text(json.dumps({"state": "failed", "error": "disk full"}))
    with pytest.raises(RuntimeError, match="generation failed"):
        wait_for_cache(tmp_path)


def test_representative_categories_skip_episodes_with_zero_windows():
    dataset = SimpleNamespace(
        records=[{"source": "suite", "metadata": {"ground": {"valid": True}}},
                 {"source": "suite", "metadata": {"ground": {"valid": True}}},
                 {"source": "hm", "metadata": {"ground": {"valid": False}}}],
        idx_to_sidx_fidx=[(1, 0), (1, 30), (1, 60), (2, 0), (2, 30), (2, 60)])
    class Indexed:
        records = dataset.records
        idx_to_sidx_fidx = dataset.idx_to_sidx_fidx
        def __len__(self):
            return len(self.idx_to_sidx_fidx)
    assert representative_indices(Indexed(), 1) == list(range(6))


def test_cpu_cannot_be_reported_as_a_bf16_gpu_acceptance():
    with pytest.raises(ValueError, match="CUDA"):
        gpu_equivalence(smplh_motion_expert_config(), [], [], "cpu")


def test_validation_error_names_field_episode_and_native_start():
    import torch
    expected = {"misc": {"motion": torch.zeros(11, 245), "traj": torch.ones(11, 18),
                         "episode_index": 7150, "start_idx": 30, "source": "hm"}}
    actual = deepcopy(expected)
    actual["misc"]["traj"].zero_()
    with pytest.raises(AssertionError, match=r"misc.traj.*episode=7150 start=30"):
        compare_sample(actual, expected)


def test_required_episodes_extend_representatives_without_filtering_scope():
    class Indexed:
        records = [{"source": "suite", "metadata": {"ground": {"valid": True}}, "episode_index": i}
                   for i in (0, 65, 7150)]
        idx_to_sidx_fidx = [(s, start) for s in range(3) for start in (0, 30, 60)]
        def __len__(self):
            return len(self.idx_to_sidx_fidx)
    dataset = Indexed()
    baseline = set(representative_indices(dataset, 1))
    expanded = set(representative_indices(dataset, 1, [65, 7150]))
    assert baseline < expanded == set(range(9))
    assert len(dataset.records) == 3 and len(dataset) == 9


def test_validator_reuses_verified_archive_without_reconstructing_or_changing_samples(cached_archive, monkeypatch):
    from pathlib import Path
    from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
    from MotionExpert.dataset.window_cache import source_code_binding
    baseline = dataset_from_config(cached_archive, "val")
    archive = baseline.archive
    checks = []
    original_assert = archive.assert_current

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("Reused validation split reconstructed the whole archive")

    def current(record=None):
        checks.append(record)
        return original_assert(record)

    monkeypatch.setattr(SMPLHParameterArchive, "__init__", forbidden_constructor)
    monkeypatch.setattr(archive, "assert_current", current)
    train = dataset_from_config(cached_archive, "train", verified_archive=archive)
    validation = dataset_from_config(cached_archive, "val", verified_archive=archive)
    assert checks == [None, None]
    assert train.archive is validation.archive is archive
    assert train.window_cache is not validation.window_cache
    assert validation.window_cache is not baseline.window_cache
    assert validation.records == baseline.records
    assert validation.idx_to_sidx_fidx == baseline.idx_to_sidx_fidx
    assert {item["episode_index"] for item in train.records}.isdisjoint(
        item["episode_index"] for item in validation.records)
    compare_sample(validation[0], baseline[0])
    compare_sample(train[0], OriginalWindowPath(train)[0])
    manifest = json.loads((Path(cached_archive.DATA.WINDOW_CACHE_ROOT) / "manifest.json").read_text())
    assert source_code_binding() == manifest["binding"]["representation_sha256"]


def test_verified_archive_rejects_identity_scope_or_current_source_mismatch(cached_archive, tmp_path, monkeypatch):
    import os
    from pathlib import Path
    archive = dataset_from_config(cached_archive, "train").archive
    kwargs = dict(archive_root=cached_archive.DATA.ARCHIVE_ROOT,
                  vision_profile_root=cached_archive.DATA.VISION_PROFILE_ROOT,
                  image_profile=cached_archive.DATA.IMAGE_PROFILE,
                  min_valid_image_fraction=cached_archive.DATA.MIN_VALID_IMAGE_FRACTION,
                  window_cache_root=cached_archive.DATA.WINDOW_CACHE_ROOT,
                  verified_archive=archive, do_normalization=False)
    other_root = tmp_path / "other-root"
    other_root.mkdir()
    with pytest.raises(TypeError, match="SMPLHParameterArchive"):
        SMPLHMotionDataset(**dict(kwargs, verified_archive=object()))
    changes = (({"archive_root": str(other_root)}, "root differs"),
               ({"vision_profile_root": str(other_root)}, "vision profile root differs"),
               ({"vision_profile_root": None}, "vision profile differs"),
               ({"image_profile": "256x320"}, "image profile differs"),
               ({"allow_unpublished": True}, "publication/diagnostic mode"),
               ({"allow_unpublished": 0}, "publication/diagnostic mode"),
               ({"row_group_cache_mb": 0}, "reader cache settings"),
               ({"episode_indices": [999]}, "episode selection"))
    for change, message in changes:
        with pytest.raises(ValueError, match=message):
            SMPLHMotionDataset(**dict(kwargs, **change))
    with monkeypatch.context() as patch:
        patch.setitem(archive.info, "schema_version", "wrong-schema")
        with pytest.raises(ValueError, match="schema/model dimensions"):
            SMPLHMotionDataset(**kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(archive, "model_sha256", "0" * 64)
        with pytest.raises(ValueError, match="model SHA256"):
            SMPLHMotionDataset(**kwargs)
    wrong_config = deepcopy(cached_archive)
    wrong_config.DATA.MODEL_SHA256 = "0" * 64
    with pytest.raises(ValueError, match="frozen configuration"):
        dataset_from_config(wrong_config, "val", verified_archive=archive)
    # The requested split is val, but a changed train episode must still fail
    # the full archive.assert_current call used before reusing the snapshot.
    path = Path(archive._record_paths[0])
    before = path.stat()
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 1000))
    with pytest.raises(ValueError, match="episode record changed"):
        SMPLHMotionDataset(**dict(kwargs, split="val"))


def test_verified_archive_still_checks_each_splits_cache_index(cached_archive, monkeypatch):
    from pathlib import Path
    from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
    archive = dataset_from_config(cached_archive, "train").archive
    root = Path(cached_archive.DATA.WINDOW_CACHE_ROOT)
    manifest = json.loads((root / "manifest.json").read_text())
    path = root / manifest["indices"]["val"]["path"]
    with path.open("ab") as stream:
        stream.write(b"corrupt-split-index")

    def forbidden_constructor(*args, **kwargs):
        raise AssertionError("Reused validation split reconstructed the whole archive")

    monkeypatch.setattr(SMPLHParameterArchive, "__init__", forbidden_constructor)
    with pytest.raises(ValueError, match="split index digest mismatch"):
        dataset_from_config(cached_archive, "val", verified_archive=archive)
