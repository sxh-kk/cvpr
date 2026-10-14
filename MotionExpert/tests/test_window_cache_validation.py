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
    OriginalWindowPath, compare_sample, gpu_equivalence, representative_indices,
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
