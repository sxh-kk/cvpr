import numpy as np
import torch

from MotionExpert.probe_prior import condition_inputs, make_noise


def test_paired_noise_does_not_depend_on_batch_partition_or_row_order():
    rows = [{"episode_index": 7, "start_frame": 0}, {"episode_index": 9, "start_frame": 60}]
    full = make_noise(rows, 42, "archive")
    separate = torch.cat([make_noise([row], 42, "archive") for row in rows])
    assert torch.equal(full, separate)
    assert torch.equal(full.flip(0), make_noise(rows[::-1], 42, "archive"))
    assert not torch.equal(full, make_noise(rows, 42, "archive", draw=1))


def test_condition_whitelist_is_independent_of_future_labels_and_masks():
    sample = {"image_features": np.ones((6, 5, 1024), dtype=np.float32),
              "historical_image_valid": np.ones(6, dtype=bool),
              "motion": np.ones((11, 245), dtype=np.float32),
              "motion_valid": np.ones((11, 245), dtype=bool)}
    before = condition_inputs([sample])
    sample["motion"][:] = np.nan
    sample["motion_valid"][:] = False
    sample["floor_transform"] = np.full((11, 4, 4), np.nan)
    after = condition_inputs([sample])
    assert set(before) == {"img_embs", "valid_img_embs", "valid_frames"}
    for key in before:
        assert torch.equal(before[key], after[key])
