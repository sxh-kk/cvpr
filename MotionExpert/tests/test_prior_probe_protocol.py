import numpy as np
import pytest
import torch
from unittest.mock import patch

from MotionExpert.probe_prior import (FEATURE_PROTOCOL, checked_forward_hidden,
                                     condition_inputs, forward_hidden, gpu_acceptance, make_noise)


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


def _frozen_expert_and_images():
    from MotionExpert.tests.fixtures import conditioning, tiny_expert
    expert = tiny_expert()
    expert.model.requires_grad_(False)
    images = conditioning(batch=1)["img_embs"]
    return expert, {"img_embs": images}


def test_paired_probe_uses_updated_state_and_matches_public_second_call_features():
    expert, conditions = _frozen_expert_and_images()
    noise = make_noise([{"episode_index": 7, "start_frame": 0}], 42, "archive")
    with torch.inference_mode(), patch.object(expert.model, "forward", wraps=expert.model.forward) as forward:
        pooled, hidden, prediction = checked_forward_hidden(expert, conditions, noise)
    assert forward.call_count == 2
    first, second = [call.args for call in forward.call_args_list]
    torch.testing.assert_close(first[0], noise, rtol=0, atol=0)
    torch.testing.assert_close(first[1], torch.ones(1), rtol=0, atol=0)
    torch.testing.assert_close(second[1], torch.full((1,), .9), rtol=0, atol=0)
    dt = (second[1] - first[1]).view(-1, 1, 1)
    torch.testing.assert_close(second[0], noise + dt * expert.model.level, rtol=0, atol=0)
    torch.testing.assert_close(pooled, hidden.mean(1), rtol=0, atol=0)
    assert prediction.shape == (1, 11, 245)
    with torch.inference_mode():
        _, public_noise = expert._sampling_inputs(conditions, seed=42, task="fore", forecast_prefix=6,
                                                 normalized=True, output_frames=11)
        expected, expected_hidden, _ = forward_hidden(expert, conditions, public_noise)
        actual = expert.forward_features(conditions, seed=42)
    torch.testing.assert_close(actual["one_step_hidden"], expected_hidden, rtol=0, atol=0)
    torch.testing.assert_close(actual["one_step_global"], expected, rtol=0, atol=0)


@pytest.mark.parametrize("mismatch", ["single_call", "unupdated_state", "wrong_time"])
def test_probe_acceptance_rejects_legacy_or_incorrect_feature_calls(monkeypatch, mismatch):
    expert, conditions = _frozen_expert_and_images()
    noise = make_noise([{"episode_index": 7, "start_frame": 0}], 42, "archive")

    def incorrect_features(model, state, model_kwargs):
        y = model_kwargs["y"]
        time = torch.ones(state.shape[0])
        if mismatch == "single_call":
            return model(state, time, y, return_features=True)
        velocity = model(state, time, y)
        next_time = time.new_full(time.shape, .8 if mismatch == "wrong_time" else .9)
        updated = state if mismatch == "unupdated_state" else state + (next_time - time)[:, None, None] * velocity
        return model(updated, next_time, y, return_features=True)

    monkeypatch.setattr(expert.flow, "forward_features", incorrect_features)
    with torch.inference_mode(), pytest.raises(AssertionError):
        checked_forward_hidden(expert, conditions, noise)


def test_both_frozen_probe_acceptances_record_two_call_protocol():
    from MotionExpert.probe_tokens import _gpu_acceptance
    expert, conditions = _frozen_expert_and_images()
    plan = {"seed": 42, "archive_fingerprint": "archive"}
    rows = [{"episode_index": 7, "start_frame": 0}]
    for report in (gpu_acceptance(expert, conditions, rows, plan), _gpu_acceptance(expert, conditions, plan)):
        assert report["passed"] and report["feature_protocol"] == FEATURE_PROTOCOL
        assert report["expert_gradients"] is False


def test_snapshot_expert_constructs_flow_for_the_public_and_probe_feature_paths():
    from MotionExpert.probe_prior import _expert_from_snapshot
    expert, conditions = _frozen_expert_and_images()
    with patch("MotionExpert.model.UniEgoMotion", return_value=expert.model), \
            patch("MotionExpert.checkpoints.model_state", return_value=expert.model.state_dict()):
        snapshot = _expert_from_snapshot(expert.cfg, expert.normalizer, None, "/frozen.pt", {}, expert.device)
    assert snapshot.flow.num_steps == 10 and snapshot.flow.prediction_type == "velocity"
    noise = make_noise([{"episode_index": 7, "start_frame": 0}], 42, "archive")
    with torch.inference_mode():
        pooled, hidden, _ = checked_forward_hidden(snapshot, conditions, noise)
    torch.testing.assert_close(pooled, hidden.mean(1), rtol=0, atol=0)
    assert not any(parameter.requires_grad for parameter in snapshot.parameters())
