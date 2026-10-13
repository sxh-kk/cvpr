"""Ground estimation and selected-window fallback preserve usable dense data."""

import numpy as np
import pytest

from egomotion_pipeline.geometry import quat_to_matrix
from egomotion_pipeline.ground import resolve_window_ground
from egomotion_pipeline.resampling import resample
from egomotion_pipeline.storage import Archive, read_arrays
from egomotion_pipeline.validation import validate
from egomotion_pipeline.windows import prepare_window
from test_ground_and_head_reference import standing_episode, prepare


LOCAL = "observation.state.body_head_floor"
VALID = "observation.valid.body_head_floor_position"


def moving_episode():
    episode = standing_episode(duration=2.)
    episode.positions["body"][:, [10, 11], 0] = episode.timestamps[:, None]*.3
    episode.positions["body"][:, [10, 11], 2] += episode.timestamps[:, None]*.2
    return episode


def test_dbscan_keeps_a_dense_height_cluster_wider_than_four_cm():
    episode = standing_episode(floor=0., duration=2.)
    episode.positions["body"][:, 11, 2] += episode.timestamps*.04
    ground = prepare(episode).metadata["ground"]
    assert ground["valid"] and ground["method"] == "stationary_foot_height_dbscan"
    assert len(ground["clusters"]) == 1
    assert ground["clusters"][0]["spread_m"] > .04
    assert ground["height_m"] == pytest.approx(0.)


def test_dbscan_noise_is_not_a_floor_and_duplicate_density_is_preserved():
    episode = standing_episode(floor=0.)
    episode.position_valid["body"] = np.ones((len(episode.timestamps), 22), bool)
    episode.position_valid["body"][2:, 10] = False
    episode.positions["body"][:2, 10, 2] = -10
    ground = prepare(episode).metadata["ground"]
    assert ground["valid"] and ground["height_m"] == pytest.approx(0.)
    assert ground["noise_count"] == 2
    assert ground["clusters"][0]["sample_count"] == len(episode.timestamps)


def test_short_density_cluster_has_no_extra_quarter_second_gate():
    ground = prepare(standing_episode(duration=.1)).metadata["ground"]
    assert ground["valid"]
    assert ground["clusters"][0]["support_s"] < .25


@pytest.mark.parametrize("old_parameter", ["min_support_s", "max_cluster_spread_m"])
def test_removed_ground_gates_are_not_silently_reused(old_parameter):
    with pytest.raises(ValueError, match=old_parameter):
        prepare(standing_episode(), ground={old_parameter: .04})


def test_failed_dbscan_remains_unknown_in_archive_but_resolves_in_window():
    episode = prepare(standing_episode(), ground={"cluster_min_samples": 1000})
    assert episode.metadata["ground"]["reason"] == "no_density_height_cluster"
    arrays = resample(episode)
    assert not arrays[VALID].any() and (arrays[LOCAL] == 0).all()
    window = prepare_window(arrays, episode.metadata, start_frame=0, num_frames=8, fps=15)
    assert window.metadata["ground"]["window_fallback_used"]
    assert window.metadata["ground"]["height_m"] == pytest.approx(1.99, abs=1e-6)
    assert not window.metadata["ground"]["verified"]
    assert window.arrays[VALID].all() and window.trajectory_valid.all()
    assert np.allclose(window.arrays["observation.state.head_reference_height"], 1.61)
    assert not arrays[VALID].any() and not episode.metadata["ground"]["valid"]


def test_fallback_uses_selected_window_and_training_rate_not_skipped_frames():
    episode = moving_episode()
    # This finite but unsuitable value belongs to a frame excluded at 15Hz.
    episode.positions["body"][7, [10, 11], 2] = 100
    episode = prepare(episode)
    arrays = resample(episode)
    first = prepare_window(arrays, episode.metadata, start_frame=6, num_frames=4, fps=15)
    second = prepare_window(arrays, episode.metadata, start_frame=18, num_frames=4, fps=15)
    assert first.metadata["training_window"]["source_frame_indices"] == [6, 8, 10, 12]
    assert first.metadata["ground"]["height_m"] == pytest.approx(2.05, abs=1e-6)
    assert second.metadata["ground"]["height_m"] == pytest.approx(2.13, abs=1e-6)
    assert np.allclose(first.arrays["timestamp"], np.arange(4)/15)
    assert np.allclose(first.arrays["observation.state.head_reference_floor_world"][:, 2], 2.05)
    local = first.arrays[LOCAL].reshape(-1, 22, 3)
    world = first.arrays["observation.state.body_world"].reshape(-1, 22, 3)
    assert np.allclose(local[..., 2], world[..., 2]-2.05)


def test_fallback_excludes_invalid_zero_filled_feet_and_keeps_joint_masks():
    episode = moving_episode()
    episode.position_valid["body"] = np.ones((len(episode.timestamps), 22), bool)
    episode.position_valid["body"][6, 10] = False
    episode.position_valid["body"][8, 18] = False
    episode = prepare(episode)
    arrays = resample(episode)
    window = prepare_window(arrays, episode.metadata, start_frame=6, num_frames=4, fps=15)
    selected = [6, 8, 10, 12]
    feet = arrays["observation.state.body_world"].reshape(-1, 22, 3)[selected][:, [10, 11], 2]
    valid = arrays["observation.valid.body_position"][selected][:, [10, 11]]
    assert window.metadata["ground"]["height_m"] == pytest.approx(feet[valid].mean()-.02, abs=1e-6)
    assert window.metadata["ground"]["window_fallback_valid_point_count"] == 7
    assert not window.arrays[VALID][0, 10] and not window.arrays[VALID][1, 18]
    assert (window.arrays[LOCAL].reshape(-1, 22, 3)[~window.arrays[VALID]] == 0).all()


def test_a_valid_zero_floor_does_not_trigger_the_legacy_zero_sentinel_fallback():
    episode = prepare(moving_episode(), ground={"mode": "manual", "height_m": 0., "verified": True})
    window = prepare_window(resample(episode), episode.metadata, start_frame=6, num_frames=4, fps=15)
    assert not window.metadata["ground"]["window_fallback_used"]
    assert window.metadata["ground"]["height_m"] == 0
    assert window.arrays["observation.valid.floor_verified"].all()
    assert np.allclose(window.arrays["observation.state.head_reference_height"], 3.6)


@pytest.mark.parametrize("unavailable", ["feet", "world", "disabled"])
def test_fallback_does_not_fabricate_missing_feet_or_override_disabled_world(unavailable):
    episode = moving_episode()
    config = {}
    if unavailable == "feet":
        episode.position_valid["body"] = np.ones((len(episode.timestamps), 22), bool)
        episode.position_valid["body"][:, [10, 11]] = False
    elif unavailable == "world":
        config["coordinate_verified"] = False
    else:
        config["ground"] = {"mode": "disabled"}
    episode = prepare(episode, **config)
    window = prepare_window(resample(episode), episode.metadata, start_frame=0, num_frames=5)
    assert not window.metadata["ground"]["valid"]
    assert not window.arrays[VALID].any()
    assert not window.trajectory_valid.any()
    assert (window.arrays[LOCAL] == 0).all()


def test_archive_window_read_rebuilds_fields_without_rewriting_parquet(tmp_path):
    episode = prepare(moving_episode())
    archive = Archive(tmp_path/"archive", {"test": "window_read"})
    archive.add(episode, resample(episode), "test-fingerprint")
    archive.finalize()
    before = archive.data_path(0).read_bytes()
    window = archive.read_window(0, start_frame=6, num_frames=4, fps=15)
    assert window.arrays[VALID].all()
    assert window.trajectory.shape == (4, 18) and window.trajectory_valid.all()
    assert np.allclose(window.trajectory[:, 8], window.arrays["observation.state.head_reference_height"])
    assert archive.data_path(0).read_bytes() == before
    assert not read_arrays(archive.data_path(0), archive.specs)[VALID].any()
    assert validate(Archive(archive.root))["frames"] == 61
    with pytest.raises(ValueError, match="floor height unavailable"):
        validate(Archive(archive.root), require_floor=True)


def test_window_canonicalization_and_world_reconstruction_are_consistent():
    episode = moving_episode()
    episode.positions["camera"][:, 0, :2] = [4, -3]
    episode = prepare(episode)
    window = prepare_window(resample(episode), episode.metadata, start_frame=6, num_frames=4)
    arrays = window.arrays
    assert np.allclose(arrays["observation.state.head_reference_world"][0, :2], 0)
    rotation = quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"])
    rebuilt = np.einsum("tij,tkj->tki", rotation, arrays[LOCAL].reshape(-1, 22, 3))
    rebuilt += arrays["observation.state.head_reference_floor_world"][:, None]
    assert np.allclose(rebuilt, arrays["observation.state.body_world"].reshape(-1, 22, 3), atol=1e-6)


def test_missing_camera_masks_local_pose_and_both_adjacent_trajectory_deltas():
    episode = prepare(standing_episode())
    arrays = resample(episode)
    arrays["observation.valid.head_reference_rotation"][4] = False
    window = prepare_window(arrays, episode.metadata, start_frame=0, num_frames=6, fps=15)
    assert not window.arrays[VALID][2].any()
    assert not window.trajectory_valid[2, :9].any()
    assert not window.trajectory_valid[2:4, 9:].any()
    assert window.trajectory_valid[[0, 1, 4, 5]].all()
    assert (window.trajectory[~window.trajectory_valid] == 0).all()


def test_canonicalization_rejects_a_missing_first_camera_but_raw_window_keeps_masks():
    episode = prepare(standing_episode())
    arrays = resample(episode)
    arrays["observation.valid.head_reference_position"][0] = False
    with pytest.raises(ValueError, match="valid first camera"):
        prepare_window(arrays, episode.metadata, start_frame=0, num_frames=4)
    window = prepare_window(arrays, episode.metadata, start_frame=0, num_frames=4, canonicalize=False)
    assert not window.trajectory_valid[0].any()
    assert not window.trajectory_valid[1, 9:].any()
    assert window.trajectory_valid[2:].all()


@pytest.mark.parametrize("kwargs", [
    {"fps": 14}, {"fps": 0}, {"start_frame": -1}, {"num_frames": 1},
    {"num_frames": 500}, {"fps": True}, {"start_frame": 0.5},
])
def test_window_selection_rejects_unsupported_or_out_of_range_requests(kwargs):
    episode = prepare(standing_episode())
    options = {"start_frame": 0, "num_frames": 4, "fps": 15, **kwargs}
    with pytest.raises(ValueError):
        prepare_window(resample(episode), episode.metadata, **options)
