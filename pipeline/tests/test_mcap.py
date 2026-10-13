import json
from pathlib import Path
import numpy as np
import pytest
from egomotion_pipeline.cli import convert, main, load_config
from egomotion_pipeline.egosuite_mcap import (
    discover_egosuite_mcap, read_egosuite_mcap, inspect_mcap, UpperBodyOnly, UPPER_JOINTS, LOWER_JOINTS)
from egomotion_pipeline.features import extract_features
from egomotion_pipeline.storage import Archive, read_arrays
from egomotion_pipeline.validation import validate
from egomotion_pipeline.video import nearest_mcap_video_frames
from mcap_fixtures import make_mcap, BASE_NS
from test_archive_and_video import FakeEncoder


def config(root, **overrides):
    return {"type": "egosuite_mcap", "name": "egosuite", "root": str(root),
            "coordinate_verified": True, "rotation_basis_verified": True, **overrides}


def read(path, **overrides):
    return read_egosuite_mcap(path, path.stem, config(path.parent, **overrides))


def test_discovery_nested_roots_uuid_selection_limit_and_duplicates(tmp_path):
    root = tmp_path/"download"
    make_mcap(root/"task-a"/"a.mcap")
    make_mcap(root/"task-b"/"b.mcap")
    (root/"task-b"/"unfinished.mcap.part").write_bytes(b"partial")
    assert [uid for _, uid in discover_egosuite_mcap(config(root))] == ["a", "b"]
    ids = tmp_path/"ids.json"; ids.write_text('["b"]')
    assert [uid for _, uid in discover_egosuite_mcap(config(root, episode_ids=str(ids)))] == ["b"]
    assert len(list(discover_egosuite_mcap(config(root, limit=1)))) == 1
    assert len(list(discover_egosuite_mcap(config(root/"task-a"/"a.mcap")))) == 1
    make_mcap(root/"task-c"/"a.mcap")
    with pytest.raises(ValueError, match="Duplicate MCAP"):
        list(discover_egosuite_mcap(config(root)))


def test_capture_clock_precision_named_quaternions_and_structural_metadata(tmp_path):
    path = make_mcap(tmp_path/"file-uuid.mcap", reverse_body=True)
    ep = read(path)
    assert ep.uid == "file-uuid" and ep.metadata["source_episode_uuid"] == "internal-session-uuid"
    assert ep.metadata["clock"]["origin_ns"] == BASE_NS
    assert np.array_equal(ep.timestamps, np.array([round(i*1e9/30) for i in range(5)])/1e9)
    assert np.allclose(ep.positions["body"][:, 0, 0], np.arange(5)/30)
    assert np.allclose(ep.rotations["body"][:, 0], [.5, 0, 0, np.sqrt(3)/2])
    assert np.allclose(ep.rotations["camera"][:, 0], [1, 0, 0, 0])
    assert ep.positions["camera"][0, 0, 2] == 1.5
    assert ep.metadata["clock"]["topic_timestamp_fields"]["/pose/body"] == {"header.ts": 5}
    assert ep.metadata["operator_id"] == "operator-1" and ep.metadata["scene_id"] == "scene-1"
    assert ep.metadata["camera_intrinsics/head_left"]["width"] == 32
    assert "THIS_SOURCE_TEXT" not in json.dumps(ep.metadata)
    info = inspect_mcap(path, path.stem)
    assert info["body_joints"] == 22 and info["topics"]["/pose/body"]["messages"] == 5


def test_time_join_masks_missing_and_out_of_tolerance_poses(tmp_path):
    path = make_mcap(tmp_path/"gaps.mcap", missing_camera=[2], camera_offsets={3: 20_000_000}, missing_hand=[1])
    ep = read(path)
    assert ep.position_valid["camera"][:, 0].tolist() == [True, True, False, False, True]
    assert ep.rotation_valid["camera"][:, 0].tolist() == [True, True, False, False, True]
    assert not ep.position_valid["hand_left"][1].any()
    assert not ep.position_valid["hand_right"][1].any()
    assert ep.metadata["pose_alignment"]["modalities"]["camera"]["matched"] == 3


def test_vendor_frame_flags_mask_only_the_documented_body_subsets(tmp_path):
    path = make_mcap(tmp_path/"quality.mcap", bad={"hand": [1], "upper_body": [2], "lower_body": [3]})
    ep = read(path)
    assert not ep.position_valid["hand_left"][1].any() and not ep.position_valid["hand_right"][1].any()
    assert ep.position_valid["body"][1].all()
    assert not ep.position_valid["body"][2, UPPER_JOINTS].any()
    assert ep.position_valid["body"][2, LOWER_JOINTS].all()
    assert not ep.rotation_valid["body"][3, LOWER_JOINTS].any()
    assert ep.rotation_valid["body"][3, UPPER_JOINTS].all()
    assert ep.position_valid["head"].all() and ep.position_valid["camera"].all()


def test_h264_delta_frames_pts_video_gaps_and_bad_image_flags(tmp_path):
    path = make_mcap(tmp_path/"video.mcap", video_indices=[0, 1, 3, 4], bad={"image": [1]})
    ep = read(path)
    frames = list(nearest_mcap_video_frames(ep.video, np.arange(5)/30, 1/60+.001))
    assert [image is not None for image, _ in frames] == [True, False, False, True, True]
    assert frames[0][0].shape == (24, 32, 3)
    assert frames[3][0].mean() > frames[0][0].mean()+70
    assert frames[3][1] == pytest.approx(.1, abs=1e-9)
    assert frames[4][1] == pytest.approx(.133333333, abs=1e-9)


def test_camera_choice_raw_video_and_missing_extrinsics(tmp_path):
    path = make_mcap(tmp_path/"right.mcap", camera="head_right", raw_video=True)
    ep = read(path, camera="head_right", video_topic="/sensor/camera_raw/head_right/video")
    assert ep.video["preprocessing"] == {"mode": "raw"}
    assert ep.video["camera"] == "head_right"
    assert list(nearest_mcap_video_frames(ep.video, [0], .01))[0][0] is not None
    with pytest.raises(ValueError, match="missing.*head_left/extrinsic"):
        read(path)
    motion_only = read(path, require_camera=False)
    assert not motion_only.position_valid["camera"].any()


def test_upper_body_rejection_duplicate_clock_and_cli_skip(tmp_path, capsys):
    upper = make_mcap(tmp_path/"upper.mcap", body_joints=14, upper_only=True)
    with pytest.raises(UpperBodyOnly):
        read(upper)
    malformed = make_mcap(tmp_path/"malformed.mcap", body_joints=20)
    with pytest.raises(ValueError, match="22-joint"):
        read(malformed)
    duplicate = make_mcap(tmp_path/"duplicate.mcap", duplicate_body=True)
    with pytest.raises(ValueError, match="duplicate capture timestamps"):
        read(duplicate)
    cfg = tmp_path/"config.json"; cfg.write_text(json.dumps({"sources": [config(upper)]}))
    assert main(["inspect", "--config", str(cfg)]) == 0
    assert '"upper_body_14_episodes": 1' in capsys.readouterr().out
    assert main(["convert", "--config", str(cfg), "--output", str(tmp_path/"out")]) == 1
    assert "No selected full-body" in capsys.readouterr().err


def test_end_to_end_mcap_resumption_features_validation_and_changed_input(tmp_path):
    root = tmp_path/"source"
    path = make_mcap(root/"episode.mcap")
    cfg = {"sources": [config(root)]}
    output = tmp_path/"archive"
    convert(cfg, output)
    archive = Archive(output)
    assert validate(archive, strict=True)["frames"] == 5
    record = archive.records()[0]
    assert record["video"]["kind"] == "mcap" and Path(record["video"]["path"]) == path
    assert not list(output.rglob("*.mp4"))
    convert(cfg, output)
    assert len(Archive(output).records()) == 1
    extract_features(archive, {"batch_size": 2}, FakeEncoder())
    assert validate(archive, strict=True, require_dino=True)["frames"] == 5
    arrays = read_arrays(archive.data_path(0), archive.specs)
    assert np.allclose(arrays["observation.video_timestamp"], np.arange(5)/30, atol=1e-8)
    assert arrays["observation.features.dino"][4, 0] > arrays["observation.features.dino"][0, 0]
    assert all("THIS_SOURCE_TEXT" not in p.read_text() for p in output.rglob("*.json"))
    extract_features(archive, {"batch_size": 2}, FakeEncoder())
    make_mcap(path, count=6)
    with pytest.raises(ValueError, match="input changed"):
        convert(cfg, output)


def test_config_resolves_mcap_and_episode_selection_paths(tmp_path):
    cfg = tmp_path/"config.json"
    cfg.write_text(json.dumps({"sources": [config("source", episode_ids="ids.json")]}))
    loaded = load_config(cfg)["sources"][0]
    assert loaded["root"] == str(tmp_path/"source") and loaded["episode_ids"] == str(tmp_path/"ids.json")
