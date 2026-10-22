"""CPU checks for the new source-bound30Hz observation selection only."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("prepare_handover_task50", Path(__file__).resolve().parents[1] / "prepare_handover_task50.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
STATE_PERMUTATION = list(range(28, 43)) + list(range(14, 28)) + list(range(14))
SOURCE_NAMES = [f"joint_{index}" for index in range(43)] + ["neck_yaw", "neck_pitch"]


def make_state_contract(tmp_path):
    path = tmp_path / "initialization40k/meta/info.json"
    module.write_json(path, {"features": {"observation.state": {"dtype": "float32", "shape": [43],
        "names": [SOURCE_NAMES[index] for index in STATE_PERMUTATION]}}})
    return path


def test_named_state_mapping_matches_each_joint_not_input_position(tmp_path):
    source = {"features": {"observation.state": {"shape": [45], "names": SOURCE_NAMES}}}
    target = json.loads(make_state_contract(tmp_path).read_text())
    assert module.named_state_permutation(source, target) == STATE_PERMUTATION
    source["features"]["observation.state"]["names"] = SOURCE_NAMES.copy()
    source["features"]["observation.state"]["names"][0] = "unrecognized_joint"
    with pytest.raises(ValueError, match="missing"):
        module.named_state_permutation(source, target)
    source["features"]["observation.state"]["names"][0] = SOURCE_NAMES[1]
    with pytest.raises(ValueError, match="Duplicate"):
        module.named_state_permutation(source, target)
    source["features"]["observation.state"]["names"] = None
    with pytest.raises(ValueError, match="Explicit named"):
        module.named_state_permutation(source, target)


def test_descriptive_state_statistics_are_projected_without_refit_or_action_change():
    original = {"observation.state": {"mean": list(range(45)), "std": list(range(50, 95)), "count": [27440]},
                "action": {"mean": list(range(14))}, "action.body_token_v1_1": {"mean": list(range(64))}}
    projected = module.project_state_statistics(original, STATE_PERMUTATION)
    assert projected["observation.state"]["mean"] == STATE_PERMUTATION
    assert projected["observation.state"]["std"] == [50 + index for index in STATE_PERMUTATION]
    assert projected["observation.state"]["count"] == [27440]
    assert projected["action"] == original["action"]
    assert projected["action.body_token_v1_1"] == original["action.body_token_v1_1"]
    assert original["observation.state"]["mean"] == list(range(45))


@pytest.mark.parametrize("length", [1, 2, 3, 49, 50, 51, 474, 650, 879])
def test_selection_has_real_ticks_and_exact_five_hz_history(length):
    indices = module.selected_source_indices(length)
    expected = []
    ordinal = 0
    while True:
        numerator = ordinal * 50
        actual = numerator // 30 + (numerator % 30 != 0)
        if actual >= length:
            break
        expected.append(actual)
        ordinal += 1
    np.testing.assert_array_equal(indices, expected)
    assert indices[0] == 0 and indices[-1] < length
    assert np.all(np.diff(indices) > 0)
    for current in range(30, len(indices)):
        history = indices[current - np.array([30, 24, 18, 12, 6, 0])]
        np.testing.assert_array_equal(np.diff(history), 10)
    offsets = indices / 50 - np.arange(len(indices)) / 30
    assert offsets.min() >= -1e-12 and offsets.max() <= 2 / 150 + 1e-12


@pytest.mark.parametrize("length", [0, -1, True, 1.5])
def test_invalid_lengths_are_rejected(length):
    with pytest.raises(ValueError):
        module.selected_source_indices(length)


def make_table(length=121):
    pa = pytest.importorskip("pyarrow")
    return pa.table({
        "timestamp": np.arange(length, dtype=np.float32) / 50,
        "frame_index": np.arange(length, dtype=np.int64),
        "episode_index": np.full(length, 3, dtype=np.int64),
        "index": np.arange(500, 500 + length, dtype=np.int64),
        "next.done": np.arange(length) == length - 1,
        "observation.state": np.arange(length * 45, dtype=np.float32).reshape(length, 45).tolist(),
        "action": np.arange(length * 14, dtype=np.float32).reshape(length, 14).tolist(),
        "action.body_token_v1_1": np.arange(length * 64, dtype=np.float32).reshape(length, 64).tolist(),
    })


def test_table_keeps_original_commands_and_distinct_actual_clock():
    source = make_table()
    indices = module.selected_source_indices(len(source))
    out = module.select_table(source, indices, source_episode_index=3, source_original_id=39, first_global_row=1000, state_permutation=STATE_PERMUTATION)
    for key in ("action", "action.body_token_v1_1", "episode_index"):
        assert out[key].to_pylist() == source[key].take(indices).to_pylist()
    original_state = np.asarray(source["observation.state"].take(indices).to_pylist())
    np.testing.assert_array_equal(out["observation.state"].to_pylist(), original_state[:, STATE_PERMUTATION])
    np.testing.assert_array_equal(out["source_state45"].to_pylist(), original_state)
    assert out["source_episode_index"].to_pylist() == [3] * len(indices)
    assert out["source_original_id"].to_pylist() == [39] * len(indices)
    assert out["source_frame_index"].to_pylist() == indices.tolist()
    np.testing.assert_array_equal(out["source_elapsed_timestamp"].to_numpy(), source["timestamp"].take(indices).to_numpy())
    np.testing.assert_array_equal(out["timestamp"].to_numpy(), (np.arange(len(indices), dtype=np.float64) / 30).astype(np.float32))
    assert out["index"].to_pylist() == list(range(1000, 1000 + len(indices)))
    assert out["action_valid"].to_pylist() == [True] * len(indices)
    assert out["next.done"].to_pylist() == [False] * (len(indices) - 1) + [True]
    assert source["timestamp"][1].as_py() == np.float32(1 / 50)


def test_existing_source_ids_and_action_validity_are_not_overwritten():
    pa = pytest.importorskip("pyarrow")
    source = make_table().append_column("source_episode_index", pa.array([92] * 121))
    source = source.append_column("action_valid", pa.array([False] * 50 + [True] * 71))
    indices = module.selected_source_indices(len(source))
    out = module.select_table(source, indices, source_episode_index=3, source_original_id=92, first_global_row=0, state_permutation=STATE_PERMUTATION)
    assert out["source_episode_index"].to_pylist() == [92] * len(indices)
    assert out["action_valid"].to_pylist() == source["action_valid"].take(indices).to_pylist()


def test_wrong_actual_clock_is_rejected():
    pa = pytest.importorskip("pyarrow")
    source = make_table()
    source = source.set_column(0, "timestamp", pa.array(np.arange(121) / 30))
    with pytest.raises(ValueError, match="timestamps"):
        module.select_table(source, module.selected_source_indices(121), source_episode_index=0, source_original_id=0, first_global_row=0, state_permutation=STATE_PERMUTATION)


def write_video(path, images, fps=50):
    av = pytest.importorskip("av")
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264rgb", rate=fps)
        stream.width, stream.height = images[0].shape[1], images[0].shape[0]
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "preset": "ultrafast"}
        for image in images:
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_video_selects_exact_frames_without_rgb_changes(tmp_path):
    av = pytest.importorskip("av")
    images = np.random.default_rng(7).integers(0, 256, (57, 12, 16, 3), dtype=np.uint8)
    source, target = tmp_path / "source.mp4", tmp_path / "out.mp4"
    write_video(source, images)
    indices = module.selected_source_indices(len(images))
    receipt = module.select_video(source, target, indices, len(images))
    with av.open(str(target)) as container:
        actual = np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])
    np.testing.assert_array_equal(actual, images[indices])
    assert receipt["decoded_pixels_exact"] and receipt["fps"] == 30
    assert receipt["source_raw_framebuffer_pixels_claimed"] is False
    with pytest.raises(ValueError, match="frame counts"):
        module.select_video(source, tmp_path / "wrong.mp4", indices, len(images) + 1)


def test_safe_child_rejects_external_paths_and_links(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "source"
    root.mkdir()
    (root / "link").symlink_to(outside, target_is_directory=True)
    for relative in ("../outside/a", str(outside), "link/file"):
        with pytest.raises(ValueError):
            module.safe_child(root, relative)


def test_stage_copy_does_not_infer_resource_ancestors_as_repository(tmp_path):
    script = tmp_path / "resources/.pipeline/task50-stage/prepare_handover_task50.py"
    script.parent.mkdir(parents=True)
    script.write_text("test")
    assert module.repository_root(script) is None
    (tmp_path / "resources/AGENTS.md").write_text("not enough alone")
    assert module.repository_root(script) is None
    (tmp_path / "resources/VLMExpert/src").mkdir(parents=True)
    assert module.repository_root(script) == tmp_path / "resources"


def synthetic_release(tmp_path, monkeypatch):
    """Two tiny fixtures replace only the real-release45/5 counts in this test."""
    pq = pytest.importorskip("pyarrow.parquet")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(module, "SPLIT_COUNTS", {"train": 1, "val": 1})
    controller = {"model_encoder.onnx": module.ENCODER_SHA, "model_decoder.onnx": module.DECODER_SHA,
                  "observation_config.yaml": module.OBS_SHA}
    release = {"release_ready": True, "physics_acceptance_gate": "handover-physics-v2", "controller_sha256": controller,
               "encoder_sha256": module.ENCODER_SHA, "masked_action_dimensions": [78, 79]}
    module.write_json(source / "training/conversion_manifest.json", release)
    release_sha = module.sha256(source / "training/conversion_manifest.json")
    module.write_json(source / "REMOTE_PATHS.json", {"source_release_manifest_sha256": release_sha})
    module.write_json(source / "training/selection.json", {"synthetic_test": True})
    module.write_json(source / "reports/final50_delivery_status.json", {"numeric_physics_passed": 50, "episodes": 50})
    module.write_json(source / "reports/final50_training_release_summary.json", {"release_manifest_sha256": release_sha})
    module.write_json(source / "reports/final50_independent_numeric_review.json", {
        "status": "passed_full50_photo_observation_and_independent_physical_replay",
        "independent_validation": {"every_episode_passed": True}, "full_photo_audit": {"passed": True}})
    for relative in ("reports/final50_training_export_acceptance.json", "reports/final50_video_data_review.json", "replay/conversion_manifest.json"):
        module.write_json(source / relative, {"synthetic_test": True})
    for split, pack in module.PACKS.items():
        root = source / "training" / pack
        features = {k: {"shape": [n]} for k, n in (("observation.state", 45), ("action", 14), ("action.body_token_v1_1", 64), ("action.neck", 2), ("action_mask", 80))}
        features["observation.state"]["names"] = SOURCE_NAMES
        features[module.VIDEO_KEY] = {"dtype": "video", "shape": [12, 16, 3]}
        module.write_json(root / "meta/info.json", {"fps": 50, "total_episodes": 1, "total_frames": 121,
            "chunks_size": 1000, "features": features, "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"})
        for name in ("modality.json", "tasks.jsonl"):
            module.write_json(root / "meta" / name, {"synthetic_test": True})
        for name in ("stats.json", "stats_psi0.json"):
            module.write_json(root / "meta" / name, {"observation.state": {"mean": list(range(45)), "count": [121]}, "action": {"mean": list(range(14))}})
        episode = {"episode_index": 0, "source_episode_index": 0 if split == "train" else 39, "length": 121,
            "lineage_complete": True, "physics_validation": {"verified_handover_success": True},
            "lineage_group_sources": [{"source_dataset_key": "test", "source_episode_index": 0 if split == "train" else 39}]}
        (root / "meta/episodes.jsonl").write_text(json.dumps(episode) + "\n")
        parquet = root / "data/chunk-000/episode_000000.parquet"
        parquet.parent.mkdir(parents=True)
        table = make_table()
        table = table.set_column(table.column_names.index("episode_index"), "episode_index", __import__("pyarrow").array([0] * 121))
        pq.write_table(table, parquet)
        images = np.random.default_rng(8).integers(0, 256, (121, 12, 16, 3), dtype=np.uint8)
        write_video(root / "videos/chunk-000" / module.VIDEO_KEY / "episode_000000.mp4", images)
    entries = [{"path": str(path.relative_to(source)), "bytes": path.stat().st_size, "sha256": module.sha256(path)}
               for path in sorted(source.rglob("*")) if path.is_file()]
    module.write_json(source / "file_manifest.json", {"entries": entries})
    (source / "SHA256SUMS").write_text(module.sha256(source / "file_manifest.json") + "  file_manifest.json\n")
    module.write_json(source / "REMOTE_UPLOAD_VERIFICATION.json", {"status": "passed", "all_file_sha256_exact": True, "symlinks": 0})
    return source


def test_full_adaptation_is_atomic_and_keeps_original_bytes(tmp_path, monkeypatch):
    source = synthetic_release(tmp_path, monkeypatch)
    before = {str(path.relative_to(source)): module.sha256(path) for path in source.rglob("*") if path.is_file()}
    output = tmp_path / "derived"
    result = module.prepare(source, output, make_state_contract(tmp_path))
    assert result["counts"] == {"train": {"episodes": 1, "frames": 73, "complete_history_future_windows": 14},
                                "val": {"episodes": 1, "frames": 73, "complete_history_future_windows": 14}}
    assert result["new_physical_replay_performed"] is False
    assert result["dino_cache_generated"] is False
    assert result["boundary_policy"]["synthetic_prefix_frames"] == 0
    for path in source.rglob("*"):
        if path.is_file(): assert module.sha256(path) == before[str(path.relative_to(source))]
    for member, expected in result["payload_sha256"].items():
        assert module.sha256(output / member) == expected
    info = json.loads((output / "Handover-sonic11-train/meta/info.json").read_text())
    assert info["fps"] == 30 and info["features"][module.VIDEO_KEY]["video_info"]["video.fps"] == 30
    assert info["features"]["observation.state"]["shape"] == [43]
    assert info["features"]["observation.state"]["names"] == [SOURCE_NAMES[index] for index in STATE_PERMUTATION]
    statistics = json.loads((output / "Handover-sonic11-train/meta/stats_psi0.json").read_text())
    assert statistics["observation.state"]["mean"] == STATE_PERMUTATION
    assert statistics["action"]["mean"] == list(range(14))
    with pytest.raises(ValueError, match="fresh output"):
        module.prepare(source, output, make_state_contract(tmp_path))


def test_changed_source_is_rejected_before_creating_output(tmp_path, monkeypatch):
    source = synthetic_release(tmp_path, monkeypatch)
    (source / "reports/final50_delivery_status.json").write_text("{}")
    with pytest.raises(ValueError, match="sealed transfer"):
        module.prepare(source, tmp_path / "derived", make_state_contract(tmp_path))
    assert not (tmp_path / "derived").exists()
