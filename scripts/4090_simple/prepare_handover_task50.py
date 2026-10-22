"""Derive 30 Hz training observations from the sealed Handover SONIC1.1 release.

This is CPU-only observation/label selection, not another physical replay. The
50 Hz source, its 45/5 split, executed tokens and acceptance evidence stay intact.
Images are selected from decoded source video, then stored losslessly. DINO and
the joint trainer's original40k normalization are prepared separately.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np


VIDEO_KEY = "observation.images.egocentric"
PACKS = {"train": "Handover-sonic11-train", "val": "Handover-sonic11-val"}
SPLIT_COUNTS = {"train": 45, "val": 5}
ENCODER_SHA = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"
DECODER_SHA = "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"
OBS_SHA = "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def safe_child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source member: {relative}")
    path = Path(root) / relative
    if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError(f"External source member: {relative}")
    return path


def repository_root(script_path=None):
    """Stage copies outside a checkout must not turn resource ancestors into repos."""
    for parent in Path(script_path or __file__).resolve().parents:
        if ((parent / "AGENTS.md").is_file()
                and ((parent / ".git").exists() or (parent / "VLMExpert/src").is_dir())):
            return parent
    return None


def selected_source_indices(length):
    """Actual50Hz ticks at ceil(k*50/30), with no duplicated or invented tail."""
    if type(length) is not int or length <= 0:
        raise ValueError("Source episode must have a positive integer length")
    # k*50 <= (length-1)*30 is equivalent to ceil(k*50/30) < length.
    output_length = (length - 1) * 30 // 50 + 1
    ordinal = np.arange(output_length, dtype=np.int64)
    return (ordinal * 50 + 29) // 30


def named_state_permutation(source_info, target_info):
    """Map observed joints to the actual40k input order, never positional slices."""
    source = source_info["features"]["observation.state"]
    target = target_info["features"]["observation.state"]
    source_names, target_names = source.get("names"), target.get("names")
    if (not isinstance(source_names, list) or not isinstance(target_names, list)
            or source.get("shape") != [45] or target.get("shape") != [43]
            or len(source_names) != 45 or len(target_names) != 43
            or any(not isinstance(name, str) or not name for name in source_names + target_names)):
        raise ValueError("Explicit named45 source and named43 target state contracts are required")
    if len(set(source_names)) != 45 or len(set(target_names)) != 43:
        raise ValueError("Duplicate joint names in state contract")
    missing = set(target_names) - set(source_names)
    if missing:
        raise ValueError(f"Target joints missing from source: {sorted(missing)}")
    if set(source_names) - set(target_names) != {"neck_yaw", "neck_pitch"}:
        raise ValueError("Only the two synthetic neck channels may be removed")
    return [source_names.index(name) for name in target_names]


def project_state_statistics(source_statistics, permutation):
    """Describe the renamed state schema without refitting or changing hand stats."""
    projected = copy.deepcopy(source_statistics)
    state = projected["observation.state"]
    for name, values in state.items():
        if isinstance(values, list) and len(values) == 45:
            state[name] = [values[index] for index in permutation]
        elif name == "count" and (isinstance(values, (int, float)) or (isinstance(values, list) and len(values) == 1)):
            continue
        else:
            raise ValueError(f"Unexpected source45 state statistic shape: {name}")
    return projected


def inspect_release(source):
    """Bind the existing numerical acceptance to critical actual source bytes."""
    source = Path(source).resolve(strict=True)
    file_manifest_path = source / "file_manifest.json"
    checksum_rows = {}
    for line in (source / "SHA256SUMS").read_text().splitlines():
        digest, separator, member = line.partition("  ")
        if separator:
            checksum_rows[member] = digest
    if checksum_rows.get("file_manifest.json") != sha256(file_manifest_path):
        raise ValueError("Transfer file_manifest differs from SHA256SUMS")
    manifest = json.loads(file_manifest_path.read_text())
    members = {row["path"]: row for row in manifest["entries"]}
    if len(members) != len(manifest["entries"]):
        raise ValueError("Duplicate transfer manifest member")
    receipts = {}

    def bind(relative):
        path = safe_child(source, relative)
        expected = members.get(relative)
        if (not expected or not path.is_file() or path.stat().st_size != expected["bytes"]
                or sha256(path) != expected["sha256"]):
            raise ValueError(f"Source differs from sealed transfer: {relative}")
        receipts[relative] = {"bytes": path.stat().st_size, "sha256": expected["sha256"]}
        return path

    documents = {}
    for relative in (
        "REMOTE_PATHS.json", "training/conversion_manifest.json", "training/selection.json",
        "reports/final50_delivery_status.json", "reports/final50_training_release_summary.json",
        "reports/final50_independent_numeric_review.json", "reports/final50_training_export_acceptance.json",
        "reports/final50_video_data_review.json", "replay/conversion_manifest.json",
    ):
        documents[relative] = json.loads(bind(relative).read_text())
    upload_path = source / "REMOTE_UPLOAD_VERIFICATION.json"
    upload = json.loads(upload_path.read_text())
    receipts[upload_path.name] = {"sha256": sha256(upload_path), "bytes": upload_path.stat().st_size}
    if (upload.get("status") != "passed" or upload.get("all_file_sha256_exact") is not True
            or upload.get("symlinks") != 0):
        raise ValueError("Source upload is not accepted")
    delivery = documents["reports/final50_delivery_status.json"]
    numeric = documents["reports/final50_independent_numeric_review.json"]
    release = documents["training/conversion_manifest.json"]
    paths = documents["REMOTE_PATHS.json"]
    summary = documents["reports/final50_training_release_summary.json"]
    if (delivery.get("numeric_physics_passed") != 50 or delivery.get("episodes") != 50
            or numeric.get("status") != "passed_full50_photo_observation_and_independent_physical_replay"
            or numeric.get("independent_validation", {}).get("every_episode_passed") is not True
            or numeric.get("full_photo_audit", {}).get("passed") is not True
            or release.get("release_ready") is not True
            or release.get("physics_acceptance_gate") != "handover-physics-v2"):
        raise ValueError("Existing fifty-episode physical/photo acceptance is incomplete")
    release_hash = receipts["training/conversion_manifest.json"]["sha256"]
    if release_hash != paths.get("source_release_manifest_sha256") or release_hash != summary.get("release_manifest_sha256"):
        raise ValueError("Original release identities disagree")
    expected_controller = {"model_encoder.onnx": ENCODER_SHA, "model_decoder.onnx": DECODER_SHA,
                           "observation_config.yaml": OBS_SHA}
    if release.get("controller_sha256") != expected_controller:
        raise ValueError("Source SONIC controller differs from the selected v1.1 resources")
    if release.get("encoder_sha256") != ENCODER_SHA or release.get("masked_action_dimensions") != [78, 79]:
        raise ValueError("Source token or neck-mask contract differs")
    groups = {}
    datasets = {}
    for split, pack in PACKS.items():
        root = source / "training" / pack
        info = json.loads(bind(f"training/{pack}/meta/info.json").read_text())
        episodes_path = bind(f"training/{pack}/meta/episodes.jsonl")
        episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line]
        if info.get("fps") != 50 or info.get("total_episodes") != SPLIT_COUNTS[split] or len(episodes) != SPLIT_COUNTS[split]:
            raise ValueError(f"Unexpected {split} source episode count or clock")
        if [row["episode_index"] for row in episodes] != list(range(SPLIT_COUNTS[split])):
            raise ValueError("Source split episodes must be dense and chronological")
        features = info["features"]
        for key, dimension in (("observation.state", 45), ("action", 14), ("action.body_token_v1_1", 64), ("action.neck", 2), ("action_mask", 80)):
            if features[key]["shape"] != [dimension]:
                raise ValueError(f"Unexpected source feature: {key}")
        for name in ("modality.json", "stats_psi0.json", "tasks.jsonl"):
            bind(f"training/{pack}/meta/{name}")
        ancestors = set()
        for row in episodes:
            if (row.get("physics_validation", {}).get("verified_handover_success") is not True
                    or row.get("lineage_complete") is not True):
                raise ValueError("Episode lacks accepted physics or complete lineage")
            if split == "val" and (row.get("split_restriction") == "train_only" or row.get("data_category") == "task_scene_regeneration"):
                raise ValueError("Regenerated training-only source leaked into validation")
            for ancestor in row.get("lineage_group_sources", []):
                ancestors.add((ancestor["source_dataset_key"], int(ancestor["source_episode_index"])))
        groups[split] = ancestors
        datasets[split] = {"root": root, "info": info, "episodes": episodes}
    if groups["train"] & groups["val"]:
        raise ValueError("Source ancestors overlap across training and validation")
    return {"source": source, "datasets": datasets, "documents": documents,
            "members": members, "receipts": receipts, "bind": bind}


def select_table(table, indices, *, source_episode_index, source_original_id, first_global_row, state_permutation):
    """Select synchronized original arrays without interpolating any command."""
    import pyarrow as pa

    rows = table.take(pa.array(indices, type=pa.int64())).replace_schema_metadata(None)
    length = len(indices)
    required = {"timestamp", "frame_index", "episode_index", "index", "observation.state", "action", "action.body_token_v1_1"}
    if not required.issubset(rows.column_names):
        raise ValueError(f"Missing source columns: {required - set(rows.column_names)}")
    actual_times = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
    expected_times = np.arange(len(table), dtype=np.float64) / 50
    if not np.isfinite(actual_times).all() or not np.allclose(actual_times, expected_times, rtol=0, atol=2e-5):
        raise ValueError("Source timestamps do not follow the accepted50Hz clock")

    def replace(key, values, dtype):
        nonlocal rows
        values = pa.array(values, type=dtype)
        if key in rows.column_names:
            rows = rows.set_column(rows.column_names.index(key), key, values)
        else:
            rows = rows.append_column(key, values)

    if "source_state45" in rows.column_names:
        raise ValueError("Source already contains a conflicting source_state45 column")
    if len(state_permutation) != 43 or len(set(state_permutation)) != 43 or any(not 0 <= index < 45 for index in state_permutation):
        raise ValueError("A complete explicit43-joint state permutation is required")
    source_state = np.asarray(rows["observation.state"].to_pylist())
    if source_state.shape != (length, 45):
        raise ValueError("Actual source observations do not match the named45 state contract")
    state_type = rows["observation.state"].type
    replace("source_state45", source_state.tolist(), state_type)
    replace("observation.state", source_state[:, state_permutation].tolist(), state_type)
    replace("source_frame_index", indices, pa.int64())
    replace("source_elapsed_timestamp", actual_times[indices], pa.float64())
    if "source_episode_index" not in rows.column_names:
        replace("source_episode_index", np.full(length, source_episode_index), pa.int64())
    replace("source_original_id", np.full(length, source_original_id), pa.int64())
    replace("timestamp", np.arange(length, dtype=np.float64) / 30, pa.float32())
    replace("frame_index", np.arange(length), pa.int64())
    replace("index", np.arange(first_global_row, first_global_row + length), pa.int64())
    if "next.done" in rows.column_names:
        replace("next.done", np.arange(length) == length - 1, pa.bool_())
    if "action_valid" not in rows.column_names:
        replace("action_valid", np.ones(length, dtype=bool), pa.bool_())
    return rows


def select_video(source_path, output_path, indices, source_length):
    """CPU decode, select actual frames, losslessly encode and verify all pixels."""
    import av

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_hash = hashlib.sha256()
    selected_count = 0
    decoded_count = 0
    shape = None
    with av.open(str(source_path)) as source, av.open(str(output_path), "w") as destination:
        stream = destination.add_stream("libx264rgb", rate=30)
        stream.options = {"crf": "0", "preset": "fast"}
        stream.pix_fmt = "rgb24"
        stream.time_base = Fraction(1, 30)
        for native_index, frame in enumerate(source.decode(video=0)):
            decoded_count += 1
            if selected_count == len(indices) or native_index != indices[selected_count]:
                continue
            rgb = frame.to_ndarray(format="rgb24")
            if shape is None:
                shape = list(rgb.shape)
                stream.width, stream.height = shape[1], shape[0]
            elif list(rgb.shape) != shape:
                raise ValueError("Source RGB dimensions changed mid-episode")
            selected_hash.update(rgb.tobytes(order="C"))
            encoded_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            encoded_frame.pts = selected_count
            encoded_frame.time_base = Fraction(1, 30)
            for packet in stream.encode(encoded_frame):
                destination.mux(packet)
            selected_count += 1
        if selected_count != len(indices) or decoded_count != source_length:
            raise ValueError("Source video and accepted numeric frame counts differ")
        for packet in stream.encode():
            destination.mux(packet)
    actual_hash = hashlib.sha256()
    output_count = 0
    with av.open(str(output_path)) as encoded:
        if encoded.streams.video[0].average_rate != 30:
            raise ValueError("Output video clock differs from30Hz")
        for frame in encoded.decode(video=0):
            actual_hash.update(frame.to_ndarray(format="rgb24").tobytes(order="C"))
            output_count += 1
    if output_count != len(indices) or actual_hash.digest() != selected_hash.digest():
        raise ValueError("Derived video failed exact selected decoded-RGB verification")
    return {"source_decoded_frames": decoded_count, "frames": selected_count, "shape": shape,
            "selected_decoded_rgb_sha256": selected_hash.hexdigest(), "output_decoded_rgb_sha256": actual_hash.hexdigest(),
            "decoded_pixels_exact": True, "source_raw_framebuffer_pixels_claimed": False,
            "codec": "libx264rgb", "crf": 0, "fps": 30, "sha256": sha256(output_path)}


def prepare(source, output, state_contract):
    import pyarrow.parquet as pq

    source = Path(source).resolve(strict=True)
    output = Path(output).absolute()
    repository = repository_root()
    if (output.exists() or output.is_symlink() or output.resolve().is_relative_to(source)
            or (repository is not None and output.resolve().is_relative_to(repository))):
        raise ValueError("Use a fresh output outside the source release and repository")
    accepted = inspect_release(source)
    state_contract = Path(state_contract).resolve(strict=True)
    target_info = json.loads(state_contract.read_text())
    state_mappings = {split: named_state_permutation(data["info"], target_info)
                      for split, data in accepted["datasets"].items()}
    staging = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    staging.mkdir(parents=True)
    episode_receipts = []
    counts = {}
    try:
        for split, dataset in accepted["datasets"].items():
            pack = PACKS[split]
            target = staging / pack
            (target / "meta").mkdir(parents=True)
            info = copy.deepcopy(dataset["info"])
            info["fps"] = 30
            info["features"]["source_state45"] = copy.deepcopy(info["features"]["observation.state"])
            info["features"]["observation.state"] = copy.deepcopy(target_info["features"]["observation.state"])
            info["features"]["source_frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
            info["features"]["source_elapsed_timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
            for key in ("source_episode_index", "source_original_id"):
                info["features"].setdefault(key, {"dtype": "int64", "shape": [1], "names": None})
            info["features"].setdefault("action_valid", {"dtype": "bool", "shape": [1], "names": None})
            video_metadata = {"video.fps": 30, "video.codec": "h264", "video.pix_fmt": "gbrp",
                              "video.is_depth_map": False, "has_audio": False}
            info["features"][VIDEO_KEY]["info"] = dict(video_metadata)
            info["features"][VIDEO_KEY]["video_info"] = dict(video_metadata)
            info["adaptation"] = {"source_fps": 50, "target_fps": 30, "source_tick_rule": "ceil",
                                  "tokens_reencoded": False, "new_physical_replay_performed": False}
            for path in sorted(dataset["root"].rglob("*")):
                relative = path.relative_to(dataset["root"])
                if not path.is_file() or relative.parts[0] in {"data", "videos"} or relative.as_posix() in {"meta/info.json", "meta/episodes.jsonl", "meta/stats.json", "meta/stats_psi0.json"}:
                    continue
                original = accepted["bind"](f"training/{pack}/{relative.as_posix()}")
                destination = safe_child(target, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
            # These are source50Hz metadata statistics with the new named state
            # order. They are not the joint trainer's inherited40k normalization.
            for name in ("stats.json", "stats_psi0.json"):
                original = accepted["bind"](f"training/{pack}/meta/{name}")
                projected = project_state_statistics(json.loads(original.read_text()), state_mappings[split])
                write_json(target / "meta" / name, projected)
            modality_path = target / "meta/modality.json"
            modality = json.loads(modality_path.read_text())
            state_modality = modality.get("state", {}).get("joint_positions")
            if state_modality is not None:
                if state_modality.get("start") != 0 or state_modality.get("end") != 45:
                    raise ValueError("Source state modality differs from the named45 contract")
                state_modality["end"] = 43
                write_json(modality_path, modality)
            global_row = 0
            output_episodes = []
            for episode in dataset["episodes"]:
                episode_id = episode["episode_index"]
                variables = {"episode_index": episode_id, "episode_chunk": episode_id // info["chunks_size"], "video_key": VIDEO_KEY}
                parquet_relative = info["data_path"].format(**variables)
                video_relative = info["video_path"].format(**variables)
                source_parquet = accepted["bind"](f"training/{pack}/{parquet_relative}")
                source_video = accepted["bind"](f"training/{pack}/{video_relative}")
                table = pq.read_table(source_parquet)
                source_length = len(table)
                if source_length != episode["length"]:
                    raise ValueError("Source metadata and actual Parquet lengths differ")
                indices = selected_source_indices(source_length)
                selected = select_table(table, indices, source_episode_index=episode_id,
                                        source_original_id=episode["source_episode_index"], first_global_row=global_row,
                                        state_permutation=state_mappings[split])
                destination = safe_child(target, parquet_relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(selected, destination, compression="zstd")
                # This verifies newly transformed data only; original physical acceptance is reused.
                reread = pq.read_table(destination)
                if not reread.equals(selected):
                    raise ValueError("New selected Parquet failed lossless roundtrip")
                video_receipt = select_video(source_video, safe_child(target, video_relative), indices, source_length)
                converted = copy.deepcopy(episode)
                converted["length"] = len(indices)
                converted["source_50hz_length"] = source_length
                converted["observation_semantics"] = "Selected actual50Hz pre-action observations and executed labels; no new physics or warmup frames"
                converted["adaptation"] = {"source_tick_rule": "ceil", "source_fps": 50, "fps": 30,
                                           "source_first_action_primer_preserved_as_provenance": True}
                output_episodes.append(converted)
                episode_receipts.append({"split": split, "episode_index": episode_id,
                    "source_original_id": episode["source_episode_index"], "source_identity": episode.get("source_identity"),
                    "source_parquet_sha256": accepted["receipts"][f"training/{pack}/{parquet_relative}"]["sha256"],
                    "source_video_sha256": accepted["receipts"][f"training/{pack}/{video_relative}"]["sha256"],
                    "source_frames": source_length, "frames": len(indices), "source_frame_indices": indices.tolist(),
                    "maximum_selection_time_offset_s": float(np.max(indices / 50 - np.arange(len(indices)) / 30)),
                    "video": video_receipt, "parquet_sha256": sha256(destination)})
                global_row += len(indices)
                print(json.dumps({"split": split, "episode": episode_id, "frames": len(indices)}, ensure_ascii=False), flush=True)
            info["total_frames"] = global_row
            write_json(target / "meta/info.json", info)
            (target / "meta/episodes.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output_episodes))
            counts[split] = {"episodes": len(output_episodes), "frames": global_row,
                             "complete_history_future_windows": sum(max(0, row["length"] - 59) for row in output_episodes)}
        evidence = staging / "source_evidence"
        for relative in accepted["documents"]:
            destination = safe_child(evidence, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        for name in ("REMOTE_UPLOAD_VERIFICATION.json", "file_manifest.json", "SHA256SUMS"):
            shutil.copyfile(source / name, evidence / name)
        for split, pack in PACKS.items():
            for name in ("stats.json", "stats_psi0.json"):
                original = accepted["bind"](f"training/{pack}/meta/{name}")
                destination = evidence / "original_statistics" / pack / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(original, destination)
        shutil.copyfile(state_contract, evidence / "initialization40k_info.json")
        ready = {"contract": "cvpr-handover-task50-30hz-selection-v1", "status": "cpu_30hz_adaptation_complete_dino_and_joint_preflight_pending",
            "created_utc": datetime.now(timezone.utc).isoformat(), "task_id": "simple/G1WholebodyHandoverTeleop-v0",
            "source_root": str(source), "output_root": str(output), "source_fps": 50, "fps": 30,
            "source_frame_rule": "ceil", "source_timestamp_column": "source_elapsed_timestamp", "counts": counts,
            "source_acceptance_reused_not_recomputed": True, "source_receipts": accepted["receipts"],
            "source_file_manifest_sha256": sha256(source / "file_manifest.json"),
            "controller_sha256": accepted["documents"]["training/conversion_manifest.json"]["controller_sha256"],
            "state_contract": {"path": str(state_contract), "sha256": sha256(state_contract),
                "target_joint_names": target_info["features"]["observation.state"]["names"],
                "source_joint_names": {split: dataset["info"]["features"]["observation.state"]["names"] for split, dataset in accepted["datasets"].items()},
                "target_from_source_permutation": state_mappings, "source45_preserved_column": "source_state45"},
            "metadata_statistics": {"state45_to43_projection_only_no_refit": True,
                "other_feature_statistics_unchanged": True, "source_fps": 50,
                "policy": "Source50Hz TRAIN descriptive statistics projected to43 named state channels; unchanged14-hand/64-token stats. Original files preserved under source_evidence/original_statistics. These are not joint training normalization: trainer must bind original40k run/dataset_statistics.json explicitly."},
            "tokens_reencoded": False, "command_interpolation": False, "new_physical_replay_performed": False,
            "official_benchmark_score_claimed": False, "dino_cache_generated": False, "training_started": False,
            "boundary_policy": {"actual_first_second_retained_for_history": True, "synthetic_prefix_frames": 0,
                "history_offsets": [-30, -24, -18, -12, -6, 0], "first_eligible_current_row": 30,
                "future_action_rows": 30, "full_future_required_by_trainer": True, "tail_repetition_permitted": False},
            "clock_limits": "Nominal30Hz timestamp=k/30; selected actual source time=ceil(k*50/30)/50, up to13.334ms later. History six5Hz ticks retain exact0.2s native spacing. Original50Hz executed labels are sampled; the30Hz sample-and-hold controller trajectory has not been physically revalidated.",
            "video_limits": "Exact preservation of original H264 decodedRGB after lossless reencoding; not a claim of raw framebuffer equality.",
            "episodes": episode_receipts, "script_sha256": sha256(__file__)}
        ready["payload_sha256"] = {str(path.relative_to(staging)): sha256(path) for path in sorted(staging.rglob("*")) if path.is_file()}
        write_json(staging / "ADAPTATION_READY.json", ready)
        staging.rename(output)
        return ready
    except Exception as error:
        write_json(staging / "FAILED.json", {"status": "failed", "error": str(error), "source_root": str(source), "output_requested": str(output)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--state-contract", type=Path, required=True,
                        help="Actual40k training pack meta/info.json with43 named input joints")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args()
    if args.inspect_only:
        release = inspect_release(args.source_root)
        target_info = json.loads(args.state_contract.read_text())
        permutations = {key: named_state_permutation(value["info"], target_info) for key, value in release["datasets"].items()}
        print(json.dumps({"source_acceptance": "bound", "split_episode_counts": {key: len(value["episodes"]) for key, value in release["datasets"].items()},
                          "consumed_metadata_files": len(release["receipts"]), "source_fps": 50, "target_state_dim": 43,
                          "state_permutations": permutations, "training_started": False}, ensure_ascii=False))
    else:
        if args.output is None:
            parser.error("--output is required unless --inspect-only")
        result = prepare(args.source_root, args.output, args.state_contract)
        print(json.dumps({"status": result["status"], "output": str(args.output), "counts": result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
