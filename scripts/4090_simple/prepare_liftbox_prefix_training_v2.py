"""CPU-only preparation of real warmup-prefix v2 supervision for the joint launcher.

This entrypoint never trains. It executes the simulator environment's independent
v2 verifier, retaining accepted numerical evidence while binding raw/code/resource
bytes and checking new packed transformations. It then binds all DINO rows
including the real observation-only warmup prefix. The old
CarryBox and Lift v1 preparation/training entrypoints are not unlocked here.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

import finetune_carrybox as shared
import finetune_lift_box_hold as existing

CONTRACT = "cvpr-simple-lift-real-warmup-prefix-v2"
TASK_ID = "simple/G1WholebodyLiftBoxHoldSonic-v0"
SPLITS = {"train": [0, 1, 2], "val": [3], "test": [4]}
TRAIN_SOURCES = frozenset((0, 1, 2, 5, 6, 7))
TEST_SOURCES = frozenset((4, 9, 10, 11))
PREFIX_ROWS = 30
SOURCE_STEP = 31000
SOURCE_SHA = "e43fd2fb07f093dbf7a1e00ee737d5378f02f374c1b520056a65c34f4747daa6"
VERIFIER = Path(__file__).with_name("prepare_liftbox_prefix_v2.py")
SIM_PYTHON = existing.SIM_PYTHON


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validated_splits(pack):
    splits = pack.get("splits", {})
    train, test = splits.get("train"), splits.get("test")
    require(isinstance(train, list) and len(train) == 3
            and all(type(source) is int and source >= 0 for source in train)
            and len(set(train)) == 3 and set(train) <= TRAIN_SOURCES
            and splits.get("val") == [3] and isinstance(test, list) and len(test) == 1
            and type(test[0]) is int and test[0] in TEST_SOURCES and not set(train) & set(test)
            and set(splits) == {"train", "val", "test"},
            "Require three unique actual TRAIN candidates0/1/2/5/6/7, fixed VAL3 and one distinct TEST candidate4/9/10/11")
    return {"train": list(train), "val": [3], "test": list(test)}


def audit_quality(dataset_root, audit_python):
    """Run the real v2 verifier; reading an asserted status/contract never suffices."""
    dataset_root, audit_python = (Path(p).resolve(strict=True) for p in (dataset_root, audit_python))
    script = VERIFIER.resolve(strict=True)
    command = [str(audit_python), str(script), "--verify-root", str(dataset_root)]
    started = time.monotonic()
    completed = subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1"),
                               capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("Real warmup-prefix v2 verification failed:\n" + completed.stderr[-6000:])
    try:
        summary = json.loads(completed.stdout)
    except ValueError as error:
        raise RuntimeError("v2 verifier did not return its actual structured result") from error
    require(isinstance(summary, dict) and summary.get("contract") == CONTRACT and summary.get("task_id") == TASK_ID
            and len(summary.get("episodes", [])) == 5,
            "Verifier must validate accepted evidence and transformed bytes for the complete v2 3/1/1 pilot; v1/status-only claims are ineligible")
    validated_splits(summary)
    ready_path = dataset_root / "LIFT_COMMANDED_ROLLOUT_READY.json"
    declared = json.loads(ready_path.read_text())
    validated_splits(declared)
    require(all(summary[key] == declared[key] for key in ("contract", "task_id", "splits", "episodes", "packed_files", "train_demo_curation", "test_demo_curation")),
            "Actual verifier result differs from this packed v2 data")
    return {"verifier_script": str(script), "verifier_script_sha256": shared.sha256(script),
            "verifier_python": str(audit_python), "command": command, "returncode": completed.returncode,
            "wall_time_s": time.monotonic() - started, "dataset_contract_sha256": shared.sha256(ready_path),
            "summary": summary,
            "scope": "accepted numerical evidence retained + actual raw/packed byte binding/new transform verification",
            "execution_scope": "Actual SIM CPU verifier execution checks accepted raw/code/resource/report SHA bindings and new packed labels, clocks and lossless video. Existing548-state force/encoder/warmup numerical acceptance is retained, not repeated; a status JSON alone is insufficient."}


def validate_base_and_pack(base_run, pack, dataset_root):
    ready = json.loads((base_run / "LOCAL_RUN_READY.json").read_text())
    require(ready.get("status") == "local_joint_run_prepared" and ready.get("step") == SOURCE_STEP
            and ready.get("joint_sha256") == SOURCE_SHA and ready.get("optimizer_rng_restored") is False,
            "Prefix v2 preparation requires the actual protected local weight-only 31k base")
    require(pack.get("contract") == CONTRACT and pack.get("task_id") == TASK_ID,
            "Only the newly audited real-prefix v2 pack can be prepared")
    validated_splits(pack)
    expected_stats = shared.sha256(base_run / "dataset_statistics.json")
    require(pack.get("source_joint_sha256") == SOURCE_SHA and pack.get("source_statistics_sha256") == expected_stats
            and shared.sha256(dataset_root / "source_dataset_statistics.json") == expected_stats,
            "v2 pack/base must use the same actual 31k weights and unchanged A100 statistics")
    return ready


def validate_episode_arrays(columns, tokens, hands, source_episode):
    """Metadata/label contract, independent of the raw v2 verifier and DINO reader."""
    frames = np.asarray(columns["frame_index"])
    valid = np.asarray(columns["action_valid"])
    count = len(frames)
    require(count >= PREFIX_ROWS + 30 and frames.shape == (count,) and np.array_equal(frames, np.arange(count)),
            "Every v2 episode needs all real prefix rows and thirty teacher future labels")
    require(valid.dtype == np.dtype(bool) and valid.shape == frames.shape,
            "action_valid must be an explicit non-null bool column, never numeric or a cast")
    require(not valid[:PREFIX_ROWS].any() and valid[PREFIX_ROWS:].all(),
            "Exactly the first thirty rows are history-only; every later row is a real teacher label")
    expected_native = (frames * 50 + 29) // 30
    require(np.array_equal(columns["source_frame_index"], expected_native),
            "Physical source_frame_index must be the global ceil30-to50 tick, distinct from teacher reference")
    reference = np.r_[np.full(PREFIX_ROWS, -1, dtype=np.int64),
                      ((frames[PREFIX_ROWS:] - PREFIX_ROWS) * 50 + 29) // 30]
    require(np.array_equal(columns["source_reference_frame_index"], reference),
            "Teacher source frame must start at zero at global row30, with no prefix labels")
    phases = np.asarray(columns["phase"])
    require(phases.shape == frames.shape and np.all(phases[:PREFIX_ROWS] == "warmup_observation_prefix")
            and np.all(phases[PREFIX_ROWS:] == "oracle_teacher_token"), "Explicit warmup/teacher phases differ")
    times = np.asarray(columns["timestamp"])
    tolerance = np.maximum(2e-6, np.abs(np.spacing((frames / 30).astype(np.float32))))
    require(times.shape == frames.shape and np.isfinite(times).all()
            and np.all(np.abs(times - frames / 30) <= tolerance), "Nominal episode clock must remain full native30Hz")
    require(np.all(np.asarray(columns["source_episode_index"]) == source_episode), "Source episode identity differs")
    require(np.allclose(columns["source_elapsed_timestamp"], expected_native / 50, rtol=0, atol=1e-8),
            "Actual event clock must include the real warmup prefix")
    teacher_time = np.r_[np.full(PREFIX_ROWS, -1.), reference[PREFIX_ROWS:] / 50]
    require(np.allclose(columns["teacher_source_timestamp"], teacher_time, rtol=0, atol=1e-8),
            "Teacher source time must be independent of the global observation clock")
    tokens, hands = np.asarray(tokens), np.asarray(hands)
    require(tokens.shape == (count, 64) and hands.shape == (count, 14)
            and np.isfinite(tokens).all() and np.isfinite(hands).all(), "Finite dense body64/hand14 storage required")
    require(not np.any(tokens[:PREFIX_ROWS]) and not np.any(hands[:PREFIX_ROWS]),
            "History-only dense action storage must contain zero placeholders, never invented labels")
    candidates = [i for i in range(PREFIX_ROWS, count)
                  if shared.future_is_complete(i, count, frames, times, np.asarray(columns["source_frame_index"]),
                                               np.asarray(columns["source_episode_index"]), source_frame_rule="ceil", label_valid=valid)]
    require(candidates and candidates[0] == PREFIX_ROWS, "The first eligible complete real history/future sample must be row30")
    return {"rows": count, "history_only_rows": int((~valid).sum()), "valid_teacher_rows": int(valid.sum()),
            "first_eligible_frame": candidates[0], "eligible_future_windows": len(candidates),
            "first_future_frames": list(range(30, 60)), "first_future_action_valid": True,
            "future_tail_rows_excluded": int(valid.sum()) - len(candidates)}, candidates


def valid_train_bounds(episodes, field):
    selected, excluded = [], 0
    for values, valid in episodes:
        values, valid = np.asarray(values), np.asarray(valid)
        require(valid.dtype == np.dtype(bool) and valid.shape == (len(values),),
                "TRAIN representability requires the actual bool validity column")
        selected.append(values[valid])
        excluded += int((~valid).sum())
    report = existing.command_bounds(np.concatenate(selected), field)
    report.update(label_filter="TRAIN action_valid=true only; real observation-only prefix placeholders excluded",
                  excluded_history_only_train_rows=excluded)
    return report


def inspect_labels_and_history(args, field, splits):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from psi.data.lerobot.dino_cache import load_cache_manifest, load_episode_cache
    from psi.data.lerobot.motion_history import history_rows
    selected_labels, episodes = [], []
    names = ("frame_index", "timestamp", "source_frame_index", "source_reference_frame_index",
             "source_episode_index", "source_elapsed_timestamp", "teacher_source_timestamp", "action_valid", "phase")
    for split, sources in splits.items():
        cache_root = args.dino_cache_root / split
        manifest = load_cache_manifest(cache_root, args.dataset_root / split, expected_profile="256x320")
        for episode, source in enumerate(sources):
            path = args.dataset_root / split / f"data/chunk-000/episode_{episode:06d}.parquet"
            table = pq.read_table(path, columns=[*names, "action.body_token_v1_1", "action"])
            require(pa.types.is_boolean(table.schema.field("action_valid").type) and table["action_valid"].null_count == 0,
                    "Packed action_valid must have actual Arrow bool dtype and no null labels")
            columns = {name: np.asarray(table[name].to_pylist()) for name in names}
            tokens, hands = (np.asarray(table[name].to_pylist(), dtype=np.float32)
                             for name in ("action.body_token_v1_1", "action"))
            row, candidates = validate_episode_arrays(columns, tokens, hands, source)
            cached = load_episode_cache(cache_root, episode, manifest=manifest, dataset_root=args.dataset_root / split,
                                        verify_checksums=True)
            try:
                require(len(cached.features) == table.num_rows and np.array_equal(cached.frame_index, columns["frame_index"])
                        and np.allclose(cached.timestamp, columns["timestamp"], rtol=0, atol=1e-7),
                        "DINO must encode every real prefix and teacher RGB row at the same time")
                current, histories = history_rows(cached.frame_index, cached.timestamp, cached.source_frame_index,
                                                 cached.video_timestamp, cached.valid, fps=30)
                selected = {int(frame): history for frame, history in zip(current, histories, strict=True)}
                require(30 in selected and np.array_equal(selected[30], [0, 6, 12, 18, 24, 30]),
                        "Production DINO reader must supply real prefix history [0,6,12,18,24,30] for first teacher row30")
                require(all(frame in selected for frame in candidates), "A future-eligible teacher row lacks actual complete DINO history")
                require(cached.features[selected[30]].shape == (6, 5, 1024)
                        and np.isfinite(cached.features[selected[30]]).all(), "First real DINO history must be finite6x5x1024")
                row.update(split=split, episode_index=episode, source_episode_index=source,
                           first_history_frames=columns["frame_index"][selected[30]].astype(int).tolist(),
                           first_history_physical_ticks=columns["source_frame_index"][selected[30]].astype(int).tolist(),
                           first_history_times_s=columns["timestamp"][selected[30]].astype(float).tolist(),
                           parquet_sha256=shared.sha256(path), feature_identity=cached.receipt["outputs"]["features.npy"],
                           dino_source_index_scope="DINO sidecar native video ordinal is separate from the packed global50Hz physical source_frame_index")
                episodes.append(row)
            finally:
                mmap = getattr(cached.features, "_mmap", None)
                if mmap is not None:
                    mmap.close()
            if split == "train":
                selected_labels.append((np.concatenate((tokens, hands), axis=1), columns["action_valid"]))
    representability = valid_train_bounds(selected_labels, field)
    return {"contract": CONTRACT, "episodes": episodes, "first_eligible_frame": 30,
            "first_history_frames": [0, 6, 12, 18, 24, 30], "first_future_action_valid": True,
            "history_only_train_rows": PREFIX_ROWS * 3, "source_frame_rule": "ceil",
            "bounds_use_only_valid_teacher_train_rows": True}, representability


def snapshot_sources(args, pack):
    snapshot = existing.snapshot_sources(args, pack)
    target = args.output / "source_at_launch"
    additions = [Path(__file__), VERIFIER, Path(__file__).with_name("collect_liftbox_prefix_v2.py"),
                 shared.REPO / "VLMExpert/src/psi/data/lerobot/motion_history.py",
                 shared.REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/live_pose_sonic_warmup.py",
                 shared.REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/dual_source_psi0.py"]
    for path in additions:
        path = path.resolve(strict=True)
        require(path.is_relative_to(shared.REPO) and path.stat().st_size < 2 << 20,
                "Source snapshot contains bounded actual code only")
        relative = Path("tree") / path.relative_to(shared.REPO)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        snapshot["files"][str(relative)] = {"actual_path": str(path), "sha256": shared.sha256(path), "size_bytes": path.stat().st_size}
    manifest = {"scope": "Actual prefix v2 collector/packer/preparer plus production history/controller/task source bytes; excludes data/weights/environments",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=shared.REPO, text=True).strip(),
                "files": snapshot["files"]}
    shared.write_json(target / "manifest.json", manifest)
    snapshot["manifest_sha256"] = shared.sha256(target / "manifest.json")
    return snapshot


def quality_gate_report(pack):
    splits = validated_splits(pack)
    curation = pack.get("train_demo_curation")
    require(isinstance(curation, dict) and curation.get("selected_train_sources") == splits["train"],
            "Actual named TRAIN sources must match the complete verified demonstration-curation record")
    test_curation = pack.get("test_demo_curation")
    require(isinstance(test_curation, dict) and type(test_curation.get("selected_test_source")) is int
            and test_curation["selected_test_source"] == splits["test"][0]
            and test_curation.get("candidate_order") == [4, 9, 10, 11],
            "Actual TEST source must match the complete verified test-demonstration curation record")
    return {"status": "actual_new_task_evidence_recomputed", "task_id": TASK_ID,
            "status_scope": "Compatibility status for the existing joint launcher: accepted full numerical evidence retained and actually rebound here; not a claim to repeat accepted force/encoder computation",
            "dataset_contract": CONTRACT, "source_frame_rule": "ceil",
            "split_episode_counts": {"train": 3, "val": 1, "test": 1},
            "training_source_episodes": splits["train"], "train_demo_curation": curation,
            "test_source_episodes": splits["test"], "test_demo_curation": test_curation,
            "required_physical_before_after_source_episodes": sorted({4, 8, *splits["test"]}),
            "measurement_hz": 50, "history_only_prefix_rows_per_episode": 30, "first_eligible_frame": 30,
            "test_used_for_control_selection": False,
            "test_used_for_control_selection_scope": "Only the current v2 collection/controller protocol frozen before this TRAIN/VAL/TEST collection; this is not a statement that source4 was never inspected",
            "previously_diagnosed_sources": [4, 8], "not_blind_test": True,
            "prior_diagnostic_scope": "Sources4/8 already participated in earlier physical and cold-start diagnostics, including the open-hand warmup decision; retain that history",
            "test_data_policy": f"Actual TEST source{splits['test'][0]} supplies no training gradients or fitted statistics; a curated successful expert reference does not replace original4/8 physical evaluation failures or make this a blind benchmark",
            "preparation_entrypoint_can_train": False}


def prepare(args):
    args.dataset_root, args.base_run, args.audit_python = (p.resolve(strict=True) for p in
                                                        (args.dataset_root, args.base_run, args.audit_python))
    quality = audit_quality(args.dataset_root, args.audit_python)
    contract_path = args.dataset_root / "LIFT_COMMANDED_ROLLOUT_READY.json"
    pack = json.loads(contract_path.read_text())
    validate_base_and_pack(args.base_run, pack, args.dataset_root)
    config, argv, report = shared.prepare(args, dataset_contract_path=contract_path,
                                         episode_counts={"train": 3, "val": 1, "test": 1},
                                         experiment="lift_box_hold_real_prefix_v2_prepared")
    source_encoder = json.loads((args.base_run / "dino_training_contract.json").read_text())["encoder"]
    dino = existing.verify_dino(args, source_encoder)
    prefix, representability = inspect_labels_and_history(args, config.data.transform.field, validated_splits(pack))
    snapshot = snapshot_sources(args, pack)
    shared.write_json(args.output / "prefix_training_contract.json", prefix)
    shared.write_json(args.output / "train_command_representability.json", representability)
    report.update(task_id=TASK_ID, dataset_contract=CONTRACT, prepare_only=True,
                  intended_training_entrypoint="finetune_lift_box_hold_joint.py --prepared-run <this output>",
                  launcher_source_sha256=shared.sha256(__file__), shared_training_mechanics_sha256=shared.sha256(Path(shared.__file__)),
                  dino_checksums=dino, quality_recomputation=quality, train_command_representability=representability,
                  source_at_launch=snapshot, real_prefix_training=prefix,
                  quality_gate=quality_gate_report(pack))
    shared.write_json(args.output / "preparation.json", report)
    return config, argv, report


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("base-run", "dataset-root", "dino-cache-root", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--audit-python", type=Path, default=SIM_PYTHON)
    result.add_argument("--max-steps", type=int, default=300)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--grad-accum", type=int, default=4)
    result.add_argument("--lr", type=float, default=1e-5)
    result.add_argument("--warmup-steps", type=int, default=20)
    result.add_argument("--seed", type=int, default=20261021)
    result.add_argument("--test-windows", type=int, default=20)
    result.add_argument("--prepare-only", action="store_true", required=True)
    return result


def main():
    args = parser().parse_args()
    sys.path[:0] = [str(shared.REPO / "VLMExpert/src"), str(shared.REPO)]
    _, _, report = prepare(args)
    print(json.dumps({"status": "cpu_config_prepared", "contract": CONTRACT, "output": str(args.output),
                      "source_step": report["source_step"], "gpu_training": "not_run"}))


if __name__ == "__main__":
    main()
