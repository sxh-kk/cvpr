#!/usr/bin/env python3
"""Export an explicit TRAIN-only modified teacher, preserving its source prefix.

The native50Hz reference and the actual30Hz sample-held execution are distinct
arrays. This is a reference artifact, never admission of an RGB training episode
or a learned-policy evaluation. Existing source parquets are read, never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import numpy as np


CONTRACT = "cvpr-handover-modified-teacher-reference-v1"
SCOPE = "TRAIN_teacher_only"
ENCODER_SHA256 = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"
DECODER_SHA256 = "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"
HELDOUT = {39, 65, 80, 92, 94}
HAND_NAMES = tuple(f"{side}_hand_{part}_joint" for side in ("left", "right")
                   for part in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"))
REPO = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype="<f4").tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def held_commands(reference):
    """Actual50Hz execution of a true30Hz command stream, with no tail padding."""
    rows = np.arange(len(reference), dtype=np.int64)
    indices = ((rows * 30 // 50) * 50 + 29) // 30
    require(bool(np.all(indices < len(reference))), "Sample-and-hold selected an unrecorded tail")
    return reference[indices]


def original_identity(source):
    require(source.get("source_split") == "train", "Only original TRAIN sources are allowed")
    row, record = source["row"], source["source_record"]
    source_id = row["source_scene_episode_index"]
    require(type(source_id) is int and source_id not in HELDOUT
            and source_id == record["source_episode_index"], "Heldout or ambiguous source ancestry")
    commands = np.asarray(source["commands"])
    require(commands.dtype == np.float32 and commands.ndim == 2 and commands.shape[1] == 78
            and len(commands) > 0 and np.isfinite(commands).all(), "Original source requires finite native50Hz float32 commands78")
    paths = [Path(p) for p in source["sources"] if Path(p).suffix == ".parquet"]
    require(len(paths) == 1 and paths[0].is_absolute(), "One bound original TRAIN parquet is required")
    original_sha = sha256(paths[0])
    require(original_sha == source["sources"][str(paths[0])] == record["parquet"]["sha256"], "Original source parquet SHA changed")
    lineage = row["lineage_group_sources"]
    require(row.get("lineage_complete") is True and lineage == [{"source_dataset_key": row["source_dataset_key"],
            "source_episode_index": source_id}], "Original TRAIN ancestor must have complete singleton lineage")
    return commands, source_id, original_sha, lineage


def payload_file(directory, descriptor):
    relative = Path(descriptor["path"])
    require(not relative.is_absolute() and ".." not in relative.parts and relative.suffix == ".npy", "Unsafe command payload path")
    path = (directory / relative).resolve(strict=True)
    require(path.is_relative_to(directory) and sha256(path) == descriptor["sha256"], "Command payload escapes bundle or SHA changed")
    values = np.load(path, allow_pickle=False)
    require(descriptor.get("dtype") == "float32" and descriptor.get("columns") == 78
            and type(descriptor.get("rows")) is int and values.dtype == np.float32
            and values.shape == (descriptor["rows"], 78) and np.isfinite(values).all(), "Command payload must be finite float32 rows78")
    return values, path


def candidate_binding(value):
    names = ("candidate_id", "candidate_spec_sha256", "plan_sha256")
    require(isinstance(value, dict) and all(isinstance(value.get(k), str) and value[k] for k in names),
            "A complete frozen candidate identity is required")
    return {name: value[name] for name in names}


def validate_reference(path, original_source, *, expected_candidate=None, expected_warmup=None,
                       allow_new_execution_candidate=False):
    """Return reference array and independently checked provenance receipt.

    ``original_source`` must come from the producer's strict original loader;
    collectors should independently load its original parquet before calling.
    Optional candidate/startup expectations come from the frozen queue recipe,
    never from this reference's self-description. Reusing a repair on another
    candidate requires the explicit ``allow_new_execution_candidate=True``;
    the original TRAIN source remains strictly bound. Origin success never
    certifies the new execution. No admission is implied.
    """
    path = Path(path).resolve(strict=True)
    document = read(path)
    original, source_id, original_sha, lineage = original_identity(original_source)
    require(document.get("contract") == CONTRACT and document.get("scope") == SCOPE
            and document.get("evaluation_oracle_allowed") is False
            and document.get("no_admission") is True and document.get("dataset_ready") is False,
            "Modified reference is TRAIN-only and cannot grant evaluation/admission")
    require(document.get("source_episode_index") == source_id
            and document.get("parent_source_parquet_sha256") == original_sha
            and document.get("parent_source_commands") == len(original)
            and document.get("original_prefix_rows") == len(original)
            and document.get("lineage_group_sources") == lineage, "Reference parent/source/prefix identity differs")
    require(document.get("command_rate_hz") == 50 and document.get("execution_command_rate_hz") == 30
            and document.get("decoder_hz") == 50 and document.get("physics_hz") == 200
            and document.get("body_token_version") == "body_token_v1_1"
            and document.get("hand_joint_names") == list(HAND_NAMES), "Reference clock or named SONICv1.1 action schema differs")
    reference, command_path = payload_file(path.parent, document["commands"])
    applied, applied_path = payload_file(path.parent, document["actual_executed_commands"])
    require(len(reference) > len(original) and reference.shape == applied.shape, "Reference requires a separately executed new tail")
    require(np.array_equal(reference[:len(original)], original), "Original native50Hz command prefix was modified")
    prefix = document["prefix_verified"]
    require(prefix.get("exact_native50_source_array_equal") is True
            and prefix.get("array_sha256") == array_sha256(original), "Original prefix receipt is stale")
    require(np.array_equal(held_commands(reference), applied), "Reapplying30Hz sample-and-hold does not reproduce actual execution")
    startup = document["startup_recipe"]
    require(startup.get("control") in {"measured_pose_pd", "live_pose_sonic_v1_1_repeat_first_snapshot"}
            and startup.get("seconds") == 1.
            and startup.get("startup_padding_is_synthetic") is (startup.get("control") == "live_pose_sonic_v1_1_repeat_first_snapshot"),
            "Unsupported modified-teacher startup recipe")
    if expected_warmup is not None:
        require(startup["control"] == expected_warmup, "Reference startup differs from frozen TRAIN schedule")
    require(type(allow_new_execution_candidate) is bool, "Candidate reuse permission must be explicit bool")
    binding = candidate_binding(document.get("repair_origin_candidate", document.get("candidate_reset_binding")))
    require(document.get("candidate_reset_binding", binding) == binding, "Origin candidate aliases disagree")
    executed = None
    if expected_candidate is not None:
        executed = candidate_binding(expected_candidate)
        if allow_new_execution_candidate or "source_episode_index" in expected_candidate:
            require(expected_candidate.get("source_episode_index") == source_id, "Execution candidate has a different source")
        require(allow_new_execution_candidate or binding == executed, "Reference belongs to a different candidate reset")
    require(not allow_new_execution_candidate or executed is not None,
            "Explicit new execution requires its frozen candidate identity")
    fixed = document["fixed_encoder"]
    require(fixed.get("sha256") == ENCODER_SHA256 and fixed.get("decoder_sha256") == DECODER_SHA256,
            "Modified teacher must retain fixed official SONICv1.1 identities")
    require(isinstance(document.get("method"), dict) and document["method"].get("kind")
            and isinstance(document.get("repair_inputs_sha256"), dict) and document["repair_inputs_sha256"], "Repair method/input provenance is missing")
    for original_path, expected in document["repair_inputs_sha256"].items():
        require(Path(original_path).is_absolute() and sha256(original_path) == expected, "Repair input identity changed")
    # Tie the exported arrays to actual immutable research execution, rather
    # than accepting two self-consistent arrays with newly invented tail data.
    results = [Path(p) for p in document["repair_inputs_sha256"] if Path(p).name == "result.json"]
    require(len(results) == 1, "One immutable repair result is required")
    result = read(results[0])
    recorded_path = results[0].parent / result["modified_command_schema"]["file"]
    require(str(recorded_path) in document["repair_inputs_sha256"]
            and sha256(recorded_path) == result["modified_command_schema"]["sha256"], "Recorded execution is not bound")
    recorded = np.load(recorded_path, allow_pickle=False)
    require(recorded.dtype == np.float32 and recorded.ndim == 2 and recorded.shape[1] == 78
            and len(applied) <= len(recorded) and np.array_equal(applied, recorded[:len(applied)]),
            "Exported execution is not an exact prefix of recorded actual commands")
    require(candidate_binding(result["candidate_reset_identity"]) == binding, "Repair origin differs from immutable result")
    crop = document.get("recorded_causal_prefix")
    if len(applied) < len(recorded):
        require(isinstance(crop, dict) and crop.get("selected_command_rows") == len(applied)
                and crop.get("original_recorded_command_rows") == len(recorded)
                and crop.get("selected_tail_ticks50") == len(applied) - len(original), "Cropped execution needs an explicit causal-prefix receipt")
    if crop is not None:
        independently_computed = causal_prefix_quality(results[0].parent, result, len(applied), len(recorded))
        require(crop == independently_computed, "Causal-prefix physics receipt changed")
    receipt = {"contract": CONTRACT, "path": str(path), "sha256": sha256(path), "source_episode_index": source_id,
        "parent_source_parquet_sha256": original_sha, "parent_source_commands": len(original),
        "commands_path": str(command_path), "commands_sha256": sha256(command_path), "commands_rows": len(reference),
        "actual_executed_commands_path": str(applied_path), "actual_executed_commands_sha256": sha256(applied_path),
        "original_prefix_rows": len(original), "prefix_exact_independently_verified": True,
        "samplehold_reapplication_exact_verified": True, "startup_recipe": startup, "candidate_reset_binding": binding,
        "repair_origin_candidate": binding, "executed_candidate": executed,
        "new_execution_differs_from_repair_origin": executed is not None and executed != binding,
        "new_execution_explicitly_allowed": allow_new_execution_candidate,
        "execution_requires_independent_admission": True, "recorded_causal_prefix": crop,
        "scope": SCOPE, "evaluation_oracle_allowed": False, "no_admission": True, "dataset_ready": False}
    return reference, receipt


def causal_prefix_quality(diagnostic_dir, result, selected_rows, original_rows):
    """Audit already recorded causal observations; this does not run physics."""
    from handover_contact_quality import evaluate_capture
    rows = [json.loads(line) for line in (diagnostic_dir/"quality_samples.jsonl").read_text().splitlines()]
    require(len(rows) == original_rows and 0 < selected_rows <= original_rows,
            "Quality samples must correspond one-to-one to recorded control ticks")
    require([row.get("completed_steps") for row in rows] == list(range(1, original_rows + 1)),
            "Quality samples must be actual consecutive50Hz teacher observations without warmup")
    quality = evaluate_capture(rows[:selected_rows], initial_target_z_m=result["initial_target_z_m"])
    require(quality.get("eligible_physics") is True, "Cropped causal prefix fails the complete strict physics gate")
    return {"selected_command_rows": selected_rows, "original_recorded_command_rows": original_rows,
        "selected_tail_ticks50": selected_rows - result["source_prefix_ticks50"],
        "last_actual_sim_time_s": rows[selected_rows-1]["sim_time_s"],
        "quality_samples_sha256": sha256(diagnostic_dir/"quality_samples.jsonl"), "strict_gate": quality,
        "new_physics_replay_executed": False, "new_RGB_capture_executed": False,
        "production_replay_and_RTX_admission_required": True}


def export_reference(diagnostic_dir, original_source, output, *, tail_ticks50=None):
    diagnostic_dir = Path(diagnostic_dir).resolve(strict=True)
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink() and not output.resolve().is_relative_to(REPO)
            and not output.resolve().is_relative_to(diagnostic_dir), "Use a fresh external reference bundle")
    result_path = diagnostic_dir / "result.json"; result = read(result_path)
    require(result.get("status") == "modified_teacher_physical_diagnostic_completed"
            and result.get("strict_physical_success") is True and result.get("finite_physics") is True
            and result.get("fell") is False and result.get("all_external_forces_zero") is True
            and result.get("force_guard_steps") == result.get("expected_force_guard_steps")
            and result.get("code_unchanged_during_run") is True and result.get("no_admission") is True,
            "Only completed strict physical research results may export a no-admission reference")
    original, source_id, original_sha, lineage = original_identity(original_source)
    require(source_id == result["source_episode_index"] and result["source_prefix_ticks50"] == len(original), "Repair run parent differs")
    controller_artifacts = result.get("controller_artifact_sha256", {})
    for filename, expected in (("model_encoder.onnx", ENCODER_SHA256), ("model_decoder.onnx", DECODER_SHA256)):
        matches = [value for name, value in controller_artifacts.items() if Path(name).name == filename]
        require(matches == [expected], "Research run did not bind the fixed official encoder/decoder")
    applied_path = diagnostic_dir / result["modified_command_schema"]["file"]
    require(sha256(applied_path) == result["modified_command_schema"]["sha256"], "Actual repair command table changed")
    applied = np.load(applied_path, allow_pickle=False)
    require(applied.dtype == np.float32 and applied.ndim == 2 and applied.shape[1] == 78
            and len(applied) > len(original) and np.isfinite(applied).all(), "Need complete actual modified execution commands")
    reference = applied.copy(); reference[:len(original)] = original
    require(np.array_equal(held_commands(reference), applied), "Original prefix plus actual tail cannot reproduce this execution clock")
    required = ["result.json", "trajectory.npz", "physical_quality.json", "quality_samples.jsonl", "applied_force_samples.jsonl",
                "modified_reference_events.jsonl", result["modified_command_schema"]["file"]]
    inputs = {str(diagnostic_dir/name): sha256(diagnostic_dir/name) for name in required}
    for name in required[1:]:
        require(result["artifact_sha256"].get(name) == inputs[str(diagnostic_dir/name)], "Recorded repair artifact changed")
    causal_prefix = None
    if tail_ticks50 is not None:
        require(type(tail_ticks50) is int and tail_ticks50 > 0, "Tail length must be a positive integer50Hz tick count")
        selected = len(original) + tail_ticks50
        causal_prefix = causal_prefix_quality(diagnostic_dir, result, selected, len(applied))
        applied = applied[:selected].copy()
        reference = reference[:selected].copy()
        require(np.array_equal(held_commands(reference), applied), "Cropped tail cannot reproduce the actual execution clock")
    staging = output.parent / ("." + output.name + ".partial-" + uuid.uuid4().hex)
    staging.mkdir(parents=True, exist_ok=False)
    try:
        for name, values in (("reference_commands_native50.npy", reference), ("actual_executed_commands_native50.npy", applied)):
            np.save(staging/name, values, allow_pickle=False)
        def descriptor(name):
            return {"path": name, "sha256": sha256(staging/name), "rows": len(reference), "columns": 78, "dtype": "float32"}
        padded = result.get("startup_padding_is_synthetic") is True
        startup = {"control": "live_pose_sonic_v1_1_repeat_first_snapshot" if padded else "measured_pose_pd", "seconds": 1.,
            "startup_padding_is_synthetic": padded, "research_receipt_kind": result.get("warmup_receipt", {}).get("kind"),
            "padding_policy": "explicit copies of first actual snapshot at identicalt0; never fabricated past timestamps" if padded else "none"}
        binding = result["candidate_reset_identity"]
        document = {"contract": CONTRACT, "scope": SCOPE, "source_episode_index": source_id,
            "parent_source_parquet_sha256": original_sha, "parent_source_commands": len(original),
            "command_rate_hz": 50, "execution_command_rate_hz": 30, "decoder_hz": 50, "physics_hz": 200,
            "body_token_version": "body_token_v1_1", "hand_joint_names": list(HAND_NAMES),
            "commands": descriptor("reference_commands_native50.npy"),
            "actual_executed_commands": descriptor("actual_executed_commands_native50.npy"),
            "original_prefix_rows": len(original), "prefix_verified": {"exact_native50_source_array_equal": True,
                "array_sha256": array_sha256(original), "basis": "repacked named original source replay_action78, before30Hz hold"},
            "lineage_group_sources": lineage, "split": "train_only", "startup_recipe": startup,
            "candidate_reset_binding": {name: binding[name] for name in ("candidate_id", "candidate_spec_sha256", "plan_sha256")},
            "repair_origin_candidate": {name: binding[name] for name in ("candidate_id", "candidate_spec_sha256", "plan_sha256")},
            "fixed_encoder": {"sha256": ENCODER_SHA256, "decoder_sha256": DECODER_SHA256},
            "method": {"kind": "mode0_left_arm_bounded_IK_terminal_retreat", "parameters": result["parameters"],
                "IK_reference": result.get("IK_reference"), "code_sha256": result["code_sha256"],
                "desired_reference_is_not_measured_history": True, "privileged_geometry_scope": "offline TRAIN teacher only"},
            "repair_inputs_sha256": inputs, "exporter_source_sha256": sha256(__file__),
            "recorded_causal_prefix": causal_prefix,
            "evaluation_oracle_allowed": False, "no_admission": True, "dataset_ready": False,
            "evidence_scope": "Completed CPU physical research; trueRTX recapture and independent admission still required"}
        write(staging/"teacher_reference.json", document)
        validate_reference(staging/"teacher_reference.json", original_source)
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return validate_reference(output/"teacher_reference.json", original_source)[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tail-ticks50", type=int, help="Crop an already recorded successful causal prefix; still requires production replay+RTX")
    args = parser.parse_args()
    from replay_handover_teacher import load_source
    source_id = read(args.diagnostic_dir/"result.json")["source_episode_index"]
    print(json.dumps(export_reference(args.diagnostic_dir, load_source(args.snapshot_root, source_episode=source_id),
                                     args.output, tail_ticks50=args.tail_ticks50)))


if __name__ == "__main__":
    main()
