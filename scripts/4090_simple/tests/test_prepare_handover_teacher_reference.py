"""CPU reference-bundle tests use synthetic inputs, not real demonstrations."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_handover_teacher_reference as reference


def fixture(tmp_path):
    original_file = tmp_path / "original.parquet"
    original_file.write_bytes(b"synthetic identity fixture, not an actual parquet")
    original = (np.arange(64 * 78).reshape(64, 78) / 10000).astype(np.float32)
    lineage = [{"source_dataset_key": "unit-test-not-real", "source_episode_index": 17}]
    source = {"source_split": "train", "commands": original, "sources": {str(original_file): reference.sha256(original_file)},
        "row": {"source_scene_episode_index": 17, "source_dataset_key": "unit-test-not-real", "lineage_complete": True,
                "lineage_group_sources": lineage},
        "source_record": {"source_episode_index": 17, "parquet": {"sha256": reference.sha256(original_file)}}}
    reference_rows = np.vstack((original, np.full((100, 78), .25, np.float32)))
    applied = reference.held_commands(reference_rows)
    diagnostic = tmp_path / "diagnostic"; diagnostic.mkdir()
    np.save(diagnostic / "modified_teacher_applied_commands_native50.npy", applied, allow_pickle=False)
    for name in ("trajectory.npz", "physical_quality.json", "quality_samples.jsonl", "applied_force_samples.jsonl", "modified_reference_events.jsonl"):
        (diagnostic / name).write_bytes(b"synthetic test-only input; no physical success claim")
    result = {"status": "modified_teacher_physical_diagnostic_completed", "strict_physical_success": True,
        "finite_physics": True, "fell": False, "all_external_forces_zero": True, "force_guard_steps": 10,
        "expected_force_guard_steps": 10, "code_unchanged_during_run": True, "no_admission": True,
        "source_episode_index": 17, "source_prefix_ticks50": len(original),
        "modified_command_schema": {"file": "modified_teacher_applied_commands_native50.npy",
            "sha256": reference.sha256(diagnostic / "modified_teacher_applied_commands_native50.npy")},
        "controller_artifact_sha256": {"/unit/model_encoder.onnx": reference.ENCODER_SHA256,
            "/unit/model_decoder.onnx": reference.DECODER_SHA256},
        "artifact_sha256": {p.name: reference.sha256(p) for p in diagnostic.iterdir()},
        "startup_padding_is_synthetic": True, "warmup_receipt": {"kind": "research_repeat_first_snapshot_live_v1_1"},
        "candidate_reset_identity": {"candidate_id": "test", "candidate_spec_sha256": "spec", "plan_sha256": "plan"},
        "parameters": {"horizontal_retreat_mm": 15, "upward_retreat_mm": 10}, "IK_reference": {}, "code_sha256": {}}
    reference.write(diagnostic / "result.json", result)
    return source, diagnostic, tmp_path / "bundle"


def export_fixture(tmp_path):
    source, diagnostic, output = fixture(tmp_path)
    receipt = reference.export_reference(diagnostic, source, output)
    return source, diagnostic, output / "teacher_reference.json", receipt


def recorded_quality_fixture(diagnostic, *, released=True):
    """Synthetic solved-force records exercise gate composition, never data."""
    from handover_contact_quality import SOLVED_STATE_STAGE
    rows = []
    for i in range(164):
        left, right = (False, True) if i < 5 else (True, True) if i < 10 else (True, False) if i < 15 else (False, False)
        if not released and i == 100:
            left = True
        rows.append({"completed_steps": i + 1, "sim_time_s": 1. + (i + 1) / 50,
            "actual_target_xyz_m": [0., 0., .7 if i < 15 else .61], "pelvis_z_m": .75,
            "contacts": {"left_hand": left, "right_hand": right, "container": i >= 15},
            "container_geometry": {"inside_container_footprint": True, "signed_margin_m": .02,
                "collision_vertical_ranges_overlap": True, "container_upright_cosine": 1.,
                "container_support_weight_fraction": 1., "support_measurement": "live_solved_mujoco_contact_force"},
            "solver_stage": SOLVED_STATE_STAGE})
    (diagnostic / "quality_samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = reference.read(diagnostic / "result.json")
    result["initial_target_z_m"] = .55
    result["artifact_sha256"]["quality_samples.jsonl"] = reference.sha256(diagnostic / "quality_samples.jsonl")
    reference.write(diagnostic / "result.json", result)


def test_crop_uses_complete_strict_gate_and_records_no_new_execution(tmp_path):
    source, diagnostic, output = fixture(tmp_path)
    recorded_quality_fixture(diagnostic)
    receipt = reference.export_reference(diagnostic, source, output, tail_ticks50=50)
    values, _ = reference.validate_reference(output / "teacher_reference.json", source)
    assert len(values) == 114
    crop = receipt["recorded_causal_prefix"]
    assert crop["selected_tail_ticks50"] == 50 and crop["strict_gate"]["eligible_physics"]
    assert not crop["new_physics_replay_executed"] and crop["production_replay_and_RTX_admission_required"]


def test_crop_rejects_persistent_contact_even_when_full_original_eventually_passed(tmp_path):
    source, diagnostic, output = fixture(tmp_path)
    recorded_quality_fixture(diagnostic, released=False)
    with pytest.raises(ValueError, match="complete strict physics gate"):
        reference.export_reference(diagnostic, source, output, tail_ticks50=50)
    assert not output.exists()


def test_export_retains_original_reference_and_proves_actual_execution_reapplication(tmp_path):
    source, _, path, receipt = export_fixture(tmp_path)
    values, again = reference.validate_reference(path, source,
        expected_candidate={"candidate_id": "test", "candidate_spec_sha256": "spec", "plan_sha256": "plan"},
        expected_warmup="live_pose_sonic_v1_1_repeat_first_snapshot")
    np.testing.assert_array_equal(values[:64], source["commands"])
    actual = np.load(path.parent / "actual_executed_commands_native50.npy", allow_pickle=False)
    assert not np.array_equal(actual[:64], values[:64])
    np.testing.assert_array_equal(reference.held_commands(values), actual)
    assert again == dict(receipt, executed_candidate=again["repair_origin_candidate"])
    assert receipt["prefix_exact_independently_verified"] and receipt["no_admission"]


@pytest.mark.parametrize("change", [
    {"scope": "MODEL_VAL"}, {"evaluation_oracle_allowed": True}, {"dataset_ready": True},
    {"source_episode_index": 68}, {"parent_source_parquet_sha256": "wrong"},
    {"hand_joint_names": list(reversed(reference.HAND_NAMES))}, {"execution_command_rate_hz": 50},
])
def test_changed_scope_parent_names_or_clock_is_rejected(tmp_path, change):
    source, _, path, _ = export_fixture(tmp_path)
    document = reference.read(path); document.update(change); reference.write(path, document)
    with pytest.raises(ValueError): reference.validate_reference(path, source)


def test_original_prefix_cannot_be_forged_even_after_updating_payload_sha(tmp_path):
    source, _, path, _ = export_fixture(tmp_path)
    document = reference.read(path); payload = path.parent / document["commands"]["path"]
    values = np.load(payload, allow_pickle=False); values[3, 2] += .01; np.save(payload, values, allow_pickle=False)
    document["commands"]["sha256"] = reference.sha256(payload); reference.write(path, document)
    with pytest.raises(ValueError, match="prefix was modified"):
        reference.validate_reference(path, source)


def test_actual_command_table_cannot_change_even_after_updating_payload_sha(tmp_path):
    source, _, path, _ = export_fixture(tmp_path)
    document = reference.read(path); payload = path.parent / document["actual_executed_commands"]["path"]
    values = np.load(payload, allow_pickle=False); values[-1, 0] += .0625; np.save(payload, values, allow_pickle=False)
    document["actual_executed_commands"]["sha256"] = reference.sha256(payload); reference.write(path, document)
    with pytest.raises(ValueError, match="does not reproduce actual execution"):
        reference.validate_reference(path, source)


def test_parent_and_repair_files_are_bound_and_unsafe_payload_is_rejected(tmp_path):
    source, diagnostic, path, _ = export_fixture(tmp_path)
    document = reference.read(path); saved = deepcopy(document)
    document["commands"]["path"] = "../original.parquet"; reference.write(path, document)
    with pytest.raises(ValueError, match="Unsafe"): reference.validate_reference(path, source)
    reference.write(path, saved)
    (diagnostic / "quality_samples.jsonl").write_text("changed")
    with pytest.raises(ValueError, match="Repair input identity changed"): reference.validate_reference(path, source)


def test_frozen_candidate_and_startup_cannot_be_overridden_by_reference(tmp_path):
    source, _, path, _ = export_fixture(tmp_path)
    with pytest.raises(ValueError, match="startup differs"):
        reference.validate_reference(path, source, expected_warmup="measured_pose_pd")
    with pytest.raises(ValueError, match="different candidate"):
        reference.validate_reference(path, source, expected_candidate={"candidate_id": "other", "candidate_spec_sha256": "newspec", "plan_sha256": "newplan"})


def test_reference_reuse_requires_explicit_permission_and_same_source(tmp_path):
    source, _, path, _ = export_fixture(tmp_path)
    current = {"candidate_id": "new-reset", "candidate_spec_sha256": "new-spec", "plan_sha256": "new-plan", "source_episode_index": 17}
    with pytest.raises(ValueError, match="different candidate"):
        reference.validate_reference(path, source, expected_candidate=current)
    _, receipt = reference.validate_reference(path, source, expected_candidate=current, allow_new_execution_candidate=True)
    assert receipt["repair_origin_candidate"]["candidate_id"] == "test"
    assert receipt["executed_candidate"]["candidate_id"] == "new-reset"
    assert receipt["new_execution_differs_from_repair_origin"] and receipt["execution_requires_independent_admission"]
    with pytest.raises(ValueError, match="different source"):
        reference.validate_reference(path, source, expected_candidate=dict(current, source_episode_index=68), allow_new_execution_candidate=True)
    with pytest.raises(ValueError, match="requires its frozen"):
        reference.validate_reference(path, source, allow_new_execution_candidate=True)


def test_changing_both_exported_arrays_cannot_invent_an_unexecuted_tail(tmp_path):
    source, _, path, _ = export_fixture(tmp_path)
    document = reference.read(path)
    native = np.load(path.parent/document["commands"]["path"], allow_pickle=False)
    native[-10:, 0] += .125
    for key, values in (("commands", native), ("actual_executed_commands", reference.held_commands(native))):
        file = path.parent/document[key]["path"]
        np.save(file, values, allow_pickle=False); document[key]["sha256"] = reference.sha256(file)
    reference.write(path, document)
    with pytest.raises(ValueError, match="exact prefix of recorded actual commands"):
        reference.validate_reference(path, source)


def test_origin_candidate_must_match_immutable_research_result(tmp_path):
    source, _, path, _ = export_fixture(tmp_path)
    document = reference.read(path)
    document["candidate_reset_binding"]["candidate_id"] = "invented-origin"
    document["repair_origin_candidate"]["candidate_id"] = "invented-origin"
    reference.write(path, document)
    with pytest.raises(ValueError, match="origin differs"):
        reference.validate_reference(path, source)


def test_crop_cannot_export_nonpositive_or_beyond_recorded_execution(tmp_path):
    source, diagnostic, output = fixture(tmp_path)
    with pytest.raises(ValueError, match="positive integer"):
        reference.export_reference(diagnostic, source, output, tail_ticks50=0)
    # Invalid synthetic rows are rejected before a gate can be claimed.
    with pytest.raises((ValueError, json.JSONDecodeError)):
        reference.export_reference(diagnostic, source, output, tail_ticks50=101)


def test_research_failure_and_modified_original_file_cannot_export(tmp_path):
    source, diagnostic, output = fixture(tmp_path)
    result = reference.read(diagnostic / "result.json"); result["strict_physical_success"] = False
    reference.write(diagnostic / "result.json", result)
    with pytest.raises(ValueError, match="Only completed strict"):
        reference.export_reference(diagnostic, source, output)
    result["strict_physical_success"] = True; reference.write(diagnostic / "result.json", result)
    Path(next(iter(source["sources"]))).write_bytes(b"changed source")
    with pytest.raises(ValueError, match="Original source parquet SHA changed"):
        reference.export_reference(diagnostic, source, output)
