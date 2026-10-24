from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pack_handover_supplement import (validate_admission, collector, legacy,
    validate_pack_selection, template_contract_bindings, CONTRACT_FILES)


def accepted():
    return {
        "contract": collector.AUDIT_CONTRACT, "authority_revision": collector.AUTHORITY_REVISION,
        "status": "independent_capture_audit_passed", "eligible_dataset": True,
        "source_episode_index": 0, "capture_content_sha256": "a" * 64,
        "bindings": {"/actual/capture.json": "b" * 64},
        "physical_gate": {
            "acceptance_gate_version": "handover-physics-v2", "verified_handover_success": True,
            "eligible_physics": True, "capture_validation_errors": [],
            "contact_evidence": {"strict_handover_success": True},
            "final_placement_geometric_validation": {
                "passed": True, "v2_passed": True, "require_3d_evidence": True}},
        "renderer_pose_audit": {"within_predeclared_render_pose_tolerances": True}}


def test_keeps_independent_physics_and_render_admission():
    value = accepted()
    before = deepcopy(value)
    validate_admission(value, allowed_ancestors={0, 9})
    assert value == before


@pytest.mark.parametrize("source", [39, 65, 80, 92, 94, 999])
def test_evaluation_or_unregistered_ancestry_never_becomes_train(source):
    value = accepted(); value["source_episode_index"] = source
    with pytest.raises(ValueError, match="TRAIN ancestor"):
        validate_admission(value, allowed_ancestors={0, 9, 39})


@pytest.mark.parametrize("field", ["passed", "v2_passed", "require_3d_evidence"])
def test_outer_xy_or_contact_flag_cannot_replace_full_3d_gate(field):
    value = accepted()
    value["physical_gate"]["final_placement_geometric_validation"][field] = False
    with pytest.raises(ValueError, match="physical handover"):
        validate_admission(value, allowed_ancestors={0})


def test_physics_only_screening_is_not_a_rendered_training_example():
    value = accepted(); value["renderer_pose_audit"] = {}
    with pytest.raises(ValueError, match="RTX"):
        validate_admission(value, allowed_ancestors={0})


def test_contact_error_rejects_capture_even_when_status_claims_success():
    value = accepted(); value["physical_gate"]["capture_validation_errors"] = ["release missing"]
    with pytest.raises(ValueError, match="physical handover"):
        validate_admission(value, allowed_ancestors={0})


@pytest.mark.parametrize("field", ["contract", "authority_revision"])
def test_obsolete_or_invented_authority_is_rejected(field):
    value = accepted(); value[field] = "obsolete-or-invented"
    with pytest.raises(ValueError, match="independent admission"):
        validate_admission(value, allowed_ancestors={0})


def test_complete_bundle_requires_the_selected_target_and_six_sources():
    items = [{"audit": {"source_episode_index": i}} for i in range(6)]
    validate_pack_selection(items, {"target_accepted_count": 6}, {"capture_target_satisfied": True})
    with pytest.raises(ValueError, match="Complete supplement"):
        validate_pack_selection(items[:1], {"target_accepted_count": 6}, {"capture_target_satisfied": True})
    items[-1]["audit"]["source_episode_index"] = 0
    with pytest.raises(ValueError, match="Complete supplement"):
        validate_pack_selection(items, {"target_accepted_count": 6}, {"capture_target_satisfied": True})


def test_self_consistent_replaced_normalization_still_fails_template_sha(tmp_path):
    for name in CONTRACT_FILES:
        (tmp_path / name).write_text("original")
    ready = {"payload_sha256": {name: legacy.sha(tmp_path / name) for name in CONTRACT_FILES}}
    template_contract_bindings(tmp_path, ready)
    for name in CONTRACT_FILES:
        (tmp_path / name).write_text("internally consistent replacement")
    with pytest.raises(ValueError, match="template payload SHA"):
        template_contract_bindings(tmp_path, ready)
