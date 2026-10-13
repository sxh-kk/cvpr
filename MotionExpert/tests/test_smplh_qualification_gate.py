"""Publication proofs distinguish a new full decision from the retained old failure."""
from copy import deepcopy
import json

import pytest

from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, file_sha256
from MotionExpert.tests.test_smplh_dataset import owned_archive, _json


@pytest.fixture
def qualified_archive(owned_archive, tmp_path):
    archive = SMPLHParameterArchive(owned_archive)
    population = {"episodes": 4, "frames": 364, "sources": {
        "egosuite": {"episodes": 2, "frames": 182}, "egoexo4d_hm": {"episodes": 2, "frames": 182}}}
    proof_root = tmp_path / "qualification-job"
    documents = {
        "latest_me_baseline": {"format": "latest-motionexpert-baseline-v1", "files": {"model.py": "1" * 64}},
        "authorization": {"format": "smplh-requalification-authorization-v1", "scope": "latest_motionexpert_baseline_only",
                          "preserve_original_failed_report": True, "retain_original_numeric_policy": True},
        "parameter_integrity_audit": {"format": "independent-standalone-smplh-parameter-full-audit-v1", "scope": "full",
                                     "passed": False, "parameter_and_source_integrity_passed": True,
                                     "original_egosuite_full_fit_acceptance_passed": False, "old_dense_dependency_absent": True,
                                     "population": population, "binding": deepcopy(archive.binding),
                                     "failures": [{"stage": "original_egosuite_full_fit_acceptance"}]},
        "original_fit_qualification": {"format": "original-smplh-fit-requalification-v1", "scope": "full",
                                       "passed": True, "numeric_quality_passed": True, "data_preservation_passed": True,
                                       "frozen_producer_passed": True, "old_registered_acceptance_passed": False,
                                       "episodes": 2, "frames": 182, "model_sha256": archive.binding["model_sha256"], "failures": []},
        "latest_motionexpert_acceptance": {"format": "latest-motionexpert-acceptance-v1", "scope": "full", "passed": True,
                                           "population": population, "binding": deepcopy(archive.binding), "failures": [],
                                           **{key: {"path": f"/removed-source-job/{key}.json", "sha256": "2" * 64}
                                              for key in ("full_population_probe", "regression", "model_smoke")}},
    }
    proofs = {}
    for name, value in documents.items():
        if name in ("original_fit_qualification", "latest_motionexpert_acceptance"):
            value["latest_baseline_sha256"] = proofs["latest_me_baseline"]["sha256"]
        path = proof_root / f"{name}.json"
        _json(path, value)
        proofs[name] = {"path": str(path), "sha256": file_sha256(path)}
    report = {"format": "qualified-standalone-smplh-parameter-full-audit-v1", "scope": "full", "failures": [],
              **{key: True for key in ("passed", "cleanup_allowed", "parameter_and_source_integrity_passed",
                                      "original_fit_numerical_quality_passed", "latest_motionexpert_accepted", "old_dense_dependency_absent")},
              "old_registered_fit_acceptance_passed": False, "population": population,
              "binding": deepcopy(archive.binding), "proofs": proofs}
    path = proof_root / "qualified.json"
    _json(path, report)
    state_path = owned_archive / ".pipeline/dataset-state.json"
    state = json.loads(state_path.read_text())
    state["acceptance_report"] = str(path)
    _json(state_path, state)
    return owned_archive, path, report


def _change_proof(case, name, change):
    _, report_path, report = case
    path = report_path.parent / report["proofs"][name]["path"]
    proof = json.loads(path.read_text())
    change(proof)
    _json(path, proof)
    report["proofs"][name]["sha256"] = file_sha256(path)
    _json(report_path, report)


def test_qualified_decision_accepts_preserved_original_failure(qualified_archive):
    root, _, report = qualified_archive
    assert SMPLHParameterArchive(root).frames == 364
    assert report["old_registered_fit_acceptance_passed"] is False
    assert report["proofs"]["parameter_integrity_audit"]


@pytest.mark.parametrize("field", ["passed", "cleanup_allowed", "parameter_and_source_integrity_passed",
                                   "original_fit_numerical_quality_passed", "latest_motionexpert_accepted",
                                   "old_dense_dependency_absent", "old_registered_fit_acceptance_passed"])
def test_qualified_decision_requires_every_explicit_flag(qualified_archive, field):
    root, path, report = qualified_archive
    del report[field]
    _json(path, report)
    with pytest.raises(ValueError, match="acceptance|qualification"):
        SMPLHParameterArchive(root)


@pytest.mark.parametrize("change", [lambda r: r.update(scope="expanded"), lambda r: r.update(partial=True),
                                    lambda r: r.update(old_registered_fit_acceptance_passed=True),
                                    lambda r: r.update(original_egosuite_full_fit_acceptance_passed=True),
                                    lambda r: r["population"].update(episodes=3),
                                    lambda r: r["population"]["sources"]["egosuite"].update(frames=181)])
def test_partial_or_rewritten_original_decision_is_rejected(qualified_archive, change):
    root, path, report = qualified_archive
    change(report)
    _json(path, report)
    with pytest.raises(ValueError, match="acceptance|qualification"):
        SMPLHParameterArchive(root)


@pytest.mark.parametrize("key", ["info_sha256", "records_manifest_sha256", "model_sha256", "episodes_metadata_sha256",
                                 "split_metadata_sha256"])
def test_qualified_decision_binds_actual_archive(qualified_archive, key):
    root, path, report = qualified_archive
    report["binding"][key] = "0" * 64
    _json(path, report)
    with pytest.raises(ValueError, match=key):
        SMPLHParameterArchive(root)


@pytest.mark.parametrize("name,change", [
    ("parameter_integrity_audit", lambda p: p.update(scope="pilot")),
    ("parameter_integrity_audit", lambda p: p.update(parameter_and_source_integrity_passed=False)),
    ("parameter_integrity_audit", lambda p: p.update(failures=[{"stage": "source_preservation"}])),
    ("parameter_integrity_audit", lambda p: p.update(original_egosuite_full_fit_acceptance_passed=True)),
    ("parameter_integrity_audit", lambda p: p["binding"].update(model_sha256="0" * 64)),
    ("original_fit_qualification", lambda p: p.update(old_registered_acceptance_passed=True)),
    ("original_fit_qualification", lambda p: p.update(episodes=1)),
    ("original_fit_qualification", lambda p: p.update(model_sha256="0" * 64)),
    ("original_fit_qualification", lambda p: p.update(latest_baseline_sha256="0" * 64)),
    ("latest_motionexpert_acceptance", lambda p: p.update(scope="expanded")),
    ("latest_motionexpert_acceptance", lambda p: p.update(passed=False)),
    ("latest_motionexpert_acceptance", lambda p: p.pop("full_population_probe")),
    ("latest_motionexpert_acceptance", lambda p: p["regression"].update(sha256="not-a-sha")),
    ("latest_motionexpert_acceptance", lambda p: p["binding"].update(info_sha256="0" * 64)),
    ("authorization", lambda p: p.update(retain_original_numeric_policy=False)),
    ("latest_me_baseline", lambda p: p.update(files={})),
])
def test_rehashed_but_invalid_proof_is_rejected(qualified_archive, name, change):
    root, _, _ = qualified_archive
    _change_proof(qualified_archive, name, change)
    with pytest.raises(ValueError, match="qualification"):
        SMPLHParameterArchive(root)


@pytest.mark.parametrize("name", ["parameter_integrity_audit", "original_fit_qualification", "latest_motionexpert_acceptance",
                                 "authorization", "latest_me_baseline"])
def test_every_proof_descriptor_is_required(qualified_archive, name):
    root, path, report = qualified_archive
    del report["proofs"][name]
    _json(path, report)
    with pytest.raises(ValueError, match="proof"):
        SMPLHParameterArchive(root)


def test_available_proof_sha_is_checked_but_relocated_external_evidence_is_optional(qualified_archive):
    root, path, report = qualified_archive
    proof_path = path.parent / report["proofs"]["authorization"]["path"]
    proof_path.write_text(proof_path.read_text() + " ")
    with pytest.raises(ValueError, match="proof SHA mismatch"):
        SMPLHParameterArchive(root)
    proof_path.unlink()
    assert SMPLHParameterArchive(root).frames == 364
    path.unlink()
    assert SMPLHParameterArchive(root).frames == 364


def test_missing_owned_proof_is_rejected(qualified_archive):
    root, path, report = qualified_archive
    report["proofs"]["authorization"]["path"] = str(root / ".pipeline/publication/missing.json")
    _json(path, report)
    with pytest.raises(ValueError, match="owned.*proof"):
        SMPLHParameterArchive(root)


def test_relative_proof_cannot_borrow_external_source(qualified_archive):
    root, path, report = qualified_archive
    report["proofs"]["authorization"]["path"] = "../qualification-job/authorization.json"
    _json(path, report)
    with pytest.raises(ValueError, match="relative proof escapes"):
        SMPLHParameterArchive(root)


def test_owned_proof_symlink_cannot_borrow_external_source(qualified_archive):
    root, path, report = qualified_archive
    relative = ".pipeline/acceptance-evidence/authorization.json"
    target = root / relative
    target.parent.mkdir(parents=True)
    target.symlink_to(report["proofs"]["authorization"]["path"])
    report["proofs"]["authorization"]["path"] = relative
    _json(path, report)
    with pytest.raises(ValueError, match="owned.*proof escapes"):
        SMPLHParameterArchive(root)


def test_owned_relative_evidence_survives_missing_original_job(qualified_archive):
    root, original_path, report = qualified_archive
    for name, descriptor in report["proofs"].items():
        original_proof = original_path.parent / descriptor["path"]
        relative = f".pipeline/acceptance-evidence/{name}.json"
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original_proof.read_bytes())
        original_proof.unlink()
        descriptor["path"] = relative
    owned_report = root / ".pipeline/acceptance-evidence/full-qualified-acceptance.json"
    _json(owned_report, report)
    state_path = root / ".pipeline/dataset-state.json"
    state = json.loads(state_path.read_text())
    state["acceptance_report"] = str(owned_report.relative_to(root))
    _json(state_path, state)
    original_path.unlink()
    assert SMPLHParameterArchive(root).frames == 364
    owned_report.unlink()
    with pytest.raises(ValueError, match="owned.*report"):
        SMPLHParameterArchive(root)
