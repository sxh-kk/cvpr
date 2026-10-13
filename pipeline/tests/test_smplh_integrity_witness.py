"""Reporting-only recovery: no numeric audit, data-page read or model call."""
from copy import deepcopy
import datetime
import json
from pathlib import Path
import time

import pytest

from test_cleanup_smplh_dense import accepted, cleanup, put
from test_smplh_requalification import qualified, qualification


@pytest.fixture
def witness_case(qualified, monkeypatch):
    job, root, source = qualified["job"], qualified["root"], qualified["source_job"]
    info = cleanup.read(root / "meta/info.json")
    info["hand_basis"]["source_asset"] = str(root / info["hand_basis"]["path"])
    put(root / "meta/info.json", info)
    put(source / "references/hm/manifest.json", {"files": []})
    wrapper = job / "code/run_integrity_audit.py"; wrapper.write_text("known fixture control flow")
    monkeypatch.setattr(qualification, "WITNESS_AUDITOR_SHA256", cleanup.sha(cleanup.AUDITOR))
    monkeypatch.setattr(qualification, "WITNESS_WRAPPER_SHA256", cleanup.sha(wrapper))
    ctx = qualification.context(job)
    records = qualified["records"]
    raw = deepcopy(cleanup.read(job / "reports/parameter-integrity-audit.json"))
    raw.update(original_execution_failed=True, numerical_checks_rerun=False, diagnostic_numeric_metrics_recovered=False)
    raw["binding"]["info_sha256"] = cleanup.sha(root / "meta/info.json")
    required = {Path(name) for name in ctx["contract"]["files"]}
    required.update([source / "plan.json", ctx["contract_path"], source / "inventory/combined.json",
        root / "meta/info.json", root / "meta/episodes.parquet", *sorted((root / "meta").glob("splits*.json")),
        root / info["model"]["path"], root / info["hand_basis"]["path"], cleanup.AUDITOR, cleanup.READER, wrapper,
        qualified["fitjob"] / "reports/full-acceptance.json", qualified["fitjob"] / "reports/full-sidecar-audit.json",
        qualified["fitjob"] / "status.json"])
    data = []; receipt_summary = []
    for record, row in zip(records, raw["episode_reports"]):
        path = root / ".pipeline/episodes" / f"{record['episode_index']:08d}.json"
        receipt_path = (root / record["data_path"]).parent / "receipt.json"
        required.update((path, receipt_path)); receipt = cleanup.read(receipt_path)
        meta = record.get("metadata", {})
        quarantined = bool(meta.get("training_eligible") is False or meta.get("canonical_training_eligible") is False or meta.get("quarantine_reasons"))
        row.update(native_assertions_passed=True, diagnostics_available=False, assertion_basis="frozen_control_flow_witness",
            quarantine_basis="owned_record_metadata", body={"quarantined": quarantined})
        receipt_summary.append(dict(episode_index=record["episode_index"], receipt_sha256=row["receipt_sha256"]))
        for role in ("motion", "vision"):
            file = root / record["data_path" if role == "motion" else "vision_path"]
            data.append(dict(path=str(file), sha256=receipt[role]["sha256"], identity=cleanup.identity(file), role=role))
    metadata = [dict(path=str(path), sha256=cleanup.sha(path), identity=cleanup.identity(path)) for path in sorted(required)]
    time.sleep(.01)
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    # Filesystem timestamp precision can lag the wall clock by one millisecond.
    time.sleep(.01)
    fresh = job / "reports/standalone-fresh-reader.json"
    put(fresh, cleanup.read(fresh))
    time.sleep(.01)
    dry = job / "reports/dense-cleanup-dry-run.json"
    put(dry, {"dry_run": True, "deletion_executed": False, "exact_dense_file_count": 0, "bytes": 0, "paths": []})
    time.sleep(.01)
    finish = datetime.datetime.now(datetime.timezone.utc).isoformat()
    executable = "/fixture/python"
    command = [executable, str(cleanup.AUDITOR), "--job", str(source), "--workers", "24", "--output", str(job / "reports/parameter-integrity-audit.json")]
    execution_path = job / "reports/parameter-integrity-execution.json"
    execution = dict(format="unchanged-producer-full-integrity-execution-v1", returncode=1, command=command,
        data_integrity_passed=False, original_gate_failure_preserved=False, audited_episodes=None, audited_frames=None,
        output_sha256=None, completed_at=finish)
    put(execution_path, execution)
    process = job / "integrity-process.json"; log = job / "logs/full-integrity-audit.log"
    put(process, dict(command=[executable, str(wrapper), str(job)], started_at=start, log=str(log)))
    log.parent.mkdir(exist_ok=True)
    log.write_text('Traceback (most recent call last):\n'
        f'  File "{cleanup.AUDITOR}", line 993, in full_audit\n'
        f'  File "{cleanup.AUDITOR}", line 879, in write_json\n'
        'ValueError: Out of range float values are not JSON compliant: inf\n'
        '\x1b[0m' + json.dumps(execution) + '\n')
    witness = dict(format=qualification.WITNESS_FORMAT, scope="full", passed=True, native_full_assertions_completed=True,
        per_episode_assertions_inferred=True, original_report_written=False, original_execution_returncode=1,
        numerical_checks_rerun=False, diagnostic_numeric_metrics_recovered=False, population=ctx["population"], binding=raw["binding"],
        original_report_path=command[-1], started_at=start, completed_at=finish,
        metadata_pre_run_files=metadata, owned_data_pre_run_files=data, receipt_manifest_sha256=cleanup.digest(receipt_summary),
        serialization_failure=dict(type="ValueError", message="Out of range float values are not JSON compliant: inf", full_audit_line=993, writer_line=879),
        native_progress=dict(events=0, first=100, last=0, expected=3, failures=0),
        proofs={key: qualification.reference(path) for key, path in dict(log=log, execution=execution_path, process=process,
            auditor=cleanup.AUDITOR, wrapper=wrapper, fresh_reader=fresh, dry_run=dry).items()})
    path = job / "reports/parameter-integrity-execution-witness.json"; put(path, witness)
    raw["execution_witness_recovery"] = qualification.reference(path)
    return dict(ctx=ctx, raw=raw, records=records, witness=witness, witness_path=path, log=log, execution_path=execution_path,
        old_failed=qualified["fitjob"] / "reports/full-acceptance.json", old_failed_sha=qualified["original_failed_sha"])


def test_complete_witness_preserves_failed_execution_and_never_hashes_data(witness_case, monkeypatch):
    case = witness_case
    data = {row["path"] for row in case["witness"]["owned_data_pre_run_files"]}
    original_sha = cleanup.sha
    def metadata_only_sha(path):
        assert str(path) not in data, "Recovery must not rehash motion/vision or read their data pages"
        return original_sha(path)
    monkeypatch.setattr(cleanup, "sha", metadata_only_sha)
    checks = qualification.verify_execution_witness(case["ctx"], case["raw"], case["records"])
    assert case["witness_path"] in checks
    assert cleanup.read(case["execution_path"])["returncode"] == 1
    assert original_sha(case["old_failed"]) == case["old_failed_sha"]
    assert cleanup.read(case["old_failed"])["passed"] is False


@pytest.mark.parametrize("change", ["replaced_log", "missing_frozen_metadata"])
def test_recovery_rejects_tampered_or_missing_evidence(witness_case, change):
    case = witness_case
    witness = case["witness"]
    if change == "replaced_log":
        case["log"].write_text(case["log"].read_text().replace("line 993", "line 962"))
        witness["proofs"]["log"] = qualification.reference(case["log"])
    else:
        witness["metadata_pre_run_files"].pop()
    put(case["witness_path"], witness)
    case["raw"]["execution_witness_recovery"] = qualification.reference(case["witness_path"])
    with pytest.raises(ValueError):
        qualification.verify_execution_witness(case["ctx"], case["raw"], case["records"])
