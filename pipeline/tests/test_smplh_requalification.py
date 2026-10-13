"""Real-file qualification and conditional-cleanup tests; no fake old pass."""
from copy import deepcopy
import importlib
from pathlib import Path
import shutil
import sys

import pytest

from test_cleanup_smplh_dense import accepted, cleanup, blob, put, mutate, parquet

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
qualification = importlib.import_module("requalify_smplh_parameters")
qualified_cleanup = importlib.import_module("cleanup_smplh_qualified")


@pytest.fixture
def qualified(accepted, monkeypatch):
    monkeypatch.setattr(qualification, "legacy", cleanup)
    monkeypatch.setattr(qualified_cleanup, "legacy", cleanup)
    monkeypatch.setattr(qualification, "EXPECTED_EGO", {"episodes": 2, "frames": 8})
    monkeypatch.setattr(qualification, "EXPECTED_WINDOWS", {"eligible": 2, "available": 1, "excluded": 1})
    source, fitjob = accepted["job"], accepted["fitjob"]
    job = source.with_name("qualified"); job.mkdir()
    root, mix = accepted["root"], accepted["mix"]
    originalmodel = accepted["preserved"][-2]
    oldplan = cleanup.read(source / "plan.json")
    oldplan.update(model_path=str(originalmodel), model_sha256=cleanup.MODEL_SHA256)
    put(source / "plan.json", oldplan)
    producer_files = {}
    for index in range(54):
        path = blob(source / "code" / f"producer-{index}.py", b"frozen producer code")
        producer_files[str(path)] = cleanup.sha(path)
    contract = {"old_fitting_frozen_files_preserved": 19, "files": producer_files,
                "plan_sha256": cleanup.sha(source / "plan.json")}
    put(source / "code/producer-contract.json", contract)
    me = mix.parent / "MotionExpert"
    files = {str(blob(me / f"file-{index}.py", b"latest authorized client")): None for index in range(22)}
    files = {path: cleanup.sha(path) for path in files}
    baselinepath = job / "code/latest-me-baseline.json"
    put(baselinepath, {"format": "latest-motionexpert-baseline-v1", "root": str(me), "files": files,
                       "include_suffixes": [".py", ".yaml", ".yml"], "excluded_dirs": ["__pycache__"]})
    fitplan = cleanup.read(fitjob / "plan.json")
    fitplan.update(implementation_sha256=dict(list(producer_files.items())[:19]), model_sha256=cleanup.MODEL_SHA256)
    put(fitjob / "plan.json", fitplan)
    status = {"job_signature": fitplan["job_signature"], "state": "full_candidate_quality_failed", "independent_acceptance_pending": False}
    put(fitjob / "status.json", status)
    policy = {"body_observation_mean_max_m": .01, "body_observation_p95_max_m": .03,
              "native_hand_to_fitted_wrist_p95_max_m": .01, "continuous_local_step_over45deg_fraction_max": .001}
    put(fitjob / "quality-policy.json", policy)
    put(fitjob / "quality-policy-registration.json", {"job_signature": fitplan["job_signature"],
        "policy_sha256": cleanup.sha(fitjob / "quality-policy.json"), "policy_identity": qualification._old_identity(fitjob / "quality-policy.json"),
        "policy_predates_first_output": True, "registered_at_unix_ns": 0})
    put(accepted["fitted"] / "manifest.json", {"actual immutable model": True})
    before = {"content_sha256": {str(accepted["preserved"][-1]): cleanup.sha(accepted["preserved"][-1])},
              "identities": {str(path): qualification._old_identity(path) for path in [*accepted["oldfiles"], accepted["preserved"][5]]}}
    put(fitjob / "production-before.json", before)
    put(fitjob / "motionexpert-before.json", {name: "earlier client sha" for name in files})
    gate = cleanup.read(fitjob / "reports/full-acceptance.json")
    gate.update(passed=False, full_candidate_quality_passed=False,
        failures=["Original v5 or MotionExpert preservation check failed"],
        preservation={"passed": False, "failures": ["Original metadata/reader/training asset changed: " + name for name in files]},
        policy_sha256=cleanup.sha(fitjob / "quality-policy.json"),
        body_observation_residual_m={"count": 176, "mean": .0068, "p95": .0184},
        native_hand_to_fitted_wrist_m={"count": 16, "mean": .0012, "p95": .0032},
        continuous_rotation_pair_count=100000, continuous_local_step_above45deg_count=4,
        continuous_local_step_above45deg_fraction=.00004)
    put(fitjob / "reports/full-acceptance.json", gate)
    authorizationpath = job / "authorization.json"
    put(authorizationpath, {"format": "smplh-requalification-authorization-v1", "scope": "latest_motionexpert_baseline_only",
        "user_request": "根据最新 ME 整体验收", "source_migration_job": str(source),
        "original_fit_acceptance_sha256": cleanup.sha(fitjob / "reports/full-acceptance.json")})
    plan = {**oldplan, "source_migration_job": str(source), "source_plan_sha256": cleanup.sha(source / "plan.json"),
        "source_producer_contract_sha256": cleanup.sha(source / "code/producer-contract.json"),
        "latest_me_baseline": qualification.reference(baselinepath), "authorization": qualification.reference(authorizationpath)}
    put(job / "plan.json", plan)
    record = accepted["records"][-1]; record["metadata"]["training_eligible"] = False
    receiptpath = root / record["data_path"]; receiptpath = receiptpath.parent / "receipt.json"
    receipt = cleanup.read(receiptpath); receipt["record"] = record; put(receiptpath, receipt)
    put(root / ".pipeline/episodes/00000002.json", record)
    raw = deepcopy(accepted["report"])
    raw["binding"]["records_manifest_sha256"] = cleanup.digest(accepted["records"])
    raw["episode_reports"][-1]["receipt_sha256"] = cleanup.sha(receiptpath)
    probe = job / "reports/standalone-fresh-reader.json"
    put(probe, cleanup.read(source / "reports/standalone-fresh-reader.json"))
    raw.update(passed=False, cleanup_allowed=False, original_egosuite_full_fit_acceptance_passed=False,
               failures=[{"stage": qualification.ALLOWED_STAGE, "error": "Historical gate remains failed"}], fresh_reader_report=str(probe))
    for result in raw["episode_reports"]:
        result["passed"] = True
    raw["binding"].update(plan_sha256=cleanup.sha(source / "plan.json"), fresh_reader_sha256=cleanup.sha(probe),
        original_full_fit_acceptance_sha256=cleanup.sha(fitjob / "reports/full-acceptance.json"))
    put(job / "reports/parameter-integrity-audit.json", raw)
    full = {"passed": True, "latest_baseline_sha256": plan["latest_me_baseline"]["sha256"], "expected_episodes": 3,
        "scanned_episodes": 3, "eligible_episodes": 2, "window_episodes": 1, "legitimate_no_window_episodes": 1,
        "rows": [{**{key: r[key] for key in ("episode_index", "source_key", "length", "split")},
            "eligible": r["episode_index"] < 2, "windows": int(r["episode_index"] == 0), "first_last_passed": True,
            "exclusion_reason": "Source feet and ground absent" if r["episode_index"] == 1 else None} for r in accepted["records"]]}
    fullpath = job / "reports/latest-me-full-reader.json"; put(fullpath, full)
    evidence = {"passed": True, "latest_baseline_sha256": plan["latest_me_baseline"]["sha256"]}
    regression = job / "reports/me-regression.json"; smoke = job / "reports/me-smoke.json"
    put(regression, evidence); put(smoke, evidence)
    meproof = {"format": "latest-motionexpert-acceptance-v1", "scope": "full", "passed": True,
        "population": raw["population"], "binding": raw["binding"], "latest_baseline_sha256": plan["latest_me_baseline"]["sha256"],
        "full_population_probe": qualification.reference(fullpath), "regression": qualification.reference(regression), "model_smoke": qualification.reference(smoke)}
    put(job / "reports/latest-motionexpert-acceptance.json", meproof)
    fitproof = qualification.fit_qualification(job); put(job / "reports/original-fit-qualification.json", fitproof)
    report = qualification.qualification_context(job)["report"]; put(job / "reports/full-qualified-acceptance.json", report)
    # These publication artifacts must also survive and remain unchanged.
    published = deepcopy(report)
    for key, descriptor in published["proofs"].items():
        destination = root / ".pipeline/acceptance-evidence" / f"{key}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(descriptor["path"], destination)
        descriptor["path"] = str(destination.relative_to(root))
    published["original_qualification_report_sha256"] = cleanup.sha(job / "reports/full-qualified-acceptance.json")
    pubpath = root / ".pipeline/acceptance-evidence/qualified.json"; put(pubpath, published)
    put(root / ".pipeline/dataset-state.json", {"schema_version": qualified_cleanup.PARAMETER_SCHEMA, "state": "accepted",
        "independent_acceptance_pending": False, "episodes": report["population"]["episodes"], "frames": report["population"]["frames"],
        "sources": report["population"]["sources"], "training_eligible_episodes": 2,
        "acceptance_report": ".pipeline/acceptance-evidence/qualified.json", "acceptance_report_sha256": cleanup.sha(pubpath)})
    def structural_only(ctx, manifest):
        report = dict(format=qualified_cleanup.AVAILABILITY_FORMAT, scope="structural_availability_only", passed=True,
            fresh_process=True, all_population_files_checked=True, numerical_checks_rerun=False, fk_or_model_executed=False,
            training_windows_revalidated=False, motionexpert_revalidated=False, data_pages_read=False,
            root=str(root), population=ctx["population"], model_sha256=ctx["report"]["binding"]["model_sha256"],
            manifest_sha256=manifest["manifest_sha256"], protected_snapshot_sha256=cleanup.digest(manifest["protected_files"]),
            original_numeric_qualification_sha256=ctx["report"]["proofs"]["original_fit_qualification"]["sha256"],
            episodes=len(ctx["records"]), frames=ctx["population"]["frames"], failures=[], pipeline_imports=[], old_source_open_attempts=[],
            blocked_roots=[str(ctx["mix"]), ctx["plan"]["egosuite_fitted_root"], ctx["plan"]["hm_root"]],
            rows=[{**{key:r[key] for key in ("episode_index","source","source_key","length","split")},
                   "motion_footer_rows":r["length"],"vision_footer_rows":r["length"],"passed":True} for r in ctx["records"]])
        qualified_cleanup.verify_availability(report, ctx, manifest)
        path = ctx["job"] / "reports/post-cleanup-availability-fixture.json"; put(path, report)
        return {"path":str(path), "sha256":cleanup.sha(path), "passed":True,"scope":"structural_availability_only"}
    monkeypatch.setattr(qualified_cleanup, "post_availability_probe", structural_only)
    def forbidden_numeric(*args, **kwargs):
        raise AssertionError("Legacy numeric/FK post probe must never run")
    monkeypatch.setattr(cleanup, "post_cleanup_probe", forbidden_numeric)
    return {**accepted, "job": job, "source_job": source, "report": report, "fitproof": fitproof, "me": me, "plan": plan,
            "original_failed_sha": cleanup.sha(fitjob / "reports/full-acceptance.json")}


def test_truthful_qualification_does_not_change_old_failed_report(qualified):
    report = qualification.qualification_context(qualified["job"])["report"]
    assert report["passed"] is True and report["cleanup_allowed"] is True
    assert report["old_registered_fit_acceptance_passed"] is False
    assert report["original_egosuite_full_fit_acceptance_passed"] is False
    assert report["formal_training_ready"] is False
    old = qualified["fitjob"] / "reports/full-acceptance.json"
    assert cleanup.sha(old) == qualified["original_failed_sha"] and cleanup.read(old)["passed"] is False


@pytest.mark.parametrize("change", ["body_mean", "frame_count", "other_preservation"])
def test_only_registered_numeric_complete_and_me_only_failure_is_eligible(qualified, change):
    path = qualified["fitjob"] / "reports/full-acceptance.json"; gate = cleanup.read(path)
    if change == "body_mean": gate["body_observation_residual_m"]["mean"] = .011
    elif change == "frame_count": gate["frames"] -= 1
    else: gate["preservation"]["failures"][0] = "Original archive file changed: raw"
    put(path, gate)
    # Rebind only authorization to isolate the numeric/preservation check.
    authpath = qualified["job"] / "authorization.json"; auth = cleanup.read(authpath)
    auth["original_fit_acceptance_sha256"] = cleanup.sha(path); put(authpath, auth)
    plan = cleanup.read(qualified["job"] / "plan.json"); plan["authorization"] = qualification.reference(authpath); put(qualified["job"] / "plan.json", plan)
    with pytest.raises(ValueError): qualification.fit_qualification(qualified["job"])


def test_changed_authorization_non_me_source_and_added_latest_client_file_refuse(qualified):
    auth = qualified["job"] / "authorization.json"
    old = auth.read_bytes(); auth.write_bytes(old + b" ")
    with pytest.raises(ValueError, match="checksum"): qualification.fit_qualification(qualified["job"])
    auth.write_bytes(old)
    added = blob(qualified["me"] / "new.py")
    with pytest.raises(ValueError, match="file set changed"): qualification.fit_qualification(qualified["job"])
    added.unlink()
    blob(qualified["oldfiles"][0], b"source replaced")
    with pytest.raises(ValueError, match="archive identity changed"): qualification.fit_qualification(qualified["job"])


@pytest.mark.parametrize("kind", ["other_integrity_failure", "partial_me", "wrong_model", "producer_changed"])
def test_qualified_gate_cannot_hide_any_other_failure(qualified, kind):
    job = qualified["job"]
    if kind == "other_integrity_failure":
        path = job / "reports/parameter-integrity-audit.json"; raw = cleanup.read(path)
        raw["failures"].append({"stage": "source_integrity"}); put(path, raw)
    elif kind == "partial_me":
        path = job / "reports/latest-me-full-reader.json"; raw = cleanup.read(path); raw["rows"].pop(); put(path, raw)
        path = job / "reports/latest-motionexpert-acceptance.json"; me = cleanup.read(path)
        me["full_population_probe"] = qualification.reference(job / "reports/latest-me-full-reader.json"); put(path, me)
    elif kind == "wrong_model":
        path = job / "reports/latest-motionexpert-acceptance.json"; me = cleanup.read(path); me["binding"]["model_sha256"] = "0" * 64; put(path, me)
    else:
        blob(qualified["source_job"] / "code/producer-53.py", b"mutated frozen source")
    with pytest.raises(ValueError): qualification.qualification_context(job)


def test_qualified_cleanup_precise_allowlist_and_all_evidence_preserved(qualified):
    before = {str(p): cleanup.sha(p) for p in qualified["preserved"]}
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    assert qualified_cleanup.prepare_manifest(qualified["job"]) == manifest
    assert {Path(e["path"]) for e in manifest["files"]} == set(qualified["oldfiles"])
    assert str(qualified["root"] / ".pipeline/dataset-state.json") in manifest["protected_files"]
    receipt = qualified_cleanup.execute_cleanup(qualified["job"])
    assert receipt["state"] == "complete" and receipt["deleted_files"] == 7
    assert all(not path.exists() for path in qualified["oldfiles"])
    assert all(cleanup.sha(path) == sha for path, sha in before.items())
    assert cleanup.sha(qualified["fitjob"] / "reports/full-acceptance.json") == qualified["original_failed_sha"]
    assert qualified_cleanup.execute_cleanup(qualified["job"]) == receipt


def test_qualified_cleanup_durable_pending_resume_and_unrecorded_loss_refused(qualified, monkeypatch):
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    original = cleanup.secure_unlink
    calls = []
    def interrupted(mix, entry):
        original(mix, entry); calls.append(entry["path"])
        if len(calls) == 2: raise RuntimeError("power interruption after unlink")
    monkeypatch.setattr(cleanup, "secure_unlink", interrupted)
    with pytest.raises(RuntimeError): qualified_cleanup.execute_cleanup(qualified["job"])
    execution = cleanup.read(qualified_cleanup.execution_path(qualified["job"]))
    assert len(execution["deleted"]) == 1 and execution["pending"] == calls[1]
    monkeypatch.setattr(cleanup, "secure_unlink", original)
    assert qualified_cleanup.execute_cleanup(qualified["job"])["deleted_files"] == len(manifest["files"])


def test_absent_source_without_durable_intent_and_changed_publication_are_blockers(qualified):
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    put(qualified["root"] / ".pipeline/dataset-state.json", {"status": "arbitrary changed publication"})
    with pytest.raises(ValueError, match="accepted owned publication|Protected owned artifact"): qualified_cleanup.execute_cleanup(qualified["job"])
    assert all(Path(e["path"]).exists() for e in manifest["files"])


def test_unrecorded_missing_original_and_symlink_refuse_without_deleting(qualified):
    qualified_cleanup.prepare_manifest(qualified["job"])
    qualified["oldfiles"][0].unlink()
    with pytest.raises((ValueError, FileNotFoundError)): qualified_cleanup.execute_cleanup(qualified["job"])
    assert all(path.exists() for path in qualified["oldfiles"][1:])


def test_symlink_candidate_and_tampered_progress_refuse(qualified):
    link = qualified["mix"] / "data/external.parquet"; link.symlink_to(qualified["preserved"][1])
    with pytest.raises(ValueError, match="Symlink"): qualified_cleanup.prepare_manifest(qualified["job"])
    link.unlink()
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    put(qualified_cleanup.execution_path(qualified["job"]), {"format": qualified_cleanup.EXECUTION_FORMAT,
        "manifest_sha256": manifest["manifest_sha256"], "state": "deleting", "deleted": [str(qualified["preserved"][1])], "pending": None})
    with pytest.raises(ValueError, match="progress"): qualified_cleanup.execute_cleanup(qualified["job"])
    assert all(path.exists() for path in qualified["oldfiles"])


def test_retained_old_metadata_cannot_change_during_cleanup(qualified):
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    ctx = qualified_cleanup.check_manifest(qualified["job"], manifest)
    oldjson = qualified["preserved"][5]
    assert str(oldjson) in manifest["retained_original_files"]
    blob(oldjson, b"changed preserved metadata")
    with pytest.raises(ValueError, match="Retained original archive identity changed"):
        qualified_cleanup.check_manifest(qualified["job"], manifest, context=ctx)
    assert all(path.exists() for path in qualified["oldfiles"])


def test_latest_reader_cannot_swap_quarantine_and_eligible_membership(qualified):
    job = qualified["job"]; path = job / "reports/latest-me-full-reader.json"
    full = cleanup.read(path)
    # Preserve all global counts while upgrading the quarantined source.
    full["rows"][1]["eligible"] = False
    full["rows"][2].update(eligible=True, exclusion_reason="fabricated no-window")
    put(path, full)
    path = job / "reports/latest-motionexpert-acceptance.json"; me = cleanup.read(path)
    me["full_population_probe"] = qualification.reference(job / "reports/latest-me-full-reader.json"); put(path, me)
    with pytest.raises(ValueError, match="upgraded quarantine"):
        qualification.latest_me_acceptance(job, qualified["plan"])


@pytest.mark.parametrize("kind", ["pending", "not_accepted", "missing_main", "missing_proof", "modified_proof", "external_proof", "modified_main"])
def test_cleanup_requires_complete_owned_accepted_publication(qualified, kind):
    root = qualified["root"]; statepath = root / ".pipeline/dataset-state.json"
    pubpath = root / ".pipeline/acceptance-evidence/qualified.json"
    if kind == "pending": mutate(statepath, "independent_acceptance_pending", True)
    elif kind == "not_accepted": mutate(statepath, "state", "converted_pending_acceptance")
    elif kind == "missing_main": pubpath.unlink()
    elif kind == "missing_proof": (pubpath.parent / "authorization.json").unlink()
    elif kind == "modified_proof": blob(pubpath.parent / "authorization.json", b"changed authorization bytes")
    else:
        value = cleanup.read(pubpath)
        if kind == "external_proof": value["proofs"]["authorization"]["path"] = str(qualified["job"] / "authorization.json")
        else: value["formal_training_ready"] = True
        put(pubpath, value)
        mutate(statepath, "acceptance_report_sha256", cleanup.sha(pubpath))
    with pytest.raises((ValueError, FileNotFoundError)):
        qualified_cleanup.prepare_manifest(qualified["job"])
    assert all(path.exists() for path in qualified["oldfiles"])


@pytest.mark.parametrize("kind", ["incomplete", "pending", "wrong_count", "wrong_bytes"])
def test_complete_cleanup_receipt_cannot_claim_partial_work(qualified, kind):
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    deleted = [entry["path"] for entry in manifest["files"]]
    receipt = {"format": qualified_cleanup.EXECUTION_FORMAT, "manifest_sha256": manifest["manifest_sha256"],
        "state": "complete", "deleted": deleted, "pending": None, "deleted_files": manifest["file_count"], "bytes_deleted": manifest["bytes"]}
    if kind == "incomplete":
        receipt["deleted"] = deleted[:-1]
        receipt["bytes_deleted"] = sum(entry["bytes"] for entry in manifest["files"][:-1])
    elif kind == "pending": receipt["pending"] = deleted[0]
    elif kind == "wrong_count": receipt["deleted_files"] -= 1
    else: receipt["bytes_deleted"] -= 1
    put(qualified_cleanup.execution_path(qualified["job"]), receipt)
    with pytest.raises(ValueError, match="full manifest|byte progress"):
        qualified_cleanup.progress(qualified["job"], manifest)
    assert all(path.exists() for path in qualified["oldfiles"])
