#!/usr/bin/env python3
"""Qualify immutable fitted data against an explicitly authorized latest client.

Original failed reports, snapshots, thresholds and producer code remain intact.
This script does not refit, convert data, run training, or delete any file.
"""
import argparse
import datetime
import json
from pathlib import Path
import re

import cleanup_smplh_dense as legacy


FIT_FORMAT = "original-smplh-fit-requalification-v1"
QUALIFIED_FORMAT = "qualified-standalone-smplh-parameter-full-audit-v1"
EXPECTED_EGO = {"episodes": 6614, "frames": 26841893}
EXPECTED_ME_DIFFERENCES = 22
ALLOWED_STAGE = "original_egosuite_full_fit_acceptance"
EXPECTED_WINDOWS = {"eligible": 9216, "available": 9205, "excluded": 11}
OWNED_BINDING_KEYS = ("info_sha256", "records_manifest_sha256", "episodes_metadata_sha256",
                      "model_sha256", "hand_basis_sha256", "split_metadata_sha256")
WITNESS_FORMAT = "standalone-smplh-full-audit-execution-witness-v1"
WITNESS_AUDITOR_SHA256 = "596546cb4b497591e1760dda439223d81e738e3ba6449c5c1683dbfc365a3ea9"
WITNESS_WRAPPER_SHA256 = "8b5381022e9cb17d8a93c1991fef67af7b5e7a57eb281d98235ac3a4123978f0"


def proof(value):
    path = legacy.nonsymlink(value["path"])
    if legacy.sha(path) != value["sha256"]:
        raise ValueError("Proof checksum changed: " + str(path))
    return path, legacy.read(path)


def reference(path):
    path = legacy.nonsymlink(path)
    return {"path": str(path), "sha256": legacy.sha(path)}


def _timestamp_ns(value):
    moment = datetime.datetime.fromisoformat(value)
    if moment.tzinfo is None:
        raise ValueError("Execution witness timestamps require a timezone")
    delta = moment - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    return (delta.days * 86400 + delta.seconds) * 10**9 + delta.microseconds * 1000


def verify_execution_witness(ctx, raw, records):
    """Verify reporting recovery without rerunning assertions or reading data pages.

    A newly persisted full reader can only follow all successful native futures
    in the pinned auditor. Pre-run identities exclude its later frozen-input
    failure. Numeric diagnostics never written by that run remain unavailable.
    """
    descriptor = raw["execution_witness_recovery"]
    witness_path, witness = proof(descriptor)
    job, source = ctx["job"], ctx["source"]
    if witness_path != job / "reports/parameter-integrity-execution-witness.json":
        raise ValueError("Recovered execution witness escapes its qualification job")
    true_fields = ("passed", "native_full_assertions_completed", "per_episode_assertions_inferred")
    false_fields = ("original_report_written", "numerical_checks_rerun", "diagnostic_numeric_metrics_recovered")
    if (witness.get("format") != WITNESS_FORMAT or witness.get("scope") != "full"
            or any(witness.get(k) is not True for k in true_fields)
            or any(witness.get(k) is not False for k in false_fields)
            or witness.get("original_execution_returncode") != 1
            or witness.get("population") != ctx["population"] or witness.get("binding") != raw["binding"]
            or raw.get("original_execution_failed") is not True
            or raw.get("numerical_checks_rerun") is not False
            or raw.get("diagnostic_numeric_metrics_recovered") is not False):
        raise ValueError("Recovered report misrepresents the original failed execution")
    expected_paths = dict(log=job / "logs/full-integrity-audit.log",
        execution=job / "reports/parameter-integrity-execution.json", process=job / "integrity-process.json",
        auditor=legacy.AUDITOR, wrapper=job / "code/run_integrity_audit.py",
        fresh_reader=job / "reports/standalone-fresh-reader.json", dry_run=job / "reports/dense-cleanup-dry-run.json")
    if set(witness.get("proofs", {})) != set(expected_paths):
        raise ValueError("Recovered execution evidence is incomplete")
    checks = {witness_path: descriptor["sha256"]}
    for name, path in expected_paths.items():
        item = witness["proofs"][name]
        path = legacy.nonsymlink(path)
        if (set(item) != {"path", "sha256"} or legacy.nonsymlink(item["path"]) != path
                or legacy.sha(path) != item["sha256"]):
            raise ValueError("Recovered execution evidence changed: " + name)
        checks[path] = item["sha256"]
    if (checks[legacy.nonsymlink(legacy.AUDITOR)] != WITNESS_AUDITOR_SHA256
            or checks[expected_paths["wrapper"]] != WITNESS_WRAPPER_SHA256):
        raise ValueError("Recovery does not use the known unchanged full-audit control flow")
    execution = legacy.read(expected_paths["execution"]); process = legacy.read(expected_paths["process"])
    executable = process["command"][0]
    command = [executable, str(legacy.AUDITOR), "--job", str(source), "--workers", "24",
               "--output", str(job / "reports/parameter-integrity-audit.json")]
    if (execution.get("format") != "unchanged-producer-full-integrity-execution-v1"
            or execution.get("returncode") != 1 or execution.get("command") != command
            or process.get("command") != [executable, str(expected_paths["wrapper"]), str(job)]
            or process.get("log") != str(expected_paths["log"])
            or execution.get("data_integrity_passed") is not False
            or execution.get("original_gate_failure_preserved") is not False
            or any(execution.get(k) is not None for k in ("output_sha256", "audited_episodes", "audited_frames"))
            or witness.get("started_at") != process["started_at"]
            or witness.get("completed_at") != execution["completed_at"]
            or witness.get("original_report_path") != command[-1]):
        raise ValueError("Original failed full-audit execution was replaced or partial")
    start = _timestamp_ns(process["started_at"]); finish = _timestamp_ns(execution["completed_at"])
    if finish <= start:
        raise ValueError("Recovered execution has an invalid time window")
    log = expected_paths["log"].read_text()
    lines = [re.sub(r"\x1b\[[0-9;]*m", "", line) for line in log.splitlines()]
    events = []
    for line in lines:
        if line.startswith("{"):
            event = json.loads(line)
            if event.get("event") == "parameter_audit":
                events.append(event)
    count = ctx["population"]["episodes"]
    wanted = [dict(event="parameter_audit", checked=n, expected=count, failures=0) for n in range(100, count + 1, 100)]
    failure = dict(type="ValueError", message="Out of range float values are not JSON compliant: inf", full_audit_line=993, writer_line=879)
    if (events != wanted or log.count("Traceback (most recent call last):") != 1
            or f'File "{legacy.AUDITOR}", line 993, in full_audit' not in log
            or f'File "{legacy.AUDITOR}", line 879, in write_json' not in log
            or "ValueError: " + failure["message"] not in log
            or json.loads(lines[-1]) != execution or witness.get("serialization_failure") != failure
            or witness.get("native_progress") != dict(events=len(wanted), first=100, last=(count // 100) * 100, expected=count, failures=0)):
        raise ValueError("Saved execution log does not prove complete native assertions and reporting-only failure")
    for name in ("fresh_reader", "dry_run"):
        ident = legacy.identity(expected_paths[name])
        if not start < ident["ctime_ns"] <= finish or not start < ident["mtime_ns"] <= finish:
            raise ValueError("Full reader evidence was not produced in this execution")
    if legacy.identity(expected_paths["fresh_reader"])["mtime_ns"] >= legacy.identity(expected_paths["dry_run"])["mtime_ns"]:
        raise ValueError("Saved full-reader/dry-run order differs from frozen control flow")
    dryrun = legacy.read(expected_paths["dry_run"])
    if (dryrun.get("dry_run") is not True or dryrun.get("deletion_executed") is not False
            or dryrun.get("exact_dense_file_count") != len(dryrun.get("paths", []))
            or dryrun.get("bytes") != sum(r["bytes"] for r in dryrun.get("paths", []))):
        raise ValueError("Recovered cleanup evidence is incomplete or destructive")
    legacy.verify_probe(legacy.read(expected_paths["fresh_reader"]), records,
        (ctx["plan"]["egosuite_archive"], ctx["plan"]["egosuite_fitted_root"], ctx["plan"]["hm_root"]))
    root = legacy.nonsymlink(ctx["plan"]["output_root"]); info = legacy.read(root / "meta/info.json")
    fitjob = legacy.nonsymlink(ctx["plan"]["egosuite_fitting_job"])
    required = {legacy.nonsymlink(name) for name in ctx["contract"]["files"]}
    required.update([source / "plan.json", ctx["contract_path"], source / "inventory/combined.json",
        root / "meta/info.json", root / "meta/episodes.parquet", *sorted((root / "meta").glob("splits*.json")),
        legacy.owned(root, info["model"]["path"]), legacy.owned(root, info["hand_basis"]["path"]),
        legacy.nonsymlink(info["hand_basis"]["source_asset"]), legacy.AUDITOR, legacy.READER,
        expected_paths["wrapper"], fitjob / "reports/full-acceptance.json", fitjob / "reports/full-sidecar-audit.json", fitjob / "status.json"])
    inventory = legacy.read(source / "inventory/combined.json")
    for item in inventory.get("source_inventory_identities", {}).values():
        required.add(legacy.nonsymlink(item["path"]))
        if legacy.sha(item["path"]) != item["sha256"]:
            raise ValueError("Recovered source population evidence changed")
    references = legacy.read(source / "references/hm/manifest.json")
    items = references.get("files", references.get("items", [])) if isinstance(references, dict) else references
    for item in items:
        value = item.get("local_path", item.get("saved_path", item.get("file")))
        if value:
            path = Path(value); path = path if path.is_absolute() else source / "references/hm" / path
            required.add(legacy.nonsymlink(path))
            if legacy.sha(path) != item["sha256"]:
                raise ValueError("Recovered pinned source evidence changed")
    data = []; receipt_summary = []; row_reports = {row["episode_index"]: row for row in raw["episode_reports"]}
    for record in records:
        path = root / ".pipeline/episodes" / f"{record['episode_index']:08d}.json"
        receipt_path = legacy.owned(root, record["data_path"]).parent / "receipt.json"
        required.update((path, receipt_path)); receipt = legacy.read(receipt_path)
        row = row_reports[record["episode_index"]]
        if (receipt.get("record") != record or row.get("receipt_sha256") != legacy.sha(receipt_path)
                or row.get("native_assertions_passed") is not True or row.get("diagnostics_available") is not False
                or row.get("assertion_basis") != "frozen_control_flow_witness"
                or row.get("quarantine_basis") != "owned_record_metadata"):
            raise ValueError("Recovered per-episode assertion evidence differs from immutable receipts")
        meta = record.get("metadata", {})
        quarantined = bool(meta.get("training_eligible") is False or meta.get("canonical_training_eligible") is False or meta.get("quarantine_reasons"))
        if row.get("body") != {"quarantined": quarantined}:
            raise ValueError("Recovery invents unavailable numerical diagnostics")
        receipt_summary.append(dict(episode_index=record["episode_index"], receipt_sha256=row["receipt_sha256"]))
        for role in ("motion", "vision"):
            path = legacy.owned(root, record["data_path" if role == "motion" else "vision_path"])
            ident = legacy.identity(path)
            if (ident != receipt["output_identities"][role] or record[role + "_sha256"] != receipt[role]["sha256"]
                    or ident["ctime_ns"] >= start or ident["mtime_ns"] >= start):
                raise ValueError("Owned data changed during or after the original audit")
            data.append(dict(path=str(path), identity=ident, sha256=receipt[role]["sha256"], role=role))
    if (witness.get("owned_data_pre_run_files") != data
            or witness.get("receipt_manifest_sha256") != legacy.digest(receipt_summary)):
        raise ValueError("Recovered owned data population or receipt binding is incomplete")
    metadata = witness.get("metadata_pre_run_files", [])
    actual = [legacy.nonsymlink(item["path"]) for item in metadata]
    if len(actual) != len(set(actual)) or set(actual) != required:
        raise ValueError("Recovered pre-run metadata does not cover every frozen input")
    for item, path in zip(metadata, actual):
        ident = legacy.identity(path)
        if (ident != item["identity"] or ident["ctime_ns"] >= start or ident["mtime_ns"] >= start
                or legacy.sha(path) != item["sha256"]):
            raise ValueError("Frozen inputs may have changed during the original audit: " + str(path))
        # Owned records/receipts are already protected by cleanup's snapshot;
        # keep only retained external evidence in the periodic gate checksum set.
        if not path.is_relative_to(root):
            checks[path] = item["sha256"]
    return checks


def current_me(plan):
    path, baseline = proof(plan["latest_me_baseline"])
    if baseline.get("format") != "latest-motionexpert-baseline-v1":
        raise ValueError("Unknown latest MotionExpert baseline")
    root = legacy.nonsymlink(baseline["root"])
    suffixes = set(baseline.get("include_suffixes", [".py", ".yaml", ".yml"]))
    excluded = set(baseline.get("excluded_dirs", []))
    if suffixes != {".py", ".yaml", ".yml"} or not excluded <= {".git", "__pycache__", "exp", "checkpoints", ".pytest_cache"}:
        raise ValueError("Latest client file population policy differs")
    actual = set()
    for item in root.rglob("*"):
        relative = item.relative_to(root)
        if any(part in excluded for part in relative.parts):
            continue
        legacy.nonsymlink(item)
        if item.is_file() and item.suffix in suffixes:
            actual.add(str(item))
    files = baseline["files"]
    if not files or set(files) != actual:
        raise ValueError("Latest MotionExpert source file set changed")
    for name, checksum in files.items():
        if not legacy.nonsymlink(name).is_relative_to(root) or legacy.sha(name) != checksum:
            raise ValueError("Latest MotionExpert source changed: " + name)
    return path, baseline


def context(job):
    job = legacy.nonsymlink(job); plan = legacy.read(job / "plan.json")
    source = legacy.nonsymlink(plan["source_migration_job"])
    old_plan = legacy.read(source / "plan.json")
    for key in ("output_root", "egosuite_archive", "egosuite_fitting_job", "egosuite_fitted_root", "hm_root", "model_path", "model_sha256"):
        if plan[key] != old_plan[key]:
            raise ValueError("Requalification changes immutable migration routing: " + key)
    rows = legacy.read(source / "inventory/combined.json")["rows"]
    population = legacy.population(rows)
    contract_path = source / "code/producer-contract.json"
    contract = legacy.read(contract_path)
    if (contract.get("old_fitting_frozen_files_preserved") != 19 or len(contract["files"]) != 54
            or legacy.sha(source / "plan.json") != contract["plan_sha256"]
            or plan.get("source_plan_sha256") != legacy.sha(source / "plan.json")
            or plan.get("source_producer_contract_sha256") != legacy.sha(contract_path)):
        raise ValueError("Original frozen migration contract changed")
    for name, checksum in contract["files"].items():
        if legacy.sha(legacy.nonsymlink(name)) != checksum:
            raise ValueError("Original frozen producer/inventory changed: " + name)
    return {"job": job, "plan": plan, "source": source, "old_plan": old_plan, "rows": rows,
            "population": population, "contract_path": contract_path, "contract": contract}


def fit_qualification(job, *, removed=()):
    ctx = context(job); plan = ctx["plan"]
    baseline_path, baseline = current_me(plan)
    authorization_path, authorization = proof(plan["authorization"])
    fitjob = legacy.nonsymlink(plan["egosuite_fitting_job"])
    gate_path = fitjob / "reports/full-acceptance.json"; gate = legacy.read(gate_path)
    if (authorization.get("format") != "smplh-requalification-authorization-v1"
            or authorization.get("scope") != "latest_motionexpert_baseline_only"
            or not isinstance(authorization.get("user_request"), str) or not authorization["user_request"].strip()
            or authorization.get("source_migration_job") != str(ctx["source"])
            or authorization.get("original_fit_acceptance_sha256") != legacy.sha(gate_path)):
        raise ValueError("Current client requalification lacks bound explicit authorization")
    fit_plan_path = fitjob / "plan.json"; fit_plan = legacy.read(fit_plan_path)
    fit_status_path = fitjob / "status.json"; status = legacy.read(fit_status_path)
    audit_path = fitjob / "reports/full-sidecar-audit.json"; audit = legacy.read(audit_path)
    policy_path = fitjob / "quality-policy.json"; policy = legacy.read(policy_path)
    registry_path = fitjob / "quality-policy-registration.json"; registry = legacy.read(registry_path)
    if (fit_plan.get("model_sha256") != plan["model_sha256"] or plan["model_sha256"] != legacy.MODEL_SHA256
            or legacy.sha(legacy.nonsymlink(plan["model_path"])) != plan["model_sha256"]
            or len(fit_plan["implementation_sha256"]) != 19):
        raise ValueError("Original fitting model or frozen producer population changed")
    if (gate.get("format") != "smplh-candidate-acceptance-v2" or gate.get("scope") != "full"
            or gate.get("passed") is not False or gate.get("full_candidate_quality_passed") is not False
            or gate.get("failures") != ["Original v5 or MotionExpert preservation check failed"]
            or status.get("state") != "full_candidate_quality_failed" or status.get("independent_acceptance_pending") is not False
            or not gate["job_signature"] == status["job_signature"] == fit_plan["job_signature"]
            or gate.get("episodes") != EXPECTED_EGO["episodes"] or gate.get("frames") != EXPECTED_EGO["frames"]):
        raise ValueError("Original failure is not exclusively the authorized client snapshot difference")
    before_path = fitjob / "production-before.json"; before = legacy.read(before_path)
    me_before_path = fitjob / "motionexpert-before.json"; me_before = legacy.read(me_before_path)
    prefix = "Original metadata/reader/training asset changed: "
    historical = gate["preservation"].get("failures", [])
    allowed = {message[len(prefix):] for message in historical if message.startswith(prefix)}
    if (gate["preservation"].get("passed") is not False or len(historical) != EXPECTED_ME_DIFFERENCES
            or len(allowed) != EXPECTED_ME_DIFFERENCES or not allowed <= set(me_before)
            or any(not legacy.nonsymlink(name).is_relative_to(legacy.nonsymlink(baseline["root"])) for name in allowed)):
        raise ValueError("Original preservation failures include unapproved data or producer changes")
    for name, checksum in before["content_sha256"].items():
        if legacy.sha(legacy.nonsymlink(name)) != checksum:
            raise ValueError("Original non-client metadata/reader changed: " + name)
    # Only already authorized, durably deleted dense files may be absent during
    # cleanup recovery. The caller derives this set from its signed manifest.
    removed = {str(legacy.nonsymlink(name)) for name in removed}
    for name, wanted in before["identities"].items():
        if name in removed:
            if Path(name).exists():
                raise ValueError("An authorized deleted source file was recreated")
            continue
        value = legacy.identity(name); value["size"] = value.pop("bytes")
        if value != wanted:
            raise ValueError("Original archive identity changed: " + name)
    for name, checksum in fit_plan["implementation_sha256"].items():
        if legacy.sha(legacy.nonsymlink(name)) != checksum:
            raise ValueError("Original fitting producer changed: " + name)
    required = {(r["source_episode_index"], r["source_key"], r["length"]) for r in ctx["rows"] if r["source"] == "egosuite"}
    actual = {(r["episode_index"], r["source_key"], r["frames"]) for r in audit["results"]}
    if (audit.get("passed") is not True or audit.get("episodes_checked") != EXPECTED_EGO["episodes"]
            or audit.get("frames_checked") != EXPECTED_EGO["frames"] or audit.get("selection_sha256")
            or audit.get("completeness_failures") or audit.get("failures") or actual != required or len(audit["results"]) != len(required)
            or not all(r.get("passed") is True for r in audit["results"])
            or audit.get("auditor_sha256") != legacy.sha(legacy.FIT_AUDITOR)
            or (audit.get("shared_train_prior_audit") or {}).get("passed") is not True
            or legacy.nonsymlink(audit["sidecar_root"]) != legacy.nonsymlink(plan["egosuite_fitted_root"])
            or gate["independent_audit_sha256"] != legacy.sha(audit_path)
            or not gate["policy_sha256"] == registry["policy_sha256"] == legacy.sha(policy_path)
            or registry.get("policy_identity") != _old_identity(policy_path)
            or registry.get("job_signature") != fit_plan["job_signature"] or registry.get("policy_predates_first_output") is not True):
        raise ValueError("Original full audit/policy/provenance binding changed")
    execution_path = fitjob / "full-execution.json"; execution = legacy.read(execution_path)
    manifest_path = legacy.nonsymlink(plan["egosuite_fitted_root"]) / "manifest.json"
    first_time = legacy.identity(manifest_path)["ctime_ns"]
    if (execution.get("state") != "complete" or execution.get("scope") != "full"
            or execution.get("episodes") != EXPECTED_EGO["episodes"] or execution.get("frames") != EXPECTED_EGO["frames"]
            or execution.get("job_signature") != fit_plan["job_signature"] or execution.get("output_root") != plan["egosuite_fitted_root"]
            or legacy.identity(policy_path)["ctime_ns"] >= first_time or legacy.identity(registry_path)["ctime_ns"] >= first_time
            or registry.get("registered_at_unix_ns", 0) >= first_time):
        raise ValueError("Original complete execution/pre-output policy registration changed")
    checks = []
    for name, stats, key, threshold in (
            ("body_mean", gate["body_observation_residual_m"], "mean", policy["body_observation_mean_max_m"]),
            ("body_p95", gate["body_observation_residual_m"], "p95", policy["body_observation_p95_max_m"]),
            ("wrist_p95", gate["native_hand_to_fitted_wrist_m"], "p95", policy["native_hand_to_fitted_wrist_p95_max_m"])):
        value = stats[key]
        if not stats["count"] or value is None or not 0 <= value <= threshold:
            raise ValueError("Original registered numeric gate failed: " + name)
        checks.append({"name": name, "value": value, "threshold": threshold})
    pairs = gate["continuous_rotation_pair_count"]; count = gate["continuous_local_step_above45deg_count"]
    if (not pairs or count < 0 or gate["continuous_local_step_above45deg_fraction"] != count / pairs
            or count / pairs > policy["continuous_local_step_over45deg_fraction_max"]):
        raise ValueError("Original continuous rotation gate failed")
    checks.append({"name": "local_rotation_jump_fraction", "value": count / pairs, "threshold": policy["continuous_local_step_over45deg_fraction_max"]})
    legacy.no_writers(ctx["source"], fitjob, legacy.nonsymlink(plan["egosuite_archive"]))
    paths = [gate_path, audit_path, policy_path, registry_path, fit_plan_path, fit_status_path, execution_path,
             manifest_path, before_path, me_before_path, ctx["contract_path"], ctx["source"] / "plan.json", baseline_path, authorization_path,
             legacy.nonsymlink(plan["model_path"])]
    return {"format": FIT_FORMAT, "scope": "full", "passed": True, "episodes": EXPECTED_EGO["episodes"], "frames": EXPECTED_EGO["frames"],
            "job_signature": fit_plan["job_signature"], "model_sha256": plan["model_sha256"], "old_registered_acceptance_passed": False,
            "numeric_quality_passed": True, "data_preservation_passed": True, "frozen_producer_passed": True,
            "numeric_checks": checks, "authorized_historical_motionexpert_differences": sorted(allowed),
            "original_data_identities_checked": len(before["identities"]), "authorized_deleted_identities": sorted(removed),
            "latest_baseline_sha256": legacy.sha(baseline_path), "authorization_sha256": legacy.sha(authorization_path),
            "binding": {str(path): legacy.sha(path) for path in paths}, "motion_truth_certified": False}


def _old_identity(path):
    value = legacy.identity(path); value["size"] = value.pop("bytes")
    return value


def verify_fit_proof(job, *, removed=()):
    path = legacy.nonsymlink(job) / "reports/original-fit-qualification.json"
    saved = legacy.read(path); current = fit_qualification(job, removed=removed)
    # Deletion progress changes availability, never the original proof bindings.
    if any(saved.get(key) != current.get(key) for key in current if key != "authorized_deleted_identities"):
        raise ValueError("Original qualification evidence changed")
    return path, saved


def latest_me_acceptance(job, plan):
    path = legacy.nonsymlink(job) / "reports/latest-motionexpert-acceptance.json"
    report = legacy.read(path); current_me(plan)
    if (report.get("format") != "latest-motionexpert-acceptance-v1" or report.get("scope") != "full"
            or report.get("passed") is not True or report.get("latest_baseline_sha256") != plan["latest_me_baseline"]["sha256"]):
        raise ValueError("Latest MotionExpert acceptance has not passed under this client baseline")
    _, full = proof(report["full_population_probe"])
    for name in ("regression", "model_smoke"):
        _, evidence = proof(report[name])
        if evidence.get("passed") is not True or evidence.get("latest_baseline_sha256") != plan["latest_me_baseline"]["sha256"]:
            raise ValueError("Latest MotionExpert " + name + " failed")
    ctx = context(job)
    expected = {(r["episode_index"], r["source_key"], r["length"], r["split"]) for r in ctx["rows"]}
    actual = {(r["episode_index"], r["source_key"], r["length"], r["split"]) for r in full.get("rows", [])}
    total = ctx["population"]["episodes"]
    if (full.get("passed") is not True or full.get("expected_episodes") != total or full.get("scanned_episodes") != total
            or full.get("latest_baseline_sha256") != plan["latest_me_baseline"]["sha256"]
            or full.get("eligible_episodes") != EXPECTED_WINDOWS["eligible"] or full.get("window_episodes") != EXPECTED_WINDOWS["available"]
            or full.get("legitimate_no_window_episodes") != EXPECTED_WINDOWS["excluded"] or len(full.get("rows", [])) != total or actual != expected
            or report.get("population") != ctx["population"]):
        raise ValueError("Latest MotionExpert reader did not cover the complete original population")
    eligible = [r for r in full["rows"] if r.get("eligible") is True]
    root = legacy.nonsymlink(plan["output_root"])
    records = [legacy.read(path) for path in sorted((root / ".pipeline/episodes").glob("*.json"))]
    record_map = {r["episode_index"]: r for r in records}
    for row in full["rows"]:
        meta = record_map[row["episode_index"]].get("metadata", {})
        wanted = (meta.get("training_eligible") is not False and meta.get("canonical_training_eligible") is not False
                  and not meta.get("quarantine_reasons", []))
        count = row.get("windows")
        if (row.get("eligible") is not wanted or isinstance(count, bool) or not isinstance(count, int) or count < 0
                or (not wanted and count != 0)):
            raise ValueError("Latest client upgraded quarantine or reported invalid window availability")
    available = [r for r in eligible if r.get("windows", 0) > 0]
    exclusions = [r for r in eligible if not r.get("windows", 0)]
    if (len(eligible) != EXPECTED_WINDOWS["eligible"] or len(available) != EXPECTED_WINDOWS["available"] or len(exclusions) != EXPECTED_WINDOWS["excluded"]
            or any(r.get("first_last_passed") is not True for r in available)
            or any(not r.get("exclusion_reason") for r in exclusions)):
        raise ValueError("Latest MotionExpert window masks/exclusions are incompletely verified")
    return path, report


def qualification_context(job, *, removed=()):
    ctx = context(job); fit_path, fit_proof = verify_fit_proof(job, removed=removed)
    me_path, me_proof = latest_me_acceptance(job, ctx["plan"])
    integrity_path = ctx["job"] / "reports/parameter-integrity-audit.json"; raw = legacy.read(integrity_path)
    if (raw.get("format") != "independent-standalone-smplh-parameter-full-audit-v1" or raw.get("scope") != "full"
            or raw.get("population") != ctx["population"] or raw.get("audited_episodes") != ctx["population"]["episodes"] or raw.get("audited_frames") != ctx["population"]["frames"]
            or raw.get("parameter_and_source_integrity_passed") is not True or raw.get("old_dense_dependency_absent") is not True
            or raw.get("passed") is not False or raw.get("original_egosuite_full_fit_acceptance_passed") is not False
            or len(raw.get("failures", [])) != 1 or raw["failures"][0].get("stage") != ALLOWED_STAGE):
        raise ValueError("Full parameter integrity report includes an unapproved failure or incomplete population")
    root = legacy.nonsymlink(ctx["plan"]["output_root"])
    records = [legacy.read(path) for path in sorted((root / ".pipeline/episodes").glob("*.json"))]
    expected = {(r["episode_index"], r["source"], r["source_key"], r["length"], r["split"]) for r in ctx["rows"]}
    actual = {(r["episode_index"], r["source"], r["source_key"], r["frames"], r["split"]) for r in raw["episode_reports"]}
    record_join = {(r["episode_index"], r["source"], r["source_key"], r["length"], r["split"]) for r in records}
    if actual != expected or record_join != expected or len(records) != len(ctx["rows"]) or len(raw["episode_reports"]) != len(ctx["rows"]) or not all(r.get("passed") is True for r in raw["episode_reports"]):
        raise ValueError("Parameter integrity audit differs from exact full source membership")
    binding = raw["binding"]
    fitjob = legacy.nonsymlink(ctx["plan"]["egosuite_fitting_job"])
    if (binding.get("original_full_fit_acceptance_sha256") != legacy.sha(fitjob / "reports/full-acceptance.json")
            or binding.get("original_full_fit_audit_sha256") != legacy.sha(fitjob / "reports/full-sidecar-audit.json")
            or binding.get("original_full_fit_signature") != fit_proof["job_signature"]
            or fit_proof["model_sha256"] != binding["model_sha256"]
            or any(me_proof.get("binding", {}).get(key) != binding[key] for key in OWNED_BINDING_KEYS)):
        raise ValueError("Fit/latest client qualification is bound to another model or archive")
    checks = {ctx["source"] / "plan.json": binding["plan_sha256"], ctx["source"] / "inventory/combined.json": binding["combined_inventory_sha256"],
              legacy.AUDITOR: binding["auditor_sha256"], legacy.READER: binding["reader_sha256"], root / "meta/info.json": binding["info_sha256"],
              root / "meta/episodes.parquet": binding["episodes_metadata_sha256"]}
    if legacy.digest(records) != binding["records_manifest_sha256"]:
        raise ValueError("Owned records changed after full parameter integrity audit")
    info = legacy.read(root / "meta/info.json")
    if info["model"]["model_sha256"] != legacy.MODEL_SHA256 or binding["model_sha256"] != legacy.MODEL_SHA256:
        raise ValueError("Full parameter asset is not the actual16 immutable model")
    checks[legacy.owned(root, info["model"]["path"])] = legacy.MODEL_SHA256
    if info["hand_basis"]["sha256"] != binding["hand_basis_sha256"] or not binding["split_metadata_sha256"]:
        raise ValueError("Owned hand basis or split binding is absent or inconsistent")
    checks[legacy.owned(root, info["hand_basis"]["path"])] = binding["hand_basis_sha256"]
    for name, checksum in binding["split_metadata_sha256"].items():
        path = legacy.owned(root, name)
        if not path.is_relative_to(root / "meta"):
            raise ValueError("Audited split escapes owned metadata")
        checks[path] = checksum
    probe_path = legacy.nonsymlink(raw["fresh_reader_report"])
    if not probe_path.is_relative_to(ctx["job"] / "reports"):
        raise ValueError("Full independent reader report escapes the new qualification job")
    checks[probe_path] = binding["fresh_reader_sha256"]
    legacy.verify_probe(legacy.read(probe_path), records, (ctx["plan"]["egosuite_archive"], ctx["plan"]["egosuite_fitted_root"], ctx["plan"]["hm_root"]))
    for path, checksum in checks.items():
        if legacy.sha(legacy.nonsymlink(path)) != checksum:
            raise ValueError("Full integrity binding changed: " + str(path))
    if "execution_witness_recovery" in raw:
        checks.update(verify_execution_witness(ctx, raw, records))
    report = {**raw, "format": QUALIFIED_FORMAT, "passed": True, "cleanup_allowed": True, "failures": [],
              "original_fit_numerical_quality_passed": True, "latest_motionexpert_accepted": True,
              "old_registered_fit_acceptance_passed": False, "original_egosuite_full_fit_acceptance_passed": False,
              "qualified_authorization_scope": "latest_motionexpert_baseline_only", "formal_training_ready": False}
    report["binding"] = {**binding, "source_migration_plan_sha256": binding["plan_sha256"],
                         "plan_sha256": legacy.sha(ctx["job"] / "plan.json"), "source_migration_producer_contract_sha256": legacy.sha(ctx["contract_path"])}
    report["proofs"] = {"parameter_integrity_audit": reference(integrity_path), "original_fit_qualification": reference(fit_path),
                        "latest_motionexpert_acceptance": reference(me_path), "authorization": ctx["plan"]["authorization"],
                        "latest_me_baseline": ctx["plan"]["latest_me_baseline"]}
    ctx.update(root=root, records=records, report=report, checks=checks)
    # Keep confidence and unavailable finite values intact; freshness is checked
    # using audited receipt identities rather than reinterpreting parameters.
    legacy.owned_snapshot(ctx)
    return ctx


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("fit", "accept")); parser.add_argument("--job", required=True)
    args = parser.parse_args(); job = legacy.nonsymlink(args.job)
    if args.command == "fit":
        report = fit_qualification(job); path = job / "reports/original-fit-qualification.json"
    else:
        report = qualification_context(job)["report"]; path = job / "reports/full-qualified-acceptance.json"
    if path.exists() and legacy.read(path) != report:
        raise ValueError("Do not overwrite different qualification evidence")
    legacy.write(path, report)
    print({"path": str(path), "passed": report["passed"], "scope": report["scope"]})


if __name__ == "__main__":
    main()
