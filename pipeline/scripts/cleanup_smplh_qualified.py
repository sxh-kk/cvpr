#!/usr/bin/env python3
"""Delete superseded dense caches only under a separate authorized qualification.

The historical acceptance remains failed. This tool consumes the newly bound
full-population evidence and uses the original cleanup's filesystem primitives.
Planning is reversible; execution is an explicit lifecycle step.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid

import cleanup_smplh_dense as legacy
import requalify_smplh_parameters as qualification

MANIFEST_FORMAT = "qualified-smplh-dense-cleanup-manifest-v1"
EXECUTION_FORMAT = "qualified-smplh-dense-cleanup-execution-v1"
PARAMETER_SCHEMA = "egomotion-smplh16-bodyparams-hands21-camera-floor-dino5-v1"
AVAILABILITY_SCRIPT = Path(__file__).with_name("probe_smplh_availability.py")
AVAILABILITY_FORMAT = "standalone-smplh-structural-availability-v1"


def manifest_path(job):
    return Path(job) / "reports/dense-cleanup-manifest.json"


def execution_path(job):
    return Path(job) / "reports/dense-cleanup-execution.json"


def progress(job, manifest):
    """Accept an absent source only after a registered durable unlink intent."""
    path = execution_path(job)
    if not path.exists():
        return None, set()
    value = legacy.read(path)
    allowed = {entry["path"] for entry in manifest["files"]}
    deleted = value.get("deleted", [])
    pending = value.get("pending")
    if (value.get("format") != EXECUTION_FORMAT or value.get("manifest_sha256") != manifest["manifest_sha256"]
            or len(deleted) != len(set(deleted)) or not set(deleted) <= allowed
            or (pending is not None and pending not in allowed)
            or value.get("state") not in ("deleting", "complete")):
        raise ValueError("Qualified cleanup progress is not bound to this manifest")
    bypath = {entry["path"]: entry for entry in manifest["files"]}
    if value.get("bytes_deleted") != sum(bypath[name]["bytes"] for name in deleted):
        raise ValueError("Qualified cleanup byte progress is inconsistent")
    if value["state"] == "complete" and (set(deleted) != allowed or pending is not None
            or value.get("deleted_files") != manifest["file_count"] or value["bytes_deleted"] != manifest["bytes"]):
        raise ValueError("Completed qualified cleanup does not cover the full manifest")
    removed = set(deleted)
    if pending is not None and not legacy.nonsymlink(pending).exists():
        removed.add(pending)
    for name in removed:
        if legacy.nonsymlink(name).exists():
            raise ValueError("A deleted dense file was recreated")
    return value, removed


def assert_publication(ctx):
    """Require a durable owned publication, preserving every original proof byte."""
    root = ctx["root"]
    state = legacy.read(root / ".pipeline/dataset-state.json")
    if (state.get("schema_version") != PARAMETER_SCHEMA or state.get("state") != "accepted"
            or state.get("independent_acceptance_pending") is not False
            or state.get("episodes") != ctx["population"]["episodes"] or state.get("frames") != ctx["population"]["frames"]
            or state.get("sources") != ctx["population"]["sources"]
            or state.get("training_eligible_episodes") != qualification.EXPECTED_WINDOWS["eligible"]):
        raise ValueError("Qualified cleanup requires accepted owned publication of the complete population")
    evidence = root / ".pipeline/acceptance-evidence"
    main = legacy.owned(root, state.get("acceptance_report", ""))
    if not main.is_relative_to(evidence):
        raise ValueError("Published qualified report must be owned acceptance evidence")
    if state.get("acceptance_report_sha256", legacy.sha(main)) != legacy.sha(main):
        raise ValueError("Published qualified report checksum differs from publication state")
    published = legacy.read(main)
    original_path = ctx["job"] / "reports/full-qualified-acceptance.json"
    if published.get("original_qualification_report_sha256") != legacy.sha(original_path):
        raise ValueError("Published qualification does not bind the original independent report")
    normalized = dict(published); normalized.pop("original_qualification_report_sha256")
    descriptors = normalized.get("proofs", {})
    if set(descriptors) != set(ctx["report"]["proofs"]):
        raise ValueError("Published qualification lacks all five owned proof descriptors")
    for key, original in ctx["report"]["proofs"].items():
        descriptor = descriptors[key]
        if set(descriptor) != {"path", "sha256"} or descriptor["sha256"] != original["sha256"]:
            raise ValueError("Published proof descriptor differs from independent evidence: " + key)
        path = legacy.owned(root, descriptor["path"])
        if not path.is_relative_to(evidence) or legacy.sha(path) != original["sha256"]:
            raise ValueError("Published proof is missing, altered or outside owned evidence: " + key)
    normalized["proofs"] = ctx["report"]["proofs"]
    if normalized != ctx["report"]:
        raise ValueError("Published qualified report changes fields beyond owned proof paths")
    return main


def _gate_files(ctx):
    checks = dict(ctx["checks"])
    checks[ctx["job"] / "plan.json"] = legacy.sha(ctx["job"] / "plan.json")
    checks[ctx["job"] / "reports/full-qualified-acceptance.json"] = legacy.sha(ctx["job"] / "reports/full-qualified-acceptance.json")
    # All numerical/source/frozen client proofs remain immutable during cleanup.
    fit = legacy.read(ctx["job"] / "reports/original-fit-qualification.json")
    for name, checksum in fit["binding"].items():
        checks[legacy.nonsymlink(name)] = checksum
    for descriptor in ctx["report"]["proofs"].values():
        path, _ = qualification.proof(descriptor)
        checks[path] = descriptor["sha256"]
    me = legacy.read(ctx["job"] / "reports/latest-motionexpert-acceptance.json")
    for key in ("full_population_probe", "regression", "model_smoke"):
        path, _ = qualification.proof(me[key]); checks[path] = me[key]["sha256"]
    for name, checksum in ctx["contract"]["files"].items():
        checks[legacy.nonsymlink(name)] = checksum
    _, baseline = qualification.current_me(ctx["plan"])
    for name, checksum in baseline["files"].items():
        checks[legacy.nonsymlink(name)] = checksum
    fitjob = legacy.nonsymlink(ctx["plan"]["egosuite_fitting_job"])
    before = legacy.read(fitjob / "production-before.json")
    for name, checksum in before["content_sha256"].items():
        checks[legacy.nonsymlink(name)] = checksum
    fitplan = legacy.read(fitjob / "plan.json")
    for name, checksum in fitplan["implementation_sha256"].items():
        checks[legacy.nonsymlink(name)] = checksum
    for name in ("active-child.json", "full-process.json"):
        path = fitjob / name; checks[path] = legacy.sha(path)
    for path in (Path(__file__), Path(qualification.__file__), Path(legacy.__file__), AVAILABILITY_SCRIPT):
        checks[path] = legacy.sha(path)
    return {str(path): checksum for path, checksum in checks.items()}


def gate_context(job, *, removed=()):
    ctx = qualification.qualification_context(job, removed=removed)
    saved = legacy.read(ctx["job"] / "reports/full-qualified-acceptance.json")
    if saved != ctx["report"]:
        raise ValueError("Qualified full acceptance changed or differs from its evidence")
    mix = legacy.nonsymlink(ctx["plan"]["egosuite_archive"])
    roots = [mix / "data", *(mix / "derived" / name for name in legacy.DERIVED_NAMES)]
    if ctx["root"] == mix or any(ctx["root"].is_relative_to(p) or p.is_relative_to(ctx["root"]) for p in roots):
        raise ValueError("Standalone archive overlaps qualified deletion scope")
    ctx.update(mix=mix, fitjob=legacy.nonsymlink(ctx["plan"]["egosuite_fitting_job"]), roots=roots)
    assert_publication(ctx)
    ctx["gate_files"] = _gate_files(ctx)
    return ctx


def protected_snapshot(ctx):
    entries = legacy.owned_snapshot(ctx)
    root = ctx["root"]
    # Publication proofs and its state are independent owned data, not caches.
    folder = root / ".pipeline/acceptance-evidence"
    if folder.exists():
        for parent, dirs, files in os.walk(folder, followlinks=False):
            for name in [*dirs, *files]:
                legacy.nonsymlink(Path(parent) / name)
            for name in files:
                path = legacy.nonsymlink(Path(parent) / name)
                entries[str(path)] = dict(identity=legacy.identity(path), sha256=legacy.sha(path), binary=False)
    state = root / ".pipeline/dataset-state.json"
    if state.exists():
        entries[str(state)] = dict(identity=legacy.identity(state), sha256=legacy.sha(state), binary=False)
    return entries


def prepare_manifest(job):
    job = legacy.nonsymlink(job)
    with legacy.cleanup_lock(job):
        path = manifest_path(job)
        if path.exists():
            value = legacy.read(path); check_manifest(job, value)
            return value
        ctx = gate_context(job); protected = protected_snapshot(ctx)
        files = legacy.candidates(ctx, protected)
        before = legacy.read(ctx["fitjob"] / "production-before.json")
        deletion = {entry["path"] for entry in files}
        retained = {name: wanted for name, wanted in before["identities"].items() if name not in deletion}
        value = dict(format=MANIFEST_FORMAT, job=str(job), output_root=str(ctx["root"]), mix_root=str(ctx["mix"]),
                     population=ctx["population"], created_at=legacy.now(), gate_files=ctx["gate_files"],
                     cleanup_code_sha256=legacy.sha(__file__), qualification_code_sha256=legacy.sha(qualification.__file__),
                     protected_files=protected, retained_original_files=retained, files=files, file_count=len(files), bytes=sum(e["bytes"] for e in files),
                     delete_directories=False, old_registered_fit_acceptance_passed=False,
                     preserved="raw MCAP/HM/Aria/DINO, weights, all jobs and failed reports, JSON, NPZ and owned standalone archive")
        value["manifest_sha256"] = legacy.digest(value); legacy.write(path, value)
        return value


def check_manifest(job, manifest, *, context=None):
    value = dict(manifest); checksum = value.pop("manifest_sha256", None)
    if (value.get("format") != MANIFEST_FORMAT or checksum != legacy.digest(value)
            or value.get("cleanup_code_sha256") != legacy.sha(__file__)
            or value.get("qualification_code_sha256") != legacy.sha(qualification.__file__)
            or value.get("delete_directories") is not False or value.get("old_registered_fit_acceptance_passed") is not False):
        raise ValueError("Qualified cleanup manifest content/code changed")
    _, removed = progress(job, manifest)
    ctx = context if context is not None else gate_context(job, removed=removed)
    if (value["job"] != str(legacy.nonsymlink(job)) or value["output_root"] != str(ctx["root"])
            or value["mix_root"] != str(ctx["mix"]) or value["population"] != ctx["population"]
            or value["gate_files"] != ctx["gate_files"]):
        raise ValueError("Qualified cleanup gate/plan/population changed")
    for name, wanted in value["gate_files"].items():
        if legacy.sha(legacy.nonsymlink(name)) != wanted:
            raise ValueError("Qualified cleanup evidence or frozen source changed: " + name)
    # Detect added/deleted client files, in addition to per-file frozen SHAs.
    qualification.current_me(ctx["plan"])
    legacy.no_writers(ctx["source"], ctx["fitjob"], ctx["mix"])
    legacy.verify_owned(value["protected_files"])
    assert_publication(ctx)
    for name, wanted in value["retained_original_files"].items():
        if qualification._old_identity(name) != wanted:
            raise ValueError("Retained original archive identity changed: " + name)
    seen = set()
    for entry in value["files"]:
        path = legacy.nonsymlink(entry["path"])
        if path.suffix != ".parquet" or not any(path.is_relative_to(root) for root in ctx["roots"]):
            raise ValueError("Qualified cleanup entry escapes exact dense-Parquet allowlist")
        if str(path) in seen or str(path) in value["protected_files"] or entry["bytes"] != entry["identity"]["bytes"]:
            raise ValueError("Qualified cleanup duplicates or targets a protected artifact")
        seen.add(str(path))
    if len(seen) != value["file_count"] or sum(e["bytes"] for e in value["files"]) != value["bytes"]:
        raise ValueError("Qualified cleanup file population changed")
    return ctx


def verify_availability(report, ctx, manifest):
    """Do not confuse post-delete file access with repeated numeric validation."""
    fields = ("episode_index", "source", "source_key", "length", "split")
    expected = {tuple(record[key] for key in fields) for record in ctx["records"]}
    rows = report.get("rows", [])
    actual = {tuple(row[key] for key in fields) for row in rows}
    blocked = {str(legacy.nonsymlink(path)) for path in (ctx["mix"], ctx["plan"]["egosuite_fitted_root"], ctx["plan"]["hm_root"])}
    if (report.get("format") != AVAILABILITY_FORMAT or report.get("scope") != "structural_availability_only"
            or report.get("passed") is not True or report.get("fresh_process") is not True or report.get("all_population_files_checked") is not True
            or any(report.get(key) is not False for key in ("numerical_checks_rerun", "fk_or_model_executed", "training_windows_revalidated", "motionexpert_revalidated", "data_pages_read"))
            or report.get("root") != str(ctx["root"]) or report.get("population") != ctx["population"]
            or report.get("model_sha256") != ctx["report"]["binding"]["model_sha256"]
            or report.get("manifest_sha256") != manifest["manifest_sha256"]
            or report.get("protected_snapshot_sha256") != legacy.digest(manifest["protected_files"])
            or report.get("original_numeric_qualification_sha256") != ctx["report"]["proofs"]["original_fit_qualification"]["sha256"]
            or report.get("episodes") != len(ctx["records"]) or report.get("frames") != ctx["population"]["frames"]
            or actual != expected or len(rows) != len(ctx["records"]) or report.get("failures")
            or report.get("old_source_open_attempts") != [] or report.get("pipeline_imports") != []
            or {str(legacy.nonsymlink(path)) for path in report.get("blocked_roots", [])} != blocked
            or any(row.get("passed") is not True or row.get("motion_footer_rows") != row["length"] or row.get("vision_footer_rows") != row["length"] for row in rows)):
        raise ValueError("Post-cleanup availability proof lacks its complete structural-only contract")


def post_availability_probe(ctx, manifest):
    fields = ("episode_index", "source", "source_key", "length", "split", "data_path", "vision_path")
    envelope = dict(root=str(ctx["root"]), records=[{key: r[key] for key in fields} for r in ctx["records"]],
        population=ctx["population"], model_sha256=ctx["report"]["binding"]["model_sha256"],
        protected_files=manifest["protected_files"], manifest_sha256=manifest["manifest_sha256"],
        blocked_roots=[str(ctx["mix"]), ctx["plan"]["egosuite_fitted_root"], ctx["plan"]["hm_root"]],
        original_numeric_qualification_sha256=ctx["report"]["proofs"]["original_fit_qualification"]["sha256"])
    result = subprocess.run([sys.executable, str(AVAILABILITY_SCRIPT)], input=legacy.canonical(envelope).decode(), capture_output=True, text=True)
    output = ctx["job"] / "reports" / ("post-cleanup-availability-" + uuid.uuid4().hex + ".json")
    try: report = json.loads(result.stdout)
    except ValueError as error:
        legacy.write(output, dict(format=AVAILABILITY_FORMAT, scope="structural_availability_only", passed=False,
                                  failures=[(result.stderr or result.stdout)[-4000:]], numerical_checks_rerun=False))
        raise ValueError("Post-cleanup availability subprocess failed; evidence: " + str(output)) from error
    legacy.write(output, report)  # Unique attempts preserve every failure report.
    if result.returncode:
        raise ValueError("Post-cleanup owned-file availability failed; evidence: " + str(output))
    verify_availability(report, ctx, manifest)
    return dict(path=str(output), sha256=legacy.sha(output), passed=True, scope="structural_availability_only")


def execute_cleanup(job):
    job = legacy.nonsymlink(job)
    with legacy.cleanup_lock(job):
        manifest = legacy.read(manifest_path(job)); ctx = check_manifest(job, manifest)
        receipt, _ = progress(job, manifest)
        if receipt is None:
            receipt = dict(format=EXECUTION_FORMAT, manifest_sha256=manifest["manifest_sha256"], state="deleting",
                           deleted=[], pending=None, started_at=legacy.now(), bytes_deleted=0)
        bypath = {entry["path"]: entry for entry in manifest["files"]}
        # Preflight every remaining file before even one additional unlink.
        for entry in manifest["files"]:
            target = legacy.nonsymlink(entry["path"])
            if entry["path"] in receipt["deleted"]:
                if target.exists():
                    raise ValueError("A deleted dense file was recreated")
            elif not target.exists():
                if receipt["pending"] != entry["path"]:
                    raise ValueError("Dense file disappeared without durable cleanup progress")
            elif legacy.identity(target) != entry["identity"]:
                raise ValueError("A remaining dense file changed after qualified registration")
        if receipt["state"] == "complete":
            probe = receipt.get("post_cleanup_probe", {})
            if probe.get("passed") is not True or legacy.sha(legacy.nonsymlink(probe["path"])) != probe.get("sha256"):
                raise ValueError("Completed cleanup post-probe changed")
            verify_availability(legacy.read(probe["path"]), ctx, manifest)
            return receipt
        for number, entry in enumerate(manifest["files"]):
            if entry["path"] in receipt["deleted"]:
                continue
            if number % 100 == 0:
                check_manifest(job, manifest, context=ctx)
            receipt.update(state="deleting", pending=entry["path"], updated_at=legacy.now())
            legacy.write(execution_path(job), receipt)
            if Path(entry["path"]).exists():
                legacy.secure_unlink(ctx["mix"], entry)
            receipt["deleted"].append(entry["path"])
            receipt.update(pending=None, bytes_deleted=sum(bypath[p]["bytes"] for p in receipt["deleted"]), updated_at=legacy.now())
            legacy.write(execution_path(job), receipt)
        check_manifest(job, manifest, context=ctx)
        probe = post_availability_probe(ctx, manifest)
        check_manifest(job, manifest, context=ctx)
        receipt.update(state="complete", post_cleanup_probe=probe, completed_at=legacy.now(), deleted_files=len(receipt["deleted"]))
        legacy.write(execution_path(job), receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("command", choices=("plan", "execute")); parser.add_argument("--job", required=True)
    args = parser.parse_args()
    value = prepare_manifest(args.job) if args.command == "plan" else execute_cleanup(args.job)
    print(json.dumps({key: value[key] for key in ("format", "file_count", "bytes", "manifest_sha256", "state", "deleted_files", "bytes_deleted") if key in value}))


if __name__ == "__main__":
    main()
