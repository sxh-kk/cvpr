#!/usr/bin/env python3
"""Manifest and remove only accepted, superseded dense-body Parquet caches.

This tool never relaxes the registered full-population quality gates. It keeps
raw inputs, weights, metadata and research evidence. Planning and execution are
separate; interrupted deletion resumes only through its durable pending record.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid

import pyarrow.parquet as pq


MODEL_SHA256 = "be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947"
EXPECTED_POPULATION = {"egosuite": {"episodes": 6614, "frames": 26841893},
                       "egoexo4d_hm": {"episodes": 2649, "frames": 11295991}}
DERIVED_NAMES = ("egosuite-native-dense22-v1", "smplh-body22-sequence-v2",
                 "egosuite-native-dense22-export-smoke-agent-20261011",
                 "egosuite-native-dense22-pilot-20261011T045036Z")
MANIFEST_FORMAT = "accepted-smplh-dense-cleanup-manifest-v1"
EXECUTION_FORMAT = "accepted-smplh-dense-cleanup-execution-v1"
WORKSPACE = Path(__file__).resolve().parents[2]
AUDITOR = Path(__file__).with_name("audit_smplh_parameters.py")
READER = Path(__file__).resolve().parents[1] / "src/egomotion_pipeline/smplh_parameters.py"
FIT_AUDITOR = Path(__file__).with_name("audit_smplh_sidecars.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def nonsymlink(path):
    path = Path(path)
    if ".." in path.parts:
        raise ValueError("Path traversal is forbidden")
    path = path.absolute()
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError("Symlink cleanup dependencies are forbidden: " + str(parent))
    return path


def owned(root, value):
    value = Path(value)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError("Owned dataset dependencies require relative paths")
    result = nonsymlink(root / value)
    if not result.is_relative_to(root):
        raise ValueError("Owned dependency escapes standalone archive")
    return result


def identity(path):
    path = nonsymlink(path)
    value = path.lstat()
    if not stat.S_ISREG(value.st_mode):
        raise ValueError("Expected a regular file: " + str(path))
    return dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                device=value.st_dev, inode=value.st_ino)


def read(path):
    path = nonsymlink(path)
    if not path.is_file():
        raise ValueError("Required acceptance/provenance is missing: " + str(path))
    return json.loads(path.read_text())


def write(path, value):
    path = nonsymlink(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temp.open("wb") as stream:
        stream.write(canonical(value) + b"\n"); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def live_processes():
    processes = []
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid():
            continue
        try:
            state = (folder / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state in ("Z", "X"):
                continue
            command = (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            processes.append({"pid": int(folder.name), "command": command, "state": state})
        except FileNotFoundError:
            continue
        except PermissionError:
            processes.append({"pid": int(folder.name), "command": None, "state": "unknown"})
    return processes


def no_writers(job, fitjob, mix):
    processes = live_processes()
    scopes = (str(job), str(fitjob), str(mix))
    scripts = ("process_smplh_dataset.py", "process_smplh_parameters.py", "run_full_candidate.py")
    for process in processes:
        cmd = process["command"] or ""
        if any(name in cmd for name in scripts) and any(scope in cmd for scope in scopes):
            raise ValueError("A dataset/fitting writer or its wrapper remains active: " + str(process["pid"]))
    # A stale PID is harmless only when a live reused PID has a readable,
    # demonstrably unrelated command. Never kill or signal the process.
    for name in ("active-child.json", "full-process.json"):
        record = read(fitjob / name)
        pid = record.get("pid")
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise ValueError("Old fitting process identity is invalid")
        for process in processes:
            if process["pid"] == pid and (not process["command"] or str(fitjob) in process["command"]):
                raise ValueError("Old fitting process has not demonstrably exited: " + str(pid))


def population(rows):
    seen, indices, groups, operators, sources = set(), set(), {}, {}, {}
    for row in rows:
        key = (row["source"], row["source_key"])
        if key in seen or row["episode_index"] in indices:
            raise ValueError("Duplicate source/group population")
        seen.add(key); indices.add(row["episode_index"])
        if row["split"] not in ("train", "val", "test") or row["length"] <= 0:
            raise ValueError("Invalid split or frame population")
        item = sources.setdefault(row["source"], dict(episodes=0, frames=0))
        item["episodes"] += 1; item["frames"] += row["length"]
        for group in row.get("split_group_ids", []):
            groups.setdefault((row["source"], group), set()).add(row["split"])
        operator = (row.get("operator_id") or row.get("metadata", {}).get("operator_id")
                    or row.get("source_record", {}).get("metadata", {}).get("operator_id"))
        if operator:
            operators.setdefault((row["source"], operator), set()).add(row["split"])
    if sources != EXPECTED_POPULATION or indices != set(range(len(rows))):
        raise ValueError("Full registered source/frame population is incomplete or changed")
    if any(len(values) > 1 for values in [*groups.values(), *operators.values()]):
        raise ValueError("Participant/capture/operator group leaks across splits")
    ordered = sorted(rows, key=lambda r: r["episode_index"])
    offset = 0
    for row in ordered:
        if row["dataset_from_index"] != offset or row["dataset_to_index"] != offset + row["length"]:
            raise ValueError("Full global frame population contains gaps or overlaps")
        offset += row["length"]
    return dict(episodes=len(rows), frames=offset, sources=sources,
                published_source_groups_disjoint=True, cross_source_person_identity_linked=False,
                physical_human_identity_independently_certified=False)


def verify_probe(probe, records, blocked_roots):
    expected = {(r["episode_index"], r["source_key"], r["length"], r["split"]) for r in records}
    actual = {(r["episode_index"], r["source_key"], r["length"], r["split"]) for r in probe.get("rows", [])}
    blocked = {str(nonsymlink(path)) for path in blocked_roots}
    if (probe.get("passed") is not True or probe.get("fresh_process") is not True
            or probe.get("all_population_interfaces_checked") is not True
            or probe.get("episodes") != len(records) or probe.get("old_source_open_attempts") != []
            or {str(nonsymlink(path)) for path in probe.get("blocked_roots", [])} != blocked
            or len(probe.get("rows", [])) != len(records) or actual != expected):
        raise ValueError("Standalone probe lacks full fresh-process blocked-source membership verification")


def gate_context(job):
    job = nonsymlink(job); plan_path = job / "plan.json"; plan = read(plan_path)
    mix, root, fitjob = [nonsymlink(plan[key]) for key in
                         ("egosuite_archive", "output_root", "egosuite_fitting_job")]
    roots = [mix / "data", *(mix / "derived" / name for name in DERIVED_NAMES)]
    if root == mix or any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in roots):
        raise ValueError("Standalone root overlaps deletion scope")
    combined_path = job / "inventory/combined.json"; rows = read(combined_path)["rows"]
    expected = population(rows)
    report_path = job / "reports/full-parameter-acceptance.json"; report = read(report_path)
    if (report.get("format") != "independent-standalone-smplh-parameter-full-audit-v1"
            or report.get("scope") != "full" or report.get("population") != expected
            or report.get("audited_episodes") != expected["episodes"] or report.get("audited_frames") != expected["frames"]
            or report.get("failures") or not all(report.get(key) is True for key in
                ("passed", "cleanup_allowed", "parameter_and_source_integrity_passed",
                 "original_egosuite_full_fit_acceptance_passed", "old_dense_dependency_absent"))):
        raise ValueError("Complete independent parameter acceptance has not passed")
    binding = report["binding"]
    checks = {plan_path: binding["plan_sha256"], combined_path: binding["combined_inventory_sha256"],
              AUDITOR: binding["auditor_sha256"], READER: binding["reader_sha256"],
              root / "meta/info.json": binding["info_sha256"]}
    checks[root / "meta/episodes.parquet"] = binding["episodes_metadata_sha256"]
    probe_path = nonsymlink(report["fresh_reader_report"])
    if not probe_path.is_relative_to(job / "reports"):
        raise ValueError("Standalone probe must be owned by this cleanup job")
    checks[probe_path] = binding["fresh_reader_sha256"]
    probe = read(probe_path)
    if probe.get("passed") is not True or probe.get("episodes") != expected["episodes"]:
        raise ValueError("Every episode must pass the blocked-old-path standalone probe")
    splits = binding.get("split_metadata_sha256")
    if not isinstance(splits, dict) or not splits:
        raise ValueError("Independent split metadata binding is missing")
    for relative, checksum in splits.items():
        path = owned(root, relative)
        if not path.is_relative_to(root / "meta"):
            raise ValueError("Split metadata path escapes owned meta")
        checks[path] = checksum
    records = [read(p) for p in sorted((root / ".pipeline/episodes").glob("*.json"))]
    if digest(records) != binding["records_manifest_sha256"]:
        raise ValueError("Audited owned episode records changed")
    actual = {(r["episode_index"], r["source"], r["source_key"], r["length"], r["split"]) for r in records}
    required = {(r["episode_index"], r["source"], r["source_key"], r["length"], r["split"]) for r in rows}
    audited = {(r["episode_index"], r["source"], r["source_key"], r["frames"], r["split"]) for r in report["episode_reports"]}
    if actual != required or audited != required or len(records) != len(rows) or len(report["episode_reports"]) != len(rows):
        raise ValueError("Exact accepted source episode/frame/split membership differs")
    verify_probe(probe, records, (mix, plan["egosuite_fitted_root"], plan["hm_root"]))
    info = read(root / "meta/info.json")
    model = owned(root, info["model"]["path"])
    if info["model"]["model_sha256"] != MODEL_SHA256 or binding["model_sha256"] != MODEL_SHA256:
        raise ValueError("Accepted model identity differs from the shared actual16 asset")
    checks[model] = MODEL_SHA256
    hand = info["hand_basis"]
    if binding["hand_basis_sha256"] != hand["sha256"]:
        raise ValueError("Independent hand basis binding differs from owned asset")
    checks[owned(root, hand["path"])] = hand["sha256"]
    fit_gate_path = fitjob / "reports/full-acceptance.json"; fit_gate = read(fit_gate_path)
    checks[fit_gate_path] = binding["original_full_fit_acceptance_sha256"]
    fit_plan_path = fitjob / "plan.json"; fit_plan = read(fit_plan_path)
    fit_status_path = fitjob / "status.json"; fit_status = read(fit_status_path)
    ego = expected["sources"]["egosuite"]
    if (fit_gate.get("format") != "smplh-candidate-acceptance-v2" or fit_gate.get("scope") != "full" or fit_gate.get("passed") is not True
            or fit_gate.get("full_candidate_quality_passed") is not True or fit_gate.get("failures")
            or (fit_gate.get("preservation") or {}).get("passed") is not True
            or fit_gate.get("episodes") != ego["episodes"] or fit_gate.get("frames") != ego["frames"]
            or fit_gate.get("job_signature") != fit_plan["job_signature"]
            or fit_status.get("job_signature") != fit_plan["job_signature"]
            or fit_status.get("state") != "full_candidate_complete" or fit_status.get("independent_acceptance_pending") is not False):
        raise ValueError("The original full fitting job has not completed and passed its registered quality gate")
    fit_audit_path = fitjob / "reports/full-sidecar-audit.json"; fit_audit = read(fit_audit_path)
    checks[fit_audit_path] = fit_gate["independent_audit_sha256"]
    if (binding["original_full_fit_audit_sha256"] != fit_gate["independent_audit_sha256"]
            or binding["original_full_fit_signature"] != fit_plan["job_signature"]):
        raise ValueError("Parameter acceptance differs from original full fitting audit/signature")
    policy_path = fitjob / "quality-policy.json"
    checks[policy_path] = fit_gate["policy_sha256"]
    if (fit_audit.get("passed") is not True or fit_audit.get("episodes_checked") != ego["episodes"]
            or fit_audit.get("frames_checked") != ego["frames"]
            or fit_audit.get("selection_sha256") or fit_audit.get("failures")
            or (fit_audit.get("shared_train_prior_audit") or {}).get("passed") is not True
            or fit_audit.get("auditor_sha256") != sha(FIT_AUDITOR)
            or nonsymlink(fit_audit["sidecar_root"]) != nonsymlink(plan["egosuite_fitted_root"])):
        raise ValueError("Original full fitting acceptance is not bound to its independent full audit")
    ego_join = {(r["source_episode_index"], r["source_key"], r["length"]) for r in rows if r["source"] == "egosuite"}
    fit_join = {(r["episode_index"], r["source_key"], r["frames"]) for r in fit_audit["results"]}
    if (fit_join != ego_join or len(fit_audit["results"]) != ego["episodes"]
            or not all(row.get("passed") is True for row in fit_audit["results"])):
        raise ValueError("The original full fitting audit covers a different source population")
    registration_path = fitjob / "quality-policy-registration.json"
    registration = read(registration_path)
    execution_path = fitjob / "full-execution.json"; execution = read(execution_path)
    fitted_root = nonsymlink(plan["egosuite_fitted_root"])
    first_output = fitted_root / "manifest.json"; first_output_identity = identity(first_output)
    if (registration.get("policy_sha256") != fit_gate["policy_sha256"]
            or registration.get("policy_predates_first_output") is not True
            or registration.get("job_signature") != fit_plan["job_signature"]
            or registration.get("policy_identity") != identity(policy_path)
            or identity(policy_path)["ctime_ns"] >= first_output_identity["ctime_ns"]
            or identity(registration_path)["ctime_ns"] >= first_output_identity["ctime_ns"]
            or (registration.get("registered_at_unix_ns") is not None
                and registration["registered_at_unix_ns"] >= first_output_identity["ctime_ns"])
            or execution.get("state") != "complete" or execution.get("scope") != "full"
            or execution.get("episodes") != ego["episodes"] or execution.get("frames") != ego["frames"]
            or execution.get("job_signature") != fit_plan["job_signature"]
            or nonsymlink(execution["output_root"]) != fitted_root):
        raise ValueError("Original full execution or its preregistered quality policy changed")
    for path in (registration_path, execution_path, first_output):
        checks[path] = sha(path)
    for path, checksum in checks.items():
        if sha(nonsymlink(path)) != checksum:
            raise ValueError("Acceptance/code/owned input SHA changed: " + str(path))
    no_writers(job, fitjob, mix)
    for path in (report_path, fit_plan_path, fit_status_path, fitjob / "active-child.json", fitjob / "full-process.json"):
        checks[path] = sha(path)
    return dict(job=job, plan=plan, mix=mix, root=root, fitjob=fitjob, rows=rows, records=records,
                population=expected, report=report, roots=roots,
                gate_files={str(path): checksum for path, checksum in checks.items()})


def owned_snapshot(context):
    root = context["root"]; audited = {r["episode_index"]: r for r in context["report"]["episode_reports"]}
    entries = {}
    def save(path, checksum=None, *, binary=False):
        path = nonsymlink(path)
        if not path.is_relative_to(root):
            raise ValueError("Protected output path escapes standalone archive")
        entries[str(path)] = {"identity": identity(path), "sha256": checksum if binary else sha(path), "binary": binary}
    for folder in (root / "meta", root / "weights"):
        for parent, dirs, files in os.walk(folder, followlinks=False):
            for name in [*dirs, *files]:
                nonsymlink(Path(parent) / name)
            for name in files:
                save(Path(parent) / name)
    for record in context["records"]:
        record_path = root / ".pipeline/episodes" / f"{record['episode_index']:08d}.json"
        motion = owned(root, record["data_path"]); vision = owned(root, record["vision_path"])
        receipt_path = motion.parent / "receipt.json"; receipt = read(receipt_path)
        if sha(receipt_path) != audited[record["episode_index"]]["receipt_sha256"] or receipt["record"] != record:
            raise ValueError("Audited episode receipt or record changed")
        save(record_path); save(receipt_path)
        for role, path in (("motion", motion), ("vision", vision)):
            if identity(path) != receipt["output_identities"][role] or record[role + "_sha256"] != receipt[role]["sha256"]:
                raise ValueError("Audited owned motion/vision identity changed")
            save(path, receipt[role]["sha256"], binary=True)
    return entries


def verify_owned(entries):
    for path, expected in entries.items():
        if identity(path) != expected["identity"]:
            raise ValueError("Protected owned artifact changed: " + path)
        if not expected["binary"] and sha(path) != expected["sha256"]:
            raise ValueError("Protected owned metadata/model SHA changed: " + path)


def candidates(context, protected):
    original = {}
    for record in context["records"]:
        if record["source"] == "egosuite":
            provenance = record["source_provenance"]
            original[str(nonsymlink(provenance["source_path"]))] = provenance["source_identity"]
            original[str(nonsymlink(provenance["fitted_path"]))] = provenance["fitted_identity"]
    entries = []
    protected_inodes = {(value["identity"]["device"], value["identity"]["inode"]) for value in protected.values()}
    for root in context["roots"]:
        nonsymlink(root)
        if not root.exists():
            continue
        for parent, dirs, files in os.walk(root, followlinks=False):
            for name in [*dirs, *files]:
                nonsymlink(Path(parent) / name)
            for name in files:
                path = nonsymlink(Path(parent) / name)
                if path.suffix != ".parquet":
                    continue
                columns = pq.ParquetFile(path).schema_arrow.names
                if not any("body_world" in column or "body_floor_rot6d_xyz" in column for column in columns):
                    continue
                current = identity(path)
                if (current["device"], current["inode"]) in protected_inodes:
                    raise ValueError("Old candidate is hard-linked to a protected owned artifact")
                if str(path) in original and current != original[str(path)]:
                    raise ValueError("Accepted original dense source was replaced or changed")
                entries.append(dict(path=str(path), identity=current, bytes=current["bytes"],
                                    kind="accepted_original" if str(path) in original else "authorized_legacy_dense_cache"))
    if not set(original) <= {entry["path"] for entry in entries}:
        raise ValueError("An accepted original dense source is absent or outside the exact deletion allowlist")
    return sorted(entries, key=lambda row: row["path"])


def manifest_path(job):
    return Path(job) / "reports/dense-cleanup-manifest.json"


def prepare_manifest(job):
    job = nonsymlink(job)
    with cleanup_lock(job):
        path = manifest_path(job)
        if path.exists():
            value = read(path); check_manifest(job, value)
            return value
        context = gate_context(job); protected = owned_snapshot(context)
        entries = candidates(context, protected)
        value = dict(format=MANIFEST_FORMAT, job=str(context["job"]), output_root=str(context["root"]),
                     mix_root=str(context["mix"]), population=context["population"], created_at=now(),
                     gate_files=context["gate_files"], cleanup_code_sha256=sha(__file__),
                     protected_files=protected, files=entries, file_count=len(entries), bytes=sum(e["bytes"] for e in entries),
                     delete_directories=False, preserved="raw MCAP/HM/Aria/DINO, weights, jobs, JSON metadata, NPZ/research evidence, standalone archive")
        value["manifest_sha256"] = digest(value)
        write(path, value)
        return value


def check_manifest(job, manifest):
    value = dict(manifest); signature = value.pop("manifest_sha256", None)
    if value.get("format") != MANIFEST_FORMAT or signature != digest(value) or value["cleanup_code_sha256"] != sha(__file__):
        raise ValueError("Cleanup manifest content/code signature changed")
    context = gate_context(job)
    if (value["job"] != str(context["job"]) or value["output_root"] != str(context["root"])
            or value["mix_root"] != str(context["mix"]) or value["population"] != context["population"]
            or value["gate_files"] != context["gate_files"]):
        raise ValueError("Cleanup gate/plan/inventory changed after manifest registration")
    verify_owned(value["protected_files"])
    seen = set()
    for entry in value["files"]:
        path = nonsymlink(entry["path"])
        if path.suffix != ".parquet" or not any(path.is_relative_to(root) for root in context["roots"]):
            raise ValueError("Cleanup entry escapes exact dense-Parquet allowlist")
        if str(path) in seen or str(path) in value["protected_files"]:
            raise ValueError("Cleanup manifest duplicates or targets a protected artifact")
        seen.add(str(path))
    if len(seen) != value["file_count"] or sum(e["bytes"] for e in value["files"]) != value["bytes"]:
        raise ValueError("Cleanup manifest size/population differs")
    return context


def secure_unlink(mix, entry):
    path = nonsymlink(entry["path"]); relative = path.relative_to(mix)
    descriptor = os.open(mix, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = child
        value = os.stat(relative.name, dir_fd=descriptor, follow_symlinks=False)
        current = dict(bytes=value.st_size, mtime_ns=value.st_mtime_ns, ctime_ns=value.st_ctime_ns,
                       device=value.st_dev, inode=value.st_ino)
        if not stat.S_ISREG(value.st_mode) or current != entry["identity"]:
            raise ValueError("Dense file changed immediately before deletion")
        os.unlink(relative.name, dir_fd=descriptor); os.fsync(descriptor)
    finally:
        os.close(descriptor)


def post_cleanup_probe(context):
    output = context["job"] / "reports/post-cleanup-standalone.json"
    command = [sys.executable, str(AUDITOR), "--reader-probe", "--root", str(context["root"]),
               "--blocked-roots", json.dumps([str(context["mix"]), context["plan"]["egosuite_fitted_root"], context["plan"]["hm_root"]]),
               "--output", str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Post-cleanup standalone reader failed: " + (result.stderr or result.stdout)[-4000:])
    report = read(output)
    verify_probe(report, context["records"], (context["mix"], context["plan"]["egosuite_fitted_root"], context["plan"]["hm_root"]))
    return dict(path=str(output), sha256=sha(output), passed=True)


@contextmanager
def cleanup_lock(job):
    path = nonsymlink(Path(job) / "cleanup.lock")
    with path.open("a") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another cleanup process is active") from error
        yield


def execute_cleanup(job):
    job = nonsymlink(job)
    with cleanup_lock(job):
        manifest = read(manifest_path(job)); context = check_manifest(job, manifest)
        path = job / "reports/dense-cleanup-execution.json"
        receipt = read(path) if path.exists() else dict(format=EXECUTION_FORMAT, manifest_sha256=manifest["manifest_sha256"],
            state="deleting", deleted=[], pending=None, started_at=now(), bytes_deleted=0)
        if receipt.get("format") != EXECUTION_FORMAT or receipt.get("manifest_sha256") != manifest["manifest_sha256"]:
            raise ValueError("Cleanup execution belongs to a different manifest")
        bypath = {entry["path"]: entry for entry in manifest["files"]}
        if (len(receipt["deleted"]) != len(set(receipt["deleted"])) or not set(receipt["deleted"]) <= set(bypath)
                or (receipt["pending"] is not None and receipt["pending"] not in bypath)):
            raise ValueError("Cleanup progress references an unknown file")
        # Preflight every remaining file before performing even one deletion.
        for entry in manifest["files"]:
            target = Path(entry["path"])
            if entry["path"] in receipt["deleted"]:
                if target.exists(): raise ValueError("A deleted dense file was recreated")
            elif not target.exists():
                if receipt["pending"] != entry["path"]:
                    raise ValueError("Dense file disappeared without durable cleanup progress")
            elif identity(target) != entry["identity"]:
                raise ValueError("A remaining dense file changed after cleanup registration")
        for number, entry in enumerate(manifest["files"]):
            if entry["path"] in receipt["deleted"]:
                continue
            if number % 100 == 0:
                check_manifest(job, manifest)
            receipt.update(state="deleting", pending=entry["path"], updated_at=now())
            write(path, receipt)
            if Path(entry["path"]).exists():
                secure_unlink(context["mix"], entry)
            receipt["deleted"].append(entry["path"])
            receipt.update(pending=None, bytes_deleted=sum(bypath[p]["bytes"] for p in receipt["deleted"]), updated_at=now())
            write(path, receipt)
        check_manifest(job, manifest)
        probe = post_cleanup_probe(context)
        receipt.update(state="complete", post_cleanup_probe=probe, completed_at=now(), deleted_files=len(receipt["deleted"]))
        write(path, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "execute"))
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    value = prepare_manifest(args.job) if args.command == "plan" else execute_cleanup(args.job)
    print(json.dumps({key: value[key] for key in ("format", "file_count", "bytes", "manifest_sha256") if key in value} or
                     {key: value[key] for key in ("state", "deleted_files", "bytes_deleted") if key in value}))


if __name__ == "__main__":
    main()
