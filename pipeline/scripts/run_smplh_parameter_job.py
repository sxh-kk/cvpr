#!/usr/bin/env python3
"""Run conversion, both independent gates and authorized conditional cleanup."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from process_smplh_parameters import assert_frozen_job, atomic_json, file_sha, read_json

WORKSPACE = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent
children = {}
job_lock = None


def stamp(): return datetime.now(timezone.utc).isoformat()


def update(job, phase, **fields):
    report = {"state": phase, "updated_at": stamp(), "job_dir": str(job),
              "output_root": read_json(job / "plan.json")["output_root"],
              "pid": os.getpid(), "acceptance_pending": phase != "complete", **fields}
    atomic_json(job / "status.json", report)
    atomic_json(job.parent / "current_smplh_parameters.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def launch(job, name, argv):
    assert_frozen_job(job)
    log = (job / "logs" / (name + ".log")).open("a", buffering=1)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(WORKSPACE / "pipeline/src") + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else "")
    process = subprocess.Popen(argv, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, env=environment)
    log.close(); children[name] = process
    atomic_json(job / ("child-" + name + ".json"), {"pid": process.pid, "command": argv,
                 "started_at": stamp()})
    print(json.dumps({"event": "started", "phase": name, "pid": process.pid,
                      "command": argv, "at": stamp()}), flush=True)
    return process


def ensure_child_ok(name):
    code = children[name].poll()
    if code is not None and code != 0: raise RuntimeError(f"{name} failed with exit code {code}; see its log")
    return code is not None


def wait_children(job, names, phase):
    while True:
        done = [ensure_child_ok(name) for name in names]
        if all(done): return
        update(job, phase, children={name: children[name].pid for name in names},
               source_status={name: read_json(job / f"status-{name}.json") for name in ("egosuite", "egoexo4d_hm")
                              if (job / f"status-{name}.json").exists()})
        time.sleep(30)


def sequential(job, name, argv):
    launch(job, name, argv); wait_children(job, [name], name)


def old_fit_failed(status):
    return status.get("state") in ("failed", "stopped", "full_candidate_quality_failed")


def assert_no_orphan_jobs(job):
    """A killed wrapper can leave its session-owning workers alive."""
    scripts = ("process_smplh_parameters.py", "audit_smplh_parameters.py", "cleanup_smplh_dense.py")
    for folder in Path("/proc").iterdir():
        if not folder.name.isdigit() or int(folder.name) == os.getpid(): continue
        try:
            state = (folder / "stat").read_text().rsplit(")", 1)[1].split()[0]
            if state in ("Z", "X"): continue
            command = (folder / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except FileNotFoundError: continue
        if str(job) in command and any(name in command for name in scripts):
            raise RuntimeError(f"Previous job child {folder.name} is still running; resume after it exits")


def finish_cleanup(args, job, plan):
    """Resume deletion without reopening deleted source files or re-signing gates."""
    cleanup = str(SCRIPTS / "cleanup_smplh_dense.py")
    if not (job / "reports/dense-cleanup-manifest.json").exists():
        sequential(job, "cleanup-plan", [args.python, cleanup, "plan", "--job", str(job)])
    sequential(job, "authorized-cleanup", [args.python, cleanup, "execute", "--job", str(job)])
    result = read_json(job / "reports/dense-cleanup-execution.json")
    if result.get("state") != "complete": raise RuntimeError("Cleanup is not complete")
    proof = result["post_cleanup_probe"]
    if not proof.get("passed") or file_sha(proof["path"]) != proof["sha256"]:
        raise RuntimeError("Post-cleanup reader proof changed")
    probe = read_json(proof["path"])
    if not probe.get("passed") or probe.get("episodes") != 9263:
        raise RuntimeError("Post-cleanup reader did not cover the full original population")
    root = Path(plan["output_root"]); audit = job / "reports/full-parameter-acceptance.json"
    state = read_json(root / ".pipeline/dataset-state.json")
    state.update(state="accepted", independent_acceptance_pending=False,
                 acceptance_report=str(audit), old_dense_deleted=True, updated_at=stamp())
    atomic_json(root / ".pipeline/dataset-state.json", state)
    atomic_json(Path(plan["egosuite_archive"]) / "current_dataset.json", {
                "path": str(root), "schema_version": state["schema_version"],
                "episodes": 9263, "frames": 38137884, "state": "accepted",
                "acceptance_report": str(audit), "old_dense_deleted": True,
                "motionexpert_parameter_adapter_pending": True, "updated_at": stamp()})
    update(job, "complete", episodes=9263, frames=38137884, old_dense_deleted=True,
           acceptance_report=str(audit), post_cleanup_reader=proof["path"])


def stop(signum, frame):
    for child in children.values():
        if child.poll() is None: os.killpg(child.pid, signum)
    raise KeyboardInterrupt


def run(args):
    global job_lock
    job = Path(args.job).resolve(); plan = read_json(job / "plan.json")
    lock = (job / "lifecycle.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    job_lock = lock
    assert_frozen_job(job)
    assert_no_orphan_jobs(job)
    if (job / "reports/dense-cleanup-manifest.json").exists():
        update(job, "resuming_authorized_cleanup")
        finish_cleanup(args, job, plan)
        return
    root = Path(plan["output_root"])
    oldjob = Path(plan["egosuite_fitting_job"])
    converter = str(SCRIPTS / "process_smplh_parameters.py")
    # Owned model/PCA assets must finish before independent source jobs start.
    sequential(job, "initialize", [args.python, converter, "initialize", "--job", str(job)])
    for source in ("egosuite", "egoexo4d_hm"):
        python = args.hm_python if source == "egoexo4d_hm" else args.python
        launch(job, source, [python, converter, "run", "--job", str(job), "--source", source,
                            "--workers", str(args.workers)])
    # Existing fitting remains independent. Migrate committed episodes while
    # its remaining episodes finish; only the existing full quality gate counts.
    while True:
        for source in ("egosuite", "egoexo4d_hm"): ensure_child_ok(source)
        fit = read_json(oldjob / "status.json")
        if old_fit_failed(fit): raise RuntimeError("Original full fitting/audit failed; old dense data is retained")
        if fit.get("completed") == fit.get("total") == 6614: break
        update(job, "migrating_and_waiting_for_fitting", original_fitting=fit)
        time.sleep(30)
    wait_children(job, ["egosuite", "egoexo4d_hm"], "converting")
    ego = read_json(job / "reports/migration-egosuite.json")
    if not ego["full_population_complete"]:
        sequential(job, "egosuite-final", [args.python, converter, "run", "--job", str(job),
                   "--source", "egosuite", "--workers", str(args.workers)])
    sequential(job, "finalize", [args.python, converter, "finalize", "--job", str(job)])
    while True:
        fit = read_json(oldjob / "status.json")
        if old_fit_failed(fit): raise RuntimeError("Original full numerical acceptance failed; old dense data is retained")
        if fit.get("state") == "full_candidate_complete": break
        update(job, "waiting_for_original_full_acceptance", original_fitting=fit)
        time.sleep(30)
    audit = job / "reports/full-parameter-acceptance.json"
    sequential(job, "independent-full-audit", [args.hm_python, str(SCRIPTS / "audit_smplh_parameters.py"),
               "--job", str(job), "--workers", str(args.audit_workers), "--output", str(audit)])
    accepted = read_json(audit)
    if not accepted.get("passed") or not accepted.get("cleanup_allowed"):
        raise RuntimeError("Independent full acceptance did not authorize cleanup")
    finish_cleanup(args, job, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True)
    parser.add_argument("--python", default="/tmp/egomotion-pipeline-venv/bin/python")
    parser.add_argument("--hm-python", default="/tmp/egomotion-hm-camera-venv/bin/python")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--audit-workers", type=int, default=12)
    args = parser.parse_args()
    if min(args.workers, args.audit_workers) < 1: parser.error("workers must be positive")
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    try: run(args)
    except BaseException as exc:
        for child in children.values():
            if child.poll() is None: os.killpg(child.pid, signal.SIGTERM)
        if job_lock is not None:
            update(Path(args.job), "stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                   error=repr(exc), old_dense_deletion_not_assumed=True)
        else:
            print(json.dumps({"event": "start_rejected", "error": repr(exc)}), flush=True)
        raise
    finally:
        if job_lock is not None: job_lock.close()


if __name__ == "__main__": main()
