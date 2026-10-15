"""Stop one frozen eight-GPU run after its exact natural saved epoch.

Checkpoint contents are never opened. This one-shot supervisor publishes a
separate stop pointer, never rewrites current_run, and never starts training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import yaml

from .switch_to_window_cache import (DEFAULT_POINTER, MAX_CHECKPOINT_AGE_SECONDS,
    _process_fields, assert_original, completed_checkpoint, file_hash,
    log_progress, stop_original, training_ranks)
from .watch_dino import (REPO_ROOT, atomic_json, locked, now, process_alive,
                         process_identity, read_json)

CONTRACT = "motionexpert-stop-at-natural-epoch-v1"


def checkpoint_identity(checkpoint):
    value = Path(checkpoint["path"]).stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def assert_frozen_run(plan):
    assert_original(plan)
    current = read_json(plan["current_run"])
    if (Path(current.get("config", "")).resolve() != Path(plan["source_config"])
            or Path(current.get("output", "")).resolve() != Path(plan["output"])
            or current.get("watcher") != plan["guardian"]):
        raise RuntimeError("Frozen training pointer/configuration/guardian changed")
    if plan["guardian"] is not None and not process_alive(plan["guardian"]):
        raise RuntimeError("Frozen training guardian exited before the stop")
    if file_hash(plan["source_config"]) != plan["source_config_sha256"]:
        raise RuntimeError("Frozen source configuration changed")


def exact_saved_boundary(plan, checkpoint, progress, *, now_seconds=None):
    target = plan["completed_epochs"]
    if checkpoint is None or checkpoint["completed_epochs"] < target:
        return False
    if checkpoint["completed_epochs"] != target:
        raise RuntimeError("Selected natural checkpoint was missed; refusing to stop a later epoch")
    if checkpoint["global_step"] != target * plan["steps_per_epoch"]:
        raise RuntimeError("Natural target checkpoint has an unexpected global step")
    if Path(checkpoint["path"]).stat().st_size <= 0:
        raise RuntimeError("Natural target checkpoint has an empty file stat")
    if progress["steps_per_epoch"] != plan["steps_per_epoch"]:
        raise RuntimeError("Training steps per epoch changed")
    age = (time.time() if now_seconds is None else now_seconds) - Path(checkpoint["path"]).stat().st_mtime
    if age > MAX_CHECKPOINT_AGE_SECONDS or age < -5:
        raise RuntimeError("Selected natural checkpoint is stale; refusing to discard later training")
    return ((progress["epoch"] == target - 1
             and progress["batch"] == progress["steps_per_epoch"])
            or (progress["epoch"] == target
                and progress["batch"] < progress["steps_per_epoch"]))


def prepare_plan(pointer, completed_epochs, job_dir):
    if type(completed_epochs) is not int or completed_epochs <= 0:
        raise ValueError("--completed-epochs must be a positive integer")
    current = read_json(pointer)
    trainer = current.get("trainer")
    if (current.get("training_running") is not True or trainer is None
            or trainer != process_identity(current.get("pid", -1))):
        raise ValueError("Current run must identify an active original trainer")
    pid = trainer["pid"]
    if os.getpgid(pid) != pid or os.getsid(pid) != pid:
        raise ValueError("Trainer must own an isolated process group and session")
    guardian = current.get("watcher")
    if guardian is not None and not process_alive(guardian):
        raise ValueError("Current run guardian is no longer alive")
    gpus = current.get("gpus")
    if (not isinstance(gpus, list) or len(set(gpus)) != 8
            or any(type(item) is not int or item < 0 for item in gpus)):
        raise ValueError("Expected eight distinct GPU IDs")
    source = Path(current["config"]).resolve(strict=True)
    config = yaml.safe_load(source.read_text())
    train = config["TRAIN"]
    interval = train.get("SAVE_EVERY_N_EPOCHS")
    if (type(interval) is not int or interval != 20
            or completed_epochs % interval or completed_epochs > train["NUM_EPOCHS"]):
        raise ValueError("Target must be an existing natural twenty-epoch save boundary")
    output = Path(current["output"]).resolve(strict=True)
    if Path(train["EXP_PATH"]).resolve() != output or train.get("NUM_GPUS") != 8:
        raise ValueError("Source config disagrees with the eight-GPU output run")
    progress = log_progress(current["log"])
    if progress["epoch"] > completed_epochs or (progress["epoch"] == completed_epochs
            and progress["batch"] == progress["steps_per_epoch"]):
        raise ValueError("Target epoch has already been missed")
    root = Path(pointer).resolve().parent.parent
    stop_pointer = root / ".pipeline/current_training_stop.json"
    if stop_pointer.exists():
        previous = read_json(stop_pointer)
        if (process_alive(previous.get("watcher"))
                or (previous.get("trainer") == trainer
                    and previous.get("completed_epochs") == completed_epochs)):
            raise FileExistsError("A stop supervisor already exists for this run/target; no automatic retry")
    marker = output / f"training-stop-epoch-{completed_epochs}.json"
    if marker.exists():
        raise FileExistsError("The target stop was already attempted; manual review is required")
    plan = {"contract": CONTRACT, "created_at": now(), "current_run": str(Path(pointer).resolve()),
            "root": str(root), "output": str(output), "job_dir": str(Path(job_dir).resolve()),
            "trainer": trainer, "guardian": guardian, "pgid": pid, "ranks": training_ranks(pid),
            "source_config": str(source), "source_config_sha256": file_hash(source),
            "old_log": current["log"], "gpus": gpus, "old_run": deepcopy(current),
            "completed_epochs": completed_epochs, "steps_per_epoch": progress["steps_per_epoch"],
            "initial_progress": progress, "baseline_checkpoint": completed_checkpoint(output),
            "stop_marker": str(marker), "automatic_restart_enabled": False,
            "checkpoint_inspection": "natural_last_symlink_filename_and_stat_only"}
    assert_frozen_run(plan)
    atomic_json(Path(job_dir) / "plan.json", plan)
    return plan


def run_stop(plan, *, poll_seconds=.5, stable_seconds=3., wait_timeout=86400.,
             shutdown_timeout=180., force_timeout=30., stop=None, shutdown=stop_original):
    stop = stop or threading.Event()
    root, job = Path(plan["root"]), Path(plan["job_dir"])
    state = {**deepcopy(plan), "watcher": process_identity(os.getpid()), "state": "armed",
             "pid": plan["trainer"]["pid"], "training_running": True, "checkpoint": None,
             "daemon_log": str(job / "daemon.log"), "automatic_restart_enabled": False,
             "safe_forced_shutdown_available": hasattr(os, "pidfd_open")
                 and hasattr(signal, "pidfd_send_signal")}
    frozen = shutdown_started = shutdown_complete = False
    started, last_publish = time.monotonic(), 0.
    observed, stable_since = None, None

    def publish(phase, **fields):
        nonlocal last_publish
        changed = state.get("state") != phase
        state.update(state=phase, updated_at=now(), **fields)
        atomic_json(job / "state.json", state)
        atomic_json(root / ".pipeline/current_training_stop.json", state)
        last_publish = time.monotonic()
        if changed:
            print(f"[{now()}] {phase}: {fields.get('reason', '')}", flush=True)

    try:
        assert_frozen_run(plan)
        publish("waiting_checkpoint", reason=f"Waiting for the natural saved epoch {plan['completed_epochs']}; no extra save or resume")
        while not stop.is_set():
            assert_frozen_run(plan)
            if time.monotonic() - started > wait_timeout:
                raise TimeoutError("Natural target checkpoint wait timed out; original training was not signalled")
            progress = log_progress(plan["old_log"])
            target = plan["completed_epochs"]
            if progress["epoch"] > target or (progress["epoch"] == target
                    and progress["batch"] == progress["steps_per_epoch"]):
                raise RuntimeError("Target boundary has been missed; refusing to stop a later saved epoch")
            checkpoint = completed_checkpoint(plan["output"])
            candidate = exact_saved_boundary(plan, checkpoint, progress)
            identity = (checkpoint["path"], *checkpoint_identity(checkpoint)) if candidate else None
            if identity != observed:
                observed, stable_since = identity, time.monotonic() if identity is not None else None
            if candidate and time.monotonic() - stable_since >= stable_seconds:
                with locked(root / "training/.launcher-reservation.lock"):
                    assert_frozen_run(plan)
                    latest = completed_checkpoint(plan["output"])
                    progress = log_progress(plan["old_log"])
                    if (latest != checkpoint or (latest["path"], *checkpoint_identity(latest)) != identity
                            or not exact_saved_boundary(plan, latest, progress)):
                        raise RuntimeError("Natural checkpoint changed before stopping; no stop was issued")
                    if stop.is_set():
                        break
                    marker = {"contract": CONTRACT, "created_at": now(), "trainer": plan["trainer"],
                              "completed_epochs": target, "checkpoint": checkpoint,
                              "checkpoint_file_identity": list(identity[1:]),
                              "automatic_restart_enabled": False, "job_dir": str(job)}
                    with Path(plan["stop_marker"]).open("x") as stream:
                        json.dump(marker, stream, indent=2); stream.flush(); os.fsync(stream.fileno())
                    assert_frozen_run(plan)
                    os.killpg(plan["pgid"], signal.SIGSTOP)
                    frozen = True
                    assert_frozen_run(plan)
                    if (completed_checkpoint(plan["output"]) != checkpoint
                            or (checkpoint["path"], *checkpoint_identity(checkpoint)) != identity
                            or not exact_saved_boundary(plan, checkpoint, log_progress(plan["old_log"]))):
                        raise RuntimeError("Natural target checkpoint changed while freezing the original session")
                    if stop.is_set():
                        os.killpg(plan["pgid"], signal.SIGCONT)
                        frozen = False
                        publish("cancelled", reason="Cancelled while freezing; original training continues")
                        return 0
                    shutdown_started = True
                    publish("stopping_original", checkpoint=checkpoint["path"], saved_checkpoint=checkpoint,
                            checkpoint_file_identity=list(identity[1:]), unsaved_progress=progress,
                            original_shutdown_started=True, training_running=False,
                            reason="Natural target save is stable; stopping only frozen original processes")
                    result = shutdown(plan, timeout=shutdown_timeout, force_timeout=force_timeout)
                    shutdown_complete, frozen = True, False
                    final = read_json(plan["current_run"])
                    atomic_json(job / "previous-run-final.json", final)
                    publish("stopped", training_running=False, finished_at=now(), shutdown_result=result,
                            original_shutdown_complete=True,
                            reason=f"Stopped after natural epoch {target}; no automatic training launch")
                    return 0
            if time.monotonic() - last_publish > 30:
                publish("waiting_checkpoint", progress=progress, observed_checkpoint=checkpoint)
            stop.wait(poll_seconds)
        publish("cancelled", reason="Stop supervision cancelled; original training continues")
        return 0
    except Exception as error:
        if frozen:
            # SIGCONT cannot terminate a process; still restrict it to the frozen
            # session while at least one captured original member remains alive.
            if any(process_alive(item) and (_process_fields(item["pid"]) or {}).get("session") == plan["pgid"]
                   and (_process_fields(item["pid"]) or {}).get("group") == plan["pgid"]
                   for item in plan["ranks"].values()):
                try:
                    os.killpg(plan["pgid"], signal.SIGCONT)
                except ProcessLookupError:
                    pass
        remaining = [item for item in plan["ranks"].values() if process_alive(item)]
        intact = not shutdown_started and len(remaining) == len(plan["ranks"])
        publish("failed", error=f"{type(error).__name__}: {error}", training_running=intact,
                original_shutdown_started=shutdown_started, original_shutdown_complete=shutdown_complete,
                remaining_original_ranks=remaining,
                manual_cleanup_required=shutdown_started and not shutdown_complete,
                reason="Stop failed; no automatic retry, no extra save, and no training was launched")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed-epochs", type=int, required=True)
    parser.add_argument("--current-run", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--job-dir", type=Path)
    parser.add_argument("--poll-seconds", type=float, default=.5)
    parser.add_argument("--stable-seconds", type=float, default=3.)
    parser.add_argument("--wait-timeout-seconds", type=float, default=86400.)
    parser.add_argument("--shutdown-timeout-seconds", type=float, default=180.)
    parser.add_argument("--force-timeout-seconds", type=float, default=30.)
    parser.add_argument("--background", action="store_true")
    args = parser.parse_args()
    for name in ("poll_seconds", "stable_seconds", "wait_timeout_seconds", "shutdown_timeout_seconds", "force_timeout_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.completed_epochs <= 0:
        parser.error("--completed-epochs must be positive")
    pointer = args.current_run.resolve(strict=True)
    root = pointer.parent.parent
    job = (args.job_dir or root / ".pipeline/jobs" / (
        "motion-stop-epoch-" + str(args.completed_epochs) + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    if job.is_relative_to(REPO_ROOT) or not job.is_relative_to(root / ".pipeline/jobs"):
        parser.error("--job-dir must be owned by MixEgoMotion/.pipeline/jobs outside the repository")
    job.mkdir(parents=True, exist_ok=True)
    if args.background:
        if any(job.iterdir()):
            parser.error("Use a new empty stop job directory; existing jobs cannot be restarted")
        command = [sys.executable, "-u", "-m", "MotionExpert.stop_at_epoch", "--completed-epochs", str(args.completed_epochs),
                   "--current-run", str(pointer), "--job-dir", str(job)]
        for name in ("poll_seconds", "stable_seconds", "wait_timeout_seconds", "shutdown_timeout_seconds", "force_timeout_seconds"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        with (job / "daemon.log").open("x") as stream:
            child = subprocess.Popen(command, cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (job / "state.json").is_file():
                state = read_json(job / "state.json")
                print(json.dumps({"pid": child.pid, "state": state["state"], "job_dir": str(job)}), flush=True)
                return 1 if state["state"] == "failed" else 0
            if child.poll() is not None:
                print((job / "daemon.log").read_text(), file=sys.stderr)
                return 1
            time.sleep(.1)
        print(json.dumps({"pid": child.pid, "state": "starting", "job_dir": str(job)}), flush=True)
        return 0
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        with locked(root / "training/.training-stop.lock"):
            if any((job / name).exists() for name in ("plan.json", "state.json")):
                raise FileExistsError("Existing stop jobs cannot be restarted")
            plan = prepare_plan(pointer, args.completed_epochs, job)
            return run_stop(plan, poll_seconds=args.poll_seconds, stable_seconds=args.stable_seconds,
                            wait_timeout=args.wait_timeout_seconds, shutdown_timeout=args.shutdown_timeout_seconds,
                            force_timeout=args.force_timeout_seconds, stop=stop)
    except Exception as error:
        print(f"[{now()}] Stop supervisor failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
