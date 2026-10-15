"""Natural target-stop control flow, using harmless isolated CPU processes."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from MotionExpert import stop_at_epoch as guard


def publish_checkpoint(plan, completed=100, *, linked=True, step=None):
    output = Path(plan["output"])
    path = output / f"epoch={completed-1}-step={completed*488 if step is None else step}.ckpt"
    path.write_bytes(b"opaque fixture; supervisor must not open checkpoint contents")
    if linked:
        temporary = output / ".last.tmp"
        temporary.symlink_to(path.name)
        temporary.replace(output / "last.ckpt")
    return path


@pytest.fixture
def setup(tmp_path, request):
    root = tmp_path / "MixEgoMotion"
    output = root / "training/run"
    job = root / ".pipeline/jobs/stop100"
    output.mkdir(parents=True)
    job.mkdir(parents=True)
    config = root / "source.yaml"
    config.write_text(yaml.safe_dump({"TRAIN": {"NUM_GPUS": 8, "NUM_EPOCHS": 200,
        "SAVE_EVERY_N_EPOCHS": 20, "EXP_PATH": str(output)}}))
    ignore_term = getattr(request, "param", None) == "ignore_term"
    code = ("import time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "print('ready',flush=True); time.sleep(600)" if ignore_term else "import time; time.sleep(600)")
    old = subprocess.Popen([sys.executable, "-c", code], start_new_session=True,
                           stdout=subprocess.PIPE if ignore_term else None)
    if ignore_term:
        assert old.stdout.readline() == b"ready\n"
    identity = guard.process_identity(old.pid)
    log = root / "training.log"
    log.write_text("Epoch 95: 12%| | 60/488\r")
    pointer = root / "training/current_run.json"
    guard.atomic_json(pointer, {"training_running": True, "pid": old.pid, "trainer": identity,
        "watcher": None, "gpus": list(range(8)), "config": str(config), "output": str(output), "log": str(log)})
    plan = {"contract": guard.CONTRACT, "created_at": guard.now(), "current_run": str(pointer),
        "root": str(root), "output": str(output), "job_dir": str(job), "trainer": identity,
        "guardian": None, "pgid": old.pid, "ranks": {"0": identity}, "source_config": str(config),
        "source_config_sha256": guard.file_hash(config), "old_log": str(log), "gpus": list(range(8)),
        "completed_epochs": 100, "steps_per_epoch": 488,
        "stop_marker": str(output / "training-stop-epoch-100.json")}
    publish_checkpoint(plan, 80)
    try:
        yield plan, old
    finally:
        if guard.process_identity(old.pid):
            os.killpg(old.pid, signal.SIGTERM)
            os.killpg(old.pid, signal.SIGCONT)
        try:
            old.wait(timeout=5)
        except subprocess.TimeoutExpired:
            old.kill()
            old.wait(timeout=5)
        if old.stdout is not None:
            old.stdout.close()


def run(plan, **kwargs):
    return guard.run_stop(plan, poll_seconds=.005, stable_seconds=.02, wait_timeout=2,
                          shutdown_timeout=.1, force_timeout=.1, **kwargs)


def test_stops_only_published_stable_exact100_without_checkpoint_read_or_pointer_write(setup, monkeypatch):
    plan, old = setup
    pointer_before = Path(plan["current_run"]).read_bytes()
    target = publish_checkpoint(plan, linked=False)
    original_open = Path.open
    def no_checkpoint_read(path, *args, **kwargs):
        if path.suffix == ".ckpt":
            raise AssertionError("Checkpoint contents were accessed")
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_checkpoint_read)
    observations = []
    def finish():
        observations.append(old.poll() is None)
        Path(plan["old_log"]).write_text("Epoch 99: 100%| | 488/488\r")
        temporary = Path(plan["output"]) / ".last.tmp"
        temporary.symlink_to(target.name)
        temporary.replace(Path(plan["output"]) / "last.ckpt")
    timer = threading.Timer(.05, finish)
    timer.start()
    try:
        assert run(plan) == 0
    finally:
        timer.cancel()
    assert observations == [True]
    assert guard.process_identity(old.pid) is None
    assert Path(plan["current_run"]).read_bytes() == pointer_before
    state = guard.read_json(Path(plan["job_dir"]) / "state.json")
    assert state["state"] == "stopped" and state["checkpoint"] == str(target)
    assert state["saved_checkpoint"]["global_step"] == 48800
    assert state["automatic_restart_enabled"] is False
    assert state["shutdown_result"]["termination"] == "sigterm"


@pytest.mark.parametrize("epoch,batch", [(99, 488), (100, 0)])
def test_log100_without_natural_save_never_signals(setup, epoch, batch):
    plan, old = setup
    Path(plan["old_log"]).write_text(f"Epoch {epoch}: 100%| | {batch}/488\r")
    stop = threading.Event()
    timer = threading.Timer(.08, stop.set)
    timer.start()
    try:
        assert run(plan, stop=stop) == 0
    finally:
        timer.cancel()
    assert old.poll() is None
    assert not Path(plan["stop_marker"]).exists()
    assert guard.read_json(Path(plan["job_dir"]) / "state.json")["state"] == "cancelled"


def test_pointer_identity_replacement_refuses_stop(setup):
    plan, old = setup
    replacement = guard.read_json(plan["current_run"])
    replacement["trainer"] = {**replacement["trainer"], "start_ticks": "replacement"}
    guard.atomic_json(plan["current_run"], replacement)
    with pytest.raises(RuntimeError, match="replaced"):
        run(plan)
    assert old.poll() is None and not Path(plan["stop_marker"]).exists()


def test_stale_target_checkpoint_fails_without_stopping(setup):
    plan, old = setup
    path = publish_checkpoint(plan)
    os.utime(path, (time.time()-200, time.time()-200))
    Path(plan["old_log"]).write_text("Epoch 100: 1%| | 1/488\r")
    with pytest.raises(RuntimeError, match="stale"):
        run(plan)
    assert old.poll() is None


@pytest.mark.parametrize("completed,step", [(120, None), (100, 49000)])
def test_different_save_or_step_is_never_stopped(setup, completed, step):
    plan, old = setup
    publish_checkpoint(plan, completed, step=step)
    Path(plan["old_log"]).write_text("Epoch 99: 100%| | 488/488\r")
    with pytest.raises(RuntimeError, match="missed|unexpected global step"):
        run(plan)
    assert old.poll() is None


def test_wait_timeout_does_not_stop_original(setup):
    plan, old = setup
    with pytest.raises(TimeoutError, match="wait timed out"):
        guard.run_stop(plan, poll_seconds=.005, stable_seconds=.01, wait_timeout=.025)
    assert old.poll() is None
    state = guard.read_json(Path(plan["job_dir"]) / "state.json")
    assert state["original_shutdown_started"] is False
    assert state["manual_cleanup_required"] is False


def test_shutdown_failure_records_manual_cleanup_and_continues_frozen_original(setup):
    plan, old = setup
    publish_checkpoint(plan)
    Path(plan["old_log"]).write_text("Epoch 99: 100%| | 488/488\r")
    def unavailable(*args, **kwargs):
        raise RuntimeError("safe pidfd escalation is unavailable")
    with pytest.raises(RuntimeError, match="pidfd"):
        run(plan, shutdown=unavailable)
    assert old.poll() is None
    state = guard.read_json(Path(plan["job_dir"]) / "state.json")
    assert state["original_shutdown_started"] is True
    assert state["original_shutdown_complete"] is False
    assert state["manual_cleanup_required"] is True
    assert state["remaining_original_ranks"] == list(plan["ranks"].values())
    assert Path(plan["stop_marker"]).exists()


def test_source_config_change_fails_before_signal(setup):
    plan, old = setup
    with Path(plan["source_config"]).open("a") as stream:
        stream.write("# changed\n")
    with pytest.raises(RuntimeError, match="configuration changed"):
        run(plan)
    assert old.poll() is None


def test_identity_replacement_during_freeze_never_sends_term(setup, monkeypatch):
    plan, old = setup
    publish_checkpoint(plan)
    Path(plan["old_log"]).write_text("Epoch 99: 100%| | 488/488\r")
    original_killpg, calls = os.killpg, []
    def signal_with_replacement(pgid, signum):
        calls.append(signum)
        original_killpg(pgid, signum)
        if signum == signal.SIGSTOP:
            replacement = guard.read_json(plan["current_run"])
            replacement["trainer"] = {**replacement["trainer"], "start_ticks": "changed-after-freeze"}
            guard.atomic_json(plan["current_run"], replacement)
    monkeypatch.setattr(guard.os, "killpg", signal_with_replacement)
    with pytest.raises(RuntimeError, match="replaced"):
        run(plan)
    assert signal.SIGSTOP in calls and signal.SIGCONT in calls
    assert signal.SIGTERM not in calls and signal.SIGKILL not in calls
    assert old.poll() is None


def test_prepare_freezes_eight_rank_and_configuration_identity(setup, monkeypatch):
    plan, old = setup
    monkeypatch.setattr(guard, "training_ranks", lambda pid: {str(rank): plan["trainer"] for rank in range(8)})
    result = guard.prepare_plan(plan["current_run"], 100, plan["job_dir"])
    assert len(result["ranks"]) == 8
    assert result["completed_epochs"] == 100 and result["steps_per_epoch"] == 488
    assert result["source_config_sha256"] == guard.file_hash(result["source_config"])
    guard.atomic_json(Path(plan["root"]) / ".pipeline/current_training_stop.json",
                      {"trainer": plan["trainer"], "completed_epochs": 100, "watcher": None})
    with pytest.raises(FileExistsError, match="already exists"):
        guard.prepare_plan(plan["current_run"], 100, plan["job_dir"])
    assert old.poll() is None


def test_existing_attempt_marker_refuses_rearm(setup, monkeypatch):
    plan, old = setup
    monkeypatch.setattr(guard, "training_ranks", lambda pid: {"0": plan["trainer"]})
    Path(plan["stop_marker"]).write_text("{}")
    with pytest.raises(FileExistsError, match="already attempted"):
        guard.prepare_plan(plan["current_run"], 100, plan["job_dir"])
    assert old.poll() is None


@pytest.mark.skipif(not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"),
                    reason="Python build does not expose Linux pidfd signalling")
@pytest.mark.parametrize("setup", ["ignore_term"], indirect=True)
def test_real_pidfd_escalation_stops_only_captured_ignoring_cpu_process(setup):
    plan, old = setup
    publish_checkpoint(plan)
    Path(plan["old_log"]).write_text("Epoch 99: 100%| | 488/488\r")
    assert run(plan) == 0
    state = guard.read_json(Path(plan["job_dir"]) / "state.json")
    assert state["state"] == "stopped" and state["safe_forced_shutdown_available"] is True
    assert state["shutdown_result"] == {"termination": "sigkill", "forced_processes": [plan["trainer"]]}
    assert guard.process_identity(old.pid) is None
