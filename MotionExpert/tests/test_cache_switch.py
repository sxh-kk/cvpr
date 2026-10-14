"""Stop at a fixed natural save, then resume after cache acceptance; CPU only."""
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from MotionExpert.config import as_dict, smplh_motion_expert_config
from MotionExpert import switch_to_window_cache as switch


def _checkpoint(output, completed, *, link=True):
    path = output / f"epoch={completed-1}-step={completed*488}.ckpt"
    # Deliberately not a torch/ZIP checkpoint: the supervisor must never read it.
    path.write_bytes(b"opaque synthetic checkpoint; contents are not inspected")
    if link:
        temporary = output / ".last.tmp"
        temporary.symlink_to(path.name)
        temporary.replace(output / "last.ckpt")
    return path


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "MixEgoMotion"
    output = root / "training/run"
    job = root / ".pipeline/jobs/switch"
    cache = root / "training-cache/derived"
    vision = root / "archive/vision"
    for path in (output, job, cache, vision):
        path.mkdir(parents=True, exist_ok=True)
    config = as_dict(smplh_motion_expert_config())
    config["DATA"]["ARCHIVE_ROOT"] = str(root / "archive")
    config["DATA"]["DATA_DIR"] = str(root / "archive")
    config["DATA"]["VISION_PROFILE_ROOT"] = str(vision)
    config["TRAIN"]["EXP_PATH"] = str(output)
    source = root / "source-config.yaml"
    source.write_text(yaml.safe_dump(config, sort_keys=False))
    resumed = deepcopy(config)
    resumed["DATA"]["WINDOW_CACHE_ROOT"] = str(cache)
    snapshot = job / "resume-config.yaml"
    snapshot.write_text(yaml.safe_dump(resumed, sort_keys=False))
    old = subprocess.Popen([sys.executable, "-u", "-c", "import time; time.sleep(600)"],
                           start_new_session=True)
    log = root / "old-training.log"
    log.write_text("Epoch 31: 55%|_____ | 270/488 [00:20<00:10]\r")
    pointer = root / "training/current_run.json"
    identity = switch.process_identity(old.pid)
    current = {"training_running": True, "pid": old.pid, "trainer": identity,
               "watcher": None, "gpus": list(range(8)), "config": str(source),
               "output": str(output), "log": str(log), "command": [sys.executable, "-m", "MotionExpert.train"]}
    switch.atomic_json(pointer, current)
    plan = {"contract": switch.CONTRACT, "current_run": str(pointer), "root": str(root),
            "output": str(output), "old_run": current, "trainer": identity, "guardian": None,
            "pgid": old.pid, "ranks": {"0": identity}, "old_log": str(log),
            "gpus": list(range(8)), "python": sys.executable,
            "cache_root": str(cache), "acceptance": str(cache / "acceptance.json"),
            "source_config": str(source), "source_config_sha256": switch.file_hash(source),
            "resume_config": str(snapshot), "resume_config_sha256": switch.file_hash(snapshot),
            "job_dir": str(job), "created_at": switch.now(), "completed_epochs": 40}
    switch.atomic_json(cache / "manifest.json", {"contract": switch.CACHE_CONTRACT, "state": "complete",
        "scope": "all", "diagnostic": False, "splits": ["train", "val", "test"]})
    switch.atomic_json(plan["acceptance"], {"format": switch.ACCEPTANCE_FORMAT, "passed": True,
        "cache_root": str(cache), "cache_manifest_sha256": switch.file_hash(cache / "manifest.json"),
        "source_config_sha256": plan["source_config_sha256"], "resume_config_sha256": plan["resume_config_sha256"]})
    _checkpoint(output, 20)
    children = []
    try:
        yield plan, old, children
    finally:
        for child in [old, *children]:
            if switch.process_identity(child.pid):
                os.killpg(child.pid, signal.SIGTERM)
                os.killpg(child.pid, signal.SIGCONT)
            child.wait(timeout=5)



def _cache_pending(plan):
    switch.atomic_json(Path(plan["cache_root"]) / "manifest.json", {"state": "building"})
    switch.atomic_json(plan["acceptance"], {"format": switch.ACCEPTANCE_FORMAT, "passed": False,
                                         "state": "waiting_cache", "started_at": switch.now()})


def _cache_ready(plan):
    path = Path(plan["cache_root"]) / "manifest.json"
    switch.atomic_json(path, {"contract": switch.CACHE_CONTRACT, "state": "complete", "scope": "all",
                             "diagnostic": False, "splits": ["train", "val", "test"]})
    switch.atomic_json(plan["acceptance"], {"format": switch.ACCEPTANCE_FORMAT, "passed": True,
        "state": "complete", "cache_root": plan["cache_root"], "finished_at": switch.now(),
        "cache_manifest_sha256": switch.file_hash(path),
        "source_config_sha256": plan["source_config_sha256"], "resume_config_sha256": plan["resume_config_sha256"]})


def _launch(plan, old, children, calls=None, after=None):
    def launch(command, **options):
        assert old.poll() is not None
        assert "--resume" in command and "--from-scratch" not in command
        if calls is not None: calls.append((command, options))
        child = subprocess.Popen([sys.executable, "-c", "print('resumed once')"], **options)
        children.append(child)
        if after is not None: after(child)
        return child
    return launch


def test_cache_ready_before40_waits_for_the_published_symlink_without_reading_contents(setup, monkeypatch):
    plan, old, children = setup
    path = _checkpoint(Path(plan["output"]), 40, link=False)
    calls, waiting = [], []
    original_open = Path.open
    def no_checkpoint_reads(self, *args, **kwargs):
        if self.suffix == ".ckpt":
            raise AssertionError("Supervisor attempted to open checkpoint contents")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_checkpoint_reads)
    monkeypatch.setenv("MOTIONEXPERT_DATA_DIR", "/incorrect")
    monkeypatch.setenv("LOCAL_RANK", "3")
    def published():
        waiting.append(old.poll() is None)
        temp = Path(plan["output"]) / ".last.tmp"
        temp.symlink_to(path.name); temp.replace(Path(plan["output"]) / "last.ckpt")
    timer = threading.Timer(.1, published); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children, calls)) == 0
    finally:
        timer.cancel()
    assert waiting == [True] and len(calls) == 1
    command, options = calls[0]
    assert command == [sys.executable, "-u", "-m", "MotionExpert.train", "--config", plan["resume_config"],
                       "--resume", str(path), "--output-dir", plan["output"]]
    assert len(options["pass_fds"]) == 2 and options["start_new_session"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "LOCAL_RANK" not in options["env"] and "MOTIONEXPERT_DATA_DIR" not in options["env"]
    state = switch.read_json(plan["current_run"])
    assert state["state"] == "completed" and state["initialization"] == "resume"
    assert not state["automatic_restart_enabled"]
    assert state["checkpoint_inspection"] == "natural_last_symlink_and_filename_only"


def test_40_stops_while_cache_builds_then_resumes_exactly_that_saved_path(setup):
    plan, old, children = setup
    _cache_pending(plan)
    checkpoint = _checkpoint(Path(plan["output"]), 40)
    observed, errors = [], []
    def accept_later():
        try:
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                state = switch.read_json(plan["current_run"])
                if state.get("state") == "waiting_cache_after_stop": break
                time.sleep(.01)
            assert old.poll() is not None
            assert state["training_running"] is False and state["checkpoint"] == str(checkpoint)
            with pytest.raises(BlockingIOError):
                with switch.locked(Path(plan["root"]) / "training/.dino-to-motion.lock"): pass
            # Other cooperating watchers take the gate, then fail the held run lock.
            with switch.locked(Path(plan["root"]) / "training/.launcher-reservation.lock"):
                with pytest.raises(BlockingIOError):
                    with switch.locked(Path(plan["root"]) / "training/.dino-to-motion.lock"): pass
            observed.append(state)
            _cache_ready(plan)
        except Exception as error:
            errors.append(error)
    timer = threading.Timer(.1, accept_later); timer.start()
    watchdog = threading.Timer(4, lambda: stop.set()); stop = threading.Event(); watchdog.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, stop=stop,
                                 launch=_launch(plan, old, children)) == 0
    finally:
        timer.cancel(); watchdog.cancel()
    assert not errors and len(observed) == 1
    assert switch.read_json(plan["current_run"])["checkpoint"] == str(checkpoint)


def test_40_without_cache_remains_stopped_if_supervisor_is_cancelled(setup):
    plan, old, _ = setup
    _cache_pending(plan)
    _checkpoint(Path(plan["output"]), 40)
    stop = threading.Event()
    timer = threading.Timer(.3, stop.set); timer.start()
    calls = []
    try:
        assert switch.run_switch(plan, poll_seconds=.01, stop=stop, launch=lambda *a, **k: calls.append(a)) == 0
    finally:
        timer.cancel()
    assert old.poll() is not None and calls == []
    state = switch.read_json(plan["current_run"])
    assert state["state"] == "supervisor_stopped" and state["training_running"] is False
    assert state["cache_acceptance_pending"] is True and state["saved_checkpoint"]["completed_epochs"] == 40


def test_existing40_is_used_even_if_the_next_epoch_log_has_started(setup):
    plan, old, children = setup
    checkpoint = _checkpoint(Path(plan["output"]), 40)
    Path(plan["old_log"]).write_text("Epoch 40: 1%| | 5/488\r")
    calls = []
    assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children, calls)) == 0
    assert calls[0][0][calls[0][0].index("--resume") + 1] == str(checkpoint)
    assert switch.read_json(plan["current_run"])["boundary"] == 40


@pytest.mark.parametrize("failure", ["partial", "failed_with_building", "manifest_hash"])
def test_failed_acceptance_after40_keeps_training_stopped_without_launch(setup, failure):
    plan, old, _ = setup
    _checkpoint(Path(plan["output"]), 40)
    if failure == "partial":
        path = Path(plan["cache_root"]) / "manifest.json"
        data = switch.read_json(path); data.update(diagnostic=True, scope="selected_episodes")
    else:
        path = Path(plan["acceptance"])
        data = switch.read_json(path)
        if failure == "failed_with_building":
            _cache_pending(plan)
            data.update(passed=False, state="failed", error="builder exited", finished_at=switch.now())
        else:
            data["cache_manifest_sha256"] = "0" * 64
    switch.atomic_json(path, data)
    calls = []
    with pytest.raises(ValueError):
        switch.run_switch(plan, poll_seconds=.01, launch=lambda *a, **k: calls.append(a))
    assert old.poll() is not None and calls == []
    state = switch.read_json(plan["current_run"])
    assert state["state"] == "failed" and state["training_running"] is False
    assert (Path(plan["output"]) / "window_cache_launch.json").is_file()


def test_cancellation_before40_preserves_the_running_old_session(setup):
    plan, old, _ = setup
    _cache_pending(plan)
    stop = threading.Event(); timer = threading.Timer(.08, stop.set); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, stop=stop) == 0
    finally:
        timer.cancel()
    assert old.poll() is None
    assert switch.read_json(plan["current_run"])["trainer"] == plan["trainer"]


def test_identity_change_during_marker_is_checked_before_sigstop(setup, monkeypatch):
    plan, old, _ = setup
    _checkpoint(Path(plan["output"]), 40)
    signals = []; original_signal, original_fsync = os.killpg, os.fsync
    def signal_group(group, signum):
        signals.append((group, signum)); original_signal(group, signum)
    def replace_pointer(fd):
        original_fsync(fd)
        if Path(os.readlink(f"/proc/self/fd/{fd}")) == Path(plan["output"]) / "window_cache_launch.json":
            current = switch.read_json(plan["current_run"])
            current["trainer"] = {**current["trainer"], "start_ticks": "new identity"}
            switch.atomic_json(plan["current_run"], current)
    monkeypatch.setattr(os, "killpg", signal_group)
    monkeypatch.setattr(os, "fsync", replace_pointer)
    with pytest.raises(RuntimeError, match="Current run was replaced"):
        switch.run_switch(plan, poll_seconds=.01)
    assert old.poll() is None and signals == []


def test_busy_launcher_gate_cannot_stop_old_training(setup, monkeypatch):
    plan, old, _ = setup
    _checkpoint(Path(plan["output"]), 40)
    signals = []; original_signal = os.killpg
    def signal_group(group, signum):
        signals.append((group, signum)); original_signal(group, signum)
    monkeypatch.setattr(os, "killpg", signal_group)
    with switch.locked(Path(plan["root"]) / "training/.launcher-reservation.lock"), pytest.raises(BlockingIOError):
        switch.run_switch(plan, poll_seconds=.01)
    assert old.poll() is None and signals == []
    assert not (Path(plan["output"]) / "window_cache_launch.json").exists()


def test_failed_launch_is_attempted_once_after_stop(setup):
    plan, old, _ = setup
    _checkpoint(Path(plan["output"]), 40)
    calls = []
    def fail(command, **options):
        calls.append(command); raise OSError("launch failed")
    with pytest.raises(OSError, match="launch failed"):
        switch.run_switch(plan, poll_seconds=.01, launch=fail)
    assert old.poll() is not None and len(calls) == 1
    assert switch.read_json(plan["current_run"])["training_running"] is False
    assert (Path(plan["output"]) / "window_cache_launch.json").is_file()


def test_configurable_completed_boundary_is_frozen(setup):
    plan, old, children = setup
    plan["completed_epochs"] = 60
    checkpoint = _checkpoint(Path(plan["output"]), 60)
    calls = []
    assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children, calls)) == 0
    assert calls[0][0][calls[0][0].index("--resume") + 1] == str(checkpoint)


def test_snapshot_reuses_config_bytes_and_changes_only_cache_root(setup, monkeypatch):
    plan, _, _ = setup
    source = yaml.safe_load(Path(plan["source_config"]).read_text())
    resumed = yaml.safe_load(Path(plan["resume_config"]).read_text())
    switch.validate_resume_config(source, resumed, plan["cache_root"])
    changed = deepcopy(resumed); changed["DATA"]["NUM_WORKERS"] = 32
    with pytest.raises(ValueError, match="NUM_WORKERS"):
        switch.validate_resume_config(source, changed, plan["cache_root"])
    monkeypatch.setattr(switch, "training_ranks", lambda _: plan["ranks"])
    job = Path(plan["job_dir"]).parent / "frozen-plan"; job.mkdir()
    frozen = switch.prepare_plan(plan["current_run"], plan["cache_root"], plan["acceptance"], job,
                                 resume_config=plan["resume_config"], completed_epochs=40)
    assert Path(frozen["resume_config"]).read_bytes() == Path(plan["resume_config"]).read_bytes()
    assert frozen["completed_epochs"] == 40
