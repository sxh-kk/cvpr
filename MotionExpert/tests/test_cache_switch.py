"""Accept the cache before a fresh natural save handoff; harmless CPU processes."""
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
            "job_dir": str(job), "created_at": switch.now()}
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


def _publish_boundary(plan, completed, *, epoch=None, batch=0):
    epoch = completed - 1 if epoch is None else epoch
    Path(plan["old_log"]).write_text(f"Epoch {epoch}: 1%| | {batch}/488\r")
    return _checkpoint(Path(plan["output"]), completed)


def test_cache_ready_waits_for_last_publication_without_reading_checkpoint(setup, monkeypatch):
    plan, old, children = setup
    path = _checkpoint(Path(plan["output"]), 40, link=False)
    calls, waiting = [], []
    original_open = Path.open
    def no_checkpoint_reads(self, *args, **kwargs):
        if self.suffix == ".ckpt": raise AssertionError("Supervisor opened checkpoint contents")
        return original_open(self, *args, **kwargs)
    monkeypatch.setattr(Path, "open", no_checkpoint_reads)
    monkeypatch.setenv("MOTIONEXPERT_DATA_DIR", "/incorrect")
    monkeypatch.setenv("LOCAL_RANK", "3")
    def published():
        waiting.append(old.poll() is None)
        Path(plan["old_log"]).write_text("Epoch 39: 100%| | 488/488\r")
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


def test_cache_pending_keeps_old_running_until_accepted_at_current_boundary(setup):
    plan, old, children = setup
    _cache_pending(plan)
    path = _publish_boundary(plan, 40)
    observations = []
    def ready():
        observations.append(old.poll() is None)
        observations.append(switch.read_json(Path(plan["job_dir"]) / "state.json")["state"] == "waiting_cache")
        _cache_ready(plan)
    timer = threading.Timer(.1, ready); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children)) == 0
    finally:
        timer.cancel()
    assert observations == [True, True]
    assert switch.read_json(plan["current_run"])["checkpoint"] == str(path)


def test_cancelling_pending_cache_does_not_stop_training_even_with40_saved(setup):
    plan, old, _ = setup
    _cache_pending(plan); _publish_boundary(plan, 40)
    stop = threading.Event(); timer = threading.Timer(.1, stop.set); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, stop=stop) == 0
    finally:
        timer.cancel()
    assert old.poll() is None
    assert switch.read_json(plan["current_run"])["trainer"] == plan["trainer"]
    assert not (Path(plan["output"]) / "window_cache_launch.json").exists()


def test_acceptance_after_next_epoch_started_waits60_then_accepts_epoch60_zero_batch(setup):
    plan, old, children = setup
    _publish_boundary(plan, 40, epoch=40, batch=8)
    observations = []
    def published():
        state = switch.read_json(Path(plan["job_dir"]) / "state.json")
        observations.extend([old.poll() is None, state["boundary"] == 60])
        _publish_boundary(plan, 60, epoch=60, batch=0)
    timer = threading.Timer(.1, published); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children)) == 0
    finally:
        timer.cancel()
    state = switch.read_json(plan["current_run"])
    assert observations == [True, True] and state["saved_checkpoint"]["completed_epochs"] == 60
    assert state["unsaved_progress"] == {"epoch": 60, "batch": 0, "steps_per_epoch": 488}


def test_stale_current_save_waits_for_next_fresh_natural_boundary(setup):
    plan, old, children = setup
    path = _publish_boundary(plan, 40)
    previous = time.time() - 180; os.utime(path, (previous, previous))
    observations = []
    def next_save():
        observations.extend([old.poll() is None,
            switch.read_json(Path(plan["job_dir"]) / "state.json")["boundary"] == 60])
        _publish_boundary(plan, 60, epoch=60, batch=1)
    timer = threading.Timer(.1, next_save); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children)) == 0
    finally:
        timer.cancel()
    assert observations == [True, True]
    assert switch.read_json(plan["current_run"])["boundary"] == 60


def test_later_completed_epoch_during_preparation_defers_without_signalling_old(setup, monkeypatch):
    plan, old, children = setup
    _publish_boundary(plan, 40)
    checks, signals, observations = [], [], []
    original_validate, original_signal = switch.validate_acceptance, os.killpg
    def validate(plan):
        result = original_validate(plan); checks.append(True)
        if len(checks) == 2:
            Path(plan["old_log"]).write_text("Epoch 41: 1%| | 1/488\r")
        return result
    def signal_group(group, signum):
        signals.append((group, signum)); original_signal(group, signum)
    monkeypatch.setattr(switch, "validate_acceptance", validate)
    monkeypatch.setattr(os, "killpg", signal_group)
    def published():
        observations.extend([old.poll() is None, signals == []])
        _publish_boundary(plan, 60, epoch=60, batch=0)
    timer = threading.Timer(.15, published); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children)) == 0
    finally:
        timer.cancel()
    assert observations == [True, True]
    assert switch.read_json(plan["current_run"])["boundary"] == 60


@pytest.mark.parametrize("failure", ["partial", "failed_with_building", "manifest_hash"])
def test_cache_acceptance_failure_preserves_running_training(setup, failure):
    plan, old, _ = setup
    _publish_boundary(plan, 40)
    if failure == "partial":
        path = Path(plan["cache_root"]) / "manifest.json"
        data = switch.read_json(path); data.update(diagnostic=True, scope="selected_episodes")
    else:
        path = Path(plan["acceptance"]); data = switch.read_json(path)
        if failure == "failed_with_building":
            _cache_pending(plan)
            data.update(passed=False, state="failed", error="builder exited", finished_at=switch.now())
        else: data["cache_manifest_sha256"] = "0" * 64
    switch.atomic_json(path, data)
    with pytest.raises(ValueError): switch.run_switch(plan, poll_seconds=.01)
    assert old.poll() is None
    assert not (Path(plan["output"]) / "window_cache_launch.json").exists()
    assert switch.read_json(plan["current_run"])["trainer"] == plan["trainer"]


def test_identity_change_during_marker_is_checked_before_sigstop(setup, monkeypatch):
    plan, old, _ = setup
    _publish_boundary(plan, 40)
    signals = []; original_signal, original_fsync = os.killpg, os.fsync
    def signal_group(group, signum):
        signals.append((group, signum)); original_signal(group, signum)
    def replace_pointer(fd):
        original_fsync(fd)
        if Path(os.readlink(f"/proc/self/fd/{fd}")) == Path(plan["output"]) / "window_cache_launch.json":
            current = switch.read_json(plan["current_run"])
            current["trainer"] = {**current["trainer"], "start_ticks": "new identity"}
            switch.atomic_json(plan["current_run"], current)
    monkeypatch.setattr(os, "killpg", signal_group); monkeypatch.setattr(os, "fsync", replace_pointer)
    with pytest.raises(RuntimeError, match="Current run was replaced"):
        switch.run_switch(plan, poll_seconds=.01)
    assert old.poll() is None and signals == []


def test_busy_launcher_gate_cannot_stop_old_training(setup, monkeypatch):
    plan, old, _ = setup
    _publish_boundary(plan, 40)
    signals = []; original_signal = os.killpg
    def signal_group(group, signum):
        signals.append((group, signum)); original_signal(group, signum)
    monkeypatch.setattr(os, "killpg", signal_group)
    with switch.locked(Path(plan["root"]) / "training/.launcher-reservation.lock"), pytest.raises(BlockingIOError):
        switch.run_switch(plan, poll_seconds=.01)
    assert old.poll() is None and signals == []


def test_failed_launch_is_attempted_once_after_stop(setup):
    plan, old, _ = setup
    _publish_boundary(plan, 40)
    calls = []
    def fail(command, **options):
        calls.append(command); raise OSError("launch failed")
    with pytest.raises(OSError, match="launch failed"):
        switch.run_switch(plan, poll_seconds=.01, launch=fail)
    assert old.poll() is not None and len(calls) == 1
    assert (Path(plan["output"]) / "window_cache_launch.json").is_file()


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
                                 resume_config=plan["resume_config"])
    assert Path(frozen["resume_config"]).read_bytes() == Path(plan["resume_config"]).read_bytes()
    assert "completed_epochs" not in frozen


@pytest.mark.parametrize("epoch,completed,expected", [(31,20,40),(39,40,40),(40,40,60),(59,40,60),(60,60,80)])
def test_acceptance_boundary_selection(epoch, completed, expected):
    assert switch.next_boundary({"epoch": epoch}, {"completed_epochs": completed}) == expected


def test_acceptance_after_next_epoch_race_reads_latest_progress(setup, monkeypatch):
    plan, old, children = setup
    _publish_boundary(plan, 40)
    original, reads, observed = switch.completed_checkpoint, [], []
    def read_last(output):
        result = original(output)
        reads.append(True)
        if len(reads) == 1:
            # Lightning enters its next epoch between the link/progress reads.
            Path(plan["old_log"]).write_text("Epoch 40: 0%| | 0/488\r")
        return result
    monkeypatch.setattr(switch, "completed_checkpoint", read_last)
    def saved_next():
        observed.extend([old.poll() is None,
            switch.read_json(Path(plan["job_dir"]) / "state.json")["boundary"] == 60])
        _publish_boundary(plan, 60, epoch=60, batch=0)
    timer = threading.Timer(.1, saved_next); timer.start()
    try:
        assert switch.run_switch(plan, poll_seconds=.01, launch=_launch(plan, old, children)) == 0
    finally:
        timer.cancel()
    assert observed == [True, True]
    assert switch.read_json(plan["current_run"])["boundary"] == 60


def test_completed_unsaved_epoch_is_not_allowed_as_microsteps(setup):
    plan, _, _ = setup
    _publish_boundary(plan, 40)
    checkpoint = switch.completed_checkpoint(plan["output"])
    progress = {"epoch": 40, "batch": 488, "steps_per_epoch": 488}
    assert not switch.recent_boundary(checkpoint, 40, progress)
    progress["batch"] = 0
    assert switch.recent_boundary(checkpoint, 40, progress)
    progress.update(epoch=39, batch=488)
    assert switch.recent_boundary(checkpoint, 40, progress)
