"""Exercise the new handoff only; never launch the real model or touch GPUs."""
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from MotionExpert.config import load_config
from MotionExpert import watch_dino as watch


@pytest.fixture
def setup(tmp_path, monkeypatch):
    cfg = load_config(watch.DEFAULT_CONFIG)
    root = tmp_path / "MixEgoMotion"
    cfg.DATA.DATA_DIR = cfg.DATA.ARCHIVE_ROOT = str(root / "archive")
    cfg.DATA.VISION_PROFILE_ROOT = str(root / "archive/vision")
    cfg.TRAIN.EXP_PATH = str(root / "training/run")
    vision = Path(cfg.DATA.VISION_PROFILE_ROOT)
    vision.mkdir(parents=True)
    manifest = {"contract": watch.CONTRACT, "scope": "all", "state": "complete", "fps": 30,
                "native_fps": 30, "sampling_step": 1, "frame_phase": 0, "shape": [5, 1024],
                "storage_dtype": "float32", "binding": "test-binding",
                "source_root": cfg.DATA.ARCHIVE_ROOT, "output_root": str(vision),
                "completed_episodes": 2, "source_episode_count": 2,
                "completed_frames": 6, "total_frames": 6, "episode_indices": [0, 1],
                "validity_equal_source": True}
    watch.atomic_json(vision / "manifest.json", manifest)
    job = root / ".pipeline/jobs/watch"
    job.mkdir(parents=True)
    monkeypatch.setattr(watch, "active_trainers", lambda: [])
    kwargs = dict(job_dir=job, extraction_state=root / ".pipeline/current_dino_extraction.json",
                  gpus=list(range(8)), poll_seconds=0.01, stop=threading.Event())
    return cfg, manifest, kwargs


@pytest.mark.parametrize("change", [
    {"state": "failed", "error": "decoder failed"},
    {"scope": "subset"}, {"fps": 5}, {"sampling_step": 6},
    {"completed_episodes": 1}, {"completed_frames": 5},
    {"validity_equal_source": False}, {"episode_indices": [1, 0]},
    {"binding": "different"}, {"source_root": "/different"},
])
def test_invalid_or_partial_manifest_never_launches(setup, change):
    cfg, manifest, kwargs = setup
    manifest.update(change)
    with pytest.raises((ValueError, RuntimeError)):
        watch.complete_manifest(manifest, cfg, "test-binding")


def test_exit_before_completion_fails_without_launch(setup):
    cfg, manifest, kwargs = setup
    manifest["state"] = "extracting"
    watch.atomic_json(Path(cfg.DATA.VISION_PROFILE_ROOT) / "manifest.json", manifest)
    calls = []
    with pytest.raises(RuntimeError, match="exited before"):
        watch.run_watch(cfg, **kwargs, check=lambda c: calls.append("check"),
                        launch=lambda *a, **k: calls.append("launch"))
    assert calls == []
    assert watch.read_json(kwargs["job_dir"] / "state.json")["state"] == "failed"
    assert not Path(cfg.TRAIN.EXP_PATH).exists()


def test_live_extraction_completes_and_exits_then_training_starts(setup):
    cfg, manifest, kwargs = setup
    vision = Path(cfg.DATA.VISION_PROFILE_ROOT)
    manifest["state"] = "extracting"
    manifest["completed_episodes"] = manifest["completed_frames"] = 0
    watch.atomic_json(vision / "manifest.json", manifest)
    code = """
import fcntl, json, pathlib, sys, time
root = pathlib.Path(sys.argv[2])
with (root / '.extraction.lock').open('a') as stream:
    fcntl.flock(stream, fcntl.LOCK_EX)
    time.sleep(.15)
    data = json.loads((root / 'manifest.json').read_text())
    data.update(state='complete', completed_episodes=2, completed_frames=6)
    tmp = root / 'manifest.tmp'
    tmp.write_text(json.dumps(data))
    tmp.replace(root / 'manifest.json')
    time.sleep(.15)
"""
    extractor = subprocess.Popen([sys.executable, "-c", code, "extract-smplh-features", str(vision)])
    watch.atomic_json(kwargs["extraction_state"], {"pid": extractor.pid, "output": str(vision)})
    checked = []

    def check(_):
        assert not watch.process_alive(watch.process_identity(extractor.pid))
        assert watch.read_json(vision / "manifest.json")["state"] == "complete"
        checked.append(True)

    def launch(command, **options):
        return subprocess.Popen([sys.executable, "-c", "print('training after extraction')"], **options)

    try:
        assert watch.run_watch(cfg, **kwargs, check=check, launch=launch) == 0
        assert checked == [True]
    finally:
        if extractor.poll() is None:
            extractor.terminate()
        extractor.wait(timeout=5)


def test_success_launches_once_with_exact_training_command_and_lock_inheritance(setup, monkeypatch):
    cfg, _, kwargs = setup
    checks, launches = [], []
    monkeypatch.setenv("MOTIONEXPERT_DATA_DIR", "/wrong/archive")
    original = copy.deepcopy(watch.as_dict(cfg))

    def launch(command, **options):
        launches.append((command, options))
        # A real harmless subprocess tests fd inheritance, log redirection and
        # detached supervision without constructing a model or invoking CUDA.
        code = "import os; print('handoff-ok', os.environ['CUDA_VISIBLE_DEVICES'], flush=True)"
        return subprocess.Popen([sys.executable, "-u", "-c", code], **options)

    assert watch.run_watch(cfg, **kwargs, check=lambda c: checks.append(c), launch=launch) == 0
    assert len(checks) == len(launches) == 1
    command, options = launches[0]
    assert command == [sys.executable, "-u", "-m", "MotionExpert.train", "--config",
                       str(kwargs["job_dir"] / "config.yaml"), "--from-scratch"]
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "MOTIONEXPERT_DATA_DIR" not in options["env"]
    assert options["start_new_session"] and len(options["pass_fds"]) == 2
    assert options["cwd"] == watch.REPO_ROOT
    monkeypatch.delenv("MOTIONEXPERT_DATA_DIR")
    frozen = watch.load_config(kwargs["job_dir"] / "config.yaml")
    assert watch.as_dict(frozen.DATA) == original["DATA"]
    assert watch.as_dict(frozen.TRAIN) == original["TRAIN"]
    state = watch.read_json(kwargs["job_dir"] / "state.json")
    assert state["state"] == "completed" and state["exit_code"] == 0
    assert not state["training_running"] and not state["automatic_restart_enabled"]
    assert "handoff-ok 0,1,2,3,4,5,6,7" in (kwargs["job_dir"] / "training.log").read_text()
    assert (Path(cfg.TRAIN.EXP_PATH) / "auto_train_launch.json").is_file()
    with pytest.raises(FileExistsError, match="existing run"):
        watch.run_watch(cfg, **kwargs, check=lambda _: None, launch=launch)
    assert len(launches) == 1


def test_preflight_failure_cannot_launch(setup):
    cfg, _, kwargs = setup
    calls = []

    def fail(_):
        raise ValueError("profile differs")

    with pytest.raises(ValueError, match="profile differs"):
        watch.run_watch(cfg, **kwargs, check=fail, launch=lambda *a, **k: calls.append(1))
    assert calls == []
    assert not Path(cfg.TRAIN.EXP_PATH).exists()
    assert watch.read_json(kwargs["job_dir"] / "state.json")["state"] == "failed"


def test_preflight_io_error_is_not_mistaken_for_busy_extraction(setup):
    cfg, _, kwargs = setup

    def fail(_):
        raise BlockingIOError("preflight IO failure")

    with pytest.raises(BlockingIOError, match="preflight IO failure"):
        watch.run_watch(cfg, **kwargs, check=fail)
    assert watch.read_json(kwargs["job_dir"] / "state.json")["state"] == "failed"
    assert not Path(cfg.TRAIN.EXP_PATH).exists()


@pytest.mark.parametrize("still_running", [True, False])
def test_complete_still_waits_for_pid_and_extraction_lock(setup, monkeypatch, still_running):
    cfg, _, kwargs = setup
    calls = []
    monkeypatch.setattr(watch, "extractor_identity", lambda *a: {"pid": os.getpid()})
    monkeypatch.setattr(watch, "process_alive", lambda _: still_running)

    def stop_when_waiting():
        with watch.locked(Path(cfg.DATA.VISION_PROFILE_ROOT) / ".extraction.lock"):
            # A timer interrupts the wait while the real flock remains held.
            timer = threading.Timer(0.1, kwargs["stop"].set)
            timer.start()
            try:
                return watch.run_watch(cfg, **kwargs, check=lambda _: calls.append("check"),
                                       launch=lambda *a, **k: calls.append("launch"))
            finally:
                timer.cancel()

    assert stop_when_waiting() == 0
    assert calls == [] and not Path(cfg.TRAIN.EXP_PATH).exists()
    assert watch.read_json(kwargs["job_dir"] / "state.json")["state"] == "stopped"


def test_training_failure_is_recorded_and_never_restarted(setup):
    cfg, _, kwargs = setup
    launches = []

    def launch(command, **options):
        launches.append(command)
        return subprocess.Popen([sys.executable, "-c", "raise SystemExit(7)"], **options)

    assert watch.run_watch(cfg, **kwargs, check=lambda _: None, launch=launch) == 1
    state = watch.read_json(kwargs["job_dir"] / "state.json")
    assert state["state"] == "failed" and state["exit_code"] == 7
    assert len(launches) == 1 and not state["automatic_restart_enabled"]


def test_stopping_watcher_preserves_training_and_prevents_duplicate_launch(setup):
    cfg, _, kwargs = setup
    children = []

    def launch(command, **options):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"], **options)
        children.append(child)
        kwargs["stop"].set()
        return child

    try:
        assert watch.run_watch(cfg, **kwargs, check=lambda _: None, launch=launch) == 0
        assert children[0].poll() is None
        state = watch.read_json(kwargs["job_dir"] / "state.json")
        assert state["state"] == "watcher_stopped" and state["training_running"]
        with pytest.raises(BlockingIOError):
            with watch.locked(Path(cfg.DATA.ARCHIVE_ROOT).parent / "training/.dino-to-motion.lock"):
                pass
        with pytest.raises(BlockingIOError):
            with watch.locked(Path(cfg.DATA.VISION_PROFILE_ROOT) / ".extraction.lock"):
                pass
    finally:
        # Only our deliberately created harmless test process is terminated.
        for child in children:
            child.terminate()
            child.wait(timeout=5)


def test_reused_pid_is_not_alive():
    identity = watch.process_identity(os.getpid())
    assert watch.process_alive(identity)
    identity["start_ticks"] = "wrong-start"
    assert not watch.process_alive(identity)


def test_existing_trainer_blocks_watch(setup, monkeypatch):
    cfg, _, kwargs = setup
    monkeypatch.setattr(watch, "active_trainers", lambda: [123])
    with pytest.raises(RuntimeError, match="already running"):
        watch.run_watch(cfg, **kwargs)


def test_background_cli_normalizes_paths_and_stops_without_touching_extractor(setup):
    cfg, manifest, kwargs = setup
    job = kwargs["job_dir"]
    manifest["state"] = "extracting"
    watch.atomic_json(Path(cfg.DATA.VISION_PROFILE_ROOT) / "manifest.json", manifest)
    config = job.parent / "source.yaml"
    config.write_text(yaml.safe_dump(watch.as_dict(cfg)))
    extractor = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)",
                                  "extract-smplh-features", cfg.DATA.VISION_PROFILE_ROOT])
    watch.atomic_json(kwargs["extraction_state"], {"pid": extractor.pid, "output": cfg.DATA.VISION_PROFILE_ROOT})
    env = os.environ.copy()
    env["PYTHONPATH"] = str(watch.REPO_ROOT)
    env.pop("MOTIONEXPERT_DATA_DIR", None)
    watcher = None
    try:
        result = subprocess.run(
            [sys.executable, "-m", "MotionExpert.watch_dino", "--config", "source.yaml",
             "--job-dir", "watch", "--extraction-state", str(kwargs["extraction_state"]),
             "--poll-seconds", "0.1", "--background"],
            cwd=job.parent, env=env, capture_output=True, text=True, timeout=20,
        )
        assert result.returncode == 0, result.stderr
        reply = json.loads(result.stdout)
        assert reply["state"] == "waiting_dino" and reply["job_dir"] == str(job)
        watcher = watch.process_identity(reply["watcher_pid"])
        assert watcher is not None
        os.kill(watcher["pid"], signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and watch.read_json(job / "state.json")["state"] != "stopped":
            time.sleep(0.02)
        assert watch.read_json(job / "state.json")["state"] == "stopped"
        assert extractor.poll() is None
        assert not Path(cfg.TRAIN.EXP_PATH).exists()
    finally:
        if watcher and watch.process_alive(watcher):
            os.kill(watcher["pid"], signal.SIGTERM)
        extractor.terminate()
        extractor.wait(timeout=5)


def test_background_rejects_stale_job_before_spawning(setup, monkeypatch):
    cfg, _, kwargs = setup
    watch.atomic_json(kwargs["job_dir"] / "state.json", {"state": "waiting_dino", "watcher": {"pid": 123}})
    monkeypatch.setattr(watch, "load_config", lambda _: cfg)
    monkeypatch.setattr(sys, "argv", ["watch_dino", "--job-dir", str(kwargs["job_dir"]), "--background"])
    with pytest.raises(SystemExit) as error:
        watch.main()
    assert error.value.code == 2
    assert not (kwargs["job_dir"] / "daemon.log").exists()
