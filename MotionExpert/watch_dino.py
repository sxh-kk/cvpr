"""One-shot Linux supervisor: completed native30Hz DINO -> eight-GPU training."""
import argparse
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time

from .config import as_dict, load_config, validate_architecture

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "MotionExpert/configs/smplh_k12_psi0_pretrain.yaml"
CONTRACT = "smplh-dino-shared-rgb-cache-v1"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def process_identity(pid):
    """Do not mistake a reused PID or an unreaped zombie for our process."""
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": pid, "start_ticks": fields[19],
                "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_alive(identity):
    return identity is not None and process_identity(identity["pid"]) == identity


def command_line(pid):
    try:
        return (Path("/proc") / str(pid) / "cmdline").read_bytes().decode(errors="replace").strip("\0").split("\0")
    except (FileNotFoundError, ProcessLookupError):
        return []


def active_trainers():
    result = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        argv = command_line(int(path.name))
        module = any(a == "-m" and b == "MotionExpert.train" for a, b in zip(argv, argv[1:]))
        script = any(a.endswith("/MotionExpert/train.py") for a in argv)
        if (module or script) and process_identity(int(path.name)):
            result.append(int(path.name))
    return result


@contextmanager
def locked(path, *, shared=False, allow_busy=False):
    """Closing our fd preserves the lock if the training child inherited it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError:
            if not allow_busy:
                raise
            yield None
            return
        yield stream


def complete_manifest(manifest, cfg, binding):
    """Cheap publication checks; the training reader validates episode receipts."""
    if manifest.get("state") == "failed":
        raise RuntimeError(f"DINO extraction failed: {manifest.get('error', 'unknown error')}")
    expected = {"contract": CONTRACT, "scope": "all", "fps": 30, "native_fps": 30,
                "sampling_step": 1, "frame_phase": 0, "shape": [5, 1024], "storage_dtype": "float32"}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"DINO manifest {key} must be {value!r}")
    for key, value in (("source_root", cfg.DATA.ARCHIVE_ROOT), ("output_root", cfg.DATA.VISION_PROFILE_ROOT)):
        if not manifest.get(key) or Path(manifest[key]).resolve() != Path(value).resolve():
            raise ValueError(f"DINO manifest {key} differs from the training configuration")
    if not binding or manifest.get("binding") != binding:
        raise ValueError("DINO extraction binding changed while the watcher was waiting")
    if manifest.get("state") == "extracting":
        return False
    if manifest.get("state") != "complete":
        raise ValueError(f"Unexpected DINO state: {manifest.get('state')!r}")
    for done, total in (("completed_episodes", "source_episode_count"), ("completed_frames", "total_frames")):
        if (type(manifest.get(total)) is not int or manifest[total] <= 0
                or type(manifest.get(done)) is not int or manifest[done] != manifest[total]):
            raise ValueError(f"DINO cache claims completion but {done}/{total} disagree")
    if (manifest.get("validity_equal_source") is not True
            or manifest.get("episode_indices") != list(range(manifest["source_episode_count"]))):
        raise ValueError("DINO cache must preserve source masks and cover every episode")
    return True


def extractor_identity(pointer, cfg):
    if not Path(pointer).is_file():
        return None
    state = read_json(pointer)
    if Path(state.get("output", "")).resolve() != Path(cfg.DATA.VISION_PROFILE_ROOT).resolve():
        raise ValueError("Extraction state points to a different DINO cache")
    pid = state.get("pid")
    if type(pid) is not int or pid <= 0:
        raise ValueError("Extraction state requires a positive process PID")
    identity = process_identity(pid)
    if identity:
        argv = command_line(pid)
        if "extract-smplh-features" not in argv or str(Path(cfg.DATA.VISION_PROFILE_ROOT).resolve()) not in argv:
            raise ValueError("Extraction PID no longer identifies the expected DINO command")
    return identity


def require_new_run(output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to start from scratch over an existing run: {output}")
    running = active_trainers()
    if running:
        raise RuntimeError(f"MotionExpert training is already running, PIDs: {running}")


def preflight(cfg):
    # Do not build training windows twice or repeat physical numeric acceptance.
    # train.py validates all receipts/splits/statistics before allocating the model.
    import torch
    from .readiness import validate_paths
    from .visual_contract import training_visual_contract
    validate_paths(cfg)
    training_visual_contract(cfg)
    if torch.cuda.device_count() != 8:
        raise RuntimeError(f"Eight visible GPUs required, found {torch.cuda.device_count()}")


def run_watch(cfg, *, job_dir, extraction_state, gpus, poll_seconds, stop,
              check=preflight, launch=subprocess.Popen):
    import yaml
    job_dir = Path(job_dir).resolve()
    root = Path(cfg.DATA.ARCHIVE_ROOT).parent
    output = Path(cfg.TRAIN.EXP_PATH)
    vision = Path(cfg.DATA.VISION_PROFILE_ROOT)
    snapshot = job_dir / "config.yaml"
    log = job_dir / "training.log"
    # A global lock serializes watchers; the trainer inherits it so stopping the
    # watcher after handoff cannot allow a second watcher to launch another run.
    with ExitStack() as locks:
        # Share the launch reservation with checkpoint-boundary handoffs. The
        # reservation covers lock acquisition only; the inherited run lock
        # remains held throughout this watcher and its training child.
        with locked(root / "training/.launcher-reservation.lock"):
            run_lock = locks.enter_context(locked(root / "training/.dino-to-motion.lock"))
        require_new_run(output)
        if any((job_dir / name).exists() for name in ("state.json", "config.yaml", "training.log")):
            raise FileExistsError("Use a new job directory; existing watcher state is immutable")
        snapshot.write_text(yaml.safe_dump(as_dict(cfg), sort_keys=False), encoding="utf-8")
        command = [sys.executable, "-u", "-m", "MotionExpert.train", "--config", str(snapshot), "--from-scratch"]
        env = os.environ.copy()
        env.pop("MOTIONEXPERT_DATA_DIR", None)  # All dataset paths are frozen in the snapshot.
        env["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        env["PYTHONUNBUFFERED"] = "1"
        state = {"contract": "dino-to-motion-one-shot-v1", "watcher": process_identity(os.getpid()),
                 "job_dir": str(job_dir), "daemon_log": str(job_dir / "daemon.log"),
                 "log": str(log), "output": str(output), "config": str(snapshot),
                 "source_config": getattr(cfg, "SOURCE_CONFIG", None), "command": command, "gpus": gpus,
                 "started_at": now(), "training_running": False, "automatic_restart_enabled": False,
                 "pid": None, "checkpoint": None}
        child = None

        def publish(phase, **fields):
            changed = state.get("state") != phase
            state.update(state=phase, updated_at=now(), **fields)
            state["automatic_start_enabled"] = phase in {"waiting_dino", "waiting_release", "preflight"}
            for path in (job_dir / "state.json", root / ".pipeline/current_dino_to_motion.json",
                         root / "training/current_run.json"):
                atomic_json(path, state)
            if changed:
                print(f"[{now()}] {phase}: {fields.get('reason', '')}", flush=True)

        try:
            manifest = read_json(vision / "manifest.json")
            binding = manifest.get("binding")
            extractor = extractor_identity(extraction_state, cfg)
            state.update(extractor=extractor, extraction_binding=binding)
            while not stop.is_set():
                manifest = read_json(vision / "manifest.json")
                complete = complete_manifest(manifest, cfg, binding)
                alive = process_alive(extractor)
                if not complete and not alive:
                    # Completion can be published between the first read and exit.
                    manifest = read_json(vision / "manifest.json")
                    complete = complete_manifest(manifest, cfg, binding)
                    if not complete:
                        raise RuntimeError("DINO process exited before publishing a complete cache")
                counters = {key: manifest.get(key) for key in
                            ("completed_episodes", "source_episode_count", "completed_frames", "total_frames")}
                publish("waiting_release" if complete else "waiting_dino", **counters,
                        reason="Waiting for the DINO process to exit" if complete else "Waiting for full DINO extraction")
                if complete and not alive:
                    # Also reject a new extractor that acquired this cache lock.
                    with locked(vision / ".extraction.lock", shared=True, allow_busy=True) as cache_lock:
                        if cache_lock is None:
                            publish("waiting_release", reason="DINO cache is still locked by extraction")
                        else:
                            if not complete_manifest(read_json(vision / "manifest.json"), cfg, binding):
                                raise RuntimeError("DINO completion was withdrawn before handoff")
                            publish("preflight", reason="Checking shared visual profile, paths and eight GPUs")
                            check(cfg)
                            require_new_run(output)
                            if stop.is_set():
                                break
                            output.mkdir(parents=True, exist_ok=True)
                            publish("starting_training", reason="Launching eight-GPU training from scratch")
                            # Permanent, exclusive marker: a crash during Popen must
                            # never cause an unattended second from-scratch run.
                            with (output / "auto_train_launch.json").open("x", encoding="utf-8") as marker:
                                json.dump(state, marker, ensure_ascii=False, indent=2)
                                marker.flush()
                                os.fsync(marker.fileno())
                            with log.open("x", encoding="utf-8") as stream:
                                child = launch(command, cwd=REPO_ROOT, env=env, stdin=subprocess.DEVNULL,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                               pass_fds=(run_lock.fileno(), cache_lock.fileno()))
                            publish("training", pid=child.pid, trainer=process_identity(child.pid),
                                    training_running=True, training_started_at=now(), reason="Training launched")
                            while child.poll() is None and not stop.wait(poll_seconds):
                                publish("training", training_running=True)
                            if child.poll() is None:
                                publish("watcher_stopped", training_running=True,
                                        reason="Watcher stopped; the training process continues independently")
                                return 0
                            code = child.returncode
                            publish("completed" if code == 0 else "failed", training_running=False,
                                    exit_code=code, finished_at=now(),
                                    reason="Training finished" if code == 0 else f"Training failed; inspect {log}")
                            return 0 if code == 0 else 1
                print(f"[{now()}] DINO episodes={counters['completed_episodes']}/{counters['source_episode_count']} "
                      f"frames={counters['completed_frames']}/{counters['total_frames']}", flush=True)
                stop.wait(poll_seconds)
            publish("stopped", reason="Watcher stopped before launch; DINO remains untouched")
            return 0
        except Exception as error:
            publish("failed", error=f"{type(error).__name__}: {error}",
                    training_running=child is not None and child.poll() is None, reason=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--extraction-state", type=Path, help="Defaults to the archive parent's .pipeline/current_dino_extraction.json")
    parser.add_argument("--job-dir", type=Path, help="New external directory for state, frozen config and logs")
    parser.add_argument("--gpus", nargs=8, type=int, default=list(range(8)))
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--background", action="store_true", help="Detach and return after the watcher has armed")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds <= 0:
        parser.error("--poll-seconds must be finite and positive")
    if len(set(args.gpus)) != 8 or any(g < 0 for g in args.gpus):
        parser.error("--gpus requires eight distinct nonnegative GPU IDs")
    cfg = load_config(args.config.resolve())
    validate_architecture(cfg)
    if cfg.TRAIN.NUM_GPUS != 8:
        parser.error("The training configuration must request eight GPUs")
    cfg.MODEL.CKPT_PATH = cfg.MODEL.INIT_CKPT_PATH = None
    cfg.SOURCE_CONFIG = str(args.config.resolve())
    root = Path(cfg.DATA.ARCHIVE_ROOT).parent
    job_dir = (args.job_dir or root / ".pipeline/jobs" / (
        "dino-to-motion-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))).resolve()
    if job_dir.is_relative_to(REPO_ROOT):
        parser.error("--job-dir must be outside the Git repository")
    job_dir.mkdir(parents=True, exist_ok=True)
    if args.background:
        if any(job_dir.iterdir()):
            parser.error("--job-dir must be empty for a new background watcher")
        # Normalize paths before changing cwd in the detached child.
        argv = ["--config", str(args.config.resolve()), "--job-dir", str(job_dir),
                "--poll-seconds", str(args.poll_seconds), "--gpus", *map(str, args.gpus)]
        if args.extraction_state:
            argv.extend(["--extraction-state", str(args.extraction_state.resolve())])
        with (job_dir / "daemon.log").open("x", encoding="utf-8") as stream:
            child = subprocess.Popen([sys.executable, "-u", "-m", "MotionExpert.watch_dino", *argv],
                                     cwd=REPO_ROOT, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if (job_dir / "state.json").is_file():
                state = read_json(job_dir / "state.json")
                if state.get("watcher", {}).get("pid") != child.pid:
                    print("Watcher state does not belong to the launched process", file=sys.stderr)
                    return 1
                print(json.dumps({"watcher_pid": child.pid, "state": state["state"],
                                  "job_dir": str(job_dir), "state_file": str(job_dir / "state.json"),
                                  "daemon_log": str(job_dir / "daemon.log")}, ensure_ascii=False))
                return 1 if state["state"] == "failed" else 0
            if child.poll() is not None:
                print((job_dir / "daemon.log").read_text(), file=sys.stderr)
                return 1
            time.sleep(0.1)
        print(f"Watcher PID {child.pid} is starting; inspect {job_dir / 'daemon.log'}", file=sys.stderr)
        return 0
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    try:
        return run_watch(cfg, job_dir=job_dir, extraction_state=args.extraction_state or
                         root / ".pipeline/current_dino_extraction.json", gpus=args.gpus,
                         poll_seconds=args.poll_seconds, stop=stop)
    except Exception as error:
        print(f"[{now()}] Watcher failed: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
