"""Run unified migration, grouped splits, train-only stats and reader checks.

The status/log directory is in the dataset. Training statistics stay in the
MotionExpert experiment directory, separate from the immutable archive.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline/src"))
sys.path.insert(0, str(REPO))

from egomotion_pipeline.storage import atomic_json, read_json
from egomotion_pipeline.unified_migration import upgrade_dense_v5


def _now():
    return datetime.now(timezone.utc).isoformat()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--migration-dir", required=True)
    parser.add_argument("--native-root")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--stats-workers", type=int, default=16)
    parser.add_argument("--experiment-dir", default=str(REPO / "MotionExpert/exp/dense_k12"))
    parser.add_argument("--config", default=str(REPO / "MotionExpert/configs/dense_k12_global.yaml"))
    args = parser.parse_args(argv)
    if min(args.workers, args.stats_workers) < 1:
        parser.error("worker counts must be positive")
    root, job, experiment = Path(args.dataset).resolve(), Path(args.job_dir).resolve(), Path(args.experiment_dir).resolve()
    if experiment == root or root in experiment.parents:
        parser.error("training artifacts must be outside the immutable archive")
    job.mkdir(parents=True, exist_ok=True)
    experiment.mkdir(parents=True, exist_ok=True)
    status_path = job / "status.json"
    status = {"stage": "starting", "started_at": _now(), "pid": os.getpid(),
              "dataset": str(root), "job_dir": str(job), "migration_dir": str(Path(args.migration_dir).resolve()),
              "experiment_dir": str(experiment), "workers": args.workers, "stats_workers": args.stats_workers}
    children = []

    def publish(stage=None, **values):
        if stage:
            status["stage"] = stage
        status.update(values, updated_at=_now())
        atomic_json(status_path, status)

    def run(command, name):
        print(json.dumps({"stage": status["stage"], "command": command, "log": name}), flush=True)
        with (job / name).open("a") as log:
            result = subprocess.run(command, cwd=REPO, env=child_env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Command failed ({result.returncode}); see {job / name}")

    child_env = dict(os.environ)
    child_env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
                     PYTHONUNBUFFERED="1")
    child_env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "pipeline/src"), child_env.get("PYTHONPATH", "")])
    try:
        publish("migrating")

        def progress(state):
            publish("migrating", migration_state=state.get("state"),
                    completed_episodes=state.get("completed_episodes", 0), total_episodes=state.get("total_episodes", 0),
                    completed_frames=state.get("completed_frames", 0), total_frames=state.get("total_frames", 0),
                    migration_elapsed_seconds=state.get("elapsed_seconds"), last_episode=state.get("last_episode"),
                    migration_error=state.get("last_error"))
            print(json.dumps({"stage": "migrating", **{key: state.get(key) for key in (
                "state", "completed_episodes", "total_episodes", "completed_frames", "elapsed_seconds", "last_episode", "last_error")}}), flush=True)

        result = upgrade_dense_v5(root, workers=args.workers, native_root=args.native_root,
                                   progress=progress, job_dir=args.migration_dir)
        atomic_json(job / "migration-result.json", result)
        info = read_json(root / "meta/info.json")
        publish("splitting", completed_episodes=info["total_episodes"], completed_frames=info["total_frames"],
                total_episodes=info["total_episodes"], total_frames=info["total_frames"])
        split_path, stats_path = experiment / "splits.json", experiment / "stats.pt"
        run([sys.executable, "-m", "MotionExpert.split_manifest", "--archive-root", str(root),
             "--output", str(split_path)], "splits.log")
        split = read_json(split_path)
        atomic_json(job / "split-summary.json", {key: value for key, value in split.items() if key != "splits"})
        publish("statistics", splits_path=str(split_path), stats_path=str(stats_path), completed_stats_shards=0)
        shards_dir = experiment / f"stats-shards-{job.name}"
        shards_dir.mkdir(parents=True, exist_ok=True)
        stem = "dense245_joint_masked_train_stats"
        handles = []
        try:
            for index in range(args.stats_workers):
                output = shards_dir / f"{stem}.shard-{index:03d}-of-{args.stats_workers:03d}.pt"
                command = [sys.executable, "-m", "MotionExpert.compute_joint_motion_stats", "--config", args.config,
                           "--archive-root", str(root), "--split-manifest", str(split_path), "--output", str(output),
                           "--num-shards", str(args.stats_workers), "--shard-index", str(index),
                           "--shard-mode", "contiguous"]
                handle = (job / f"stats-{index:03d}.log").open("a")
                handles.append(handle)
                children.append(subprocess.Popen(command, cwd=REPO, env=child_env, stdout=handle, stderr=subprocess.STDOUT))
            while True:
                codes = [child.poll() for child in children]
                failed = [index for index, code in enumerate(codes) if code not in (None, 0)]
                if failed:
                    raise RuntimeError(f"Train statistics shard(s) failed: {failed}; see stats-NNN.log")
                count = sum(code == 0 for code in codes)
                publish("statistics", completed_stats_shards=count,
                        stats_pids=[child.pid for child, code in zip(children, codes) if code is None])
                if count == len(children):
                    break
                time.sleep(5)
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
            for handle in handles:
                handle.close()
        run([sys.executable, "-m", "MotionExpert.merge_joint_motion_stats", "--input-dir", str(shards_dir),
             "--num-shards", str(args.stats_workers), "--output", str(stats_path)], "stats-merge.log")
        publish("checking_readers", stats_shards_dir=str(shards_dir), stats_sha256=hashlib.sha256(stats_path.read_bytes()).hexdigest())
        for split_name in ("train", "val", "test"):
            run([sys.executable, "-m", "MotionExpert.check_data", "--config", args.config,
                 "--split", split_name, "--samples-per-source", "3"], f"reader-{split_name}.log")
        publish("complete", completed_at=_now(), reader_checks=["train", "val", "test"], last_error=None)
        print(json.dumps(status, indent=2), flush=True)
        return 0
    except BaseException as exc:
        publish("failed", last_error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc()[-6000:])
        print(json.dumps(status, indent=2), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
