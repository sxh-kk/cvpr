"""Native30Hz visual re-extraction from the accepted SMPL-H archive."""

import argparse
import json
import os
from pathlib import Path
import signal
import sys

from .io import read_json


def load_config(path):
    path = Path(path).resolve(strict=True)
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError("DINO configuration must be a JSON object")
    if config.get("revision") == "local" and isinstance(config.get("checkpoint"), str):
        config["checkpoint"] = str((path.parent / config["checkpoint"]).resolve())
    profile = config.get("image_profile", "256x480")
    if isinstance(profile, str) and profile not in {"256x480", "256x320"}:
        config["image_profile"] = str((path.parent / profile).resolve())
    return config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native30Hz DINO cache from accepted SMPL-H motion and original cameras")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract-smplh-features", help="Extract the shared-RGB CLS + 4-register cache")
    extract.add_argument("--dataset", required=True, help="Accepted native30Hz SMPL-H parameter archive")
    extract.add_argument("--output", required=True, help="New dedicated visual-profile directory")
    extract.add_argument("--config", required=True, help="Pinned DINO JSON configuration")
    extract.add_argument("--episodes", type=int, nargs="+", help="Small diagnostic subset; use a separate output")
    extract.add_argument("--gpus", type=int, nargs="+", help="GPU IDs visible to this process")
    extract.add_argument("--workers-per-gpu", type=int, default=1,
                         help="DINO processes per GPU; benchmark before increasing (default: 1)")
    extract.add_argument("--dry-run", action="store_true", help="Verify provenance and plan without loading DINO or writing files")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        cpu_threads = config.get("cpu_threads", 2)
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("DINO cpu_threads must be a positive integer")
        # Apply defaults before NumPy/Torch imports and multiprocessing spawn.
        # GPU workers already parallelize clips and PIL/video preparation.
        for name, value in (("OPENBLAS_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"),
                            ("OMP_NUM_THREADS", str(cpu_threads)), ("NUMEXPR_NUM_THREADS", str(cpu_threads))):
            os.environ.setdefault(name, value)
        from .smplh_features import run_extraction
        def stop(_signum, _frame):
            raise InterruptedError("DINO extraction stopped by signal")
        previous = signal.signal(signal.SIGTERM, stop)
        try:
            result = run_extraction(args.dataset, args.output, config,
                                    episode_indices=args.episodes, gpu_ids=args.gpus,
                                    workers_per_gpu=args.workers_per_gpu, dry_run=args.dry_run)
        finally:
            signal.signal(signal.SIGTERM, previous)
        keys = ("state", "scope", "source_episode_count", "completed_episodes", "total_frames", "valid_frames",
                "fps", "frame_phase", "image_process", "output_root", "estimated_feature_bytes")
        print(json.dumps({key: result[key] for key in keys if key in result}, indent=2, ensure_ascii=False))
        return 0
    except (ValueError, FileNotFoundError, OSError, RuntimeError, InterruptedError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
