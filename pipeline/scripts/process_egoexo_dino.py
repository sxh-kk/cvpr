"""Extract EgoExo4D-HM ego-video DINO sidecars with the EgoSuite encoder."""

import argparse
from pathlib import Path
import signal

from egomotion_pipeline.storage import atomic_json, read_json
from egomotion_pipeline.video_dino import plan_videos, run_videos, validate_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("--dataset-root", required=True)
    plan.add_argument("--output-root", required=True)
    plan.add_argument("--plan", required=True)
    plan.add_argument("--workers", type=int, default=16)
    run = commands.add_parser("run")
    run.add_argument("--plan", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--job-dir", required=True)
    run.add_argument("--gpus", type=int, nargs="+", required=True)
    run.add_argument("--reference-encoder", help="Completed EgoSuite encoder metadata for exact compatibility check")
    validate = commands.add_parser("validate")
    validate.add_argument("--plan", required=True)
    validate.add_argument("--report", required=True)
    validate.add_argument("--workers", type=int, default=8)
    status = commands.add_parser("status")
    status.add_argument("--job-dir", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        data = plan_videos(args.dataset_root, args.output_root, args.workers)
        atomic_json(Path(args.plan), data)
        print(f"{data['video_count']} ego videos, {data['total_frames']} target frames", flush=True)
    elif args.command == "run":
        def stop(*_):
            raise InterruptedError("DINO job received a stop signal")
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        reference = read_json(args.reference_encoder) if args.reference_encoder else None
        report = run_videos(read_json(args.plan), read_json(args.config), args.job_dir, args.gpus, reference)
        return 0 if report["passed"] else 1
    elif args.command == "validate":
        report = validate_plan(read_json(args.plan), args.workers)
        atomic_json(Path(args.report), report)
        print(report, flush=True)
        return 0 if report["passed"] else 1
    else:
        state = read_json(Path(args.job_dir) / "status.json")
        print(f"{state['stage']}: {state['completed_videos']}/{state['total_videos']} videos, "
              f"{state['completed_frames']}/{state['total_frames']} frames, "
              f"{state.get('committed_frames_per_second', 0):.1f} frames/s, "
              f"ETA {state.get('estimated_remaining_seconds')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
