#!/usr/bin/env python3
"""Re-decode and independently validate every live v5 numeric column."""

import argparse
import json
from egomotion_pipeline.revalidation import revalidate_unified_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=3000)
    parser.add_argument("--episodes", nargs="+", type=int, help="Pilot only: does not certify the whole dataset")
    args = parser.parse_args(argv)
    report = revalidate_unified_dataset(args.dataset, args.job_dir, workers=args.workers,
                                       batch_size=args.batch_size, episode_indices=args.episodes)
    print(json.dumps({key: report.get(key) for key in ("stage", "passed", "full_dataset_validation",
          "all_live_episodes_validated", "completed_episodes", "completed_frames", "failed_episodes",
          "validated_physical_bytes", "decoded_numeric_bytes", "elapsed_seconds", "errors")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
