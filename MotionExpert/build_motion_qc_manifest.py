"""Build a deterministic sequence allow-list from joint-statistics QC records.

The first robust stats pass records non-finite values and per-sequence extrema
without allowing them into sufficient statistics.  This utility turns that
report into an explicit manifest for the second pass and for training.  The
manifest is JSON on purpose: it is easy to inspect, diff, and keep beside the
motion pack without modifying the pack itself.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-terrain", action="store_true")
    parser.add_argument("--allow-nonfinite", action="store_true")
    parser.add_argument("--max-abs-motion", type=float, default=None)
    parser.add_argument("--max-abs-traj", type=float, default=None)
    args = parser.parse_args()

    stats = torch.load(args.stats, weights_only=False)
    sequence_qc = stats.get("sequence_qc", {})
    if not isinstance(sequence_qc, dict) or not sequence_qc:
        raise ValueError(f"stats file has no sequence_qc records: {args.stats}")

    accepted = []
    rejected = {}
    sequence_status = {}
    source_counts = {}
    for name in sorted(sequence_qc):
        value = sequence_qc[name]
        reasons = []
        if not args.allow_nonfinite and (
            int(value.get("nonfinite_motion_values", 0))
            or int(value.get("nonfinite_traj_frames", 0))
        ):
            reasons.append("nonfinite_input")
        if args.exclude_terrain and bool(value.get("discard_terrain", False)):
            reasons.append("terrain_interaction")
        if args.max_abs_motion is not None and float(value.get("max_abs_motion", 0.0)) > args.max_abs_motion:
            reasons.append("motion_outlier")
        if args.max_abs_traj is not None and float(value.get("max_abs_traj", 0.0)) > args.max_abs_traj:
            reasons.append("trajectory_outlier")

        status = {
            **value,
            "accepted": not reasons,
            "reasons": reasons,
        }
        sequence_status[name] = status
        if reasons:
            rejected[name] = status
        else:
            accepted.append(name)
        source = str(value.get("source", "unknown"))
        source_counts.setdefault(source, {"total": 0, "accepted": 0, "rejected": 0})
        source_counts[source]["total"] += 1
        source_counts[source]["accepted" if not reasons else "rejected"] += 1

    if not accepted:
        raise ValueError("QC thresholds rejected every sequence")
    report = {
        "schema_version": 1,
        "stats_file": str(args.stats.resolve()),
        "stats_schema_version": int(stats.get("stats_schema_version", -1)),
        "accepted_sequences": accepted,
        "rejected_sequences": rejected,
        "sequence_status": sequence_status,
        "source_counts": source_counts,
        "options": {
            "exclude_terrain": bool(args.exclude_terrain),
            "allow_nonfinite": bool(args.allow_nonfinite),
            "max_abs_motion": args.max_abs_motion,
            "max_abs_traj": args.max_abs_traj,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "source_counts": source_counts,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
