import argparse
import json
import os
from pathlib import Path

import numpy as np


def collect_action_stats(data_root: Path, output_path: Path) -> None:
    stats_path = data_root / "meta" / "episodes_stats.jsonl"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing {stats_path}.")

    min_val = None
    max_val = None
    with stats_path.open("r") as f:
        for line in f:
            row = json.loads(line)
            action_stats = row.get("stats", {}).get("action", {})
            if "min" not in action_stats or "max" not in action_stats:
                continue
            ep_min = np.asarray(action_stats["min"], dtype=np.float32)
            ep_max = np.asarray(action_stats["max"], dtype=np.float32)
            min_val = ep_min if min_val is None else np.minimum(min_val, ep_min)
            max_val = ep_max if max_val is None else np.maximum(max_val, ep_max)

    if min_val is None or max_val is None:
        raise RuntimeError("No action stats found in episodes_stats.jsonl.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump({"lerobot": {"min": min_val.tolist(), "max": max_val.tolist()}}, f, indent=2)

    print(f"Wrote action stats to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default=os.environ.get("LEROBOT_DATA_ROOT", ""))
    parser.add_argument("--output_path", default=os.environ.get("LEROBOT_STAT_PATH", ""))
    args = parser.parse_args()

    data_root = Path(args.data_root).expanduser()
    if not data_root.exists():
        raise FileNotFoundError("LeRobot data root not found.")

    output_path = Path(args.output_path) if args.output_path else data_root / "meta" / "action_stats.json"
    collect_action_stats(data_root, output_path)


if __name__ == "__main__":
    main()
