#!/usr/bin/env python3
"""Download DINO weights using the encoder version in the project configuration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path,
        default=Path(__file__).resolve().parents[1] / "pipeline/configs/dino.json",
        help="DINO JSON configuration (default: pipeline/configs/dino.json)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path.home() / "sigma0/models/dinov2-with-registers-large",
        help="Model output directory (default: ~/sigma0/models/dinov2-with-registers-large)",
    )
    args = parser.parse_args(argv)
    config = json.loads(args.config.expanduser().read_text(encoding="utf-8"))
    checkpoint, revision = config.get("checkpoint"), config.get("revision")
    if not all(isinstance(value, str) and value.strip() for value in (checkpoint, revision)):
        parser.error("DINO configuration requires non-empty checkpoint and revision values")

    from huggingface_hub import snapshot_download

    output = args.output.expanduser().resolve()
    snapshot_download(repo_id=checkpoint, revision=revision, local_dir=str(output))
    print(output)


if __name__ == "__main__":
    main()
