"""Inventory expected local resources; never downloads or substitutes a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    root = args.root.expanduser().resolve()
    expected = [
        ("joint20k/checkpoints/ckpt_20000/model.safetensors", "exact 20k joint model; strict load and identity pending"),
        ("joint20k/run_config.json", "remap construction paths locally; retain architecture/statistics contracts"),
        ("joint20k/argv.txt", "original launch provenance"),
        ("joint20k/dataset_statistics.json", "original bounds and state/action order; no test-set refit"),
        ("joint20k/qwen", "local tokenizer/config/processor; joint checkpoint holds VLM weights"),
        ("joint20k/motion", "portable construction checkpoint + metadata and normalization"),
        ("dinov2-with-registers-large", "same frozen DINO identity and processor as training"),
        ("sonic-body-token-v1.1", "matched controller/decoder identity; shape alone is insufficient"),
        ("simple", "fixed task, robot/USD/textures, evaluation initial conditions"),
    ]
    items = []
    for relative, contract in expected:
        path = root / relative
        present = path.is_file() or (path.is_dir() and any(path.iterdir()))
        item = {"path": str(path), "present": present, "contract": contract,
                "semantic_validation": "not_run"}
        if path.is_file():
            item["bytes"] = path.stat().st_size
            digest = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
            item["sha256"] = digest.hexdigest()
        items.append(item)
    report = {"root": str(root), "resources": items,
              "boundary": "inventory only; presence and hash never prove compatibility or successful loading"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
