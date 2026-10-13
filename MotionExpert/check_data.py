"""Validate preprocessed EE4D/NymeriaPlus packs, DINO and CLIP sidecars."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .config import load_config
from .dataset.ee4d_motion_dataset import EE4D_Motion_DataModule


def validate_paths(cfg, splits=("train", "val")):
    keys = ["EE4D_PACK", "EE4D_DINO_PACK", "STATS_PATH"]
    if cfg.DATA.DATASET_NAME == "ee4d_nymeria":
        keys += ["NYMERIA_PACK", "NYMERIA_DINO_PACK"]
    if cfg.DATA.COND_TEXT:
        keys.append("TEXT_FEATURES_PATH")
    if getattr(cfg.DATA, "SEQUENCE_MANIFEST", None):
        keys.append("SEQUENCE_MANIFEST")
    missing = []
    for split in splits:
        for key in keys:
            value = getattr(cfg.DATA, key, None)
            if not value or not Path(str(value).format(split=split)).is_file():
                missing.append(f"DATA.{key} ({split}): {value}")
    if missing:
        raise FileNotFoundError("Missing training inputs:\n" + "\n".join(missing))


def inspect_dataset(dataset, samples_per_source=2):
    counts, selected, text_valid = Counter(), Counter(), Counter()
    for index, (seq_idx, start) in enumerate(dataset.idx_to_sidx_fidx):
        name = dataset.seq_names[int(seq_idx)]
        source = dataset.motion_data[name].get("source", "ee4d")
        counts[source] += 1
        if dataset.text_features is not None:
            row = dataset.text_features.index.get(name, {}).get(str(int(start)))
            if row is None or not 0 <= int(row) < len(dataset.text_features.valid):
                raise ValueError(f"Missing/invalid CLIP index for {name}:{start}")
            text_valid[source] += int(dataset.text_features.valid[int(row)])
        if selected[source] >= samples_per_source:
            continue
        sample = dataset[index]
        for key, shape in (("traj", (dataset.window, 18)), ("img_embs", (dataset.window, 1024)),
                           ("text_tokens", (16, 768)), ("text_pooled", (768,))):
            value = sample["y"][key]
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid {key} in {name}:{start}, shape={tuple(value.shape)}")
        motion = sample["misc"]["motion"]
        if motion.shape != (dataset.window, 153) or not torch.isfinite(motion).all():
            raise ValueError(f"Invalid K12 target in {name}:{start}")
        if not sample["y"]["valid_img_embs"].any():
            raise ValueError(f"No valid DINO frames in {name}:{start}")
        selected[source] += 1
    return {"windows": dict(counts), "text_valid_windows": dict(text_valid), "checked_samples": dict(selected)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=("train", "val", "both"), default="both")
    parser.add_argument("--samples-per-source", type=int, default=2)
    args = parser.parse_args()
    if args.samples_per_source < 1:
        parser.error("--samples-per-source must be positive")
    torch.set_num_threads(4)
    cfg = load_config(args.config, data_dir=args.data_dir)
    splits = ("train", "val") if args.split == "both" else (args.split,)
    validate_paths(cfg, splits)
    report = {}
    for split in splits:
        dm = EE4D_Motion_DataModule(cfg)
        dm.setup("train" if split == "train" else "validate")
        report[split] = inspect_dataset(getattr(dm, f"{split}_dataset"), args.samples_per_source)
        del dm
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
