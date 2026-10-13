"""Export strict E7 model weights, using checkpoint EMA weights by default."""

import argparse
import json
from pathlib import Path

import torch

from .checkpoints import read_checkpoint, model_state, checkpoint_config
from .config import architecture_metadata, as_dict
from .dataset.normalization import MotionNormalizer
from .model import UniEgoMotion


def export(checkpoint_path, output_path, *, use_ema=True):
    torch.set_num_threads(4)
    checkpoint = read_checkpoint(checkpoint_path)
    cfg = checkpoint_config(checkpoint)
    with torch.device("meta"):
        model = UniEgoMotion(cfg)
    state = model_state(model, checkpoint, use_ema=use_ema)
    model.load_state_dict(state, strict=True, assign=True)
    state = {name: value.detach().float().contiguous() for name, value in state.items()}
    output = Path(output_path)
    if output.resolve() == Path(checkpoint_path).resolve():
        raise ValueError("Export destination must differ from the training checkpoint")
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = checkpoint.get("motionexpert_normalization")
    if stats is not None:
        stats = dict(stats, normalization_clip=checkpoint.get("motionexpert_normalization_clip", 8.0))
    elif Path(checkpoint_path).with_suffix(".stats.pt").is_file():
        stats = torch.load(Path(checkpoint_path).with_suffix(".stats.pt"), weights_only=True)
    normalizer = MotionNormalizer(stats)
    weights_kind = "ema" if use_ema and "ema_state_format" in checkpoint else checkpoint.get("weights", "model")
    architecture = architecture_metadata(cfg)
    torch.save({
        "format": "motionexpert_weights_v2",
        "state_dict": state,
        "hyper_parameters": as_dict(cfg),
        "motionexpert_architecture": architecture,
        "motionexpert_normalization": normalizer.stats,
        "motionexpert_normalization_clip": normalizer.clip,
        "weights": weights_kind,
    }, output)
    torch.save(dict(normalizer.stats, normalization_clip=normalizer.clip), output.with_suffix(".stats.pt"))
    metadata = {
        "source": str(Path(checkpoint_path).resolve()),
        "epoch": checkpoint.get("epoch"), "global_step": checkpoint.get("global_step"),
        "weights": weights_kind,
        **architecture,
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": "float32", "bytes": output.stat().st_size,
        "normalization": output.with_suffix(".stats.pt").name,
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()
    export(args.checkpoint, args.output, use_ema=not args.no_ema)


if __name__ == "__main__":
    main()
