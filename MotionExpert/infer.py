"""Run E7 inference from a tensor input or a preprocessed dataset window."""

import argparse
from pathlib import Path

import torch

from .motion_expert import MotionExpert


def _to_cpu(value):
    if isinstance(value, torch.Tensor):
        return value.cpu()
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_cpu(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help=".pt mapping of batched conditioning tensors")
    source.add_argument("--config", type=Path, help="Use a dataset window from this training config")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir")
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sequence")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--text", nargs="+")
    parser.add_argument("--text-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--task", choices=("recon", "fore", "gen"), default="recon")
    parser.add_argument("--forecast-prefix", type=int, default=20)
    parser.add_argument("--cond-scale", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--unnormalized-traj", action="store_true")
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--features-only", action="store_true", help="Extract t=1 hidden/global with one model call")
    args = parser.parse_args()
    if args.features_only and args.all_steps:
        parser.error("--features-only cannot be combined with --all-steps")
    torch.set_num_threads(4)
    metadata = {"task": args.task, "seed": args.seed, "joint_indices": [0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21]}
    stats = args.stats
    if args.config:
        from .config import load_config
        from .check_data import validate_paths
        from .dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
        from .utils.torch_utils import careful_collate_fn
        cfg = load_config(args.config, data_dir=args.data_dir)
        if cfg.MODEL.GLOBAL_TOKEN_COUNT and not args.weights:
            parser.error("Global-token inference requires --weights from the global training run")
        validate_paths(cfg, (args.split,))
        dm = EE4D_Motion_DataModule(cfg)
        dm.setup("train" if args.split == "train" else "validate")
        dataset = getattr(dm, f"{args.split}_dataset")
        sample = (dataset.get_from_seq_and_st(args.sequence, args.start, -1)
                  if args.sequence else dataset[args.index])
        y = careful_collate_fn([sample])["y"]
        metadata.update({key: sample["misc"][key] for key in ("seq_name", "source", "start_idx", "end_idx")})
        stats = stats or dict(dataset.stats, normalization_clip=dataset.normalization_clip)
        if args.unnormalized_traj:
            parser.error("Dataset windows are already normalized")
    else:
        payload = torch.load(args.input, weights_only=True, map_location="cpu")
        y = payload.get("y", payload)
    if args.text:
        y = {key: value for key, value in y.items() if not key.startswith("text_") and key != "text"}
    expert = MotionExpert(args.weights, device=args.device, dtype=args.dtype, stats=stats,
                          text_model=args.text_model, local_files_only=args.local_files_only)
    if args.config and cfg.MODEL.GLOBAL_TOKEN_COUNT != expert.cfg.MODEL.GLOBAL_TOKEN_COUNT:
        parser.error("The dataset configuration and inference weights have different global-token architectures")
    text = args.text[0] if args.text and len(args.text) == 1 else args.text
    kwargs = dict(text=text, seed=args.seed, task=args.task, forecast_prefix=args.forecast_prefix,
                  cond_scale=args.cond_scale, normalized=not args.unnormalized_traj)
    output = (expert.extract_features(y, **kwargs) if args.features_only else
              expert.sample(y, **kwargs, denormalize=True, return_all_pred_xstart=args.all_steps))
    output = _to_cpu(output)
    output["metadata"].update(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    shapes = {key: tuple(value.shape) for key, value in output.items() if isinstance(value, torch.Tensor)}
    print(f"Saved {args.output}: {shapes}")


if __name__ == "__main__":
    main()
