"""Generate versioned motion from tensors or a MixEgoMotion archive window."""

import argparse
import json
from pathlib import Path

import torch

from .config import K12_JOINTS, architecture_metadata
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
    source.add_argument("--config", type=Path, help="Read a dataset window using a training config")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--stats", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", help="MixEgoMotion archive root")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sequence")
    parser.add_argument("--start", type=int, default=0, help="Frame index aligned to the configured native30Hz sampling step")
    parser.add_argument("--task", choices=("recon", "fore", "gen"), default=None,
                        help="Default: checkpoint/config training task (fore for the 5Hz profile)")
    parser.add_argument("--forecast-prefix", type=int, default=None,
                        help="Observed motion-grid frames (5Hz profile: 6, including current)")
    parser.add_argument("--frames", type=int, help="Output frames; allows a prefix-only fore/gen input")
    parser.add_argument("--cond-scale", type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default=None)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--unnormalized-traj", action="store_true")
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--features-only", action="store_true", help="Extract t=1 hidden/global with one model call")
    parser.add_argument("--decode", action="store_true", help="Also decode physical body/hand/floor trajectories")
    args = parser.parse_args()
    if args.features_only and (args.all_steps or args.decode):
        parser.error("--features-only cannot be combined with --all-steps or --decode")
    torch.set_num_threads(4)
    metadata = {"task": args.task, "seed": args.seed, "joint_indices": list(K12_JOINTS)}
    stats = args.stats
    decode_kwargs = {}
    if args.config:
        from .config import load_config
        from .dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
        from .utils.torch_utils import careful_collate_fn
        cfg = load_config(args.config, data_dir=args.data_dir)
        args.task = args.task or getattr(cfg.DATA, "TRAIN_TASK", None) or "recon"
        if args.split_manifest:
            cfg.DATA.SPLIT_MANIFEST = str(args.split_manifest.resolve())
        if args.forecast_prefix is not None and args.forecast_prefix != cfg.DATA.FORECAST_PREFIX:
            parser.error("Archive inference must use the stats/config forecast prefix; use --input for a different observed duration")
        dm = EE4D_Motion_DataModule(cfg)
        dm.setup("fit" if args.split == "train" else "test" if args.split == "test" else "validate")
        dataset = getattr(dm, f"{args.split}_dataset")
        sample = (dataset.get_from_seq_and_st(args.sequence, args.start, task=args.task)
                  if args.sequence else dataset.get_sample(args.index, task=args.task))
        y = careful_collate_fn([sample])["y"]
        requested_frames = cfg.DATA.WINDOW if args.frames is None else args.frames
        if requested_frames < 1:
            parser.error("--frames must be positive")
        if args.task == "recon" and requested_frames > cfg.DATA.WINDOW:
            parser.error("Reconstruction needs camera/image conditions for every output frame")
        observed = (cfg.DATA.FORECAST_PREFIX if args.task == "fore"
                    else 1 if args.task == "gen" else requested_frames)
        if args.task == "fore" and requested_frames <= observed:
            parser.error("Forecasting needs --frames greater than the observed prefix")
        # Keep trained statistics/window policy, but send only observed conditions.
        y = {key: value[:, :observed] if isinstance(value, torch.Tensor)
             and value.ndim >= 2 and value.shape[1] == cfg.DATA.WINDOW else value
             for key, value in y.items()}
        args.frames = requested_frames
        misc = sample["misc"]
        metadata.update({key: misc[key] for key in ("seq_name", "source", "start_idx", "end_idx") if key in misc})
        if "metadata_json" in misc:
            metadata["window"] = json.loads(misc["metadata_json"])
        stats = stats or dict(dataset.normalizer.stats, normalization_clip=dataset.normalizer.clip)
        if args.unnormalized_traj:
            parser.error("Dataset windows are already normalized")
        if args.decode:
            decode_kwargs = {"window_to_world": misc["world_from_window"],
                             "floor_height": misc["floor_height"]}
            # Reuse the last observed floor frame for future-only forecasting decode.
            if args.task == "fore":
                prefix = args.forecast_prefix or cfg.DATA.FORECAST_PREFIX
                anchor_valid = torch.as_tensor(misc["trajectory_valid"])[prefix - 1, :9]
                if not bool(anchor_valid.bool().all()):
                    parser.error("Forecast world decode requires a valid last-observed camera-floor reference")
                decode_kwargs.update(start_frame=prefix,
                                     initial_floor_transform=misc["floor_transform"][prefix - 1])
    else:
        if args.data_dir or args.split_manifest:
            parser.error("--data-dir and --split-manifest require --config")
        payload = torch.load(args.input, weights_only=True, map_location="cpu")
        y = payload.get("y", payload)
        if args.decode:
            decode_kwargs = payload.get("decode_kwargs", {})
    expert = MotionExpert(args.weights, device=args.device, dtype=args.dtype, stats=stats)
    args.task = args.task or getattr(expert.cfg.DATA, "TRAIN_TASK", None) or "recon"
    metadata["task"] = args.task
    metadata["joint_indices"] = list(expert.cfg.SPARSE_JOINTS.INDICES)
    if args.task == "fore" and expert.cfg.DATA.FPS == 5 and args.frames is None:
        args.frames = expert.cfg.DATA.WINDOW
    if args.config and architecture_metadata(cfg) != architecture_metadata(expert.cfg):
        parser.error("Dataset config and weights have different representation or architecture protocols")
    prefix = (cfg.DATA.FORECAST_PREFIX if args.config and args.forecast_prefix is None
              else args.forecast_prefix)
    kwargs = dict(seed=args.seed, task=args.task, forecast_prefix=prefix,
                  output_frames=args.frames, cond_scale=args.cond_scale,
                  normalized=not args.unnormalized_traj)
    output = (expert.extract_features(y, **kwargs) if args.features_only else
              expert.sample(y, **kwargs, denormalize=True, decode=args.decode,
                            decode_kwargs=decode_kwargs, return_all_pred_xstart=args.all_steps))
    output = _to_cpu(output)
    output["metadata"].update(metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    shapes = {key: tuple(value.shape) for key, value in output.items() if isinstance(value, torch.Tensor)}
    print(f"Saved {args.output}: {shapes}")


if __name__ == "__main__":
    main()
