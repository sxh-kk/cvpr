"""Strict local 20k GPU loading and optional inference on a captured real history.

This is a model probe, never an episode-success or closed-loop evaluation.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

EXPECTED_SHA = "a000c559a0de0a3be477933aa31ac5c681a5a79fe380c344adb89dde399d535a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=Path, help="NPZ: rgb, sim_time_s, state43; genuine single-episode observations")
    parser.add_argument("--instruction", default="pick up the object")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    args.resources = args.resources.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "joint-gpu.json"
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "resources": str(args.resources), "batch": 1, "gpu_count": 1,
              "boundary": "strict GPU model probe; not a G1 control/episode-success evaluation"}

    def save():
        report_path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        from export_bundle import verify_bundle
        verified = verify_bundle(args.resources)
        manifest = json.loads((args.resources / "manifest.json").read_text())
        assert manifest["step"] == 20000 and manifest["joint_sha256"] == EXPECTED_SHA
        report.update(bundle_verification=verified, checkpoint_sha256=EXPECTED_SHA)
        print("Bundle verified; loading original config and full checkpoint", flush=True)
        import numpy as np
        import torch
        from psi.models.psi0 import Psi0Model
        from psi.utils import parse_args_to_tyro_config, apply_legacy_model_config_defaults

        run = args.resources / "joint20k"
        schema = parse_args_to_tyro_config(run / "argv.txt")
        config = schema.model_validate(apply_legacy_model_config_defaults(json.loads((run / "run_config.json").read_text())))
        assert config.model.action_architecture == "dual_source_v1"
        assert (config.model.action_dim, config.model.odim, config.model.action_chunk_size) == (80, 45, 30)
        for path in (config.model.model_name_or_path, config.model.motion_checkpoint):
            assert Path(path).resolve().is_relative_to(args.resources), path
        torch.manual_seed(config.seed or 42)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        model = Psi0Model.from_pretrained(run, 20000, config, "cuda:0").to("cuda:0").eval()
        torch.cuda.synchronize()
        assert all(not m.training for m in model.modules())
        report.update(strict_gpu_load="passed", load_seconds=time.perf_counter() - start,
                      parameters=sum(p.numel() for p in model.parameters()),
                      cuda_runtime=torch.version.cuda, torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save()
        if not args.history:
            report.update(status="strict_gpu_load_passed", inference="not_run_no_real_history")
            save()
            print(json.dumps(report, indent=2), flush=True)
            return

        print("Loading captured episode history and frozen local DINO", flush=True)
        from psi.deploy.dual_source_runtime import LocalDinoEncoder
        from PIL import Image
        local_encoder = LocalDinoEncoder(run / "dino_training_contract.json", device="cuda:0")
        encoder = local_encoder.encoder
        with np.load(args.history, allow_pickle=False) as data:
            rgb, timestamps, state43 = data["rgb"], data["sim_time_s"], data["state43"]
        assert rgb.ndim == 4 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
        assert len(rgb) == len(timestamps) == len(state43) and len(rgb) >= 6
        assert timestamps.ndim == 1 and np.isfinite(timestamps).all() and np.all(np.diff(timestamps) > 0)
        assert state43.shape == (len(rgb), 43) and np.isfinite(state43).all()
        end = float(timestamps[-1])
        targets = end - np.arange(5, -1, -1) / 5
        # Causal selection: never use an image from after the requested simulated time.
        indices = np.searchsorted(timestamps, targets + 1e-9, side="right") - 1
        assert (indices >= 0).all() and len(set(indices)) == 6
        assert np.max(targets - timestamps[indices]) <= 1 / 30 + .002
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        window = encoder.encode_window(list(rgb[indices]), stage="posttrain", training=False)
        embs = torch.from_numpy(window["img_embs"]).unsqueeze(0).to("cuda:0")
        assert embs.shape == (1, 6, 5, 1024) and torch.isfinite(embs).all()
        torch.cuda.synchronize()
        dino_seconds = time.perf_counter() - start
        state = np.pad(state43[-1:].astype(np.float32), ((0, 0), (0, 2)))
        field = config.data.transform.field
        if field.normalize_state:
            state = field.normalize_state_func(state)
        states = torch.as_tensor(state, device="cuda:0").unsqueeze(0)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            out = model.predict_action(
                observations=[[Image.fromarray(window["rgb"][-1])]], states=states,
                instructions=[args.instruction.lower()], num_inference_steps=args.steps, traj2ds=None,
                motion_img_embs=embs, return_vlm_features=True)
        torch.cuda.synchronize()
        inference_seconds = time.perf_counter() - start
        action = out.action.float().cpu().numpy()
        assert action.shape == (1, 30, 80) and np.isfinite(action).all()
        denormalized = field.denormalize(action[0])
        assert denormalized.shape == (30, 80) and np.isfinite(denormalized).all()
        motion, vlm = out.motion_features, out.vlm_features
        assert motion.tokens.shape == (1, 11, 768)
        assert torch.equal(motion.p_m, motion.tokens.float().mean(1).to(motion.tokens.dtype))
        assert vlm.tokens.shape[1] == 1 and vlm.attention_mask.shape == (1, vlm.tokens.shape[2])
        assert torch.equal(vlm.tokens[:, 0], vlm.last_hidden_state)
        valid = vlm.attention_mask.bool()
        mean = vlm.tokens.float().masked_fill(~valid[:, None, :, None], 0).sum(2) / valid.sum(-1)[:, None, None]
        assert torch.equal(vlm.pooled_tokens, mean.to(vlm.tokens.dtype))
        np.savez_compressed(args.output / "joint-action.npz", normalized=action, denormalized=denormalized)
        report.update(status="model_inference_passed", history_file=str(args.history.resolve()),
                      history_timestamps=timestamps[indices].tolist(), requested_timestamps=targets.tolist(),
                      dino_shape=list(embs.shape), action_shape=list(action.shape), state_shape=list(states.shape),
                      dino_optimization=encoder.metadata.get("optimization"),
                      motion_tokens=list(motion.tokens.shape), vlm_tokens=list(vlm.tokens.shape),
                      full_tokens_and_pooling="passed", action_denormalization="finite",
                      dino_seconds=dino_seconds, inference_seconds=inference_seconds,
                      inference_steps=args.steps, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      instruction=args.instruction, control_evaluation="not_run")
        encoder.close()
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        save()
        raise


if __name__ == "__main__":
    main()
