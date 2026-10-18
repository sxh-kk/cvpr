#!/usr/bin/env python3
"""Bounded acceptance of REAL UnifoLM/Qwen/Motion/Psi0 components, never a run recipe.

Run after sourcing scripts/train/psi0/unifolm-local.env.sh. Each invocation has
one complete model resident. Outputs/checkpoints must be outside the repository.

  "$PSI_PYTHON" scripts/validation/validate_unifolm_full_model.py \
      --mode train --output-dir /external/acceptance/single
  "$PSI_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
      scripts/validation/validate_unifolm_full_model.py \
      --mode train --output-dir /external/acceptance/ddp
  # Use the checkpoint path printed by train; resume uses the SAME rank count.
  "$PSI_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
      scripts/validation/validate_unifolm_full_model.py --mode resume \
      --checkpoint /external/.../checkpoints/ckpt_4 --output-dir /external/acceptance/resume
  "$PSI_PYTHON" scripts/validation/validate_unifolm_full_model.py --mode inference \
      --checkpoint /external/.../checkpoints/ckpt_4 --output-dir /external/acceptance/inference

Train/resume use real PosttrainTrainer methods and Accelerator preparation,
with a finite subset of the real LeRobot wrapper and the production collator.
Inference additionally exercises Psi0Model.from_pretrained. No W&B, git tagging,
unbounded epoch loop, online DINO, synthetic inputs, or architectural downsizing.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback


VLM_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = VLM_ROOT.parent
for _directory in (VLM_ROOT / "src", VLM_ROOT, REPO_ROOT):
    sys.path.insert(0, str(_directory))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=("train", "resume", "inference"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--vlm", default=os.getenv("EGO_DEX_VLM_PATH"))
    parser.add_argument("--motion", default=os.getenv("MOTION_EXPERT_CHECKPOINT"))
    parser.add_argument("--data-root", default=os.getenv("UNIFOLM_DINO_ROOT"))
    parser.add_argument("--cache-root", default=os.getenv("UNIFOLM_DINO_CACHE"))
    parser.add_argument("--pack", default=os.getenv("UNIFOLM_DINO_TRAIN_ID", "unifolm_sonic_lerobot_train"))
    parser.add_argument("--val-pack", default=os.getenv("UNIFOLM_DINO_VAL_ID", "unifolm_sonic_lerobot_val"))
    parser.add_argument("--binding", default=os.getenv("UNIFOLM_INIT_WEIGHTS_RECORD"))
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1, help="Per-rank microbatch size")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--seed", type=int, default=292285)
    parser.add_argument("--comparison-atol", type=float, default=0.005)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir == REPO_ROOT or REPO_ROOT in args.output_dir.parents:
        parser.error("output-dir must be outside the repository")
    if not 4 <= args.updates <= 8 or not 1 <= args.batch_size <= 16 or args.accumulation != 2:
        parser.error("bounded gate requires 4..8 updates, batch-size 1..16, accumulation=2")
    if args.mode == "train" and not all((args.vlm, args.motion, args.data_root, args.cache_root)):
        parser.error("train requires explicit local --vlm/--motion/--data-root/--cache-root or their sourced env variables")
    if args.mode == "train" and not args.binding:
        parser.error("train requires the stage2 --binding record or UNIFOLM_INIT_WEIGHTS_RECORD")
    if args.mode != "train" and args.checkpoint is None:
        parser.error("resume/inference require --checkpoint from this script")
    if args.mode != "train" and args.binding is None:
        args.binding = str(args.checkpoint / "validation_binding.json")
    if not math.isfinite(args.comparison_atol) or not 0 < args.comparison_atol <= 0.01:
        parser.error("comparison-atol must be in (0, .01]")
    return args


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _tensor_hash(hasher, value):
    """Stream through CPU in bounded chunks, including BF16 without NumPy casts."""
    import torch

    value = value.detach()
    hasher.update(f"{value.dtype}:{tuple(value.shape)}".encode())
    if not value.numel():
        return
    if value.ndim == 0:
        parts = (value.reshape(1),)
    else:
        row_size = max(1, value[0].numel())
        parts = value.split(max(1, (1 << 20) // row_size), dim=0)
    for part in parts:
        raw = part.contiguous().reshape(-1).view(torch.uint8).cpu().numpy()
        hasher.update(memoryview(raw))


def tree_hash(value):
    import numpy as np
    import torch

    hasher = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            _tensor_hash(hasher, item)
        elif isinstance(item, np.ndarray):
            hasher.update(f"{item.dtype}:{tuple(item.shape)}".encode())
            hasher.update(memoryview(np.ascontiguousarray(item)))
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                hasher.update(str(key).encode() + b"\0")
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            hasher.update(str(len(item)).encode() + b"[")
            for entry in item:
                visit(entry)
        else:
            hasher.update(repr(item).encode() + b"\0")

    visit(value)
    return hasher.hexdigest()


def rng_hash():
    import numpy as np
    import torch

    return tree_hash({"python": random.getstate(), "numpy": np.random.get_state(),
                      "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()})


def checkpoint_rng_hash(checkpoint, process_index, local_device_index):
    """Compare restore against the serialized snapshot, before post-save logs.

    Accelerator stores all visible CUDA generators in each rank's RNG file;
    this process's active generator is selected using LOCAL_RANK, not RANK.
    """
    import torch

    states = torch.load(Path(checkpoint) / f"random_states_{process_index}.pkl",
                        map_location="cpu", weights_only=False)
    cuda = states["torch_cuda_manual_seed"]
    if not 0 <= local_device_index < len(cuda):
        raise ValueError("Saved RNG state does not include this process's local CUDA device")
    return tree_hash({"python": states["random_state"], "numpy": states["numpy_random_seed"],
                      "torch": states["torch_manual_seed"], "cuda": cuda[local_device_index]})


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def config(args):
    from psi.config.train.posttrain_sonic_psi0_config import DynamicLaunchConfig

    if args.mode != "train":
        cfg = DynamicLaunchConfig.model_validate_json((args.checkpoint / "validation_config.json").read_text())
        cfg.train.output_dir = str(args.output_dir / "runs")
        return cfg
    data_root, cache_root = Path(args.data_root).resolve(), Path(args.cache_root).resolve()
    vlm, motion = Path(args.vlm).resolve(), Path(args.motion).resolve()
    if not vlm.is_dir() or not motion.is_file():
        raise ValueError("VLM directory and Motion checkpoint must be real local artifacts")
    stats = data_root / args.pack / "meta/stats_psi0.json"
    return DynamicLaunchConfig.model_validate({
        "exp": "bounded_full_model_acceptance", "timestamp": "bounded", "seed": args.seed,
        "auto_tag_run": False, "log": {"report_to": None}, "wandb": {},
        "train": {"name": "posttrain", "output_dir": str(args.output_dir / "runs"),
                  "mixed_precision": "bf16", "data_parallel": "ddp", "num_workers": 0,
                  "train_batch_size": args.batch_size, "val_batch_size": args.batch_size,
                  "gradient_accumulation_steps": args.accumulation, "max_training_steps": args.updates,
                  "learning_rate": 1e-4, "warmup_steps": 0, "warmup_ratio": None,
                  "lr_scheduler_type": "constant", "max_grad_norm": 1.0, "optimizer_foreach": False,
                  "lr_scheduler_kwargs": {"betas": [0.95, 0.999], "weight_decay": 1e-6},
                  "max_checkpoints_to_keep": 1, "preserve_checkpoint_every": None},
        "data": {"root_dir": str(data_root), "train_repo_ids": [args.pack], "val_repo_ids": [args.val_pack],
                 "dino_cache_root": str(cache_root), "shared_image_profile": "256x320",
                 "dino_max_open_episodes": 2, "transform": {
                     "repack": {"dataset_name": "unifolm_sonic", "image_keys": ["observation.images.egocentric"],
                                "instruction_key": "task_description", "state_keys": ["observation.state"],
                                "action_keys": ["action.body_token_v1_1", "action[:14]"],
                                "action_chunk_size": 30, "pad_action_dim": 80, "pad_state_dim": 45},
                     "field": {"stat_path": str(stats), "stat_state_keys": ["observation.state"],
                               "stat_action_keys": ["action.body_token_v1_1", "action[:14]"],
                               "action_norm_type": "bounds", "pad_action_dim": 80,
                               "pad_state_dim": 45, "normalize_state": True},
                     "model": {"motion_image_policy": "vlm_color_clean_motion", "img_aug": True,
                               "view_aug": False, "adaptive_resize": False}}},
        "model": {"action_architecture": "dual_source_v1", "model_name_or_path": str(vlm),
                  "motion_checkpoint": str(motion), "motion_use_ema": True, "motion_feature_seed": 0,
                  "tune_vlm": True, "tune_motion": True, "gradient_checkpointing": True,
                  "action_dim": 80, "action_chunk_size": 30, "action_exec_horizon": 30, "odim": 45,
                  "hidden_dim": 1536, "num_blocks": 6, "nhead": 24, "view_feature_dim": 2048,
                  "n_conditions": 0, "noise_scheduler": "flow", "eval_diffusion_steps": 3,
                  "dropout": 0.0, "state_feature_dropout": 0.0, "qk_norm": "rms_norm",
                  "state_drop_prob": 0.5, "state_null_token": True,
                  "lang_backbone_lr": 1e-6, "vision_tower_lr": 1e-5, "mm_projector_lr": 1e-4,
                  "motion_expert_lr": 1e-5}})


def check_binding(args, cfg):
    """Check the prehashed pin; never reread many GB on every rank."""
    path = Path(args.binding)
    binding = json.loads(path.read_text())
    for section, artifact, size_key, mtime_key in (
        ("vlm", Path(cfg.model.model_name_or_path) / "model.safetensors", "model_bytes", "model_mtime_ns"),
        ("motion", Path(cfg.model.motion_checkpoint), "bytes", "mtime_ns"),
    ):
        item = binding[section]
        expected_path = Path(cfg.model.model_name_or_path) if section == "vlm" else artifact
        if Path(item["path"]).resolve() != expected_path.resolve():
            raise ValueError(f"{section} path differs from the pinned binding")
        stat = artifact.stat()
        if stat.st_size != item[size_key] or stat.st_mtime_ns != item[mtime_key]:
            raise ValueError(f"{section} bytes/mtime differ from the prehashed binding")
    if binding["motion"]["use_ema"] is not True or not cfg.model.motion_use_ema:
        raise ValueError("This gate requires the pinned EMA Motion initialization")
    if binding["motion"]["completed_epochs"] != 100 or binding["motion"]["hidden_shape"] != [11, 768]:
        raise ValueError("This gate requires the pinned 100-epoch, eleven-token MotionExpert")
    if not binding["motion"]["cross_size_adaptation_explicit"]:
        raise ValueError("Motion256x480 -> UnifoLM256x320 adaptation must be explicitly bound")
    for name, expected in binding["vlm"]["config_tokenizer_hashes"].items():
        if hashlib.sha256((Path(cfg.model.model_name_or_path) / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"VLM {name} differs from its pinned SHA256")
    return {"record": str(path.resolve()), "record_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "identities": binding, "verification": "prehashed identity plus current path/size/mtime"}


def accelerator_for(trainer):
    # Reuse the real entrypoint's precision/DDP policy without invoking train().
    spec = importlib.util.spec_from_file_location("_bounded_psi0_training_entrypoint", VLM_ROOT / "scripts/train.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._initialize_accelerator(trainer)


def make_evaluation_data(cfg, processor, batch_size):
    """Load only the real validation pack for cold inference and its reference."""
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

    val_data = cfg.data(split="val", transform_kwargs={"vlm_processor": processor})
    if len(val_data) < batch_size:
        raise ValueError(f"Need {batch_size} eligible validation rows, found {len(val_data)}")
    collator = PaddedCollatorForTogether(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id)
    samples = [val_data[index] for index in range(batch_size)]
    batch = collator(samples)
    metadata = {"eligible_val_rows": len(val_data), "val_packs": cfg.data.val_repo_ids,
                "selected_val_filtered_rows": list(range(batch_size)),
                "selected_val_native_rows": [int(val_data.raw_dataset.motion_history.original_indices[i]) for i in range(batch_size)],
                "batch_size_per_rank": batch_size, "rgb_policy": cfg.data.transform.model.motion_image_policy,
                "history_shape": list(batch["motion_img_embs"].shape), "workers": 0}
    return samples, batch, metadata


def make_data(cfg, processor, world_size, batch_size, updates, accumulation):
    from torch.utils.data import DataLoader, Subset
    from psi.trainers.qwen3vl_mixin import PaddedCollatorForTogether

    train_data = cfg.data(split="train", transform_kwargs={"vlm_processor": processor})
    count = world_size * batch_size * updates * accumulation
    if len(train_data) < count:
        raise ValueError(f"Need {count} distinct eligible native rows, found {len(train_data)}")
    indices = [i * (len(train_data) - 1) // max(count - 1, 1) for i in range(count)]
    collator = PaddedCollatorForTogether(processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id)
    # Main-process decode only: avoid LeRobot's documented parent/worker decoder mix.
    loader = DataLoader(Subset(train_data, indices), batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collator)
    evaluation_samples, batch, evaluation_metadata = make_evaluation_data(cfg, processor, batch_size)
    metadata = {**evaluation_metadata, "eligible_rows": len(train_data), "selected_filtered_rows": indices,
                "selected_native_rows": [int(train_data.raw_dataset.motion_history.original_indices[i]) for i in indices],
                "train_packs": cfg.data.train_repo_ids, "val_packs": cfg.data.val_repo_ids,
                "batch_size_per_rank": batch_size,
                "rgb_policy": cfg.data.transform.model.motion_image_policy,
                "history_shape": list(batch["motion_img_embs"].shape), "workers": 0}
    return loader, evaluation_samples, batch, metadata


def head_contract(head):
    """Inspect the constructed modules, independently of the requested config."""
    signature = head._architecture_signature.detach().cpu().tolist()
    branch_heads, branch_widths, modulation_widths, ffn_widths = [], [], [], []
    for block in head.transformer_blocks:
        branches = (block.self_attn, block.vlm_attn, block.motion_attn)
        branch_heads.append([branch.heads for branch in branches])
        branch_widths.append([branch.to_q.out_features for branch in branches])
        modulation_widths.append(block.mod_head[-1].out_features)
        ffn_widths.append(block.ff_act[0].out_features)
    if (head.action_hidden_dim != 1536 or len(head.transformer_blocks) != 6
            or branch_heads != [[24, 24, 24]] * 6 or branch_widths != [[1536, 1536, 1536]] * 6
            or modulation_widths != [12 * 1536] * 6 or ffn_widths != [4 * 1536] * 6
            or signature[2:10] != [80, 30, 1536, 6, 24, 2048, 45, 256]):
        raise AssertionError("Acceptance requires the actual complete 1536x6/24 head with four residuals and 12d modulation per block")
    return {"class": type(head).__name__, "architecture_id": head.architecture_id,
            "hidden_dim": head.action_hidden_dim, "blocks": len(head.transformer_blocks),
            "heads_per_self_vlm_motion_branch": branch_heads, "branch_widths": branch_widths,
            "modulation_widths": modulation_widths, "ffn_widths": ffn_widths,
            "architecture_signature": signature}


def parameter_inventory(model, *, require_fp32):
    import torch

    components, metadata = {}, []
    for name, parameter in model.named_parameters():
        if name.startswith("action_header."):
            component = "action_header"
        elif name.startswith("motion_conditioner."):
            component = "motion_expert"
        elif name.startswith("vlm_model."):
            component = "mm_projector" if "merger" in name else "vision_tower" if "visual" in name else "lang_backbone"
        else:
            component = "other"
        item = components.setdefault(component, {"named_parameter_tensors": 0, "elements": 0,
                                                 "trainable_tensors": 0, "trainable_elements": 0,
                                                 "frozen_tensors": 0, "frozen_elements": 0, "elements_by_dtype": {}})
        item["named_parameter_tensors"] += 1
        item["elements"] += parameter.numel()
        kind = "trainable" if parameter.requires_grad else "frozen"
        item[f"{kind}_tensors"] += 1
        item[f"{kind}_elements"] += parameter.numel()
        dtype = str(parameter.dtype)
        item["elements_by_dtype"][dtype] = item["elements_by_dtype"].get(dtype, 0) + parameter.numel()
        metadata.append((name, tuple(parameter.shape), dtype, parameter.requires_grad))
        if require_fp32 and parameter.requires_grad and parameter.dtype != torch.float32:
            raise AssertionError(f"Training requires FP32 master parameters: {name} is {parameter.dtype}")
    return {"components": components, "named_parameter_tensors": len(metadata),
            "elements": sum(item["elements"] for item in components.values()),
            "named_parameter_metadata_sha256": hashlib.sha256(json.dumps(metadata).encode()).hexdigest(),
            "trainable_fp32_master_check": "passed" if require_fp32 else "deployment dtype inventory only"}


def model_contract(model, *, require_fp32):
    # Qwen4.57 dispatches both attention implementations using EACH attention
    # module's config. Checking only the outer configuration would be weaker.
    attention = {}
    for module in model.vlm_model.modules():
        name = type(module).__name__
        if name in ("Qwen3VLTextAttention", "Qwen3VLVisionAttention"):
            backend = module.config._attn_implementation
            if backend != "flash_attention_2":
                raise AssertionError(f"Actual {name} dispatch is {backend}, expected flash_attention_2")
            item = attention.setdefault(name, {"modules": 0, "backend": backend})
            item["modules"] += 1
    if set(attention) != {"Qwen3VLTextAttention", "Qwen3VLVisionAttention"}:
        raise AssertionError("Expected real Qwen3-VL language AND vision attention modules")
    return {"head": head_contract(model.action_header), "qwen_attention": attention,
            "parameters": parameter_inventory(model, require_fp32=require_fp32)}


class FeatureAudit:
    def __init__(self, model):
        self.current = None
        self.calls = {"vlm": 0, "motion": 0}
        self.shapes = None
        self.handles = [model.action_header.register_forward_pre_hook(self._head, with_kwargs=True),
                        model.vlm_model.register_forward_hook(self._vlm),
                        model.motion_conditioner.register_forward_hook(self._motion)]

    def _vlm(self, *unused):
        self.calls["vlm"] += 1

    def _motion(self, *unused):
        self.calls["motion"] += 1

    def _head(self, module, positional, kwargs):
        import torch

        context = kwargs["joint_attention_kwargs"]
        vlm, motion = context["vlm_features"], context["motion_features"]
        mask = vlm.attention_mask.bool()
        expected_v = vlm.tokens[:, 0].masked_fill(~mask[..., None], 0).sum(1, dtype=torch.float32)
        expected_v = (expected_v / mask.sum(1, keepdim=True)).to(vlm.p_v.dtype)
        torch.testing.assert_close(vlm.tokens[:, 0], vlm.last_hidden_state, rtol=0, atol=0)
        torch.testing.assert_close(vlm.p_v, expected_v, rtol=0, atol=0)
        torch.testing.assert_close(motion.p_m, motion.tokens.mean(1, dtype=torch.float32).to(motion.p_m.dtype), rtol=0, atol=0)
        if tuple(motion.tokens.shape[1:]) != (11, 768) or vlm.tokens.shape[1] != 1:
            raise AssertionError("Gate requires last VLM layer and ALL eleven motion hidden tokens")
        self.shapes = {"vlm_tokens": list(vlm.tokens.shape), "p_v": list(vlm.p_v.shape),
                       "motion_tokens": list(motion.tokens.shape), "p_m": list(motion.p_m.shape)}
        self.current = {"vlm_tokens": vlm.tokens, "p_v": vlm.pooled_tokens,
                        "motion_tokens": motion.tokens, "p_m": motion.p_m}
        if torch.is_grad_enabled():
            for value in self.current.values():
                if not value.requires_grad:
                    raise AssertionError("A trainable source was detached")
                value.retain_grad()
        elif any(value.requires_grad for value in self.current.values()):
            raise AssertionError("Sampling unexpectedly retained a source graph")

    def gradients(self):
        import torch

        result = {}
        for name, value in self.current.items():
            if value.grad is None or not torch.isfinite(value.grad).all():
                raise AssertionError(f"Missing/nonfinite {name} gradient")
            result[name] = float(value.grad.float().norm())
        self.current = None
        return result

    def close(self):
        self.current = None
        for handle in self.handles:
            handle.remove()


def group_norms(optimizer, *, gradients):
    import torch

    result = {}
    for group in optimizer.param_groups:
        values = [p.grad for p in group["params"] if p.grad is not None] if gradients else group["params"]
        with torch.no_grad():
            norms = torch.stack(torch._foreach_norm(values)) if values else torch.zeros(1, device="cuda")
        if not torch.isfinite(norms).all():
            raise AssertionError(f"Nonfinite {'gradient' if gradients else 'parameter'} in {group['group_name']}")
        result[group["group_name"]] = float(norms.double().square().sum().sqrt())
    return result


def integer_probe_indices(n, count, *, device=None):
    """Evenly cover a flattened parameter without float32 endpoint rounding."""
    import torch

    if n < 0 or count < 0:
        raise ValueError("Probe length and count must be nonnegative")
    count = min(n, count)
    return torch.arange(count, dtype=torch.long, device=device) * (n - 1) // max(count - 1, 1)


def probes(optimizer):
    import torch

    result = {}
    for group in optimizer.param_groups:
        samples = []
        for parameter in group["params"]:
            flat = parameter.detach().reshape(-1)
            indices = integer_probe_indices(flat.numel(), 32, device=flat.device)
            samples.append(flat[indices].float().cpu())
        result[group["group_name"]] = torch.cat(samples)
    return result


def consistency(model, accelerator):
    import torch.distributed as dist

    digest = tree_hash(model.state_dict())
    if accelerator.num_processes > 1:
        digests = [None] * accelerator.num_processes
        dist.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise AssertionError(f"Full model state diverged across ranks: {digests}")
    return digest


def memory(device):
    import torch

    torch.cuda.synchronize(device)
    return {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device)}


@contextmanager
def deterministic_sampling(seed):
    import numpy as np
    import torch

    python_state, numpy_state = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def trainer_inference(trainer, batch, audit, seed):
    import torch
    from psi.utils import move_to_device

    raw = trainer.accelerator.unwrap_model(trainer.model)
    raw.eval()
    before = dict(audit.calls)
    with deterministic_sampling(seed), trainer.forward_autocast():
        prediction = trainer.inference(raw, move_to_device(batch, trainer.device))
    if prediction.requires_grad or not torch.isfinite(prediction).all():
        raise AssertionError("Three-step real sampling must be finite without a graph")
    if {name: audit.calls[name] - before[name] for name in before} != {"vlm": 1, "motion": 1}:
        raise AssertionError("Sources must run once per chunk, not per diffusion step")
    if tuple(prediction.shape) != (batch["states"].shape[0], 30, 80):
        raise AssertionError("Incorrect sampled action shape")
    audit.current = None
    return prediction.detach().float().cpu()


def predict(model, samples, batch, device, seed):
    import torch
    from psi.models.psi0 import HumanFoundationModelOutput

    with deterministic_sampling(seed):
        output = model.predict_action(observations=[item["observations"] for item in samples],
                                      instructions=[item["instruction"] for item in samples],
                                      states=batch["states"].to(device), num_inference_steps=3, traj2ds=None,
                                      motion_img_embs=batch["motion_img_embs"].to(device), return_vlm_features=True)
    # The production API returns BaseOutput with .action when this flag is true,
    # and a plain Tensor otherwise. Never reinterpret its legacy to_tuple().
    if not isinstance(output, HumanFoundationModelOutput) or output.vlm_features is None or output.motion_features is None:
        raise AssertionError("predict_action(return_vlm_features=True) must return the real complete source output")
    if (output.action.requires_grad or not torch.isfinite(output.action).all()
            or tuple(output.action.shape) != (len(samples), 30, 80)):
        raise AssertionError("Deployment sampling must be finite without a graph")
    return output.action.detach().float().cpu()


def run(args, report):
    import numpy as np
    import PIL
    import torch
    from psi.utils import move_to_device
    from psi.trainers.posttrain import PosttrainTrainer

    if np.__version__ != "1.26.4" or PIL.__version__ != "11.3.0":
        raise RuntimeError("Real cache contract requires NumPy1.26.4 and Pillow11.3.0")
    if not torch.cuda.is_available():
        raise RuntimeError("This real-model gate requires CUDA; CPU substitutes are unsupported")
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    seed_all(args.seed)
    cfg = config(args)
    report.update(mode=args.mode, binding=check_binding(args, cfg), torch=torch.__version__,
                  numpy=np.__version__, pillow=PIL.__version__, device=torch.cuda.get_device_name(device),
                  precision="FP32 master parameters / BF16 autocast", optimizer_foreach=False,
                  scope="bounded acceptance updates only; no formal training", inference_steps=3,
                  deployment_api={"method": "Psi0Model.predict_action", "return_vlm_features": True,
                                  "required_return_type": "HumanFoundationModelOutput", "action_layout": "B,30,80"},
                  timing_is_instrumented_acceptance_not_throughput=True)
    if args.mode == "inference":
        if int(os.getenv("WORLD_SIZE", "1")) != 1:
            raise ValueError("Cold inference restore is a separate single-GPU invocation")
        from psi.models.psi0 import Psi0Model

        checkpoint = args.checkpoint.resolve()
        model = Psi0Model.from_pretrained(str(checkpoint.parent.parent), checkpoint.name.removeprefix("ckpt_"), cfg, device)
        model = model.to(device).eval()
        report["actual_model_contract"] = model_contract(model, require_fp32=False)
        report["precision"] = "BF16 VLM / FP32 Motion+head, BF16 sampling autocast"
        samples, batch, metadata = make_evaluation_data(cfg, model.vlm_processor, cfg.train.val_batch_size)
        reference = torch.load(checkpoint / "validation_reference_rank0.pt", map_location="cpu", weights_only=False)
        if tree_hash(batch) != reference["eval_batch_hash"]:
            raise AssertionError("The live validation data/tokenization changed since save")
        audit = FeatureAudit(model)
        result = predict(model, samples, batch, device, reference["sampling_seed"])
        if audit.calls != {"vlm": 1, "motion": 1}:
            raise AssertionError("Cold restored prediction repeated source extraction")
        torch.testing.assert_close(result, reference["deployment_prediction"], rtol=0, atol=args.comparison_atol)
        report.update(data=metadata, max_prediction_difference=float((result - reference["deployment_prediction"]).abs().max()),
                      comparison_atol=args.comparison_atol, feature_shapes=audit.shapes,
                      inference_source_calls_per_chunk=audit.calls, memory=memory(device), passed=True)
        audit.close()
        return

    trainer = PosttrainTrainer(cfg, device)
    trainer.init_models()
    accelerator = accelerator_for(trainer)
    report["rank"], report["world_size"] = accelerator.process_index, accelerator.num_processes
    loader, samples, eval_batch, metadata = make_data(cfg, trainer.vlm_processor, accelerator.num_processes,
                                                    cfg.train.train_batch_size, cfg.train.max_training_steps,
                                                    cfg.train.gradient_accumulation_steps)
    report["data"] = metadata
    trainer.train_dataloader = loader
    trainer.create_optimizer_and_scheduler(cfg.train.max_training_steps)
    prepared_loader = trainer.prepare(accelerator)
    raw = accelerator.unwrap_model(trainer.model)
    report["actual_model_contract"] = model_contract(raw, require_fp32=True)
    if any(not p.requires_grad for p in raw.vlm_model.model.language_model.norm.parameters()):
        raise AssertionError("The consumed final Qwen normalization must remain trainable")
    tied_head = raw.vlm_model.lm_head.weight is raw.vlm_model.get_input_embeddings().weight
    if not tied_head and raw.vlm_model.lm_head.weight.requires_grad:
        raise AssertionError("An unused untied vocabulary head must remain frozen")
    frozen = {name: p for name, p in raw.named_parameters() if not p.requires_grad}
    frozen_hash = tree_hash(frozen)
    report["parameters"] = {"total": sum(p.numel() for p in raw.parameters()),
                            "trainable": sum(p.numel() for p in raw.parameters() if p.requires_grad),
                            "groups": {g["group_name"]: {"parameters": sum(p.numel() for p in g["params"]), "lr": g["lr"]}
                                       for g in trainer.optimizer.param_groups}}
    audit = FeatureAudit(raw)
    sampling_seed = args.seed + 991

    if args.mode == "resume":
        reference = torch.load(args.checkpoint / f"validation_reference_rank{accelerator.process_index}.pt",
                               map_location="cpu", weights_only=False)
        if reference["world_size"] != accelerator.num_processes:
            raise ValueError("Accelerator resume must use the saved rank count")
        if tree_hash(eval_batch) != reference["eval_batch_hash"]:
            raise AssertionError("The live validation data/tokenization changed since save")
        expected_rng = checkpoint_rng_hash(args.checkpoint, accelerator.process_index, device.index)
        report["rng_restore"] = {"source": f"random_states_{accelerator.process_index}.pkl",
                                 "expected_checkpoint_hash": expected_rng,
                                 "reference_sidecar_hash": reference["rng_hash"],
                                 "legacy_reference_rng_drift": expected_rng != reference["rng_hash"]}
        accelerator.load_state(str(args.checkpoint))
        report["rng_restore"]["actual_hash"] = rng_hash()
        if report["rng_restore"]["actual_hash"] != expected_rng:
            raise AssertionError("Accelerator did not restore Python/NumPy/CPU/CUDA RNG state")
        if consistency(raw, accelerator) != reference["model_hash"]:
            raise AssertionError("Accelerator restored a different full model state")
        if tree_hash(trainer.optimizer.state_dict()) != reference["optimizer_hash"]:
            raise AssertionError("Accelerator did not restore all optimizer slots/groups")
        if tree_hash(trainer.lr_scheduler.state_dict()) != reference["scheduler_hash"]:
            raise AssertionError("Accelerator did not restore the scheduler")
        result = trainer_inference(trainer, eval_batch, audit, reference["sampling_seed"])
        torch.testing.assert_close(result, reference["training_prediction"], rtol=0, atol=0)
        report.update(restored_model_optimizer_scheduler_rng=True, max_prediction_difference=0.0)
        iterations = 1
    else:
        iterations = cfg.train.max_training_steps

    initial_probes = probes(trainer.optimizer)
    reports, feature_reports = [], []
    iterator = iter(prepared_loader)
    raw.train()
    for update in range(iterations):
        start = time.perf_counter()
        for micro in range(cfg.train.gradient_accumulation_steps):
            batch = next(iterator)
            if batch["motion_img_embs"].requires_grad:
                raise AssertionError("DINO cache must remain a frozen input")
            history_hash = tree_hash(batch["motion_img_embs"])
            with accelerator.accumulate(trainer.model):
                with trainer.forward_autocast():
                    losses = trainer.forward_and_loss(trainer.model, move_to_device(batch, device))
                if not torch.isfinite(losses["loss"]):
                    raise AssertionError("Nonfinite real-model loss")
                accelerator.backward(losses["loss"])
                feature_reports.append(audit.gradients())
                if accelerator.sync_gradients:
                    gradient_norms = group_norms(trainer.optimizer, gradients=True)
                    trainer.clip_grad_norms()
                    trainer.optimizer.step()
                    trainer.lr_scheduler.step()
                    trainer.optimizer.zero_grad(set_to_none=True)
            if tree_hash(batch["motion_img_embs"]) != history_hash:
                raise AssertionError("An update mutated the clean six-frame DINO cache input")
        group_norms(trainer.optimizer, gradients=False)
        digest = consistency(raw, accelerator)
        reports.append({"update": update + 1, "loss": float(losses["loss"].detach()),
                        "gradient_norms_before_clip": gradient_norms, "full_model_sha256": digest,
                        "seconds": time.perf_counter() - start, "memory": memory(device)})
        print(json.dumps({"rank": accelerator.process_index, **reports[-1]}), flush=True)

    final_probes = probes(trainer.optimizer)
    changes = {key: float((final_probes[key] - value).abs().max()) for key, value in initial_probes.items()}
    required = ("action_header", "motion_expert", "lang_backbone", "vision_tower", "mm_projector")
    if any(changes.get(key, 0) <= 0 for key in required):
        raise AssertionError(f"Every tuned component must actually update: {changes}")
    if any(max(item[key] for item in feature_reports) <= 0 for key in ("p_v", "p_m", "vlm_tokens", "motion_tokens")):
        raise AssertionError("Source tokens and BOTH mean-pool branches must receive nonzero gradients after gates open")
    training_prediction = trainer_inference(trainer, eval_batch, audit, sampling_seed)
    if tree_hash(frozen) != frozen_hash:
        raise AssertionError("An optimizer update changed frozen/inactive parameters")
    report.update(updates=reports, feature_shapes=audit.shapes, feature_gradients=feature_reports,
                  sampled_parameter_max_changes=changes, inference_source_calls_per_chunk={"vlm": 1, "motion": 1},
                  frozen_parameters_unchanged=True, memory=memory(device))
    if args.mode == "train":
        checkpoint = Path(trainer.save_checkpoint(iterations))
        if accelerator.is_main_process:
            (checkpoint / "validation_config.json").write_text(cfg.model_dump_json(indent=2))
            write_json(checkpoint / "validation_binding.json", report["binding"]["identities"])
        reference = {"world_size": accelerator.num_processes, "sampling_seed": sampling_seed,
                     "rng_hash": checkpoint_rng_hash(checkpoint, accelerator.process_index, device.index),
                     "rng_reference_point": "serialized_accelerator_checkpoint",
                     "model_hash": reports[-1]["full_model_sha256"],
                     "optimizer_hash": tree_hash(trainer.optimizer.state_dict()),
                     "scheduler_hash": tree_hash(trainer.lr_scheduler.state_dict()),
                     "training_prediction": training_prediction, "eval_batch_hash": tree_hash(eval_batch)}
        # from_pretrained constructs BF16 VLM weights. Compare against the SAME
        # BF16 deployment conversion, after saving unmodified FP32 training state.
        raw.vlm_model.to(dtype=torch.bfloat16)
        raw.vlm_processor, raw.noise_scheduler = trainer.vlm_processor, trainer.noise_scheduler
        raw.action_horizon, raw.action_dim, raw.device = 30, 80, device
        reference["deployment_prediction"] = predict(raw, samples, eval_batch, device, sampling_seed)
        torch.save(reference, checkpoint / f"validation_reference_rank{accelerator.process_index}.pt")
        report["checkpoint"] = str(checkpoint)
    audit.close()
    report["passed"] = True
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank = int(os.getenv("RANK", "0"))
    report_path = args.output_dir / f"{args.mode}_rank{rank}.json"
    if report_path.exists():
        raise FileExistsError(f"Choose a fresh output directory; preserving {report_path}")
    report = {"passed": False, "rank": rank, "started_unix": time.time()}
    started = time.perf_counter()
    try:
        run(args, report)
        print(json.dumps({"mode": args.mode, "rank": rank, "passed": report["passed"],
                          "checkpoint": report.get("checkpoint"), "report": str(report_path)}), flush=True)
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        write_json(report_path, report)


if __name__ == "__main__":
    main()
