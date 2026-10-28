"""Deterministic, episode-disjoint Mobile overfit diagnostics.

The caller supplies a production val/no_aug reader and already frozen filtered
row indices. Full FM loss uses every row of five VAL and five TRAIN-probe
episodes; generated-action L1 uses eight uniform rows per episode. This module
does not select a split, change the model, or write a training/evaluation receipt.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
from numbers import Integral
import re
import random

CONTRACT = "cvpr-mobile-pp95-5-overfit-metrics-v1"
ROLES = ("val", "train_probe")
EPISODES_PER_ROLE = 5
SAMPLES_PER_EPISODE = 8
# FM sum/count, body absolute-error sum/count, hand sum/count, sampled windows.
COLUMNS = 7


@contextlib.contextmanager
def _isolated_rng(seed=None, device="cpu"):
    """Preserve all RNG used by this rank without touching other CUDA devices.

    torch.manual_seed and the shared all-visible-device fork would seed or query
    nonlocal CUDA generators. Production forward uses trainer.device only, so
    nonlocal generators remain untouched rather than initialized for a snapshot.
    """
    import numpy as np
    import torch

    device = torch.device(device)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_index = (device.index if device.index is not None else torch.cuda.current_device()) \
        if device.type == "cuda" else None
    cuda_state = torch.cuda.get_rng_state(cuda_index) if cuda_index is not None else None
    try:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.random.default_generator.manual_seed(seed)
            if cuda_index is not None:
                with torch.cuda.device(cuda_index):
                    torch.cuda.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, cuda_index)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _seed(split_sha256, role, episode, stream, rank, batch, rows):
    identity = [split_sha256, role, episode, stream, rank, batch, list(rows)]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def _row_sha(rows):
    return hashlib.sha256(json.dumps(list(rows), separators=(",", ":")).encode()).hexdigest()


def _rank_world(trainer):
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    accelerator = getattr(trainer, "accelerator", None)
    _require(getattr(accelerator, "num_processes", 1) == 1,
             "Distributed diagnostics require the initialized training process group")
    return 0, 1


def _reduce_sum(packed, world):
    if world > 1:
        import torch.distributed as dist
        dist.all_reduce(packed, op=dist.ReduceOp.SUM)


def _prepare_batch(batch, device):
    # Use the same encoded-string decoding and device transfer as production.
    from psi.utils import batch_str_to_tensor, move_to_device
    return move_to_device(batch_str_to_tensor(batch), device)


def _specification(trainer, bundle, completed_update, split_sha256):
    _require(type(completed_update) is int and completed_update >= 0,
             "completed_update must count nonnegative completed optimizer updates")
    _require(isinstance(split_sha256, str) and re.fullmatch(r"[0-9a-f]{64}", split_sha256),
             "Bind diagnostics to the frozen split SHA256")
    dataset = _get(bundle, "clean_dataset")
    _require(dataset is not None, "Supply the production val/no_aug clean_dataset")
    batch_size = _get(bundle, "eval_batch_size", None)
    if batch_size is None:
        batch_size = getattr(getattr(trainer, "train_cfg", None), "val_batch_size", 1)
    _require(type(batch_size) is int and batch_size > 0, "eval_batch_size must be positive")
    collate = _get(bundle, "collate_fn", None)
    if collate is None:
        collate = getattr(getattr(trainer, "train_dataloader", None), "collate_fn", None)
    _require(callable(collate), "Use the production collate_fn")
    result = {"dataset": dataset, "batch_size": batch_size, "collate": collate, "roles": {}}
    seen_rows, seen_episodes = set(), set()
    length = len(dataset)
    for role in ROLES:
        candidates = _get(bundle, role + "_rows_by_episode")
        _require(isinstance(candidates, dict) and len(candidates) == EPISODES_PER_ROLE,
                 role + " must contain exactly five complete source episodes")
        episodes = {}
        for key, values in candidates.items():
            _require(isinstance(key, (str, Integral)) and not isinstance(key, bool),
                     "Episode identities must be strings or integer source indices")
            episode = str(key)
            _require(episode and episode not in episodes and episode not in seen_episodes,
                     "VAL and TRAIN probe source episodes must be distinct")
            _require(isinstance(values, (list, tuple)) and len(values) >= SAMPLES_PER_EPISODE,
                     "Each complete source episode needs at least eight eligible rows")
            _require(all(isinstance(row, Integral) and not isinstance(row, bool) for row in values),
                     "Filtered production row indices must be integers")
            rows = [int(row) for row in values]
            _require(rows == sorted(set(rows)) and all(0 <= row < length for row in rows),
                     "Episode rows must be unique, sorted, and inside the clean reader")
            _require(not seen_rows.intersection(rows), "Source rows overlap across diagnostic episodes")
            sampled = [rows[i * (len(rows) - 1) // (SAMPLES_PER_EPISODE - 1)]
                       for i in range(SAMPLES_PER_EPISODE)]
            episodes[episode] = {"rows": rows, "sampled": sampled}
            seen_rows.update(rows)
            seen_episodes.add(episode)
        result["roles"][role] = dict(sorted(episodes.items()))
    return result


def _check_batch(batch, expected_size):
    import torch
    actions, mask = batch["actions"], batch["actions_mask"]
    _require(torch.is_tensor(actions) and tuple(actions.shape) == (expected_size, 30, 80)
             and torch.isfinite(actions).all().item(), "Expected finite production actions Bx30x80")
    _require(torch.is_tensor(mask) and mask.shape == actions.shape
             and torch.isfinite(mask).all().item()
             and ((mask == 0) | (mask == 1)).all().item(), "Invalid production supervision mask")
    valid = mask.bool()
    _require(valid[:, :, :78].all().item() and not valid[:, :, 78:].any().item(),
             "Require body64/hand14 supervision with both padding channels excluded")
    return actions


def _accumulate(trainer, model, spec, packed, rank, world, split_sha256):
    import torch
    from torch.utils.data import DataLoader, Subset

    slot = 0
    for role in ROLES:
        for episode, selection in spec["roles"][role].items():
            for stream, source_rows in (("full_fm", selection["rows"]),
                                        ("sampled_l1", selection["sampled"])):
                local_rows = source_rows[rank::world]
                if not local_rows:
                    continue
                generator = torch.Generator().manual_seed(
                    _seed(split_sha256, role, episode, stream, rank, -1, local_rows))
                loader = DataLoader(Subset(spec["dataset"], local_rows),
                                    batch_size=spec["batch_size"], shuffle=False, num_workers=0,
                                    drop_last=False, collate_fn=spec["collate"], generator=generator)
                iterator = iter(loader)
                for batch_index, start in enumerate(range(0, len(local_rows), spec["batch_size"])):
                    batch_rows = local_rows[start:start + spec["batch_size"]]
                    # Fetch/collate are inside the fixed seed context too. The
                    # outer isolation additionally protects DataLoader creation.
                    seed = _seed(split_sha256, role, episode, stream, rank, batch_index, batch_rows)
                    with _isolated_rng(seed, trainer.device):
                        batch = _prepare_batch(next(iterator), trainer.device)
                        actions = _check_batch(batch, len(batch_rows))
                        with trainer.forward_autocast():
                            if stream == "full_fm":
                                loss = trainer.forward_and_loss(model, batch)["loss"]
                            else:
                                prediction = trainer.inference(model, batch)
                        if stream == "full_fm":
                            _require(torch.is_tensor(loss) and loss.numel() == 1
                                     and torch.isfinite(loss).all().item(), "Nonfinite or nonscalar FM loss")
                            packed[slot, 0] += loss.detach().double().reshape(()) * len(batch_rows)
                            packed[slot, 1] += len(batch_rows)
                        else:
                            _require(torch.is_tensor(prediction) and prediction.shape == actions.shape
                                     and torch.isfinite(prediction).all().item(), "Invalid sampled action prediction")
                            error = (prediction.detach().double() - actions.double()).abs()
                            body, hand = error[:, :, :64], error[:, :, 64:78]
                            packed[slot, 2] += body.sum()
                            packed[slot, 3] += body.numel()
                            packed[slot, 4] += hand.sum()
                            packed[slot, 5] += hand.numel()
                            packed[slot, 6] += len(batch_rows)
            slot += 1


def _report(packed, spec, completed_update, split_sha256, world):
    totals = packed.cpu().tolist()
    result = {"contract": CONTRACT, "completed_update": completed_update,
              "split_sha256": split_sha256, "world_size": world,
              "eval_batch_size": spec["batch_size"], "metrics": {},
              "dataset_role": "caller_bound_production_val_no_aug",
              "full_fm_scope": "all eligible filtered rows in each of five episodes per role",
              "sampled_l1_scope": "eight fixed uniform rows per episode; 40 per role; normalized channels",
              "padding_channels_in_metrics": False,
              "seed_policy": "sha256(split,role,episode,stream,rank,batch,rows); independent of checkpoint/update",
              "rng_isolation": "Python, NumPy, torch CPU and rank-local CUDA restored; nonlocal CUDA never queried or seeded",
              "distributed_policy": "rows[rank::world], no padding/repeats; one final SUM all_reduce",
              "roles": {}}
    slot = 0
    for role in ROLES:
        episodes = {}
        for episode, selection in spec["roles"][role].items():
            fm_sum, count, body_sum, body_n, hand_sum, hand_n, sampled_n = totals[slot]
            _require(count == len(selection["rows"]) and sampled_n == SAMPLES_PER_EPISODE,
                     "Distributed evaluation lost or duplicated source windows")
            _require(body_n == sampled_n * 30 * 64 and hand_n == sampled_n * 30 * 14,
                     "Sampled channel-element counts differ")
            episodes[episode] = {"full_fm_loss": fm_sum / count, "full_fm_windows": int(count),
                "sampled_normalized_body_token_l1": body_sum / body_n,
                "sampled_normalized_hand_joint_l1": hand_sum / hand_n,
                "sampled_l1_windows": int(sampled_n), "body_elements": int(body_n), "hand_elements": int(hand_n),
                "full_rows_sha256": _row_sha(selection["rows"]), "sampled_rows": selection["sampled"]}
            slot += 1
        segment = totals[slot - EPISODES_PER_ROLE:slot]
        counts = {"episodes": EPISODES_PER_ROLE, "full_fm_windows": int(sum(x[1] for x in segment)),
                  "sampled_l1_windows": int(sum(x[6] for x in segment)),
                  "body_elements": int(sum(x[3] for x in segment)),
                  "hand_elements": int(sum(x[5] for x in segment))}
        metrics = {"full_fm_loss_micro": sum(x[0] for x in segment) / sum(x[1] for x in segment),
                   "full_fm_loss_episode_macro": sum(x["full_fm_loss"] for x in episodes.values()) / EPISODES_PER_ROLE}
        for name, numerator, denominator in (("body_token", 2, 3), ("hand_joint", 4, 5)):
            prefix = "sampled_normalized_" + name + "_l1"
            metrics[prefix + "_micro"] = sum(x[numerator] for x in segment) / sum(x[denominator] for x in segment)
            metrics[prefix + "_episode_macro"] = sum(x[prefix] for x in episodes.values()) / EPISODES_PER_ROLE
        result["roles"][role] = {"counts": counts, "per_episode": episodes, "metrics": metrics}
        result["metrics"].update({role + "_" + key: value for key, value in metrics.items()})
    for key, value in result["roles"]["val"]["metrics"].items():
        result["metrics"]["gap_" + key] = value - result["roles"]["train_probe"]["metrics"][key]
    _require(all(math.isfinite(value) for value in result["metrics"].values()), "Nonfinite reduced diagnostics")
    return result


def evaluate_diagnostics(trainer, bundle, completed_update, split_sha256):
    """Return collective full-loss and sampled-L1 diagnostics on every rank.

    Required bundle fields: clean_dataset, val_rows_by_episode, and
    train_probe_rows_by_episode (five complete episodes each). Optional
    eval_batch_size/collate_fn default to the trainer's production settings.
    No per-batch collective is used, including when rank row counts differ.
    """
    import torch

    rank, world = _rank_world(trainer)
    _require(world > 0 and 0 <= rank < world, "Invalid distributed rank/world")
    packed = torch.zeros((2 * EPISODES_PER_ROLE + 1, COLUMNS), dtype=torch.float64, device=trainer.device)
    modes, spec, failure = [], None, None
    try:
        with _isolated_rng(device=trainer.device), torch.no_grad():
            spec = _specification(trainer, bundle, completed_update, split_sha256)
            model = trainer.unwrap_model()
            modes = [(part, part.training) for part in model.modules()]
            try:
                model.eval()
                _accumulate(trainer, model, spec, packed, rank, world, split_sha256)
                _require(torch.isfinite(packed).all().item(), "Nonfinite local diagnostic accumulators")
            finally:
                for part, training in modes:
                    part.training = training
    except Exception as error:
        failure = error
        packed[-1, 0] = 1
    # Rank-local errors still join this one collective, instead of stranding
    # healthy ranks after unequal-length local evaluation loops.
    _reduce_sum(packed, world)
    if packed[-1, 0].item():
        if world == 1 and failure is not None:
            raise failure
        detail = f"{type(failure).__name__}: {failure}" if failure is not None else "another rank failed"
        raise RuntimeError(f"Mobile diagnostic evaluation failed collectively on rank {rank}: {detail}") from failure
    return _report(packed, spec, completed_update, split_sha256, world)
