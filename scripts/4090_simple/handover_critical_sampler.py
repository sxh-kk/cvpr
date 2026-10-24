"""Bind critical-window sampling to the production real-history Dataset index.

Install only for a NEW run, after ``create_dataloaders`` and before Accelerator
prepares/shards the loader.  Source RGB/actions, loss weights, validation and
normalization do not change.  All ranks draw the same epoch-seeded global stream;
Accelerator retains sole responsibility for distributed batch sharding.

The builder's original physical-record receipts remain provenance.  Relocated
A100 consumers need only the immutable sidecar and matching packed metadata, not
identical absolute paths to the builder's original capture directories.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import hashlib
import inspect
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import BatchSampler, DataLoader, Sampler, Subset

CONTRACT = "cvpr-handover-critical-windows-v2"
HELDOUT_VAL = frozenset((39, 65, 80, 92, 94))
STAGES = frozenset(("lift", "handover", "placement"))


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def _int(value, name, *, minimum=0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite(value, name, *, positive=False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    value = float(value)
    if value < 0 or positive and value <= 0:
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return value


def _train_root(path) -> Path:
    root = Path(path).resolve(strict=True)
    if (root / "train/meta/episodes.jsonl").is_file():
        root = (root / "train").resolve(strict=True)
    if not (root / "meta/episodes.jsonl").is_file():
        raise ValueError("Dataset root must identify the packed TRAIN metadata")
    return root


def weights_digest(rows) -> str:
    payload = [[r["episode_index"], r["current_row"], r["weight"]] for r in rows]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CriticalSidecar:
    directory: Path
    metadata: dict[str, Any]
    rows: tuple[dict[str, Any], ...]


def _verify_source(sidecar: CriticalSidecar, dataset_root) -> Path:
    train = _train_root(dataset_root)
    meta = sidecar.metadata
    if _sha(train.parent / "PACK_READY.json") != meta["source_pack_ready_sha256"]:
        raise ValueError("Sidecar source PACK_READY bytes differ from this dataset")
    episodes_path = train / "meta/episodes.jsonl"
    if _sha(episodes_path) != meta["source_metadata_sha256"]:
        raise ValueError("Sidecar source episodes metadata differs from this dataset")
    episodes = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    expected = {}
    sources = {}
    for e in episodes:
        ep = _int(e["episode_index"], "source episode_index")
        source = _int(e["source_episode_index"], "source ancestor")
        if ep in sources or e.get("split") != "train" or source in HELDOUT_VAL:
            raise ValueError("Source must contain unique TRAIN episodes without heldout ancestors")
        sources[ep] = source
        length = _int(e["length"], "source episode length", minimum=1)
        eligible = e["eligible_current_rows"]
        for value in eligible:
            row = _int(value, "eligible current_row", minimum=30)
            if row + 29 >= length or (ep, row) in expected:
                raise ValueError("Source eligible rows repeat or lack complete real future/history")
            expected[(ep, row)] = source
    actual = {(r["episode_index"], r["current_row"]): r["source_episode_index"] for r in sidecar.rows}
    if actual != expected:
        raise ValueError("Sidecar does not cover exactly the authoritative TRAIN eligible rows/ancestors")
    return train


def load_sidecar(directory, *, dataset_root=None, expected_window_count=32794) -> CriticalSidecar:
    """Validate immutable weights and optionally relocated packed-source identity.

    ``expected_window_count`` may be changed explicitly for a separately admitted
    release; the existing quality100 release defaults to exactly32,794 windows.
    No absolute raw-physics path in the build receipt is opened here.
    """
    root = Path(directory).resolve(strict=True)
    metadata = json.loads((root / "metadata.json").read_text())
    if metadata.get("contract") != CONTRACT:
        raise ValueError("Unrecognized critical-window sidecar contract")
    rows_path = root / "rows.jsonl"
    if _sha(rows_path) != metadata.get("rows_sha256"):
        raise ValueError("Critical-window rows SHA differs from metadata")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    if not rows or _int(metadata.get("window_count"), "window_count", minimum=1) != len(rows):
        raise ValueError("Critical-window count differs from metadata")
    if expected_window_count is not None and len(rows) != _int(expected_window_count, "expected_window_count", minimum=1):
        raise ValueError("Critical-window count differs from the explicitly expected release")
    keys = set()
    for r in rows:
        key = (_int(r["episode_index"], "episode_index"), _int(r["current_row"], "current_row", minimum=30))
        source = _int(r["source_episode_index"], "source_episode_index")
        if key in keys or source in HELDOUT_VAL:
            raise ValueError("Repeated critical-window key or heldout ancestor")
        keys.add(key)
        stages = r["critical_stages"]
        if (not isinstance(stages, list) or len(stages) != len(set(stages))
                or any(s not in STAGES for s in stages)):
            raise ValueError("Critical stages must be unique known TRAIN stage tags")
        weight = _finite(r["weight"], "weight", positive=True)
        base = _finite(r["base_weight"], "base_weight", positive=True)
        focus = r["focus_weights"]
        if not isinstance(focus, dict) or set(focus) - STAGES:
            raise ValueError("Invalid focus weight stages")
        for stage, value in focus.items():
            if _finite(value, "focus_weight") > 0 and stage not in stages:
                raise ValueError("Nonzero focus weight is not backed by a critical stage tag")
        if not math.isclose(weight, base + sum(focus.values()), rel_tol=1e-10, abs_tol=1e-13):
            raise ValueError("Weight differs from its full-trajectory plus focus components")
        _finite(r["sim_time_s"], "sim_time_s")
        if not isinstance(r["observed_stage"], str) or not r["observed_stage"]:
            raise ValueError("Observed stage must identify a past/current physical observation")
    if not math.isclose(math.fsum(r["weight"] for r in rows), 1., rel_tol=0, abs_tol=1e-10):
        raise ValueError("Critical probabilities must sum to one")
    if weights_digest(rows) != metadata.get("weights_digest"):
        raise ValueError("Critical probability digest differs from metadata")
    for name in ("source_pack_ready_sha256", "source_metadata_sha256"):
        value = metadata.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Critical sidecar requires SHA-bound packed-source metadata")
    sidecar = CriticalSidecar(root, metadata, tuple(rows))
    if dataset_root is not None:
        _verify_source(sidecar, dataset_root)
    return sidecar


def _production_base(subset):
    if not isinstance(subset, Subset):
        raise ValueError("Expected the admitted production real-history Subset")
    resolved = list(range(len(subset)))
    current = subset
    while isinstance(current, Subset):
        resolved = [_int(current.indices[i], "Subset index") for i in resolved]
        if any(i >= len(current.dataset) for i in resolved):
            raise ValueError("Subset index escapes its underlying Dataset")
        current = current.dataset
    if len(set(resolved)) != len(resolved):
        raise ValueError("Production Subset repeats an underlying real-history row")
    history = getattr(getattr(current, "raw_dataset", None), "motion_history", None)
    if history is None or len(history.source_episode_ranges) != 1 or len(history.pack_info) != 1:
        raise ValueError("Require the single-pack production MotionHistoryIndex")
    if len(history.original_indices) != len(current):
        raise ValueError("Production history mapping length differs from its Dataset")
    return current, resolved, history


def weights_for_subset(subset, sidecar, *, dataset_root=None, expected_window_count=32794):
    """Return probabilities in Dataset/Subset order, never Parquet row order."""
    if not isinstance(sidecar, CriticalSidecar):
        sidecar = load_sidecar(sidecar, expected_window_count=expected_window_count)
    _, resolved, history = _production_base(subset)
    root = dataset_root if dataset_root is not None else history.pack_info[0][1]
    _verify_source(sidecar, root)
    ranges = sorted((int(start), int(stop), int(ep)) for ep, (start, stop) in history.source_episode_ranges[0].items())
    if (not ranges or any(start < 0 or stop <= start for start, stop, _ in ranges)
            or any(a[1] > b[0] for a, b in zip(ranges, ranges[1:]))):
        raise ValueError("Invalid or overlapping production source episode ranges")
    starts = [r[0] for r in ranges]
    by_key = {(r["episode_index"], r["current_row"]): r for r in sidecar.rows}
    actual_keys, weights = [], []
    for index in resolved:
        original = _int(history.original_indices[index], "history original index")
        j = bisect_right(starts, original) - 1
        if j < 0 or original >= ranges[j][1]:
            raise ValueError("History original row is outside every bound source episode")
        start, _, ep = ranges[j]
        key = (ep, original - start)
        if key not in by_key:
            raise ValueError("Production Dataset includes a non-admitted critical-window row")
        actual_keys.append(key)
        weights.append(by_key[key]["weight"])
    if len(set(actual_keys)) != len(actual_keys) or set(actual_keys) != set(by_key):
        raise ValueError("Production Subset must cover every admitted window exactly once")
    if expected_window_count is not None and len(weights) != expected_window_count:
        raise ValueError("Production sampling count differs from the expected release")
    return torch.tensor(weights, dtype=torch.float64, device="cpu")


class EpochWeightedSampler(Sampler[int]):
    """Identical global weighted stream on every rank; no internal rank slicing."""
    def __init__(self, weights, *, seed, num_samples=None):
        self.weights = torch.as_tensor(weights, dtype=torch.float64, device="cpu").clone()
        if (self.weights.ndim != 1 or not self.weights.numel() or not torch.isfinite(self.weights).all()
                or not (self.weights > 0).all()):
            raise ValueError("Sampler weights must be a positive finite vector")
        self.weights /= self.weights.sum()
        self.num_samples = _int(len(self.weights) if num_samples is None else num_samples, "num_samples", minimum=1)
        self.seed = _int(seed, "seed")
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = _int(epoch, "epoch")

    def __iter__(self):
        generator = torch.Generator(device="cpu")
        generator.manual_seed((self.seed + self.epoch) % (2**63))
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True, generator=generator).tolist())

    def __len__(self):
        return self.num_samples


def install_critical_sampler(trainer, sidecar_dir, *, seed=None, dataset_root=None, expected_window_count=32794):
    """Replace only the unprepared training loader's sampling policy.

    Call from a new trainer's ``create_dataloaders`` override, immediately after
    ``super().create_dataloaders``.  Do not call on an active/prepared loader.
    Returns a serializable receipt; the caller may save it beside new run config.
    """
    old = trainer.train_dataloader
    train_config = getattr(trainer.cfg, "train", None)
    resume = getattr(train_config, "resume_from_checkpoint", None)
    if (type(old) is not DataLoader or old.batch_size is None or type(old.batch_sampler) is not BatchSampler
            or getattr(old, "_iterator", None) is not None or getattr(trainer, "completed_updates", 0) > 0
            or resume is not None or hasattr(trainer, "critical_sampling_receipt")):
        raise ValueError("Install critical sampling once on a fresh plain loader before Accelerator.prepare")
    sidecar = load_sidecar(sidecar_dir, expected_window_count=expected_window_count)
    weights = weights_for_subset(old.dataset, sidecar, dataset_root=dataset_root,
                                 expected_window_count=expected_window_count)
    if seed is None:
        seed = getattr(trainer.cfg, "seed", None)
        seed = 42 if seed is None else seed
    sampler = EpochWeightedSampler(weights, seed=seed)
    options = {"batch_size": old.batch_size, "sampler": sampler, "collate_fn": old.collate_fn,
               "num_workers": old.num_workers, "pin_memory": old.pin_memory,
               "drop_last": old.drop_last, "timeout": old.timeout,
               "worker_init_fn": old.worker_init_fn, "generator": old.generator,
               "persistent_workers": old.persistent_workers}
    if old.num_workers:
        options.update(prefetch_factor=old.prefetch_factor, multiprocessing_context=old.multiprocessing_context)
    for name in ("pin_memory_device", "in_order"):
        if name in inspect.signature(DataLoader).parameters and hasattr(old, name):
            options[name] = getattr(old, name)
    validation = trainer.val_dataloader
    trainer.train_dataloader = DataLoader(old.dataset, **options)
    trainer.train_sampler = sampler
    assert trainer.val_dataloader is validation
    receipt = {"contract": "cvpr-handover-critical-sampler-adapter-v1", "sidecar_directory": str(sidecar.directory),
               "rows_sha256": sidecar.metadata["rows_sha256"], "weights_digest": sidecar.metadata["weights_digest"],
               "window_count": len(weights), "seed": sampler.seed, "replacement": True,
               "distributed_sharding": "Accelerator only; identical global epoch stream on all ranks",
               "validation_loader_changed": False, "rgb_actions_normalization_changed": False,
               "stage_labels_are_model_inputs": False}
    trainer.critical_sampling_receipt = receipt
    return receipt
