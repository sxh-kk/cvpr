"""Bounded CPU diagnostics and collective finite checks for DDP training.

Healthy steps keep only the current accumulation window in memory. Numerical
failures are agreed across ranks before backward or gradient clipping. Exception
dumps use CPU material captured before forward and never query CUDA again.

A native CUDA assertion can abort the NCCL watchdog before Python receives an
exception. Such SIGABRT exits cannot be caught here. With CUDA_LAUNCH_BLOCKING=1,
a locally caught CUDA error can still write its already captured CPU diagnostic;
this does not guarantee a dump from other ranks or repair the native assertion.
For a bounded native-crash investigation, PSI0_DIAGNOSTIC_BATCH_RANGE=start:end
also atomically replaces one rolling CPU batch/RNG file per rank before forward.
Only that inclusive, zero-based optimizer-step range performs healthy-step I/O.
PSI0_ROLLING_DIAGNOSTICS_DIR enables full-run capture in a separate absolute
directory (prefer a run-specific tmpfs directory). Two files per rank retain the
current and previous update; these survive process aborts, not host restarts.
"""

from collections import defaultdict, deque
from collections.abc import Mapping
from datetime import timedelta
import os
from pathlib import Path
import random
import re

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist


def _cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, Mapping):
        return {key: _cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, Image.Image):
        return value.copy()
    if value is None or isinstance(value, (str, bytes, bool, int, float, np.generic)):
        return value
    # Convert unknown metadata while CUDA is still healthy; never pickle an
    # opaque object that might retain GPU tensors or an autograd graph.
    return {"type": type(value).__qualname__, "repr": repr(value)}


class TrainingSafetyGuard:
    def __init__(self, diagnostics_dir, *, device, rank=0, world_size=1, accumulation_steps=1):
        self.diagnostics_dir = Path(diagnostics_dir)
        self.device = torch.device(device)
        self.rank, self.world_size = int(rank), int(world_size)
        self.snapshots = deque(maxlen=max(1, int(accumulation_steps)))
        self.context = {}
        self.stage = "capture"
        self.last_dump = None
        window = os.getenv("PSI0_DIAGNOSTIC_BATCH_RANGE")
        self.persist_range = None
        rolling = os.getenv("PSI0_ROLLING_DIAGNOSTICS_DIR")
        self.rolling_dir = Path(rolling) if rolling else None
        if self.rolling_dir is not None and not self.rolling_dir.is_absolute():
            raise ValueError("PSI0_ROLLING_DIAGNOSTICS_DIR must be an absolute, run-specific directory")
        if window:
            if not re.fullmatch(r"[0-9]+:[0-9]+", window):
                raise ValueError("PSI0_DIAGNOSTIC_BATCH_RANGE must be start:end")
            start, end = map(int, window.split(":"))
            if start > end:
                raise ValueError("PSI0_DIAGNOSTIC_BATCH_RANGE start must not exceed end")
            self.persist_range = (start, end)
        # Create the CPU coordinator while CUDA is healthy. It is used only
        # after a collective finite failure, so torchrun cannot kill slower
        # writers when the first rank finishes its diagnostic and raises.
        self._diagnostic_group = (dist.new_group(backend="gloo", timeout=timedelta(seconds=30))
                                  if self.world_size > 1 else None)

    def capture(self, batch, *, global_step, local_step):
        self.stage = "capture"
        self.last_dump = None
        if self.context.get("global_step") != global_step:
            self.snapshots.clear()
        self.context = {"global_step": int(global_step), "local_step": int(local_step), "rank": self.rank}
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": (torch.cuda.get_rng_state(self.device).clone()
                           if self.device.type == "cuda" else None),
        }
        self.snapshots.append({**self.context, "batch": _cpu_snapshot(batch), "rng": rng})
        if self.persist_range is not None and self.persist_range[0] <= global_step <= self.persist_range[1]:
            self._persist_snapshot(self.diagnostics_dir / f"latest_batch_rank_{self.rank:03d}.pt")
        if self.rolling_dir is not None:
            # Keep the previous update intact while replacing the current one.
            # Unlike the bounded investigation window, this remains active at
            # every step; tmpfs avoids writing full RGB batches to shared disk.
            self._persist_snapshot(self.rolling_dir / (
                f"latest_batch_rank_{self.rank:03d}_slot_{int(global_step) % 2}.pt"))

    def _persist_snapshot(self, path):
        # Persist before launching the next model kernels. A process-level
        # abort cannot run dump(), but this file survives independently.
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        torch.save({"schema_version": 1, **self.context, "stage": "pre_forward",
                    "microbatches": list(self.snapshots)}, temporary)
        os.replace(temporary, path)

    def dump(self, reason, **details):
        """Write only cached CPU material, including when CUDA is poisoned."""
        if self.last_dump is not None:
            return self.last_dump
        self.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        stage = re.sub(r"[^a-zA-Z0-9_-]", "_", self.stage)
        name = (f"step_{self.context.get('global_step', -1):08d}_"
                f"local_{self.context.get('local_step', -1):08d}_rank_{self.rank:03d}_{stage}.pt")
        path = self.diagnostics_dir / name
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        torch.save({"schema_version": 1, **self.context, "stage": self.stage, "reason": reason,
                    "details": details, "microbatches": list(self.snapshots)}, temporary)
        os.replace(temporary, path)
        self.last_dump = path
        return path

    def _agree_finite(self, finite, **details):
        # Each rank owns one slot; one collective identifies every failing rank
        # without individual scalar transfers for parameters or optimizer groups.
        flags = torch.ones(self.world_size, dtype=torch.int32, device=self.device)
        flags[self.rank] = finite.to(device=self.device, dtype=torch.int32)
        if self.world_size > 1:
            dist.all_reduce(flags, op=dist.ReduceOp.MIN)
        failed = [rank for rank, value in enumerate(flags.detach().cpu().tolist()) if not value]
        if failed:
            try:
                path = self.dump("nonfinite", nonfinite_ranks=failed, **details)
            finally:
                if self._diagnostic_group is not None:
                    # No CUDA work after detecting the failure, including when
                    # a rank's local file write fails. This barrier is bounded.
                    dist.barrier(group=self._diagnostic_group)
            raise FloatingPointError(f"Non-finite {self.stage} on ranks {failed}; CPU diagnostic: {path}")

    def check_loss(self, loss):
        self.stage = "loss"
        self._agree_finite(torch.isfinite(loss.detach()).all())

    def check_gradients(self, parameters):
        self.stage = "gradients"
        groups = defaultdict(list)
        for parameter in parameters:
            if parameter.grad is not None:
                gradient = parameter.grad.detach()
                if gradient.is_sparse:
                    gradient = gradient.coalesce().values()
                groups[(gradient.device, gradient.dtype)].append(gradient)
        checks = []
        for gradients in groups.values():
            # Infinity norms test every element without squared-sum overflow;
            # this does not replace or change the trainer's L2 clipping budget.
            norms = torch._foreach_norm(gradients, float("inf"))
            checks.append(torch.isfinite(torch.stack(norms)).all().to(self.device))
        finite = torch.stack(checks).all() if checks else torch.ones((), dtype=torch.bool, device=self.device)
        self._agree_finite(finite, gradient_tensors_checked=sum(map(len, groups.values())))
