"""Reconstruct the original task loader stream by replaying CPU batches.

This deliberately fetches discarded samples, including their random transforms.
It requires a fresh copy of the *original* map-style loader: identical dataset,
sampler seed, rank/sharding, worker count, start method and prefetch settings.
It does not make arbitrary dataset side effects or unseeded RNG reproducible.

Keep ``trainer.train_dataloader = manager.dataloader`` and iterate the iterable
returned by ``for_epoch`` separately. Accelerator's active loader is the original
prepared loader, not the continuation iterable. Do not prepare either a second
time, iterate the original loader separately, or combine this with index skipping.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset


@contextmanager
def preserve_cpu_rng_state():
    """Protect model-side RNG without reverting loader/worker generators.

    No CUDA API is called here. Production may supply ``extra_rng_context`` to
    protect its already initialized CUDA RNG in addition to these CPU states.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


class _EpochContinuation:
    """Own one existing iterator, retaining its prefetched/lookahead batch."""

    def __init__(self, iterator, loader, remaining):
        self._iterator = iterator
        self.active_dataloader = loader
        self._remaining = remaining
        self.started = False
        self.finished = False

    def __len__(self):
        return self._remaining

    def __iter__(self):
        if self.started:
            raise RuntimeError("An epoch continuation may only be iterated once")
        self.started = True
        yield from self._iterator
        self.finished = True


class ReplayResumedMapDataLoader:
    """Replay all consumed microbatches once, then continue the same iterator.

    ``completed_microbatches`` counts batches actually returned to this rank,
    excluding DataLoader and Accelerator lookahead. For the task's accumulation
    of one this is the checkpoint's completed optimizer-update count. The caller
    restores model/optimizer/scheduler/process RNG before calling ``for_epoch``.

    At an exact epoch boundary the preceding iterator is fully exhausted before
    starting the next epoch, matching an uninterrupted outer training loop. With
    persistent workers this retains their advancing RNG states across epochs.

    Persistent multiprocessing workers are required: their independently seeded
    random streams can be reconstructed from epoch zero. A zero-worker loader
    would instead consume the checkpoint-restored main process RNG, which is not
    the initial RNG state and cannot reproduce earlier random transforms.

    ``extra_rng_context`` is an optional zero-argument context-manager factory;
    use it for CUDA RNG protection without importing/initializing CUDA here.
    ``progress(done, total)`` is optional and runs inside the RNG protection.
    """

    # Production must retain the original DataLoaderShard for Accelerator's
    # active-loader/accumulation bookkeeping while iterating our continuation.
    keep_prepared_loader = True

    def __init__(self, dataloader, *, completed_microbatches,
                 extra_rng_context=None, progress=None):
        if isinstance(dataloader.dataset, IterableDataset):
            raise TypeError("Task replay requires a map-style dataset")
        if getattr(dataloader, "use_stateful_dataloader", False):
            raise ValueError("Do not replay a loader with a restored stateful cursor")
        if type(completed_microbatches) is not int or completed_microbatches < 0:
            raise ValueError("completed_microbatches must be a nonnegative integer")
        device = getattr(dataloader, "device", None)
        if device is not None and torch.device(device).type != "cpu":
            raise ValueError("CPU replay requires device_placement=False")
        if getattr(dataloader, "iteration", 0) != 0 or getattr(dataloader, "_iterator", None) is not None:
            raise ValueError("Replay requires a fresh, uniterated loader")
        if not isinstance(getattr(dataloader, "generator", None), torch.Generator):
            raise ValueError("Replay requires the original independently seeded DataLoader generator")
        if not getattr(dataloader, "num_workers", 0) or not getattr(dataloader, "persistent_workers", False):
            raise ValueError("Task replay requires the original persistent multiprocessing workers")
        self.epoch_length = len(dataloader)
        if self.epoch_length <= 0:
            raise ValueError("Replay requires at least one actual batch per epoch")
        self.dataloader = dataloader
        self.completed_microbatches = completed_microbatches
        self.resume_epoch, self.resume_offset = divmod(completed_microbatches, self.epoch_length)
        self.extra_rng_context = extra_rng_context
        self.progress = progress
        self.replayed_microbatches = 0
        self._last_epoch = None
        self._continuation = None
        self._failed = False

    def _iterator_for_epoch(self, epoch):
        # DataLoaderShard.__iter__ calls set_epoch(self.iteration). Set iteration
        # first, while preserving the original sampler and generator objects.
        if hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)
        return iter(self.dataloader)

    def _record_replayed_batch(self):
        self.replayed_microbatches += 1
        if self.progress is not None:
            self.progress(self.replayed_microbatches, self.completed_microbatches)

    def _replay(self):
        with preserve_cpu_rng_state():
            with self.extra_rng_context() if self.extra_rng_context is not None else nullcontext():
                for epoch in range(self.resume_epoch):
                    count = 0
                    for _ in self._iterator_for_epoch(epoch):
                        count += 1
                        self._record_replayed_batch()
                    if count != self.epoch_length:
                        raise RuntimeError("Actual epoch length changed during replay")
                iterator = self._iterator_for_epoch(self.resume_epoch)
                for _ in range(self.resume_offset):
                    next(iterator)
                    self._record_replayed_batch()
                # Keep this exact iterator: restarting it would discard the
                # Accelerator lookahead and reseed/reassign worker batches.
                return iterator

    def for_epoch(self, epoch):
        """Return ``(single_use_epoch_iterable, original_local_step_start)``."""
        if self._failed:
            raise RuntimeError("A failed replay cannot be reused; recreate the loader")
        if type(epoch) is not int or epoch < self.resume_epoch:
            raise ValueError("Cannot iterate before the checkpoint's actual epoch")
        expected = self.resume_epoch if self._last_epoch is None else self._last_epoch + 1
        if epoch != expected:
            raise ValueError("Replay epochs must be requested consecutively")
        if self._continuation is not None and not self._continuation.finished:
            raise RuntimeError("Exhaust the current epoch continuation before advancing")
        try:
            iterator = self._replay() if self._last_epoch is None else self._iterator_for_epoch(epoch)
        except BaseException:
            self._failed = True
            # An exception interrupts DataLoaderShard before its ordinary end()
            # cleanup. Do not leave a dead loader active in Accelerator state.
            state = getattr(self.dataloader, "gradient_state", None)
            if state is not None and state.active_dataloader is self.dataloader:
                self.dataloader.end()
            raise
        offset = self.resume_offset if self._last_epoch is None else 0
        self._continuation = _EpochContinuation(iterator, self.dataloader, self.epoch_length - offset)
        self._last_epoch = epoch
        return self._continuation, offset
