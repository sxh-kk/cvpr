"""Resume map-style batches without fetching the already-consumed samples.

The sampler order is retained; worker augmentation/prefetch RNG is not saved by
the ordinary DataLoader and is therefore not reconstructed by this helper.
"""
from __future__ import annotations

from accelerate import skip_first_batches
from accelerate.data_loader import DataLoaderShard, SkipBatchSampler
from torch.utils.data import BatchSampler, IterableDataset


class ResumedMapDataLoader:
    """Keep one prepared loader, with an offset only in the resumed epoch.

    Reusing the wrapper on subsequent epochs also preserves persistent workers
    and avoids drawing an extra DataLoader base seed when removing the offset.
    ``for_epoch`` returns the loader and the original microbatch number at which
    enumeration should start. It never advances Accelerator's accumulation step.
    """

    def __init__(self, dataloader, *, skipped_batches: int, resume_epoch: int):
        if isinstance(dataloader.dataset, IterableDataset):
            raise TypeError("Sampler-level resume requires a map-style dataset")
        if getattr(dataloader, "use_stateful_dataloader", False):
            raise ValueError("Do not combine batch skipping with a stateful DataLoader cursor")
        if not isinstance(skipped_batches, int) or skipped_batches < 0:
            raise ValueError("skipped_batches must be a nonnegative integer")
        if skipped_batches >= len(dataloader) and skipped_batches:
            raise ValueError("Resume offset must be inside the selected epoch")
        if not isinstance(resume_epoch, int) or resume_epoch < 0:
            raise ValueError("resume_epoch must be a nonnegative integer")

        self.original_dataloader = dataloader
        self.resume_epoch = resume_epoch
        self.skipped_batches = skipped_batches
        self.dataloader = dataloader
        self._skip_sampler = None
        if skipped_batches:
            self.dataloader = skip_first_batches(dataloader, skipped_batches)
            self._skip_sampler = (
                self.dataloader.sampler
                if isinstance(self.dataloader.sampler, BatchSampler)
                else self.dataloader.batch_sampler
            )
            if not isinstance(self._skip_sampler, SkipBatchSampler):
                raise TypeError("Accelerate did not install a sampler-level batch offset")

            # Accelerate 1.7's skip wrapper omits the shard's metadata. Retain
            # total_batch_size/remainder and device-transfer behavior, without
            # changing the already-sharded sampler itself.
            sampler = self._skip_sampler.batch_sampler
            for name in ("batch_size", "drop_last", "split_batches", "num_processes"):
                if hasattr(sampler, name):
                    setattr(self._skip_sampler, name, getattr(sampler, name))
            if isinstance(dataloader, DataLoaderShard):
                self.dataloader._drop_last = dataloader._drop_last
                self.dataloader._non_blocking = dataloader._non_blocking

    def for_epoch(self, epoch: int):
        if epoch < self.resume_epoch:
            raise ValueError("Cannot resume an epoch before the checkpoint data position")
        offset = self.skipped_batches if epoch == self.resume_epoch else 0
        if self._skip_sampler is not None:
            self._skip_sampler.skip_batches = offset

        # The skip wrapper adds a level of BatchSampler nesting which can hide
        # DistributedSampler/SeedableRandomSampler.set_epoch from Accelerate.
        # Set it through the original prepared loader before iterating the same
        # shared sampler through the wrapper.
        if hasattr(self.original_dataloader, "set_epoch"):
            self.original_dataloader.set_epoch(epoch)
        if self.dataloader is not self.original_dataloader and hasattr(self.dataloader, "set_epoch"):
            self.dataloader.set_epoch(epoch)
        return self.dataloader, offset
