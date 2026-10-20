"""Sampler-level resume using real Accelerate 1.7 CPU/Gloo data loaders."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, IterableDataset

from psi.trainers.resume_data import ResumedMapDataLoader


class _CountedRows(Dataset):
    def __init__(self, length=64):
        self.length = length
        # A persistent spawn worker must report actual __getitem__ calls too.
        self.hits = torch.zeros(length, dtype=torch.int64).share_memory_()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        self.hits[index] += 1
        return {"row": index, "input": torch.tensor([index / 64, 1.0])}


def _copy_state(model, optimizer, scheduler, accelerator):
    return deepcopy({"model": accelerator.unwrap_model(model).state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "accumulation_step": accelerator.step})


def _run_case(*, workers, resumed, keep_cpu, checkpoint=None, seedable=False, length=64):
    # Production seeds every rank identically before loader/model construction;
    # SeedableRandomSampler also reads this seed when no data_seed is specified.
    torch.manual_seed(292285)
    accelerator = Accelerator(
        cpu=True, mixed_precision="no", gradient_accumulation_steps=2,
        dataloader_config=DataLoaderConfiguration(dispatch_batches=False, use_seedable_sampler=seedable),
    )
    dataset = _CountedRows(length)
    options = dict(num_workers=workers, persistent_workers=workers > 0)
    if workers:
        options["multiprocessing_context"] = "spawn"
    raw_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True,
                            generator=torch.Generator().manual_seed(292285), **options)
    loader = accelerator.prepare_data_loader(raw_loader, device_placement=not keep_cpu)
    loader._non_blocking = True  # Also retain the prepared loader's transfer policy.
    model = torch.nn.Linear(2, 1, bias=False)
    torch.nn.init.constant_(model.weight, 0.125)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (1 + step / 10))
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    if resumed:
        accelerator.unwrap_model(model).load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        accelerator.step = checkpoint["accumulation_step"]
        assert accelerator.step == 4
    manager = ResumedMapDataLoader(loader, skipped_batches=4, resume_epoch=0) if resumed else None
    trainer = SimpleNamespace(train_dataloader=loader)
    epochs, local_ids, sync_flags, updates = [], [], [], 2 if resumed else 0
    saved = None
    first_wrapper = None
    batches_per_rank = length // 8
    for epoch in (0, 1):
        if manager is not None:
            trainer.train_dataloader, start = manager.for_epoch(epoch)
        else:
            trainer.train_dataloader.set_epoch(epoch)
            start = 0
        if resumed:
            assert len(trainer.train_dataloader) == (batches_per_rank - 4 if epoch == 0 else batches_per_rank)
            assert trainer.train_dataloader._drop_last is True
            assert trainer.train_dataloader._non_blocking is True
            assert trainer.train_dataloader.total_batch_size == 8
            assert trainer.train_dataloader.device == loader.device
            if epoch == 0:
                first_wrapper = trainer.train_dataloader
            else:
                # Reuse persistent workers; clear only the sampler offset.
                assert trainer.train_dataloader is first_wrapper
        rows, ids, flags = [], [], []
        for local, batch in enumerate(trainer.train_dataloader, start=start):
            assert accelerator.gradient_state.active_dataloader is trainer.train_dataloader
            rows.extend(batch["row"].tolist())
            ids.append(local)
            with accelerator.accumulate(model):
                loss = (model(batch["input"]) - 0.1).square().mean()
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                flags.append(accelerator.sync_gradients)
                if accelerator.sync_gradients:
                    updates += 1
                    if not resumed and epoch == 0 and local == 3:
                        saved = _copy_state(model, optimizer, scheduler, accelerator)
        epochs.append(rows)
        local_ids.append(ids)
        sync_flags.append(flags)
        if resumed and epoch == 0:
            # No skipped sample was decoded, including by prefetched workers.
            torch.testing.assert_close(dataset.hits, torch.bincount(torch.tensor(rows), minlength=length))
        assert accelerator.gradient_state.active_dataloader is None
    final = _copy_state(model, optimizer, scheduler, accelerator)
    assert updates == 2 * ((batches_per_rank + 1) // 2)
    assert not torch.cuda.is_initialized()
    result = {"epochs": epochs, "local_ids": local_ids, "sync_flags": sync_flags,
              "checkpoint": saved, "final": final}
    # Promptly close persistent CPU workers instead of retaining one pool per case.
    iterator = getattr(trainer.train_dataloader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()
    accelerator.free_memory()
    return result


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_tree_equal(a, b)
    else:
        assert left == right


def _gloo_worker(rank, directory, port, workers, length=64):
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE="2", LOCAL_WORLD_SIZE="2",
                      MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    root = Path(directory)
    dist.init_process_group("gloo", init_method=(root / "rendezvous").as_uri(), rank=rank,
                            world_size=2, timeout=timedelta(seconds=90))
    try:
        # Odd-tail acceptance needs only the production RandomSampler path.
        for seedable in ((False, True) if length == 64 else (False,)):
            baseline = _run_case(workers=workers, resumed=False, keep_cpu=not bool(workers),
                                 seedable=seedable, length=length)
            resumed = _run_case(workers=workers, resumed=True, keep_cpu=not bool(workers),
                                checkpoint=baseline["checkpoint"], seedable=seedable, length=length)
            assert resumed["epochs"][0] == baseline["epochs"][0][16:]
            assert resumed["epochs"][1] == baseline["epochs"][1]
            batches_per_rank = length // 8
            assert resumed["local_ids"] == [list(range(4, batches_per_rank)), list(range(batches_per_rank))]
            assert resumed["sync_flags"] == [
                [local % 2 == 1 or local == batches_per_rank - 1 for local in range(4, batches_per_rank)],
                [local % 2 == 1 or local == batches_per_rank - 1 for local in range(batches_per_rank)],
            ]
            _assert_tree_equal(baseline["final"], resumed["final"])
            all_rows = [None, None]
            dist.all_gather_object(all_rows, baseline["epochs"])
            for epoch in (0, 1):
                assert set(all_rows[0][epoch]).isdisjoint(all_rows[1][epoch])
                assert sorted(all_rows[0][epoch] + all_rows[1][epoch]) == list(range(length))
        (root / f"rank{rank}.json").write_text(json.dumps({"passed": True, "workers": workers}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("workers", [0, 1])
def test_two_rank_resume_skips_decode_preserves_shards_accumulation_and_next_epoch(tmp_path, workers):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    mp.spawn(_gloo_worker, args=(str(tmp_path), port, workers), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"rank{rank}.json").read_text())["passed"] for rank in (0, 1))


def test_two_rank_odd_tail_forces_sync_then_next_epoch_restarts_accumulation_without_skip(tmp_path):
    # 72 / batch4 / world2 = 9 local microbatches. Resume leaves five,
    # so the last single microbatch must synchronize before the next epoch.
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    mp.spawn(_gloo_worker, args=(str(tmp_path), port, 0, 72), nprocs=2, join=True)
    assert all(json.loads((tmp_path / f"rank{rank}.json").read_text())["passed"] for rank in (0, 1))


class _Stream(IterableDataset):
    def __iter__(self):
        yield from range(4)


def test_sampler_resume_rejects_iterable_and_invalid_offsets():
    with pytest.raises(TypeError, match="map-style"):
        ResumedMapDataLoader(DataLoader(_Stream()), skipped_batches=1, resume_epoch=0)
    loader = DataLoader(list(range(8)), batch_size=2)
    for count in (-1, 4, 5, 0.5):
        with pytest.raises(ValueError):
            ResumedMapDataLoader(loader, skipped_batches=count, resume_epoch=0)


def test_zero_offset_keeps_original_loader_and_original_local_step():
    loader = DataLoader(list(range(8)), batch_size=2)
    resumed = ResumedMapDataLoader(loader, skipped_batches=0, resume_epoch=2)
    assert resumed.for_epoch(2) == (loader, 0)
    assert resumed.for_epoch(3) == (loader, 0)
    with pytest.raises(ValueError, match="before"):
        resumed.for_epoch(1)
