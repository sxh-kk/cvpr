"""CPU acceptance of RandomSampler and four persistent spawn-worker RNG replay."""
from contextlib import contextmanager
import importlib.util
from pathlib import Path
import random
import sys
import unittest

from accelerate.data_loader import prepare_data_loader
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


_PATH = Path(__file__).resolve().parents[1] / "replay_task_data.py"
_SPEC = importlib.util.spec_from_file_location("replay_task_data", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
ReplayResumedMapDataLoader = _MODULE.ReplayResumedMapDataLoader


class _RandomRows(Dataset):
    def __len__(self):
        return 105

    def __getitem__(self, index):
        # Variable consumption catches accidental worker reassignment/reseeding.
        for _ in range(1 + index % 3):
            python = random.random()
        numpy = np.random.random(1 + index % 4)[-1]
        tensor = torch.rand(1 + index % 5)[-1].item()
        return {"index": index, "augmentation": torch.tensor([python, numpy, tensor], dtype=torch.float64)}


def _seed_worker(worker):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _loader(*, rank=1, workers=4):
    options = dict(num_workers=workers, persistent_workers=workers > 0)
    if workers:
        options.update(multiprocessing_context="spawn", prefetch_factor=4)
    raw = DataLoader(_RandomRows(), batch_size=4, shuffle=True, drop_last=True,
                     generator=torch.Generator().manual_seed(42), worker_init_fn=_seed_worker, **options)
    # Real Accelerate sharding and lookahead, with CPU virtual ranks: no process
    # group or model/GPU work is required to verify each rank's input stream.
    return prepare_data_loader(raw, num_processes=2, process_index=rank,
                               put_on_device=False, rng_types=[])


def _close(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def _rng():
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _restore_rng(states):
    random.setstate(states[0])
    np.random.set_state(states[1])
    torch.set_rng_state(states[2])


def _assert_rng(left, right):
    assert left[0] == right[0]
    assert left[1][0] == right[1][0]
    np.testing.assert_array_equal(left[1][1], right[1][1])
    assert left[1][2:] == right[1][2:]
    assert torch.equal(left[2], right[2])


def _model_random():
    return random.random(), float(np.random.random()), float(torch.rand(()))


def _check_four_spawn_workers_replay(completed):
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    baseline = _loader()
    assert len(baseline) == 13  # floor(105 / (4 * 2)), not trainer's ceil14.
    records, checkpoint_rng = [], _rng() if completed == 0 else None
    checkpoint_generator = baseline.generator.get_state().clone() if completed == 0 else None
    try:
        for epoch in range(4):
            baseline.set_epoch(epoch)
            for local, batch in enumerate(baseline):
                assert baseline.gradient_state.active_dataloader is baseline
                records.append((epoch, local, batch, _model_random()))
                if len(records) == completed:
                    checkpoint_rng = _rng()
                    checkpoint_generator = baseline.generator.get_state().clone()
            assert baseline.gradient_state.active_dataloader is None
        final_generator = baseline.generator.get_state().clone()
    finally:
        _close(baseline)

    resumed = _loader()
    _restore_rng(checkpoint_rng)
    manager = ReplayResumedMapDataLoader(resumed, completed_microbatches=completed)
    assert manager.resume_epoch == completed // 13
    seen, progress = [], []
    manager.progress = lambda done, total: progress.append((done, total))
    try:
        for epoch in range(manager.resume_epoch, 4):
            before = _rng()
            continuation, start = manager.for_epoch(epoch)
            _assert_rng(_rng(), before)
            assert continuation.active_dataloader is resumed
            assert len(continuation) == 13 - start
            if epoch == manager.resume_epoch:
                assert torch.equal(resumed.generator.get_state(), checkpoint_generator)
            for local, batch in enumerate(continuation, start=start):
                assert resumed.gradient_state.active_dataloader is manager.dataloader
                seen.append((epoch, local, batch, _model_random()))
            assert resumed.gradient_state.active_dataloader is None
        assert manager.replayed_microbatches == completed
        assert progress == [(i, completed) for i in range(1, completed + 1)]
        assert torch.equal(resumed.generator.get_state(), final_generator)
    finally:
        _close(resumed)
    assert len(seen) == len(records) - completed
    for actual, expected in zip(seen, records[completed:]):
        assert actual[:2] == expected[:2]
        assert torch.equal(actual[2]["index"], expected[2]["index"])
        assert torch.equal(actual[2]["augmentation"], expected[2]["augmentation"])
        assert actual[3] == expected[3]
    assert not torch.cuda.is_initialized()


def _check_failure_rng(case):
    loader = _loader()
    entered = []

    @contextmanager
    def extra_rng_context():
        entered.append("enter")
        try:
            yield
        finally:
            entered.append("exit")

    def fail(done, total):
        _model_random()
        raise RuntimeError("callback failure")

    manager = ReplayResumedMapDataLoader(loader, completed_microbatches=2,
        extra_rng_context=extra_rng_context, progress=fail)
    before = _rng()
    with case.assertRaisesRegex(RuntimeError, "callback failure"):
        manager.for_epoch(0)
    _assert_rng(_rng(), before)
    assert entered == ["enter", "exit"]
    assert loader.gradient_state.active_dataloader is None
    with case.assertRaisesRegex(RuntimeError, "cannot be reused"):
        manager.for_epoch(0)
    _close(loader)


class _Stream(IterableDataset):
    def __iter__(self):
        yield from range(8)


def _check_epoch_guards(case):
    with case.assertRaisesRegex(TypeError, "map-style"):
        ReplayResumedMapDataLoader(DataLoader(_Stream()), completed_microbatches=0)
    for completed in [-1, True, 1.5]:
        with case.assertRaisesRegex(ValueError, "nonnegative integer"):
            ReplayResumedMapDataLoader(_loader(), completed_microbatches=completed)
    loader = _loader()
    manager = ReplayResumedMapDataLoader(loader, completed_microbatches=14)
    with case.assertRaisesRegex(ValueError, "before"):
        manager.for_epoch(0)
    with case.assertRaisesRegex(ValueError, "consecutively"):
        manager.for_epoch(2)
    continuation, offset = manager.for_epoch(1)
    assert offset == 1
    with case.assertRaisesRegex(RuntimeError, "Exhaust"):
        manager.for_epoch(2)
    list(continuation)
    with case.assertRaisesRegex(RuntimeError, "only be iterated once"):
        list(continuation)
    list(manager.for_epoch(2)[0])
    with case.assertRaisesRegex(ValueError, "fresh"):
        ReplayResumedMapDataLoader(loader, completed_microbatches=0)
    _close(loader)


def _check_input_guards(case):
    loader = _loader(workers=0)
    loader.device = torch.device("cuda:0")
    with case.assertRaisesRegex(ValueError, "device_placement"):
        ReplayResumedMapDataLoader(loader, completed_microbatches=0)
    loader.device = None
    loader.use_stateful_dataloader = True
    with case.assertRaisesRegex(ValueError, "stateful"):
        ReplayResumedMapDataLoader(loader, completed_microbatches=0)
    with case.assertRaisesRegex(ValueError, "generator"):
        ReplayResumedMapDataLoader(DataLoader(range(8)), completed_microbatches=0)
    with case.assertRaisesRegex(ValueError, "persistent multiprocessing"):
        ReplayResumedMapDataLoader(_loader(workers=0), completed_microbatches=0)
    assert not torch.cuda.is_initialized()


class ReplayTaskDataTests(unittest.TestCase):
    def test_four_spawn_workers_replay_across_epochs(self):
        for completed in (0, 26, 29, 38):
            with self.subTest(completed=completed):
                _check_four_spawn_workers_replay(completed)

    def test_process_rng_and_external_context_on_failure(self):
        _check_failure_rng(self)

    def test_epoch_guards_and_iterator_reuse(self):
        _check_epoch_guards(self)

    def test_device_stateful_and_generator_guards(self):
        _check_input_guards(self)

    def test_actual_task_cursor_uses_431_batches_without_fetching_data(self):
        raw = DataLoader(range(13808), batch_size=4, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(42), num_workers=4,
            multiprocessing_context="spawn", persistent_workers=True, prefetch_factor=4)
        loader = prepare_data_loader(raw, num_processes=8, process_index=7,
                                    put_on_device=False, rng_types=[])
        manager = ReplayResumedMapDataLoader(loader, completed_microbatches=4000)
        assert manager.epoch_length == 431
        assert (manager.resume_epoch, manager.resume_offset) == (9, 121)
        assert manager.replayed_microbatches == 0


if __name__ == "__main__":
    unittest.main()
