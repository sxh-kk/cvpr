"""Rolling CPU inputs survive a native process abort without full-run I/O."""

import subprocess
import sys
import errno
import random

import numpy as np
import pytest
import torch

from psi.trainers.training_safety import TrainingSafetyGuard


def test_bounded_window_rotates_one_file_and_preserves_failure_dump(tmp_path, monkeypatch):
    monkeypatch.setenv("PSI0_DIAGNOSTIC_BATCH_RANGE", "2:3")
    guard = TrainingSafetyGuard(tmp_path, device="cpu", accumulation_steps=2)
    for step in range(5):
        for micro in range(2):
            value = torch.tensor([step, micro])
            guard.capture({"actions": value}, global_step=step, local_step=step * 2 + micro)
            value.fill_(-1)
        if step < 2:
            assert not list(tmp_path.iterdir())
    paths = list(tmp_path.glob("*.pt"))
    assert len(paths) == 1
    saved = torch.load(paths[0], weights_only=False)
    assert saved["global_step"] == 3 and saved["local_step"] == 7
    assert len(saved["microbatches"]) == 2
    assert [entry["batch"]["actions"].tolist() for entry in saved["microbatches"]] == [[3, 0], [3, 1]]
    guard.stage = "backward"
    failure = guard.dump("caught CUDA error")
    assert failure != paths[0]
    assert torch.load(failure, weights_only=False)["reason"] == "caught CUDA error"


def test_cpu_batch_and_rng_survive_native_abort(tmp_path, monkeypatch):
    monkeypatch.setenv("PSI0_DIAGNOSTIC_BATCH_RANGE", "0:0")
    code = '''
import os, resource, sys, torch
from psi.trainers.training_safety import TrainingSafetyGuard
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
guard = TrainingSafetyGuard(sys.argv[1], device="cpu", accumulation_steps=2)
guard.capture({"input_ids": torch.tensor([[1, 2, 3]])}, global_step=0, local_step=0)
os.abort()
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], capture_output=True)
    assert result.returncode < 0
    saved = torch.load(tmp_path / "latest_batch_rank_000.pt", weights_only=False)
    assert saved["stage"] == "pre_forward"
    assert saved["microbatches"][0]["batch"]["input_ids"].tolist() == [[1, 2, 3]]
    assert set(saved["microbatches"][0]["rng"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_full_run_capture_retains_previous_update_while_replacing_current(tmp_path, monkeypatch):
    rolling = tmp_path / "rolling"
    monkeypatch.delenv("PSI0_DIAGNOSTIC_BATCH_RANGE", raising=False)
    monkeypatch.setenv("PSI0_ROLLING_DIAGNOSTICS_DIR", str(rolling))
    guard = TrainingSafetyGuard(tmp_path / "failures", device="cpu", rank=5, accumulation_steps=2)
    for step in range(10):
        for micro in range(2):
            guard.capture({"value": torch.tensor([step, micro])}, global_step=step, local_step=step * 2 + micro)
    # Starting an eleventh update retains both microbatches of the tenth one.
    guard.capture({"value": torch.tensor([10, 0])}, global_step=10, local_step=20)
    paths = list(rolling.iterdir())
    assert len(paths) == 2
    saved = {item["global_step"]: item for item in (
        torch.load(path, weights_only=False) for path in paths)}
    assert set(saved) == {9, 10}
    assert [entry["batch"]["value"].tolist() for entry in saved[9]["microbatches"]] == [[9, 0], [9, 1]]
    assert len(saved[10]["microbatches"]) == 1
    assert not (tmp_path / "failures").exists()


def test_full_run_snapshot_survives_abort_outside_bounded_range(tmp_path, monkeypatch):
    rolling = tmp_path / "rolling"
    monkeypatch.setenv("PSI0_DIAGNOSTIC_BATCH_RANGE", "0:1")
    monkeypatch.setenv("PSI0_ROLLING_DIAGNOSTICS_DIR", str(rolling))
    code = '''
import os, resource, sys, torch
from psi.trainers.training_safety import TrainingSafetyGuard
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
guard = TrainingSafetyGuard(sys.argv[1], device="cpu", rank=5, accumulation_steps=2)
for step in (15107, 15108):
    for micro in range(2):
        guard.capture({"actions": torch.tensor([step, micro])}, global_step=step, local_step=step*2+micro)
os.abort()
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "failures")], capture_output=True)
    assert result.returncode < 0
    saved = [torch.load(path, weights_only=False) for path in rolling.glob("*.pt")]
    assert sorted(item["global_step"] for item in saved) == [15107, 15108]
    assert all(len(item["microbatches"]) == 2 for item in saved)
    assert not (tmp_path / "failures").exists()


def test_relative_rolling_directory_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("PSI0_ROLLING_DIAGNOSTICS_DIR", "relative-path")
    with pytest.raises(ValueError, match="PSI0_ROLLING_DIAGNOSTICS_DIR"):
        TrainingSafetyGuard(tmp_path, device="cpu")


def test_rolling_persistence_does_not_advance_random_streams(tmp_path, monkeypatch):
    monkeypatch.setenv("PSI0_ROLLING_DIAGNOSTICS_DIR", str(tmp_path / "rolling"))
    guard = TrainingSafetyGuard(tmp_path / "failures", device="cpu")
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    guard.capture({"actions": torch.ones(2)}, global_step=15108, local_step=30216)
    saved = torch.load(next((tmp_path / "rolling").glob("*.pt")), weights_only=False)
    rng = saved["microbatches"][0]["rng"]
    assert random.getstate() == python_rng == rng["python"]
    np.testing.assert_equal(np.random.get_state(), numpy_rng)
    np.testing.assert_equal(rng["numpy"], numpy_rng)
    assert torch.equal(torch.get_rng_state(), torch_rng)
    assert torch.equal(rng["torch_cpu"], torch_rng)


def test_failed_atomic_replace_raises_and_keeps_both_committed_updates(tmp_path, monkeypatch):
    monkeypatch.setenv("PSI0_ROLLING_DIAGNOSTICS_DIR", str(tmp_path / "rolling"))
    guard = TrainingSafetyGuard(tmp_path / "failures", device="cpu")
    for step in (15106, 15107):
        guard.capture({"actions": torch.tensor([step])}, global_step=step, local_step=step * 2)

    def fail_replace(*_args):
        raise OSError(errno.ENOSPC, "simulated full diagnostic filesystem")

    monkeypatch.setattr("psi.trainers.training_safety.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated full"):
        guard.capture({"actions": torch.tensor([15108])}, global_step=15108, local_step=30216)
    committed = [torch.load(path, weights_only=False) for path in (tmp_path / "rolling").glob("*.pt")]
    assert sorted(item["global_step"] for item in committed) == [15106, 15107]


@pytest.mark.parametrize("window", ["3:2", "-1:3", "1", "1:x"])
def test_invalid_window_is_rejected_before_training(tmp_path, monkeypatch, window):
    monkeypatch.setenv("PSI0_DIAGNOSTIC_BATCH_RANGE", window)
    with pytest.raises(ValueError, match="PSI0_DIAGNOSTIC_BATCH_RANGE"):
        TrainingSafetyGuard(tmp_path, device="cpu")
