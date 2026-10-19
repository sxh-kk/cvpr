"""Rolling CPU inputs survive a native process abort without full-run I/O."""

import subprocess
import sys

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


@pytest.mark.parametrize("window", ["3:2", "-1:3", "1", "1:x"])
def test_invalid_window_is_rejected_before_training(tmp_path, monkeypatch, window):
    monkeypatch.setenv("PSI0_DIAGNOSTIC_BATCH_RANGE", window)
    with pytest.raises(ValueError, match="PSI0_DIAGNOSTIC_BATCH_RANGE"):
        TrainingSafetyGuard(tmp_path, device="cpu")
