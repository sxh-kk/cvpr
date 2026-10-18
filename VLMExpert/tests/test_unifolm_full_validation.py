"""CPU checks of checkpoint-verification helpers, never substitutes for the GPU gate."""

import importlib.util
import io
from pathlib import Path
import random

import numpy as np
import pytest
import torch


def _script():
    path = Path(__file__).parents[1] / "scripts/validation/validate_unifolm_full_model.py"
    spec = importlib.util.spec_from_file_location("_real_full_validation_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_full_state_hash_handles_bfloat16_and_noncontiguous_values():
    gate = _script()
    value = torch.arange(96).reshape(8, 12).to(torch.bfloat16)
    assert gate.tree_hash(value.T) == gate.tree_hash(value.T.contiguous())
    assert gate.tree_hash(value) != gate.tree_hash(value + 1)
    assert gate.tree_hash(value) != gate.tree_hash(value.float())


def test_rng_hash_includes_numpy_values_hidden_by_abbreviated_repr():
    gate = _script()
    first = np.zeros(4096, dtype=np.uint32)
    second = first.copy()
    second[2048] = 1
    assert repr(first) == repr(second)
    assert gate.tree_hash(first) != gate.tree_hash(second)


def test_real_adamw_state_roundtrip_and_slot_mutation_are_detected():
    gate = _script()
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    model(torch.ones(1, 4)).square().sum().backward()
    optimizer.step()
    original = optimizer.state_dict()
    stream = io.BytesIO()
    torch.save(original, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert gate.tree_hash(original) == gate.tree_hash(restored)
    restored["state"][0]["exp_avg"].reshape(-1)[0] += 1
    assert gate.tree_hash(original) != gate.tree_hash(restored)


def test_parameter_inventory_counts_frozen_tensors_and_rejects_bf16_masters():
    gate = _script()
    model = torch.nn.ModuleDict({"action_header": torch.nn.Linear(3, 2),
                                 "motion_conditioner": torch.nn.Linear(2, 4)})
    model["motion_conditioner"].bias.requires_grad_(False)
    report = gate.parameter_inventory(model, require_fp32=True)
    motion = report["components"]["motion_expert"]
    assert motion["named_parameter_tensors"] == 2
    assert motion["elements"] == 12
    assert motion["trainable_elements"] == 8 and motion["frozen_elements"] == 4
    model.bfloat16()
    with pytest.raises(AssertionError, match="FP32 master"):
        gate.parameter_inventory(model, require_fp32=True)
    deployment = gate.parameter_inventory(model, require_fp32=False)
    assert deployment["components"]["action_header"]["elements_by_dtype"] == {"torch.bfloat16": 8}


def test_architecture_gate_rejects_a_real_reduced_action_head():
    from psi.models.dual_source_action import DualSourceActionTransformer

    head = DualSourceActionTransformer(action_dim=80, action_pred_horizon=30,
                                       action_hidden_dim=16, action_num_blocks=2, action_nheads=4,
                                       view_feature_dim=2048, odim=45)
    with pytest.raises(AssertionError, match="complete 1536x6/24 head"):
        _script().head_contract(head)


def test_large_embedding_probe_indices_keep_exact_integer_endpoints():
    gate = _script()
    n = 153792 * 2048
    indices = gate.integer_probe_indices(n, 32)
    assert indices.dtype == torch.long and indices.shape == (32,)
    assert indices[0].item() == 0 and indices[-1].item() == n - 1
    assert torch.all((indices >= 0) & (indices < n))
    assert indices.tolist() == [i * (n - 1) // 31 for i in range(32)]


@pytest.mark.parametrize(("n", "count", "expected"), [(0, 32, []), (1, 32, [0]), (10, 0, []), (10, 1, [0])])
def test_probe_indices_handle_empty_and_single_boundaries(n, count, expected):
    indices = _script().integer_probe_indices(n, count)
    assert indices.dtype == torch.long
    assert indices.tolist() == expected


def test_checkpoint_rng_reference_uses_serialized_snapshot_before_logging_draws(tmp_path):
    gate = _script()
    python = random.Random(1234)
    states = {"step": 8, "random_state": python.getstate(),
              "numpy_random_seed": np.random.RandomState(1234).get_state(),
              "torch_manual_seed": torch.Generator().manual_seed(1234).get_state(),
              "torch_cuda_manual_seed": [torch.tensor([10], dtype=torch.uint8),
                                          torch.tensor([20], dtype=torch.uint8)]}
    torch.save(states, tmp_path / "random_states_0.pkl")
    expected = {"python": states["random_state"], "numpy": states["numpy_random_seed"],
                "torch": states["torch_manual_seed"], "cuda": states["torch_cuda_manual_seed"][1]}
    canonical = gate.checkpoint_rng_hash(tmp_path, 0, 1)
    assert canonical == gate.tree_hash(expected)
    for _ in range(7):
        python.getrandbits(32)  # Rich's path hyperlink IDs advance the global Python RNG.
    later_sidecar_hash = gate.tree_hash({**expected, "python": python.getstate()})
    assert canonical != later_sidecar_hash
    assert canonical != gate.checkpoint_rng_hash(tmp_path, 0, 0)
    with pytest.raises(ValueError, match="local CUDA device"):
        gate.checkpoint_rng_hash(tmp_path, 0, 2)
