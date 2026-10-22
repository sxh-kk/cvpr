"""Change only AdamW execution flags after a complete task-state restoration."""
from __future__ import annotations

import math

import torch


def select_adamw_backend(optimizer, *, fused):
    """Keep parameters, moments, individual counters and group options intact.

    Ordinary AdamW stores counters on CPU; the fused CUDA kernel requires them
    on the parameter device. Counter values and moment tensors are preserved.
    """
    if type(fused) is not bool:
        raise TypeError("fused must be explicitly true or false")
    actual = getattr(optimizer, "optimizer", optimizer)
    if not isinstance(actual, torch.optim.AdamW):
        raise TypeError("Only an existing torch.optim.AdamW can be switched")
    counters = []
    # Validate everything before changing any execution flags.
    for group in actual.param_groups:
        if group.get("capturable", False) or group.get("differentiable", False):
            raise ValueError("This task uses noncapturable, nondifferentiable AdamW")
        for parameter in group["params"]:
            if fused and (parameter.device.type != "cuda" or parameter.dtype != torch.float32):
                raise ValueError("The accepted fused task backend requires CUDA FP32 master parameters")
            state = actual.state.get(parameter)
            if not state:
                continue  # An unused routed parameter must remain uninitialized.
            step = state.get("step")
            if not isinstance(step, torch.Tensor) or step.numel() != 1:
                raise ValueError("Expected a scalar saved Adam counter")
            value = float(step.detach().cpu())
            if not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError("Invalid Adam counter")
            counters.append((parameter, state, value))
    for parameter, state, value in counters:
        target = parameter.device if fused else torch.device("cpu")
        state["step"] = state["step"].to(device=target, dtype=torch.float32)
        if float(state["step"].detach().cpu()) != value:
            raise ValueError("Adam counter cannot be represented exactly")
    actual.defaults.update(fused=fused, foreach=False)
    for group in actual.param_groups:
        group.update(fused=fused, foreach=False)
    return {"backend": "fused" if fused else "single_tensor", "foreach": False,
            "initialized_parameter_states": len(counters),
            "minimum_parameter_step": min((value for _, _, value in counters), default=None),
            "maximum_parameter_step": max((value for _, _, value in counters), default=None),
            "moments_reinitialized": False}


def optimizer_state_identity(optimizer):
    """Small metadata receipt; not a claim of full moment equality."""
    actual = getattr(optimizer, "optimizer", optimizer)
    return [{"group_name": group["group_name"], "lr": float(group["lr"]),
             "betas": list(group["betas"]), "eps": group["eps"],
             "weight_decay": group["weight_decay"], "fused": group.get("fused"),
             "foreach": group.get("foreach"), "parameter_tensors": len(group["params"])}
            for group in actual.param_groups]


def exact_tensor_equal(actual, expected, *, chunk_elements=262144):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    left, right = actual.detach().reshape(-1), expected.detach().reshape(-1)
    return all(torch.equal(left[offset:offset + chunk_elements].cpu(),
                           right[offset:offset + chunk_elements].cpu())
               for offset in range(0, left.numel(), chunk_elements))


def audit_optimizer_restoration(optimizer, saved):
    """Full moment/counter equality before selecting a different execution path."""
    actual = getattr(optimizer, "optimizer", optimizer)
    if len(actual.param_groups) != len(saved["param_groups"]):
        raise ValueError("Optimizer group count changed")
    checked, elements = 0, 0
    for group, reference in zip(actual.param_groups, saved["param_groups"]):
        options = {key: value for key, value in group.items() if key != "params"}
        expected_options = {key: value for key, value in reference.items() if key != "params"}
        if options != expected_options or len(group["params"]) != len(reference["params"]):
            raise ValueError("Saved optimizer options/order changed")
        for parameter, index in zip(group["params"], reference["params"]):
            state, expected = actual.state.get(parameter, {}), saved["state"].get(index, {})
            if state.keys() != expected.keys():
                raise ValueError("Missing or recreated optimizer state")
            for key, tensor in expected.items():
                if not isinstance(tensor, torch.Tensor) or not exact_tensor_equal(state[key], tensor):
                    raise ValueError(f"Restored optimizer state differs: {key}")
                checked += 1
                elements += tensor.numel()
    return {"status": "passed", "all_saved_state_tensors_exactly_equal": True,
            "state_tensors_compared": checked, "state_elements_compared": elements,
            "groups_compared": len(actual.param_groups)}
