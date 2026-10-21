"""Single-process FP32 AdamW offload using PyTorch's unmodified CPU optimizer.

Public parameter groups retain the original model Parameter objects, so ordinary
Accelerate/DDP, gradient checks, clipping and LR schedulers see the real model.
Only synchronous copies surround native torch.optim.AdamW.step(). No Adam update
equations, loss scaling, clipping or accumulation are implemented here.

New tasks only: from_optimizer() refuses existing optimizer state. A wrapper
checkpoint can be restored after restoring the model's FP32 state first; native
A100 AdamW state is deliberately not accepted. As with ordinary optimizer state,
model weights are not included in state_dict(). CPU masters are refreshed from
the restored public model parameters by load_state_dict().
"""
from __future__ import annotations

from collections import defaultdict
import copy

import torch


class CpuMasterAdamW(torch.optim.Optimizer):
    SCHEMA = "cvpr_cpu_master_adamw_v1"

    def __init__(self, params, **adamw_kwargs):
        # Delegate argument/default validation to the actual PyTorch optimizer.
        # PyTorch 2.7 stores this Adam-internal flag in AdamW.defaults even
        # though AdamW.__init__ does not accept it as a keyword argument.
        adamw_kwargs = dict(adamw_kwargs)
        if adamw_kwargs.pop("decoupled_weight_decay", True) is not True:
            raise ValueError("CpuMasterAdamW preserves AdamW decoupled weight decay")
        prototype = torch.optim.AdamW(params, **adamw_kwargs)
        self._initializing = True
        super().__init__(prototype.param_groups, prototype.defaults)
        self._initializing = False
        self._public_layout = tuple(tuple(group["params"]) for group in self.param_groups)
        flat = [parameter for group in self._public_layout for parameter in group]
        if len({id(parameter) for parameter in flat}) != len(flat):
            raise ValueError("Each shared Parameter must occur once in the optimizer")
        cpu_groups = []
        for group in self.param_groups:
            self._check_group(group)
            masters = []
            for parameter in group["params"]:
                if parameter.dtype != torch.float32:
                    raise ValueError("CpuMasterAdamW requires the original FP32 model parameters")
                masters.append(torch.nn.Parameter(parameter.detach().to("cpu", copy=True), requires_grad=False))
            cpu_groups.append({**{key: copy.deepcopy(value) for key, value in group.items() if key != "params"}, "params": masters})
        self.cpu_optimizer = torch.optim.AdamW(cpu_groups, **adamw_kwargs)
        self.completed_steps = 0
        self._refresh_public_state()

    @classmethod
    def from_optimizer(cls, optimizer):
        if type(optimizer) is not torch.optim.AdamW or optimizer.state:
            raise ValueError("Wrap a fresh native torch.optim.AdamW; do not restore A100 optimizer state")
        return cls(optimizer.param_groups, **optimizer.defaults)

    def add_param_group(self, param_group):
        if not getattr(self, "_initializing", False):
            raise ValueError("CpuMasterAdamW parameter groups are fixed for this new run")
        return super().add_param_group(param_group)

    @staticmethod
    def _check_group(group):
        if any(group.get(key, False) for key in ("capturable", "differentiable", "fused")):
            raise ValueError("CPU offload requires capturable=False, differentiable=False, fused=False")
        if torch.is_tensor(group["lr"]):
            raise ValueError("Use ordinary scalar learning rates for the CPU-master optimizer")

    def _validate_layout(self):
        if len(self.param_groups) != len(self._public_layout):
            raise ValueError("Optimizer parameter groups changed")
        for group, original in zip(self.param_groups, self._public_layout):
            self._check_group(group)
            if len(group["params"]) != len(original) or any(a is not b for a, b in zip(group["params"], original)):
                raise ValueError("Optimizer Parameter identities/order changed")
            if any(parameter.dtype != torch.float32 for parameter in original):
                raise ValueError("Model parameters must remain FP32")

    def _refresh_public_state(self):
        # Diagnostics expecting optimizer.state[model_parameter] see real CPU
        # moments. Do not create empty native states for unused/routed experts.
        self.state = defaultdict(dict)
        for public, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for parameter, master in zip(public["params"], cpu["params"]):
                if master in self.cpu_optimizer.state:
                    self.state[parameter] = self.cpu_optimizer.state[master]

    @torch.no_grad()
    def sync_from_model(self):
        """Call after any external model-weight replacement; optimizer state stays."""
        self._validate_layout()
        for public, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for parameter, master in zip(public["params"], cpu["params"]):
                master.copy_(parameter.detach(), non_blocking=False)

    @torch.no_grad()
    def step(self, closure=None):
        self._validate_layout()
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for public, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for key, value in public.items():
                if key != "params":
                    cpu[key] = copy.deepcopy(value)
            for parameter, master in zip(public["params"], cpu["params"]):
                gradient = parameter.grad
                if gradient is None:
                    master.grad = None
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                if gradient.dtype != torch.float32:
                    raise ValueError("CPU-master gradients must be FP32 after the original unscale/clip")
                if master.grad is None:
                    master.grad = torch.empty_like(master)
                master.grad.copy_(gradient.detach(), non_blocking=False)
        self.cpu_optimizer.step()
        for public, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            for parameter, master in zip(public["params"], cpu["params"]):
                if parameter.grad is not None:
                    parameter.copy_(master, non_blocking=False)
        self.completed_steps += 1
        self._refresh_public_state()
        return loss

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.cpu_optimizer.zero_grad(set_to_none=set_to_none)

    def _layout_metadata(self):
        return [{"group_name": group.get("group_name"), "shapes": [list(p.shape) for p in group["params"]]}
                for group in self.param_groups]

    def state_dict(self):
        self._validate_layout()
        result = self.cpu_optimizer.state_dict()
        # LR scheduler changes to the public groups must be included even when
        # saving between scheduler.step() and the next optimizer.step().
        result["param_groups"] = [{**group, **{k: copy.deepcopy(v) for k, v in public.items() if k != "params"}}
                                  for group, public in zip(result["param_groups"], self.param_groups)]
        result["cpu_master_adamw"] = {"schema": self.SCHEMA, "completed_steps": self.completed_steps,
            "layout": self._layout_metadata(), "restore_order": "restore FP32 model, then optimizer, then scheduler"}
        return result

    def load_state_dict(self, state_dict):
        metadata = state_dict.get("cpu_master_adamw", {})
        if metadata.get("schema") != self.SCHEMA:
            raise ValueError("Only CpuMasterAdamW state is accepted; A100/native optimizer resume is unsupported")
        self._validate_layout()
        if metadata.get("layout") != self._layout_metadata():
            raise ValueError("CPU-master optimizer checkpoint parameter-group layout changed")
        steps = metadata.get("completed_steps")
        if not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid CPU-master completed_steps")
        groups = state_dict["param_groups"]
        if len(groups) != len(self.param_groups):
            raise ValueError("CPU-master optimizer checkpoint group count changed")
        for public, saved in zip(self.param_groups, groups):
            if len(public["params"]) != len(saved["params"]):
                raise ValueError("CPU-master optimizer checkpoint parameter count changed")
            for parameter, key in zip(public["params"], saved["params"]):
                entry = state_dict["state"].get(key, {})
                for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                    if name in entry and entry[name].shape != parameter.shape:
                        raise ValueError("CPU-master optimizer moment shape changed")
        def cpu_tree(value):
            if torch.is_tensor(value):
                return value.detach().to("cpu")
            if isinstance(value, dict):
                return {k: cpu_tree(v) for k, v in value.items()}
            if isinstance(value, list):
                return [cpu_tree(v) for v in value]
            if isinstance(value, tuple):
                return tuple(cpu_tree(v) for v in value)
            return copy.deepcopy(value)
        self.cpu_optimizer.load_state_dict(cpu_tree({"state": state_dict["state"], "param_groups": groups}))
        for public, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            public.update({k: copy.deepcopy(v) for k, v in cpu.items() if k != "params"})
        self.sync_from_model()
        self.completed_steps = steps
        self._refresh_public_state()

    def receipt(self):
        groups = []
        for public, cpu in zip(self.param_groups, self.cpu_optimizer.param_groups):
            states = [self.cpu_optimizer.state[p] for p in cpu["params"] if p in self.cpu_optimizer.state]
            steps = [int(state["step"].item()) for state in states if "step" in state]
            groups.append({"group_name": public.get("group_name"), "lr": float(public["lr"]),
                "parameter_tensors": len(public["params"]), "numel": sum(p.numel() for p in public["params"]),
                "public_devices": sorted({str(p.device) for p in public["params"]}),
                "master_devices": sorted({str(p.device) for p in cpu["params"]}),
                "initialized_states": len(states), "state_devices": sorted({str(v.device) for state in states for v in state.values() if torch.is_tensor(v)}),
                "parameter_step_min": min(steps, default=0), "parameter_step_max": max(steps, default=0)})
        return {"schema": self.SCHEMA, "native_optimizer": "torch.optim.AdamW", "completed_steps": self.completed_steps,
                "model_and_master_dtype": "float32", "copy_mode": "synchronous", "groups": groups}
