"""Bounded AdamW microbenchmark; does not load a model or touch a training run.

CUDA must be explicitly selected. Both implementations start from identical,
synthetic *nonzero* FP32 Adam moments, parameters and per-parameter steps. This
checks optimizer math/state migration, not full-model loss or training speed.
Example (only after scheduling a GPU slot)::

    python benchmark_fused_adamw.py --device cuda:0 --run-config RUN/run_config.json \
        --output /outside/repository/fused_adamw.json

CPU plumbing check::

    python benchmark_fused_adamw.py --device cpu --candidate standard \
        --profile tiny --output /tmp/adamw_cpu.json
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import torch


GROUP_NAMES = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")
DEFAULT_LRS = (1e-5, 1e-7, 1e-6, 1e-5, 1e-6)
# Representative dense matrix/vector sizes, not the complete production layout.
MATRIX_SHAPES = ((1024, 1024), (1536, 1536), (1280, 1280), (1536, 1280), (512, 512))
MAX_ELEMENTS = 8_000_000  # two FP32 (parameter, gradient, m, v) copies <= 245 MiB


def optimizer_recipe(run_config=None):
    """Read the same kwargs/LR fields used by PosttrainTrainer.create_optimizers."""
    if run_config is None:
        return {"lrs": dict(zip(GROUP_NAMES, DEFAULT_LRS)),
                "kwargs": {"betas": (0.95, 0.999), "eps": 1e-8, "weight_decay": 1e-6},
                "source": "explicit handover task defaults; no run config bound"}
    path = Path(run_config).resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    train, model = config["train"], config["model"]
    kwargs = dict(train["lr_scheduler_kwargs"])
    if set(kwargs) - {"betas", "eps", "weight_decay"}:
        raise ValueError("Unexpected optimizer kwargs; extend the benchmark explicitly")
    kwargs["betas"] = tuple(kwargs.get("betas", (0.9, 0.999)))
    kwargs.setdefault("eps", 1e-8)
    kwargs.setdefault("weight_decay", 0.01)
    lrs = dict(zip(GROUP_NAMES, (train["learning_rate"], model["lang_backbone_lr"],
        model["vision_tower_lr"], model["mm_projector_lr"], model["motion_expert_lr"])))
    return {"lrs": lrs, "kwargs": kwargs, "source": str(path),
            "source_sha256": hashlib.sha256(raw).hexdigest()}


def layout(profile="representative"):
    if profile not in {"tiny", "representative"}:
        raise ValueError("Unknown size profile")
    shapes = MATRIX_SHAPES if profile == "representative" else ((7, 11),) * 5
    # Third tensor deliberately receives None on some numerical-check steps.
    return [{"group_name": name, "shapes": [shape, (shape[0],), (3,)]}
            for name, shape in zip(GROUP_NAMES, shapes)]


def make_fixture(recipe, profile="representative", seed=20261022, initial_step=938):
    if initial_step < 1:
        raise ValueError("Use a nonzero initial Adam step")
    description = layout(profile)
    numel = sum(math.prod(shape) for group in description for shape in group["shapes"])
    if numel > MAX_ELEMENTS:
        raise ValueError("Fixture exceeds the bounded-memory allocation limit")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    parameters, gradients, state, groups = [], [], {}, []
    for group in description:
        ids = []
        for shape in group["shapes"]:
            index = len(parameters)
            ids.append(index)
            parameters.append(torch.randn(shape, generator=generator) * 0.1 + 0.25)
            gradients.append(torch.randn(shape, generator=generator) * 0.03 + 0.005)
            state[index] = {"step": torch.tensor(float(initial_step)),
                "exp_avg": torch.randn(shape, generator=generator) * 0.01,
                "exp_avg_sq": torch.rand(shape, generator=generator) * 0.003 + 0.001}
        groups.append({"params": ids, "group_name": group["group_name"],
                       "lr": recipe["lrs"][group["group_name"]]})
    return {"parameters": parameters, "gradients": gradients,
            "optimizer_state": {"state": state, "param_groups": groups},
            "layout": description, "numel": numel, "initial_step": initial_step}


def build_optimizer(fixture, recipe, device, *, fused):
    """Restore nonzero state while deliberately retaining the target backend.

    load_state_dict replaces group options. Setting only the constructor's
    fused=True would silently restore fused=False from a production checkpoint.
    Patching a private copy *before* load also lets PyTorch place step tensors on
    the correct device for the fused backend. The source fixture stays intact.
    """
    parameters = [torch.nn.Parameter(p.to(device=device, copy=True)) for p in fixture["parameters"]]
    groups = []
    restored = {"state": fixture["optimizer_state"]["state"], "param_groups": []}
    for group in fixture["optimizer_state"]["param_groups"]:
        groups.append({"params": [parameters[i] for i in group["params"]],
                       "group_name": group["group_name"], "lr": group["lr"]})
        restored["param_groups"].append({**copy.deepcopy(recipe["kwargs"]),
            **copy.deepcopy(group), "foreach": False, "fused": fused,
            "capturable": False, "differentiable": False, "maximize": False,
            "amsgrad": False})
    optimizer = torch.optim.AdamW(groups, foreach=False, fused=fused, **recipe["kwargs"])
    # Deep-copy on CPU: two CPU reference branches must not alias Adam moments.
    optimizer.load_state_dict(copy.deepcopy(restored))
    for p, template in zip(parameters, fixture["gradients"]):
        p.grad = torch.empty_like(p)
        p.grad.copy_(template)
    return parameters, optimizer


@torch.no_grad()
def set_gradients(parameters, fixture, update, *, allow_none=True):
    for index, (parameter, template) in enumerate(zip(parameters, fixture["gradients"])):
        if allow_none and index % 3 == 2 and update % 3 != 0:
            parameter.grad = None
        else:
            if parameter.grad is None:
                parameter.grad = torch.empty_like(parameter)
            # Distinct signed, nonzero gradients across steps; no RNG on CUDA.
            parameter.grad.copy_(template * (1.0 + (update % 5) * 0.07))


def tensor_comparison(reference, candidate, *, atol, rtol, chunk_elements):
    """Bound host scratch space; never materialize full-device difference tensors."""
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        return {"passed": False, "reason": "shape_or_dtype_mismatch"}
    left, right = reference.detach().view(-1), candidate.detach().view(-1)
    result = {"passed": True, "finite": True, "max_abs": 0.0,
              "max_relative": 0.0, "max_tolerance_ratio": 0.0}
    for offset in range(0, left.numel(), chunk_elements):
        a = left[offset:offset + chunk_elements].to("cpu", copy=True).double()
        b = right[offset:offset + chunk_elements].to("cpu", copy=True).double()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        if not finite:
            result.update(passed=False, finite=False)
            continue
        difference = (a - b).abs()
        if atol == 0 and rtol == 0:
            ratio = torch.where(difference == 0, 0.0, float("inf"))
        else:
            ratio = difference / (atol + rtol * a.abs())
        result["max_abs"] = max(result["max_abs"], difference.max().item())
        result["max_relative"] = max(result["max_relative"],
            (difference / a.abs().clamp_min(1e-12)).max().item())
        result["max_tolerance_ratio"] = max(result["max_tolerance_ratio"], ratio.max().item())
        result["passed"] = result["passed"] and bool((ratio <= 1).all())
    return result


def compare_branches(reference, candidate, fixture, updates, *, atol, rtol, chunk_elements,
                     fully_active_steps=0):
    left_params, left_opt = reference
    right_params, right_opt = candidate
    results, passed = [], True
    for index, (left, right) in enumerate(zip(left_params, right_params)):
        checks = {}
        for name, a, b in (("parameter", left, right),
                          *((key, left_opt.state[left][key], right_opt.state[right][key])
                            for key in ("exp_avg", "exp_avg_sq", "step"))):
            checks[name] = tensor_comparison(a, b, atol=0.0 if name == "step" else atol,
                rtol=0.0 if name == "step" else rtol, chunk_elements=chunk_elements)
        # A step counter is state too; inactive routed-expert-like tensors must
        # retain their counter/moments when gradient is None.
        active_steps = sum(index % 3 != 2 or update % 3 == 0 for update in range(1, updates + 1))
        expected_step = fixture["initial_step"] + active_steps + fully_active_steps
        observed_steps = [int(opt.state[p]["step"].item())
                          for p, opt in ((left, left_opt), (right, right_opt))]
        changed = not tensor_comparison(fixture["parameters"][index], right,
            atol=1e-30, rtol=0.0, chunk_elements=chunk_elements)["passed"]
        entry_passed = (all(check["passed"] for check in checks.values())
                        and observed_steps == [expected_step] * 2 and changed)
        passed &= entry_passed
        results.append({"group_name": GROUP_NAMES[index // 3], "index": index,
            "shape": list(left.shape), "checks": checks,
            "expected_step": expected_step, "observed_steps": observed_steps,
            "nonzero_parameter_update": changed, "passed": entry_passed})
    return {"passed": bool(passed), "updates": updates,
            "additional_fully_active_steps": fully_active_steps, "tensors": results,
            "atol": atol, "rtol": rtol,
            "gradient_pattern": "each third tensor skips updates not divisible by three"}


def cuda_memory(device):
    free, total = torch.cuda.mem_get_info(device)
    return {"allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "global_free_bytes": free, "global_total_bytes": total}


def time_optimizer(optimizer, device, iterations):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        before = cuda_memory(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()
        start.record()
        for _ in range(iterations):
            optimizer.step()
        end.record()
        end.synchronize()
        elapsed_ms = start.elapsed_time(end)
        wall_ms = (time.perf_counter() - wall_start) * 1000
        memory = cuda_memory(device)
        memory["temporary_peak_above_live_bytes"] = max(0,
            memory["peak_allocated_bytes"] - before["allocated_bytes"])
    else:
        wall_start = time.perf_counter()
        for _ in range(iterations):
            optimizer.step()
        elapsed_ms = wall_ms = (time.perf_counter() - wall_start) * 1000
        memory = None
    return {"milliseconds_per_step": elapsed_ms / iterations,
            "wall_milliseconds_per_step": wall_ms / iterations, "memory": memory}


def benchmark(args):
    if (args.correctness_steps < 3 or args.warmup < 1 or args.iterations < 1 or args.rounds < 2
            or args.chunk_elements < 1 or args.atol <= 0 or args.rtol < 0):
        raise ValueError("Require >=3 correctness steps, positive warmup/iterations/chunks/tolerances, >=2 rounds")
    device = torch.device(args.device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only explicit CPU or CUDA devices are supported")
    recipe = optimizer_recipe(args.run_config)
    fixture = make_fixture(recipe, args.profile, args.seed, args.initial_step)
    # Estimated live parameter, gradient, and moments for two branches only.
    estimated_live_bytes = fixture["numel"] * 4 * 4 * 2
    if estimated_live_bytes > args.max_allocated_mib * 1024 ** 2:
        raise ValueError("Fixture exceeds requested CUDA allocation budget before CUDA initialization")
    receipt = {"schema": "cvpr_fused_adamw_microbenchmark_v1",
        "started_utc": datetime.now(timezone.utc).isoformat(), "torch": torch.__version__,
        "cuda_build": torch.version.cuda, "device": str(device), "seed": args.seed,
        "profile": args.profile, "recipe": recipe, "layout": fixture["layout"],
        "dtype": "float32", "numel_per_branch": fixture["numel"],
        "estimated_live_tensor_bytes_both_branches": estimated_live_bytes,
        "source_state": "synthetic nonzero moments; not production checkpoint tensors",
        "initial_step": args.initial_step, "baseline": "AdamW foreach=False fused=False",
        "candidate": f"AdamW foreach=False fused={args.candidate == 'fused'}",
        "scope": "single-device optimizer-only; no model forward/backward, DDP, scheduler or end-to-end throughput",
        "limitation": "shapes, tensor count and active-expert pattern are a bounded sample; no training-speed prediction"}
    if device.type == "cuda":
        torch.cuda.set_device(device)
        receipt["gpu_name"] = torch.cuda.get_device_name(device)
        receipt["memory_before_fixture"] = cuda_memory(device)
    reference = build_optimizer(fixture, recipe, device, fused=False)
    candidate = build_optimizer(fixture, recipe, device, fused=args.candidate == "fused")
    if device.type == "cuda":
        receipt["memory_after_fixture"] = cuda_memory(device)
        if torch.cuda.memory_allocated(device) > args.max_allocated_mib * 1024 ** 2:
            raise RuntimeError("Actual CUDA allocations exceeded the requested budget")
    for update in range(1, args.correctness_steps + 1):
        for parameters, optimizer in (reference, candidate):
            set_gradients(parameters, fixture, update)
            optimizer.step()
    receipt["correctness"] = compare_branches(reference, candidate, fixture, args.correctness_steps,
        atol=args.atol, rtol=args.rtol, chunk_elements=args.chunk_elements)
    if not receipt["correctness"]["passed"]:
        receipt["status"] = "failed_numerical_comparison"
        return receipt
    for parameters, optimizer in (reference, candidate):
        set_gradients(parameters, fixture, 1, allow_none=False)
        for _ in range(args.warmup):
            optimizer.step()
    # Alternate order to expose some shared-GPU contention/order sensitivity.
    timings = {"standard": [], "candidate": []}
    for index in range(args.rounds):
        order = (("standard", reference), ("candidate", candidate))
        for name, (_, optimizer) in order if index % 2 == 0 else reversed(order):
            timings[name].append(time_optimizer(optimizer, device, args.iterations))
    receipt["timing"] = {"warmup_steps": args.warmup, "iterations_per_round": args.iterations,
        "rounds": args.rounds, "method": "CUDA events plus synchronized wall clock" if device.type == "cuda" else "CPU perf_counter",
        "gradient_preparation_included": False, "all_tensors_active_for_timing": True,
        "samples": timings,
        "median_milliseconds_per_step": {key: statistics.median(x["milliseconds_per_step"] for x in values)
                                          for key, values in timings.items()}}
    # Check the fully active warmup/timing phase too, including finite moments.
    extra_steps = args.warmup + args.iterations * args.rounds
    receipt["post_timing_correctness"] = compare_branches(reference, candidate, fixture,
        args.correctness_steps, fully_active_steps=extra_steps,
        atol=args.atol, rtol=args.rtol, chunk_elements=args.chunk_elements)
    final_finite = all(check.get("finite", False)
        for tensor in receipt["post_timing_correctness"]["tensors"]
        for check in tensor["checks"].values())
    receipt["timing"]["additional_steps_per_branch"] = extra_steps
    receipt["timing"]["final_all_parameters_and_moments_finite"] = bool(final_finite)
    if device.type == "cuda":
        receipt["memory_final"] = cuda_memory(device)
        if torch.cuda.memory_allocated(device) > args.max_allocated_mib * 1024 ** 2:
            raise RuntimeError("Actual CUDA allocations exceeded the requested budget")
    receipt["status"] = ("optimizer_microbenchmark_passed"
        if receipt["post_timing_correctness"]["passed"] else "failed_post_timing_numerical_comparison")
    if device.type == "cpu" and args.candidate == "standard":
        if receipt["post_timing_correctness"]["passed"]:
            receipt["status"] = "cpu_harness_self_check_passed"
    receipt["finished_utc"] = datetime.now(timezone.utc).isoformat()
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, help="Explicit cpu or cuda:N; never auto-selects a GPU")
    parser.add_argument("--run-config")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate", choices=("fused", "standard"), default="fused")
    parser.add_argument("--profile", choices=("tiny", "representative"), default="representative")
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--initial-step", type=int, default=938)
    parser.add_argument("--correctness-steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--chunk-elements", type=int, default=131072)
    parser.add_argument("--max-allocated-mib", type=int, default=384)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--rtol", type=float, default=1e-5)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing receipt: {args.output}")
    try:
        receipt = benchmark(args)
    except Exception as error:
        receipt = {"status": "failed", "error_type": type(error).__name__, "error": str(error),
                   "device": args.device, "scope": "optimizer-only microbenchmark; no training modified"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output.resolve())}))
    return 0 if receipt["status"] in {"optimizer_microbenchmark_passed", "cpu_harness_self_check_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
