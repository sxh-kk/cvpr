"""One physical episode per production server process, with post-load RNG reset/audit."""
from __future__ import annotations
import argparse
from base64 import b64decode
import hashlib
import json
import math
import os
from pathlib import Path
import pickle
import random
import sys
import threading
import time
import numpy as np
from numpy.lib.format import descr_to_dtype
import torch
from fastapi import HTTPException, Request


def rng_sha():
    digest = lambda value: hashlib.sha256(value).hexdigest()
    return {"python": digest(pickle.dumps(random.getstate(), protocol=4)),
            "numpy": digest(pickle.dumps(np.random.get_state(), protocol=4)),
            "torch_cpu": digest(torch.get_rng_state().numpy().tobytes()),
            "torch_cuda": [digest(state.cpu().numpy().tobytes()) for state in torch.cuda.get_rng_state_all()]
                          if torch.cuda.is_initialized() else []}


def audited_server(factory, seed, seed_function, output, source_sha):
    server = factory()  # Complete all production model/DINO construction first.
    seed_function(seed)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    receipt = {"weight_sha256": source_sha, "sampling_seed": seed, "pid": os.getpid(),
               "rng_sha256": rng_sha(), "host": "127.0.0.1", "Ta": server.Ta, "Tp": server.Tp,
               "rtc": server.enable_rtc, "rtc_mode": server.rtc_mode,
               "rtc_guidance_action_scope": getattr(server, "rtc_guidance_action_scope", "all"),
               "rtc_guidance_action_proof": getattr(server, "rtc_guidance_action_proof",
                   dict(scope="all", action_dim=getattr(server, "Da", 80), guided_action_dimensions=getattr(server, "Da", 80),
                        excluded_action_indices=[])),
               "inference_steps": server.num_inference_steps,
               "rtc_exec_shift": server.exec_shift if server.enable_rtc else None,
               "rtc_inference_delay": server.rtc_inference_delay if server.enable_rtc else None,
               "rtc_min_exec_horizon": server.min_exec_horizon if server.enable_rtc else None,
               "pig_mask_schedule": server.pig_mask_schedule if server.rtc_mode == "test_time" else None,
               "pig_guidance_alpha": server.pig_guidance_alpha if server.rtc_mode == "test_time" else None}
    # Count the sampler methods actually invoked by the production handler. The
    # dual-source previous plan lives in SimulationHistory, not previous_action.
    sampler_names = ("predict_action", "predict_action_with_rtc_flow", "predict_action_with_training_rtc_flow")
    sampler_paths = dict(zip(sampler_names, ("plain", "test_time", "train")))
    sampler_counts = {name: 0 for name in sampler_names}
    sampler_action_audit = {}

    def action_values(value):
        return value.detach().float().cpu().tolist() if isinstance(value, torch.Tensor) else np.asarray(value).tolist()
    for name in sampler_names:
        method = getattr(server.model, name, None)
        if callable(method):
            def wrap(original, sampler_name):
                def call(*args, **kwargs):
                    sampler_counts[sampler_name] += 1
                    # Diagnostic values are copied after actual production calls;
                    # never replace the sampler input, RNG or normalized cache.
                    if sampler_name == "predict_action_with_rtc_flow":
                        mask = kwargs.get("guidance_action_mask")
                        actual_mask = None if mask is None else mask.detach().cpu().tolist()
                        sampler_action_audit["actual_rtc_guidance_action_mask"] = actual_mask
                        expected_mask = receipt["rtc_guidance_action_proof"].get("guidance_action_mask")
                        if actual_mask != expected_mask:
                            raise RuntimeError("Actual RTC sampler action mask differs from accepted scope proof")
                    previous = kwargs.get("prev_actions")
                    if previous is not None:
                        sampler_action_audit["normalized_shifted_previous"] = action_values(previous)
                    result = original(*args, **kwargs)
                    sampler_action_audit["normalized_prediction"] = action_values(result)
                    return result
                return call
            setattr(server.model, name, wrap(method, name))
    receipt["audited_sampler_methods"] = [name for name in sampler_names if callable(getattr(server.model, name, None))]
    parameters = list(server.model.parameters())
    receipt["parameter_tensor_count"] = len(parameters)
    receipt["trainable_parameter_tensor_count"] = sum(value.requires_grad for value in parameters)
    receipt["parameter_gradient_tensor_count"] = sum(value.grad is not None for value in parameters)
    online_dino = getattr(server, "online_dino", None)
    if online_dino is not None:
        # Record the encoder actually constructed in this GPU process, rather
        # than treating a cached training contract as runtime-version evidence.
        receipt["online_dino_runtime"] = dict(online_dino.encoder.metadata)
    (output / "startup.json").write_text(json.dumps(receipt, indent=2) + "\n")
    lock, owner_lock, owner, ordinal = threading.Lock(), threading.Lock(), None, 0

    def claim(history):
        nonlocal owner
        identity = (history.get("session_id"), history.get("episode_id")) if isinstance(history, dict) else (None, None)
        if not all(isinstance(value, str) and value for value in identity):
            raise HTTPException(400, "Evaluation requires session_id and episode_id")
        with owner_lock:
            if owner is None:
                owner = identity
            if owner != identity:
                raise HTTPException(409, "One physical episode per process; start a fresh evaluation server")

    original_predict, original_observe, original_reset = server.predict_action, server.observe, server.reset

    def predict_action(payload: dict, http_request: Request):
        nonlocal ordinal
        with lock:
            ordinal += 1
            history = payload.get("history") or {}
            row = {"ordinal": ordinal, "episode_id": history.get("episode_id"), "session_id": history.get("session_id"),
                   "sim_time_s": str(history.get("sim_time_s")), "rng_before": rng_sha(), "status": "exception"}
            before_samplers = sampler_counts.copy()
            sampler_action_audit.clear()
            started = time.monotonic()
            try:
                claim(history)
                response = original_predict(payload, http_request)
                row["status"] = response.status_code
                if response.status_code == 200:
                    action = json.loads(response.body)["action"]
                    if isinstance(action, dict) and "__numpy__" in action:
                        action = np.frombuffer(b64decode(action["__numpy__"]),
                            descr_to_dtype(action["dtype"])).reshape(action["shape"])
                    row["response_action_shape"] = list(np.asarray(action).shape)
                    row["denormalized_returned_action"] = np.asarray(action).tolist()
                return response
            except BaseException as error:
                row["status"] = getattr(error, "status_code", "exception")
                row["exception_type"] = type(error).__name__
                raise
            finally:
                calls = {sampler_paths[name]: sampler_counts[name] - before_samplers[name] for name in sampler_names}
                row["sampler_calls"] = calls
                row["rtc_guidance_action_scope"] = receipt["rtc_guidance_action_scope"]
                row["rtc_guidance_action_proof"] = receipt["rtc_guidance_action_proof"]
                row.update(sampler_action_audit)
                paths = [name for name, count in calls.items() if count]
                row["sampling_path"] = paths[0] if len(paths) == 1 else None
                row["parameter_gradient_tensor_count"] = sum(value.grad is not None for value in parameters)
                row.update(rng_after=rng_sha(), wall_time_s=time.monotonic() - started)
                with (output / "requests.jsonl").open("a") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")

    def observe(payload: dict):
        claim(payload.get("history"))
        return original_observe(payload)

    def reset(payload: dict):
        claim(payload)
        return original_reset(payload)

    server.predict_action, server.observe, server.reset = predict_action, observe, reset
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ckpt-step", type=int, required=True)
    parser.add_argument("--sampling-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=21074)
    parser.add_argument("--rtc-mode", choices=("off", "test_time"), default="off")
    parser.add_argument("--rtc-guidance-action-scope", choices=("all", "validated_repack"), default="all")
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument("--rtc-inference-delay", type=int, default=0)
    parser.add_argument("--min-exec-horizon", type=int)
    parser.add_argument("--pig-mask-schedule", choices=("exponential", "linear", "hard"), default="exponential")
    parser.add_argument("--pig-guidance-alpha", type=float, default=.9)
    args = parser.parse_args()
    validate_runtime_options(args)
    repo = Path(__file__).resolve().parents[2]
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(repo) or args.output.exists():
        raise ValueError("Use a fresh absolute audit directory outside the repository")
    sys.path[:0] = [str(repo / "VLMExpert/src"), str(repo)]
    from psi.deploy.serve_psi0_simple import Server
    from psi.utils import seed_everything
    from finetune_carrybox import sha256
    weights = args.run_dir / "checkpoints" / f"ckpt_{args.ckpt_step}" / "model.safetensors"
    server = audited_server(lambda: Server("psi0", args.run_dir, args.ckpt_step, "cuda:0",
                            enable_rtc=args.rtc_mode != "off", action_exec_horizon=args.execute_steps,
                            rtc_mode=args.rtc_mode, num_inference_steps=10,
                            rtc_inference_delay=args.rtc_inference_delay, min_exec_horizon=args.min_exec_horizon,
                            pig_mask_schedule=args.pig_mask_schedule, pig_guidance_alpha=args.pig_guidance_alpha,
                            rtc_guidance_action_scope=args.rtc_guidance_action_scope),
                            args.sampling_seed, seed_everything, args.output, sha256(weights))
    server.run("127.0.0.1", args.port)


def validate_runtime_options(args):
    scope = getattr(args, "rtc_guidance_action_scope", "all")
    if scope not in ("all", "validated_repack") or (scope == "validated_repack" and args.rtc_mode != "test_time"):
        raise ValueError("validated_repack action guidance requires test_time RTC")
    if not 1 <= args.execute_steps <= 30:
        raise ValueError("execute-steps must be in [1,30]")
    if not math.isfinite(args.pig_guidance_alpha) or not 0 <= args.pig_guidance_alpha <= 1:
        raise ValueError("RTC guidance alpha must be finite and in [0,1]")
    if args.rtc_mode == "test_time":
        if args.execute_steps == 30 or args.execute_steps * 50 % 30:
            raise ValueError("Test-time RTC requires a shorter window exactly aligned to 50Hz control ticks")
        horizon = args.execute_steps if args.min_exec_horizon is None else args.min_exec_horizon
        if not args.execute_steps <= horizon <= 30 or not 0 <= args.rtc_inference_delay <= 30-horizon:
            raise ValueError("RTC requires execute-steps <= min-exec-horizon and d+s <=30")
        if args.rtc_inference_delay != 0:
            raise ValueError("Synchronous paused-clock RTC requires zero inference delay")


if __name__ == "__main__":
    main()
