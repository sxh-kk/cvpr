"""One physical episode per production server process, with post-load RNG reset/audit."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import threading
import time
import numpy as np
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
               "rng_sha256": rng_sha(), "host": "127.0.0.1", "Ta": 30, "rtc": False, "inference_steps": 10}
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
            started = time.monotonic()
            try:
                claim(history)
                response = original_predict(payload, http_request)
                row["status"] = response.status_code
                return response
            except BaseException as error:
                row["status"] = getattr(error, "status_code", "exception")
                row["exception_type"] = type(error).__name__
                raise
            finally:
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
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(repo) or args.output.exists():
        raise ValueError("Use a fresh absolute audit directory outside the repository")
    sys.path[:0] = [str(repo / "VLMExpert/src"), str(repo)]
    from psi.deploy.serve_psi0_simple import Server
    from psi.utils import seed_everything
    from finetune_carrybox import sha256
    weights = args.run_dir / "checkpoints" / f"ckpt_{args.ckpt_step}" / "model.safetensors"
    server = audited_server(lambda: Server("psi0", args.run_dir, args.ckpt_step, "cuda:0", enable_rtc=False,
                            action_exec_horizon=30, rtc_mode="off", num_inference_steps=10),
                            args.sampling_seed, seed_everything, args.output, sha256(weights))
    server.run("127.0.0.1", args.port)


if __name__ == "__main__":
    main()
