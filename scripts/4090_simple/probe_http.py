"""Replay captured real SIMPLE history through a localhost model service.

This CPU client never loads a model or runs CUDA. Successful requests verify
GPU-backed HTTP/history/reset behavior only when the separately started server
uses the genuine checkpoint. Replayed observations do not form live control or
an episode-success evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
import uuid

import numpy as np
import requests

from simple.baselines.client import RequestMessage, ResponseMessage


REPO = Path(__file__).resolve().parents[2]
BOUNDARY = ("real SIMPLE history replay through an independently started model service; "
            "HTTP/history/reset acceptance, not live control or episode-success evaluation")


def load_capture(path: Path):
    """Require capture_simple's provenance, one episode, native simulator time."""
    with np.load(path, allow_pickle=False) as data:
        rgb = data["rgb"]
        timestamps = data["sim_time_s"]
        state43 = data["state43"]
        episode_id = str(data["episode_id"].item())
        provenance = json.loads(str(data["source_metadata_json"].item()))
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:3]) < 1:
        raise ValueError("Capture must contain genuine nonempty uint8 RGB [T,H,W,3]")
    if timestamps.shape != (len(rgb),) or len(rgb) < 6 or not np.isfinite(timestamps).all():
        raise ValueError("Capture sim_time_s must be a finite [T] sequence")
    delta = np.diff(timestamps)
    if timestamps[0] < 0 or not (delta > 0).all() or delta.max() > 1 / 30 + .001:
        raise ValueError("Capture must contain advancing real simulator observations at >=30Hz")
    if timestamps[-1] - timestamps[0] < 1.0:
        raise ValueError("Capture needs at least one complete second of real history")
    if state43.shape != (len(rgb), 43) or not np.isfinite(state43).all():
        raise ValueError("Capture state43 must contain finite [T,43] mapped joint radians")
    if (not episode_id or provenance.get("episode_id") != episode_id
            or provenance.get("timestamp_source") != "MuJoCo mjData.time"
            or provenance.get("state_units") != "radians"
            or provenance.get("frame_count") != len(rgb)):
        raise ValueError("Capture provenance does not bind one real MuJoCo episode and radian state")
    targets = timestamps[-1] - np.arange(5, -1, -1) / 5
    selected = np.searchsorted(timestamps, targets + 1e-8, side="right") - 1
    if ((selected < 0).any() or len(set(selected.tolist())) != 6
            or np.max(targets - timestamps[selected]) > 1 / 30 + .001):
        raise ValueError("Capture does not supply six causal 5Hz target observations")
    return rgb, timestamps, state43, episode_id, provenance


def verify_info(info: dict) -> dict:
    action = info.get("action", {})
    history = info.get("online_history", {})
    required = info.get("expected_keys", {}).get("history", {})
    if (info.get("architecture", "dual_source_v1" if history else None) != "dual_source_v1"
            or history.get("frames") != 6
            or history.get("offsets_s") != [-1., -.8, -.6, -.4, -.2, 0.]
            or not {"session_id", "episode_id", "sim_time_s"}.issubset(required)):
        raise ValueError("/info must describe dual_source_v1 with real six-frame simulator history")
    if (action.get("action_dim"), action.get("action_chunk_size"),
            action.get("action_exec_horizon")) != (80, 30, 30):
        raise ValueError("/info must bind full returned 30x80 chunks for this probe")
    # Older service snapshots exposed padding in the existing state description;
    # newer snapshots may also expose its structured numerical value.
    state_dim = action.get("state_dim")
    if state_dim is None:
        description = info.get("expected_keys", {}).get("state", {}).get("states", "")
        match = re.search(r"padded to\s+(\d+)\b", description)
        state_dim = int(match.group(1)) if match else None
    if state_dim != 45:
        raise ValueError("/info must bind the model state dimension to 45")
    if info.get("rtc_enabled") or info.get("rtc_mode") != "off":
        raise ValueError("This full-window reset probe requires RTC off")
    if str(info.get("ckpt_step")) != "20000":
        raise ValueError("/info must identify ckpt_step=20000")
    if (history.get("observe_endpoint"), history.get("reset_endpoint"),
            history.get("warming_up_http_status")) != ("/observe", "/reset", 425):
        raise ValueError("/info observation/reset protocol differs")
    return {"architecture": "dual_source_v1", "state_dim": 45,
            "returned_action_shape": [30, 80], "rtc_mode": "off", "ckpt_step": 20000}


def replay(client, base_url: str, capture, *, instruction: str, timeout_s: float,
           report: dict, save):
    """Exercise the real wire contract without importing any model implementation."""
    rgb, timestamps, state43, source_episode, provenance = capture
    session_id = "http-probe-" + uuid.uuid4().hex
    episode_a, episode_b = "replay-a-" + uuid.uuid4().hex, "replay-b-" + uuid.uuid4().hex
    report.update(session_id=session_id, replay_episode_ids=[episode_a, episode_b],
                  source_episode_id=source_episode, instruction=instruction,
                  source_provenance=provenance, frames_per_replay=len(rgb),
                  sim_time_range_s=[float(timestamps[0]), float(timestamps[-1])],
                  requests=[], observe_summaries=[])

    def request(method, endpoint, *, payload=None, expected=200, name=None, summarize=False):
        started = time.perf_counter()
        response = client.request(method, base_url + endpoint, json=payload, timeout=timeout_s)
        elapsed = time.perf_counter() - started
        event = {"check": name or endpoint, "endpoint": endpoint,
                 "status_code": response.status_code, "expected_status_code": expected,
                 "wall_seconds": elapsed}
        if not summarize:
            report["requests"].append(event)
        if response.status_code != expected:
            report["failed_response"] = response.text[:2000]
            raise RuntimeError(f"{event['check']}: expected HTTP{expected}, got HTTP{response.status_code}")
        body = response.json()
        if expected in (425, 409):
            event["detail"] = body.get("detail")
        if expected == 425 and body.get("detail", {}).get("status") != "warming_up":
            raise RuntimeError("HTTP425 must identify genuine history warming_up")
        if not summarize:
            save()
        return body, elapsed

    def wire(index: int, episode: str):
        return RequestMessage(
            {"observation.images.egocentric": rgb[index]}, instruction,
            {"session_id": session_id, "episode_id": episode, "sim_time_s": float(timestamps[index])},
            {"states": state43[index:index + 1]}, {}, [], "simple", datetime.now(timezone.utc).isoformat(),
        ).serialize()

    def observe_remaining(episode: str):
        latencies = []
        for index in range(1, len(rgb)):
            body, elapsed = request("POST", "/observe", payload=wire(index, episode), summarize=True)
            if body.get("status") != "observed":
                raise RuntimeError("/observe did not acknowledge the genuine observation")
            latencies.append(elapsed)
        report["observe_summaries"].append({
            "episode_id": episode, "count": len(latencies), "wall_seconds": float(sum(latencies)),
            "maximum_request_wall_seconds": float(max(latencies))})
        save()

    def action(episode: str, label: str):
        body, elapsed = request("POST", "/act", payload=wire(len(rgb) - 1, episode), name=label)
        response = ResponseMessage.deserialize(body)
        values = np.asarray(response.action)
        if values.shape != (30, 80) or not np.isfinite(values).all() or response.err != 0.0:
            raise ValueError("/act must return finite denormalized 30x80 actions and err=0")
        report["requests"][-1].update(response_shape=list(values.shape), dtype=str(values.dtype),
                                      finite=True, action_wall_seconds=elapsed)
        save()
        return values.copy()

    info, _ = request("GET", "/info", name="service_contract")
    report["service_info"] = info
    report["verified_contract"] = verify_info(info)
    save()
    request("POST", "/reset", payload={"session_id": session_id, "episode_id": episode_a}, name="reset_a")
    request("POST", "/act", payload=wire(0, episode_a), expected=425, name="episode_a_first_frame_warms_up")
    observe_remaining(episode_a)
    first = action(episode_a, "episode_a_full_history_action")
    request("POST", "/act", payload=wire(len(rgb) - 1, episode_a), expected=409,
            name="same_timestamp_action_rejected")
    request("POST", "/reset", payload={"session_id": session_id, "episode_id": episode_b}, name="reset_b")
    request("POST", "/observe", payload=wire(len(rgb) - 1, episode_a), expected=409,
            name="old_episode_observation_rejected")
    request("POST", "/act", payload=wire(0, episode_b), expected=425,
            name="episode_b_first_frame_has_no_cross_episode_history")
    observe_remaining(episode_b)
    second = action(episode_b, "episode_b_full_history_action")
    return np.stack([first, second]), np.asarray([episode_a, episode_b])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--host", choices=("127.0.0.1", "localhost", "::1"), default="127.0.0.1")
    parser.add_argument("--port", type=int, default=21074)
    parser.add_argument("--output", type=Path, required=True, help="Absolute output directory outside the repository")
    parser.add_argument("--instruction", help="Defaults to the captured official task instruction")
    parser.add_argument("--timeout", type=float, default=180.)
    args = parser.parse_args()
    if not 0 < args.port < 65536 or not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("Invalid localhost port or timeout")
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(REPO):
        parser.error("--output must be an absolute directory outside the repository")
    args.output.mkdir(parents=True, exist_ok=True)
    report_path, actions_path = args.output / "http-history-reset.json", args.output / "http-actions.npz"
    if report_path.exists() or actions_path.exists():
        parser.error("Use a fresh output directory to preserve previous acceptance records")
    host = "[::1]" if args.host == "::1" else args.host
    report = {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
              "base_url": f"http://{host}:{args.port}", "boundary": BOUNDARY,
              "client_uses_gpu": False, "live_control": "not_run", "episode_success": "not_evaluated"}

    def save():
        temporary = report_path.with_suffix(".json.partial")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(report_path)

    save()
    try:
        args.history = args.history.resolve(strict=True)
        capture = load_capture(args.history)
        instruction = args.instruction or capture[4].get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Capture lacks a task instruction; supply --instruction explicitly")
        report.update(history_file=str(args.history), history_sha256=hashlib.sha256(args.history.read_bytes()).hexdigest())
        with requests.Session() as client:
            client.trust_env = False  # localhost must bypass all HTTP(S)_PROXY settings
            actions, episodes = replay(client, report["base_url"], capture, instruction=instruction,
                                       timeout_s=args.timeout, report=report, save=save)
        np.savez_compressed(actions_path, denormalized_actions=actions, replay_episode_ids=episodes,
                            source_sim_time_s=capture[1], source_episode_id=np.asarray(capture[3]))
        report.update(status="http_history_reset_replay_passed", finished_at=datetime.now(timezone.utc).isoformat(),
                      saved_action_shape=list(actions.shape), actions_file=str(actions_path))
        save()
        print(json.dumps(report, indent=2), flush=True)
    except Exception as error:
        report.update(status="failed", error=repr(error))
        save()
        raise


if __name__ == "__main__":
    main()
