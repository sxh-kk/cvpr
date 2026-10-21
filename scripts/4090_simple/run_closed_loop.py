"""Bounded local SIMPLE/20k/SONIC v1.1 integration smoke, not a benchmark score.

Start the real local model server separately. HTTP inference is synchronous:
MuJoCo and Isaac do not advance while waiting for it. Wall-clock latency is
recorded independently; 50 Hz simulation control is not a real-time claim.
Each task runs in a separate child to preserve Isaac SimulationApp lifecycle.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
import uuid

from capture_simple import load_fixed_episode
from video_capture import RawVideoRecorder


TASKS = ("G1WholebodyXMovePickTeleop-v0", "G1WholebodyXMoveBendPickTeleop-v0",
         "G1WholebodyHandoverTeleop-v0")
CARRYBOX_TASK = "G1WholebodyXMoveBendCarryBoxSonic-v0"
LIFTBOX_TASK = "G1WholebodyLiftBoxHoldSonic-v0"
SUPPORTED_TASKS = (*TASKS, CARRYBOX_TASK, LIFTBOX_TASK)


def simulation_render_contract(raw, args):
    """Bind actual render path separately from the authoritative physical loop."""
    import inspect
    isaac = raw.isaac is not None
    only = bool(getattr(raw,"isaac_observation_only",False))
    def source_hash(cls):
        return hashlib.sha256(Path(inspect.getsourcefile(cls)).read_bytes()).hexdigest()
    return {"sim_mode":"mujoco_isaac" if isaac else "mujoco",
        "render_engine":"Isaac RTX" if isaac else "MuJoCo EGL",
        "isaac_observation_only":only, "authoritative_physics":"MuJoCo",
        "sync_no_time_advance":bool(isaac and only and raw.sync_mujoco_observations),
        "measurement_hz":float(raw.task.render_hz),"render_hz":float(raw.task.render_hz),
        "geometry_precision":"full authoritative MuJoCo robot q/root, no decimal rounding; engine representation precision applies",
        "bridge_source_sha256":source_hash(type(raw.isaac)) if isaac else None,
        "environment_source_sha256":source_hash(type(raw))}


def _json_value(value):
    # Task state_dict includes NumPy table extents, light transforms and scalars.
    # Preserve their values without allowing NaN/Inf or arbitrary repr strings.
    if isinstance(value, Path):
        return str(value)
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def write_json(path: Path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False, default=_json_value) + "\n")
    temporary.replace(path)


def write_report_and_close(path: Path, result, close):
    """Keep failure evidence and release simulation resources even on JSON errors."""
    try:
        try:
            write_json(path, result)
        except Exception as error:
            write_json(path, {"status": "error", "report_serialization_error": repr(error),
                              "previous_status": str(result.get("status")),
                              "boundary": "report failed; existing trace/video are diagnostic artifacts only"})
            raise
    finally:
        close()


def worker_result_errors(result, expected_episodes: int):
    """Isaac shutdown may exit(0) after a Python failure; audit saved evidence."""
    errors = []
    if not isinstance(result, dict):
        return ["worker result is missing or is not a JSON object"]
    if result.get("status") != "bounded_smoke_finished":
        errors.append(f"worker status is {result.get('status')!r}")
    episodes = result.get("episodes")
    if not isinstance(episodes, list) or len(episodes) != expected_episodes or not episodes:
        return errors + [f"expected {expected_episodes} completed episodes"]
    terminal_statuses = {"sim_time_limit", "task_terminated", "env_truncated", "stopped_on_fall"}
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            errors.append(f"episode {index} is not an object")
            continue
        if episode.get("status") not in terminal_statuses:
            errors.append(f"episode {index} status is {episode.get('status')!r}")
        if episode.get("finite_states") is not True:
            errors.append(f"episode {index} did not retain finite physics")
        if episode.get("diagnostic_loop_executed") is not True or episode.get("decoder_calls", 0) < 1:
            errors.append(f"episode {index} did not execute the real model/controller loop")
        if episode.get("simulator_unchanged_during_http") is not True:
            errors.append(f"episode {index} did not confirm the synchronous simulation clock")
    return errors


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sonic-resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=21074)
    parser.add_argument("--expected-checkpoint-step", type=int, default=20000)
    parser.add_argument("--video", action="store_true", help="Record every genuine camera frame to a checked ffmpeg MP4")
    parser.add_argument("--sync-observations", action="store_true",
                        help="Opt in to synchronized integrated MuJoCo qpos/derived state/render observations")
    parser.add_argument("--isaac-observation-only", action="store_true",
                        help="Render Isaac only at each synchronized observation; keep all physics in MuJoCo")
    parser.add_argument("--initial-physics-run", type=Path,
                        help="Restore only row 0 of a same-task/same-seed run's complete physical state once, after exact model checks")
    parser.add_argument("--warmup-control", choices=("measured_pose_pd", "live_pose_sonic_v1_1"),
                        default="measured_pose_pd", help="Optional live-pose v1.1 balance reference while genuine RGB history fills")
    parser.add_argument("--body-token-projection", choices=("none", "fsq_v1_1"), default="none",
                        help="Optional denormalized body64 projection to the fixed SONIC v1.1 32-level FSQ grid")
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=list(TASKS))
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--dr-level", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--worker-wall-timeout", type=float, default=900.0)
    parser.add_argument("--fall-height", type=float, default=.35)
    parser.add_argument("--fall-tilt-degrees", type=float, default=75.)
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac", "isaacsim"), default="mujoco")
    parser.add_argument("--compatibility-evidence", type=Path)
    parser.add_argument("--diagnostic-unverified-sonic", action="store_true",
                        help="Explicitly allow official v1.1 with unverified training-encoder binding; no formal score")
    parser.add_argument("--pd-profile", choices=("fixed_cpp_baseline", "official_v1_1_ankle_pitch_1_5"),
                        default="fixed_cpp_baseline")
    parser.add_argument("--hand-torque-profile", choices=("mjcf_limits", "recording_0p7_nm"),
                        default="mjcf_limits")
    parser.add_argument("--worker-task", choices=SUPPORTED_TASKS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.isaac_observation_only and (not args.sync_observations or args.sim_mode == "mujoco"):
        parser.error("--isaac-observation-only requires --sync-observations and Isaac render mode")
    if args.warmup_control == "live_pose_sonic_v1_1" and not args.sync_observations:
        parser.error("--warmup-control live_pose_sonic_v1_1 requires --sync-observations")
    if args.initial_physics_run is not None:
        if (not args.initial_physics_run.is_absolute() or not args.sync_observations or
                args.episodes_per_task != 1 or len(args.tasks) != 1 or
                args.tasks[0] not in (CARRYBOX_TASK, LIFTBOX_TASK) or args.dr_level != 0):
            parser.error("--initial-physics-run needs an absolute source, synchronized observations, one local DR0 task and one episode")
        args.initial_physics_run = args.initial_physics_run.resolve(strict=True)
    args.data_root = args.data_root.resolve(strict=True)
    args.sonic_resources = args.sonic_resources.resolve(strict=True)
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("--output must be an absolute directory outside the repository")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in ("seconds", "http_timeout", "worker_wall_timeout", "fall_height", "fall_tilt_degrees"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive")
    if args.seconds < 1.2 or args.episodes_per_task < 1 or args.episode_start < 0 or not 1 <= args.execute_steps <= 30:
        parser.error("Need >=1.2 simulated seconds, positive episode count and execute-steps in [1,30]")
    return args


def run_worker(args):
    os.environ["SIMPLE_DATA_DIR"] = str(args.data_root)
    import numpy as np
    import gymnasium as gym
    from PIL import Image
    import simple.envs
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.baselines.dual_source_psi0 import (
        ACTION_FPS, DualSourceHttpActionClient, DualSourcePsi0Agent, HistoryWarmingUp,
    )
    from simple.baselines.sonic_v1_1 import SonicV11OnnxController

    class MeasuredClient(DualSourceHttpActionClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.events = []
            self.sim_clock = lambda: None

        def _post(self, endpoint, payload):
            event = {"endpoint": endpoint, "history": payload.get("history", {
                key: payload.get(key) for key in ("session_id", "episode_id")})}
            start, before = time.perf_counter(), self.sim_clock()
            try:
                result = super()._post(endpoint, payload)
                event["http_status"] = 200
                if endpoint == "/reset":
                    if result.get("history_frames") != 0 or result.get("rtc_cleared") is not True:
                        raise ValueError("Server reset did not confirm empty real history and RTC reset")
                    event["reset_response"] = result
                return result
            except HistoryWarmingUp as error:
                event.update(http_status=425, detail=error.detail)
                raise
            except Exception as error:
                event.update(http_status=getattr(getattr(error, "response", None), "status_code", None),
                             error=repr(error))
                raise
            finally:
                after = self.sim_clock()
                event.update(wall_seconds=time.perf_counter() - start,
                             simulator_before_s=before, simulator_after_s=after)
                self.events.append(event)
                if before is not None and after != before:
                    raise RuntimeError("Simulator advanced while the synchronous local HTTP request was pending")

    task_dir = args.output / args.worker_task
    task_dir.mkdir(parents=True, exist_ok=True)
    path = task_dir / "result.json"
    result = {"task": args.worker_task, "status": "running", "episodes": [],
              "boundary": "bounded integration smoke; not a formal success-rate benchmark",
              "synchronization": "paused simulation while synchronous localhost HTTP waits",
              "requested_sim_mode": args.sim_mode}
    result["observation_phase"] = ("post_step_mj_forward_same_qpos" if args.sync_observations
                                     else "legacy_mj_step_derived_phase")
    if args.initial_physics_run is not None:
        result["initial_physics_run"] = str(args.initial_physics_run)
    write_json(path, result)
    env, client = None, None
    try:
        controller = SonicV11OnnxController(args.sonic_resources,
            compatibility_evidence=args.compatibility_evidence,
            allow_unverified_compatibility=args.diagnostic_unverified_sonic, pd_profile=args.pd_profile,
            sync_mujoco_observations=args.sync_observations, hand_torque_profile=args.hand_torque_profile)
        result["controller_identity"] = controller.identity
        result["formal_evaluation_eligible"] = False
        client = MeasuredClient(server_ip="127.0.0.1", server_port=args.port, timeout_s=args.http_timeout)
        health = client.session.get(client.base_url + "/health", timeout=5)
        health.raise_for_status()
        result["server_health"] = health.json()
        server_info = client.session.get(client.base_url + "/info", timeout=5)
        server_info.raise_for_status()
        server_info = server_info.json()
        result["server_info"] = server_info
        action_contract = server_info.get("action", {})
        if (server_info.get("ckpt_step") != args.expected_checkpoint_step or
                action_contract.get("action_dim") != 80 or
                action_contract.get("action_chunk_size") != 30 or
                action_contract.get("action_exec_horizon") != 30 or
                server_info.get("online_history", {}).get("frames") != 6):
            raise ValueError("Smoke requires the requested checkpoint step, 6-frame history and 30x80 response")
        if server_info.get("rtc_enabled") is not False:
            raise ValueError("Bounded synchronous smoke requires RTC disabled; restart the service with RTC off")
        result["initial_metadata_get_requests"] = 2
        metadata = args.data_root / "eval" / args.worker_task / f"dr-level-{args.dr_level}" / "meta/episodes.jsonl"
        carrybox = args.worker_task in (CARRYBOX_TASK, LIFTBOX_TASK)
        if carrybox:
            if args.dr_level != 0:
                raise ValueError("CarryBox local v1 supports DR level 0 only")
            first_config = {"metadata": {"physics_dt": .005}}
        else:
            first_config, _ = load_fixed_episode(metadata, args.episode_start, args.data_root)
        config = SimLoopConfig().load_wbc_yaml()
        config["ENV_NAME"] = "simple"
        config["SIMULATE_DT"] = float(first_config["metadata"]["physics_dt"])
        mode = "mujoco_isaac" if args.sim_mode == "isaacsim" else args.sim_mode
        env = gym.make("simple/" + args.worker_task, sonic_config=config, sim_mode=mode,
                       headless=True, dr_level=args.dr_level,
                       sync_mujoco_observations=args.sync_observations,
                       isaac_observation_only=args.isaac_observation_only)
        raw = env.unwrapped
        result["render_contract"] = simulation_render_contract(raw,args)
        if args.sync_observations and raw.isaac is not None:
            result["isaac_observation_bridge"] = {"time_delta_s":0,
                "robot_pose_precision":"full authoritative MuJoCo values; no decimal rounding; Isaac/USD representation precision still applies"}
        client.sim_clock = lambda: float(raw.mujoco.mjData.time)
        warmup_control = None
        if args.warmup_control == "live_pose_sonic_v1_1":
            from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
            warmup_control = LivePoseSonicWarmup(args.sonic_resources)
        agent = DualSourcePsi0Agent(raw.task.robot, client=client, controller=controller,
                                   execute_steps=args.execute_steps, warmup_control=warmup_control,
                                   body_token_projection=args.body_token_projection)
        result["body_token_projection"] = deepcopy(agent.projection_receipt)
        for episode_index in range(args.episode_start, args.episode_start + args.episodes_per_task):
            seed = args.seed + episode_index - args.episode_start if carrybox else args.seed
            if carrybox:
                obs, info = env.reset(seed=seed)
                env_config = deepcopy(raw.task.state_dict())
                source_episode = {"kind": "independent_local_seed", "seed": seed,
                                  "task_adapter": raw.task.metadata["task_adapter"]}
            else:
                env_config, source_episode = load_fixed_episode(metadata, episode_index, args.data_root)
                obs, info = env.reset(seed=seed, options={"state_dict": env_config, "dr_level": args.dr_level})
            initial_physics = None
            if args.initial_physics_run is not None:
                from initial_physics_state import InitialPhysicsState
                initializer = InitialPhysicsState(args.initial_physics_run,
                    task_id="simple/" + args.worker_task, seed=seed)
                result["initial_physics_source"] = initializer.receipt
                obs, info, initial_physics = initializer.restore(raw)
            episode_id = f"{args.worker_task}:dr{args.dr_level}:episode{episode_index}:{uuid.uuid4().hex}"
            episode_dir = task_dir / f"episode-{episode_index}"
            episode_dir.mkdir(exist_ok=True)
            event_start = len(client.events)
            agent.reset(episode_id=episode_id)
            episode = {"episode_id": episode_id, "episode_index": episode_index, "seed": seed,
                "dr_level": args.dr_level, "source_episode": source_episode,
                "local_environment_config": env_config,
                "source_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest() if not carrybox else None,
                "physics_dt_s": float(raw.mujoco.mjModel.opt.timestep),
                "render_hz": float(raw.task.render_hz), "action_fps": ACTION_FPS,
                "sonic_decoder_hz": 50, "execute_steps": args.execute_steps,
                "controller_history_length_after_reset": len(controller._history),
                "agent_chunk_cleared_after_reset": agent._chunk is None,
                "finite_states": True, "fell": False, "task_success_observed": False,
                "warmup_frames": 0, "policy_frames": 0, "frames": 0,
                "fell_before_first_policy_command": False,
                "status": "running", "instruction": raw.task.instruction}
            if initial_physics is not None:
                episode["initial_physics"] = initial_physics
                episode["local_environment_config_scope"] = "same-seed reset model; actual initial coordinates/velocities are in initial_physics"
            pelvis_id = raw.mujoco.mjModel.body("pelvis").id
            episode["initial_pelvis_position_m"] = raw.mujoco.mjData.xpos[pelvis_id].tolist()
            episode["initial_pelvis_quaternion_wxyz"] = raw.mujoco.mjData.xquat[pelvis_id].tolist()
            result["episodes"].append(episode)
            write_json(path, result)
            first_time = float(obs["sim_time_s"])
            start = time.perf_counter()
            last_png_second, last_obs = -1, obs
            trace_path = episode_dir / "trace.jsonl"
            video = None
            try:
                if args.video:
                    video = RawVideoRecorder(episode_dir / "rollout.mp4", fps=raw.task.render_hz,
                                             shape=np.asarray(obs["head_stereo_left"]).shape)
                with trace_path.open("w", buffering=1) as trace:
                    while True:
                        sim_time = float(obs["sim_time_s"])
                        if video:
                            video.write(obs["head_stereo_left"], sim_time_s=sim_time)
                        data = raw.mujoco.mjData
                        finite = all(np.isfinite(value).all() for value in (
                            data.qpos, data.qvel, data.qacc, data.ctrl, data.xpos, data.xmat))
                        episode["finite_states"] = bool(episode["finite_states"] and finite)
                        if not finite:
                            episode["status"] = "nonfinite_physics"
                            break
                        pelvis = raw.mujoco.mjModel.body("pelvis").id
                        height = float(data.xpos[pelvis, 2])
                        tilt = float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3, 3)[2, 2], -1, 1))))
                        fell_now = height < args.fall_height or tilt > args.fall_tilt_degrees
                        if fell_now and not episode["fell"]:
                            episode["first_fall_sim_time_s"] = sim_time
                        episode["fell"] = bool(episode["fell"] or fell_now)
                        episode["minimum_pelvis_height_m"] = min(episode.get("minimum_pelvis_height_m", height), height)
                        episode["maximum_pelvis_tilt_degrees"] = max(episode.get("maximum_pelvis_tilt_degrees", tilt), tilt)
                        if episode["policy_frames"] == 0 and fell_now:
                            episode["fell_before_first_policy_command"] = True
                        episode["frames"] += 1
                        current_second = int(math.floor(sim_time - first_time))
                        if current_second > last_png_second:
                            Image.fromarray(np.asarray(obs["head_stereo_left"])).save(episode_dir / f"rgb-{current_second:02}.png")
                            last_png_second = current_second
                        if sim_time - first_time >= args.seconds:
                            episode["status"] = "sim_time_limit"
                            break
                        if fell_now and args.stop_on_fall:
                            episode["status"] = "stopped_on_fall"
                            break
                        before_action = time.perf_counter()
                        command = agent.get_action(obs, instruction=raw.task.instruction, info=info)
                        action_wall = time.perf_counter() - before_action
                        phase = agent.warmup_phase if agent.warming_up else "policy_latent"
                        if not agent.warming_up and episode["policy_frames"] == 0:
                            episode["first_policy_command_sim_time_s"] = sim_time
                            episode["first_policy_command_pelvis_height_m"] = height
                            episode["first_policy_command_pelvis_tilt_degrees"] = tilt
                            episode["warmup_decoder_calls"] = controller.decoder_calls
                        if agent.warming_up:
                            phase_counts = episode.setdefault("warmup_phase_counts", {})
                            phase_counts[phase] = phase_counts.get(phase, 0) + 1
                        episode["warmup_frames" if agent.warming_up else "policy_frames"] += 1
                        index = (None if agent._chunk_time is None else
                                 int(math.floor((sim_time - agent._chunk_time) * ACTION_FPS + 1e-7)))
                        record = {"sim_time_s": sim_time, "elapsed_sim_time_s": sim_time - first_time,
                            "phase": phase, "chunk_start_sim_time_s": agent._chunk_time,
                            "action_index_at_30hz": index, "action_type": command.type,
                            "action_wait_wall_seconds": action_wall, "decoder_calls_before_step": controller.decoder_calls,
                            "pelvis_height_m": height, "pelvis_tilt_degrees": tilt, "fell": fell_now}
                        obs, reward, terminated, truncated, info = env.step(command)
                        last_obs = obs
                        record.update(next_sim_time_s=float(obs["sim_time_s"]),
                                      decoder_calls_after_step=controller.decoder_calls,
                                      terminated=bool(terminated), truncated=bool(truncated), reward=float(reward))
                        if "carry_box_progress" in info or "lift_box_hold_progress" in info:
                            record["task_progress"] = info.get("lift_box_hold_progress", info.get("carry_box_progress"))
                        trace.write(json.dumps(record, allow_nan=False) + "\n")
                        if terminated or truncated:
                            episode["task_success_observed"] = bool(terminated)
                            episode["status"] = "task_terminated" if terminated else "env_truncated"
                            break
                if episode["status"] == "running":
                    episode["status"] = "stopped"
            except Exception as error:
                episode.update(status="error", error=repr(error), traceback=traceback.format_exc())
                raise
            finally:
                if video:
                    try:
                        video.write(last_obs["head_stereo_left"], sim_time_s=float(last_obs["sim_time_s"]))
                        episode["video"] = video.close()
                    except Exception as error:
                        episode.update(status="error", video_error=repr(error))
                        try:
                            video.close()
                        finally:
                            raise
                elapsed = float(last_obs["sim_time_s"]) - first_time
                final_data = raw.mujoco.mjData
                final_finite = all(np.isfinite(value).all() for value in (
                    final_data.qpos, final_data.qvel, final_data.qacc, final_data.ctrl,
                    final_data.xpos, final_data.xmat))
                episode["finite_states"] = bool(episode["finite_states"] and final_finite)
                if final_finite:
                    final_height = float(final_data.xpos[pelvis_id, 2])
                    final_tilt = float(np.degrees(np.arccos(np.clip(
                        final_data.xmat[pelvis_id].reshape(3, 3)[2, 2], -1, 1))))
                    final_fall = final_height < args.fall_height or final_tilt > args.fall_tilt_degrees
                    if final_fall and not episode["fell"]:
                        episode["first_fall_sim_time_s"] = float(last_obs["sim_time_s"])
                    episode["fell"] = bool(episode["fell"] or final_fall)
                    episode["minimum_pelvis_height_m"] = min(episode.get("minimum_pelvis_height_m", final_height), final_height)
                    episode["maximum_pelvis_tilt_degrees"] = max(episode.get("maximum_pelvis_tilt_degrees", final_tilt), final_tilt)
                    episode["final_pelvis_position_m"] = final_data.xpos[pelvis_id].tolist()
                    episode["final_pelvis_quaternion_wxyz"] = final_data.xquat[pelvis_id].tolist()
                else:
                    episode["final_pelvis_position_m"] = None
                    episode["final_pelvis_quaternion_wxyz"] = None
                events = client.events[event_start:]
                counts = Counter(event["endpoint"] for event in events)
                latencies = [event["wall_seconds"] for event in events if event["endpoint"] == "/act" and event.get("http_status") == 200]
                episode.update(simulated_seconds=elapsed, wall_seconds=time.perf_counter() - start,
                    http_counts=dict(counts), http_425_count=sum(event.get("http_status") == 425 for event in events),
                    policy_chunks=sum(event["endpoint"] == "/act" and event.get("http_status") == 200 for event in events),
                    decoder_calls=controller.decoder_calls, clipped_torque_elements=controller.clipped_torque_elements,
                    http_wall_seconds=sum(event["wall_seconds"] for event in events),
                    successful_action_http_latencies_s=latencies,
                    simulator_unchanged_during_http=all(event["simulator_before_s"] == event["simulator_after_s"] for event in events),
                    task_success_attributed_to_policy=bool(episode["task_success_observed"] and not episode["fell"] and
                        episode["finite_states"] and controller.decoder_calls > 0 and
                        controller.identity["compatibility_status"] == "verified"),
                    diagnostic_loop_executed=bool(controller.decoder_calls > 0 and episode["policy_frames"] > 0),
                    formal_evaluation_eligible=False)
                if carrybox:
                    episode["final_task_progress"] = dict(raw.task.progress)
                episode["warmup_control"] = (deepcopy(warmup_control.receipt) if warmup_control is not None
                    else {"kind":"measured_pose_pd", "balance_decoder_during_history":False})
                episode["warmup_control"]["actual_warmup_decoder_calls"] = episode.get("warmup_decoder_calls", controller.decoder_calls)
                episode["body_token_projection"] = deepcopy(agent.projection_receipt)
                episode["strategy_quality_assessment"] = (
                    "ineligible: robot already unstable before first learned action" if episode["fell_before_first_policy_command"]
                    else "ineligible: bounded integration smoke with explicit controller compatibility boundary")
                write_json(episode_dir / "http.json", events)
                Image.fromarray(np.asarray(last_obs["head_stereo_left"])).save(episode_dir / "rgb-last.png")
                write_json(path, result)
                print(json.dumps({"task": args.worker_task, "episode": episode_index,
                                  "status": episode["status"], "simulated_seconds": elapsed,
                                  "decoder_calls": controller.decoder_calls, "fell": episode["fell"]}), flush=True)
        result["status"] = "bounded_smoke_finished"
    except Exception as error:
        result.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        result["http_request_count"] = len(client.events) if client else 0
        def close_resources():
            try:
                if client:
                    client.close()
            finally:
                if env is not None:
                    env.close()
        write_report_and_close(path, result, close_resources)


def supervise(args):
    summary = {"status": "running", "tasks": [],
               "boundary": "sequential bounded integration smoke, not a formal evaluation score",
               "synchronization": "simulation paused during local HTTP inference; no real-time claim"}
    path = args.output / "summary.json"
    write_json(path, summary)
    original = sys.argv[1:]
    for task in args.tasks:
        task_dir = args.output / task
        task_dir.mkdir(exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), *original, "--worker-task", task]
        record = {"task": task, "status": "running", "log": str(task_dir / "worker.log")}
        summary["tasks"].append(record)
        write_json(path, summary)
        print(json.dumps({"task": task, "status": "worker_started", "log": record["log"]}), flush=True)
        with (task_dir / "worker.log").open("w") as log:
            try:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                           timeout=args.worker_wall_timeout, check=False)
                record.update(status="finished" if completed.returncode == 0 else "error", returncode=completed.returncode)
            except subprocess.TimeoutExpired:
                record["status"] = "worker_wall_timeout"
        result_path = task_dir / "result.json"
        try:
            record["result"] = json.loads(result_path.read_text()) if result_path.is_file() else None
            errors = worker_result_errors(record["result"], args.episodes_per_task)
        except (OSError, ValueError) as error:
            errors = [f"Cannot read worker result: {error!r}"]
        record["result_validation_errors"] = errors
        if errors and record["status"] == "finished":
            record["status"] = "error"
        write_json(path, summary)
    summary["status"] = "bounded_smoke_finished" if all(task["status"] == "finished" for task in summary["tasks"]) else "errors_recorded"
    write_json(path, summary)
    print(json.dumps({"status": summary["status"], "summary": str(path)}), flush=True)
    return 0 if summary["status"] == "bounded_smoke_finished" else 1


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.worker_task:
        run_worker(arguments)
    else:
        raise SystemExit(supervise(arguments))
