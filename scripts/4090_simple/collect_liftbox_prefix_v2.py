"""New Lift teacher collection with a real deployment warmup observation prefix.

This producer never changes the v1 collector, task, controller or model server.
Rows0..29 contain real history observations, not supervised 30Hz commands:
initial PD has no latent, and live-pose warmup updates at50Hz. Row30 is the
first executed teacher command at original reference frame0. No state/controller
reset occurs at that transition. Raw capture closure never authorizes training.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time
import traceback
import uuid

import numpy as np

CONTRACT = "cvpr-simple-lift-real-warmup-prefix-v2"
TASK_ID = "simple/G1WholebodyLiftBoxHoldSonic-v0"
OBSERVATION_PHASE = "post_step_mj_forward_same_qpos"
WARMUP_TICKS = 50
PREFIX_ROWS = 30
REPO = Path(__file__).resolve().parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    from run_closed_loop import write_json as write
    write(Path(path), value)


def native_tick(row):
    if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or row < 0:
        raise ValueError("Row must be a nonnegative integer")
    return (int(row) * 50 + 29) // 30


def tick_plan(tick):
    if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)) or tick < 0:
        raise ValueError("Physical observation tick must be a nonnegative integer")
    row = int(tick) * 30 // 50
    teacher = tick >= WARMUP_TICKS
    return {"native_frame_index": int(tick), "event_index": row,
            "is_event": native_tick(row) == tick,
            "phase": "oracle_teacher_token" if teacher else "warmup_observation_prefix",
            "action_valid": teacher,
            "source_ref_frame": int(tick) - WARMUP_TICKS if teacher else None,
            "teacher_source_time_s": (int(tick) - WARMUP_TICKS) / 50 if teacher else None}


def vector(value, width, name):
    array = np.asarray(value)
    if array.shape != (width,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite{width}-vector")
    return array


def validate_events(events, *, require_window=True):
    """CPU schema/clock gate; physical/token/RGB audits remain separate."""
    if not events or (require_window and len(events) < PREFIX_ROWS + 30):
        raise ValueError("Need a real one-second prefix and thirty teacher labels")
    first = float(events[0]["sim_time_s"])
    if not math.isfinite(first):
        raise ValueError("Nonfinite observation clock")
    source = events[0]["source_episode"]
    for index, row in enumerate(events):
        tick = native_tick(index)
        expected = tick_plan(tick)
        if (row.get("event_index") != index or row.get("native_frame_index") != tick
                or row.get("observation_hz") != 30 or row.get("source_episode") != source
                or row.get("phase") != expected["phase"]
                or row.get("action_valid") is not expected["action_valid"]
                or row.get("source_ref_frame") != expected["source_ref_frame"]
                or row.get("teacher_source_time_s") != expected["teacher_source_time_s"]
                or row.get("observation_phase") != OBSERVATION_PHASE):
            raise ValueError("Prefix/teacher phase, source identity or ceil clock differs")
        now = float(row["sim_time_s"])
        if (not math.isfinite(now) or abs(now - first - tick / 50) > 1e-7
                or row.get("collection_elapsed_s") != tick / 50
                or row.get("nominal_timestamp_s") != index / 30):
            raise ValueError("Observation clock must follow genuine50Hz ceil-selected ticks")
        for key, width in (("state43", 43), ("base_pose7", 7), ("box_pose7", 7)):
            value = vector(row[key], width, key)
            if key.endswith("pose7") and not np.isclose(np.linalg.norm(value[3:]), 1, atol=1e-5):
                raise ValueError("Physical pose must contain a unit wxyz quaternion")
        if expected["action_valid"]:
            vector(row["executed_token64"], 64, "executed_token64")
            vector(row["hand_targets14"], 14, "hand_targets14")
            runtime = row.get("runtime_encoder", {})
            if (row.get("command_hz") != 30 or runtime.get("mode") != "feedback-heading-v11"
                    or runtime.get("source_frame_index") != expected["source_ref_frame"]
                    or runtime.get("changed_input_slice") != [584, 644]
                    or len(runtime.get("encoder_input_sha256", "")) != 64):
                raise ValueError("Teacher label lacks its actual official encoder reference")
        elif (row.get("executed_token64") is not None or row.get("hand_targets14") is not None
              or row.get("runtime_encoder") is not None or row.get("command_hz") is not None):
            raise ValueError("History-only warmup cannot claim supervised30Hz executed labels")
        if index >= 6 and abs(now - float(events[index - 6]["sim_time_s"]) - .2) > 1e-7:
            raise ValueError("Six observation rows must span actual0.2s")
    return source


def eligible_window(events, current, horizon=30):
    """Only teacher current+future is supervised; prefix supplies history only."""
    if current < PREFIX_ROWS or horizon != 30 or current + horizon > len(events):
        return False
    past = [current - offset for offset in (30, 24, 18, 12, 6, 0)]
    if any(index < 0 for index in past):
        return False
    times = np.asarray([events[index]["sim_time_s"] for index in past], dtype=float)
    if (not np.isfinite(times).all()
            or not np.allclose(times[-1] - times, [1., .8, .6, .4, .2, 0.], atol=1e-7, rtol=0)):
        return False
    future = events[current:current + horizon]
    return all(row.get("action_valid") is True and row.get("phase") == "oracle_teacher_token"
               and row.get("source_episode") == events[current]["source_episode"]
               and row.get("event_index") == current + offset
               and row.get("source_ref_frame") == native_tick(current + offset) - WARMUP_TICKS
               for offset, row in enumerate(future))


class PrefixTeacherCommands:
    """Share one live controller across warmup and teacher; no reset hook exists."""
    def __init__(self, teacher, provider, warmup, controller, hand_names):
        self.teacher, self.provider, self.warmup, self.controller = teacher, provider, warmup, controller
        self.hand_names = tuple(hand_names)
        self.token = self.hands = self.runtime = None
        self.control_phase = None
        self.previous_tick = -1

    def command(self, plan, raw, observation):
        tick = plan["native_frame_index"]
        if tick != self.previous_tick + 1:
            raise ValueError("Command dispatch requires every actual50Hz tick")
        self.previous_tick = tick
        if not plan["action_valid"]:
            command = self.warmup.command(raw.mujoco, self.controller, observation)
            self.control_phase = self.warmup.phase
            return command
        if tick == WARMUP_TICKS:
            if (len(self.controller._history) != WARMUP_TICKS or self.controller.decoder_calls != 40
                    or self.warmup.receipt["encoder_calls"] != 40):
                raise ValueError("Expected50 real proprioception samples,10PD+40live encoder/decoder warmup ticks")
        if plan["is_event"]:
            index = plan["source_ref_frame"]
            self.token, self.runtime = self.provider(index, raw)
            self.hands = dict(zip(self.hand_names, self.teacher["action"][index]))
        self.control_phase = "oracle_teacher_token"
        return self.controller.step(self.token, self.hands, observation)


class PrefixRecorder:
    """Lossless real observations with explicit unavailable prefix action labels."""
    def __init__(self, output, instruction, metadata):
        self.output = Path(output)
        if (not self.output.is_absolute() or self.output.exists()
                or self.output.resolve().is_relative_to(REPO)):
            raise ValueError("Use a fresh absolute capture directory outside the repository")
        (self.output / "rgb").mkdir(parents=True)
        shutil.copy2(__file__, self.output / "collector_source.py")
        self.instruction, self.metadata, self.events, self.closed = instruction, metadata, [], False

    def record(self, row, rgb):
        from PIL import Image
        if self.closed or row["event_index"] != len(self.events):
            raise ValueError("Capture requires each observation row exactly once")
        image = np.asarray(rgb)
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Require real RGB uint8 HWC without conversion")
        row = deepcopy(row)
        validate_events(self.events + [row], require_window=False)
        relative = Path("rgb") / f"event_{len(self.events):06d}.png"
        Image.fromarray(image).save(self.output / relative)
        row.update(rgb_lossless_png=str(relative), rgb_shape=list(image.shape), rgb_dtype="uint8",
                   rgb_sha256=hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest(),
                   rgb_png_file_sha256=sha256(self.output / relative))
        self.events.append(row)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        return row

    def close(self):
        if self.closed:
            return
        self.closed = True
        # No fabricated body arrays for the initial PD/history-only rows.
        keys = ("event_index", "native_frame_index", "sim_time_s", "collection_elapsed_s",
                "nominal_timestamp_s", "state43", "base_pose7", "box_pose7", "action_valid", "phase")
        np.savez_compressed(self.output / "observations.npz", **{
            key: np.asarray([row[key] for row in self.events]) for key in keys})
        from prepare_carrybox_v11 import TRAIN_STATE_NAMES, TRAIN_HAND_NAMES
        write_json(self.output / "capture.json", {"contract": CONTRACT, "state": "capture_closed",
            "instruction": self.instruction, "metadata": self.metadata, "event_count": len(self.events),
            "state43_joint_names": list(TRAIN_STATE_NAMES), "hand14_joint_names": list(TRAIN_HAND_NAMES),
            "prefix_rows": PREFIX_ROWS, "label_policy": "prefix action_valid=false; no latent labels; teacher only",
            "events_jsonl_sha256": sha256(self.output / "events.jsonl") if self.events else None,
            "observations_npz_sha256": sha256(self.output / "observations.npz"),
            "collector_sha256": sha256(self.output / "collector_source.py"),
            "acceptance": "raw capture only; independent physics/token/pixel audit required"})


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "contract", "data-root", "sonic-resources", "output", "record-observations"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=60., help="Teacher reference duration, excluding the fixed1s warmup")
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"), default="mujoco_isaac")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    if args.episode < 0 or not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("Need nonnegative source episode and finite positive teacher duration")
    if args.sim_mode != "mujoco_isaac" and not args.qualification_only:
        parser.error("EGL is qualification-only; pilot collection requires actual RTX")
    for key in ("source", "contract", "data_root", "sonic_resources"):
        setattr(args, key, getattr(args, key).resolve(strict=True))
    for key in ("output", "record_observations"):
        value = getattr(args, key)
        if not value.is_absolute() or value.exists() or value.resolve().is_relative_to(REPO):
            parser.error("Outputs must be fresh absolute directories outside the repository")
    a, b = args.output.resolve(), args.record_observations.resolve()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        parser.error("Run and capture directories must be separate")
    return args


def run(args):
    """Real simulator producer; deliberately not invoked by CPU contract tests."""
    os.environ["SIMPLE_DATA_DIR"] = str(args.data_root)
    import gymnasium as gym
    import mujoco
    import simple.envs
    import simple.baselines.sonic_v1_1 as controller_module
    from simple.baselines.dual_source_psi0 import TRAINING_HAND_JOINT_NAMES, TRAINING_STATE_JOINT_NAMES, state43_to45
    from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
    from simple.tasks import g1_wholebody_lift_box_hold_sonic as task_module
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from replay_carrybox_native50 import load_native, FeedbackHeadingEncoder
    from replay_carrybox_v11 import initialize_source_state, save_physics_model, box_contacts, finalize_replay_report
    from run_closed_loop import simulation_render_contract
    from video_capture import RawVideoRecorder

    args.output.mkdir(parents=True)
    seed = 10001 + args.episode
    result = {"contract": CONTRACT, "status": "running", "task_id": TASK_ID, "seed": seed,
        "purpose": "new teacher physical collection with real deployment warmup; not model evaluation",
        "run_role": "qualification" if args.qualification_only else "pilot_collection",
        "requested_sim_mode": args.sim_mode, "command_capture_root": str(args.record_observations),
        "finite_states": True, "fell": False, "task_success_observed": False,
        "warmup_seconds_requested": 1., "warmup_hand_policy": "open", "measurement_hz": 50,
        "observation_hz": 30, "command_hz": 30, "prefix_rows": PREFIX_ROWS,
        "observation_phase": OBSERVATION_PHASE, "training_authorized": False,
        "source_clock": "frozen until native tick50; teacher reference0 at dataset row30; no controller or state reset"}
    result["attempt_log"] = {"path": str(args.output.parent / "lift-box-prefix-v2-attempts.jsonl"),
                             "attempt_id": uuid.uuid4().hex}
    def attempt(phase):
        with Path(result["attempt_log"]["path"]).open("a") as stream:
            stream.write(json.dumps({"attempt_id": result["attempt_log"]["attempt_id"], "phase": phase,
                "source_episode": args.episode, "run_output": str(args.output), "status": result["status"],
                "run_role": result["run_role"], "frozen_contract_sha256": result.get("frozen_contract", {}).get("sha256")}) + "\n")
    path = args.output / "result.json"
    env = recorder = video = trace = None
    arrays = []
    started = time.perf_counter()
    try:
        teacher, receipt = load_native(args.source, args.episode, args.sonic_resources, args.contract, "feedback-heading-v11")
        provider = FeedbackHeadingEncoder(teacher, args.sonic_resources)
        result["teacher"] = receipt
        evidence = args.output / "teacher_encoder_compatibility.json"
        write_json(evidence, {"compatibility_verified": True, "decoder_sha256": controller_module.OFFICIAL_DECODER_SHA256,
            "body_token_version": "body_token_v1_1", "evidence": receipt,
            "scope": "fresh teacher labels only; no A100 token reproduction claim"})
        controller = controller_module.SonicV11OnnxController(args.sonic_resources, compatibility_evidence=evidence,
            pd_profile="official_v1_1_ankle_pitch_1_5", hand_torque_profile="mjcf_limits", sync_mujoco_observations=True)
        warmup = LivePoseSonicWarmup(args.sonic_resources, hand_policy="open")
        result["controller_identity"] = controller.identity
        config = SimLoopConfig().load_wbc_yaml()
        config.update(ENV_NAME="simple", SIMULATE_DT=.005, ENABLE_ELASTIC_BAND=False)
        env = gym.make(TASK_ID, sonic_config=config, sim_mode=args.sim_mode, headless=True, dr_level=0,
                       sync_mujoco_observations=True, isaac_observation_only=args.sim_mode == "mujoco_isaac")
        obs, info = env.reset(seed=seed)
        raw = env.unwrapped
        result["render_contract"] = simulation_render_contract(raw, args)
        result["instruction"] = raw.task.instruction
        result["environment_config_before_source_initialization"] = deepcopy(raw.task.state_dict())
        result["physics_model"] = save_physics_model(raw.mujoco.mjModel, raw.mujoco.mjSpec, args.output)
        result["initialization"] = initialize_source_state(raw, teacher)
        controller.reset()
        warmup.reset()
        obs, info = raw._get_obs(), raw._get_info()
        first = float(obs["sim_time_s"])
        result.update(initial_sim_time_s=first, planned_teacher_start_sim_time_s=first + 1., physics_dt_s=.005, decoder_hz=50)
        if not np.isclose(raw.mujoco.mjModel.opt.timestep, .005) or not np.isclose(raw.task.render_hz, 50):
            raise ValueError("Require200Hz physics and real50Hz camera/controller")
        result["task_contract"] = deepcopy(task_module.TASK_CONTRACT)
        result["controller_contract"] = {"physics_dt_s": .005, "decoder_hz": 50, "teacher_command_hz": 30,
            "warmup_control_hz": 50, "prefix_observation_hz": 30, "warmup_seconds": 1., "hand_policy": "open",
            "observation_phase": OBSERVATION_PHASE, "transition": "continuous; no reset; reference frame0 at row30",
            "pd_profile": controller.identity["pd_profile"], "hand_torque_profile": "mjcf_limits"}
        snapshots = {}
        source_paths = (Path(__file__), Path(controller_module.__file__), Path(task_module.__file__),
                        Path(__import__(LivePoseSonicWarmup.__module__, fromlist=["x"]).__file__),
                        REPO / "scripts/4090_simple/replay_carrybox_native50.py",
                        REPO / "scripts/4090_simple/replay_carrybox_v11.py")
        (args.output / "source_at_launch").mkdir()
        for source_path in source_paths:
            destination = args.output / "source_at_launch" / source_path.name
            shutil.copy2(source_path, destination)
            snapshots[str(source_path)] = {"sha256": sha256(destination), "snapshot": str(destination)}
        declaration = {"contract": CONTRACT, "task_contract": result["task_contract"],
            "controller_contract": result["controller_contract"], "render_contract": result["render_contract"],
            "source_files": snapshots, "robot_mjcf_sha256": sha256(args.data_root / raw.task.robot.mjcf_path)}
        frozen = args.output / "task_controller_contract.json"
        write_json(frozen, declaration)
        result["frozen_contract"] = {"path": str(frozen), "sha256": sha256(frozen)}
        attempt("started")
        write_json(path, result)
        recorder = PrefixRecorder(args.record_observations, raw.task.instruction,
            {"run_output": str(args.output), "source_episode": args.episode, "seed": seed,
             "task_id": TASK_ID, "run_role": result["run_role"], "frozen_contract": result["frozen_contract"]})
        if args.video:
            video = RawVideoRecorder(args.output / "oracle-replay.mp4", fps=50, shape=obs["head_stereo_left"].shape)
        trace = (args.output / "trace.jsonl").open("w", buffering=1)
        endpoint = min(float(args.seconds), float(receipt["replay_last_timestamp_s"]))
        policy = PrefixTeacherCommands(teacher, provider, warmup, controller, TRAINING_HAND_JOINT_NAMES)
        tick = 0

        def snapshot(phase):
            data, model = raw.mujoco.mjData, raw.mujoco.mjModel
            now = float(obs["sim_time_s"])
            if abs(now - first - tick / 50) > 1e-7:
                raise ValueError("Missing, repeated or shifted real50Hz observation tick")
            finite = all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.ctrl, data.xpos, data.xmat))
            if not finite:
                result["finite_states"] = False
                raise ValueError("Nonfinite actual physics")
            height = float(data.body("pelvis").xpos[2])
            tilt = float(np.degrees(np.arccos(np.clip(data.body("pelvis").xmat.reshape(3, 3)[2, 2], -1, 1))))
            result["fell"] |= height < .35 or tilt > 75
            result["minimum_pelvis_height_m"] = min(result.get("minimum_pelvis_height_m", height), height)
            result["maximum_pelvis_tilt_degrees"] = max(result.get("maximum_pelvis_tilt_degrees", tilt), tilt)
            state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
            q = state43_to45(obs["joint_qpos"], obs["joint_names"])[0, :43].copy()
            base = np.r_[data.body("pelvis").xpos, data.body("pelvis").xquat]
            box = np.r_[data.body("cube").xpos, data.body("cube").xquat]
            arrays.append((now, q, base.copy(), box.copy(), phase, state, tick))
            return now, q, base, box, height, tilt

        while True:
            plan = tick_plan(tick)
            if plan["action_valid"] and plan["teacher_source_time_s"] + .02 > endpoint + 1e-8:
                result["status"] = "source_end" if endpoint == receipt["replay_last_timestamp_s"] else "sim_time_limit"
                break
            now, q, base, box, height, tilt = snapshot(plan["phase"])
            if video:
                video.write(obs["head_stereo_left"], sim_time_s=now)
            if result["fell"]:
                result["status"] = "stopped_on_fall"
                break
            before = controller.decoder_calls
            command = policy.command(plan, raw, obs)
            token, hands, runtime, control_phase = policy.token, policy.hands, policy.runtime, policy.control_phase
            if tick == WARMUP_TICKS:
                result.update(actual_warmup_sim_seconds=now - first, teacher_start_sim_time_s=now,
                    warmup_decoder_calls=controller.decoder_calls, controller_reset_at_transition=False)
            actual_token = command["body_token"]
            actual_hands = command["hand_targets"]
            row = None
            if plan["is_event"]:
                row = {"event_index": plan["event_index"], "observation_hz": 30,
                    "nominal_timestamp_s": plan["event_index"] / 30, "native_frame_index": tick,
                    "collection_elapsed_s": tick / 50, "sim_time_s": now, "source_episode": args.episode,
                    "source_ref_frame": plan["source_ref_frame"], "teacher_source_time_s": plan["teacher_source_time_s"],
                    "phase": plan["phase"], "control_phase": control_phase, "action_valid": plan["action_valid"],
                    "observation_phase": OBSERVATION_PHASE, "state43": q.tolist(), "base_pose7": base.tolist(),
                    "box_pose7": box.tolist(), "command_hz": 30 if plan["action_valid"] else None,
                    "executed_token64": np.asarray(token).tolist() if plan["action_valid"] else None,
                    "hand_targets14": [float(hands[name]) for name in TRAINING_HAND_JOINT_NAMES] if plan["action_valid"] else None,
                    "runtime_encoder": runtime if plan["action_valid"] else None,
                    "alignment": "same real pre-command observation; post-step mj_forward/render; no time advance"}
                if raw.isaac is not None:
                    row["render_pose"] = deepcopy(raw.isaac.last_observation_pose_diagnostics)
                row = recorder.record(row, obs["head_stereo_left"])
            record = {"sim_time_s": now, "native_frame_index": tick, "phase": plan["phase"],
                "control_phase": control_phase, "source_ref_frame": plan["source_ref_frame"],
                "teacher_source_time_s": plan["teacher_source_time_s"], "observation_event_index": plan["event_index"] if plan["is_event"] else None,
                "observation_phase": OBSERVATION_PHASE, "decoder_calls_before_step": before,
                "applied_body_token": np.asarray(actual_token).tolist() if actual_token is not None else None,
                "applied_hand_targets": {name: float(value) for name, value in actual_hands.items()},
                "command_updated": bool(plan["action_valid"] and plan["is_event"]),
                "runtime_encoder": runtime if plan["action_valid"] and plan["is_event"] else None,
                "pelvis_height_m": height, "pelvis_tilt_degrees": tilt,
                "box_contacts": box_contacts(raw.mujoco.mjModel, raw.mujoco.mjData)}
            if raw.isaac is not None:
                record["render_pose"] = deepcopy(raw.isaac.last_observation_pose_diagnostics)
            obs, reward, terminated, truncated, info = env.step(command)
            if (abs(float(obs["sim_time_s"]) - now - .02) > 1e-7
                    or controller.decoder_calls - before != int(tick >= 10)):
                raise ValueError("Actual physics/control step differs from10PD then one decoder call per50Hz tick")
            record.update(next_sim_time_s=float(obs["sim_time_s"]), decoder_calls_after_step=controller.decoder_calls,
                reward=float(reward), terminated=bool(terminated), truncated=bool(truncated),
                task_progress=info.get("lift_box_hold_progress"))
            trace.write(json.dumps(record, allow_nan=False) + "\n")
            tick += 1
            if terminated or truncated:
                result.update(status="task_terminated" if terminated else "env_truncated",
                              task_success_observed=bool(terminated), terminated_after_teacher=bool(terminated and tick > 50))
                break
        # Exact terminal integration state is required even when it is not a30Hz event.
        if arrays and abs(arrays[-1][0] - float(obs["sim_time_s"])) < 1e-9:
            arrays.pop()
        snapshot("task_terminal")
        result.update(physical_outcome_status=result["status"], decoder_calls=controller.decoder_calls,
            final_task_progress=deepcopy(raw.task.progress), actual_sim_seconds=float(obs["sim_time_s"]) - first,
            actual_teacher_source_seconds=max(0., float(obs["sim_time_s"]) - first - 1.),
            warmup_control=deepcopy(warmup.receipt), observation_rows=len(recorder.events),
            teacher_command_events=max(0, len(recorder.events) - PREFIX_ROWS),
            oracle_success_without_fall=bool(result["task_success_observed"] and not result["fell"] and result["finite_states"]))
        if result["task_success_observed"] and not result.get("terminated_after_teacher"):
            raise ValueError("Warmup termination cannot be accepted as a teacher episode")
        validate_events(recorder.events, require_window=not args.qualification_only)
    except BaseException as error:
        result.setdefault("physical_outcome_status", result.get("status", "error"))
        result.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if trace is not None:
                trace.close()
            if arrays:
                keys = ("sim_time_s", "actual_state43", "actual_base_pose", "actual_box_pose", "phase", "physics_state_integration", "native_frame_index")
                saved = {key: np.stack([row[index] for row in arrays]) for index, key in enumerate(keys)}
                saved.update(state43_joint_names=np.asarray(TRAINING_STATE_JOINT_NAMES),
                             physics_state_spec=np.asarray(int(mujoco.mjtState.mjSTATE_INTEGRATION)))
                np.savez_compressed(args.output / "trajectory.npz", **saved)
                result["trajectory_sha256"] = sha256(args.output / "trajectory.npz")
            if video is not None:
                video.write(obs["head_stereo_left"], sim_time_s=float(obs["sim_time_s"]))
                result["video"] = video.close()
        except BaseException as error:
            result.update(status="error", artifact_close_error=repr(error))
            raise
        finally:
            result["wall_seconds"] = time.perf_counter() - started
            finalize_replay_report(path, result, recorder, env.close if env is not None else lambda: None,
                                   before_report=lambda: attempt("finished"))
    return 0 if (result.get("oracle_success_without_fall") or
                 (args.qualification_only and result["status"] in ("source_end", "sim_time_limit"))) else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
