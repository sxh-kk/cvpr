"""Physical CarryBox oracle replay of freshly encoded v1.1 demonstration labels.

This is a teacher-reference/controller diagnostic, not learned model performance.
Only the initial state is set from the recording. Subsequent frames are produced
by real MuJoCo dynamics, official 50 Hz SONIC decoding and 200 Hz torque PD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from copy import deepcopy
import time
import traceback
import uuid

import numpy as np

from run_closed_loop import CARRYBOX_TASK, LIFTBOX_TASK, write_json, write_report_and_close, simulation_render_contract
from video_capture import RawVideoRecorder

ENCODER_SHA256 = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _feature_names(feature):
    names = feature.get("names")
    if isinstance(names, dict) and len(names) == 1:
        names = next(iter(names.values()))
    return names


def load_teacher(dataset_root, episode_index, conversion_provenance, contract_path):
    import pyarrow.parquet as pq
    from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, TRAINING_STATE_JOINT_NAMES, TRAINING_HAND_JOINT_NAMES

    info = json.loads((dataset_root / "meta/info.json").read_text())
    features = info["features"]
    if _feature_names(features["observation.state"]) != list(TRAINING_STATE_JOINT_NAMES):
        raise ValueError("Converted teacher must declare all 43 training-order joint names")
    if _feature_names(features["action"]) != list(TRAINING_HAND_JOINT_NAMES):
        raise ValueError("Converted teacher must declare the 14 training-order hand action names")
    if _feature_names(features["observation.body_joint_velocities"]) != list(BODY_JOINT_NAMES):
        raise ValueError("Converted body qvel must declare the physical body29 name order")
    if info.get("fps") != 30 or "action.body_token_v1_1" not in features:
        raise ValueError("Teacher must be the explicit 30 Hz v1.1 conversion, never the legacy token bundle")
    provenance_text = conversion_provenance.read_text()
    provenance = json.loads(provenance_text)
    if provenance.get("encoder_sha256") != ENCODER_SHA256:
        raise ValueError("Conversion provenance does not identify the fixed official v1.1 encoder SHA")
    if (provenance.get("output_state_names") != list(TRAINING_STATE_JOINT_NAMES) or
            provenance.get("output_action_hand_names") != list(TRAINING_HAND_JOINT_NAMES)):
        raise ValueError("Conversion provenance and output named state/action contracts disagree")
    if provenance.get("token_reproduction_performed") is not True:
        raise ValueError("Teacher conversion must pass the independent encoder token reproduction check")
    if provenance.get("state_action_contract_sha256") != sha256(contract_path):
        raise ValueError("Source state/action audit changed since the conversion receipt; use its fixed snapshot")
    contract = json.loads(contract_path.read_text())
    names = contract.get("joint_names_canonical", contract.get("source_joint_names"))
    if not isinstance(names, list) or len(names) != 43 or set(names) != set(TRAINING_STATE_JOINT_NAMES):
        raise ValueError("Source state/action contract lacks the independently audited 43 named joints")
    matches = list(dataset_root.glob(f"data/**/episode_{episode_index:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError(f"Expected one converted episode {episode_index}; found {len(matches)}")
    path = matches[0]
    table = pq.read_table(path)
    widths = {"observation.state": 43, "action": 14, "action.body_token_v1_1": 64,
              "observation.base_pose": 7, "observation.base_vel": 6,
              "observation.body_joint_velocities": 29, "observation.object_poses": 7}
    values = {}
    for name, width in widths.items():
        if name not in table.column_names:
            raise ValueError(f"Converted teacher is missing required real source column: {name}")
        value = np.asarray(table[name].to_pylist(), dtype=np.float64)
        if value.shape != (table.num_rows, width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid teacher feature {name}: {value.shape}")
        values[name] = value
    for name in ("timestamp", "source_episode_index", "source_frame_index", "source_timestamp"):
        if name not in table.column_names:
            raise ValueError(f"Converted teacher lost source identity column {name}")
        values[name] = np.asarray(table[name].to_pylist())
    timestamps = values["timestamp"].astype(float)
    if len(timestamps) < 2 or not np.allclose(timestamps, np.arange(len(timestamps))/30, atol=1e-5, rtol=0):
        raise ValueError("Converted timestamps must explicitly use 30 Hz from zero")
    if len(np.unique(values["source_episode_index"])) != 1:
        raise ValueError("Teacher episode mixes source episodes")
    for name in ("source_episode_index", "source_frame_index", "source_timestamp"):
        if not np.isfinite(values[name]).all():
            raise ValueError(f"Nonfinite source provenance: {name}")
    for name in ("source_episode_index", "source_frame_index"):
        if not np.all(values[name] == values[name].astype(np.int64)):
            raise ValueError(f"Source identity must contain integral indices: {name}")
    if np.any(np.diff(values["source_frame_index"]) <= 0) or np.any(np.diff(values["source_timestamp"]) <= 0):
        raise ValueError("Teacher source frame/timestamp provenance is not chronological")
    for name in ("observation.base_pose", "observation.object_poses"):
        if not np.allclose(np.linalg.norm(values[name][:, 3:], axis=1), 1, atol=1e-4, rtol=0):
            raise ValueError(f"Source {name} quaternion is not unit length")
    receipt = {"converted_parquet": str(path), "converted_parquet_sha256": sha256(path),
               "converted_info_sha256": sha256(dataset_root / "meta/info.json"),
               "conversion_provenance": str(conversion_provenance), "conversion_provenance_sha256": sha256(conversion_provenance),
               "state_action_contract": str(contract_path), "state_action_contract_sha256": sha256(contract_path),
               "source_episode_index": int(values["source_episode_index"][0]), "frames": len(timestamps),
               "converted_last_timestamp_s": (len(timestamps)-1)/30,
               "stored_last_timestamp_s": float(timestamps[-1]), "encoder_sha256": ENCODER_SHA256}
    return values, receipt


def initialize_source_state(raw, teacher):
    """The sole kinematic state assignment, before any controller history exists."""
    import mujoco
    from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, TRAINING_STATE_JOINT_NAMES

    model, data = raw.mujoco.mjModel, raw.mujoco.mjData
    initial_q = teacher["observation.state"][0]
    initial_base = teacher["observation.base_pose"][0]
    initial_object = teacher["observation.object_poses"][0]
    pre = {"base_pose": data.qpos[:7].tolist(), "object_pose": np.r_[data.body("cube").xpos, data.body("cube").xquat].tolist()}
    root_joint = model.joint("floating_base_joint")
    if int(root_joint.type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("Teacher replay requires the real floating G1 root")
    root_q, root_dq = int(root_joint.qposadr[0]), int(root_joint.dofadr[0])
    data.qpos[root_q:root_q+7] = initial_base
    data.qvel[:] = 0
    data.qvel[root_dq:root_dq+6] = teacher["observation.base_vel"][0]
    for name, value in zip(TRAINING_STATE_JOINT_NAMES, initial_q):
        data.joint(name).qpos[0] = value
    for name, value in zip(BODY_JOINT_NAMES, teacher["observation.body_joint_velocities"][0]):
        data.joint(name).qvel[0] = value
    data.joint("cube_joint").qpos[:] = initial_object
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)
    raw.task._reset_progress()
    raw.task.layout.actors["target"].pose.position = initial_object[:3].tolist()
    raw.task.layout.actors["target"].pose.quaternion = initial_object[3:].tolist()
    if raw.isaac is not None:
        if getattr(raw,"isaac_observation_only",False):
            raw.isaac.refresh_from_mujoco(raw.mujoco,update_kinematics=True)
        else:
            raw.isaac.step(raw.mujoco)
    return {"initialization_count": 1, "before_source_assignment": pre,
            "source_state43": initial_q.tolist(), "source_base_pose": initial_base.tolist(),
            "source_base_vel_qvel_order": teacher["observation.base_vel"][0].tolist(),
            "source_object_pose": initial_object.tolist(),
            "initial_hand_joint_velocities": "zero; source does not provide measured hand qvel",
            "initial_body_joint_velocities": "source50Hz numerical derivative from converted data",
            "initial_object_velocity": "zero; source does not record object velocity",
            "unrecovered_source_domain": "training archive has no scene/material/lighting state; local task uses fixed official scene0/table recipe",
            "subsequent_state_assignment": "none; physical PD and decoder only"}


def quaternion_error_degrees(actual, reference):
    cosine = np.clip(abs(np.dot(actual, reference)) / (np.linalg.norm(actual)*np.linalg.norm(reference)), 0, 1)
    return float(np.degrees(2*np.arccos(cosine)))


def box_contacts(model, data):
    """Actual current MuJoCo contacts, including palms fused into wrist bodies."""
    import mujoco
    target = model.body("cube").id
    contacts = []
    for index in range(data.ncon):
        contact = data.contact[index]
        a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if target not in (a, b):
            continue
        other_geom = int(contact.geom2 if a == target else contact.geom1)
        other_body = model.body(int(model.geom_bodyid[other_geom])).name
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        contacts.append({"other_body": other_body, "other_geom_id": other_geom,
                         "point_m": contact.pos.tolist(), "distance_m": float(contact.dist),
                         "normal_force_n": float(force[0]),
                         "hand_or_palm": "hand" in other_body or "wrist_yaw" in other_body})
    return contacts


def save_physics_model(model, spec, output):
    """Save the exact compiled model, including meshes, and verify CPU reload.

    Attached MuJoCo 3.3 MjSpec XML can lose mesh directories and write an
    unnamed nested default. Its XML is retained as a diagnostic spec dump;
    the independently reloadable MJB is the authoritative physical model.
    """
    import mujoco
    binary_path = output / "physics_model.mjb"
    mujoco.mj_saveModel(model, str(binary_path))
    loaded = mujoco.MjModel.from_binary_path(str(binary_path))
    if (loaded.nq, loaded.nv, loaded.nu) != (model.nq, model.nv, model.nu):
        raise RuntimeError("Saved physics MJB roundtrip changed the dimensions")
    for field in ("body_mass", "body_inertia", "body_pos", "body_quat", "jnt_type", "jnt_qposadr",
                  "geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction", "actuator_gear", "mesh_vert"):
        if not np.array_equal(getattr(model, field), getattr(loaded, field)):
            raise RuntimeError(f"Saved physics MJB roundtrip changed {field}")
    if loaded.opt.timestep != model.opt.timestep or not np.array_equal(loaded.opt.gravity, model.opt.gravity):
        raise RuntimeError("Saved physics MJB roundtrip changed timestep/gravity")
    xml_path = output / "physics_spec_dump.xml"
    xml_path.write_text(spec.to_xml())
    return {"path": str(binary_path), "sha256": sha256(binary_path), "format": "compiled MuJoCo MJB",
            "cpu_reload_verified": True, "nq": model.nq, "nv": model.nv, "nu": model.nu,
            "state_storage": "trajectory.npz; MJB stores model rather than time-varying state",
            "diagnostic_spec_xml": str(xml_path), "diagnostic_spec_xml_sha256": sha256(xml_path),
            "diagnostic_xml_limitation": "attached MjSpec XML may lose mesh directories/named default classes; use MJB for exact reload"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="One converted train/val/test split directory")
    parser.add_argument("--conversion-provenance", type=Path, required=True)
    parser.add_argument("--state-action-contract", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sonic-resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--task", choices=(CARRYBOX_TASK,LIFTBOX_TASK), default=CARRYBOX_TASK)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup-seconds", type=float, default=.2)
    parser.add_argument("--seed", type=int, default=10001)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--sync-observations", action="store_true")
    parser.add_argument("--isaac-observation-only", action="store_true")
    parser.add_argument("--qualification-only", action="store_true")
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac", "isaacsim"), default="mujoco")
    parser.add_argument("--pd-profile", choices=("fixed_cpp_baseline", "official_v1_1_ankle_pitch_1_5"),
                        default="official_v1_1_ankle_pitch_1_5")
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--fall-height", type=float, default=.35)
    parser.add_argument("--fall-tilt-degrees", type=float, default=75)
    args = parser.parse_args()
    if args.isaac_observation_only and (not args.sync_observations or args.sim_mode == "mujoco"):
        parser.error("Observation-only Isaac requires synchronized Isaac render mode")
    for field in ("dataset_root", "conversion_provenance", "state_action_contract", "data_root", "sonic_resources"):
        setattr(args, field, getattr(args, field).resolve(strict=True))
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("--output must be an absolute directory outside the repository")
    for field in ("seconds", "warmup_seconds", "fall_height", "fall_tilt_degrees"):
        if not math.isfinite(getattr(args, field)) or getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be finite and positive")
    if args.warmup_seconds < .2 or args.episode < 0:
        parser.error("Need >=.2s genuine proprioception warmup and a nonnegative episode index")
    args.output.mkdir(parents=True, exist_ok=True)
    return args


def command_event_index(source_time_s, command_hz):
    """Command clock on real 50 Hz observation ticks, with no interpolation."""
    if command_hz not in (30, 50):
        raise ValueError("Command rate must be 30 or 50 Hz")
    ordinal = int(math.floor(source_time_s * command_hz + 1e-7))
    native_frame = (ordinal * 50 + command_hz - 1) // command_hz
    return ordinal, native_frame


def finalize_replay_report(path, result, command_recorder, close_environment, before_report=None):
    """Seal raw capture and write its failure state before Isaac can exit(0)."""
    try:
        if command_recorder is not None:
            try:
                command_recorder.close()
            except Exception as error:
                result.update(status="error", command_capture_close_error=repr(error))
                raise
    finally:
        try:
            if before_report is not None:
                before_report()
        finally:
            write_report_and_close(path, result, close_environment)


def run(args, *, teacher_loader=load_teacher, action_hz=30, continue_after_termination=False,
        token_provider=None, command_hz=50, command_recorder=None):
    os.environ["SIMPLE_DATA_DIR"] = str(args.data_root)
    from PIL import Image
    import mujoco
    import gymnasium as gym
    import simple.envs
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.baselines.dual_source_psi0 import TRAINING_HAND_JOINT_NAMES, TRAINING_STATE_JOINT_NAMES, state43_to45
    from simple.baselines.sonic_v1_1 import OFFICIAL_DECODER_SHA256, SonicV11OnnxController

    task_name = getattr(args,"task",CARRYBOX_TASK)
    lift_task = task_name == LIFTBOX_TASK
    result = {"status": "running", "purpose": "oracle teacher-reference physical replay; not learned policy evaluation",
              "formal_model_evaluation_eligible": False, "task": task_name, "task_id":"simple/"+task_name, "seed": args.seed,
              "requested_sim_mode": args.sim_mode, "finite_states": True, "fell": False,
              "task_success_observed": False, "warmup_camera_frames": 0, "teacher_control_frames": 0,
              "command_hz": command_hz, "command_events": 0,
              "run_role":"qualification" if getattr(args,"qualification_only",False) else ("pilot_collection" if lift_task else "controller_diagnostic"),
              "observation_phase":("post_step_mj_forward_same_qpos" if getattr(args,"sync_observations",False)
                                     else "legacy_mj_step_derived_phase"),
              "source_clock": "starts after genuine measured-pose proprioception warmup; no source-time advances during warmup"}
    path = args.output / "result.json"
    if command_recorder is not None:
        result["command_capture_root"] = str(args.record_commands)
    if lift_task:
        result["attempt_log"] = {"path":str(args.output.parent / "lift-box-attempts.jsonl"),"attempt_id":uuid.uuid4().hex}
    def log_attempt(phase):
        if lift_task:
            with Path(result["attempt_log"]["path"]).open("a") as stream:
                stream.write(json.dumps({"attempt_id":result["attempt_log"]["attempt_id"],"phase":phase,
                    "source_episode":args.episode,"run_output":str(args.output),"status":result["status"],
                    "run_role":result["run_role"],"physical_outcome_status":result.get("physical_outcome_status"),
                    "frozen_contract_sha256":result.get("frozen_contract",{}).get("sha256")})+"\n")
    write_json(path, result)
    env = video = None
    measurements = []
    physics_states = []
    started = time.perf_counter()
    try:
        teacher, receipt = teacher_loader(args.dataset_root, args.episode, args.conversion_provenance, args.state_action_contract)
        result["teacher"] = receipt
        token_key = receipt.get("action_token_key", "action.body_token_v1_1")
        last_timestamp = receipt.get("replay_last_timestamp_s", receipt.get("converted_last_timestamp_s"))
        if receipt.get("compatibility_verified", True):
            evidence_path = args.output / "teacher_encoder_compatibility.json"
            write_json(evidence_path, {"body_token_version": "body_token_v1_1", "decoder_sha256": OFFICIAL_DECODER_SHA256,
                       "compatibility_verified": True, "evidence": receipt,
                       "scope": "freshly encoded teacher dataset only; does not establish A100 model training-token identity"})
            controller = SonicV11OnnxController(args.sonic_resources, compatibility_evidence=evidence_path, pd_profile=args.pd_profile,
                hand_torque_profile=getattr(args, "hand_torque_profile", "mjcf_limits"),
                sync_mujoco_observations=getattr(args,"sync_observations",False))
        else:
            controller = SonicV11OnnxController(args.sonic_resources, allow_unverified_compatibility=True, pd_profile=args.pd_profile,
                hand_torque_profile=getattr(args, "hand_torque_profile", "mjcf_limits"),
                sync_mujoco_observations=getattr(args,"sync_observations",False))
            result["purpose"] = "unverified recorded-token controller diagnostic; not model evaluation or training data acceptance"
        result["source_token_field"] = token_key
        result["controller_identity"] = controller.identity
        config = SimLoopConfig().load_wbc_yaml()
        config.update(ENV_NAME="simple", SIMULATE_DT=.005, ENABLE_ELASTIC_BAND=False)
        env = gym.make("simple/" + task_name, sonic_config=config,
                       sim_mode="mujoco_isaac" if args.sim_mode == "isaacsim" else args.sim_mode,
                       headless=True, dr_level=0,
                       sync_mujoco_observations=getattr(args,"sync_observations",False),
                       isaac_observation_only=getattr(args,"isaac_observation_only",False))
        obs, info = env.reset(seed=args.seed)
        raw = env.unwrapped
        result["render_contract"] = simulation_render_contract(raw,args)
        result["instruction"] = raw.task.instruction
        if getattr(args,"sync_observations",False) and raw.isaac is not None:
            result["isaac_observation_bridge"] = {"time_delta_s":0,
                "robot_pose_precision":"full authoritative MuJoCo values; no decimal rounding; Isaac/USD representation precision still applies"}
        result["environment_config_before_source_initialization"] = deepcopy(raw.task.state_dict())
        result["physics_model"] = save_physics_model(raw.mujoco.mjModel, raw.mujoco.mjSpec, args.output)
        result["initialization"] = initialize_source_state(raw, teacher)
        if lift_task:
            from copy import deepcopy as copy_contract
            from simple.tasks import g1_wholebody_lift_box_hold_sonic as lift_module
            import simple.baselines.sonic_v1_1 as controller_module
            result["task_contract"] = copy_contract(lift_module.TASK_CONTRACT)
            result["task_contract_id"] = lift_module.TASK_CONTRACT["metric"]
            result["measurement_hz"] = 50
            keys=("body_token_version","decoder_sha256","pd_profile","pd_profile_source_sha256",
                  "body_kp","body_kd","hand_torque_profile","hand_torque_limit_nm","observation_phase")
            result["controller_contract"] = {key:controller.identity[key] for key in keys if key in controller.identity}
            result["controller_contract"].update(physics_dt_s=float(raw.mujoco.mjModel.opt.timestep),
                decoder_hz=50, command_hz=command_hz, controller_source_sha256=sha256(Path(controller_module.__file__)),
                robot_mjcf_sha256=sha256(args.data_root / raw.task.robot.mjcf_path))
            declaration={"task_contract":result["task_contract"],"controller_contract":result["controller_contract"],
                         "render_contract":result["render_contract"],
                         "task_source_sha256":sha256(Path(lift_module.__file__))}
            declaration_path=args.output/"task_controller_contract.json"
            write_json(declaration_path,declaration)
            result["frozen_contract"]={"path":str(declaration_path),"sha256":sha256(declaration_path)}
            log_attempt("started")
        controller.reset()
        obs, info = raw._get_obs(), raw._get_info()
        initial_time = float(obs["sim_time_s"])
        source_start = None
        result.update(physics_dt_s=float(raw.mujoco.mjModel.opt.timestep), decoder_hz=50,
                      camera_hz=float(raw.task.render_hz), action_hz=action_hz, warmup_seconds_requested=args.warmup_seconds)
        if not np.isclose(result["camera_hz"], 50) or not np.isclose(result["physics_dt_s"], .005):
            raise ValueError("Teacher replay requires actual 200 Hz physics and 50 Hz camera/decoder")
        if args.video:
            video = RawVideoRecorder(args.output / "oracle-replay.mp4", fps=50, shape=obs["head_stereo_left"].shape)
        Image.fromarray(obs["head_stereo_left"]).save(args.output / "rgb-first.png")
        endpoint = min(args.seconds, last_timestamp)
        trace = (args.output / "trace.jsonl").open("w", buffering=1)
        command_trace = (args.output / "command_events.jsonl").open("w", buffering=1)
        command_ordinal = -1
        body_token = hands = None
        capture_command_rgb = getattr(args, "command_hz", None) is not None and command_recorder is None
        if capture_command_rgb:
            (args.output / "command_rgb").mkdir()
        try:
            while True:
                now = float(obs["sim_time_s"])
                if video:
                    video.write(obs["head_stereo_left"], sim_time_s=now)
                data, model = raw.mujoco.mjData, raw.mujoco.mjModel
                finite = all(np.isfinite(value).all() for value in (data.qpos, data.qvel, data.qacc, data.ctrl, data.xpos, data.xmat))
                result["finite_states"] = bool(result["finite_states"] and finite)
                if not finite:
                    result["status"] = "nonfinite_physics"
                    break
                pelvis = model.body("pelvis").id
                height = float(data.xpos[pelvis, 2])
                tilt = float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3,3)[2,2], -1, 1))))
                fell_now = height < args.fall_height or tilt > args.fall_tilt_degrees
                if fell_now and not result["fell"]:
                    result["first_fall_sim_time_s"] = now
                result["fell"] = bool(result["fell"] or fell_now)
                result["minimum_pelvis_height_m"] = min(result.get("minimum_pelvis_height_m", height), height)
                result["maximum_pelvis_tilt_degrees"] = max(result.get("maximum_pelvis_tilt_degrees", tilt), tilt)
                if fell_now and args.stop_on_fall:
                    result["status"] = "stopped_on_fall"
                    break
                if source_start is None and now - initial_time >= args.warmup_seconds - 1e-8:
                    if len(controller._history) < 10:
                        raise ValueError("Warmup did not collect ten real 50 Hz physical frames")
                    source_start = now
                    result.update(teacher_start_sim_time_s=now, actual_warmup_sim_seconds=now-initial_time,
                                  real_proprioception_frames_before_teacher=len(controller._history),
                                  pelvis_height_at_teacher_start_m=height, pelvis_tilt_at_teacher_start_degrees=tilt)
                source_time = 0 if source_start is None else now - source_start
                if source_start is not None and source_time + 1/50 > endpoint + 1e-8:
                    result["status"] = "source_end" if endpoint == last_timestamp else "sim_time_limit"
                    break
                index = int(math.floor(source_time * action_hz + 1e-7))
                if not 0 <= index < len(teacher["timestamp"]):
                    raise ValueError("Teacher action index is outside the genuine converted trajectory")
                actual_q = state43_to45(obs["joint_qpos"], obs["joint_names"])[0, :43]
                reference_q = teacher["observation.state"][index]
                base = np.r_[data.xpos[pelvis], data.xquat[pelvis]]
                box = np.r_[data.body("cube").xpos, data.body("cube").xquat]
                base_ref, box_ref = teacher["observation.base_pose"][index], teacher["observation.object_poses"][index]
                record = {"sim_time_s": now, "source_time_s": float(source_time), "converted_frame_index": index,
                          "source_episode_index": int(teacher["source_episode_index"][index]),
                          "source_frame_index": int(teacher["source_frame_index"][index]),
                          "source_timestamp": float(teacher["source_timestamp"][index]),
                          "phase": "measured_pose_proprioception_warmup" if source_start is None else "oracle_teacher_token",
                          "joint_rmse_rad": float(np.sqrt(np.mean((actual_q-reference_q)**2))),
                          "body_joint_rmse_rad": float(np.sqrt(np.mean((actual_q[:29]-reference_q[:29])**2))),
                          "hand_joint_rmse_rad": float(np.sqrt(np.mean((actual_q[29:]-reference_q[29:])**2))),
                          "base_position_error_m": float(np.linalg.norm(base[:3]-base_ref[:3])),
                          "base_orientation_error_degrees": quaternion_error_degrees(base[3:], base_ref[3:]),
                          "box_position_error_m": float(np.linalg.norm(box[:3]-box_ref[:3])),
                          "box_orientation_error_degrees": quaternion_error_degrees(box[3:], box_ref[3:]),
                          "pelvis_height_m": height, "pelvis_tilt_degrees": tilt,
                          "fell": fell_now, "decoder_calls_before_step": controller.decoder_calls}
                record["box_contacts"] = box_contacts(model, data)
                record["observation_phase"] = result["observation_phase"]
                if getattr(raw,"isaac_observation_only",False):
                    record["render_pose"]=raw.isaac.last_observation_pose_diagnostics
                record["action_hz"] = action_hz
                record["source_token_field"] = token_key
                if source_start is None:
                    command = controller.hold(obs)
                    result["warmup_camera_frames"] += 1
                else:
                    event_ordinal, event_native_frame = command_event_index(source_time, command_hz)
                    new_command = event_ordinal != command_ordinal
                    if new_command:
                        if event_ordinal != command_ordinal + 1:
                            raise ValueError("Command clock skipped or repeated an ordinal")
                        actual_native_frame = int(math.floor(source_time * 50 + 1e-7))
                        if actual_native_frame != event_native_frame:
                            raise ValueError("Command event was not generated on its first real 50 Hz tick")
                        command_ordinal = event_ordinal
                        hands = dict(zip(TRAINING_HAND_JOINT_NAMES, teacher["action"][index]))
                        if token_provider is None:
                            body_token = teacher[token_key][index]
                            token_metadata = None
                        else:
                            token_started = time.perf_counter()
                            body_token, token_metadata = token_provider(index, raw)
                            record["runtime_encoder"] = token_metadata
                            record["token_encoding_wall_s"] = time.perf_counter()-token_started
                    record.update(command_event_index=command_ordinal, command_updated=new_command,
                                  command_hold="zero order hold between command events; decoder remains 50 Hz")
                    record["applied_body_token"] = np.asarray(body_token).tolist()
                    record["applied_hand_targets"] = hands
                    command = controller.step(body_token, hands, obs)
                    if new_command:
                        hand14 = np.asarray([hands[name] for name in TRAINING_HAND_JOINT_NAMES])
                        event = {"event_index":command_ordinal, "command_hz":command_hz,
                            "nominal_source_time_s":command_ordinal/command_hz,
                            "source_time_s":float(source_time), "sim_time_s":now,
                            "native_frame_index":actual_native_frame,
                            "expected_native_frame_index":event_native_frame,
                            "source_ref_frame":int(teacher["source_frame_index"][index]),
                            "source_episode":int(teacher["source_episode_index"][index]),
                            "state43":actual_q.tolist(), "base_pose7":base.tolist(), "box_pose7":box.tolist(),
                            "executed_token64":np.asarray(body_token).tolist(), "hand_targets14":hand14.tolist(),
                            "runtime_encoder":token_metadata,
                            "observation_phase":result["observation_phase"],
                            "rgb_sha256":hashlib.sha256(np.ascontiguousarray(obs["head_stereo_left"]).tobytes()).hexdigest(),
                            "rgb_shape":list(obs["head_stereo_left"].shape), "rgb_dtype":"uint8",
                            "observation_command_alignment":"same real pre-step observation and physics tick",
                            "reference_anchor_A":"identity; original world reference remains fixed"}
                        if getattr(raw,"isaac_observation_only",False):
                            event["render_pose"]=raw.isaac.last_observation_pose_diagnostics
                        if capture_command_rgb:
                            rgb_path = Path("command_rgb") / f"event_{command_ordinal:06d}.png"
                            Image.fromarray(obs["head_stereo_left"]).save(args.output / rgb_path)
                            event["rgb_lossless_png"] = str(rgb_path)
                        if command_recorder is not None:
                            command_recorder.record(event_index=command_ordinal, sim_time_s=now,
                                rgb_uint8HWC=obs["head_stereo_left"], state43=actual_q,
                                executed_token64=np.asarray(body_token), hand_targets14=hand14,
                                base_pose7=base, box_pose7=box, source_ref_frame=event["source_ref_frame"],
                                source_episode=event["source_episode"], source_time_s=float(source_time))
                        command_trace.write(json.dumps(event, allow_nan=False) + "\n")
                        result["command_events"] += 1
                    result["teacher_control_frames"] += 1
                measurements.append((now, float(source_time), index, actual_q.copy(), reference_q.copy(),
                                     base.copy(), base_ref.copy(), box.copy(), box_ref.copy(), record["phase"]))
                if lift_task:
                    state=np.empty(mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION))
                    mujoco.mj_getState(model,data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
                    physics_states.append(state)
                obs, reward, terminated, truncated, info = env.step(command)
                record.update(next_sim_time_s=float(obs["sim_time_s"]), decoder_calls_after_step=controller.decoder_calls,
                              task_progress=info.get("lift_box_hold_progress",info.get("carry_box_progress")), reward=float(reward),
                              terminated=bool(terminated), truncated=bool(truncated))
                trace.write(json.dumps(record, allow_nan=False) + "\n")
                result["task_success_observed"] = bool(result["task_success_observed"] or terminated)
                if (terminated or truncated) and (lift_task or not continue_after_termination):
                    result.update(status="task_terminated" if terminated else "env_truncated", task_success_observed=bool(terminated))
                    break
        finally:
            trace.close()
            command_trace.close()
        result["physical_outcome_status"]=result["status"]
        final_data = raw.mujoco.mjData
        finite = all(np.isfinite(value).all() for value in (final_data.qpos, final_data.qvel, final_data.qacc, final_data.ctrl, final_data.xpos, final_data.xmat))
        result["finite_states"] = bool(result["finite_states"] and finite)
        if finite:
            final_height = float(final_data.xpos[pelvis, 2])
            final_tilt = float(np.degrees(np.arccos(np.clip(final_data.xmat[pelvis].reshape(3,3)[2,2], -1, 1))))
            final_fall = final_height < args.fall_height or final_tilt > args.fall_tilt_degrees
            if final_fall and not result["fell"]:
                result["first_fall_sim_time_s"] = float(obs["sim_time_s"])
            result["fell"] = bool(result["fell"] or final_fall)
            result["minimum_pelvis_height_m"] = min(result.get("minimum_pelvis_height_m", final_height), final_height)
            result["maximum_pelvis_tilt_degrees"] = max(result.get("maximum_pelvis_tilt_degrees", final_tilt), final_tilt)
            result["final_base_pose"] = np.r_[final_data.xpos[pelvis], final_data.xquat[pelvis]].tolist()
            result["final_box_pose"] = np.r_[final_data.body("cube").xpos, final_data.body("cube").xquat].tolist()
        result.update(decoder_calls=controller.decoder_calls, clipped_torque_elements=controller.clipped_torque_elements,
                      actual_sim_seconds=float(obs["sim_time_s"])-initial_time,
                      actual_teacher_source_seconds=0 if source_start is None else float(obs["sim_time_s"])-source_start,
                      final_task_progress=dict(raw.task.progress),
                      oracle_success_without_fall=bool(result["task_success_observed"] and not result["fell"] and result["finite_states"]),
                      label_controller_tracking_acceptance="unscored diagnostic; inspect errors/video before accepting labels for SFT")
        if getattr(raw,"isaac_observation_only",False):
            result["final_render_pose"]=raw.isaac.last_observation_pose_diagnostics
        if lift_task:
            terminal_time=float(obs["sim_time_s"])
            terminal_source=0.0 if source_start is None else terminal_time-source_start
            terminal_index=min(len(teacher["timestamp"])-1,max(0,int(math.floor(terminal_source*action_hz+1e-7))))
            measurements.append((terminal_time,terminal_source,terminal_index,
                state43_to45(obs["joint_qpos"],obs["joint_names"])[0,:43].copy(),
                teacher["observation.state"][terminal_index].copy(),np.r_[final_data.xpos[pelvis],final_data.xquat[pelvis]],
                teacher["observation.base_pose"][terminal_index].copy(),np.r_[final_data.body("cube").xpos,final_data.body("cube").xquat],
                teacher["observation.object_poses"][terminal_index].copy(),"task_terminal"))
            state=np.empty(mujoco.mj_stateSize(raw.mujoco.mjModel,mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(raw.mujoco.mjModel,final_data,state,mujoco.mjtState.mjSTATE_INTEGRATION)
            physics_states.append(state)
        Image.fromarray(obs["head_stereo_left"]).save(args.output / "rgb-last.png")
        if measurements:
            arrays = {name: np.stack([row[index] for row in measurements]) for index, name in enumerate(
                ("sim_time_s", "source_time_s", "converted_frame_index", "actual_state43", "reference_state43",
                 "actual_base_pose", "reference_base_pose", "actual_box_pose", "reference_box_pose", "phase"))}
            converted_indices = arrays["converted_frame_index"].astype(int)
            for name in ("source_episode_index", "source_frame_index", "source_timestamp"):
                arrays[name] = teacher[name][converted_indices]
            arrays["state43_joint_names"] = np.asarray(TRAINING_STATE_JOINT_NAMES)
            if lift_task:
                arrays["physics_state_integration"]=np.stack(physics_states)
                arrays["physics_state_spec"]=np.asarray(int(mujoco.mjtState.mjSTATE_INTEGRATION))
            np.savez_compressed(args.output / "trajectory.npz", **arrays)
            mask = arrays["phase"] == "oracle_teacher_token"
            if np.any(mask):
                difference = arrays["actual_state43"][mask] - arrays["reference_state43"][mask]
                result["teacher_tracking_rmse_rad"] = float(np.sqrt(np.mean(difference**2)))
                result["teacher_body_tracking_rmse_rad"] = float(np.sqrt(np.mean(difference[:, :29]**2)))
                result["teacher_hand_tracking_rmse_rad"] = float(np.sqrt(np.mean(difference[:, 29:]**2)))
    except Exception as error:
        result.setdefault("physical_outcome_status","error_before_physical_terminal")
        result.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if video:
                video.write(obs["head_stereo_left"], sim_time_s=float(obs["sim_time_s"]))
                result["video"] = video.close()
        except Exception as error:
            result.update(status="error", video_error=repr(error))
            if video:
                video.close()
            raise
        finally:
            result["wall_seconds"] = time.perf_counter()-started
            finalize_replay_report(path, result, command_recorder, env.close if env is not None else lambda: None,
                                   before_report=(lambda:log_attempt("finished")) if lift_task else None)
    print(json.dumps({"status": result["status"], "result": str(path),
                      "oracle_success_without_fall": result.get("oracle_success_without_fall", False)}), flush=True)
    return 0 if result["status"] in {"sim_time_limit", "source_end", "task_terminated", "stopped_on_fall"} and result["finite_states"] else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
