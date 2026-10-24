"""Execute a new object-feedback expert in a frozen randomized TRAIN scene.

This independent entrypoint never edits legacy captures. Its labels are the
commands actually generated online and executed; raw source references are
not described as actions executed in this new scene. Admission is separate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

import replay_handover_teacher as base

CONTRACT = "cvpr-handover-random100-online-execution-v1"


def run(args):
    from handover_random100_sources import load_raw_source
    from handover_random100_scene import (randomize_config, transformed_initial, audit_scene_readback,
                                          apply_scene_visual_parameters)
    from handover_random100_expert import HandoverObjectFeedbackExpert, ExpertConfig
    from handover_eval_runtime import initialize, measured_task_state
    from handover_contact_quality import sample_live_state, evaluate_capture
    from handover_geometry_quality import build_container_geometry_cache
    import gymnasium as gym
    import mujoco
    import simple.envs
    import simple.baselines.sonic_v1_1 as sonic_module
    from simple.baselines.sonic_v1_1 import SonicV11OnnxController
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

    output = args.output.absolute()
    repo = Path(__file__).resolve().parents[2]
    if (output.exists() or output.resolve().is_relative_to(repo)
            or any(output.resolve().is_relative_to(p.resolve()) for p in
                   (args.raw_root, args.data_root, args.sonic_resources))):
        raise ValueError("Use a fresh external output, separate from all inputs")
    if not 5 <= args.max_seconds <= 35:
        raise ValueError("Bound each physical trial to5..35 seconds")
    projection_phases = tuple(args.latent_projection_phases.split(","))
    if not projection_phases or any(name not in (
            "approach", "right_lift", "handover", "left_transfer", "placement", "release", "hold") for name in projection_phases):
        raise ValueError("Declare valid, comma-separated expert phases for optional projection")
    source = load_raw_source(args.source_episode, raw_root=args.raw_root, asset_root=args.data_root)
    source["values"]["hand_condition"] = args.hand_condition
    config = deepcopy(source["config"])
    config["uid"] = "g1_wholebody_handover_teleop"
    config = base.remap_paths(config, args.data_root, args.data_root)
    config, proposal = randomize_config(config, args.seed, args.data_root,
        xy_max_m=args.xy_max_m, yaw_max_rad=args.yaw_max_rad)
    initial = transformed_initial(source["initial"], proposal)
    calibration_ids = base.calibration_source_identities()
    data_root, calibration_shas = base.bound_calibration_assets(args.data_root, calibration_ids)
    output.mkdir(parents=True)
    base.write_json(output / "scene_proposal.json", proposal)
    base.write_json(output / "environment_config.json", config)
    base.write_json(output / "initialization_input.json", initial)
    base.write_json(output / "source_evidence.json", source["evidence"])
    execution_args = {key: str(value.resolve()) if isinstance(value, Path) else value
                      for key, value in vars(args).items()}
    base.write_json(output / "execution_args.json", execution_args)
    script_dir = Path(__file__).parent
    simple_dir = Path(sonic_module.__file__).parents[1]
    code = [Path(__file__), Path(base.__file__), *(script_dir / name for name in (
        "handover_random100_sources.py", "handover_random100_scene.py", "handover_random100_expert.py",
        "audit_handover_random100.py",
        "feedback_wrist_ik_v11.py", "prepare_carrybox_v11.py", "handover_eval_runtime.py",
        "handover_contact_quality.py", "handover_geometry_quality.py")), Path(sonic_module.__file__),
        *(simple_dir / name for name in ("core/action.py", "engines/isaacsim.py", "engines/mujoco.py",
        "sensors/head_camera_mount.py", "envs/sonic_loco_manip.py", "baselines/live_pose_sonic_warmup.py",
        "robots/g1_sonic.py")), *(simple_dir.parents[1] / name for name in (
        "third_party/gear_sonic/utils/mujoco_sim/configs.py",
        "third_party/gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml"))]
    expert_dependencies = {}
    if args.placement_encoder_mode == "teleop":
        from handover_sonic_teleop_reference import verify_official_sources
        teleop_sources = verify_official_sources()["source_sha256"]
        code.append(script_dir / "handover_sonic_teleop_reference.py")
        code.extend(Path(path) for path in teleop_sources)
    if args.latent_projector_batch_view is not None:
        code.append(script_dir / "handover_sonic_latent_projector.py")
        view = args.latent_projector_batch_view.resolve(strict=True)
        for path in (view, Path(str(view) + ".json")):
            expert_dependencies[str(path)] = base.file_sha256(path)
    code_sha = {str(p.resolve()): base.file_sha256(p) for p in code}
    snapshots = []
    for i, p in enumerate(code):
        destination = output / "provenance/code" / f"{i:02d}-{p.name}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, destination)
        if base.file_sha256(destination) != code_sha[str(p.resolve())]:
            raise RuntimeError("Source changed during snapshot")
        snapshots.append({"original_path": str(p.resolve()), "path": str(destination.relative_to(output)),
                          "sha256": code_sha[str(p.resolve())]})
    base.write_json(output / "provenance/code_manifest.json", snapshots)
    source_bindings = source.get("sources", source.get("source_sha256", {}))
    result = {"contract": CONTRACT, "status": "initializing", "started_utc": datetime.now(timezone.utc).isoformat(),
        "source_episode_index": args.source_episode, "source_split": "train", "source_sha256": source_bindings,
        "capture_code_sha256": code_sha, "code_snapshots": snapshots,
        "resource_data_root": str(data_root), "calibration_source_asset_sha256": calibration_shas,
        "source_reference_frames": len(source["values"]["reference_full_state43"]),
        "expert_dependency_sha256": expert_dependencies,
        "hand_command_condition": args.hand_condition,
        "placement_encoder_mode": args.placement_encoder_mode,
        "placement_locomotion": args.placement_locomotion,
        "scene_proposal_sha256": base.file_sha256(output / "scene_proposal.json"),
        "execution_args_sha256": base.file_sha256(output / "execution_args.json"),
        "source_evidence_sha256": base.file_sha256(output / "source_evidence.json"),
        "source_tokens_reencoded": True, "commands_generated_online": True,
        "command_rate_hz": 30, "physics_hz": 200, "decoder_hz": 50, "render_mode": args.render_mode,
        "object_restores_during_control": False, "learned_policy_used": False,
        "pd_profile": "fixed_cpp_baseline", "hand_torque_profile": "recording_0p7_nm",
        "scope": "New source-informed object-feedback expert; no claim of recovered human commands or policy performance",
        "accepted_new_dataset": False}
    base.write_json(output / "result.json", result)
    os.environ["SIMPLE_DATA_DIR"] = str(data_root)
    env = training_capture = force_stream = quality_stream = trace_stream = expert_stream = None
    states, times, joints, roots, velocities, quality_rows = [], [], [], [], [], []
    completed = force_count = 0
    rendered = args.render_mode == "isaac"
    try:
        controller = SonicV11OnnxController(args.sonic_resources, allow_unverified_compatibility=True,
            pd_profile="fixed_cpp_baseline", hand_torque_profile="recording_0p7_nm", sync_mujoco_observations=True)
        result["controller_identity"] = controller.identity
        sonic_config = SimLoopConfig().load_wbc_yaml()
        sonic_config.update(ENV_NAME="simple", SIMULATE_DT=.005)
        env = gym.make("simple/" + base.TASK, sonic_config=sonic_config,
            sim_mode="mujoco_isaac" if rendered else "mujoco", headless=True, dr_level=0,
            sync_mujoco_observations=True, isaac_observation_only=rendered, disable_env_checker=not rendered)
        raw = env.unwrapped
        if not rendered:
            raw.task.sensor_cfgs = {}
            raw._render_frame = lambda: {}
        obs, info = env.reset(seed=args.seed, options={"state_dict": config, "dr_level": None})
        if rendered:
            applied = apply_scene_visual_parameters(raw, proposal)
            base.write_json(output / "scene_visual_application.json", applied)
            result["scene_visual_application_sha256"] = base.file_sha256(output / "scene_visual_application.json")
        obs, info, receipt = initialize(raw, initial, {"scope": "one TRAIN cold start before real warmup",
            "teacher_actions_used": True, "restores_during_control": False})
        result["initialization"] = receipt
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        if model.neq or not np.isclose(model.opt.timestep, .005) or not np.array_equal(model.opt.gravity, [0, 0, -9.81]):
            raise ValueError("Require200Hz gravity physics with no equality attachment")
        mujoco.mj_saveModel(model, str(output / "physics_model.mjb"))
        result["physics_model"] = {"sha256": base.file_sha256(output / "physics_model.mjb"),
                                   "equalities": [], "target_robot_external_attachment": False}
        cache = build_container_geometry_cache(model, data, container_body=raw.mujoco.mj_objects["container"].id,
                                               target_body=raw.mujoco.mj_objects["target"].id)
        base.write_json(output / "container_geometry_cache.json", cache)
        if rendered:
            scene_receipt = audit_scene_readback(raw, proposal, expected_initial=initial)
        else:
            measured = {role: {"pose7": [*raw.mujoco.mj_objects[role].xpos.tolist(),
                *raw.mujoco.mj_objects[role].xquat.tolist()],
                "mass": float(model.body_mass[raw.mujoco.mj_objects[role].id])}
                for role in ("target", "container")}
            for role, item in measured.items():
                if (not np.allclose(item["pose7"], initial["objects"][role]["pose7"], atol=1e-10, rtol=0)
                        or not np.isclose(item["mass"], initial["objects"][role]["mass"], atol=1e-7, rtol=1e-5)):
                    raise ValueError("CPU reset differs from frozen randomized proposal")
            scene_receipt = {"status": "physical_reset_verified_RTX_readback_pending", "objects": measured,
                "eligible_training_capture": False, "material_light_application_certified": False}
        base.write_json(output / "scene_readback.json", scene_receipt)
        result["scene_readback"] = scene_receipt
        if args.scene_only:
            if not rendered:
                raise ValueError("Scene-only evidence requires a real RTX scene")
            from PIL import Image
            Image.fromarray(np.asarray(obs["head_stereo_left"])).save(output / "initial_rgb.png")
            result.update(status="scene_readback_diagnostic_complete", completed_control_ticks=0,
                          scene_only=True, accepted_new_dataset=False)
            return result
        initial_z = float(raw.mujoco.mj_objects["target"].xpos[2])
        result["initial_target_z_m"] = initial_z
        force_stream = (output / "applied_force_samples.jsonl").open("w", buffering=1)
        quality_stream = (output / "quality_samples.jsonl").open("w", buffering=1)
        phase = {"tick": None}

        def check_force():
            nonlocal force_count
            value = base.applied_force_state(data); force_count += 1
            force_stream.write(base.json_text({"physics_step": force_count, "substep_index": force_count,
                                               "sim_time_s": float(data.time), **value}) + "\n")
            if not value["all_zero"]:
                raise RuntimeError("Nonzero external applied force")

        def observe_quality(count):
            row = sample_live_state(model, data, cache=cache, completed_steps=count,
                target_body=raw.mujoco.mj_objects["target"].id, container_body=raw.mujoco.mj_objects["container"].id,
                solver_stage="post_mj_step_before_observation_refresh")
            quality_rows.append(row); quality_stream.write(base.json_text(row) + "\n")

        raw.mujoco.step = base.solved_step_hook(raw.mujoco.step, lambda: phase["tick"], observe_quality,
                                              post_substep=check_force)
        controller.reset()
        from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
        warmup = LivePoseSonicWarmup(args.sonic_resources, hand_policy="measured")
        expert = HandoverObjectFeedbackExpert(source["values"], args.sonic_resources,
            config=ExpertConfig(placement_encoder_mode=args.placement_encoder_mode,
                                placement_locomotion=args.placement_locomotion))
        projector = None
        if args.latent_projector_batch_view is not None:
            from handover_sonic_latent_projector import SonicLatentProjector
            projector = SonicLatentProjector(jacobian_batch_view=args.latent_projector_batch_view)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        names = list(initial["joint_names"])
        if tuple(names[-14:]) != base.MODEL_HAND_NAMES:
            raise ValueError("Source must return actual MODEL_HAND_NAMES ordering")
        pelvis = model.body("pelvis").id
        root_joint = int(model.body_jntadr[pelvis]); root_dof = int(model.jnt_dofadr[root_joint])
        start_time = float(data.time)
        if rendered:
            training_capture = base.TrainingCapture(output / "capture", metadata={
                "execution_contract": CONTRACT, "source_episode_index": args.source_episode,
                "source_sha256": source_bindings, "capture_code_sha256": code_sha,
                "commands_generated_online": True, "source_tokens_reencoded": True,
                "scene_proposal_sha256": result["scene_proposal_sha256"], "instruction": raw.task.instruction,
                "resource_data_root": str(data_root), "calibration_source_asset_sha256": calibration_shas,
                "initial_target_z_m": initial_z}, joint_names=names, warmup_ticks=50, start_time=start_time)

        def save_state():
            state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, data, state, spec)
            states.append(state); times.append(float(data.time))
            joints.append([float(data.joint(name).qpos[0]) for name in names])
            roots.append([*data.xpos[pelvis].tolist(), *data.xquat[pelvis].tolist()]); velocities.append(data.qvel.copy())

        save_state()
        trace_stream = (output / "trace.jsonl").open("w", buffering=1)
        expert_stream = (output / "expert_commands.jsonl").open("w", buffering=1)
        held = None; last_ordinal = -1
        result["status"] = "running"
        for tick in range(round(args.max_seconds * 50)):
            teacher_tick = tick - 50
            if teacher_tick < 0:
                phase["tick"] = None
                command = warmup.command(raw.task.robot, controller, obs)
                control_phase = warmup.phase
                expert_diagnostic = None
            else:
                phase["tick"] = teacher_tick
                ordinal = teacher_tick * 30 // 50
                if ordinal != last_ordinal:
                    feedback = dict(quality_rows[-1]) if quality_rows else {}
                    # The expert reads the original solved-stage marker and
                    # dense completed_steps as well as contacts and timestamps.
                    feedback["recent_contact_samples"] = quality_rows[-30:]
                    token, hands, expert_diagnostic = expert(teacher_tick, raw, feedback=feedback)
                    if projector is not None:
                        unprojected = np.asarray(token, dtype=np.float32).copy()
                        if expert_diagnostic["phase"] in projection_phases:
                            token, projection = projector.project(token, expert_diagnostic["desired_body_q29"], raw, controller)
                        else:
                            projection = {"state": "skipped_phase_filter", "phase": expert_diagnostic["phase"],
                                          "enabled_phases": list(projection_phases)}
                        expert_diagnostic = {**expert_diagnostic, "unprojected_body_token64": unprojected.tolist(),
                            "latent_projection": projection,
                            "applied_body_token_sha256": base.hashlib.sha256(np.asarray(token, dtype=np.float32).tobytes()).hexdigest()}
                    held = (token, hands, expert_diagnostic)
                    expert_stream.write(base.json_text({"command_index30": ordinal, "teacher_tick50": teacher_tick,
                        "sim_time_s": float(data.time), "body_token64": token, "hand14": hands,
                        "hand_joint_names": list(base.HAND_NAMES), "diagnostics": expert_diagnostic}) + "\n")
                    last_ordinal = ordinal
                token, hands, expert_diagnostic = held
                command = controller.step(token, {n: float(v) for n, v in zip(base.HAND_NAMES, hands)}, obs)
                control_phase = "online_object_feedback_expert"
            if training_capture and base.capture_tick_index(tick) is not None:
                renderer = getattr(raw.isaac, "last_observation_pose_diagnostics", None)
                authoritative = {"links": {name: {"position_m": data.xpos[model.body(name).id].copy(),
                    "quaternion_wxyz": data.xquat[model.body(name).id].copy()} for name in base.RENDERER_LINKS},
                    "cameras": {"head_stereo_left": {"position_m": data.cam_xpos[model.camera("head_stereo_left").id].copy(),
                    "rotation_usd_axes": data.cam_xmat[model.camera("head_stereo_left").id].copy()}}}
                training_capture.append(tick50=tick, sim_time_s=float(data.time), rgb=obs["head_stereo_left"],
                    observed_joint_names=obs["joint_names"], observed_qpos=obs["joint_qpos"],
                    physical_qpos=joints[-1], physical_qvel=[float(data.joint(n).qvel[0]) for n in names],
                    root_pose7=roots[-1], root_qvel6=data.qvel[root_dof:root_dof+6],
                    objects={role: {"pose7": [*raw.mujoco.mj_objects[role].xpos, *raw.mujoco.mj_objects[role].xquat]}
                             for role in ("target", "container")}, command=command if teacher_tick >= 0 else None,
                    renderer_pose_diagnostics=renderer, renderer_authoritative_poses=authoritative,
                    observation_sim_time_s=obs.get("sim_time_s"))
            row = {"tick50": tick, "phase": control_phase, "sim_time_s": float(data.time),
                "elapsed_s": tick / 50, "physics_state_row_before": tick, "measured_before": measured_task_state(raw),
                "actual_body_token64": command["body_token"], "actual_hand_targets": command["hand_targets"],
                "expert_command_index30": last_ordinal if teacher_tick >= 0 else None,
                "decoder_calls_before": controller.decoder_calls}
            obs, reward, terminated, truncated, info = env.step(command)
            save_state(); completed = tick + 1
            finite = all(np.isfinite(a).all() for a in (data.qpos, data.qvel, data.qacc, data.ctrl))
            height = float(data.xpos[pelvis, 2]); tilt = float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3, 3)[2, 2], -1, 1))))
            row.update(next_sim_time_s=float(data.time), physics_state_row_after=tick + 1,
                measured_after=measured_task_state(raw), decoder_calls_after=controller.decoder_calls,
                reward=float(reward), official_success=bool(terminated), truncated=bool(truncated),
                pelvis_z_m=height, pelvis_tilt_degrees=tilt, finite_physics=bool(finite))
            trace_stream.write(base.json_text(row) + "\n")
            if completed % 100 == 0:
                print(base.json_text({"completed_ticks": completed, "source": args.source_episode,
                    "expert_phase": expert_diagnostic.get("phase") if expert_diagnostic else "warmup"}), flush=True)
            if not finite or height < .35 or tilt > 75 or truncated:
                result["early_stop"] = "nonfinite_fall_or_truncation"; break
            if len(quality_rows) >= 200 and len(quality_rows) % 25 == 0:
                if evaluate_capture(quality_rows, initial_target_z_m=initial_z)["eligible_physics"]:
                    from audit_handover_random100 import full_outer_footprint_final50
                    footprint = full_outer_footprint_final50({"model": model, "states": states,
                        "spec": spec, "roles": {role: raw.mujoco.mj_objects[role].id
                                                 for role in ("target", "container")}})
                    if footprint["passed"]:
                        result["actual_full_outer_footprint_before_early_stop"] = footprint
                        result["early_stop"] = "strict_full_placement_gate_satisfied"; break
        result.update(status="execution_complete", completed_control_ticks=completed,
            expert_control_ticks=max(0, completed - 50), source_commands=max(0, completed - 50),
            source_commands_interpretation="generated online50Hz control count; not original source row count",
            decoder_calls=controller.decoder_calls, warmup_control=warmup.receipt,
            physical_quality=evaluate_capture(quality_rows, initial_target_z_m=initial_z))
        base.write_json(output / "physical_quality.json", result["physical_quality"])
    except Exception as error:
        result.update(status="error", error=repr(error), traceback=traceback.format_exc())
        if hasattr(error, "snapshot"):
            base.write_json(output / "scene_readback_failure_snapshot.json", error.snapshot)
        if hasattr(error, "receipt"):
            base.write_json(output / "scene_readback_rejection.json", error.receipt)
        raise
    finally:
        for stream in (force_stream, quality_stream, trace_stream, expert_stream):
            if stream: stream.close()
        result["code_unchanged_during_run"] = all(base.file_sha256(path) == value for path, value in code_sha.items())
        result["expert_dependencies_unchanged_during_run"] = all(base.file_sha256(path) == value
            for path, value in expert_dependencies.items())
        result["calibration_source_assets_unchanged"] = base.bound_calibration_assets(data_root, calibration_ids)[1] == calibration_shas
        if states:
            np.savez_compressed(output / "trajectory.npz", physics_state_spec=np.asarray(spec),
                physics_state_integration=np.stack(states), sim_time_s=np.asarray(times), joint_names=np.asarray(names),
                actual_joint_q43=np.asarray(joints), root_pose7=np.asarray(roots), actual_qvel=np.asarray(velocities))
            result["trajectory"] = {"rows": len(states), "sha256": base.file_sha256(output / "trajectory.npz")}
        if force_stream:
            result["force_guard"] = {"path": "applied_force_samples.jsonl", "sha256": base.file_sha256(output / "applied_force_samples.jsonl"),
                "steps": force_count, "all_applied_forces_zero": force_count == 4 * completed,
                "equality_count": int(model.neq), "target_robot_attachment": False}
        if training_capture:
            training_capture.metadata.update(code_unchanged_during_run=result["code_unchanged_during_run"],
                calibration_source_assets_unchanged=result["calibration_source_assets_unchanged"], force_guard=result.get("force_guard"))
            result["training_capture"] = training_capture.close(completed=result["status"] == "execution_complete")
        result["ended_utc"] = datetime.now(timezone.utc).isoformat()
        base.write_json(output / "result.json", result)
        if env: env.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw-root", "data-root", "sonic-resources", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--source-episode", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--xy-max-m", type=float, default=.03)
    p.add_argument("--yaw-max-rad", type=float, default=.10)
    p.add_argument("--max-seconds", type=float, default=30.)
    p.add_argument("--render-mode", choices=("none", "isaac"), default="none")
    p.add_argument("--hand-condition", choices=("declared", "inferred_effective"), default="declared")
    p.add_argument("--placement-encoder-mode", choices=("g1", "teleop"), default="g1")
    p.add_argument("--placement-locomotion", choices=("source_future", "actual_hold"), default="source_future")
    p.add_argument("--scene-only", action="store_true")
    p.add_argument("--latent-projector-batch-view", type=Path)
    p.add_argument("--latent-projection-phases", default="placement")
    result = run(p.parse_args())
    print(base.json_text({k: result.get(k) for k in ("status", "source_episode_index", "expert_control_ticks", "early_stop")}))


if __name__ == "__main__":
    main()
