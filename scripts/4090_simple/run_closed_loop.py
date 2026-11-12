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
MOBILE_PP_TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
BEND_PICK_TASK = "G1SonicBendPickPico-v0"
BEND_PICK_SOURCE_TASK = "G1WholebodyBendPickMP-v0"
PICK_EVAL_TASKS = ("G1SonicXMovePickPico-v0", "G1SonicXMoveBendPickPico-v0")
SUPPORTED_TASKS = (*TASKS, CARRYBOX_TASK, LIFTBOX_TASK, MOBILE_PP_TASK, BEND_PICK_TASK, *PICK_EVAL_TASKS)


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


def episode_reset_options(environment_config, dr_level, *, recorded_handover=False, recorded_mobile=False, recorded_bend=False, recorded_pick=False):
    # SIMPLE DR0 deliberately resamples materials and distractors. A recorded
    # heldout source used dr_level=None to restore every recorded randomizer.
    return {"state_dict": environment_config,
            "dr_level": None if recorded_handover or recorded_mobile or recorded_bend or recorded_pick else dr_level}


def bend_official_protocol(environment_config):
    """Original reward/time budget on the explicitly declared SONIC adaptation."""
    metadata = environment_config['metadata']
    steps = metadata.get('max_episode_steps')
    if (environment_config.get('uid') != 'g1_wholebody_bend_pick_mp'
            or metadata.get('physics_dt') != .002 or metadata.get('render_hz') != 50
            or type(steps) is not int or steps != 400):
        raise ValueError('BendPick requires native MP metadata with a positive exact step budget')
    return {'task_id': 'cvpr/'+BEND_PICK_TASK, 'source_task_id': 'simple/'+BEND_PICK_SOURCE_TASK,
        'max_episode_steps': steps, 'observation_hz': 50, 'maximum_simulated_seconds': steps/50,
        'reset_dr_level': None, 'source_physics_dt': .002, 'physics_dt': .005,
        'source_metadata_success_criteria': metadata.get('success_criteria'), 'effective_success_criteria': .9,
        'source_metadata_split': metadata.get('split'), 'world_translation_m': [0., 0., .405],
        'success_semantics': 'unchanged original check_success, target lift .1m scale, threshold .9; no contact/hold gate',
        'initialization': 'saved official scene reset only; no teacher actions/states or PICO physical-state injection',
        'robot_physics_equivalent': False, 'formal_evaluation_eligible': False}


def derive_bend_eval_reset(source_config):
    """Remove only stale spatial entries that have no real scene actor.

    Official eval L0/L1 carries unused DR keys left by source generation.
    All actual target/distractor/robot poses remain byte-value identical.
    Three optional camera serialization fields unsupported by CameraCfg are
    removed only when None; nonempty calibration must never be discarded.
    The unchanged training adapter still validates the complete derived reset.
    """
    source = deepcopy(source_config)
    bend_official_protocol(source)
    dr, actors = source['dr_state_dict'], source['layout']['actors']
    selected = {str(dr['target']['uid']), *map(str, dr['distractors'])}
    actor_uids = [str(actor['asset']['uid']) for actor in actors.values() if 'asset' in actor]
    if (source.get('robot_cfg', {}).get('uid') != 'g1_wholebody' or len(dr['distractors']) != 3
            or len(actor_uids) != 4 or len(set(actor_uids)) != 4 or set(actor_uids) != selected
            or set(actors) != {'table', 'target', *[name for name in actors if name.startswith('distractor_')]}
            or 'g1_wholebody' not in dr['spatial'] or 'g1_sonic' in dr['spatial']):
        raise ValueError('Bend eval reset must preserve the exact selected target, three distractors, table and native robot')
    keep = selected | {'g1_wholebody'}
    if not keep <= set(dr['spatial']):
        raise ValueError('Bend eval is missing an actual actor spatial pose')
    removed = sorted(set(dr['spatial']) - keep)
    if any(uid.startswith('g1_') for uid in removed):
        raise ValueError('An unexpected robot spatial key may not be removed')
    derived = deepcopy(source)
    derived['dr_state_dict']['spatial'] = {uid: deepcopy(value) for uid, value in dr['spatial'].items() if uid in keep}
    removed_sensor_fields = {}
    for name, sensor in derived['sensor_cfgs'].items():
        for field in ('intrinsics', 'depth_intrinsics', 'depth_to_color_extrinsics'):
            if field not in sensor:
                continue
            if sensor[field] is not None:
                raise ValueError('Bend eval may not discard nonempty camera calibration: sensor_cfgs.'+name+'.'+field)
            removed_sensor_fields.setdefault(name, {})[field] = sensor.pop(field)
    # Exact source/runtime compatibility, valid only with complete saved DR and
    # reset dr_level=None. These are not generic sampling-policy overrides.
    compatible_dr_fields = {
        ('spatial', 'placement_attempts'): (int, 100,
            'Current SpatialDR has the same hard-coded range(100); saved real poses are restored'),
        ('scene', 'randomize_scene_pose'): (bool, True,
            'Saved scene pose is restored and inner-state return bypasses scene-pose sampling'),
        ('material', 'table_material'): (type(None), None,
            'No configuration override; complete saved table/robot/object material state is restored'),
    }
    removed_dr_fields = {}
    dr_field_reasons = {}
    for (name, field), (expected_type, expected_value, reason) in compatible_dr_fields.items():
        config = derived['dr_cfgs'].get(name, {})
        if field not in config:
            continue
        if type(config[field]) is not expected_type or config[field] != expected_value:
            raise ValueError('Unsupported non-equivalent Bend eval DR option: dr_cfgs.'+name+'.'+field)
        removed_dr_fields.setdefault(name, {})[field] = config.pop(field)
        dr_field_reasons[name+'.'+field] = reason
    digest = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=_json_value).encode()).hexdigest()
    return derived, {'contract': 'cvpr-bend-pick-official-eval-reset-derivation-v3',
        'source_config_sha256': digest(source), 'derived_config_sha256': digest(derived),
        'removed_unused_spatial_uids': removed,
        'removed_unused_spatial_entries': {uid: deepcopy(dr['spatial'][uid]) for uid in removed},
        'removed_optional_sensor_fields': removed_sensor_fields,
        'removed_optional_sensor_field_paths': [['sensor_cfgs', name, field]
            for name, fields in removed_sensor_fields.items() for field in fields],
        'camera_geometry_or_nonempty_calibration_modified': False,
        'removed_compatible_dr_fields': removed_dr_fields,
        'removed_compatible_dr_field_paths': [['dr_cfgs', name, field]
            for name, fields in removed_dr_fields.items() for field in fields],
        'compatible_dr_field_reasons': dr_field_reasons,
        'compatibility_scope': 'complete saved DR restore with dr_level=None; no fresh randomization equivalence claim',
        'preserved_real_spatial_uids': sorted(keep), 'actual_actor_poses_modified': False,
        'metadata_split_or_dr_level_modified': False, 'reward_or_success_modified': False,
        'reason': 'Removed only unused spatial keys, three named None-only camera fields, and three exact-value source/runtime DR options with explicit saved-reset equivalence evidence; real poses, camera geometry, saved materials and original reward remain unchanged'}


def prepare_bend_reset(source_config, *, initialization_profile='official_original', support_geometry=None):
    from pico_bend_pick_task import adapt_official_config
    derived, derivation = derive_bend_eval_reset(source_config)
    support_receipt = None
    if initialization_profile == 'stable_support_v1':
        from bend_support_initialization import derive_stable_support_reset
        if support_geometry is None:
            raise ValueError('Stable-support initialization requires its verified geometry manifest')
        derived, support_receipt = derive_stable_support_reset(derived, support_geometry)
    elif initialization_profile != 'official_original' or support_geometry is not None:
        raise ValueError('Unknown or mismatched BendPick initialization profile')
    adapted, adaptation = adapt_official_config(derived)
    return adapted, derivation, adaptation, support_receipt


def install_bend_support_checks(env, support_receipt):
    """Validate compiled initialization before the native reset step; observe only."""
    from bend_support_initialization import verify_actual_support_model
    from bend_initialization_monitor import BendInitializationMonitor
    raw = env.unwrapped
    raw._bend_support_pending_receipt = deepcopy(support_receipt)
    raw._bend_support_model_verification = None
    raw._bend_initialization_monitor = None
    original_update = raw.mujoco.update_layout
    original_step = raw.mujoco.step

    def update(*args, **kwargs):
        raw._bend_support_model_verification = None
        raw._bend_initialization_monitor = None
        result = original_update(*args, **kwargs)
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        raw._bend_support_model_verification = verify_actual_support_model(
            model, data, raw._bend_support_pending_receipt)
        raw._bend_initialization_monitor = BendInitializationMonitor(
            model, data, int(raw.mujoco.mj_objects['target'].id))
        return result

    def step(*args, **kwargs):
        monitor = raw._bend_initialization_monitor
        verification = raw._bend_support_model_verification
        if (monitor is None or monitor.model is not raw.mujoco.mjModel or
                not verification or verification.get('status') != 'passed'):
            raise ValueError('BendPick physics stepped before its initialization verification')
        result = original_step(*args, **kwargs)
        monitor.after_step(raw.mujoco.mjModel, raw.mujoco.mjData)
        return result

    raw.mujoco.update_layout, raw.mujoco.step = update, step


def make_bend_environment(sonic_config, source_config, *, sim_mode, sync_observations, isaac_observation_only,
                          initialization_profile='official_original', support_geometry=None):
    from pico_bend_pick_task import make_bend_pick_task, install_frame_adapter
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv
    import gymnasium as gym
    protocol = bend_official_protocol(source_config)
    adapted, derivation, receipt, support_receipt = prepare_bend_reset(source_config,
        initialization_profile=initialization_profile, support_geometry=support_geometry)
    task = make_bend_pick_task(sonic_config, adapted_config=adapted)
    env = SonicLocoManipEnv(task=task, sonic_config=sonic_config, sim_mode=sim_mode, headless=True,
        dr_level=0, sync_mujoco_observations=sync_observations, isaac_observation_only=isaac_observation_only)
    try:
        binding = install_frame_adapter(env)
        if support_receipt is not None:
            install_bend_support_checks(env, support_receipt)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=protocol['max_episode_steps'])
    except BaseException:
        env.close()
        raise
    return env, {'adaptation': receipt, 'reset_derivation': derivation, 'frame_binding': binding, 'protocol': protocol,
        'initialization_profile': initialization_profile, 'bend_support_initialization': support_receipt}


def mobile_official_protocol(environment_config):
    """Keep original reset metadata; use the actual upstream CLI time budget.

    The archive's success_criteria is descriptive metadata. SIMPLE reset only
    restores randomizers; the live task keeps its constructor's .9 threshold.
    """
    metadata = environment_config["metadata"]
    if (environment_config.get("uid") != "g1_wholebody_locomotion_pick_between_tables_teleop"
            or metadata.get("max_episode_steps") != 1200
            or metadata.get("render_hz") != 50 or metadata.get("physics_dt") != .005):
        raise ValueError("Mobile official-case protocol requires the unchanged1200-step/50Hz task metadata")
    return {"max_episode_steps": 1200, "observation_hz": 50, "maximum_simulated_seconds": 24.,
            "reset_dr_level": None, "source_metadata_success_criteria": metadata.get("success_criteria"),
            "source_metadata_split": metadata.get("split"), "effective_success_criteria": .9,
            "success_semantics": "upstream task.check_success; stop on native terminated",
            "initialization": "official task reset; no teacher row or PICO cold-profile injection",
            "boundary": "official fixed cases with dual-source SONIC controller adaptation; not certified original-controller benchmark"}


def mobile_time_limit(env):
    import gymnasium as gym
    return gym.wrappers.TimeLimit(env, max_episode_steps=1200)


def mobile_matched_protocol(initializer):
    """Same native budget, with an explicit source-row-zero diagnostic scope."""
    protocol = mobile_official_protocol(initializer.environment_config)
    protocol.update(initialization="one admitted Mobile source row0 before new history; no teacher actions/images/future states",
        boundary="matched source initial state with dual-source SONIC adaptation; not an official fixed-case benchmark",
        initialization_case_index=initializer.receipt["case_index"],
        source_episode_index=initializer.receipt["source_episode_index"],
        source_human_demo_uuid=initializer.receipt["human_demo_uuid"],
        experiment_role=initializer.receipt["role"], formal_evaluation_eligible=False)
    return protocol


def bind_mobile_training_split(initializer, expected_sha256):
    """Bind evaluation roles to the actual current model's frozen train split."""
    if initializer.receipt["split_manifest"]["sha256"] != expected_sha256:
        raise ValueError("Mobile initialization split differs from the expected actual model training split")
    initializer.receipt["expected_mobile_split_sha256"] = expected_sha256


def reset_mobile_matched_episode(env, initializer, agent, episode_id, receipt_path=None):
    """Reset static scene, restore once, then clear/start new causal history."""
    env.reset(seed=initializer.seed, options=episode_reset_options(
        deepcopy(initializer.environment_config), 0, recorded_mobile=True))
    obs, info, receipt = initializer.restore(env.unwrapped)
    if receipt_path is not None:
        write_json(receipt_path, receipt)
    agent.reset(episode_id=episode_id)
    return obs, info, receipt


def episode_termination(episode, terminated, truncated, sim_time, *, continue_after_success=False):
    if terminated:
        episode["task_success_observed"] = True
        episode.setdefault("first_task_success_sim_time_s", float(sim_time))
        if not continue_after_success:
            return "task_terminated"
    if truncated:
        return "env_truncated"
    return None


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


def requested_rtc_contract(args):
    """Declare synchronous plan stitching; never invent an asynchronous delay."""
    enabled = args.rtc_mode == "test_time"
    if not math.isfinite(args.pig_guidance_alpha) or not 0 <= args.pig_guidance_alpha <= 1:
        raise ValueError("RTC guidance alpha must be finite and in [0,1]")
    horizon = args.execute_steps if args.min_exec_horizon is None else args.min_exec_horizon
    if enabled:
        if args.execute_steps >= 30 or args.execute_steps * 50 % 30:
            raise ValueError("Test-time RTC requires a shorter execution window exactly aligned to 50Hz ticks")
        if not args.execute_steps <= horizon <= 30 or not 0 <= args.rtc_inference_delay <= 30-horizon:
            raise ValueError("RTC requires execute-steps <= min-exec-horizon and d+s <=30")
        if args.rtc_inference_delay != 0:
            raise ValueError("Synchronous paused-clock RTC requires zero inference delay")
    return {"rtc_enabled": enabled, "rtc_mode": args.rtc_mode,
            "rtc_exec_shift": args.execute_steps if enabled else None,
            "rtc_inference_delay": args.rtc_inference_delay if enabled else None,
            "rtc_min_exec_horizon": horizon if enabled else None,
            "pig_mask_schedule": args.pig_mask_schedule if enabled else None,
            "pig_guidance_alpha": args.pig_guidance_alpha if enabled else None,
            "num_inference_steps": 10}


def validate_server_info(server_info, args):
    expected = requested_rtc_contract(args)
    action = server_info.get("action", {})
    if (server_info.get("ckpt_step") != args.expected_checkpoint_step or
            action.get("action_dim") != 80 or action.get("action_chunk_size") != 30 or
            action.get("action_exec_horizon") != args.execute_steps or
            server_info.get("online_history", {}).get("frames") != 6):
        raise ValueError("Evaluation requires the requested checkpoint, six-frame history and matching action window")
    actual = {name: server_info.get(name) for name in expected}
    if actual != expected:
        raise ValueError(f"Actual service RTC configuration differs: expected={expected}, actual={actual}")
    if getattr(args,'pick_warmup_profile','native')=='ready_upper_v1' and server_info.get('rtc_guidance_action_scope')!='all':
        raise ValueError('ready_upper_v1 requires actual legacy all80 RTC guidance; do not mix a guidance-mask change')
    if getattr(args,'pick_budget_profile','native_16s')=='xmove_bend_failed7_24s' and server_info.get('rtc_guidance_action_scope')!='all':
        raise ValueError('24s diagnostic requires actual original all80 RTC guidance')
    return {"status": "passed", "requested": expected, "actual": actual,
            "action_exec_horizon": args.execute_steps, "action_chunk_size": 30,
            "synchronization": "synchronous HTTP pauses simulation; no asynchronous latency compensation claim"}


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
    parser.add_argument("--handover-initialization", type=Path,
                        help="Once before history: source-informed held-out Handover cold start; never teacher actions")
    parser.add_argument("--mobile-initialization-manifest", type=Path,
                        help="Mobile matched diagnostic: one admitted first integration state before new causal history")
    parser.add_argument("--mobile-initialization-case", type=int, choices=range(10),
                        help="Frozen manifest case0..4=train probes, case5..9=heldout; requires the manifest")
    parser.add_argument("--expected-mobile-split-sha256",
                        help="Required with Mobile initialization; take this SHA from the current model's actual95/5 training receipt")
    parser.add_argument("--warmup-control", choices=("measured_pose_pd", "live_pose_sonic_v1_1"),
                        default="measured_pose_pd", help="Optional live-pose v1.1 balance reference while genuine RGB history fills")
    parser.add_argument("--warmup-hand-policy", choices=("measured", "open"), default="measured",
                        help="Optional neutral zero hand targets during live-pose SONIC warmup only")
    parser.add_argument('--pick-warmup-profile', choices=('native', 'ready_upper_v1'), default='native',
                        help='Explicit XMovePick5000 diagnostic only; changes startup reference, never the saved scene')
    parser.add_argument('--pick-ready-planner-model', type=Path,
                        help='Pinned official CPU neutral planner; required only for ready_upper_v1')
    parser.add_argument("--body-token-projection", choices=("none", "fsq_v1_1"), default="none",
                        help="Optional denormalized body64 projection to the fixed SONIC v1.1 32-level FSQ grid")
    parser.add_argument("--tasks", nargs="+", choices=SUPPORTED_TASKS, default=list(TASKS))
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--episode-start", type=int, default=0)
    parser.add_argument("--dr-level", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--execute-steps", type=int, default=30)
    parser.add_argument("--rtc-mode", choices=("off", "test_time"), default="off")
    parser.add_argument("--rtc-inference-delay", type=int, default=0)
    parser.add_argument("--min-exec-horizon", type=int)
    parser.add_argument("--pig-mask-schedule", choices=("exponential", "linear", "hard"), default="exponential")
    parser.add_argument("--pig-guidance-alpha", type=float, default=.9)
    parser.add_argument("--http-timeout", type=float, default=120.0)
    parser.add_argument("--worker-wall-timeout", type=float, default=900.0)
    parser.add_argument("--fall-height", type=float, default=.35)
    parser.add_argument("--fall-tilt-degrees", type=float, default=75.)
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument("--continue-after-task-success", action="store_true",
                        help="Handover only: retain official success flag but continue to the observation time limit")
    parser.add_argument("--mobile-official-protocol", action="store_true",
                        help="Mobile only: original case DR state, native .9 success and exact1200-step TimeLimit")
    parser.add_argument('--pick-official-protocol', action='store_true',
                        help='Moving Pick private task with exact saved scene, native success and800 control ticks')
    parser.add_argument('--pick-profile', choices=('xmove_pick', 'xmove_bend_pick'))
    parser.add_argument('--pick-budget-profile', choices=('native_16s','xmove_bend_failed7_24s'), default='native_16s',
                        help='Explicit XMoveBendPick failed7 diagnostic only:1200 total control ticks/24s including50 warmup')
    parser.add_argument('--pick-interaction-diagnostics', action='store_true',
                        help='Read solved200Hz contacts and actual sent actions; no physical intervention')
    parser.add_argument('--bend-official-protocol', action='store_true',
                        help='BendPick private SONIC adapter: saved native scene, original reward and metadata TimeLimit')
    parser.add_argument('--initialization-profile', choices=('official_original', 'stable_support_v1'),
                        default='official_original', help='Explicitly separate revised stable-support BendPick initialization')
    parser.add_argument('--support-geometry-manifest', type=Path,
                        help='Pinned CPU-compiled target geometry required only by stable_support_v1')
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
    if args.pick_interaction_diagnostics and not args.pick_official_protocol:
        parser.error('Pick interaction diagnostics require an explicit private Pick task')
    from pick_eval_protocol import profile_spec
    if args.pick_official_protocol != (args.pick_profile is not None):
        parser.error('Moving Pick requires both --pick-official-protocol and --pick-profile')
    if args.pick_official_protocol:
        pick_task = profile_spec(args.pick_profile)['task']
        if (args.tasks != [pick_task] or args.worker_task not in (None, pick_task)
                or args.bend_official_protocol or args.mobile_official_protocol or args.stop_on_fall
                or args.mobile_initialization_manifest is not None or args.initial_physics_run is not None
                or args.handover_initialization is not None or not args.sync_observations
                or not args.isaac_observation_only or args.sim_mode == 'mujoco'):
            parser.error('Moving Pick requires its sole private task, synchronized observation-only Isaac and no foreign initialization/fall gate')
    elif any(task in PICK_EVAL_TASKS for task in args.tasks) or args.worker_task in PICK_EVAL_TASKS:
        parser.error('Private moving Pick tasks require their explicit official profile')
    ready_upper = args.pick_warmup_profile == 'ready_upper_v1'
    if ready_upper != (args.pick_ready_planner_model is not None):
        parser.error('ready_upper_v1 requires exactly its explicit planner model')
    if ready_upper:
        if (not args.pick_official_protocol or args.pick_profile != 'xmove_pick'
                or args.warmup_control != 'live_pose_sonic_v1_1' or args.warmup_hand_policy != 'measured'
                or not args.pick_interaction_diagnostics or args.body_token_projection != 'none'
                or args.expected_checkpoint_step != 5000 or args.execute_steps != 15
                or args.rtc_mode != 'test_time' or args.seconds != 16.
                or args.pd_profile != 'official_v1_1_ankle_pitch_1_5' or args.hand_torque_profile != 'mjcf_limits'
                or args.rtc_inference_delay != 0 or args.min_exec_horizon not in (None,15)
                or args.pig_mask_schedule != 'exponential' or args.pig_guidance_alpha != .9):
            parser.error('ready_upper_v1 is an explicit XMovePick5000 test_time/15-action diagnostic with all physical contacts recorded')
        args.pick_ready_planner_model = args.pick_ready_planner_model.resolve(strict=True)
    if args.pick_budget_profile == 'xmove_bend_failed7_24s':
        from pick_eval_protocol import EXTENDED_BUDGET_CASES
        if (not args.pick_official_protocol or args.pick_profile != 'xmove_bend_pick'
                or args.seconds != 24. or args.seed != 0 or args.episodes_per_task != 1 or (args.dr_level,args.episode_start) not in EXTENDED_BUDGET_CASES
                or args.pick_warmup_profile != 'native' or args.expected_checkpoint_step != 5000
                or args.warmup_control != 'live_pose_sonic_v1_1' or args.warmup_hand_policy != 'measured'
                or args.rtc_mode != 'test_time' or args.execute_steps != 15 or args.body_token_projection != 'none'
                or args.pd_profile != 'official_v1_1_ankle_pitch_1_5' or args.hand_torque_profile != 'mjcf_limits'
                or args.rtc_inference_delay != 0 or args.min_exec_horizon not in (None,15)
                or args.pig_mask_schedule != 'exponential' or args.pig_guidance_alpha != .9):
            parser.error('24s requires only the fixed failed7 XMoveBendPick5000 native warmup/test_time/15-action diagnostic')
    stable_support = args.initialization_profile == 'stable_support_v1'
    if stable_support != (args.support_geometry_manifest is not None) or (stable_support and not args.bend_official_protocol):
        parser.error('Stable-support initialization requires only BendPick plus --support-geometry-manifest')
    if args.support_geometry_manifest is not None:
        args.support_geometry_manifest = args.support_geometry_manifest.resolve(strict=True)
    if (args.bend_official_protocol != (args.tasks == [BEND_PICK_TASK])
            or BEND_PICK_TASK in args.tasks and args.tasks != [BEND_PICK_TASK]):
        parser.error('The private BendPick task requires exactly --bend-official-protocol and no other task')
    if args.bend_official_protocol and (args.worker_task not in (None, BEND_PICK_TASK)
            or args.stop_on_fall or args.mobile_official_protocol or args.mobile_initialization_manifest is not None
            or args.initial_physics_run is not None or args.handover_initialization is not None
            or not args.sync_observations or not args.isaac_observation_only or args.sim_mode == 'mujoco'):
        parser.error('BendPick requires synchronized observation-only Isaac, no fall gate or foreign initialization')
    if args.continue_after_task_success and args.tasks != ["G1WholebodyHandoverTeleop-v0"]:
        parser.error("Continuing after official success is only supported for Handover observation")
    if args.mobile_official_protocol and (args.tasks != [MOBILE_PP_TASK] or args.seconds < 24
            or args.stop_on_fall or args.worker_task not in (None, MOBILE_PP_TASK)):
        parser.error("--mobile-official-protocol requires only Mobile P&P, --seconds >=24 and no --stop-on-fall; TimeLimit remains1200 steps")
    matched = args.mobile_initialization_manifest is not None
    if matched != (args.mobile_initialization_case is not None) or matched != (args.expected_mobile_split_sha256 is not None):
        parser.error("Mobile initialization requires manifest, case and --expected-mobile-split-sha256 from the current model training receipt")
    if matched:
        if (len(args.expected_mobile_split_sha256) != 64
                or any(c not in "0123456789abcdef" for c in args.expected_mobile_split_sha256)):
            parser.error("Expected Mobile training split must be a lowercase SHA256")
        if (args.tasks != [MOBILE_PP_TASK] or args.worker_task not in (None, MOBILE_PP_TASK)
                or args.episodes_per_task != 1 or args.dr_level != 0 or args.seconds < 24 or args.stop_on_fall
                or not args.sync_observations or not args.isaac_observation_only
                or args.sim_mode == "mujoco" or args.mobile_official_protocol
                or args.initial_physics_run is not None or args.handover_initialization is not None):
            parser.error("Mobile matched initialization requires one Mobile DR0 episode, synchronized observation-only Isaac, >=24s/no fall stop, and no other initialization/official protocol")
        if not args.mobile_initialization_manifest.is_absolute():
            parser.error("Mobile initialization manifest must be absolute")
        args.mobile_initialization_manifest = args.mobile_initialization_manifest.resolve(strict=True)
    if args.isaac_observation_only and (not args.sync_observations or args.sim_mode == "mujoco"):
        parser.error("--isaac-observation-only requires --sync-observations and Isaac render mode")
    if args.warmup_control == "live_pose_sonic_v1_1" and not args.sync_observations:
        parser.error("--warmup-control live_pose_sonic_v1_1 requires --sync-observations")
    if args.warmup_hand_policy == "open" and args.warmup_control != "live_pose_sonic_v1_1":
        parser.error("--warmup-hand-policy open requires --warmup-control live_pose_sonic_v1_1")
    if args.initial_physics_run is not None:
        if (not args.initial_physics_run.is_absolute() or not args.sync_observations or
                args.episodes_per_task != 1 or len(args.tasks) != 1 or
                args.tasks[0] not in (CARRYBOX_TASK, LIFTBOX_TASK) or args.dr_level != 0):
            parser.error("--initial-physics-run needs an absolute source, synchronized observations, one local DR0 task and one episode")
        args.initial_physics_run = args.initial_physics_run.resolve(strict=True)
    if args.handover_initialization is not None:
        if (not args.sync_observations or args.dr_level != 0 or
                args.tasks != ["G1WholebodyHandoverTeleop-v0"] or args.initial_physics_run is not None):
            parser.error("Handover initialization requires synchronized Handover DR0 only")
        args.handover_initialization = args.handover_initialization.resolve(strict=True)
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
    try:
        requested_rtc_contract(args)
    except ValueError as error:
        parser.error(str(error))
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
    matched_manifest = getattr(args, "mobile_initialization_manifest", None)
    if matched_manifest is not None:
        result["mobile_initialization_manifest"] = str(matched_manifest)
        result["mobile_initialization_case"] = args.mobile_initialization_case
        result["expected_mobile_split_sha256"] = args.expected_mobile_split_sha256
    write_json(path, result)
    env, client = None, None
    try:
        controller = SonicV11OnnxController(args.sonic_resources,
            compatibility_evidence=args.compatibility_evidence,
            allow_unverified_compatibility=args.diagnostic_unverified_sonic, pd_profile=args.pd_profile,
            sync_mujoco_observations=args.sync_observations, hand_torque_profile=args.hand_torque_profile)
        result["controller_identity"] = controller.identity
        result["formal_evaluation_eligible"] = False
        client = MeasuredClient(server_ip="127.0.0.1", server_port=args.port, timeout_s=args.http_timeout,
                                execute_steps=args.execute_steps)
        health = client.session.get(client.base_url + "/health", timeout=5)
        health.raise_for_status()
        result["server_health"] = health.json()
        server_info = client.session.get(client.base_url + "/info", timeout=5)
        server_info.raise_for_status()
        server_info = server_info.json()
        result["server_info"] = server_info
        result["rtc_contract"] = validate_server_info(server_info, args)
        result["initial_metadata_get_requests"] = 2
        metadata_task = BEND_PICK_SOURCE_TASK if args.worker_task == BEND_PICK_TASK else args.worker_task
        if args.pick_official_protocol:
            from pick_eval_protocol import profile_spec
            metadata_task = profile_spec(args.pick_profile)['source_task']
        metadata = args.data_root / "eval" / metadata_task / f"dr-level-{args.dr_level}" / "meta/episodes.jsonl"
        mobile_initializer = None
        if matched_manifest is not None:
            from mobile_pp_initial_state import load_initialization
            mobile_initializer = load_initialization(matched_manifest, args.mobile_initialization_case)
            bind_mobile_training_split(mobile_initializer, args.expected_mobile_split_sha256)
        carrybox = args.worker_task in (CARRYBOX_TASK, LIFTBOX_TASK)
        if mobile_initializer is not None:
            first_config = deepcopy(mobile_initializer.environment_config)
            result["mobile_protocol"] = mobile_matched_protocol(mobile_initializer)
        elif carrybox:
            if args.dr_level != 0:
                raise ValueError("CarryBox local v1 supports DR level 0 only")
            first_config = {"metadata": {"physics_dt": .005}}
        else:
            first_config, _ = load_fixed_episode(metadata, args.episode_start, args.data_root)
        if args.mobile_official_protocol:
            result["mobile_protocol"] = mobile_official_protocol(first_config)
        config = SimLoopConfig().load_wbc_yaml()
        config["ENV_NAME"] = "simple"
        config["SIMULATE_DT"] = .005 if args.bend_official_protocol else float(first_config["metadata"]["physics_dt"])
        if args.worker_task in (MOBILE_PP_TASK, BEND_PICK_TASK, *PICK_EVAL_TASKS):
            config["ENABLE_ELASTIC_BAND"] = False
        mode = "mujoco_isaac" if args.sim_mode == "isaacsim" else args.sim_mode
        support_geometry = None
        if args.pick_official_protocol:
            from pick_eval_protocol import make_pick_environment, INITIALIZATION_PROFILE
            result['initialization_profile'] = INITIALIZATION_PROFILE
            result['pick_profile'] = args.pick_profile
            env, result['pick_adapter'] = make_pick_environment(config, first_config, args.pick_profile,
                sim_mode=mode, sync_observations=args.sync_observations,
                isaac_observation_only=args.isaac_observation_only, budget_profile=args.pick_budget_profile)
        elif args.bend_official_protocol:
            if args.initialization_profile == 'stable_support_v1':
                from bend_support_initialization import load_geometry_manifest
                support_geometry = load_geometry_manifest(args.support_geometry_manifest, args.data_root)
            result['initialization_profile'] = args.initialization_profile
            env, result['bend_adapter'] = make_bend_environment(config, first_config, sim_mode=mode,
                sync_observations=args.sync_observations, isaac_observation_only=args.isaac_observation_only,
                initialization_profile=args.initialization_profile, support_geometry=support_geometry)
        else:
            env = gym.make("simple/" + args.worker_task, sonic_config=config, sim_mode=mode,
                           headless=True, dr_level=args.dr_level,
                           sync_mujoco_observations=args.sync_observations,
                           isaac_observation_only=args.isaac_observation_only)
        if args.mobile_official_protocol or mobile_initializer is not None:
            env = mobile_time_limit(env)
        raw = env.unwrapped
        if args.pick_interaction_diagnostics:
            from pick_interaction_observer import install_interaction_observer
            install_interaction_observer(raw)
        result["render_contract"] = simulation_render_contract(raw,args)
        if args.sync_observations and raw.isaac is not None:
            result["isaac_observation_bridge"] = {"time_delta_s":0,
                "robot_pose_precision":"full authoritative MuJoCo values; no decimal rounding; Isaac/USD representation precision still applies"}
        client.sim_clock = lambda: float(raw.mujoco.mjData.time)
        warmup_control = None
        if args.pick_warmup_profile == 'ready_upper_v1':
            from xmove_ready_upper_warmup import ReadyUpperSonicWarmup
            warmup_control = ReadyUpperSonicWarmup(args.sonic_resources, args.pick_ready_planner_model)
            result.update(pick_warmup_profile='ready_upper_v1', native_saved_scene_initialization_preserved=True,
                original_official_initialization=False, warmup_protocol_modified=True)
        elif args.warmup_control == "live_pose_sonic_v1_1":
            from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
            warmup_control = LivePoseSonicWarmup(args.sonic_resources, hand_policy=args.warmup_hand_policy)
        agent = DualSourcePsi0Agent(raw.task.robot, client=client, controller=controller,
                                   execute_steps=args.execute_steps, warmup_control=warmup_control,
                                   body_token_projection=args.body_token_projection)
        result["body_token_projection"] = deepcopy(agent.projection_receipt)
        for episode_index in range(args.episode_start, args.episode_start + args.episodes_per_task):
            episode_id = f"{args.worker_task}:dr{args.dr_level}:episode{episode_index}:{uuid.uuid4().hex}"
            episode_dir = task_dir / f"episode-{episode_index}"
            episode_dir.mkdir(exist_ok=True)
            event_start = len(client.events)
            mobile_initial = None
            seed = args.seed + episode_index - args.episode_start if carrybox else args.seed
            if mobile_initializer is not None:
                seed = mobile_initializer.seed
                env_config = deepcopy(mobile_initializer.environment_config)
                protocol = mobile_matched_protocol(mobile_initializer)
                obs, info, mobile_initial = reset_mobile_matched_episode(env, mobile_initializer, agent, episode_id,
                    episode_dir / "mobile_initialization_receipt.json")
                source_episode = {"kind": "frozen_mobile_first_state_only",
                    "source_episode_index": mobile_initial["source_episode_index"],
                    "human_demo_uuid": mobile_initial["human_demo_uuid"], "experiment_role": mobile_initial["role"],
                    "initialization_case_index": mobile_initial["case_index"],
                    "initialization_manifest_sha256": mobile_initial["manifest_sha256"]}
            elif carrybox:
                obs, info = env.reset(seed=seed)
                env_config = deepcopy(raw.task.state_dict())
                source_episode = {"kind": "independent_local_seed", "seed": seed,
                                  "task_adapter": raw.task.metadata["task_adapter"]}
            else:
                env_config, source_episode = load_fixed_episode(metadata, episode_index, args.data_root)
                bend_source = None
                if args.pick_official_protocol:
                    from pick_eval_protocol import prepare_pick_reset, environment_binding_receipt
                    pick_source = deepcopy(env_config)
                    env_config, pick_derivation, pick_adaptation, pick_protocol = prepare_pick_reset(pick_source, args.pick_profile, args.pick_budget_profile)
                    if args.seconds < pick_protocol['maximum_simulated_seconds']:
                        raise ValueError('Moving Pick budget may not shorten its declared control ticks')
                if args.bend_official_protocol:
                    from pico_bend_pick_task import bend_pick_environment_binding_receipt
                    bend_source = deepcopy(env_config)
                    protocol = bend_official_protocol(bend_source)
                    if args.seconds < protocol['maximum_simulated_seconds']:
                        raise ValueError('BendPick wall budget may not shorten the original metadata step budget')
                    env_config, bend_derivation, bend_receipt, bend_support_receipt = prepare_bend_reset(bend_source,
                        initialization_profile=args.initialization_profile, support_geometry=support_geometry)
                    if bend_support_receipt is not None:
                        raw._bend_support_pending_receipt = deepcopy(bend_support_receipt)
                    if protocol['max_episode_steps'] != result['bend_adapter']['protocol']['max_episode_steps']:
                        raise ValueError('One worker must preserve its original BendPick TimeLimit')
                if args.mobile_official_protocol:
                    protocol = mobile_official_protocol(env_config)
                obs, info = env.reset(seed=seed, options=episode_reset_options(
                    env_config, args.dr_level, recorded_handover=args.handover_initialization is not None,
                    recorded_mobile=args.worker_task == MOBILE_PP_TASK, recorded_bend=args.bend_official_protocol, recorded_pick=args.pick_official_protocol))
                if args.pick_official_protocol:
                    pick_binding = environment_binding_receipt(env_config, raw.task.state_dict(), args.pick_profile)
                    if pick_binding.get('matched') is not True or raw.task.success_criteria != .9:
                        raise ValueError('Moving Pick saved scene/native success threshold differs')
                if args.bend_official_protocol:
                    actual_binding = bend_pick_environment_binding_receipt(env_config, raw.task.state_dict())
                    if actual_binding.get('matched') is not True or raw.task.success_criteria != .9:
                        raise ValueError('BendPick saved environment binding/native threshold differs')
                if args.mobile_official_protocol and not math.isclose(float(raw.task.success_criteria), .9):
                    raise ValueError("Actual Mobile native success threshold differs from upstream CLI default .9")
            initial_physics = None
            handover_initial = None
            if args.handover_initialization is not None:
                from handover_eval_runtime import load_initialization, initialize
                row, receipt = load_initialization(args.handover_initialization, episode_index)
                obs, info, handover_initial = initialize(raw, row, receipt)
            if args.initial_physics_run is not None:
                from initial_physics_state import InitialPhysicsState
                initializer = InitialPhysicsState(args.initial_physics_run,
                    task_id="simple/" + args.worker_task, seed=seed)
                result["initial_physics_source"] = initializer.receipt
                obs, info, initial_physics = initializer.restore(raw)
            if mobile_initial is None:
                agent.reset(episode_id=episode_id)
            episode = {"episode_id": episode_id, "episode_index": episode_index, "seed": seed,
                "dr_level": args.dr_level, "source_episode": source_episode,
                "local_environment_config": env_config,
                "source_metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest()
                    if not carrybox and mobile_initial is None else None,
                "physics_dt_s": float(raw.mujoco.mjModel.opt.timestep),
                "render_hz": float(raw.task.render_hz), "action_fps": ACTION_FPS,
                "sonic_decoder_hz": 50, "execute_steps": args.execute_steps,
                "controller_history_length_after_reset": len(controller._history),
                "agent_chunk_cleared_after_reset": agent._chunk is None,
                "finite_states": True, "fell": False, "task_success_observed": False,
                "warmup_frames": 0, "policy_frames": 0, "frames": 0,
                "fell_before_first_policy_command": False,
                "status": "running", "instruction": raw.task.instruction}
            episode["source_scene_reset_dr_level"] = (None if args.handover_initialization is not None
                or args.worker_task == MOBILE_PP_TASK or args.bend_official_protocol or args.pick_official_protocol else args.dr_level)
            if args.pick_official_protocol:
                from pick_eval_protocol import profile_spec, CONTROL_POLICY, INITIALIZATION_PROFILE
                episode.update(initialization_profile=INITIALIZATION_PROFILE, pick_profile=args.pick_profile, task_id=profile_spec(args.pick_profile)['task_id'],
                    pick_protocol=pick_protocol, pick_source_environment_config=pick_source,
                    pick_reset_derivation=pick_derivation, pick_adaptation=pick_adaptation,
                    pick_environment_binding=pick_binding, evaluation_control_policy=CONTROL_POLICY)
            if args.pick_budget_profile == 'xmove_bend_failed7_24s':
                from pick_eval_protocol import extended_budget_runtime_receipt
                result['pick_budget_profile'] = args.pick_budget_profile
                episode['pick_extended_budget'] = extended_budget_runtime_receipt(env,pick_protocol,args.seconds)
            if args.pick_warmup_profile == 'ready_upper_v1':
                episode.update(pick_warmup_profile='ready_upper_v1', native_saved_scene_initialization_preserved=True,
                    original_official_initialization=False, warmup_protocol_modified=True)
            if args.bend_official_protocol:
                episode.update(bend_protocol=protocol, bend_source_environment_config=bend_source,
                    bend_adaptation=bend_receipt, bend_environment_binding=actual_binding,
                    bend_reset_derivation=bend_derivation,
                    task_id='cvpr/'+BEND_PICK_TASK, initialization_profile=args.initialization_profile)
                if bend_support_receipt is not None:
                    episode.update(bend_support_initialization=bend_support_receipt,
                        bend_support_geometry_manifest=deepcopy(bend_support_receipt['geometry_manifest_identity']),
                        bend_support_model_verification=deepcopy(raw._bend_support_model_verification))
            if args.mobile_official_protocol or mobile_initial is not None:
                episode["mobile_protocol"] = protocol
            if mobile_initial is not None:
                receipt_path = episode_dir / "mobile_initialization_receipt.json"
                episode["mobile_initialization"] = mobile_initial
                episode["mobile_initialization_receipt"] = str(receipt_path)
                episode["local_environment_config_scope"] = "static saved scene; actual first physical state is the admitted row0 initialization receipt"
            if initial_physics is not None:
                episode["initial_physics"] = initial_physics
                episode["local_environment_config_scope"] = "same-seed reset model; actual initial coordinates/velocities are in initial_physics"
            if handover_initial is not None:
                episode["handover_initialization"] = handover_initial
            pelvis_id = raw.mujoco.mjModel.body("pelvis").id
            episode["initial_pelvis_position_m"] = raw.mujoco.mjData.xpos[pelvis_id].tolist()
            episode["initial_pelvis_quaternion_wxyz"] = raw.mujoco.mjData.xquat[pelvis_id].tolist()
            if args.pick_interaction_diagnostics:
                episode['initial_solved_contacts'] = raw._pick_interaction_observer.drain()
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
                        if args.pick_interaction_diagnostics:
                            raw._pick_interaction_observer.phase = phase
                        obs, reward, terminated, truncated, info = env.step(command)
                        last_obs = obs
                        record.update(next_sim_time_s=float(obs["sim_time_s"]),
                                      decoder_calls_after_step=controller.decoder_calls,
                                      terminated=bool(terminated), truncated=bool(truncated), reward=float(reward))
                        if args.pick_official_protocol:
                            record['pick_native_state'] = {'target_xyz_m': list(map(float, info['target'][:3])),
                                'baseline_target_z_m': float(raw.task._init_target_height),
                                'native_reward': float(reward), 'native_success': bool(terminated)}
                        if args.pick_interaction_diagnostics:
                            from pick_interaction_observer import measured_control_snapshot
                            record['pick_interaction_substeps'] = raw._pick_interaction_observer.drain()
                            record['pick_measured_control'] = measured_control_snapshot(raw,command,obs)
                        if args.bend_official_protocol:
                            record['bend_native_state'] = {'target_xyz_m': list(map(float, info['target'][:3])),
                                'baseline_target_z_m': float(raw.task._init_target_height),
                                'native_reward': float(reward), 'native_success': bool(terminated)}
                        if args.worker_task == "G1WholebodyHandoverTeleop-v0":
                            from handover_eval_runtime import measured_task_state
                            record["handover_measured_state"] = measured_task_state(raw)
                        if args.worker_task == MOBILE_PP_TASK:
                            from handover_eval_runtime import measured_task_state
                            record["mobile_measured_state"] = measured_task_state(raw)
                        if "carry_box_progress" in info or "lift_box_hold_progress" in info:
                            record["task_progress"] = info.get("lift_box_hold_progress", info.get("carry_box_progress"))
                        trace.write(json.dumps(record, allow_nan=False) + "\n")
                        terminal = episode_termination(episode, terminated, truncated, obs["sim_time_s"],
                            continue_after_success=args.continue_after_task_success)
                        if terminal:
                            episode["status"] = terminal
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
                episode["continue_after_official_success"] = args.continue_after_task_success
                if args.pick_official_protocol:
                    from pick_eval_protocol import initialization_receipt
                    episode['pick_initialization_safety'] = initialization_receipt(raw)
                if args.pick_budget_profile == 'xmove_bend_failed7_24s':
                    from pick_eval_protocol import extended_budget_runtime_receipt
                    episode['pick_extended_budget'] = extended_budget_runtime_receipt(env,pick_protocol,args.seconds)
                if args.pick_interaction_diagnostics:
                    observer = raw._pick_interaction_observer
                    episode['pick_interaction_observer'] = observer.receipt()
                    if episode['status'] not in ('error', 'running'):
                        episode['pick_interaction_observer'] = observer.completed_receipt(
                            episode['warmup_frames']+episode['policy_frames'])
                if args.bend_official_protocol and support_geometry is not None:
                    episode['bend_initialization_safety'] = raw._bend_initialization_monitor.receipt()
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
