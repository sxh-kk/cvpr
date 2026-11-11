"""Official fixed30 BendPick scenes with explicit SONIC adaptation and audited RTC.

Only saved scene metadata is used. Native AMO robot/physics are adapted with the
same frozen PICO task as training; this is not an AMO benchmark or a claim that
the source archive's split=train cases prove independent heldout generalization.
Each case owns a fresh inference server and simulator; no automatic retry.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from urllib.error import URLError
from urllib.request import build_opener, ProxyHandler

import evaluate_mobile_pp30 as common
from run_closed_loop import (BEND_PICK_TASK, BEND_PICK_SOURCE_TASK, bend_official_protocol,
    derive_bend_eval_reset, _json_value)
from pico_bend_pick_task import adapt_official_config, OFFICIAL_TASK_SOURCE_SHA256

REPO = Path(__file__).resolve().parents[2]
TASK = BEND_PICK_TASK
TASK_ID = 'cvpr/'+TASK
SOURCE_TASK = BEND_PICK_SOURCE_TASK
CONTRACT = 'cvpr-bend-pick-fixed-official30-sonic-rtc-diagnostic-v1'
STABLE_SUPPORT_CONTRACT = 'cvpr-bend-pick-fixed30-stable-support-v1-sonic-rtc-diagnostic-v1'
STABLE_SUPPORT_BOUNDARY = ('Uniform target XY-only minimum translation into table support with5mm margin; '
    'all30 original cases retained; revised initialization, not the original official30 scene protocol. ')
READY_CONTRACT = 'cvpr-bend-pick-local-deployment-v1'
READY_STATUS = 'local_bend_pick_checkpoint_prepared'
OFFICIAL_REVISION = 'e78fb93cc28912a3031a10b8656d32d7f0a2b867'
OFFICIAL_ARCHIVE_SHA256 = '6bcdb9ddfa1d3f851144f585bda19507ad9efb53e71514dad98836fcefadc5e2'
OFFICIAL_METADATA_SHA256 = (
    '521bc985fa88569984b06f00ef36493fc56e89599eff0d4f6db16240efc7ceb6',
    'd0df0d3df180e67ee80d7553b94aeafccbff2c889d513f692f4b4947aee0fb46',
    '483b872dfa7bac3baa6f84df27ba20a91a5b626c8fbb3e898f2281a9bd6adc54')
BOUNDARY = ('Official fixed BendPick L0/L1/L2 scenes with explicit PICO SONIC robot/physics/frame adaptation; '
    'source split=train preserved, not proof of independent heldout or new-scene generalization; not native AMO benchmark')
MODEL_PY, SIM_PY = common.MODEL_PY, common.SIM_PY
DEPLOYMENT_ARTIFACTS = common.DEPLOYMENT_ARTIFACTS
require, now, read, write, file_identity, unchanged = (common.require, common.now, common.read, common.write,
    common.file_identity, common.unchanged)
json_lines, close_number = common.json_lines, common.close_number
runtime_gate, validate_rtc_execution = common.runtime_gate, common.validate_rtc_execution
mark_owned, stop_owned = common.mark_owned, common.stop_owned
preflight, environment, finish = common.preflight, common.environment, common.finish


def evaluation_policy(args):
    policy = common.evaluation_policy(args)
    require(policy['rtc_mode'] == 'test_time' and policy['execute_steps'] == 15,
        'This BendPick campaign requires the requested test_time RTC / exact15-action execution window')
    return policy


def initialization_profile(args):
    profile = getattr(args, 'initialization_profile', 'official_original')
    manifest = getattr(args, 'support_geometry_manifest', None)
    prioritize = getattr(args, 'prioritize_support_adjusted', False)
    require(profile in ('official_original', 'stable_support_v1'), 'Unknown BendPick initialization profile')
    require((profile == 'stable_support_v1') == (manifest is not None),
        'Only stable_support_v1 requires a support geometry manifest')
    require(not prioritize or profile == 'stable_support_v1',
        'Support-adjusted ordering requires stable_support_v1')
    return profile


def order_support_cases(cases, prioritize):
    if not prioritize:
        return cases
    # Stable sort by geometry-derived receipt only; never inspect model outcomes.
    return sorted(cases, key=lambda case: not case['bend_support_initialization']['applied'])


def source_identities():
    sources = {name: file_identity(Path(__file__).with_name(name)) for name in (
        'evaluate_bend_pick30.py', 'evaluate_mobile_pp30.py', 'run_closed_loop.py', 'serve_evaluation.py',
        'pico_bend_pick_task.py', 'capture_simple.py', 'video_capture.py', 'bend_support_initialization.py', 'bend_initialization_monitor.py')}
    for relative in ('deploy/serve_psi0_simple.py', 'deploy/dual_source_runtime.py', 'models/psi0.py', 'models/dual_source_action.py'):
        sources['psi/'+relative] = file_identity(REPO/'VLMExpert/src/psi'/relative)
    spec = importlib.util.find_spec('simple')
    require(spec is not None and spec.submodule_search_locations, 'Installed SIMPLE source is unavailable')
    simple = Path(next(iter(spec.submodule_search_locations)))
    for relative in ('tasks/g1_wholebody_bend_pick_mp.py', 'envs/sonic_loco_manip.py', 'envs/base_dual_env.py',
            'baselines/dual_source_psi0.py', 'baselines/sonic_v1_1.py', 'baselines/live_pose_sonic_warmup.py'):
        sources['SIMPLE/'+relative] = file_identity(simple/relative)
    require(sources['SIMPLE/tasks/g1_wholebody_bend_pick_mp.py']['sha256'] == OFFICIAL_TASK_SOURCE_SHA256,
        'The unchanged native BendPick task source differs')
    return sources


def freeze_inputs(args):
    from capture_simple import remap_episode_paths
    profile = initialization_profile(args)
    support_geometry = support_identity = None
    if profile == 'stable_support_v1':
        from bend_support_initialization import load_geometry_manifest
        support_path = Path(args.support_geometry_manifest).resolve(strict=True)
        support_identity = file_identity(support_path)
        support_geometry = load_geometry_manifest(support_path, args.data_root)
        unchanged(support_identity)
    ready_path = args.deployment/'LOCAL_RUN_READY.json'
    ready = read(ready_path)
    require(ready.get('contract') == READY_CONTRACT and ready.get('status') == READY_STATUS and
        ready.get('task_id') == TASK_ID and ready.get('step') == ready.get('task_step') == args.checkpoint_step and
        ready.get('joint_sha256') == args.expected_model_sha256 and ready.get('checkpoint_save_completed') is True and
        ready.get('training_completed') is True and ready.get('component_load_validation', {}).get('status') == 'passed',
        'Independent BendPick CPU deployment/save completion is required')
    manifest = file_identity(args.training_manifest)
    require(manifest['sha256'] == ready.get('training_manifest_sha256'), 'Training DINO manifest differs from deployment')
    encoder = read(args.training_manifest).get('encoder')
    require(isinstance(encoder, dict) and encoder.get('shape') == [5, 1024], 'Training encoder contract missing')
    model = file_identity(args.deployment/'checkpoints'/f'ckpt_{args.checkpoint_step}'/'model.safetensors')
    require(model['sha256'] == args.expected_model_sha256, 'Actual checkpoint bytes differ')
    sources = source_identities()
    sources['deployment_ready'] = file_identity(ready_path)
    if support_identity is not None:
        sources['support_geometry_manifest'] = support_identity
    artifacts = ready.get('deployment_artifacts')
    require(isinstance(artifacts, dict) and set(artifacts) == DEPLOYMENT_ARTIFACTS, 'Seven complete deployment artifacts are required')
    for name, record in artifacts.items():
        path = args.deployment/name
        require(path.resolve(strict=True).is_relative_to(args.deployment), 'Deployment artifact escapes its directory')
        identity = file_identity(path)
        require(identity['sha256'] == record.get('sha256') and identity['size_bytes'] == record.get('bytes'),
            'Deployment artifact differs from CPU preparation: '+name)
        sources['deployment/'+name] = identity
    config = read(args.deployment/'run_config.json')
    dino_contract = read(args.deployment/'dino_training_contract.json')
    require(config['data']['transform']['model'].get('motion_image_policy') ==
        dino_contract.get('motion_image_policy') == 'vlm_color_clean_motion', 'Trained image policy may not revert to shared_rgb')
    cases, metadata_files = [], []
    for level in range(3):
        path = args.data_root/'eval'/SOURCE_TASK/f'dr-level-{level}'/'meta/episodes.jsonl'
        identity = file_identity(path)
        require(identity['sha256'] == OFFICIAL_METADATA_SHA256[level], 'Require the frozen official BendPick metadata')
        rows = json_lines(path)
        require(len(rows) == 10 and sorted(row['episode_index'] for row in rows) == list(range(10)),
            'Each level requires exactly ten original cases0..9')
        for row in sorted(rows, key=lambda value: value['episode_index']):
            original = json.loads(row['environment_config'])
            native = remap_episode_paths(original, args.data_root)
            protocol = bend_official_protocol(native)
            derived, derivation = derive_bend_eval_reset(native)
            support_receipt = None
            if profile == 'stable_support_v1':
                from bend_support_initialization import derive_stable_support_reset
                derived, support_receipt = derive_stable_support_reset(derived, support_geometry)
                require(type(support_receipt.get('applied')) is bool, 'Support derivation must declare applied bool')
            adapted, adaptation = adapt_official_config(derived)
            cases.append({'level': level, 'case_index': row['episode_index'], 'flat_index': level*10+row['episode_index'],
                'source_row': row, 'source_environment_config': native, 'local_environment_config': adapted,
                'reset_derivation': derivation, 'adaptation': adaptation, 'protocol': protocol,
                'metadata_sha256': identity['sha256'], 'environment_seed': 0,
                'sampling_seed': args.sampling_seed+level*10+row['episode_index'],
                'initialization_profile': profile,
                'original_official_initialization': profile == 'official_original',
                'bend_support_initialization': support_receipt,
                'bend_support_geometry_manifest': support_identity})
        unchanged(identity)
        metadata_files.append(identity)
    cases = order_support_cases(cases, getattr(args, 'prioritize_support_adjusted', False))
    boundary = (STABLE_SUPPORT_BOUNDARY if profile == 'stable_support_v1' else '') + BOUNDARY
    return {'contract': STABLE_SUPPORT_CONTRACT if profile == 'stable_support_v1' else CONTRACT, 'task_id': TASK_ID, 'source_task_id': 'simple/'+SOURCE_TASK, 'prepared_utc': now(),
        'checkpoint_step': args.checkpoint_step, 'model': model, 'deployment': str(args.deployment),
        'training_manifest': manifest, 'metadata_files': metadata_files, 'sources': sources, 'cases': cases,
        'official_repository': 'USC-PSI-Lab/psi-data', 'official_revision': OFFICIAL_REVISION,
        'official_archive_sha256': OFFICIAL_ARCHIVE_SHA256, 'boundary': boundary, 'formal_evaluation_eligible': False,
        'maximum_steps_including_warmup': 400, 'maximum_simulated_seconds': 8., 'effective_native_success_criteria': .9,
        'source_split_labels_preserved': True, 'independent_heldout_split_proven': False,
        'initialization_profile': profile, 'original_official_initialization': profile == 'official_original',
        'support_geometry_manifest': support_identity,
        'support_adjusted_cases': sum(case['bend_support_initialization']['applied'] for case in cases)
            if profile == 'stable_support_v1' else 0,
        'case_order': 'support_adjusted_first_stable_original_order' if getattr(args, 'prioritize_support_adjusted', False)
            else 'original_level_case_order',
        'evaluation_policy': evaluation_policy(args)}


def commands(args, case, directory):
    policy = evaluation_policy(args)
    profile = initialization_profile(args)
    options = ['--rtc-mode', 'test_time', '--execute-steps', '15', '--rtc-inference-delay', '0',
        '--min-exec-horizon', '15', '--pig-mask-schedule', 'exponential', '--pig-guidance-alpha', '0.9']
    server = [str(MODEL_PY), str(REPO/'scripts/4090_simple/serve_evaluation.py'), '--run-dir', str(args.deployment),
        '--ckpt-step', str(args.checkpoint_step), '--sampling-seed', str(case['sampling_seed']),
        '--output', str(directory/'server'), '--port', str(args.port), *options]
    simulator = [str(SIM_PY), str(REPO/'scripts/4090_simple/run_closed_loop.py'), '--data-root', str(args.data_root),
        '--sonic-resources', str(args.runtime_root/'resources/sonic_v1_1'), '--output', str(directory/'simulation'),
        '--tasks', TASK, '--episode-start', str(case['case_index']), '--episodes-per-task', '1',
        '--dr-level', str(case['level']), '--seed', '0', '--seconds', str(case['protocol']['maximum_simulated_seconds']),
        *options, '--expected-checkpoint-step', str(args.checkpoint_step), '--port', str(args.port),
        '--bend-official-protocol', '--sync-observations', '--isaac-observation-only', '--sim-mode', 'mujoco_isaac',
        '--diagnostic-unverified-sonic', '--pd-profile', 'official_v1_1_ankle_pitch_1_5', '--hand-torque-profile', 'mjcf_limits',
        '--warmup-control', 'live_pose_sonic_v1_1', '--warmup-hand-policy', 'measured', '--body-token-projection', 'none',
        '--video', '--worker-wall-timeout', '1800', '--worker-task', TASK]
    if profile == 'stable_support_v1':
        simulator += ['--initialization-profile', profile, '--support-geometry-manifest',
            str(Path(args.support_geometry_manifest).resolve(strict=True))]
    return {'server': server, 'simulator': simulator}


def validate_native_reward_ledger(ticks):
    """Cross-check the unchanged native reward; geometry never replaces returned success."""
    baselines = []
    for tick in ticks:
        actual = tick.get('bend_native_state', {})
        baseline = actual.get('baseline_target_z_m')
        require(type(baseline) in (int, float) and math.isfinite(baseline), 'Missing original BendPick reward baseline')
        baselines.append(baseline)
        target = actual.get('target_xyz_m')
        require(isinstance(target, list) and len(target) == 3 and all(math.isfinite(value) for value in target),
            'Missing measured BendPick target position')
        expected_reward = max(0., min(1., (target[2]-baseline)/.1))
        close_number(tick['reward'], expected_reward, 'Original BendPick reward changed')
        close_number(actual['native_reward'], tick['reward'], 'BendPick reward evidence differs')
        require(actual['native_success'] is tick['terminated'] and tick['terminated'] == (tick['reward'] >= .9),
            'Original BendPick success flag changed')
    require(all(value == baselines[0] for value in baselines), 'Native BendPick baseline changed')


def validate_initialization_safety(safety):
    require(isinstance(safety, dict) and
        safety.get('contract') == 'cvpr-bend-pick-revised-startup-passive-verification-v1' and
        safety.get('status') == 'passed', 'Missing or failed revised initialization safety verification')
    for key, expected in (('observed_physics_substeps', 201), ('required_physics_substeps', 201),
            ('warmup_control_ticks', 50), ('native_reset_physics_substeps', 1),
            ('robot_target_contact_count', 0), ('physics_steps_added', 0)):
        require(type(safety.get(key)) is int and safety[key] == expected,
            'Initialization safety substep/contact/budget ledger differs: '+key)
    require(safety.get('engineering_displacement_limit_m') == .002 and
        safety.get('engineering_rotation_limit_degrees') == 2., 'Initialization diagnostic thresholds differ')
    for key, limit in (('maximum_target_displacement_m', .002), ('maximum_target_rotation_degrees', 2.)):
        value = safety.get(key)
        require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit,
            'Initialization target drift or rotation exceeds diagnostic threshold')
    samples = safety.get('samples')
    require(isinstance(samples, list) and len(samples) == 201 and safety.get('contacts') == [],
        'Initialization safety lacks complete contact/sample evidence')
    displacements, rotations = [], []
    for index, sample in enumerate(samples, 1):
        require(type(sample.get('substep')) is int and sample['substep'] == index,
            'Initialization sample substep sequence differs')
        sample_time = sample.get('sim_time_s')
        require(type(sample_time) in (int, float) and math.isfinite(sample_time),
            'Initialization sample timestamp is nonfinite')
        close_number(sample_time, .005*index, 'Initialization sample time differs from actual200Hz physics')
        for key, length in (('target_xyz_m', 3), ('target_quaternion_wxyz', 4)):
            value = sample.get(key)
            require(isinstance(value, list) and len(value) == length and
                all(type(item) in (int, float) and math.isfinite(item) for item in value),
                'Initialization sample target pose is incomplete or nonfinite')
        close_number(sum(value*value for value in sample['target_quaternion_wxyz']), 1.,
            'Initialization sample target quaternion is not unit length')
        require(type(sample.get('robot_target_contacts')) is int and sample['robot_target_contacts'] == 0,
            'Initialization sample contains robot-target contact')
        for key, limit, values in (('target_displacement_m', .002, displacements),
                ('target_rotation_degrees', 2., rotations)):
            value = sample.get(key)
            require(type(value) in (int, float) and math.isfinite(value) and 0 <= value <= limit,
                'Initialization sample drift or rotation exceeds diagnostic threshold')
            values.append(value)
    close_number(max(displacements), safety['maximum_target_displacement_m'],
        'Initialization displacement maximum disagrees with sample ledger')
    close_number(max(rotations), safety['maximum_target_rotation_degrees'],
        'Initialization rotation maximum disagrees with sample ledger')
    require(safety.get('finite_states') is True and safety.get('private_FK_only') is True and
        all(safety.get(key) is False for key in ('live_physical_state_writes', 'controller_commands_modified',
            'policy_phase_interventions')), 'Initialization observer changed live state/control or used nonfinite states')


def validate_support_identity(result, episode, case, frozen):
    profile = frozen.get('initialization_profile', 'official_original')
    require(case.get('initialization_profile', 'official_original') == profile and
        result.get('initialization_profile', 'official_original') == profile and
        episode.get('initialization_profile', 'official_original') == profile,
        'BendPick initialization profile differs')
    if profile == 'official_original':
        require(episode.get('bend_support_initialization') is None and
            episode.get('bend_support_model_verification') is None,
            'Original initialization contains an undeclared support revision')
        return
    require(profile == 'stable_support_v1' and frozen.get('contract') == STABLE_SUPPORT_CONTRACT,
        'Stable support must use its independent revised-initialization contract')
    receipt = case.get('bend_support_initialization')
    require(isinstance(receipt, dict) and receipt.get('profile') == profile and
        receipt.get('contract') == 'cvpr-bend-pick-stable-support-initialization-v1' and
        episode.get('bend_support_initialization') == receipt,
        'Frozen support derivation differs from actual worker receipt')
    identity = case.get('bend_support_geometry_manifest')
    require(isinstance(identity, dict) and frozen.get('support_geometry_manifest') == identity,
        'Frozen support geometry manifest identity differs')
    minimal_identity = {key: identity[key] for key in ('path', 'sha256', 'size_bytes')}
    require(receipt.get('geometry_manifest_identity') == minimal_identity and
        episode.get('bend_support_geometry_manifest') == minimal_identity,
        'Actual support geometry manifest differs')
    validate_initialization_safety(episode.get('bend_initialization_safety'))
    gate = episode.get('bend_support_model_verification', {})
    receipt_sha = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()
    require(gate.get('status') == 'passed' and gate.get('profile') == profile and
        gate.get('applied') is receipt['applied'] and gate.get('read_only') is True and
        type(gate.get('physics_steps_taken')) is int and gate['physics_steps_taken'] == 0 and
        gate.get('initialization_receipt_sha256') == receipt_sha,
        'Actual support model was not verified read-only before first physics')
    expected_com = [value+shift for value, shift in zip(receipt['after_com_xyz_m'],
        receipt['actual_world_translation_m'])]
    actual_com = gate.get('actual_target_com_world_m')
    require(isinstance(actual_com, list) and len(actual_com) == len(expected_com) == 3,
        'Missing actual target support center evidence')
    for actual, expected in zip(actual_com, expected_com):
        close_number(actual, expected, 'Actual target support center differs from frozen derivation')


def validate_case(directory, case, frozen):
    """Read actual trace, both HTTP sides and decoded video; never infer success from geometry."""
    from run_closed_loop import worker_result_errors
    base = directory / "simulation" / TASK
    result_path = base / "result.json"
    result = read(result_path)
    errors = worker_result_errors(result, 1)
    episodes = result.get("episodes", [])
    # The generic smoke gate requires learned actions. Official native success
    # during causal warmup is still native success, not a technical/task failure.
    if len(episodes) == 1 and episodes[0].get("task_success_observed") is True:
        errors = [error for error in errors if "did not execute the real model/controller loop" not in error]
    require(not errors, "Simulator result is incomplete: " + repr(errors))
    episode = result["episodes"][0]
    validate_support_identity(result, episode, case, frozen)
    require(episode["episode_index"] == case["case_index"] and episode["dr_level"] == case["level"]
            and episode["seed"] == 0 and episode["source_scene_reset_dr_level"] is None, "Wrong case/reset identity")
    require(episode["source_metadata_sha256"] == case["metadata_sha256"]
            and episode["local_environment_config"] == case["local_environment_config"], "Frozen official configuration differs")
    protocol = episode['bend_protocol']
    require(episode.get('task_id') == TASK_ID and protocol == case['protocol'], 'Native BendPick protocol differs')
    require(episode.get('bend_source_environment_config') == case['source_environment_config'] and
        episode.get('bend_reset_derivation') == case['reset_derivation'] and
        episode.get('bend_adaptation') == case['adaptation'] and
        episode.get('bend_environment_binding', {}).get('matched') is True, 'BendPick reset/frame binding differs')
    require(result.get('bend_adapter', {}).get('frame_binding', {}).get('installed') is True,
        'BendPick frame hooks were not installed before reset')
    render = result["render_contract"]
    require(render["render_engine"] == "Isaac RTX" and render["authoritative_physics"] == "MuJoCo"
            and render["sync_no_time_advance"] is True and render["isaac_observation_only"] is True, "Render/physics binding differs")
    if "sources" in frozen:
        require(render.get("environment_source_sha256") == frozen["sources"]["SIMPLE/envs/sonic_loco_manip.py"]["sha256"],
                "Actual simulator environment source differs from frozen source")
    episode_dir = base / f"episode-{case['case_index']}"
    ticks = json_lines(episode_dir / "trace.jsonl")
    require(0 < len(ticks) <= 400 and len(ticks) == episode["frames"] == episode["warmup_frames"] + episode["policy_frames"],
            "Incomplete tick ledger")
    require(episode["warmup_frames"] == min(len(ticks), 50), "Causal warmup was removed or extended beyond the official budget")
    require(all(type(tick["terminated"]) is bool and type(tick["truncated"]) is bool for tick in ticks), "Native flags are not bool")
    for index, tick in enumerate(ticks):
        require(tick["phase"] == ("warmup_measured_pose_pd" if index < 10 else
                                  "warmup_live_pose_sonic_v1_1" if index < 50 else "policy_latent"),
                "Warmup/policy phase ledger differs")
        close_number(tick["next_sim_time_s"] - tick["sim_time_s"], .02, "Wrong simulation tick")
        if index:
            close_number(tick["sim_time_s"], ticks[index-1]["next_sim_time_s"], "Discontinuous simulation clock")
    validate_native_reward_ledger(ticks)
    require(all(not tick["terminated"] and not tick["truncated"] for tick in ticks[:-1]), "Continued after native terminal")
    native = ticks[-1]["terminated"]
    require(native == episode["task_success_observed"] and (episode["status"] == "task_terminated") == native, "Native success mismatch")
    if not native:
        require(episode["status"] == "env_truncated" and len(ticks) == 400 and ticks[-1]["truncated"], "Unsuccessful case lacks full official budget")
    close_number(episode["simulated_seconds"], len(ticks) * .02, "Wrong total time including warmup")
    # Native success even during warmup is retained; policy execution is not an extra success gate.
    requests = json_lines(directory / "server/requests.jsonl")
    http = read(episode_dir / "http.json")
    require([row["ordinal"] for row in requests] == list(range(1, len(requests)+1)), "Server request ordinal gap")
    require(all(row["status"] in (200, 425) for row in requests), "Server HTTP technical failure")
    require(all(row["rng_before"] == row["rng_after"] for row in requests if row["status"] == 425), "Warmup rejection consumed sampling RNG")
    require(sum(row["status"] == 200 for row in requests) == episode["policy_chunks"]
            and sum(row["status"] == 425 for row in requests) == episode["http_425_count"], "Server request count differs")
    identities = {(row["session_id"], row["episode_id"]) for row in requests}
    require(len(identities) == 1 and next(iter(identities))[1] == episode["episode_id"], "Cross-episode server history")
    close_number(float(requests[0]["sim_time_s"]), ticks[0]["sim_time_s"], "Server history does not start at the first observation")
    session_id = next(iter(identities))[0]
    require(dict(Counter(row["endpoint"] for row in http)) == episode["http_counts"]
            and len(http) == result["http_request_count"] and episode["http_counts"].get("/reset") == 1
            and episode["http_counts"].get("/observe") == len(ticks), "Client causal history counts differ")
    require(all(row["history"]["episode_id"] == episode["episode_id"] and row["history"]["session_id"] == session_id
                and row["simulator_before_s"] == row["simulator_after_s"] for row in http), "HTTP identity/paused simulation differs")
    require(all(row["http_status"] == 200 for row in http if row["endpoint"] in ("/reset", "/observe")), "Observe/reset technical error")
    reset = next(row for row in http if row["endpoint"] == "/reset")["reset_response"]
    require(reset.get("history_frames") == 0 and reset.get("rtc_cleared") is True, "Fresh server history reset is unproven")
    acts = [row for row in http if row["endpoint"] == "/act"]
    require(len(acts) == len(requests), "Client/server action count differs")
    for client, server in zip(acts, requests):
        require(client["http_status"] == server["status"], "Client/server action status differs")
        close_number(float(client["history"]["sim_time_s"]), float(server["sim_time_s"]), "Client/server timestamp differs")
    observed = [row["history"]["sim_time_s"] for row in http if row["endpoint"] == "/observe"]
    require(observed == [row["sim_time_s"] for row in ticks], "History is not sourced from each current observation")
    policy = frozen["evaluation_policy"]
    rtc_execution = (validate_rtc_execution(result, ticks, requests, policy) if policy["rtc_enabled"] else None)
    video = Path(episode["video"]["path"]).resolve(strict=True)
    require(video == (episode_dir / "rollout.mp4").resolve() and episode["video"]["ffmpeg_returncode"] == 0, "Wrong/incomplete case video")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video)], timeout=120))["streams"]
    require(len(probe) == 1 and probe[0]["r_frame_rate"] == "50/1" and
            (probe[0]["width"], probe[0]["height"]) == (640, 360) and
            int(probe[0]["nb_read_frames"]) == episode["video"]["frames"] == len(ticks) + 1, "Decoded video frame/rate/geometry differs")
    paths = [result_path, episode_dir / "trace.jsonl", episode_dir / "http.json", video,
             directory / "server/startup.json", directory / "server/requests.jsonl", directory / "ONLINE_DINO_RUNTIME_ACCEPTANCE.json"]
    return {"status": "complete", "level": case["level"], "case_index": case["case_index"],
            "native_success": native, "fell": episode["fell"], "fell_before_first_policy": episode["fell_before_first_policy_command"],
            "ticks": len(ticks), "warmup_ticks": episode["warmup_frames"], "policy_ticks": episode["policy_frames"],
            "simulated_seconds": episode["simulated_seconds"], "video_probe": probe[0], "video": str(video),
            "evidence": [file_identity(path) for path in paths], "formal_evaluation_eligible": False,
            "model_sha256": frozen["model"]["sha256"], "evaluation_policy": policy, "rtc_execution": rtc_execution,
            "contract": frozen.get('contract', CONTRACT),
            "initialization_profile": frozen.get('initialization_profile', 'official_original'),
            "original_official_initialization": frozen.get('initialization_profile', 'official_original') == 'official_original',
            "source_environment_config": case['source_environment_config'],
            "reset_derivation": case['reset_derivation'],
            "bend_support_initialization": case.get('bend_support_initialization'),
            "bend_support_geometry_manifest": case.get('bend_support_geometry_manifest'),
            "bend_support_model_verification": episode.get('bend_support_model_verification'),
            "bend_initialization_safety": episode.get('bend_initialization_safety')}



def execute(args, case, frozen, state):
    directory = args.output / f"level{case['level']}" / f"case_{case['case_index']:02}"
    directory.mkdir(parents=True, exist_ok=False)
    row = {"level": case["level"], "case_index": case["case_index"], "status": "starting",
           "directory": str(directory), "started_utc": now(), "sampling_seed": case["sampling_seed"],
           "initialization_profile": frozen.get('initialization_profile', 'official_original'),
           "bend_support_initialization": case.get('bend_support_initialization')}
    state["cases"].append(row)
    write(args.output / "state.json", state)
    command = commands(args, case, directory)
    write(directory / "COMMANDS.json", command)
    write(directory / "FROZEN_CASE.json", case)
    processes, interrupted = [], False
    try:
        for identity in [frozen["model"], frozen["training_manifest"], *frozen["sources"].values(), *frozen["metadata_files"]]:
            unchanged(identity)
        preflight(args)
        with (directory / "server.log").open("xb") as log:
            server = subprocess.Popen(command["server"], cwd=REPO, env=environment(args, MODEL_PY), stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(server)
            mark_owned(server)
            row["server_identity"] = {"pid": server.pid, **server.owned_identity}
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 480
            while True:
                require(server.poll() is None, "Model server exited; inspect retained log")
                if (directory / "server/startup.json").is_file():
                    try:
                        with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError, OSError):
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model server startup exceeded480s")
                time.sleep(1)
            acceptance = runtime_gate(read(directory / "server/startup.json"), read(args.training_manifest)["encoder"],
                args.expected_model_sha256, case["sampling_seed"], server.pid, evaluation_policy(args))
            acceptance["training_manifest_sha256"] = frozen["training_manifest"]["sha256"]
            write(directory / "ONLINE_DINO_RUNTIME_ACCEPTANCE.json", acceptance)
            row["status"] = "simulating"
            write(args.output / "state.json", state)
            with (directory / "simulation.log").open("xb") as simlog:
                simulator = subprocess.Popen(command["simulator"], cwd=REPO, env=environment(args, SIM_PY), stdout=simlog,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(simulator)
                mark_owned(simulator)
                row["simulator_identity"] = {"pid": simulator.pid, **simulator.owned_identity}
                require(simulator.wait(timeout=2100) == 0, "Simulator failed; inspect retained log/result")
            for identity in [frozen["model"], frozen["training_manifest"], *frozen["sources"].values(), *frozen["metadata_files"]]:
                unchanged(identity)
            row.update(validate_case(directory, case, frozen))
    except BaseException as error:
        interrupted = not isinstance(error, Exception)
        row.update(status="technical_error", error=repr(error), traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        for process in reversed(processes):
            try:
                stop_owned(process)
            except Exception as error:
                cleanup_errors.append(repr(error))
        if cleanup_errors:
            row.update(status="technical_error", cleanup_errors=cleanup_errors)
        row["finished_utc"] = now()
        write(directory / ("CASE_SUMMARY.json" if row["status"] == "complete" else "TECHNICAL_FAILURE.json"), row)
        write(args.output / "state.json", state)
    if interrupted:
        raise KeyboardInterrupt("Campaign interrupted; owned children cleaned, evidence retained")



def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment", "data-root", "training-manifest", "output", "runtime-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=5000)
    parser.add_argument("--port", type=int, default=21115)
    parser.add_argument("--sampling-seed", type=int, default=20261111)
    parser.add_argument("--rtc-mode", choices=("test_time",), default="test_time")
    parser.add_argument("--execute-steps", type=int, help="Exact15 actions with test_time RTC, consuming25 simulation ticks per window")
    parser.add_argument('--initialization-profile', choices=('official_original', 'stable_support_v1'), default='official_original')
    parser.add_argument('--support-geometry-manifest', type=Path)
    parser.add_argument('--prioritize-support-adjusted', action='store_true',
        help='Stable-support profile only: execute geometry-adjusted cases first, retaining all30 original cases/seeds')
    args = parser.parse_args(argv)
    initialization_profile(args)
    require(len(args.expected_model_sha256) == 64 and all(char in "0123456789abcdef" for char in args.expected_model_sha256), "Expected SHA must be lowercase SHA256")
    require(args.checkpoint_step > 0 and 1024 <= args.port <= 65535 and 0 <= args.sampling_seed <= 2**32 - 30, "Invalid step/port/seed")
    policy = evaluation_policy(args)
    for name in ("deployment", "data_root", "training_manifest", "runtime_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    require(args.output.is_absolute() and not args.output.exists(), "Campaign must be a fresh absolute directory")
    args.output = args.output.resolve()
    require(not args.output.is_relative_to(REPO) and not args.output.is_relative_to(args.deployment)
            and not args.output.is_relative_to(args.data_root), "Output overlaps source/deployment/resources")
    frozen = json.loads(json.dumps(freeze_inputs(args), default=_json_value, allow_nan=False))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "FROZEN_CAMPAIGN.json", frozen)
    state = {"contract": frozen["contract"], "status": "running", "created_utc": now(), "planned_cases": 30,
             "checkpoint_step": args.checkpoint_step, "model_sha256": args.expected_model_sha256,
             "task_id": TASK_ID, "cases": [], "formal_evaluation_eligible": False, "boundary": frozen["boundary"],
             "technical_errors_are_task_failures": False, "automatic_retries": 0, "evaluation_policy": policy,
             "initialization_profile": frozen['initialization_profile'],
             "original_official_initialization": frozen['original_official_initialization'],
             "support_geometry_manifest": frozen['support_geometry_manifest'],
             "support_adjusted_cases": frozen['support_adjusted_cases'], "case_order": frozen['case_order']}
    write(args.output / "state.json", state)
    try:
        for case in frozen["cases"]:
            execute(args, case, frozen, state)
    finally:
        code = finish(state)
        write(args.output / "FINAL_EVALUATION_ACCEPTANCE.json", state)
        write(args.output / "state.json", state)
    return code


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Supervisor interrupted; retain evidence and clean only owned children")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
