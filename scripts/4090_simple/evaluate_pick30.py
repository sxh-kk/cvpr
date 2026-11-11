"""Official fixed30 XMove Pick scenes, private training tasks and audited RTC.

Only saved scene metadata is used. Native SONIC scenes are bound to the
same frozen PICO task as training; this is not archived-controller equivalence or a claim that
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
from run_closed_loop import _json_value
from pick_eval_protocol import profile_spec, pick_official_protocol, derive_pick_eval_reset, adapt_pick_config
REPO = Path(__file__).resolve().parents[2]
CONTRACT = 'cvpr-pick-fixed-official30-sonic-rtc-diagnostic-v1'
READY_CONTRACT = 'cvpr-pick-local-deployment-v1'
READY_STATUS = 'local_pick_checkpoint_prepared'
INITIALIZATION_PROFILE = 'native_saved_scene_observed_v1'
SAVE_STEPS = (1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 10000)
OFFICIAL_REVISION = 'e78fb93cc28912a3031a10b8656d32d7f0a2b867'
OFFICIAL_ARCHIVE_SHA256 = {
 'xmove_pick': 'af6f1048d1cb8070a03b33fa5ca9c4eb8f903b0a5f863eab4e9fbe21c389eb79',
 'xmove_bend_pick': 'a0c597efa3c5bfbe2e7b75af7ba23c0e94236834f2c474d4d40181eefcdd20bc'}
OFFICIAL_METADATA_SHA256 = {
 'xmove_pick': ('9d00683983dbdc967551bcc792ec32cd25df4b3e705f577458feb5810b1930e3',
 '6e3c981e46157e635e57e440ca096f722be67b5f703d636313b65aede0407e5e',
 '155d397303641ed1ec7cbae05b9376411e22a4a9778026009f5107c78d3ad32b'),
 'xmove_bend_pick': ('e0467ce60e1b4dddd97e62fc1790cc644e937c469861e3d9acda91f43fdaeaf0',
 'c3ad6b63cbeb25e86f2be4a98638ff708a0f3389baf415d020c02a9c124367f1',
 'afc0bcb899c0981a8e69a98aebf19bd20f3cdbb3b31a57e746a963c9453e7fc5')}
BOUNDARY = ('Official fixed XMove task L0/L1/L2 scenes with explicit private PICO SONIC controller adaptation; '
 'source split=train preserved, not proof of independent heldout or new-scene generalization; '
 'no teacher action/state replay; original target/table/object poses and native reward preserved; '
 'native_saved_scene_observed_v1 keeps startup displacement/rotation as engineering warnings, not native-success filters')
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
        'This Pick campaign requires the requested test_time RTC / exact15-action execution window')
    return policy


def source_identities(profile):
    profile_data = profile_spec(profile)
    adapter = importlib.import_module(profile_data["adapter_module"])
    sources = {name: file_identity(Path(__file__).with_name(name)) for name in (
        'evaluate_pick30.py', 'pick_eval_protocol.py', 'prepare_pick_eval.py', 'evaluate_mobile_pp30.py', 'run_closed_loop.py', 'serve_evaluation.py',
        profile_data['adapter_module']+'.py', 'capture_simple.py', 'video_capture.py', 'bend_initialization_monitor.py')}
    for relative in ('deploy/serve_psi0_simple.py', 'deploy/dual_source_runtime.py', 'models/psi0.py', 'models/dual_source_action.py'):
        sources['psi/'+relative] = file_identity(REPO/'VLMExpert/src/psi'/relative)
    spec = importlib.util.find_spec('simple')
    require(spec is not None and spec.submodule_search_locations, 'Installed SIMPLE source is unavailable')
    simple = Path(next(iter(spec.submodule_search_locations)))
    task_source = 'tasks/'+('g1_wholebody_xmove_pick_teleop.py' if profile == 'xmove_pick' else 'g1_wholebody_xmove_bend_pick_teleop.py')
    for relative in (task_source, 'envs/sonic_loco_manip.py', 'envs/base_dual_env.py',
            'baselines/dual_source_psi0.py', 'baselines/sonic_v1_1.py', 'baselines/live_pose_sonic_warmup.py'):
        sources['SIMPLE/'+relative] = file_identity(simple/relative)
    require(sources['SIMPLE/'+task_source]['sha256'] == adapter.OFFICIAL_TASK_SOURCE_SHA256,
        'The unchanged native Pick task source differs')
    return sources


def freeze_inputs(args):
    from capture_simple import remap_episode_paths
    profile = args.profile
    spec = profile_spec(profile)
    require(type(args.checkpoint_step) is int and args.checkpoint_step in SAVE_STEPS, 'Unselected checkpoint step')
    ready_path = args.deployment/'LOCAL_RUN_READY.json'
    ready = read(ready_path)
    require(ready.get('contract') == READY_CONTRACT and ready.get('status') == READY_STATUS and
        ready.get('profile') == profile and ready.get('task_id') == spec['task_id'] and
        ready.get('source_task_id') == spec['source_task_id'] and ready.get('step') == ready.get('task_step') == args.checkpoint_step and
        ready.get('joint_sha256') == args.expected_model_sha256 and ready.get('checkpoint_save_completed') is True and
        ready.get('training_completed') is True and
        ready.get('selected_checkpoint_is_final') is (args.checkpoint_step == 10000) and ready.get('component_load_validation', {}).get('status') == 'passed',
        'Independent Pick CPU deployment/save completion is required')
    manifest = file_identity(args.training_manifest)
    require(manifest['sha256'] == ready.get('training_manifest_sha256'), 'Training DINO manifest differs from deployment')
    encoder = read(args.training_manifest).get('encoder')
    require(isinstance(encoder, dict) and encoder.get('shape') == [5, 1024], 'Training encoder contract missing')
    model = file_identity(args.deployment/'checkpoints'/f'ckpt_{args.checkpoint_step}'/'model.safetensors')
    require(model['sha256'] == args.expected_model_sha256, 'Actual checkpoint bytes differ')
    sources = source_identities(profile)
    sources['deployment_ready'] = file_identity(ready_path)
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
        path = args.data_root/'eval'/spec['source_task']/f'dr-level-{level}'/'meta/episodes.jsonl'
        identity = file_identity(path)
        require(identity['sha256'] == OFFICIAL_METADATA_SHA256[profile][level], 'Require the frozen official Pick metadata')
        rows = json_lines(path)
        require(len(rows) == 10 and sorted(row['episode_index'] for row in rows) == list(range(10)),
            'Each level requires exactly ten original cases0..9')
        for row in sorted(rows, key=lambda value: value['episode_index']):
            original = json.loads(row['environment_config'])
            native = remap_episode_paths(original, args.data_root)
            protocol = pick_official_protocol(native, profile)
            derived, derivation = derive_pick_eval_reset(native, profile)
            adapted, adaptation = adapt_pick_config(derived, profile)
            cases.append({'level': level, 'case_index': row['episode_index'], 'flat_index': level*10+row['episode_index'],
                'source_row': row, 'source_environment_config': native, 'local_environment_config': adapted,
                'reset_derivation': derivation, 'adaptation': adaptation, 'protocol': protocol,
                'metadata_sha256': identity['sha256'], 'environment_seed': 0,
                'sampling_seed': args.sampling_seed+level*10+row['episode_index'],
                'profile': profile, 'task_id': spec['task_id'], 'source_task_id': spec['source_task_id'],
                'initialization_profile': INITIALIZATION_PROFILE})
        unchanged(identity)
        metadata_files.append(identity)
    return {'contract': CONTRACT, 'profile': profile, 'task_id': spec['task_id'], 'source_task_id': spec['source_task_id'],
        'prepared_utc': now(), 'checkpoint_step': args.checkpoint_step, 'model': model, 'deployment': str(args.deployment),
        'training_manifest': manifest, 'metadata_files': metadata_files, 'sources': sources, 'cases': cases,
        'official_repository': 'USC-PSI-Lab/psi-data', 'official_revision': OFFICIAL_REVISION,
        'official_archive_sha256': OFFICIAL_ARCHIVE_SHA256[profile], 'boundary': BOUNDARY, 'formal_evaluation_eligible': False,
        'maximum_steps_including_warmup': 800, 'maximum_simulated_seconds': 16., 'effective_native_success_criteria': .9,
        'source_split_labels_preserved': True, 'independent_heldout_split_proven': False,
        'original_official_initialization': True, 'initialization_profile': INITIALIZATION_PROFILE,
        'world_translation_m': [0., 0., 0.],
        'case_order': 'original_level_case_order', 'automatic_retries': 0,
        'evaluation_policy': evaluation_policy(args)}


def commands(args, case, directory):
    policy = evaluation_policy(args)
    task = profile_spec(args.profile)['task']
    options = ['--rtc-mode', 'test_time', '--execute-steps', '15', '--rtc-inference-delay', '0',
        '--min-exec-horizon', '15', '--pig-mask-schedule', 'exponential', '--pig-guidance-alpha', '0.9']
    server = [str(MODEL_PY), str(REPO/'scripts/4090_simple/serve_evaluation.py'), '--run-dir', str(args.deployment),
        '--ckpt-step', str(args.checkpoint_step), '--sampling-seed', str(case['sampling_seed']),
        '--output', str(directory/'server'), '--port', str(args.port), *options]
    simulator = [str(SIM_PY), str(REPO/'scripts/4090_simple/run_closed_loop.py'), '--data-root', str(args.data_root),
        '--sonic-resources', str(args.runtime_root/'resources/sonic_v1_1'), '--output', str(directory/'simulation'),
        '--tasks', task, '--episode-start', str(case['case_index']), '--episodes-per-task', '1',
        '--dr-level', str(case['level']), '--seed', '0', '--seconds', str(case['protocol']['maximum_simulated_seconds']),
        *options, '--expected-checkpoint-step', str(args.checkpoint_step), '--port', str(args.port),
        '--pick-official-protocol', '--pick-profile', args.profile, '--sync-observations', '--isaac-observation-only', '--sim-mode', 'mujoco_isaac',
        '--diagnostic-unverified-sonic', '--pd-profile', 'official_v1_1_ankle_pitch_1_5', '--hand-torque-profile', 'mjcf_limits',
        '--warmup-control', 'live_pose_sonic_v1_1', '--warmup-hand-policy', 'measured', '--body-token-projection', 'none',
        '--video', '--worker-wall-timeout', '1800', '--worker-task', task]
    return {'server': server, 'simulator': simulator}


def validate_native_reward_ledger(ticks):
    """Cross-check the unchanged native reward; geometry never replaces returned success."""
    baselines = []
    for tick in ticks:
        actual = tick.get('pick_native_state', {})
        baseline = actual.get('baseline_target_z_m')
        require(type(baseline) in (int, float) and math.isfinite(baseline), 'Missing original Pick reward baseline')
        baselines.append(baseline)
        target = actual.get('target_xyz_m')
        require(isinstance(target, list) and len(target) == 3 and all(math.isfinite(value) for value in target),
            'Missing measured Pick target position')
        expected_reward = max(0., min(1., (target[2]-baseline)/.1))
        close_number(tick['reward'], expected_reward, 'Original Pick reward changed')
        close_number(actual['native_reward'], tick['reward'], 'Pick reward evidence differs')
        require(actual['native_success'] is tick['terminated'] and tick['terminated'] == (tick['reward'] >= .9),
            'Original Pick success flag changed')
    require(all(value == baselines[0] for value in baselines), 'Native Pick baseline changed')


def quaternion_angle_degrees(a, b):
    return math.degrees(2*math.acos(min(1., abs(sum(x*y for x,y in zip(a,b))))))


def close_initialization_metric(actual, expected, key):
    # acos amplifies last-bit dot-product differences near identical rotations.
    # Ten microdegrees cannot conceal the two-degree engineering reference.
    if key.endswith('degrees'):
        require(math.isfinite(actual) and math.isclose(actual,expected,rel_tol=0,abs_tol=1e-5),
            'Initialization angle disagrees with independently reconstructed pose: '+key)
    else:
        close_number(actual,expected,'Initialization metric disagrees with independently reconstructed pose: '+key)


def validate_initialization_safety(safety):
    """Require genuine passive observation; report engineering drift separately."""
    require(isinstance(safety, dict) and
        safety.get('contract') == 'cvpr-pick-official-startup-passive-verification-v1' and
        safety.get('initialization_profile') == INITIALIZATION_PROFILE and
        safety.get('status') == 'observation_complete', 'Missing or incomplete native initialization observation')
    for key, expected in (('observed_physics_substeps',201),('required_physics_substeps',201),
            ('warmup_control_ticks',50),('native_reset_physics_substeps',1),
            ('robot_target_contact_count',0),('physics_steps_added',0)):
        require(type(safety.get(key)) is int and safety[key] == expected,
            'Initialization substep/contact/budget ledger differs: '+key)
    require(safety.get('engineering_displacement_limit_m') == .002 and
        safety.get('engineering_rotation_limit_degrees') == 2., 'Engineering reference thresholds differ')
    for key in ('finite_states','private_FK_only'):
        require(safety.get(key) is True, 'Incomplete/nonfinite/private observer evidence: '+key)
    require(all(safety.get(key) is False for key in ('live_physical_state_writes',
        'controller_commands_modified','policy_phase_interventions')), 'Observer changed state/control/policy')
    samples=safety.get('samples')
    require(isinstance(samples,list) and len(samples)==201 and safety.get('contacts')==[],
        'Initialization lacks complete contact/sample evidence')
    def vector(value,length):
        require(isinstance(value,list) and len(value)==length and
            all(type(v) in (int,float) and math.isfinite(v) for v in value), 'Nonfinite/incomplete initialization pose')
        if length==4:
            close_number(sum(v*v for v in value),1.,'Initialization quaternion is not unit length')
        return value
    initial=vector(safety.get('initial_target_xyz_m'),3)
    initial_quat=vector(safety.get('initial_target_quaternion_wxyz'),4)
    displacements,rotations=[],[]
    for index,row in enumerate(samples,1):
        require(type(row.get('substep')) is int and row['substep']==index, 'Initialization sample sequence differs')
        stamp=row.get('sim_time_s')
        require(type(stamp) in (int,float) and math.isfinite(stamp), 'Initialization sample time is nonfinite')
        close_number(stamp,.005*index,'Initialization sample time differs from real200Hz physics')
        xyz=vector(row.get('target_xyz_m'),3);quat=vector(row.get('target_quaternion_wxyz'),4)
        require(type(row.get('robot_target_contacts')) is int and row['robot_target_contacts']==0,
            'Initialization sample contains robot-target contact')
        displacement=math.dist(initial,xyz);rotation=quaternion_angle_degrees(initial_quat,quat)
        for key,actual in (('target_displacement_m',displacement),('target_rotation_degrees',rotation)):
            value=row.get(key)
            require(type(value) in (int,float) and math.isfinite(value) and value>=0., 'Nonfinite/negative initialization metric')
            close_initialization_metric(value,actual,key)
        displacements.append(displacement);rotations.append(rotation)
    maximum=max(displacements);maximum_rotation=max(rotations)
    for key,actual in (('maximum_target_displacement_m',maximum),('maximum_target_rotation_degrees',maximum_rotation)):
        value=safety.get(key)
        require(type(value) in (int,float) and math.isfinite(value) and value>=0., 'Nonfinite initialization maximum')
        close_initialization_metric(value,actual,key)
    warnings=[]
    if maximum>.002:warnings.append('target_displacement_exceeds_2mm')
    if maximum_rotation>2.:warnings.append('target_rotation_exceeds_2deg')
    require(safety.get('engineering_reference_warnings')==warnings and
        safety.get('engineering_reference_status')==('failed_or_incomplete' if warnings else 'passed'),
        'Engineering reference warning/status disagrees with independently reconstructed samples')
    tail=[row for row in samples if row['sim_time_s']>=samples[-1]['sim_time_s']-.2-1e-12]
    metrics={'window_duration_s':.2,'sample_count':len(tail),'start_sim_time_s':tail[0]['sim_time_s'],
        'end_sim_time_s':tail[-1]['sim_time_s'],
        'maximum_pairwise_displacement_m':max(math.dist(a['target_xyz_m'],b['target_xyz_m']) for a in tail for b in tail),
        'endpoint_displacement_m':math.dist(tail[0]['target_xyz_m'],tail[-1]['target_xyz_m']),
        'maximum_pairwise_rotation_degrees':max(quaternion_angle_degrees(a['target_quaternion_wxyz'],
            b['target_quaternion_wxyz']) for a in tail for b in tail),
        'endpoint_rotation_degrees':quaternion_angle_degrees(tail[0]['target_quaternion_wxyz'],tail[-1]['target_quaternion_wxyz'])}
    stored=safety.get('tail_settling_metrics')
    require(isinstance(stored,dict) and set(stored)==set(metrics), 'Missing/extra tail settling metrics')
    require(type(stored.get('sample_count')) is int and stored['sample_count']==41, 'Tail window sample count differs')
    for key,value in metrics.items():
        require(type(stored[key]) in (int,float) and math.isfinite(stored[key]), 'Nonfinite tail metric')
        close_initialization_metric(stored[key],value,key)
    return warnings


def validate_pick_identity(result, episode, case, frozen):
    spec = profile_spec(frozen['profile'])
    require(frozen.get('contract') == CONTRACT and frozen.get('task_id') == spec['task_id'] and
        frozen.get('source_task_id') == spec['source_task_id'] and case.get('profile') == frozen['profile'] and
        case.get('task_id') == spec['task_id'] and case.get('source_task_id') == spec['source_task_id'],
        'Pick frozen profile/task identity differs')
    require(all(value.get('initialization_profile') == INITIALIZATION_PROFILE for value in
        (frozen,case,result,episode,episode.get('pick_protocol',{}))), 'Native initialization profile identity differs')
    adapter = result.get('pick_adapter', {})
    require(adapter.get('initialization_profile') == INITIALIZATION_PROFILE, 'Private adapter initialization profile differs')
    require(adapter.get('adaptation') == case['adaptation'] and adapter.get('reset_derivation') == case['reset_derivation'] and
        adapter.get('protocol') == case['protocol'], 'Private Pick adapter differs from frozen reset')
    require(episode.get('pick_initialization_safety', {}).get('profile') == frozen['profile'],
        'Initialization monitor profile differs')
    safety=episode.get('pick_initialization_safety')
    validate_initialization_safety(safety)
    origin=case['local_environment_config']['layout']['actors']['target']['pose']
    require(math.dist(safety['initial_target_xyz_m'],origin['position'])<=1e-6 and
        quaternion_angle_degrees(safety['initial_target_quaternion_wxyz'],origin['quaternion'])<=1e-3,
        'Zero-step observed target pose differs from untouched saved initialization')
    require(case['adaptation'].get('world_translation_m') == [0., 0., 0.], 'XMove may not inherit Bend world shift')
    require(result.get('server_info', {}).get('ckpt_step') == frozen['checkpoint_step'] and
        result['server_info'].get('run_dir') == frozen['deployment'], 'Actual server checkpoint step/deployment differs')


def validate_case(directory, case, frozen):
    """Read actual trace, both HTTP sides and decoded video; never infer success from geometry."""
    from run_closed_loop import worker_result_errors
    spec = profile_spec(frozen['profile'])
    TASK, TASK_ID = spec['task'], spec['task_id']
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
    validate_pick_identity(result, episode, case, frozen)
    startup = read(directory/'server/startup.json')
    runtime_gate(startup, read(frozen['training_manifest']['path'])['encoder'], frozen['model']['sha256'],
        case['sampling_seed'], startup['pid'], frozen['evaluation_policy'])
    require(episode["episode_index"] == case["case_index"] and episode["dr_level"] == case["level"]
            and episode["seed"] == 0 and episode["source_scene_reset_dr_level"] is None, "Wrong case/reset identity")
    require(episode["source_metadata_sha256"] == case["metadata_sha256"]
            and episode["local_environment_config"] == case["local_environment_config"], "Frozen official configuration differs")
    protocol = episode['pick_protocol']
    require(episode.get('task_id') == TASK_ID and protocol == case['protocol'], 'Native Pick protocol differs')
    require(episode.get('pick_source_environment_config') == case['source_environment_config'] and
        episode.get('pick_reset_derivation') == case['reset_derivation'] and
        episode.get('pick_adaptation') == case['adaptation'] and
        episode.get('pick_environment_binding', {}).get('matched') is True, 'Pick reset/frame binding differs')
    render = result["render_contract"]
    require(render["render_engine"] == "Isaac RTX" and render["authoritative_physics"] == "MuJoCo"
            and render["sync_no_time_advance"] is True and render["isaac_observation_only"] is True, "Render/physics binding differs")
    if "sources" in frozen:
        require(render.get("environment_source_sha256") == frozen["sources"]["SIMPLE/envs/sonic_loco_manip.py"]["sha256"],
                "Actual simulator environment source differs from frozen source")
    episode_dir = base / f"episode-{case['case_index']}"
    ticks = json_lines(episode_dir / "trace.jsonl")
    require(0 < len(ticks) <= 800 and len(ticks) == episode["frames"] == episode["warmup_frames"] + episode["policy_frames"],
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
        require(episode["status"] == "env_truncated" and len(ticks) == 800 and ticks[-1]["truncated"], "Unsuccessful case lacks full official budget")
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
            'profile': frozen['profile'], 'task_id': spec['task_id'], 'source_task_id': spec['source_task_id'],
            'checkpoint_step': frozen['checkpoint_step'], 'original_official_initialization': True,
            'initialization_profile': INITIALIZATION_PROFILE,
            'initialization_warnings': episode['pick_initialization_safety']['engineering_reference_warnings'],
            'initialization_warning_count': len(episode['pick_initialization_safety']['engineering_reference_warnings']),
            'source_environment_config': case['source_environment_config'],
            'reset_derivation': case['reset_derivation'], 'adaptation': case['adaptation'],
            'pick_initialization_safety': episode['pick_initialization_safety']}


def execute(args, case, frozen, state):
    directory = args.output / f"level{case['level']}" / f"case_{case['case_index']:02}"
    directory.mkdir(parents=True, exist_ok=False)
    row = {"level": case["level"], "case_index": case["case_index"], "status": "starting",
           "directory": str(directory), "started_utc": now(), "sampling_seed": case["sampling_seed"],
           'profile': frozen['profile'], 'task_id': frozen['task_id'], 'checkpoint_step': args.checkpoint_step,
           'initialization_profile': INITIALIZATION_PROFILE}
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


def finish(state):
    """Keep all30 native outcomes; initialization warnings do not filter cases."""
    code=common.finish(state)
    state['initialization_profile']=INITIALIZATION_PROFILE
    warnings=[{'level':row['level'],'case_index':row['case_index'],
        'warnings':list(row['initialization_warnings'])} for row in state['cases'] if row.get('initialization_warnings')]
    state['initialization_warnings']=warnings
    state['initialization_warning_case_count']=len(warnings)
    state['initialization_warning_count']=sum(len(row['warnings']) for row in warnings)
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment", "data-root", "training-manifest", "output", "runtime-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, choices=SAVE_STEPS, required=True)
    parser.add_argument("--profile", choices=tuple(OFFICIAL_METADATA_SHA256), required=True)
    parser.add_argument("--port", type=int, default=21115)
    parser.add_argument("--sampling-seed", type=int, default=20261111)
    parser.add_argument("--rtc-mode", choices=("test_time",), default="test_time")
    parser.add_argument("--execute-steps", type=int, help="Exact15 actions with test_time RTC, consuming25 simulation ticks per window")
    args = parser.parse_args(argv)
    require(len(args.expected_model_sha256) == 64 and all(char in "0123456789abcdef" for char in args.expected_model_sha256), "Expected SHA must be lowercase SHA256")
    require(args.checkpoint_step > 0 and 1024 <= args.port <= 65535 and 0 <= args.sampling_seed <= 2**32 - 30, "Invalid step/port/seed")
    policy = evaluation_policy(args)
    for name in ("deployment", "data_root", "training_manifest", "runtime_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    require(args.output.is_absolute() and not args.output.exists() and not args.output.is_symlink(), "Campaign must be a fresh absolute directory")
    args.output = args.output.resolve()
    require(all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in
        (REPO, args.deployment, args.data_root, args.runtime_root)), 'Output overlaps source/deployment/resources')
    frozen = json.loads(json.dumps(freeze_inputs(args), default=_json_value, allow_nan=False))
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "FROZEN_CAMPAIGN.json", frozen)
    state = {"contract": frozen["contract"], "status": "running", "created_utc": now(), "planned_cases": 30,
             "checkpoint_step": args.checkpoint_step, "model_sha256": args.expected_model_sha256,
             "task_id": frozen["task_id"], "source_task_id": frozen["source_task_id"], "profile": args.profile, "cases": [], "formal_evaluation_eligible": False, "boundary": frozen["boundary"],
             "technical_errors_are_task_failures": False, "automatic_retries": 0, "evaluation_policy": policy,
             'original_official_initialization': True, 'initialization_profile': INITIALIZATION_PROFILE}
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
