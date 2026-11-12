"""Saved-scene, native-reward evaluation adapters for the two moving Pick tasks.

Collection joystick/mode-switch provenance is historical. During evaluation
all locomotion/manipulation actions come from the learned body64 + fingers14
closed loop; no PICO input, navigation planner, or scripted phase switch.
"""
from copy import deepcopy
import hashlib
import importlib
import json

import numpy as np

PROFILES = {
    'xmove_pick': {'task': 'G1SonicXMovePickPico-v0', 'task_id': 'cvpr/G1SonicXMovePickPico-v0',
        'source_task': 'G1WholebodyXMovePickTeleop-v0', 'source_task_id': 'simple/G1WholebodyXMovePickTeleop-v0',
        'uid': 'g1_wholebody_xmove_pick_teleop', 'adapter_module': 'pico_xmove_pick_task',
        'factory': 'make_xmove_pick_task', 'binding': 'xmove_pick_environment_binding_receipt'},
    'xmove_bend_pick': {'task': 'G1SonicXMoveBendPickPico-v0', 'task_id': 'cvpr/G1SonicXMoveBendPickPico-v0',
        'source_task': 'G1WholebodyXMoveBendPickTeleop-v0', 'source_task_id': 'simple/G1WholebodyXMoveBendPickTeleop-v0',
        'uid': 'g1_wholebody_xmove_bend_pick_teleop', 'adapter_module': 'pico_xmove_bend_pick_task',
        'factory': 'make_xmove_bend_pick_task', 'binding': 'xmove_bend_pick_environment_binding_receipt'},
}
TASKS = tuple(spec['task'] for spec in PROFILES.values())
INITIALIZATION_CONTRACT = 'cvpr-pick-official-startup-passive-verification-v1'
INITIALIZATION_PROFILE = 'native_saved_scene_observed_v1'
CONTROL_POLICY = 'learned_body64_fingers14_closed_loop_no_joystick_or_scripted_navigation'
NATIVE_BUDGET_PROFILE = 'native_16s'
EXTENDED_BUDGET_PROFILE = 'xmove_bend_failed7_24s'
EXTENDED_BUDGET_CONTRACT = 'cvpr-xmove-bend-pick-failed7-24s-diagnostic-v1'
EXTENDED_BUDGET_ROLE = 'fixed_failed7_extended_time_budget_diagnostic'
EXTENDED_BUDGET_CASES = ((0,5),(1,6),(1,7),(1,8),(2,0),(2,4),(2,9))


def require(condition, message):
    if not condition: raise ValueError(message)


def _json(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False, default=_json).encode()).hexdigest()


def profile_spec(profile):
    require(profile in PROFILES, 'Unknown moving Pick evaluation profile')
    return deepcopy(PROFILES[profile])


def _adapter(profile):
    return importlib.import_module(profile_spec(profile)['adapter_module'])


def pick_official_protocol(source, profile):
    spec = profile_spec(profile); metadata = source['metadata']
    require(source.get('uid') == spec['uid'] and source.get('robot_cfg', {}).get('uid') == 'g1_sonic'
        and metadata.get('physics_dt') == .005 and metadata.get('render_hz') == 50,
        'Moving Pick evaluation requires its native SONIC 200Hz/50Hz saved scene')
    budget = metadata.get('max_episode_steps')
    if 'max_episode_steps' in metadata:
        require(type(budget) is int and budget == 800, 'Native moving Pick budget must be exact800 control ticks')
        budget_source = 'saved_metadata'
    else:
        # Some original XMoveBend L0 rows omit this key. Bind the real unchanged
        # native task default rather than inserting a fabricated metadata value.
        official = _adapter(profile)._load_official()
        require(type(official.metadata.get('max_episode_steps')) is int and
            official.metadata['max_episode_steps'] == 800, 'Native task default budget differs')
        budget_source = 'unchanged_native_task_default_metadata'
    adapter = _adapter(profile)
    return {'profile': profile, 'task_id': spec['task_id'], 'source_task_id': spec['source_task_id'],
        'max_episode_steps': 800, 'observation_hz': 50, 'maximum_simulated_seconds': 16.,
        'source_metadata_max_episode_steps': budget, 'budget_source': budget_source,
        'native_default_max_episode_steps': 800, 'budget_default_source_sha256': adapter.OFFICIAL_TASK_SOURCE_SHA256,
        'source_metadata_control_hz': metadata.get('control_hz'),
        'control_tick_semantics': 'one real observation/control step is .02s, four .005s physical substeps',
        'reset_dr_level': None, 'source_physics_dt': .005, 'physics_dt': .005,
        'effective_success_criteria': .9, 'source_metadata_success_criteria': metadata.get('success_criteria'),
        'source_metadata_split': metadata.get('split'), 'world_translation_m': [0., 0., 0.],
        'success_semantics': 'unchanged native check_success: clip(target lift/.1,0,1)>=.9; no additional navigation/bend/contact/hold gate',
        'evaluation_control_policy': CONTROL_POLICY,
        'collection_control_identity_scope': 'task adapter records historical PICO collection control; it is not the evaluation action source',
        'initialization': 'saved official scene only; no teacher actions/states or PICO recorded physical-state injection',
        'initialization_profile': INITIALIZATION_PROFILE,
        'robot_physics_equivalent': False, 'formal_evaluation_eligible': False}



def pick_evaluation_protocol(source, profile, budget_profile=NATIVE_BUDGET_PROFILE):
    """Native bytes remain intact; a separately named diagnostic changes only time."""
    require(budget_profile in (NATIVE_BUDGET_PROFILE, EXTENDED_BUDGET_PROFILE), 'Unknown Pick budget profile')
    native = pick_official_protocol(source, profile)
    if budget_profile == NATIVE_BUDGET_PROFILE:
        return native
    require(profile == 'xmove_bend_pick', '24s diagnostic budget is only for XMoveBendPick')
    return dict(native, max_episode_steps=1200, maximum_simulated_seconds=24.,
        budget_profile=EXTENDED_BUDGET_PROFILE, diagnostic_contract=EXTENDED_BUDGET_CONTRACT,
        diagnostic_role=EXTENDED_BUDGET_ROLE, native_max_episode_steps=800,
        native_maximum_simulated_seconds=16., additional_control_ticks=400,
        warmup_control_ticks=50, original_official_time_budget=False, official_budget_modified=True,
        source_metadata_modified=False, scene_or_reward_or_success_modified=False,
        budget_override_scope='Explicit fixed failed7 diagnostic only;1200 real control ticks include original50 warmup; no full30 result claim')


def extended_budget_runtime_receipt(env, protocol, supervisor_seconds):
    """Read the real outer TimeLimit, not merely a requested --seconds value."""
    require(protocol.get('budget_profile') == EXTENDED_BUDGET_PROFILE and
        protocol.get('profile') == 'xmove_bend_pick' and protocol.get('max_episode_steps') == 1200 and
        protocol.get('maximum_simulated_seconds') == supervisor_seconds == 24., 'Wrong explicit24s protocol/cutoff')
    maximum, elapsed = env._max_episode_steps, env._elapsed_steps
    require(type(maximum) is int and maximum == 1200 and type(elapsed) is int and 0 <= elapsed <= maximum,
        'Actual environment TimeLimit is not the diagnostic1200 control ticks')
    return {'budget_profile': EXTENDED_BUDGET_PROFILE, 'native_max_episode_steps': 800,
        'native_maximum_simulated_seconds': 16., 'environment_time_limit_steps': maximum,
        'supervisor_cutoff_seconds': supervisor_seconds, 'actual_environment_elapsed_control_ticks': elapsed,
        'warmup_control_ticks': 50, 'original_official_time_budget': False, 'official_budget_modified': True,
        'scene_or_reward_or_success_modified': False}


def derive_pick_eval_reset(source, profile):
    """Reversible serialization compatibility only, without moving any actor."""
    pick_official_protocol(source, profile)
    derived = deepcopy(source); dr = source['dr_state_dict']; actors = source['layout']['actors']
    selected = {str(dr['target']['uid']), *map(str, dr['distractors'])}
    actual = [str(actor['asset']['uid']) for actor in actors.values() if 'asset' in actor]
    require(len(dr['distractors']) == 3 and len(actual) == len(set(actual)) == 4 and set(actual) == selected,
        'Moving Pick reset must preserve the exact actual target and three distractors')
    require(str(actors['target']['asset']['uid']) == str(dr['target']['uid']), 'Moving Pick target role differs')
    keep = selected | {'g1_sonic'}
    require(keep <= set(dr['spatial']) and 'g1_wholebody' not in dr['spatial'], 'Missing actual actor/native SONIC spatial pose')
    removed = sorted(set(dr['spatial'])-keep)
    require(not any(uid.startswith('g1_') for uid in removed), 'Unexpected robot spatial key may not be removed')
    derived['dr_state_dict']['spatial'] = {uid: deepcopy(pose) for uid, pose in dr['spatial'].items() if uid in keep}
    removed_sensors = {}
    for name, config in derived['sensor_cfgs'].items():
        for field in ('intrinsics', 'depth_intrinsics', 'depth_to_color_extrinsics'):
            if field in config:
                require(config[field] is None, 'May not discard nonempty camera calibration: '+name+'.'+field)
                removed_sensors.setdefault(name, {})[field] = config.pop(field)
    removed_layout_cameras = {}
    for name, camera in derived['layout']['cameras'].items():
        for field in ('intrinsics', 'depth_intrinsics', 'depth_to_color_extrinsics'):
            config = camera.get('cam_cfg', {})
            if field in config:
                require(config[field] is None, 'May not discard nonempty redundant camera calibration: '+name+'.'+field)
                removed_layout_cameras.setdefault(name, {})[field] = config.pop(field)
    compatibility = {('spatial','placement_attempts'): (int, 100),
        ('scene','randomize_scene_pose'): (bool, True), ('material','table_material'): (type(None), None)}
    removed_dr = {}
    for (name, field), (kind, value) in compatibility.items():
        config = derived['dr_cfgs'].get(name, {})
        if field in config:
            require(type(config[field]) is kind and config[field] == value,
                'Unsupported non-equivalent saved-reset DR option: '+name+'.'+field)
            removed_dr.setdefault(name, {})[field] = config.pop(field)
    receipt = {'contract': 'cvpr-moving-pick-eval-saved-reset-compatibility-v1', 'profile': profile,
        'source_config_sha256': digest(source), 'derived_config_sha256': digest(derived),
        'removed_unused_spatial_uids': removed,
        'removed_unused_spatial_entries': {uid: deepcopy(dr['spatial'][uid]) for uid in removed},
        'removed_optional_sensor_fields': removed_sensors,
        'removed_optional_layout_camera_fields': removed_layout_cameras, 'removed_compatible_dr_fields': removed_dr,
        'preserved_real_spatial_uids': sorted(keep), 'actual_actor_poses_modified': False,
        'camera_geometry_or_nonempty_calibration_modified': False, 'saved_materials_or_lighting_modified': False,
        'metadata_modified': False, 'reward_or_success_modified': False,
        'world_translation_m': [0., 0., 0.], 'compatibility_scope': 'complete saved reset only with dr_level=None',
        'DR_compatibility_evidence': {'placement_attempts': 'exact100 equals current SpatialDR hardcoded loop',
            'randomize_scene_pose': 'saved scene state restores full pose and bypasses scene sampling',
            'table_material': 'None gives no override; complete saved material state retained'}}
    return derived, receipt


def adapt_pick_config(derived, profile):
    return _adapter(profile).adapt_official_config(derived)


def prepare_pick_reset(source, profile, budget_profile=NATIVE_BUDGET_PROFILE):
    protocol = pick_evaluation_protocol(source, profile, budget_profile)
    derived, derivation = derive_pick_eval_reset(source, profile)
    adapted, adaptation = adapt_pick_config(derived, profile)
    return adapted, derivation, adaptation, protocol


def environment_binding_receipt(expected, actual, profile):
    return getattr(_adapter(profile), profile_spec(profile)['binding'])(expected, actual)


def install_pick_initialization_monitor(env, profile):
    """Use the proven generic passive observer, with a task-specific contract."""
    from bend_initialization_monitor import BendInitializationMonitor
    raw = env.unwrapped
    original_update, original_step = raw.mujoco.update_layout, raw.mujoco.step
    raw._pick_initialization_monitor = None
    def update(*args, **kwargs):
        raw._pick_initialization_monitor = None
        result = original_update(*args, **kwargs)
        raw._pick_initialization_monitor = BendInitializationMonitor(raw.mujoco.mjModel,
            raw.mujoco.mjData, int(raw.mujoco.mj_objects['target'].id))
        return result
    def step(*args, **kwargs):
        monitor = raw._pick_initialization_monitor
        require(monitor is not None and monitor.model is raw.mujoco.mjModel,
            'Moving Pick physics preceded its passive initialization monitor')
        result = original_step(*args, **kwargs)
        monitor.after_step(raw.mujoco.mjModel, raw.mujoco.mjData)
        return result
    raw.mujoco.update_layout, raw.mujoco.step = update, step
    raw._pick_initialization_profile = profile


def initialization_receipt(raw):
    """Separate untouched native-scene observation from Bend engineering limits.

    The shared observer remains unchanged. Its original strict diagnosis is
    retained; displacement/rotation are diagnostic warnings for these native
    saved scenes, never a replacement reward, reset or scene-validity rule.
    """
    monitor = raw._pick_initialization_monitor
    receipt = monitor.receipt()
    receipt['contract'] = INITIALIZATION_CONTRACT
    receipt['profile'] = raw._pick_initialization_profile
    receipt['initialization_profile'] = INITIALIZATION_PROFILE
    receipt['initial_target_xyz_m'] = monitor.initial_xyz.tolist()
    receipt['initial_target_quaternion_wxyz'] = monitor.initial_quaternion.tolist()
    receipt['engineering_reference_status'] = receipt['status']
    receipt['engineering_reference_warnings'] = []
    for value, limit, warning in (
        ('maximum_target_displacement_m', 'engineering_displacement_limit_m', 'target_displacement_exceeds_2mm'),
        ('maximum_target_rotation_degrees', 'engineering_rotation_limit_degrees', 'target_rotation_exceeds_2deg')):
        if receipt[value] > receipt[limit]:
            receipt['engineering_reference_warnings'].append(warning)
    complete = (receipt['observed_physics_substeps'] == receipt['required_physics_substeps'] == 201
        and len(receipt['samples']) == 201 and receipt['finite_states']
        and receipt['robot_target_contact_count'] == 0 and not receipt['contacts']
        and receipt['private_FK_only'] and not receipt['live_physical_state_writes']
        and receipt['physics_steps_added'] == 0 and not receipt['controller_commands_modified']
        and not receipt['policy_phase_interventions'])
    receipt['status'] = 'observation_complete' if complete else 'observation_failed_or_incomplete'
    receipt['threshold_scope'] = ('2mm/2deg are retained Bend engineering references only; '
        'native saved scenes and real50tick warmup unchanged; no scene-validity/native-success gate')
    samples = receipt['samples']
    if samples:
        end = samples[-1]['sim_time_s']
        tail = [sample for sample in samples if sample['sim_time_s'] >= end - .2 - 1e-12]
        xyz = np.asarray([sample['target_xyz_m'] for sample in tail], dtype=float)
        quat = np.asarray([sample['target_quaternion_wxyz'] for sample in tail], dtype=float)
        angles = np.degrees(2 * np.arccos(np.clip(np.abs(quat @ quat.T), 0., 1.)))
        receipt['tail_settling_metrics'] = {
            'window_duration_s': .2, 'sample_count': len(tail),
            'start_sim_time_s': tail[0]['sim_time_s'], 'end_sim_time_s': end,
            'maximum_pairwise_displacement_m': float(np.linalg.norm(xyz[:, None] - xyz[None, :], axis=-1).max()),
            'endpoint_displacement_m': float(np.linalg.norm(xyz[-1] - xyz[0])),
            'maximum_pairwise_rotation_degrees': float(angles.max()),
            'endpoint_rotation_degrees': float(angles[0, -1])}
    else:
        receipt['tail_settling_metrics'] = None
    return receipt


def make_pick_environment(sonic_config, source, profile, *, sim_mode, sync_observations, isaac_observation_only,
        budget_profile=NATIVE_BUDGET_PROFILE):
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv
    import gymnasium as gym
    adapted, derivation, adaptation, protocol = prepare_pick_reset(source, profile, budget_profile)
    task = getattr(_adapter(profile), profile_spec(profile)['factory'])(sonic_config, adapted_config=adapted)
    env = SonicLocoManipEnv(task=task, sonic_config=sonic_config, sim_mode=sim_mode, headless=True,
        dr_level=0, sync_mujoco_observations=sync_observations, isaac_observation_only=isaac_observation_only)
    try:
        install_pick_initialization_monitor(env, profile)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=protocol['max_episode_steps'])
    except BaseException:
        env.close()
        raise
    return env, {'adaptation': adaptation, 'reset_derivation': derivation, 'protocol': protocol,
        'world_translation_m': [0., 0., 0.], 'evaluation_control_policy': CONTROL_POLICY,
        'initialization_profile': INITIALIZATION_PROFILE}
