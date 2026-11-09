"""Explicit pick-task RTX batches; historical Grasp producer stays unchanged.

One private environment per task/process. Each episode independently resets and
binds its archived scene, restores complete physical states without stepping,
and writes the original per-frame state/event/RGB hash bridge. No output deletion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import re
import shutil
import traceback
from types import SimpleNamespace

import numpy as np
import render_deferred_pico as original
from pico_simple_teleop import (actual_renderer_poses, bind_saved_scene_manager,
    configure_unrelated_viewport, disable_instance_render_rate_limits, root_pose)
from replay_handover_teacher import (TrainingCapture, bound_calibration_assets,
    calibration_source_identities, file_sha256, write_json)

CONTRACT = 'cvpr-pico-pick-deferred-rtx-batch-v1'
COMPLETION_CONTRACT = 'cvpr-pico-pick-deferred-rtx-completion-v1'
FROZEN_CONTRACT = 'cvpr-pico-pick-rtx-frozen-inputs-v1'
ORIGINAL_RENDERER_SHA256 = 'c4c4735aaccb111e3ef08d48c69194b02af2f29994d8a6fc4e9e766575664215'
PROFILES = {
    'xmove_pick': 'cvpr/G1SonicXMovePickPico-v0',
    'bend_pick': 'cvpr/G1SonicBendPickPico-v0',
    'xmove_bend_pick': 'cvpr/G1SonicXMoveBendPickPico-v0',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def profile_for(task_id):
    matches = [profile for profile, identity in PROFILES.items() if identity == task_id]
    require(len(matches) == 1, 'Unsupported pick task identity: ' + str(task_id))
    return matches[0]


def task_module(task_id):
    return importlib.import_module('pico_' + profile_for(task_id) + '_task')


def producer_sources(task_id):
    require(file_sha256(original.__file__) == ORIGINAL_RENDERER_SHA256, 'Historical zero-step helper bytes changed')
    profile = profile_for(task_id)
    paths = [Path(original.__file__), Path(task_module(task_id).__file__)]
    paths.extend(Path(__file__).with_name(name) for name in
        ('pico_' + profile + '_quality.py', 'pico_deferred_capture.py',
         'pico_simple_teleop.py', 'replay_handover_teacher.py'))
    return {str(path.resolve()): file_sha256(path) for path in paths}


def recorded_task(metadata):
    from pico_deferred_capture import evidence_object_roles
    task_id = metadata.get('task_id')
    profile = profile_for(task_id)
    quality = importlib.import_module('pico_' + profile + '_quality')
    check = getattr(quality, 'validate_' + profile + '_task_identity')(metadata)
    require(check.get('passed') is True, 'Recorded pick identity invalid: ' + '; '.join(check.get('errors', [])))
    current = task_module(task_id).adapter_identity()
    require(metadata.get(profile + '_adapter_identity') == current, 'Captured adapter bytes differ from rendering adapter')
    require(evidence_object_roles(metadata) == ('target',), 'Pick rendering requires explicit target-only evidence')
    return task_id


def recorded_environment(path, data_root, task_id):
    """Use the archived already-adapted payload unchanged; never translate twice."""
    environment = load_json(path)
    task_module(task_id).validate_adapted_config(environment)
    data_root = Path(data_root).resolve(strict=True)
    resources = {}
    def visit(value, field='environment'):
        if isinstance(value, dict):
            for key, item in value.items():
                label = field + '.' + key
                if key in ('usd_path', 'mdl_path', 'path', 'data_dir') and isinstance(item, str):
                    require('://' not in item, 'Remote resource in archived environment')
                    candidate = Path(item) if Path(item).is_absolute() else data_root / item.removeprefix('data/')
                    resources[label] = str(candidate.resolve(strict=True))
                else:
                    visit(item, label)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, field + '[' + str(index) + ']')
    visit(environment)
    scene = environment['dr_state_dict']['scene']
    resources['authoritative_scene_usd'] = str((data_root / scene['data_dir'] / (scene['name'] + '.usd')).resolve(strict=True))
    return environment, {'local_resource_paths': resources, 'already_adapted_payload_unchanged': True}


def recorded_adaptation(metadata, environment):
    task_id = metadata['task_id']
    receipt = task_module(task_id).validate_adapted_config(environment)
    profile = profile_for(task_id)
    key = profile + ('_frame_adaptation' if profile == 'bend_pick' else '_adaptation')
    require(metadata.get(key) == receipt, 'Recorded adaptation differs from archived environment')
    return receipt


def make_recorded_renderer(task_id, sonic_config, environment):
    """Every task has its own private constructor; no registry impersonation."""
    from simple.envs.sonic_loco_manip import SonicLocoManipEnv
    module = task_module(task_id)
    task = getattr(module, 'make_' + profile_for(task_id) + '_task')(sonic_config, adapted_config=environment)
    env = SonicLocoManipEnv(task=task, sonic_config=sonic_config, sim_mode='mujoco_isaac',
        headless=True, dr_level=0, sync_mujoco_observations=True, isaac_observation_only=True)
    try:
        binding = (module.install_frame_adapter(env) if profile_for(task_id) == 'bend_pick' else
            {**module.adapter_identity(), 'installed': True, 'world_translation_m': [0., 0., 0.],
             'scope': 'native SONIC world frame; no coordinate hook required'})
    except BaseException:
        env.close()
        raise
    return env, binding


def recorded_environment_binding(task_id, expected, actual):
    return getattr(task_module(task_id), profile_for(task_id) + '_environment_binding_receipt')(expected, actual)


def recorded_initialization_index(metadata, profile):
    identity = metadata.get('official_initialization')
    if isinstance(identity, dict):
        index = identity.get('official_episode_index')
        require(type(index) is int and index >= 0, 'Invalid recorded official initialization index')
        return index
    # The selected XMove supplement was captured through its explicit fixed
    # reset-config entry. Authenticate that original config; never add an
    # invented initialization field to its immutable acquisition metadata.
    require(profile == 'xmove_pick', 'Missing recorded official initialization identity')
    source = metadata.get('environment_config_source', {})
    path = Path(source['path']).resolve(strict=True)
    match = re.fullmatch(r'episode_(\d{3})\.json', path.name)
    require(match is not None and path.is_file() and file_sha256(path) == source.get('sha256'),
            'Supplemental XMove reset config path/hash differs')
    return int(match.group(1))


RecordedStateView = original.RecordedStateView
require_equal = original.require_equal
restore_recorded_state = original.restore_recorded_state
assert_recorded_state_unchanged = original.assert_recorded_state_unchanged
recorded_objects = original.recorded_objects
write_frame_bindings = original.write_frame_bindings
canonical_event_sha256 = original.canonical_event_sha256


class CompatiblePickResources:
    """Keep cached robot/camera/light/primitive/scene/asset identities fixed."""
    def __init__(self):
        self.fixed, self.scenes, self.assets = None, {}, {}

    def add(self, environment):
        fixed = {key: environment[key] for key in ('uid', 'metadata', 'robot_cfg', 'sensor_cfgs')}
        fixed['light_topology'] = [{key: light[key] for key in ('uid', 'type')} for light in environment['layout']['lights']]
        fixed['primitives'] = {role: {key: actor[key] for key in ('uid', 'size') if key in actor}
            for role, actor in environment['layout']['actors'].items() if 'asset' not in actor}
        if self.fixed is None:
            self.fixed = canonical(fixed)
        require(self.fixed == canonical(fixed), 'Batch robot/camera/light/primitive geometry differs')
        scene = environment['dr_state_dict']['scene']
        identity = canonical({key: scene[key] for key in ('uid', 'conf', 'name', 'data_dir')})
        require(self.scenes.setdefault(scene['uid'], identity) == identity, 'Scene UID aliases different geometry/config')
        for actor in environment['layout']['actors'].values():
            if 'asset' in actor:
                asset = actor['asset']; label = asset['label']; identity = canonical(asset)
                require(self.assets.setdefault(label, identity) == identity, 'Object label aliases different asset: ' + label)


class EnvironmentLease:
    def __init__(self, owner):
        self.owner = owner

    @property
    def unwrapped(self):
        return self.owner.environment.unwrapped

    def reset(self, *args, **kwargs):
        return self.owner.environment.reset(*args, **kwargs)

    def close(self):
        pass  # Owner closes after report flush, including hard-exit close.


class RendererPool:
    def __init__(self, factory):
        self.factory, self.environment, self.frame_binding = factory, None, None
        self.task_id, self.sonic, self.leases = None, None, 0
        self.compatibility = CompatiblePickResources()

    def acquire(self, task_id, sonic_config, environment):
        profile_for(task_id)
        self.compatibility.add(environment)
        sonic = canonical(sonic_config)
        if self.environment is None:
            self.environment, self.frame_binding = self.factory(task_id, sonic_config, environment)
            self.task_id, self.sonic = task_id, sonic
        else:
            require((self.task_id, self.sonic) == (task_id, sonic), 'One process must render one task and SONIC configuration')
        self.leases += 1
        return EnvironmentLease(self), self.frame_binding

    def close(self):
        if self.environment is not None:
            env, self.environment = self.environment, None
            env.close()


def render_episode(args, pool):
    from pico_deferred_capture import load_deferred, evidence_object_roles
    episode=args.episode.resolve(strict=True)
    destination=episode/'capture'
    receipt_path=episode/'rtx_completion.json'
    if any(path.exists() or path.is_symlink() for path in (destination, receipt_path,
            episode/'rtx_frame_bindings.jsonl', episode/'deferred_render_environment.json')):
        raise ValueError('RTX output already exists; preserve it and use a fresh episode copy for retry')
    source=load_deferred(episode/'deferred_capture',verify=True)
    manifest,events,states=source['manifest'],source['events'],source['states']
    metadata=manifest['metadata']
    task_id = recorded_task(metadata)
    object_roles = evidence_object_roles(metadata)
    if manifest['status'] != 'raw_physics_capture_completed_pending_rtx':
        raise ValueError('Only explicitly completed physical captures may be rendered for audit')
    original_outcome=json.loads((episode/'episode.json').read_text())
    if not original_outcome['state']['pending_audit']:
        raise ValueError('Source episode was not saved for audit')
    data_root,assets=bound_calibration_assets(args.data_root,calibration_source_identities())
    if metadata['calibration_source_asset_sha256'] != assets:
        raise ValueError('Camera source asset identities differ from the physical capture')
    os.environ['SIMPLE_DATA_DIR']=str(data_root)
    expected_env_path=episode/'deferred_render_environment.json'
    write_json(expected_env_path,metadata['environment_config'])
    environment,_=recorded_environment(expected_env_path,data_root,task_id)
    frame_receipt = recorded_adaptation(metadata, environment)
    result={'contract':COMPLETION_CONTRACT,'status':'initializing',
        'accepted_new_dataset':False,'eligible_for_packing':False,
        'source_kind':metadata['source_kind'],'observation_domain_feedback':'MuJoCo EGL preview',
        'training_rgb_domain':'Isaac RTX deferred','rgb_was_live_operator_feedback':False,
        'archived_physics_steps':0,'sonic_inferences':0,'completed_frames':0,
        'source_manifest_sha256':file_sha256(episode/'deferred_capture/capture.json'),
        'source_events_sha256':file_sha256(episode/'deferred_capture/events.jsonl'),
        'source_model_sha256':file_sha256(episode/'deferred_capture/model.mjb'),
        'source_states_sha256':file_sha256(episode/'deferred_capture/states.npy'),
        'task_id':task_id,
        'render_code_path':str(Path(__file__).resolve()), 'render_code_sha256':file_sha256(__file__),
        'original_renderer_code_path':str(Path(original.__file__).resolve()),
        'original_renderer_code_sha256':file_sha256(original.__file__),
        'producer_source_sha256':producer_sources(task_id)}
    if frame_receipt is not None:
        result['task_adaptation'] = frame_receipt
        result['evidence_object_roles'] = list(object_roles)
    env=capture=None
    try:
        import mujoco
        import simple.envs
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from simple.core.action import ActionCmd
        model=source['model']; data=mujoco.MjData(model)
        view=RecordedStateView(model,data,source['joint_names'],metadata['mujoco_object_body_names'],metadata['mujoco_obj_names'])
        sonic=SimLoopConfig().load_wbc_yaml()
        sonic.update(ENV_NAME='simple',SIMULATE_DT=.005,ENABLE_ELASTIC_BAND=False)
        env,frame_binding=pool.acquire(task_id,sonic,environment)
        raw=env.unwrapped
        result['task_adapter_identity'] = task_module(task_id).adapter_identity()
        result['frame_adapter_binding'] = frame_binding
        result['actual_task_source_sha256'] = result['task_adapter_identity']['official_task_source_sha256']
        if metadata.get('actual_task_source_sha256') is not None and result['actual_task_source_sha256'] != metadata['actual_task_source_sha256']:
            raise ValueError('Loaded RTX task module differs from the original capture')
        bind_saved_scene_manager(raw.task,environment)
        raw._render_frame=lambda:raw.isaac.render()
        env.reset(seed=metadata['physics_seed'],options={'state_dict':copy.deepcopy(environment),'dr_level':None})
        result['environment_binding']=recorded_environment_binding(task_id,environment,raw.task.state_dict())
        if not result['environment_binding']['matched']:
            raise ValueError('Deferred renderer did not reconstruct the same room/assets/layout')
        import carb
        from omni.kit.viewport.utility import get_active_viewport
        result['instance_render_rate_limits']=disable_instance_render_rate_limits(carb.settings.get_settings())
        result['unrelated_viewport']=configure_unrelated_viewport(get_active_viewport(),raw.isaac.cameras,disable=True)
        capture_metadata={**metadata,'render_engine':'Isaac RTX','rgb_capture_mode':'deferred_from_exact_recorded_integration_state',
            'observation_domain_feedback':'MuJoCo EGL preview','training_rgb_domain':'Isaac RTX deferred',
            'rgb_was_live_operator_feedback':False,'deferred_source_manifest_sha256':result['source_manifest_sha256']}
        capture=TrainingCapture(destination,metadata=capture_metadata,joint_names=source['joint_names'],
            warmup_ticks=manifest['warmup_ticks'],start_time=manifest['start_time'])
        binding=object()  # Provenance carrier only; never apply/step this command.
        state_spec=mujoco.mjtState(source['state_spec'])
        for index,event in enumerate(events):
            # Only derived FK/cameras are refreshed. There is no mj_step,
            # env.step, controller history, torque control or physical replay.
            state_before=restore_recorded_state(model,data,states[index],state_spec)
            require_equal([data.time],[event['sim_time_s']],'time',1e-9)
            q=np.array([data.joint(name).qpos[0] for name in source['joint_names']])
            dq=np.array([data.joint(name).qvel[0] for name in source['joint_names']])
            require_equal(q,event['physical_joint_q43_float64'],'q43')
            require_equal(dq,event['physical_joint_qvel43'],'qvel43')
            require_equal(root_pose(model,data),event['root_pose7'],'root pose')
            objects=recorded_objects(view,event,object_roles)
            raw.isaac.refresh_from_mujoco(view,update_kinematics=True)
            rgb=raw.isaac.render()['head_stereo_left']
            assert_recorded_state_unchanged(model,data,state_before,state_spec)
            action=event['action']
            command=None if action is None else ActionCmd('sonic_v1_1',controller=binding,
                body_token=np.array(action['body_token64'],dtype=np.float64),
                hand_targets=dict(zip(action['hand_joint_names'],action['hand14'])))
            capture.append(tick50=event['physical_tick50'],sim_time_s=float(data.time),rgb=rgb,
                observed_joint_names=source['joint_names'],observed_qpos=q.astype(np.float32),
                physical_qpos=q,physical_qvel=dq,root_pose7=root_pose(model,data),root_qvel6=event['root_qvel6'],
                objects=objects,command=command,
                renderer_pose_diagnostics=raw.isaac.last_observation_pose_diagnostics,
                renderer_authoritative_poses=actual_renderer_poses(model,data),observation_sim_time_s=float(data.time))
            result['completed_frames']=index+1
        capture_manifest=capture.close(completed=True)
        result.update(write_frame_bindings(episode/'rtx_frame_bindings.jsonl',events,destination/'events.jsonl'))
        # This receipt replaces neither the original physics outcome nor its
        # data-source identity. The independent audit must bind both phases.
        result.update(status='deferred_rtx_completed_pending_independent_audit',
            capture=capture_manifest,capture_manifest_sha256=file_sha256(destination/'capture.json'),
            state_event_numeric_checks_passed=True)
    except BaseException as error:
        result.update(status='rejected',error=repr(error),traceback=traceback.format_exc())
        if capture:
            capture.close(completed=False)
        raise
    finally:
        write_json(receipt_path,result)
        if env:
            env.close()
    return result


def frozen_inputs(path):
    manifest = load_json(path)
    require(manifest.get('contract') == FROZEN_CONTRACT, 'Unknown frozen pick input contract')
    rows = manifest.get('episodes', [])
    require(type(manifest.get('count')) is int and 1 <= manifest['count'] <= 300
            and len(rows) == manifest['count'], 'Frozen processing count differs')
    uuids = {row['uuid'] for row in rows}
    require(len(uuids) == len(rows), 'Repeated frozen UUID')
    require(len({str(Path(row['directory']).resolve()) for row in rows}) == len(rows), 'Repeated frozen directory')
    counts = dict(Counter(row['profile'] for row in rows))
    expected = manifest.get('expected_task_counts')
    require(isinstance(expected, dict) and all(profile in PROFILES and type(value) is int and 0 < value <= 100
            for profile, value in expected.items()) and expected == counts, 'Explicit expected task counts differ')
    if len(rows) < 300:
        candidate_path = Path(manifest['candidate_manifest_path']).resolve(strict=True)
        require(file_sha256(candidate_path) == manifest['candidate_manifest_sha256'], 'Candidate manifest changed')
        candidate = load_json(candidate_path)
        candidates = candidate['episodes']
        require(candidate.get('contract') == FROZEN_CONTRACT and candidate.get('count') == 300
                and manifest.get('candidate_count') == 300 and len(candidates) == 300,
                'Processing subset must bind its complete candidate300 manifest')
        by_uuid = {row['uuid']: row for row in candidates}
        excluded = manifest.get('excluded_episodes', [])
        excluded_ids = {row['uuid'] for row in excluded}
        require(len(by_uuid) == 300 and len(excluded_ids) == len(excluded)
                and excluded_ids == set(by_uuid) - uuids and uuids <= set(by_uuid), 'Processing UUIDs do not equal candidate minus exclusions')
        for row in rows:
            require(canonical(row) == canonical(by_uuid[row['uuid']]), 'Processing row changed from candidate manifest')
        for row in excluded:
            require(row.get('excluded_from_processing') is True and bool(row.get('reason')),
                    'Excluded candidate requires an explicit preserved exclusion reason')
            require(all(row.get(key) == by_uuid[row['uuid']].get(key) for key in ('task_id', 'directory', 'number')),
                    'Excluded candidate identity differs')
    compatibility = {profile: CompatiblePickResources() for profile in counts}
    required_files = {'metadata.json', 'episode.json', 'deferred_capture/capture.json',
        'deferred_capture/events.jsonl', 'deferred_capture/model.mjb', 'deferred_capture/states.npy',
        'raw_quality_samples.jsonl', 'applied_force_samples.jsonl', 'commands.jsonl'}
    for row in rows:
        require(PROFILES.get(row['profile']) == row['task_id'], 'Frozen profile/task identity differs')
        episode = Path(row['directory']).resolve(strict=True)
        require(required_files <= set(row['source_files']), 'Frozen source inventory lacks mandatory original evidence')
        for name, frozen in row['source_files'].items():
            relative = Path(name)
            source = episode / relative
            require(not relative.is_absolute() and '..' not in relative.parts and not source.is_symlink(), 'Unsafe frozen source path')
            source = source.resolve(strict=True)
            require(source.is_relative_to(episode) and source.is_file(), 'Frozen source escapes episode')
            stat = source.stat()
            require(stat.st_size == frozen['size_bytes'] and stat.st_mtime_ns == frozen['mtime_ns'], 'Frozen source stat changed: ' + name)
            require(file_sha256(source) == frozen['sha256'], 'Frozen source SHA changed: ' + name)
        metadata = load_json(episode/'metadata.json')
        capture = load_json(episode/'deferred_capture/capture.json')
        require(capture['metadata'] == metadata and metadata['human_demo_uuid'] == row['uuid'], 'Frozen original metadata/UUID differs')
        require(recorded_task(metadata) == row['task_id'], 'Frozen task changed')
        require(recorded_initialization_index(metadata, row['profile']) == row['official_episode_index'], 'Frozen official initialization changed')
        compatibility[row['profile']].add(metadata['environment_config'])
    return manifest


def completed_output(episode):
    """Only exact completed bytes from this producer can be reused."""
    episode = Path(episode)
    paths = [episode/'capture', episode/'rtx_completion.json', episode/'rtx_frame_bindings.jsonl',
        episode/'deferred_render_environment.json']
    if not any(path.exists() or path.is_symlink() for path in paths):
        return None
    require(all(path.exists() and not path.is_symlink() for path in paths), 'Partial RTX output exists; preserve it and retry in a fresh copy')
    receipt = load_json(paths[1])
    require(receipt.get('contract') == COMPLETION_CONTRACT
            and receipt.get('status') == 'deferred_rtx_completed_pending_independent_audit', 'Rejected/unknown RTX output exists; never overwrite')
    require(receipt.get('archived_physics_steps') == receipt.get('sonic_inferences') == 0
            and receipt.get('full_integration_state_unchanged') is True
            and receipt.get('state_event_numeric_checks_passed') is True
            and receipt.get('environment_binding', {}).get('matched') is True, 'Incomplete zero-step renderer proof')
    require(receipt.get('render_code_path') == str(Path(__file__).resolve())
            and receipt.get('render_code_sha256') == file_sha256(__file__), 'Completed producer changed')
    require(receipt.get('original_renderer_code_path') == str(Path(original.__file__).resolve())
            and receipt.get('original_renderer_code_sha256') == ORIGINAL_RENDERER_SHA256,
            'Completed historical helper identity differs')
    require(receipt.get('producer_source_sha256') == producer_sources(receipt['task_id']), 'Completed producer dependency source changed')
    for kind, filename in [('manifest', 'capture.json'), ('events', 'events.jsonl'), ('model', 'model.mjb'), ('states', 'states.npy')]:
        require(receipt.get('source_' + kind + '_sha256') == file_sha256(episode/'deferred_capture'/filename), 'Completed original source changed: ' + kind)
    capture = load_json(episode/'capture/capture.json')
    require(receipt.get('capture_manifest_sha256') == file_sha256(episode/'capture/capture.json')
            and receipt.get('capture') == capture, 'Completed RTX manifest changed')
    require(capture.get('events_sha256') == file_sha256(episode/'capture/events.jsonl'), 'Completed RTX events changed')
    require(receipt.get('frame_bindings_path') == paths[2].name
            and receipt.get('frame_bindings_sha256') == file_sha256(paths[2]), 'Completed frame bridge changed')
    events = [json.loads(line) for line in (episode/'capture/events.jsonl').read_text().splitlines()]
    source = [json.loads(line) for line in (episode/'deferred_capture/events.jsonl').read_text().splitlines()]
    bridge = [json.loads(line) for line in paths[2].read_text().splitlines()]
    require(len(source) == len(events) == len(bridge) == receipt['completed_frames'] == receipt['frame_bindings_count'] > 0, 'Completed frame counts differ')
    for index, (raw, event, binding) in enumerate(zip(source, events, bridge)):
        require(raw['event_index'] == event['event_index'] == binding['event_index'] == index,
                'Completed frame order changed')
        require(binding['source_event_sha256'] == canonical_event_sha256(raw)
                and binding['rtx_event_sha256'] == canonical_event_sha256(event)
                and binding['integration_state_sha256'] == raw['integration_state_sha256'], 'Completed source/RTX event bridge changed')
        relative = Path(event['rgb']['path']); png = episode/'capture'/relative
        require(not relative.is_absolute() and '..' not in relative.parts and not png.is_symlink()
                and png.resolve(strict=True).is_relative_to((episode/'capture').resolve()), 'RTX PNG escapes capture')
        require(file_sha256(png) == event['rgb']['file_sha256'] == binding['rgb_file_sha256'], 'Completed PNG bytes changed')
    return receipt


def capacity_preflight(rows, estimated_bytes_per_frame=450_000, reserve_free_bytes=2*1024**3):
    require(type(estimated_bytes_per_frame) is int and estimated_bytes_per_frame > 0
            and type(reserve_free_bytes) is int and reserve_free_bytes >= 0, 'Invalid capacity estimate')
    groups = {}
    for row in rows:
        episode = Path(row['directory']).resolve(strict=True)
        if completed_output(episode) is not None:
            continue
        group = groups.setdefault(episode.stat().st_dev, {'path': str(episode), 'frames': 0})
        group['frames'] += load_json(episode/'deferred_capture/capture.json')['event_count']
    report = []
    for group in groups.values():
        free = shutil.disk_usage(group['path']).free
        estimate = group['frames'] * estimated_bytes_per_frame
        require(free >= estimate + reserve_free_bytes, 'Insufficient persistent output space: need estimate plus reserve ' + str(estimate + reserve_free_bytes))
        report.append({**group, 'free_bytes': free, 'estimated_png_bytes': estimate,
            'estimated_bytes_per_frame': estimated_bytes_per_frame, 'reserve_free_bytes': reserve_free_bytes,
            'scope': 'Conservative PNG estimate; pilot sizes must be checked before continuing'})
    return report


def run(args):
    manifest_path = args.manifest.resolve(strict=True)
    manifest = frozen_inputs(manifest_path)
    require(args.profile in PROFILES, 'Select one explicit task profile')
    selected = [row for row in manifest['episodes'] if row['profile'] == args.profile]
    require(bool(selected), 'Frozen manifest has no rows for selected task')
    offset = getattr(args, 'offset', 0)
    require(type(offset) is int and 0 <= offset < len(selected), 'Invalid task offset')
    require(args.limit is None or 1 <= args.limit <= len(selected) - offset, 'Invalid chunk limit')
    rows = selected[offset:None if args.limit is None else offset + args.limit]
    capacity = capacity_preflight(rows, getattr(args, 'estimated_bytes_per_frame', 450_000),
        getattr(args, 'reserve_free_bytes', 2*1024**3))
    report = {'contract': CONTRACT, 'status': 'running', 'accepted_new_dataset': False,
        'manifest_path': str(manifest_path), 'manifest_sha256': file_sha256(manifest_path),
        'batch_code_path': str(Path(__file__).resolve()), 'batch_code_sha256': file_sha256(__file__),
        'original_renderer_code_path': str(Path(original.__file__).resolve()),
        'original_renderer_code_sha256': ORIGINAL_RENDERER_SHA256, 'profile': args.profile,
        'task_id': PROFILES[args.profile], 'selected_count': len(rows), 'frozen_count': manifest['count'],
        'offset': offset, 'selected_uuids': [row['uuid'] for row in rows],
        'created_utc': datetime.now(timezone.utc).isoformat(), 'capacity_preflight': capacity,
        'reuse_scope': 'one task-specific environment, full reset and identity binding per episode', 'episodes': []}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open('x') as stream:
        json.dump(report, stream, indent=2)
    pool = RendererPool(make_recorded_renderer)
    try:
        for row in rows:
            episode = Path(row['directory']).resolve(strict=True)
            receipt = completed_output(episode)
            reused = receipt is not None
            if receipt is None:
                receipt = render_episode(SimpleNamespace(episode=episode, data_root=args.data_root), pool)
                receipt['batch_execution'] = {'contract': CONTRACT,
                    'batch_code_path': report['batch_code_path'], 'batch_code_sha256': report['batch_code_sha256'],
                    'original_renderer_code_path': report['original_renderer_code_path'],
                    'original_renderer_code_sha256': report['original_renderer_code_sha256'],
                    'frozen_manifest_path': str(manifest_path), 'frozen_manifest_sha256': report['manifest_sha256'],
                    'environment_lease_number': pool.leases, 'real_environment_count': 1,
                    'original_reset_and_environment_binding_per_episode': True,
                    'factory_override': 'task-specific private environment lease; defer close until batch finally',
                    'batch_report_path': str(args.report.resolve())}
                write_json(episode/'rtx_completion.json', receipt)
            report['episodes'].append({'uuid': row['uuid'], 'task_id': row['task_id'], 'number': row['number'],
                'directory': str(episode), 'reused_completed_output': reused,
                'completed_frames': receipt['completed_frames'], 'environment_binding': receipt['environment_binding'],
                'rtx_completion_sha256': file_sha256(episode/'rtx_completion.json')})
            write_json(args.report, report)
            print(json.dumps(report['episodes'][-1], ensure_ascii=False), flush=True)
        report['status'] = 'render_completed_pending_process_exit'
        report['render_finished_utc'] = datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        report.update(status='rejected', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report.update(new_environment_leases=pool.leases, cleanup_requested=True, cleanup_return_verified=False)
        write_json(args.report, report)
        try:
            pool.close()
        except BaseException as error:
            report.update(status='rejected', close_error=repr(error), close_traceback=traceback.format_exc())
            raise
        finally:
            report['cleanup_return_verified'] = 'close_error' not in report
            if report['status'] == 'render_completed_pending_process_exit':
                report['status'] = 'completed_pending_independent_audit'
            report['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(args.report, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--profile', choices=tuple(PROFILES), required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--limit', type=int, help='Pilot prefix after validating the complete frozen processing cohort')
    parser.add_argument('--offset', type=int, default=0, help='Start offset in the chosen task, for independently audited chunks')
    parser.add_argument('--estimated-bytes-per-frame', type=int, default=450_000)
    parser.add_argument('--reserve-free-bytes', type=int, default=2*1024**3)
    run(parser.parse_args())


if __name__ == '__main__':
    main()

