"""Local PICO -> fixed SONIC v1.1 -> SIMPLE human demonstration collector.

The simulator is paused when tracking is unavailable. Recorded-input mode is
an explicitly rejected-for-training diagnostic, never a new human episode.
Imports are side-effect free; all SDK, simulator and rendering startup is in run.
"""
from __future__ import annotations

import argparse
from collections import deque
import copy
from dataclasses import replace
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np

from pico_live_input import PicoReceiver, parse_tracking, validate_frame
from pico_sonic_reference import PicoSonicReference, WRIST_NAMES, GEAR, body24_to_smpl
from pico_teleop_recording import AsyncTrainingCapture, CommandLedger, RecordingState
from pico_hand_control import named_hand_targets, profile_receipt, hand_feedback
from replay_handover_teacher import (MODEL_HAND_NAMES, RENDERER_LINKS, applied_force_state,
    bound_calibration_assets, calibration_source_identities, capture_tick_index,
    file_sha256, json_text, write_json)


class RecordedInput:
    """Time-causal diagnostic playback; it cannot create new human data."""
    def __init__(self, path):
        rows = json.loads(Path(path).read_text())
        self.frames = [parse_tracking(row['data'], 0., i+1) for i, row in enumerate(rows)]
        source_kinds = set()
        for row in rows:
            value = row['data'].get('value', row['data'])
            value = json.loads(value) if isinstance(value, str) else value
            kind = value.get('diagnostic_source_kind', 'recorded_pico_diagnostic')
            if not isinstance(kind, str) or not kind:
                raise ValueError('Diagnostic source kind must be a nonempty string')
            source_kinds.add(kind)
        if len(source_kinds) != 1:
            raise ValueError('Mixed diagnostic source kinds are not allowed')
        self.source_kind = next(iter(source_kinds))
        if len(self.frames) < 2:
            raise ValueError('Recorded diagnostic needs at least two actual packets')
        self.times = np.array([f.device_timestamp_ns for f in self.frames], dtype=np.int64)
        if np.any(np.diff(self.times) <= 0):
            raise ValueError('Recorded device timestamps must strictly increase')
        self.times = (self.times-self.times[0])/1e9

    def at(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Playback time must be finite and nonnegative')
        if elapsed > self.times[-1]:
            raise EOFError('Recorded input ended; refusing to repeat or extrapolate it')
        index = max(0, int(np.searchsorted(self.times, elapsed, side='right'))-1)
        return replace(self.frames[index], host_received_s=time.monotonic()-(elapsed-self.times[index]))


def root_pose(model, data):
    body = model.body('pelvis').id
    return np.r_[data.xpos[body], data.xquat[body]].copy()


def physical_snapshot(model, data, body_names):
    """Small actual-state receipt; no renderer, model inference, or state mutation."""
    pelvis = model.body('pelvis').id
    root = root_pose(model, data)
    tilt = float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3,3)[2,2], -1, 1))))
    return {'sim_time_s': float(data.time), 'root_pose7': root.tolist(),
        'pelvis_z_m': float(data.xpos[pelvis,2]), 'pelvis_tilt_deg': tilt,
        'body_joint_names': list(body_names),
        'body_q29': [float(data.joint(name).qpos[0]) for name in body_names],
        'body_dq29': [float(data.joint(name).qvel[0]) for name in body_names],
        'hand_joint_names': list(MODEL_HAND_NAMES),
        'hand_q14': [float(data.joint(name).qpos[0]) for name in MODEL_HAND_NAMES],
        'state_finite': all(np.isfinite(value).all() for value in
            (data.qpos, data.qvel, data.qacc, data.ctrl))}


def require_standing_warmup(before, after):
    """Readiness for the standing initial reference, not a task-success gate."""
    minimum_z = max(.45, float(before['pelvis_z_m']) - .12)
    receipt = {'scope': 'standing warmup readiness only; not task success',
        'minimum_pelvis_z_m': minimum_z, 'maximum_pelvis_tilt_deg': 25.,
        'passed': bool(after['state_finite'] and after['pelvis_z_m'] >= minimum_z
                       and after['pelvis_tilt_deg'] <= 25.)}
    if not receipt['passed']:
        raise ValueError(f"Standing warmup not ready: pelvis_z={after['pelvis_z_m']:.3f}m, "
                         f"tilt={after['pelvis_tilt_deg']:.1f}deg, finite={after['state_finite']}")
    return receipt


def require_neutral_calibration(reference, body24x7):
    """Require an upright initial human reference; never remove real roll/pitch."""
    from scipy.spatial.transform import Rotation
    _, root, _ = body24_to_smpl(body24x7, reference.rest_joints, reference.fk_parents)
    matrix = Rotation.from_quat(root[[1,2,3,0]]).as_matrix()
    tilt = float(np.degrees(np.arccos(np.clip(matrix[2,2], -1, 1))))
    if not np.isfinite(tilt) or tilt > 15.:
        raise ValueError(f'校准被拒绝：人体参考倾角 humanTilt={tilt:.1f}°，要求≤15°。'
                         '请戴好头显、站直并完成全身标定后重试。')
    return {'scope':'initial human reference only; not task success',
        'reference_root_wxyz':root.tolist(),'reference_root_tilt_deg':tilt,
        'maximum_tilt_deg':15.,'passed':True}


def source_code_identities():
    """Source files present when this session starts, not a Git cleanliness claim."""
    names = ('pico_simple_teleop.py','pico_live_input.py','pico_sonic_reference.py',
             'pico_teleop_recording.py','pico_teleop_ui.py','pico_deferred_capture.py',
             'render_deferred_pico.py','finish_pico_session.py','audit_pico_teleop_episode.py',
             'start_pico_teleop.sh','handover_contact_quality.py','handover_geometry_quality.py',
             'pico_hand_control.py')
    directory = Path(__file__).resolve().parent
    return {name:file_sha256(directory/name) for name in names}


def disable_instance_render_rate_limits(settings):
    """Change only this running Kit instance; never edit shared Kit files."""
    receipt = {'scope':'current process Kit settings only; no physical clock change',
        'settings':{}}
    for loop in ('main','present','rendering_0'):
        key = f'/app/runLoops/{loop}/rateLimitEnabled'
        before = settings.get(key)
        settings.set_bool(key,False)
        receipt['settings'][key] = {'before':before,'after':settings.get(key)}
    return receipt


def configure_unrelated_viewport(viewport, cameras, *, disable=False):
    """Inspect by default; disable only a proven separate viewport product.

    The caller obtains the viewport from the current process. Neither camera
    products, resolution, synchronous rendering nor physics settings are changed.
    This helper is not enabled by the runner until a separate profiling decision.
    """
    receipt = {'scope':'current viewport only; head camera products remain enabled',
        'requested_disable':bool(disable),'disabled':False,'cameras':{},'reasons':[]}
    if viewport is None:
        receipt['status'] = 'no_viewport'
        return receipt
    viewport_product = viewport.render_product_path
    receipt.update(viewport_render_product_path=viewport_product,
        updates_enabled_before=bool(viewport.updates_enabled))
    if not isinstance(viewport_product,str) or not viewport_product:
        receipt['reasons'].append('viewport_render_product_unknown')
    for name,camera in cameras.items():
        product = camera.get_render_product_path()
        resolution = tuple(int(value) for value in camera.get_resolution())
        receipt['cameras'][name] = {'render_product_path':product,'resolution_wh':list(resolution)}
        if not isinstance(product,str) or not product:
            receipt['reasons'].append(name+':render_product_unknown')
        elif product == viewport_product:
            receipt['reasons'].append(name+':shares_viewport_render_product')
    heads = ('head_stereo_left','head_stereo_right')
    for name in heads:
        camera = receipt['cameras'].get(name)
        if camera is None:
            receipt['reasons'].append(name+':missing')
        elif camera['resolution_wh'] != [640,360]:
            receipt['reasons'].append(name+':expected_640x360')
    if all(name in receipt['cameras'] for name in heads):
        if receipt['cameras'][heads[0]]['render_product_path'] == receipt['cameras'][heads[1]]['render_product_path']:
            receipt['reasons'].append('head_stereo_products_not_distinct')
    if receipt['reasons']:
        receipt['status'] = 'refused'
    elif disable:
        viewport.updates_enabled = False
        receipt['updates_enabled_after'] = bool(viewport.updates_enabled)
        receipt['disabled'] = not receipt['updates_enabled_after']
        receipt['status'] = 'disabled' if receipt['disabled'] else 'disable_failed'
    else:
        receipt['status'] = 'inspected_without_change'
        receipt['updates_enabled_after'] = bool(viewport.updates_enabled)
    return receipt


def load_environment_config(path, data_root):
    """Validate a saved Handover layout and all explicit local resource paths."""
    path, data_root = Path(path).resolve(strict=True), Path(data_root).resolve(strict=True)
    state = json.loads(path.read_text())
    if not isinstance(state,dict) or state.get('uid') != 'g1_wholebody_handover_teleop':
        raise ValueError('Environment config must be the G1 wholebody Handover task')
    if state.get('robot_cfg',{}).get('uid') != 'g1_sonic':
        raise ValueError('Environment config must use the matching G1 SONIC robot')
    if not isinstance(state.get('dr_state_dict'),dict) or not isinstance(state.get('layout'),dict):
        raise ValueError('Environment config needs saved DR state and layout')
    local_paths = {}
    def visit(value, location='environment'):
        if isinstance(value,dict):
            for key,item in value.items():
                label = location+'.'+key
                if key in ('usd_path','mdl_path','path','data_dir') and isinstance(item,str):
                    if '://' in item:
                        raise ValueError('Remote resource in environment config: '+label)
                    candidate = Path(item)
                    candidate = candidate if candidate.is_absolute() else data_root/item.removeprefix('data/')
                    candidate = candidate.resolve(strict=True)
                    local_paths[label] = str(candidate)
                else:
                    visit(item,label)
        elif isinstance(value,list):
            for index,item in enumerate(value):
                visit(item,f'{location}[{index}]')
    visit(state)
    scene = state['dr_state_dict'].get('scene',{})
    conf = scene.get('conf',{})
    if not scene.get('name') or conf.get('name') != scene['name'] or scene.get('uid') != 'hssd:'+str(conf.get('uid')):
        raise ValueError('Saved HSSD scene uid/name/config identity is incomplete or inconsistent')
    scene_file = (data_root/scene['data_dir']/(scene['name']+'.usd')).resolve(strict=True)
    local_paths['authoritative_scene_usd'] = str(scene_file)
    receipt = {'path':str(path),'sha256':file_sha256(path),'uid':state['uid'],
        'scene_name':scene['name'],'local_resource_paths':local_paths,
        'scope':'fixed saved environment for diagnostics; not randomized evaluation'}
    return state,receipt


def bind_saved_scene_manager(task, state):
    """Clone only this task's manager so UID lookup preserves saved room config."""
    scene = state['dr_state_dict']['scene']
    randomizer = task.dr.get_randomizer('scene')
    manager = copy.copy(randomizer.scene_manager)
    if not hasattr(manager,'hssd_scenes'):
        raise ValueError('Saved Handover scene binding requires the local HSSD manager')
    manager.hssd_scenes = [copy.deepcopy(scene['conf'])]
    randomizer.scene_manager = manager
    return {'scope':'task-local manager copy; no shared resource/config mutation',
        'uid':scene['uid'],'name':scene['name']}


def environment_binding_receipt(expected, actual):
    """Compare the real reconstructed DR/layout, including room and materials."""
    def digest(state):
        selected = {key:state[key] for key in ('dr_state_dict','layout')}
        plain = json.loads(json_text(selected))
        return hashlib.sha256(json.dumps(plain,sort_keys=True,separators=(',',':'),
            allow_nan=False).encode()).hexdigest()
    expected_sha, actual_sha = digest(expected),digest(actual)
    return {'scope':'exact saved DR state plus actual reconstructed layout',
        'expected_state_sha256':expected_sha,'actual_state_sha256':actual_sha,
        'matched':expected_sha==actual_sha,
        'expected_scene':expected['dr_state_dict']['scene']['name'],
        'actual_scene':actual['dr_state_dict']['scene']['name']}


def configure_mujoco_egl(environ, *, mujoco_already_loaded):
    """Select EGL before MuJoCo imports; never pretend a loaded backend changed."""
    before = environ.get('MUJOCO_GL')
    if mujoco_already_loaded and before != 'egl':
        raise ValueError('MuJoCo was already imported with another GL backend; use a fresh EGL process')
    environ['MUJOCO_GL'] = 'egl'
    return {'scope':'current process preview backend only','before':before,'after':environ['MUJOCO_GL']}


def integration_state_snapshot(mujoco_module, model, data, observation_sim_time_s):
    """Archive the observed pre-action state without advancing or recomputing it."""
    if not np.isfinite(observation_sim_time_s) or not np.isclose(
            float(observation_sim_time_s),float(data.time),rtol=0,atol=1e-9):
        raise ValueError('Deferred observation and actual integration state timestamps differ')
    spec = mujoco_module.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco_module.mj_stateSize(model,spec),dtype=np.float64)
    mujoco_module.mj_getState(model,data,value,spec)
    if not np.isfinite(value).all():
        raise ValueError('Deferred integration state contains nonfinite values')
    return value


def read_live_frame(receiver):
    """A malformed new packet is no input, never a reason to reuse older poses."""
    try:
        return receiver.snapshot(),None
    except ValueError as error:
        return None,('原始PICO数据包无效，已暂停并拒收当前示例；请恢复有效追踪后重新校准。'
                     f' {type(error).__name__}: {str(error)[:500]}')


def wait_fresh_command_frame(receiver, previous, last_timestamp_ns, deadline_s, *,
                             now=time.monotonic, sleep=time.sleep):
    """Wait within the existing lateness budget; never pad a new command.

    A short catch-up step may inspect the same 90 Hz packet twice. Waiting for
    its successor is valid; recording the repeated packet as a new 30 Hz sample
    is not. Malformed/invalid successors still fail immediately.
    """
    while now() < deadline_s:
        candidate, error = read_live_frame(receiver)
        if error:
            raise ValueError(error)
        if now() >= deadline_s:
            break
        if candidate is not None:
            quality = validate_frame(candidate, previous=previous, now=now())
            if not quality['valid']:
                raise ValueError('Tracking rejected while waiting: '+','.join(quality['reasons']))
            if (candidate.device_timestamp_ns != last_timestamp_ns and quality['recordable']
                    and now() < deadline_s):
                return candidate, quality
            previous = candidate
        remaining = deadline_s-now()
        if remaining > 0:
            sleep(min(.0015, remaining))
    raise RuntimeError('等待新的PICO动作包超时，已暂停；不会重复动作包充当新采集帧')


def rising_operator_commands(buttons, last_buttons):
    return [action for key, action in (('X','calibrate'),('Y','toggle_follow'),('A','toggle_record'))
        if buttons.get(key,False) and not last_buttons.get(key,False)]


def prepare_deferred_cache(prepare_function, output, index, *, model, objects, joint_names):
    """Do model I/O and verifier allocation before live control or recording."""
    directory = Path(output)/'prepared_models'/f'{index:04d}'
    bindings = {role:model.body(objects[role].id).name for role in ('target','container')}
    started = time.perf_counter()
    prepared = prepare_function(directory,model=model,joint_names=list(joint_names),
        metadata={'object_body_names':bindings})
    return prepared,{'directory':str(directory),'object_body_names':bindings,
        'preparation_wall_s':time.perf_counter()-started,
        'scope':'prepared outside live control; reader independently revalidates archived files'}


def prepare_quality_geometry(cache, output, index):
    """Serialize/hash the large immutable geometry cache outside live control."""
    path = Path(output)/'geometry_caches'/f'{index:04d}.json'
    if path.exists():
        raise ValueError('Refusing to overwrite a geometry cache linked by an earlier episode')
    write_json(path,cache)
    return {'path':str(path),'sha256':file_sha256(path)}


def quality_geometry_source_identities(directory):
    directory = Path(directory)
    return {key:file_sha256(directory/name) for key,name in (
        ('raw_samples_sha256','raw_quality_samples.jsonl'),
        ('cache_sha256','container_geometry_cache.json'),
        ('completed_samples_sha256','quality_samples.jsonl'))}


def sample_original_solved_quality(model, data, *, completed_steps, target_body,
                                   container_body, solver_stage):
    """Cheap live evidence only: original solved contacts/forces and body poses."""
    from handover_contact_quality import SOLVED_STATE_STAGE
    from handover_geometry_quality import _body_id, target_contacts, target_support_status
    if solver_stage != SOLVED_STATE_STAGE:
        raise ValueError('Original solved support requires sampling before observation refresh')
    if isinstance(completed_steps,bool) or not isinstance(completed_steps,(int,np.integer)) or completed_steps < 1:
        raise ValueError('Completed50Hz steps must be positive integers')
    target,container,pelvis = (_body_id(model,value) for value in (target_body,container_body,'pelvis'))
    target_name,container_name = model.body(target).name,model.body(container).name
    return {'completed_steps':int(completed_steps),'sim_time_s':float(data.time),
        'actual_target_xyz_m':data.xpos[target].tolist(),'pelvis_z_m':float(data.xpos[pelvis,2]),
        'contacts':target_contacts(model,data,target_name,container_label=container_name),
        'original_live_support':target_support_status(model,data,target_name,container_label=container_name),
        'target_pose7':np.r_[data.xpos[target],data.xquat[target]].tolist(),
        'container_pose7':np.r_[data.xpos[container],data.xquat[container]].tolist(),
        'target_body_name':target_name,'container_body_name':container_name,
        'solver_stage':SOLVED_STATE_STAGE}


def complete_quality_geometry(raw_samples, cache):
    """Pure geometry on archived poses, retaining the original solved support.

    This function never creates a simulator, calls forward dynamics, or estimates
    contact forces. It can be re-run by an independent archive auditor.
    """
    from handover_contact_quality import SOLVED_STATE_STAGE
    from handover_geometry_quality import container_footprint_from_poses
    result = []
    support_keys = {'support_measurement','target_mass_kg','target_weight_n',
        'container_target_contact_count','container_force_on_target_world_n',
        'container_upward_support_force_n','container_support_weight_fraction'}
    previous_time = None
    for index,raw in enumerate(raw_samples):
        if raw.get('solver_stage') != SOLVED_STATE_STAGE or type(raw.get('completed_steps')) is not int or raw.get('completed_steps') != index+1:
            raise ValueError('Raw solved quality stage or dense50Hz step sequence differs')
        stamp = float(raw['sim_time_s'])
        if not np.isfinite(stamp) or (previous_time is not None and not np.isclose(stamp-previous_time,.02,rtol=0,atol=1e-6)):
            raise ValueError('Raw solved quality is not a real50Hz clock')
        previous_time = stamp
        for key,cache_key in (('target_body_name','target_root_body_name'),('container_body_name','container_root_body_name')):
            if raw.get(key) != cache.get(cache_key):
                raise ValueError('Raw solved quality body identity differs from geometry cache')
        for key in ('target_pose7','container_pose7'):
            pose = np.asarray(raw[key],dtype=np.float64)
            if pose.shape != (7,) or not np.isfinite(pose).all() or not np.isclose(np.linalg.norm(pose[3:]),1.,rtol=0,atol=1e-6):
                raise ValueError('Invalid original solved pose: '+key)
        target_xyz = np.asarray(raw['actual_target_xyz_m'],dtype=float)
        if target_xyz.shape != (3,) or not np.allclose(target_xyz,raw['target_pose7'][:3],rtol=0,atol=1e-9):
            raise ValueError('Original solved target pose and measured XYZ differ')
        if not isinstance(raw.get('contacts'),dict) or any(type(raw['contacts'].get(key)) is not bool
                for key in ('left_hand','right_hand','container')):
            raise ValueError('Original solved contact booleans are incomplete')
        support = raw.get('original_live_support')
        if not isinstance(support,dict) or set(support) != support_keys or support.get('support_measurement') != 'live_solved_mujoco_contact_force':
            raise ValueError('Original solved support evidence is incomplete or replaced')
        force = np.asarray(support['container_force_on_target_world_n'],dtype=float)
        if force.shape != (3,) or not np.isfinite(force).all():
            raise ValueError('Original solved support force is not finite')
        for key in support_keys-{'support_measurement','container_force_on_target_world_n'}:
            if not np.isfinite(float(support[key])):
                raise ValueError('Original solved support metric is not finite')
        if (support['target_mass_kg'] <= 0 or support['target_weight_n'] <= 0
                or type(support['container_target_contact_count']) is not int or support['container_target_contact_count'] < 0):
            raise ValueError('Original solved mass/weight/contact count is invalid')
        geometry = container_footprint_from_poses(cache,
            container_pose_wxyz=raw['container_pose7'],target_pose_wxyz=raw['target_pose7'])
        geometry.update(copy.deepcopy(support))
        geometry['placement_geometry_measurement'] = 'deferred_geometry_with_original_solved_support'
        geometry['limitations'].append('Geometry is computed later from original solved poses; support is the original live solved force, never reconstructed.')
        row = copy.deepcopy(raw)
        row['container_geometry'] = geometry
        row['quality_geometry_mode'] = 'deferred_geometry_with_original_solved_support'
        result.append(row)
    return result


class RenderProfiler:
    """Optional API wall-time probes; no extra render, sync, or physics calls."""
    def __init__(self, path):
        self.stream = Path(path).open('x',buffering=1)
        self.patches = []
        self.statistics = {}
        self.errors = []
        self.context = {}
        self.stack = []
        self.next_call = 0

    def set_context(self, phase, tick50, sim_time_s):
        self.context = {'phase':phase,'tick50':tick50,'sim_time_s':float(sim_time_s)}

    def wrap(self, owner, name, label):
        original = getattr(owner,name)
        @functools.wraps(original)
        def measured(*args,**kwargs):
            self.next_call += 1
            call_id = self.next_call
            row = {**self.context,'name':label,'call_id':call_id,
                'parent_call_id':self.stack[-1] if self.stack else None,
                'depth':len(self.stack)}
            self.stack.append(call_id)
            started = time.perf_counter()
            try:
                return original(*args,**kwargs)
            except BaseException as error:
                row['error'] = repr(error)
                raise
            finally:
                elapsed = time.perf_counter()-started
                self.stack.pop()
                row['elapsed_s'] = elapsed
                stat = self.statistics.setdefault(label,{'calls':0,'total_s':0.,'maximum_s':0.})
                stat['calls'] += 1
                stat['total_s'] += elapsed
                stat['maximum_s'] = max(stat['maximum_s'],elapsed)
                try:
                    self.stream.write(json_text(row)+'\n')
                except Exception as error:
                    self.errors.append(repr(error))
                    self.errors = self.errors[-8:]
        setattr(owner,name,measured)
        self.patches.append((owner,name,original,measured))

    def install(self, isaac, orchestrator):
        if not hasattr(isaac,'world'):
            raise ValueError('Render profiling requires env.reset() to create the Isaac world first')
        for owner,name,label in ((isaac,'sync_states','isaac.sync_states'),
            (isaac.world,'render','isaac.world.render'),
            (isaac,'observation_pose_diagnostics','isaac.observation_pose_diagnostics'),
            (isaac,'render','isaac.render'),
            (orchestrator,'step','rep.orchestrator.step')):
            self.wrap(owner,name,label)
        return self

    def receipt(self):
        return {'scope':'inclusive CPU API wall time, nested rows are not additive; no extra GPU sync',
            'statistics':self.statistics,'errors':list(self.errors)}

    def close(self):
        for owner,name,original,measured in reversed(self.patches):
            if getattr(owner,name) is measured:
                setattr(owner,name,original)
        self.patches.clear()
        self.stream.close()


def actual_renderer_poses(model, data):
    camera = model.camera('head_stereo_left').id
    return {'links': {name: {'position_m': data.xpos[model.body(name).id].copy(),
        'quaternion_wxyz': data.xquat[model.body(name).id].copy()} for name in RENDERER_LINKS},
        'cameras': {'head_stereo_left': {'position_m': data.cam_xpos[camera].copy(),
            'rotation_usd_axes': data.cam_xmat[camera].copy()}}}


def initialize_pose(raw, default_angles, body_names):
    """One cold start before controller history. No object or mid-episode restore."""
    import mujoco
    model, data = raw.mujoco.mjModel, raw.mujoco.mjData
    for name, value in zip(body_names, default_angles):
        data.joint(name).qpos[0] = value
    for name in MODEL_HAND_NAMES:
        data.joint(name).qpos[0] = 0.
    data.qvel[:] = 0.
    data.qfrc_applied[:] = 0.
    data.xfrc_applied[:] = 0.
    mujoco.mj_forward(model, data)
    # Analytic transformed local AABB corners of the foot collision geometry.
    feet = {model.body(side+'_ankle_roll_link').id for side in ('left', 'right')}
    heights = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) not in feet or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        center, half = model.geom_aabb[geom, :3], model.geom_aabb[geom, 3:]
        corners = center + np.array([[a,b,c] for a in (-1,1) for b in (-1,1) for c in (-1,1)])*half
        heights.extend((corners @ data.geom_xmat[geom].reshape(3,3).T + data.geom_xpos[geom])[:,2])
    if not heights:
        raise ValueError('Cannot identify actual foot collision bounds')
    pelvis = model.body('pelvis').id
    root_joint = int(model.body_jntadr[pelvis])
    z_index = int(model.jnt_qposadr[root_joint])+2
    offset = .002-float(min(heights))
    data.qpos[z_index] += offset
    mujoco.mj_forward(model, data)
    return {'kind':'one_time_default_sonic_pose_before_measured_warmup',
        'root_z_offset_m':offset,'root_pose7':root_pose(model,data).tolist(),
        'object_restores_during_control':False,'external_assistance':False}


def run(args):
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.exists() or output.is_relative_to(repo):
        raise ValueError('Use a new output directory outside Git')
    if args.diagnostic_record and (not args.pico_replay or args.render_mode == 'none'):
        raise ValueError('Diagnostic capture requires recorded input and an explicit preview/capture mode')
    output.mkdir(parents=True)
    resources = args.sonic_resources.resolve(strict=True)
    data_root, asset_shas = bound_calibration_assets(args.data_root, calibration_source_identities())
    environment_state = environment_receipt = None
    if args.environment_config:
        environment_state,environment_receipt = load_environment_config(args.environment_config,data_root)
    os.environ['SIMPLE_DATA_DIR'] = str(data_root)
    rendered = args.render_mode == 'isaac'
    deferred = args.render_mode == 'mujoco_deferred'
    replay = RecordedInput(args.pico_replay) if args.pico_replay else None
    result = {'status':'initializing','task_id':'simple/G1WholebodyHandoverTeleop-v0',
        'source_kind':replay.source_kind if replay else 'live_pico_human',
        'accepted_new_dataset':False,'physics_hz':200,'decoder_hz':50,'command_rgb_hz':30,
        'absolute_human_root_translation_encoded':False,'render_mode':args.render_mode,
        'raw_pico_va_wva_used':False,'resource_data_root':str(data_root),
        'preview_engine':'mujoco_egl' if deferred else 'isaac_rtx' if rendered else None,
        'training_rgb_engine':'pending_offline_isaac_rtx' if deferred else 'online_isaac_rtx' if rendered else None,
        'preview_is_training_rgb':False if deferred else bool(rendered),
        'render_engine':'deferred_rtx_pending' if deferred else 'online_isaac_rtx' if rendered else 'none',
        'calibration_source_asset_sha256':asset_shas,'episodes':[],
        'source_code_sha256_at_session_start':source_code_identities(),
        'environment_config_source':environment_receipt,
        'replay_source_sha256':file_sha256(args.pico_replay) if replay else None}
    write_json(output/'result.json', result)
    env = ui = receiver = video_bridge = render_profiler = None
    capture = ledger = None
    episode = None
    quality_rows = []
    quality_stream = force_stream = control_trace = None
    wall_periods = deque(maxlen=3000)
    wall_period_count = 0
    maximum_wall_period = 0.
    prepared_deferred = None
    deferred_prepare_index = 0
    geometry_cache_index = 0
    reference = PicoSonicReference(resources, args.human_info)
    pairing = {'body_token_version':'body_token_v1_1','compatibility_verified':True,
        'decoder_sha256':'34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509',
        'evidence':{'scope':'Only new tokens generated here by fixed matching official encoder, not old dataset tokens',
            'encoder_identity':reference.identities}}
    write_json(output/'new_producer_binding.json', pairing)

    def finalize(completed=False, reason=None):
        nonlocal capture, ledger, episode, quality_stream, force_stream, quality_rows, following
        if episode is None:
            return
        closing_episode, closing_capture, closing_ledger = episode, capture, ledger
        following = False
        if ui:
            ui.update({'status':'saving','recording':None,'accepted_new_dataset':False,
                'note':'仿真已暂停，保存原始证据并计算几何；此后仍需RTX补图和独立验收'})
        streams = (quality_stream, force_stream)
        rows = quality_rows
        # Detach first: a writer failure must not reuse or close an episode twice.
        capture = ledger = episode = quality_stream = force_stream = None
        quality_rows = []
        outcome = {'accepted_new_dataset':False,'source_kind':result['source_kind']}
        errors = {}
        try:
            if reason:
                closing_episode.reject(reason)
            state = closing_episode.close(completed=completed)
            outcome['state'] = state
        except Exception as error:
            errors['recording_state'] = repr(error)
            state = {'pending_audit':False,'status':'rejected'}
            outcome['state'] = state
        try:
            outcome['capture'] = closing_capture.close(
                completed=completed and state['pending_audit']) if closing_capture else None
        except Exception as error:
            errors['capture'] = repr(error)
        for index, stream in enumerate(streams):
            if stream:
                try:
                    stream.close()
                except Exception as error:
                    errors[f'quality_or_force_stream_{index}'] = repr(error)
        try:
            outcome['ledger'] = closing_ledger.close() if closing_ledger else None
        except Exception as error:
            errors['ledger'] = repr(error)
        if rows:
            try:
                # Physics is paused. Only geometry is deferred; original solved
                # forces are read solely in measured_step and retained verbatim.
                completed_quality = complete_quality_geometry(rows,cache)
                directory = output/'episodes'/closing_episode.human_demo_uuid
                with (directory/'quality_samples.jsonl').open('x') as stream:
                    for row in completed_quality:
                        stream.write(json_text(row)+'\n')
                outcome['quality_geometry_sources'] = quality_geometry_source_identities(directory)
                if outcome['quality_geometry_sources']['cache_sha256'] != prepared_geometry_cache['sha256']:
                    raise ValueError('Geometry cache changed after startup preparation')
                outcome['quality_geometry_mode'] = 'deferred_geometry_with_original_solved_support'
                outcome['physical_gate'] = evaluate_capture(completed_quality, initial_target_z_m=initial_z)
            except Exception as error:
                outcome['physical_gate_error'] = repr(error)
        if errors:
            outcome['finalization_errors'] = errors
            outcome['state']['pending_audit'] = False
            outcome['state']['status'] = 'rejected'
        if deferred:
            pending = (completed and not errors and state.get('pending_audit')
                and outcome.get('capture',{}).get('status') == 'raw_physics_capture_completed_pending_rtx')
            outcome['status'] = 'pending_rtx' if pending else 'rejected'
            outcome['render_engine'] = 'deferred_rtx_pending'
            outcome['rtx_ready'] = False
        directory = output/'episodes'/closing_episode.human_demo_uuid
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'episode.json', outcome)
        result['episodes'].append({'directory':str(directory),**outcome})
        write_json(output/'result.json', result)
        return outcome

    try:
        if deferred:
            result['egl_backend_selection'] = configure_mujoco_egl(os.environ,
                mujoco_already_loaded='mujoco' in sys.modules)
        random.seed(args.seed)
        np.random.seed(args.seed)
        import gymnasium as gym
        import mujoco
        if deferred:
            from pico_deferred_capture import DeferredStateCapture, prepare_deferred_model
            result['actual_mujoco_gl_module'] = mujoco.GLContext.__module__
            if result['actual_mujoco_gl_module'] != 'mujoco.egl':
                raise ValueError('Deferred preview requires the actual MuJoCo EGL backend')
        import simple.envs
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from simple.baselines.sonic_v1_1 import SonicV11OnnxController, DEFAULT_ANGLES, BODY_JOINT_NAMES
        from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
        from simple.baselines.dual_source_psi0 import TRAINING_STATE_JOINT_NAMES
        from handover_contact_quality import evaluate_capture
        from handover_geometry_quality import build_container_geometry_cache
        from pico_teleop_ui import LocalTeleopUI
        ui = LocalTeleopUI(port=args.port).start()
        result['control_url'] = ui.url
        controller = SonicV11OnnxController(resources, compatibility_evidence=output/'new_producer_binding.json',
            sync_mujoco_observations=True, pd_profile='official_v1_1_ankle_pitch_1_5')
        config = SimLoopConfig().load_wbc_yaml()
        config.update(ENV_NAME='simple', SIMULATE_DT=.005, ENABLE_ELASTIC_BAND=False)
        env = gym.make(result['task_id'], sonic_config=config,
            sim_mode='mujoco_isaac' if rendered else 'mujoco',headless=True,dr_level=0,
            sync_mujoco_observations=True,isaac_observation_only=rendered,disable_env_checker=True)
        raw = env.unwrapped
        if environment_state:
            result['saved_scene_manager_binding'] = bind_saved_scene_manager(raw.task,environment_state)
        if args.render_mode == 'none':
            raw.task.sensor_cfgs = {}
            raw._render_frame = lambda: {}
        elif rendered:
            raw._render_frame = lambda: raw.isaac.render()
        reset_options = {'state_dict':copy.deepcopy(environment_state),'dr_level':None} if environment_state else None
        obs, _ = env.reset(seed=args.seed,options=reset_options)
        renderer_environment_config = copy.deepcopy(raw.task.state_dict())
        write_json(output/'environment_config.json',renderer_environment_config)
        if environment_state:
            actual_environment = renderer_environment_config
            result['environment_config_binding'] = environment_binding_receipt(environment_state,actual_environment)
            write_json(output/'result.json',result)
            if not result['environment_config_binding']['matched']:
                raise ValueError('Saved environment DR/layout did not exactly reconstruct; refusing same-scene comparison')
        if rendered:
            import carb
            result['instance_render_rate_limits'] = disable_instance_render_rate_limits(carb.settings.get_settings())
            if args.disable_unrelated_viewport:
                from omni.kit.viewport.utility import get_active_viewport
                result['unrelated_viewport'] = configure_unrelated_viewport(
                    get_active_viewport(),raw.isaac.cameras,disable=True)
            if args.profile_render:
                import importlib
                isaac_module = importlib.import_module('simple.engines.isaacsim')
                render_profiler = RenderProfiler(output/'render_profile.jsonl')
                render_profiler.install(raw.isaac,isaac_module.rep.orchestrator)
                render_profiler.set_context('after_initial_environment_reset',None,raw.mujoco.mjData.time)
        raw_get_obs = raw._get_obs
        render_next = True
        def selective_obs():
            if render_next:
                return raw_get_obs()
            mujoco.mj_forward(raw.mujoco.mjModel, raw.mujoco.mjData)
            q = raw.mujoco.get_robot_qpos()
            return {'joint_names':tuple(q),'joint_qpos':np.array(list(q.values()),dtype=np.float32),
                'sim_time_s':float(raw.mujoco.mjData.time)}
        raw._get_obs = selective_obs
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        result['hand_control_profile'] = profile_receipt()
        if not np.isclose(model.opt.timestep,.005) or raw.control_decimal != 4 or not np.array_equal(model.opt.gravity,[0,0,-9.81]):
            raise ValueError('Require true 200Hz gravity physics / 50Hz controller')
        result['initialization'] = initialize_pose(raw, DEFAULT_ANGLES, BODY_JOINT_NAMES)
        if deferred:
            prepared_deferred,preparation = prepare_deferred_cache(prepare_deferred_model,
                output,deferred_prepare_index,model=model,objects=raw.mujoco.mj_objects,
                joint_names=TRAINING_STATE_JOINT_NAMES)
            result['prepared_deferred_models'] = [preparation]
        result['initial_physical_state'] = physical_snapshot(model, data, BODY_JOINT_NAMES)
        warmup_baseline = result['initial_physical_state']
        obs = raw_get_obs()
        result['instruction'] = raw.task.instruction
        mujoco.mj_saveModel(model,str(output/'physics_model.mjb'))
        result['physics_model_sha256'] = file_sha256(output/'physics_model.mjb')
        cache = build_container_geometry_cache(model,data,container_body=raw.mujoco.mj_objects['container'].id,
            target_body=raw.mujoco.mj_objects['target'].id)
        prepared_geometry_cache = prepare_quality_geometry(cache,output,geometry_cache_index)
        result['prepared_quality_geometry_caches'] = [prepared_geometry_cache]
        original_step = raw.mujoco.step
        substeps = 0
        active_local = None
        def measured_step(*a, **kw):
            nonlocal substeps
            value = original_step(*a, **kw)
            substeps += 1
            force = applied_force_state(data)
            if not force['all_zero']:
                raise RuntimeError('Nonzero externally applied force')
            if force_stream:
                force_stream.write(json_text({'sim_time_s':float(data.time),**force})+'\n')
            if active_local is not None and substeps % 4 == 0:
                sample_started = time.perf_counter()
                row = sample_original_solved_quality(model,data,completed_steps=active_local+1,
                    target_body=raw.mujoco.mj_objects['target'].id,
                    container_body=raw.mujoco.mj_objects['container'].id,
                    solver_stage='post_mj_step_before_observation_refresh')
                row['live_contacts_support_sampling_wall_s'] = time.perf_counter()-sample_started
                quality_rows.append(row)
                quality_stream.write(json_text(row)+'\n')
            return value
        raw.mujoco.step = measured_step
        control_trace = (output/'control_trace.jsonl').open('x', buffering=1)
        warmup = LivePoseSonicWarmup(resources,hand_policy='measured')
        for tick in range(50):
            render_next = capture_tick_index(tick+1) is not None
            if render_profiler:
                render_profiler.set_context('startup_warmup',tick,data.time)
            obs, *_ = env.step(warmup.command(raw.task.robot,controller,obs))
            control_trace.write(json_text({'phase':'startup_warmup','tick50':tick,
                'actual_after_step':physical_snapshot(model,data,BODY_JOINT_NAMES)})+'\n')
        result['post_warmup_physical_state'] = physical_snapshot(model,data,BODY_JOINT_NAMES)
        write_json(output/'result.json',result)
        result['standing_warmup_readiness'] = require_standing_warmup(
            result['initial_physical_state'], result['post_warmup_physical_state'])
        result.update(controller_identity=controller.identity,warmup=warmup.receipt,
            encoder_identity=reference.identities,status='waiting_for_calibration')
        if not replay:
            receiver = PicoReceiver().start()
        if args.headset_video:
            from pico_teleop_ui import HeadsetVideoBridge
            video_bridge = HeadsetVideoBridge(expected_peer_ip=args.pico_ip).start()
            # Let the operator see the paused initial scene before calibrating.
            # The latest-frame buffer retains this until the camera connects.
            video_bridge.update(obs)
        frame = previous = None
        last_buttons = {}
        pending_input_commands = []
        following = False
        error_text = None
        global_tick = 0
        wall_deadline = None
        source_elapsed = 0.
        diagnostic_started = False
        max_lateness = 0.
        replay_calibrated = False
        started_wall = time.monotonic()
        command = None
        hands = None
        last_step_wall = None
        quality = {'valid':False,'reasons':['waiting_for_input']}
        initial_z = float(raw.mujoco.mj_objects['target'].xpos[2])
        reference_evidence = None
        calibration_fresh = False
        physics_ready = True
        last_result_write_wall = 0.
        last_preview_time = None
        last_snapshot_error = None

        def invalidate_calibration():
            nonlocal following, command, calibration_fresh, reference_evidence
            following = False
            command = reference_evidence = None
            calibration_fresh = False
            reference.calibration = None
            reference.last_body = reference.last_timestamp_ns = None

        def reject_and_pause(reason):
            # A bad episode or input cannot close the UI/receiver during live use.
            # A fresh calibration is required even for a short operator pause.
            nonlocal error_text, active_local
            error_text = str(reason)
            active_local = None
            invalidate_calibration()
            result.setdefault('rejections',[]).append({'sim_time_s':float(data.time),
                'tick50':global_tick,'reason':error_text})
            result['rejections'] = result['rejections'][-128:]
            try:
                control_trace.write(json_text({'phase':'rejected_and_paused',
                    **result['rejections'][-1]})+'\n')
            except Exception as trace_error:
                result['trace_write_error'] = repr(trace_error)
            try:
                finalize(False,error_text)
            except Exception as finalization_error:
                result['episode_finalize_error'] = repr(finalization_error)
        while True:
            snapshot_error = None
            if args.seconds and time.monotonic()-started_wall >= args.seconds:
                break
            if replay:
                try:
                    frame = replay.at(source_elapsed)
                except EOFError:
                    if episode and episode.status == 'recording':
                        episode.request_save(); finalize(True)
                    break
            else:
                frame,snapshot_error = read_live_frame(receiver)
                if snapshot_error:
                    if snapshot_error != last_snapshot_error:
                        reject_and_pause(snapshot_error)
                    last_snapshot_error = snapshot_error
                else:
                    last_snapshot_error = None
            commands, pending_input_commands = pending_input_commands, []
            commands.extend(ui.commands())
            if frame:
                quality = validate_frame(frame,previous=previous)
                if previous is None or frame.sequence != previous.sequence:
                    if quality['valid']:
                        commands.extend(rising_operator_commands(frame.buttons,last_buttons))
                        last_buttons = dict(frame.buttons)
                    previous = frame
                if replay and not replay_calibrated:
                    commands += ['calibrate','follow']
                    replay_calibrated = True
            else:
                quality = {'valid':False,'reasons':['malformed_tracking_packet' if snapshot_error else 'no_tracking_packet']}
            if args.diagnostic_record and following and not diagnostic_started:
                commands.append('record'); diagnostic_started = True
            if 'quit' in commands:
                break
            for action in commands:
                if action == 'toggle_follow':
                    action = 'pause' if following else 'follow'
                if action == 'toggle_record':
                    action = 'save' if episode else 'record'
                try:
                    if action == 'calibrate':
                        if episode:
                            raise ValueError('Discard or save the current episode before calibration')
                        if not quality['valid']:
                            raise ValueError('Calibration requires valid live full-body tracking')
                        if not physics_ready:
                            raise ValueError('Reset the simulator after a physical readiness failure')
                        human_readiness = require_neutral_calibration(reference,frame.body24x7)
                        robot_readiness = require_standing_warmup(warmup_baseline,
                            physical_snapshot(model,data,BODY_JOINT_NAMES))
                        reference.calibrate(frame.body24x7,root_pose(model,data),
                            {name:float(data.joint(name).qpos[0]) for name in WRIST_NAMES})
                        reference.calibration.update(human_readiness=human_readiness,
                            robot_readiness=robot_readiness)
                        following = False; error_text = None; calibration_fresh = True
                        write_json(output/'last_calibration.json',reference.calibration)
                    elif action == 'follow':
                        if not calibration_fresh or not reference.calibration or not quality['valid'] or not physics_ready:
                            raise ValueError('Calibrate valid tracking before following')
                        require_neutral_calibration(reference,frame.body24x7)
                        require_standing_warmup(warmup_baseline,
                            physical_snapshot(model,data,BODY_JOINT_NAMES))
                        following = True; wall_deadline = None; command = None
                        error_text = None
                    elif action == 'pause':
                        reject_and_pause('operator_pause')
                    elif action == 'record':
                        if episode or not following or not (rendered or deferred):
                            raise ValueError('Recording needs active calibrated control and an explicit rendering/capture mode')
                        episode = RecordingState(); episode.request_start()
                    elif action == 'save':
                        if not episode:
                            raise ValueError('No active episode')
                        episode.request_save(); finalize(True); invalidate_calibration()
                    elif action == 'discard':
                        finalize(False,'operator_discard'); invalidate_calibration()
                    elif action == 'reset':
                        if episode:
                            raise ValueError('Discard or save before reset')
                        # Reset is explicit, between demos only; control histories/calibration are cleared.
                        invalidate_calibration(); physics_ready = False
                        reference = PicoSonicReference(resources,args.human_info)
                        raw.mujoco.step = original_step
                        render_next = True
                        args.seed += 1
                        if render_profiler:
                            render_profiler.set_context('explicit_environment_reset',None,data.time)
                        reset_options = {'state_dict':copy.deepcopy(environment_state),'dr_level':None} if environment_state else None
                        obs, _ = env.reset(seed=args.seed,options=reset_options)
                        renderer_environment_config = copy.deepcopy(raw.task.state_dict())
                        if environment_state:
                            binding = environment_binding_receipt(environment_state,raw.task.state_dict())
                            result.setdefault('reset_environment_bindings',[]).append(binding)
                            if not binding['matched']:
                                raise ValueError('Reset failed to reconstruct the bound saved environment')
                        if rendered:
                            result['instance_render_rate_limits'] = disable_instance_render_rate_limits(carb.settings.get_settings())
                            if args.disable_unrelated_viewport:
                                result['unrelated_viewport'] = configure_unrelated_viewport(
                                    get_active_viewport(),raw.isaac.cameras,disable=True)
                        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
                        reset_initialization = initialize_pose(raw,DEFAULT_ANGLES,BODY_JOINT_NAMES)
                        if deferred:
                            deferred_prepare_index += 1
                            prepared_deferred,preparation = prepare_deferred_cache(prepare_deferred_model,
                                output,deferred_prepare_index,model=model,objects=raw.mujoco.mj_objects,
                                joint_names=TRAINING_STATE_JOINT_NAMES)
                            result['prepared_deferred_models'].append(preparation)
                        reset_before = physical_snapshot(model,data,BODY_JOINT_NAMES)
                        warmup_baseline = reset_before
                        obs = raw_get_obs()
                        controller.reset(); warmup.reset()
                        for tick in range(50):
                            render_next = capture_tick_index(tick+1) is not None
                            if render_profiler:
                                render_profiler.set_context('reset_warmup',tick,data.time)
                            obs, *_ = env.step(warmup.command(raw.task.robot,controller,obs))
                            control_trace.write(json_text({'phase':'reset_warmup','tick50':tick,
                                'actual_after_step':physical_snapshot(model,data,BODY_JOINT_NAMES)})+'\n')
                        reset_after = physical_snapshot(model,data,BODY_JOINT_NAMES)
                        result.setdefault('resets',[]).append({'initialization':reset_initialization,
                            'before':reset_before,'after':reset_after})
                        write_json(output/'result.json',result)
                        result['resets'][-1]['standing_warmup_readiness'] = require_standing_warmup(reset_before, reset_after)
                        cache = build_container_geometry_cache(model,data,container_body=raw.mujoco.mj_objects['container'].id,
                            target_body=raw.mujoco.mj_objects['target'].id)
                        geometry_cache_index += 1
                        prepared_geometry_cache = prepare_quality_geometry(cache,output,geometry_cache_index)
                        result['prepared_quality_geometry_caches'].append(prepared_geometry_cache)
                        raw.mujoco.step = measured_step
                        substeps=0; global_tick=0; command=None; previous=None
                        last_preview_time=None
                        physics_ready=True; last_buttons={}; reference_evidence=None
                        pending_input_commands=[]
                        wall_deadline=None; last_step_wall=None
                    else:
                        raise ValueError('Unknown command: '+action)
                except Exception as error:
                    error_text = str(error)
                    if action in ('calibrate','follow','reset') or episode or following:
                        reject_and_pause(error_text)
                    if replay:
                        raise
            if following and not quality['valid']:
                reject_and_pause('Tracking rejected: '+','.join(quality['reasons']))
            if following:
                try:
                    if wall_deadline is None:
                        wall_deadline = time.monotonic()
                    lateness = time.monotonic()-wall_deadline
                    max_lateness = max(max_lateness,lateness)
                    if lateness > args.max_lateness:
                        reject_and_pause(f'Cannot maintain real time: lag={lateness:.3f}s')
                        if replay:
                            raise RuntimeError(error_text)
                        continue
                    state = episode.on_tick(global_tick) if episode else None
                    if state and state['started']:
                        directory = output/'episodes'/episode.human_demo_uuid
                        directory.mkdir(parents=True)
                        initial_z = float(raw.mujoco.mj_objects['target'].xpos[2])
                        metadata = {**result,'episodes':[],
                            'human_demo_uuid':episode.human_demo_uuid,'pico_calibration':reference.calibration,
                            'hand_profile':result['hand_control_profile']['profile_id'],
                            'physics_seed':args.seed,'initial_target_z_m':initial_z,
                            'mujoco_object_body_names':{role:obj.name for role,obj in raw.mujoco.mj_objects.items()},
                            'object_body_names':{role:model.body(raw.mujoco.mj_objects[role].id).name
                                for role in ('target','container')},
                            'mujoco_obj_names':list(raw.mujoco.obj_names),
                            'quality_geometry_mode':'deferred_geometry_with_original_solved_support',
                            'container_geometry_cache_sha256':prepared_geometry_cache['sha256'],
                            'environment_config':copy.deepcopy(renderer_environment_config),
                            'renderer_scene_config_scope':'initial after env.reset before controller warmup; dynamic frame poses come only from archived integration states'}
                        os.link(prepared_geometry_cache['path'],directory/'container_geometry_cache.json',follow_symlinks=False)
                        write_json(directory/'metadata.json',metadata)
                        if deferred:
                            capture = DeferredStateCapture(directory/'deferred_capture',metadata=metadata,
                                joint_names=list(TRAINING_STATE_JOINT_NAMES),warmup_ticks=50,
                                start_time=float(data.time),model=model,prepared_model=prepared_deferred)
                        else:
                            capture = AsyncTrainingCapture(directory/'capture',metadata=metadata,
                                joint_names=list(TRAINING_STATE_JOINT_NAMES),warmup_ticks=50,start_time=float(data.time))
                        ledger = CommandLedger(directory/'commands.jsonl',human_demo_uuid=episode.human_demo_uuid,metadata=metadata)
                        quality_stream=(directory/'raw_quality_samples.jsonl').open('x',buffering=1)
                        force_stream=(directory/'applied_force_samples.jsonl').open('w',buffering=1)
                    due = capture_tick_index(global_tick) is not None
                    control_tick_started = time.perf_counter()
                    encode_seconds = 0.
                    input_wait_seconds = 0.
                    actual_before = physical_snapshot(model,data,BODY_JOINT_NAMES)
                    new_command = due or command is None
                    if new_command:
                        if reference.last_timestamp_ns == frame.device_timestamp_ns:
                            if replay:
                                raise ValueError('No new recorded PICO frame for a new30Hz command')
                            wait_started = time.perf_counter()
                            frame, quality = wait_fresh_command_frame(receiver, frame,
                                reference.last_timestamp_ns, wall_deadline+args.max_lateness)
                            # A short button press in this inspected packet must
                            # survive even if the next callback is its release.
                            pending_input_commands.extend(rising_operator_commands(frame.buttons,last_buttons))
                            last_buttons = dict(frame.buttons)
                            previous = frame
                            input_wait_seconds = time.perf_counter()-wait_started
                        encode_started = time.perf_counter()
                        token, evidence = reference.encode(frame.body24x7,root_pose(model,data),frame.device_timestamp_ns)
                        encode_seconds = time.perf_counter()-encode_started
                        reference_evidence = evidence
                        hands = named_hand_targets(raw.task.robot,frame)
                        for name,value in hands.items():
                            low,high = model.joint(name).range
                            if not low-1e-6 <= value <= high+1e-6:
                                raise ValueError('Hand target exceeds actual joint range: '+name)
                        command = controller.step(token,hands,obs)
                        if ledger:
                            ledger.append('command',physical_tick50=state['local_tick50'],sim_time_s=float(data.time),
                                source_sequence=frame.sequence,device_timestamp_ns=frame.device_timestamp_ns,
                                source_host_received_s=frame.host_received_s,source_raw_packet=frame.raw_payload,
                                body_token64=token,named_hand_targets=hands,reference_evidence=evidence)
                    if capture and state['capture_due']:
                        pelvis=model.body('pelvis').id; root_joint=int(model.body_jntadr[pelvis]); dof=int(model.jnt_dofadr[root_joint])
                        capture_values = dict(tick50=state['local_tick50'],sim_time_s=float(data.time),
                            observed_joint_names=obs['joint_names'],observed_qpos=obs['joint_qpos'],
                            physical_qpos=[float(data.joint(name).qpos[0]) for name in TRAINING_STATE_JOINT_NAMES],
                            physical_qvel=[float(data.joint(name).qvel[0]) for name in TRAINING_STATE_JOINT_NAMES],
                            root_pose7=root_pose(model,data),root_qvel6=data.qvel[dof:dof+6],
                            objects={role:{'pose7':np.r_[raw.mujoco.mj_objects[role].xpos,raw.mujoco.mj_objects[role].xquat]}
                                for role in ('target','container')},command=command if state['action_valid'] else None,
                            )
                        if deferred:
                            integration_state = integration_state_snapshot(mujoco,model,data,obs['sim_time_s'])
                            capture.append(**capture_values,integration_state=integration_state,
                                preview_rgb=obs['head_stereo_left'],source_fields={
                                    'source_raw_packet':frame.raw_payload,
                                    'source_host_received_s':frame.host_received_s,
                                    'source_sequence':frame.sequence,'device_timestamp_ns':frame.device_timestamp_ns,
                                    'reference_evidence':reference_evidence,
                                    'observation_sim_time_s':obs['sim_time_s']})
                        else:
                            capture.enqueue(**capture_values,rgb=obs['head_stereo_left'],
                                renderer_pose_diagnostics=getattr(raw.isaac,'last_observation_pose_diagnostics',None),
                                renderer_authoritative_poses=actual_renderer_poses(model,data),observation_sim_time_s=obs['sim_time_s'])
                    active_local = state['local_tick50'] if state and state['recording'] else None
                    render_next = capture_tick_index(global_tick+1) is not None
                    if render_profiler:
                        render_profiler.set_context('pico_follow',global_tick,data.time)
                    step_started = time.perf_counter()
                    obs,reward,terminated,truncated,info = env.step(command)
                    step_seconds = time.perf_counter()-step_started
                    active_local = None
                    control_trace.write(json_text({'phase':'pico_follow','tick50':global_tick,
                        'source_sequence':frame.sequence,'source_device_timestamp_ns':frame.device_timestamp_ns,
                        'source_inputs':dict(frame.inputs),'named_hand_targets':dict(hands),
                        'new_command30':new_command,'reference':reference_evidence if new_command else None,
                        'timing_s':{'reference_encode':encode_seconds,'environment_step':step_seconds,
                            'fresh_input_wait':input_wait_seconds,
                            'control_tick_before_trace':time.perf_counter()-control_tick_started},
                        'actual_before_step':actual_before,
                        'actual_after_step':physical_snapshot(model,data,BODY_JOINT_NAMES),
                        'terminated':bool(terminated),'truncated':bool(truncated)})+'\n')
                    finite = all(np.isfinite(x).all() for x in (data.qpos,data.qvel,data.qacc,data.ctrl))
                    pelvis=model.body('pelvis').id
                    tilt=float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3,3)[2,2],-1,1))))
                    if not finite or data.xpos[pelvis,2] < .35 or tilt > 75 or truncated:
                        physics_ready = False
                        raise RuntimeError(f'Physical safety check failed: finite={finite}, pelvis_z={data.xpos[pelvis,2]:.3f}, tilt={tilt:.1f}')
                    if capture and capture.failed:
                        raise RuntimeError('Capture failed: '+str(capture.error))
                    if 'head_stereo_left' in obs:
                        if video_bridge:
                            video_bridge.update(obs)
                    now=time.monotonic()
                    if last_step_wall is not None:
                        period = now-last_step_wall
                        wall_periods.append(period)
                        wall_period_count += 1
                        maximum_wall_period = max(maximum_wall_period,period)
                    last_step_wall=now
                    global_tick+=1; source_elapsed+=.02; wall_deadline+=.02
                    delay=wall_deadline-time.monotonic()
                    if delay>0:
                        time.sleep(delay)
                except Exception as error:
                    reject_and_pause(f'{type(error).__name__}: {error}')
                    wall_deadline=None; last_step_wall=None
                    if replay:
                        raise
            else:
                wall_deadline=None; last_step_wall=None
                time.sleep(.01)
            feedback = None
            feedback_error = None
            if frame is not None:
                try:
                    feedback = hand_feedback(frame,
                        {name:float(data.joint(name).qpos[0]) for name in MODEL_HAND_NAMES},
                        hands,following=following)
                except (ValueError, TypeError) as invalid_feedback:
                    # Tracking/physics guards already reject invalid control.
                    # Invalid telemetry must not shut down the operator UI.
                    feedback_error = str(invalid_feedback)
            preview_rgb = obs.get('head_stereo_left')
            preview_time = (id(model),float(obs.get('sim_time_s',data.time)))
            if preview_rgb is not None:
                if last_preview_time == preview_time:
                    preview_rgb = None
                else:
                    last_preview_time = preview_time
            ui.update({'status':'following' if following else 'paused','source_kind':result['source_kind'],
                'tracking':quality,'calibrated':calibration_fresh,'error':error_text,
                'hand_control':feedback,'hand_feedback_error':feedback_error,
                'recording':episode.status if episode else None,'episode_count':len(result['episodes']),
                'sim_time_s':float(data.time),'decoder_calls':controller.decoder_calls,
                'maximum_lag_s':max_lateness,'output':str(output),'accepted_new_dataset':False,
                'render_engine':result['render_engine'],
                'headset_video':{'enabled':True,**video_bridge.status} if video_bridge else {'enabled':False},
                'note':('MuJoCo实时预览；保存后待RTX同状态补图及独立验收，尚非训练数据' if deferred
                        else '保存后仍需独立物理、图像和数据验收；不会自动标为成功数据')},preview_rgb)
            if time.monotonic()-last_result_write_wall >= 1.:
                if not following and frame is not None:
                    # One bounded latest packet from this same SDK consumer.
                    # No second diagnostic client may steal its subscription.
                    write_json(output/'latest_pico_input.json',{
                        'source_sequence':frame.sequence,'device_timestamp_ns':frame.device_timestamp_ns,
                        'input_mode':frame.input_mode,'body_joint_count':
                            len(frame.body24x7) if frame.body24x7 is not None else 0,
                        'raw_packet':frame.raw_payload})
                result.update(status='following' if following else 'paused',decoder_calls=controller.decoder_calls,
                    simulated_control_ticks=global_tick,maximum_lag_s=max_lateness,last_error=error_text)
                # The full receipt grows with saved/rejected episodes. Serialize
                # it only while paused; real-time control writes a bounded status.
                if following:
                    write_json(output/'live_status.json',{
                        'status':'following','decoder_calls':controller.decoder_calls,
                        'simulated_control_ticks':global_tick,'maximum_lag_s':max_lateness,
                        'last_error':error_text,'hand_control':feedback,
                        'accepted_new_dataset':False})
                else:
                    write_json(output/'result.json',result)
                last_result_write_wall = time.monotonic()
        result['status']='stopped'
    except BaseException as error:
        result.update(status='error',error=repr(error),traceback=traceback.format_exc())
        raise
    finally:
        try:
            finalize(False,'session_ended_without_save' if episode else None)
        except Exception as finalization_error:
            result['episode_finalize_error'] = repr(finalization_error)
            result['status'] = 'error'
        # Isaac SimulationApp.close() can terminate the process. Persist all
        # final receipts and close writers before handing control to env.close.
        if render_profiler:
            result['render_profile'] = render_profiler.receipt()
        for name, component in (('receiver',receiver),('headset_video',video_bridge),
                                ('ui',ui),('control_trace',control_trace),('render_profiler',render_profiler)):
            if component:
                try:
                    component.close()
                except Exception as cleanup_error:
                    result.setdefault('cleanup_errors',{})[name] = repr(cleanup_error)
        periods=list(wall_periods)
        result['wall_timing']={'measured_inter_step_periods':wall_period_count,
            'distribution_scope':'last at most 3000 inter-step periods; maximum covers whole session',
            'distribution_sample_count':len(periods),
            'median_s':float(np.median(periods)) if periods else None,
            'p99_s':float(np.percentile(periods,99)) if periods else None,
            'maximum_s':maximum_wall_period if wall_period_count else None}
        if 'global_tick' in locals():
            result.update(simulated_control_ticks=global_tick,maximum_lag_s=max_lateness,
                decoder_calls=controller.decoder_calls,last_error=error_text)
        result['environment_close_requested'] = env is not None
        write_json(output/'result.json',result)
        if env:
            env.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    runtime=Path('/home/zxc/CVPR_Evaluation_runtime')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path,default=runtime/'resources/simple_local_usd_v1')
    parser.add_argument('--sonic-resources',type=Path,default=runtime/'resources/sonic_v1_1')
    parser.add_argument('--human-info',type=Path,default=GEAR/'data/human/human_joints_info.pkl')
    parser.add_argument('--render-mode',choices=('none','isaac','mujoco_deferred'),default='mujoco_deferred')
    parser.add_argument('--pico-replay',type=Path,help='Diagnostic only: recorded actual packets, no SDK')
    parser.add_argument('--diagnostic-record',action='store_true')
    parser.add_argument('--headset-video',action='store_true')
    parser.add_argument('--profile-render',action='store_true',help='Diagnostic API wall-time profiling without extra render calls')
    parser.add_argument('--disable-unrelated-viewport',action='store_true',help='Opt-in diagnostic: disable only a proven separate viewport render product')
    parser.add_argument('--environment-config',type=Path,help='Reuse a local saved Handover environment for scene-bound diagnostics')
    parser.add_argument('--pico-ip',default='172.16.20.83',help='Expected headset peer for video bridge')
    parser.add_argument('--seed',type=int,default=20261025)
    parser.add_argument('--port',type=int,default=43100)
    parser.add_argument('--seconds',type=float,default=0,help='Wall-clock limit after simulator warmup; 0 waits for operator')
    parser.add_argument('--max-lateness',type=float,default=.10)
    args=parser.parse_args()
    if not args.output.is_absolute() or args.seconds<0 or not 0<args.max_lateness<=.5:
        parser.error('Require absolute output, nonnegative seconds, and max lateness in (0,.5]')
    run(args)


if __name__=='__main__':
    main()
