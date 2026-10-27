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
from pico_mobile_task import (task_identity, collection_buttons, require_mobile_transition,
    require_manipulation_pose_match, mobile_reference_diagnostics,
    MobileStoppedWindow, MobilePreparationReference)
from pico_cpu_runtime import configure_cpu_runtime
from pico_capture_start import CaptureStartGate
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
        return replace(self.frames[index], host_received_s=float(time.monotonic()-(elapsed-self.times[index])))


def root_pose(model, data):
    body = model.body('pelvis').id
    return np.r_[data.xpos[body], data.xquat[body]].copy()


def actual_root_velocity(model, data):
    body = model.body('pelvis').id
    joint = int(model.body_jntadr[body])
    dof = int(model.jnt_dofadr[joint])
    return np.asarray(data.qvel[dof:dof+6]).copy()


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
             'pico_hand_control.py','pico_episode_history.py','pico_episode_review.py',
             'pico_cpu_runtime.py','pico_capture_start.py','pico_async_preview.py',
             'pico_mobile_task.py')
    directory = Path(__file__).resolve().parent
    return {name:file_sha256(directory/name) for name in names}


EPISODE_SESSION_FIELDS = (
    'task_id','source_kind','accepted_new_dataset','physics_hz','decoder_hz','command_rgb_hz',
    'absolute_human_root_translation_encoded','render_mode','raw_pico_va_wva_used','resource_data_root',
    'preview_engine','training_rgb_engine','preview_is_training_rgb','render_engine',
    'calibration_source_asset_sha256','source_code_sha256_at_session_start',
    'environment_config_source','replay_source_sha256','controller_identity','encoder_identity',
    'hand_control_profile','egl_backend_selection','actual_mujoco_gl_module',
    'review_mode','review_collection_enabled','cpu_runtime','preview_protocol',
    'capture_start_protocol','live_status_transport','collection_task_profile',
    'mobile_control_protocol','mobile_planner_model_sha256','actual_task_source_sha256',
    'actual_native_success_criteria','mobile_cold_start_profile')


def episode_session_identity(session_result):
    """Snapshot only fixed identities, never the growing session history."""
    required = ('task_id','source_kind','accepted_new_dataset','physics_hz','decoder_hz',
        'command_rgb_hz','render_mode','raw_pico_va_wva_used','preview_engine',
        'training_rgb_engine','preview_is_training_rgb','calibration_source_asset_sha256',
        'source_code_sha256_at_session_start','replay_source_sha256',
        'controller_identity','encoder_identity','hand_control_profile')
    missing = [name for name in required if name not in session_result]
    if missing:
        raise ValueError('Episode session identities are incomplete: '+','.join(missing))
    return copy.deepcopy({name:session_result[name] for name in EPISODE_SESSION_FIELDS if name in session_result})


def episode_scene_metadata(*, initialization, standing_warmup_readiness, warmup,
        environment_config, physics_seed, instruction, mujoco_object_body_names,
        object_body_names, mujoco_obj_names, container_geometry_cache_sha256,
        physics_model_sha256=None):
    """Prepare the current scene receipt while paused, before the next record tick."""
    values = {'initialization':initialization,'standing_warmup_readiness':standing_warmup_readiness,
        'warmup':warmup,'environment_config':environment_config,'physics_seed':physics_seed,
        'instruction':instruction,'mujoco_object_body_names':mujoco_object_body_names,
        'object_body_names':object_body_names,'mujoco_obj_names':mujoco_obj_names,
        'quality_geometry_mode':'deferred_geometry_with_original_solved_support',
        'container_geometry_cache_sha256':container_geometry_cache_sha256,
        'renderer_scene_config_scope':'initial after env.reset before controller warmup; dynamic frame poses come only from archived integration states'}
    if physics_model_sha256 is not None:
        values.update(physics_model_sha256=physics_model_sha256,
            physics_model_identity_scope='current scene prepared model used by deferred capture')
    return copy.deepcopy(values)


def bounded_episode_metadata(session_identity, scene_metadata, *, human_demo_uuid,
                             pico_calibration, initial_target_z_m):
    """A complete per-episode binding whose size does not grow with older attempts."""
    fixed = {name:session_identity[name] for name in EPISODE_SESSION_FIELDS if name in session_identity}
    return copy.deepcopy({**fixed,**scene_metadata,
        'episode_metadata_contract':'cvpr-pico-current-scene-metadata-v1',
        'human_demo_uuid':human_demo_uuid,'pico_calibration':pico_calibration,
        'hand_profile':session_identity['hand_control_profile']['profile_id'],
        'initial_target_z_m':initial_target_z_m})


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


def load_environment_config(path, data_root, *, profile='handover'):
    """Validate the selected task's saved layout and explicit local resources."""
    path, data_root = Path(path).resolve(strict=True), Path(data_root).resolve(strict=True)
    state = json.loads(path.read_text())
    _, expected_uid = task_identity(profile)
    if not isinstance(state,dict) or state.get('uid') != expected_uid:
        label = 'Handover task' if profile == 'handover' else expected_uid
        raise ValueError('Environment config must match the explicit collection '+label)
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
    return collection_buttons(buttons,last_buttons)


def paused_operator_message(reason, *, physics_ready):
    if not physics_ready:
        return '已暂停：物理状态检查未通过，请先重置场景，再按X标定、Y跟随。原因：'+str(reason)
    if 'Calibration requires valid live full-body tracking' in str(reason):
        return '已暂停：需要有效的手柄和全身追踪数据。请检查头显XRoboToolkit前台、Send和手柄模式，再按X标定。'
    return '已暂停；请查看原因，恢复后按X重新标定、Y跟随。原因：'+str(reason)


def recording_command_guard(action, *, episode_active, episode_status=None):
    """A calibration request cannot alter or reject an unfinished demonstration."""
    if action == 'calibrate' and episode_active:
        return '当前示例尚未结束，请先保存后再校准；已忽略本次校准请求，录制继续。'
    if action == 'save' and episode_active and episode_status in ('start_requested','warmup'):
        return '正在准备起录或记录一秒起始姿势，请稍候再保存；本次提前保存已忽略，录制继续。'
    return None


def preview_receipt(frame):
    return {'status':'ready','epoch':frame.epoch,'source_tick50':frame.source_tick50,
        'sim_time_s':frame.sim_time_s,'frame_id':frame.frame_id,'request_id':frame.request_id,
        'state_sha256':frame.state_sha256,'submitted_monotonic_s':frame.submitted_monotonic_s,
        'received_monotonic_s':frame.received_monotonic_s,'age_s':frame.receive_age_s,
        'render_wall_s':frame.render_wall_s,'training_observation':False,'max_age_s':.1,
        'delivery_scope':'collector poll only; no browser/headset viewing acknowledgement',
        'paused_render_request':frame.evidence.get('paused_render_request',False),
        'repeated_physical_source':frame.evidence.get('repeated_physical_source',False),
        'state_restoration':{key:frame.evidence.get(key) for key in
            ('full_integration_state_unchanged','physics_steps','forward_dynamics_calls','model_sha256')}}


def current_preview_feedback(receipt, *, following, now=None):
    feedback = copy.deepcopy(receipt or {'status':'waiting'})
    if receipt and following:
        feedback['age_s'] = (time.monotonic() if now is None else now)-receipt['submitted_monotonic_s']
        if feedback['age_s']>.1:
            feedback['status']='waiting'
    return feedback


def capture_preparation_interrupt(commands):
    """Act on operator stop intentions before the first command is committed."""
    for action in ('quit','discard','reset','pause','toggle_follow'):
        if action in commands:
            return action
    return None


class LoopWallTimer:
    """Bounded host timing only; never advances or changes the physics clock."""
    STAGES = ('input_validation','operator_actions','control','clock_sleep',
              'review_poll','preview_status','status_disk')

    def __init__(self, *, clock_ns=None, wall_clock_ns=None):
        self._clock = clock_ns or time.monotonic_ns
        self.start_ns = self._clock()
        self.start_wall_time_ns = (wall_clock_ns or time.time_ns)()
        self._stage = 'input_validation'
        self._stage_start_ns = self.start_ns
        self._durations_ns = {name:0 for name in self.STAGES}
        self._finished = None

    def enter(self, stage):
        if stage not in self.STAGES or self._finished is not None:
            raise ValueError('Invalid or already finished loop timing stage')
        now = self._clock()
        self._durations_ns[self._stage] += max(0,now-self._stage_start_ns)
        self._stage,self._stage_start_ns = stage,now

    def snapshot(self):
        if self._finished is not None:
            return copy.deepcopy(self._finished)
        now = self._clock()
        durations = dict(self._durations_ns)
        durations[self._stage] += max(0,now-self._stage_start_ns)
        timings = {name:value/1e9 for name,value in durations.items()}
        timings['total'] = max(0,now-self.start_ns)/1e9
        return {'complete':False,'loop_start_monotonic_ns':self.start_ns,
            'loop_start_pc_wall_time_ns':self.start_wall_time_ns,'timing_s':timings,
            'loop_critical_wall_s':max(0,timings['total']-timings['clock_sleep']),
            'tail_after_control_wall_s':sum(timings[name] for name in
                ('review_poll','preview_status','status_disk')),
            'active_stage_at_snapshot':self._stage}

    def finish(self):
        if self._finished is None:
            self._finished = self.snapshot()
            self._finished['complete'] = True
        return copy.deepcopy(self._finished)


def scene_bound_commands(commands, transition_started):
    """Never apply commands collected before a scene transition to its successor."""
    for action in commands:
        if transition_started():
            return
        yield action


def seal_episode_writers(closing_episode, closing_capture, closing_ledger, streams, *,
                         completed, reason, source_kind, capture_started, deferred):
    """Drain every raw writer before reset; leave all geometry and review to CPU worker."""
    started = time.perf_counter()
    outcome = {'accepted_new_dataset':False,'source_kind':source_kind,
        'capture_started':capture_started,'geometry_pending':bool(capture_started),
        'quality_geometry_mode':'deferred_geometry_with_original_solved_support'}
    errors,timing = {},{}
    try:
        if reason:
            closing_episode.reject(reason)
        state = closing_episode.close(completed=completed)
    except Exception as error:
        errors['recording_state'] = repr(error)
        state = {'pending_audit':False,'status':'rejected'}
    outcome['state'] = state
    capture_started_wall = time.perf_counter()
    try:
        outcome['capture'] = closing_capture.close(
            completed=completed and state.get('pending_audit',False)) if closing_capture else None
    except Exception as error:
        errors['capture'] = repr(error)
    timing['capture_close'] = time.perf_counter()-capture_started_wall
    streams_started = time.perf_counter()
    for index,stream in enumerate(streams):
        if stream:
            try:
                stream.close()
            except Exception as error:
                errors[f'quality_or_force_stream_{index}'] = repr(error)
    timing['quality_force_close'] = time.perf_counter()-streams_started
    ledger_started = time.perf_counter()
    try:
        outcome['ledger'] = closing_ledger.close() if closing_ledger else None
    except Exception as error:
        errors['ledger'] = repr(error)
    timing['ledger_close'] = time.perf_counter()-ledger_started
    if errors:
        outcome['finalization_errors'] = errors
        state.update(pending_audit=False,status='rejected')
    if deferred:
        pending = (completed and not errors and state.get('pending_audit')
            and (outcome.get('capture') or {}).get('status') == 'raw_physics_capture_completed_pending_rtx')
        outcome.update(status='pending_rtx' if pending else 'rejected',
            render_engine='deferred_rtx_pending',rtx_ready=False)
    timing['total'] = time.perf_counter()-started
    outcome['seal_wall_timing_s'] = timing
    return outcome


def episode_finalization_error(outcome, *, requested_complete):
    """Keep technical writer/history failures visible after an automatic reset."""
    problems = []
    if outcome.get('finalization_errors'):
        problems.append('写盘错误：'+json_text(outcome['finalization_errors']))
    capture = outcome.get('capture') or {}
    failure = capture.get('failure') or (capture.get('async_capture') or {}).get('failure')
    if failure:
        problems.append('采集写盘失败：'+str(failure))
    if requested_complete and capture and capture.get('status') == 'partial_capture_rejected' and not failure:
        problems.append('采集未完整写入，本条已拒收')
    collection = outcome.get('collection') or {}
    if collection.get('status') == 'error' or collection.get('error'):
        problems.append('归档记录失败：'+str(collection.get('error') or 'unknown error'))
    return '；'.join(problems) or None


def end_episode_and_reset(finalize_callback, reset_callback, *, discard):
    """Finish once, then reset once; never turn a writer failure into success."""
    outcome = finalize_callback(completed=not discard,reason='operator_discard' if discard else None)
    problem = episode_finalization_error(outcome,requested_complete=not discard)
    try:
        reset_callback()
    except Exception as error:
        prefix = (problem+'；') if problem else ''
        raise RuntimeError(prefix+'本条已结束，但自动重置失败：'+str(error)) from error
    return outcome,problem


def completed_episode_message(action, collection):
    number = collection.get('number')
    prefix = '本条' if number is None else '第'+str(number).zfill(3)+'条'
    if action == 'discard':
        message = prefix+'已标记丢弃'
    elif collection.get('pending') or collection.get('status') == 'pending':
        message = prefix+'原始证据已封存，后台正在物理复核和编号归档'
    elif collection.get('disposition') == 'kept':
        message = prefix+'已通过物理验收并归档，仍待图像和数据验收'
    elif collection.get('disposition') == 'discarded':
        reasons = '；'.join(map(str,collection.get('reasons') or []))
        message = prefix+'未通过物理验收，已保留原始证据并归档到 discarded'+('：'+reasons if reasons else '')
    else:
        message = prefix+'原始证据已保存，尚未成为合格训练数据'
    return message+'。新场景已准备，请站直后重新校准。'


def episode_has_started_capture(episode, output):
    return bool(episode is not None and episode.start_tick50 is not None and
        (Path(output)/'episodes'/episode.human_demo_uuid).is_dir())


def submit_finished_episode(review_queue, directory, *, capture_started, collection_enabled=True):
    """Submit only sealed captures with explicit human/diagnostic collection policy."""
    if review_queue is None or not capture_started:
        return None
    try:
        receipt=review_queue.submit(directory,collection_enabled=collection_enabled)
        if not isinstance(receipt,dict) or receipt.get('uuid') != Path(directory).name:
            raise ValueError('Background review submission returned a different episode identity')
        return receipt
    except Exception as error:
        return {'status':'error','available':False,'path':str(directory),
            'uuid':Path(directory).name,'pending':False,'error':repr(error)}


def discard_finished_episode(review_queue, latest_finished_uuid):
    """A pending review is still the latest attempt; never fall back to an older one."""
    if review_queue is None:
        return {'status':'disabled_diagnostic','available':False}
    if not latest_finished_uuid:
        return {'status':'empty','available':False}
    receipt=review_queue.request_discard(latest_finished_uuid)
    if not isinstance(receipt,dict) or receipt.get('uuid') != latest_finished_uuid:
        raise ValueError('Background discard returned a different episode identity')
    return receipt


def latest_review_status(snapshot, latest_finished_uuid, previous):
    """An old worker result cannot replace the latest finished UUID on the UI."""
    latest_finished_uuid = latest_finished_uuid or snapshot.get('latest_finished_uuid')
    if not latest_finished_uuid:
        return None,previous
    candidates = [snapshot.get('latest_episode'),snapshot.get('latest_completed')]
    candidates.extend(snapshot.get('latest_results') or [])
    for candidate in candidates:
        if not isinstance(candidate,dict) or candidate.get('uuid') != latest_finished_uuid:
            continue
        if (previous.get('uuid') == latest_finished_uuid
                and previous.get('status') == 'pending_discard'
                and candidate.get('status') != 'error'
                and not candidate.get('discarded') and candidate.get('disposition') != 'discarded'):
            return latest_finished_uuid,previous
        return latest_finished_uuid,copy.deepcopy(candidate)
    return latest_finished_uuid,previous


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


def load_mobile_cold_start(path, body_names):
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    q = np.asarray(value.get('body_q29'),dtype=float)
    if (value.get('contract') != 'cvpr-mobile-pp-cold-start-profile-v1'
            or value.get('body_joint_names') != list(body_names)
            or q.shape != (29,) or not np.isfinite(q).all()
            or value.get('hand_joint_names') != list(MODEL_HAND_NAMES)
            or value.get('hand_q14') != [0.]*14
            or value.get('before_first_physics_step_required') is not True
            or value.get('maximum_initialization_count') != 1
            or value.get('online_state_overwrite_allowed') is not False):
        raise ValueError('Invalid Mobile-only one-time cold start profile')
    return value,{'path':str(path),'sha256':file_sha256(path),'profile':value}


def install_mobile_cold_start(raw, cold_profile, body_names):
    """Initialize once before reset's implicit first physics step, then unhook."""
    original = raw.mujoco.step
    receipt = {}
    def first_step(*args, **kwargs):
        if receipt or float(raw.mujoco.mjData.time) != 0.:
            raise ValueError('Mobile cold start attempted after physical stepping')
        if cold_profile['body_joint_names'] != list(body_names):
            raise ValueError('Mobile cold start BODY29 order differs')
        for name,value in zip(body_names,cold_profile['body_q29']):
            low,high = raw.mujoco.mjModel.joint(name).range
            if not low <= value <= high:
                raise ValueError('Mobile cold start joint out of range: '+name)
        initialized = initialize_pose(raw,np.asarray(cold_profile['body_q29']),body_names)
        initialized.update(kind='one_time_mobile_profile_pose_before_first_physics_step',
            profile_id=cold_profile['profile_id'],initialization_time_s=0.,initialization_count=1,
            body_joint_names=list(body_names),body_q29=list(cold_profile['body_q29']))
        receipt.update(initialization=initialized)
        raw.mujoco.step = original
        return original(*args,**kwargs)
    raw.mujoco.step = first_step
    return receipt


def run(args):
    output = args.output.resolve()
    repo = Path(__file__).resolve().parents[2]
    if output.exists() or output.is_relative_to(repo):
        raise ValueError('Use a new output directory outside Git')
    if args.diagnostic_record and (not args.pico_replay or args.render_mode == 'none'):
        raise ValueError('Diagnostic capture requires recorded input and an explicit preview/capture mode')
    output.mkdir(parents=True)
    cpu_runtime = configure_cpu_runtime()
    resources = args.sonic_resources.resolve(strict=True)
    data_root, asset_shas = bound_calibration_assets(args.data_root, calibration_source_identities())
    profile = getattr(args,'task','handover')
    task_id,_ = task_identity(profile)
    mobile = profile == 'mobile_pp'
    if mobile and (not args.environment_config or args.render_mode != 'mujoco_deferred'):
        raise ValueError('Mobile P&P requires its saved official scene and deferred capture')
    environment_state = environment_receipt = None
    if args.environment_config:
        environment_state,environment_receipt = load_environment_config(args.environment_config,data_root,profile=profile)
    os.environ['SIMPLE_DATA_DIR'] = str(data_root)
    rendered = args.render_mode == 'isaac'
    deferred = args.render_mode == 'mujoco_deferred'
    replay = RecordedInput(args.pico_replay) if args.pico_replay else None
    result = {'status':'initializing','task_id':task_id,'collection_task_profile':profile,
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
    result.update(cpu_runtime=cpu_runtime.receipt,
        capture_start_protocol='cvpr-pico-frozen-capture-start-v1' if deferred else None,
        live_status_transport='localhost HTTP; periodic disk status only while paused')
    write_json(output/'result.json', result)
    env = ui = receiver = video_bridge = render_profiler = preview_worker = mobile_control = None
    mobile_mode = 'pose'
    mobile_tick = 0
    mobile_preparation = None
    mobile_entry_smpl = None
    mobile_stopped = MobileStoppedWindow()
    preview_epoch = -1
    preview_frame_receipt = None
    capture_start_receipt = None
    capture = ledger = None
    episode = None
    quality_stream = force_stream = control_trace = None
    pending_mobile_quality_row = None
    wall_periods = deque(maxlen=3000)
    wall_period_count = 0
    maximum_wall_period = 0.
    prepared_deferred = None
    deferred_prepare_index = 0
    geometry_cache_index = 0
    session_identity = scene_metadata = None
    review_queue = None
    latest_finished_uuid = None
    review_status = {'status':'disabled_diagnostic' if replay else 'initializing','pending_count':0}
    last_review_poll_wall = 0.
    history_status = {'status':'disabled_diagnostic' if replay else 'empty','available':False,
        'path':None,'uuid':None,'discarded':False,'error':None}
    operator_message = '请戴好头显、站直并校准，再开始下一条录制。'
    reference = PicoSonicReference(resources, args.human_info)
    if mobile:
        from pico_mobile_control import MobilePlannerControl, CONTRACT as MOBILE_CONTROL_CONTRACT
        from sonic_locomotion_planner import MODEL_SHA256
        if not getattr(args,'planner_model',None):
            raise ValueError('Mobile P&P requires the explicit pinned planner model')
        mobile_control = MobilePlannerControl(args.planner_model)
        result.update(mobile_control_protocol=MOBILE_CONTROL_CONTRACT,
            mobile_planner_model_sha256=MODEL_SHA256)
        for name in ('pico_mobile_control.py','pico_locomotion_reference.py',
                'sonic_locomotion_planner.py','pico_mobile_quality.py','start_pico_mobile_pp.sh'):
            result['source_code_sha256_at_session_start'][name]=file_sha256(Path(__file__).with_name(name))
    def operator_buttons(buttons, previous):
        return collection_buttons(buttons,previous,mobile=mobile)
    pairing = {'body_token_version':'body_token_v1_1','compatibility_verified':True,
        'decoder_sha256':'34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509',
        'evidence':{'scope':'Only new tokens generated here by fixed matching official encoder, not old dataset tokens',
            'encoder_identity':reference.identities}}
    write_json(output/'new_producer_binding.json', pairing)

    def current_scene_receipt(initialization, readiness):
        return episode_scene_metadata(initialization=initialization,
            standing_warmup_readiness=readiness,warmup=warmup.receipt,
            environment_config=renderer_environment_config,physics_seed=args.seed,
            instruction=raw.task.instruction,
            mujoco_object_body_names={role:obj.name for role,obj in raw.mujoco.mj_objects.items()},
            object_body_names={role:model.body(raw.mujoco.mj_objects[role].id).name
                for role in ('target','container')},mujoco_obj_names=list(raw.mujoco.obj_names),
            container_geometry_cache_sha256=prepared_geometry_cache['sha256'],
            physics_model_sha256=prepared_deferred.model_sha256 if deferred else None)

    def finalize(completed=False, reason=None):
        nonlocal capture, ledger, episode, quality_stream, force_stream, following
        nonlocal history_status, latest_finished_uuid, operator_message
        nonlocal capture_start_receipt
        if episode is None:
            return
        closing_episode, closing_capture, closing_ledger = episode, capture, ledger
        capture_started = episode_has_started_capture(closing_episode,output)
        following = False
        operator_message = '仿真已暂停，正在封存原始证据；物理复核将在后台独立进行。'
        if ui:
            ui.update({'status':'saving','recording':None,'accepted_new_dataset':False,
                'operator_message':operator_message,'last_episode':history_status,
                'note':'仅等待原始写入结束，随后重置；物理复核、编号归档在后台进行，仍需RTX补图和独立验收'})
        streams = (quality_stream, force_stream)
        # Detach first: a writer failure must not reuse or close an episode twice.
        capture = ledger = episode = quality_stream = force_stream = None
        outcome = seal_episode_writers(closing_episode,closing_capture,closing_ledger,streams,
            completed=completed,reason=reason,source_kind=result['source_kind'],
            capture_started=capture_started,deferred=deferred)
        if capture_started:
            outcome['quality_geometry_expected_cache_sha256'] = prepared_geometry_cache['sha256']
        directory = output/'episodes'/closing_episode.human_demo_uuid
        directory.mkdir(parents=True,exist_ok=True)
        if capture_start_receipt is not None:
            receipt_path = directory/'capture_start_receipt.json'
            write_json(receipt_path,capture_start_receipt)
            outcome['capture_start_receipt'] = {'path':receipt_path.name,
                'sha256':file_sha256(receipt_path),'contract':capture_start_receipt['contract']}
        capture_start_receipt = None
        write_json(directory/'episode.json', outcome)
        if review_queue is not None and capture_started:
            # The raw evidence is immutable from here. The independent worker
            # writes a separate review receipt, never this sealed episode file.
            latest_finished_uuid = closing_episode.human_demo_uuid
            history_status = submit_finished_episode(review_queue,directory,capture_started=capture_started,
                collection_enabled=not replay)
            outcome['collection'] = copy.deepcopy(history_status)
            operator_message = '原始证据已封存并提交后台复核；正在准备下一条的新场景。'
        result['episodes'].append({'directory':str(directory),**outcome})
        write_json(output/'result.json', result)
        return outcome

    try:
        from pico_episode_review import ReviewQueue
        review_options = {'spool_directory':output/'.review_diagnostic'} if replay else {}
        review_queue = ReviewQueue(output.parent/'pico_episode_history.json',**review_options).start()
        review_status = review_queue.snapshot()
        latest_finished_uuid,history_status = latest_review_status(
            review_status,latest_finished_uuid,history_status)
        result['review_mode'] = 'independent_cpu_worker_after_raw_writers_sealed'
        result['review_collection_enabled'] = not replay
        if not replay:
            result['episode_history_index'] = str(output.parent/'pico_episode_history.json')
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
        result['actual_task_source_sha256'] = file_sha256(__import__('inspect').getfile(type(raw.task)))
        if mobile:
            result['actual_native_success_criteria'] = float(raw.task.success_criteria)
        if environment_state:
            result['saved_scene_manager_binding'] = bind_saved_scene_manager(raw.task,environment_state)
        if args.render_mode == 'none':
            raw.task.sensor_cfgs = {}
            raw._render_frame = lambda: {}
        elif deferred:
            # Operator feedback is a private no-step renderer. The control
            # process still observes/archives every original physical state.
            raw._render_frame = lambda: {}
        elif rendered:
            raw._render_frame = lambda: raw.isaac.render()
        cold_profile = None
        if mobile:
            if not getattr(args,'mobile_cold_start',None):
                raise ValueError('Mobile P&P requires its verified cold-start profile')
            cold_profile,result['mobile_cold_start_profile'] = load_mobile_cold_start(
                args.mobile_cold_start,BODY_JOINT_NAMES)
            cold_receipt = install_mobile_cold_start(raw,cold_profile,BODY_JOINT_NAMES)
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
        result['initialization'] = (cold_receipt['initialization'] if mobile
            else initialize_pose(raw,DEFAULT_ANGLES,BODY_JOINT_NAMES))
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
            nonlocal substeps,pending_mobile_quality_row
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
                if mobile:
                    row['actual_root_xyz_m'] = root_pose(model,data)[:3].tolist()
                    if pending_mobile_quality_row is not None:
                        raise RuntimeError('Unpublished Mobile quality row from previous control step')
                    # Contacts/forces remain the original solved-state values.
                    # The native task flag is not computed until env.step
                    # returns; bind it then, rather than shifting it by a tick.
                    pending_mobile_quality_row = row
                else:
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
        if mobile_control:
            # Load and verify the separate planner while physics is paused,
            # before any receiver/recording or real-time deadline exists.
            mobile_control.boot(root_pose(model,data),
                physical_snapshot(model,data,BODY_JOINT_NAMES)['body_q29'])
            mobile_boot_deadline = time.monotonic()+70.
            while not mobile_control.poll()['ready']:
                if time.monotonic() >= mobile_boot_deadline:
                    raise RuntimeError('Mobile planner did not become ready before live collection')
                time.sleep(.005)
            result['mobile_planner_startup'] = mobile_control.status()
            mobile_control.pause('startup_prepared_waiting_for_operator')
        def configure_preview():
            nonlocal preview_worker,preview_epoch,preview_frame_receipt
            if not deferred:
                return
            from pico_async_preview import AsyncMujocoPreview, CONTRACT
            if preview_worker is None:
                cpus = sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None
                preview_worker = AsyncMujocoPreview(cpu_affinity=cpus[-2:-1] if cpus and len(cpus)>1 else cpus).start()
            preview_epoch += 1
            preview_frame_receipt = None
            if ui:
                ui.update({'status':'resetting','recording':None,
                    'operator_message':'正在准备新场景相机，完成后再开始录制。',
                    'preview_feedback':{'status':'waiting','epoch':preview_epoch}},clear_image=True)
            cameras = {'head_stereo_left':(640,360)}
            if args.headset_video:
                cameras['head_stereo_right'] = (640,360)
            status = preview_worker.configure(prepared_deferred.model_path,prepared_deferred.model_sha256,
                int(mujoco.mjtState.mjSTATE_INTEGRATION),
                mujoco.mj_stateSize(model,mujoco.mjtState.mjSTATE_INTEGRATION),
                cameras=cameras,epoch=preview_epoch)
            result['preview_protocol'] = {'contract':CONTRACT,'independent_process':True,
                'physics_steps':0,'forward_dynamics_calls':0,'latest_only_feedback':True,
                'training_observation':False,'cameras':cameras,
                'worker_policy':status.get('worker'),'maximum_frame_delivery_age_s':.1}
            initial_state = integration_state_snapshot(mujoco,model,data,float(data.time))
            preview_worker.submit_paused(initial_state,float(data.time),preview_epoch,0)
            # Initialization/reset is paused; require an actual camera frame
            # before offering the next episode to the operator.
            timeout = time.monotonic()+5.
            retry_at = time.monotonic()+.15
            while time.monotonic()<timeout:
                ready = preview_worker.poll()
                if ready is not None:
                    preview_frame_receipt = preview_receipt(ready)
                    ui.update({'status':'resetting','recording':None,
                        'preview_feedback':preview_frame_receipt},ready.images['head_stereo_left'])
                    if video_bridge:
                        video_bridge.update({**ready.images,'sim_time_s':ready.sim_time_s})
                    return
                worker_status = preview_worker.status()
                if worker_status.get('error'):
                    raise RuntimeError('Preview worker: '+worker_status['error'])
                if time.monotonic()>=retry_at:
                    preview_worker.submit_paused(initial_state,float(data.time),preview_epoch,0)
                    retry_at = time.monotonic()+.15
                time.sleep(.002)
            raise RuntimeError('相机预览未就绪，仿真保持暂停')

        if args.headset_video:
            from pico_teleop_ui import HeadsetVideoBridge
            video_bridge = HeadsetVideoBridge(expected_peer_ip=args.pico_ip).start()
            if 'head_stereo_left' in obs:
                video_bridge.update(obs)
        configure_preview()
        result.update(controller_identity=controller.identity,warmup=warmup.receipt,
            encoder_identity=reference.identities,status='waiting_for_calibration')
        session_identity = episode_session_identity(result)
        scene_metadata = current_scene_receipt(result['initialization'],result['standing_warmup_readiness'])
        if not replay:
            receiver = PicoReceiver().start()
        frame = previous = None
        last_buttons = {}
        pending_input_commands = []
        following = False
        error_text = None
        global_tick = 0
        last_command_timestamp_ns = None
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
        reset_generation = 0
        loop_timer = None
        last_loop_wall_timing = None

        def invalidate_calibration():
            nonlocal following, command, calibration_fresh, reference_evidence
            nonlocal mobile_mode,mobile_tick,mobile_preparation,mobile_entry_smpl
            nonlocal last_command_timestamp_ns
            following = False
            command = reference_evidence = None
            calibration_fresh = False
            reference.calibration = None
            reference.last_body = reference.last_timestamp_ns = None
            if mobile_control:
                mobile_control.pause('collector_paused')
            mobile_mode,mobile_tick = 'pose',0
            mobile_preparation = None
            mobile_entry_smpl = None
            last_command_timestamp_ns = None
            mobile_stopped.clear()

        def reject_and_pause(reason):
            # A bad episode or input cannot close the UI/receiver during live use.
            # A fresh calibration is required even for a short operator pause.
            nonlocal error_text, active_local, operator_message
            had_started_capture = not replay and episode_has_started_capture(episode,output)
            error_text = str(reason)
            operator_message = paused_operator_message(reason,physics_ready=physics_ready)
            active_local = None
            invalidate_calibration()
            result.setdefault('rejections',[]).append({'sim_time_s':float(data.time),
                'tick50':global_tick,'reason':error_text,'pc_wall_time_ns':time.time_ns(),
                'current_loop_wall_timing':loop_timer.snapshot() if loop_timer else None,
                'previous_loop_wall_timing':last_loop_wall_timing})
            result['rejections'] = result['rejections'][-128:]
            try:
                control_trace.write(json_text({'phase':'rejected_and_paused',
                    **result['rejections'][-1]})+'\n')
            except Exception as trace_error:
                result['trace_write_error'] = repr(trace_error)
            try:
                outcome = finalize(False,error_text)
            except Exception as finalization_error:
                result['episode_finalize_error'] = repr(finalization_error)
                error_text += '；保存失败：'+repr(finalization_error)
                return
            if had_started_capture:
                problem = episode_finalization_error(outcome,requested_complete=False)
                if problem:
                    error_text += '；'+problem
                try:
                    reset_scene(reason='after_rejected_capture')
                    operator_note = completed_episode_message('save',outcome.get('collection') or {})
                    operator_message=operator_note+' 原始拒收原因：'+str(reason)
                except Exception as reset_error:
                    # reset_scene already cleared readiness. Never recursively
                    # call this rejection path to retry a failed reset.
                    error_text += '；自动重置失败：'+str(reset_error)
                    operator_message='本条已结束，自动重置失败；仿真保持暂停，请查看错误。'
        def reset_scene(*, reason='operator_reset'):
            nonlocal physics_ready,reference,render_next,obs,renderer_environment_config
            nonlocal model,data,deferred_prepare_index,prepared_deferred,warmup_baseline,cache
            nonlocal geometry_cache_index,prepared_geometry_cache,substeps,global_tick,command
            nonlocal previous,last_preview_time,last_buttons,reference_evidence,pending_input_commands
            nonlocal wall_deadline,last_step_wall,hands,active_local,quality,operator_message
            nonlocal pending_mobile_quality_row
            pending_mobile_quality_row = None
            nonlocal reset_generation
            nonlocal scene_metadata
            if episode is not None:
                raise ValueError('Finish the current episode before resetting the scene')
            # Invalidate old UI/handset commands even if reset fails partway.
            reset_generation += 1
            pending_input_commands=[]
            operator_message = '仿真已暂停，正在重置新场景并预热控制器。'
            if ui:
                ui.update({'status':'resetting','recording':None,'accepted_new_dataset':False,
                    'operator_message':operator_message,'last_episode':history_status,'error':error_text})
            invalidate_calibration(); physics_ready = False
            reference = PicoSonicReference(resources,args.human_info)
            raw.mujoco.step = original_step
            render_next = True
            args.seed += 1
            if render_profiler:
                render_profiler.set_context('explicit_environment_reset',None,data.time)
            reset_options = {'state_dict':copy.deepcopy(environment_state),'dr_level':None} if environment_state else None
            if mobile:
                cold_receipt = install_mobile_cold_start(raw,cold_profile,BODY_JOINT_NAMES)
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
            reset_initialization = (cold_receipt['initialization'] if mobile
                else initialize_pose(raw,DEFAULT_ANGLES,BODY_JOINT_NAMES))
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
            result.setdefault('resets',[]).append({'reason':reason,'initialization':reset_initialization,
                'before':reset_before,'after':reset_after})
            write_json(output/'result.json',result)
            reset_readiness = require_standing_warmup(reset_before, reset_after)
            result['resets'][-1]['standing_warmup_readiness'] = reset_readiness
            cache = build_container_geometry_cache(model,data,container_body=raw.mujoco.mj_objects['container'].id,
                target_body=raw.mujoco.mj_objects['target'].id)
            geometry_cache_index += 1
            prepared_geometry_cache = prepare_quality_geometry(cache,output,geometry_cache_index)
            result['prepared_quality_geometry_caches'].append(prepared_geometry_cache)
            scene_metadata = current_scene_receipt(reset_initialization,reset_readiness)
            configure_preview()
            raw.mujoco.step = measured_step
            substeps=0; global_tick=0; command=None; previous=None
            last_preview_time=None
            physics_ready=True; last_buttons=dict(frame.buttons) if frame is not None else {}; reference_evidence=None
            pending_input_commands=[]
            wall_deadline=None; last_step_wall=None
            hands=None; active_local=None
            quality={'valid':False,'reasons':['waiting_for_tracking_after_reset']}
            if video_bridge and 'head_stereo_left' in obs:
                video_bridge.update(obs)
            operator_message = '新场景已准备，请站直后重新校准，再开始下一条录制。'

        while True:
            loop_timer = LoopWallTimer()
            reset_generation_at_loop_start = reset_generation
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
                        commands.extend(operator_buttons(frame.buttons,last_buttons))
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
            loop_timer.enter('operator_actions')
            reset_this_batch = reset_generation != reset_generation_at_loop_start
            for action in scene_bound_commands(commands,lambda:
                    reset_this_batch or reset_generation != reset_generation_at_loop_start):
                if action == 'toggle_follow':
                    action = 'pause' if following else 'follow'
                if action == 'toggle_record':
                    action = 'save' if episode else 'record'
                command_notice = recording_command_guard(action,episode_active=episode is not None,
                    episode_status=episode.status if episode else None)
                if command_notice:
                    operator_message = command_notice
                    continue
                try:
                    if action == 'calibrate':
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
                        operator_message='校准完成，可以开始跟随，再按录制键开始新示例。'
                        write_json(output/'last_calibration.json',reference.calibration)
                    elif action == 'follow':
                        if not calibration_fresh or not reference.calibration or not quality['valid'] or not physics_ready:
                            raise ValueError('Calibrate valid tracking before following')
                        require_neutral_calibration(reference,frame.body24x7)
                        require_standing_warmup(warmup_baseline,
                            physical_snapshot(model,data,BODY_JOINT_NAMES))
                        following = True; wall_deadline = None; command = None
                        error_text = None
                        operator_message='正在跟随；按录制键开始新示例。'
                    elif action == 'pause':
                        reject_and_pause('operator_pause')
                    elif action == 'toggle_navigation':
                        if not mobile or not following or not quality['valid']:
                            raise ValueError('移动模式需要有效追踪、标定并开始跟随')
                        if not mobile_stopped.ready or (episode and episode.status != 'recording'):
                            operator_message='请松开摇杆并连续站稳半秒后再按B；当前录制继续。'
                            continue
                        try:
                            require_mobile_transition(frame,actual_root_velocity(model,data),
                                recording_warmup=bool(episode and episode.status != 'recording'))
                        except ValueError as transition_error:
                            operator_message=str(transition_error)+'；当前录制继续。'
                            continue
                        snapshot = physical_snapshot(model,data,BODY_JOINT_NAMES)
                        return_pose_receipt = None
                        if mobile_mode == 'pose':
                            mobile_entry_smpl,_,_ = body24_to_smpl(frame.body24x7,
                                reference.rest_joints,reference.fk_parents)
                            mobile_preparation = MobilePreparationReference(root_pose(model,data),
                                snapshot['body_q29'],source_sequence=frame.sequence,
                                device_timestamp_ns=frame.device_timestamp_ns)
                            mobile_control.boot(root_pose(model,data),snapshot['body_q29'])
                            mobile_mode,mobile_tick = 'preparing_navigation',0
                            operator_message='正在准备移动模式，请松开摇杆；准备好后左杆向前、右杆转向。'
                        elif mobile_mode == 'navigation':
                            try:
                                return_human_readiness = require_neutral_calibration(reference,frame.body24x7)
                                candidate_smpl,_,_ = body24_to_smpl(frame.body24x7,
                                    reference.rest_joints,reference.fk_parents)
                                return_pose_receipt = require_manipulation_pose_match(mobile_entry_smpl,
                                    candidate_smpl)
                                return_pose_receipt['human_upright'] = return_human_readiness
                                return_pose_receipt['entry_source_sequence'] = mobile_preparation.source_sequence
                                return_pose_receipt['entry_device_timestamp_ns'] = mobile_preparation.device_timestamp_ns
                            except ValueError as pose_error:
                                operator_message=str(pose_error)+'；保持移动模式，当前录制继续。'
                                continue
                            mobile_control.pause('return_to_manipulation')
                            # Preserve the new robot heading after walking. The
                            # same live body/wrist calibration avoids snapping
                            # back to the pre-navigation world orientation.
                            reference.calibrate(frame.body24x7,root_pose(model,data),
                                {name:float(data.joint(name).qpos[0]) for name in WRIST_NAMES})
                            mobile_mode = 'pose'
                            mobile_preparation = None
                            mobile_entry_smpl = None
                            operator_message='已恢复手臂操作，可以放入第二张桌子的容器。'
                        else:
                            operator_message='移动模式正在准备，请等候；也可以按Y暂停。'
                            continue
                        control_trace.write(json_text({'phase':'mobile_mode_transition',
                            'tick50':global_tick,'mode':mobile_mode,
                            'actual_state':snapshot,'source_sequence':frame.sequence,
                            'source_device_timestamp_ns':frame.device_timestamp_ns,
                            'source_raw_packet_sha256':hashlib.sha256(json_text(frame.raw_payload).encode()).hexdigest(),
                            'return_pose_check':return_pose_receipt,
                            'calibration':reference.calibration})+'\n')
                    elif action == 'record':
                        if episode or not following or not (rendered or deferred):
                            raise ValueError('Recording needs active calibrated control and an explicit rendering/capture mode')
                        episode = RecordingState(); episode.request_start()
                        operator_message='正在准备录制；起录后先保持姿势一秒。'
                    elif action == 'save':
                        if not episode:
                            raise ValueError('No active episode')
                        reset_this_batch=True
                        episode.request_save()
                        outcome,problem = end_episode_and_reset(finalize,
                            lambda:reset_scene(reason='after_save'),discard=False)
                        error_text=problem
                        operator_message=completed_episode_message('save',outcome.get('collection') or {})
                        if problem:
                            operator_message='本条未正常完成：'+problem+'。新场景已重置，原始失败证据保留。'
                    elif action == 'discard':
                        if episode is not None:
                            reset_this_batch=True
                            outcome,problem = end_episode_and_reset(finalize,
                                lambda:reset_scene(reason='after_active_discard'),discard=True)
                            error_text=problem
                            operator_message=completed_episode_message('discard',outcome.get('collection') or {})
                            if problem:
                                operator_message+=' 写盘或归档问题：'+problem
                        elif replay or review_queue is None:
                            operator_message='诊断输入不操作真人历史记录；当前场景未重置。'
                        else:
                            if following:
                                invalidate_calibration()
                            history_status=discard_finished_episode(review_queue,latest_finished_uuid)
                            if history_status.get('status') == 'error':
                                raise RuntimeError('提交丢弃意图失败：'+str(history_status.get('error') or 'unknown error'))
                            error_text=None
                            if history_status.get('status') == 'empty':
                                operator_message='目前没有可丢弃的上一条记录；当前场景未重置。'
                            else:
                                label='第'+str(history_status.get('number')).zfill(3)+'条' if history_status.get('number') is not None else '上一条记录'
                                operator_message=label+'已提交后台丢弃请求，原始证据保留；当前场景未重置。'
                            for item in reversed(result['episodes']):
                                if (item.get('collection') or {}).get('uuid') == history_status.get('uuid') and history_status.get('uuid') is not None:
                                    item['collection']=copy.deepcopy(history_status)
                                    break
                    elif action == 'reset':
                        if not reset_this_batch:
                            reset_this_batch=True
                            reset_scene()
                            error_text=None
                    else:
                        raise ValueError('Unknown command: '+action)
                except Exception as error:
                    error_text = str(error)
                    if action in ('save','discard','reset'):
                        operator_message='操作未正常完成，仿真保持暂停；原始证据不会删除。'+error_text
                    if action in ('calibrate','follow','reset') or episode or following:
                        reject_and_pause(error_text)
                    if replay:
                        raise
            loop_timer.enter('control')
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
                        last_loop_wall_timing = loop_timer.finish()
                        result['last_loop_wall_timing'] = last_loop_wall_timing
                        continue
                    state = episode.on_tick(global_tick) if episode else None
                    if state and state['started']:
                        setup_started_wall = time.monotonic()
                        start_gate = CaptureStartGate(episode.human_demo_uuid,global_tick,
                            float(data.time),substeps,
                            integration_state_snapshot(mujoco,model,data,obs['sim_time_s']),frame) if deferred else None
                        directory = output/'episodes'/episode.human_demo_uuid
                        directory.mkdir(parents=True)
                        initial_z = float(raw.mujoco.mj_objects['target'].xpos[2])
                        metadata = bounded_episode_metadata(session_identity,scene_metadata,
                            human_demo_uuid=episode.human_demo_uuid,pico_calibration=reference.calibration,
                            initial_target_z_m=initial_z)
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
                        if start_gate is not None:
                            # Preparatory file I/O owns no physical samples.
                            # Revalidate the same receiver without resetting the
                            # encoder's last accepted pose/time or earlier lag.
                            if replay:
                                # Diagnostic source time advances through the
                                # frozen boundary; these skipped input packets
                                # are never represented as captured grid frames.
                                source_elapsed += time.monotonic()-setup_started_wall
                                after_frame = replay.at(source_elapsed)
                            else:
                                after_frame,_ = wait_fresh_command_frame(receiver,frame,
                                    frame.device_timestamp_ns,setup_started_wall+.20)
                            after_quality = validate_frame(after_frame,previous=frame)
                            setup_commands = ui.commands()+operator_buttons(after_frame.buttons,last_buttons)
                            last_buttons = dict(after_frame.buttons)
                            interrupt = capture_preparation_interrupt(setup_commands)
                            if interrupt == 'quit':
                                break
                            if interrupt is not None:
                                reason = 'operator_discard' if interrupt == 'discard' else 'operator_'+interrupt+'_during_preparation'
                                if interrupt == 'discard':
                                    episode.discard(reason)
                                reject_and_pause(reason)
                                # A reset request is still honored when the
                                # zero-frame raw attempt did not auto-reset.
                                if interrupt == 'reset' and reset_generation == reset_generation_at_loop_start:
                                    reset_scene(reason='operator_reset_during_preparation')
                                continue
                            pending_input_commands.extend(setup_commands)
                            wall_deadline,capture_start_receipt = start_gate.finish(wall_deadline,
                                after_global_tick=global_tick,after_sim_time=float(data.time),after_substeps=substeps,
                                after_state=integration_state_snapshot(mujoco,model,data,obs['sim_time_s']),
                                frame=after_frame,quality=after_quality,event_count=capture.count,
                                ledger_event_count=ledger._count,quality_bytes=quality_stream.tell(),
                                force_bytes=force_stream.tell(),max_lateness=args.max_lateness)
                            frame = previous = after_frame
                            quality = after_quality
                            lateness = time.monotonic()-wall_deadline
                    due = capture_tick_index(global_tick) is not None
                    control_tick_started = time.perf_counter()
                    encode_seconds = 0.
                    input_wait_seconds = 0.
                    actual_before = physical_snapshot(model,data,BODY_JOINT_NAMES)
                    new_command = due or command is None
                    if new_command:
                        if last_command_timestamp_ns == frame.device_timestamp_ns:
                            if replay:
                                raise ValueError('No new recorded PICO frame for a new30Hz command')
                            wait_started = time.perf_counter()
                            frame, quality = wait_fresh_command_frame(receiver, frame,
                                last_command_timestamp_ns, wall_deadline+args.max_lateness)
                            # A short button press in this inspected packet must
                            # survive even if the next callback is its release.
                            pending_input_commands.extend(operator_buttons(frame.buttons,last_buttons))
                            last_buttons = dict(frame.buttons)
                            previous = frame
                            input_wait_seconds = time.perf_counter()-wait_started
                    mobile_value = mobile_evidence = None
                    if mobile_mode == 'preparing_navigation':
                        preparing = mobile_control.poll()
                        from pico_mobile_control import neutral_controls
                        if preparing['ready'] and new_command and neutral_controls(frame):
                            mobile_mode = 'navigation'
                            mobile_tick = 0
                            operator_message='移动模式：左杆向前、右杆转向，松杆停步；上身保持进入时姿态，手指仍可控。站稳并恢复入场人体姿态后按B恢复手臂。'
                        else:
                            mobile_value,mobile_evidence = mobile_preparation.encoder_reference(root_pose(model,data))
                            if preparing['ready']:
                                operator_message='移动规划已就绪；请先松开两个摇杆，进入移动模式后再推杆。当前姿态保持、录制继续。'
                    if mobile_mode == 'navigation':
                        mobile_value,mobile_evidence = mobile_control.step(mobile_tick,
                            root_pose(model,data),actual_before['body_q29'],frame,
                            tracking_valid=quality['valid'])
                    if new_command:
                        encode_started = time.perf_counter()
                        if mobile_value is not None:
                            from pico_sonic_reference import ENCODER_SHA256
                            reference_range_diagnostics = mobile_reference_diagnostics(mobile_value,
                                BODY_JOINT_NAMES,[model.joint(name).range for name in BODY_JOINT_NAMES])
                            encoded = np.asarray(reference.session.run([reference.output_name],
                                {reference.input_name:mobile_value})[0],dtype=np.float32)
                            if encoded.shape != (1,64) or not np.isfinite(encoded).all():
                                raise ValueError('Invalid Mobile SONIC token')
                            token = encoded[0].copy()
                            evidence = dict(mobile_evidence,model_inference_performed=True,
                                encoder_sha256=ENCODER_SHA256,token_shape=[64],
                                reference_range_diagnostics=reference_range_diagnostics,
                                token_sha256=hashlib.sha256(token.tobytes()).hexdigest())
                        else:
                            token, evidence = reference.encode(frame.body24x7,root_pose(model,data),frame.device_timestamp_ns)
                        last_command_timestamp_ns = frame.device_timestamp_ns
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
                                body_token64=token,named_hand_targets=hands,reference_evidence=evidence,
                                operator_preview_feedback=current_preview_feedback(preview_frame_receipt,following=True) if deferred else None,
                                **({'capture_start_receipt':capture_start_receipt} if state['local_tick50']==0 and capture_start_receipt else {}))
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
                                preview_rgb=None,source_fields={
                                    'source_raw_packet':frame.raw_payload,
                                    'source_host_received_s':frame.host_received_s,
                                    'source_sequence':frame.sequence,'device_timestamp_ns':frame.device_timestamp_ns,
                                    'operator_preview_feedback':current_preview_feedback(preview_frame_receipt,following=True),
                                    'reference_evidence':reference_evidence,
                                    'observation_sim_time_s':obs['sim_time_s']})
                        else:
                            capture.enqueue(**capture_values,rgb=obs['head_stereo_left'],
                                renderer_pose_diagnostics=getattr(raw.isaac,'last_observation_pose_diagnostics',None),
                                renderer_authoritative_poses=actual_renderer_poses(model,data),observation_sim_time_s=obs['sim_time_s'])
                    if preview_worker and due:
                        preview_state = integration_state_snapshot(mujoco,model,data,obs['sim_time_s'])
                        preview_worker.submit(preview_state,float(data.time),preview_epoch,global_tick)
                    active_local = state['local_tick50'] if state and state['recording'] else None
                    render_next = capture_tick_index(global_tick+1) is not None
                    if render_profiler:
                        render_profiler.set_context('pico_follow',global_tick,data.time)
                    step_started = time.perf_counter()
                    obs,reward,terminated,truncated,info = env.step(command)
                    if pending_mobile_quality_row is not None:
                        pending_mobile_quality_row['official_task_success'] = bool(terminated)
                        quality_stream.write(json_text(pending_mobile_quality_row)+'\n')
                        pending_mobile_quality_row = None
                    if mobile_mode == 'navigation':
                        mobile_tick += 1
                    step_seconds = time.perf_counter()-step_started
                    active_local = None
                    control_trace.write(json_text({'phase':'pico_follow','tick50':global_tick,
                        'source_sequence':frame.sequence,'source_device_timestamp_ns':frame.device_timestamp_ns,
                        'source_inputs':dict(frame.inputs),'named_hand_targets':dict(hands),
                        'latest_delivered_preview':current_preview_feedback(preview_frame_receipt,following=True) if deferred else None,
                        'new_command30':new_command,'reference':reference_evidence if new_command else None,
                        'timing_s':{'reference_encode':encode_seconds,'environment_step':step_seconds,
                            'fresh_input_wait':input_wait_seconds,
                            'control_tick_before_trace':time.perf_counter()-control_tick_started},
                        'loop_wall_timing':{'loop_start_monotonic_ns':loop_timer.start_ns,
                            'loop_start_pc_wall_time_ns':loop_timer.start_wall_time_ns,
                            'pc_wall_time_ns':time.time_ns(),'lag_s':lateness,
                            'previous_complete_loop':last_loop_wall_timing},
                        'actual_before_step':actual_before,
                        'actual_after_step':physical_snapshot(model,data,BODY_JOINT_NAMES),
                        'terminated':bool(terminated),'truncated':bool(truncated)})+'\n')
                    finite = all(np.isfinite(x).all() for x in (data.qpos,data.qvel,data.qacc,data.ctrl))
                    pelvis=model.body('pelvis').id
                    tilt=float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3,3)[2,2],-1,1))))
                    if mobile:
                        mobile_stopped.observe(global_tick,frame,actual_root_velocity(model,data),
                            tilt_deg=tilt,tracking_valid=quality['valid'])
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
                    loop_timer.enter('clock_sleep')
                    if delay>0:
                        time.sleep(delay)
                except Exception as error:
                    reject_and_pause(f'{type(error).__name__}: {error}')
                    wall_deadline=None; last_step_wall=None
                    if replay:
                        raise
            else:
                wall_deadline=None; last_step_wall=None
                loop_timer.enter('clock_sleep')
                time.sleep(.01)
            # A bounded status file is the only worker observation in this
            # process: no history lock, geometry, hashes, or state-file loading.
            loop_timer.enter('review_poll')
            if review_queue and time.monotonic()-last_review_poll_wall >= .5:
                last_review_poll_wall=time.monotonic()
                try:
                    review_status=review_queue.snapshot()
                    latest_finished_uuid,history_status=latest_review_status(
                        review_status,latest_finished_uuid,history_status)
                except Exception as review_poll_error:
                    review_status={'status':'error','error':repr(review_poll_error),
                        'latest_finished_uuid':latest_finished_uuid}
                result['background_review']=copy.deepcopy(review_status)
            loop_timer.enter('preview_status')
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
            preview_feedback = None
            if preview_worker:
                ready = preview_worker.poll()
                if ready is not None:
                    preview_rgb = ready.images['head_stereo_left']
                    preview_time = (ready.epoch,ready.frame_id)
                    preview_frame_receipt = preview_receipt(ready)
                    if video_bridge:
                        video_bridge.update({**ready.images,'sim_time_s':ready.sim_time_s})
                preview_feedback = current_preview_feedback(preview_frame_receipt,following=following)
                worker_status = preview_worker.status()
                if worker_status.get('error'):
                    preview_feedback.update(status='error',error=worker_status['error'])
            if preview_rgb is not None:
                if last_preview_time == preview_time:
                    preview_rgb = None
                else:
                    last_preview_time = preview_time
            ui.update({'status':'following' if following else 'paused','source_kind':result['source_kind'],
                'task_id':task_id,'collection_task_profile':profile,'mobile_mode':mobile_mode,
                'mobile_planner':mobile_control.status() if mobile_control else None,
                'mobile_stopped_for_mode_transition':mobile_stopped.ready if mobile else None,
                'operator_message':operator_message,'last_episode':history_status,
                'background_review':review_status,
                'tracking':quality,'calibrated':calibration_fresh,'error':error_text,
                'hand_control':feedback,'hand_feedback_error':feedback_error,
                'recording':episode.status if episode else None,'episode_count':len(result['episodes']),
                'sim_time_s':float(data.time),'decoder_calls':controller.decoder_calls,
                'maximum_lag_s':max_lateness,'output':str(output),'accepted_new_dataset':False,
                'render_engine':result['render_engine'],
                'preview_feedback':preview_feedback,
                'headset_video':{'enabled':True,**video_bridge.status} if video_bridge else {'enabled':False},
                'note':('MuJoCo实时预览；保存后待RTX同状态补图及独立验收，尚非训练数据' if deferred
                        else '保存后仍需独立物理、图像和数据验收；不会自动标为成功数据')},preview_rgb)
            loop_timer.enter('status_disk')
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
                if not following:
                    write_json(output/'result.json',result)
                last_result_write_wall = time.monotonic()
            last_loop_wall_timing = loop_timer.finish()
            result['last_loop_wall_timing'] = last_loop_wall_timing
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
        if review_queue:
            try:
                review_queue.close(wait=False)
            except Exception as review_close_error:
                result.setdefault('cleanup_errors',{})['background_review'] = repr(review_close_error)
        for name, component in (('receiver',receiver),('mobile_planner',mobile_control),('headset_video',video_bridge),
                                ('preview_worker',preview_worker),('ui',ui),('control_trace',control_trace),('render_profiler',render_profiler)):
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
        cpu_runtime.close()
        if env:
            env.close()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    runtime=Path('/home/zxc/CVPR_Evaluation_runtime')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task',choices=('handover','mobile_pp'),default='handover')
    parser.add_argument('--planner-model',type=Path,help='Pinned official planner for Mobile P&P only')
    parser.add_argument('--mobile-cold-start',type=Path,help='Verified one-time initial BODY29 profile')
    parser.add_argument('--data-root',type=Path,default=runtime/'resources/simple_local_usd_v1')
    parser.add_argument('--sonic-resources',type=Path,default=runtime/'resources/sonic_v1_1')
    parser.add_argument('--human-info',type=Path,default=GEAR/'data/human/human_joints_info.pkl')
    parser.add_argument('--render-mode',choices=('none','isaac','mujoco_deferred'),default='mujoco_deferred')
    parser.add_argument('--pico-replay',type=Path,help='Diagnostic only: recorded actual packets, no SDK')
    parser.add_argument('--diagnostic-record',action='store_true')
    parser.add_argument('--headset-video',action='store_true')
    parser.add_argument('--profile-render',action='store_true',help='Diagnostic API wall-time profiling without extra render calls')
    parser.add_argument('--disable-unrelated-viewport',action='store_true',help='Opt-in diagnostic: disable only a proven separate viewport render product')
    parser.add_argument('--environment-config',type=Path,help='Reuse a local saved environment matching --task')
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
