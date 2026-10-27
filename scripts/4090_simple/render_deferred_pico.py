"""Render archived real PICO physics states in RTX, without replaying dynamics.

The human saw MuJoCo EGL preview. These training RGB frames are generated later
from the exact archived integration states, not claimed to be live feedback.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import traceback
import inspect

import numpy as np

from pico_simple_teleop import (actual_renderer_poses, bind_saved_scene_manager,
    configure_unrelated_viewport, disable_instance_render_rate_limits,
    environment_binding_receipt, load_environment_config, root_pose)
from replay_handover_teacher import (TrainingCapture, bound_calibration_assets,
    calibration_source_identities, file_sha256, write_json)


def recorded_task(metadata):
    """Dispatch by recorded task identity; never render Mobile in a Handover room."""
    from pico_mobile_task import HANDOVER_TASK, MOBILE_TASK
    task = metadata.get('task_id', HANDOVER_TASK)
    if task not in (HANDOVER_TASK, MOBILE_TASK):
        raise ValueError('Unsupported recorded task identity: ' + str(task))
    if task == MOBILE_TASK:
        from pico_episode_history import MOBILE_OFFICIAL_SOURCE_SHA256, MOBILE_NATIVE_SUCCESS_CRITERIA
        criteria = metadata.get('actual_native_success_criteria')
        if (metadata.get('actual_task_source_sha256') != MOBILE_OFFICIAL_SOURCE_SHA256
                or type(criteria) not in (int,float) or criteria != MOBILE_NATIVE_SUCCESS_CRITERIA):
            raise ValueError('Recorded Mobile official source or success threshold identity differs')
    return task


def recorded_environment(path, data_root, task_id):
    from pico_mobile_task import MOBILE_TASK
    return load_environment_config(path, data_root,
        profile='mobile_pp' if task_id == MOBILE_TASK else 'handover')


class RecordedStateView:
    """Only the read interface used by Isaac's zero-time state synchronization."""
    def __init__(self, model, data, joint_names, role_bodies, object_names):
        self.mjModel, self.mjData = model, data
        self.joint_names = tuple(joint_names)
        self.mj_objects = {role:data.body(name) for role,name in role_bodies.items()}
        self.obj_names = list(object_names)
        if len(self.obj_names) != len(self.mj_objects):
            raise ValueError('Recorded object order does not match role bindings')
        self.render_step = 0
        self.articulated_objects = None

    def get_robot_qpos(self):
        return {name:float(self.mjData.joint(name).qpos[0]) for name in self.joint_names}

    def get_states(self):
        objects = list(self.mj_objects.values())
        return [self.render_step,self.get_robot_qpos(),self.obj_names,
            [body.xpos.copy() for body in objects],[body.xquat.copy() for body in objects],
            self.mjData.qpos[:7].copy()]


def require_equal(actual, expected, label, atol=1e-10):
    a,b=np.asarray(actual,dtype=np.float64),np.asarray(expected,dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.allclose(a,b,rtol=0,atol=atol):
        raise ValueError('Archived physics state differs from recorded event: '+label)


def restore_recorded_state(model, data, state, state_spec):
    """Restore the whole archive and refresh FK/cameras, without integration."""
    import mujoco
    state_before=np.asarray(state,dtype=np.float64).copy()
    mujoco.mj_setState(model,data,state_before,state_spec)
    mujoco.mj_kinematics(model,data)
    mujoco.mj_camlight(model,data)
    assert_recorded_state_unchanged(model,data,state_before,state_spec)
    return state_before


def assert_recorded_state_unchanged(model, data, state_before, state_spec):
    import mujoco
    state_after=np.empty_like(state_before)
    mujoco.mj_getState(model,data,state_after,state_spec)
    require_equal(state_after,state_before,'render changed full integration state',0.)


def canonical_event_sha256(event):
    """Stable digest of decoded JSON, independently recomputed by the auditor."""
    encoded=json.dumps(event,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def write_frame_bindings(path, source_events, capture_events_path):
    """Bind every zero-step state acquisition to its resulting RTX PNG event."""
    rendered=[json.loads(line) for line in Path(capture_events_path).read_text().splitlines()]
    if not source_events or len(rendered)!=len(source_events):
        raise ValueError('Deferred RTX and original physical frame counts differ')
    bindings=[]
    shared=('event_index','capture_index30','physical_tick50','sim_time_s','nominal_elapsed_s',
        'actual_elapsed_s','phase','action_valid','source_command_index','joint_names',
        'state43','physical_joint_q43_float64','physical_joint_qvel43','root_pose7',
        'root_qvel6','objects','action')
    for index,(source,event) in enumerate(zip(source_events,rendered)):
        if source['event_index']!=index or any(source[key]!=event[key] for key in shared):
            raise ValueError('RTX capture changed original physical/state/command fields at frame '+str(index))
        rgb=event['rgb']
        bindings.append({'event_index':index,'physical_tick50':source['physical_tick50'],
            'sim_time_s':source['sim_time_s'],'integration_state_sha256':source['integration_state_sha256'],
            'source_event_sha256':canonical_event_sha256(source),
            'rtx_event_sha256':canonical_event_sha256(event),
            'rgb_pixel_sha256':rgb['pixel_sha256'],'rgb_file_sha256':rgb['file_sha256'],
            'full_integration_state_unchanged':True})
    path=Path(path)
    with path.open('x') as stream:
        for binding in bindings:
            stream.write(json.dumps(binding,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    return {'frame_bindings_path':path.name,'frame_bindings_sha256':file_sha256(path),
        'frame_bindings_count':len(bindings),'full_integration_state_unchanged':True}


def render(args):
    from pico_deferred_capture import load_deferred
    episode=args.episode.resolve(strict=True)
    destination=episode/'capture'
    receipt_path=episode/'rtx_completion.json'
    if destination.exists() or receipt_path.exists():
        raise ValueError('RTX output already exists; preserve it and use a fresh episode copy for retry')
    source=load_deferred(episode/'deferred_capture',verify=True)
    manifest,events,states=source['manifest'],source['events'],source['states']
    metadata=manifest['metadata']
    task_id = recorded_task(metadata)
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
    result={'contract':'cvpr-pico-deferred-rtx-completion-v1','status':'initializing',
        'accepted_new_dataset':False,'eligible_for_packing':False,
        'source_kind':metadata['source_kind'],'observation_domain_feedback':'MuJoCo EGL preview',
        'training_rgb_domain':'Isaac RTX deferred','rgb_was_live_operator_feedback':False,
        'archived_physics_steps':0,'sonic_inferences':0,'completed_frames':0,
        'source_manifest_sha256':file_sha256(episode/'deferred_capture/capture.json'),
        'source_events_sha256':file_sha256(episode/'deferred_capture/events.jsonl'),
        'source_model_sha256':file_sha256(episode/'deferred_capture/model.mjb'),
        'source_states_sha256':file_sha256(episode/'deferred_capture/states.npy'),
        'task_id':task_id,
        'render_code_sha256':file_sha256(__file__)}
    env=capture=None
    try:
        import gymnasium as gym
        import mujoco
        import simple.envs
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from simple.core.action import ActionCmd
        model=source['model']; data=mujoco.MjData(model)
        view=RecordedStateView(model,data,source['joint_names'],metadata['mujoco_object_body_names'],metadata['mujoco_obj_names'])
        sonic=SimLoopConfig().load_wbc_yaml()
        sonic.update(ENV_NAME='simple',SIMULATE_DT=.005,ENABLE_ELASTIC_BAND=False)
        env=gym.make(task_id,sonic_config=sonic,
            sim_mode='mujoco_isaac',headless=True,dr_level=0,sync_mujoco_observations=True,
            isaac_observation_only=True,disable_env_checker=True)
        raw=env.unwrapped
        result['actual_task_source_sha256'] = file_sha256(inspect.getfile(type(raw.task)))
        if metadata.get('actual_task_source_sha256') is not None and result['actual_task_source_sha256'] != metadata['actual_task_source_sha256']:
            raise ValueError('Loaded RTX task module differs from the original capture')
        bind_saved_scene_manager(raw.task,environment)
        raw._render_frame=lambda:raw.isaac.render()
        env.reset(seed=metadata['physics_seed'],options={'state_dict':copy.deepcopy(environment),'dr_level':None})
        result['environment_binding']=environment_binding_receipt(environment,raw.task.state_dict())
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
            objects={role:{'pose7':np.r_[view.mj_objects[role].xpos,view.mj_objects[role].xquat]}
                for role in ('target','container')}
            for role,value in objects.items():
                require_equal(value['pose7'],event['objects'][role]['pose7'],role)
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--data-root',type=Path,default=Path('/home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1'))
    render(parser.parse_args())


if __name__=='__main__':
    main()
