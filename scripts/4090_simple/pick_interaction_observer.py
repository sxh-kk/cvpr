"""Passive solved-contact diagnostics; never forward or write the live model.

Contact forces describe the last solver interval. Pose snapshots are taken by
the worker after its existing synchronized control step, not manufactured here.
"""
from collections import Counter
from copy import deepcopy
import math

import numpy as np


def descendants(model, root):
    roots = {int(root)}
    for body in range(int(root)+1, model.nbody):
        if int(model.body_parentid[body]) in roots:
            roots.add(body)
    return roots


class PickInteractionObserver:
    def __init__(self, model, data, target_body, distractor_bodies=()):
        if data.model is not model:
            raise ValueError('Observer must bind the actual live model/data')
        self.model, self.data = model, data
        self.robot = descendants(model, model.body('pelvis').id)
        self.target = descendants(model, target_body)
        self.distractors = set().union(*(descendants(model, body) for body in distractor_bodies))
        self.table_geoms = {i for i in range(model.ngeom) if (model.geom(i).name or '').startswith('table')}
        self.phase = 'native_reset'
        self.steps = 0
        self.counts = Counter()
        self.first_events = {}
        self.pending = []
        self.last_time = float(data.time)

    def after_step(self, model, data):
        import mujoco
        if model is not self.model or data is not self.data:
            raise ValueError('Interaction observer model/data changed')
        stamp, dt = float(data.time), float(model.opt.timestep)
        if not math.isclose(stamp-self.last_time, dt, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Interaction observer missed a real physical step')
        self.last_time = stamp
        self.steps += 1
        events = []
        for index in range(data.ncon):
            contact = data.contact[index]
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(model.geom_bodyid[g]) for g in geoms)
            groups = []
            if any(b in self.robot for b in bodies):
                if any(b in self.target for b in bodies):groups.append('robot_target')
                if any(g in self.table_geoms for g in geoms):groups.append('robot_table')
                if any(b in self.distractors for b in bodies):groups.append('robot_distractor')
                if all(b in self.robot for b in bodies):groups.append('robot_self')
            if any(b in self.target for b in bodies):
                if any(g in self.table_geoms for g in geoms):groups.append('target_table')
                if any(b in self.distractors for b in bodies):groups.append('target_distractor')
            if not groups:
                continue
            wrench = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, index, wrench)
            if not np.isfinite(wrench).all():
                raise ValueError('Nonfinite measured contact force')
            if float(np.linalg.norm(wrench[:3])) <= 1e-6:
                continue
            event = dict(categories=groups, body_names=[model.body(b).name for b in bodies],
                geom_names=[model.geom(g).name for g in geoms],
                position_m=np.asarray(contact.pos).tolist(),distance_m=float(contact.dist),
                normal_force_N=float(wrench[0]),force_norm_N=float(np.linalg.norm(wrench[:3])))
            events.append(event)
            for group in groups:
                key = self.phase+':'+group
                self.counts[key] += 1
                if key not in self.first_events:
                    self.first_events[key] = dict(solved_interval_end_s=stamp,
                        solved_interval_start_s=stamp-dt,physics_substep=self.steps,**deepcopy(event))
        if events:
            self.pending.append(dict(physics_substep=self.steps,phase=self.phase,
                solved_interval_start_s=stamp-dt,solved_interval_end_s=stamp,contacts=events))

    def drain(self):
        events, self.pending = self.pending, []
        return events

    def receipt(self):
        return dict(contract='cvpr-pick-passive-interaction-observer-v1',
            observed_physics_substeps=self.steps,solved_interval_end_s=self.last_time,
            force_count_semantics='positive-force contact points per physical substep; not unique collision events',
            counts=dict(self.counts),first_contacts=deepcopy(self.first_events),
            live_state_writes=False,physics_steps_added=0,controller_commands_modified=False,
            contact_phase='query existing solved contacts after the real mj_step; no mj_forward')

    def completed_receipt(self, control_ticks):
        if type(control_ticks) is not int or control_ticks < 1:
            raise ValueError('Completed diagnostics require the actual positive control-step count')
        if self.steps != 1+4*control_ticks or not math.isclose(self.last_time,
                .005+.02*control_ticks,rel_tol=0,abs_tol=1e-8):
            raise ValueError('Incomplete interaction physical-step/time ledger')
        if not math.isclose(float(self.data.time),self.last_time,rel_tol=0,abs_tol=1e-9):
            raise ValueError('Unobserved physical step after interaction ledger')
        result=self.receipt()
        result['control_ticks']=control_ticks
        result['status']='complete'
        return result


def install_interaction_observer(raw):
    """Compose with existing wrappers, create a new observer for each real reset."""
    original_update, original_step = raw.mujoco.update_layout, raw.mujoco.step
    raw._pick_interaction_observer = None
    def update(*args, **kwargs):
        raw._pick_interaction_observer = None
        result = original_update(*args, **kwargs)
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        actors = raw.task.state_dict()['layout']['actors']
        distractors = [model.body(actor['asset']['label']).id for role,actor in actors.items()
            if role != 'target' and 'asset' in actor]
        raw._pick_interaction_observer = PickInteractionObserver(model,data,
            int(raw.mujoco.mj_objects['target'].id),distractors)
        return result
    def step(*args, **kwargs):
        observer = raw._pick_interaction_observer
        if observer is None:
            raise ValueError('Physics preceded the diagnostic observer binding')
        result = original_step(*args, **kwargs)
        observer.after_step(raw.mujoco.mjModel,raw.mujoco.mjData)
        return result
    raw.mujoco.update_layout, raw.mujoco.step = update, step


def measured_control_snapshot(raw, command, observation):
    """Read existing post-step authoritative poses, actual qpos and sent targets."""
    data, model = raw.mujoco.mjData, raw.mujoco.mjModel
    names = ('pelvis','left_wrist_yaw_link','right_wrist_yaw_link')
    bodies = {name:dict(xyz_m=data.xpos[model.body(name).id].tolist(),
        quaternion_wxyz=data.xquat[model.body(name).id].tolist()) for name in names}
    body = command['body_token']
    return dict(sim_time_s=float(data.time),bodies=bodies,
        target_xyz_m=data.xpos[int(raw.mujoco.mj_objects['target'].id)].tolist(),
        joint_names=list(observation['joint_names']),joint_qpos=np.asarray(observation['joint_qpos']).tolist(),
        sent_body_token64=None if body is None else np.asarray(body).tolist(),
        sent_hand_targets=dict(command['hand_targets'] or {}))
