"""Opt-in ready_upper_v1: native reset, neutral planner, fixed generic upper body.

This is a startup reference generator, not a task policy. It never reads PICO,
recorded demonstrations, reward, object poses, future observations or teacher
states. The caller owns the unchanged real50tick startup budget without changing the native
zero-joint reset or adding physical steps. No live state is written.
"""
from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import mujoco

from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES, state43_to45
from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
from handover_sonic_teleop_reference import (
    conjugate, multiply_wxyz, rotation, vr_points_from_body_poses,
)
from pico_locomotion_reference import OfflinePlannerReference
from sonic_locomotion_planner import MODEL_SHA256

PROFILE = 'ready_upper_v1'
BODY_NAMES = ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link')


class _NeutralMode1Warmup(LivePoseSonicWarmup):
    """First ten real samples use PD, then neutral mode1 through the same decoder.

    Desired wrist/head targets are frozen from the first actual FK in root-local
    coordinates and lifted by the current actual root on every subsequent tick.
    This mirrors an unchanged neutral VR target after calibration, without any
    PICO packet. Neutral planner forecasts are references, never measured history.
    """
    def __init__(self, resource_dir, planner_model, *, seed=20261027):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError('Nonnegative int64 neutral planner seed required')
        self.planner_model = Path(planner_model).resolve(strict=True)
        self.seed = seed
        self.reference = OfflinePlannerReference(self.planner_model, seed=seed)
        super().__init__(resource_dir, hand_policy='measured')

    def reset(self):
        super().reset()
        self.local_points = self.local_quaternions = None
        self.tick = 0
        self.first_time = None
        self.reference.active = None
        self.reference.cursor = 0
        self.reference.last_tick = self.reference.last_command = None
        self.reference.last_plan_tick = 0
        self.reference.next_check = 0
        self.receipt.update(kind=PROFILE, initialization_profile=PROFILE,
            helper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            encoder_mode=1, planner_model_path=str(self.planner_model), planner_model_sha256=MODEL_SHA256,
            planner_seed=self.seed, source_kind='current actual FK plus pinned neutral planner only',
            PICO_or_teacher_data_read=False, live_physical_state_writes=False,
            physics_steps_added=0, policy_phase_interventions=False,
            reference='frozen first actual root-local wrist-tip/head FK; neutral planner lower12',
            target_or_scene_or_reward_inputs_used=False, warmup_budget_control_ticks=50)

    def command(self, robot, controller, observation):
        state43_to45(observation['joint_qpos'], observation['joint_names'])
        t = float(observation['sim_time_s'])
        if not np.isfinite(t) or self.tick >= 50:
            raise ValueError('Canonical candidate is restricted to exactly the real50tick startup window')
        if self.first_time is None:
            self.first_time = t
        if not np.isclose(t, self.first_time + .02*self.tick, rtol=0, atol=1e-9):
            raise ValueError('Canonical warmup requires consecutive real50Hz observations')
        data = robot.mjData
        root = np.r_[data.body('pelvis').xpos, data.body('pelvis').xquat].astype(float)
        q = dict(zip(observation['joint_names'], observation['joint_qpos']))
        body = np.array([q[name] for name in BODY_JOINT_NAMES], float)
        if self.local_points is None:
            positions = np.array([data.body(name).xpos for name in BODY_NAMES], float)
            quaternions = np.array([data.body(name).xquat for name in BODY_NAMES], float)
            points, quaternions = vr_points_from_body_poses(positions, quaternions)
            self.local_points = (points-root[:3]) @ rotation(root[3:])
            self.local_quaternions = np.array([multiply_wxyz(conjugate(root[3:]), value) for value in quaternions])
            self.hands = {name:float(q[name]) for name in SIMPLE_HAND_JOINT_NAMES}
            self.receipt.update(initialization_count=1, first_sim_time_s=t,
                first_actual_root_pose7=root.tolist(), first_actual_body_q29=body.tolist(),
                frozen_root_local_points_m=self.local_points.tolist(),
                frozen_root_local_quaternions_wxyz=self.local_quaternions.tolist(),
                first_actual_measured_hands14=list(self.hands.values()))
        self.tick += 1
        if len(controller._history) < 10:
            self.phase = 'warmup_measured_pose_pd'
            return controller.hold(observation)
        if self.tick < 11:
            raise ValueError('Controller history predates canonical warmup; reset it before startup')
        points = self.local_points @ rotation(root[3:]).T + root[:3]
        quaternions = np.array([multiply_wxyz(root[3:], value) for value in self.local_quaternions])
        r = rotation(self.reference.alignment.initial_actual_pose7[3:]) if self.reference.active is not None else rotation(root[3:])
        facing = np.r_[r[:2, 0],0.];facing /= np.linalg.norm(facing)
        command = {'mode':0, 'movement_direction':[0.,0.,0.], 'facing_direction':facing.tolist(), 'speed':-1., 'height':-1.}
        value, evidence = self.reference.step(self.tick-11, root, body, command,
            vr_world_points=points, vr_world_quaternions=quaternions)
        token = np.asarray(self.session.run([self.output_name], {self.input_name:value})[0], np.float32)
        if token.shape != (1,64) or not np.isfinite(token).all():
            raise ValueError('Invalid canonical neutral mode1 token')
        self.phase = 'warmup_ready_upper_v1'
        self.receipt['encoder_calls'] += 1
        self.receipt['events'].append({'sim_time_s':t, 'actual_root_wxyz':root[3:].tolist(),
            'token64':token[0].tolist(), 'reference_evidence':evidence,
            'hand_targets14':list(self.hands.values()), 'state_forcing':False, 'history_injection':False})
        return controller.step(token[0], self.hands, observation)


# Universal preset fixed before the CPU screen, not fitted per scene.
READY_JOINTS={'left_elbow_joint':.6,'right_elbow_joint':.6,
              'left_shoulder_roll_joint':.3,'right_shoulder_roll_joint':-.3}


def blend(tick):
    if type(tick)is not int or not 0<=tick<50:
        raise ValueError('Exactly50 real control ticks required')
    x=float(np.clip((tick-10)/39,0.,1.))
    return x*x*(3.-2.*x)


class ReadyUpperSonicWarmup(_NeutralMode1Warmup):
    def reset(self):
        super().reset()
        self.private=None
        self.initial_named=None
        self.receipt.update(kind=PROFILE,initialization_profile=PROFILE,
            helper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            fixed_ready_joint_targets_rad=dict(READY_JOINTS),
            source_kind='native measured robot plus universal fixed ready upper reference and pinned neutral planner',
            reference='native initial measured joint reference, four universal upper targets; private FK smoothstep ticks10..49',
            private_fk=True,live_state_writes=False,object_pose_inputs_used=False,
            cold_DEFAULT_initialization=False,teacher_reference_inputs_used=False)

    def command(self,robot,controller,observation):
        # Populate the existing neutral mode1 generator's root-local three-point
        # targets using virtual robot FK before it performs its normal command.
        state43_to45(observation['joint_qpos'],observation['joint_names'])
        tick=self.tick
        alpha=blend(tick)
        model,data=robot.mjModel,robot.mjData
        if self.private is None:
            self.private=mujoco.MjData(model)
            self.initial_named=dict(zip(observation['joint_names'],map(float,observation['joint_qpos'])))
            self.receipt.update(initial_joint_names43=list(observation['joint_names']),
                initial_measured_joint_qpos43=list(self.initial_named.values()))
        if self.private.model is not model:
            raise ValueError('Candidate private FK model identity changed')
        # Copy only the actual robot floating root and named robot joints; scene
        # object coordinates are never consulted and live qpos is never written.
        self.private.qpos[:7]=data.qpos[:7]
        for name in (*BODY_JOINT_NAMES,*SIMPLE_HAND_JOINT_NAMES):
            q=self.initial_named[name]
            if name in READY_JOINTS:q=q+alpha*(READY_JOINTS[name]-q)
            self.private.joint(name).qpos[0]=q
        mujoco.mj_forward(model,self.private)
        root=np.r_[data.body('pelvis').xpos,data.body('pelvis').xquat].astype(float)
        positions=np.array([self.private.body(name).xpos for name in BODY_NAMES],float)
        quats=np.array([self.private.body(name).xquat for name in BODY_NAMES],float)
        points,quats=vr_points_from_body_poses(positions,quats)
        local=(points-root[:3])@rotation(root[3:])
        localq=np.array([multiply_wxyz(conjugate(root[3:]),q)for q in quats])
        # The parent initializes frozen hands/first measured receipt only once.
        if tick==0:
            result=super().command(robot,controller,observation)
            self.local_points,self.local_quaternions=local,localq
        else:
            self.local_points,self.local_quaternions=local,localq
            result=super().command(robot,controller,observation)
        self.receipt.setdefault('ready_reference_events',[]).append(dict(
            control_tick=tick,sim_time_s=float(observation['sim_time_s']),blend=alpha,
            virtual_ready_targets_rad={n:self.initial_named[n]+alpha*(v-self.initial_named[n])for n,v in READY_JOINTS.items()},
            local_points_m=local.tolist(),local_quaternions_wxyz=localq.tolist(),
            private_fk_only=True,live_qpos_writes=False))
        return result
