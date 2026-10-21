"""Named-joint MuJoCo adapter for the official SONIC v1.1 ONNX decoder.

Decoder input and actuator math follow the pinned repository C++ deployment:
``gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/{src/g1_deploy_onnx_ref.cpp,
include/policy_parameters.hpp}``. This module does not import a second SONIC
checkout or overwrite SIMPLE's older WBC. The physics hook is called every
MuJoCo tick; decoder/proprioception history run at 50 Hz in simulator time.

Official artifact identity and compatibility with a dataset's latent field are
separate checks. Unverified dataset binding requires an explicit diagnostic
opt-in and may never be reported as a completed model evaluation.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path

import numpy as np

from simple.baselines.dual_source_psi0 import (
    BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES, simple_hand_targets,
)
from simple.core.action import ActionCmd


OFFICIAL_DECODER_SHA256 = "34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509"
OFFICIAL_OBSERVATION_SHA256 = "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0"
OFFICIAL_README_SHA256 = "0a69d0a418fddc2d96e9a70244876c32d40a44bee2b20c70c21747807c2972fd"
# For each policy/IsaacLab joint, index in BODY_JOINT_NAMES (MuJoCo/hardware).
POLICY_TO_BODY = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22,
                           4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
BODY_TO_POLICY = np.argsort(POLICY_TO_BODY)
DEFAULT_ANGLES = np.array([
    -.312, 0, 0, .669, -.363, 0, -.312, 0, 0, .669, -.363, 0,
    0, 0, 0, .2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0,
], dtype=np.float64)
_ARMATURE = {"5020": .003609725, "7520_14": .010177520,
             "7520_22": .025101925, "4010": .00425}
_EFFORT = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_MOTOR = (["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"] * 2 +
          ["7520_14", "5020", "5020"] + ["5020"] * 5 + ["4010"] * 2 +
          ["5020"] * 5 + ["4010"] * 2)
_OMEGA = 10 * 2 * 3.1415926535
_STIFFNESS = np.array([_ARMATURE[motor] * _OMEGA ** 2 for motor in _MOTOR])
_DAMPING = np.array([4 * _ARMATURE[motor] * _OMEGA for motor in _MOTOR])
_GAIN_MULTIPLIER = np.ones(29)
_GAIN_MULTIPLIER[[4, 5, 10, 11, 13, 14]] = 2
BODY_KP = _STIFFNESS * _GAIN_MULTIPLIER
BODY_KD = _DAMPING * _GAIN_MULTIPLIER
ACTION_SCALE = np.array([.25 * _EFFORT[motor] for motor in _MOTOR]) / _STIFFNESS


def decoder_input(latent64, history) -> np.ndarray:
    """Build official 994-vector from ten real chronological 50 Hz samples.

    A sample is (time, body angular velocity3, relative joint positions29 in
    policy order, joint velocities29 in policy order, previous raw policy
    action29, pelvis-frame gravity3). History is grouped by term, not by frame.
    """
    latent = np.asarray(latent64, dtype=np.float32)
    if latent.shape != (64,) or len(history) != 10:
        raise ValueError("SONIC v1.1 requires 64 latents and ten real proprioception samples")
    times = np.asarray([sample[0] for sample in history])
    if not np.isfinite(times).all() or not np.allclose(np.diff(times), .02, rtol=0, atol=.0001):
        raise ValueError("SONIC proprioception must be real chronological samples at 50 Hz")
    widths = (3, 29, 29, 29, 3)
    for sample in history:
        if len(sample) != 6 or any(np.asarray(value).shape != (width,) for value, width in zip(sample[1:], widths)):
            raise ValueError("Invalid SONIC proprioception sample layout")
    value = np.concatenate([latent] + [np.stack([sample[index] for sample in history]).reshape(-1)
                                      for index in range(1, 6)]).astype(np.float32)
    if value.shape != (994,) or not np.isfinite(value).all():
        raise ValueError("SONIC decoder input must be finite [64+30+290+290+290+30]")
    return value[None]


def body_targets(raw_policy_action) -> np.ndarray:
    value = np.asarray(raw_policy_action)
    if value.shape != (29,) or not np.isfinite(value).all():
        raise ValueError("SONIC decoder must produce 29 finite raw policy-order joint actions")
    return DEFAULT_ANGLES + value[BODY_TO_POLICY] * ACTION_SCALE


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class SonicV11OnnxController:
    """Execute token64 with official 50 Hz proprioception and per-tick torque PD.

    ``compatibility_evidence`` is a local JSON recording independent dataset
    encoder binding. Without it only diagnostic opt-in is accepted. Artifact
    SHA verification always runs. Initial history warm-up uses the measured
    joint pose with official PD gains; it never invents ten proprioception
    frames or assumes a checkpoint-specific standing latent.
    """
    def __init__(self, resource_dir, *, compatibility_evidence=None,
                 allow_unverified_compatibility=False, providers=None,
                 pd_profile="fixed_cpp_baseline", hand_torque_profile="mjcf_limits",
                 sync_mujoco_observations=False):
        root = Path(resource_dir).resolve(strict=True)
        decoder_path = root / "model_decoder.onnx"
        if _file_sha256(decoder_path) != OFFICIAL_DECODER_SHA256:
            raise ValueError("SONIC v1.1 decoder SHA differs from the fixed official artifact")
        if _file_sha256(root / "observation_config.yaml") != OFFICIAL_OBSERVATION_SHA256:
            raise ValueError("SONIC v1.1 observation configuration identity mismatch")
        source = json.loads((root / "PROVENANCE.json").read_text())
        if source.get("revision") != "6733128a3d8a523b1418b06bca3cdf61c8b0987f":
            raise ValueError("Unexpected SONIC official provenance revision")
        verified = False
        evidence_path = None
        if compatibility_evidence is not None:
            evidence_path = Path(compatibility_evidence).resolve(strict=True)
            evidence = json.loads(evidence_path.read_text())
            verified = (evidence.get("body_token_version") == "body_token_v1_1" and
                        evidence.get("decoder_sha256") == OFFICIAL_DECODER_SHA256 and
                        evidence.get("compatibility_verified") is True and
                        bool(evidence.get("evidence")))
            if not verified:
                raise ValueError("Dataset/encoder compatibility evidence is incomplete or mismatched")
        if not verified and not allow_unverified_compatibility:
            raise ValueError("Dataset body_token_v1_1 encoder binding remains unverified; use diagnostic opt-in only")
        self.identity = {
            "body_token_version": "body_token_v1_1", "decoder_sha256": OFFICIAL_DECODER_SHA256,
            "provenance": str(root / "PROVENANCE.json"),
            "compatibility_status": "verified" if verified else "unverified_diagnostic_only",
            "compatibility_evidence": str(evidence_path) if evidence_path else "",
        }
        self.body_kp, self.body_kd = BODY_KP.copy(), BODY_KD.copy()
        if pd_profile == "official_v1_1_ankle_pitch_1_5":
            if _file_sha256(root / "README.md") != OFFICIAL_README_SHA256:
                raise ValueError("Official v1.1 gain tuning requires the fixed README identity")
            self.body_kp[[4, 10]] *= 1.5
            self.body_kd[[4, 10]] *= 1.5
            self.identity["pd_profile_source_sha256"] = OFFICIAL_README_SHA256
        elif pd_profile != "fixed_cpp_baseline":
            raise ValueError("Unknown SONIC PD profile")
        self.identity.update(pd_profile=pd_profile, body_kp=self.body_kp.tolist(), body_kd=self.body_kd.tolist())
        self.sync_mujoco_observations = bool(sync_mujoco_observations)
        self.identity["observation_phase"] = ("post_step_mj_forward_same_qpos" if self.sync_mujoco_observations
                                               else "legacy_mj_step_derived_phase")
        if hand_torque_profile not in ("mjcf_limits", "recording_0p7_nm"):
            raise ValueError("Unknown hand torque profile")
        self.hand_torque_profile = hand_torque_profile
        self.identity["hand_torque_profile"] = hand_torque_profile
        if hand_torque_profile == "recording_0p7_nm":
            self.identity["hand_torque_limit_nm"] = {name: 2.45 if "thumb_0" in name else .7 for name in SIMPLE_HAND_JOINT_NAMES}
            self.identity["hand_torque_source"] = "static fixed SONIC model12 YAML/deploy MJCF cap comparison; original per-episode teleop runtime profile is not stamped; diagnostic opt-in only"
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(decoder_path), sess_options=options,
                                           providers=providers or ["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or list(inputs[0].shape) != [1, 994] or inputs[0].type != "tensor(float)":
            raise ValueError("Unexpected official decoder input contract")
        if len(outputs) != 1 or list(outputs[0].shape) != [1, 29]:
            raise ValueError("Unexpected official decoder output contract")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.reset()

    def reset(self):
        self._history = deque(maxlen=60)
        self._last_tick = None
        self._last_time = None
        self._last_raw_action = np.zeros(29, dtype=np.float32)
        self._body_target = None
        self.decoder_calls = 0
        self.clipped_torque_elements = 0

    def hold(self, observation):
        state = dict(zip(observation["joint_names"], observation["joint_qpos"]))
        return self._command(None, {name: float(state[name]) for name in SIMPLE_HAND_JOINT_NAMES})

    def step(self, body_token, hand_targets, observation):
        simple_hand_targets(hand_targets)  # validate named hand action contract
        token = np.asarray(body_token, dtype=np.float32)
        if token.shape != (64,) or not np.isfinite(token).all():
            raise ValueError("Expected 64 finite v1.1 latents")
        return self._command(token.copy(), dict(hand_targets))

    def _command(self, body_token, hand_targets):
        return ActionCmd("sonic_v1_1", controller=self, body_token=body_token, hand_targets=hand_targets)

    def apply_physics(self, robot, body_token, hand_targets):
        """Called by G1Sonic.apply_action at each physical substep (normally 200Hz)."""
        import mujoco
        model, data = robot.mjModel, robot.mjData
        t, dt = float(data.time), float(model.opt.timestep)
        if not 0 < dt <= .02 or abs(.02 / dt - round(.02 / dt)) > 1e-6:
            raise ValueError("SONIC physics timestep must divide the official 0.02-second control period")
        if self._last_time is not None and t < self._last_time - 1e-8:
            raise ValueError("Physics time moved backwards; reset the SONIC controller")
        self._last_time = t
        if (self.sync_mujoco_observations and
                (self._last_tick is None or t - self._last_tick >= .02 - 1e-8)):
            # Body orientation/angular velocity must describe the same integrated
            # qpos/qvel as joint proprioception at each real 50 Hz control tick.
            mujoco.mj_forward(model, data)
        body_q = np.asarray([data.joint(name).qpos[0] for name in BODY_JOINT_NAMES])
        body_dq = np.asarray([data.joint(name).qvel[0] for name in BODY_JOINT_NAMES])
        if self._body_target is None:
            self._body_target = body_q.copy()
            # Warm-up applies a measured-pose PD target, which may differ from
            # the official default stance. Record that actual previous target
            # in policy coordinates instead of claiming a zero residual.
            self._last_raw_action = np.asarray(
                ((self._body_target - DEFAULT_ANGLES) / ACTION_SCALE)[POLICY_TO_BODY], dtype=np.float32)
        if self._last_tick is None or t - self._last_tick >= .02 - 1e-8:
            if self._last_tick is not None and t - self._last_tick > .02 + 1e-6:
                raise ValueError("Missed SONIC 50 Hz physical state sample; refusing padded history")
            pelvis = model.body("pelvis").id
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, pelvis, velocity, 1)
            gravity = data.xmat[pelvis].reshape(3, 3).T @ np.array([0., 0., -1.])
            self._history.append((t, velocity[:3].copy(),
                                  (body_q - DEFAULT_ANGLES)[POLICY_TO_BODY],
                                  body_dq[POLICY_TO_BODY], self._last_raw_action.copy(), gravity))
            self._last_tick = t
            if body_token is not None:
                if len(self._history) < 10:
                    raise ValueError("Need ten real 50 Hz proprioception frames before SONIC latent control")
                actor_obs = decoder_input(body_token, list(self._history)[-10:])
                raw_action = self.session.run([self.output_name], {self.input_name: actor_obs})[0]
                self._body_target = body_targets(np.asarray(raw_action)[0])
                self._last_raw_action = np.asarray(raw_action[0], dtype=np.float32).copy()
                self.decoder_calls += 1
        # G1Sonic body and hand actuator indices may differ from both training
        # and policy order. Resolve their actual transmission joint by name.
        targets = dict(zip(BODY_JOINT_NAMES, self._body_target))
        targets.update(hand_targets)
        gains = dict(zip(BODY_JOINT_NAMES, zip(self.body_kp, self.body_kd)))
        gains.update({name: (5.0 if "thumb" in name else 2.5, 1.0) for name in SIMPLE_HAND_JOINT_NAMES})
        joint_to_actuator = {}
        for aid in range(model.nu):
            if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
                jid = int(model.actuator_trnid[aid, 0])
                name = model.joint(jid).name
                if name in targets:
                    if name in joint_to_actuator or not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
                        raise ValueError("SONIC adapter requires one unit-gear torque motor per named joint")
                    joint_to_actuator[name] = aid
        if set(joint_to_actuator) != set(targets):
            raise ValueError("SIMPLE model does not expose all 43 named SONIC torque actuators")
        for name, target in targets.items():
            aid = joint_to_actuator[name]
            joint = data.joint(name)
            kp, kd = gains[name]
            torque = kp * (target - joint.qpos[0]) - kd * joint.qvel[0]
            if not np.isfinite(torque):
                raise ValueError("Nonfinite SONIC PD torque")
            lower, upper = model.actuator_ctrlrange[aid] if model.actuator_ctrllimited[aid] else (-np.inf, np.inf)
            if self.hand_torque_profile == "recording_0p7_nm" and name in SIMPLE_HAND_JOINT_NAMES:
                cap = 2.45 if "thumb_0" in name else .7
                lower, upper = max(lower, -cap), min(upper, cap)
            if np.isfinite(lower) or np.isfinite(upper):
                bounded = np.clip(torque, lower, upper)
                self.clipped_torque_elements += int(bounded != torque)
                torque = bounded
            data.ctrl[aid] = torque
