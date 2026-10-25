"""Causal PICO body-pose reference for the fixed SONIC v1.1 SMPL encoder.

This module never initializes XRoboToolkit, launches a service, creates a
simulator, or uses PICO's invalid ``va/wva`` fields.  Its ten reference frames
are an explicitly held *desired pose*, not ten measured future observations.
The decoder must still use its own real simulator proprioception history.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from handover_sonic_teleop_reference import (
    compute_apply_delta_heading, heading_wxyz, multiply_wxyz,
    verify_official_sources,
)


CONTRACT = "cvpr-pico-sonic11-smpl-held-reference-v1"
REPO = Path(__file__).resolve().parents[2]
GEAR = REPO / "VLMExpert/third_party/SIMPLE/third_party/gear_sonic"
ENCODER_SHA256 = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"
OBSERVATION_SHA256 = "4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0"
HUMAN_INFO_SHA256 = "4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975"
SOURCE_SHA256 = {
    "scripts/pico_manager_thread_server.py": "5cda47575ce9d9d84932d51d1d40cce8b1627fa3224133543f8419c8433de484",
    "trl/utils/torch_transform.py": "4d40e3ac765cb2541aacd8af60f1162914e4ce97b4cc8fadc871a7c6a2fe7b41",
    "trl/utils/rotation_conversion.py": "a2c350e42cc4e7eb906755867f1d977733d190925f22419e4b2e285ffac2bfa1",
    "isaac_utils/rotations.py": "88d08511c5ef4f64d406adfa632716deb7f259001f87ea8200bf3099bf5d61f2",
}
BODY_PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
                         9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])
OUTPUT_SMPL_INDICES = tuple(range(22)) + (39, 54)
# Official GatherMotionJointPositionsMultiFrame uses policy indices 23..28.
WRIST_NAMES = ("left_wrist_roll_joint", "right_wrist_roll_joint",
               "left_wrist_pitch_joint", "right_wrist_pitch_joint",
               "left_wrist_yaw_joint", "right_wrist_yaw_joint")
LAYOUT = {"encoder_mode_4": [0, 4], "smpl_joints_10frame_step1": [911, 1631],
          "smpl_anchor_orientation_heading_10frame_step1": [1631, 1691],
          "motion_joint_positions_wrists_10frame_step1": [1691, 1751]}
_PURE_MATH = None


def _require(value, message):
    if not value:
        raise ValueError(message)


def _finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    _require(array.shape == shape and np.isfinite(array).all(), f"Require finite {name}{shape}")
    return array.copy()


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verify_hash(path, expected):
    path = Path(path).resolve(strict=True)
    actual = _sha(path)
    _require(actual == expected, f"Pinned reference identity changed: {path}")
    return str(path), actual


def verify_runtime_sources(sonic_resources, human_info_path):
    """Bind exact encoder/config, skeleton, upstream math, and C++ gather sources."""
    root = Path(sonic_resources).resolve(strict=True)
    bindings = dict(_verify_hash(GEAR / name, value) for name, value in SOURCE_SHA256.items())
    bindings.update([_verify_hash(human_info_path, HUMAN_INFO_SHA256),
                     _verify_hash(root / "model_encoder.onnx", ENCODER_SHA256),
                     _verify_hash(root / "observation_config.yaml", OBSERVATION_SHA256)])
    return {"source_sha256": bindings, "official_gather": verify_official_sources()}


def load_human_info(path):
    """Only deserialize the fixed small CPU skeleton after its hash has passed."""
    _verify_hash(path, HUMAN_INFO_SHA256)
    import torch
    info = torch.load(path, map_location="cpu", weights_only=True)
    rest = _finite(info["J"].detach().cpu().numpy(), (55, 3), "human rest joints")
    parents = np.asarray(info["parents_list"], dtype=np.int64)
    _require(parents.shape == (55,) and parents[0] == -1 and
             all(0 <= int(parents[i]) < i for i in range(1, 55)), "Invalid fixed human FK tree")
    return rest, parents


def validate_body_pose(body24x7):
    """Validate pose geometry, without pretending quaternion validity is tracking quality."""
    body = _finite(body24x7, (24, 7), "PICO body xyz/xyzw")
    norm = np.linalg.norm(body[:, 3:], axis=1)
    _require(np.all(np.abs(norm - 1.) <= 1e-4), "PICO body requires unit xyzw quaternions")
    _require(np.max(np.linalg.norm(body[:, :3], axis=1)) <= 10., "PICO positions exceed room-scale metres")
    # Main articulated segments; hands 22/23 can be different endpoint definitions.
    for child in range(1, 22):
        length = np.linalg.norm(body[child, :3] - body[BODY_PARENTS[child], :3])
        _require(.005 <= length <= 1.5, "PICO body has an invalid skeletal segment")
    body[:, 3:] /= norm[:, None]
    return body


def _root_pose(actual_root_pose7):
    root = _finite(actual_root_pose7, (7,), "actual root xyz/wxyz")
    _require(abs(np.linalg.norm(root[3:]) - 1.) <= 1e-5, "Actual root requires unit wxyz")
    return root


def _wrist_reference(local_rotations):
    """Upstream elbow-twist/wrist XYZ mapping, in official policy wrist order."""
    sides = []
    for elbow, wrist, mirror in ((18, 20, 1.), (19, 21, -1.)):
        q = local_rotations[elbow].as_quat()  # xyzw
        twist = np.array([0., q[1], 0., q[3]])
        _require(np.linalg.norm(twist) > 1e-6, "Singular PICO elbow twist")
        twist /= np.linalg.norm(twist)
        swing = Rotation.from_quat(twist).inv() * local_rotations[elbow]
        swing_xyz = swing.as_euler("XYZ")
        wrist_xyz = local_rotations[wrist].as_euler("XYZ")
        # This matches manager lines 1393..1407, including its double pitch minus.
        sides.append(np.array([mirror * (swing_xyz[0] + wrist_xyz[0]), wrist_xyz[1],
                               swing_xyz[2] + wrist_xyz[2]]))
    return np.asarray(sides).T.reshape(6)


def _pure_math():
    """Import CPU quaternion math once, never the PICO application or SDK."""
    global _PURE_MATH
    if _PURE_MATH is None:
        from gear_sonic.trl.utils import torch_transform
        from gear_sonic.isaac_utils import rotations
        _verify_hash(torch_transform.__file__, SOURCE_SHA256["trl/utils/torch_transform.py"])
        _verify_hash(rotations.__file__, SOURCE_SHA256["isaac_utils/rotations.py"])
        _PURE_MATH = torch_transform, rotations
    return _PURE_MATH


def body24_to_smpl(body24x7, rest_joints, fk_parents):
    """Pinned float32 CPU PICO/SMPL FK, without importing its application.

    The output 24 points are first 22 SMPL joints plus *thumb tips* 39/54;
    passing PICO's raw 22/23 positions in those slots would be a different input.
    """
    body = validate_body_pose(body24x7)
    rest = _finite(rest_joints, (55, 3), "rest joints")
    parents = np.asarray(fk_parents, dtype=np.int64)
    _require(parents.shape == (55,), "Expected the fixed 55-joint FK tree")
    global_rot = Rotation.from_quat(body[:, 3:]) * Rotation.from_euler("y", np.pi)
    local = [global_rot[0]] + [global_rot[int(BODY_PARENTS[i])].inv() * global_rot[i]
                            for i in range(1, 24)]
    _require(parents[0] == -1 and all(0 <= int(parents[i]) < i for i in range(1, 55)),
             "Invalid causal human FK parent")
    import torch
    transform, rotations = _pure_math()
    # Retain the upstream float32 conversion and quaternion operations. A
    # float64 SciPy-only FK is close, but differs near pi due to upstream
    # axis-angle rounding; no need to introduce that numerical divergence.
    pose_aa = np.stack([rotation.as_rotvec() for rotation in local])
    body_pose = torch.from_numpy(pose_aa[1:22].reshape(1, 63)).float()
    root_aa = torch.from_numpy(pose_aa[0].reshape(1, 3)).float()
    root_quat = rotations.smpl_root_ytoz_up(transform.angle_axis_to_quaternion(root_aa))
    global_orient = transform.quaternion_to_angle_axis(root_quat)
    full_pose = torch.cat([global_orient, body_pose, torch.zeros((1, 99))], dim=1)
    matrices = transform.angle_axis_to_rotation_matrix(full_pose.reshape(1, 55, 3))
    relative = torch.from_numpy(rest.astype(np.float32)).clone().unsqueeze(0)
    relative[:, 1:] -= relative[:, parents[1:]]
    transforms = torch.nn.functional.pad(torch.cat([matrices, relative[..., None]], dim=-1), [0, 0, 0, 1])
    transforms[..., 3, 3] = 1.
    chain = [transforms[:, 0]]
    for i in range(1, 55):
        chain.append(torch.matmul(chain[int(parents[i])], transforms[:, i]))
    joints = torch.stack(chain, dim=-3)[..., :3, 3][:, list(OUTPUT_SMPL_INDICES)]
    reference_root = rotations.remove_smpl_base_rot(root_quat, w_last=False)
    inverse = transform.quat_inv(reference_root).unsqueeze(1).repeat(1, 24, 1)
    joints = transform.quat_apply(inverse, joints).numpy()[0].astype(np.float64)
    root_wxyz = reference_root.numpy()[0].astype(np.float64)
    wrist = _wrist_reference(local)
    _require(np.isfinite(joints).all() and np.isfinite(wrist).all(), "Nonfinite pose retarget result")
    return joints, root_wxyz, wrist


class PicoSonicReference:
    """Encode full PICO body pose through SONIC v1.1 SMPL mode 2.

    ``prepare`` does not advance accepted pose/time state; ``encode`` advances
    it only after valid encoder output.  Input/focus/packet freshness is the
    responsibility of the single raw receiver, because getters cache old data.
    """
    def __init__(self, sonic_resources, human_info_path, *, providers=None):
        self.identities = verify_runtime_sources(sonic_resources, human_info_path)
        self.rest_joints, self.fk_parents = load_human_info(human_info_path)
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(Path(sonic_resources) / "model_encoder.onnx"),
            sess_options=options, providers=providers or ["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        _require(len(inputs) == len(outputs) == 1 and list(inputs[0].shape) == [1, 1751]
                 and inputs[0].type == "tensor(float)" and list(outputs[0].shape) == [1, 64],
                 "Fixed SONIC v1.1 encoder contract changed")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.calibration = None
        self.last_body = self.last_timestamp_ns = None

    def calibrate(self, frame_body24x7, actual_root_pose7, actual_named_wrist_q=None):
        body, root = validate_body_pose(frame_body24x7), _root_pose(actual_root_pose7)
        joints, ref_root, wrist = body24_to_smpl(body, self.rest_joints, self.fk_parents)
        if actual_named_wrist_q is None:
            # No invented claim of actual joints: use the upstream absolute mapping.
            wrist_offset = np.zeros(6)
            wrist_source = "uncorrected_pinned_upstream_wrist_mapping"
        else:
            _require(isinstance(actual_named_wrist_q, Mapping) and
                     set(WRIST_NAMES).issubset(actual_named_wrist_q), "Provide six named actual wrist joints")
            actual = _finite([actual_named_wrist_q[n] for n in WRIST_NAMES], (6,), "actual wrist targets")
            wrist_offset = actual - wrist
            wrist_source = "actual_named_simulator_wrist_q_at_calibration_plus_human_angle_delta"
        self.apply_delta_heading = compute_apply_delta_heading(root[3:], ref_root)
        self.wrist_offset = wrist_offset
        self.calibration = {"actual_root_pose7": root.tolist(), "reference_root_wxyz": ref_root.tolist(),
            "apply_delta_heading_wxyz": self.apply_delta_heading.tolist(), "wrist_offset6": wrist_offset.tolist(),
            "wrist_names": list(WRIST_NAMES), "wrist_source": wrist_source,
            "body_pose_sha256": hashlib.sha256(body.tobytes()).hexdigest(),
            "initial_smpl_joints24_sha256": hashlib.sha256(joints.tobytes()).hexdigest()}
        self.last_body = self.last_timestamp_ns = None
        return dict(self.calibration)

    def _validate_continuity(self, body, timestamp):
        if timestamp is None:
            return None
        _require(type(timestamp) is int and timestamp > 0, "Positive integer device_timestamp_ns required")
        if self.last_timestamp_ns is None:
            return None
        dt = (timestamp - self.last_timestamp_ns) / 1e9
        _require(0 < dt <= .25, "PICO timestamp is duplicated, reversed, or has a gap over 250ms")
        position_speed = np.linalg.norm(body[:, :3] - self.last_body[:, :3], axis=1) / dt
        q0, q1 = self.last_body[:, 3:], body[:, 3:]
        angle = 2 * np.arccos(np.clip(np.abs(np.sum(q0 * q1, axis=1)), 0, 1))
        _require(float(position_speed.max()) <= 5., "PICO body position discontinuity exceeds 5m/s")
        _require(float(angle.max() / dt) <= 20., "PICO body orientation discontinuity exceeds 20rad/s")
        return {"dt_s": dt, "maximum_joint_speed_m_s": float(position_speed.max()),
                "maximum_joint_angular_speed_rad_s": float(angle.max() / dt)}

    def prepare(self, body24x7, actual_root_pose7, device_timestamp_ns=None):
        _require(self.calibration is not None, "Explicit PICO calibration is required")
        body, root = validate_body_pose(body24x7), _root_pose(actual_root_pose7)
        continuity = self._validate_continuity(body, device_timestamp_ns)
        joints, ref_root, wrist = body24_to_smpl(body, self.rest_joints, self.fk_parents)
        wrist = wrist + self.wrist_offset
        _require(np.max(np.abs(wrist)) <= np.pi, "PICO wrist conditioning exceeds absolute pi radians")
        aligned_root = multiply_wxyz(self.apply_delta_heading, ref_root)
        heading_inverse = heading_wxyz(root[3:]) * np.array([1., -1., -1., -1.])
        relative_root = multiply_wxyz(heading_inverse, aligned_root)
        six = Rotation.from_quat(relative_root[[1, 2, 3, 0]]).as_matrix()[:, :2].reshape(6)
        value = np.zeros((1, 1751), dtype=np.float32)
        value[0, 0] = 2.  # SMPL encoder mode, not ZMQ packet protocol version 3.
        value[0, 911:1631] = np.tile(joints.reshape(72), 10)
        value[0, 1631:1691] = np.tile(six, 10)
        value[0, 1691:1751] = np.tile(wrist, 10)
        _require(np.isfinite(value).all(), "PICO reference exceeds float32 range")
        evidence = {"contract": CONTRACT, "encoder_mode": 2, "layout": LAYOUT,
            "input_shape": [1, 1751], "input_dtype": "float32", "device_timestamp_ns": device_timestamp_ns,
            "reference_horizon": {"frames": 10, "spacing_s": .02, "kind": "causal_constant_desired_pose_hold",
                "offsets_s": [i / 50 for i in range(10)], "measured_future_frames": 0,
                "not_recorded_future": True},
            "raw_body_joint_order": "PICO SMPL24 xyz/xyzw; fixed human FK uses first22 plus thumb tips39/54",
            "smpl_output_indices": list(OUTPUT_SMPL_INDICES), "wrist_names": list(WRIST_NAMES),
            "wrist_targets6": wrist.tolist(), "reference_root_wxyz": ref_root.tolist(),
            "heading_normalization": "actual_robot_heading_inverse_times_initial_delta_heading_times_reference_root",
            "rotation_6d": "first_two_columns_flattened_rowwise", "PICO_raw_va_wva_used": False,
            "PICO_joint_t_used": False, "absolute_human_root_translation_encoded": False,
            "reference_input_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "continuity": continuity, "calibration": self.calibration,
            "identities": self.identities, "model_inference_performed": False,
            "physics_performed": False, "task_success_certified": False}
        return value, evidence

    def encode(self, body24x7, actual_root_pose7, device_timestamp_ns):
        _require(type(device_timestamp_ns) is int and device_timestamp_ns > 0,
                 "Positive integer device_timestamp_ns required")
        value, evidence = self.prepare(body24x7, actual_root_pose7, device_timestamp_ns)
        token = np.asarray(self.session.run([self.output_name], {self.input_name: value})[0], dtype=np.float32)
        _require(token.shape == (1, 64) and np.isfinite(token).all(), "Invalid SONIC v1.1 token")
        self.last_body = validate_body_pose(body24x7)
        self.last_timestamp_ns = device_timestamp_ns
        evidence.update(model_inference_performed=True, encoder_sha256=ENCODER_SHA256,
            token_sha256=hashlib.sha256(token[0].tobytes()).hexdigest(), token_shape=[64])
        return token[0].copy(), evidence
