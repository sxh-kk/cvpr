"""Pure PICO body -> measured-FK VR3-point targets for continuous Mobile mode1.

No SDK, encoder, planner, MuJoCo object, or physics is constructed here. The
caller supplies one already validated, fresh Body24 packet and current actual
robot FK. Head/controller poses are deliberately not substituted for the
official Body22/23 wrists and Body12 neck. These are desired control targets,
not observed future states or a physical/task acceptance certificate.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


CONTRACT = 'cvpr-pico-mobile-measured-fk-vr3pt-reference-v1'
REPO = Path(__file__).resolve().parents[2]
GEAR = REPO / 'VLMExpert/third_party/SIMPLE/third_party/gear_sonic'
SOURCE_PATHS = {
    'pico_manager': GEAR / 'scripts/pico_manager_thread_server.py',
    'vr3pt_fk': GEAR / 'utils/teleop/vis/vr3pt_pose_visualizer.py',
}
SOURCE_SHA256 = {
    'pico_manager': '5cda47575ce9d9d84932d51d1d40cce8b1627fa3224133543f8419c8433de484',
    'vr3pt_fk': 'f223df8e7e1863e37a423aa80619e0ea42a67f1569df9c1d6e2487af08c4d3a6',
}
POINT_ORDER = ('left_wrist_tip', 'right_wrist_tip', 'head_from_torso')
ACTUAL_BODY_NAMES = ('left_wrist_yaw_link', 'right_wrist_yaw_link', 'torso_link')
BODY_KEYPOINT_INDICES = (0, 22, 23, 12)
VR_OFFSETS = np.array([[.18, -.025, 0.], [.18, .025, 0.], [0., 0., .35]])
UNITY_TO_ROBOT = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
# Postmultiplication matches pinned manager OFFSETS, not Euler axis negation.
_ORIENTATION_OFFSETS = Rotation.from_euler('xyz',
    [[0., 0., -90.], [90., 0., 0.], [-90., 0., 180.], [0., 0., -90.]], degrees=True)


def _finite(value, shape, name):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f'Finite {name} with shape {shape} required')
    return value.copy()


def _unit_quaternions(value, shape, name, *, tolerance=1e-6):
    value = _finite(value, shape, name)
    if np.any(np.abs(np.linalg.norm(value, axis=-1) - 1.) > tolerance):
        raise ValueError(f'Unit {name} quaternions required')
    return value


def _actual_root(value):
    value = _finite(value, (7,), 'actual root xyz/wxyz')
    _unit_quaternions(value[3:], (4,), 'actual root wxyz')
    return value, Rotation.from_quat(value[3:], scalar_first=True)


def _sha_array(value):
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float64).tobytes()).hexdigest()


def _sha_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def verify_official_sources(paths=None):
    """Read and bind pinned upstream math/FK definitions without importing them."""
    paths = SOURCE_PATHS if paths is None else paths
    if set(paths) != set(SOURCE_SHA256):
        raise ValueError('Both pinned PICO VR3-point source files must be bound')
    bindings = {}
    for name, expected in SOURCE_SHA256.items():
        path = Path(paths[name]).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError('Pinned PICO VR3-point source changed: ' + name)
        bindings[str(path)] = actual
    return {'official_sources_rehashed': True, 'source_sha256': bindings}


def body24_to_root_local_vr(body24x7):
    """Pinned ``_process_3pt_pose`` math, including its float32 intermediates.

    Input is Unity xyz/xyzw. Output is [left wrist, right wrist, neck],
    xyz/wxyz in the *current human pelvis frame*. Input arrays are never edited.
    Shape/quaternion checks do not replace the caller's tracking/freshness gate.
    """
    body = _finite(body24x7, (24, 7), 'PICO Body24 xyz/xyzw')
    _unit_quaternions(body[:, 3:], (24, 4), 'PICO xyzw', tolerance=1e-4)
    converted = np.zeros((24, 7), dtype=np.float32)
    for index, pose in enumerate(body):
        converted[index, :3] = UNITY_TO_ROBOT @ pose[:3]
        human_matrix = Rotation.from_quat(pose[3:]).as_matrix()
        converted[index, 3:] = Rotation.from_matrix(
            UNITY_TO_ROBOT @ human_matrix @ UNITY_TO_ROBOT.T).as_quat(scalar_first=True)
    keypoints = np.zeros((4, 7), dtype=np.float32)
    for index, joint in enumerate(BODY_KEYPOINT_INDICES):
        keypoints[index, :3] = converted[joint, :3]
        keypoints[index, 3:] = (Rotation.from_quat(converted[joint, 3:], scalar_first=True)
                               * _ORIENTATION_OFFSETS[index]).as_quat()
    inverse_root = Rotation.from_quat(keypoints[0, 3:]).inv()
    for index in range(1, 4):
        keypoints[index, :3] = inverse_root.apply(keypoints[index, :3] - keypoints[0, :3])
        keypoints[index, 3:] = (inverse_root * Rotation.from_quat(
            keypoints[index, 3:])).as_quat(scalar_first=True)
    return keypoints[1:].copy()


class MobileVRReference:
    """Explicit X calibration and stateless-per-packet mode1 VR target assembly.

    Wrist calibration follows the pinned initial-neck-inverse/offset algebra.
    Neck calibration additionally aligns to the measured torso FK orientation
    and root-local origin, instead of assuming the official identity/0.05m
    initial torso. All three first-frame targets therefore match actual FK.
    This measured-FK neck adaptation is recorded explicitly in every receipt.
    """

    def __init__(self):
        self.identities = verify_official_sources()
        self.calibration = None

    @property
    def identity(self):
        """Independent JSON source receipt for the collector's session metadata."""
        return deepcopy(self.identities)

    def calibrate(self, body24x7, actual_root7, actual_body_positions_world,
                  actual_body_quats_world):
        """Calibrate from actual wrist_yaw_link L/R and torso_link body origins.

        Actual orientations are wxyz; the official local tip offsets are added
        exactly once here. Do not pass wrist tips in place of body origins.
        No simulator state is modified, and no all-zero/default FK is invented.
        """
        human = body24_to_root_local_vr(body24x7)
        root, root_rotation = _actual_root(actual_root7)
        positions = _finite(actual_body_positions_world, (3, 3), 'actual body world origins')
        quaternions = _unit_quaternions(actual_body_quats_world, (3, 4), 'actual body world wxyz')
        body_rotations = Rotation.from_quat(quaternions, scalar_first=True)
        actual_tips_world = positions + body_rotations.apply(VR_OFFSETS)
        inverse_root = root_rotation.inv()
        actual_local_points = inverse_root.apply(actual_tips_world - root[:3])
        actual_local_rotations = inverse_root * body_rotations
        human_rotations = Rotation.from_quat(human[:, 3:], scalar_first=True)
        neck_inverse = human_rotations[2].inv()
        corrected_positions = neck_inverse.apply(human[:, :3])
        corrected_rotations = neck_inverse * human_rotations
        wrist_position_offsets = corrected_positions[:2] - actual_local_points[:2]
        rotation_offsets = actual_local_rotations * corrected_rotations.inv()
        torso_origin_local = inverse_root.apply(positions[2] - root[:3])

        receipt = {
            'contract': CONTRACT, 'point_order': list(POINT_ORDER),
            'actual_body_names': list(ACTUAL_BODY_NAMES),
            'actual_root_pose7': root.tolist(),
            'actual_body_origins_world': positions.tolist(),
            'actual_body_quaternions_world_wxyz': quaternions.tolist(),
            'actual_tip_points_world': actual_tips_world.tolist(),
            'actual_tip_points_root_local': actual_local_points.tolist(),
            'actual_point_quaternions_root_local_wxyz': actual_local_rotations.as_quat(scalar_first=True).tolist(),
            'official_body_local_tip_offsets_m': VR_OFFSETS.tolist(),
            'tip_offsets_applied_once_to_actual_body_origins': True,
            'body24x7_float64_sha256': _sha_array(body24x7),
            'initial_human_root_local_xyz_wxyz': human.tolist(),
            'initial_human_neck_inverse_wxyz': neck_inverse.as_quat(scalar_first=True).tolist(),
            'wrist_position_offsets_m': wrist_position_offsets.tolist(),
            'point_rotation_offsets_wxyz': rotation_offsets.as_quat(scalar_first=True).tolist(),
            'actual_initial_torso_origin_root_local': torso_origin_local.tolist(),
            'neck_calibration': 'measured_actual_torso_FK_origin_and_orientation_adaptation',
            'official_identity_neck_and_fixed_005m_torso_assumed': False,
            'actual_state_modified': False, 'source_identities': deepcopy(self.identities),
            'tracking_and_freshness_checked_by_helper': False,
            'physical_or_task_success_certified': False,
        }
        receipt['calibration_sha256'] = _sha_json(receipt)
        # Publish only after every input and derived value has been validated.
        self._neck_inverse = neck_inverse
        self._wrist_position_offsets = wrist_position_offsets.copy()
        self._rotation_offsets = rotation_offsets
        self._torso_origin_local = torso_origin_local.copy()
        self.calibration = receipt
        return deepcopy(receipt)

    def prepare(self, body24x7, actual_root7):
        """Return already-offset world VR targets attached to this actual root.

        Pass both arrays and the SAME ``actual_root7`` to mode1's encoder
        reference builder. Its full-root inverse then recovers these desired
        local targets while robot translation and turning carry them along.
        Neither stale packet reuse nor packet timing is authorized by this API.
        """
        if self.calibration is None:
            raise ValueError('Explicit Mobile VR calibration is required')
        human = body24_to_root_local_vr(body24x7)
        root, root_rotation = _actual_root(actual_root7)
        human_rotations = Rotation.from_quat(human[:, 3:], scalar_first=True)
        local_rotations = self._rotation_offsets * (self._neck_inverse * human_rotations)
        local_points = self._neck_inverse.apply(human[:, :3])
        local_points[:2] -= self._wrist_position_offsets
        local_points[2] = self._torso_origin_local + local_rotations[2].apply(VR_OFFSETS[2])
        world_points = root_rotation.apply(local_points) + root[:3]
        world_quaternions = (root_rotation * local_rotations).as_quat(scalar_first=True)
        if not np.isfinite(world_points).all() or not np.isfinite(world_quaternions).all():
            raise ValueError('Nonfinite calibrated Mobile VR targets')
        evidence = {
            'contract': CONTRACT, 'point_order': list(POINT_ORDER),
            'human_body_keypoint_indices_root_left_right_neck': list(BODY_KEYPOINT_INDICES),
            'human_source': 'same_packet_PicoFrame_Body24_xyz_xyzw',
            'raw_Head_or_Controller_poses_substituted': False,
            'body24x7_float64_sha256': _sha_array(body24x7),
            'human_root_local_xyz_wxyz': human.tolist(),
            'desired_points_root_local': local_points.tolist(),
            'desired_quaternions_root_local_wxyz': local_rotations.as_quat(scalar_first=True).tolist(),
            'actual_root_pose7_used_for_world_lift': root.tolist(),
            'world_points': world_points.tolist(), 'world_quaternions_wxyz': world_quaternions.tolist(),
            'world_points_already_tip_offset': True,
            'root_attachment': 'same_tick_current_actual_full_root_translation_and_rotation',
            'absolute_human_root_translation_encoded': False,
            'human_current_pelvis_position_and_orientation_removed': True,
            'neck_calibration': self.calibration['neck_calibration'],
            'calibration_sha256': self.calibration['calibration_sha256'],
            'calibration': deepcopy(self.calibration),
            'world_points_float64_sha256': _sha_array(world_points),
            'world_quaternions_float64_sha256': _sha_array(world_quaternions),
            'tracking_and_freshness_checked_by_helper': False,
            'caller_must_validate_same_packet_tracking': True,
            'actual_state_modified': False, 'model_inference_performed': False,
            'physics_steps_performed': 0, 'physical_or_task_success_certified': False,
        }
        return world_points.copy(), world_quaternions.copy(), evidence
