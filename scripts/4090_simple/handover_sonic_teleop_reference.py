"""Pure official SONIC1.1 mode1 reference assembly, without model inference.

The three world points passed to ``build_teleop_reference`` are already offset
VR points: left wrist tip, right wrist tip, torso/head point. Do not add wrist
offsets twice. ``vr_points_from_body_poses`` is the separate explicit adapter
for unoffset body origins. Root-local VR uses the full actual root rotation;
the separate six-value anchor uses actual robot heading only, as in C++.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

CONTRACT = "cvpr-sonic11-pure-teleop-reference-v1"
REPO = Path(__file__).resolve().parents[2]
RESOURCES = Path('/home/zxc/CVPR_Evaluation_runtime/resources')
DEPLOY = REPO / 'VLMExpert/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
SOURCE_PATHS = {
    'gather_cpp': RESOURCES / 'carrybox_source_audit_20261021/official_controller/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/src/g1_deploy_onnx_ref.cpp',
    'math_header': DEPLOY / 'include/math_utils.hpp',
    'joint_mapping_header': DEPLOY / 'include/policy_parameters.hpp',
    'observation_config': RESOURCES / 'sonic_v1_1/observation_config.yaml'}
SOURCE_SHA256 = {
    'gather_cpp': '7efde47fb9ff20b07c2d1bee5d2e4a096e205326dbb7100ccd8fef39648d5411',
    'math_header': 'd85de60b9f2c6a4d26b1380264b172b3380200138173998f2a64660310f3c4d5',
    'joint_mapping_header': 'b9332adf07c2c9b75c9b1e0756e57c7a1c2a890d8bb0aa53c3f1905fb739b791',
    'observation_config': '4a67713b310932e50aca81f19188c8d76013148e98b15c8b5bbea995f12e59f0'}
LOWER_BODY_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left', 'right')
                         for joint in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
FUTURE_OFFSETS = tuple(range(0, 46, 5))
VR_POINT_ORDER = ('left_wrist_tip', 'right_wrist_tip', 'head_from_torso')
VR_OFFSETS = ((.18, -.025, 0.), (.18, .025, 0.), (0., 0., .35))
LAYOUT = {'encoder_mode_4': (0, 4), 'motion_anchor_orientation_heading': (644, 650),
    'motion_joint_positions_lowerbody_10frame_step5': (650, 770),
    'motion_joint_velocities_lowerbody_10frame_step5': (770, 890),
    'vr_3point_local_target': (890, 899), 'vr_3point_local_orn_target': (899, 911)}


def require(value, message):
    if not value: raise ValueError(message)


def finite(value, shape, name):
    array = np.asarray(value, dtype=np.float64)
    require(array.shape == shape and np.isfinite(array).all(), f'Require finite {name}{shape}')
    return array.copy()


def quaternion(value, shape, name):
    q = finite(value, shape, name)
    require(np.all(np.abs(np.linalg.norm(q, axis=-1) - 1.) <= 1e-6), f'Require unit wxyz {name}')
    return q


def multiply_wxyz(a, b):
    """Hamilton product, retaining the input quaternion sign convention."""
    w, x, y, z = a; v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v], dtype=np.float64)


def conjugate(q): return np.asarray(q) * np.array([1., -1., -1., -1.])


def rotate(q, v):
    # Exact algebra used by official quat_rotate_d; unit inputs are required.
    return v * (2 * q[0]**2 - 1) + 2 * q[0] * np.cross(q[1:], v) + 2 * q[1:] * np.dot(q[1:], v)


def rotation(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def heading_wxyz(q):
    """Official calc_heading_quat_d: atan2 of the rotated local X axis."""
    q = quaternion(q, (4,), 'heading quaternion')
    direction = rotate(q, np.array([1., 0., 0.]))
    yaw = math.atan2(float(direction[1]), float(direction[0]))
    return np.array([math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)])


def compute_apply_delta_heading(initial_robot_root_wxyz, initial_reference_root_wxyz, delta_heading_rad=0.):
    """Official initial-heading alignment, explicit and independent of VR points."""
    require(type(delta_heading_rad) in (float, int) and math.isfinite(delta_heading_rad), 'Finite heading delta required')
    align = multiply_wxyz(heading_wxyz(initial_robot_root_wxyz), conjugate(heading_wxyz(initial_reference_root_wxyz)))
    delta = np.array([math.cos(delta_heading_rad / 2), 0., 0., math.sin(delta_heading_rad / 2)])
    return multiply_wxyz(delta, align)


def encoder_mode4(mode):
    require(type(mode) is int and mode in (0, 1, 2), 'Supported scalar encoder mode required')
    return np.array([mode, 0., 0., 0.], dtype=np.float32)


def vr_points_from_body_poses(body_positions_world, body_quaternions_world_wxyz):
    """Explicitly add official offsets once: left wrist, right wrist, torso."""
    positions = finite(body_positions_world, (3, 3), 'unoffset body world positions')
    quaternions = quaternion(body_quaternions_world_wxyz, (3, 4), 'body world orientations')
    points = positions + np.stack([rotate(q, np.asarray(offset)) for q, offset in zip(quaternions, VR_OFFSETS)])
    return points, quaternions.copy()


def verify_official_sources(paths=None):
    """Separate read-only freeze check; pure assembly below does not read files."""
    paths = SOURCE_PATHS if paths is None else paths
    require(set(paths) == set(SOURCE_SHA256), 'All official assembly sources must be bound')
    bindings = {}
    for name, path in paths.items():
        path = Path(path).resolve(strict=True)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == SOURCE_SHA256[name], 'Pinned official mode1 source changed: ' + name)
        bindings[str(path)] = actual
    return {'official_sources_rehashed': True, 'source_sha256': bindings}


def build_teleop_reference(lower_body_q, lower_body_dq, current, actual_root_pose7,
                           vr_world_points, vr_world_quaternions_wxyz, reference_root_wxyz,
                           *, apply_delta_heading_wxyz=(1., 0., 0., 0.), lower_joint_names=LOWER_BODY_NAMES):
    """Return float32[1,1751] and source/mapping evidence without side effects.

    ``lower_body_q/dq[N,12]`` use MuJoCo left-leg6 then right-leg6 order,
    absolute radians and radians/s. Full current+45 native50Hz samples are
    required; no implicit tail clamp occurs. Explicit synthesized horizons
    must be declared by the caller rather than described as recorded futures.

    World VR goals already include tip/head offsets. Their root-local transform
    uses actual root roll/pitch/yaw, while the anchor is heading-only inverse
    times explicit initial-heading alignment times the current reference root.
    """
    q = np.asarray(lower_body_q, dtype=np.float64); dq = np.asarray(lower_body_dq, dtype=np.float64)
    require(q.ndim == 2 and q.shape[1] == 12 and dq.shape == q.shape
            and np.isfinite(q).all() and np.isfinite(dq).all(), 'Require finite aligned lower-body q/dq[N,12]')
    require(tuple(lower_joint_names) == LOWER_BODY_NAMES, 'Lower-body reference must use named MuJoCo leg order')
    require(type(current) is int and current >= 0 and current + 45 < len(q), 'Complete current+45 future required; no implicit tail clamp')
    root = finite(actual_root_pose7, (7,), 'actual root pose7')
    root_q = quaternion(root[3:], (4,), 'actual root wxyz')
    points = finite(vr_world_points, (3, 3), 'already-offset world VR points')
    point_q = quaternion(vr_world_quaternions_wxyz, (3, 4), 'world VR orientations')
    ref_q = quaternion(reference_root_wxyz, (4,), 'current reference root')
    delta_q = quaternion(apply_delta_heading_wxyz, (4,), 'explicit heading alignment')
    require(np.allclose(delta_q[1:3], 0., atol=1e-12, rtol=0), 'Heading alignment must be yaw-only')
    inverse = conjugate(root_q)
    local_points = np.stack([rotate(inverse, point - root[:3]) for point in points])
    local_quaternions = np.stack([multiply_wxyz(inverse, orient) for orient in point_q])
    anchor_q = multiply_wxyz(conjugate(heading_wxyz(root_q)), multiply_wxyz(delta_q, ref_q))
    indices = current + np.asarray(FUTURE_OFFSETS)
    value = np.zeros((1, 1751), dtype=np.float32)
    value[0, :4] = encoder_mode4(1)
    value[0, 644:650] = rotation(anchor_q)[:, :2].reshape(-1)
    value[0, 650:770] = q[indices].reshape(-1)
    value[0, 770:890] = dq[indices].reshape(-1)
    value[0, 890:899] = local_points.reshape(-1)
    value[0, 899:911] = local_quaternions.reshape(-1)
    require(np.isfinite(value).all(), 'Reference exceeds float32 range')
    evidence = {'contract': CONTRACT, 'encoder_mode': 1, 'encoder_mode_encoding': 'scalar_mode_plus_three_zero_fillers',
        'input_shape': [1, 1751], 'input_dtype': 'float32', 'encoder_input_sha256': hashlib.sha256(value.tobytes()).hexdigest(),
        'layout': {key: list(bounds) for key, bounds in LAYOUT.items()}, 'future_reference_indices': indices.tolist(),
        'lower_joint_names': list(LOWER_BODY_NAMES), 'VR_point_order': list(VR_POINT_ORDER),
        'world_VR_points_already_offset': True, 'offsets_applied_by_builder': False,
        'VR_transform': 'full_actual_root_inverse', 'anchor_transform': 'actual_robot_heading_inverse_times_apply_delta_heading_times_current_reference_root',
        'anchor_6D_layout': 'first_two_rotation_matrix_columns_flattened_row_wise',
        'inactive_encoder_branches_zeroed': True, 'tail_clamped_by_builder': False,
        'official_source_sha256': {str(SOURCE_PATHS[key]): value for key, value in SOURCE_SHA256.items()},
        'official_source_files_rehashed_by_builder': False, 'source_freeze_instruction': 'Call verify_official_sources once and bind its actual file SHAs before execution',
        'model_inference_performed': False, 'physics_steps_performed': 0, 'GPU_used': False,
        'task_success_or_safe_control_certified': False}
    return value, evidence
