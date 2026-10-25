"""Named SONIC hand posture with an explicit local PICO analogue input policy.

The fixed closed posture is the pinned upstream middle-close gesture. Combining
trigger and grip with max() is this collector's local policy, not a claim about
upstream PICO button behavior. This module does not import an SDK or simulator,
change torque/physics settings, or initialize any device.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from numbers import Real
from pathlib import Path


PROFILE_ID = "sonic_v1_1_middle_close_local_max_trigger_grip_v1"
JOINT_SUFFIXES = ("thumb_0_joint", "thumb_1_joint", "thumb_2_joint",
                  "index_0_joint", "index_1_joint", "middle_0_joint", "middle_1_joint")
LEFT_CLOSE = (0., .7, .7, -1., -1.5, -1., -1.5)
RIGHT_CLOSE = tuple(-value for value in LEFT_CLOSE)
HAND_JOINT_NAMES = tuple(f"{side}_hand_{suffix}" for side in ("left", "right")
                         for suffix in JOINT_SUFFIXES)
UPSTREAM_SOURCE_PATH = (Path(__file__).resolve().parents[2] / "VLMExpert/third_party/SIMPLE/third_party/"
                        "gear_sonic/utils/teleop/solver/hand/g1_gripper_ik_solver.py")
UPSTREAM_SOURCE_SHA256 = "1819bc011d022e2e9567e5209bc8c5b4f3f328c1275ab5cfeb0a8e0a2e9b9c25"


def _amount(inputs, side):
    values = []
    for control in ("trigger", "grip"):
        key = f"{side}_{control}"
        value = inputs.get(key)
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not math.isfinite(float(value)) or not 0. <= value <= 1.):
            raise ValueError(f"{key} must be a finite numeric value in [0,1]")
        values.append(float(value))
    # Both original values have been validated. In particular, max(valid, NaN)
    # must never silently accept the valid value and hide a malformed packet.
    return max(values)


def named_hand_targets(robot, frame):
    """Return fourteen radian targets by name, independent of config ordering.

    Each side's actual robot config must expose exactly the expected seven
    distinct joint names. Its legacy init/close poses are deliberately unused:
    release is explicit all-zero and closure uses the fixed SONIC gesture.
    """
    inputs = getattr(frame, "inputs", None)
    if not isinstance(inputs, Mapping):
        raise ValueError("PICO inputs must provide both trigger and grip for each hand")
    amounts = {side: _amount(inputs, side) for side in ("left", "right")}
    for side in ("left", "right"):
        expected = {f"{side}_hand_{suffix}" for suffix in JOINT_SUFFIXES}
        try:
            names = getattr(robot.controller_cfg, f"{side}_eef_cfg").joint_names
        except AttributeError as error:
            raise ValueError(f"Robot is missing {side}_eef_cfg.joint_names") from error
        if (not isinstance(names, (list, tuple)) or len(names) != 7
                or any(not isinstance(name, str) for name in names) or set(names) != expected):
            raise ValueError(f"Robot {side} hand must expose exactly the seven named SONIC joints")
    targets = {}
    for side, closed in (("left", LEFT_CLOSE), ("right", RIGHT_CLOSE)):
        amount = amounts[side]
        for suffix, angle in zip(JOINT_SUFFIXES, closed):
            targets[f"{side}_hand_{suffix}"] = float(angle*amount) if amount else 0.
    return targets


def _projection(named_qpos, side, close_pose, label):
    if not isinstance(named_qpos, Mapping):
        raise ValueError(f"{label} must provide named joint positions")
    numerator = denominator = 0.
    for suffix, close in zip(JOINT_SUFFIXES, close_pose):
        name = f"{side}_hand_{suffix}"
        value = named_qpos.get(name)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"{label} {name} must be finite")
        numerator += float(value)*close
        denominator += close*close
    # Deliberately retain negative values and values above one. Clamping would
    # hide motion in the wrong direction or travel beyond the desired closure.
    return numerator/denominator


def hand_feedback(frame, named_actual_qpos, named_commanded_targets=None, *, following=False):
    """Separate requested inputs, applied targets and actual finger positions.

    Closure is a projection onto each side's fixed close vector, not a measure
    of object contact, holding force, or grasp success. Inputs remain visible
    when paused; an unapplied or missing command is represented by None.
    """
    if type(following) is not bool:
        raise ValueError("following must be an explicit boolean")
    inputs = getattr(frame,"inputs",None)
    if not isinstance(inputs,Mapping):
        raise ValueError("PICO inputs must provide both trigger and grip for each hand")
    result = {"profile":PROFILE_ID,"following":following,
        "closure_measurement":"projection onto named official close pose; not grasp success"}
    for side,closed in (("left",LEFT_CLOSE),("right",RIGHT_CLOSE)):
        requested = _amount(inputs,side)
        result[side] = {"trigger":float(inputs[f"{side}_trigger"]),
            "grip":float(inputs[f"{side}_grip"]),"requested_closed_fraction":requested,
            "applied_closed_fraction":(_projection(named_commanded_targets,side,closed,"Commanded hand")
                if following and named_commanded_targets is not None else None),
            "actual_closed_fraction":_projection(named_actual_qpos,side,closed,"Actual hand")}
    return result


def profile_receipt():
    """Bind the posture source; call once during session preparation/metadata."""
    source = Path(UPSTREAM_SOURCE_PATH).resolve(strict=True)
    actual_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    if actual_sha != UPSTREAM_SOURCE_SHA256:
        raise ValueError("Pinned upstream SONIC middle-close source identity changed")
    return {"profile_id": PROFILE_ID,
        "policy": "per_hand_amount=max(trigger,grip); both raw values finite in [0,1]; no clamping",
        "policy_scope": "explicit local PICO analogue control mapping; upstream supplies only the fixed middle-close pose",
        "joint_names": list(HAND_JOINT_NAMES), "angle_unit": "radian",
        "open_pose": [0.]*14, "left_close_pose": list(LEFT_CLOSE), "right_close_pose": list(RIGHT_CLOSE),
        "upstream": {"path": str(source), "sha256": actual_sha,
            "method": "G1GripperInverseKinematicsSolver._get_middle_close_q_desired",
            "pose_order": "thumb0,thumb1,thumb2,index0,index1,middle0,middle1; right mirrored"},
        "torque_collision_physics_modified": False,
        "physical_grasp_success_certified": False}
