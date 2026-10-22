"""One-time source-informed Handover cold start and measured contact logging.

This is not complete teacher state restoration. No teacher actions or future
trajectory rows are loaded, and initialization never runs during policy control.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

RECORDED_LIMIT_PENETRATION_RAD = .001

def vector(value, size):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite coordinates")
    return result


def pose(value):
    result = vector(value, 7)
    if not np.isclose(np.linalg.norm(result[3:]), 1, rtol=0, atol=1e-4):
        raise ValueError("Initial pose quaternion must be unit wxyz")
    return result


def load_initialization(path, episode):
    path = Path(path).resolve(strict=True)
    doc = json.loads(path.read_text())
    if doc.get("contract") != "cvpr-handover-heldout-initial-v1":
        raise ValueError("Unknown Handover initialization contract")
    row = doc["episodes"][str(episode)]
    if row.get("source_episode_index") not in (39, 65, 80, 92, 94):
        raise ValueError("Only the accepted five held-out source scenes are allowed")
    names = row["joint_names"]
    if len(names) != 43 or len(set(names)) != 43 or not all(isinstance(n, str) for n in names):
        raise ValueError("Require 43 unique named physical joints")
    vector(row["joint_qpos"], 43)
    vector(row.get("joint_qvel", [0] * 43), 43)
    pose(row["root_pose7"])
    vector(row.get("root_qvel6", [0] * 6), 6)
    if set(row["objects"]) != {"target", "container"}:
        raise ValueError("Require target and container initial poses")
    for item in row["objects"].values():
        pose(item["pose7"])
        if not np.isfinite(item["mass"]) or not 0 < item["mass"] < 10:
            raise ValueError("Invalid recorded object mass")
    if not row.get("sources"):
        raise ValueError("Require source-file identity bindings")
    for name, expected in row["sources"].items():
        source = Path(name).resolve(strict=True)
        if source.stat().st_size > 64 * 1024**2:
            raise ValueError("Initialization evidence must be bounded metadata")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Initialization evidence changed: {source}")
    return row, {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "source_episode_index": row["source_episode_index"],
                 "scope": "source-informed cold start; not complete teacher physics/RNG/controller restoration",
                 "teacher_actions_used": False, "restores_during_control": False,
                 "missing_velocity_policy": "zero if not explicitly recorded"}


def initialize(raw, row, receipt):
    import mujoco
    model, data = raw.mujoco.mjModel, raw.mujoco.mjData
    initial_time = float(data.time)
    root_pose = pose(row["root_pose7"])
    root_velocity = vector(row.get("root_qvel6", [0] * 6), 6)
    joint_positions = vector(row["joint_qpos"], 43)
    joint_velocities = vector(row.get("joint_qvel", [0] * 43), 43)
    pelvis = model.body("pelvis").id
    root = int(model.body_jntadr[pelvis])
    if root < 0 or model.jnt_type[root] != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError("Pelvis must own the floating root joint")
    assignments = []
    limit_penetrations = []
    for name, q, dq in zip(row["joint_names"], joint_positions, joint_velocities):
        joint = model.joint(name)
        if model.jnt_type[joint.id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Expected scalar hinge: {name}")
        if model.jnt_limited[joint.id]:
            lower, upper = model.jnt_range[joint.id]
            penetration = max(float(lower - q), float(q - upper), 0.)
            if penetration > RECORDED_LIMIT_PENETRATION_RAD:
                raise ValueError(f"Source reset joint exceeds local model limits: {name}")
            if penetration:
                # Recorded physical states can penetrate MuJoCo's soft limits.
                # Keep their exact values; this tolerance never clips actions.
                limit_penetrations.append({"joint": name, "penetration_rad": penetration})
        assignments.append((int(model.jnt_qposadr[joint.id]), int(model.jnt_dofadr[joint.id]), q, dq))
    objects = []
    for name, item in row["objects"].items():
        body = raw.mujoco.mj_objects[name]
        joint = int(model.body_jntadr[body.id])
        if joint < 0 or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Expected floating object: {name}")
        if not np.isclose(model.body_mass[body.id], item["mass"], rtol=1e-5, atol=1e-7):
            raise ValueError(f"Local/source object mass mismatch: {name}")
        objects.append((joint, pose(item["pose7"])))
    # Resolve and validate every source/model field before mutating live state.
    root_q, root_dq = int(model.jnt_qposadr[root]), int(model.jnt_dofadr[root])
    data.qpos[root_q:root_q + 7] = root_pose
    data.qvel[:] = 0
    data.qvel[root_dq:root_dq + 6] = root_velocity
    for qadr, dadr, q, dq in assignments:
        data.qpos[qadr], data.qvel[dadr] = q, dq
    for joint, initial_pose in objects:
        adr = int(model.jnt_qposadr[joint])
        data.qpos[adr:adr + 7] = initial_pose
    data.qacc_warmstart[:] = 0
    data.ctrl[:] = 0
    data.qfrc_applied[:] = 0
    data.xfrc_applied[:] = 0
    mujoco.mj_forward(model, data)
    observation, info = raw._get_obs(), raw._get_info()
    if float(data.time) != initial_time:
        raise RuntimeError("Cold-start observation advanced physics time")
    return observation, info, {**receipt, "applied_before_history_collection": True,
                              "sim_time_s": initial_time, "object_masses_verified": True,
                              "source_joint_values_clipped": False,
                              "maximum_recorded_limit_penetration_rad": RECORDED_LIMIT_PENETRATION_RAD,
                              "recorded_limit_penetrations": limit_penetrations}


def measured_task_state(raw):
    """Record measured poses/contacts, independently of the task's reward counter."""
    model, data = raw.mujoco.mjModel, raw.mujoco.mjData
    target, container = (raw.mujoco.mj_objects[n] for n in ("target", "container"))
    contacts = []
    for contact in data.contact:
        bodies = [int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2)]
        if target.id in bodies:
            other = bodies[1] if bodies[0] == target.id else bodies[0]
            contacts.append({"other_body": model.body(other).name,
                             "distance_m": float(contact.dist)})
    others = [c["other_body"] for c in contacts]
    return {"target_pose7": [*target.xpos.tolist(), *target.xquat.tolist()],
            "container_pose7": [*container.xpos.tolist(), *container.xquat.tolist()],
            "target_contacts": contacts,
            "left_hand_contact": any("left_hand" in n for n in others),
            "right_hand_contact": any("right_hand" in n for n in others),
            "container_contact": container.name in others,
            "scope": "measured camera/control-frame contacts; not the release's complete strict physics gate"}
