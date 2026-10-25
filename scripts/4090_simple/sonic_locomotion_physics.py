"""Isolated SONIC walking model helpers and recorded-evidence numeric gates.

Importing this module loads neither MuJoCo nor a policy, renderer or SDK.
Models/assets are supplied explicitly. Only ``initialize`` creates a new,
cold MjData; none of these helpers restores a running simulator's root state.
The pure audit checks saved measurements, not planner forecast displacement.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np


BODY_JOINT_NAMES = tuple(
    [f"{side}_{joint}_joint" for side in ("left", "right")
     for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{joint}_joint" for joint in ("yaw", "roll", "pitch")]
    + [f"{side}_{joint}_joint" for side in ("left", "right")
       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                     "wrist_roll", "wrist_pitch", "wrist_yaw")])
HAND_JOINT_NAMES = tuple(f"{side}_hand_{finger}_{joint}_joint"
    for side in ("left", "right") for finger, count in (("thumb", 3), ("index", 2), ("middle", 2))
    for joint in range(count))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    _require(result.shape == shape and np.isfinite(result).all(), f"Require finite {name}{shape}")
    return result


def _number(value, name):
    _require(type(value) in (float, int) and math.isfinite(value), f"Require finite numeric {name}")
    return float(value)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_model(environment, *, official_scene_path=None, simple_robot_path=None, expected_sha256=None):
    """Build the selected object-free model using explicit existing XML paths.

    Official scene options are preserved except the official 200Hz timestep.
    SIMPLE gets its existing engine/ground options. Optional identities map
    explicit asset paths to expected digests; no runtime path is implicit.
    """
    import mujoco
    _require(environment in ("official", "simple"), "Choose official or simple model")
    for path, expected in (expected_sha256 or {}).items():
        _require(sha256(Path(path).resolve(strict=True)) == expected, "Model resource SHA changed: " + str(path))
    if environment == "official":
        _require(official_scene_path is not None, "Explicit official_scene_path is required")
        model = mujoco.MjModel.from_xml_path(str(Path(official_scene_path).resolve(strict=True)))
        model.opt.timestep = .005
    else:
        _require(simple_robot_path is not None, "Explicit simple_robot_path is required")
        spec = mujoco.MjSpec.from_file(str(Path(simple_robot_path).resolve(strict=True)))
        spec.option.timestep = .005
        spec.option.impratio = 10
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.noslip_iterations = 2
        for geom in spec.geoms:
            name = (geom.name or "") + (geom.meshname or "")
            if "hand" in name or "gripper" in name:
                geom.solimp[:3] = [.9, .99, .0001]
                geom.solref[:2] = [.005, 1.]
        spec.worldbody.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[0, 0, 1], pos=[0, 0, 0], friction=[1., .005, .0001], material="groundplane")
        model = spec.compile()
    _require((model.nq, model.nv, model.nu, model.njnt) == (50, 49, 43, 44), "Require free-root43-joint model")
    _require(model.neq == model.nmocap == 0 and np.array_equal(model.opt.gravity, [0, 0, -9.81]),
             "Require gravity and no equality/mocap assistance")
    actual = {model.joint(i).name for i in range(model.njnt)
              if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE}
    _require(actual == set(BODY_JOINT_NAMES + HAND_JOINT_NAMES), "Model joint identity differs")
    actuator_mapping(model)
    return model


def actuator_mapping(model):
    import mujoco
    result = {}
    for actuator in range(model.nu):
        _require(model.actuator_trntype[actuator] == mujoco.mjtTrn.mjTRN_JOINT, "Require joint torque actuators")
        name = model.joint(int(model.actuator_trnid[actuator, 0])).name
        _require(name not in result and np.array_equal(model.actuator_gear[actuator], [1, 0, 0, 0, 0, 0]),
                 "Require unique named unit-gear actuators")
        result[name] = {"actuator_index": actuator, "actuator_name": model.actuator(actuator).name,
                        "ctrlrange": model.actuator_ctrlrange[actuator].tolist()}
    _require(set(result) == set(BODY_JOINT_NAMES + HAND_JOINT_NAMES), "Missing43-joint transmissions")
    return result


def collision_bounds(kind, size, position, rotation):
    """Pure box/sphere world bounds, including rotations of box collision soles."""
    p, r = _finite(position, (3,), "geom position"), _finite(rotation, (3, 3), "geom rotation")
    _require(np.allclose(r.T @ r, np.eye(3), rtol=0, atol=1e-6) and abs(np.linalg.det(r)-1) < 1e-6,
             "Require proper geom rotation")
    size = _finite(size, (3,), "geom size")
    if kind == "box":
        _require(np.all(size > 0), "Box half extents must be positive")
        half = np.abs(r) @ size
    elif kind == "sphere":
        _require(size[0] > 0, "Sphere radius must be positive")
        half = np.full(3, size[0])
    else:
        raise ValueError("Unsupported foot collision geometry: " + str(kind))
    return p-half, p+half


def foot_geometry(model, data, side, *, ground_z_m=0.):
    """Read actual FK collision soles; do not step physics or use reference feet."""
    import mujoco
    _require(side in ("left", "right"), "Unknown foot side")
    ground = _number(ground_z_m, "ground_z_m")
    body = model.body(side + "_ankle_roll_link").id
    geoms = np.flatnonzero((model.geom_bodyid == body)
        & ((model.geom_contype != 0) | (model.geom_conaffinity != 0)))
    _require(len(geoms) > 0, "Foot collision geometry is missing or disabled")
    bounds = []
    for geom in geoms:
        kind = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}.get(model.geom_type[geom])
        bounds.append(collision_bounds(kind, model.geom_size[geom], data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3)))
    lower, upper = np.min([row[0] for row in bounds], axis=0), np.max([row[1] for row in bounds], axis=0)
    return {"body": int(body), "geom_indices": geoms.tolist(),
        "ankle_world_xyz": _finite(data.xpos[body], (3,), "ankle xyz").tolist(),
        "minimum_collision_z_m": float(lower[2]), "ground_clearance_m": float(lower[2]-ground),
        "ground_z_m": ground, "world_collision_bounds_min_m": lower.tolist(),
        "world_collision_bounds_max_m": upper.tolist()}


def initialize(model, default_angles, *, body_joint_names=BODY_JOINT_NAMES, hand_joint_names=HAND_JOINT_NAMES):
    """Create/align only new cold MjData; never reset an externally supplied state."""
    import mujoco
    angles = _finite(default_angles, (len(body_joint_names),), "default angles")
    _require(tuple(body_joint_names) == BODY_JOINT_NAMES and set(hand_joint_names) == set(HAND_JOINT_NAMES),
             "Explicit joint names differ from the fixed43-joint contract")
    data = mujoco.MjData(model)
    root = model.joint("floating_base_joint")
    address = int(root.qposadr[0])
    data.qpos[address:address+7] = [0, 0, 0, 1, 0, 0, 0]
    for name, angle in zip(body_joint_names, angles):
        _require(model.joint(name).range[0] <= angle <= model.joint(name).range[1], "Default joint angle out of range")
        data.joint(name).qpos[0] = angle
    for name in hand_joint_names:
        data.joint(name).qpos[0] = 0.
    data.qvel[:] = 0.
    data.ctrl[:] = 0.
    mujoco.mj_forward(model, data)
    data.qpos[address+2] = -min(foot_geometry(model, data, side)["minimum_collision_z_m"] for side in ("left", "right"))
    mujoco.mj_forward(model, data)
    feet = {side: foot_geometry(model, data, side) for side in ("left", "right")}
    _require(all(abs(foot["minimum_collision_z_m"]) < 1e-7 for foot in feet.values()), "Both cold soles must touch ground")
    _require(model.neq == model.nmocap == 0 and not np.any(data.qfrc_applied) and not np.any(data.xfrc_applied),
             "Cold initialization has external assistance")
    return data, {"root_pose7": data.qpos[address:address+7].tolist(), "BODY29": angles.tolist(),
        "hands14": {name: 0. for name in hand_joint_names}, "qvel_all_zero": True,
        "initialization_only": True, "state_restored_after_first_physics_step": False,
        "ground_alignment": "single cold free-root Z from actual collision sole FK", **feet}


def model_evidence(model):
    names = BODY_JOINT_NAMES + HAND_JOINT_NAMES
    return {"dimensions": {name: int(getattr(model, name)) for name in ("nbody", "njnt", "nq", "nv", "nu", "ngeom", "nmesh", "neq", "nmocap")},
        "robot_mass_kg": float(np.sum(model.body_mass)),
        "options": {name: np.asarray(getattr(model.opt, name)).tolist() for name in
            ("timestep", "integrator", "solver", "cone", "impratio", "noslip_iterations", "iterations", "ls_iterations", "tolerance", "gravity")},
        "actuator_mapping": actuator_mapping(model),
        "joint_properties": {name: {"range": model.joint(name).range.tolist(),
            "armature": float(model.dof_armature[int(model.joint(name).dofadr[0])]),
            "damping": float(model.dof_damping[int(model.joint(name).dofadr[0])]),
            "frictionloss": float(model.dof_frictionloss[int(model.joint(name).dofadr[0])])} for name in names}}


def audit_locomotion(samples50, force_guard200, *, movement_world,
                     walk_tick_range=(100, 400), stop_tick_range=(400, 500), warmup_ticks50=50,
                     ground_z_m=0.):
    """Pure fail-closed gates over dense actual50Hz samples and200Hz force guards.

    Tick ranges are half-open, in ``post_warmup_tick50`` coordinates. Progress
    includes the real state immediately before the first walking interval.
    Last0.5s stopping uses the maximum measured world-XY speed, not its mean.
    Passing is a walking-control diagnostic, never human/data admission.
    """
    report = {"contract": "cvpr-sonic-actual-walk-physical-gates-v1", "passed": False,
        "checks": {}, "metrics": {}, "reasons": [], "accepted_new_dataset": False,
        "human_following_validated": False, "planner_forecast_used_as_actual": False}
    def check(name, passed, reason):
        report["checks"][name] = bool(passed)
        if not passed:
            report["reasons"].append(reason)
    try:
        direction = _finite(movement_world, (3,), "world direction")
        _require(abs(direction[2]) < 1e-12 and np.linalg.norm(direction[:2]) > 1e-9, "Require horizontal nonzero movement")
        direction = direction[:2] / np.linalg.norm(direction[:2])
        ground = _number(ground_z_m, "ground height")
        _require(type(warmup_ticks50) is int and warmup_ticks50 >= 50, "Require at least1s real warmup")
        for limits in (walk_tick_range, stop_tick_range):
            _require(len(limits) == 2 and all(type(x) is int for x in limits) and 0 <= limits[0] < limits[1], "Invalid half-open tick range")
        start, end = walk_tick_range
        stop_start, stop_end = stop_tick_range
        _require(stop_start == end and stop_end-stop_start >= 25, "Require a contiguous stop segment of at least0.5s")
        samples, forces = list(samples50), list(force_guard200)
        _require(len(samples) >= warmup_ticks50+stop_end, "Recorded sample prefix is incomplete")
        times, roots, velocities, tilts, clearance = [], [], [], [], {"left": [], "right": []}
        for index, row in enumerate(samples):
            _require(type(row["global_tick50"]) is int and row["global_tick50"] == index, "50Hz global tick gap")
            _require(type(row["post_warmup_tick50"]) is int and row["post_warmup_tick50"] == index-warmup_ticks50, "50Hz post-warmup tick gap")
            _require(type(row["physics_state_row"]) is int and row["physics_state_row"] == 4*(index+1), "50Hz sample not bound to4 physical substeps")
            times.append(_number(row["sim_time_s"], "sample time"))
            pose = _finite(row["actual_root_pose7"], (7,), "actual root pose")
            _require(abs(np.linalg.norm(pose[3:])-1) < 1e-5, "Root quaternion is not unit wxyz")
            roots.append(pose)
            velocities.append(_finite(row["actual_root_qvel6"], (6,), "actual root velocity"))
            tilt = math.acos(float(np.clip(1-2*(pose[4]**2+pose[5]**2), -1, 1)))
            _require(abs(_number(row["pelvis_tilt_rad"], "pelvis tilt")-tilt) < 1e-6, "Pelvis tilt differs from actual root quaternion")
            tilts.append(tilt)
            for side in clearance:
                foot = row["feet"][side]
                lower = _finite(foot["world_collision_bounds_min_m"], (3,), "foot lower bounds")
                upper = _finite(foot["world_collision_bounds_max_m"], (3,), "foot upper bounds")
                _require(np.all(upper >= lower) and abs(_number(foot["minimum_collision_z_m"], "sole z")-lower[2]) < 1e-9, "Invalid actual collision bounds")
                _finite(foot["ankle_world_xyz"], (3,), "actual ankle position")
                height = float(lower[2]-ground)
                if "ground_clearance_m" in foot:
                    _require(abs(_number(foot["ground_clearance_m"], "foot clearance")-height) < 1e-9, "Foot clearance uses a different ground")
                clearance[side].append(height)
        times = np.asarray(times)
        _require(np.allclose(np.diff(times), .02, rtol=0, atol=1e-6), "50Hz sample clock gap")
        check("dense_actual_50hz", True, "")
        _require(len(forces) == 4*len(samples), "200Hz force guard is incomplete")
        origin = times[0]-.02
        for index, row in enumerate(forces):
            _require(type(row["physics_step200"]) is int and row["physics_step200"] == index+1, "200Hz force step gap")
            for phase, offset in (("before", index), ("after", index+1)):
                state = row[phase]
                _require(abs(_number(state["sim_time_s"], "force time")-(origin+offset/200)) < 1e-6, "200Hz force clock gap")
                _require(type(state["neq"]) is int and type(state["nmocap"]) is int and state["neq"] == state["nmocap"] == 0,
                         "Equality/mocap assistance is present")
                _require(_number(state["maximum_xfrc_applied"], "external spatial force") == 0
                         and _number(state["maximum_qfrc_applied"], "external generalized force") == 0, "External force is nonzero")
        check("dense_zero_force_200hz", True, "")
        roots, velocities = np.asarray(roots), np.asarray(velocities)
        base, finish = warmup_ticks50+start-1, warmup_ticks50+end-1
        displacement = roots[finish, :3]-roots[base, :3]
        progress = float(np.dot(displacement[:2], direction))
        speed = np.linalg.norm(velocities[warmup_ticks50+stop_end-25:warmup_ticks50+stop_end, :2], axis=1)
        report["metrics"].update(walk_tick_range=list(walk_tick_range), stop_tick_range=list(stop_tick_range),
            walk_actual_displacement_xyz_m=displacement.tolist(), signed_direction_progress_m=progress,
            stop_last_0p5s_max_xy_speed_m_s=float(speed.max()),
            minimum_actual_root_height_m=float(roots[:, 2].min()), maximum_actual_tilt_deg=float(np.degrees(tilts).max()),
            samples50=len(samples), force_samples200=len(forces), feet={})
        check("actual_direction_progress", progress >= .1, "Actual direction progress is below0.1m")
        check("stopped_last_0p5s", bool(np.all(speed < .1)), "Actual planar speed has not stayed below0.1m/s for the final0.5s")
        check("root_height", bool(np.all(roots[:, 2] > .6)), "Actual root height is not above0.6m")
        check("upright", bool(np.all(np.degrees(tilts) < 25)), "Actual pelvis tilt reaches25degrees")
        for side, values in clearance.items():
            values = np.asarray(values)
            lifted = np.flatnonzero(values[warmup_ticks50+start:warmup_ticks50+end] > .01)+warmup_ticks50+start
            completed_swing = any(np.any(values[base:int(at)] <= .005)
                and np.any(values[int(at)+1:warmup_ticks50+stop_end] <= .005) for at in lifted)
            report["metrics"]["feet"][side] = {"maximum_walk_ground_clearance_m": float(values[warmup_ticks50+start:warmup_ticks50+end].max()),
                "lifted_samples50": len(lifted), "lifted_over1cm_and_landed": bool(completed_swing)}
            check(side+"_foot_swing_and_landing", bool(completed_swing), side+" foot lacks measured ground→over1cm→ground sequence")
        report["passed"] = all(report["checks"].values())
    except (KeyError, TypeError, ValueError, IndexError) as error:
        check("evidence_integrity", False, f"{type(error).__name__}: {error}")
    return report
