# Portions derived from SIMPLE, Copyright (c) 2025 Songlin Wei and Contributors.
# Source snapshots are retained without changing their numerical acceptance rules.
"""Read-only outer-container footprint diagnostics for physical replay.

Compiled MuJoCo mesh vertices are already scaled, recentered and rotated.
Applying geom_xmat/geom_xpos reconstructs their world coordinates; applying
mesh_pos/mesh_quat again would double-count the compiler's transform.

These are additional necessary geometry/support gates, not official task rules
or a complete test of the basket's exact inner cavity.
"""

from __future__ import annotations

SOURCE_SHA256 = 'f91666f18b8bea796cc3a9d0d9e24c2dabf17a81a45a723e0115c3f73e1ba26a'

from numbers import Integral
from typing import Any

import numpy as np


def _array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"container geometry {name} must be finite with shape {shape}")
    return array


def _body_id(model, value) -> int:
    if isinstance(value, str):
        try:
            body = int(model.body(value).id)
        except (KeyError, ValueError) as error:
            raise ValueError(f"Unknown container geometry body {value!r}") from error
    elif isinstance(value, Integral) and not isinstance(value, bool):
        body = int(value)
    elif hasattr(value, "id"):
        body = int(value.id)
    else:
        raise ValueError("container geometry body must be a body name or id")
    if not 0 < body < model.nbody:
        raise ValueError("container geometry requires a non-world body")
    return body


def _descendants(model, root: int) -> list[int]:
    result = []
    for candidate in range(1, model.nbody):
        current = candidate
        while current and current != root:
            current = int(model.body_parentid[current])
        if current == root:
            result.append(candidate)
    return result


def _collision_vertices_world(model, data, root: int):
    import mujoco

    body_ids = _descendants(model, root)
    explicit_pairs = set(np.asarray(model.pair_geom1).tolist()) | set(np.asarray(model.pair_geom2).tolist())
    geom_ids = [g for g in range(model.ngeom)
                if int(model.geom_bodyid[g]) in body_ids
                and (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0 or g in explicit_pairs)]
    pieces, mesh_count = [], 0
    for geom in geom_ids:
        geom_type = int(model.geom_type[geom])
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = np.asarray(model.mesh_vert[start:start + count], dtype=np.float64)
            mesh_count += 1
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            # Exact primitive-box support; curved primitives deliberately fail
            # rather than silently omitting collision geometry from the hull.
            signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
            vertices = signs * np.asarray(model.geom_size[geom])
        else:
            raise ValueError(f"Unsupported container collision geom type {geom_type}; mesh or box required")
        rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
        pieces.append(vertices @ rotation.T + np.asarray(data.geom_xpos[geom]))
    if not pieces:
        raise ValueError("Container subtree has no supported collision geometry")
    vertices = np.concatenate(pieces)
    if not np.isfinite(vertices).all():
        raise ValueError("Container collision world vertices are not finite")
    return vertices, geom_ids, mesh_count, body_ids


def _hull_xy(vertices: np.ndarray) -> np.ndarray:
    """Andrew monotone-chain convex hull, counterclockwise without duplicate end."""
    points = sorted(set(map(tuple, vertices[:, :2])))
    if len(points) < 3:
        raise ValueError("Container horizontal footprint is degenerate")

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    lower, upper = [], []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        raise ValueError("Container horizontal footprint is collinear")
    return hull


def _status(vertices, target_com_world, inset_margin_m, *, geom_count, mesh_count):
    vertices = np.asarray(vertices, dtype=np.float64)
    target = _array(target_com_world, (3,), "target COM")
    inset = float(inset_margin_m)
    if not np.isfinite(inset) or inset < 0:
        raise ValueError("Container inset margin must be finite and nonnegative")
    hull = _hull_xy(vertices)
    edges = np.roll(hull, -1, axis=0) - hull
    offsets = target[:2] - hull
    # CCW edge cross product is positive in the interior. The minimum
    # supporting-line clearance implements exact half-plane inset testing.
    clearances = (edges[:, 0] * offsets[:, 1] - edges[:, 1] * offsets[:, 0]) / np.linalg.norm(edges, axis=1)
    signed_margin = float(clearances.min())
    return {
        "diagnostic": "target COM inside horizontal convex hull of container collision geometry",
        "inside_container_footprint": signed_margin >= inset,
        "inside_outer_footprint": signed_margin >= 0.0,
        "signed_margin_m": signed_margin,
        "signed_margin_definition": "minimum signed distance to the CCW hull's supporting edge lines; positive inside",
        "required_inset_margin_m": inset,
        "target_com_world_m": target.tolist(),
        "world_bounds_m": {"min": vertices.min(axis=0).tolist(), "max": vertices.max(axis=0).tolist()},
        "world_hull_xy_m": hull.tolist(),
        "collision_geom_count": int(geom_count),
        "collision_mesh_geom_count": int(mesh_count),
        "collision_vertex_count": len(vertices),
        "official_task_flag_modified": False,
        "limitations": [
            "Outer convex footprint is a necessary horizontal gate, not a test of the basket's exact inner cavity.",
            "Target COM inclusion does not prove full object extent fits, correct height, support direction, release, or stability.",
            "Convexification fills concave footprint gaps; use together with contact, release and stability evidence.",
        ],
    }


def container_footprint_status(
    model,
    data,
    *,
    container_body,
    target_body=None,
    target_com_world=None,
    inset_margin_m: float = 0.005,
) -> dict[str, Any]:
    """Inspect current MuJoCo geometry without stepping or changing state.

    The caller must have refreshed forward kinematics after any qpos edits.
    For target_body, use its mass-weighted subtree COM, including descendants;
    target_com_world is an explicit alternative and must actually represent COM.
    """
    root = _body_id(model, container_body)
    if (target_body is None) == (target_com_world is None):
        raise ValueError("Provide exactly one of target_body or target_com_world")
    target = (np.asarray(data.subtree_com[_body_id(model, target_body)])
              if target_body is not None else target_com_world)
    vertices, geom_ids, mesh_count, _ = _collision_vertices_world(model, data, root)
    result = _status(vertices, target, inset_margin_m, geom_count=len(geom_ids), mesh_count=mesh_count)
    result.update(container_root_body_id=root, container_root_body_name=model.body(root).name,
                  target_com_source="mass-weighted target subtree COM" if target_body is not None else "explicit caller-provided COM")
    return result


def build_container_geometry_cache(model, data, *, container_body, target_body=None) -> dict[str, Any]:
    """Cache rigid collision vertices and optional target COM offset for offline poses.

    The JSON-serializable cache is valid only for unchanged rigid geometry.
    A joint below either root is rejected because a single root pose would not
    reconstruct an articulated subtree. Root free joints are permitted.
    """
    root = _body_id(model, container_body)
    vertices, geom_ids, mesh_count, bodies = _collision_vertices_world(model, data, root)
    if any(model.body_jntnum[body] for body in bodies if body != root):
        raise ValueError("Container geometry cache requires a rigid subtree")
    rotation = np.asarray(data.xmat[root]).reshape(3, 3)
    local = (vertices - np.asarray(data.xpos[root])) @ rotation
    cache = {
        "format": "rigid-container-collision-vertices-v1",
        "container_root_body_name": model.body(root).name,
        "root_local_collision_vertices_m": local.tolist(),
        "collision_geom_count": len(geom_ids),
        "collision_mesh_geom_count": mesh_count,
        "collision_geom_ids": geom_ids,
        "upright_axis_local": (rotation.T @ np.array([0., 0., 1.])).tolist(),
        "upright_reference": "world up at caller's initial recorded container pose; assumes that initial pose is upright",
        "note": "Vertices already include MuJoCo compiler mesh transforms and fixed descendant transforms.",
    }
    if target_body is not None:
        target = _body_id(model, target_body)
        if any(model.body_jntnum[body] for body in _descendants(model, target) if body != target):
            raise ValueError("Target COM cache requires a rigid subtree")
        target_rotation = np.asarray(data.xmat[target]).reshape(3, 3)
        cache.update(target_root_body_name=model.body(target).name,
                     target_com_offset_local_m=((np.asarray(data.subtree_com[target]) - data.xpos[target]) @ target_rotation).tolist())
        try:
            target_vertices, target_geoms, _, _ = _collision_vertices_world(model, data, target)
        except ValueError as error:
            if not str(error).startswith("Unsupported container collision geom type"):
                raise
            # Keep the existing COM-only API usable for curved primitives;
            # their unknown exact vertices cannot satisfy the v2 3D gate.
            cache["target_collision_vertices_unavailable_reason"] = str(error)
        else:
            cache.update(target_root_local_collision_vertices_m=((target_vertices - data.xpos[target]) @ target_rotation).tolist(),
                         target_collision_geom_count=len(target_geoms))
    return cache


def _pose(pose, name):
    pose = _array(pose, (7,), name)
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-12:
        raise ValueError(f"Container geometry {name} quaternion has zero norm")
    w, x, y, z = pose[3:] / norm
    rotation = np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y)],
        [2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x)],
        [2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y)],
    ])
    return pose[:3], rotation


def container_footprint_from_poses(
    cache,
    *,
    container_pose_wxyz,
    target_pose_wxyz=None,
    target_com_world=None,
    inset_margin_m: float = 0.005,
) -> dict[str, Any]:
    """Offline equivalent using body-root poses [xyz,qw,qx,qy,qz].

    Use target_pose_wxyz only with a cache containing the compiled target COM
    offset. Passing a target body origin as COM would be an unverified shortcut.
    """
    if cache.get("format") != "rigid-container-collision-vertices-v1":
        raise ValueError("Unrecognized container geometry cache")
    if (target_pose_wxyz is None) == (target_com_world is None):
        raise ValueError("Provide exactly one of target_pose_wxyz or target_com_world")
    local = np.asarray(cache["root_local_collision_vertices_m"], dtype=np.float64)
    if local.ndim != 2 or local.shape[1] != 3 or len(local) < 3 or not np.isfinite(local).all():
        raise ValueError("Invalid cached container vertices")
    position, rotation = _pose(container_pose_wxyz, "container pose")
    vertices = local @ rotation.T + position
    if target_pose_wxyz is not None:
        if "target_com_offset_local_m" not in cache:
            raise ValueError("Target COM offset was not cached; do not substitute a body origin")
        target_position, target_rotation = _pose(target_pose_wxyz, "target pose")
        offset = _array(cache["target_com_offset_local_m"], (3,), "target COM offset")
        target_com_world = target_position + target_rotation @ offset
    result = _status(vertices, target_com_world, inset_margin_m,
                     geom_count=cache["collision_geom_count"], mesh_count=cache["collision_mesh_geom_count"])
    result.update(container_root_body_name=cache["container_root_body_name"],
                  target_com_source="cached mass-weighted target subtree COM offset" if target_pose_wxyz is not None else "explicit caller-provided COM")
    if target_pose_wxyz is not None and "target_root_local_collision_vertices_m" in cache:
        target_vertices = np.asarray(cache["target_root_local_collision_vertices_m"], dtype=np.float64)
        if target_vertices.ndim != 2 or target_vertices.shape[1] != 3 or not np.isfinite(target_vertices).all():
            raise ValueError("Invalid cached target collision vertices")
        target_vertices = target_vertices @ target_rotation.T + target_position
        hull = np.asarray(result["world_hull_xy_m"])
        edges = np.roll(hull, -1, axis=0) - hull
        offsets = target_vertices[:, None, :2] - hull[None]
        clearance = (edges[None, :, 0] * offsets[:, :, 1] - edges[None, :, 1] * offsets[:, :, 0]) / np.linalg.norm(edges, axis=1)[None]
        tzmin, tzmax = float(target_vertices[:, 2].min()), float(target_vertices[:, 2].max())
        czmin, czmax = float(vertices[:, 2].min()), float(vertices[:, 2].max())
        result.update(target_collision_world_bounds_m={"min": target_vertices.min(axis=0).tolist(), "max": target_vertices.max(axis=0).tolist()},
                      collision_vertical_ranges_overlap=tzmax >= czmin - .005 and tzmin <= czmax + .005,
                      collision_vertical_overlap_m=min(tzmax, czmax) - max(tzmin, czmin),
                      target_vertices_minimum_outer_xy_margin_m=float(clearance.min()),
                      target_vertices_inside_outer_xy_fraction=float(np.mean(np.min(clearance, axis=1) >= 0)),
                      target_collision_vertex_count=len(target_vertices),
                      vertex_containment_used_as_acceptance_gate=False)
    if "upright_axis_local" in cache:
        up = _array(cache["upright_axis_local"], (3,), "upright axis")
        if not np.isclose(np.linalg.norm(up), 1.0, atol=1e-6, rtol=0):
            raise ValueError("Cached upright axis is not a unit vector")
        cosine = float(np.clip((rotation @ up)[2], -1, 1))
        result.update(container_upright_cosine=cosine,
                      container_tilt_from_initial_upright_deg=float(np.degrees(np.arccos(cosine))),
                      upright_reference=cache["upright_reference"])
    return result


def container_placement_status(model, data, *, cache, container_body, target_body):
    """Read necessary XY/3D/upright geometry plus actual solved support force.

    Build ``cache`` once at the known upright initial pose. Call on live solved
    dynamics after a physics step; offline pose reconstruction is geometry only.
    Target vertex ranges are measured, but full XY containment is deliberately
    not required: a tall or tilted box may validly extend above/beyond the rim.
    """

    container, target = _body_id(model, container_body), _body_id(model, target_body)
    if cache.get("container_root_body_name") != model.body(container).name or cache.get("target_root_body_name") != model.body(target).name:
        raise ValueError("Placement cache body names do not match the live model")
    result = container_footprint_from_poses(
        cache, container_pose_wxyz=np.r_[data.xpos[container], data.xquat[container]],
        target_pose_wxyz=np.r_[data.xpos[target], data.xquat[target]])
    result.update(target_support_status(model, data, model.body(target).name,
                                        container_label=model.body(container).name))
    result["placement_geometry_measurement"] = "live_collision_vertices_and_solved_contact_force"
    result["limitations"].append("3D extent overlap and upward support are necessary evidence, not a proof of exact inner-cavity mesh containment.")
    return result


def final_placement_geometric_validation(samples, *, window_frames=50,
                                       require_3d_evidence=False,
                                       maximum_container_tilt_deg=30.0,
                                       minimum_support_weight_fraction=.5):
    """Summarize legacy outer XY evidence and the stronger measured v2 gate.

    v2 requires a known upright reference, collision height overlap, and basket
    contact supporting at least half the target weight in >=49/50 final frames.
    Missing live force/geometry evidence fails v2; it is never filled from poses.
    The opt-in preserves legacy callers while they add the new live metrics.
    """
    if isinstance(window_frames, bool) or not isinstance(window_frames, Integral) or window_frames < 1:
        raise ValueError("Placement window_frames must be a positive integer")
    if (not np.isfinite(maximum_container_tilt_deg) or not 0 <= maximum_container_tilt_deg < 90
            or not np.isfinite(minimum_support_weight_fraction) or not 0 < minimum_support_weight_fraction <= 1):
        raise ValueError("Invalid placement v2 thresholds")
    final = samples[-window_frames:]
    geometry = [s.get("container_geometry") for s in final]
    measured = [g for g in geometry if isinstance(g, dict)]
    inside = sum(g.get("inside_container_footprint") is True for g in measured)
    legacy_pass = len(final) == window_frames and len(measured) == window_frames and inside == window_frames
    fields = ("collision_vertical_ranges_overlap", "container_upright_cosine",
              "container_support_weight_fraction", "support_measurement")
    missing = sorted({key for g in geometry for key in fields if not isinstance(g, dict) or key not in g})
    complete = len(final) == window_frames and not missing
    for g in measured:
        try:
            valid = ("collision_vertical_ranges_overlap" not in g or isinstance(g["collision_vertical_ranges_overlap"], bool))
            for key in ("container_upright_cosine", "container_support_weight_fraction"):
                if key in g:
                    valid = valid and np.isscalar(g[key]) and bool(np.isfinite(g[key]))
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Invalid finite placement v2 geometry/support measurements")
    minimum_up = float(np.cos(np.radians(maximum_container_tilt_deg)))
    upright = sum(g.get("container_upright_cosine", -1) >= minimum_up for g in measured)
    overlap = sum(g.get("collision_vertical_ranges_overlap") is True for g in measured)
    support = sum(g.get("support_measurement") == "live_solved_mujoco_contact_force"
                  and g.get("container_support_weight_fraction", -1) >= minimum_support_weight_fraction for g in measured)
    v2_checks = {"outer_xy_com_inside": legacy_pass, "v2_evidence_complete": complete,
                 "container_upright": complete and upright == window_frames,
                 "collision_vertical_ranges_overlap": complete and overlap == window_frames,
                 "basket_supports_target_weight": complete and support >= max(1, window_frames - 1)}
    v2_pass = all(v2_checks.values())
    return {
        "gate_version": "placement-necessary-geometry-v2",
        "passed": v2_pass if require_3d_evidence else legacy_pass,
        "require_3d_evidence": bool(require_3d_evidence),
        "legacy_outer_footprint_passed": legacy_pass,
        "v2_passed": v2_pass, "v2_checks": v2_checks,
        "v2_missing_fields": missing,
        "window_frames": window_frames, "measured_frames": len(measured), "inside_frames": inside,
        "minimum_signed_margin_m": min((g["signed_margin_m"] for g in measured), default=None),
        "upright_frames": upright, "height_overlap_frames": overlap, "weight_supported_frames": support,
        "v2_thresholds": {"maximum_container_tilt_deg": maximum_container_tilt_deg,
                          "minimum_support_weight_fraction": minimum_support_weight_fraction,
                          "minimum_support_frames": max(1, window_frames - 1)},
        "condition": "target mass-weighted COM stays inside the container collision footprint with 5 mm inset",
        "limitation": "Necessary placement evidence; outer XY hull, vertical extent overlap, upright pose and support force do not certify exact inner-cavity containment. Contact handover, release, stability and video remain required.",
    }


# These three live contact/force helpers retain replay_objects source semantics.
OBJECT_SOURCE_SHA256 = '77f0b5373c54026fbd60c9ccc673e2e60cfcb674ad59435b8bf69a2007c17dbe'

def target_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_label: str,
    *,
    container_label: str,
) -> dict[str, bool]:
    """Read contacts of target/container subtrees and complete physical hands.

    The G1 palm collision geom belongs to ``*_wrist_yaw_link``. Matching only
    bodies named ``*_hand_*`` would miss an object still resting on a palm.
    This diagnostic never changes the simulator's official reward/flag.
    """
    if not target_label or not container_label or target_label == container_label:
        raise ValueError("distinct nonempty target_label and container_label are required")
    target = _body_subtree(model, int(model.body(target_label).id))
    container = _body_subtree(model, int(model.body(container_label).id))
    hands = {}
    for side in ("left", "right"):
        try:
            hands[side] = _body_subtree(model, int(model.body(f"{side}_wrist_yaw_link").id))
        except (KeyError, ValueError):
            # Small CPU test models and other embodiments may expose finger
            # bodies without a G1 wrist. They do not establish a palm subtree.
            hands[side] = {b for b in range(1, model.nbody)
                           if model.body(b).name.startswith(f"{side}_hand")}
    result = {"left_hand": False, "right_hand": False, "container": False}
    for index in range(data.ncon):
        contact = data.contact[index]
        body1 = int(model.geom_bodyid[contact.geom1])
        body2 = int(model.geom_bodyid[contact.geom2])
        others = []
        if body1 in target:
            others.append(body2)
        if body2 in target:
            others.append(body1)
        for other in others:
            result["left_hand"] |= other in hands["left"]
            result["right_hand"] |= other in hands["right"]
            result["container"] |= other in container
    return result

def _body_subtree(model: mujoco.MjModel, root: int) -> set[int]:
    result = set()
    for body in range(1, model.nbody):
        ancestor = body
        while ancestor and ancestor != root:
            ancestor = int(model.body_parentid[ancestor])
        if ancestor == root:
            result.add(body)
    return result

def target_support_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_label: str,
    *,
    container_label: str,
) -> dict[str, Any]:
    """Read solved basket contact force on the target, projected against gravity.

    Call on the live solved dynamics state after stepping. Reconstructing poses
    and calling ``mj_forward`` later does not recover the original forces and
    must not be presented as measured rollout support. Contact-frame forces are
    on geom2; negate when the target is geom1. Frictional support is included.
    """
    import mujoco

    if not target_label or not container_label or target_label == container_label:
        raise ValueError("distinct nonempty target/container labels are required")
    target = _body_subtree(model, int(model.body(target_label).id))
    container = _body_subtree(model, int(model.body(container_label).id))
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    mass = float(np.asarray(model.body_mass)[list(target)].sum())
    if not np.isfinite(gravity).all() or gravity_norm <= 0 or not np.isfinite(mass) or mass <= 0:
        raise ValueError("support evidence requires finite gravity and positive target mass")
    total = np.zeros(3)
    count = 0
    contact_force = np.zeros(6)
    for index in range(data.ncon):
        c = data.contact[index]
        b1, b2 = int(model.geom_bodyid[c.geom1]), int(model.geom_bodyid[c.geom2])
        sign = 1 if b2 in target and b1 in container else -1 if b1 in target and b2 in container else 0
        if not sign:
            continue
        mujoco.mj_contactForce(model, data, index, contact_force)
        total += sign * np.asarray(c.frame).reshape(3, 3).T @ contact_force[:3]
        count += 1
    if not np.isfinite(total).all():
        raise ValueError("nonfinite solved container contact force")
    upward = float(total @ (-gravity / gravity_norm))
    weight = mass * gravity_norm
    return {"support_measurement": "live_solved_mujoco_contact_force",
            "target_mass_kg": mass, "target_weight_n": weight,
            "container_target_contact_count": count,
            "container_force_on_target_world_n": total.tolist(),
            "container_upward_support_force_n": upward,
            "container_support_weight_fraction": upward / weight}
