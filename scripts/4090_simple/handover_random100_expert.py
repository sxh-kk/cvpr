"""Bounded object-feedback Handover teacher with freshly encoded SONIC1.1 actions.

This is a new task controller, not a claim that fixed world-wrist IK already
solves Handover. Only private FK/IK data and references are changed. Acceptance
remains the collector's unchanged solved-contact/geometry/stability audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from feedback_wrist_ik_v11 import WorldWristIK, clipped_vector, rotation_error, rotation_from_vector
from prepare_carrybox_v11 import (ENCODER_SHA, FUTURE_OFFSETS, SIMPLE_HAND_NAMES,
                                 TRAIN_STATE_NAMES, encoder_input, matrix_wxyz, sha256)

PHASES = ("approach", "right_lift", "handover", "left_transfer", "placement", "release", "hold")


def finite(value, shape, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite with shape {shape}")
    return result.copy()


@dataclass(frozen=True)
class ExpertConfig:
    contact_dwell_s: float = .08
    minimum_lift_m: float = .05
    object_feedback_gain: float = .8
    feedback_delta_limit_m: float = .035
    joint_delta_limit_rad: float = .45
    position_delta_limit_m: float = .12
    orientation_delta_limit_rad: float = np.deg2rad(20.)
    ik_iterations: int = 15
    release_seconds: float = .4
    release_support_gap_seconds: float = .12
    release_wrist_retreat_m: float = .12
    release_wrist_retreat_speed_m_s: float = .08
    maximum_delta_velocity_rad_s: float = 2.
    placement_xy_tolerance_m: float = .025
    grasp_alignment_tolerance_m: float = .03
    grasp_close_seconds: float = .25
    left_grip_dwell_s: float = .20
    right_release_seconds: float = .3
    placement_xy_speed_m_s: float = .08
    placement_vertical_speed_m_s: float = .025
    placement_clearance_m: float = .025
    placement_servo_tracking_margin_m: float = .035
    placement_support_seek_depth_m: float = .035
    placement_support_dwell_s: float = .20
    placement_geometry_inset_m: float = .015
    placement_ik_iterations: int = 45
    placement_joint_step_limit_rad: float = .06
    placement_joint_velocity_limit_rad_s: float = 1.5
    placement_posture_prior_gain: float = .025
    placement_torso_velocity_limit_rad_s: float = .35
    placement_COM_margin_m: float = .005
    placement_footprint_margin_m: float = .01
    placement_goal_interior_slack_m: float = .01
    placement_rotation_speed_rad_s: float = .45
    placement_orientation_tolerance_rad: float = .15
    placement_locomotion: str = "source_future"
    placement_encoder_mode: str = "g1"
    placement_orientation_feedback_gain: float = 1.5
    placement_orientation_feedback_limit_rad: float = .25
    placement_orientation_feedback_speed_rad_s: float = .6
    placement_position_integral_gain_s_inv: float = .5
    placement_position_integral_limit_m: float = .08
    placement_position_integral_speed_m_s: float = .02
    hand_condition: str = "inferred_effective"

    def __post_init__(self):
        for name, value in vars(self).items():
            if name == "placement_encoder_mode":
                if value not in ("g1", "teleop"):
                    raise ValueError("Select g1 or teleop placement encoder mode")
                continue
            if name == "placement_locomotion":
                if value not in ("source_future", "actual_hold"):
                    raise ValueError("Select source_future or actual_hold placement locomotion")
                continue
            if name == "hand_condition":
                if value not in ("declared", "inferred_effective"):
                    raise ValueError("Select explicit declared or inferred_effective hands")
                continue
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"Expert configuration {name} must be finite and positive")
        if type(self.ik_iterations) is not int or self.ik_iterations > 50:
            raise ValueError("Use at most50 integer IK iterations")
        if type(self.placement_ik_iterations) is not int or self.placement_ik_iterations > 100:
            raise ValueError("Use at most100 integer placement IK iterations")


class BoundedPositionIntegral:
    """Slow reference correction; never a simulator or force modification."""
    def __init__(self, config):
        self.config = config
        self.value = np.zeros(3, dtype=np.float64)

    def step(self, error, dt, active_axes, *, reset=False):
        error = finite(error, (3,), "Actual object COM tracking error")
        axes = np.asarray(active_axes)
        if axes.shape != (3,) or axes.dtype != np.bool_ or not np.isfinite(dt) or dt <= 0:
            raise ValueError("Integral requires boolean XYZ gates and positive elapsed time")
        previous = self.value.copy()
        attempted_increment = np.where(axes, error * self.config.placement_position_integral_gain_s_inv * dt, 0.)
        speed_clamped = np.linalg.norm(attempted_increment) > self.config.placement_position_integral_speed_m_s * dt
        offset_clamped = False
        if reset:
            self.value[:] = 0.
        else:
            increment = clipped_vector(attempted_increment, self.config.placement_position_integral_speed_m_s * dt)
            offset_clamped = np.linalg.norm(self.value + increment) > self.config.placement_position_integral_limit_m
            self.value = clipped_vector(self.value + increment, self.config.placement_position_integral_limit_m)
        return self.value.copy(), {"error_world_m": error.tolist(), "active_XYZ": axes.tolist(),
            "reset": bool(reset), "previous_offset_world_m": previous.tolist(),
            "offset_world_m": self.value.tolist(),
            "attempted_increment_world_m": attempted_increment.tolist(),
            "applied_increment_world_m": (self.value - previous).tolist(),
            "speed_clamped": bool(speed_clamped and not reset), "offset_clamped": bool(offset_clamped),
            "antiwindup": "reset" if reset else "radial_clamp" if offset_clamped else "axis_gate" if not axes.all() else "none",
            "maximum_offset_norm_m": self.config.placement_position_integral_limit_m,
            "maximum_increment_speed_m_s": self.config.placement_position_integral_speed_m_s,
            "reference_only_no_live_state_or_force_change": True}


def physical_feedback(feedback):
    """Missing solved contact/support cannot authorize a phase transition."""
    feedback = feedback or {}
    contacts, geometry = feedback.get("contacts", {}), feedback.get("container_geometry", {})
    def contact(name, alias):
        result = contacts.get(name, contacts.get(alias, False))
        if not isinstance(result, (bool, np.bool_)):
            raise ValueError("Physical contact flags must be boolean")
        return bool(result)
    fraction = geometry.get("container_support_weight_fraction", 0.)
    if not np.isfinite(fraction):
        raise ValueError("Nonfinite actual support feedback")
    return {"left": contact("left_hand", "left"), "right": contact("right_hand", "right"),
            "container": contact("container", "container"),
            "inside": geometry.get("inside_container_footprint") is True,
            "supported": geometry.get("support_measurement") == "live_solved_mujoco_contact_force"
                         and fraction >= .5,
            "support_fraction": float(fraction),
            "complete_projection_inside": float(geometry.get("target_vertices_minimum_outer_xy_margin_m", -1.)) >= .005}


def solved_contact_durations(feedback):
    """Trailing consecutive real50Hz contact durations, when supplied by runner."""
    rows = (feedback or {}).get("recent_contact_samples")
    if rows is None:
        return None
    if not rows:
        return {name: 0. for name in ("right_only", "both", "left_only", "left", "right", "supported", "unsupported", "released_supported", "neither")}
    evidence = []
    previous = None
    for row in rows:
        step = row.get("completed_steps")
        if (type(step) is not int or row.get("solver_stage") != "post_mj_step_before_observation_refresh"
                or (previous is not None and step != previous + 1)):
            raise ValueError("Contact history must be consecutive solved50Hz samples")
        previous = step;e = physical_feedback(row)
        supported = e["inside"] and e["container"] and e["supported"] and e["complete_projection_inside"]
        evidence.append({"right_only": e["right"] and not e["left"], "both": e["left"] and e["right"],
            "left_only": e["left"] and not e["right"], "left": e["left"], "right": e["right"],
            "supported": supported, "unsupported": not supported,
            "released_supported": supported and not e["left"] and not e["right"],
            "neither": not e["left"] and not e["right"]})
    result = {}
    for name in evidence[-1]:
        count = 0
        for row in reversed(evidence):
            if not row[name]:break
            count += 1
        result[name] = count / 50.
    return result


def yaw_rotation(rotation):
    # A toppled object's local X can be vertical. Use the better horizontal
    # axis projection so floating-point noise cannot create a180-degree yaw.
    if np.linalg.norm(rotation[:2, 0]) >= np.linalg.norm(rotation[:2, 1]):
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        yaw = np.arctan2(-rotation[0, 1], rotation[1, 1])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def foot_contact_support_polygon(model, data):
    """Actual contact geometry only; never claim solved support force here."""
    def subtree(root):
        result = set()
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != root:ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:result.add(body)
        return result
    robot = subtree(model.body("pelvis").id)
    feet = subtree(model.body("left_ankle_roll_link").id) | subtree(model.body("right_ankle_roll_link").id)
    points = []
    for i in range(data.ncon):
        contact = data.contact[i];a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
        if ((a in feet and b not in robot) or (b in feet and a not in robot)) and contact.dist <= .003 and abs(contact.frame[2]) >= .8:
            points.append(tuple(np.asarray(contact.pos[:2], dtype=float)))
    points = sorted(set(points))
    if len(points) < 3:return None
    def cross(o, a, b):return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    low, high = [], []
    for point in points:
        while len(low) >= 2 and cross(low[-2], low[-1], point) <= 0:low.pop()
        low.append(point)
    for point in reversed(points):
        while len(high) >= 2 and cross(high[-2], high[-1], point) <= 0:high.pop()
        high.append(point)
    result = np.asarray(low[:-1] + high[:-1], dtype=float)
    return result if len(result) >= 3 else None


def support_margin(polygon, xy):
    edge = np.roll(polygon, -1, axis=0) - polygon
    return float(np.min((edge[:, 0] * (xy[1] - polygon[:, 1]) - edge[:, 1] * (xy[0] - polygon[:, 0])) /
                        np.maximum(np.linalg.norm(edge, axis=1), 1e-12)))


def clip_halfspace(polygon, normal, boundary):
    result = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        da, db = float(normal @ a - boundary), float(normal @ b - boundary)
        if da >= -1e-10:result.append(a)
        if (da >= 0) != (db >= 0):result.append(a + (b - a) * da / (da - db))
    return np.asarray(result, dtype=float).reshape(-1, 2)


def eroded_COM_region(container_hull, target_offsets_world, margin):
    """Exact convex halfspaces for every real projected target vertex."""
    hull = np.asarray(container_hull, dtype=float)
    target = np.asarray(target_offsets_world, dtype=float)
    if (hull.ndim != 2 or hull.shape[1] != 2 or len(hull) < 3
            or target.ndim != 2 or target.shape[1] != 3 or not np.isfinite(hull).all() or not np.isfinite(target).all()):
        raise ValueError("Require finite actual hull and target collision vertices")
    area = np.sum(hull[:, 0] * np.roll(hull[:, 1], -1) - np.roll(hull[:, 0], -1) * hull[:, 1])
    if area < 0:hull = hull[::-1]
    region = hull.copy();normals, boundaries = [], []
    for point, following in zip(hull, np.roll(hull, -1, axis=0)):
        edge = following - point;length = np.linalg.norm(edge)
        if length < 1e-10:continue
        inward = np.array([-edge[1], edge[0]]) / length
        boundary = float(inward @ point + margin - np.min(target[:, :2] @ inward))
        normals.append(inward);boundaries.append(boundary)
        region = clip_halfspace(region, inward, boundary)
        if len(region) < 3:raise ValueError("No COM region can contain the real target footprint at the planned upright orientation")
    return region, np.asarray(normals), np.asarray(boundaries)


def project_to_convex_region(point, polygon):
    point = np.asarray(point, dtype=float)
    if support_margin(polygon, point) >= -1e-10:return point.copy()
    candidates = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a;t = np.clip((point - a) @ edge / max(float(edge @ edge), 1e-15), 0., 1.)
        candidates.append(a + t * edge)
    return min(candidates, key=lambda p: np.linalg.norm(p - point)).copy()


def geometry_upright_rotation(neutral_rotation, local_offsets, hull, current_rotation):
    """Align the neutral object's long footprint axis to the basket's axis."""
    neutral = finite(neutral_rotation, (3, 3), "Source neutral object rotation")
    def axis(points):
        centered = points - points.mean(axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        direction = vectors[:, -1]
        return float(np.arctan2(direction[1], direction[0]))
    target_angle = axis((local_offsets @ neutral.T)[:, :2])
    basket_angle = axis(np.asarray(hull))
    candidates = []
    for half_turn in range(-2, 3):
        yaw = basket_angle - target_angle + half_turn * np.pi
        candidate = rotation_from_vector(np.array([0., 0., yaw])) @ neutral
        candidates.append(candidate)
    return min(candidates, key=lambda r: np.linalg.norm(rotation_error(r, current_rotation)))


def container_goal_geometry(feedback, config):
    """Independent necessary geometry goal; never use a source target waypoint."""
    geometry = (feedback or {}).get("container_geometry", {})
    bounds = geometry.get("world_bounds_m", {})
    lower = finite(bounds.get("min"), (3,), "Measured container lower world bound")
    upper = finite(bounds.get("max"), (3,), "Measured container upper world bound")
    if np.any(upper <= lower):
        raise ValueError("Measured container world bounds must have positive extent")
    center = (lower[:2] + upper[:2]) / 2
    hull = geometry.get("world_hull_xy_m")
    margin = None
    if hull is not None:
        polygon = np.asarray(hull, dtype=np.float64)
        if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3 or not np.isfinite(polygon).all():
            raise ValueError("Measured container hull must contain finite world XY vertices")
        following = np.roll(polygon, -1, axis=0)
        cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
        area2 = float(cross.sum())
        if abs(area2) < 1e-10:
            raise ValueError("Measured container hull is degenerate")
        center = ((polygon + following) * cross[:, None]).sum(axis=0) / (3 * area2)
        edge = following - polygon
        distances = np.sign(area2) * (edge[:, 0] * (center[1] - polygon[:, 1]) - edge[:, 1] * (center[0] - polygon[:, 0])) / np.maximum(np.linalg.norm(edge, axis=1), 1e-12)
        margin = float(distances.min())
        if margin < config.placement_geometry_inset_m:
            raise ValueError("Container geometry has no verified safe centroid inset")
    com = finite(geometry.get("target_com_world_m"), (3,), "Measured target world COM")
    target_bounds = geometry.get("target_collision_world_bounds_m", {})
    target_lower = finite(target_bounds.get("min"), (3,), "Measured target lower collision bound")
    below_com = float(com[2] - target_lower[2])
    if below_com <= 0 or below_com > .5:
        raise ValueError("Invalid measured target COM-to-bottom extent")
    return {"center_xy": center, "container_bottom_z": float(lower[2]), "container_top_z": float(upper[2]),
            "target_below_com_m": below_com, "actual_com": com, "centroid_inset_m": margin,
            "goal_source": "actual_container_world_collision_hull_centroid_or_bounds_center"}


class ContainerPlacementReference:
    """Carry an observed left grasp to a measured basket, then seek support.

    The grasp transform is latched once. Fallen-object poses never redefine
    that transform or the intended terminal basket location.
    """
    def __init__(self, target_pose, wrist_points, wrist_rotations, feedback, config, *, neutral_rotation=None):
        geometry = container_goal_geometry(feedback, config)
        self.config = config
        self.target_rotation = matrix_wxyz(target_pose[3:])
        self.grasp_offset = self.target_rotation.T @ (wrist_points[0] - geometry["actual_com"])
        self.grasp_rotation = self.target_rotation.T @ wrist_rotations[0]
        self.right_point = wrist_points[1].copy();self.right_rotation = wrist_rotations[1].copy()
        self.command_com = geometry["actual_com"].copy()
        self.entry_com = self.command_com.copy();self.elapsed = 0.
        cache = (feedback or {}).get("target_geometry_cache", {})
        vertices = np.asarray(cache.get("target_root_local_collision_vertices_m"), dtype=float)
        offset = finite(cache.get("target_com_offset_local_m"), (3,), "Real target local COM offset")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4 or not np.isfinite(vertices).all():
            raise ValueError("Require real target root-local collision vertices for complete-footprint placement")
        self.local_offsets = vertices - offset
        hull = np.asarray(feedback["container_geometry"]["world_hull_xy_m"], dtype=float)
        self.goal_rotation = geometry_upright_rotation(np.eye(3) if neutral_rotation is None else neutral_rotation,
                                                      self.local_offsets, hull, self.target_rotation)
        self.command_rotation = self.target_rotation.copy()
        self.safe_region, self.region_normals, self.region_boundaries = eroded_COM_region(
            hull, self.local_offsets @ self.goal_rotation.T, config.placement_footprint_margin_m)
        nearest = project_to_convex_region(self.command_com[:2], self.safe_region)
        following = np.roll(self.safe_region, -1, axis=0)
        cross = self.safe_region[:, 0] * following[:, 1] - following[:, 0] * self.safe_region[:, 1]
        centroid = ((self.safe_region + following) * cross[:, None]).sum(axis=0) / (3 * float(cross.sum()))
        # Use the safe region's interior, not its closest boundary. The whole
        # target needs room for measured grasp slip and servo tracking error.
        self.initial_goal_xy = .75 * centroid + .25 * nearest
        self.initial_container_center = geometry["center_xy"].copy()
        self.orientation_feedback = np.zeros(3)
        self.position_integral = BoundedPositionIntegral(config)
        self.release_point = self.release_rotation = None
        self.release_retreat = 0.
        self.rim_clearance_observed = (geometry["actual_com"][2] - geometry["target_below_com_m"] >=
                                       geometry["container_top_z"] + config.placement_clearance_m)
        self.descent_started = False
        goal_below = -float(np.min((self.local_offsets @ self.goal_rotation.T)[:, 2]))
        self.hover_z = max(float(self.command_com[2]), geometry["container_top_z"] + goal_below + config.placement_clearance_m + config.placement_servo_tracking_margin_m)
        self.geometry_cache_sha256 = hashlib.sha256(np.ascontiguousarray(np.r_[vertices.ravel(), offset]).tobytes()).hexdigest()
        self.grasp_geometry = {"left_tip_in_target_frame_m": self.grasp_offset.tolist(),
            "left_rotation_in_target_frame": self.grasp_rotation.tolist(),
            "target_rotation_world_at_left_takeover": self.target_rotation.tolist(),
            "target_COM_world_at_left_takeover": geometry["actual_com"].tolist()}

    def step(self, feedback, evidence, dt, *, releasing=False, release_progress=0.):
        geometry = container_goal_geometry(feedback, self.config)
        com = geometry["actual_com"]
        self.elapsed += dt
        hull = np.asarray(feedback["container_geometry"]["world_hull_xy_m"], dtype=float)
        goal_offsets = self.local_offsets @ self.goal_rotation.T
        self.safe_region, self.region_normals, self.region_boundaries = eroded_COM_region(hull, goal_offsets, self.config.placement_footprint_margin_m)
        goal_xy = project_to_convex_region(self.initial_goal_xy + geometry["center_xy"] - self.initial_container_center, self.safe_region)
        angular_delta = clipped_vector(rotation_error(self.goal_rotation, self.command_rotation), self.config.placement_rotation_speed_rad_s * dt)
        self.command_rotation = rotation_from_vector(angular_delta) @ self.command_rotation
        actual_rotation = finite((feedback or {}).get("actual_target_rotation_world", self.target_rotation), (3, 3), "Actual target orientation")
        object_angle_error = rotation_error(self.goal_rotation, actual_rotation)
        requested_feedback = clipped_vector(self.config.placement_orientation_feedback_gain * object_angle_error,
                                             self.config.placement_orientation_feedback_limit_rad)
        self.orientation_feedback += clipped_vector(requested_feedback - self.orientation_feedback,
            self.config.placement_orientation_feedback_speed_rad_s * dt)
        servo_rotation = rotation_from_vector(self.orientation_feedback) @ self.command_rotation
        actual_offsets = self.local_offsets @ actual_rotation.T
        actual_vertex_margin = min(support_margin(hull, com[:2] + point[:2]) for point in actual_offsets)
        orientation_error = float(np.linalg.norm(rotation_error(self.goal_rotation, actual_rotation)))
        xy_error = goal_xy - com[:2]
        rim_clear = com[2] - geometry["target_below_com_m"] >= geometry["container_top_z"] + self.config.placement_clearance_m
        self.rim_clearance_observed = self.rim_clearance_observed or bool(rim_clear)
        projection_aligned = (np.linalg.norm(xy_error) <= self.config.placement_xy_tolerance_m
                              and actual_vertex_margin >= .005)
        if projection_aligned and rim_clear:
            self.descent_started = True
        aligned = projection_aligned and self.descent_started
        if not aligned:
            self.hover_z = max(float(self.entry_com[2]), geometry["container_top_z"] + max(geometry["target_below_com_m"], -float(goal_offsets[:, 2].min())) + self.config.placement_clearance_m + self.config.placement_servo_tracking_margin_m)
        # Losing the actual left grip before basket support holds the last
        # commanded task reference; it does not follow a fallen target down.
        basket_support = evidence["inside"] and evidence["container"] and evidence["supported"]
        lost = not evidence["left"] and not basket_support and not releasing
        previous = self.command_com.copy()
        # A bounded below-floor *reference* permits seeking solved support in
        # the presence of servo error. Actual contact/force stops descent;
        # neither object state nor the physical acceptance plane is altered.
        floor_z = geometry["container_bottom_z"] - float(goal_offsets[:, 2].min()) - self.config.placement_support_seek_depth_m
        if not lost and not releasing:
            # Raise the actual collision bottom above the rim before crossing
            # sideways. Once clear, final-XY error gates descent, never XY.
            if rim_clear or self.descent_started:
                self.command_com[:2] += clipped_vector(goal_xy - self.command_com[:2], self.config.placement_xy_speed_m_s * dt)
            if self.command_com[2] < self.hover_z - .001 and not aligned and not self.descent_started:
                self.command_com[2] = min(self.hover_z, self.command_com[2] + self.config.placement_vertical_speed_m_s * dt)
            elif aligned and not evidence["supported"]:
                self.command_com[2] = max(floor_z, self.command_com[2] - self.config.placement_vertical_speed_m_s * dt)
        tracking_error = self.command_com - com
        integral_axes = np.array([rim_clear or self.descent_started,
                                  rim_clear or self.descent_started,
                                  aligned and self.descent_started], dtype=bool)
        integral_reset = evidence["supported"] or lost or releasing or not evidence["left"]
        integral, integral_diagnostic = self.position_integral.step(
            tracking_error, dt, integral_axes, reset=integral_reset)
        applied_integral = integral.copy()
        applied_integral[~integral_axes] = 0.
        correction = self.config.object_feedback_gain * clipped_vector(tracking_error, self.config.feedback_delta_limit_m)
        if not aligned:
            correction[2] = max(0., correction[2])
        if evidence["supported"] or lost or releasing:
            correction[:] = 0.
        desired_com = self.command_com + correction + applied_integral
        if aligned or self.descent_started:
            desired_com[:2] = project_to_convex_region(desired_com[:2], self.safe_region)
        left_point = desired_com + servo_rotation @ self.grasp_offset
        left_rotation = servo_rotation @ self.grasp_rotation
        retreat = np.clip(self.elapsed / .6, 0., 1.)
        right_point = self.right_point + retreat * np.array([0., -.06, .025])
        if releasing:
            if self.release_point is None:
                self.release_point = finite(feedback["actual_wrist_points_left_right"], (2, 3), "Actual release wrist points")[0]
                self.release_rotation = finite(feedback["actual_wrist_rotations_left_right"], (2, 3, 3), "Actual release wrist rotations")[0]
            # Open at the measured support pose before withdrawing. Lifting
            # a still-closed hand unloads the basket and causes regrasp.
            if release_progress >= .999:
                self.release_retreat = min(self.config.release_wrist_retreat_m,
                    self.release_retreat + self.config.release_wrist_retreat_speed_m_s * dt)
            left_point = self.release_point + np.array([0., 0., self.release_retreat])
            left_rotation = self.release_rotation.copy()
        else:
            self.release_point = self.release_rotation = None
            self.release_retreat = 0.
        diagnostic = {"contract": "cvpr-complete-footprint-placement-control-v5", "goal_source": "real_collision_vertices_halfspace_eroded_COM_region",
            "terminal_container_goal_xy_m": goal_xy.tolist(),
            "container_collision_centroid_xy_m": geometry["center_xy"].tolist(),
            "safe_complete_footprint_COM_region_xy_m": self.safe_region.tolist(),
            "complete_footprint_planning_margin_m": self.config.placement_footprint_margin_m,
            "actual_complete_footprint_margin_m": float(actual_vertex_margin),
            "bounded_position_integral": integral_diagnostic,
            "applied_position_integral_world_m": applied_integral.tolist(),
            "release_wrist_pose_latched": self.release_point is not None,
            "release_wrist_retreat_m": self.release_retreat,
            "release_wrist_retreat_waits_for_fully_open_command": True,
            "upright_target_rotation_world": self.goal_rotation.tolist(),
            "command_target_rotation_world": servo_rotation.tolist(),
            "base_upright_rotation_reference_world": self.command_rotation.tolist(),
            "bounded_object_orientation_feedback_rotvec_world": self.orientation_feedback.tolist(),
            "actual_object_orientation_error_rotvec_world": object_angle_error.tolist(),
            "actual_target_orientation_error_rad": orientation_error,
            "real_target_geometry_cache_sha256": self.geometry_cache_sha256,
            "actual_target_COM_world_m": com.tolist(), "command_target_COM_world_m": self.command_com.tolist(),
            "previous_command_target_COM_world_m": previous.tolist(), "desired_target_COM_with_feedback_m": desired_com.tolist(),
            "terminal_target_COM_floor_z_m": floor_z, "container_top_z_m": geometry["container_top_z"],
            "hover_target_COM_z_m": self.hover_z, "actual_terminal_XY_error_m": float(np.linalg.norm(xy_error)),
            "physical_rim_clearance_m": self.config.placement_clearance_m,
            "command_servo_tracking_margin_m": self.config.placement_servo_tracking_margin_m,
            "below_floor_support_seek_reference_depth_m": self.config.placement_support_seek_depth_m,
            "actual_terminal_XY_aligned": bool(aligned), "descent_gated_by_actual_container_XY": not aligned,
            "actual_target_bottom_clears_container_rim": bool(rim_clear),
            "actual_rim_clearance_observed_before_descent": self.rim_clearance_observed,
            "descent_started_after_actual_XY_projection_and_current_rim_clear": self.descent_started,
            "object_orientation_is_descent_gate": False,
            "XY_transport_gated_only_by_actual_rim_clearance": not (rim_clear or aligned),
            "left_grasp_lost_before_support": bool(lost), "latched_actual_left_grasp": self.grasp_geometry,
            "source_target_waypoint_used_as_terminal_goal": False,
            "source_style_clock_controls_task_XY": False, "physical_success_certified": False}
        return np.asarray([left_point, right_point]), np.asarray([left_rotation, self.right_rotation]), diagnostic


class ObjectRelativeReference:
    """Pure reference planner; source time advances only within contact gates."""
    def __init__(self, points, rotations, objects, hands, *, stage_frames=None, config=None):
        self.config = config or ExpertConfig()
        n = len(points)
        if n < 46:
            raise ValueError("Need at least46 real source samples to initialize a motion style")
        self.points = finite(points, (n, 2, 3), "Source wrist FK")
        self.rotations = finite(rotations, (n, 2, 3, 3), "Source wrist rotations")
        self.objects = {role: finite(objects[role], (n, 7), f"Source {role} pose")
                        for role in ("target", "container")}
        self.object_rotations = {role: matrix_wxyz(value[:, 3:]) for role, value in self.objects.items()}
        self.hands = finite(hands, (n, 14), "Source effective SIMPLE hand commands")
        self.stage_frames = self._anchors(stage_frames)
        self.open_hands = self.hands[0].copy()
        self.grip_hands = self.open_hands.copy()
        self.grip_frames = []
        for side in range(2):
            part = slice(side * 7, side * 7 + 7)
            # Recorded closing direction is retained, without assuming signs or
            # swapping named fingers. This is a reference prior, not contact.
            index = int(np.argmax(np.linalg.norm(self.hands[:, part] - self.open_hands[part], axis=1)))
            self.grip_hands[part] = self.hands[index, part];self.grip_frames.append(index)
        self.phase_index = 0;self.reference_frame = 0
        self.previous_tick = None;self.dwell = 0.;self.release_progress = 0.
        self.initial_actual_objects = None;self.previous_target = None
        self.previous_goals = None;self.alignment_hold = False
        self.grasp_progress = 0.;self.right_release_progress = 0.
        self.contact_streaks = {};self.previous_contact_flags = {}
        self.placement_controller = None

    def _anchors(self, explicit):
        n = len(self.points)
        if explicit is not None:
            values = [int(explicit[name]) for name in PHASES]
            if values[0] != 0 or any(a > b for a, b in zip(values, values[1:])) or values[-1] >= n:
                raise ValueError("Reference phase anchors must be ordered real source indices")
            self.anchor_method = "explicit_source_reference_priors_not_live_phase_evidence"
            return values
        target = self.objects["target"][:, :3]
        lifted = np.flatnonzero(target[:, 2] >= target[0, 2] + self.config.minimum_lift_m)
        lift = int(lifted[0]) if len(lifted) else n // 4
        lift = int(np.clip(lift, 5, n - 31))
        distance = np.linalg.norm(self.points[:, 0] - target, axis=1)
        candidates = np.flatnonzero((np.arange(n) >= lift) & (distance < .16))
        both = int(candidates[0]) if len(candidates) else n // 2
        both = int(np.clip(both, lift + 5, n - 21))
        right_distance = np.linalg.norm(self.points[:, 1] - target, axis=1)
        candidates = np.flatnonzero((np.arange(n) >= both + 5) & (right_distance > distance + .04))
        left = int(candidates[0]) if len(candidates) else 2 * n // 3
        left = int(np.clip(left, both + 5, n - 11))
        self.anchor_method = "source_hand_closing_object_height_and_wrist_distance_priors_not_live_contact_labels"
        placement = min(n - 2, max(left + 5, 3 * n // 4))
        closing = np.linalg.norm(self.hands[:, 7:] - self.hands[0, 7:], axis=1)
        closing_frames = np.flatnonzero(closing >= max(.05, .2 * float(closing.max())))
        grasp = int(closing_frames[0]) if len(closing_frames) else max(0, lift - 10)
        grasp = int(np.clip(grasp, 1, lift - 1))
        return [0, grasp, both, left, placement, n - 1, n - 1]

    def step(self, tick, actual_objects, feedback=None):
        if isinstance(tick, bool) or int(tick) != tick or tick < 0:
            raise ValueError("Elapsed native50Hz tick must be a nonnegative integer")
        tick = int(tick)
        if self.previous_tick is not None and tick <= self.previous_tick:
            raise ValueError("Task controller calls need strictly increasing elapsed ticks")
        actual = {role: finite(actual_objects[role], (7,), f"Actual {role} pose")
                  for role in ("target", "container")}
        rotations = {role: matrix_wxyz(value[3:]) for role, value in actual.items()}
        if self.initial_actual_objects is None:
            self.initial_actual_objects = {role: value.copy() for role, value in actual.items()}
        elapsed = 0 if self.previous_tick is None else tick - self.previous_tick
        dt = max(elapsed, 1) / 50.
        target = actual["target"][:3]
        speed = 0. if self.previous_target is None else np.linalg.norm(target - self.previous_target) / dt
        evidence = physical_feedback(feedback)
        contact_age = solved_contact_durations(feedback)
        supported_now = evidence["inside"] and evidence["container"] and evidence["supported"] and evidence["complete_projection_inside"]
        flags = {"right_only": evidence["right"] and not evidence["left"],
            "both": evidence["left"] and evidence["right"], "left_only": evidence["left"] and not evidence["right"],
            "left": evidence["left"], "right": evidence["right"], "supported": supported_now,
            "unsupported": not supported_now,
            "released_supported": supported_now and not evidence["left"] and not evidence["right"],
            "neither": not evidence["left"] and not evidence["right"]}
        if contact_age is None:
            for name, present in flags.items():
                self.contact_streaks[name] = (self.contact_streaks.get(name, 0.) + min(dt, 1 / 30.)
                    if present and self.previous_contact_flags.get(name) else .02 if present else 0.)
            contact_age = self.contact_streaks.copy()
        self.previous_contact_flags = flags
        raised = target[2] - self.initial_actual_objects["target"][2] >= self.config.minimum_lift_m
        support = evidence["inside"] and evidence["container"] and evidence["supported"] and evidence["complete_projection_inside"]
        conditions = (evidence["right"] and not evidence["left"],
                      evidence["right"] and not evidence["left"] and raised,
                      evidence["right"] and evidence["left"] and raised,
                      evidence["left"] and not evidence["right"] and raised,
                      support and speed <= .1,
                      support and not evidence["left"] and not evidence["right"], False)
        previous_phase = PHASES[self.phase_index]
        previous_reference = self.reference_frame
        if (self.phase_index in (1, 2, 3) and contact_age["neither"] >= .12
                and not raised):
            self.phase_index = 0;self.reference_frame = min(self.reference_frame, self.stage_frames[1])
            self.grasp_progress = 0.;self.right_release_progress = 0.;self.dwell = 0.
        geometry = (feedback or {}).get("container_geometry", {})
        release_geometry_safe = (evidence["inside"] and evidence["complete_projection_inside"]
            and geometry.get("collision_vertical_ranges_overlap") is True
            and float(geometry.get("container_tilt_from_initial_upright_deg", float("inf"))) <= 30.)
        release_grace = (self.phase_index == 5 and not support and release_geometry_safe
            and bool((feedback or {}).get("recent_contact_samples"))
            and contact_age["unsupported"] < self.config.release_support_gap_seconds)
        if self.phase_index == 5 and (not release_geometry_safe or (not support and not release_grace)):
            # Sustained solved support loss or unsafe geometry recloses the
            # hand. A bounded in-basket load transfer gap cannot certify hold.
            self.phase_index = 4;self.release_progress = 0.;self.dwell = 0.
        else:
            gate = ("right_only", "right_only", "both", "left_only", "supported", "released_supported", None)[self.phase_index]
            self.dwell = contact_age[gate] if gate is not None and conditions[self.phase_index] else 0.
            threshold = (self.config.left_grip_dwell_s if self.phase_index == 2 else
                         self.config.placement_support_dwell_s if self.phase_index == 4 else self.config.contact_dwell_s)
            if self.phase_index == 3 and self.right_release_progress < .99:
                self.dwell = 0.
            if self.phase_index < 6 and self.dwell >= threshold:
                self.phase_index += 1;self.dwell = 0.
        if self.phase_index == 5:
            self.release_progress = min(1., self.release_progress + dt / self.config.release_seconds)
        elif self.phase_index == 6:
            self.release_progress = 1.
        end = self.stage_frames[min(self.phase_index + 1, 6)]
        self.reference_frame = min(self.reference_frame + elapsed, end)
        frame = self.reference_frame
        source_target = self.objects["target"][frame]
        pregrasp = self.phase_index <= 1
        target_delta = (yaw_rotation(rotations["target"]) @ yaw_rotation(self.object_rotations["target"][frame]).T
                        if pregrasp else rotations["target"] @ self.object_rotations["target"][frame].T)
        initial_delta = matrix_wxyz(self.initial_actual_objects["target"][3:]) @ self.object_rotations["target"][0].T
        planned_target = self.initial_actual_objects["target"][:3] + initial_delta @ (source_target[:3] - self.objects["target"][0, :3])
        if self.phase_index == 0:
            planned_target = target.copy()
        container_delta = rotations["container"] @ self.object_rotations["container"][frame].T
        placement = self.phase_index >= 4
        error = planned_target - target
        correction = self.config.object_feedback_gain * clipped_vector(error, self.config.feedback_delta_limit_m)
        alignment_hold = placement and np.linalg.norm(error[:2]) > self.config.placement_xy_tolerance_m
        if alignment_hold:
            # Correct XY before commanding any additional descent.
            correction[2] = max(0., correction[2])
        if placement and evidence["supported"]:
            correction[2] = max(0., correction[2])
        goals, goal_rotations, anchors = [], [], []
        for side in range(2):
            point = target + target_delta @ (self.points[frame, side] - source_target[:3])
            rotation = target_delta @ self.rotations[frame, side]
            anchor = "actual_target_relative_plus_source_motion_tracking"
            point = point + correction
            if alignment_hold and self.previous_goals is not None:
                point[2] = max(point[2], self.previous_goals[side, 2])
            if self.phase_index >= 5:
                point[2] += .025 * self.release_progress
            goals.append(point);goal_rotations.append(rotation);anchors.append(anchor)
        placement_diagnostic = None
        if placement:
            feedback = dict(feedback or {})
            feedback["actual_target_rotation_world"] = rotations["target"]
            actual_wrists = finite((feedback or {}).get("actual_wrist_points_left_right"), (2, 3), "Actual takeover wrist FK")
            actual_rotations = finite((feedback or {}).get("actual_wrist_rotations_left_right"), (2, 3, 3), "Actual takeover wrist rotations")
            if self.placement_controller is None:
                self.placement_controller = ContainerPlacementReference(actual["target"], actual_wrists, actual_rotations, feedback, self.config,
                    neutral_rotation=self.object_rotations["target"][0])
            goals, goal_rotations, placement_diagnostic = self.placement_controller.step(feedback, evidence, dt,
                releasing=self.phase_index >= 5, release_progress=self.release_progress)
            anchors = ["latched_actual_left_grasp_to_independent_container_COM", "right_arm_free_to_natural_source_posture"]
            planned_target = np.asarray(placement_diagnostic["command_target_COM_world_m"])
            error = planned_target - np.asarray(placement_diagnostic["actual_target_COM_world_m"])
            correction = np.asarray(placement_diagnostic["desired_target_COM_with_feedback_m"]) - planned_target
            alignment_hold = not placement_diagnostic["actual_terminal_XY_aligned"]
        hands = self.hands[frame].copy()
        wrist_error = None
        if feedback is not None and "actual_wrist_points_left_right" in feedback:
            actual_wrists = finite(feedback["actual_wrist_points_left_right"], (2, 3), "Actual wrist FK")
            wrist_error = np.linalg.norm(actual_wrists - np.asarray(goals), axis=1)
        if self.phase_index == 0 and frame >= self.stage_frames[1]:
            aligned = wrist_error is None or wrist_error[1] <= self.config.grasp_alignment_tolerance_m
            if aligned and speed <= .10:
                self.grasp_progress = min(1., self.grasp_progress + dt / self.config.grasp_close_seconds)
            elif not evidence["right"]:
                self.grasp_progress = max(0., self.grasp_progress - dt / self.config.grasp_close_seconds)
            hands[7:] = (1 - self.grasp_progress) * self.open_hands[7:] + self.grasp_progress * self.grip_hands[7:]
        if 1 <= self.phase_index <= 2:
            hands[7:] = self.grip_hands[7:]
        left_stable = evidence["left"] and contact_age["left"] >= self.config.left_grip_dwell_s and raised and speed <= .10
        if self.phase_index == 3:
            change = dt / self.config.right_release_seconds
            self.right_release_progress = (min(1., self.right_release_progress + change) if left_stable
                                           else max(0., self.right_release_progress - 2 * change))
        elif self.phase_index >= 4:
            self.right_release_progress = 1.
        if self.phase_index >= 3:
            hands[7:] = (1 - self.right_release_progress) * self.grip_hands[7:] + self.right_release_progress * self.open_hands[7:]
            hands[:7] = (1 - self.release_progress) * self.grip_hands[:7] + self.release_progress * self.open_hands[:7]
        # Placement has a new task-controlled horizon. The current real source
        # pose remains its style prior, but future source descent must not leak
        # through an encoder lookahead while the actual object is transporting.
        control_end = frame if placement else end
        future = frame + FUTURE_OFFSETS
        indices = np.minimum(future, control_end)
        synthetic = future > control_end
        if placement:
            synthetic[:] = True
        metadata = {"phase": PHASES[self.phase_index], "previous_phase": previous_phase,
            "phase_transition": previous_phase != PHASES[self.phase_index], "elapsed_native50Hz_tick": tick,
            "previous_reference_frame": previous_reference,
            "actual_reference_frame": frame, "reference_phase_end": end,
            "reference_control_end": control_end,
            "reference_stage_frames": dict(zip(PHASES, self.stage_frames)), "anchor_method": self.anchor_method,
            "physical_feedback": evidence, "actual_target_lift_m": float(target[2] - self.initial_actual_objects["target"][2]),
            "contact_duration_seconds": contact_age,
            "contact_duration_basis": "consecutive_solved50Hz_samples" if (feedback or {}).get("recent_contact_samples") is not None else "conservative_control_rate_observations_not_complete50Hz_history",
            "actual_target_speed_m_s": float(speed), "phase_evidence_dwell_s": self.dwell,
            "release_support_grace_active": bool(release_grace),
            "release_support_loss_duration_s": contact_age["unsupported"],
            "actual_objects_pose7": {role: pose.tolist() for role, pose in actual.items()},
            "planned_target_xyz_m": planned_target.tolist(), "actual_target_error_xyz_m": error.tolist(),
            "task_feedback_delta_xyz_m": correction.tolist(), "alignment_holds_additional_descent": bool(alignment_hold),
            "independent_container_placement": placement_diagnostic,
            "desired_wrist_points_left_right": np.asarray(goals).tolist(),
            "desired_wrist_rotations_left_right": np.asarray(goal_rotations).tolist(), "wrist_anchor_left_right": anchors,
            "release_progress": self.release_progress, "hand_grip_reference_frames_left_right": self.grip_frames,
            "grasp_close_progress": self.grasp_progress, "right_release_progress": self.right_release_progress,
            "left_grip_stable_candidate": bool(left_stable),
            "actual_wrist_goal_errors_m": None if wrist_error is None else wrist_error.tolist(),
            "target_orientation_feedback": "yaw_only_until_stable_grasp_and_lift" if pregrasp else "full_SE3_after_lifted_grasp",
            "encoder_source_reference_indices": indices.tolist(), "generated_control_horizon_mask": synthetic.tolist(),
            "future_reference_semantics": "source style samples plus new task control; phase/tail holds are synthesized references, never real future labels",
            "placement_encoder_horizon": "current_task_corrected_q29_hold_with_current_source_style_prior" if placement else None,
            "task_success_certified": False}
        self.previous_tick = tick;self.previous_target = target.copy()
        self.previous_goals = np.asarray(goals).copy();self.alignment_hold = bool(alignment_hold)
        return np.asarray(goals), np.asarray(goal_rotations), hands, indices, synthetic, dt, metadata


class ObjectWristIK(WorldWristIK):
    """Private FK from the tracked helper; configurable bounded two-arm solve."""
    def solve_current(self, frame, position, rotation, goals, goal_rotations, config):
        source = self.states[frame]
        self._set_reference(source, position, rotation)
        correction, reports = np.zeros(14), []
        for side in range(2):
            indices = np.arange(15 + side * 7, 22 + side * 7)
            qadr, dofs = self.qadr[indices], self.arm_dofs[side]
            original = source[indices]
            low, high = original - config.joint_delta_limit_rad, original + config.joint_delta_limit_rad
            for i, name in enumerate(np.asarray(TRAIN_STATE_NAMES)[indices]):
                joint = self.model.joint(name)
                if bool(joint.limited[0]):
                    low[i] = max(low[i], joint.range[0]);high[i] = min(high[i], joint.range[1])
            if np.any(low > high):
                raise ValueError("Source arm lies beyond the bounded joint correction radius")
            point, orientation = self._pose(side)
            goal = point + clipped_vector(goals[side] - point, config.position_delta_limit_m)
            goal_rotation = rotation_from_vector(clipped_vector(rotation_error(goal_rotations[side], orientation),
                config.orientation_delta_limit_rad)) @ orientation
            before = np.linalg.norm(goals[side] - point)
            jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            iterations = 0
            for iteration in range(config.ik_iterations):
                point, orientation = self._pose(side)
                error = np.r_[goal - point, .15 * rotation_error(goal_rotation, orientation)]
                if np.linalg.norm(error[:3]) < 1e-4 and np.linalg.norm(error[3:]) < .00015:
                    break
                mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.wrists[side])
                jacobian = np.vstack((jacp[:, dofs], .15 * jacr[:, dofs]))
                delta = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + .01 ** 2 * np.eye(6), error)
                # Preserve the descent direction: independent component clipping
                # can destroy cancellation in a nearly singular right-arm pose.
                delta *= min(1., .04 / max(float(np.max(np.abs(delta))), 1e-15))
                original_q = self.data.qpos[qadr].copy();accepted = False
                for scale in (1., .5, .25, .125, .0625):
                    self.data.qpos[qadr] = np.clip(original_q + scale * delta, low, high)
                    mujoco.mj_forward(self.model, self.data)
                    candidate_point, candidate_rotation = self._pose(side)
                    candidate_error = np.r_[goal - candidate_point, .15 * rotation_error(goal_rotation, candidate_rotation)]
                    if np.linalg.norm(candidate_error) < np.linalg.norm(error) - 1e-12:
                        accepted = True;break
                if not accepted:
                    self.data.qpos[qadr] = original_q;mujoco.mj_forward(self.model, self.data);break
                iterations = iteration + 1
            after, _ = self._pose(side)
            correction[side * 7:side * 7 + 7] = self.data.qpos[qadr] - original
            reports.append({"iterations": iterations, "requested_point": goals[side].tolist(), "bounded_point": goal.tolist(),
                "position_error_before_m": float(before), "position_error_after_m": float(np.linalg.norm(goals[side] - after)),
                "bounded_position_residual_m": float(np.linalg.norm(goal - after)),
                "position_goal_clipped": bool(np.linalg.norm(goals[side] - goal) > 1e-9),
                "joint_delta": correction[side * 7:side * 7 + 7].tolist()})
        return correction, reports

    def solve_placement(self, frame, actual_state43, position, rotation, goal, goal_rotation,
                        previous_command29, dt, config, *, support_polygon=None):
        """Full-range left IK from actual posture, then bounded command slew.

        The source is a weak null-space posture prior and supplies the free
        right-arm retreat. It is never a reach bound for the carrying arm.
        """
        actual = finite(actual_state43, (43,), "Actual named state43")
        previous = finite(previous_command29, (29,), "Previous body command29")
        self._set_reference(actual, position, rotation)
        initial_point, initial_rotation = self._pose(0)
        seeded_fk = initial_point.copy()
        pelvis = self.model.body("pelvis").id
        base_com = self.data.subtree_com[pelvis, :2].copy()
        baseline_margin = None if support_polygon is None else support_margin(support_polygon, base_com)
        required_margin = None if baseline_margin is None else min(config.placement_COM_margin_m, baseline_margin)
        # Torso is enabled only with an actual foot-contact support polygon.
        indices = np.arange(12 if support_polygon is not None else 15, 22)
        qadr = self.qadr[indices]
        dofs = [int(self.model.joint(name).dofadr[0]) for name in np.asarray(TRAIN_STATE_NAMES)[indices]]
        dimensions = len(indices)
        lower, upper = np.full(dimensions, -np.inf), np.full(dimensions, np.inf)
        for i, name in enumerate(np.asarray(TRAIN_STATE_NAMES)[indices]):
            joint = self.model.joint(name)
            if bool(joint.limited[0]):lower[i], upper[i] = joint.range
        self.data.qpos[qadr] = np.clip(self.data.qpos[qadr], lower, upper)
        mujoco.mj_forward(self.model, self.data)
        prior = self.states[frame, indices].copy()
        if support_polygon is not None:prior[:3] = actual[12:15]
        jacp, jacr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
        iterations = 0
        for iteration in range(config.placement_ik_iterations):
            point, orientation = self._pose(0)
            error = np.r_[goal - point, .12 * rotation_error(goal_rotation, orientation)]
            if np.linalg.norm(error[:3]) < 2e-4 and np.linalg.norm(error[3:]) < .0003:break
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.wrists[0])
            jacobian = np.vstack((jacp[:, dofs], .12 * jacr[:, dofs]))
            # Re-solve on free joints at an active physical bound. Merely
            # clipping an outward torso step can invalidate a coupled DLS
            # descent direction and stall still-free arm joints.
            free = np.ones(dimensions, dtype=bool)
            current_q = self.data.qpos[qadr].copy()
            for _ in range(dimensions + 1):
                reduced = jacobian[:, free]
                inverse = reduced.T @ np.linalg.solve(reduced @ reduced.T + .012 ** 2 * np.eye(6), np.eye(6))
                free_step = inverse @ error
                free_step += (np.eye(int(free.sum())) - inverse @ reduced) @ (config.placement_posture_prior_gain * (prior[free] - current_q[free]))
                step = np.zeros(dimensions);step[free] = free_step
                blocked = free & (((current_q <= lower + 1e-8) & (step < 0)) | ((current_q >= upper - 1e-8) & (step > 0)))
                if not blocked.any():break
                free[blocked] = False
                if not free.any():break
            step *= min(1., .10 / max(float(np.max(np.abs(step))), 1e-15))
            old = self.data.qpos[qadr].copy();accepted = False
            for scale in (1., .5, .25, .125, .0625):
                self.data.qpos[qadr] = np.clip(old + scale * step, lower, upper)
                mujoco.mj_forward(self.model, self.data)
                trial_point, trial_rotation = self._pose(0)
                trial = np.r_[goal - trial_point, .12 * rotation_error(goal_rotation, trial_rotation)]
                margin_ok = support_polygon is None or support_margin(support_polygon, self.data.subtree_com[pelvis, :2]) >= required_margin - 1e-9
                if margin_ok and np.linalg.norm(trial) < np.linalg.norm(error) - 1e-12:
                    accepted = True;break
            if not accepted:
                self.data.qpos[qadr] = old;mujoco.mj_forward(self.model, self.data);break
            iterations = iteration + 1
        full_point, full_rotation = self._pose(0)
        full_solution = self.data.qpos[qadr].copy()
        full_margin = None if support_polygon is None else support_margin(support_polygon, self.data.subtree_com[pelvis, :2])
        desired = actual[:29].copy()
        desired[indices] = full_solution
        desired[22:29] = self.states[frame, 22:29]
        # Continuous limits apply to the outgoing reference, not to task reach.
        maximum_step = min(config.placement_joint_step_limit_rad, config.placement_joint_velocity_limit_rad_s * dt)
        command = previous + np.clip(desired - previous, -maximum_step, maximum_step)
        torso_step = min(maximum_step, config.placement_torso_velocity_limit_rad_s * dt)
        command[12:15] = previous[12:15] + np.clip(desired[12:15] - previous[12:15], -torso_step, torso_step)
        physical_lower, physical_upper = [], []
        for i, name in enumerate(TRAIN_STATE_NAMES[:29]):
            joint = self.model.joint(name)
            lo, hi = joint.range if bool(joint.limited[0]) else (-np.inf, np.inf)
            command[i] = np.clip(command[i], lo, hi)
            physical_lower.append(float(lo));physical_upper.append(float(hi))
        predicted = actual.copy();predicted[:29] = command
        self._set_reference(predicted, position, rotation)
        command_point, command_rotation = self._pose(0)
        right_point, right_rotation = self._pose(1)
        command_margin = None if support_polygon is None else support_margin(support_polygon, self.data.subtree_com[pelvis, :2])
        full_left = full_solution[-7:]
        reports = [{"mode": "actual_posture_physical_range_torso_left_task_IK", "iterations": iterations,
            "requested_point": np.asarray(goal).tolist(), "bounded_point": np.asarray(goal).tolist(),
            "position_goal_clipped": False, "source_reach_radius_applied": False,
            "actual_posture_seed_wrist_point": seeded_fk.tolist(),
            "position_error_before_m": float(np.linalg.norm(goal - initial_point)),
            "full_physical_range_solution_point": full_point.tolist(),
            "full_physical_range_solution_residual_m": float(np.linalg.norm(goal - full_point)),
            "full_physical_range_orientation_residual_rad": float(np.linalg.norm(rotation_error(goal_rotation, full_rotation))),
            "rate_limited_command_wrist_point": command_point.tolist(),
            "position_error_after_m": float(np.linalg.norm(goal - command_point)),
            "bounded_position_residual_m": float(np.linalg.norm(goal - command_point)),
            "full_solution_arm_q7": full_left.tolist(), "command_arm_q7": command[15:22].tolist(),
            "joint_delta": (command[15:22] - self.states[frame, 15:22]).tolist(),
            "maximum_command_step_rad": maximum_step,
            "actual_maximum_command_step_rad": float(np.max(np.abs(command - previous))),
            "actual_maximum_command_velocity_rad_s": float(np.max(np.abs(command - previous)) / dt),
            "style_prior": "weak source arm/actual torso posture in Jacobian nullspace; no source displacement box"},
            {"mode": "unconstrained_wrist_natural_right_source_posture_with_command_slew",
             "position_goal_clipped": False, "requested_point": right_point.tolist(),
             "position_error_before_m": None, "position_error_after_m": 0.,
             "joint_delta": (command[22:29] - self.states[frame, 22:29]).tolist()}]
        metadata = {"actual_state43_seed": actual.tolist(), "previous_body_command29": previous.tolist(),
            "physical_body_joint_lower": physical_lower, "physical_body_joint_upper": physical_upper,
            "full_range_left_solution_source_deviation_rad": (full_left - self.states[frame, 15:22]).tolist(),
            "full_range_torso_left_solution_q": full_solution.tolist(),
            "full_range_torso_left_solution_indices": indices.tolist(),
            "torso_IK_enabled": support_polygon is not None,
            "foot_support_polygon_xy_m": None if support_polygon is None else support_polygon.tolist(),
            "support_constraint_scope": "contact geometry and quasi-static robot COM projection; not solved-force or dynamic balance certification",
            "actual_robot_COM_margin_m": baseline_margin, "required_robot_COM_margin_m": required_margin,
            "full_solution_robot_COM_margin_m": full_margin, "command_robot_COM_margin_m": command_margin,
            "torso_maximum_command_velocity_rad_s": float(np.max(np.abs(command[12:15] - previous[12:15])) / dt),
            "command_body_velocity29": ((command - previous) / dt).tolist(),
            "source_root_xyz": self.positions[frame].tolist(), "actual_root_xyz": np.asarray(position).tolist(),
            "actual_minus_source_root_xyz_m": (position - self.positions[frame]).tolist(),
            "actual_to_source_root_rotation_error_rad": float(np.linalg.norm(rotation_error(rotation, self.rotations[frame]))),
            "right_wrist_task_constraint_enabled": False, "simulator_state_mutated": False}
        return command, reports, metadata, right_point, right_rotation


class HandoverObjectFeedbackExpert:
    def __init__(self, values, sonic, *, config=None):
        import onnxruntime as ort
        self.config = config or ExpertConfig()
        self.states = np.asarray(values["reference_full_state43"], dtype=np.float64).copy()
        n = len(self.states)
        self.states = finite(self.states, (n, 43), "Measured source state43")
        self.dq = finite(values["reference_full_body_dq"], (n, 29), "Measured source body velocity")
        self.xyz = finite(values["reference_full_root_xyz"], (n, 3), "Measured source root position")
        self.roots = finite(values["reference_full_root_wxyz"], (n, 4), "Measured source root orientation")
        matrix_wxyz(self.roots)
        if tuple(values.get("reference_full_state_joint_names", TRAIN_STATE_NAMES)) != tuple(TRAIN_STATE_NAMES):
            raise ValueError("Source state43 named order differs")
        if tuple(values.get("reference_full_hand_command_names", SIMPLE_HAND_NAMES)) != tuple(SIMPLE_HAND_NAMES):
            raise ValueError("Require effective SIMPLE thumb/index/middle hand order")
        self.objects = values["reference_full_object_poses"]
        self.hand_condition = values.get("hand_condition", self.config.hand_condition)
        if self.hand_condition not in ("declared", "inferred_effective"):
            raise ValueError("Select explicit declared or inferred_effective hands")
        hand_key = "reference_full_declared_hand_commands14" if self.hand_condition == "declared" else "reference_full_hand_commands14"
        self.hands = values[hand_key]
        self.stage_frames = values.get("reference_stage_frames")
        path = Path(sonic) / "model_encoder.onnx"
        if sha256(path) != ENCODER_SHA:
            raise ValueError("Object-feedback expert requires the pinned official SONIC1.1 encoder")
        options = ort.SessionOptions();options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        if list(self.session.get_inputs()[0].shape) != [1, 1751] or list(self.session.get_outputs()[0].shape) != [1, 64]:
            raise ValueError("Official SONIC1.1 encoder input/output contract differs")
        self.ik = self.planner = self.previous_delta = None
        self.previous_body_command = None
        self.target_geometry_cache = None
        self.locomotion_start = None
        self.locomotion_frame = None
        self.locomotion_raw_frame = None
        self.locomotion_last_tick = None
        self.locomotion_root_start = None
        self.locomotion_alignment = None
        self.teleop_official_sources = None
        if self.config.placement_encoder_mode == "teleop":
            from handover_sonic_teleop_reference import verify_official_sources
            self.teleop_official_sources = verify_official_sources()

    def __call__(self, frame, raw, feedback=None):
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        if self.ik is None:
            self.ik = ObjectWristIK(model, self.states, self.xyz, self.roots)
            self.planner = ObjectRelativeReference(self.ik.points, self.ik.orientations, self.objects,
                self.hands, stage_frames=self.stage_frames, config=self.config)
        actual_objects = {}
        for role in ("target", "container"):
            body = raw.mujoco.mj_objects[role]
            actual_objects[role] = np.r_[np.asarray(body.xpos), np.asarray(body.xquat)]
        feedback = dict(feedback or {})
        if self.target_geometry_cache is None and self.planner.phase_index >= 3:
            if "target_geometry_cache" in feedback:
                self.target_geometry_cache = feedback["target_geometry_cache"]
            else:
                from handover_geometry_quality import build_container_geometry_cache
                private = mujoco.MjData(model)
                mujoco.mj_forward(model, private)
                cache = build_container_geometry_cache(model, private,
                    container_body=raw.mujoco.mj_objects["container"].id,
                    target_body=raw.mujoco.mj_objects["target"].id)
                self.target_geometry_cache = {key: cache[key] for key in
                    ("target_root_local_collision_vertices_m", "target_com_offset_local_m")}
        if self.target_geometry_cache is not None:
            feedback["target_geometry_cache"] = self.target_geometry_cache
        actual_wrists = []
        actual_wrist_rotations = []
        from feedback_wrist_ik_v11 import TIP_OFFSETS
        for side, body in enumerate(self.ik.wrists):
            actual_wrists.append(data.xpos[body] + data.xmat[body].reshape(3, 3) @ TIP_OFFSETS[side])
            actual_wrist_rotations.append(data.xmat[body].reshape(3, 3).copy())
        feedback["actual_wrist_points_left_right"] = np.asarray(actual_wrists)
        feedback["actual_wrist_rotations_left_right"] = np.asarray(actual_wrist_rotations)
        goals, rotations, hands, indices, synthetic, dt, metadata = self.planner.step(frame, actual_objects, feedback)
        pelvis = model.body("pelvis").id
        position = data.xpos[pelvis].copy();rotation = data.xmat[pelvis].reshape(3, 3).copy()
        ref = metadata["actual_reference_frame"]
        is_placement = metadata["independent_container_placement"] is not None
        placement_ik = None
        if is_placement:
            actual_state = np.asarray(data.qpos[self.ik.qadr]).copy()
            free_root = np.asarray(data.qpos[self.ik.root_qadr:self.ik.root_qadr + 7]).copy()
            position = free_root[:3];rotation = matrix_wxyz(free_root[3:])
            previous = actual_state[:29] if self.previous_body_command is None else self.previous_body_command
            source_locomotion = self.config.placement_locomotion == "source_future"
            command29, ik_reports, placement_ik, right_goal, right_rotation = self.ik.solve_placement(
                ref, actual_state, position, rotation, goals[0], rotations[0], previous, dt, self.config,
                support_polygon=None if source_locomotion else foot_contact_support_polygon(model, data))
            goals[1] = right_goal;rotations[1] = right_rotation
            metadata["desired_wrist_points_left_right"] = goals.tolist()
            metadata["desired_wrist_rotations_left_right"] = rotations.tolist()
            metadata["actual_wrist_goal_errors_m"] = np.linalg.norm(np.asarray(actual_wrists) - goals, axis=1).tolist()
            delta = command29[15:29] - self.states[ref, 15:29]
            q = np.tile(command29, (46, 1))
            # A held future pose has zero future reference velocity. The
            # outgoing command slew is a separate diagnostic, not a forecast.
            dq = np.zeros((46, 29), dtype=np.float64)
            roots = np.tile(free_root[3:], (46, 1))
            delta_velocity = np.asarray(placement_ik["command_body_velocity29"])[15:29].copy()
            if source_locomotion:
                if self.locomotion_start is None:
                    self.locomotion_start = self.locomotion_frame = int(ref)
                    self.locomotion_raw_frame = int(ref)
                    self.locomotion_root_start = position.copy()
                    self.locomotion_alignment = yaw_rotation(rotation) @ yaw_rotation(matrix_wxyz(self.roots[ref])).T
                elif self.locomotion_last_tick is not None:
                    self.locomotion_raw_frame += int(frame) - self.locomotion_last_tick
                    self.locomotion_frame = min(len(self.states) - 1, self.locomotion_raw_frame)
                self.locomotion_last_tick = int(frame)
                future_native = self.locomotion_raw_frame + np.arange(46)
                native_indices = np.minimum(future_native, len(self.states) - 1)
                real = future_native < len(self.states)
                # Preserve the source's actual leg AND waist future/dq. The
                # arm task hold must not erase the original lateral stepping.
                q[:, :15] = self.states[native_indices, :15]
                dq[:, :15] = self.dq[native_indices, :15]
                dq[~real, :15] = 0.
                # The arm horizon is a pose hold, so its reference velocity
                # must be zero rather than the finite difference of commands.
                dq[:, 15:29] = 0.
                delta_velocity[:] = 0.
                unclipped_lower = q[:, :15].copy()
                for i, name in enumerate(TRAIN_STATE_NAMES[:15]):
                    joint = model.joint(name)
                    if bool(joint.limited[0]):q[:, i] = np.clip(q[:, i], *joint.range)
                aligned_roots = self.locomotion_alignment[None] @ matrix_wxyz(self.roots[native_indices])
                for i, R in enumerate(aligned_roots):mujoco.mju_mat2Quat(roots[i], np.ascontiguousarray(R).reshape(-1))
                target_root = self.locomotion_root_start + self.locomotion_alignment @ (self.xyz[self.locomotion_frame] - self.xyz[self.locomotion_start])
                placement_ik["task_command_limit_scope"] = "task arm14 only; lower15 retain separately bound real source q/dq"
                placement_ik["task_torso_IK_enabled"] = False
                metadata["locomotion_reference"] = {
                    "contract": "cvpr-source-lower15-native-future-v6", "source_start_frame": self.locomotion_start,
                    "source_current_frame": self.locomotion_frame,
                    "requested_native_frame": self.locomotion_raw_frame,
                    "source_future_frames": native_indices[FUTURE_OFFSETS].tolist(),
                    "real_native_future_mask": real[FUTURE_OFFSETS].tolist(),
                    "generated_tail_semantics": "source-end lower-body pose hold with zero dq, never real future labels",
                    "lower_body_joint_names": list(TRAIN_STATE_NAMES[:15]),
                    "lower15_future_q": q[FUTURE_OFFSETS, :15].tolist(),
                    "lower15_future_dq": dq[FUTURE_OFFSETS, :15].tolist(),
                    "source_root_motion_target_xyz_m": target_root.tolist(),
                    "actual_root_xyz_m": position.tolist(),
                    "actual_minus_source_motion_target_xyz_m": (position - target_root).tolist(),
                    "source_relative_root_displacement_m": (target_root - self.locomotion_root_start).tolist(),
                    "actual_relative_root_displacement_m": (position - self.locomotion_root_start).tolist(),
                    "root_translation_is_diagnostic_not_directly_set": True,
                    "task_waist_residual_applied": False,
                    "source_lower15_physical_range_clipped": bool(np.any(q[:, :15] != unclipped_lower)),
                    "task_arm_horizon_velocity_semantics": "zero dq for held task-arm q; command slew logged separately",
                    "reference_scope": "actual source native lower15 q/dq and aligned root orientations after measured left takeover; task arms remain online"}
                metadata["placement_encoder_horizon"] = "source_lower15_real_future_plus_online_arm_task_hold; generated source-tail explicitly masked"
            else:
                metadata["locomotion_reference"] = {"contract": "actual-posture-hold-v5", "source_locomotion_enabled": False}
        else:
            delta, ik_reports = self.ik.solve_current(ref, position, rotation, goals, rotations, self.config)
        # Approximation: solve only current two wrists and reuse their bounded
        # joint delta over the10 future style references. No future observation.
            dense = np.arange(46)
            source_indices = np.minimum(ref + dense, metadata["reference_control_end"])
            q, dq, roots = self.states[source_indices, :29].copy(), self.dq[source_indices].copy(), self.roots[source_indices].copy()
            generated = ref + dense > metadata["reference_control_end"]
            dq[generated] = 0.
            q[:, 15:29] += delta
            for i, name in enumerate(TRAIN_STATE_NAMES[15:29]):
                joint = model.joint(name)
                if bool(joint.limited[0]):
                    q[:, 15 + i] = np.clip(q[:, 15 + i], *joint.range)
            delta_velocity = np.zeros(14) if self.previous_delta is None else np.clip(
                (delta - self.previous_delta) / dt, -self.config.maximum_delta_velocity_rad_s,
                self.config.maximum_delta_velocity_rad_s)
            dq[:, 15:29] += delta_velocity
        value = encoder_input(q, dq, roots, 0)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0]);c, s = np.cos(yaw), np.sin(yaw)
        heading_inverse = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
        value[0, 584:644] = (heading_inverse[None] @ matrix_wxyz(roots[FUTURE_OFFSETS]))[..., :2].reshape(-1)
        encoder_mode = 0
        teleop_diagnostic = None
        if is_placement and self.config.placement_encoder_mode == "teleop":
            from handover_sonic_teleop_reference import build_teleop_reference
            # Preserve measured source torso/head style at the current actual
            # root. This FK touches only the expert's private MjData.
            self.ik._set_reference(self.states[ref], position, rotation)
            torso = model.body("torso_link").id
            head_rotation = self.ik.data.xmat[torso].reshape(3, 3).copy()
            head_point = self.ik.data.xpos[torso].copy() + head_rotation @ np.array([0., 0., .35])
            world_points = np.asarray([goals[0], goals[1], head_point])
            world_rotations = np.asarray([rotations[0], rotations[1], head_rotation])
            world_quaternions = np.empty((3, 4), dtype=np.float64)
            for i, R in enumerate(world_rotations):
                mujoco.mju_mat2Quat(world_quaternions[i], np.ascontiguousarray(R).reshape(-1))
            value, builder_evidence = build_teleop_reference(q[:, :12], dq[:, :12], 0,
                free_root, world_points, world_quaternions, roots[0],
                apply_delta_heading_wxyz=(1., 0., 0., 0.))
            encoder_mode = 1
            teleop_diagnostic = {"world_points_left_right_head_m": world_points.tolist(),
                "world_point_quaternions_wxyz": world_quaternions.tolist(),
                "actual_root_pose7": free_root.tolist(), "aligned_reference_root_wxyz": roots[0].tolist(),
                "apply_delta_heading_wxyz": [1., 0., 0., 0.],
                "head_source": "private FK of current real source q43/torso at the actual free root; torso-local+0.35m applied once",
                "wrist_points_already_include_tip_offsets": True,
                "official_builder_evidence": builder_evidence,
                "official_source_verification": self.teleop_official_sources,
                "G1_arm_joint_IK_encoded": False,
                "unused_G1_IK_scope": "diagnostics and natural right-retreat goal only; mode1 encodes direct world VR goals, not arm q"}
        token = np.asarray(self.session.run(None, {self.session.get_inputs()[0].name: value})[0][0], dtype=np.float32)
        if token.shape != (64,) or not np.isfinite(token).all() or not np.isfinite(hands).all():
            raise ValueError("Object-feedback expert produced a nonfinite command")
        self.previous_delta = delta.copy()
        self.previous_body_command = q[0].copy()
        metadata.update(contract="cvpr-handover-object-feedback-expert-v1",
            mode="online_object_task_teleop_sonic11_mode1" if encoder_mode == 1 else "online_object_relative_wrist_ik_sonic11_mode0",
            encoder_mode=encoder_mode, placement_encoder_mode=self.config.placement_encoder_mode,
            placement_teleop_reference=teleop_diagnostic,
            encoder_sha256=ENCODER_SHA, encoder_input_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            body_token_sha256=hashlib.sha256(token.tobytes()).hexdigest(),
            hand_command_sha256=hashlib.sha256(np.asarray(hands, dtype=np.float32).tobytes()).hexdigest(),
            hand_condition=self.hand_condition, historical_hand_mapping_certified=False,
            actual_root_xyz=position.tolist(), actual_root_rotation=rotation.tolist(),
            current_arm_joint_delta=delta.tolist(), current_arm_delta_velocity=delta_velocity.tolist(),
            desired_body_q29=q[0].tolist(), desired_body_q29_names=list(TRAIN_STATE_NAMES[:29]),
            desired_body_q29_direct_encoder_mask=([True] * 12 + [False] * 17) if encoder_mode == 1 else [True] * 29,
            G1_arm_IK_solution_used_for_token=encoder_mode == 0,
            current_two_wrist_ik=ik_reports, hand_command_names=list(SIMPLE_HAND_NAMES),
            placement_actual_posture_IK=placement_ik,
            hands14=np.asarray(hands).tolist(), configuration=vars(self.config),
            approximation=("direct task world wrists plus source-style torso/head via official mode1; lower12 future retained; arm IK is not encoded"
                           if encoder_mode == 1 else "actual-posture full-range task arms with native lower15 source future and no task torso lean"
                           if is_placement and self.config.placement_locomotion == "source_future" else
                           "actual-posture full-range left IK plus slew-limited body29 task hold; free right source-posture retreat"
                           if is_placement else "two current-arm IK solves; constant bounded arm delta across10 newly encoded future references"),
            simulator_state_mutated=False, object_state_mutated=False, external_force_applied=False,
            acceptance_gate_modified=False, physical_success_claim=False)
        return token, np.asarray(hands, dtype=np.float32), metadata
