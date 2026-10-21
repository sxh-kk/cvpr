"""Isolated CarryBox backport; see docs/CARRYBOX_LOCAL.md for metric differences."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Any, Type

import numpy as np

from simple.assets.carry_box import CarryBoxAsset
from simple.core.actor import ObjectActor
from simple.core.randomizer import Randomizer
from simple.core.task import Task
from simple.dr import CameraDRCfg, LanguageDRCfg, LightingDRCfg, MaterialDRCfg, SpatialDRCfg, TabletopSceneDRCfg, TargetDRCfg
from simple.dr.manager import DRManager
from simple.dr.types import Box
from simple.robots.registry import RobotRegistry
from simple.tasks.g1_wholebody_xmove_pick_teleop import G1WholebodyXMovePickTaskTeleop
from simple.tasks.registry import TaskRegistry


class _CarryBoxTargetDR(Randomizer):
    def __call__(self, split, **kwargs):
        if self._inner_state is None:
            self._inner_state = CarryBoxAsset(self.cfg.size)
        return self._inner_state

    def state_dict(self):
        return self._inner_state.to_dict() if self._inner_state is not None else {}

    def load_state_dict(self, state):
        if state.get("uid") != "cube":
            raise ValueError("CarryBox accepts only the primitive:cube target")
        self._inner_state = CarryBoxAsset(state.get("size", self.cfg.size))


@dataclass
class _CarryBoxTargetCfg(TargetDRCfg):
    size: tuple = (0.23, 0.38, 0.434)
    randmizer_class: Type[Randomizer] = _CarryBoxTargetDR


class _FloorBoxSpatialDR(Randomizer):
    def __call__(self, split, layout, table_height=0.0):
        if self._inner_state is None:
            # The official source writes reversed x bounds; sort the same range.
            low, high = np.asarray(self.cfg.target_region.low), np.asarray(self.cfg.target_region.high)
            xy = np.random.uniform(np.minimum(low, high), np.maximum(low, high)).tolist()
            self._inner_state = {
                "g1_sonic": {"position": [-1.2, 0.0, 0.0], "quaternion": [1., 0., 0., 0.]},
                "cube": {"position": xy + [layout.actors["target"].asset.size[2] / 2],
                         "quaternion": [1., 0., 0., 0.]},
            }
        for key, actor in (("g1_sonic", layout.robot), ("cube", layout.actors["target"])):
            state = self._inner_state[key]
            position, quaternion = np.asarray(state["position"], dtype=float), np.asarray(state["quaternion"], dtype=float)
            if position.shape != (3,) or quaternion.shape != (4,) or not np.isfinite(np.r_[position, quaternion]).all():
                raise ValueError(f"Invalid CarryBox spawn pose for {key}")
            if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-5):
                raise ValueError(f"CarryBox spawn quaternion is not unit length: {key}")
            actor.pose.position, actor.pose.quaternion = position.tolist(), quaternion.tolist()


@dataclass
class _FloorBoxSpatialCfg(SpatialDRCfg):
    randmizer_class: Type[Randomizer] = _FloorBoxSpatialDR


class _CarryBoxDRManager(DRManager):
    def __init__(self, **configs):
        # The old base manager keeps a class-level dict; isolate this task.
        self.randomizers = {}
        super().__init__(0, **configs)


@TaskRegistry.register("g1_wholebody_xmove_bend_carry_box_sonic")
class G1WholebodyXMoveBendCarryBoxTaskSonic(G1WholebodyXMovePickTaskTeleop):
    uid = "g1_wholebody_xmove_bend_carry_box_sonic"
    label = "G1 SONIC XMove Bend Carry Box (local v1)"
    description = "Lift the floor box and release it on the table."
    metadata = {**G1WholebodyXMovePickTaskTeleop.metadata,
                "task_adapter": "carry_box_local_v1", "success_metric": "true_tabletop_continuous_simtime_v1",
                "upstream_commit": "6d10628794d9c7de4596b4f2afb2c054a637c2bc"}
    dr_cfgs = dict(
        language=LanguageDRCfg(instructions=["xmove to the table and bend to pick up the {}"]),
        target=_CarryBoxTargetCfg(asset_id="primitive:cube"),
        spatial=_FloorBoxSpatialCfg(target_region=Box([-0.6, -0.05], [-0.5, 0.05])),
        camera=CameraDRCfg(cam_id="franka_camera"),
        scene=TabletopSceneDRCfg(scene_mode="fixed", table_position=Box([0.3, 0], [0.3, 0]),
                                table_height=Box(0.45, 0.45), room_choices=["hssd:scene0"], scene_manager="hssd"),
        lighting=LightingDRCfg(light_mode="random", light_num=(2, 3),
                              light_color_temperature=Box(6001, 8001), light_intensity=Box(8000, 12000),
                              light_radius=Box(0.08, 0.12), light_length=Box(0.51, 2.1),
                              light_spacing=Box((1., 1.), (2.5, 2.5)),
                              light_position=Box((-1.1, -1.1, 1.3), (1.1, 1.1, 1.5)),
                              light_eulers=Box((0, 0, -0.5*np.pi), (0, 0, 0.5*np.pi))),
        material=MaterialDRCfg(material_mode="rand_all"),
    )

    def __init__(self, robot_uid="g1_sonic", scene_uid="hssd:scene0", target=None,
                 split="val", render_hz=None, dr_level=0, success_criteria=0.9, **kwargs):
        if robot_uid != "g1_sonic" or scene_uid != "hssd:scene0" or target not in (None, "primitive:cube"):
            raise ValueError("CarryBox local v1 fixes g1_sonic, hssd:scene0 and primitive:cube")
        if dr_level != 0:
            raise ValueError("CarryBox local v1 implements DR level 0 only")
        if not np.isfinite(success_criteria) or success_criteria <= 0:
            raise ValueError("success_criteria must be a positive hold duration in simulation seconds")
        self.metadata = deepcopy(type(self).metadata)
        self.robot_cfg = deepcopy(type(self).robot_cfg)
        self.sensor_cfgs = deepcopy(type(self).sensor_cfgs)
        self.dr_cfgs = deepcopy(type(self).dr_cfgs)
        self._instruction = self._target = self._layout = None
        self._robot = RobotRegistry.make(uid=robot_uid, **kwargs)
        self.success_criteria = float(success_criteria)
        self._reset_progress()
        Task.__init__(self, dr=_CarryBoxDRManager(**self.dr_cfgs), split=split,
                      render_hz=render_hz, dr_level=dr_level)

    def _reset_progress(self):
        self.reward, self._last_reward_time = 0.0, None
        self._previously_placed = False
        self.progress = {"metric": "true_tabletop_continuous_simtime_v1", "hold_sim_s": 0.0}

    def reset(self, seed=None, options=None):
        # Existing DR helpers use NumPy/Python global RNG. Seed locally, restore
        # caller RNGs, and serialize the resulting exact initial state.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            if seed is not None:
                np.random.seed(seed)
                random.seed(seed)
            super().reset(seed, options)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self._reset_progress()
        self.metadata["reset_seed"] = seed

    def preload_objects(self):
        return [ObjectActor(CarryBoxAsset(self.dr_cfgs["target"].size))]

    def decompose(self):
        raise NotImplementedError("CarryBox local v1 evaluates policies; it has no synthetic demonstration generator")

    def compute_reward(self, info: dict[str, Any], *args, mujoco_env=None, **kwargs):
        if mujoco_env is None:
            return self.reward
        model, data = mujoco_env.mjModel, mujoco_env.mjData
        now = float(data.time)
        if not np.isfinite(now) or (self._last_reward_time is not None and now < self._last_reward_time):
            raise ValueError("CarryBox reward requires finite, monotonic simulation time; reset the task for a new episode")
        table = self.layout.actors["table"]
        table_top = float(table.pose.position[2] + 0.5 * table.size[2])
        target_id = model.body(self.target.asset.label).id
        hands_touch, on_top = False, False
        for contact in data.contact[:data.ncon]:
            body_a, body_b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if target_id not in (body_a, body_b):
                continue
            other = model.body(body_b if body_a == target_id else body_a).name
            # This MJCF fuses each palm mesh into the wrist_yaw body.
            hands_touch |= "hand" in other or "wrist_yaw" in other
            on_top |= "table" in other and abs(float(contact.pos[2]) - table_top) <= 0.05
        target_z = float(data.body(self.target.asset.label).xpos[2])
        placed = not hands_touch and on_top and np.isfinite(target_z) and target_z >= table_top
        delta = 0.0 if self._last_reward_time is None else now - self._last_reward_time
        # Repeated queries at one simulation timestamp do not earn extra reward.
        self.reward = self.reward + delta if placed and self._previously_placed else 0.0
        self._last_reward_time = now
        self._previously_placed = bool(placed)
        self.progress = {"metric": "true_tabletop_continuous_simtime_v1", "sim_time_s": now,
                         "target_z": target_z, "table_top_z": table_top, "hands_off": not hands_touch,
                         "on_table_top": bool(on_top), "com_above_table": target_z >= table_top,
                         "hold_sim_s": self.reward, "success": self.reward >= self.success_criteria}
        info["carry_box_progress"] = dict(self.progress)
        return self.reward

    def check_success(self, info, *args, **kwargs):
        return self.reward >= self.success_criteria
