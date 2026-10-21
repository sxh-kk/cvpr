"""Independent airborne box hold task; never redefines CarryBox placement."""
from copy import deepcopy

import mujoco
import numpy as np

from simple.dr import LanguageDRCfg
from simple.tasks.g1_wholebody_xmove_bend_carry_box_sonic import G1WholebodyXMoveBendCarryBoxTaskSonic
from simple.tasks.registry import TaskRegistry

LIFT_BOX_INSTRUCTION = ("Lift the box so its lowest point is at least 10 cm above the floor, "
                        "and maintain hand contact without floor or table support for 1 second.")
TASK_CONTRACT = {
    "task_id":"simple/G1WholebodyLiftBoxHoldSonic-v0",
    "uid":"g1_wholebody_lift_box_hold_sonic",
    "metric":"lift_box_airborne_hand_contact_v1", "instruction":LIFT_BOX_INSTRUCTION,
    "floor_z_m":0.0, "minimum_oriented_box_clearance_m":0.10,
    "minimum_hand_contact_normal_force_n_exclusive":0.01,
    "required_continuous_sim_seconds":1.0, "measurement_hz":50,
    "floor_table_contacts_allowed":False, "knee_other_robot_assistance_allowed":True,
    "stationary_hold_required":False, "placement_required":False,
    "geometry_and_control":"inherited CarryBox geometry and controller; independent success condition",
}


@TaskRegistry.register("g1_wholebody_lift_box_hold_sonic")
class G1WholebodyLiftBoxHoldTaskSonic(G1WholebodyXMoveBendCarryBoxTaskSonic):
    uid = "g1_wholebody_lift_box_hold_sonic"
    label = "G1 SONIC Lift Box Hold (local v1)"
    description = LIFT_BOX_INSTRUCTION
    metadata = {**G1WholebodyXMoveBendCarryBoxTaskSonic.metadata,
                "task_adapter":"lift_box_hold_local_v1", "success_metric":TASK_CONTRACT["metric"]}
    dr_cfgs = deepcopy(G1WholebodyXMoveBendCarryBoxTaskSonic.dr_cfgs)
    dr_cfgs["language"] = LanguageDRCfg(instructions=[LIFT_BOX_INSTRUCTION])

    def __init__(self, *args, success_criteria=1.0, render_hz=None, **kwargs):
        if success_criteria != 1.0 or render_hz not in (None, 50):
            raise ValueError("LiftBoxHold fixes one second of continuous observation at 50 Hz")
        super().__init__(*args, success_criteria=1.0, render_hz=render_hz, **kwargs)

    @property
    def instruction(self):
        return LIFT_BOX_INSTRUCTION

    def _reset_progress(self):
        self.reward, self._last_reward_time = 0.0, None
        self._previously_airborne = False
        self.progress = {"metric":TASK_CONTRACT["metric"], "hold_sim_s":0.0, "success":False}

    def _invalid(self, message):
        self._reset_progress()
        raise ValueError(message)

    def compute_reward(self, info, *args, mujoco_env=None, **kwargs):
        if mujoco_env is None:
            return self.reward
        model, data = mujoco_env.mjModel, mujoco_env.mjData
        now = float(data.time)
        if not np.isfinite(now) or (self._last_reward_time is not None and now < self._last_reward_time):
            self._invalid("LiftBoxHold requires finite monotonic simulation time; reset after a new episode")
        target = model.body(self.target.asset.label).id
        geoms = [i for i in range(model.ngeom) if model.geom_bodyid[i] == target]
        if len(geoms) != 1 or model.geom_type[geoms[0]] != mujoco.mjtGeom.mjGEOM_BOX:
            self._invalid("LiftBoxHold requires its single oriented analytic box geometry")
        geom = geoms[0]
        center, rotation, half_size = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3,3), model.geom_size[geom]
        if not np.isfinite(np.r_[center, rotation.reshape(-1), half_size]).all():
            self._invalid("LiftBoxHold received a nonfinite box geometry")
        lowest = float(center[2] - np.sum(np.abs(rotation[2]) * half_size))
        loaded_hand, floor_touch, table_touch = False, False, False
        hand_force, auxiliary_bodies = 0.0, set()
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
            if target not in (a, b):
                continue
            other_geom = int(contact.geom2 if a == target else contact.geom1)
            other = model.body(int(model.geom_bodyid[other_geom])).name
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, index, force)
            if not np.isfinite(force).all():
                self._invalid("LiftBoxHold received a nonfinite real contact force")
            hand = other.startswith(("left_hand_", "right_hand_")) or other in ("left_wrist_yaw_link", "right_wrist_yaw_link")
            floor = model.geom_type[other_geom] == mujoco.mjtGeom.mjGEOM_PLANE or model.geom(other_geom).name in ("ground", "floor")
            table = "table" in other.lower()
            floor_touch |= bool(floor)
            table_touch |= table
            if hand:
                hand_force += max(float(force[0]), 0.0)
                loaded_hand |= bool(force[0] > .01)
            elif not floor and not table:
                auxiliary_bodies.add(other)
        condition = bool(lowest >= .10 and loaded_hand and not floor_touch and not table_touch)
        delta = 0.0 if self._last_reward_time is None else now-self._last_reward_time
        # A missed observation interval cannot establish continuous contact.
        gap = delta > 1.0/50 + 1e-7
        if condition and self._previously_airborne and not gap:
            self.reward += delta
        else:
            self.reward = 0.0
        self._last_reward_time, self._previously_airborne = now, condition
        success = bool(self.reward >= 1.0-1e-9)
        self.progress = {"metric":TASK_CONTRACT["metric"], "sim_time_s":now,
            "target_z":float(center[2]), "box_lowest_z_m":lowest, "floor_z_m":0.0,
            "clearance_ge_10cm":bool(lowest >= .10), "loaded_hand_contact":bool(loaded_hand),
            "hand_normal_force_n":hand_force, "floor_contact":bool(floor_touch), "table_contact":bool(table_touch),
            "other_robot_contact_bodies":sorted(auxiliary_bodies), "condition":condition,
            "measurement_hz":50, "continuity_gap":bool(gap), "hold_sim_s":self.reward, "success":success}
        info["lift_box_hold_progress"] = dict(self.progress)
        return self.reward

    def check_success(self, info, *args, **kwargs):
        return bool(self.progress.get("success", False))
