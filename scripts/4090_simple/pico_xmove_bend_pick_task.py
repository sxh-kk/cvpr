"""Private native-SONIC XMoveBendPick task with explicit PICO controller identity.

All saved world coordinates, robot spawn and native reward/check_success stay
unchanged. This does not certify equivalence to the archived teleop controller.
No teacher action/state or global task registry is used.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

TASK_ID = "cvpr/G1SonicXMoveBendPickPico-v0"
SOURCE_OFFICIAL_TASK_ID = "simple/G1WholebodyXMoveBendPickTeleop-v0"
TASK_UID = "g1_wholebody_xmove_bend_pick_teleop"
OFFICIAL_TASK_SOURCE_SHA256 = "a50c845c59abb8e8c577e13beb70a91f9025036e22ae3d530a73336ccb09881e"
CONTRACT = "cvpr-pico-xmove-bend-pick-sonic-controller-adapter-v1"
FRAME_KEY = "pico_xmove_bend_pick_adaptation"
WORLD_TRANSLATION_M = (0.0, 0.0, 0.0)
MATERIAL_RESTORATION = "saved table/object layout materials only; unsaved ground/robot use backend defaults"
LIGHTING_RESTORATION = "saved layout light instances only; original empty DR lighting state preserved, no random light generation"
_RESET_KEYS = ("uid", "metadata", "robot_cfg", "sensor_cfgs", "dr_cfgs", "dr_state_dict")


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=_plain).encode()).hexdigest()


def adapter_identity():
    return {"contract": CONTRACT, "task_id": TASK_ID,
        "source_official_task_id": SOURCE_OFFICIAL_TASK_ID,
        "official_task_source_sha256": OFFICIAL_TASK_SOURCE_SHA256,
        "adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def _xyz(value):
    result = np.asarray(value, dtype=float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Expected a finite world XYZ position")
    return result


def _reset_digest(state):
    # Layout is a redundant serialization: SIMPLE renumbers distractor labels and
    # reconstructs float32 stereo extrinsics. Bind its physical poses separately.
    return _sha({key: state[key] for key in _RESET_KEYS})


def _validate_layout(state):
    spatial = state["dr_state_dict"]["spatial"]
    actors = state["layout"]["actors"]
    if "container" in actors or "target" not in actors:
        raise ValueError("XMoveBendPick requires a target and no synthetic container")
    target = actors["target"]
    saved_target = state["dr_state_dict"].get("target", {})
    target_asset = target.get("asset", {})
    if (not saved_target.get("res_id") or not saved_target.get("uid")
            or target_asset.get("res_id") != saved_target["res_id"]
            or str(target_asset.get("uid")) != str(saved_target["uid"])
            or str(target.get("uid")) != str(saved_target["uid"])):
        raise ValueError("XMoveBendPick target role must match the saved DR target asset identity")
    uids = []
    for name, actor in actors.items():
        if "asset" not in actor:
            continue
        uid = str(actor["asset"]["uid"])
        uids.append(uid)
        expected = spatial.get(uid)
        if expected is None or not np.allclose(actor["pose"]["position"],
                expected["position"], rtol=0, atol=1e-7) or not np.allclose(
                actor["pose"]["quaternion"], expected["quaternion"], rtol=0, atol=1e-7):
            raise ValueError(f"XMoveBendPick actor {name} is not its saved initial spatial pose")
    if len(set(uids)) != len(uids) or set(uids) != set(spatial) - {"g1_sonic"}:
        raise ValueError("Missing or duplicate XMoveBendPick object spatial identity")
    if not np.allclose(actors["table"]["pose"]["position"],
            state["dr_state_dict"]["scene"]["table"]["pose"]["position"], rtol=0, atol=1e-7):
        raise ValueError("Saved table layout and DR disagree")


def adapt_official_config(source_config):
    """Bind a native SONIC reset without translating any coordinate or pose."""
    source = copy.deepcopy(source_config)
    if FRAME_KEY in source or source.get("uid") != TASK_UID:
        raise ValueError("Expected an unadapted official XMoveBendPick reset configuration")
    if source.get("robot_cfg", {}).get("uid") != "g1_sonic":
        raise ValueError("XMoveBendPick requires the original SONIC robot identity")
    if (source.get("metadata", {}).get("physics_dt") != .005
            or source.get("metadata", {}).get("render_hz") != 50):
        raise ValueError("Expected saved native XMoveBendPick 200Hz physics / 50Hz observations")
    table = source["dr_state_dict"]["scene"]["table"]
    top = float(table["pose"]["position"][2]) + .5 * float(table["size"][2])
    robot_z = _xyz(source["dr_state_dict"]["spatial"]["g1_sonic"]["position"])[2]
    if not np.isclose(top, .45, rtol=0, atol=1e-12) or not np.isclose(robot_z, 0., rtol=0, atol=1e-12):
        raise ValueError("XMoveBendPick requires native table top .45 and robot attachment z=0")
    _validate_layout(source)
    receipt = {**adapter_identity(), "world_translation_m": list(WORLD_TRANSLATION_M),
        "source_robot_uid": "g1_sonic", "robot_uid": "g1_sonic",
        "sonic_control_policy": "joystick_mode0_full_body_mode2", "sonic_encoder_modes": [0, 2],
        "native_ground_z_m": 0., "native_robot_spawn_z_m": 0., "native_table_top_z_m": .45,
        "source_config_sha256": _sha(source), "adapted_payload_sha256": _reset_digest(source),
        "adapted_payload_scope": list(_RESET_KEYS), "source_physics_dt": .005,
        "physics_dt": .005, "render_hz": 50, "native_reward_unmodified": True,
        "robot_physics_equivalent": False, "coordinate_transform_applied": False,
        "source_material_state_missing": source["dr_state_dict"].get("material") is None,
        "material_restoration": MATERIAL_RESTORATION,
        "source_lighting_state_missing": source["dr_state_dict"].get("lighting") == {},
        "lighting_restoration": LIGHTING_RESTORATION,
        "scope": "native SONIC saved scene and unmodified reward/check_success; explicit PICO controller adaptation, no official control equivalence claim"}
    source[FRAME_KEY] = copy.deepcopy(receipt)
    return source, receipt


def validate_adapted_config(state):
    receipt = state.get(FRAME_KEY, {})
    if any(receipt.get(k) != v for k, v in adapter_identity().items()):
        raise ValueError("XMoveBendPick adapter/source identity mismatch")
    if state.get("uid") != TASK_UID or state.get("robot_cfg", {}).get("uid") != "g1_sonic":
        raise ValueError("XMoveBendPick task/robot identity mismatch")
    expected = {"sonic_control_policy": "joystick_mode0_full_body_mode2",
        "native_ground_z_m": 0., "native_robot_spawn_z_m": 0., "native_table_top_z_m": .45,
        "sonic_encoder_modes": [0, 2], "world_translation_m": [0., 0., 0.], "source_robot_uid": "g1_sonic",
        "robot_uid": "g1_sonic", "source_physics_dt": .005, "physics_dt": .005,
        "render_hz": 50, "native_reward_unmodified": True,
        "robot_physics_equivalent": False, "coordinate_transform_applied": False,
        "adapted_payload_scope": list(_RESET_KEYS)}
    if any(receipt.get(k) != v or (type(v) is bool and type(receipt.get(k)) is not bool)
            for k, v in expected.items()):
        raise ValueError("XMoveBendPick coordinate/controller declaration mismatch")
    if (not isinstance(receipt.get("sonic_encoder_modes"), list)
            or any(type(value) is not int for value in receipt["sonic_encoder_modes"])):
        raise ValueError("XMoveBendPick encoder modes must be exact integers")
    if (type(receipt.get("source_material_state_missing")) is not bool
            or receipt["source_material_state_missing"] != (state["dr_state_dict"].get("material") is None)
            or receipt.get("material_restoration") != MATERIAL_RESTORATION):
        raise ValueError("XMoveBendPick missing material provenance declaration mismatch")
    if (type(receipt.get("source_lighting_state_missing")) is not bool
            or receipt["source_lighting_state_missing"] != (state["dr_state_dict"].get("lighting") == {})
            or receipt.get("lighting_restoration") != LIGHTING_RESTORATION):
        raise ValueError("XMoveBendPick missing lighting provenance declaration mismatch")
    if (state["metadata"].get("physics_dt") != .005 or state["metadata"].get("render_hz") != 50
            or receipt.get("adapted_payload_sha256") != _reset_digest(state)):
        raise ValueError("XMoveBendPick adapted reset payload changed")
    _validate_layout(state)
    return copy.deepcopy(receipt)


def xmove_bend_pick_environment_binding_receipt(expected, actual):
    """Compare real reconstructed DR/scene/actors/cameras/lights, without edits.

    Only distractor labels are normalized by immutable asset UID; target keeps
    its fixed role and DR-bound asset identity. Camera stereo creation
    in upstream SIMPLE casts positions to float32. Only numeric values receive
    an absolute 1e-6 tolerance (zero relative tolerance); every string/key/list
    length/material field remains exact. Robot spawn is bound in spatial DR.
    """
    validate_adapted_config(expected)
    validate_adapted_config(actual)

    def normalize(state):
        out = copy.deepcopy({"dr_state_dict": state["dr_state_dict"], "layout": state["layout"]})
        layout = out["layout"]
        actors = {}
        for name, actor in layout["actors"].items():
            key = ("distractor_asset:" + str(actor["asset"]["uid"])
                if name.startswith("distractor_") and "asset" in actor else name)
            if key in actors:
                raise ValueError("Duplicate XMoveBendPick actor asset identity")
            actors[key] = actor
        layout["actors"] = actors
        # Derived cam_cfg poses describe the same transformed left-camera frame.
        return json.loads(json.dumps(out, default=_plain, allow_nan=False))

    left, right = normalize(expected), normalize(actual)
    errors = []

    def compare(a, b, path):
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                errors.append(path + ":keys")
            for key in sorted(set(a) & set(b)):
                compare(a[key], b[key], path + "." + key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                errors.append(path + ":length")
            for i, (x, y) in enumerate(zip(a, b)):
                compare(x, y, path + f"[{i}]")
        elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            if not np.isfinite([a, b]).all() or not np.isclose(a, b, rtol=0, atol=1e-6):
                errors.append(path)
        elif type(a) is not type(b) or a != b:
            errors.append(path)

    compare(left, right, "environment")
    return {"contract": CONTRACT, "matched": not errors,
        "scope": "complete saved DR and reconstructed scene/actor/camera/light/material fields; target role fixed, distractor labels by asset UID",
        "absolute_numeric_tolerance": 1e-6, "relative_numeric_tolerance": 0,
        "expected_state_sha256": _sha(left), "actual_state_sha256": _sha(right),
        "expected_scene": expected["dr_state_dict"]["scene"]["name"],
        "actual_scene": actual["dr_state_dict"]["scene"]["name"], "mismatches": errors}


def _load_official():
    module = importlib.import_module("simple.tasks.g1_wholebody_xmove_bend_pick_teleop")
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != OFFICIAL_TASK_SOURCE_SHA256:
        raise ValueError("Official XMoveBendPick task source changed")
    return module.G1WholebodyXMoveBendPickTaskTeleop


def _cfg_value(value):
    from simple.dr.types import Box
    if isinstance(value, dict) and set(value) == {"low", "high"}:
        return Box(copy.deepcopy(value["low"]), copy.deepcopy(value["high"]))
    if isinstance(value, list):
        return [_cfg_value(x) for x in value]
    return copy.deepcopy(value)


def make_xmove_bend_pick_task(sonic_config, *, source_config=None, adapted_config=None):
    """Create an unregistered task-local subclass, without modifying global task registration."""
    if (source_config is None) == (adapted_config is None):
        raise ValueError("Provide exactly one source_config or adapted_config")
    state = adapt_official_config(source_config)[0] if source_config is not None else copy.deepcopy(adapted_config)
    validate_adapted_config(state)
    if not np.isclose(sonic_config.get("SIMULATE_DT", float("nan")), 0.005, rtol=0, atol=1e-12):
        raise ValueError("XMoveBendPick SONIC requires declared 200Hz physics")
    official = _load_official()
    from simple.core.task import Task
    from simple.dr.manager import TabletopGraspDRManager
    from simple.dr.material import MaterialDR
    from simple.dr.lighting import LightingDR
    from simple.core.actor import ObjectActor
    from simple.robots.registry import RobotRegistry
    from simple.sensors.config import CameraCfg, StereoCameraCfg

    class PrivateXMoveBendPick(official):
        # uid intentionally remains the original task UID for Task.load_state_dict.
        adapter_task_id = TASK_ID

        def __init__(self):
            self.metadata = copy.deepcopy(state["metadata"])
            self.robot_cfg = copy.deepcopy(state["robot_cfg"])
            self.sensor_cfgs = {k: (StereoCameraCfg if "baseline" in v else CameraCfg)(**copy.deepcopy(v))
                for k, v in state["sensor_cfgs"].items()}
            self.dr_cfgs = {}
            for key, raw in state["dr_cfgs"].items():
                default = official.dr_cfgs[key]
                if raw["randmizer_class"] != default.to_dict()["randmizer_class"]:
                    raise ValueError("Unexpected saved randomizer class")
                self.dr_cfgs[key] = type(default)(**{k: _cfg_value(v) for k, v in raw.items() if k != "randmizer_class"})
            # Upstream DRManager otherwise writes a shared class dictionary.
            class LocalDR(TabletopGraspDRManager):
                def __init__(self, **kw):
                    self.randomizers = {}
                    super().__init__(**kw)
            self._robot = RobotRegistry.make(uid="g1_sonic", sonic_config=copy.deepcopy(sonic_config))
            self._instruction = self._target = self._layout = self._init_target_height = None
            self.reward = 0
            self.success_criteria = .9
            self._contact_started = False
            self._adapted_reset = copy.deepcopy(state)
            self.frame_receipt = copy.deepcopy(state[FRAME_KEY])
            Task.__init__(self, dr=LocalDR(level=0, **self.dr_cfgs), split=self.metadata.get("split", "train"),
                render_hz=50, dr_level=0, physics_dt=0.005)
            # These source100 records omitted MaterialDR state while retaining
            # actual table/object materials in layout. Do not let upstream turn
            # None into a new random visual initialization. The missing robot
            # and ground values remain backend defaults, without invented DR.
            owner = self
            original_material = self.dr.get_randomizer("material")
            class SavedLayoutMaterial(MaterialDR):
                def __call__(self, split, layout, **kwargs):
                    if self._inner_state is not None:
                        return super().__call__(split, layout, **kwargs)
                    source_actors = owner._adapted_reset["layout"]["actors"]
                    saved_objects = {str(actor["asset"]["uid"]): actor["material"]
                        for actor in source_actors.values() if "asset" in actor}
                    layout.actors["table"].set_material(copy.deepcopy(source_actors["table"]["material"]))
                    for actor in layout.actors.values():
                        if isinstance(actor, ObjectActor):
                            actor.set_material(copy.deepcopy(saved_objects[str(actor.asset.uid)]))
                    return None
            local_material = SavedLayoutMaterial.__new__(SavedLayoutMaterial)
            local_material.__dict__ = copy.deepcopy(original_material.__dict__)
            self.dr.randomizers["material"] = local_material
            class SavedLayoutLighting(LightingDR):
                def __call__(self, split, *args, **kwargs):
                    if owner._adapted_reset["dr_state_dict"].get("lighting") != {}:
                        return super().__call__(split, *args, **kwargs)
                    # Loading the source {} must stay {} in DR provenance.
                    # Reconstruct only its separately saved actual light list.
                    saved = LightingDR(copy.deepcopy(self.cfg))
                    saved.load_state_dict({light["uid"]: copy.deepcopy(light)
                        for light in owner._adapted_reset["layout"]["lights"]})
                    return saved(split, *args, **kwargs)
            original_lighting = self.dr.get_randomizer("lighting")
            local_lighting = SavedLayoutLighting.__new__(SavedLayoutLighting)
            local_lighting.__dict__ = copy.deepcopy(original_lighting.__dict__)
            self.dr.randomizers["lighting"] = local_lighting

        def reset(self, seed=None, options=None):
            opts = copy.deepcopy(options or {})
            if opts.get("dr_level") is not None:
                raise ValueError("XMoveBendPick must restore all saved DR; no dr_level override")
            reset_state = opts.get("state_dict") or self._adapted_reset
            validate_adapted_config(reset_state)
            # No global HSSD manager or scene configuration is mutated.
            rnd = self.dr.get_randomizer("scene")
            manager = copy.copy(rnd.scene_manager)
            if not hasattr(manager, "hssd_scenes"):
                raise ValueError("XMoveBendPick saved reset requires HSSD scene manager")
            manager.hssd_scenes = [copy.deepcopy(reset_state["dr_state_dict"]["scene"]["conf"])]
            rnd.scene_manager = manager
            # Sensors vary with saved camera DR/source configuration on each reset.
            self.sensor_cfgs = {k: (StereoCameraCfg if "baseline" in v else CameraCfg)(**copy.deepcopy(v))
                for k, v in reset_state["sensor_cfgs"].items()}
            self._adapted_reset = copy.deepcopy(reset_state)
            Task.reset(self, seed, {"state_dict": copy.deepcopy(reset_state), "dr_level": None})
            self._adapted_reset = copy.deepcopy(reset_state)
            self.frame_receipt = copy.deepcopy(reset_state[FRAME_KEY])
            self._target = self.layout.actors["target"]
            self._instruction = self.dr.get_randomizer("language")(self.metadata.get("split", "train")).format(self._target.asset.name)
            self._init_target_height = None
            self.reward = 0
            self.robot.reset(spawn_pose=copy.deepcopy(self.layout.actors["robot"].pose))

        def state_dict(self):
            actual = Task.state_dict(self)
            # Keep authoritative original config values; DR state and layout here
            # remain the actual reconstructed task, not a fabricated source copy.
            actual["dr_cfgs"] = copy.deepcopy(self._adapted_reset["dr_cfgs"])
            actual[FRAME_KEY] = copy.deepcopy(self.frame_receipt)
            return actual

    return PrivateXMoveBendPick()
