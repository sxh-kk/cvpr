"""Private PICO/SONIC adapter for the unchanged SIMPLE MP Grasp reward.

This is an explicit robot/backend adaptation, not the official AMO benchmark.
Only world coordinates and the robot XML attachment origin are transformed.
The two robots' geometry, dynamics, PD gains and attached camera mounts differ.
No policy, teacher action, future demonstration frame or global registry is used.
"""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

TASK_ID = "cvpr/G1SonicTabletopGraspPico-v0"
SOURCE_OFFICIAL_TASK_ID = "simple/G1WholebodyTabletopGraspMP-v0"
TASK_UID = "g1_wholebody_tabletop_grasp_mp"
OFFICIAL_TASK_SOURCE_SHA256 = "a0f5d66a79e8afcc984bb20754639f2b157118aa8d31dc8886c2b952238975f4"
CONTRACT = "cvpr-pico-grasp-sonic-full-world-frame-adapter-v1"
FRAME_KEY = "pico_grasp_frame_adaptation"
WORLD_TRANSLATION_M = (0.0, 0.0, 0.785)
NATIVE_XML_ROOT_M = (0.0, 0.0, 0.0)
SONIC_XML_ROOT_M = (0.0, 0.0, 0.793)
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


def _shift_pose(pose):
    pose["position"] = (_xyz(pose["position"]) + WORLD_TRANSLATION_M).tolist()


def _robot_pose(native_pose):
    from transforms3d.quaternions import quat2mat
    pose = copy.deepcopy(native_pose)
    q = np.asarray(pose["quaternion"], dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or not np.isclose(np.linalg.norm(q), 1):
        raise ValueError("Expected a unit robot quaternion (wxyz)")
    # Translate the actual XML pelvis, not merely its enclosing world frame.
    compensation = quat2mat(q) @ (np.asarray(NATIVE_XML_ROOT_M) - SONIC_XML_ROOT_M)
    pose["position"] = (_xyz(pose["position"]) + WORLD_TRANSLATION_M + compensation).tolist()
    return pose, compensation.tolist()


def _reset_digest(state):
    # Layout is a redundant serialization: SIMPLE renumbers distractor labels and
    # reconstructs float32 stereo extrinsics. Bind its physical poses separately.
    return _sha({key: state[key] for key in _RESET_KEYS})


def _validate_layout(state):
    spatial = state["dr_state_dict"]["spatial"]
    actors = state["layout"]["actors"]
    if "container" in actors or "target" not in actors:
        raise ValueError("Grasp requires a target and no synthetic container")
    target = actors["target"]
    saved_target = state["dr_state_dict"].get("target", {})
    target_asset = target.get("asset", {})
    if (not saved_target.get("res_id") or not saved_target.get("uid")
            or target_asset.get("res_id") != saved_target["res_id"]
            or str(target_asset.get("uid")) != str(saved_target["uid"])
            or str(target.get("uid")) != str(saved_target["uid"])):
        raise ValueError("Grasp target role must match the saved DR target asset identity")
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
            raise ValueError(f"Grasp actor {name} is not its saved initial spatial pose")
    if len(set(uids)) != len(uids) or set(uids) != set(spatial) - {"g1_wholebody", "g1_sonic"}:
        raise ValueError("Missing or duplicate Grasp object spatial identity")
    if not np.allclose(actors["table"]["pose"]["position"],
            state["dr_state_dict"]["scene"]["table"]["pose"]["position"], rtol=0, atol=1e-7):
        raise ValueError("Saved table layout and DR disagree")


def adapt_official_config(source_config):
    """Copy one original saved initialization; never change the source dictionary."""
    source = copy.deepcopy(source_config)
    if FRAME_KEY in source or source.get("uid") != TASK_UID:
        raise ValueError("Expected an unadapted official MP Grasp configuration")
    if source.get("robot_cfg", {}).get("uid") != "g1_wholebody":
        raise ValueError("Expected the original AMO robot identity")
    if source.get("metadata", {}).get("physics_dt") != .002 or source.get("metadata", {}).get("render_hz") != 50:
        raise ValueError("Expected the saved official Grasp 500Hz physics / 50Hz observation configuration")
    _validate_layout(source)
    source_sha = _sha(source)
    out = copy.deepcopy(source)
    spatial = out["dr_state_dict"]["spatial"]
    if "g1_sonic" in spatial or "g1_wholebody" not in spatial:
        raise ValueError("Missing or ambiguous original robot spatial identity")
    native_robot = spatial.pop("g1_wholebody")
    if not np.isclose(_xyz(native_robot["position"])[2], 0., rtol=0, atol=1e-12):
        raise ValueError("Official Grasp floor translation requires its saved robot attachment z=0")
    sonic_robot, compensation = _robot_pose(native_robot)
    for pose in spatial.values():
        _shift_pose(pose)
    spatial["g1_sonic"] = sonic_robot
    out["robot_cfg"]["uid"] = "g1_sonic"
    # SONIC production uses 200Hz physics and 50Hz real observations. This change
    # is declared, not presented as numerical equivalence to native AMO 500Hz.
    out["metadata"].update(physics_dt=0.005, render_hz=50)
    for scene in (out["dr_state_dict"]["scene"], out["layout"]["scene"]):
        scene["center_offset"] = (_xyz(scene["center_offset"]) + WORLD_TRANSLATION_M).tolist()
        for table in ("table", "table2"):
            if scene.get(table) is not None:
                _shift_pose(scene[table]["pose"])
    for actor in out["layout"]["actors"].values():
        _shift_pose(actor["pose"])
    for collection in (out["dr_state_dict"]["lighting"].values(), out["layout"]["lights"]):
        for light in collection:
            light["center_light_postion"] = (_xyz(light["center_light_postion"]) + WORLD_TRANSLATION_M).tolist()
            # light.pose is local to its shared light-center transform.
    for name, cfg in out["sensor_cfgs"].items():
        if cfg["mount"] == "eye_on_base":
            left = name + "_left" if "baseline" in cfg else name
            cfg["pose"] = copy.deepcopy(source["layout"]["cameras"][left]["pose"])
            _shift_pose(cfg["pose"])
    for cam in out["layout"]["cameras"].values():
        if cam["mount"] == "eye_on_base":
            _shift_pose(cam["pose"])
        cam["cam_cfg"] = copy.deepcopy(out["sensor_cfgs"][cam["cam_id"]])
    receipt = {**adapter_identity(), "world_translation_m": list(WORLD_TRANSLATION_M),
        "robot_anchor_compensation_m": compensation,
        "native_xml_root_m": list(NATIVE_XML_ROOT_M), "sonic_xml_root_m": list(SONIC_XML_ROOT_M),
        "source_config_sha256": source_sha, "adapted_payload_sha256": _reset_digest(out),
        "adapted_payload_scope": list(_RESET_KEYS),
        "source_physics_dt": source["metadata"]["physics_dt"], "physics_dt": 0.005,
        "render_hz": 50, "native_reward_unmodified": True,
        "robot_physics_equivalent": False,
        "camera_scope": "world cameras translated; robot-local cameras retain actual SONIC mounts",
        "required_hooks": ["mujoco_world_wrist_camera", "isaac_room_translation"]}
    out[FRAME_KEY] = copy.deepcopy(receipt)
    validate_adapted_config(out)
    return out, receipt


def validate_adapted_config(state):
    """Validate reset-authoritative fields and redundant per-asset layout poses."""
    receipt = state.get(FRAME_KEY, {})
    if any(receipt.get(k) != v for k, v in adapter_identity().items()):
        raise ValueError("Grasp adapter/source identity mismatch")
    if state.get("uid") != TASK_UID or state.get("robot_cfg", {}).get("uid") != "g1_sonic":
        raise ValueError("Adapted Grasp task/robot identity mismatch")
    if receipt.get("world_translation_m") != list(WORLD_TRANSLATION_M):
        raise ValueError("Wrong Grasp world translation")
    if (receipt.get("native_xml_root_m") != list(NATIVE_XML_ROOT_M)
            or receipt.get("sonic_xml_root_m") != list(SONIC_XML_ROOT_M)
            or receipt.get("native_reward_unmodified") is not True
            or receipt.get("robot_physics_equivalent") is not False
            or state["metadata"].get("physics_dt") != 0.005
            or state["metadata"].get("render_hz") != 50):
        raise ValueError("Grasp frame/physics declaration mismatch")
    if receipt.get("adapted_payload_sha256") != _reset_digest(state):
        raise ValueError("Adapted Grasp reset payload changed")
    _validate_layout(state)
    return copy.deepcopy(receipt)


def grasp_environment_binding_receipt(expected, actual):
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
                raise ValueError("Duplicate Grasp actor asset identity")
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
    module = importlib.import_module("simple.tasks.g1_wholebody_tabletop_grasp_mp")
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != OFFICIAL_TASK_SOURCE_SHA256:
        raise ValueError("Official Grasp task source changed")
    return module.G1WholebodyTabletopGraspMP


def _cfg_value(value):
    from simple.dr.types import Box
    if isinstance(value, dict) and set(value) == {"low", "high"}:
        return Box(copy.deepcopy(value["low"]), copy.deepcopy(value["high"]))
    if isinstance(value, list):
        return [_cfg_value(x) for x in value]
    return copy.deepcopy(value)


def make_grasp_task(sonic_config, *, source_config=None, adapted_config=None):
    """Create an unregistered task-local subclass, without constructing AMO."""
    if (source_config is None) == (adapted_config is None):
        raise ValueError("Provide exactly one source_config or adapted_config")
    state = adapt_official_config(source_config)[0] if source_config is not None else copy.deepcopy(adapted_config)
    validate_adapted_config(state)
    if not np.isclose(sonic_config.get("SIMULATE_DT", float("nan")), 0.005, rtol=0, atol=1e-12):
        raise ValueError("Grasp SONIC requires declared 200Hz physics")
    official = _load_official()
    from simple.core.task import Task
    from simple.dr.manager import TabletopGraspDRManager
    from simple.robots.registry import RobotRegistry
    from simple.sensors.config import CameraCfg, StereoCameraCfg

    class PrivateGrasp(official):
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
            self._adapted_reset = copy.deepcopy(state)
            self.frame_receipt = copy.deepcopy(state[FRAME_KEY])
            Task.__init__(self, dr=LocalDR(level=0, **self.dr_cfgs), split=self.metadata.get("split", "train"),
                render_hz=50, dr_level=0, physics_dt=0.005)

        def reset(self, seed=None, options=None):
            opts = copy.deepcopy(options or {})
            if opts.get("dr_level") is not None:
                raise ValueError("Grasp must restore all saved DR; no dr_level override")
            reset_state = opts.get("state_dict") or self._adapted_reset
            validate_adapted_config(reset_state)
            # No global HSSD manager or scene configuration is mutated.
            rnd = self.dr.get_randomizer("scene")
            manager = copy.copy(rnd.scene_manager)
            if not hasattr(manager, "hssd_scenes"):
                raise ValueError("Grasp saved reset requires HSSD scene manager")
            manager.hssd_scenes = [copy.deepcopy(reset_state["dr_state_dict"]["scene"]["conf"])]
            rnd.scene_manager = manager
            # Sensors vary with saved camera DR/source configuration on each reset.
            self.sensor_cfgs = {k: (StereoCameraCfg if "baseline" in v else CameraCfg)(**copy.deepcopy(v))
                for k, v in reset_state["sensor_cfgs"].items()}
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

    return PrivateGrasp()


def install_frame_adapter(env):
    """Install per-environment hooks before reset/first physics or RGB step.

    MuJoCo's world-fixed wrist placeholder is shifted once per new model;
    Isaac's SONIC-only room z=0 override is undone after every layout reset.
    Local head/hand mounts remain the actual robot backend's mounts.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    if getattr(raw.task, "adapter_task_id", None) != TASK_ID:
        raise ValueError("Frame adapter can only bind the private Grasp task")
    if getattr(raw, "_pico_grasp_frame_adapter_installed", False):
        raise ValueError("Grasp frame adapter is already installed")
    raw._pico_grasp_frame_adapter_installed = True
    original = raw.mujoco.update_layout

    def update_mujoco(*args, **kwargs):
        result = original(*args, **kwargs)
        import mujoco
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        for name, camera in raw.task.layout.cameras.items():
            if camera.mount != "eye_in_hand":
                continue
            cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
            if cid < 0 or model.cam_bodyid[cid] != 0:
                raise ValueError("Upstream world-fixed wrist camera contract changed")
            if not np.allclose(model.cam_pos[cid], [1.5, 0., .8], rtol=0, atol=1e-12):
                raise ValueError("Upstream wrist camera position changed or frame hook applied twice")
            model.cam_pos[cid] += WORLD_TRANSLATION_M
        mujoco.mj_forward(model, data)
        return result

    raw.mujoco.update_layout = update_mujoco
    isaac = getattr(raw, "isaac", None)
    if isaac is not None:
        original_isaac = isaac.update_layout

        def update_isaac(*args, **kwargs):
            result = original_isaac(*args, **kwargs)
            scene = raw.task.layout.scene
            prim, obb = isaac.scenes[scene.uid]
            xyz = -_xyz(obb["position"]) + _xyz(scene.center_offset)
            prim.GetAttribute("xformOp:translate").Set(tuple(xyz))
            # Isaac eye_on_base is parented to the robot; MuJoCo's same named
            # camera is world-fixed. The robot parent already supplies the world
            # translation in Isaac, so retain its original local camera offset.
            for name, camera in raw.task.layout.cameras.items():
                if camera.mount == "eye_on_base":
                    isaac.cameras[name].set_local_pose(
                        translation=_xyz(camera.pose.position) - WORLD_TRANSLATION_M,
                        orientation=np.asarray(camera.pose.quaternion))
            return result

        isaac.update_layout = update_isaac
    return {**adapter_identity(), "installed": True, "scope": "this environment only; before reset stepping"}


def cpu_frame_geometry_proof(source_config, data_root):
    """Compile both actual robot XMLs on CPU, without controllers or renderers.

    Proves initial world anchors/table/object/camera/light translation only.
    It deliberately does not compare robot link shapes, masses or PD behavior.
    Object marker geoms represent their saved origins, not collision meshes;
    subsequent real-scene contact/standing checks remain required.
    """
    import mujoco
    adapted, _ = adapt_official_config(source_config)
    root = Path(data_root).resolve(strict=True)
    records = []
    for native, state, relative in ((True, source_config, "robots/g1/g1_29dof_wholebody_dex3.xml"),
            (False, adapted, "robots/g1_sonic/g1_29dof_with_hand.xml")):
        path = (root / relative).resolve(strict=True)
        spec = mujoco.MjSpec()
        robot = mujoco.MjSpec.from_file(str(path))
        spatial = state["dr_state_dict"]["spatial"]
        pose = spatial["g1_wholebody" if native else "g1_sonic"]
        frame = spec.worldbody.add_frame(pos=pose["position"], quat=pose["quaternion"])
        spec.attach(robot, frame=frame)
        floor_z = -.785 if native else 0.
        spec.worldbody.add_geom(name="proof_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[2, 2, .1], pos=[0, 0, floor_z])
        table = state["dr_state_dict"]["scene"]["table"]
        spec.worldbody.add_geom(name="proof_table", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.asarray(table["size"]) / 2, pos=table["pose"]["position"], quat=table["pose"]["quaternion"])
        for uid, obj in spatial.items():
            if uid.startswith("g1_"):
                continue
            spec.worldbody.add_geom(name="proof_object_" + uid, type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[.001, 0, 0], pos=obj["position"])
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        points = {"pelvis": data.body("pelvis").xpos.copy(), "floor": np.array([0., 0., floor_z]),
            "table_top": data.geom("proof_table").xpos + np.array([0, 0, table["size"][2] / 2])}
        for uid in spatial:
            if not uid.startswith("g1_"):
                points["object_" + uid] = data.geom("proof_object_" + uid).xpos.copy()
        records.append({"robot_xml": str(path), "robot_xml_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "nq": model.nq, "nv": model.nv, "points": {k: v.tolist() for k, v in points.items()}})
    errors = {name: (np.asarray(records[1]["points"][name]) - records[0]["points"][name]
        - WORLD_TRANSLATION_M).tolist() for name in records[0]["points"]}
    maximum = max(float(np.max(np.abs(value))) for value in errors.values())
    return {"contract": CONTRACT, "passed": maximum <= 1e-12, "maximum_translation_error_m": maximum,
        "per_anchor_error_m": errors, "models": records, "physics_steps": 0,
        "renderers_created": 0, "robot_physics_equivalent": False,
        "scope": "actual robot XML pelvis and static saved-origin markers; not a collision or standing acceptance"}
