"""Explicit reset-time scene proposals for a separate TRAIN random100 release.

This module never edits an asset, starts a simulator or adds a control-time
force. Apply the returned configuration before gym.reset and use
``transformed_initial`` with the teacher cold start. A proposal is not evidence
of application: ``audit_scene_readback`` must observe the actual new scene.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
import fnmatch
import hashlib
import json
import math
from pathlib import Path
import random

REPO = Path(__file__).resolve().parents[2]
MATERIAL_CATALOG = (REPO / "VLMExpert/third_party/SIMPLE/src/simple/resources"
                    / "vMaterials_2/material_split.npy")
CONTRACT = "cvpr-handover-random100-scene-proposal-v2"
AUDIT_CONTRACT = "cvpr-handover-random100-scene-readback-v2"
PBR_KEYS = ("reflection_roughness_constant", "metallic_constant", "specular_level")
LIGHT_KEYS = ("light_intensity", "light_color_temperature", "light_radius", "light_length")
TABLE_MATERIAL_CATEGORIES = frozenset(("Wood", "Metal", "Stone", "Concrete", "Plastic", "Composite", "Ceramic", "Paint"))
EXCLUDED_MATERIAL_TERMS = ("glass", "mirror", "translucent", "transparent", "water", "liquid", "ice",
                           "emissive", "emission", "glow", "luminous", "thin", "clear", "crystal", "subsurface", "frosted")


class SceneReadbackRejected(ValueError):
    """Retain the actual composed-stage evidence when application is rejected."""
    def __init__(self, message, snapshot):
        super().__init__(message)
        self.snapshot = deepcopy(snapshot)
        self.receipt = {"contract": AUDIT_CONTRACT, "status": "actual_scene_readback_rejected",
                        "application_verified": False, "error": message,
                        "actual_readback": self.snapshot}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def finite(value, name):
    require(type(value) in (int, float) and math.isfinite(value), f"Require finite {name}")
    return float(value)


def vector(value, width, name):
    require(isinstance(value, (list, tuple)) and len(value) == width, f"Require {name}[{width}]")
    return [finite(item, name) for item in value]


def pose7(value, name):
    result = vector(value, 7, name)
    require(abs(math.sqrt(sum(x * x for x in result[3:])) - 1) < 1e-4,
            f"Require normalized {name} quaternion")
    return result


def _pose_dict(value):
    return {"position": value[:3], "quaternion": value[3:]}


def _pose_vector(value, name):
    return pose7([*value["position"], *value["quaternion"]], name)


def _same_pose(first, second, *, atol=1e-8):
    a, b = pose7(first, "actual pose"), pose7(second, "expected pose")
    sign = 1 if sum(x * y for x, y in zip(a[3:], b[3:])) >= 0 else -1
    return (max(abs(x - y) for x, y in zip(a[:3], b[:3])) <= atol
            and max(abs(x - sign * y) for x, y in zip(a[3:], b[3:])) <= atol)


def _catalog_train_materials(resources_root):
    import numpy as np
    split = np.load(MATERIAL_CATALOG, allow_pickle=True).item()
    result = []
    for material in split["train"]["table"]:
        description = (material["path"] + " " + material["name"]).lower()
        if any(term in description for term in EXCLUDED_MATERIAL_TERMS):
            continue
        relative = Path(material["path"])
        if str(relative).startswith("data/"):
            relative = Path(*relative.parts[1:])
        require(not relative.is_absolute() and ".." not in relative.parts
                and relative.parts[0] == "vMaterials_2" and relative.suffix == ".mdl",
                "TRAIN material catalogue contains an unsafe resource path")
        if len(relative.parts) < 3 or relative.parts[1] not in TABLE_MATERIAL_CATEGORIES:
            continue
        path = resources_root / relative
        if path.is_file():
            result.append({"path": str(path), "name": material["name"]})
    require(result, "No actual local TRAIN table MDL is available")
    return result


def _proposal_identity(proposal):
    return digest({key: value for key, value in proposal.items() if key != "proposal_sha256"})


def _verify_proposal(proposal):
    require(proposal.get("contract") == CONTRACT and proposal.get("split") == "train"
            and proposal.get("proposal_sha256") == _proposal_identity(proposal),
            "Scene proposal contract or frozen identity changed")
    require(proposal.get("material_seed_is_application_evidence") is False
            and proposal.get("application_verified") is False,
            "A frozen proposal must not self-certify actual application")


def randomize_config(source_config, rng_seed, resources_root, *, xy_max_m=.03, yaw_max_rad=.10):
    """Return a frozen, continuous TRAIN proposal and pre-reset configuration.

    XY bounds apply to each axis independently, not to the radial distance.
    No global RNG is used. Container shaders remain unchanged because the
    current asset has no portable, verified matching shader path.
    """
    require(type(rng_seed) is int and 0 <= rng_seed < 2**63, "Require a nonnegative explicit RNG seed")
    xy = finite(xy_max_m, "XY maximum"); yaw = finite(yaw_max_rad, "yaw maximum")
    require(.01 <= xy <= .06 and .05 <= yaw <= .35, "Use v2 XY .01..06m and yaw .05..35rad bounds")
    root = Path(resources_root).resolve(strict=True)
    require(root.is_dir(), "Local scene resources must exist")
    source = deepcopy(source_config); config = deepcopy(source_config)
    require(source.get("uid") in ("g1_wholebody_handover_teleop", "g1_wholebody_handover_sonic"),
            "Require the Handover source configuration")
    state, actors = config["dr_state_dict"], config["layout"]["actors"]
    require(state["scene"]["uid"] == "hssd:scene0" and config["layout"]["scene"]["uid"] == "hssd:scene0",
            "This release has only the actual locally installed scene0")
    rng = random.Random(rng_seed)
    poses = {}
    for role in ("target", "container"):
        actor = actors[role]
        old = _pose_vector(actor["pose"], role)
        key = f"container_{actor['uid']}" if role == "container" else str(actor["uid"])
        require(_same_pose(_pose_vector(state["spatial"][key], role + " DR"), old),
                "Source layout and restored spatial state disagree")
        dx, dy, turn = rng.uniform(-xy, xy), rng.uniform(-xy, xy), rng.uniform(-yaw, yaw)
        applied = old.copy(); applied[0] += dx; applied[1] += dy
        c, s = math.cos(turn / 2), math.sin(turn / 2)
        w, x, y, z = old[3:]
        applied[3:] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]
        actors[role]["pose"] = _pose_dict(applied)
        state["spatial"][key] = _pose_dict(applied)
        poses[role] = {"source_pose7": old, "applied_pose7": applied, "xy_delta_m": [dx, dy],
                       "world_z_yaw_delta_rad": turn, "spatial_key": key,
                       "asset_label": actor["asset"]["label"]}

    materials = state["material"]
    available = _catalog_train_materials(root)
    original_table = materials["table_material"]
    choices = [item for item in available if (Path(item["path"]).resolve(), item["name"])
               != (Path(original_table["path"]).resolve(), original_table["name"])]
    require(choices, "Need a different actual TRAIN MDL for the new reset")
    table = deepcopy(rng.choice(choices)); materials["table_material"] = table
    def pbr():
        return {"reflection_roughness_constant": rng.uniform(.15, .90),
                "metallic_constant": rng.uniform(0., .55), "specular_level": rng.uniform(.05, .75)}
    materials["robot_shader_params"] = pbr()
    object_roles = [role for role, actor in actors.items() if "asset" in actor]
    require(len(materials["object_shader_params"]) == len(object_roles)
            and "target" in object_roles and "container" in object_roles,
            "Material slots must match the restored ObjectActor order")
    materials["object_shader_params"][object_roles.index("target")] = pbr()
    # MaterialDR applies the same values stored on the layout during reset.
    actors["table"]["material"] = deepcopy(table)
    for role, parameters in zip(object_roles, materials["object_shader_params"]):
        actors[role]["material"] = deepcopy(parameters)

    lights = state["lighting"]
    require(len(lights) == 6 and {item["uid"] for item in lights.values()} == set(lights),
            "Keep the installed six-light scene topology")
    center = [rng.uniform(-1.1, 1.1), rng.uniform(-1.1, 1.1), rng.uniform(1.3, 1.8)]
    angle = rng.uniform(-math.pi / 2, math.pi / 2)
    center_q = [math.cos(angle / 2), 0., 0., math.sin(angle / 2)]
    spacing_x, spacing_y = rng.uniform(1., 2.5), rng.uniform(1., 2.5)
    for ordinal, (uid, light) in enumerate(lights.items()):
        require(light["type"] == "CylinderLight", "Only actual installed CylinderLights are supported")
        i, j = divmod(ordinal, 3)
        light["pose"] = {"position": [(j - 1) * spacing_x, (i - .5) * spacing_y, 0.]}
        light.update(light_intensity=rng.uniform(25000., 75000.),
                     light_color_temperature=rng.uniform(4500., 8500.),
                     light_radius=rng.uniform(.06, .15), light_length=rng.uniform(.5, 2.1),
                     center_light_postion=list(center), center_light_orientation=list(center_q))
    config["layout"]["lights"] = deepcopy(list(lights.values()))
    proposal = {"contract": CONTRACT, "split": "train", "rng_seed": rng_seed,
        "source_config_sha256": digest(source), "config_sha256": digest(config),
        "resources_root": str(root), "scene_uid": "hssd:scene0", "multiple_rooms_claimed": False,
        "distributions": {"xy_per_axis_m": [-xy, xy], "world_z_yaw_rad": [-yaw, yaw],
            "bounds_contract": "random100-reset-bounds-v2", "maximum_supported_xy_per_axis_m": .06,
            "maximum_supported_world_z_yaw_rad": .35,
            "light_intensity": [25000., 75000.], "light_color_temperature_K": [4500., 8500.],
            "light_radius_m": [.06, .15], "light_length_m": [.5, 2.1],
            "table_MDL_policy": "existing TRAIN table-surface categories and opaque-name filtering; not proof of optical opacity",
            "table_category_whitelist": sorted(TABLE_MATERIAL_CATEGORIES),
            "eligible_existing_TRAIN_table_material_count": len(available),
            "alternative_existing_TRAIN_table_material_count": len(choices),
            "excluded_material_name_terms": list(EXCLUDED_MATERIAL_TERMS),
            "sampling": "independent continuous uniform draws; no direction template"},
        "objects": poses, "object_shader_roles": object_roles,
        "material": {"table_material": table, "table_mdl_sha256": sha(table["path"]),
            "catalog_sha256": sha(MATERIAL_CATALOG), "catalog_split": "train",
            "robot_shader_params": deepcopy(materials["robot_shader_params"]),
            "target_shader_params": deepcopy(materials["object_shader_params"][object_roles.index("target")]),
            "container_material_randomized": False, "ground_material_randomized": False},
        "lighting": deepcopy(lights), "preserved_sensor_cfgs": deepcopy(source["sensor_cfgs"]),
        "preserved_robot_cfg": deepcopy(source["robot_cfg"]), "object_assets_changed": False,
        "mass_or_contact_properties_randomized": False, "camera_calibration_changed": False,
        "robot_initial_pose_changed": False, "control_time_restores_or_forces": False,
        "application_verified": False, "material_seed_is_application_evidence": False}
    proposal["proposal_sha256"] = _proposal_identity(proposal)
    return config, proposal


def transformed_initial(initial, proposal):
    """Keep the source robot/velocities/masses while applying the same reset poses."""
    _verify_proposal(proposal)
    result = deepcopy(initial)
    for role, item in proposal["objects"].items():
        require(_same_pose(initial["objects"][role]["pose7"], item["source_pose7"]),
                "Cold-start source and scene proposal disagree")
        result["objects"][role]["pose7"] = list(item["applied_pose7"])
    return result


def audit_scene_snapshot(snapshot, proposal, expected_initial):
    """Pure admission of actual readback; fake/missing shader application fails."""
    _verify_proposal(proposal)
    require(snapshot.get("phase") == "before_warmup" and snapshot.get("clock_advanced") is False,
            "Scene reset evidence must precede warmup without advancing physics")
    require(snapshot.get("scene_uid") == proposal["scene_uid"] == "hssd:scene0",
            "Actual scene differs from the installed scene0 proposal")
    require(snapshot.get("sensor_cfgs") == proposal["preserved_sensor_cfgs"]
            and snapshot.get("robot_cfg") == proposal["preserved_robot_cfg"],
            "Source camera calibration/robot configuration changed")
    require(_same_pose(snapshot["root_pose7"], expected_initial["root_pose7"]), "Source robot root pose changed")
    for role, item in proposal["objects"].items():
        actual = snapshot["objects"][role]
        require(_same_pose(actual["pose7"], item["applied_pose7"])
                and _same_pose(expected_initial["objects"][role]["pose7"], item["applied_pose7"]),
                f"Actual reset pose differs from the frozen proposal: {role}")
        require(math.isclose(finite(actual["mass"], "actual mass"),
                             finite(expected_initial["objects"][role]["mass"], "source mass"),
                             rel_tol=1e-5, abs_tol=1e-7), f"Source object mass changed: {role}")
    require(snapshot["joint_names"] == expected_initial["joint_names"]
            and len(snapshot["joint_qpos"]) == len(expected_initial["joint_qpos"])
            and all(math.isclose(finite(a, "actual joint"), finite(b, "source joint"), rel_tol=0, abs_tol=1e-8)
                    for a, b in zip(snapshot["joint_qpos"], expected_initial["joint_qpos"])),
            "Source named robot joint pose changed")
    require(set(snapshot["lighting"]) == set(proposal["lighting"]), "Actual light topology changed")
    for uid, expected in proposal["lighting"].items():
        actual = snapshot["lighting"][uid]
        require(actual.get("enable_color_temperature") is True,
                f"Actual light color temperature is disabled: {uid}")
        require(all(math.isclose(finite(actual[key], key), finite(expected[key], key), rel_tol=1e-6, abs_tol=1e-6)
                    for key in LIGHT_KEYS), f"Actual USD light parameters differ: {uid}")
        for key, wanted in (("local_position", expected["pose"]["position"]),
                            ("center_position", expected["center_light_postion"])):
            require(all(math.isclose(a, b, rel_tol=0, abs_tol=1e-6)
                        for a, b in zip(vector(actual[key], 3, key), wanted)), f"Actual light position differs: {uid}")
        require(_same_pose([*actual["center_position"], *actual["center_quaternion_wxyz"]],
                           [*expected["center_light_postion"], *expected["center_light_orientation"]], atol=1e-6),
                f"Actual light group rotation differs: {uid}")
    material = snapshot["material"]
    require(material.get("table_bound") is True and material.get("table_mdl_shader_count", 0) > 0,
            "Table has no actual bound MDL shader")
    require(Path(material["table_mdl_path"]).resolve(strict=True)
            == Path(proposal["material"]["table_material"]["path"]).resolve(strict=True)
            and material["table_mdl_name"] == proposal["material"]["table_material"]["name"]
            and sha(material["table_mdl_path"]) == proposal["material"]["table_mdl_sha256"],
            "Actual table MDL binding differs from the selected TRAIN resource")
    for role in ("target", "robot"):
        shaders = material.get(role + "_shaders", [])
        require(shaders, f"Zero matching actual {role} shaders; parameter dictionaries are not application evidence")
        expected = proposal["material"][role + "_shader_params"]
        for shader in shaders:
            require(shader.get("bound_consumers"), f"Actual {role} shader has no visible bound geometry")
            mapping = shader_parameter_mapping(shader.get("identity", {}))
            require(shader.get("logical_to_shader_input") == mapping,
                    f"Actual {role} shader parameter semantics are missing")
            require(shader.get("path") and all(math.isclose(finite(shader["inputs"].get(key), key), expected[key],
                                                           rel_tol=1e-6, abs_tol=1e-6) for key in PBR_KEYS),
                    f"Actual {role} shader values differ")
    require(snapshot.get("camera_intrinsics_match_source") is True, "Actual camera intrinsics changed")
    return {"contract": AUDIT_CONTRACT, "status": "actual_reset_lighting_material_readback_passed",
        "proposal_sha256": proposal["proposal_sha256"], "snapshot_sha256": digest(snapshot),
        "application_verified": True, "table_MDL_binding_verified": True,
        "target_and_robot_PBR_inputs_verified": True, "container_material_randomized": False,
        "ground_material_randomized": False, "multiple_rooms_claimed": False,
        "camera_calibration_changed": False, "robot_initial_pose_changed": False,
        "mass_verified_against_source": True, "rendered_pixel_counterfactual_verified": False,
        "full_renderer_pose_and_physics_admission_still_required": True,
        "scope": "Actual pre-warmup CPU physics and composed Isaac stage readback; physical trajectory and RGB quality still require full independent capture admission",
        "actual_readback": deepcopy(snapshot)}


def _plain(value):
    if is_dataclass(value):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def engine_robot_shader_namespace(workspace, robot_ns):
    """Match add_robot/__update_robot before wrapper articulation initialization.

    IsaacRobot.prim_path can resolve to a descendant articulation root after
    initialize(). Looks belongs to the source namespace, never that descendant.
    This is the engine's original write target, not a fallback shader search.
    """
    require(isinstance(robot_ns, str) and robot_ns and "/" not in robot_ns and robot_ns not in (".", ".."),
            "Require the exact source robot namespace")
    return workspace.rstrip("/") + "/Robot/" + robot_ns


def matching_shader_paths(paths, prefix, pattern):
    """Keep the engine's exact semantic write pattern for readback."""
    return [path for path in paths if fnmatch.fnmatch(path, prefix + pattern)]


def shader_parameter_mapping(identity):
    """Support only a verified consuming shader, never invented input names.

    UsdPreviewSurface has a different specular workflow and is deliberately
    rejected until a separately reviewed complete mapping exists.
    """
    require(identity.get("implementation_source") == "sourceAsset"
            and Path(identity.get("source_asset", "")).name == "OmniPBR.mdl"
            and identity.get("source_subidentifier") == "OmniPBR"
            and identity.get("shader_id") in (None, ""),
            "Unsupported shader semantics: require actual MDL OmniPBR; UsdPreviewSurface is not an MDL alias")
    return {key: key for key in PBR_KEYS}


def _shader_identity(prim):
    asset = prim.GetAttribute("info:mdl:sourceAsset").Get()
    return {"shader_id": prim.GetAttribute("info:id").Get(),
        "implementation_source": prim.GetAttribute("info:implementationSource").Get(),
        "source_asset": getattr(asset, "path", str(asset)) if asset is not None else "",
        "source_subidentifier": prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()}


def _all_shaders(stage):
    from pxr import Usd, UsdShade
    return {str(prim.GetPath()): prim for prim in
            Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
            if prim.IsA(UsdShade.Shader)}


def _physics_identity(raw):
    model, data = raw.mujoco.mjModel, raw.mujoco.mjData
    values = {"sim_time_s": float(data.time)}
    for owner, names in ((data, ("qpos", "qvel", "qacc", "ctrl", "qfrc_applied", "xfrc_applied")),
                         (model, ("body_mass", "body_inertia", "geom_friction", "body_pos", "body_quat", "cam_fovy"))):
        for name in names:
            value = getattr(owner, name)
            values[name] = hashlib.sha256(value.tobytes()).hexdigest()
    return values


def _bound_consuming_shaders(stage, prefix):
    """Follow actual visible geometry -> bound material -> MDL surface output."""
    from pxr import Usd, UsdGeom, UsdShade
    materials = {}
    for candidate in Usd.PrimRange(stage.GetPrimAtPath(prefix), Usd.TraverseInstanceProxies()):
        if not candidate.IsA(UsdGeom.Gprim):
            continue
        imageable = UsdGeom.Imageable(candidate)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible or imageable.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        bound, _ = UsdShade.MaterialBindingAPI(candidate).ComputeBoundMaterial()
        if bound:
            entry = materials.setdefault(str(bound.GetPath()), {"material": bound, "consumers": []})
            entry["consumers"].append(str(candidate.GetPath()))
    found, skipped = [], []
    for path, entry in materials.items():
        output = entry["material"].GetSurfaceOutput("mdl")
        connected = output.GetConnectedSource() if output else None
        if not connected:
            skipped.append({"material": path, "reason": "no_connected_mdl_surface", "consumers": entry["consumers"]})
            continue
        prim = connected[0].GetPrim(); identity = _shader_identity(prim)
        try:
            mapping = shader_parameter_mapping(identity)
        except ValueError:
            skipped.append({"material": path, "shader": str(prim.GetPath()), "identity": identity,
                            "reason": "unsupported_consuming_shader_unchanged", "consumers": entry["consumers"]})
            continue
        found.append({"path": str(prim.GetPath()), "prim": prim, "identity": identity,
                      "mapping": mapping, "consumers": entry["consumers"], "material": path})
    return found, skipped


def apply_scene_visual_parameters(raw, proposal):
    """Apply verified MDL inputs in the new stage before any control/history.

    Does not render, step, change physics, edit source assets or bind a different
    material. The caller must refresh the actual observation afterward.
    """
    from pxr import Sdf, UsdShade
    _verify_proposal(proposal)
    require(raw.isaac is not None and float(raw.mujoco.mjData.time) <= .005 + 1e-9
            and getattr(raw, "step_count", 0) == 0, "Visual changes are reset-time only, before warmup/control")
    before = _physics_identity(raw); isaac = raw.isaac; stage = isaac.world.stage
    planned, skipped = [], {}
    for role in ("target", "robot"):
        prefix = (engine_robot_shader_namespace(isaac.workspace_prim_path, raw.task.robot.robot_ns) if role == "robot"
                  else isaac.objects[raw.task.layout.actors[role].asset.label]["object_prim_path"])
        matches, skipped[role] = _bound_consuming_shaders(stage, prefix)
        require(matches, f"Zero bound consuming {role} OmniPBR shaders; no application can be certified")
        planned.extend((role, value["path"], value["prim"], value["identity"], value["mapping"], value["consumers"])
                       for value in matches)
    changes = []
    for role, path, prim, identity, mapping, consumers in planned:
        shader = UsdShade.Shader(prim); old = {key: prim.GetAttribute("inputs:" + name).Get() for key, name in mapping.items()}
        values = proposal["material"][role + "_shader_params"]
        for key, name in mapping.items():
            shader.CreateInput(name, Sdf.ValueTypeNames.Float).Set(values[key])
        actual = {key: prim.GetAttribute("inputs:" + name).Get() for key, name in mapping.items()}
        require(all(math.isclose(finite(actual[key], key), values[key], rel_tol=1e-6, abs_tol=1e-6) for key in mapping),
                "Actual consuming shader did not retain the requested parameter")
        changes.append({"role": role, "path": path, "identity": identity,
                        "logical_to_shader_input": mapping, "bound_consumers": consumers,
                        "before": _plain(old), "after": _plain(actual)})
    require(_physics_identity(raw) == before, "Visual application changed physics/time")
    return {"contract": "cvpr-handover-random100-visual-application-v1", "proposal_sha256": proposal["proposal_sha256"],
        "status": "actual_bound_OmniPBR_inputs_written_and_read_back", "changes": changes,
        "unsupported_or_non_surface_materials_unchanged": skipped,
        "physics_before_and_after": before, "physics_unchanged": True, "source_assets_edited": False,
        "observation_refresh_required_before_capture": True, "pixel_effect_certified": False}


def audit_scene_readback(raw, proposal, *, expected_initial):
    """Read an already initialized actual environment; never render or step it.

    Call immediately after the cold-start initializer and before warmup. The
    caller retains this returned JSON receipt with the frozen config/proposal.
    This function imports USD only when an actual Isaac scene already exists.
    """
    from pxr import Usd, UsdShade
    require(raw.isaac is not None, "A CPU screen alone cannot certify material/light application")
    model, data, isaac = raw.mujoco.mjModel, raw.mujoco.mjData, raw.isaac
    before = float(data.time); stage = isaac.world.stage
    all_shaders = _all_shaders(stage)
    robot_prefix = engine_robot_shader_namespace(isaac.workspace_prim_path, raw.task.robot.robot_ns)
    snapshot = {"phase": "before_warmup", "sim_time_s": before, "scene_uid": raw.task.layout.scene.uid,
        "sensor_cfgs": _plain(raw.task.sensor_cfgs), "robot_cfg": _plain(raw.task.robot_cfg),
        "root_pose7": [*data.body("pelvis").xpos.tolist(), *data.body("pelvis").xquat.tolist()],
        "joint_names": list(expected_initial["joint_names"]),
        "joint_qpos": [float(data.joint(name).qpos[0]) for name in expected_initial["joint_names"]],
        "objects": {}, "lighting": {}, "material": {}}
    snapshot["shader_discovery"] = {"robot_api_prim_path": str(isaac.robot.prim_path),
        "robot_engine_namespace": robot_prefix,
        "method": "source engine namespace/pattern; include actual USD instance proxies; no shader writes",
        "robot_shader_inventory": [{"path": path,
            "input_names": [str(value.GetBaseName()) for value in UsdShade.Shader(prim).GetInputs()],
            "PBR_readback": {key: prim.GetAttribute("inputs:" + key).Get() for key in PBR_KEYS}}
            for path, prim in all_shaders.items() if path.startswith(isaac.workspace_prim_path + "/Robot/")]}
    for role in ("target", "container"):
        body = raw.mujoco.mj_objects[role]
        geoms = [index for index, body_id in enumerate(model.geom_bodyid) if int(body_id) == body.id]
        snapshot["objects"][role] = {"pose7": [*body.xpos.tolist(), *body.xquat.tolist()],
            "mass": float(model.body_mass[body.id]), "body_inertia": model.body_inertia[body.id].tolist(),
            "geom_friction": {str(index): model.geom_friction[index].tolist() for index in geoms}}
    center_p, center_q = isaac.light_center.get_local_pose()
    require(len(isaac.lights) == len(raw.task.layout.lights) == len(proposal["lighting"]),
            "Actual Isaac light count differs from the frozen topology")
    for prim, expected in zip(isaac.lights, raw.task.layout.lights):
        snapshot["lighting"][expected.uid] = {
            **{key: float(prim.GetAttribute("inputs:" + name).Get()) for key, name in (
                ("light_intensity", "intensity"), ("light_color_temperature", "colorTemperature"),
                ("light_radius", "radius"), ("light_length", "length"))},
            "enable_color_temperature": bool(prim.GetAttribute("inputs:enableColorTemperature").Get()),
            "local_position": list(prim.GetAttribute("xformOp:translate").Get()),
            "center_position": _plain(center_p), "center_quaternion_wxyz": _plain(center_q)}
    table = stage.GetPrimAtPath(isaac.workspace_prim_path + "/table_cuboid")
    bound, _ = UsdShade.MaterialBindingAPI(table).ComputeBoundMaterial()
    material = snapshot["material"]; material.update(table_bound=bool(bound), table_mdl_shader_count=0)
    if bound:
        candidates = []
        for path, prim in all_shaders.items():
            if not path.startswith(str(bound.GetPath()) + "/"):
                continue
            asset = prim.GetAttribute("info:mdl:sourceAsset").Get()
            name = prim.GetAttribute("info:mdl:sourceAsset:subIdentifier").Get()
            if asset is not None and name:
                candidates.append((getattr(asset, "resolvedPath", "") or getattr(asset, "path", str(asset)), str(name)))
        material["table_mdl_shader_count"] = len(candidates)
        require(len(candidates) == 1, "Require one unambiguous actual table MDL source")
        material["table_mdl_path"], material["table_mdl_name"] = candidates[0]
    for role in ("target", "robot"):
        prefix = (robot_prefix if role == "robot" else
                  isaac.objects[raw.task.layout.actors[role].asset.label]["object_prim_path"])
        matches, skipped = _bound_consuming_shaders(stage, prefix)
        snapshot["shader_discovery"][role + "_unsupported_materials_unchanged"] = skipped
        snapshot["shader_discovery"][role + "_application_selection"] = "visible bound geometry -> MDL surface connection -> verified OmniPBR"
        material[role + "_shaders"] = [{"path": item["path"], "identity": item["identity"],
            "logical_to_shader_input": item["mapping"], "bound_consumers": item["consumers"],
            "inputs": {key: item["prim"].GetAttribute("inputs:" + key).Get() for key in PBR_KEYS}} for item in matches]
    intrinsics = []
    for name, camera in isaac.cameras.items():
        source = raw.task.layout.cameras[name]
        fovy = math.degrees(2 * math.atan(source.resolution[1] / (2 * source.fy)))
        intrinsics.append(math.isclose(float(model.cam_fovy[model.camera(name).id]), fovy, rel_tol=0, abs_tol=1e-6)
            and math.isclose(float(camera.get_focal_length()), source.focal_length, rel_tol=1e-6, abs_tol=1e-6)
            and math.isclose(float(camera.get_horizontal_aperture()), source.focal_length * source.resolution[0] / source.fx,
                             rel_tol=1e-6, abs_tol=1e-6)
            and math.isclose(float(camera.get_vertical_aperture()), source.focal_length * source.resolution[1] / source.fy,
                             rel_tol=1e-6, abs_tol=1e-6))
    snapshot["camera_intrinsics_match_source"] = bool(intrinsics) and all(intrinsics)
    snapshot["clock_advanced"] = float(data.time) != before
    snapshot = _plain(snapshot)
    try:
        return audit_scene_snapshot(snapshot, proposal, expected_initial)
    except (ValueError, KeyError, TypeError) as error:
        raise SceneReadbackRejected(str(error), snapshot) from error
