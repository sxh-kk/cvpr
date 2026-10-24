"""CPU contract checks; fake fixtures never certify actual RTX application."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

SPEC = importlib.util.spec_from_file_location("handover_random100_scene",
    Path(__file__).resolve().parents[1] / "handover_random100_scene.py")
scene = importlib.util.module_from_spec(SPEC); SPEC.loader.exec_module(scene)


def source_config(root):
    pbr = dict(zip(scene.PBR_KEYS, [.5, .1, .2]))
    actors = {"table": {"pose": {"position": [0., 0., .5], "quaternion": [1., 0., 0., 0.]}, "size": [1., 1., .1]},
        "container": {"uid": "0", "asset": {"label": "basket"},
            "pose": {"position": [-.5, .4, .55], "quaternion": [1., 0., 0., 0.]}},
        "target": {"uid": "0", "asset": {"label": "cracker_box"},
            "pose": {"position": [-.5, 0., .65], "quaternion": [1., 0., 0., 0.]}}}
    spatial = {"container_0": deepcopy(actors["container"]["pose"]), "0": deepcopy(actors["target"]["pose"]),
               "g1_sonic": {"position": [-.7, 0., 0.], "quaternion": [1., 0., 0., 0.]}}
    lights = {f"Light_{i}_{j}": {"uid": f"Light_{i}_{j}", "type": "CylinderLight",
        "pose": {"position": [0., 0., 0.]}, "light_intensity": 50000., "light_color_temperature": 6500.,
        "light_radius": .1, "light_length": 1., "center_light_postion": [0., 0., 1.5],
        "center_light_orientation": [1., 0., 0., 0.]} for i in range(2) for j in range(3)}
    return {"uid": "g1_wholebody_handover_teleop", "robot_cfg": {"uid": "g1_sonic"},
        "sensor_cfgs": {"head_stereo": {"width": 640, "height": 360, "fov": 1.92, "baseline": .05}},
        "layout": {"actors": actors, "scene": {"uid": "hssd:scene0"}, "lights": list(lights.values())},
        "dr_state_dict": {"spatial": spatial, "scene": {"uid": "hssd:scene0"}, "lighting": lights,
            "material": {"table_material": {"path": str(root / "vMaterials_2/old.mdl"), "name": "Old"},
                "ground_material": {"path": "ground", "name": "unchanged"},
                "robot_shader_params": deepcopy(pbr), "object_shader_params": [deepcopy(pbr), deepcopy(pbr)]}}}


def initial(config):
    return {"root_pose7": [0., 0., .75, 1., 0., 0., 0.], "root_qvel6": [0.] * 6,
        "joint_names": [f"joint{i}" for i in range(43)], "joint_qpos": [0.] * 43, "joint_qvel": [0.] * 43,
        "objects": {role: {"mass": .1, "pose7": scene._pose_vector(config["layout"]["actors"][role]["pose"], role)}
                    for role in ("target", "container")}}


def accepted_snapshot(proposal, cold):
    lights = {uid: {**{key: value[key] for key in scene.LIGHT_KEYS}, "enable_color_temperature": True,
        "local_position": value["pose"]["position"], "center_position": value["center_light_postion"],
        "center_quaternion_wxyz": value["center_light_orientation"]} for uid, value in proposal["lighting"].items()}
    material = {"table_bound": True, "table_mdl_shader_count": 1,
        "table_mdl_path": proposal["material"]["table_material"]["path"],
        "table_mdl_name": proposal["material"]["table_material"]["name"]}
    for role in ("target", "robot"):
        material[role + "_shaders"] = [{"path": "/fixture/" + role,
            "bound_consumers": ["/fixture/" + role + "/mesh"],
            "identity": {"implementation_source": "sourceAsset", "source_asset": "OmniPBR.mdl",
                         "source_subidentifier": "OmniPBR", "shader_id": None},
            "logical_to_shader_input": {key: key for key in scene.PBR_KEYS},
            "inputs": deepcopy(proposal["material"][role + "_shader_params"])}]
    return {"phase": "before_warmup", "clock_advanced": False, "scene_uid": "hssd:scene0", "sensor_cfgs": proposal["preserved_sensor_cfgs"],
        "robot_cfg": proposal["preserved_robot_cfg"], "root_pose7": cold["root_pose7"],
        "joint_names": cold["joint_names"], "joint_qpos": cold["joint_qpos"],
        "objects": deepcopy(cold["objects"]), "lighting": lights, "material": material,
        "camera_intrinsics_match_source": True}


class SceneRandomizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        (self.root / "vMaterials_2").mkdir()
        (self.root / "vMaterials_2/Wood").mkdir()
        for name in ("old", "new", "heldout"):
            (self.root / f"vMaterials_2/{name}.mdl").write_text("fixture " + name)
        (self.root / "vMaterials_2/Wood/new.mdl").write_text("fixture new")
        self.catalog = self.root / "material_split.npy"
        np.save(self.catalog, {"train": {"table": [{"path": "data/vMaterials_2/Wood/new.mdl", "name": "New"}]},
            "val": {"table": [{"path": "data/vMaterials_2/heldout.mdl", "name": "Heldout"}]}})
        self.patch = patch.object(scene, "MATERIAL_CATALOG", self.catalog); self.patch.start()
        self.source = source_config(self.root)

    def tearDown(self):
        self.patch.stop(); self.temp.cleanup()

    def build(self, seed=3, **kwargs):
        return scene.randomize_config(self.source, seed, self.root, **kwargs)

    def test_deterministic_continuous_reset_and_all_proposal_bounds(self):
        source_bytes = json.dumps(self.source, sort_keys=True)
        a, proposal = self.build(); b, second = self.build()
        self.assertEqual((a, proposal), (b, second))
        self.assertEqual(json.dumps(self.source, sort_keys=True), source_bytes)
        quadrants = {role: set() for role in ("target", "container")}
        unique = set()
        for seed in range(100):
            config, item = self.build(seed)
            unique.add(item["proposal_sha256"])
            for role, value in item["objects"].items():
                dx, dy = value["xy_delta_m"]
                quadrants[role].add((dx > 0, dy > 0))
                self.assertLessEqual(max(abs(dx), abs(dy)), .03)
                self.assertLessEqual(abs(value["world_z_yaw_delta_rad"]), .1)
                self.assertEqual(config["dr_state_dict"]["spatial"][value["spatial_key"]],
                                 config["layout"]["actors"][role]["pose"])
        self.assertEqual(len(unique), 100)
        self.assertTrue(all(len(value) == 4 for value in quadrants.values()))
        self.assertFalse(proposal["application_verified"])

    def test_only_train_existing_material_and_preserved_source_contracts(self):
        config, proposal = self.build()
        self.assertEqual(proposal["material"]["table_material"]["name"], "New")
        self.assertEqual(config["sensor_cfgs"], self.source["sensor_cfgs"])
        self.assertEqual(config["robot_cfg"], self.source["robot_cfg"])
        self.assertEqual(config["dr_state_dict"]["spatial"]["g1_sonic"], self.source["dr_state_dict"]["spatial"]["g1_sonic"])
        self.assertEqual(config["dr_state_dict"]["material"]["object_shader_params"][0],
                         self.source["dr_state_dict"]["material"]["object_shader_params"][0])
        self.assertEqual(config["dr_state_dict"]["material"]["ground_material"],
                         self.source["dr_state_dict"]["material"]["ground_material"])
        cold = initial(self.source); updated = scene.transformed_initial(cold, proposal)
        for key in ("root_pose7", "root_qvel6", "joint_names", "joint_qpos", "joint_qvel"):
            self.assertEqual(cold[key], updated[key])
        for role in ("target", "container"):
            self.assertEqual(updated["objects"][role]["mass"], .1)
            self.assertEqual(updated["objects"][role]["pose7"], proposal["objects"][role]["applied_pose7"])
        self.assertEqual(scene.digest(config), proposal["config_sha256"])

    def test_fail_closed_on_unavailable_material_room_or_unknown_initial(self):
        (self.root / "vMaterials_2/Wood/new.mdl").unlink()
        with self.assertRaisesRegex(ValueError, "No actual local TRAIN"):
            self.build()
        (self.root / "vMaterials_2/Wood/new.mdl").write_text("fixture new")
        self.source["dr_state_dict"]["scene"]["uid"] = "hssd:scene1"
        with self.assertRaisesRegex(ValueError, "scene0"):
            self.build()
        self.source["dr_state_dict"]["scene"]["uid"] = "hssd:scene0"
        _, proposal = self.build(); cold = initial(self.source); cold["objects"]["target"]["pose7"][0] += .2
        with self.assertRaisesRegex(ValueError, "Cold-start"):
            scene.transformed_initial(cold, proposal)

    def test_explicit_transparent_material_names_are_not_quality_proposals(self):
        np.save(self.catalog, {"train": {"table": [{"path": "data/vMaterials_2/Wood/new.mdl", "name": "Clear_Glass"}]}})
        with self.assertRaisesRegex(ValueError, "No actual local TRAIN"):
            self.build()
        np.save(self.catalog, {"train": {"table": [{"path": "data/vMaterials_2/Gems/Emerald.mdl", "name": "Emerald"}]}})
        with self.assertRaisesRegex(ValueError, "No actual local TRAIN"):
            self.build()

    def test_real_readback_contract_rejects_zero_shaders_or_unapplied_fields(self):
        _, proposal = self.build(); cold = scene.transformed_initial(initial(self.source), proposal)
        snapshot = accepted_snapshot(proposal, cold)
        receipt = scene.audit_scene_snapshot(snapshot, proposal, cold)
        self.assertTrue(receipt["application_verified"])
        self.assertFalse(receipt["container_material_randomized"])
        for change in (lambda s: s["material"].update(target_shaders=[]),
                       lambda s: s["material"].update(robot_shaders=[]),
                       lambda s: s["material"].update(table_bound=False),
                       lambda s: s["lighting"]["Light_0_0"].update(light_intensity=1.),
                       lambda s: s["lighting"]["Light_0_0"].update(enable_color_temperature=False),
                       lambda s: s["objects"]["target"].update(mass=2.),
                       lambda s: s.update(scene_uid="hssd:scene1"),
                       lambda s: s.update(camera_intrinsics_match_source=False),
                       lambda s: s.update(clock_advanced=True)):
            bad = deepcopy(snapshot); change(bad)
            with self.assertRaises(ValueError):
                scene.audit_scene_snapshot(bad, proposal, cold)

    def test_tampered_proposal_and_unsafe_bounds_rejected(self):
        _, proposal = self.build(); proposal["objects"]["target"]["xy_delta_m"][0] = .04
        with self.assertRaisesRegex(ValueError, "frozen identity"):
            scene.transformed_initial(initial(self.source), proposal)
        config, extended = self.build(xy_max_m=.05, yaw_max_rad=.25)
        self.assertEqual(extended["distributions"]["bounds_contract"], "random100-reset-bounds-v2")
        self.assertEqual(extended["distributions"]["xy_per_axis_m"], [-.05, .05])
        for kwargs in ({"xy_max_m": .061}, {"yaw_max_rad": .36}, {"xy_max_m": float("nan")}):
            with self.assertRaises(ValueError):
                self.build(**kwargs)

    def test_engine_namespace_does_not_follow_initialized_pelvis_wrapper(self):
        namespace = scene.engine_robot_shader_namespace("/World/workspace", "g1_29dof_wholebody_dex3")
        paths = [namespace + "/Looks/material_CAD1EE/Shader", namespace + "/Looks/DefaultMaterial/DefaultMaterial",
                 "/World/workspace/other/Looks/material_CAD1EE/Shader"]
        pattern = "/Looks/material_*/Shader"
        self.assertEqual(scene.matching_shader_paths(paths, namespace, pattern), paths[:1])
        self.assertEqual(scene.matching_shader_paths(paths, namespace + "/pelvis", pattern), [])
        error = scene.SceneReadbackRejected("actual mismatch", {"shader_discovery": {"paths": paths}})
        self.assertFalse(error.receipt["application_verified"])
        self.assertEqual(error.snapshot["shader_discovery"]["paths"], paths)

    def test_shader_inputs_require_real_consuming_MDL_identity(self):
        identity = {"implementation_source": "sourceAsset", "source_asset": "OmniPBR.mdl",
                    "source_subidentifier": "OmniPBR", "shader_id": None}
        self.assertEqual(scene.shader_parameter_mapping(identity), dict(zip(scene.PBR_KEYS, scene.PBR_KEYS)))
        for changed in ({**identity, "shader_id": "UsdPreviewSurface"},
                        {**identity, "source_asset": "Other.mdl"},
                        {**identity, "source_subidentifier": "Other"}):
            with self.assertRaisesRegex(ValueError, "Unsupported shader semantics"):
                scene.shader_parameter_mapping(changed)


if __name__ == "__main__":
    unittest.main()
