"""Official TRAIN initial-pose derivation, frozen identity and safe reload tests."""
from copy import deepcopy
import contextlib
import io
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mobile_collection_plan_tests", SCRIPTS / "pico_mobile_collection_plan.py")
plan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plan)


def pose(x, y=0., z=.7):
    return {"position": [x, y, z], "quaternion": [1., 0., 0., 0.]}


def config(index):
    # Source role ordinals deliberately do not match the spatial asset keys.
    selections = {"container": ("0", "objects"), "target": ("0", "graspnet1b"),
        "distractor_0": ("18", "graspnet1b"), "distractor_1": ("17", "graspnet1b"),
        "distractor_2": ("37", "graspnet1b"), "distractor_3": ("30", "graspnet1b")}
    dr = {"spatial": {"g1_sonic": pose(-.644 + index * .00001, z=0.)}, "distractors": {},
        "scene": {"uid": "hssd:scene6", "table": {"size": [.79, 1.99, .1]}, "literal": "preserve this"},
        "material": {"ground": "same.mdl", "object_shader_params": [{"roughness": .23}]},
        "lighting": {"Light_0": {"intensity": 50000., "pose": pose(0.)}}}
    actors = {"table": {"uid": "box", "size": [.79, 1.99, .1], "pose": pose(0., z=.6)},
              "table2": {"uid": "box", "size": [1., 2.39, .1], "pose": pose(-2.94, z=.6)}}
    for number, (role, (uid, namespace)) in enumerate(selections.items()):
        asset = {"uid": uid, "res_id": namespace, "usd_path": "/home/wenke/SIMPLE/data/assets/" + uid + ".usd"}
        actors[role] = {"uid": uid, "asset": deepcopy(asset), "pose": pose(-2.5 + number * .001),
                        "material": {"roughness": .1 + number * .1}}
        key = "container_" + uid if role == "container" else uid
        dr["spatial"][key] = pose(-2.47 if role == "container" else -.31 - number * .01, y=number * .02)
        if role.startswith("distractor_"):
            dr["distractors"][uid] = deepcopy(asset)
        else:
            dr[role] = deepcopy(asset)
    return {"uid": plan.TASK_UID, "metadata": {"split": "train", "physics_dt": .005, "render_hz": 50},
        "robot_cfg": {"uid": "g1_sonic"}, "dr_cfgs": {"preserved": {"range": [1, 2, 3]}},
        "dr_state_dict": dr, "layout": {"actors": actors, "scene": deepcopy(dr["scene"]),
            "lights": [deepcopy(dr["lighting"]["Light_0"])], "cameras": {"head_stereo_left": {"resolution": [640, 360]}},
            "scene_info": {}}, "container_uid": "0"}


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.output = self.root / "episodes.jsonl", self.root / "frozen_plan"
        self.configs = [config(index) for index in range(99)]
        self.records = [{"episode_index": index, "length": 600,
                         "environment_config": json.dumps(value)} for index, value in enumerate(self.configs)]
        self.seal_source()
        self.patcher = patch.object(plan, "SOURCE_SHA256", self.source_sha)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def seal_source(self):
        self.source.write_text("".join(json.dumps(row) + "\n" for row in self.records))
        self.source_sha = plan.digest(self.source.read_bytes())

    def changed_source(self):
        self.seal_source()
        plan.SOURCE_SHA256 = self.source_sha

    def build(self):
        return plan.build_plan(self.source, self.output)

    def manifest(self):
        return plan.parse_json((self.output / "manifest.json").read_text())

    def write_manifest(self, value):
        (self.output / "manifest.json").write_bytes(plan.json_bytes(value))

    def test_build_reload99_source_bytes_hashes_absolute_paths_and_initial_not_terminal(self):
        before = self.source.read_bytes()
        manifest_path = self.build()
        rows, manifest = plan.load_plan(manifest_path), self.manifest()
        self.assertEqual([row["official_episode_index"] for row in rows], list(range(99)))
        self.assertEqual(len({row["dr_state_sha256"] for row in rows}), 99)
        self.assertEqual(len({row["spatial_state_sha256"] for row in rows}), 99)
        self.assertEqual(len({row["plan_sha256"] for row in rows}), 1)
        self.assertEqual(rows[0]["plan_sha256"], plan.digest(manifest_path.read_bytes()))
        self.assertEqual((self.output / "source_episodes.jsonl").read_bytes(), before)
        self.assertEqual(self.source.read_bytes(), before)
        self.assertEqual(Path(rows[0]["original_config_path"]).read_bytes(), self.records[0]["environment_config"].encode())
        original, actual = self.configs[0], plan.parse_json(Path(rows[0]["reset_config_path"]).read_text())
        self.assertNotEqual(original["layout"]["actors"]["target"]["pose"], original["dr_state_dict"]["spatial"]["0"])
        self.assertEqual(actual["layout"]["actors"]["target"]["pose"], original["dr_state_dict"]["spatial"]["0"])
        self.assertEqual(actual["layout"]["actors"]["container"]["pose"], original["dr_state_dict"]["spatial"]["container_0"])
        self.assertFalse(manifest["derivation"]["resource_paths_rebound"])
        self.assertTrue(Path(rows[0]["reset_config_path"]).is_absolute())

    def test_only_six_actor_poses_change_all_dr_robot_cameras_materials_preserved(self):
        original = config(0)
        original_before = deepcopy(original)
        derived, mapping = plan.derive_reset(original)
        self.assertEqual(original, original_before)
        self.assertEqual(derived["dr_state_dict"], original["dr_state_dict"])
        self.assertEqual(len(mapping), 6)
        for entry in mapping:
            role = entry["layout_role"]
            self.assertEqual(derived["layout"]["actors"][role]["pose"], original["dr_state_dict"]["spatial"][entry["initial_dr_spatial_key"]])
            derived["layout"]["actors"][role]["pose"] = deepcopy(original["layout"]["actors"][role]["pose"])
        self.assertEqual(derived, original)

    def test_distractor_uid_mapping_survives_role_reordering_and_never_uses_ordinal(self):
        original = config(0)
        actors = original["layout"]["actors"]
        actors["distractor_0"], actors["distractor_3"] = actors["distractor_3"], actors["distractor_0"]
        derived, mapping = plan.derive_reset(original)
        by_role = {row["layout_role"]: row for row in mapping}
        self.assertEqual(by_role["distractor_0"]["initial_dr_spatial_key"], "30")
        self.assertEqual(derived["layout"]["actors"]["distractor_0"]["pose"], original["dr_state_dict"]["spatial"]["30"])
        self.assertEqual(by_role["distractor_3"]["initial_dr_spatial_key"], "18")

    def test_external_original_source_can_move_after_freeze(self):
        path = self.build()
        self.source.rename(self.root / "moved_episodes.jsonl")
        self.assertEqual(len(plan.load_plan(path)), 99)

    def test_wrong_official_source_fingerprint_fails_before_creating_output(self):
        self.source.write_text(self.source.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            self.build()
        self.assertFalse(self.output.exists())

    def test_missing_duplicate_bool_episode_indices_and_duplicate_dr_fail(self):
        baseline = deepcopy(self.records)
        for change in ("missing", "duplicate", "bool", "duplicate_dr"):
            self.records = deepcopy(baseline)
            if change == "missing": self.records.pop()
            elif change == "duplicate": self.records[1]["episode_index"] = 0
            elif change == "bool": self.records[0]["episode_index"] = False
            else: self.records[1]["environment_config"] = self.records[0]["environment_config"]
            self.changed_source()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.build()
            self.assertFalse(self.output.exists())

    def test_wrong_task_split_missing_spatial_uid_mismatch_and_pose_fail_without_fallback(self):
        for change in ("task", "split", "container_key", "uid", "namespace", "target_missing", "extra_spatial", "pose"):
            original = config(0)
            if change == "task": original["uid"] = "handover"
            elif change == "split": original["metadata"]["split"] = "test"
            elif change == "container_key": original["dr_state_dict"]["spatial"].pop("container_0")
            elif change == "uid": original["layout"]["actors"]["distractor_0"]["uid"] = "17"
            elif change == "namespace": original["layout"]["actors"]["target"]["asset"]["res_id"] = "objects"
            elif change == "target_missing": original["dr_state_dict"]["spatial"].pop("0")
            elif change == "extra_spatial": original["dr_state_dict"]["spatial"]["new"] = pose(0.)
            else: original["dr_state_dict"]["spatial"]["0"]["position"] = [0., 1.]
            with self.subTest(change=change), self.assertRaises(ValueError):
                plan.derive_reset(original)

    def test_nonfinite_or_duplicate_json_values_are_not_silently_used(self):
        for value in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
            with self.subTest(value=value), self.assertRaises(ValueError): plan.parse_json(value)
        original = config(0)
        original["dr_state_dict"]["spatial"]["0"]["quaternion"] = [0., 0., 0., 0.]
        with self.assertRaises(ValueError): plan.derive_reset(original)

    def test_payload_change_or_resealed_wrong_mapping_still_refused(self):
        path = self.build()
        doc = self.manifest()
        reset_path = self.output / doc["episodes"][0]["reset_config_path"]
        reset = plan.parse_json(reset_path.read_text())
        reset["layout"]["actors"]["target"]["pose"] = deepcopy(self.configs[0]["layout"]["actors"]["target"]["pose"])
        reset_path.write_bytes(plan.json_bytes(reset))
        # Updating the claimed hash cannot replace independent derivation.
        doc["episodes"][0]["reset_config_sha256"] = plan.digest(reset_path.read_bytes())
        self.write_manifest(doc)
        with self.assertRaisesRegex(ValueError, "independent official"):
            plan.load_plan(path)

    def test_original_reset_frozen_source_and_symlink_tamper_refused(self):
        path = self.build()
        for relative in ("original/episode_000.json", "reset/episode_000.json", "source_episodes.jsonl"):
            target = self.output / relative
            payload = target.read_bytes()
            target.write_bytes(payload + b" ")
            with self.subTest(relative=relative), self.assertRaises(ValueError): plan.load_plan(path)
            target.write_bytes(payload)
        target = self.output / "reset/episode_000.json"
        real = self.root / "real_config.json"
        target.rename(real); target.symlink_to(real)
        with self.assertRaisesRegex(ValueError, "symlinks"): plan.load_plan(path)

    def test_plan_row_order_paths_or_contract_tamper_refused(self):
        path = self.build()
        baseline = self.manifest()
        for change in ("order", "path", "task", "dr", "derivation"):
            value = deepcopy(baseline)
            if change == "order": value["episodes"].reverse()
            elif change == "path": value["episodes"][0]["reset_config_path"] = "../external.json"
            elif change == "task": value["task_id"] = "handover"
            elif change == "dr": value["episodes"][0]["dr_state_sha256"] = "0" * 64
            else: value["derivation"]["resource_paths_rebound"] = True
            self.write_manifest(value)
            with self.subTest(change=change), self.assertRaises(ValueError): plan.load_plan(path)

    def test_fresh_outside_repo_directory_no_overwrite_and_cli(self):
        self.output.mkdir()
        (self.output / "user_file").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh"):
            self.build()
        self.assertEqual((self.output / "user_file").read_text(), "keep")
        repo = self.root / "repo"; repo.mkdir()
        with patch.object(plan, "REPO", repo), self.assertRaisesRegex(ValueError, "outside"):
            plan.build_plan(self.source, repo / "new_plan")
        output = self.root / "cli_plan"
        with patch.object(sys, "argv", ["plan", "build", "--source-jsonl", str(self.source), "--output-dir", str(output)]), \
                contextlib.redirect_stdout(io.StringIO()) as stream:
            plan.main()
        response = json.loads(stream.getvalue())
        self.assertEqual(response["official_episodes"], 99)
        self.assertFalse(response["physics_or_GPU_used"])


if __name__ == "__main__":
    unittest.main()
