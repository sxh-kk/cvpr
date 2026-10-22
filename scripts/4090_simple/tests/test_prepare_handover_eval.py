"""CPU evidence for initial-state selection; no policy or physics is executed."""
import importlib.util
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


spec = importlib.util.spec_from_file_location("prepare_handover_eval",
    Path(__file__).resolve().parents[1] / "prepare_handover_eval.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write(path, value):
    module.write_json(path, value)


def fixture(root):
    source, assets, output = root / "source", root / "assets", root / "prepared"
    source.mkdir()
    model = assets / "robots/g1_sonic/g1_29dof_with_hand.xml"
    model.parent.mkdir(parents=True)
    names = [f"joint_{i}" for i in range(43)]
    model.write_text("<mujoco><worldbody>" + "".join(f'<joint name="{n}"/>' for n in names) + "</worldbody></mujoco>")
    asset = assets / "assets/item.usd"
    asset.parent.mkdir()
    asset.write_text("local resource")
    write(source / "conversion_manifest.json", {"status": "complete", "is_official": False,
        "accepted_source_episode_indices": list(module.VAL_IDS),
        "control_profiles": {"hand_pd_profile": "collection"}})
    write(source / "meta/info.json", {"fps": 50, "features": {
        "observation.object_poses": {"names": module.OBJECT_NAMES}}})
    rows = []
    for i, original in enumerate(module.VAL_IDS):
        cfg = {"uid": "g1_wholebody_handover_sonic", "robot_cfg": {"uid": "g1_sonic"},
            "layout": {"actors": {"table": {},
                "target": {"uid": "0", "pose": {"position": [8, 9, 10], "quaternion": [1, 0, 0, 0]},
                    "asset": {"usd_path": "/home/old/SIMPLE/data/assets/item.usd"}},
                "container": {"uid": "0", "pose": {"position": [11, 12, 13], "quaternion": [1, 0, 0, 0]}}}},
            "dr_state_dict": {"spatial": {"0": {}, "container_0": {},
                "g1_sonic": {"position": [-.75, 0, 0], "quaternion": [1, 0, 0, 0]}}}}
        rows.append({"episode_index": i, "source_episode_index": original,
            "environment_config": json.dumps(cfg), "length": 2})
        prefix = source / "validation" / f"source_episode_{original:06d}"
        reset = np.arange(43, dtype=float) / 100
        write(prefix.with_name(prefix.name + "_physics.json"), {"reset_joint_names": names,
            "reset_joint_state43": reset.tolist(), "objects": {
                role: {"mass": .1, "external_force": [0] * 6} for role in ("container", "target")}})
        prefix.parent.mkdir(exist_ok=True)
        np.save(prefix.with_name(prefix.name + "_reset_joint_state43.npy"), reset)
        write(prefix.with_name(prefix.name + "_controller_reset.json"), {
            "vector_sha256": hashlib.sha256(reset.astype("<f8").tobytes()).hexdigest(),
            "output_replay_reset_state_key": "replay.initial_state43"})
        objects = {"container": [1, 2, 3, 1, 0, 0, 0], "target": [4, 5, 6, 1, 0, 0, 0]}
        write(prefix.with_name(prefix.name + "_initial_objects.json"), {
            "object_restore_during_control": False, "objects": {role: {
                "recorded_pose7": value, "actual_pose7": value} for role, value in objects.items()}})
        path = source / "data/chunk-000" / f"episode_{i:06d}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"observation.base_pose": [[-.8, 0, .74, 1, 0, 0, 0], [99] * 7],
            "observation.base_vel": [[.01, .02, .03, .04, .05, .06], [99] * 6],
            "observation.object_poses": [objects["container"] + objects["target"], [99] * 14],
            "action": [[float("nan")], [float("nan")]]}), path)
    episodes = source / "meta/episodes.jsonl"
    episodes.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return source, assets, output


class PrepareHandoverEvalTest(unittest.TestCase):
    def test_all_five_cases_use_row_zero_and_preserve_asset_and_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, assets, output = fixture(Path(temporary))
            before = {str(p): module.sha256(p) for p in source.rglob("*") if p.is_file()}
            result = module.prepare(source, assets, output)
            initial = json.loads((output / "handover_initialization.json").read_text())
            rows = [json.loads(s) for s in (output / f"eval/{module.TASK}/dr-level-0/meta/episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 5)
            self.assertEqual(result["source_episode_ids"], list(module.VAL_IDS))
            self.assertFalse(initial["compatibility_verified"])
            self.assertTrue((output / "assets").is_symlink())
            case = initial["episodes"]["0"]
            self.assertEqual(case["root_pose7"], [-.8, 0, .74, 1, 0, 0, 0])
            self.assertEqual(case["root_qvel6"], [.01, .02, .03, .04, .05, .06])
            self.assertFalse(case["teacher_primer_consumed"])
            cfg = json.loads(rows[0]["environment_config"])
            self.assertEqual(cfg["layout"]["actors"]["target"]["pose"]["position"], [4, 5, 6])
            self.assertEqual(cfg["uid"], "g1_wholebody_handover_teleop")
            self.assertEqual(rows[0]["source_task_uid"], "g1_wholebody_handover_sonic")
            self.assertEqual(cfg["dr_state_dict"]["spatial"]["0"]["position"], [4, 5, 6])
            self.assertEqual(cfg["dr_state_dict"]["spatial"]["g1_sonic"]["position"], [-.75, 0, 0])
            self.assertEqual(cfg["layout"]["actors"]["target"]["asset"]["usd_path"], str(output / "assets/item.usd"))
            self.assertEqual(before, {str(p): module.sha256(p) for p in source.rglob("*") if p.is_file()})

    def test_subset_is_dense_and_keeps_original_heldout_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, assets, output = fixture(Path(temporary))
            module.prepare(source, assets, output, [80])
            initial = json.loads((output / "handover_initialization.json").read_text())
            self.assertEqual(list(initial["episodes"]), ["0"])
            self.assertEqual(initial["episodes"]["0"]["source_episode_index"], 80)
            row = json.loads((output / f"eval/{module.TASK}/dr-level-0/meta/episodes.jsonl").read_text())
            self.assertEqual((row["episode_index"], row["source_heldout_episode_index"]), (0, 2))

    def test_reset_hash_and_initial_object_conflicts_fail_before_output(self):
        for conflict in ("hash", "object"):
            with self.subTest(conflict=conflict), tempfile.TemporaryDirectory() as temporary:
                source, assets, output = fixture(Path(temporary))
                suffix = "controller_reset" if conflict == "hash" else "initial_objects"
                path = source / "validation" / f"source_episode_000039_{suffix}.json"
                d = json.loads(path.read_text())
                if conflict == "hash":
                    d["vector_sha256"] = "0" * 64
                else:
                    d["objects"]["target"]["actual_pose7"][0] += .1
                write(path, d)
                with self.assertRaises(ValueError):
                    module.prepare(source, assets, output)
                self.assertFalse(output.exists())

    def test_nonheldout_or_existing_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, assets, output = fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "five original"):
                module.prepare(source, assets, output, [0])
            output.mkdir()
            (output / "keep.txt").write_text("retain")
            with self.assertRaisesRegex(ValueError, "fresh"):
                module.prepare(source, assets, output)
            self.assertEqual((output / "keep.txt").read_text(), "retain")


if __name__ == "__main__":
    unittest.main()
