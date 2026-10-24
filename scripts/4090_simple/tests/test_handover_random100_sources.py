"""CPU raw-reference regressions; no simulator stepping, GPU or training."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handover_random100_sources as sources


class RawSourcesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.root = base / "raw"; self.assets = base / "assets"; self.archive = base / "raw.zip"
        self.provenance = base / "receipt.json"
        self.action_names = list(sources.BODY_NAMES[:22]) + [f"left_hand_{finger}_{i}_joint"
            for finger, count in (("index", 2), ("middle", 2), ("thumb", 3)) for i in range(count)]
        self.action_names += list(sources.BODY_NAMES[22:]) + [f"right_hand_{finger}_{i}_joint"
            for finger, count in (("index", 2), ("middle", 2), ("thumb", 3)) for i in range(count)]
        info = {"fps": 50, "total_episodes": 100, "chunks_size": 1000,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {"observation.state": {"shape": [43], "names": list(sources.STATE_SOURCE_NAMES)},
                         "action": {"shape": [43], "names": self.action_names},
                         "observation.object_poses": {"names": [f"{role}.{term}"
                             for role in ("container", "target") for term in sources.OBJECT_TERMS]}}}
        config = {"uid": "g1_wholebody_handover_sonic", "robot_cfg": {"uid": "g1_sonic"},
                  "layout": {"actors": {}}, "dr_state_dict": {"spatial": {}}}
        for role, family, res in (("target", "graspnet", "graspnet1b"), ("container", "objects", "objects")):
            usd = self.assets / f"assets/{family}/ruled_models/000_ruled.usd"
            mesh = self.assets / f"assets/{family}/models/000/coacd_0.05/convex_piece_0.obj"
            for p in (usd, mesh): p.parent.mkdir(parents=True, exist_ok=True); p.write_text("unit-test asset")
            config["layout"]["actors"][role] = {"uid": "0", "asset": {"res_id": res, "uid": "0",
                "usd_path": f"/home/original/SIMPLE/data/assets/{family}/ruled_models/000_ruled.usd"},
                "pose": {"position": [9., 9., 9.], "quaternion": [1., 0., 0., 0.]}}
        xml = self.assets / "robots/g1_sonic/g1_29dof_with_hand.xml"
        xml.parent.mkdir(parents=True)
        xml.write_text("<mujoco><worldbody><body>" + "".join(f'<joint name="{n}"/>' for n in sources.TRAIN_HAND_NAMES)
                       + "</body></worldbody></mujoco>")
        episodes = [{"episode_index": i, "length": 1 if i == 38 else 60,
                     "tasks": ["Handover"], "environment_config": json.dumps(config)} for i in range(100)]
        meta = self.root / "meta"; meta.mkdir(parents=True)
        (meta / "info.json").write_text(json.dumps(info))
        (meta / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
        for episode in episodes:
            n = episode["length"]; t = np.arange(n) / 50
            state = np.arange(43)[None] * .001 + t[:, None] * .001
            action = np.broadcast_to(np.arange(43, dtype=float), (n, 43))
            roots = np.broadcast_to([-.7, 0., .75, 1., 0., 0., 0.], (n, 7))
            objects = np.broadcast_to([-.5, .4, .56, 1., 0., 0., 0., -.5, 0., .66, 1., 0., 0., 0.], (n, 14))
            values = {"observation.state": state.tolist(), "action": action.tolist(),
                "observation.base_pose": roots.tolist(), "observation.base_vel": np.zeros((n, 6)).tolist(),
                "observation.object_poses": objects.tolist(), "timestamp": t.tolist(),
                "frame_index": list(range(n)), "episode_index": [episode["episode_index"]] * n}
            p = self.root / f"data/chunk-000/episode_{episode['episode_index']:06d}.parquet"
            p.parent.mkdir(parents=True, exist_ok=True); pq.write_table(pa.table(values), p)
        with zipfile.ZipFile(self.archive, "w") as archive:
            for p in self.root.rglob("*"):
                if p.is_file(): archive.write(p, f"{sources.PREFIX}/{p.relative_to(self.root)}")
        self.archive_sha = sources.sha256(self.archive)
        self.provenance.write_text(json.dumps({"source_url": f"https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/{sources.REVISION}/simple-teleop/G1WholebodyHandoverTeleop-v0.zip",
            "source_revision": sources.REVISION, "archive_sha256": self.archive_sha, "hf_lfs_sha256": self.archive_sha}))
        self.pin = patch.object(sources, "ARCHIVE_SHA256", self.archive_sha); self.pin.start(); self.addCleanup(self.pin.stop)
        self.catalog = sources.RawSourceCatalog(self.root, archive_path=self.archive, provenance_path=self.provenance)

    def load(self, index=1):
        return sources.load_raw_source(index, catalog=self.catalog, asset_root=self.assets)

    def test_returns_real_named_q_root_objects_commands_and_row0_environment(self):
        source = self.load()
        value = source["values"]
        self.assertEqual(value["reference_full_state43"].shape, (60, 43))
        self.assertEqual(value["reference_full_body_dq"].shape, (60, 29))
        np.testing.assert_allclose(value["reference_full_body_dq"], .001, atol=1e-12)
        self.assertEqual(value["reference_full_hand_command_names"], list(sources.SIMPLE_HAND_NAMES))
        self.assertEqual(source["initial"]["joint_names"], list(sources.TRAIN_STATE_NAMES))
        index = sources.STATE_SOURCE_NAMES.index("left_hand_middle_0_joint")
        target = sources.TRAIN_STATE_NAMES.index("left_hand_middle_0_joint")
        self.assertEqual(value["reference_full_state43"][0, target], index * .001)
        self.assertEqual(source["config"]["layout"]["actors"]["target"]["pose"]["position"], [-.5, 0., .66])
        self.assertEqual(source["config"]["dr_state_dict"]["spatial"]["0"]["position"], [-.5, 0., .66])
        self.assertEqual(source["initial"]["objects"]["target"]["mass"], .1)
        self.assertIn("not_recorded_source_mass", source["evidence"]["object_mass_policy"]["origin"])
        self.assertFalse(source["evidence"]["hand_commands"]["historical_per_episode_recipe_binding_verified"])
        self.assertFalse(source["evidence"]["new_successful_demonstration"])

    def test_hand_declared_effective_differ_and_never_scale_angles_by_torque(self):
        source = self.load(); values = source["values"]
        np.testing.assert_array_equal(values["reference_full_declared_hand_commands14"][0],
            [26, 27, 28, 22, 23, 24, 25, 40, 41, 42, 36, 37, 38, 39])
        np.testing.assert_array_equal(values["reference_full_hand_commands14"][0],
            [26, 27, 28, 24, 25, 22, 23, 40, 41, 42, 38, 39, 36, 37])
        self.assertFalse(np.array_equal(values["reference_full_hand_commands14"], values["reference_full_state43"][:, -14:]))

    def test_registered_resource_alias_and_actual_payload_are_both_hash_bound(self):
        usd = self.assets / "assets/graspnet/ruled_models/000_ruled.usd"
        actual = Path(self.temp.name) / "immutable_source_asset.usd"
        actual.write_bytes(usd.read_bytes()); usd.unlink(); usd.symlink_to(actual)
        source = self.load()
        self.assertEqual(source["sources"][str(usd)], source["sources"][str(actual)])

    def test_all_five_VAL_and_incomplete_record_never_load_as_train(self):
        for index in (*sources.HELDOUT, 38, -1, 100, True):
            with self.subTest(index=index), self.assertRaises(ValueError): self.load(index)

    def test_extracted_parquet_mutation_is_rejected_against_archive(self):
        p = self.root / "data/chunk-000/episode_000001.parquet"
        with p.open("ab") as stream: stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "differ from official archive"): self.load()

    def test_wrong_old_simple_namespace_is_rejected(self):
        receipt = json.loads(self.provenance.read_text()); receipt["source_url"] = receipt["source_url"].replace("simple-teleop/", "simple/")
        self.provenance.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "namespace"):
            sources.RawSourceCatalog(self.root, archive_path=self.archive, provenance_path=self.provenance)

    def test_complete_real_future_is_required_and_original_tail_cannot_be_padded(self):
        source = self.load(); values = source["values"]
        current = sources.encoder_input(values["reference_full_state43"][:, :29], values["reference_full_body_dq"],
            values["reference_full_root_wxyz"], 14)
        self.assertEqual(current.shape, (1, 1751)); self.assertTrue(np.isfinite(current).all())
        with self.assertRaisesRegex(ValueError, "no tail padding"):
            sources.encoder_input(values["reference_full_state43"][:, :29], values["reference_full_body_dq"],
                values["reference_full_root_wxyz"], 15)

    def test_inventory_binds100_and_marks94_candidates_without_claiming_success(self):
        inventory = sources.inventory(self.catalog)
        self.assertEqual(inventory["raw_episodes"], 100)
        self.assertEqual(inventory["TRAIN_candidate_count"], 94)
        self.assertEqual(inventory["new_successes_accepted"], 0)
        self.assertEqual(sum(r["status"] == "protected_VAL" for r in inventory["records"]), 5)
        self.assertFalse(inventory["independent_human_sessions_proven"])
        source = self.load(1)
        parquets = [path for path in source["sources"] if path.endswith(".parquet")]
        self.assertEqual(parquets, [str(self.root / "data/chunk-000/episode_000001.parquet")])


if __name__ == "__main__":
    unittest.main()
