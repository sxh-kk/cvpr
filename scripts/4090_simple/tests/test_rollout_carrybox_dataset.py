"""CPU unit fixtures only; none of these fixtures authorize real data or training."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("rollout_carrybox_dataset", SCRIPTS / "rollout_carrybox_dataset.py")
packing = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packing)


def events(count=61):
    result = []
    for index in range(count):
        tick = packing.native_tick(index)
        result.append({"event_index": index, "command_hz": 30, "native_frame_index": tick,
                       "expected_native_frame_index": tick, "source_ref_frame": tick, "source_episode": 0,
                       "source_time_s": tick / 50, "sim_time_s": .205 + tick / 50,
                       "state43": np.zeros(43).tolist(), "executed_token64": np.full(64, index/100).tolist(),
                       "hand_targets14": np.zeros(14).tolist(), "base_pose7": [0, 0, .8, 1, 0, 0, 0],
                       "box_pose7": [0, 0, .55, 1, 0, 0, 0]})
    return result


class PackingContracts(unittest.TestCase):
    def test_ceil_command_clock_and_true_five_hz_history(self):
        rows = events()
        self.assertEqual([packing.native_tick(index) for index in range(7)], [0, 2, 4, 5, 7, 9, 10])
        self.assertEqual(packing.validate_events(rows), 0)
        for index in range(6, len(rows)):
            self.assertAlmostEqual(rows[index]["sim_time_s"] - rows[index-6]["sim_time_s"], .2)
        self.assertAlmostEqual(rows[30]["sim_time_s"] - rows[0]["sim_time_s"], 1.)
        broken = copy.deepcopy(rows)
        broken[1]["source_time_s"] = .02  # Old floor selection, not its actual command update tick.
        with self.assertRaises(ValueError):
            packing.validate_events(broken)
        for key in ("sim_time_s", "source_time_s"):
            broken = copy.deepcopy(rows); broken[3][key] = np.nan
            with self.assertRaisesRegex(ValueError, "finite"):
                packing.validate_events(broken)

    def test_full_source_extent_is_derived_from_real_rows_not_short_run_claims(self):
        teacher = {"source_native_frames": 1272, "replay_frames": 1227, "excluded_future_tail_frames": 45,
                   "replay_last_timestamp_s": 24.52, "comparison_valid_prefix_last_timestamp_s": 24.52}
        active = [{"source_time_s": index/50} for index in range(1226)]
        self.assertEqual(packing.validate_source_extent(1272, teacher, active)["physical_active_ticks"], 1226)
        # A self-consistent short success receipt must not impersonate a full source.
        short = dict(teacher, replay_frames=1200, replay_last_timestamp_s=23.98,
                     comparison_valid_prefix_last_timestamp_s=23.98)
        with self.assertRaisesRegex(ValueError, "actual source row count"):
            packing.validate_source_extent(1272, short, active[:1199])
        with self.assertRaisesRegex(ValueError, "actual source row count"):
            packing.validate_source_extent(1272, dict(teacher, replay_last_timestamp_s=np.nan), active)

    def test_raw_recorder_lossless_pixels_and_numeric_capture(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / "capture"
            recorder = packing.LosslessCommandRecorder(target, instruction="unit fixture")
            rows = events()
            for row in rows:
                rgb = np.full((12, 16, 3), row["event_index"], dtype=np.uint8)
                recorder.record(**{key: row[key] for key in ("event_index", "sim_time_s", "state43", "executed_token64",
                                "hand_targets14", "base_pose7", "box_pose7", "source_ref_frame", "source_episode", "source_time_s")},
                                rgb_uint8HWC=rgb)
            recorder.close()
            recorder.close()
            manifest = json.loads((target / "capture.json").read_text())
            self.assertEqual(manifest["state"], "capture_closed")
            self.assertIn("not_evaluated", manifest["acceptance"])
            numeric = np.load(target / "events.npz", allow_pickle=False)
            self.assertEqual(numeric["state43"].shape, (61, 43))
            with Image.open(target / "rgb/event_000010.png") as image:
                self.assertTrue(np.array_equal(np.asarray(image), np.full((12, 16, 3), 10, dtype=np.uint8)))
            with self.assertRaises(ValueError):
                recorder.record(**{key: rows[0][key] for key in ("event_index", "sim_time_s", "state43", "executed_token64",
                                "hand_targets14", "base_pose7", "source_ref_frame", "source_episode", "source_time_s")},
                                rgb_uint8HWC=np.zeros((12, 16, 3), dtype=np.uint8))

    def test_lossy_or_non_rgb_input_is_not_silently_cast(self):
        with self.assertRaises(ValueError):
            packing.raw_rgb_sha(np.zeros((12, 16, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            packing.raw_rgb_sha(np.zeros((12, 16), dtype=np.uint8))

    def test_video_and_parquet_preserve_executed_event_values(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rows, pngs = events(), []
            for row in rows:
                rgb = np.zeros((12, 16, 3), dtype=np.uint8)
                rgb[:, :, 0] = row["event_index"]
                rgb[:, ::2, 1] = 127
                path = root / f"event_{row['event_index']:06d}.png"
                Image.fromarray(rgb).save(path)
                pngs.append(path)
                row["rgb_sha256"] = packing.raw_rgb_sha(rgb)
            target = root / "packed"
            entry, _, receipt = packing.pack_episode(target, 0, 0, rows, pngs,
                                                     {"source_episode": 0, "instruction": "unit fixture"})
            table = pq.read_table(target / "data/chunk-000/episode_000000.parquet")
            self.assertEqual(entry["length"], 61)
            self.assertEqual(table["source_frame_index"].to_pylist(), [packing.native_tick(index) for index in range(61)])
            self.assertEqual(table["source_timestamp"].to_pylist(), [row["sim_time_s"] for row in rows])
            self.assertEqual(table["action.body_token_v1_1"].to_pylist()[40], np.full(64, .4, dtype=np.float32).tolist())
            self.assertTrue(receipt["video"]["all_decoded_rgb_equal_actual_event"])

    def test_missing_failed_and_old_observation_phase_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            with self.assertRaises(FileNotFoundError):
                packing.audit_run(root)
            packing.write_json(root / "result.json", {"status": "error", "task_success_observed": True})
            with self.assertRaises(ValueError):
                packing.audit_run(root)
            packing.write_json(root / "result.json", {"status": "source_end", "command_hz": 30,
                               "finite_states": True, "fell": False, "observation_phase": "old_stale_xpos"})
            with self.assertRaisesRegex(ValueError, "synchronized"):
                packing.audit_run(root)

    def test_partial_source_pilot_cannot_be_packed_as_complete(self):
        with tempfile.TemporaryDirectory() as root:
            with mock.patch.object(packing, "audit_run", return_value=([], [], {"source_episode": 0})):
                with self.assertRaisesRegex(ValueError, "TRAIN0/1/2"):
                    packing.pack([Path(root)], Path(root) / "output", Path(root))

    def test_geometric_placement_reconstruction_does_not_trust_success_bool(self):
        import mujoco
        children = []
        for index, name in enumerate(packing.TRAIN_STATE_NAMES):
            children.append(f'<body name="j{index}"><joint name="{name}" axis="0 0 1"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0"/></body>')
        xml = '<mujoco><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>' + ''.join(children) + '</body><body name="table" pos="0 0 .4"><geom type="box" size=".5 .5 .05"/></body><body name="cube"><freejoint name="cube_joint"/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "model.mjb"
            mujoco.mj_saveModel(model, str(path))
            trajectory = {"sim_time_s": np.arange(61)/50, "state43_joint_names": np.array(packing.TRAIN_STATE_NAMES),
                          "actual_state43": np.zeros((61, 43)),
                          "actual_base_pose": np.tile([0, 0, .8, 1, 0, 0, 0], (61, 1)),
                          "actual_box_pose": np.tile([0, 0, .55-1e-6, 1, 0, 0, 0], (61, 1))}
            accepted = packing.reconstruct_placement(path, trajectory)
            self.assertGreaterEqual(accepted["final_reconstructed_hold_s"], .9)
            original_time = trajectory["sim_time_s"].copy()
            for time in (np.nan, np.inf, original_time[29]):
                trajectory["sim_time_s"][30] = time
                with self.assertRaisesRegex(ValueError, "finite, increasing"):
                    packing.reconstruct_placement(path, trajectory)
            trajectory["sim_time_s"] = original_time
            trajectory["actual_box_pose"][:, 2] = .1
            with self.assertRaisesRegex(ValueError, "tabletop release"):
                packing.reconstruct_placement(path, trajectory)


if __name__ == "__main__":
    unittest.main()
