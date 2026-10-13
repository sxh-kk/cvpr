import copy
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from psi.config.transform import WBTNativePosttrainRepackTransform
from psi.data.humanoid.wbt_hand_protocol import (
    G1_BODY_JOINT_NAMES,
    NATIVE_DIM,
    PROTOCOL_NAMES,
    canonicalize_hand,
    protocol_metadata,
)

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - project data dependency
    pa = None
    pq = None


def _robot_rows(length: int) -> np.ndarray:
    values = np.zeros((length, 36), dtype=np.float32)
    values[:, 3] = 1.0  # WXYZ root quaternion
    return values


class WBTNativeProtocolTests(unittest.TestCase):
    @staticmethod
    def _raw_sample(**overrides):
        robot = _robot_rows(1)
        sample = {
            "observation.state.robot_q_current": robot,
            "observation.state.hand_state": np.zeros((1, 12), dtype=np.float32),
            "action.robot_q_desired": robot.copy(),
            "action.hand_cmd": np.zeros((1, 12), dtype=np.float32),
            "observation.images.head_stereo_left": np.zeros(
                (1, 8, 8, 3), dtype=np.uint8
            ),
            "task": "test task",
        }
        sample.update(overrides)
        return sample

    def test_inspire_and_brainco_layouts_are_explicit(self):
        raw = np.arange(12, dtype=np.float32)
        inspire, inspire_meta = canonicalize_hand(raw, "inspire-6d")
        brainco, brainco_meta = canonicalize_hand(raw, "brainco-6d")
        np.testing.assert_array_equal(inspire[:6], [4, 5, 0, 1, 2, 3])
        np.testing.assert_array_equal(brainco[:6], [0, 1, 2, 3, 4, 5])
        self.assertEqual(inspire_meta["canonical_order"][0], "thumb_flex")
        self.assertEqual(brainco_meta["source_hand_type"], "brainco")

    def test_non_12d_hand_is_rejected(self):
        with self.assertRaises(ValueError):
            canonicalize_hand(np.zeros(2, dtype=np.float32), "inspire-6d")

    def test_repack_produces_48d_sequences_and_complete_masks(self):
        repack = WBTNativePosttrainRepackTransform(
            hand_layout_id="inspire-6d", num_past_frames=1, action_chunk_size=4
        )
        robot = _robot_rows(2)
        hand = np.arange(24, dtype=np.float32).reshape(2, 12) / 24.0
        data = {
            "observation.state.robot_q_current": robot,
            "observation.state.hand_state": hand,
            "action.robot_q_desired": robot.copy(),
            "action.hand_cmd": hand.copy(),
            "observation.images.head_stereo_left": np.zeros(
                (2, 8, 8, 3), dtype=np.uint8
            ),
            # A reader may already provide a per-dimension mask.
            "action_mask": np.ones((2, NATIVE_DIM), dtype=np.float32),
            "task": "Pick Up Medicine",
        }
        output = repack(data)
        self.assertEqual(output["states"].shape, (2, NATIVE_DIM))
        self.assertEqual(output["actions"].shape, (4, NATIVE_DIM))
        self.assertEqual(output["actions_mask"].shape, (4, NATIVE_DIM))
        np.testing.assert_array_equal(output["actions_mask"][:2], 1.0)
        np.testing.assert_array_equal(output["actions_mask"][2:], 0.0)
        self.assertEqual(output["instruction"], "pick up medicine")
        self.assertEqual(output["protocol"], "wbt-native-v1")

    def test_nonuniform_dimension_mask_is_rejected(self):
        repack = WBTNativePosttrainRepackTransform(hand_layout_id="brainco-6d")
        data = self._raw_sample(
            action_mask=np.concatenate(
                [np.ones((1, 1), dtype=np.float32), np.zeros((1, 1), dtype=np.float32)],
                axis=1,
            )
        )
        with self.assertRaises(ValueError):
            repack(data)

    def test_nonbinary_mask_is_rejected(self):
        repack = WBTNativePosttrainRepackTransform(hand_layout_id="brainco-6d")
        with self.assertRaises(ValueError):
            repack(self._raw_sample(action_mask=np.asarray([0.5], dtype=np.float32)))

    def test_out_of_range_image_is_rejected(self):
        repack = WBTNativePosttrainRepackTransform(hand_layout_id="brainco-6d")
        image = np.full((1, 8, 8, 3), 256.0, dtype=np.float32)
        with self.assertRaises(ValueError):
            repack(self._raw_sample(**{"observation.images.head_stereo_left": image}))

    def test_protocol_metadata_freezes_body_order(self):
        metadata = protocol_metadata(
            state_hand_layout_id="brainco-6d",
            action_hand_layout_id="brainco-6d",
        )
        self.assertEqual(metadata["dimensions"], list(PROTOCOL_NAMES))
        self.assertEqual(metadata["body"]["joint_names"], list(G1_BODY_JOINT_NAMES))
        self.assertEqual(metadata["body"]["source_vector_slice"], [7, 36])
        self.assertTrue(metadata["body"]["joint_order_sources"])
        self.assertEqual(metadata["sampling"]["fps"], 30.0)

    def test_camera_and_fps_are_protocol_constants(self):
        from psi.data.humanoid.wbt_dataset import WBTDataset

        with self.assertRaises(ValueError):
            WBTDataset("unused", camera_key="observation.images.wrist_left")
        repack = WBTNativePosttrainRepackTransform(hand_layout_id="brainco-6d")
        with self.assertRaises(ValueError):
            repack.delta_timestamps(15)

    def test_wrong_protocol_metadata_is_rejected(self):
        repack = WBTNativePosttrainRepackTransform(hand_layout_id="brainco-6d")
        metadata = protocol_metadata(
            state_hand_layout_id="brainco-6d",
            action_hand_layout_id="brainco-6d",
        )
        metadata["dimensions"][7] = "wrong_joint"
        with self.assertRaises(ValueError):
            repack(self._raw_sample(protocol_metadata=metadata))

    def test_foreign_dataset_is_rejected(self):
        repack = WBTNativePosttrainRepackTransform(hand_layout_id="brainco-6d")
        with self.assertRaises(ValueError):
            repack({"dataset_name": "humanoid-everyday"})

    def test_wbt_stats_and_model_guards(self):
        from psi.config.train.posttrain_wbt_psi0_config import (
            WBTPsi0ModelConfig,
            WBTActionStateTransform,
            _load_wbt_stats,
            _validate_wbt_stats_payload,
        )

        _, stats = _load_wbt_stats("assets/stats/wbt_native_stats.json")
        broken = copy.deepcopy(stats)
        broken["dimensions"][7] = "wrong_joint"
        with self.assertRaises(ValueError):
            _validate_wbt_stats_payload(broken, Path("broken.json"))
        broken = copy.deepcopy(stats)
        first_repo = broken["source"]["repo_ids"][0]
        broken["source"]["protocol_metadata"][first_repo]["hand_layout_ids"]["state"] = "inspire-6d"
        with self.assertRaises(ValueError):
            _validate_wbt_stats_payload(broken, Path("broken.json"))
        broken = copy.deepcopy(stats)
        broken["source"]["fps_by_repo"][first_repo] = 15.0
        with self.assertRaises(ValueError):
            _validate_wbt_stats_payload(broken, Path("broken.json"))
        with TemporaryDirectory() as temporary:
            missing = Path(temporary) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                WBTActionStateTransform(stat_path=str(missing))
        with self.assertRaises(ValueError):
            WBTPsi0ModelConfig(pretrained_action_header_path="he-action-header")

    def test_empty_episode_selection_is_rejected(self):
        from psi.config.data_wbt import WBTDataConfig

        with self.assertRaises(ValueError):
            WBTDataConfig._episode_ids([], "repo")


@unittest.skipIf(pa is None or pq is None, "pyarrow is required")
class WBTDatasetTests(unittest.TestCase):
    def _write_repo(self, root: Path) -> Path:
        repo = root / "G1_WBT_Inspire_Test"
        (repo / "meta").mkdir(parents=True)
        (repo / "data" / "chunk-000").mkdir(parents=True)
        info = {
            "codebase_version": "v3.0",
            "robot_type": "unitree_g1",
            "total_episodes": 2,
            "total_frames": 5,
            "fps": 30,
            "features": {
                "observation.images.head_stereo_left": {
                    "dtype": "video",
                    "shape": [8, 8, 3],
                    "info": {"video.fps": 30},
                },
                "observation.state.robot_q_current": {
                    "dtype": "float32",
                    "shape": [36],
                },
                "observation.state.hand_state": {
                    "dtype": "float32",
                    "shape": [12],
                },
                "action.robot_q_desired": {
                    "dtype": "float32",
                    "shape": [36],
                },
                "action.hand_cmd": {"dtype": "float32", "shape": [12]},
            },
        }
        (repo / "meta" / "info.json").write_text(json.dumps(info))
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array([0, 1], type=pa.int64()),
                    "__index_level_0__": pa.array(["first task", "second task"]),
                }
            ),
            repo / "meta" / "tasks.parquet",
        )

        data_path = repo / "data" / "chunk-000" / "file-000.parquet"
        writer = None
        global_index = 0
        try:
            for episode_index, length in enumerate((3, 2)):
                robot = _robot_rows(length)
                robot[:, 7] = np.arange(global_index, global_index + length)
                action = robot.copy()
                action[:, 7] += 100.0
                table = pa.table(
                    {
                        "observation.state.robot_q_current": pa.array(robot.tolist()),
                        "observation.state.hand_state": pa.array(
                            np.zeros((length, 12), dtype=np.float32).tolist()
                        ),
                        "action.robot_q_desired": pa.array(action.tolist()),
                        "action.hand_cmd": pa.array(
                            np.zeros((length, 12), dtype=np.float32).tolist()
                        ),
                        "timestamp": pa.array(
                            np.arange(length, dtype=np.float64) / 30.0
                        ),
                        "frame_index": pa.array(np.arange(length), type=pa.int64()),
                        "episode_index": pa.array(
                            [episode_index] * length, type=pa.int64()
                        ),
                        "index": pa.array(
                            np.arange(global_index, global_index + length), type=pa.int64()
                        ),
                        "task_index": pa.array(
                            [episode_index] * length, type=pa.int64()
                        ),
                    }
                )
                if writer is None:
                    writer = pq.ParquetWriter(data_path, table.schema)
                writer.write_table(table)
                global_index += length
        finally:
            if writer is not None:
                writer.close()
        return repo

    def test_episode_tail_padding_never_reads_next_episode(self):
        from psi.data.humanoid.wbt_dataset import WBTDataset

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._write_repo(root)
            with patch.object(
                WBTDataset,
                "_read_image",
                return_value=np.zeros((8, 8, 3), dtype=np.uint8),
            ):
                dataset = WBTDataset(
                    root,
                    repo_ids=["G1_WBT_Inspire_Test"],
                    episodes=[0],
                    num_past_frames=1,
                    action_chunk_size=3,
                )
                sample = dataset[2]
        np.testing.assert_array_equal(sample["action_mask"], [1.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            sample["action.robot_q_desired"][:, 7], [102.0, 102.0, 102.0]
        )
        self.assertEqual(sample["episode_index"], 0)
        self.assertEqual(sample["task"], "first task")


if __name__ == "__main__":
    unittest.main()
