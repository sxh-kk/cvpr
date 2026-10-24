"""CPU regression tests for variable curation and lossless legacy reindexing."""
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SCRIPT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT))
import pack_handover_curated as curated


class CuratedPackTests(unittest.TestCase):
    def selection(self):
        entries = [dict(candidate_id="new-a", run="/a", admission_path="/a.json", admission_sha256="a" * 64),
                   dict(candidate_id="new-b", run="/b", admission_path="/b.json", admission_sha256="b" * 64)]
        ready = {"episodes": [dict(episode_index=0, candidate_id="old-a"),
                              dict(episode_index=1, candidate_id="old-b")]}
        selection = dict(contract=curated.SELECTION_CONTRACT, old_episode_indices=[1], new14=copy.deepcopy(entries),
                         retained=[dict(origin="new14", candidate_id=e["candidate_id"]) for e in entries]
                         + [dict(origin="original100", candidate_id="old-b", old_episode_index=1)],
                         new_count=2, old_count=1, total_count=3)
        return selection, entries, ready

    def test_variable_count_and_quality_order_not_exact100(self):
        selection, entries, ready = self.selection()
        before = copy.deepcopy((selection, entries, ready))
        self.assertEqual(curated.validate_selection(selection, entries, ready), [1])
        self.assertEqual((selection, entries, ready), before)

    def test_all_qualified_required_in_original_order(self):
        selection, entries, ready = self.selection()
        selection["new14"].reverse()
        with self.assertRaisesRegex(ValueError, "Every qualified"):
            curated.validate_selection(selection, entries, ready)
        selection, entries, ready = self.selection()
        selection["new14"].pop()
        with self.assertRaisesRegex(ValueError, "Every qualified"):
            curated.validate_selection(selection, entries, ready)

    def test_repeated_or_substituted_old_candidate_rejected(self):
        selection, entries, ready = self.selection()
        selection["old_episode_indices"] = [1, 1]
        with self.assertRaisesRegex(ValueError, "Repeated"):
            curated.validate_selection(selection, entries, ready)
        selection, entries, ready = self.selection()
        selection["retained"][-1]["candidate_id"] = "old-a"
        with self.assertRaisesRegex(ValueError, "identity"):
            curated.validate_selection(selection, entries, ready)

    def table(self):
        return pa.table({"episode_index": pa.array([8, 8, 8], pa.int64()),
                         "index": pa.array([99, 100, 101], pa.int64()),
                         "task_index": pa.array([4, 4, 4], pa.int64()),
                         "action.body_token_v1_1": pa.array([[float("nan"), -0.0], [1.5, 2.5], [3.5, 4.5]],
                                                          type=pa.list_(pa.float32())),
                         "observation.base_pose": pa.array([[1., 2., 3.]] * 3, type=pa.list_(pa.float64())),
                         "source_frame_index": pa.array([50, 52, 53], pa.int64()),
                         "action_valid": [False, True, True], "action_mask": [[False, False], [True, True], [True, True]],
                         "phase": ["warmup", "approach", "right_lift"]}).replace_schema_metadata({b"source": b"frozen"})

    def test_reindex_only_three_indices_preserves_nan_clock_masks_schema(self):
        original = self.table()
        result = curated.reindex_table(original, 17, 500, 0)
        self.assertEqual(result.schema, original.schema)
        self.assertEqual(result["episode_index"].to_pylist(), [17] * 3)
        self.assertEqual(result["index"].to_pylist(), [500, 501, 502])
        self.assertEqual(result["task_index"].to_pylist(), [0] * 3)
        self.assertEqual(original["episode_index"].to_pylist(), [8] * 3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reindexed.parquet"; pq.write_table(result, path)
            curated.check_unmodified_columns(original, pq.read_table(path))
        # IEEE sign and NaN payloads of Arrow arrays are retained in memory.
        self.assertEqual(original["action.body_token_v1_1"].chunk(0).values.buffers()[1].to_pybytes(),
                         result["action.body_token_v1_1"].chunk(0).values.buffers()[1].to_pybytes())

    def test_changed_real_label_is_rejected(self):
        original = self.table(); changed = original.set_column(original.schema.get_field_index("source_frame_index"),
                                                               "source_frame_index", pa.array([50, 51, 53], pa.int64()))
        with self.assertRaisesRegex(ValueError, "source_frame_index"):
            curated.check_unmodified_columns(original, changed)

    def test_source_sha_and_input_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source"; path.write_bytes(b"bound raw events")
            bindings = curated.Bindings(); bindings.verify(path, curated.legacy.sha(path))
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                bindings.verify(path, "a" * 64)
            path.write_bytes(b"changed raw events")
            with self.assertRaisesRegex(ValueError, "Input changed"):
                bindings.unchanged()
            with self.assertRaisesRegex(ValueError, "Bound bytes"):
                curated.Bindings().verify(path, "b" * 64)

    def test_copy_evidence_keeps_physics_and_exact_events_without_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); run = root / "original"; (run / "capture/rgb").mkdir(parents=True)
            files = {"physics_model.mjb": b"physics", "trajectory.npz": b"actual states", "applied_force_samples.jsonl": b"forces",
                     "capture/capture.json": b"{}", "capture/events.jsonl": b"raw double state43\n", "capture/rgb/frame.png": b"png"}
            for name, value in files.items(): (run / name).write_bytes(value)
            admission = root / "admission.json"; admission.write_text("{}")
            bindings = {str(run / name): curated.legacy.sha(run / name) for name in files}
            item = dict(kind="new_random", entry=dict(candidate_id="new-physical", admission_sha256=curated.legacy.sha(admission)),
                        admission=admission, run=run, capture=run / "capture", audit=dict(bindings=bindings), source={})
            out = root / "out"; out.mkdir(); receipt = curated.copy_evidence(item, out)
            self.assertEqual((out / receipt["evidence_files"]["capture/events.jsonl"]).read_bytes(), files["capture/events.jsonl"])
            self.assertIn("physics_model.mjb", receipt["evidence_files"])
            self.assertIn("applied_force_samples.jsonl", receipt["evidence_files"])
            self.assertFalse(list(out.rglob("*.png")))

    def test_namespace_is_not_numeric_ancestor_deduplication(self):
        # Different raw namespaces and payloads may have the same numeric ID.
        a = dict(source_dataset_key="archive-a", source_episode_index=9, source_parquet_sha256="a" * 64)
        b = dict(source_dataset_key="archive-b", source_episode_index=9, source_parquet_sha256="b" * 64)
        self.assertNotEqual(a, b)
        self.assertNotEqual(a["source_parquet_sha256"], b["source_parquet_sha256"])
        with self.assertRaises(ValueError): curated.safe_candidate("../../escape")


if __name__ == "__main__":
    unittest.main()
