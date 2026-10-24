import importlib.util
from pathlib import Path
import sys
import json
import tempfile

import numpy as np
import unittest

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("curated_selector_tested", SCRIPTS / "select_handover_curated.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def descriptor(joint, fk=None):
    return {"joints": np.full((64, 43), joint), "fk": np.full((64, 9), joint if fk is None else fk)}


def inputs():
    new = [{"candidate_id": f"new{i}"} for i in range(14)]
    return new, {e["candidate_id"]: descriptor(i) for i, e in enumerate(new)}


def row(index, margin):
    return {"episode_index": index, "candidate_id": f"old{index}",
            "actual_full_outer_footprint_final50": {"minimum_outer_hull_margin_m": margin},
            "final_stability_max_pairwise_displacement_m": .0001,
            "final_support_weight_fraction": {"min": .99}}


class SelectionTest(unittest.TestCase):
    def test_quality_audit_is_bound_to_the_actual_old_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            pack = Path(directory) / "pack"
            (pack / "train/meta").mkdir(parents=True)
            ready = pack / "PACK_READY.json"
            metadata = pack / "train/meta/episodes.jsonl"
            ready.write_text(json.dumps({"episodes": []}))
            metadata.write_text(json.dumps({"episode_index": 0}) + "\n")
            summary = {"source_sha256": {str(p): module.sha(p) for p in (ready, metadata)}}
            _, rows = module.validate_old_pack_binding(summary, pack)
            assert rows == {0: {"episode_index": 0}}
            other = Path(directory) / "other"
            (other / "train/meta").mkdir(parents=True)
            (other / "PACK_READY.json").write_bytes(ready.read_bytes())
            (other / "train/meta/episodes.jsonl").write_bytes(metadata.read_bytes())
            with self.assertRaisesRegex(ValueError, "does not bind"):
                module.validate_old_pack_binding(summary, other)
            metadata.write_text(json.dumps({"episode_index": 99}) + "\n")
            with self.assertRaisesRegex(ValueError, "does not bind"):
                module.validate_old_pack_binding(summary, pack)

    def test_quality_order_retains_best_old_and_mandatory_new(self):
        new, descriptors = inputs()
        descriptors.update(old0=descriptor(20), old1=descriptor(20.001), old2=descriptor(0.001))
        kept, rejected = module.choose([row(0, .01), row(1, .02), row(2, .03)], new, descriptors)
        assert [x["candidate_id"] for x in kept] == [f"new{i}" for i in range(14)] + ["old1"]
        assert {x["candidate_id"] for x in rejected} == {"old0", "old2"}
        assert next(x for x in rejected if x["candidate_id"] == "old2")["kept_candidate_id"] == "new0"


    def test_both_motion_thresholds_are_required_for_removal(self):
        new, descriptors = inputs()
        descriptors.update(old0=descriptor(.001, .02), old1=descriptor(.02, .001))
        kept, rejected = module.choose([row(0, .01), row(1, .02)], new, descriptors)
        assert len(kept) == 16
        assert not rejected


    def test_mandatory_new_are_not_silently_dropped(self):
        new, descriptors = inputs()
        descriptors["new1"] = descriptors["new0"]
        with self.assertRaisesRegex(ValueError, "mandatory new14"):
            module.choose([], new, descriptors)


    def test_nonfinite_quality_is_rejected(self):
        new, descriptors = inputs()
        descriptors["old0"] = descriptor(20)
        with self.assertRaisesRegex(ValueError, "Nonfinite quality"):
            module.choose([row(0, float("nan"))], new, descriptors)


    def test_same_numeric_source_is_not_used_as_dedup(self):
        new, descriptors = inputs()
        a, b = row(0, .01), row(1, .02)
        a["source_episode_index"] = b["source_episode_index"] = 0
        descriptors.update(old0=descriptor(20), old1=descriptor(30))
        kept, rejected = module.choose([a, b], new, descriptors)
        assert len(kept) == 16 and not rejected


if __name__ == "__main__":
    unittest.main()
