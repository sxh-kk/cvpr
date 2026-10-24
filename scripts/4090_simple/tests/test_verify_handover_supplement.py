"""Small CPU-only regression checks for independent-pack verification/mixing."""
from pathlib import Path
from copy import deepcopy
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_handover_supplement as verifier


class SupplementVerifierTests(unittest.TestCase):
    def test_transform_contract_rejects_shape_preserving_channel_and_normalization_changes(self):
        source = {"repack": {"action_keys": ["action.body_token_v1_1", "action[:14]"],
                              "state_keys": ["observation.state"], "pad_action_dim": 80, "pad_state_dim": 45},
                  "field": {"normalize_state": True, "action_norm_type": "bounds",
                            "stat_action_keys": ["action.body_token_v1_1", "action[:14]"]}}
        normalized = deepcopy(source); normalized["field"]["normalize_state"] = False
        reordered = deepcopy(source); reordered["repack"]["action_keys"].reverse()
        self.assertNotEqual(verifier.transform_contract(source), verifier.transform_contract(normalized))
        self.assertNotEqual(verifier.transform_contract(source), verifier.transform_contract(reordered))

    def test_encoder_contract_pins_real_weights_dtype_and_preprocessing(self):
        original = {"storage_dtype": "float32", "image_process": {"profile_id": "canonical"},
                    "encoder": {"weights_sha256": "actual-local-weight-sha", "preprocessing": {"do_resize": False}}}
        for mutation in (lambda m: m.update(storage_dtype="float16"),
                         lambda m: m["encoder"].update(weights_sha256="other"),
                         lambda m: m["encoder"]["preprocessing"].update(do_resize=True)):
            changed = deepcopy(original); mutation(changed)
            self.assertNotEqual(verifier.encoder_contract(original), verifier.encoder_contract(changed))

    def test_mixture_preserves_all_ancestry_and_keeps_both_packs_positive(self):
        weights, report = verifier.ancestry_preserving_mix([.1, .4, .25, .25], [0, 0, 9, 9], [0, 0, 0])
        self.assertAlmostEqual(sum(weights), 1)
        self.assertEqual(weights[:4], [.05, .2, .25, .25])
        self.assertTrue(all(value > 0 for value in weights))
        self.assertAlmostEqual(sum(weights[:2]) + sum(weights[4:]), .5)
        self.assertEqual(report["pack_count"], 2)
        self.assertFalse(report["physical_pack_merge_performed"])
        self.assertFalse(report["validation_changed"])

    def test_six_of24_new_ancestors_receive12point5_percent_total_mass(self):
        _, report = verifier.ancestry_preserving_mix([1 / 24] * 24, list(range(24)), list(range(6)))
        self.assertAlmostEqual(report["supplement_probability_mass"], .125)
        self.assertAlmostEqual(report["old_pack_probability_mass"], .875)

    def test_mixture_rejects_unknown_heldout_invalid_or_zero_mass(self):
        for old, sources, new, fraction in (([1.], [0], [9], .5), ([1.], [39], [39], .5),
                ([0.], [0], [0], .5), ([float("nan")], [0], [0], .5), ([1.], [0], [0], 0),
                ([1.], [0], [0], 1), ([1.], [0], [0], float("inf"))):
            with self.subTest(old=old, sources=sources, new=new, fraction=fraction), self.assertRaises(ValueError):
                verifier.ancestry_preserving_mix(old, sources, new, fraction=fraction)

    def test_actual_sample_and_batch_require_all_six_history_and30_future(self):
        sample = {"actions": np.zeros((30, 80)), "actions_mask": np.c_[np.ones((30, 78)), np.zeros((30, 2))],
                  "states": np.zeros((1, 45)), "motion_img_embs": np.zeros((6, 5, 1024))}
        verifier.validate_sample(sample)
        verifier.validate_sample({key: value[None] for key, value in sample.items()}, batched=True)
        for key, replacement in (("motion_img_embs", np.zeros((5, 5, 1024))),
                ("actions", np.full((30, 80), np.nan)), ("states", np.zeros((1, 43))),
                ("actions_mask", np.ones((30, 80)))):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verifier.validate_sample({**sample, key: replacement})

    def test_mapping_uses_filtered_indices_instead_of_original_or_current_row(self):
        history = SimpleNamespace(source_episode_ranges=[{0: (100, 190)}],
            original_indices=np.array([130, 160]), episode_ids=np.array([0, 0]),
            history_indices=np.array([[0, 6, 12, 18, 24, 30], [30, 36, 42, 48, 54, 60]]))
        columns = {"frame_index": np.r_[np.arange(100), np.arange(90)], "action_valid": np.ones(190, bool)}
        episodes = [{"episode_index": 0, "length": 90, "eligible_current_rows": [30, 60]}]
        entries = {0: {"history_rows": {"30": [0, 6, 12, 18, 24, 30], "60": [30, 36, 42, 48, 54, 60]}}}
        self.assertEqual(verifier.validate_mapping(history, {0: [0, 1]}, columns, episodes, entries), [0, 1])
        history.history_indices[1, 0] = 31
        with self.assertRaisesRegex(ValueError, "history differs"):
            verifier.validate_mapping(history, {0: [0, 1]}, columns, episodes, entries)
        history.history_indices[1, 0] = 30
        columns["action_valid"][-1] = False
        with self.assertRaisesRegex(ValueError, "Unavailable history/future"):
            verifier.validate_mapping(history, {0: [0, 1]}, columns, episodes, entries)


if __name__ == "__main__":
    unittest.main()
