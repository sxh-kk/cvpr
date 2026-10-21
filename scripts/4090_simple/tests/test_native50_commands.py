"""CPU contracts for native reference heading and the 30/50 Hz command clocks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prepare_carrybox_v11 import encoder_input
from replay_carrybox_native50 import feedback_encoder_input
from replay_carrybox_v11 import command_event_index


class NativeCommandTests(unittest.TestCase):
    def test_30hz_events_use_first_real_50hz_tick_and_zero_order_hold(self):
        events = []
        previous = -1
        held = None
        for native_frame in range(50):
            ordinal, expected_frame = command_event_index(native_frame / 50, 30)
            if ordinal != previous:
                self.assertEqual(ordinal, previous + 1)
                self.assertEqual(native_frame, expected_frame)
                held = native_frame
                events.append((ordinal, native_frame))
                previous = ordinal
            self.assertEqual(held, expected_frame)
        self.assertEqual(events, [(index, (index * 50 + 29) // 30) for index in range(30)])
        self.assertEqual([frame for _, frame in events[:7]], [0, 2, 4, 5, 7, 9, 10])

    def test_50hz_updates_each_tick_and_rejects_unsupported_rate(self):
        for frame in range(100):
            self.assertEqual(command_event_index(frame / 50, 50), (frame, frame))
        with self.assertRaises(ValueError):
            command_event_index(0, 60)

    def test_feedback_changes_only_heading_branch_and_same_heading_is_identical(self):
        frames = np.arange(80)
        angles = frames * .01
        roots = np.c_[np.cos(angles / 2), np.zeros((80, 2)), np.sin(angles / 2)]
        q = np.arange(80 * 29, dtype=float).reshape(80, 29) / 1000
        dq = q / 10
        frame = 20
        original = encoder_input(q, dq, roots, frame)
        same = feedback_encoder_input(q, dq, roots, frame, angles[frame])
        np.testing.assert_allclose(original, same, rtol=0, atol=1e-7)
        changed = feedback_encoder_input(q, dq, roots, frame, angles[frame] + .2)
        np.testing.assert_array_equal(original[:, :584], changed[:, :584])
        np.testing.assert_array_equal(original[:, 644:], changed[:, 644:])
        self.assertGreater(np.max(np.abs(original[:, 584:644] - changed[:, 584:644])), .1)
        self.assertTrue(np.isfinite(changed).all())


if __name__ == "__main__":
    unittest.main()
