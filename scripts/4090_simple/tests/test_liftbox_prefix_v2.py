"""CPU fixtures exercise new prefix clocks/labels; they do not accept real data."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_liftbox_prefix_v2 as producer


def event(index):
    tick = math.ceil(index * 50 / 30)
    teacher = index >= 30
    reference = tick - 50 if teacher else None
    return {"event_index": index, "native_frame_index": tick, "observation_hz": 30,
        "nominal_timestamp_s": index / 30, "collection_elapsed_s": tick / 50,
        "sim_time_s": .005 + tick / 50, "source_episode": 0,
        "source_ref_frame": reference, "teacher_source_time_s": reference / 50 if teacher else None,
        "phase": "oracle_teacher_token" if teacher else "warmup_observation_prefix",
        "action_valid": teacher, "observation_phase": producer.OBSERVATION_PHASE,
        "state43": [0.] * 43, "base_pose7": [0., 0., .8, 1., 0., 0., 0.],
        "box_pose7": [0., 0., .217, 1., 0., 0., 0.],
        "command_hz": 30 if teacher else None,
        "executed_token64": [0.] * 64 if teacher else None,
        "hand_targets14": [0.] * 14 if teacher else None,
        "runtime_encoder": {"mode": "feedback-heading-v11", "source_frame_index": reference,
                            "changed_input_slice": [584, 644], "encoder_input_sha256": "a" * 64} if teacher else None}


class PrefixContracts(unittest.TestCase):
    def test_exact_clock_first_teacher_and_actual_history(self):
        plans = [producer.tick_plan(tick) for tick in range(101)]
        recorded = [plan for plan in plans if plan["is_event"]]
        self.assertEqual([row["native_frame_index"] for row in recorded[:7]], [0, 2, 4, 5, 7, 9, 10])
        self.assertFalse(any(plan["action_valid"] for plan in plans[:50]))
        self.assertEqual(recorded[30]["native_frame_index"], 50)
        self.assertEqual(recorded[30]["source_ref_frame"], 0)
        self.assertEqual([recorded[i]["source_ref_frame"] for i in range(30, 34)], [0, 2, 4, 5])
        self.assertEqual([recorded[i]["native_frame_index"] for i in (0, 6, 12, 18, 24, 30)], [0, 10, 20, 30, 40, 50])
        self.assertEqual(producer.validate_events([event(i) for i in range(60)]), 0)

    def test_floor_stale_source_clock_or_prefix_labels_are_rejected(self):
        for mutate in (
            lambda rows: rows[1].update(native_frame_index=1),
            lambda rows: rows[30].update(source_ref_frame=50),
            lambda rows: rows[0].update(executed_token64=[0.] * 64),
            lambda rows: rows[29].update(action_valid=True),
            lambda rows: rows[30].update(teacher_source_time_s=1.),
            lambda rows: rows[31].update(command_hz=50),
        ):
            rows = [event(i) for i in range(60)]
            mutate(rows)
            with self.assertRaises(ValueError):
                producer.validate_events(rows)

    def test_nonfinite_clock_state_token_and_shifted_rgb_clock_are_rejected(self):
        for key, value in (("sim_time_s", float("nan")), ("sim_time_s", float("inf")),
                           ("sim_time_s", .123), ("state43", [float("nan")] * 43),
                           ("executed_token64", [float("inf")] * 64)):
            rows = [event(i) for i in range(60)]
            rows[30][key] = value
            with self.assertRaises(ValueError):
                producer.validate_events(rows)

    def test_full_teacher_future_only_and_no_tail_padding(self):
        rows = [event(i) for i in range(60)]
        self.assertFalse(producer.eligible_window(rows, 29))
        self.assertTrue(producer.eligible_window(rows, 30))
        self.assertFalse(producer.eligible_window(rows, 31))
        rows[45]["action_valid"] = False
        self.assertFalse(producer.eligible_window(rows, 30))
        rows[45]["action_valid"] = True
        rows[0]["sim_time_s"] += .02
        self.assertFalse(producer.eligible_window(rows, 30))
        rows[0]["sim_time_s"] = float("nan")
        self.assertFalse(producer.eligible_window(rows, 30))

    def test_actual_png_byte_identity_and_null_prefix_without_object_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture"
            recorder = producer.PrefixRecorder(capture, "fixture instruction", {"source_episode": 0})
            image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
            row = recorder.record(event(0), image)
            recorder.close()
            recorder.close()
            manifest = json.loads((capture / "capture.json").read_text())
            self.assertEqual(manifest["contract"], producer.CONTRACT)
            self.assertEqual(manifest["event_count"], 1)
            self.assertEqual(len(manifest["state43_joint_names"]), 43)
            self.assertEqual(len(manifest["hand14_joint_names"]), 14)
            np.testing.assert_array_equal(np.asarray(Image.open(capture / row["rgb_lossless_png"])), image)
            self.assertEqual(row["rgb_sha256"], hashlib.sha256(image.tobytes()).hexdigest())
            saved = json.loads((capture / "events.jsonl").read_text())
            self.assertIsNone(saved["executed_token64"])
            self.assertFalse(saved["action_valid"])
            with np.load(capture / "observations.npz", allow_pickle=False) as arrays:
                self.assertTrue(all(arrays[key].dtype.kind != "O" for key in arrays.files))
            with self.assertRaises(ValueError):
                recorder.record(event(1), image)

    def test_dispatch_continuous_controller_no_teacher_read_during_warmup(self):
        class Controller:
            def __init__(self):
                self._history, self.decoder_calls = [], 0
            def reset(self):
                raise AssertionError("Transition must not reset the controller")
            def step(self, token, hands, observation):
                return {"body_token": np.asarray(token), "hand_targets": hands}
        class Warmup:
            def __init__(self):
                self.receipt, self.phase = {"encoder_calls": 0}, ""
            def command(self, robot, controller, observation):
                if len(controller._history) < 10:
                    self.phase = "warmup_measured_pose_pd"
                    return {"body_token": None, "hand_targets": {"hand": 1.}}
                self.phase = "warmup_live_pose_sonic_v1_1"
                self.receipt["encoder_calls"] += 1
                return controller.step(np.full(64, -.5), {"hand": 0.}, observation)
        controller, warmup, references = Controller(), Warmup(), []
        raw = SimpleNamespace(mujoco=object())
        def provider(reference, context):
            self.assertIs(context, raw)
            references.append(reference)
            return np.full(64, reference), {"source_frame_index": reference}
        policy = producer.PrefixTeacherCommands({"action": np.arange(100)[:, None]}, provider, warmup, controller, ["hand"])
        commands = []
        for tick in range(55):
            command = policy.command(producer.tick_plan(tick), raw, {})
            commands.append(command)
            controller._history.append(tick)
            controller.decoder_calls += int(command["body_token"] is not None)
            if tick == 49:
                self.assertFalse(references)
                self.assertEqual(controller.decoder_calls, 40)
        self.assertEqual(warmup.receipt["encoder_calls"], 40)
        self.assertEqual(references, [0, 2, 4])
        self.assertTrue(all(command["body_token"] is None for command in commands[:10]))
        np.testing.assert_array_equal(commands[50]["body_token"], commands[51]["body_token"])
        self.assertEqual(commands[50]["hand_targets"], {"hand": 0})
        with self.assertRaisesRegex(ValueError, "every actual"):
            policy.command(producer.tick_plan(57), raw, {})


if __name__ == "__main__":
    unittest.main()
