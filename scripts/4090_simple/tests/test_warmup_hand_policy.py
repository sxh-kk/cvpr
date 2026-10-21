"""CPU contracts for explicitly opening hands during live SONIC warmup."""
from contextlib import redirect_stderr
import copy
import inspect
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_live_pose_warmup import Controller, observation, warmup_fixture
from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
from simple.baselines.dual_source_psi0 import SIMPLE_HAND_JOINT_NAMES
import run_closed_loop


class WarmupHandPolicyTests(unittest.TestCase):
    def test_measured_default_and_invalid_policy(self):
        self.assertEqual(inspect.signature(LivePoseSonicWarmup).parameters["hand_policy"].default, "measured")
        with self.assertRaisesRegex(ValueError, "hand policy"):
            LivePoseSonicWarmup("/never/read/resources", hand_policy="teacher")

    def test_open_changes_only_post_history_hand_targets_and_receipt(self):
        robot = SimpleNamespace(mjData=SimpleNamespace(body=lambda name:SimpleNamespace(xquat=np.array([1.,0,0,0]))))
        measured, opened = warmup_fixture(), warmup_fixture("open")
        left, right = Controller(), Controller()
        for index in range(11):
            obs = observation(index, offset=.3)
            unchanged = copy.deepcopy(obs)
            left._history = list(range(index))
            right._history = list(range(index))
            for helper, controller in ((measured, left), (opened, right)):
                history = list(controller._history)
                command = helper.command(robot, controller, obs)
                self.assertEqual(command.type, "pd" if index < 10 else "latent")
                self.assertEqual(controller._history, history)
                np.testing.assert_array_equal(obs["joint_qpos"], unchanged["joint_qpos"])
                np.testing.assert_array_equal(obs["head_stereo_left"], unchanged["head_stereo_left"])
                self.assertEqual(obs["sim_time_s"], unchanged["sim_time_s"])
        expected = dict(zip(SIMPLE_HAND_JOINT_NAMES, observation(0,.3)["joint_qpos"][29:]))
        self.assertEqual(left.steps[0][1], expected)
        self.assertEqual(right.steps[0][1], {name:0. for name in SIMPLE_HAND_JOINT_NAMES})
        np.testing.assert_array_equal(left.steps[0][0], right.steps[0][0])
        np.testing.assert_array_equal(measured.session.inputs, opened.session.inputs)
        np.testing.assert_array_equal(measured.body_q, opened.body_q)
        np.testing.assert_array_equal(measured.root_wxyz, opened.root_wxyz)
        self.assertEqual(measured.receipt["initial_live_reference_sha256"], opened.receipt["initial_live_reference_sha256"])
        self.assertEqual(opened.receipt["first_actual_measured_hands14"], list(expected.values()))
        self.assertEqual(opened.receipt["hand_targets14"], [0.]*14)
        event = opened.receipt["events"][0]
        self.assertEqual(event["hand_policy"], "open")
        self.assertEqual(event["hand_targets14"], [0.]*14)
        np.testing.assert_array_equal(event["actual_hand_qpos14"], obs["joint_qpos"][29:])
        self.assertFalse(event["state_forcing"] or event["history_injection"])
        opened.reset()
        self.assertEqual(opened.hand_policy, "open")
        self.assertEqual(opened.receipt["events"], [])
        self.assertNotIn("first_actual_measured_hands14", opened.receipt)

    def test_zero_is_within_every_pinned_g1_hand_joint_limit(self):
        path = (Path(__file__).resolve().parents[3] / "VLMExpert/third_party/SIMPLE/third_party/gear_sonic"
                "/data/robot_model/model_data/g1/g1_29dof_with_hand.xml")
        joints = {node.attrib["name"]:node for node in ET.parse(path).iter("joint") if "name" in node.attrib}
        for name in SIMPLE_HAND_JOINT_NAMES:
            with self.subTest(joint=name):
                low, high = map(float, joints[name].attrib["range"].split())
                self.assertLessEqual(low, 0.)
                self.assertGreaterEqual(high, 0.)

    def test_cli_rejects_open_without_live_warmup_and_keeps_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            argv = ["run_closed_loop.py", "--data-root",str(root),"--sonic-resources",str(root),
                    "--output",str(root/"output"),"--seconds","1.2"]
            with patch.object(sys,"argv",argv):
                self.assertEqual(run_closed_loop.parse_args().warmup_hand_policy,"measured")
            with patch.object(sys,"argv",argv+["--warmup-hand-policy","open"]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):run_closed_loop.parse_args()
            with patch.object(sys,"argv",argv+["--warmup-control","live_pose_sonic_v1_1",
                                               "--sync-observations","--warmup-hand-policy","open"]):
                self.assertEqual(run_closed_loop.parse_args().warmup_hand_policy,"open")


if __name__ == "__main__":
    unittest.main()
