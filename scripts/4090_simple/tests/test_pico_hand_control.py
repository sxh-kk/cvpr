"""CPU semantic checks, without importing SDKs, SIMPLE robot code or simulation."""
import ast
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_hand_control as hand


def robot_fixture():
    def config(side):
        return SimpleNamespace(joint_names=[f"{side}_hand_{suffix}" for suffix in hand.JOINT_SUFFIXES],
            init_qpos=[999.]*7, close_qpos=[-999.]*7)
    return SimpleNamespace(controller_cfg=SimpleNamespace(
        left_eef_cfg=config("left"),right_eef_cfg=config("right")))


def frame_fixture(**values):
    return SimpleNamespace(inputs={f"{side}_{control}":0. for side in ("left","right")
        for control in ("trigger","grip")} | values)


class HandControlTests(unittest.TestCase):
    def test_closed_posture_matches_only_pinned_upstream_pure_method(self):
        source = hand.UPSTREAM_SOURCE_PATH.read_text()
        self.assertEqual(hashlib.sha256(source.encode()).hexdigest(),hand.UPSTREAM_SOURCE_SHA256)
        original = next(node for node in ast.parse(source).body if isinstance(node,ast.ClassDef)
                        and node.name=="G1GripperInverseKinematicsSolver")
        method = next(node for node in original.body if isinstance(node,ast.FunctionDef)
                      and node.name=="_get_middle_close_q_desired")
        namespace = {"np":np}
        # Execute only arithmetic from the source method; never import the
        # upstream application, solver dependencies, SDK, or robot initializer.
        exec(compile(ast.Module(body=[method],type_ignores=[]),"pinned-middle-close","exec"),namespace)
        for side,expected in (("L",hand.LEFT_CLOSE),("R",hand.RIGHT_CLOSE)):
            actual = namespace[method.name](SimpleNamespace(side=side))
            np.testing.assert_array_equal(actual,expected)

    def test_sides_are_independent_and_right_closure_is_mirrored(self):
        target = hand.named_hand_targets(robot_fixture(),frame_fixture(left_trigger=1.,right_grip=.25))
        self.assertEqual(list(target),list(hand.HAND_JOINT_NAMES))
        np.testing.assert_array_equal(list(target.values())[:7],hand.LEFT_CLOSE)
        np.testing.assert_array_equal(list(target.values())[7:],np.asarray(hand.RIGHT_CLOSE)*.25)

    def test_grip_only_controls_closure_and_stronger_input_wins(self):
        a = hand.named_hand_targets(robot_fixture(),frame_fixture(left_grip=.8,right_grip=1.))
        b = hand.named_hand_targets(robot_fixture(),frame_fixture(left_trigger=.3,left_grip=.8,right_trigger=1.))
        self.assertEqual(a,b)
        self.assertEqual(a['left_hand_index_0_joint'],-.8)
        self.assertEqual(a['right_hand_thumb_1_joint'],-.7)

    def test_release_is_all_zero_and_does_not_inherit_old_close_or_open(self):
        target = hand.named_hand_targets(robot_fixture(),frame_fixture())
        self.assertEqual(list(target.values()),[0.]*14)
        self.assertFalse(np.signbit(list(target.values())).any())

    def test_validates_both_raw_inputs_even_when_the_other_is_fully_pressed(self):
        for key in frame_fixture().inputs:
            for invalid in (float('nan'),float('inf'),float('-inf'),-.001,1.001,'1',True,None):
                with self.subTest(key=key,invalid=invalid):
                    frame = frame_fixture(left_trigger=1.,left_grip=1.,right_trigger=1.,right_grip=1.)
                    frame.inputs[key] = invalid
                    with self.assertRaisesRegex(ValueError,key):
                        hand.named_hand_targets(robot_fixture(),frame)
            frame = frame_fixture()
            del frame.inputs[key]
            with self.assertRaisesRegex(ValueError,key):
                hand.named_hand_targets(robot_fixture(),frame)

    def test_named_mapping_survives_robot_config_reordering_without_mutation(self):
        robot = robot_fixture()
        frame = frame_fixture(left_trigger=.4,right_grip=.9)
        expected = hand.named_hand_targets(robot,frame)
        robot.controller_cfg.left_eef_cfg.joint_names.reverse()
        robot.controller_cfg.right_eef_cfg.joint_names[:] = np.roll(
            robot.controller_cfg.right_eef_cfg.joint_names,3).tolist()
        before_names = copy.deepcopy(vars(robot.controller_cfg))
        before_inputs = dict(frame.inputs)
        actual = hand.named_hand_targets(robot,frame)
        self.assertEqual(actual,expected)
        self.assertEqual(list(actual),list(hand.HAND_JOINT_NAMES))
        self.assertEqual(vars(robot.controller_cfg),before_names)
        self.assertEqual(frame.inputs,before_inputs)

    def test_rejects_missing_extra_duplicate_or_swapped_side_names(self):
        for replacement in (list(hand.HAND_JOINT_NAMES[:6]),list(hand.HAND_JOINT_NAMES[:7])+['extra'],
                            [hand.HAND_JOINT_NAMES[0]]*7,list(hand.HAND_JOINT_NAMES[7:])):
            robot = robot_fixture()
            robot.controller_cfg.left_eef_cfg.joint_names = replacement
            with self.assertRaisesRegex(ValueError,'seven named'):
                hand.named_hand_targets(robot,frame_fixture())
        with self.assertRaisesRegex(ValueError,'left_eef_cfg'):
            hand.named_hand_targets(SimpleNamespace(controller_cfg=SimpleNamespace()),frame_fixture())

    def test_profile_is_explicit_local_policy_and_source_bound(self):
        receipt = hand.profile_receipt()
        self.assertEqual(receipt['upstream']['sha256'],hand.UPSTREAM_SOURCE_SHA256)
        self.assertIn('local',receipt['policy_scope'])
        self.assertIn('max(trigger,grip)',receipt['policy'])
        self.assertFalse(receipt['torque_collision_physics_modified'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'changed_source.py'
            path.write_text('# mismatched source')
            with patch.object(hand,'UPSTREAM_SOURCE_PATH',path):
                with self.assertRaisesRegex(ValueError,'identity changed'):
                    hand.profile_receipt()

    def test_feedback_distinguishes_request_applied_and_actual_without_clamping(self):
        frame = frame_fixture(left_trigger=1.,right_grip=.3)
        actual = {name:value*scale for names,pose,scale in (
            (hand.HAND_JOINT_NAMES[:7],hand.LEFT_CLOSE,1.25),
            (hand.HAND_JOINT_NAMES[7:],hand.RIGHT_CLOSE,-.2)) for name,value in zip(names,pose)}
        command = hand.named_hand_targets(robot_fixture(),frame_fixture(left_trigger=.7,right_grip=.4))
        feedback = hand.hand_feedback(frame,actual,command,following=True)
        self.assertEqual(feedback['profile'],hand.PROFILE_ID)
        self.assertEqual(feedback['left']['requested_closed_fraction'],1.)
        self.assertEqual(feedback['right']['requested_closed_fraction'],.3)
        self.assertAlmostEqual(feedback['left']['applied_closed_fraction'],.7)
        self.assertAlmostEqual(feedback['right']['applied_closed_fraction'],.4)
        self.assertAlmostEqual(feedback['left']['actual_closed_fraction'],1.25)
        self.assertAlmostEqual(feedback['right']['actual_closed_fraction'],-.2)
        self.assertEqual(feedback['left']['trigger'],1.)
        self.assertEqual(feedback['left']['grip'],0.)

    def test_feedback_never_labels_paused_or_absent_command_as_applied(self):
        actual = dict.fromkeys(hand.HAND_JOINT_NAMES,0.)
        command = hand.named_hand_targets(robot_fixture(),frame_fixture(left_trigger=1.,right_trigger=1.))
        for following,targets in ((False,command),(True,None),(False,None)):
            result = hand.hand_feedback(frame_fixture(left_trigger=1.),actual,targets,following=following)
            self.assertEqual(result['left']['requested_closed_fraction'],1.)
            self.assertIsNone(result['left']['applied_closed_fraction'])
            self.assertIsNone(result['right']['applied_closed_fraction'])
            self.assertEqual(result['left']['actual_closed_fraction'],0.)

    def test_feedback_rejects_incomplete_or_nonfinite_actual_and_applied_joints(self):
        actual = dict.fromkeys(hand.HAND_JOINT_NAMES,0.)
        for values in ({**actual,hand.HAND_JOINT_NAMES[0]:float('nan')},
                       {key:value for key,value in actual.items() if key!=hand.HAND_JOINT_NAMES[-1]}):
            with self.assertRaisesRegex(ValueError,'Actual hand'):
                hand.hand_feedback(frame_fixture(),values)
        with self.assertRaisesRegex(ValueError,'Commanded hand'):
            hand.hand_feedback(frame_fixture(),actual,{},following=True)
        with self.assertRaisesRegex(ValueError,'right_grip'):
            hand.hand_feedback(frame_fixture(right_grip=2.),actual)


if __name__=='__main__':
    unittest.main()
