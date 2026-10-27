import sys
from pathlib import Path
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_mobile_task import (collection_buttons, require_mobile_transition, task_identity,
    require_manipulation_pose_match, MobileStoppedWindow)


class MobileTaskTests(unittest.TestCase):
    def frame(self, **values):
        inputs = dict(left_axis_x=0., left_axis_y=0., right_axis_x=0., right_axis_y=0.)
        inputs.update(values)
        return SimpleNamespace(inputs=inputs)

    def test_explicit_identity_never_guesses(self):
        self.assertIn('LocomotionPickBetweenTables', task_identity('mobile_pp')[0])
        self.assertIn('Handover', task_identity('handover')[0])
        with self.assertRaises(ValueError):
            task_identity('other')

    def test_b_is_mobile_only_and_rising_edge(self):
        self.assertEqual(collection_buttons({'B': True}, {}, mobile=False), [])
        self.assertEqual(collection_buttons({'B': True}, {}, mobile=True), ['toggle_navigation'])
        self.assertEqual(collection_buttons({'B': True}, {'B': True}, mobile=True), [])

    def test_return_requires_all_human_joints_to_match_entry_pose(self):
        import numpy as np
        original=np.zeros((24,3)); current=original.copy()
        current[18,0]=.02
        self.assertEqual(require_manipulation_pose_match(original,current)
            ['maximum_root_local_smpl_joint_distance_m'],.02)
        current[18,0]=.031
        with self.assertRaisesRegex(ValueError,'恢复切入移动'):
            require_manipulation_pose_match(original,current)
        with self.assertRaisesRegex(ValueError,'缺少'):
            require_manipulation_pose_match(None,current)

    def test_switch_requires_measured_stop_neutral_and_completed_warmup(self):
        self.assertTrue(require_mobile_transition(self.frame(), [0.]*6)['sticks_neutral'])
        for frame, speed, warmup in [(self.frame(left_axis_y=.4), [0.]*6, False),
                (self.frame(), [.2,0,0,0,0,0], False),
                (self.frame(), [0,0,0,0,0,.3], False),
                (self.frame(), [0.]*6, True),
                (SimpleNamespace(inputs={}), [0.]*6, False)]:
            with self.subTest(frame=frame,speed=speed,warmup=warmup):
                with self.assertRaises(ValueError):
                    require_mobile_transition(frame,speed,recording_warmup=warmup)

    def test_mode_switch_requires_full_half_second_and_restarts_after_motion_or_gap(self):
        window = MobileStoppedWindow()
        for tick in range(25):
            window.observe(tick,self.frame(),[0.]*6,tilt_deg=0.,tracking_valid=True)
        self.assertFalse(window.ready)
        window.observe(25,self.frame(),[0.]*6,tilt_deg=0.,tracking_valid=True)
        self.assertTrue(window.ready)
        window.observe(26,self.frame(),[.1,0,0,0,0,0],tilt_deg=0.,tracking_valid=True)
        self.assertFalse(window.ready)
        for tick in range(27,53):
            window.observe(tick,self.frame(),[0.]*6,tilt_deg=0.,tracking_valid=True)
        self.assertTrue(window.ready)
        window.observe(54,self.frame(),[0.]*6,tilt_deg=0.,tracking_valid=True)
        self.assertFalse(window.ready)

    def test_cold_initialization_precedes_first_physics_step_and_unhooks(self):
        from unittest.mock import patch
        import pico_simple_teleop as entry
        names=['joint'+str(i) for i in range(29)]
        calls=[]
        data=SimpleNamespace(time=0.)
        model=SimpleNamespace(joint=lambda name:SimpleNamespace(range=(-2.,2.)))
        def original():
            calls.append(('physics',data.time));data.time+=.005
        engine=SimpleNamespace(step=original,mjData=data,mjModel=model)
        raw=SimpleNamespace(mujoco=engine)
        profile={'body_joint_names':names,'body_q29':[0.]*29,'profile_id':'test-profile'}
        def initialize(*args):
            calls.append(('cold',data.time));return {}
        with patch.object(entry,'initialize_pose',side_effect=initialize):
            receipt=entry.install_mobile_cold_start(raw,profile,names)
            engine.step();engine.step()
        self.assertEqual([kind for kind,_ in calls],['cold','physics','physics'])
        self.assertEqual(receipt['initialization']['initialization_count'],1)
        self.assertIs(engine.step,original)
        entry.install_mobile_cold_start(raw,profile,names)
        with self.assertRaisesRegex(ValueError,'after physical'):
            engine.step()


if __name__ == '__main__':
    unittest.main()
