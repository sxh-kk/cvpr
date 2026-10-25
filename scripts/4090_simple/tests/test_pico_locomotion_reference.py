"""Planner/encoder timeline regressions without SDK, real models or physics."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_locomotion_reference import OfflinePlannerReference, encoder_reference
from sonic_locomotion_planner import Forecast
from handover_sonic_teleop_reference import rotation


def forecast():
    q = np.zeros((72, 36))
    q[:, 2] = .78
    q[:, 3] = 1.
    q[:, 0] = np.arange(72) * .006
    return Forecast(q, np.zeros((72, 29)), np.ones(72, bool), {})


class FakePlanner:
    def __init__(self):
        self.calls = []

    def infer(self, context, alignment, **command):
        self.calls.append((np.array(context), command))
        value = forecast()
        return value, value, {'fake_test_only': True}, {}, None


def idle():
    return {'mode': 0, 'movement_direction': [0., 0., 0.],
            'facing_direction': [1., 0., 0.], 'speed': -1., 'height': -1.}


ROOT = np.array([0., 0., .78, 1., 0., 0., 0.])


class LocomotionReferenceTests(unittest.TestCase):
    def test_mode0_actual_heading_inverse_and_future_order(self):
        q = forecast().qpos50[:46].copy()
        q[:, 7:] = np.arange(46 * 29).reshape(46, 29) / 1000
        root = ROOT.copy()
        root[3:] = [np.cos(.3), 0., 0., np.sin(.3)]
        value, evidence = encoder_reference(q, np.ones((46, 29)), root)
        self.assertEqual(value.shape, (1, 1751))
        self.assertEqual(value.dtype, np.float32)
        self.assertFalse(value[0, :4].any())
        self.assertFalse(value[0, 644:].any())
        expected = rotation(root[3:]).T[:, :2].reshape(-1)
        np.testing.assert_allclose(value[0, 584:590], expected, atol=1e-7)
        self.assertFalse(evidence['model_forecasts_are_training_labels'])

    def test_mode1_preserves_vr_world_to_root_and_lower12(self):
        q = forecast().qpos50[:46].copy()
        q[:, 7:] = np.arange(29) / 30
        points = np.array([[.4, .2, 1.], [.4, -.2, 1.], [0., 0., 1.3]])
        quats = np.tile([1., 0., 0., 0.], (3, 1))
        value, evidence = encoder_reference(q, np.ones((46, 29)), ROOT,
            vr_world_points=points, vr_world_quaternions=quats)
        self.assertEqual(value[0, 0], 1.)
        np.testing.assert_allclose(value[0, 650:662], q[0, 7:19], atol=1e-7)
        np.testing.assert_allclose(value[0, 890:899], (points-ROOT[:3]).reshape(-1), atol=1e-7)
        self.assertFalse(value[0, 911:].any())
        self.assertEqual(evidence['encoder_mode'], 1)

    def test_partial_vr_invalid_root_and_short_future_rejected(self):
        q, dq = forecast().qpos50[:46], np.zeros((46, 29))
        with self.assertRaises(ValueError):
            encoder_reference(q, dq, ROOT, vr_world_points=np.zeros((3, 3)))
        for bad in (ROOT * 2., np.full(7, np.nan)):
            with self.assertRaises(ValueError):
                encoder_reference(q, dq, bad)
        with self.assertRaises(ValueError):
            encoder_reference(q[:45], dq[:45], ROOT)

    def test_stop_mode_is_replanned_immediately_and_time_cannot_skip(self):
        fake = FakePlanner()
        reference = OfflinePlannerReference(None, planner=fake)
        body = np.zeros(29)
        reference.step(0, ROOT, body, idle())
        walking = dict(idle(), mode=1, movement_direction=[1., 0., 0.], speed=.3)
        reference.step(1, ROOT, body, walking)
        _, evidence = reference.step(2, ROOT, body, idle())
        self.assertEqual([x[1]['mode'] for x in fake.calls], [0, 1, 0])
        self.assertEqual(evidence['replan_reason'], 'official_mode_changed')
        with self.assertRaises(ValueError):
            reference.step(4, ROOT, body, idle())

    def test_low_watermark_replans_before_future_runs_out_without_padding(self):
        fake = FakePlanner()
        reference = OfflinePlannerReference(None, planner=fake)
        for tick in range(40):
            value, evidence = reference.step(tick, ROOT, np.zeros(29), idle())
            self.assertEqual(value.shape, (1, 1751))
        self.assertGreater(len(fake.calls), 1)
        self.assertLess(reference.cursor, 27)

    def test_no_unsupported_modes_unbounded_speed_or_nonplanar_intent(self):
        for command in (dict(idle(), mode=2), dict(idle(), speed=0.),
                        dict(idle(), mode=1, speed=.01, movement_direction=[1., 0., 0.]),
                        dict(idle(), mode=1, speed=2., movement_direction=[1., 0., 0.]),
                        dict(idle(), facing_direction=[0., 0., 1.])):
            with self.assertRaises(ValueError):
                OfflinePlannerReference(None, planner=FakePlanner()).step(0, ROOT, np.zeros(29), command)


if __name__ == '__main__':
    unittest.main()
