import unittest

import torch

from MotionExpert.evaluate import compute_motion_metrics, sampled_validation
from MotionExpert.tests.fixtures import tiny_expert


def physical_motion(frames=40):
    motion = torch.zeros(frames, 245)
    body = motion[:, :108].reshape(frames, 12, 9)
    body[..., 0] = body[..., 4] = 1
    body[..., 8] = 1
    motion[:, 108] = motion[:, 112] = 1
    motion[:, 114] = .1
    motion[0, 114] = 0
    return motion


class DenseEvaluationTests(unittest.TestCase):
    def test_exact_recovery_and_missing_hand_exclusion(self):
        motion = physical_motion()
        valid = torch.ones_like(motion, dtype=torch.bool)
        valid[:, 180:243] = False
        pred = motion.clone()
        pred[:, 180:243] = 100
        metrics = compute_motion_metrics(pred, motion, valid)
        self.assertEqual(metrics["body_world_mpjpe_m"], 0)
        self.assertEqual(metrics["floor_ate_m"], 0)
        self.assertEqual(metrics["body_rotation_error_deg"], 0)
        self.assertEqual(metrics["right_hand_local_mpjpe_m_count"], 0)
        self.assertNotIn("right_hand_local_mpjpe_m", metrics)

    def test_fore_ignores_predicted_prefix_and_accumulates_future_drift(self):
        motion = physical_motion()
        valid = torch.ones_like(motion, dtype=torch.bool)
        floor = torch.eye(4).repeat(40, 1, 1)
        floor[:, 0, 3] = torch.arange(40) * .1
        pred = motion.clone()
        pred[:30] = 999
        pred[30:, 114] += .1
        metrics = compute_motion_metrics(pred, motion, valid, task="fore", forecast_prefix=30, floor_transform=floor)
        self.assertEqual(metrics["body_local_mpjpe_m"], 0)
        self.assertAlmostEqual(metrics["floor_final_drift_m"], 1., places=5)
        self.assertGreater(metrics["body_world_mpjpe_m"], .5)
        # Future reference transforms are never used for reanchoring.
        floor[30:] = 987
        again = compute_motion_metrics(pred, motion, valid, task="fore", forecast_prefix=30, floor_transform=floor)
        self.assertEqual(metrics, again)

    def test_invalid_reference_stops_world_evaluation_and_reports_coverage(self):
        motion = physical_motion()
        valid = torch.ones_like(motion, dtype=torch.bool)
        valid[10, 108:117] = False
        metrics = compute_motion_metrics(motion, motion, valid)
        self.assertEqual(metrics["reference_chain_coverage"], .25)
        self.assertEqual(metrics["floor_ate_m_count"], 10)
        self.assertEqual(metrics["body_local_mpjpe_m_count"], 40 * 12)

    def test_degenerate_body_rotation_is_penalized(self):
        motion = physical_motion()
        pred = motion.clone()
        pred[:, :108].reshape(40, 12, 9)[..., :6] = 0
        metrics = compute_motion_metrics(pred, motion, torch.ones_like(motion, dtype=torch.bool))
        self.assertEqual(metrics["body_rotation_error_deg"], 180)
        self.assertEqual(metrics["body_degenerate_rotation_fraction"], 1)

    def test_contacts_and_floor_physics_are_scored_in_physical_units(self):
        motion = physical_motion()
        motion[:, 243:] = 1
        body = motion[:, :108].reshape(40, 12, 9)
        body[:, [3, 4], 8] = -.02
        metrics = compute_motion_metrics(motion, motion, torch.ones_like(motion, dtype=torch.bool))
        self.assertEqual(metrics["contact_f1"], 1)
        self.assertAlmostEqual(metrics["foot_penetration_m"], .02, places=6)
        self.assertAlmostEqual(metrics["contact_foot_slide_m_s"], 1.5, places=5)

    def test_non_finite_generated_output_fails(self):
        motion = physical_motion()
        pred = motion.clone(); pred[0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            compute_motion_metrics(pred, motion, torch.ones_like(motion, dtype=torch.bool))

    def test_sampled_validation_runs_tasks_and_restores_rng(self):
        expert = tiny_expert()
        motion = physical_motion()
        class Dataset:
            fps = 15
            forecast_prefix = 30
            normalizer = None
            def __len__(self): return 1
            def get_sample(self, index, task):
                from MotionExpert.tests.fixtures import conditioning
                from MotionExpert.utils.task_conditioning import apply_task_conditioning
                y = {key: value[0] for key, value in conditioning(1, 40).items()}
                y = apply_task_conditioning(y, task, forecast_prefix=30)
                return {"y": y, "misc": {"motion": motion,
                    "num_real_frames": 40, "feature_valid_mask": torch.ones_like(motion, dtype=torch.bool),
                    "world_from_window": torch.eye(4), "floor_transform": torch.eye(4).repeat(40, 1, 1),
                    "floor_height": 0}}
        before = torch.random.get_rng_state().clone()
        metrics = sampled_validation(expert, Dataset(), max_samples=1)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        for task in ("recon", "fore", "gen"):
            self.assertEqual(metrics[f"{task}/sample_count"], 1)
            self.assertIn(f"{task}/body_rotation_error_deg", metrics)
        self.assertEqual(expert.model.calls, 30)


if __name__ == "__main__": unittest.main()
