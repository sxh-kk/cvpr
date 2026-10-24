"""CPU-only numerical/clock checks; official replay comparison is separate."""
from dataclasses import replace
from pathlib import Path
import json
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import handover_sonic_latent_projector as projector


class ProjectorTests(unittest.TestCase):
    @staticmethod
    def coupled_decoder(token):
        output = np.zeros(29)
        output[:15] = token[:15]
        output[15:] = token[:14] + token[15:29]
        return output

    def test_nullspace_step_improves_arms_without_following_desired_legs(self):
        desired = np.r_[np.full(15, 5.), np.linspace(-.05, .05, 14)]
        original = np.zeros(64, np.float32)
        result, report = projector.optimize_latent(original, desired, self.coupled_decoder)
        self.assertTrue(report["applied"])
        self.assertLess(report["arm_rms_after_rad"], report["arm_rms_before_rad"] / 100)
        self.assertLessEqual(report["max_protected_target_change_rad"], .002)
        self.assertLessEqual(report["latent_residual_linf"], .08 + 1e-7)
        self.assertLessEqual(report["latent_residual_l2"], .25 + 1e-7)
        np.testing.assert_array_equal(original, np.zeros(64))
        self.assertEqual(result.dtype, np.float32)

    def test_large_unreachable_goal_still_obeys_global_bounds(self):
        desired = np.r_[np.zeros(15), np.full(14, 100.)]
        result, report = projector.optimize_latent(np.zeros(64), desired, self.coupled_decoder)
        self.assertGreater(report["arm_rms_after_rad"], 90)
        self.assertLessEqual(np.max(np.abs(result)), .08 + 1e-7)
        self.assertLessEqual(np.linalg.norm(result), .25 + 1e-7)
        self.assertLessEqual(report["max_protected_target_change_rad"], .002)

    def test_nonlinear_protected_motion_is_rechecked_not_only_linearized(self):
        def nonlinear(z):
            value = self.coupled_decoder(z)
            value[0] += .5 * np.sum(z[15:29].astype(float) ** 2)
            return value
        config = replace(projector.ProjectionConfig(), max_protected_target_delta_rad=.0001)
        token, report = projector.optimize_latent(np.zeros(64), np.r_[np.zeros(15), np.full(14, .06)], nonlinear, config)
        actual = nonlinear(token)
        self.assertLessEqual(np.max(np.abs(actual[:15])), config.max_protected_target_delta_rad)
        self.assertLessEqual(report["max_protected_target_change_rad"], config.max_protected_target_delta_rad)

    def test_no_possible_improvement_retains_original_token(self):
        token = np.zeros(64, np.float32)
        result, report = projector.optimize_latent(token, np.ones(29), lambda z: np.zeros(29))
        self.assertFalse(report["applied"])
        np.testing.assert_array_equal(result, token)

    def test_nonfinite_decoder_or_inputs_are_rejected(self):
        for token, desired, decoder in ((np.full(64, np.nan), np.zeros(29), self.coupled_decoder),
                (np.zeros(64), np.full(29, np.inf), self.coupled_decoder),
                (np.zeros(64), np.zeros(29), lambda z: np.full(29, np.nan)),
                (np.full(64, 2.), np.zeros(29), self.coupled_decoder)):
            with self.subTest(token=token[0], desired=desired[0]), self.assertRaises(ValueError):
                projector.optimize_latent(token, desired, decoder)

    def test_iteration_budget_remains_bounded(self):
        _, report = projector.optimize_latent(np.zeros(64), np.r_[np.zeros(15), np.full(14, .02)], self.coupled_decoder)
        self.assertLessEqual(report["decoder_evaluations"], 1 + 2 * (64 + 8))

    def test_batch_jacobian_keeps_official_acceptance_and_reduces_B1_calls(self):
        desired = np.r_[np.zeros(15), np.full(14, .02)]
        token = np.zeros(64, np.float32)
        expected, original = projector.optimize_latent(token, desired, self.coupled_decoder)
        result, report = projector.optimize_latent(token, desired, self.coupled_decoder,
            predict_targets_batch=lambda rows: np.stack([self.coupled_decoder(row) for row in rows]))
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(report["derived_batch_evaluations"], 2)
        self.assertEqual(report["decoder_evaluations"], original["decoder_evaluations"] - 128)
        self.assertLessEqual(report["max_protected_target_change_rad"], .002)

    def test_unreliable_batch_jacobian_cannot_bypass_true_nonlinear_limits(self):
        # Malicious/inaccurate derived FD rows encourage legs to move: actual
        # candidate acceptance still evaluates the supplied original B1 function.
        def wrong_batch(rows):
            output = np.stack([self.coupled_decoder(row) for row in rows])
            output[:, :15] = 0
            return output
        token, report = projector.optimize_latent(np.zeros(64), np.r_[np.zeros(15), np.ones(14)],
            self.coupled_decoder, predict_targets_batch=wrong_batch)
        self.assertLessEqual(np.max(np.abs(self.coupled_decoder(token)[:15])), .002)
        self.assertLessEqual(report["max_protected_target_change_rad"], .002)

    def test_nonfinite_batch_or_wrong_shape_is_rejected(self):
        for value in (np.full((64, 29), np.nan), np.zeros((29, 64))):
            with self.assertRaises(ValueError):
                projector.optimize_latent(np.zeros(64), np.ones(29), self.coupled_decoder,
                    predict_targets_batch=lambda rows: value)

    def test_unbound_derived_artifact_is_rejected_before_session_construction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "derived.onnx"; path.write_bytes(b"invalid model")
            path.with_suffix(".onnx.json").write_text(json.dumps({"contract": projector.BATCH_CONTRACT,
                "official_decoder_sha256": projector.OFFICIAL_DECODER_SHA,
                "only_input_output_dim0_changed": True, "restored_complete_model_equal": True,
                "official_execution_claim": False, "derived_view_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                projector.DerivedBatchJacobianView(path)

    def test_invalid_configuration_is_rejected(self):
        for changes in ({"iterations": 0}, {"iterations": 4}, {"line_search_steps": 0},
                {"finite_difference_step": 0}, {"max_latent_l2": float("inf")},
                {"max_protected_target_delta_rad": -1}, {"latent_min": 2, "latent_max": 1},
                {"latent_max": 1.}, {"max_protected_target_delta_rad": .003}, {"control_scope": "all"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                projector.ProjectionConfig(**changes)

    def test_optional_waist_scope_optimizes17_joints_and_preserves12_legs(self):
        def decoder(token):
            return token[:29].copy()
        desired=np.r_[np.full(12,10.),np.full(3,.05),np.full(14,.03)]
        token=np.zeros(64,np.float32)
        _,default=projector.optimize_latent(token,desired,decoder)
        result,report=projector.optimize_latent(token,desired,decoder,projector.ProjectionConfig.waist_arms_experiment())
        self.assertEqual(default["protected_joint_count"],15)
        self.assertAlmostEqual(default["max_waist_target_change_rad"],0.)
        self.assertEqual(report["protected_joint_count"],12);self.assertEqual(report["controlled_joint_count"],17)
        self.assertLess(report["controlled_rms_after_rad"],report["controlled_rms_before_rad"]*.01)
        self.assertLess(report["waist_rms_after_rad"],default["waist_rms_after_rad"]*.01)
        self.assertLessEqual(report["max_leg_target_change_rad"],.002)
        self.assertLessEqual(np.max(np.abs(result-token)),.16+1e-7)
        self.assertLessEqual(np.linalg.norm(result-token),.5+1e-7)

    def test_waist_scope_nonlinear_actual_legs_are_rechecked_with_B1(self):
        def decoder(token):
            output=token[:29].astype(float).copy();output[0]+=.5*np.sum(token[12:15]**2)
            return output
        desired=np.r_[np.zeros(12),np.full(3,.15),np.full(14,.10)]
        config=projector.ProjectionConfig.waist_arms_experiment()
        token,report=projector.optimize_latent(np.zeros(64),desired,decoder,config,
            predict_targets_batch=lambda rows:np.stack([decoder(row) for row in rows]))
        self.assertTrue(report["applied"])
        self.assertLessEqual(np.max(np.abs(decoder(token)[:12])),.002)
        self.assertLessEqual(report["max_leg_target_change_rad"],.002)
        self.assertEqual(report["derived_batch_evaluations"],3)

    def test_original_positional_iteration_constructor_and_default_scope_remain_compatible(self):
        config=projector.ProjectionConfig(1)
        self.assertEqual(config.iterations,1);self.assertEqual(config.control_scope,"arms")

    def clock_fixture(self, times, actual_time, *, initialized=True):
        sonic = projector.sonic_module()
        history = [(float(t), np.zeros(3), np.zeros(29), np.zeros(29), np.zeros(29, np.float32),
                    np.array([0., 0., -1.])) for t in times]
        controller = SimpleNamespace(identity={"decoder_sha256": sonic.OFFICIAL_DECODER_SHA256},
            sync_mujoco_observations=True, session=SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"]),
            _last_tick=float(times[-1]) if len(times) else None, _history=history,
            _body_target=np.zeros(29) if initialized else None)
        raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=None, mjData=SimpleNamespace(time=actual_time)))
        return raw, controller

    def test_cold_start_never_pads_real_history(self):
        raw, controller = self.clock_fixture(np.arange(8) / 50, .16)
        token = np.zeros(64, np.float32)
        result, report = projector.SonicLatentProjector().project(token, np.zeros(29), raw, controller)
        self.assertFalse(report["applied"])
        self.assertEqual(report["reason"], "insufficient_real_controller_history")
        self.assertEqual(len(controller._history), 8)
        np.testing.assert_array_equal(result, token)

    def test_current_tick_already_consumed_never_predicts_a_different_tick(self):
        raw, controller = self.clock_fixture(np.arange(9) / 50, .16)
        _, report = projector.SonicLatentProjector().project(np.zeros(64), np.zeros(29), raw, controller)
        self.assertEqual(report["reason"], "not_the_exact_next_50Hz_decoder_tick")
        self.assertEqual(len(controller._history), 9)

    def test_clock_gap_is_rejected_instead_of_filling_missing_frame(self):
        raw, controller = self.clock_fixture(np.array([0, 1, 2, 3, 5, 6, 7, 8, 9]) / 50, .20)
        _, report = projector.SonicLatentProjector().project(np.zeros(64), np.zeros(29), raw, controller)
        self.assertEqual(report["reason"], "noncontiguous_real_controller_history")


if __name__ == "__main__":
    unittest.main()
