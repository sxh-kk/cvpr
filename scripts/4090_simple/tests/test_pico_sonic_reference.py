"""CPU checks for full-body reference semantics; never initialize PICO or simulation."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_sonic_reference as reference


HUMAN = reference.GEAR / "data/human/human_joints_info.pkl"


def body_fixture():
    body = np.zeros((24, 7))
    body[:, 6] = 1.
    body[0, :3] = [0., 1., 0.]
    for i in range(1, 24):
        body[i, :3] = body[reference.BODY_PARENTS[i], :3] + [.02 * (1 if i % 2 else -1), -.1, .01]
    return body


class FakeEncoder:
    class Spec:
        def __init__(self, name, shape):
            self.name, self.shape, self.type = name, shape, "tensor(float)"

    def get_inputs(self):
        return [self.Spec("obs_dict", [1, 1751])]

    def get_outputs(self):
        return [self.Spec("encoded_tokens", [1, 64])]

    def run(self, names, inputs):
        assert names == ["encoded_tokens"] and inputs["obs_dict"][0, 0] == 2
        return [np.ones((1, 64), dtype=np.float32) / 16]


class PicoSonicReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rest, cls.parents = reference.load_human_info(HUMAN)

    def make_reference(self):
        # Unit tests exercise real fixed FK and state machine, without requiring
        # downloaded ONNX assets. Actual artifact guards have separate checks.
        with patch.object(reference, "verify_runtime_sources", return_value={"unit_test": True}), \
             patch("onnxruntime.InferenceSession", return_value=FakeEncoder()):
            return reference.PicoSonicReference("unused-by-fake-session", HUMAN)

    def test_fk_matches_pinned_upstream_cpu_math(self):
        """An independent oracle executes only two upstream pure functions.

        In particular, the full upstream application is never imported, so its
        optional SDK, visualizer and service entry points cannot be executed.
        """
        import torch
        from gear_sonic.trl.utils import torch_transform as transform
        from gear_sonic.isaac_utils.rotations import remove_smpl_base_rot, smpl_root_ytoz_up

        transform.human_joints_info = torch.load(HUMAN, map_location="cpu", weights_only=True)
        source = (reference.GEAR / "scripts/pico_manager_thread_server.py").read_text()
        nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and
                 n.name in {"process_smpl_joints", "compute_from_body_poses"}]
        namespace = {"np": np, "sRot": Rotation, "torch": torch,
                     "remove_smpl_base_rot": remove_smpl_base_rot, "smpl_root_ytoz_up": smpl_root_ytoz_up}
        for name in ("angle_axis_to_quaternion", "quaternion_to_angle_axis", "compute_human_joints",
                     "quat_inv", "quat_apply", "quaternion_to_rotation_matrix"):
            namespace[name] = getattr(transform, name)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "pinned-pure-fk-functions", "exec"), namespace)
        for seed in (1, 2, 3):
            body = body_fixture()
            body[:, 3:] = Rotation.from_rotvec(np.random.default_rng(seed).normal(0, .2, (24, 3))).as_quat()
            expected = namespace["compute_from_body_poses"](reference.BODY_PARENTS, torch.device("cpu"), body)
            actual, root, _ = reference.body24_to_smpl(body, self.rest, self.parents)
            np.testing.assert_allclose(actual, expected["smpl_joints_local"].numpy()[0], atol=4e-7, rtol=1e-6)
            oracle_root = expected["global_orient_quat"].numpy()[0]
            self.assertAlmostEqual(abs(float(np.dot(root, oracle_root))), 1., places=6)

    def test_mode2_layout_and_explicit_desired_hold(self):
        item = self.make_reference()
        body = body_fixture()
        root = [0, 0, .75, 1, 0, 0, 0]
        actual_wrists = {name: i / 20 for i, name in enumerate(reference.WRIST_NAMES)}
        item.calibrate(body, root, actual_wrists)
        value, evidence = item.prepare(body, root, 1_000_000_000)
        self.assertEqual(value.shape, (1, 1751))
        self.assertEqual(value.dtype, np.float32)
        np.testing.assert_array_equal(value[0, :4], [2, 0, 0, 0])
        self.assertFalse(value[0, 4:911].any())
        joints = value[0, 911:1631].reshape(10, 24, 3)
        np.testing.assert_array_equal(joints, np.tile(joints[0], (10, 1, 1)))
        wrists = value[0, 1691:1751].reshape(10, 6)
        np.testing.assert_allclose(wrists, np.tile(list(actual_wrists.values()), (10, 1)), atol=1e-7)
        self.assertEqual(evidence["reference_horizon"]["kind"], "causal_constant_desired_pose_hold")
        self.assertEqual(evidence["reference_horizon"]["measured_future_frames"], 0)
        self.assertFalse(evidence["PICO_raw_va_wva_used"])
        self.assertFalse(evidence["physics_performed"])

    def test_leg_motion_changes_reference_without_g1_leg_padding(self):
        item = self.make_reference()
        body, root = body_fixture(), [0, 0, .75, 1, 0, 0, 0]
        item.calibrate(body, root)
        a, _ = item.prepare(body, root)
        bent = body.copy()
        bent[4, 3:] = Rotation.from_euler("x", .5).as_quat()
        b, _ = item.prepare(bent, root)
        self.assertGreater(np.linalg.norm(a[:, 911:1631] - b[:, 911:1631]), .1)
        self.assertFalse(b[0, 650:890].any())  # No mode1 fixed-leg substitute.

    def test_root_translation_is_not_encoded_and_heading_uses_columns(self):
        item = self.make_reference()
        body, root = body_fixture(), [0, 0, .75, 1, 0, 0, 0]
        item.calibrate(body, root)
        translated = body.copy()
        translated[:, :3] += [1., .2, -.5]
        a, _ = item.prepare(body, root)
        b, _ = item.prepare(translated, root)
        np.testing.assert_allclose(a, b, atol=1e-7)
        yaw = Rotation.from_euler("z", .6).as_quat()[[3, 0, 1, 2]]
        c, _ = item.prepare(body, [0, 0, .75, *yaw])
        expected = Rotation.from_euler("z", -.6).as_matrix()[:, :2].reshape(6)
        np.testing.assert_allclose(c[0, 1631:1637], expected, atol=1e-7)

    def test_reject_duplicate_reversed_or_gapped_timestamps(self):
        item = self.make_reference()
        body, root = body_fixture(), [0, 0, .75, 1, 0, 0, 0]
        item.calibrate(body, root)
        token, evidence = item.encode(body, root, 1_000_000_000)
        self.assertEqual(token.shape, (64,))
        self.assertTrue(evidence["model_inference_performed"])
        for timestamp in (1_000_000_000, 999_000_000, 1_251_000_000, True, -1):
            with self.assertRaises(ValueError):
                item.encode(body, root, timestamp)
        self.assertEqual(item.last_timestamp_ns, 1_000_000_000)

    def test_reject_pose_and_direction_jumps_without_advancing_state(self):
        item = self.make_reference()
        body, root = body_fixture(), [0, 0, .75, 1, 0, 0, 0]
        item.calibrate(body, root)
        item.encode(body, root, 1_000_000_000)
        jumped = body.copy()
        jumped[:, 0] += .2
        with self.assertRaisesRegex(ValueError, "position discontinuity"):
            item.encode(jumped, root, 1_020_000_000)
        rotated = body.copy()
        rotated[0, 3:] = Rotation.from_euler("y", 1.).as_quat()
        with self.assertRaisesRegex(ValueError, "orientation discontinuity"):
            item.encode(rotated, root, 1_020_000_000)
        self.assertEqual(item.last_timestamp_ns, 1_000_000_000)
        np.testing.assert_array_equal(item.last_body, body)

    def test_failed_encoder_does_not_consume_the_next_tracking_frame(self):
        item = self.make_reference()
        body, root = body_fixture(), [0, 0, .75, 1, 0, 0, 0]
        item.calibrate(body, root)
        item.encode(body, root, 1_000_000_000)
        with patch.object(item.session, "run", return_value=[np.full((1, 64), np.nan)]):
            with self.assertRaisesRegex(ValueError, "Invalid SONIC"):
                item.encode(body, root, 1_020_000_000)
        self.assertEqual(item.last_timestamp_ns, 1_000_000_000)
        item.encode(body, root, 1_020_000_000)
        self.assertEqual(item.last_timestamp_ns, 1_020_000_000)

    def test_invalid_pose_and_uncalibrated_requests_fail(self):
        item = self.make_reference()
        body, root = body_fixture(), [0, 0, .75, 1, 0, 0, 0]
        with self.assertRaisesRegex(ValueError, "calibration"):
            item.prepare(body, root)
        invalid = body.copy()
        invalid[3, 3:] = 0
        with self.assertRaisesRegex(ValueError, "unit xyzw"):
            item.calibrate(invalid, root)
        invalid = body.copy()
        invalid[1, :3] = invalid[0, :3]
        with self.assertRaisesRegex(ValueError, "skeletal segment"):
            item.calibrate(invalid, root)
        with self.assertRaisesRegex(ValueError, "six named"):
            item.calibrate(body, root, {"left_wrist_roll_joint": 0})

    def test_modified_skeleton_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "human.pkl"
            path.write_bytes(b"not the fixed skeleton")
            with self.assertRaisesRegex(ValueError, "Pinned reference identity"):
                reference.load_human_info(path)


if __name__ == "__main__":
    unittest.main()
