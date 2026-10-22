"""Small real-MuJoCo checks of one-time cold starts; no policy/teacher replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import mujoco
import numpy as np

spec = importlib.util.spec_from_file_location("handover_runtime", Path(__file__).resolve().parents[1] / "handover_eval_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fixture():
    hinges = ''.join(f'<body pos="0 0 .01"><joint name="j{i}" range="-2 2"/><geom type="sphere" size=".01" mass=".01"/></body>' for i in range(43))
    xml = '<mujoco><compiler angle="radian"/><worldbody><body name="pelvis" pos="0 0 1"><freejoint/><geom type="sphere" size=".1" mass="1"/>' + hinges + '</body>'
    for name, x in (("target", 1), ("container", 2)):
        xml += f'<body name="{name}" pos="{x} 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass=".1"/></body>'
    model = mujoco.MjModel.from_xml_string(xml + '</worldbody></mujoco>')
    data = mujoco.MjData(model)
    data.time = .005
    raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=model, mjData=data,
                         mj_objects={n: data.body(n) for n in ("target", "container")}))
    raw._get_obs = lambda: {"sim_time_s": float(data.time)}
    raw._get_info = lambda: {}
    row = {"source_episode_index": 39, "joint_names": [f"j{i}" for i in range(43)],
           "joint_qpos": [.1] * 43, "root_pose7": [0, 0, 1, 1, 0, 0, 0],
           "objects": {n: {"pose7": [x, .2, 1, 1, 0, 0, 0], "mass": .1}
                       for n, x in (("target", 1), ("container", 2))}}
    return raw, row


class HandoverColdStartTests(unittest.TestCase):
    def test_named_reset_preserves_clock_and_does_not_step(self):
        raw, row = fixture()
        raw.mujoco.mjData.xfrc_applied[:] = 3
        obs, _, receipt = runtime.initialize(raw, row, {})
        self.assertEqual(obs["sim_time_s"], .005)
        self.assertTrue(receipt["applied_before_history_collection"])
        self.assertEqual(raw.mujoco.mjData.joint("j42").qpos[0], .1)
        np.testing.assert_array_equal(raw.mujoco.mjData.qvel, 0)
        np.testing.assert_array_equal(raw.mujoco.mjData.xfrc_applied, 0)
        np.testing.assert_allclose(raw.mujoco.mjData.body("target").xpos, [1, .2, 1])
        state = runtime.measured_task_state(raw)
        self.assertEqual(state["target_pose7"], [1, .2, 1, 1, 0, 0, 0])

    def test_mass_mismatch_refuses_before_live_state_mutation(self):
        raw, row = fixture()
        row["objects"]["container"]["mass"] = 1
        before = raw.mujoco.mjData.qpos.copy()
        with self.assertRaisesRegex(ValueError, "mass mismatch"):
            runtime.initialize(raw, row, {})
        np.testing.assert_array_equal(raw.mujoco.mjData.qpos, before)

    def test_small_recorded_soft_limit_penetration_retained_but_large_rejected(self):
        raw, row = fixture()
        row['joint_qpos'][0] = 2.0005
        _, _, receipt = runtime.initialize(raw, row, {})
        self.assertEqual(raw.mujoco.mjData.joint('j0').qpos[0], 2.0005)
        self.assertFalse(receipt['source_joint_values_clipped'])
        self.assertEqual(len(receipt['recorded_limit_penetrations']), 1)
        row['joint_qpos'][0] = 2.1
        before = raw.mujoco.mjData.qpos.copy()
        with self.assertRaisesRegex(ValueError, 'exceeds local model limits'):
            runtime.initialize(raw, row, {})
        np.testing.assert_array_equal(raw.mujoco.mjData.qpos, before)

    def test_invalid_joint_or_pose_refuses_before_mutation(self):
        for key in ("joint_qpos", "root_pose7"):
            raw, row = fixture()
            row[key][0] = float('nan')
            before = raw.mujoco.mjData.qpos.copy()
            with self.assertRaises(ValueError):
                runtime.initialize(raw, row, {})
            np.testing.assert_array_equal(raw.mujoco.mjData.qpos, before)

    def test_source_file_identity_and_val_only_contract(self):
        _, row = fixture()
        with tempfile.TemporaryDirectory() as tmp:
            evidence = Path(tmp) / 'source.json'
            evidence.write_text('{}')
            row['sources'] = {str(evidence): hashlib.sha256(evidence.read_bytes()).hexdigest()}
            path = Path(tmp) / 'initialization.json'
            def save():
                path.write_text(json.dumps({'contract': 'cvpr-handover-heldout-initial-v1', 'episodes': {'0': row}}))
            save()
            _, receipt = runtime.load_initialization(path, 0)
            self.assertFalse(receipt['teacher_actions_used'])
            evidence.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'evidence changed'):
                runtime.load_initialization(path, 0)
            row['source_episode_index'] = 0
            save()
            with self.assertRaisesRegex(ValueError, 'held-out'):
                runtime.load_initialization(path, 0)


if __name__ == '__main__':
    unittest.main()
