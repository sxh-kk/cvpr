"""CPU-only initial-state fidelity and exact-model rejection contracts."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from initial_physics_state import InitialPhysicsState, compare_models, file_sha256, read_initial_integration_state


XML = """<mujoco><option timestep='.005'/><worldbody>
<geom name='floor' type='plane' size='2 2 .1'/>
<body name='pelvis' pos='0 0 1'><freejoint name='root'/>
<geom name='box' type='box' size='.1 .2 .3' mass='2'/>
<camera name='head' pos='0 0 .4'/>
<body name='arm' pos='.2 0 0'><joint name='hinge' axis='0 1 0'/>
<geom name='tip' type='sphere' size='.04' mass='.1'/></body></body>
</worldbody><actuator><motor name='motor' joint='hinge'/></actuator></mujoco>"""
TASK = "simple/G1WholebodyLiftBoxHoldSonic-v0"
SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def fixture(directory):
    model = mujoco.MjModel.from_xml_string(XML)
    mujoco.mj_saveModel(model, str(directory / "physics_model.mjb"))
    data = mujoco.MjData(model)
    data.time, data.qpos[0], data.qpos[-1] = .005, -.3, .25
    data.qvel[:] = np.arange(model.nv) * .01
    data.ctrl[:] = .17
    data.xfrc_applied[1, 0] = .12
    data.qacc_warmstart[:] = .03
    mujoco.mj_forward(model, data)
    state = np.empty(mujoco.mj_stateSize(model, SPEC))
    mujoco.mj_getState(model, data, state, SPEC)
    # Future row and an object-valued action are deliberately invalid. Neither
    # may be decoded or used by a one-frame initializer.
    np.savez_compressed(directory / "trajectory.npz", physics_state_spec=np.int32(SPEC),
        physics_state_integration=np.stack([state, np.full_like(state, np.nan)]),
        action=np.array([{"must_not_read": True}], dtype=object))
    result = {"task_id": TASK, "seed": 10009, "physical_outcome_status": "source_end",
              "physics_model": {"sha256": file_sha256(directory / "physics_model.mjb")}}
    (directory / "result.json").write_text(json.dumps(result))
    return model, state


class InitialPhysicsTests(unittest.TestCase):
    def test_restore_preserves_model_data_views_state_and_gets_fresh_observation_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, expected = fixture(directory)
            model = mujoco.MjModel.from_xml_string(XML)
            data = mujoco.MjData(model)
            cached = data.joint("hinge").qpos
            calls = []
            raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=model, mjData=data), _success=True,
                                  task=SimpleNamespace(_reset_progress=lambda: calls.append("reset_progress")))
            def observe():
                calls.append("fresh_rgb")
                self.assertAlmostEqual(float(cached[0]), .25)
                self.assertAlmostEqual(float(data.time), .005)
                return {"sim_time_s": float(data.time), "head_stereo_left": np.zeros((2, 3, 3), np.uint8)}
            raw._get_obs, raw._get_info = observe, lambda: {"new_info": True}
            initializer = InitialPhysicsState(directory, task_id=TASK, seed=10009)
            observation, info, receipt = initializer.restore(raw)
            self.assertIs(raw.mujoco.mjModel, model)
            self.assertIs(raw.mujoco.mjData, data)
            self.assertTrue(np.shares_memory(cached, data.qpos))
            actual = np.empty_like(expected)
            mujoco.mj_getState(model, data, actual, SPEC)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(calls, ["reset_progress", "fresh_rgb"])
            self.assertFalse(raw._success)
            self.assertEqual(receipt["initial_state_sha256"], receipt["restored_state_sha256"])
            self.assertEqual(receipt["initialization_count"], 1)
            self.assertGreater(receipt["model_compatibility"]["fields_compared"], 500)
            with self.assertRaisesRegex(RuntimeError, "once"):
                initializer.restore(raw)

    def test_same_shape_different_static_physics_or_names_rejected_before_assignment(self):
        source = mujoco.MjModel.from_xml_string(XML)
        changes = {
            "body_mass": lambda m: m.body_mass.__setitem__(1, 3),
            "body_inertia": lambda m: m.body_inertia.__setitem__((1, 0), .9),
            "geom_friction": lambda m: m.geom_friction.__setitem__((1, 0), .3),
            "geom_size": lambda m: m.geom_size.__setitem__((1, 0), .15),
            "jnt_axis": lambda m: m.jnt_axis.__setitem__((1, 1), .5),
            "cam_pos": lambda m: m.cam_pos.__setitem__((0, 2), .41),
            "actuator_gear": lambda m: m.actuator_gear.__setitem__((0, 0), 2),
            "opt.timestep": lambda m: setattr(m.opt, "timestep", .01),
        }
        for field, mutate in changes.items():
            with self.subTest(field=field):
                target = mujoco.MjModel.from_xml_string(XML)
                mutate(target)
                with self.assertRaisesRegex(ValueError, field):
                    compare_models(source, target)
        renamed = mujoco.MjModel.from_xml_string(XML.replace("name='hinge'", "name='other'").replace("joint='hinge'", "joint='other'"))
        with self.assertRaisesRegex(ValueError, "names"):
            compare_models(source, renamed)

    def test_only_first_row_and_spec_are_opened_even_when_future_is_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, state = fixture(directory)
            opened = []
            original = zipfile.ZipFile.open
            def tracked(archive, name, *args, **kwargs):
                opened.append(name)
                return original(archive, name, *args, **kwargs)
            with mock.patch.object(zipfile.ZipFile, "open", tracked):
                actual, spec = read_initial_integration_state(directory / "trajectory.npz")
            self.assertEqual(opened, ["physics_state_spec.npy", "physics_state_integration.npy"])
            np.testing.assert_array_equal(actual, state)
            self.assertEqual(spec, SPEC)

    def test_task_seed_mjb_hash_and_incomplete_state_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            _, state = fixture(directory)
            for kwargs in ({"task_id": "wrong", "seed": 10009}, {"task_id": TASK, "seed": 10008}):
                with self.assertRaisesRegex(ValueError, "same task and seed"):
                    InitialPhysicsState(directory, **kwargs)
            result_path = directory / "result.json"
            result = json.loads(result_path.read_text())
            result["physics_model"]["sha256"] = "0" * 64
            result_path.write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, "SHA256"):
                InitialPhysicsState(directory, task_id=TASK, seed=10009)
            np.savez(directory / "bad.npz", physics_state_spec=np.int32(1), physics_state_integration=state[None])
            with self.assertRaisesRegex(ValueError, "complete"):
                read_initial_integration_state(directory / "bad.npz")


if __name__ == "__main__":
    unittest.main()
