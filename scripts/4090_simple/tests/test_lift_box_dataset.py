"""CPU fixtures test verifier mechanics; fixtures never authorize real training."""
from pathlib import Path
import sys
import tempfile
import types
import unittest
import json

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_lift_box_hold as packing
import finetune_carrybox as shared


class FakeBridge:
    pass


class FakeEnvironment:
    def __init__(self, isaac=True):
        self.isaac = FakeBridge() if isaac else None
        self.isaac_observation_only = isaac
        self.sync_mujoco_observations = True
        self.task = types.SimpleNamespace(render_hz=50)


class LiftDatasetContracts(unittest.TestCase):
    def test_actual_producer_renderer_schema_and_egl_rejection(self):
        from run_closed_loop import simulation_render_contract
        contract = simulation_render_contract(FakeEnvironment(),types.SimpleNamespace())
        packing.validate_render_contracts([contract]*5)
        with self.assertRaisesRegex(ValueError,"RTX"):
            packing.validate_render_contracts([simulation_render_contract(FakeEnvironment(False),types.SimpleNamespace())]*5)

    def test_curated_offline_val3_requires_explicit_choice_and_fixed_train_test(self):
        from run_closed_loop import simulation_render_contract
        renderer = simulation_render_contract(FakeEnvironment(),types.SimpleNamespace())
        def audited(sources):
            return {source:([],[],{"frozen_contract":{"sha256":"same"},"render_contract":renderer,
                                   "run_role":"pilot_collection"}) for source in sources}
        curated = audited((0,1,2,3,4))
        with self.assertRaisesRegex(ValueError,"explicit VAL"):
            packing.validate_pilot(curated)
        self.assertEqual(packing.validate_pilot(curated,3),{"train":[0,1,2],"val":[3],"test":[4]})
        self.assertEqual(packing.validate_pilot(audited((0,1,2,8,4))),{"train":[0,1,2],"val":[8],"test":[4]})
        with self.assertRaisesRegex(ValueError,"TRAIN0/1/2"):
            packing.validate_pilot(audited((0,1,5,3,4)),3)

    def test_repeated_attempts_are_facts_without_a_one_shot_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            contract={"task_id":packing.TASK_ID,"instruction":packing.INSTRUCTION,"metric":packing.METRIC,"measurement_hz":50}
            controller={"physics_dt_s":.005,"decoder_hz":50,"command_hz":30,"pd_profile":"official_v1_1_ankle_pitch_1_5",
                        "hand_torque_profile":"mjcf_limits","observation_phase":packing.common.OBSERVATION_PHASE}
            frozen=root/'contract.json';frozen.write_text(json.dumps({"task_contract":contract,"controller_contract":controller}))
            digest=packing.common.sha256(frozen);log=root/'attempts.jsonl'
            rows=[{"attempt_id":attempt,"phase":"finished","source_episode":4,"run_output":str(root),
                   "frozen_contract_sha256":digest,"status":status} for attempt,status in
                  (("qualification","sim_time_limit"),("failed","error"),("current","task_terminated"))]
            log.write_text(''.join(json.dumps(row)+'\n' for row in rows));before=log.read_bytes()
            result={"status":"task_terminated","task_success_observed":True,"task_id":packing.TASK_ID,"seed":10005,
                    "warmup_seconds_requested":.2,"teacher":{"source_episode_index":4},"task_contract":contract,
                    "controller_contract":controller,"frozen_contract":{"path":str(frozen),"sha256":digest},
                    "attempt_log":{"path":str(log),"attempt_id":"current"}}
            packing.LiftAdapter().validate_result(root,result)
            self.assertEqual(log.read_bytes(),before)
    def physics_fixture(self, directory):
        import mujoco
        children = ''.join(f'<body name="j{i}"><joint name="{name}"/><geom type="sphere" size=".001" mass=".001" contype="0" conaffinity="0"/></body>'
                           for i, name in enumerate(packing.TRAIN_STATE_NAMES))
        xml = '<mujoco><option timestep=".005"/><worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>' + children + '</body><body name="left_hand_thumb_0_link" pos="0 0 .15"><geom type="box" size=".3 .3 .05"/></body><body name="cube"><freejoint name="cube_joint"/><geom type="box" size=".1 .1 .1" mass="1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        data.qpos[:7] = [0, 0, .8, 1, 0, 0, 0]
        data.joint("cube_joint").qpos[:] = [0, 0, .2999, 1, 0, 0, 0]
        path = directory / "model.mjb"; mujoco.mj_saveModel(model, str(path))
        states = []
        for time in np.arange(61)/50:
            data.time = time; mujoco.mj_forward(model, data)
            state = np.empty(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
            mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION); states.append(state)
        trajectory = {"sim_time_s": np.arange(61)/50, "physics_state_integration": np.stack(states),
            "physics_state_spec": np.asarray(int(mujoco.mjtState.mjSTATE_INTEGRATION)),
            "phase": np.array(["oracle_teacher_token"]*60+["task_terminal"]),
            "state43_joint_names": np.array(packing.TRAIN_STATE_NAMES), "actual_state43": np.zeros((61,43)),
            "actual_base_pose": np.tile(data.qpos[:7], (61,1)),
            "actual_box_pose": np.tile(data.joint("cube_joint").qpos, (61,1))}
        return model, data, path, trajectory

    def test_real_solved_force_and_terminal_integration_reconstruction(self):
        with tempfile.TemporaryDirectory() as directory:
            model, data, path, trajectory = self.physics_fixture(Path(directory))
            metric = packing.lift_measurement(model, data)
            self.assertGreater(metric["maximum_hand_normal_force_n"], .01)
            self.assertTrue(metric["condition"])
            result = packing.reconstruct_lift(path, trajectory, {}, [{"next_sim_time_s":1.2}])
            self.assertAlmostEqual(result["final_reconstructed_hold_s"], 1.2)
            self.assertEqual(result["measurement_hz"], 50)
            trajectory["phase"][-1] = "oracle_teacher_token"
            with self.assertRaisesRegex(ValueError, "terminal"):
                packing.reconstruct_lift(path, trajectory, {}, [{"next_sim_time_s":1.2}])

    def test_qpos_only_nonfinite_or_wrong_clock_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, path, trajectory = self.physics_fixture(Path(directory))
            trajectory["physics_state_spec"] = np.asarray(2)
            with self.assertRaisesRegex(ValueError, "qpos-only"):
                packing.reconstruct_lift(path, trajectory, {}, [{"next_sim_time_s":1.2}])
            trajectory["physics_state_spec"] = np.asarray(8191)
            trajectory["sim_time_s"][20] = np.nan
            with self.assertRaisesRegex(ValueError, "qpos-only"):
                packing.reconstruct_lift(path, trajectory, {}, [{"next_sim_time_s":1.2}])

    def test_oriented_bottom_and_missing_loaded_hand_are_not_success(self):
        import mujoco
        with tempfile.TemporaryDirectory() as directory:
            model, data, _, _ = self.physics_fixture(Path(directory))
            data.joint("cube_joint").qpos[2] = 1.
            mujoco.mj_forward(model,data)
            metric = packing.lift_measurement(model,data)
            self.assertFalse(metric["condition"])
            self.assertEqual(metric["maximum_hand_normal_force_n"],0)
            data.joint("cube_joint").qpos[:] = [0,0,.22,np.cos(np.pi/8),0,np.sin(np.pi/8),0]
            mujoco.mj_forward(model,data)
            self.assertAlmostEqual(packing.lift_measurement(model,data)["box_bottom_z_m"],.22-.1*np.sqrt(2))
            self.assertFalse(packing.lift_measurement(model,data)["condition"])

    def test_old_status_and_manual_pass_do_not_authorize_new_task(self):
        with self.assertRaisesRegex(ValueError, "newly executed"):
            packing.LiftAdapter().validate_result(Path("/tmp"), {"status":"source_end", "task_success_observed":True,
                "task_id":packing.TASK_ID,"task_contract":{"task_id":packing.TASK_ID,"instruction":packing.INSTRUCTION,
                "metric":packing.METRIC,"measurement_hz":50}})

    def test_new_clock_complete_future_and_one_test_episode(self):
        frames=np.arange(120); times=frames.astype(np.float32)/30; sources=(frames*50+29)//30; episodes=np.zeros(120)
        self.assertTrue(shared.future_is_complete(40,120,frames,times,sources,episodes,source_frame_rule="ceil"))
        self.assertFalse(shared.future_is_complete(40,120,frames,times,sources,episodes))
        self.assertFalse(shared.future_is_complete(92,120,frames,times,sources,episodes,source_frame_rule="ceil"))
        self.assertEqual(len(shared.uniform_windows({0:list(range(100))},20,expected_episodes=1)),20)
        with self.assertRaisesRegex(RuntimeError,"blocked"):
            shared.require_training_quality()


if __name__ == "__main__":
    unittest.main()
