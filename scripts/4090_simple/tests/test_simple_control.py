"""CPU contracts for joint mapping, real-time history transport and v1.1 PD.

The deterministic decoder below is a unit fixture, never a policy evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import requests

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VLMExpert/third_party/SIMPLE/src"))
from simple.baselines.client import RequestMessage, ResponseMessage
from simple.baselines.dual_source_psi0 import (
    BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES, TRAINING_HAND_JOINT_NAMES,
    TRAINING_STATE_JOINT_NAMES, DualSourceHttpActionClient, DualSourcePsi0Agent,
    HistoryWarmingUp, simple_hand_targets, split_action80, state43_to45,
)
from simple.baselines.sonic_v1_1 import (
    ACTION_SCALE, BODY_KD, BODY_KP, BODY_TO_POLICY, DEFAULT_ANGLES,
    POLICY_TO_BODY, SonicV11OnnxController, body_targets, decoder_input,
)
from simple.core.action import ActionCmd


class FakeController:
    identity = {"body_token_version": "body_token_v1_1", "decoder_sha256": "a" * 64,
                "provenance": "unit fixture; not a controller compatibility claim"}

    def __init__(self):
        self.calls = []

    def reset(self):
        self.calls.clear()

    def hold(self, observation):
        return ActionCmd("unit_hold")

    def step(self, body_token, hand_targets, observation):
        self.calls.append((body_token.copy(), dict(hand_targets)))
        return ActionCmd("unit_action")


class FakeClient:
    def __init__(self):
        self.resets, self.observed, self.queries = [], [], []

    def reset(self, session_id, episode_id):
        self.resets.append((session_id, episode_id))

    def observe(self, images, *, history):
        self.observed.append((images, history))

    def query_action(self, images, instruction, state, *, history):
        self.queries.append((images, instruction, state, history))
        if history["sim_time_s"] < 1.0:
            raise HistoryWarmingUp({"status": "warming_up"})
        return np.repeat(np.arange(30, dtype=np.float32)[:, None], 80, axis=1)


def observation(t):
    return {"sim_time_s": t, "head_stereo_left": np.full((8, 10, 3), int(t * 10), dtype=np.uint8),
            "joint_names": BODY_JOINT_NAMES + SIMPLE_HAND_JOINT_NAMES,
            "joint_qpos": np.arange(43, dtype=np.float32)}


class SimpleControlContracts(unittest.TestCase):
    def test_named_timed_observation_matches_gymnasium_declaration(self):
        from gymnasium import spaces
        from gymnasium.utils.passive_env_checker import check_obs
        from simple.envs.observation_contract import named_sim_observation_space
        obs = observation(.02)
        obs["joint_names"] = tuple(obs["joint_names"])
        original = spaces.Dict({
            "joint_qpos": spaces.Box(-100, 100, shape=(43,), dtype=np.float32),
            "head_stereo_left": spaces.Box(0, 255, shape=(8, 10, 3), dtype=np.uint8),
        })
        declared = named_sim_observation_space(original, obs["joint_names"])
        check_obs(obs, declared, "reset")
        self.assertTrue(declared.contains(obs))
        self.assertNotIn("joint_names", original.spaces)

    def test_state_reorders_both_hands_by_name_and_pads_only_after_43(self):
        obs = observation(0)
        state = state43_to45(obs["joint_qpos"], obs["joint_names"])
        np.testing.assert_array_equal(state[0, :29], np.arange(29))
        np.testing.assert_array_equal(state[0, 29:43], [29, 30, 31, 34, 35, 32, 33, 36, 37, 38, 41, 42, 39, 40])
        np.testing.assert_array_equal(state[0, 43:], 0)
        self.assertEqual(state.shape, (1, 45))
        shuffled = np.random.default_rng(12).permutation(43)
        np.testing.assert_array_equal(state43_to45(obs["joint_qpos"][shuffled],
                                     [obs["joint_names"][i] for i in shuffled]), state)
        with self.assertRaises(ValueError):
            state43_to45(np.zeros(32), obs["joint_names"][:32])
        with self.assertRaises(ValueError):
            state43_to45(np.zeros(43), [*obs["joint_names"][:-1], obs["joint_names"][0]])

    def test_action_remaps_both_hands_and_never_uses_padding(self):
        action = np.arange(80, dtype=np.float32)
        action[78:] = [999, -999]
        latent, hands = split_action80(action)
        np.testing.assert_array_equal(latent, np.arange(64))
        left, right = simple_hand_targets(hands)
        np.testing.assert_array_equal(left, [64, 65, 66, 69, 70, 67, 68])
        np.testing.assert_array_equal(right, [71, 72, 73, 76, 77, 74, 75])
        for dim in (36, 78, 82):
            with self.assertRaises(ValueError):
                split_action80(np.zeros(dim))

    def test_frame_observation_continues_during_chunk_and_reset_clears_it(self):
        client, controller = FakeClient(), FakeController()
        agent = DualSourcePsi0Agent(None, client=client, controller=controller, session_id="a")
        agent.reset(episode_id="one")
        for i in range(56):
            cmd = agent.get_action(observation(i * .02), instruction="pick")
            self.assertEqual(cmd.type, "unit_hold" if i < 50 else "unit_action")
        self.assertEqual(len(client.observed), 56)
        self.assertEqual(len(client.queries), 51)
        np.testing.assert_array_equal([call[0][0] for call in controller.calls], [0, 0, 1, 1, 2, 3])
        self.assertEqual(client.queries[-1][2]["states"].shape, (1, 45))
        self.assertEqual(client.observed[-1][1], {"session_id": "a", "episode_id": "one", "sim_time_s": 1.1})
        agent.reset(episode_id="two")
        self.assertEqual(agent.get_action(observation(0)).type, "unit_hold")
        self.assertEqual(controller.calls, [])
        self.assertEqual(client.resets, [("a", "one"), ("a", "two")])
        with self.assertRaisesRegex(ValueError, "advance"):
            agent.get_action(observation(0))
        with self.assertRaisesRegex(ValueError, ">=30"):
            agent.get_action(observation(.1))

    def test_http_wire_and_warming_up_are_explicit(self):
        class Session:
            def __init__(self):
                self.calls = []
                self.next = {"status": "observed"}
                self.status = 200
            def post(self, url, *, json, timeout):
                self.calls.append((url, json, timeout))
                response = requests.Response()
                response.status_code = self.status
                response._content = __import__("json").dumps(self.next).encode()
                return response
        session = Session()
        client = DualSourceHttpActionClient(session=session)
        hist = {"session_id": "session", "episode_id": "episode", "sim_time_s": .5}
        client.observe({"rgb_head_stereo_left": observation(0)["head_stereo_left"]}, history=hist)
        decoded = RequestMessage.deserialize(session.calls[-1][1])
        self.assertEqual(decoded.history, hist)
        self.assertEqual(session.calls[-1][0], "http://127.0.0.1:22085/observe")
        self.assertFalse(session.trust_env)
        session.status, session.next = 425, {"detail": {"status": "warming_up"}}
        with self.assertRaises(HistoryWarmingUp):
            client.query_action({}, "pick", {}, history=hist)
        session.status, session.next = 409, {"detail": {"status": "invalid_observation"}}
        with self.assertRaises(requests.HTTPError):
            client.query_action({}, "pick", {}, history=hist)
        session.status, session.next = 200, ResponseMessage(np.zeros((30, 80)), 0).serialize()
        self.assertEqual(client.query_action({}, "pick", {}, history=hist).shape, (30, 80))
        with self.assertRaises(ValueError):
            DualSourceHttpActionClient("115.191.40.142")

    def test_decoder_proprioception_layout_is_term_major_and_real_50hz(self):
        history = [(i * .02, np.full(3, i), np.full(29, i + 10), np.full(29, i + 20),
                    np.full(29, i + 30), np.full(3, i + 40)) for i in range(10)]
        value = decoder_input(np.full(64, -1), history)
        np.testing.assert_array_equal(value[0, :64], -1)
        np.testing.assert_array_equal(value[0, 64:94].reshape(10, 3)[:, 0], np.arange(10))
        np.testing.assert_array_equal(value[0, 94:384].reshape(10, 29)[:, 0], np.arange(10) + 10)
        np.testing.assert_array_equal(value[0, -30:].reshape(10, 3)[:, 0], np.arange(10) + 40)
        with self.assertRaises(ValueError):
            decoder_input(np.zeros(64), [history[-1]] * 10)
        with self.assertRaises(ValueError):
            decoder_input(np.zeros(64), history[:-1])

    @unittest.skipUnless(shutil.which("g++"), "C++ compiler unavailable")
    def test_constants_match_fixed_official_cpp_at_runtime(self):
        header = REPO / "VLMExpert/third_party/GR00T-WholeBodyControl/gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/policy_parameters.hpp"
        code = f'''#include <vector>
#include <iostream>
#include <iomanip>
#include "{header}"
template<class T> void emit(const T& v) {{ for(auto x : v) std::cout << x << " "; std::cout << "\\n"; }}
int main() {{ std::cout << std::setprecision(17); emit(mujoco_to_isaaclab); emit(isaaclab_to_mujoco); emit(default_angles); emit(g1_action_scale); emit(kps); emit(kds); }}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "contract.cpp"
            binary = Path(directory) / "contract"
            source.write_text(code)
            subprocess.run(["g++", "-std=c++17", str(source), "-o", str(binary)], check=True, capture_output=True)
            rows = subprocess.check_output([str(binary)], text=True).splitlines()
        for actual, expected in zip(rows, [POLICY_TO_BODY, BODY_TO_POLICY, DEFAULT_ANGLES,
                                          ACTION_SCALE, BODY_KP, BODY_KD]):
            np.testing.assert_allclose(np.fromstring(actual, sep=" "), expected, rtol=1e-7, atol=1e-8)

    def test_physics_hook_samples_real_mujoco_at_50hz_and_maps_pd_by_name(self):
        import mujoco
        names = BODY_JOINT_NAMES + SIMPLE_HAND_JOINT_NAMES
        bodies = "".join(f'<body name="b{i}" pos="{i * .01} 0 0"><joint name="{name}" axis="0 1 0"/>'
                          '<geom type="sphere" size=".01" mass=".01" contype="0" conaffinity="0"/></body>'
                          for i, name in enumerate(names))
        motors = "".join(f'<motor name="m{i}" joint="{name}" ctrlrange="-150 150"/>' for i, name in enumerate(names))
        model = mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005" gravity="0 0 0"/>'
            '<worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".1" mass="5"/>' +
            bodies + '</body></worldbody><actuator>' + motors + '</actuator></mujoco>')
        data = mujoco.MjData(model)
        initial_q = DEFAULT_ANGLES + np.arange(29) * .001
        for name, q in zip(BODY_JOINT_NAMES, initial_q):
            data.joint(name).qpos[0] = q
        mujoco.mj_forward(model, data)
        controller = SonicV11OnnxController.__new__(SonicV11OnnxController)
        controller.body_kp, controller.body_kd = BODY_KP.copy(), BODY_KD.copy()
        controller.hand_torque_profile = "mjcf_limits"
        controller.sync_mujoco_observations = False
        controller.reset()
        inputs = []
        class Session:
            def run(self, outputs, values):
                inputs.append(next(iter(values.values())).copy())
                return [np.arange(29, dtype=np.float32)[None] / 100]
        controller.session = Session()
        controller.input_name, controller.output_name = "obs_dict", "action"
        robot = SimpleNamespace(mjModel=model, mjData=data)
        hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, 0.0)
        for _ in range(40):
            controller.apply_physics(robot, None, hands)
            mujoco.mj_step(model, data)
        self.assertEqual(controller.decoder_calls, 0)
        self.assertEqual(len(controller._history), 10)
        expected_hold_raw = ((initial_q - DEFAULT_ANGLES) / ACTION_SCALE)[POLICY_TO_BODY]
        for sample in controller._history:
            np.testing.assert_allclose(sample[4], expected_hold_raw, atol=1e-7)
        controller.apply_physics(robot, np.ones(64), hands)
        self.assertEqual(controller.decoder_calls, 1)
        self.assertEqual(inputs[0].shape, (1, 994))
        np.testing.assert_allclose(np.diff([row[0] for row in list(controller._history)[-10:]]), .02)
        expected = body_targets(np.arange(29) / 100)
        np.testing.assert_allclose(controller._body_target, expected)
        for name, target, kp in zip(BODY_JOINT_NAMES, expected, BODY_KP):
            aid = next(i for i in range(model.nu) if model.joint(int(model.actuator_trnid[i, 0])).name == name)
            self.assertAlmostEqual(data.ctrl[aid], kp * (target - data.joint(name).qpos[0]), places=5)
        body_before = {name: data.ctrl[next(i for i in range(model.nu) if model.joint(int(model.actuator_trnid[i, 0])).name == name)] for name in BODY_JOINT_NAMES}
        controller.hand_torque_profile = "recording_0p7_nm"
        controller.apply_physics(robot, None, dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, 100.0))
        for name in SIMPLE_HAND_JOINT_NAMES:
            aid = next(i for i in range(model.nu) if model.joint(int(model.actuator_trnid[i, 0])).name == name)
            self.assertAlmostEqual(data.ctrl[aid], 2.45 if "thumb_0" in name else .7)
        for name, value in body_before.items():
            aid = next(i for i in range(model.nu) if model.joint(int(model.actuator_trnid[i, 0])).name == name)
            self.assertEqual(data.ctrl[aid], value)
        controller.reset()
        self.assertEqual(len(controller._history), 0)
        with self.assertRaisesRegex(ValueError, "ten real"):
            controller.apply_physics(robot, np.ones(64), hands)


if __name__ == "__main__":
    unittest.main()
