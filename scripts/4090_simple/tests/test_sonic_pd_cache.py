"""CPU numerical equivalence to the original ordered SONIC torque controller.

The scalar oracle is the pre-optimization PD tail. No ONNX session, SDK,
rendering or GPU is used. Synthetic dynamics here are regression fixtures,
not demonstrations or a policy evaluation.
"""
from collections import deque
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "VLMExpert/third_party/SIMPLE/src"))
from simple.baselines.sonic_v1_1 import (BODY_JOINT_NAMES, BODY_KD, BODY_KP,
    SIMPLE_HAND_JOINT_NAMES, SonicV11OnnxController)


NAMES = BODY_JOINT_NAMES + SIMPLE_HAND_JOINT_NAMES


def scalar_pd_reference(controller, model, data, hand_targets):
    """Frozen original scalar mapping, arithmetic, clipping and error order."""
    targets = dict(zip(BODY_JOINT_NAMES, controller._body_target))
    targets.update(hand_targets)
    gains = dict(zip(BODY_JOINT_NAMES, zip(controller.body_kp, controller.body_kd)))
    gains.update({name: (5.0 if "thumb" in name else 2.5, 1.0) for name in SIMPLE_HAND_JOINT_NAMES})
    mapping = {}
    for aid in range(model.nu):
        if model.actuator_trntype[aid] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = int(model.actuator_trnid[aid, 0])
            name = model.joint(jid).name
            if name in targets:
                if name in mapping or not np.allclose(model.actuator_gear[aid], [1, 0, 0, 0, 0, 0]):
                    raise ValueError("SONIC adapter requires one unit-gear torque motor per named joint")
                mapping[name] = aid
    if set(mapping) != set(targets):
        raise ValueError("SIMPLE model does not expose all 43 named SONIC torque actuators")
    for name, target in targets.items():
        aid = mapping[name]
        joint = data.joint(name)
        kp, kd = gains[name]
        torque = kp * (target - joint.qpos[0]) - kd * joint.qvel[0]
        if not np.isfinite(torque):
            raise ValueError("Nonfinite SONIC PD torque")
        lower, upper = model.actuator_ctrlrange[aid] if model.actuator_ctrllimited[aid] else (-np.inf, np.inf)
        if controller.hand_torque_profile == "recording_0p7_nm" and name in SIMPLE_HAND_JOINT_NAMES:
            cap = 2.45 if "thumb_0" in name else .7
            lower, upper = max(lower, -cap), min(upper, cap)
        if np.isfinite(lower) or np.isfinite(upper):
            bounded = np.clip(torque, lower, upper)
            controller.clipped_torque_elements += int(bounded != torque)
            torque = bounded
        data.ctrl[aid] = torque


class ScalarController(SonicV11OnnxController):
    def _apply_named_pd(self, model, data, hand_targets):
        scalar_pd_reference(self, model, data, hand_targets)


class DecoderFixture:
    def __init__(self):
        self.inputs = []

    def run(self, outputs, values):
        observation = next(iter(values.values())).copy()
        self.inputs.append(observation)
        return [np.sin(np.arange(29, dtype=np.float32)[None] * .1 + len(self.inputs) * .01) * .1]


def make_model(seed=19):
    rng = np.random.default_rng(seed)
    joints = [NAMES[i] for i in rng.permutation(len(NAMES))] + ["unrelated_joint"]
    motors = [NAMES[i] for i in rng.permutation(len(NAMES))] + ["unrelated_joint"]
    bodies = "".join(f'<body name="link_{i}" pos="{i * .01} 0 0"><joint name="{name}" axis="0 1 0"/>'
                     '<geom type="sphere" size=".01" mass=".1" contype="0" conaffinity="0"/></body>'
                     for i, name in enumerate(joints))
    actuators = "".join(f'<motor name="motor_{i}" joint="{name}" ctrlrange="-20 20"/>'
                        for i, name in enumerate(motors))
    return mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005" gravity="0 0 0"/>'
        '<worldbody><body name="pelvis"><freejoint/><geom type="sphere" size=".05" mass="5" contype="0" conaffinity="0"/>'
        + bodies + '</body></worldbody><actuator>' + actuators + '</actuator></mujoco>')


def make_controller(kind=SonicV11OnnxController, *, profile="mjcf_limits"):
    controller = kind.__new__(kind)
    controller.body_kp, controller.body_kd = BODY_KP.copy(), BODY_KD.copy()
    controller.hand_torque_profile = profile
    controller.sync_mujoco_observations = False
    controller.session = DecoderFixture()
    controller.input_name, controller.output_name = "obs_dict", "action"
    controller.reset()
    controller._body_target = np.zeros(29, dtype=np.float64)
    return controller


class SonicPdCacheTests(unittest.TestCase):
    def pair(self, seed=19):
        model = make_model(seed)
        fast, slow = make_controller(), make_controller(ScalarController)
        first, second = mujoco.MjData(model), mujoco.MjData(model)
        first.ctrl[:] = second.ctrl[:] = 17.25  # The unrelated motor must survive.
        return model, fast, slow, first, second

    def assert_bits(self, actual, expected):
        self.assertEqual(np.asarray(actual).dtype, np.asarray(expected).dtype)
        np.testing.assert_array_equal(np.asarray(actual).view(np.uint64), np.asarray(expected).view(np.uint64))

    def compare(self, model, fast, slow, first, second, hands):
        fast._apply_named_pd(model, first, hands)
        slow._apply_named_pd(model, second, hands)
        self.assert_bits(first.ctrl, second.ctrl)
        self.assertEqual(fast.clipped_torque_elements, slow.clipped_torque_elements)

    def test_shuffled_actuators_and_joint_addresses_random_bitwise_equivalence(self):
        rng = np.random.default_rng(427)
        for profile in ("mjcf_limits", "recording_0p7_nm"):
            model, fast, slow, first, second = self.pair()
            fast.hand_torque_profile = slow.hand_torque_profile = profile
            for sample in range(100):
                first.qpos[:] = second.qpos[:] = rng.normal(0, .3, model.nq)
                first.qvel[:] = second.qvel[:] = rng.normal(0, 2., model.nv)
                fast._body_target = slow._body_target = rng.normal(0, .8, 29)
                fast.body_kp = slow.body_kp = rng.uniform(.1, 100., 29)
                fast.body_kd = slow.body_kd = rng.uniform(.1, 10., 29)
                hands = {SIMPLE_HAND_JOINT_NAMES[i]: float(rng.normal(0, 2.))
                         for i in rng.permutation(14)}
                model.actuator_ctrllimited[:] = rng.integers(0, 2, model.nu)
                model.actuator_ctrlrange[:, 0] = -rng.uniform(.01, 20., model.nu)
                model.actuator_ctrlrange[:, 1] = rng.uniform(.01, 20., model.nu)
                self.compare(model, fast, slow, first, second, hands)
            self.assertEqual(first.ctrl[-1], 17.25)

    def test_limits_gain_and_hand_profile_changes_apply_without_rebuilding_structure(self):
        model, fast, slow, first, second = self.pair()
        hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, 100.)
        fast._body_target = slow._body_target = np.ones(29)
        self.compare(model, fast, slow, first, second, hands)
        binding = fast._pd_binding
        for profile in ("recording_0p7_nm", "mjcf_limits"):
            fast.hand_torque_profile = slow.hand_torque_profile = profile
            model.actuator_ctrlrange[:] = [-.125, .25]
            model.actuator_ctrllimited[::2] = False
            fast.body_kp[:] = slow.body_kp[:] = np.arange(29) + 1.25
            fast.body_kd[:] = slow.body_kd[:] = .25
            first.qvel[:] = second.qvel[:] = 1.
            self.compare(model, fast, slow, first, second, hands)
            self.assertIs(fast._pd_binding, binding)

    def test_cached_structure_revalidates_gear_transmission_and_addresses(self):
        model, fast, slow, first, second = self.pair()
        hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, .2)
        self.compare(model, fast, slow, first, second, hands)
        binding = fast._pd_binding
        model.actuator_trnid[[0, 1], 0] = model.actuator_trnid[[1, 0], 0]
        self.compare(model, fast, slow, first, second, hands)
        self.assertIsNot(fast._pd_binding, binding)
        binding = fast._pd_binding
        first_joint, second_joint = [model.joint(name).id for name in BODY_JOINT_NAMES[:2]]
        model.jnt_qposadr[[first_joint, second_joint]] = model.jnt_qposadr[[second_joint, first_joint]]
        model.jnt_dofadr[[first_joint, second_joint]] = model.jnt_dofadr[[second_joint, first_joint]]
        self.compare(model, fast, slow, first, second, hands)
        self.assertIsNot(fast._pd_binding, binding)
        # A small permitted gear perturbation triggers the original validation.
        binding = fast._pd_binding
        model.actuator_gear[0, 0] += 1e-7
        self.compare(model, fast, slow, first, second, hands)
        self.assertIsNot(fast._pd_binding, binding)
        model.actuator_gear[0, 0] = 2.
        before = first.ctrl.copy()
        with self.assertRaisesRegex(ValueError, "unit-gear"):
            fast._apply_named_pd(model, first, hands)
        self.assert_bits(first.ctrl, before)

    def test_duplicate_missing_actuator_and_invalid_address_reject_after_cache(self):
        for corruption in ("duplicate", "missing", "address"):
            with self.subTest(corruption=corruption):
                model, fast, slow, first, second = self.pair()
                hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, .2)
                self.compare(model, fast, slow, first, second, hands)
                before = first.ctrl.copy()
                if corruption == "duplicate":
                    model.actuator_trnid[0, 0] = model.actuator_trnid[1, 0]
                elif corruption == "missing":
                    model.actuator_trntype[0] = mujoco.mjtTrn.mjTRN_SITE
                else:
                    model.jnt_qposadr[model.joint(BODY_JOINT_NAMES[0]).id] = model.nq
                with self.assertRaises(ValueError):
                    fast._apply_named_pd(model, first, hands)
                self.assert_bits(first.ctrl, before)

    def test_model_data_replacement_and_reset_drop_cached_handles(self):
        model, fast, slow, first, second = self.pair()
        hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, .2)
        self.compare(model, fast, slow, first, second, hands)
        binding = fast._pd_binding
        replacement = mujoco.MjData(model)
        fast._apply_named_pd(model, replacement, hands)
        self.assertIsNot(fast._pd_binding, binding)
        changed_model = make_model(29)
        changed_data = mujoco.MjData(changed_model)
        fast._apply_named_pd(changed_model, changed_data, hands)
        self.assertIs(fast._pd_binding["model"], changed_model)
        fast.reset()
        self.assertIsNone(fast._pd_binding)

    def test_nonfinite_inputs_preserve_original_exception_and_written_prefix(self):
        for field in ("target", "hand", "qpos", "qvel", "kp", "kd"):
            with self.subTest(field=field):
                model, fast, slow, first, second = self.pair()
                fast._body_target = slow._body_target = np.ones(29)
                hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, .2)
                self.compare(model, fast, slow, first, second, hands)
                first.ctrl[:] = second.ctrl[:] = -17.
                fast.clipped_torque_elements = slow.clipped_torque_elements = 0
                joint = BODY_JOINT_NAMES[15]
                if field == "target":
                    fast._body_target[15] = np.nan
                elif field == "hand":
                    hands[SIMPLE_HAND_JOINT_NAMES[7]] = np.inf
                elif field in ("qpos", "qvel"):
                    getattr(first.joint(joint), field)[0] = getattr(second.joint(joint), field)[0] = np.nan
                elif field == "kp":
                    fast.body_kp[15] = slow.body_kp[15] = np.inf
                else:
                    fast.body_kd[15] = slow.body_kd[15] = np.nan
                with np.errstate(invalid="ignore"):
                    with self.assertRaisesRegex(ValueError, "Nonfinite SONIC PD torque"):
                        fast._apply_named_pd(model, first, hands)
                    with self.assertRaisesRegex(ValueError, "Nonfinite SONIC PD torque"):
                        slow._apply_named_pd(model, second, hands)
                self.assert_bits(first.ctrl, second.ctrl)
                self.assertEqual(fast.clipped_torque_elements, slow.clipped_torque_elements)

    def test_signed_zero_boundary_nextafter_infinite_and_nan_limits_match_scalar(self):
        model, fast, slow, first, second = self.pair()
        fast._body_target = slow._body_target = np.zeros(29)
        hands = dict.fromkeys(SIMPLE_HAND_JOINT_NAMES, -0.)
        for bounds in ((-.7, .7), (-0., 0.), (-np.inf, np.inf), (-np.inf, .7), (-.7, np.inf),
                       (np.nan, np.nan), (np.nan, .7), (-.7, np.nan)):
            for target in (0., -0., .7, -.7, np.nextafter(.7, np.inf), np.nextafter(.7, -np.inf)):
                model.actuator_ctrllimited[:] = True
                model.actuator_ctrlrange[:] = bounds
                fast._body_target[:] = target
                hands.update(dict.fromkeys(hands, target))
                with np.errstate(invalid="ignore"):
                    self.compare(model, fast, slow, first, second, hands)

    def test_continuous_cpu_dynamics_decoder_history_and_state_remain_bitwise_equal(self):
        model = make_model()
        first, second = mujoco.MjData(model), mujoco.MjData(model)
        fast, slow = make_controller(), make_controller(ScalarController)
        fast._body_target = slow._body_target = None
        for data in (first, second):
            mujoco.mj_forward(model, data)
        first_robot, second_robot = SimpleNamespace(mjModel=model, mjData=first), SimpleNamespace(mjModel=model, mjData=second)
        for tick in range(240):
            hands = {name: float(np.sin(tick / 80 + i) * .1) for i, name in enumerate(SIMPLE_HAND_JOINT_NAMES)}
            token = np.ones(64, dtype=np.float32) if tick >= 40 else None
            if tick == 120:
                fast.hand_torque_profile = slow.hand_torque_profile = "recording_0p7_nm"
            fast.apply_physics(first_robot, token, hands)
            slow.apply_physics(second_robot, token, hands)
            self.assert_bits(first.ctrl, second.ctrl)
            mujoco.mj_step(model, first)
            mujoco.mj_step(model, second)
            self.assert_bits(first.qpos, second.qpos)
            self.assert_bits(first.qvel, second.qvel)
        self.assertEqual(fast.decoder_calls, slow.decoder_calls)
        self.assertEqual(fast.clipped_torque_elements, slow.clipped_torque_elements)
        self.assertGreater(fast.decoder_calls, 0)
        self.assertEqual(len(fast.session.inputs), len(slow.session.inputs))
        for actual, expected in zip(fast.session.inputs, slow.session.inputs):
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        for actual, expected in zip(fast._history, slow._history):
            self.assertEqual(actual[0], expected[0])
            for a, b in zip(actual[1:], expected[1:]):
                np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()
