"""Execute collector production blocks with CPU fakes; no SDK or simulator.

The AST comes from run(), so these checks cover the actual collector call
ordering rather than a second implementation of its 50/30 Hz dispatch.
"""
import ast
import copy
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_simple_teleop as entry


def run_tree():
    return ast.parse(inspect.getsource(entry.run))


def statement_list_containing(predicate):
    for parent in ast.walk(run_tree()):
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list) and value and all(isinstance(node, ast.stmt) for node in value):
                for index, node in enumerate(value):
                    if predicate(node):
                        return value, index
    raise AssertionError("Production control block was not found")


def is_assign(node, name):
    return isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)


def execute(nodes, namespace):
    module = ast.Module(body=copy.deepcopy(nodes), type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), entry.__file__, "exec"), namespace)


def command_block():
    statements, start = statement_list_containing(lambda node: is_assign(node, "due"))
    end = next(index for index in range(start + 1, len(statements))
               if isinstance(statements[index], ast.If)
               and ast.unparse(statements[index].test).startswith("capture and state"))
    return statements[start:end]


def step_and_native_block():
    statements, start = statement_list_containing(lambda node: isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "env.step"
        and any("terminated" in ast.unparse(target) for target in node.targets))
    assert isinstance(statements[start + 1], ast.If)
    assert ast.unparse(statements[start + 1].test) == "pending_mobile_quality_row is not None"
    return statements[start:start + 2]


class FakeReference:
    def __init__(self):
        self.calls = []
        self.calibrations = []
        self.continuity_calls = []
        self.input_name, self.output_name = "input", "token"
        self.session = SimpleNamespace(run=self.infer)
        self.last_timestamp_ns = None
        self.last_body = None
        self.calibration = {"initial": True}
        self.rest_joints = np.zeros((24, 3))
        self.fk_parents = np.zeros(24, dtype=int)

    def infer(self, outputs, inputs):
        self.calls.append(("planner_encode", inputs["input"].copy()))
        return [np.zeros((1, 64), dtype=np.float32)]

    def encode(self, body, root, timestamp):
        self.calls.append(("pose_encode", timestamp, root.copy()))
        self.last_timestamp_ns = timestamp
        self.last_body = body.copy()
        return np.zeros(64, dtype=np.float32), {"encoder_mode": 2}

    def calibrate(self, body, root, wrists):
        self.calibrations.append((body.copy(), root.copy(), dict(wrists)))
        self.last_timestamp_ns = self.last_body = None
        self.calibration = {"actual_root_pose7": root.tolist()}

    def _validate_continuity(self, body, timestamp):
        if self.last_timestamp_ns is not None and timestamp <= self.last_timestamp_ns:
            raise ValueError("PICO command timestamp must advance")
        self.continuity_calls.append((timestamp, body.copy()))


class FakeLiveVR:
    def __init__(self):
        self.calls = []
        self.calibration = {"cpu_fixture": True}

    def calibrate(self, body, root, positions, quaternions):
        self.calibration = {"contract": "cpu_test_vr_calibration", "calibration_sha256": "b" * 64,
                            "actual_root_pose7": root.tolist()}
        return copy.deepcopy(self.calibration)

    def prepare(self, body, root):
        if self.calibration is None:
            raise ValueError("Live VR calibration required")
        points = body[[20, 21, 15], :3] + root[None, :3]
        quaternions = np.tile([1., 0., 0., 0.], (3, 1))
        self.calls.append((body.copy(), root.copy(), points.copy()))
        return points, quaternions, {"producer": "cpu_test_current_packet_vr", "encoder_mode": 1}

class FakeMobile:
    def __init__(self):
        self.ticks = []
        self.pauses = []
        self.boots = []
        self.vr_calls = []
        self.ready = True

    def poll(self):
        return {"ready": self.ready}

    def step(self, tick, root, body, frame, *, tracking_valid,
             vr_world_points=None, vr_world_quaternions=None):
        self.ticks.append((tick, frame.device_timestamp_ns, tracking_valid))
        if vr_world_points is not None:
            from pico_locomotion_reference import encoder_reference
            q = np.repeat(np.r_[root, body][None], 46, axis=0)
            value, evidence = encoder_reference(q, np.zeros((46, 29)), root,
                vr_world_points=vr_world_points, vr_world_quaternions=vr_world_quaternions)
            self.vr_calls.append((frame.sequence, frame.device_timestamp_ns, np.asarray(vr_world_points).copy()))
            evidence.update(contract="cvpr-pico-mobile-continuous-vr-mode1-async-reference-v1", tick50=tick,
                reference_policy="continuous_vr_mode1", upper_body_frozen=False)
            return value, evidence
        return np.zeros((1, 1751), dtype=np.float32), {"encoder_mode": 1, "tick50": tick}

    def pause(self, reason):
        self.pauses.append(reason)

    def boot(self, root, body):
        self.boots.append((root.copy(), list(body)))


class CollectorMobileIntegrationTests(unittest.TestCase):
    def namespace(self):
        ledger_rows, commands = [], []
        reference, mobile = FakeReference(), FakeMobile()
        root = np.array([1., 2., .75, .7071067811865476, 0., 0., .7071067811865476])
        joint = lambda name: SimpleNamespace(range=(-10., 10.), qpos=[.1])
        namespace = dict(vars(entry), global_tick=0, command=None, last_command_timestamp_ns=None,
            model=SimpleNamespace(joint=joint), data=SimpleNamespace(time=0., joint=joint), BODY_JOINT_NAMES=["b" + str(i) for i in range(29)],
            physical_snapshot=lambda *_: {"body_q29": [0.] * 29}, root_pose=lambda *_: root.copy(),
            mobile_mode="navigation", mobile_tick=0, mobile_control=mobile, reference=reference,
            mobile_continuous=False, mobile_policy="locked_upper_mode0", mobile_vr=None,
            mobile_follow_source_timestamp=None,
            mobile_preparation_wall=None,
            mobile_preparation=SimpleNamespace(source_sequence=1, device_timestamp_ns=1_000_000_000),
            mobile_entry_smpl=np.zeros((24, 3)),
            quality={"valid": True}, replay=None, pending_input_commands=[], last_buttons={}, previous=None,
            receiver=object(), wall_deadline=0., args=SimpleNamespace(max_lateness=.1),
            named_hand_targets=lambda *_: {"hand": .2}, raw=SimpleNamespace(task=SimpleNamespace(robot=object())),
            obs={}, state={"local_tick50": 0}, deferred=False, capture_start_receipt=None,
            ledger=SimpleNamespace(append=lambda *args, **kwargs: ledger_rows.append((args, kwargs))),
            controller=SimpleNamespace(step=lambda token, hands, obs: commands.append((token.copy(), dict(hands))) or {"token": token.copy()}),
            operator_buttons=lambda *_: [], operator_message="", reference_evidence=None,
            body24_to_smpl=lambda body, *_: (body[:, :3].copy(), np.array([1., 0., 0., 0.]), np.zeros(6)),
            require_neutral_calibration=lambda *_: {"passed": True, "reference_root_tilt_deg": 0.},
            require_standing_warmup=lambda *_: {"passed": True}, warmup_baseline={},
            _commands=commands, _ledger_rows=ledger_rows)
        return namespace

    def frame(self, tick):
        return SimpleNamespace(sequence=tick + 1, device_timestamp_ns=1_000_000_000 + tick * 20_000_000,
            host_received_s=100. + tick / 50, body24x7=np.zeros((24, 7)), raw_payload={"tick": tick},
            buttons={}, inputs=dict(left_axis_x=0., left_axis_y=0., right_axis_x=0., right_axis_y=0.))

    def tick(self, namespace, tick, *, frame=None):
        namespace.update(global_tick=tick, frame=frame or self.frame(tick), state={"local_tick50": tick})
        namespace["data"].time = tick / 50
        execute(command_block(), namespace)
        if namespace["mobile_mode"] in ("navigation", "continuous"):
            namespace["mobile_tick"] += 1

    def toggle_navigation(self, namespace):
        transition = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.If)
                          and ast.unparse(node.test) == "action == 'toggle_navigation'")
        execute([ast.For(target=ast.Name(id="action", ctx=ast.Store()),
                         iter=ast.List(elts=[ast.Constant(value="toggle_navigation")], ctx=ast.Load()),
                         body=transition.body, orelse=[])], namespace)

    def operator_action(self, action, namespace):
        branch = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "action == " + repr(action))
        execute([ast.For(target=ast.Name(id="action", ctx=ast.Store()),
                         iter=ast.List(elts=[ast.Constant(value=action)], ctx=ast.Load()),
                         body=branch.body, orelse=[])], namespace)

    def continuous_preparation(self, namespace):
        branch = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.If)
                      and ast.unparse(node.test) == "mobile_continuous and mobile_mode == 'preparing_continuous'"
                      and isinstance(node.body[0], ast.Try))
        execute([branch], namespace)

    def invalidate(self, namespace):
        node = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.FunctionDef)
                    and node.name == "invalidate_calibration")
        node = copy.deepcopy(node)
        node.body = [ast.Global(names=item.names) if isinstance(item, ast.Nonlocal) else item for item in node.body]
        execute([node], namespace)
        namespace["invalidate_calibration"]()

    def test_continuous_live_arms_and_fingers_use_same_packet_at_50_30hz_without_b(self):
        namespace = self.namespace()
        vr = FakeLiveVR()
        namespace.update(mobile_continuous=True, mobile_mode="continuous", mobile_vr=vr,
            named_hand_targets=lambda _, frame: {"hand": frame.sequence / 100.})
        for tick in range(5):
            sample = self.frame(tick)
            sample.body24x7[20, 0] = tick * .05
            sample.body24x7[21, 1] = -tick * .04
            self.tick(namespace, tick, frame=sample)
        mobile = namespace["mobile_control"]
        self.assertEqual([item[0] for item in mobile.ticks], list(range(5)))
        self.assertEqual(len(vr.calls), 5)
        self.assertEqual([item[0] for item in mobile.vr_calls], [1, 2, 3, 4, 5])
        self.assertEqual(len(namespace["reference"].calls), 3)
        self.assertTrue(all(item[0] == "planner_encode" for item in namespace["reference"].calls))
        self.assertEqual([row[0] for row in namespace["reference"].continuity_calls],
                         [self.frame(tick).device_timestamp_ns for tick in (0, 2, 4)])
        self.assertEqual(len(namespace["_ledger_rows"]), 3)
        for _, row in namespace["_ledger_rows"]:
            evidence = row["reference_evidence"]
            self.assertEqual(evidence["encoder_mode"], 1)
            self.assertEqual(evidence["reference_policy"], "continuous_vr_mode1")
            self.assertEqual(evidence["live_vr"]["source_sequence"], row["source_sequence"])
            self.assertEqual(evidence["live_vr"]["source_device_timestamp_ns"], row["device_timestamp_ns"])
            self.assertEqual(row["named_hand_targets"]["hand"], row["source_sequence"] / 100.)
            self.assertFalse(evidence["upper_body_frozen"])
        encoded = [row[1] for row in namespace["reference"].calls]
        np.testing.assert_array_equal(encoded[0][0, 644:890], encoded[-1][0, 644:890])
        self.assertFalse(np.array_equal(encoded[0][0, 890:896], encoded[-1][0, 890:896]))
        self.assertEqual(namespace["reference"].last_timestamp_ns, self.frame(4).device_timestamp_ns)
        np.testing.assert_array_equal(namespace["reference"].last_body, vr.calls[-1][0])

    def test_continuous_b_has_no_effect_and_physical_b_is_not_bound(self):
        namespace = self.namespace()
        episode = SimpleNamespace(status="recording")
        namespace.update(mobile=True, mobile_continuous=True, mobile_mode="continuous", following=True,
            episode=episode, frame=self.frame(1), command={"current": True},
            mobile_stopped=SimpleNamespace(ready=False))
        old_command = namespace["command"]
        self.toggle_navigation(namespace)
        self.assertEqual(namespace["mobile_mode"], "continuous")
        self.assertIs(namespace["episode"], episode)
        self.assertIs(namespace["command"], old_command)
        self.assertEqual(namespace["mobile_control"].pauses, [])
        self.assertEqual(namespace["reference"].calibrations, [])
        self.assertIn("无需B", namespace["operator_message"])
        buttons = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.FunctionDef)
                       and node.name == "operator_buttons")
        execute([buttons], namespace)
        self.assertEqual(namespace["operator_buttons"]({"B": True}, {}), [])
        namespace["mobile_continuous"] = False
        self.assertEqual(namespace["operator_buttons"]({"B": True}, {}), ["toggle_navigation"])
        namespace["mobile"] = False
        self.assertEqual(namespace["operator_buttons"]({"B": True}, {}), [])

    def test_continuous_follow_prepares_while_paused_then_requires_fresh_neutral_packet(self):
        namespace = self.namespace()
        namespace.update(mobile_continuous=True, mobile_mode="continuous_paused", following=False,
            mobile_vr=FakeLiveVR(), calibration_fresh=True, physics_ready=True, frame=self.frame(0),
            command={"old": True}, wall_deadline=100., mobile_stopped=SimpleNamespace(clear=lambda: None))
        namespace["mobile_control"].ready = False
        self.operator_action("follow", namespace)
        self.assertFalse(namespace["following"])
        self.assertEqual(namespace["mobile_mode"], "preparing_continuous")
        self.assertEqual(len(namespace["mobile_control"].boots), 1)
        self.assertIsNone(namespace["wall_deadline"])
        self.assertIsNone(namespace["command"])
        self.continuous_preparation(namespace)
        self.assertFalse(namespace["following"])
        self.assertEqual(namespace["mobile_control"].ticks, [])
        namespace["mobile_control"].ready = True
        self.continuous_preparation(namespace)  # old packet is not enough
        self.assertFalse(namespace["following"])
        namespace["frame"] = self.frame(1)
        namespace["frame"].inputs["left_axis_y"] = .8
        self.continuous_preparation(namespace)
        self.assertFalse(namespace["following"])
        namespace["frame"] = self.frame(2)
        self.continuous_preparation(namespace)
        self.assertTrue(namespace["following"])
        self.assertEqual(namespace["mobile_mode"], "continuous")
        self.assertEqual(namespace["mobile_tick"], 0)
        self.assertIsNone(namespace["wall_deadline"])
        self.assertEqual(namespace["data"].time, 0.)
        self.assertEqual(namespace["mobile_control"].ticks, [])

    def test_continuous_pause_invalidates_vr_and_reference_and_next_follow_must_rebind(self):
        namespace = self.namespace()
        namespace.update(mobile_continuous=True, mobile_mode="continuous", following=True,
            mobile_vr=FakeLiveVR(), command={"active": True}, calibration_fresh=True,
            mobile_follow_source_timestamp=1000, mobile_preparation_wall=1.,
            mobile_stopped=SimpleNamespace(clear=lambda: None))
        self.invalidate(namespace)
        self.assertFalse(namespace["following"])
        self.assertEqual(namespace["mobile_mode"], "continuous_paused")
        self.assertIsNone(namespace["mobile_vr"].calibration)
        self.assertIsNone(namespace["reference"].calibration)
        self.assertIsNone(namespace["mobile_follow_source_timestamp"])
        self.assertIsNone(namespace["mobile_preparation_wall"])
        self.assertIsNone(namespace["command"])
        self.assertFalse(namespace["calibration_fresh"])
        namespace.update(frame=self.frame(1), physics_ready=True)
        with self.assertRaisesRegex(ValueError, "Calibrate valid tracking"):
            self.operator_action("follow", namespace)

    def test_continuous_metadata_keeps_mode1_policy_and_vr_identity_separate_from_legacy(self):
        from test_pico_simple_teleop import episode_metadata_fixture
        session, scene = episode_metadata_fixture()
        session.update(task_id="simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0",
            collection_task_profile="mobile_pp", mobile_control_policy="continuous_vr_mode1",
            mobile_encoder_mode=1, mobile_vr_source={"source_sha256": {"vr_math": "a" * 64}},
            mobile_control_protocol="cvpr-pico-mobile-continuous-vr-mode1-async-reference-v1")
        identity = entry.episode_session_identity(session)
        metadata = entry.bounded_episode_metadata(identity, scene, human_demo_uuid="cpu-test-uuid",
            pico_calibration={"encoder_mode": 1, "mobile_vr": {"calibration_sha256": "b" * 64}},
            initial_target_z_m=.55)
        self.assertEqual(metadata["mobile_control_policy"], "continuous_vr_mode1")
        self.assertEqual(metadata["mobile_encoder_mode"], 1)
        self.assertEqual(metadata["pico_calibration"]["encoder_mode"], 1)
        self.assertEqual(metadata["mobile_vr_source"], session["mobile_vr_source"])
        self.assertEqual(metadata["mobile_control_protocol"], session["mobile_control_protocol"])
        self.assertFalse(metadata["accepted_new_dataset"])
        session["mobile_vr_source"]["source_sha256"].clear()
        self.assertEqual(metadata["mobile_vr_source"]["source_sha256"], {"vr_math": "a" * 64})

    def test_continuous_x_calibrates_vr_only_and_saves_no_unused_mode2_wrist_receipt(self):
        namespace = self.namespace()
        namespace.update(mobile_continuous=True, mobile_policy="continuous_vr_mode1", mobile_mode="continuous_paused",
            mobile_vr=FakeLiveVR(), frame=self.frame(0), physics_ready=True, following=False,
            mobile_actual_vr_bodies=lambda: (np.zeros((3, 3)), np.tile([1., 0., 0., 0.], (3, 1))))
        namespace["reference"].last_timestamp_ns = 999
        namespace["reference"].last_body = np.ones((24, 7))
        with tempfile.TemporaryDirectory() as directory:
            namespace["output"] = Path(directory)
            self.operator_action("calibrate", namespace)
            saved = json.loads((Path(directory) / "last_calibration.json").read_text())
        self.assertEqual(namespace["reference"].calibrations, [])
        self.assertEqual(saved["encoder_mode"], 1)
        self.assertEqual(saved["mobile_vr"]["calibration_sha256"], "b" * 64)
        self.assertIn("human_readiness", saved)
        self.assertIn("robot_readiness", saved)
        self.assertFalse(set(saved) & {"wrist_offset6", "initial_smpl_joints24_sha256", "wrist_source"})
        self.assertIsNone(namespace["reference"].last_body)
        self.assertIsNone(namespace["reference"].last_timestamp_ns)
        self.assertTrue(namespace["calibration_fresh"])
        self.assertFalse(namespace["following"])
        self.assertEqual(namespace["mobile_mode"], "continuous_paused")

    def test_legacy_and_handover_x_keep_existing_mode2_calibrator(self):
        for mobile in (False, True):
            namespace = self.namespace()
            namespace.update(mobile=mobile, mobile_continuous=False, frame=self.frame(0),
                physics_ready=True, following=False)
            with tempfile.TemporaryDirectory() as directory:
                namespace["output"] = Path(directory)
                self.operator_action("calibrate", namespace)
            self.assertEqual(len(namespace["reference"].calibrations), 1)
            self.assertTrue(namespace["calibration_fresh"])

    def test_actual_dispatch_calls_planner_50hz_and_token_ledger_only_30hz(self):
        namespace = self.namespace()
        for tick in range(5): self.tick(namespace, tick)
        self.assertEqual([row[0] for row in namespace["mobile_control"].ticks], list(range(5)))
        self.assertEqual(len(namespace["reference"].calls), 3)
        self.assertEqual(len(namespace["_commands"]), 3)
        self.assertEqual([kwargs["physical_tick50"] for _, kwargs in namespace["_ledger_rows"]], [0, 2, 4])
        self.assertEqual([kwargs["device_timestamp_ns"] for _, kwargs in namespace["_ledger_rows"]],
                         [self.frame(tick).device_timestamp_ns for tick in (0, 2, 4)])

    def test_due_command_cannot_reuse_old_pico_timestamp(self):
        namespace = self.namespace()
        self.tick(namespace, 0)
        fresh = self.frame(2)
        waits = []
        def wait(receiver, previous, timestamp, deadline):
            waits.append(timestamp)
            return fresh, {"valid": True}
        namespace["wait_fresh_command_frame"] = wait
        self.tick(namespace, 2, frame=self.frame(0))
        self.assertEqual(waits, [self.frame(0).device_timestamp_ns])
        self.assertEqual(namespace["_ledger_rows"][-1][1]["device_timestamp_ns"], fresh.device_timestamp_ns)

    def test_mode_return_calibrates_current_world_heading_and_resets_pose_clock(self):
        namespace = self.namespace()
        self.tick(namespace, 0)
        namespace.update(mobile=True, following=True, episode=None, mobile_stopped=SimpleNamespace(ready=True),
            actual_root_velocity=lambda *_: [0.] * 6, frame=self.frame(1), control_trace=io.StringIO())
        self.toggle_navigation(namespace)
        self.assertEqual(namespace["mobile_mode"], "pose")
        self.assertEqual(namespace["mobile_control"].pauses, ["return_to_manipulation"])
        self.assertEqual(namespace["reference"].calibrations[-1][1].tolist(), namespace["root_pose"]().tolist())
        self.assertIsNone(namespace["reference"].last_timestamp_ns)
        self.assertIsNone(namespace["mobile_entry_smpl"])
        self.tick(namespace, 2)
        self.assertEqual(namespace["reference"].calls[-1][0:2], ("pose_encode", self.frame(2).device_timestamp_ns))
        self.assertEqual(len(namespace["mobile_control"].ticks), 1)

    def test_changed_human_arm_blocks_return_without_pausing_or_calibrating_recording(self):
        namespace = self.namespace()
        episode = SimpleNamespace(status="recording")
        namespace.update(mobile=True, following=True, episode=episode, mobile_stopped=SimpleNamespace(ready=True),
            actual_root_velocity=lambda *_: [0.] * 6, frame=self.frame(1), control_trace=io.StringIO(),
            command={"keep_current_navigation": True})
        original_command = namespace["command"]
        namespace["frame"].body24x7[18, 0] = .2
        self.toggle_navigation(namespace)
        self.assertEqual(namespace["mobile_mode"], "navigation")
        self.assertTrue(namespace["following"])
        self.assertIs(namespace["episode"], episode)
        self.assertIs(namespace["command"], original_command)
        self.assertEqual(namespace["mobile_control"].pauses, [])
        self.assertEqual(namespace["reference"].calibrations, [])
        self.assertIsNotNone(namespace["mobile_entry_smpl"])
        self.assertIn("当前录制继续", namespace["operator_message"])
        namespace["frame"] = self.frame(2)  # restore the entry arm/body posture
        self.toggle_navigation(namespace)
        self.assertEqual(namespace["mobile_mode"], "pose")
        self.assertEqual(namespace["mobile_control"].pauses, ["return_to_manipulation"])
        self.assertEqual(len(namespace["reference"].calibrations), 1)
        self.assertIsNone(namespace["mobile_entry_smpl"])
        self.assertIs(namespace["episode"], episode)

    def test_b_entry_caches_measured_human_pose_as_an_independent_copy(self):
        namespace = self.namespace()
        namespace.update(mobile=True, following=True, episode=SimpleNamespace(status="recording"),
            mobile_mode="pose", mobile_stopped=SimpleNamespace(ready=True),
            actual_root_velocity=lambda *_: [0.] * 6, frame=self.frame(1), control_trace=io.StringIO())
        namespace["frame"].body24x7[18, 0] = .1
        self.toggle_navigation(namespace)
        expected = namespace["frame"].body24x7[:, :3].copy()
        self.assertEqual(namespace["mobile_mode"], "preparing_navigation")
        self.assertEqual(len(namespace["mobile_control"].boots), 1)
        np.testing.assert_allclose(namespace["mobile_entry_smpl"], expected, atol=1e-7)
        namespace["frame"].body24x7[18, 0] = .4
        np.testing.assert_allclose(namespace["mobile_entry_smpl"], expected, atol=1e-7)

    def test_tilted_human_return_is_ignored_without_pausing_navigation_or_recording(self):
        namespace = self.namespace()
        episode = SimpleNamespace(status="recording")
        namespace.update(mobile=True, following=True, episode=episode, mobile_stopped=SimpleNamespace(ready=True),
            actual_root_velocity=lambda *_: [0.] * 6, frame=self.frame(1), control_trace=io.StringIO())
        def reject_tilt(*_):
            raise ValueError("人体参考倾角 humanTilt=20.0°，要求≤15°")
        namespace["require_neutral_calibration"] = reject_tilt
        self.toggle_navigation(namespace)
        self.assertEqual(namespace["mobile_mode"], "navigation")
        self.assertTrue(namespace["following"])
        self.assertIs(namespace["episode"], episode)
        self.assertEqual(namespace["mobile_control"].pauses, [])
        self.assertEqual(namespace["reference"].calibrations, [])
        self.assertIn("humanTilt", namespace["operator_message"])
        self.assertIn("当前录制继续", namespace["operator_message"])

    def test_delayed_planner_preparation_holds_measured_entry_not_moving_human(self):
        from pico_mobile_task import MobilePreparationReference
        from prepare_carrybox_v11 import POLICY_TO_BODY
        namespace = self.namespace()
        entry_body = np.linspace(-.2, .2, 29)
        namespace.update(mobile_mode="preparing_navigation", mobile_preparation=MobilePreparationReference(
            namespace["root_pose"](), entry_body, source_sequence=4, device_timestamp_ns=1234))
        namespace["mobile_control"].ready = False
        for tick in range(3):
            frame = self.frame(tick)
            frame.body24x7[:, :3] += tick * 2.
            self.tick(namespace, tick, frame=frame)
        self.assertEqual(namespace["mobile_mode"], "preparing_navigation")
        self.assertEqual(namespace["mobile_control"].ticks, [])
        self.assertEqual(len(namespace["reference"].calls), 2)
        self.assertTrue(all(call[0] == "planner_encode" for call in namespace["reference"].calls))
        for _, value in namespace["reference"].calls:
            self.assertEqual(value[0, 0], 0.)
            np.testing.assert_allclose(value[0, 4:294].reshape(10, 29),
                                      np.tile(entry_body[POLICY_TO_BODY], (10, 1)), atol=1e-7)
        for _, item in namespace["_ledger_rows"]:
            evidence = item["reference_evidence"]
            self.assertFalse(evidence["current_human_pose_used"])
            self.assertTrue(evidence["static_desired_reference"])
            self.assertFalse(evidence["complete46_future"])
            self.assertEqual(evidence["pose_source_device_timestamp_ns"], 1234)
        namespace["mobile_control"].ready = True
        self.tick(namespace, 3)  # readiness alone cannot add an off-grid token
        self.assertEqual(namespace["mobile_mode"], "preparing_navigation")
        self.tick(namespace, 4)  # actual 30 Hz grid tick: activate with a new token
        self.assertEqual(namespace["mobile_mode"], "navigation")
        self.assertEqual([row[0] for row in namespace["mobile_control"].ticks], [0])

    def test_ready_planner_waits_for_neutral_sticks_without_interrupting_capture(self):
        from pico_mobile_task import MobilePreparationReference
        namespace=self.namespace()
        namespace.update(mobile_mode='preparing_navigation',mobile_preparation=MobilePreparationReference(
            namespace['root_pose'](),np.zeros(29),source_sequence=1,device_timestamp_ns=1234))
        pushed=self.frame(0);pushed.inputs['left_axis_y']=.8
        self.tick(namespace,0,frame=pushed)
        self.assertEqual(namespace['mobile_mode'],'preparing_navigation')
        self.assertEqual(namespace['mobile_control'].ticks,[])
        self.assertIn('松开',namespace['operator_message'])
        self.assertEqual(len(namespace['_ledger_rows']),1)
        self.tick(namespace,2)
        self.assertEqual(namespace['mobile_mode'],'navigation')
        self.assertEqual([item[0] for item in namespace['mobile_control'].ticks],[0])

    def test_reference_outside_soft_range_reaches_encoder_unchanged_with_evidence(self):
        namespace=self.namespace();value=np.zeros((1,1751),dtype=np.float32)
        value[0,4+29]=10.0015
        original=value.copy()
        namespace['mobile_control'].step=lambda *args,**kwargs:(value,{'encoder_mode':0})
        self.tick(namespace,0)
        self.assertEqual(len(namespace['_commands']),1)
        np.testing.assert_array_equal(namespace['reference'].calls[0][1],original)
        report=namespace['_ledger_rows'][0][1]['reference_evidence']['reference_range_diagnostics']
        self.assertFalse(report['hard_reject_on_reference_range'])
        self.assertFalse(report['reference_modified'])
        self.assertEqual(report['outside_range_reference_count'],1)
        value[0,15]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'):
            self.tick(namespace,2)
        self.assertEqual(len(namespace['_commands']),1)

    def test_invalid_tracking_runs_real_invalidation_before_follow_control(self):
        namespace = self.namespace()
        self.tick(namespace, 0)
        namespace.update(following=True, quality={"valid": False, "reasons": ["stale"]},
            calibration_fresh=True, mobile_preparation=object(),
            mobile_stopped=SimpleNamespace(clear=lambda: None))
        invalidate = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.FunctionDef)
                          and node.name == "invalidate_calibration")
        invalidate = copy.deepcopy(invalidate)
        invalidate.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node
                           for node in invalidate.body]
        execute([invalidate], namespace)
        rejected = []
        def reject(reason):
            rejected.append(reason)
            namespace["invalidate_calibration"]()
        namespace["reject_and_pause"] = reject
        statements, index = statement_list_containing(lambda node: isinstance(node, ast.If)
            and ast.unparse(node.test) == "following and (not quality['valid'])")
        self.assertEqual(ast.unparse(statements[index + 1].test), "following")
        following = copy.deepcopy(statements[index + 1])
        following.body = command_block()
        following.orelse = []
        execute([statements[index], following], namespace)
        self.assertEqual(rejected, ["Tracking rejected: stale"])
        self.assertFalse(namespace["following"])
        self.assertIsNone(namespace["command"])
        self.assertIsNone(namespace["last_command_timestamp_ns"])
        self.assertIsNone(namespace["mobile_preparation"])
        self.assertIsNone(namespace["mobile_entry_smpl"])
        self.assertEqual(namespace["mobile_mode"], "pose")
        self.assertEqual(namespace["mobile_control"].pauses, ["collector_paused"])
        self.assertEqual(len(namespace["mobile_control"].ticks), 1)
        self.assertEqual(len(namespace["_commands"]), 1)

    def test_native_result_binds_same_solved_row_after_env_step_including_last_success(self):
        namespace = self.namespace()
        data = namespace["data"]
        buffer = io.StringIO()
        sample_rows = []
        def original_step(): data.time += .005
        def measured_sample(*args, **kwargs):
            row = {"completed_steps": kwargs["completed_steps"], "sim_time_s": data.time,
                   "contacts": {"left_hand": False, "right_hand": False, "container": True},
                   "original_live_support": {"container_support_weight_fraction": .75}}
            sample_rows.append(copy.deepcopy(row))
            return row
        namespace.update(original_step=original_step, substeps=0, mobile=True, active_local=0,
            pending_mobile_quality_row=None, quality_stream=buffer, force_stream=None,
            sample_original_solved_quality=measured_sample, applied_force_state=lambda _: {"all_zero": True},
            model=object(), raw=SimpleNamespace(_success=False, mujoco=SimpleNamespace(mj_objects={
                "target": SimpleNamespace(id=1), "container": SimpleNamespace(id=2)})))
        measured = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.FunctionDef) and node.name == "measured_step")
        measured = copy.deepcopy(measured)
        measured.body = [ast.Global(names=node.names) if isinstance(node, ast.Nonlocal) else node for node in measured.body]
        execute([measured], namespace)
        native_results = [False, True]
        def step(command):
            for _ in range(4): namespace["measured_step"]()
            self.assertEqual(len(buffer.getvalue().splitlines()), 2 - len(native_results))
            self.assertNotIn("official_task_success", namespace["pending_mobile_quality_row"])
            result = native_results.pop(0)
            namespace["raw"]._success = result
            return {}, 0., result, False, {}
        namespace["env"] = SimpleNamespace(step=step)
        for index in range(2):
            namespace["active_local"] = index
            execute(step_and_native_block(), namespace)
        recorded = [json.loads(line) for line in buffer.getvalue().splitlines()]
        self.assertEqual([row["official_task_success"] for row in recorded], [False, True])
        self.assertEqual([row["completed_steps"] for row in recorded], [1, 2])
        self.assertAlmostEqual(recorded[-1]["sim_time_s"], .04)
        self.assertEqual(recorded[-1]["contacts"], sample_rows[-1]["contacts"])
        self.assertEqual(recorded[-1]["original_live_support"], sample_rows[-1]["original_live_support"])
        self.assertIsNone(namespace["pending_mobile_quality_row"])


if __name__ == "__main__":
    unittest.main()
