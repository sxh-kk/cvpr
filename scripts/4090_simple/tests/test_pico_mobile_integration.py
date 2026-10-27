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

class FakeMobile:
    def __init__(self):
        self.ticks = []
        self.pauses = []
        self.boots = []
        self.ready = True

    def poll(self):
        return {"ready": self.ready}

    def step(self, tick, root, body, frame, *, tracking_valid):
        self.ticks.append((tick, frame.device_timestamp_ns, tracking_valid))
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
        if namespace["mobile_mode"] == "navigation":
            namespace["mobile_tick"] += 1

    def toggle_navigation(self, namespace):
        transition = next(node for node in ast.walk(run_tree()) if isinstance(node, ast.If)
                          and ast.unparse(node.test) == "action == 'toggle_navigation'")
        execute([ast.For(target=ast.Name(id="action", ctx=ast.Store()),
                         iter=ast.List(elts=[ast.Constant(value="toggle_navigation")], ctx=ast.Load()),
                         body=transition.body, orelse=[])], namespace)

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
