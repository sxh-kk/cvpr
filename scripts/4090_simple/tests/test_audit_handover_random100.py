"""CPU integrity regressions for new generated-command admission."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import audit_handover_random100 as auditor
import replay_handover_teacher as base
from prepare_carrybox_v11 import ENCODER_SHA


def ledger():
    total = 55; times = .005 + np.arange(total + 1) / 50
    names = [f"body{i}" for i in range(29)] + list(base.MODEL_HAND_NAMES)
    root = np.tile([0., 0., .75, 1., 0., 0., 0.], (total + 1, 1))
    objects = {role: np.tile([x, 0., .6, 1., 0., 0., 0.], (total + 1, 1))
               for role, x in (("target", 0.), ("container", .4))}
    history = {"times": times, "names": names, "q": np.zeros((total + 1, 43)), "root": root, "objects": objects}
    commands = [{"command_index30": i, "teacher_tick50": (i * 50 + 29) // 30,
        "sim_time_s": times[50 + (i * 50 + 29) // 30], "body_token64": [float(i)] * 64,
        "hand14": [float(i)] * 14, "hand_joint_names": list(base.HAND_NAMES)} for i in range(3)]
    warmup = {"kind": "live_initial_pose_sonic_v1_1", "encoder_calls": 40, "events": [],
        "encoder_sha256": ENCODER_SHA, "hand_policy": "measured", "initialization_count": 1,
        "helper_source_sha256": auditor.sha(auditor.REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/live_pose_sonic_warmup.py"),
        "encoder_mode": 0,
        "first_sim_time_s": times[0], "reference_body_q29": [0.] * 29, "reference_root_wxyz": [1., 0., 0., 0.],
        "reference_hands": {name: 0. for name in base.HAND_NAMES}, "state_forcing": False, "history_injection": False}
    trace = []; decoder = 0
    for tick in range(total):
        ordinal = (tick - 50) * 30 // 50 if tick >= 50 else None
        token = None if tick < 10 else ([.1] * 64 if tick < 50 else commands[ordinal]["body_token64"])
        hand = [0.] * 14 if tick < 50 else commands[ordinal]["hand14"]
        if 10 <= tick < 50:
            warmup["events"].append({"sim_time_s": times[tick], "token64": token,
                "actual_root_wxyz": [1., 0., 0., 0.], "state_forcing": False, "history_injection": False})
        increment = int(tick >= 10)
        trace.append({"tick50": tick, "physics_state_row_before": tick, "physics_state_row_after": tick + 1,
            "sim_time_s": times[tick], "next_sim_time_s": times[tick + 1], "elapsed_s": tick / 50,
            "finite_physics": True, "phase": ("warmup_measured_pose_pd" if tick < 10 else
                "warmup_live_pose_sonic_v1_1" if tick < 50 else "online_object_feedback_expert"),
            "actual_body_token64": token, "actual_hand_targets": dict(zip(base.HAND_NAMES, hand)),
            "expert_command_index30": ordinal, "decoder_calls_before": decoder, "decoder_calls_after": decoder + increment,
            "measured_before": {role + "_pose7": objects[role][tick].tolist() for role in objects},
            "measured_after": {role + "_pose7": objects[role][tick + 1].tolist() for role in objects}})
        decoder += increment
    result = {"completed_control_ticks": total, "warmup_control": warmup, "decoder_calls": decoder}
    return commands, trace, history, result


class Random100AuditTests(unittest.TestCase):
    def test_placement_encoder_mode_is_explicit_with_legacy_g1_default(self):
        self.assertEqual(auditor.placement_encoder_mode({}),"g1")
        self.assertEqual(auditor.placement_encoder_mode({"placement_encoder_mode":"teleop"}),"teleop")
        with self.assertRaises(ValueError):auditor.placement_encoder_mode({"placement_encoder_mode":"bad"})
        self.assertEqual(auditor.placement_locomotion({}),"source_future")
        self.assertEqual(auditor.placement_locomotion({"placement_locomotion":"actual_hold"}),"actual_hold")
        with self.assertRaises(ValueError):auditor.placement_locomotion({"placement_locomotion":"bad"})

    def test_mode1_helper_and_all_four_official_sources_are_required_without_onnx(self):
        import handover_sonic_teleop_reference as helper
        bindings={str(Path(helper.__file__).resolve()):auditor.sha(helper.__file__),
                  **helper.verify_official_sources()["source_sha256"]}
        with patch.dict(sys.modules,{"onnx":None}):
            self.assertEqual(auditor.projection_bindings({}),{})
            self.assertEqual(auditor.placement_encoder_bindings({"placement_encoder_mode":"teleop"},bindings),bindings)
        for missing in bindings:
            changed=dict(bindings);del changed[missing]
            with self.subTest(missing=missing),self.assertRaisesRegex(ValueError,"all four"):
                auditor.placement_encoder_bindings({"placement_encoder_mode":"teleop"},changed)
        changed=dict(bindings);changed[next(iter(bindings))]="0"*64
        with self.assertRaises(ValueError):auditor.placement_encoder_bindings({"placement_encoder_mode":"teleop"},changed)
        self.assertEqual(auditor.placement_encoder_bindings({"placement_encoder_mode":"g1"},{}),{})

    def test_independent_replay_constructs_same_placement_encoder_config(self):
        for args,expected in (({},"g1"),({"placement_encoder_mode":"teleop"},"teleop"),
                              ({"placement_locomotion":"actual_hold"},"g1")):
            with patch("handover_random100_expert.ExpertConfig") as config,\
                 patch("handover_random100_expert.HandoverObjectFeedbackExpert") as expert:
                result=auditor.make_replay_expert({"values":{"fixture":True}},{"sonic_resources":"/sonic",**args})
                config.assert_called_once_with(placement_encoder_mode=expected,
                    placement_locomotion=args.get("placement_locomotion","source_future"))
                expert.assert_called_once_with({"fixture":True},Path('/sonic'),config=config.return_value)
                self.assertIs(result,expert.return_value)

    def test_new_generated30hz_commands_hold_on_every_real50hz_tick(self):
        auditor.validate_command_ledger(*ledger())

    def test_wrong_hold_or_future_command_cannot_be_an_actual_label(self):
        for modify in (lambda c, t, h, r: t[51].update(actual_body_token64=c[1]["body_token64"]),
                       lambda c, t, h, r: c[1].update(teacher_tick50=1),
                       lambda c, t, h, r: t[52].update(expert_command_index30=0),
                       lambda c, t, h, r: t[52].update(sim_time_s=999.),
                       lambda c, t, h, r: t[50]["actual_hand_targets"].update({base.HAND_NAMES[0]: 3.}),
                       lambda c, t, h, r: t[20].update(decoder_calls_before=0),
                       lambda c, t, h, r: r["warmup_control"].update(hand_policy="open")):
            values = ledger(); modify(*values)
            with self.assertRaises(ValueError): auditor.validate_command_ledger(*values)

    def test_raw_warmup_and_teacher_hand_names_cannot_be_fabricated(self):
        values = ledger(); values[1][0]["actual_body_token64"] = [0.] * 64
        with self.assertRaisesRegex(ValueError, "First ten"):
            auditor.validate_command_ledger(*values)
        values = ledger(); values[0][0]["hand_joint_names"].reverse()
        with self.assertRaisesRegex(ValueError, "hand order"):
            auditor.validate_command_ledger(*values)

    def test_auxiliary_pose_cannot_hide_a_different_actual_object(self):
        values = ledger(); values[1][50]["measured_after"]["target_pose7"][0] += .05
        with self.assertRaisesRegex(ValueError, "integration authority"):
            auditor.validate_command_ledger(*values)

    def test_missing_or_producer_only_completion_never_admits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = auditor.audit(root)
            self.assertFalse(report["eligible_dataset"])
            self.assertEqual(report["status"], "independent_capture_rejected")
            self.assertEqual(report["physics_steps_performed"], 0)
            self.assertFalse(report["GPU_used"])

    def test_projection_replay_prefix_is_every50hz_including_warmup(self):
        commands, trace, history, result = ledger()
        class Controller:
            decoder_calls = 0
            def __init__(self): self.history = []
            def apply_physics(self, robot, token, hands):
                self.history.append(token)
                self.decoder_calls += int(token is not None)
        controller = Controller(); prepared = []; observed_prefixes = []
        auditor.replay_native_controls(trace, controller, object(), prepared.append,
            lambda tick, ordinal: observed_prefixes.append((tick, ordinal, len(controller.history))))
        self.assertEqual(prepared, list(range(55)))
        self.assertEqual(observed_prefixes, [(50, 0, 50), (52, 1, 52), (54, 2, 54)])
        self.assertEqual(controller.decoder_calls, result["decoder_calls"])

    def test_projection_comparison_ignores_only_wallclock_timing(self):
        first = {"CPU_seconds": .2, "applied": True, "targets": [1., 2.], "actor_sha": "actual"}
        self.assertEqual(auditor.stable_projection_diagnostic(first),
                         auditor.stable_projection_diagnostic({**first, "CPU_seconds": 3.}))
        self.assertNotEqual(auditor.stable_projection_diagnostic(first),
                            auditor.stable_projection_diagnostic({**first, "actor_sha": "wrong"}))
        self.assertEqual(auditor.projection_bindings({"latent_projector_batch_view": None}), {})

    def test_projection_phase_filter_is_explicit_and_legacy_missing_means_all(self):
        class Projector:
            def __init__(self): self.calls = 0
            def project(self, token, desired, raw, controller):
                self.calls += 1
                return token + 1, {"applied": True}
        projector = Projector(); token = np.zeros(64)
        actual, diagnostic = auditor.replay_projection_step({"latent_projection_phases": "placement,release"},
            token, None, "approach", None, None, projector)
        self.assertTrue(np.array_equal(actual, token)); self.assertEqual(projector.calls, 0)
        self.assertEqual(diagnostic, {"state": "skipped_phase_filter", "phase": "approach",
                                     "enabled_phases": ["placement", "release"]})
        auditor.replay_projection_step({"latent_projection_phases": "placement"}, token, None, "placement", None, None, projector)
        auditor.replay_projection_step({}, token, None, "approach", None, None, projector)
        self.assertEqual(projector.calls, 2)
        for invalid in ("", "all", " placement", None):
            with self.assertRaises(ValueError): auditor.projection_phases({"latent_projection_phases": invalid})

    def test_private_saved_prestate_rebuilds_real_foot_contact_geometry_without_step(self):
        import mujoco
        from handover_random100_expert import foot_contact_support_polygon
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
          <geom type="plane" size="2 2 .1"/>
          <body name="pelvis" pos="0 0 .02"><freejoint/>
            <body name="left_ankle_roll_link" pos="0 .1 0"><geom type="box" size=".1 .04 .02"/></body>
            <body name="right_ankle_roll_link" pos="0 -.1 0"><geom type="box" size=".1 .04 .02"/></body>
          </body></worldbody></mujoco>''')
        live = mujoco.MjData(model); live.time = .025
        mujoco.mj_forward(model, live)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, live, state, spec)
        private = mujoco.MjData(model)
        self.assertEqual(private.ncon, 0)
        history = {"model": model, "states": [state], "spec": spec}
        auditor.restore_private_expert_state(history, private, 0)
        self.assertGreater(private.ncon, 0)
        self.assertTrue(np.array_equal(foot_contact_support_polygon(model, live), foot_contact_support_polygon(model, private)))
        self.assertEqual(private.time, .025)
        self.assertEqual(live.time, .025)

    def test_final50_actual_mesh_vertices_reject_aabb_shortcut_and_one_bad_state(self):
        import mujoco
        model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
          <mesh name="triangle" vertex="-.2 -.2 -.02 .2 -.2 -.02 -.2 .2 -.02 -.2 -.2 .02 .2 -.2 .02 -.2 .2 .02"/>
          </asset><worldbody>
          <body name="basket"><freejoint/><geom type="mesh" mesh="triangle"/></body>
          <body name="target" pos="-.08 -.08 .1"><freejoint/><geom type="box" size=".02 .02 .02"/></body>
          </worldbody></mujoco>''')
        data = mujoco.MjData(model); spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        states = []
        for row in range(101):
            data.time = .005 + row / 50
            state = np.empty(mujoco.mj_stateSize(model, spec)); mujoco.mj_getState(model, data, state, spec); states.append(state)
        history = {"model": model, "states": states, "spec": spec,
                   "roles": {role: model.body(name).id for role, name in (("target", "target"), ("container", "basket"))}}
        passed = auditor.full_outer_footprint_final50(history)
        self.assertTrue(passed["passed"]); self.assertEqual(passed["checked_states"], 50)
        self.assertFalse(passed["AABB_or_source_bbox_used_as_vertices"])
        # This center and its full small-box AABB lie inside the container's
        # bounding box, but outside its actual triangular outer collision hull.
        data.joint(1).qpos[:2] = [.08, .08]
        mujoco.mj_getState(model, data, states[-1], spec)
        bad = auditor.full_outer_footprint_final50(history)
        self.assertFalse(bad["passed"]); self.assertLess(bad["minimum_outer_hull_margin_m"], 0.)
        self.assertEqual(sum(row["passed"] for row in bad["measurements"]), 49)
        # Being inside the outer hull with only3mm clearance also fails the
        # explicit5mm supplemental quality threshold.
        data.joint(1).qpos[:2] = [-.177, -.08]
        mujoco.mj_getState(model, data, states[-1], spec)
        edge = auditor.full_outer_footprint_final50(history)
        self.assertTrue(edge["all_states_all_vertices_inside_outer_hull"])
        self.assertFalse(edge["passed"])
        self.assertAlmostEqual(edge["minimum_outer_hull_margin_m"], .003, places=6)


if __name__ == "__main__": unittest.main()
