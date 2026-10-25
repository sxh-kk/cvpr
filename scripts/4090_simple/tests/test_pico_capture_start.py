"""CPU-only capture-start evidence checks, without SDK/model/physics calls."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_capture_start import CaptureStartGate
from pico_live_input import parse_tracking, validate_frame


UUID = "5ea9892a-84dc-4c46-99e0-777482a2bf81"


def packet(timestamp_ns, received, sequence, *, mode=1, focus=True):
    value = {"timeStampNs": timestamp_ns, "Input": mode,
        "appState": {"focus": focus}, "Head": {"status": 3, "pose": [0, 1.7, .1, 0, 0, 0, 1]},
        "Controller": {"left": {"pose": [-.3, 1.1, .2, 0, 0, 0, 1]},
                       "right": {"pose": [.3, 1.1, .2, 0, 0, 0, 1]}},
        "Body": {"joints": [{"p": [i*.01, 1-i*.02, .1, 0, 0, 0, 1]} for i in range(24)]}}
    return parse_tracking({"functionName": "Tracking", "value": json.dumps(value)}, received, sequence)


class CaptureStartTests(unittest.TestCase):
    def setup_case(self, *, elapsed=.135, device_gap=.135, initial_age=0.):
        self.clock = SimpleNamespace(value=1000.)
        self.before = packet(1_000_000_000, self.clock.value-initial_age, 10)
        self.state = np.arange(12, dtype=np.float64)
        gate = CaptureStartGate(UUID, 50, 2., 400, self.state, self.before,
            clock=lambda: self.clock.value)
        self.clock.value += elapsed
        self.after = packet(1_000_000_000+round(device_gap*1e9), self.clock.value, 17)
        args = dict(old_deadline=999.92, after_global_tick=50, after_sim_time=2.,
            after_substeps=400, after_state=self.state.copy(), frame=self.after,
            quality=validate_frame(self.after, now=self.clock.value, previous=self.before),
            event_count=0, ledger_event_count=1, quality_bytes=0, force_bytes=0)
        return gate, args

    def test_135ms_setup_preserves_original_80ms_lag(self):
        gate, args = self.setup_case()
        shifted, receipt = gate.finish(**args)
        self.assertAlmostEqual(shifted, 1000.055)
        self.assertAlmostEqual(receipt["setup_wall_s"], .135)
        self.assertAlmostEqual(receipt["old_lateness_s"], .08)
        self.assertAlmostEqual(self.clock.value-shifted, .08)
        self.assertAlmostEqual(receipt["preserved_lateness_s"], .08)
        self.assertTrue(receipt["frozen_before_first_capture"])
        self.assertEqual(receipt["source_sequence_delta"], 7)
        self.assertFalse(receipt["all_intermediate_sdk_packets_inspected"])
        self.assertEqual(receipt["integration_state_sha256_before"], receipt["integration_state_sha256_after"])
        self.assertFalse(receipt["accepted_new_dataset"])
        json.dumps(receipt, allow_nan=False)
        initial = receipt["source_before"]
        final = receipt["source_after"]
        checked = validate_frame(
            parse_tracking(final["raw_packet"], final["host_received_s"], final["sequence"]),
            now=receipt["setup_finished_monotonic_s"],
            previous=parse_tracking(initial["raw_packet"], initial["host_received_s"], initial["sequence"]))
        self.assertEqual(checked, receipt["source_endpoint_quality"])
        # Work after finish is charged normally, never excluded a second time.
        self.clock.value += .03
        self.assertAlmostEqual(self.clock.value-shifted, .11)

    def test_setup_never_erases_already_excessive_lag(self):
        gate, args = self.setup_case()
        args["old_deadline"] = 999.89
        with self.assertRaisesRegex(ValueError, "Existing lateness"):
            gate.finish(**args)

    def test_accepts_real_validator_numpy_booleans_but_not_numeric_truth_values(self):
        gate,args=self.setup_case()
        args['quality']={**args['quality'],**{key:np.bool_(True) for key in ('valid','fresh','recordable')}}
        gate.finish(**args)
        for key in ('valid','fresh','recordable'):
            gate,args=self.setup_case()
            args['quality']={**args['quality'],key:1}
            with self.assertRaisesRegex(ValueError,'continuity rejected'):
                gate.finish(**args)

    def test_preserves_time_ahead_of_deadline_as_well(self):
        gate, args = self.setup_case()
        args["old_deadline"] = 1000.01
        shifted, receipt = gate.finish(**args)
        self.assertAlmostEqual(self.clock.value-shifted, -.01)
        self.assertAlmostEqual(receipt["old_lateness_s"], -.01)

    def test_existing_capture_or_any_writer_payload_forbids_exclusion(self):
        for key, value in (("event_count", 1), ("ledger_event_count", 2),
                           ("ledger_event_count", 0), ("quality_bytes", 1), ("force_bytes", 1)):
            with self.subTest(key=key, value=value):
                gate, args = self.setup_case()
                args[key] = value
                with self.assertRaisesRegex(ValueError, "forbidden after"):
                    gate.finish(**args)

    def test_no_physics_tick_substep_time_or_integrated_state_change(self):
        changes = (("after_global_tick", 51), ("after_sim_time", 2.000000000001),
                   ("after_substeps", 404), ("after_state", np.arange(12)+1e-12),
                   ("after_state", np.arange(13, dtype=np.float64)))
        for key, value in changes:
            with self.subTest(key=key):
                gate, args = self.setup_case()
                args[key] = value
                with self.assertRaisesRegex(ValueError, "changed during frozen"):
                    gate.finish(**args)

    def test_gate_retains_copy_of_actual_initial_state(self):
        gate, args = self.setup_case()
        self.state[:] = 100
        _, receipt = gate.finish(**args)
        self.assertEqual(receipt["integration_state_width"], 12)

    def test_float_conversion_cannot_hide_a_changed_integrated_state(self):
        for state in (np.asarray([2**60], dtype=np.uint64), np.zeros(12, dtype=np.float32),
                      np.zeros(12, dtype=np.longdouble)):
            gate, args = self.setup_case()
            args["after_state"] = state
            with self.assertRaisesRegex(ValueError, "float64 vector"):
                gate.finish(**args)
        gate, args = self.setup_case()
        args["after_state"][0] = -0.0
        with self.assertRaisesRegex(ValueError, "state changed"):
            gate.finish(**args)

    def test_each_gate_is_one_attempt_even_after_rejection(self):
        for invalid in (False, True):
            gate, args = self.setup_case()
            if invalid:
                args["event_count"] = 1
                with self.assertRaises(ValueError):
                    gate.finish(**args)
                args["event_count"] = 0
            else:
                gate.finish(**args)
            with self.assertRaisesRegex(ValueError, "already been consumed"):
                gate.finish(**args)

    def test_preparation_wall_gap_is_bounded_even_if_device_gap_looks_short(self):
        gate, args = self.setup_case(elapsed=.201, device_gap=.03)
        with self.assertRaisesRegex(ValueError, "200ms tracking gap"):
            gate.finish(**args)

    def test_source_device_gap_rejected_even_with_fresh_arrival(self):
        gate, args = self.setup_case(elapsed=.135, device_gap=.201)
        # A caller's accidentally optimistic quality cannot hide this gap.
        args["quality"] = {"valid": True, "fresh": True, "recordable": True, "reasons": []}
        with self.assertRaisesRegex(ValueError, "device_timestamp_gap"):
            gate.finish(**args)

    def test_duplicate_regressed_or_stale_source_is_rejected(self):
        cases = ((1_000_000_000, 1000.135, 17), (999_000_000, 1000.135, 17),
                 (1_135_000_000, 1000.135, 10), (1_135_000_000, 999.8, 17))
        for timestamp, host, sequence in cases:
            gate, args = self.setup_case()
            args["frame"] = packet(timestamp, host, sequence)
            args["quality"] = {"valid": True, "fresh": True, "recordable": True, "reasons": []}
            with self.assertRaisesRegex(ValueError, "continuity rejected"):
                gate.finish(**args)

    def test_mode_focus_and_body_jumps_cannot_be_washed_by_setup(self):
        for mode, focus, jump in ((2, True, False), (1, False, False), (1, True, True)):
            gate, args = self.setup_case()
            after = packet(1_135_000_000, self.clock.value, 17, mode=mode, focus=focus)
            if jump:
                raw = after.raw_payload
                value = json.loads(raw["value"])
                value["Body"]["joints"][0]["p"][0] = 2.
                raw["value"] = json.dumps(value)
                after = parse_tracking(raw, self.clock.value, 17)
            args["frame"] = after
            with self.assertRaisesRegex(ValueError, "continuity rejected"):
                gate.finish(**args)

    def test_passed_quality_must_also_report_recordable(self):
        for quality in ({}, {"valid": True}, {"valid": True, "fresh": False, "recordable": False, "reasons": []},
                        {"valid": True, "fresh": True, "recordable": True, "reasons": ["bad"]}):
            gate, args = self.setup_case()
            args["quality"] = quality
            with self.assertRaisesRegex(ValueError, "continuity rejected"):
                gate.finish(**args)

    def test_nonfinite_bad_types_invalid_grid_and_relaxed_budget_rejected(self):
        for name, value in (("event_count", True), ("force_bytes", -1),
                ("ledger_event_count", 1.), ("old_deadline", float("nan")),
                ("after_sim_time", float("inf")), ("max_lateness", .101),
                ("after_state", [float("nan")] * 12), ("after_state", np.zeros((3, 4)))):
            gate, args = self.setup_case()
            args[name] = value
            with self.assertRaises(ValueError):
                gate.finish(**args)
        for tick, substeps in ((51, 400), (50, 401), (True, 400)):
            with self.assertRaises(ValueError):
                CaptureStartGate(UUID, tick, 2., substeps, np.zeros(12),
                    packet(1_000_000_000, 1000., 10), clock=lambda: 1000.)

    def test_invalid_before_source_and_reversed_wall_clock_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid source before"):
            self.setup_case(initial_age=.201)
        gate, args = self.setup_case()
        self.clock.value = 999.9
        with self.assertRaisesRegex(ValueError, "wall time is reversed"):
            gate.finish(**args)


if __name__ == "__main__":
    unittest.main()
