"""CPU checks that saved evidence, rather than Isaac exit(0), controls status."""
import importlib.util
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("run_closed_loop", ROOT / "run_closed_loop.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeReportingTests(unittest.TestCase):
    def test_error_running_or_empty_worker_never_passes(self):
        for status in ("error", "running", "bounded_smoke_finished"):
            self.assertTrue(smoke.worker_result_errors({"status": status, "episodes": []}, 1))
        self.assertTrue(smoke.worker_result_errors(None, 1))

    def test_complete_episode_requires_actual_controller_and_finite_final_state(self):
        episode = {"status": "sim_time_limit", "finite_states": True,
                   "diagnostic_loop_executed": True, "decoder_calls": 200,
                   "simulator_unchanged_during_http": True}
        valid = {"status": "bounded_smoke_finished", "episodes": [episode]}
        self.assertEqual(smoke.worker_result_errors(valid, 1), [])
        self.assertTrue(smoke.worker_result_errors(valid, 2))
        for field, bad in (("status", "error"), ("finite_states", False),
                           ("diagnostic_loop_executed", False), ("decoder_calls", 0),
                           ("simulator_unchanged_during_http", False)):
            with self.subTest(field=field):
                changed = {"status": "bounded_smoke_finished", "episodes": [{**episode, field: bad}]}
                self.assertTrue(smoke.worker_result_errors(changed, 1))


if __name__ == "__main__":
    unittest.main()
