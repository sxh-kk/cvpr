"""CPU-only boundaries for the new task launcher, independent of model loading."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import finetune_lift_box_hold as launch


class LiftLauncherContracts(unittest.TestCase):
    def test_bounds_report_measures_all78_raw_channels_without_changing_bounds(self):
        field = SimpleNamespace(action_min=[-1.]*80, action_max=[1.]*80)
        before = (field.action_min.copy(), field.action_max.copy())
        values = np.zeros((2,78)); values[0,0] = -1.25; values[1,77] = 1.1
        report = launch.command_bounds(values,field)
        self.assertEqual(report["groups"]["body64"]["channels"][0]["outside_fraction"],.5)
        self.assertAlmostEqual(report["groups"]["body64"]["maximum_below_minimum"],.25)
        self.assertAlmostEqual(report["groups"]["hand14"]["maximum_above_maximum"],.1)
        self.assertEqual((field.action_min,field.action_max), before)
        self.assertFalse(report["test_used_for_statistics"])

    def test_handwritten_pass_cannot_replace_real_physics_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'LIFT_COMMANDED_ROLLOUT_READY.json').write_text(json.dumps({"contract":launch.CONTRACT,
                "task_id":launch.TASK_ID,"status":"passed","training_authorized":True}))
            with self.assertRaisesRegex(RuntimeError,"verification failed"):
                launch.audit_quality(root,launch.SIM_PYTHON)

    def test_explicit_snapshot_contains_new_ignored_task_renderer_and_launcher(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);data=root/'dataset';data.mkdir()
            (data/'LIFT_COMMANDED_ROLLOUT_READY.json').write_text('{}')
            report=launch.snapshot_sources(SimpleNamespace(output=root/'output',dataset_root=data),{"episodes":[]})
            self.assertIn('tree/scripts/4090_simple/finetune_lift_box_hold.py',report['files'])
            self.assertIn('tree/scripts/4090_simple/prepare_lift_box_hold.py',report['files'])
            self.assertIn('tree/VLMExpert/third_party/SIMPLE/src/simple/tasks/g1_wholebody_lift_box_hold_sonic.py',report['files'])
            self.assertIn('tree/VLMExpert/third_party/SIMPLE/src/simple/engines/isaacsim.py',report['files'])
            self.assertTrue(all(Path(value['actual_path']).is_file() for value in report['files'].values()))


if __name__ == '__main__':
    unittest.main()
