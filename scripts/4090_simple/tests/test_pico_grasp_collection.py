"""Grasp source/round/control isolation; no SDK, model inference or simulation."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_grasp_collection_plan as plan
from pico_grasp_collection_progress import CollectionProgress
from pico_grasp_control import GraspStandingControl
from pico_mobile_control import MobileControlError


class GraspCollectionTests(unittest.TestCase):
    def test_source_cannot_be_mobile_or_invented_seed_list(self):
        with self.assertRaisesRegex(ValueError, 'SHA'):
            plan.source_rows(b'{"episode_index":0}\n')

    def test_progress_isolated_from_mobile_native_gate(self):
        rows = [{'official_episode_index': i, 'task_id': plan.TASK_ID,
            'plan_sha256': 'a'*64} for i in range(100)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'progress.json'
            progress = CollectionProgress(rows, path)
            try:
                identity = str(uuid.uuid4())
                progress.submit(identity, Path(directory)/identity)
                result = {'latest_completed': {'uuid': identity, 'pending': False, 'status': 'ready',
                    'available': True, 'collection_enabled': True, 'selection_policy': 'native_only',
                    'selection_status': 'selected', 'disposition': 'kept',
                    'official_task_success_observed': True,
                    'acceptance_gate_version': 'grasp-native-task-success-v1'}}
                self.assertTrue(progress.observe(result))
                self.assertEqual(progress.current['official_episode_index'], 1)
                self.assertEqual(progress.receipt()['total'], 100)
                self.assertIn('grasp', json.loads(path.read_text())['contract'])
            finally:
                progress.close()
            # A legacy progress file cannot impersonate this Grasp round.
            value = json.loads(path.read_text())
            value['contract'] = 'cvpr-mobile-official99-collection-progress-v1'
            path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, '冻结'):
                CollectionProgress(rows, path)

    def test_grasp_controls_record_sticks_but_hold_heading_and_idle_reference(self):
        control = GraspStandingControl('unused', planner_worker=SimpleNamespace(diagnostic_only=False))
        control.yaw = .3
        control.last_tick = 0
        control.last_source_timestamp = control.last_source_sequence = None
        frame = {'inputs': dict(left_axis_x=.4, left_axis_y=.8, right_axis_x=.7, right_axis_y=-.2),
            'host_received_s': 100., 'device_timestamp_ns': 1000000000, 'sequence': 1}
        command, evidence = control._input_command(frame, 100., True)
        self.assertEqual(command['mode'], 0)
        self.assertEqual(control.yaw, .3)
        self.assertEqual(evidence['axes_left_x_y_right_x_y'], [.4,.8,.7,-.2])
        self.assertEqual(evidence['applied_axes_left_x_y_right_x_y'], [0.,0.,0.,0.])
        self.assertEqual(frame['inputs']['left_axis_y'], .8)
        with self.assertRaises(MobileControlError):
            control._input_command(frame, 101., True)


if __name__ == '__main__':
    unittest.main()
