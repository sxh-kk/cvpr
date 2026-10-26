"""Read-only recovery planning and genuine launcher adaptation; no GPU execution."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

FILE = Path(__file__).resolve().parents[1] / 'resume_handover149_eval.py'
SPEC = importlib.util.spec_from_file_location('resume149', FILE)
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)
TEMPLATE = Path('/home/zxc/quality100_4090_20261024/mirror/home/weisheng/psi0_runtime/quality100_milestone_eval_20261023/handover149_5000_4090_eval_20261026_r2')


class PlanningTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(case_id=f'level_{level}_episode_{episode:02d}',
            official_episode_index=episode, official_level=level)
            for level in range(3) for episode in range(10)]
        ordered = sorted(self.rows, key=lambda r: (r['official_episode_index'], r['official_level']))
        self.protocol = {'ordered_seed_pairs': self.rows}
        self.state = dict(step=5000, model_sha256=p.MODEL_SHA, status='execution_error', pid=None,
            current_case='level_1_episode_09', completed=[r['case_id'] for r in ordered[:28]],
            failures=[dict(case='level_1_episode_09', error='RuntimeError: Owned cleanup failure: PermissionError',
                counted_as_policy_failure=False)])

    def test_exact_original28_and_last_two_only(self):
        original = deepcopy(self.state)
        previous, pending = p.recovery_plan(self.protocol, self.state)
        self.assertEqual(len(previous), 28)
        self.assertEqual([r['case_id'] for r in pending], list(p.CASES))
        self.assertEqual(self.state, original)

    def test_wrong_model_step_live_pid_and_failure_type_rejected(self):
        for key, value in [('step', 4000), ('model_sha256', 'a'*64), ('pid', 123),
                           ('status', 'running'), ('current_case', 'level_0_episode_09')]:
            with self.subTest(key=key):
                state = deepcopy(self.state); state[key] = value
                with self.assertRaises(RuntimeError): p.recovery_plan(self.protocol, state)
        state = deepcopy(self.state); state['failures'][0]['counted_as_policy_failure'] = True
        with self.assertRaises(RuntimeError): p.recovery_plan(self.protocol, state)
        state = deepcopy(self.state); state['failures'][0]['error'] = 'model has nonfinite outputs'
        with self.assertRaises(RuntimeError): p.recovery_plan(self.protocol, state)

    def test_cannot_drop_reorder_or_reexecute_previous_case(self):
        for completed in [self.state['completed'][:-1], self.state['completed'] + ['level_1_episode_09'],
                          list(reversed(self.state['completed']))]:
            state = deepcopy(self.state); state['completed'] = completed
            with self.assertRaises(RuntimeError): p.recovery_plan(self.protocol, state)

    def test_duplicate_or_different_final_case_rejected(self):
        for rows in [self.rows[:-1], self.rows + [self.rows[0]]]:
            with self.assertRaises(RuntimeError): p.recovery_plan({'ordered_seed_pairs': rows}, self.state)


@unittest.skipUnless(TEMPLATE.is_dir(), 'Actual accepted R2 wrapper unavailable')
class WrapperTests(PlanningTests):
    def test_real_wrapper_changes_only_reviewed_entry_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            mirror = Path(directory) / 'mirror/home'
            relative = 'weisheng/psi0_runtime/quality100_milestone_eval_20261023'
            framework = mirror / relative / TEMPLATE.name
            framework.mkdir(parents=True)
            original = (TEMPLATE / 'namespace_launch.sh').read_text()
            (framework / 'namespace_launch.sh').write_text(original)
            (framework / 'host_target.json').write_text(json.dumps({'mirror_home': str(mirror)}))
            experiment = framework.parent / 'handover149_experiment_5000_official30_4090_trt10133_20261026_r2'
            experiment.mkdir()
            for name, value in [('protocol.json', self.protocol), ('state.json', self.state),
                                ('results.json', {'status': 'partial'}), ('preflight_acceptance.json', {'fixture': True})]:
                (experiment / name).write_text(json.dumps(value))
            failed = experiment / 'cases/level_1_episode_09/attempt_001'; failed.mkdir(parents=True)
            (failed / 'owned_processes.json').write_text(json.dumps({
                'cleanup_complete': False, 'processes': [
                    {'role': 'model', 'pid': 99999998, 'process_group': 99999998},
                    {'role': 'simulator_launcher', 'pid': 99999999, 'process_group': 99999999}]}))
            output = framework / 'cleanup_recovery001'
            real_exists = Path.exists
            def exists(path):
                if str(path) == '/proc/2109022': return False
                return real_exists(path)
            with patch.object(Path, 'exists', exists):
                result = p.prepare(framework, output)
            generated = (output / 'namespace_resume.sh').read_text()
            differences = [(a,b) for a,b in zip(original.splitlines(), generated.splitlines()) if a != b]
            self.assertEqual(len(differences), 3)
            self.assertTrue(any('script_dir=' in b for _,b in differences))
            self.assertTrue(any('[[ "$phase" == run ]]' == b for _,b in differences))
            self.assertTrue(any('resume_handover149_eval.py --execute' in b for _,b in differences))
            for line in original.splitlines():
                if any(token in line for token in ('exec 9>', 'flock -n 9', 'exec unshare',
                    'mount --bind', 'mount -t tmpfs', 'host_bootstrap.py')):
                    self.assertIn(line, generated)
            self.assertEqual((framework / 'namespace_launch.sh').read_text(), original)
            self.assertEqual(p.read(experiment / 'state.json'), self.state)
            self.assertEqual(p.read(failed / 'owned_processes.json')['cleanup_complete'], False)
            admission = p.read(output / 'ROOT_RESUME_ADMISSION.json')
            self.assertEqual(len(admission['source_bindings']), 3)
            self.assertFalse(admission['preflight_rerun_claimed'])
            self.assertFalse(admission['inference_sources_modified'])
            self.assertEqual(result['recovery_namespace'], '/home/' + relative + '/' + TEMPLATE.name + '/cleanup_recovery001')
            subprocess.run(['bash', '-n', str(output / 'namespace_resume.sh')], check=True)


if __name__ == '__main__':
    unittest.main(verbosity=2)
