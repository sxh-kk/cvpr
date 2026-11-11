"""CPU campaign tests: task identity, actual artifacts, RTC and native evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_bend_pick30 as evaluate
from test_bend_eval_protocol import fixture
import test_evaluate_mobile_pp30 as mobile_tests
from test_evaluate_mobile_pp30 import encoder


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class BendCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(deployment=self.root/'deployment', data_root=self.root/'data',
            training_manifest=self.root/'manifest.json', runtime_root=self.root/'runtime', output=self.root/'out',
            checkpoint_step=5000, port=21115, sampling_seed=20261111, rtc_mode='test_time', execute_steps=None)
        model = self.args.deployment/'checkpoints/ckpt_5000/model.safetensors'
        model.parent.mkdir(parents=True); model.write_bytes(b'exact selected Bend checkpoint')
        self.args.expected_model_sha256 = sha(model)
        self.write(self.args.training_manifest, {'encoder': encoder()})
        ready = self.ready = {'contract': evaluate.READY_CONTRACT, 'status': evaluate.READY_STATUS,
            'task_id': evaluate.TASK_ID, 'step': 5000, 'task_step': 5000,
            'joint_sha256': sha(model), 'checkpoint_save_completed': True, 'training_completed': True,
            'component_load_validation': {'status': 'passed'}, 'training_manifest_sha256': sha(self.args.training_manifest),
            'deployment_artifacts': {}}
        for name in evaluate.DEPLOYMENT_ARTIFACTS:
            path = self.args.deployment/name
            value = ({'data': {'transform': {'model': {'motion_image_policy': 'vlm_color_clean_motion'}}}}
                if name == 'run_config.json' else {'motion_image_policy': 'vlm_color_clean_motion'}
                if name == 'dino_training_contract.json' else {'fixture': True})
            self.write(path, value)
            ready['deployment_artifacts'][name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        self.write(self.args.deployment/'LOCAL_RUN_READY.json', ready)
        self.metadata = []
        for level in range(3):
            rows = []
            for index in range(10):
                source = fixture()
                if level == 2: source['dr_state_dict']['spatial'].pop('99')
                rows.append({'episode_index': index, 'environment_config': json.dumps(source)})
            path = self.args.data_root/'eval'/evaluate.SOURCE_TASK/f'dr-level-{level}/meta/episodes.jsonl'
            path.parent.mkdir(parents=True); path.write_text('\n'.join(map(json.dumps, rows))+'\n')
            self.metadata.append(path)
        metadata_patch = patch.object(evaluate, 'OFFICIAL_METADATA_SHA256', tuple(map(sha, self.metadata)))
        metadata_patch.start(); self.addCleanup(metadata_patch.stop)
        # Source identity itself is tested against installed SIMPLE in the integration test.
        sources_patch = patch.object(evaluate, 'source_identities', return_value={'SIMPLE/envs/sonic_loco_manip.py': evaluate.file_identity(Path(__file__))})
        sources_patch.start(); self.addCleanup(sources_patch.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))

    def test_exact30_freeze_retains_original_split_and_independent_sonic_receipt(self):
        frozen = evaluate.freeze_inputs(self.args)
        self.assertEqual([(c['level'], c['case_index']) for c in frozen['cases']], [(l, i) for l in range(3) for i in range(10)])
        self.assertEqual({c['environment_seed'] for c in frozen['cases']}, {0})
        self.assertEqual([c['sampling_seed'] for c in frozen['cases']], list(range(20261111, 20261141)))
        self.assertEqual(frozen['maximum_steps_including_warmup'], 400)
        self.assertEqual(frozen['task_id'], 'cvpr/G1SonicBendPickPico-v0')
        self.assertFalse(frozen['independent_heldout_split_proven'])
        self.assertFalse(frozen['formal_evaluation_eligible'])
        for case in frozen['cases']:
            self.assertEqual(case['source_environment_config']['metadata']['split'], 'train')
            self.assertEqual(case['local_environment_config']['robot_cfg']['uid'], 'g1_sonic')
            self.assertEqual(case['adaptation']['world_translation_m'], [0., 0., .405])
            self.assertEqual(case['reset_derivation']['removed_unused_spatial_uids'], ['99'] if case['level'] < 2 else [])
        # The serialized frozen configuration must compare with worker JSON without ndarray ambiguity.
        json.loads(json.dumps(frozen, default=evaluate._json_value, allow_nan=False))

    def test_refuse_mobile_ready_pending_save_wrong_step_or_tampered_model(self):
        for field, value in [('contract', 'cvpr-mobile-pp-local-deployment-v1'), ('task_id', 'simple/Mobile'),
            ('training_completed', False), ('checkpoint_save_completed', False), ('task_step', 10000),
            ('training_manifest_sha256', '0'*64), ('component_load_validation', {'status': 'pending'})]:
            with self.subTest(field=field):
                self.write(self.args.deployment/'LOCAL_RUN_READY.json', dict(self.ready, **{field: value}))
                with self.assertRaises(ValueError): evaluate.freeze_inputs(self.args)
        self.write(self.args.deployment/'LOCAL_RUN_READY.json', self.ready)
        (self.args.deployment/'checkpoints/ckpt_5000/model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Actual checkpoint'): evaluate.freeze_inputs(self.args)

    def test_reject_changed_artifact_and_trained_image_policy_reversion(self):
        path = self.args.deployment/'dino_training_contract.json'
        self.write(path, {'motion_image_policy': 'shared_rgb'})
        with self.assertRaisesRegex(ValueError, 'artifact differs'): evaluate.freeze_inputs(self.args)
        self.ready['deployment_artifacts'][path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        self.write(self.args.deployment/'LOCAL_RUN_READY.json', self.ready)
        with self.assertRaisesRegex(ValueError, 'Trained image policy'): evaluate.freeze_inputs(self.args)

    def test_metadata_byte_drift_duplicate_index_and_missing_real_actor_are_rejected(self):
        path = self.metadata[0]; original = path.read_text(); path.write_text(original+'\n')
        with self.assertRaisesRegex(ValueError, 'frozen official'): evaluate.freeze_inputs(self.args)
        rows = [json.loads(line) for line in original.splitlines()]; rows[-1]['episode_index'] = 0
        path.write_text('\n'.join(map(json.dumps, rows)))
        with patch.object(evaluate, 'OFFICIAL_METADATA_SHA256', tuple(map(sha, self.metadata))):
            with self.assertRaisesRegex(ValueError, 'exactly ten'): evaluate.freeze_inputs(self.args)
        rows[-1]['episode_index'] = 9
        state = json.loads(rows[0]['environment_config']); state['dr_state_dict']['spatial'].pop('7')
        rows[0]['environment_config'] = json.dumps(state); path.write_text('\n'.join(map(json.dumps, rows)))
        with patch.object(evaluate, 'OFFICIAL_METADATA_SHA256', tuple(map(sha, self.metadata))):
            with self.assertRaisesRegex(ValueError, 'actual actor'): evaluate.freeze_inputs(self.args)

    def test_commands_use_bend_dispatch_exact8seconds_and_unchanged_rtc_server(self):
        case = evaluate.freeze_inputs(self.args)['cases'][-1]
        commands = evaluate.commands(self.args, case, self.root/'case')
        sim = commands['simulator']
        for option, value in [('--tasks', evaluate.TASK), ('--worker-task', evaluate.TASK), ('--seconds', '8.0'),
            ('--dr-level', '2'), ('--episode-start', '9'), ('--seed', '0'), ('--expected-checkpoint-step', '5000')]:
            self.assertEqual(sim[sim.index(option)+1], value)
        self.assertIn('--bend-official-protocol', sim)
        for option in ['--stop-on-fall', '--mobile-official-protocol', '--initial-physics-run', '--mobile-initialization-manifest']:
            self.assertNotIn(option, sim)
        for command in commands.values():
            for option, value in [('--rtc-mode', 'test_time'), ('--execute-steps', '15'), ('--rtc-inference-delay', '0'),
                ('--min-exec-horizon', '15'), ('--pig-mask-schedule', 'exponential'), ('--pig-guidance-alpha', '0.9')]:
                self.assertEqual(command[command.index(option)+1], value)
        for invalid in ('off', 'train'):
            self.args.rtc_mode = invalid
            with self.assertRaises(ValueError): evaluate.evaluation_policy(self.args)

    def test_actual_sampler_ledger_audit_is_shared_without_changed_success_gate(self):
        result, ticks, requests, policy = mobile_tests.CampaignTests.rtc_ledger(self)
        accepted = evaluate.validate_rtc_execution(result, ticks, requests, policy)
        self.assertEqual((accepted['plain_requests'], accepted['test_time_requests']), (1, 2))
        warmup = evaluate.validate_rtc_execution(result, [], requests[:1], policy)
        self.assertEqual(warmup['test_time_requests'], 0)
        requests[2]['sampling_path'] = 'plain'
        with self.assertRaises(ValueError): evaluate.validate_rtc_execution(result, ticks, requests, policy)

    def test_full_case_audit_preserves_warmup_success_and_budget_failure_with_real_rtc_ledger(self):
        for native, count in ((True, 8), (False, 400)):
            with self.subTest(native=native):
                directory = self.root/('native' if native else 'budget')
                with patch.object(mobile_tests, 'evaluate', evaluate):
                    directory, case, frozen, result, probe = mobile_tests.CampaignTests.evidence_fixture(
                        self, native=native, count=count, directory=directory)
                episode = result['episodes'][0]
                episode.update(task_id=evaluate.TASK_ID, bend_protocol=case['protocol'],
                    bend_source_environment_config=case['source_environment_config'],
                    bend_reset_derivation=case['reset_derivation'], bend_adaptation=case['adaptation'],
                    bend_environment_binding={'matched': True})
                result['bend_adapter'] = {'frame_binding': {'installed': True}}
                rtc_result, _, _, policy = mobile_tests.CampaignTests.rtc_ledger(self)
                result['server_info'] = rtc_result['server_info']
                epdir = directory/'simulation'/evaluate.TASK/'episode-0'
                ticks = evaluate.json_lines(epdir/'trace.jsonl')
                requests = evaluate.json_lines(directory/'server/requests.jsonl')
                filtered = []
                for index, (tick, request) in enumerate(zip(ticks, requests)):
                    z = .6 if tick['terminated'] else .5
                    reward = max(0., min(1., (z-.5)/.1))
                    tick.update(reward=reward, bend_native_state={'target_xyz_m': [0., 0., z],
                        'baseline_target_z_m': .5, 'native_reward': reward, 'native_success': tick['terminated']})
                    if index >= 50:
                        offset = index-50; start = .005+.02*(50+(offset//25)*25)
                        tick.update(chunk_start_sim_time_s=start,
                            action_index_at_30hz=int((tick['sim_time_s']-start)*30+1e-7))
                        if offset % 25: continue
                    first = index == 50
                    request.update(ordinal=len(filtered)+1, sampling_path=None if index < 50 else 'plain' if first else 'test_time',
                        sampler_calls={'plain': int(first), 'test_time': int(index > 50), 'train': 0},
                        parameter_gradient_tensor_count=0)
                    if index >= 50: request['response_action_shape'] = [15, 80]
                    filtered.append(request)
                requests = filtered
                episode['policy_chunks'] = sum(r['status'] == 200 for r in requests)
                http = evaluate.read(epdir/'http.json')
                action_times = {float(r['sim_time_s']) for r in requests}
                http = [r for r in http if r['endpoint'] != '/act' or r['history']['sim_time_s'] in action_times]
                episode['http_counts']['/act'] = len(requests)
                result['http_request_count'] = len(http)
                self.write(epdir/'http.json', http)
                (epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps, ticks)))
                (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps, requests)))
                self.write(epdir.parent/'result.json', result)
                with patch.object(evaluate.subprocess, 'check_output', return_value=json.dumps(probe).encode()):
                    accepted = evaluate.validate_case(directory, case, frozen)
                    self.assertEqual(accepted['native_success'], native)
                    self.assertEqual(accepted['ticks'], count)
                    self.assertTrue(accepted['fell'])
                    self.assertEqual(accepted['rtc_execution']['test_time_requests'], 0 if native else 13)
                    if not native:
                        ticks[-1]['truncated'] = False
                        (epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps, ticks)))
                        with self.assertRaisesRegex(ValueError, 'full official budget'):
                            evaluate.validate_case(directory, case, frozen)

    def test_native_reward_ledger_cannot_replace_or_strengthen_original_success(self):
        def tick(z):
            reward = max(0., min(1., (z-.5)/.1)); success = reward >= .9
            return {'reward': reward, 'terminated': success, 'bend_native_state': {
                'target_xyz_m': [0., 0., z], 'baseline_target_z_m': .5, 'native_reward': reward, 'native_success': success}}
        ticks = [tick(.5), tick(.58), tick(.60)]
        evaluate.validate_native_reward_ledger(ticks)  # No contact, hold or learned-action requirement.
        for mutate in [lambda t: t[-1].update(terminated=False),
            lambda t: t[0]['bend_native_state'].update(baseline_target_z_m=.4),
            lambda t: t[-1].update(reward=.5),
            lambda t: t[-1]['bend_native_state'].update(target_xyz_m=[0., 0., float('nan')])]:
            changed = deepcopy(ticks); mutate(changed)
            with self.assertRaises(ValueError): evaluate.validate_native_reward_ledger(changed)


if __name__ == '__main__': unittest.main()
