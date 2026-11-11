"""CPU campaign tests: task identity, actual artifacts, RTC and native evidence."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace, ModuleType
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

    def test_profile_cli_gate_keeps_original_default_and_requires_stable_manifest(self):
        self.assertEqual(evaluate.initialization_profile(self.args), 'official_original')
        for fields in [dict(initialization_profile='stable_support_v1'),
            dict(support_geometry_manifest=self.root/'geometry.json'),
            dict(prioritize_support_adjusted=True), dict(initialization_profile='foreign')]:
            args = SimpleNamespace(**vars(self.args)); vars(args).update(fields)
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                evaluate.initialization_profile(args)
        self.args.initialization_profile = 'stable_support_v1'
        self.args.support_geometry_manifest = self.root/'geometry.json'
        self.args.prioritize_support_adjusted = True
        self.assertEqual(evaluate.initialization_profile(self.args), 'stable_support_v1')

    def test_uniform_support_derivation_all30_and_geometric_priority_preserve_case_seed(self):
        self.args.initialization_profile = 'stable_support_v1'
        self.args.support_geometry_manifest = self.root/'geometry.json'
        self.args.prioritize_support_adjusted = True
        self.write(self.args.support_geometry_manifest, {'fixture': 'geometry'})
        helper = ModuleType('bend_support_initialization')
        calls = []
        geometry_identity = {key: evaluate.file_identity(self.args.support_geometry_manifest)[key]
            for key in ('path', 'sha256', 'size_bytes')}
        helper.load_geometry_manifest = lambda path, data_root: {'manifest_identity': geometry_identity}
        def derive(source, geometry):
            index = len(calls); calls.append(deepcopy(source))
            # This dispatcher fixture assigns a geometric adjustment every ten
            # cases; no task result or previous model evidence is supplied.
            applied = index % 10 == 1
            corrected = deepcopy(source)
            if applied:
                uid = str(source['dr_state_dict']['target']['uid'])
                corrected['dr_state_dict']['spatial'][uid]['position'][0] += .001
                corrected['layout']['actors']['target']['pose']['position'][0] += .001
            return corrected, {'profile': 'stable_support_v1', 'applied': applied,
                'geometry_manifest_identity': geometry['manifest_identity']}
        helper.derive_stable_support_reset = derive
        with patch.dict(sys.modules, {'bend_support_initialization': helper}):
            frozen = evaluate.freeze_inputs(self.args)
        self.assertEqual(len(calls), 30)
        self.assertEqual(frozen['contract'], evaluate.STABLE_SUPPORT_CONTRACT)
        self.assertFalse(frozen['original_official_initialization'])
        self.assertIn('not the original official30', frozen['boundary'])
        self.assertEqual(frozen['support_adjusted_cases'], 3)
        self.assertEqual([(c['level'], c['case_index']) for c in frozen['cases'][:3]], [(0, 1), (1, 1), (2, 1)])
        originals = [(level, index) for level in range(3) for index in range(10)]
        self.assertEqual(sorted((c['level'], c['case_index']) for c in frozen['cases']), originals)
        for case in frozen['cases']:
            flat = case['level']*10+case['case_index']
            self.assertEqual(case['flat_index'], flat)
            self.assertEqual(case['sampling_seed'], self.args.sampling_seed+flat)
            self.assertEqual(case['environment_seed'], 0)
            self.assertNotIn('99', calls[flat]['dr_state_dict']['spatial'])
            self.assertEqual(json.loads(case['source_row']['environment_config']), case['source_environment_config'])
        command = evaluate.commands(self.args, frozen['cases'][0], self.root/'case')['simulator']
        self.assertEqual(command[command.index('--initialization-profile')+1], 'stable_support_v1')
        self.assertEqual(command[command.index('--support-geometry-manifest')+1], str(self.args.support_geometry_manifest))
        self.assertEqual(command[command.index('--seconds')+1], '8.0')
        self.assertEqual(command[command.index('--warmup-control')+1], 'live_pose_sonic_v1_1')
        before = deepcopy(frozen['cases'])
        self.assertEqual(evaluate.order_support_cases(before, False), before)

    def safety_fixture(self):
        return {'contract': 'cvpr-bend-pick-revised-startup-passive-verification-v1', 'status': 'passed',
            'observed_physics_substeps': 201, 'required_physics_substeps': 201, 'warmup_control_ticks': 50,
            'native_reset_physics_substeps': 1, 'robot_target_contact_count': 0, 'physics_steps_added': 0,
            'engineering_displacement_limit_m': .002, 'engineering_rotation_limit_degrees': 2.,
            'maximum_target_displacement_m': .0003, 'maximum_target_rotation_degrees': .02,
            'finite_states': True, 'private_FK_only': True, 'live_physical_state_writes': False,
            'controller_commands_modified': False, 'policy_phase_interventions': False, 'contacts': [],
            'samples': [{'substep': index, 'sim_time_s': .005*index, 'target_xyz_m': [.0003, 0., 0.],
                'target_quaternion_wxyz': [math.cos(math.radians(.02)/2), 0., 0., math.sin(math.radians(.02)/2)],
                'robot_target_contacts': 0, 'target_displacement_m': .0003, 'target_rotation_degrees': .02}
                for index in range(1, 202)]}

    def test_stable_initialization_requires_complete_passive_monitor_without_strengthening_native_gate(self):
        evaluate.validate_initialization_safety(self.safety_fixture())
        for key, value in [('status', 'failed_or_incomplete'), ('observed_physics_substeps', 200),
            ('required_physics_substeps', 200), ('warmup_control_ticks', 49), ('native_reset_physics_substeps', 0),
            ('robot_target_contact_count', 1), ('physics_steps_added', 1), ('physics_steps_added', False),
            ('maximum_target_displacement_m', .00201), ('maximum_target_rotation_degrees', 2.01),
            ('maximum_target_displacement_m', float('nan')), ('engineering_displacement_limit_m', .003),
            ('engineering_rotation_limit_degrees', 3.), ('finite_states', False), ('private_FK_only', False),
            ('live_physical_state_writes', True), ('controller_commands_modified', True), ('policy_phase_interventions', True)]:
            changed = self.safety_fixture(); changed[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                evaluate.validate_initialization_safety(changed)
        result, episode, case, frozen = self.support_identity_fixture()
        episode.pop('bend_initialization_safety')
        with self.assertRaisesRegex(ValueError, 'Missing or failed'):
            evaluate.validate_support_identity(result, episode, case, frozen)
        # Original profile keeps its existing native success, including warmup.
        evaluate.validate_support_identity({}, {}, {}, {})

    def test_startup_sample_ledger_must_match_real_substeps_pose_contacts_and_header_maxima(self):
        evaluate.validate_initialization_safety(self.safety_fixture())
        mutations = [lambda r: r['samples'].pop(),
            lambda r: r['samples'][5].update(substep=5),
            lambda r: r['samples'][5].update(sim_time_s=.035),
            lambda r: r['samples'][5].update(sim_time_s=float('nan')),
            lambda r: r['samples'][5].update(target_xyz_m=[0., 0.]),
            lambda r: r['samples'][5].update(target_xyz_m=[0., 0., float('inf')]),
            lambda r: r['samples'][5].update(target_quaternion_wxyz=[2., 0., 0., 0.]),
            lambda r: r['samples'][5].update(target_quaternion_wxyz=[1., 0., 0.]),
            lambda r: r['samples'][5].update(robot_target_contacts=1),
            lambda r: r['samples'][5].update(robot_target_contacts=False),
            lambda r: r.update(contacts=[{'substep': 6}]),
            lambda r: r['samples'][5].update(target_displacement_m=.003),
            lambda r: r['samples'][5].update(target_rotation_degrees=3.),
            lambda r: r['samples'][5].update(target_displacement_m=-.001),
            lambda r: r['samples'][5].update(target_rotation_degrees=float('nan')),
            lambda r: r.update(maximum_target_displacement_m=.0004),
            lambda r: r.update(maximum_target_rotation_degrees=.03)]
        for mutate in mutations:
            changed = self.safety_fixture(); mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                evaluate.validate_initialization_safety(changed)

    def support_identity_fixture(self):
        identity = {'path': str(self.root/'geometry.json'), 'sha256': 'f'*64, 'size_bytes': 10,
            'mtime_ns': 123, 'device': 1, 'inode': 2}
        minimal = {key: identity[key] for key in ('path', 'sha256', 'size_bytes')}
        receipt = {'contract': 'cvpr-bend-pick-stable-support-initialization-v1',
            'profile': 'stable_support_v1', 'applied': True, 'geometry_manifest_identity': minimal,
            'after_com_xyz_m': [0., .1, .5], 'actual_world_translation_m': [0., 0., .405]}
        receipt_sha = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        episode = {'initialization_profile': 'stable_support_v1', 'bend_support_initialization': deepcopy(receipt),
            'bend_support_geometry_manifest': minimal,
            'bend_support_model_verification': {'status': 'passed', 'profile': 'stable_support_v1', 'applied': True,
                'read_only': True, 'physics_steps_taken': 0, 'initialization_receipt_sha256': receipt_sha,
                'actual_target_com_world_m': [0., .1, .905]},
            'bend_initialization_safety': self.safety_fixture()}
        case = {'initialization_profile': 'stable_support_v1', 'bend_support_initialization': receipt,
            'bend_support_geometry_manifest': identity}
        frozen = {'contract': evaluate.STABLE_SUPPORT_CONTRACT, 'initialization_profile': 'stable_support_v1',
            'support_geometry_manifest': identity}
        return {'initialization_profile': 'stable_support_v1'}, episode, case, frozen

    def test_real_model_support_gate_binds_profile_geometry_receipt_and_before_physics(self):
        result, episode, case, frozen = self.support_identity_fixture()
        evaluate.validate_support_identity(result, episode, case, frozen)
        mutations = [lambda e: e.update(initialization_profile='official_original'),
            lambda e: e['bend_support_initialization'].update(applied=False),
            lambda e: e['bend_support_geometry_manifest'].update(sha256='0'*64),
            lambda e: e['bend_support_model_verification'].update(status='pending'),
            lambda e: e['bend_support_model_verification'].update(read_only=False),
            lambda e: e['bend_support_model_verification'].update(physics_steps_taken=1),
            lambda e: e['bend_support_model_verification'].update(physics_steps_taken=False),
            lambda e: e['bend_support_model_verification'].update(initialization_receipt_sha256='0'*64),
            lambda e: e['bend_support_model_verification'].update(actual_target_com_world_m=[0., .2, .905])]
        for mutate in mutations:
            changed = deepcopy(episode); mutate(changed)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                evaluate.validate_support_identity(result, changed, case, frozen)
        evaluate.validate_support_identity({}, {}, {}, {})
        with self.assertRaisesRegex(ValueError, 'undeclared'):
            evaluate.validate_support_identity({}, {'bend_support_initialization': {}}, {}, {})
        wrong = deepcopy(frozen); wrong['contract'] = evaluate.CONTRACT
        with self.assertRaisesRegex(ValueError, 'independent'):
            evaluate.validate_support_identity(result, episode, case, wrong)

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
