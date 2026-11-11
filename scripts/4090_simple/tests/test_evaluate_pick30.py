"""CPU identity, full RTC/HTTP/native ledger and finality tests for moving Pick."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_pick30 as evaluate
from pick_eval_protocol import profile_spec
from test_pico_xmove_pick_task import fixture as pick_fixture
from test_pico_xmove_bend_pick_task import fixture as bend_fixture
from test_evaluate_mobile_pp30 import encoder
import test_evaluate_mobile_pp30 as mobile_tests
import test_evaluate_bend_pick30 as bend_tests


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(profile):
    state = (pick_fixture if profile == 'xmove_pick' else bend_fixture)()
    state['metadata'].update(max_episode_steps=800, control_hz=200, split='train')
    actors = state['layout']['actors']
    existing = actors.pop('distractor_6'); actors['distractor_0'] = existing
    for index, uid in ((1, '7'), (2, '8')):
        actor = deepcopy(existing); actor['uid'] = actor['asset']['uid'] = uid
        actor['pose']['position'][0] += index*.1
        actors['distractor_'+str(index)] = actor
        state['dr_state_dict']['spatial'][uid] = deepcopy(actor['pose'])
    state['dr_state_dict']['distractors'] = ['6', '7', '8']
    state['dr_state_dict']['spatial']['99'] = deepcopy(existing['pose'])
    return state


class PickCampaignTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.args = SimpleNamespace(deployment=self.root/'deployment', data_root=self.root/'data',
            training_manifest=self.root/'manifest.json', runtime_root=self.root/'runtime', output=self.root/'out',
            checkpoint_step=10000, profile='xmove_pick', port=21115, sampling_seed=20261105,
            rtc_mode='test_time', execute_steps=15)
        self.write(self.args.training_manifest, {'encoder': encoder()})
        self.metadata = {}
        for profile in evaluate.OFFICIAL_METADATA_SHA256:
            paths = []
            for level in range(3):
                rows = []
                for index in range(10):
                    state = fixture(profile)
                    if level == 2: state['dr_state_dict']['spatial'].pop('99')
                    rows.append({'episode_index': index, 'environment_config': json.dumps(state)})
                p = self.args.data_root/'eval'/profile_spec(profile)['source_task']/f'dr-level-{level}/meta/episodes.jsonl'
                p.parent.mkdir(parents=True); p.write_text('\n'.join(map(json.dumps, rows))+'\n'); paths.append(p)
            self.metadata[profile] = paths
        metadata_patch = patch.object(evaluate, 'OFFICIAL_METADATA_SHA256',
            {k: tuple(map(sha, v)) for k, v in self.metadata.items()})
        metadata_patch.start(); self.addCleanup(metadata_patch.stop)
        source_patch = patch.object(evaluate, 'source_identities', return_value={
            'SIMPLE/envs/sonic_loco_manip.py': evaluate.file_identity(Path(__file__))})
        source_patch.start(); self.addCleanup(source_patch.stop)
        self.prepare()

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value))

    def prepare(self):
        spec = profile_spec(self.args.profile); step = self.args.checkpoint_step
        p = self.args.deployment/f'checkpoints/ckpt_{step}/model.safetensors'
        p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(f'exact {self.args.profile} {step}'.encode())
        self.args.expected_model_sha256 = sha(p)
        self.ready = {'contract': evaluate.READY_CONTRACT, 'status': evaluate.READY_STATUS,
            'profile': self.args.profile, 'task_id': spec['task_id'], 'source_task_id': spec['source_task_id'],
            'step': step, 'task_step': step, 'selected_checkpoint_is_final': step == 10000,
            'joint_sha256': sha(p), 'training_completed': True, 'checkpoint_save_completed': True,
            'component_load_validation': {'status': 'passed'}, 'training_manifest_sha256': sha(self.args.training_manifest),
            'deployment_artifacts': {}}
        for name in evaluate.DEPLOYMENT_ARTIFACTS:
            value = ({'data': {'transform': {'model': {'motion_image_policy': 'vlm_color_clean_motion'}}}}
                if name == 'run_config.json' else {'motion_image_policy': 'vlm_color_clean_motion'}
                if name == 'dino_training_contract.json' else {'fixture': True})
            p = self.args.deployment/name; self.write(p, value)
            self.ready['deployment_artifacts'][name] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        self.write(self.args.deployment/'LOCAL_RUN_READY.json', self.ready)

    def test_each_profile_all13_steps_freezes_exact30_and_dynamic_model_identity(self):
        for profile in self.metadata:
            for step in evaluate.SAVE_STEPS:
                with self.subTest(profile=profile, step=step):
                    self.args.profile, self.args.checkpoint_step = profile, step; self.prepare()
                    frozen = evaluate.freeze_inputs(self.args)
                    self.assertEqual(frozen['profile'], profile)
                    self.assertEqual(frozen['checkpoint_step'], step)
                    self.assertEqual(frozen['task_id'], profile_spec(profile)['task_id'])
                    self.assertEqual(frozen['maximum_steps_including_warmup'], 800)
                    self.assertEqual(frozen['initialization_profile'],evaluate.INITIALIZATION_PROFILE)
                    self.assertEqual(frozen['world_translation_m'], [0., 0., 0.])
                    self.assertFalse(frozen['formal_evaluation_eligible'])
                    self.assertFalse(frozen['independent_heldout_split_proven'])
                    self.assertEqual([(c['level'],c['case_index']) for c in frozen['cases']],
                        [(l,i) for l in range(3) for i in range(10)])
                    self.assertEqual([c['sampling_seed'] for c in frozen['cases']], list(range(20261105,20261135)))
                    for c in frozen['cases']:
                        self.assertEqual(c['adaptation']['world_translation_m'], [0.,0.,0.])
                        self.assertEqual(c['protocol']['max_episode_steps'],800)
                        self.assertEqual(c['reset_derivation']['removed_unused_spatial_uids'], ['99'] if c['level']<2 else [])
                        self.assertEqual(c['source_environment_config']['metadata']['split'],'train')
                    json.loads(json.dumps(frozen, default=evaluate._json_value, allow_nan=False))

    def test_ready_must_bind_task_actual_step_finality_and_component_load(self):
        for key,value in [('profile','bend_pick'),('task_id','cvpr/G1SonicBendPickPico-v0'),
                ('source_task_id','wrong'),('task_step',5000),('selected_checkpoint_is_final',False),
                ('training_completed',False),('checkpoint_save_completed',False),
                ('component_load_validation',{'status':'pending'}),('training_manifest_sha256','0'*64)]:
            self.write(self.args.deployment/'LOCAL_RUN_READY.json',dict(self.ready,**{key:value}))
            with self.subTest(key=key), self.assertRaises(ValueError):evaluate.freeze_inputs(self.args)
        self.write(self.args.deployment/'LOCAL_RUN_READY.json',self.ready)
        self.args.checkpoint_step=7777
        with self.assertRaisesRegex(ValueError,'Unselected'):evaluate.freeze_inputs(self.args)

    def test_metadata_artifact_model_and_visual_policy_tamper_are_refused(self):
        p=self.metadata[self.args.profile][0];p.write_text(p.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'frozen official'):evaluate.freeze_inputs(self.args)
        p=self.args.deployment/'motion/joint_motion.ckpt';p.write_text('tamper')
        with self.assertRaisesRegex(ValueError,'artifact differs'):evaluate.freeze_inputs(self.args)
        self.prepare(); p=self.args.deployment/'dino_training_contract.json';self.write(p,{'motion_image_policy':'shared_rgb'})
        self.ready['deployment_artifacts'][p.name]={'sha256':sha(p),'bytes':p.stat().st_size}
        self.write(self.args.deployment/'LOCAL_RUN_READY.json',self.ready)
        with self.assertRaisesRegex(ValueError,'Trained image policy'):evaluate.freeze_inputs(self.args)
        self.prepare();p=self.args.deployment/'checkpoints/ckpt_10000/model.safetensors';p.write_bytes(b'tamper')
        with self.assertRaisesRegex(ValueError,'Actual checkpoint'):evaluate.freeze_inputs(self.args)

    def test_private_dispatch_has_exact_budget_and_no_bend_shift_navigation_or_fall_gate(self):
        for profile in self.metadata:
            self.args.profile=profile;self.prepare();c=evaluate.freeze_inputs(self.args)['cases'][-1]
            commands=evaluate.commands(self.args,c,self.root/'case');sim=commands['simulator'];task=profile_spec(profile)['task']
            for key,value in [('--tasks',task),('--worker-task',task),('--pick-profile',profile),('--seconds','16.0'),
                    ('--expected-checkpoint-step','10000'),('--dr-level','2'),('--episode-start','9')]:
                self.assertEqual(sim[sim.index(key)+1],value)
            self.assertIn('--pick-official-protocol',sim)
            for key in ('--bend-official-protocol','--mobile-official-protocol','--stop-on-fall','--initialization-profile',
                    '--support-geometry-manifest','--mobile-initialization-manifest','--initial-physics-run'):
                self.assertNotIn(key,sim)
            for command in commands.values():
                for key,value in [('--rtc-mode','test_time'),('--execute-steps','15'),('--rtc-inference-delay','0'),('--min-exec-horizon','15')]:
                    self.assertEqual(command[command.index(key)+1],value)
        for value in (10,30):
            self.args.execute_steps=value
            with self.assertRaises(ValueError):evaluate.commands(self.args,c,self.root/'case')

    def safety(self,displacement=.0003,rotation=.02):
        v=bend_tests.BendCampaignTests.safety_fixture(self)
        origin=fixture(self.args.profile)['layout']['actors']['target']['pose']
        v.update(contract='cvpr-pick-official-startup-passive-verification-v1',profile=self.args.profile,
            initialization_profile=evaluate.INITIALIZATION_PROFILE,status='observation_complete',
            initial_target_xyz_m=deepcopy(origin['position']),initial_target_quaternion_wxyz=deepcopy(origin['quaternion']),
            maximum_target_displacement_m=displacement,maximum_target_rotation_degrees=rotation)
        warnings=[]
        if displacement>.002:warnings.append('target_displacement_exceeds_2mm')
        if rotation>2.:warnings.append('target_rotation_exceeds_2deg')
        v.update(engineering_reference_status='failed_or_incomplete' if warnings else 'passed',
            engineering_reference_warnings=warnings,
            tail_settling_metrics={'window_duration_s':.2,'sample_count':41,'start_sim_time_s':.805,'end_sim_time_s':1.005,
                'maximum_pairwise_displacement_m':0.,'endpoint_displacement_m':0.,
                'maximum_pairwise_rotation_degrees':0.,'endpoint_rotation_degrees':0.})
        for row in v['samples']:
            row.update(target_xyz_m=[origin['position'][0]+displacement,*origin['position'][1:]],
                target_quaternion_wxyz=[math.cos(math.radians(rotation)/2),0.,0.,math.sin(math.radians(rotation)/2)],
                target_displacement_m=displacement,target_rotation_degrees=rotation)
        return v

    def evidence(self,native,count):
        task=profile_spec(self.args.profile)['task']
        with patch.object(mobile_tests,'evaluate',evaluate),patch.object(evaluate,'TASK',task,create=True):
            directory,c,f,result,probe=mobile_tests.CampaignTests.evidence_fixture(self,native=native,count=count)
        e=result['episodes'][0]; e.pop('mobile_protocol')
        e.update(task_id=c['task_id'],initialization_profile=evaluate.INITIALIZATION_PROFILE,pick_protocol=c['protocol'],pick_source_environment_config=c['source_environment_config'],
            pick_reset_derivation=c['reset_derivation'],pick_adaptation=c['adaptation'],
            pick_environment_binding={'matched':True},pick_initialization_safety=self.safety())
        result['initialization_profile']=evaluate.INITIALIZATION_PROFILE
        result['pick_adapter']={'adaptation':c['adaptation'],'reset_derivation':c['reset_derivation'],'protocol':c['protocol'],
            'initialization_profile':evaluate.INITIALIZATION_PROFILE}
        rtc,_,_,policy=mobile_tests.CampaignTests.rtc_ledger(self)
        result['server_info']=dict(rtc['server_info'],ckpt_step=self.args.checkpoint_step,run_dir=str(self.args.deployment))
        epdir=directory/'simulation'/task/'episode-0';ticks=evaluate.json_lines(epdir/'trace.jsonl');requests=evaluate.json_lines(directory/'server/requests.jsonl')
        filtered=[]
        for index,(tick,request) in enumerate(zip(ticks,requests)):
            z=.6 if tick['terminated'] else .5;reward=max(0.,min(1.,(z-.5)/.1))
            tick.update(reward=reward,pick_native_state={'target_xyz_m':[0.,0.,z],'baseline_target_z_m':.5,
                'native_reward':reward,'native_success':tick['terminated']})
            if index>=50:
                offset=index-50;start=.005+.02*(50+(offset//25)*25)
                tick.update(chunk_start_sim_time_s=start,action_index_at_30hz=int((tick['sim_time_s']-start)*30+1e-7))
                if offset%25:continue
            first=index==50
            request.update(ordinal=len(filtered)+1,sampling_path=None if index<50 else 'plain' if first else 'test_time',
                sampler_calls={'plain':int(first),'test_time':int(index>50),'train':0},parameter_gradient_tensor_count=0)
            if index>=50:request['response_action_shape']=[15,80]
            filtered.append(request)
        e['policy_chunks']=sum(q['status']==200 for q in filtered)
        http=evaluate.read(epdir/'http.json');times={float(q['sim_time_s']) for q in filtered}
        http=[q for q in http if q['endpoint']!='/act' or q['history']['sim_time_s'] in times]
        e['http_counts']['/act']=len(filtered);result['http_request_count']=len(http)
        self.write(epdir/'http.json',http);(epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps,ticks)))
        (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,filtered)))
        self.write(epdir.parent/'result.json',result)
        startup={'weight_sha256':self.args.expected_model_sha256,'pid':123,'sampling_seed':c['sampling_seed'],'Ta':15,'Tp':30,
            'rtc':True,'inference_steps':10,'online_dino_runtime':encoder(),'parameter_tensor_count':12,
            'trainable_parameter_tensor_count':0,'parameter_gradient_tensor_count':0}
        startup.update({k:policy[k] for k in ('rtc_mode','rtc_exec_shift','rtc_inference_delay','rtc_min_exec_horizon','pig_mask_schedule','pig_guidance_alpha')})
        self.write(directory/'server/startup.json',startup)
        return directory,c,f,result,probe

    def test_full_case_native_success_and800tick_failure_with_real_rtc_audit(self):
        for profile in self.metadata:
            for native,count in ((True,60),(False,800)):
                self.args.profile=profile;self.prepare()
                # Keep each complete CPU fixture isolated like production cases.
                old=self.root;self.root=old/f'{profile}_{count}'
                directory,c,f,result,probe=self.evidence(native,count)
                with patch.object(evaluate.subprocess,'check_output',return_value=json.dumps(probe).encode()):
                    row=evaluate.validate_case(directory,c,f)
                    self.assertEqual(row['native_success'],native);self.assertEqual(row['ticks'],count)
                    self.assertEqual(row['rtc_execution']['test_time_requests'],0 if native else 29)
                    e=result['episodes'][0];e['pick_initialization_safety']['samples'][6]['robot_target_contacts']=1
                    self.write(directory/'simulation'/profile_spec(profile)['task']/'result.json',result)
                    with self.assertRaisesRegex(ValueError,'sample contains'):evaluate.validate_case(directory,c,f)
                self.root=old

    def test_actual_step_rtc_causality_native_and_video_evidence_tamper_fail_closed(self):
        mutations = [
            ('step', lambda result, ticks, requests, http, probe: result['server_info'].update(ckpt_step=5000)),
            ('binding', lambda result, ticks, requests, http, probe: result['episodes'][0]['pick_environment_binding'].update(matched=False)),
            ('profile', lambda result, ticks, requests, http, probe: result['episodes'][0]['pick_initialization_safety'].update(profile='xmove_bend_pick')),
            ('rtc_method', lambda result, ticks, requests, http, probe: requests[-1].update(sampling_path='plain')),
            ('future_history', lambda result, ticks, requests, http, probe: http[1]['history'].update(sim_time_s=99.)),
            ('unpaused', lambda result, ticks, requests, http, probe: http[1].update(simulator_after_s=99.)),
            ('clock', lambda result, ticks, requests, http, probe: ticks[60].update(next_sim_time_s=99.)),
            ('native_reward', lambda result, ticks, requests, http, probe: ticks[60].update(reward=.5)),
            ('early_truncation', lambda result, ticks, requests, http, probe: ticks[-1].update(truncated=False)),
            ('first_baseline', lambda result, ticks, requests, http, probe: ticks[60]['pick_native_state'].update(baseline_target_z_m=.6)),
            ('video', lambda result, ticks, requests, http, probe: probe['streams'][0].update(nb_read_frames='800')),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                old=self.root;self.root=old/name
                directory,c,f,result,probe=self.evidence(False,800)
                task=profile_spec(self.args.profile)['task'];epdir=directory/'simulation'/task/'episode-0'
                ticks=evaluate.json_lines(epdir/'trace.jsonl');requests=evaluate.json_lines(directory/'server/requests.jsonl')
                http=evaluate.read(epdir/'http.json');mutate(result,ticks,requests,http,probe)
                self.write(epdir.parent/'result.json',result);self.write(epdir/'http.json',http)
                (epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps,ticks)))
                (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,requests)))
                with patch.object(evaluate.subprocess,'check_output',return_value=json.dumps(probe).encode()):
                    with self.assertRaises(ValueError):evaluate.validate_case(directory,c,f)
                self.root=old

    def test_pending_ready_cli_does_not_launch_or_create_output(self):
        self.write(self.args.deployment/'LOCAL_RUN_READY.json',dict(self.ready,status='pending'))
        self.args.runtime_root.mkdir()
        argv=[]
        for name in ('deployment','data-root','training-manifest','output','runtime-root'):
            argv.extend(['--'+name,str(getattr(self.args,name.replace('-','_')))])
        argv += ['--profile',self.args.profile,'--checkpoint-step',str(self.args.checkpoint_step),
            '--expected-model-sha256',self.args.expected_model_sha256]
        with patch.object(evaluate.subprocess,'Popen') as launch:
            with self.assertRaises(ValueError):evaluate.main(argv)
            launch.assert_not_called()
        self.assertFalse(self.args.output.exists())

    def test_initialization_profile_contract_and_all201_samples_are_required(self):
        evaluate.validate_initialization_safety(self.safety())
        for mutate in (lambda v:v.update(contract='cvpr-bend-pick-revised-startup-passive-verification-v1'),
                lambda v:v['samples'].pop(),lambda v:v['samples'][3].update(sim_time_s=.5),
                lambda v:v.update(maximum_target_displacement_m=.003),lambda v:v.update(live_physical_state_writes=True),
                lambda v:v['samples'][3]['target_xyz_m'].__setitem__(0,99.),
                lambda v:v['tail_settling_metrics'].update(endpoint_displacement_m=.1),
                lambda v:v.update(engineering_reference_warnings=['target_displacement_exceeds_2mm']),
                lambda v:v.update(engineering_reference_status='failed_or_incomplete'),
                lambda v:v.update(status='passed')):
            value=self.safety();mutate(value)
            with self.assertRaises(ValueError):evaluate.validate_initialization_safety(value)

    def test_large_uncommanded_settle_is_warning_without_scene_or_native_outcome_filter(self):
        safety=self.safety(displacement=.015,rotation=7.)
        self.assertEqual(evaluate.validate_initialization_safety(safety),
            ['target_displacement_exceeds_2mm','target_rotation_exceeds_2deg'])
        directory,c,f,result,probe=self.evidence(False,800)
        result['episodes'][0]['pick_initialization_safety']=safety
        task=profile_spec(self.args.profile)['task']
        self.write(directory/'simulation'/task/'result.json',result)
        with patch.object(evaluate.subprocess,'check_output',return_value=json.dumps(probe).encode()):
            row=evaluate.validate_case(directory,c,f)
        self.assertFalse(row['native_success']);self.assertEqual(row['status'],'complete')
        self.assertEqual(row['initialization_warning_count'],2)
        self.assertEqual(row['initialization_profile'],evaluate.INITIALIZATION_PROFILE)
        self.assertEqual(row['pick_initialization_safety'],safety)
        state={'cases':[{'level':l,'case_index':i,'status':'complete','native_success':False}
            for l in range(3) for i in range(10)]}
        state['cases'][0]=row
        self.assertEqual(evaluate.finish(state),0)
        self.assertEqual(state['technically_valid_cases'],30)
        self.assertEqual(state['native_successes'],0)
        self.assertEqual(state['initialization_warning_case_count'],1)
        self.assertEqual(state['initialization_warning_count'],2)
        self.assertEqual(state['initialization_profile'],evaluate.INITIALIZATION_PROFILE)
        # No forgery of the original Bend engineering result is accepted.
        safety['engineering_reference_status']='passed'
        with self.assertRaises(ValueError):evaluate.validate_initialization_safety(safety)

    def test_all30_zero_success_is_complete_technical_and_duplicate_are_failure(self):
        state={'cases':[{'level':l,'case_index':i,'status':'complete','native_success':False} for l in range(3) for i in range(10)]}
        self.assertEqual(evaluate.finish(state),0)
        state['cases'][0]['status']='technical_error';self.assertEqual(evaluate.finish(state),1)
        state['cases'][0]=deepcopy(state['cases'][1]);self.assertEqual(evaluate.finish(state),1)


if __name__=='__main__':unittest.main()
