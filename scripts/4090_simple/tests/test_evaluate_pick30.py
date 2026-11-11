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

    def test_explicit_diagnostic_parser_and_full30_default_rtc_gate(self):
        pairs=((0,6),(0,0),(2,7),(0,4))
        self.assertEqual(evaluate.parse_diagnostic_cases('0:6,0:0,2:7,0:4'),pairs)
        for value in ('','0:6,0:6','0:10','3:0','0:0,','0:0;2:7','-1:0','0:00'):
            with self.subTest(value=value),self.assertRaises(ValueError):evaluate.parse_diagnostic_cases(value)
        self.args.rtc_mode='off'
        with self.assertRaisesRegex(ValueError,'Full30 production'):evaluate.evaluation_policy(self.args)
        self.args.diagnostic_cases=pairs;self.args.execute_steps=None
        policy=evaluate.evaluation_policy(self.args)
        self.assertFalse(policy['rtc_enabled']);self.assertEqual((policy['execute_steps'],policy['control_ticks_per_window']),(15,25))
        self.args.execute_steps=30
        with self.assertRaises(ValueError):evaluate.evaluation_policy(self.args)

    def test_pilot_freezes_all30_but_selects_only_ordered_original_cases_and_seeds(self):
        self.args.diagnostic_cases='0:6,0:0,2:7,0:4';self.args.rtc_mode='off'
        frozen=evaluate.freeze_inputs(self.args)
        self.assertEqual(frozen['contract'],evaluate.DIAGNOSTIC_CONTRACT)
        self.assertEqual(frozen['experiment_role'],'same_seed_rtc_single_factor_pilot')
        self.assertEqual(len(frozen['cases']),30);self.assertEqual(len(frozen['metadata_files']),3)
        self.assertEqual(frozen['planned_cases'],4);self.assertEqual(frozen['evaluation_denominator'],4)
        self.assertFalse(frozen['full_official30_claim'])
        selected=evaluate.execution_cases(frozen)
        self.assertEqual([(c['level'],c['case_index']) for c in selected],[(0,6),(0,0),(2,7),(0,4)])
        self.assertEqual([c['sampling_seed'] for c in selected],[20261111,20261105,20261132,20261109])
        for command in evaluate.commands(self.args,selected[0],self.root/'case').values():
            self.assertEqual(command[command.index('--rtc-mode')+1],'off')
            self.assertEqual(command[command.index('--execute-steps')+1],'15')
        changed=deepcopy(frozen);changed['diagnostic_selected_cases'][0]['sampling_seed']+=1
        with self.assertRaisesRegex(ValueError,'seed or identity'):evaluate.execution_cases(changed)
        changed=deepcopy(frozen);changed['contract']=evaluate.CONTRACT
        with self.assertRaisesRegex(ValueError,'silently select'):evaluate.execution_cases(changed)
        self.args.rtc_mode='test_time'
        self.assertEqual(evaluate.freeze_inputs(self.args)['contract'],evaluate.DIAGNOSTIC_CONTRACT)

    def test_pilot_zero_success_technical_completion_uses_selected_denominator_and_seed(self):
        self.args.diagnostic_cases='0:6,0:0,2:7,0:4';self.args.rtc_mode='off';f=evaluate.freeze_inputs(self.args)
        state={key:deepcopy(f[key]) for key in ('contract','experiment_role','planned_cases','evaluation_denominator',
            'full_official30_claim','diagnostic_selected_cases')}
        state['cases']=[dict(c,status='complete',native_success=False) for c in f['diagnostic_selected_cases']]
        self.assertEqual(evaluate.finish(state),0);self.assertEqual(state['status'],'complete')
        self.assertEqual(state['technically_valid_cases'],4);self.assertEqual(state['native_successes'],0)
        self.assertEqual(state['native_success_rate'],0.)
        self.assertEqual([r['planned_cases'] for r in state['levels']],[3,0,1])
        for mutate in (lambda v:v['cases'][0].update(status='technical_error'),
            lambda v:v['cases'].pop(),lambda v:v['cases'].append(deepcopy(v['cases'][0])),
            lambda v:v['cases'][0].update(sampling_seed=0),
            lambda v:v['cases'].__setitem__(0,deepcopy(v['cases'][1]))):
            changed=deepcopy(state);mutate(changed)
            with self.subTest(mutation=mutate):self.assertEqual(evaluate.finish(changed),1)

    def test_pilot_main_launches_only_explicit_four_cases_with_zero_success(self):
        self.args.runtime_root.mkdir();argv=[]
        for name in ('deployment','data-root','training-manifest','output','runtime-root'):
            argv += ['--'+name,str(getattr(self.args,name.replace('-','_')))]
        argv += ['--profile',self.args.profile,'--checkpoint-step',str(self.args.checkpoint_step),
            '--expected-model-sha256',self.args.expected_model_sha256,'--rtc-mode','off',
            '--diagnostic-cases','0:6,0:0,2:7,0:4','--sampling-seed',str(self.args.sampling_seed)]
        called=[]
        def execute(args,case,frozen,state):
            called.append((case['level'],case['case_index'],case['sampling_seed']))
            state['cases'].append(dict(case,status='complete',native_success=False))
        with patch.object(evaluate,'execute',side_effect=execute),patch.object(evaluate.subprocess,'Popen') as child:
            self.assertEqual(evaluate.main(argv),0);child.assert_not_called()
        self.assertEqual(called,[(0,6,20261111),(0,0,20261105),(2,7,20261132),(0,4,20261109)])
        state=evaluate.read(self.args.output/'FINAL_EVALUATION_ACCEPTANCE.json')
        self.assertEqual(state['planned_cases'],4);self.assertEqual(state['evaluation_denominator'],4)
        self.assertEqual(state['native_successes'],0);self.assertEqual(state['contract'],evaluate.DIAGNOSTIC_CONTRACT)
        self.assertFalse(state['full_official30_claim'])

    def prepare_guidance_fixture(self):
        config=evaluate.read(self.args.deployment/'run_config.json')
        config.update(model={'action_dim':80,'action_architecture':'dual_source_v1'})
        config['data']['transform']['repack']={'action_keys':['action.body_token_v1_1','action[:14]'],
            'pad_action_dim':80,'action_mask_key':'action_mask'}
        self.write(self.args.deployment/'run_config.json',config)
        path=self.args.deployment/'run_config.json'
        self.ready['deployment_artifacts']['run_config.json']={'sha256':sha(path),'bytes':path.stat().st_size}
        hands=[side+'_hand_'+name+'_joint' for side in ('left','right') for name in
            ('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1')]
        path=self.args.deployment/'provenance/release_metadata/packed/PACK_READY.json'
        self.write(path,{'contract':'cvpr-pico-pick-lerobot-pack-v1','task_id':profile_spec(self.args.profile)['task_id'],
            'packer_sha256':evaluate.VALIDATED_PACKER_SHA256,'checkpoint_contract':{'hand_joint_names':hands}})
        self.ready['source_metadata_sha256']={'release_metadata/packed/PACK_READY.json':sha(path)}
        self.write(self.args.deployment/'LOCAL_RUN_READY.json',self.ready)

    def test_guidance_scope_is_explicit_pilot_only_and_bound_to_accepted_repack_files(self):
        self.args.rtc_guidance_action_scope='validated_repack'
        with self.assertRaisesRegex(ValueError,'explicit test_time diagnostic'):evaluate.evaluation_policy(self.args)
        self.args.diagnostic_cases='0:6,0:0,2:7,0:4';self.prepare_guidance_fixture()
        frozen=evaluate.freeze_inputs(self.args);policy=frozen['evaluation_policy'];proof=policy['rtc_guidance_action_proof']
        self.assertEqual(proof['guidance_action_mask'],[True]*78+[False]*2)
        self.assertEqual(proof['excluded_action_indices'],[78,79]);self.assertEqual(proof['guided_action_dimensions'],78)
        self.assertEqual(proof['local_run_ready_sha256'],sha(self.args.deployment/'LOCAL_RUN_READY.json'))
        self.assertEqual(proof['run_config_sha256'],sha(self.args.deployment/'run_config.json'))
        self.assertIn('guidance_pack_ready',frozen['sources'])
        evaluate.validate_guidance_proof(proof,policy)
        bad=deepcopy(proof);bad['guidance_action_mask'][0]=1
        with self.assertRaises(ValueError):evaluate.validate_guidance_proof(bad,policy)
        self.args.rtc_mode='off'
        with self.assertRaises(ValueError):evaluate.freeze_inputs(self.args)
        self.args.rtc_mode='test_time'
        path=self.args.deployment/'provenance/release_metadata/packed/PACK_READY.json';pack=evaluate.read(path)
        pack['packer_sha256']='0'*64;self.write(path,pack)
        self.ready['source_metadata_sha256']['release_metadata/packed/PACK_READY.json']=sha(path)
        self.write(self.args.deployment/'LOCAL_RUN_READY.json',self.ready)
        with self.assertRaisesRegex(ValueError,'not independently recognized'):evaluate.freeze_inputs(self.args)

    def test_validated_guidance_full_case_actual_sampler_mask_and_action_arrays_are_required(self):
        self.args.diagnostic_cases='0:0';self.args.rtc_guidance_action_scope='validated_repack';self.prepare_guidance_fixture()
        directory,c,f,result,probe=self.evidence(False,800)
        task=profile_spec(self.args.profile)['task'];epdir=directory/'simulation'/task/'episode-0'
        with patch.object(evaluate.subprocess,'check_output',return_value=json.dumps(probe).encode()):
            row=evaluate.validate_case(directory,c,f)
            self.assertEqual(row['rtc_execution']['guided_action_dimensions'],78)
            requests=evaluate.json_lines(directory/'server/requests.jsonl')
            requests[-1]['actual_rtc_guidance_action_mask'][-1]=True
            (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,requests)))
            with self.assertRaisesRegex(ValueError,'mask argument'):evaluate.validate_case(directory,c,f)
            requests[-1]['actual_rtc_guidance_action_mask'][-1]=False
            requests[-1]['normalized_prediction']=[0.]*80
            (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,requests)))
            with self.assertRaisesRegex(ValueError,'shape'):evaluate.validate_case(directory,c,f)

    def interaction_fixture(self):
        hands=[side+'_hand_'+name+'_joint' for side in ('left','right') for name in
            ('thumb_0','thumb_1','thumb_2','middle_0','middle_1','index_0','index_1')]
        contact={'categories':['target_table'],'body_names':['cracker_box','world'],'geom_names':['box','table'],
            'position_m':[0.,0.,.5],'distance_m':-.0001,'normal_force_N':1.,'force_norm_N':1.}
        initial=[{'physics_substep':1,'phase':'native_reset','solved_interval_start_s':0.,'solved_interval_end_s':.005,'contacts':[contact]}]
        ticks=[]
        for index in range(3):
            start=.005+.02*index;end=start+.02
            snapshot={'sim_time_s':end,'bodies':{name:{'xyz_m':[0.,0.,.8],'quaternion_wxyz':[1.,0.,0.,0.]}
                for name in ('pelvis','left_wrist_yaw_link','right_wrist_yaw_link')},'target_xyz_m':[0.,0.,.5],
                'joint_names':['body_'+str(i) for i in range(29)]+hands,'joint_qpos':[0.]*43,
                'sent_body_token64':[0.]*64,'sent_hand_targets':{name:0. for name in hands}}
            ticks.append({'phase':'policy_latent','sim_time_s':start,'next_sim_time_s':end,
                'chunk_start_sim_time_s':.005,'action_index_at_30hz':int(index*.02*30+1e-7),
                'pick_native_state':{'target_xyz_m':[0.,0.,.5]},'pick_interaction_substeps':[],
                'pick_measured_control':snapshot})
        episode={'initial_solved_contacts':initial,'pick_interaction_observer':{
            'contract':'cvpr-pick-passive-interaction-observer-v1',
            'contact_phase':'query existing solved contacts after the real mj_step; no mj_forward',
            'force_count_semantics':'positive-force contact points per physical substep; not unique collision events',
            'observed_physics_substeps':13,'solved_interval_end_s':.065,
            'live_state_writes':False,'physics_steps_added':0,'controller_commands_modified':False,
            'counts':{'native_reset:target_table':1},'first_contacts':{'native_reset:target_table':{
                'solved_interval_end_s':.005,'solved_interval_start_s':0.,'physics_substep':1,**contact}}}}
        requests=[{'status':200,'sim_time_s':'.005','denormalized_returned_action':[[0.]*80 for _ in range(15)]}]
        return episode,ticks,requests

    def test_passive_interaction_audit_checks_physical_count_real_tick_commands_and_no_intervention(self):
        episode,ticks,requests=self.interaction_fixture()
        accepted=evaluate.validate_interaction_observation(episode,ticks,requests)
        self.assertEqual(accepted['observed_physics_substeps'],13)
        for mutate in (lambda e,t:e['pick_interaction_observer'].update(observed_physics_substeps=12),
            lambda e,t:e['pick_interaction_observer'].update(live_state_writes=True),
            lambda e,t:t[0]['pick_measured_control']['sent_body_token64'].__setitem__(0,1.),
            lambda e,t:t[0]['pick_measured_control'].update(sent_body_token64=None),
            lambda e,t:t[0]['pick_measured_control']['sent_hand_targets'].pop(next(iter(t[0]['pick_measured_control']['sent_hand_targets']))),
            lambda e,t:e['initial_solved_contacts'][0].update(physics_substep=2),
            lambda e,t:e['pick_interaction_observer']['counts'].update(wrong=1)):
            e,t=deepcopy(episode),deepcopy(ticks);mutate(e,t)
            with self.assertRaises(ValueError):evaluate.validate_interaction_observation(e,t,requests)
        self.args.pick_interaction_diagnostics=True
        with self.assertRaises(ValueError):evaluate.evaluation_policy(self.args)
        self.args.diagnostic_cases='0:0';c=evaluate.freeze_inputs(self.args)['cases'][0]
        self.assertIn('--pick-interaction-diagnostics',evaluate.commands(self.args,c,self.root/'case')['simulator'])

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
        for request in filtered:
            request.update(rtc_guidance_action_scope=f['evaluation_policy']['rtc_guidance_action_scope'],
                rtc_guidance_action_proof=deepcopy(f['evaluation_policy']['rtc_guidance_action_proof']))
            if request['status']==200:
                request.update(normalized_prediction=[[[0.]*80 for _ in range(30)]],
                    denormalized_returned_action=[[0.]*80 for _ in range(15)])
                if request['sampling_path']=='test_time':
                    request.update(normalized_shifted_previous=[[[0.]*80 for _ in range(30)]],
                        actual_rtc_guidance_action_mask=deepcopy(f['evaluation_policy']['rtc_guidance_action_proof'].get('guidance_action_mask')))
        result['server_info'].update(rtc_guidance_action_scope=f['evaluation_policy']['rtc_guidance_action_scope'],
            rtc_guidance_action_proof=deepcopy(f['evaluation_policy']['rtc_guidance_action_proof']))
        self.write(epdir/'http.json',http);(epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps,ticks)))
        (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,filtered)))
        self.write(epdir.parent/'result.json',result)
        startup={'weight_sha256':self.args.expected_model_sha256,'pid':123,'sampling_seed':c['sampling_seed'],'Ta':15,'Tp':30,
            'rtc':True,'inference_steps':10,'online_dino_runtime':encoder(),'parameter_tensor_count':12,
            'trainable_parameter_tensor_count':0,'parameter_gradient_tensor_count':0,
            'rtc_guidance_action_scope':f['evaluation_policy']['rtc_guidance_action_scope'],
            'rtc_guidance_action_proof':deepcopy(f['evaluation_policy']['rtc_guidance_action_proof'])}
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

    def test_off_pilot_actual_plain_sampler_full800tick_ledger_and_no_hidden_rtc(self):
        self.args.diagnostic_cases='0:6,0:0,2:7,0:4';self.args.rtc_mode='off'
        directory,c,f,result,probe=self.evidence(False,800)
        task=profile_spec(self.args.profile)['task'];epdir=directory/'simulation'/task/'episode-0'
        requests=evaluate.json_lines(directory/'server/requests.jsonl')
        for row in requests:
            if row['status']==200:row.update(sampling_path='plain',sampler_calls={'plain':1,'test_time':0,'train':0})
        result['server_info'].update(rtc_enabled=False,rtc_mode='off')
        self.write(epdir.parent/'result.json',result)
        startup=evaluate.read(directory/'server/startup.json');startup.update(rtc=False,rtc_mode='off')
        self.write(directory/'server/startup.json',startup)
        (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,requests)))
        with patch.object(evaluate.subprocess,'check_output',return_value=json.dumps(probe).encode()):
            row=evaluate.validate_case(directory,c,f)
            self.assertFalse(row['native_success']);self.assertEqual(row['evaluation_denominator'],4)
            self.assertEqual((row['rtc_execution']['plain_requests'],row['rtc_execution']['test_time_requests']),(30,0))
            requests[-1].update(sampling_path='test_time',sampler_calls={'plain':0,'test_time':1,'train':0})
            (directory/'server/requests.jsonl').write_text('\n'.join(map(json.dumps,requests)))
            with self.assertRaisesRegex(ValueError,'actually use exactly one plain'):evaluate.validate_case(directory,c,f)

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
