"""Fixed failed7 diagnostic: real factory TimeLimit, strict CLI and audited1200-tick fixtures.

No model, renderer, GPU, training state, teacher action or actual evaluation is run.
"""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib,io,json,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import gymnasium as gym
import numpy as np
import evaluate_pick30 as e
import pick_eval_protocol as protocol
import run_closed_loop as worker
import test_evaluate_pick30 as base
from test_pick_eval_protocol import fixture as protocol_fixture


class Counter(gym.Env):
 def __init__(self,**kwargs):
  self.observation_space=gym.spaces.Box(-1,1,(1,),dtype=np.float32);self.action_space=gym.spaces.Discrete(1);self.steps=0
 def reset(self,*,seed=None,options=None):
  super().reset(seed=seed);self.steps=0;return np.zeros(1,dtype=np.float32),{}
 def step(self,action):
  self.steps+=1;return np.zeros(1,dtype=np.float32),0.,False,False,{}


class ExtendedPickTests(unittest.TestCase):
 def fixture(self):
  f=base.PickCampaignTests();f.setUp();self.addCleanup(f.doCleanups)
  a=f.args;a.profile='xmove_bend_pick';a.checkpoint_step=5000;f.prepare()
  a.pick_budget_profile=protocol.EXTENDED_BUDGET_PROFILE;a.diagnostic_cases=protocol.EXTENDED_BUDGET_CASES
  a.rtc_guidance_action_scope='all';a.pick_warmup_profile='native'
  return f

 def test_native_protocol_bytes_unchanged_and_explicit_override_never_changes_source(self):
  for profile in protocol.PROFILES:
   source=protocol_fixture(profile);original=deepcopy(source)
   self.assertEqual(protocol.pick_evaluation_protocol(source,profile),protocol.pick_official_protocol(source,profile))
   if profile=='xmove_bend_pick':
    for missing in (False,True):
     if missing:source['metadata'].pop('max_episode_steps')
     before=deepcopy(source);p=protocol.pick_evaluation_protocol(source,profile,protocol.EXTENDED_BUDGET_PROFILE)
     self.assertEqual((p['max_episode_steps'],p['maximum_simulated_seconds']),(1200,24.))
     self.assertEqual((p['native_max_episode_steps'],p['warmup_control_ticks'],p['effective_success_criteria']),(800,50,.9))
     self.assertTrue(p['official_budget_modified']);self.assertFalse(p['source_metadata_modified']);self.assertEqual(source,before)
   else:
    with self.assertRaises(ValueError):protocol.pick_evaluation_protocol(source,profile,protocol.EXTENDED_BUDGET_PROFILE)
   self.assertEqual(original['layout'],source['layout'])
  with self.assertRaises(ValueError):protocol.pick_evaluation_protocol(protocol_fixture(),'xmove_pick','arbitrary_30s')

 def test_real_private_factory_returns_actual1200_TimeLimit_with_native800_default(self):
  from simple.envs import sonic_loco_manip
  source=protocol_fixture('xmove_bend_pick')
  adapter=SimpleNamespace(make_xmove_bend_pick_task=lambda *args,**kwargs:object())
  for profile,steps in ((protocol.NATIVE_BUDGET_PROFILE,800),(protocol.EXTENDED_BUDGET_PROFILE,1200)):
   p=protocol.pick_evaluation_protocol(source,'xmove_bend_pick',profile)
   with patch.object(sonic_loco_manip,'SonicLocoManipEnv',Counter),patch.object(protocol,'_adapter',return_value=adapter),    patch.object(protocol,'prepare_pick_reset',return_value=(deepcopy(source),{}, {},p)),patch.object(protocol,'install_pick_initialization_monitor'):
    env,receipt=protocol.make_pick_environment({},source,'xmove_bend_pick',sim_mode='mujoco_isaac',
     sync_observations=True,isaac_observation_only=True,budget_profile=profile)
   self.addCleanup(env.close);self.assertIsInstance(env,gym.wrappers.TimeLimit);env.reset()
   self.assertEqual(env._max_episode_steps,steps)
   for tick in range(steps):self.assertEqual(env.step(0)[3],tick==steps-1)
   self.assertEqual(env.unwrapped.steps,steps)
   if steps==1200:
    ledger=protocol.extended_budget_runtime_receipt(env,p,24.)
    self.assertEqual(ledger['actual_environment_elapsed_control_ticks'],1200)
    with self.assertRaises(ValueError):protocol.extended_budget_runtime_receipt(env,p,16.)
    env._max_episode_steps=800
    with self.assertRaises(ValueError):protocol.extended_budget_runtime_receipt(env,p,24.)

 def test_worker_cli_exact24_fixed7_native_warmup_and_actualall80_before_physics(self):
  with tempfile.TemporaryDirectory()as directory:
   c=['worker','--data-root',directory,'--sonic-resources',directory,'--output',directory+'/out',
    '--tasks','G1SonicXMoveBendPickPico-v0','--pick-official-protocol','--pick-profile','xmove_bend_pick',
    '--pick-budget-profile',protocol.EXTENDED_BUDGET_PROFILE,'--seconds','24','--episode-start','5','--dr-level','0',
    '--sync-observations','--isaac-observation-only','--sim-mode','mujoco_isaac','--expected-checkpoint-step','5000',
    '--execute-steps','15','--rtc-mode','test_time','--warmup-control','live_pose_sonic_v1_1',
    '--warmup-hand-policy','measured','--pd-profile','official_v1_1_ankle_pitch_1_5','--hand-torque-profile','mjcf_limits']
   with patch.object(sys,'argv',c):args=worker.parse_args();self.assertEqual(args.seconds,24.)
   for flag,value in [('--seconds','16'),('--seconds','23'),('--seconds','25'),('--episode-start','4'),
       ('--expected-checkpoint-step','10000'),('--seed','1'),('--rtc-mode','off'),('--execute-steps','30'),
       ('--warmup-control','measured_pose_pd'),('--pick-warmup-profile','ready_upper_v1')]:
    bad=c.copy()
    if flag in bad:bad[bad.index(flag)+1]=value
    else:bad.extend([flag,value])
    with self.subTest(flag=flag,value=value),patch.object(sys,'argv',bad),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):worker.parse_args()
   actual=dict(worker.requested_rtc_contract(args),ckpt_step=5000,action=dict(action_dim=80,action_chunk_size=30,action_exec_horizon=15),
    online_history={'frames':6},rtc_guidance_action_scope='all')
   self.assertEqual(worker.validate_server_info(actual,args)['status'],'passed')
   actual['rtc_guidance_action_scope']='validated_repack'
   with self.assertRaises(ValueError):worker.validate_server_info(actual,args)

 def test_campaign_rejects_mixed_mode_reordered_cases_other_saves_and_any_other_budget(self):
  f=self.fixture();a=f.args;self.assertEqual(e.evaluation_policy(a)['rtc_mode'],'test_time')
  for name,value in [('profile','xmove_pick'),('checkpoint_step',10000),('sampling_seed',20261106),
    ('rtc_mode','off'),('rtc_guidance_action_scope','validated_repack'),('execute_steps',30),
    ('diagnostic_cases',None),('diagnostic_cases',tuple(reversed(protocol.EXTENDED_BUDGET_CASES))),
    ('diagnostic_cases',protocol.EXTENDED_BUDGET_CASES[:-1]),('pick_warmup_profile','ready_upper_v1'),
    ('pick_budget_profile','other_24s')]:
   bad=deepcopy(a);setattr(bad,name,value)
   with self.subTest(name=name),self.assertRaises(ValueError):e.evaluation_policy(bad)

 def test_freeze_keeps_all30_original_scenes_but_executes_only7_and_transmits_both_budget_gates(self):
  f=self.fixture();a=f.args;native_args=deepcopy(a);native_args.pick_budget_profile=protocol.NATIVE_BUDGET_PROFILE
  native=e.freeze_inputs(native_args);extended=e.freeze_inputs(a)
  self.assertEqual((extended['contract'],extended['experiment_role'],extended['planned_cases']),
   (protocol.EXTENDED_BUDGET_CONTRACT,protocol.EXTENDED_BUDGET_ROLE,7))
  self.assertEqual(extended['source_cases_frozen'],30);self.assertFalse(extended['full_official30_claim'])
  self.assertTrue(extended['original_official_initialization']);self.assertTrue(extended['official_budget_modified'])
  for x,y in zip(native['cases'],extended['cases']):
   for name in ('source_environment_config','local_environment_config','reset_derivation','adaptation','sampling_seed'):
    self.assertEqual(x[name],y[name])
   self.assertEqual(y['protocol']['max_episode_steps'],1200)
  selected=e.execution_cases(extended);self.assertEqual(tuple((c['level'],c['case_index'])for c in selected),protocol.EXTENDED_BUDGET_CASES)
  for c in selected:
   command=e.commands(a,c,Path('/out'))['simulator']
   self.assertEqual(command[command.index('--seconds')+1],'24.0')
   self.assertEqual(command[command.index('--pick-budget-profile')+1],protocol.EXTENDED_BUDGET_PROFILE)
  default=e.commands(native_args,native['cases'][5],Path('/out'))['simulator']
  self.assertEqual(default[default.index('--seconds')+1],'16.0');self.assertNotIn('--pick-budget-profile',default)
  bad=deepcopy(extended);bad['diagnostic_selected_cases'].reverse()
  with self.assertRaises(ValueError):e.execution_cases(bad)

 def evidence(self,native,count):
  f=self.fixture();a=f.args;extended=e.freeze_inputs(a)
  # Reuse the existing real RTC/HTTP/schema fixture producer without changing its native defaults.
  native_args=deepcopy(a);native_args.pick_budget_profile=protocol.NATIVE_BUDGET_PROFILE
  native_frozen=e.freeze_inputs(native_args)
  with patch.object(e,'freeze_inputs',return_value=native_frozen):directory,_,_,result,probe=f.evidence(native,count)
  case=extended['cases'][5];task=protocol.profile_spec(a.profile)['task']
  old=directory/'simulation'/task/'episode-0';ep=old.with_name('episode-5');old.rename(ep)
  episode=result['episodes'][0];episode.update(episode_index=5,pick_protocol=case['protocol'],
   pick_source_environment_config=case['source_environment_config'],pick_reset_derivation=case['reset_derivation'],
   pick_adaptation=case['adaptation'],local_environment_config=case['local_environment_config'])
  episode['video']['path']=str(ep/'rollout.mp4')
  episode['pick_extended_budget']=protocol.extended_budget_runtime_receipt(SimpleNamespace(_max_episode_steps=1200,_elapsed_steps=count),case['protocol'],24.)
  result['pick_budget_profile']=protocol.EXTENDED_BUDGET_PROFILE;result['pick_adapter']['protocol']=case['protocol']
  f.write(ep.parent/'result.json',result)
  startup=e.read(directory/'server/startup.json');startup['sampling_seed']=case['sampling_seed'];f.write(directory/'server/startup.json',startup)
  return f,directory,case,extended,result,probe,ep

 def test_end_to_end_validate_case_actual800_801_success_and1200_failure_plus_budget_pollution(self):
  for native,count,label in ((True,800,'native_success_within_original_16s'),(True,801,'native_success_during_extra_8s'),
    (True,1200,'native_success_during_extra_8s'),(False,1200,'no_native_success_by_24s')):
   f,directory,case,frozen,result,probe,ep=self.evidence(native,count)
   if native and count==1200:
    terminal=e.json_lines(ep/'trace.jsonl');terminal[-1]['truncated']=True
    (ep/'trace.jsonl').write_text('\n'.join(map(json.dumps,terminal)))
   with patch.object(e.subprocess,'check_output',return_value=json.dumps(probe).encode()):
    row=e.validate_case(directory,case,frozen)
    self.assertEqual(row['extended_budget_outcome'],label)
    self.assertEqual(row['first_native_success_control_tick'],count if native else None)
    if native:self.assertAlmostEqual(row['first_native_success_elapsed_s'],count*.02)
    else:self.assertIsNone(row['first_native_success_elapsed_s'])
    for field,value in [('environment_time_limit_steps',800),('supervisor_cutoff_seconds',16.),
      ('actual_environment_elapsed_control_ticks',count-1),('official_budget_modified',False)]:
     bad=deepcopy(result);bad['episodes'][0]['pick_extended_budget'][field]=value;f.write(ep.parent/'result.json',bad)
     with self.subTest(field=field),self.assertRaises(ValueError):e.validate_case(directory,case,frozen)
    f.write(ep.parent/'result.json',result)
    if not native:
     # A legacy800 tick truncation cannot pass as a completed24s retest.
     ticks=e.json_lines(ep/'trace.jsonl');ticks=ticks[:800];ticks[-1]['truncated']=True
     (ep/'trace.jsonl').write_text('\n'.join(map(json.dumps,ticks)))
     with self.assertRaises(ValueError):e.validate_case(directory,case,frozen)

 def test_finish_classifies_this_run_only_and_never_sums_original23_as_measured30(self):
  f=self.fixture();frozen=e.freeze_inputs(f.args)
  state={k:deepcopy(v)for k,v in frozen.items()if k!='cases'};state['cases']=[]
  for source,count in zip(e.execution_cases(frozen),(800,900,None,700,None,1100,801)):
   native=count is not None;n=count if native else 1200
   row=dict(level=source['level'],case_index=source['case_index'],sampling_seed=source['sampling_seed'],status='complete',
    native_success=native,ticks=n,first_native_success_control_tick=n if native else None,
    first_native_success_elapsed_s=n*.02 if native else None,extended_budget_outcome=(
     'native_success_within_original_16s'if n<=800 else'native_success_during_extra_8s')if native else'no_native_success_by_24s')
   state['cases'].append(row)
  self.assertEqual(e.finish(state),0)
  self.assertEqual((state['within_original_16s_successes'],state['extra_8s_successes'],state['no_native_success_by_24s']),(2,3,2))
  self.assertEqual(state['native_successes'],5);self.assertEqual(state['evaluation_denominator'],7)
  self.assertFalse(state['extra_8s_causal_gain_claim']);self.assertFalse(state['can_combine_with_original23_as_measured30'])
  state['cases'][0]['extended_budget_outcome']='native_success_during_extra_8s'
  with self.assertRaises(ValueError):e.finish(state)

 def test_main_dispatches_exact7_native_seed_without_real_children_or_full30_claim(self):
  f=self.fixture();a=f.args;a.runtime_root.mkdir();argv=[]
  for name in ('deployment','data-root','training-manifest','output','runtime-root'):
   argv.extend(['--'+name,str(getattr(a,name.replace('-','_')))])
  argv+=['--profile',a.profile,'--checkpoint-step','5000','--sampling-seed','20261105',
   '--expected-model-sha256',a.expected_model_sha256,'--pick-budget-profile',protocol.EXTENDED_BUDGET_PROFILE,
   '--diagnostic-cases',','.join(f'{l}:{c}'for l,c in protocol.EXTENDED_BUDGET_CASES)]
  called=[]
  def fake_execute(args,case,frozen,state):
   called.append((case['level'],case['case_index'],case['sampling_seed']))
   state['cases'].append(dict(level=case['level'],case_index=case['case_index'],sampling_seed=case['sampling_seed'],status='complete',
    native_success=False,ticks=1200,first_native_success_elapsed_s=None,first_native_success_control_tick=None,
    extended_budget_outcome='no_native_success_by_24s'))
  with patch.object(e,'execute',side_effect=fake_execute),patch.object(e.subprocess,'Popen')as children:
   self.assertEqual(e.main(argv),0);children.assert_not_called()
  self.assertEqual(called,[(l,c,20261105+l*10+c)for l,c in protocol.EXTENDED_BUDGET_CASES])
  final=e.read(a.output/'FINAL_EVALUATION_ACCEPTANCE.json')
  self.assertEqual(final['contract'],protocol.EXTENDED_BUDGET_CONTRACT);self.assertEqual(final['evaluation_denominator'],7)
  self.assertFalse(final['full_official30_claim']);self.assertEqual(final['maximum_steps_including_warmup'],1200)


if __name__=='__main__':unittest.main()
