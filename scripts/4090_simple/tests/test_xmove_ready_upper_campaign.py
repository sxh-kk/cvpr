"""Separate ready-pose protocol; default RTC/scene paths retain their behavior."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys,unittest,tempfile,contextlib,io,json
import test_evaluate_pick30 as base_tests
import run_closed_loop as worker
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_pick30 as e
import xmove_ready_upper_warmup as ready
from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES,SIMPLE_HAND_JOINT_NAMES


def args():
 return SimpleNamespace(profile='xmove_pick',checkpoint_step=5000,rtc_mode='test_time',execute_steps=15,
  sampling_seed=20261105,diagnostic_cases=e.READY_UPPER_CASES,rtc_guidance_action_scope='all',pick_interaction_diagnostics=True,
  pick_warmup_profile='ready_upper_v1',ready_planner_model=Path('/fixed/planner.onnx'),
  deployment=Path('/deployment'),runtime_root=Path('/runtime'),data_root=Path('/data'),port=21155)


def evidence():
 names=list(BODY_JOINT_NAMES)+list(SIMPLE_HAND_JOINT_NAMES);hands=list(SIMPLE_HAND_JOINT_NAMES)
 source='a'*64;model='b'*64;encoder='c'*64
 frozen=dict(contract=e.READY_UPPER_CONTRACT,experiment_role=e.READY_UPPER_ROLE,profile='xmove_pick',checkpoint_step=5000,
  warmup_profile='ready_upper_v1',evaluation_policy=e.evaluation_policy(args()),sources={
   'ready/xmove_ready_upper_warmup.py':{'sha256':source},'ready/planner_model':{'sha256':model,'path':'/fixed/planner.onnx'},
   'ready/resource/model_encoder.onnx':{'sha256':encoder}})
 result=dict(pick_warmup_profile='ready_upper_v1',original_official_initialization=False,
  native_saved_scene_initialization_preserved=True,warmup_protocol_modified=True)
 warm=dict(kind=ready.PROFILE,initialization_profile=ready.PROFILE,fixed_ready_joint_targets_rad=dict(ready.READY_JOINTS),
  helper_source_sha256=source,planner_model_sha256=model,planner_model_path='/fixed/planner.onnx',encoder_sha256=encoder,
  planner_seed=20261027,private_fk=True,physics_steps_added=0,initialization_count=1,encoder_calls=40,encoder_mode=1,
  warmup_budget_control_ticks=50,initial_joint_names43=names,initial_measured_joint_qpos43=[0.]*43,
  first_actual_root_pose7=[-.8,0.,.79,1.,0.,0.,0.],first_actual_body_q29=[0.]*29,
  hand_target_joint_names=hands,first_actual_measured_hands14=[0.]*14,ready_reference_events=[],events=[])
 for field in ('PICO_or_teacher_data_read','live_physical_state_writes','live_state_writes','object_pose_inputs_used',
  'target_or_scene_or_reward_inputs_used','cold_DEFAULT_initialization','teacher_reference_inputs_used',
  'policy_phase_interventions','state_forcing','history_injection'):warm[field]=False
 rows=[]
 for i in range(50):
  t=.005+.02*i;alpha=ready.blend(i);token=[.25]*64
  warm['ready_reference_events'].append(dict(control_tick=i,sim_time_s=t,blend=alpha,
   private_fk_only=True,live_qpos_writes=False,virtual_ready_targets_rad={n:alpha*v for n,v in ready.READY_JOINTS.items()},
   local_points_m=[[0.,0.,0.]]*3,local_quaternions_wxyz=[[1.,0.,0.,0.]]*3))
  rows.append(dict(sim_time_s=t,fell=False,pick_interaction_substeps=[],pick_measured_control=dict(joint_names=names,
   sent_body_token64=None if i<10 else token,sent_hand_targets={n:0. for n in hands})))
  if i>=10:warm['events'].append(dict(sim_time_s=t,reference_evidence={'encoder_mode':1},state_forcing=False,
   history_injection=False,token64=token,hand_targets14=[0.]*14))
 episode=dict(result,warmup_frames=50,fell_before_first_policy_command=False,initial_pelvis_position_m=[-.8,0.,.79],
  warmup_control=warm,initial_solved_contacts=[])
 return result,episode,rows,frozen


class ReadyCampaignTests(unittest.TestCase):
 def test_only_predeclared_four_same_seed_5000_testtime_all_factor(self):
  a=args();self.assertEqual(e.evaluation_policy(a)['rtc_mode'],'test_time')
  for field,value in [('profile','xmove_bend_pick'),('checkpoint_step',10000),('sampling_seed',20261106),('rtc_mode','off'),
   ('rtc_guidance_action_scope','validated_repack'),('execute_steps',30),('diagnostic_cases',None),
   ('diagnostic_cases',((0,0),)),('pick_interaction_diagnostics',False)]:
   with self.subTest(field=field,value=value):
    b=deepcopy(a);setattr(b,field,value)
    with self.assertRaises(ValueError):e.evaluation_policy(b)

 def test_worker_cli_rejects_foreign_profiles_mixed_controls_and_missing_planner(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);model=root/'planner.onnx';model.write_bytes(b'fixture only')
   command=['worker','--data-root',directory,'--sonic-resources',directory,'--output',str(root/'out'),
    '--tasks','G1SonicXMovePickPico-v0','--pick-official-protocol','--pick-profile','xmove_pick',
    '--sync-observations','--isaac-observation-only','--sim-mode','mujoco_isaac','--seconds','16',
    '--expected-checkpoint-step','5000','--execute-steps','15','--rtc-mode','test_time',
    '--warmup-control','live_pose_sonic_v1_1','--pd-profile','official_v1_1_ankle_pitch_1_5','--pick-interaction-diagnostics',
    '--pick-warmup-profile','ready_upper_v1','--pick-ready-planner-model',str(model)]
   with patch.object(sys,'argv',command):self.assertEqual(worker.parse_args().pick_warmup_profile,'ready_upper_v1')
   for option,value in [('--expected-checkpoint-step','10000'),('--rtc-mode','off'),('--execute-steps','30'),
       ('--warmup-control','measured_pose_pd'),('--pd-profile','fixed_cpp_baseline'),('--seconds','17')]:
    bad=command.copy();bad[bad.index(option)+1]=value
    with self.subTest(option=option),patch.object(sys,'argv',bad),contextlib.redirect_stderr(io.StringIO()):
     with self.assertRaises(SystemExit):worker.parse_args()
   for flag in ['--pick-interaction-diagnostics','--pick-ready-planner-model']:
    bad=command.copy();i=bad.index(flag);del bad[i:i+(2 if flag.endswith('model') else 1)]
    with self.subTest(flag=flag),patch.object(sys,'argv',bad),contextlib.redirect_stderr(io.StringIO()):
     with self.assertRaises(SystemExit):worker.parse_args()

 def test_ready_worker_checks_actual_server_scope_before_physics(self):
  a=SimpleNamespace(execute_steps=15,rtc_mode='test_time',rtc_inference_delay=0,min_exec_horizon=15,
   pig_mask_schedule='exponential',pig_guidance_alpha=.9,expected_checkpoint_step=5000,pick_warmup_profile='ready_upper_v1')
  actual=dict(worker.requested_rtc_contract(a),ckpt_step=5000,action=dict(action_dim=80,action_chunk_size=30,
   action_exec_horizon=15),online_history=dict(frames=6),rtc_guidance_action_scope='all')
  self.assertEqual(worker.validate_server_info(actual,a)['status'],'passed')
  actual['rtc_guidance_action_scope']='validated_repack'
  with self.assertRaises(ValueError):worker.validate_server_info(actual,a)

 def test_default_commands_do_not_import_or_enable_ready_and_candidate_only_adds_explicit_flags(self):
  a=args();case=dict(level=0,case_index=6,sampling_seed=20261111,protocol={'maximum_simulated_seconds':16.})
  native=deepcopy(a);native.pick_warmup_profile='native';native.ready_planner_model=None
  before=e.commands(native,case,Path('/out'));after=e.commands(a,case,Path('/out'))
  self.assertEqual(before['server'],after['server'])
  self.assertNotIn('--pick-warmup-profile',before['simulator'])
  self.assertEqual(after['simulator'][:-4],before['simulator'])
  self.assertEqual(after['simulator'][-4:],['--pick-warmup-profile','ready_upper_v1','--pick-ready-planner-model','/fixed/planner.onnx'])

 def test_actual_reference_and_sent_command_binding_and17_tamper_rejections(self):
  result,episode,rows,frozen=evidence()
  self.assertEqual(e.validate_ready_upper_evidence(result,episode,rows,frozen)['status'],'passed')
  mutations=[lambda r,x,t,f:x.update(original_official_initialization=True),
   lambda r,x,t,f:x['warmup_control'].update(teacher_reference_inputs_used=True),
   lambda r,x,t,f:x['warmup_control'].update(cold_DEFAULT_initialization=True),
   lambda r,x,t,f:x['warmup_control'].update(live_state_writes=True),
   lambda r,x,t,f:x['warmup_control'].update(physics_steps_added=1),
   lambda r,x,t,f:x['warmup_control'].update(planner_seed=123),
   lambda r,x,t,f:x['warmup_control'].update(encoder_calls=39),
   lambda r,x,t,f:x['warmup_control'].update(ready_reference_events=x['warmup_control']['ready_reference_events'][:-1]),
   lambda r,x,t,f:x['warmup_control']['ready_reference_events'][20].update(blend=.8),
   lambda r,x,t,f:x['warmup_control']['ready_reference_events'][49]['virtual_ready_targets_rad'].update(left_elbow_joint=.7),
   lambda r,x,t,f:x['warmup_control'].update(helper_source_sha256='f'*64),
   lambda r,x,t,f:x['warmup_control'].update(planner_model_path='/other/planner.onnx'),
   lambda r,x,t,f:t[10]['pick_measured_control'].update(sent_body_token64=[.2]*64),
   lambda r,x,t,f:t[2]['pick_measured_control'].update(sent_body_token64=[.2]*64),
   lambda r,x,t,f:t[3].update(fell=True),
   lambda r,x,t,f:t[12].update(pick_interaction_substeps=[{'contacts':[{'categories':['robot_self']}]}]),
   lambda r,x,t,f:f['evaluation_policy'].update(rtc_mode='off')]
  for index,mutate in enumerate(mutations):
   with self.subTest(index=index):
    r,x,t,f=deepcopy((result,episode,rows,frozen));mutate(r,x,t,f)
    with self.assertRaises(ValueError):e.validate_ready_upper_evidence(r,x,t,f)

 def test_complete_validate_case_keeps_native_result_and_audits_changed_warmup(self):
  fixture=base_tests.PickCampaignTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
  fixture.args.checkpoint_step=5000;fixture.prepare()
  directory,case,frozen,result,probe=fixture.evidence(False,800)
  template,episode,tick_refs,reference=evidence()
  frozen.update(contract=e.READY_UPPER_CONTRACT,experiment_role=e.READY_UPPER_ROLE,warmup_profile='ready_upper_v1',
   original_official_initialization=False,native_saved_scene_initialization_preserved=True,warmup_protocol_modified=True,
   full_official30_claim=False,planned_cases=4,evaluation_denominator=4,interaction_diagnostics_requested=True)
  frozen['sources'].update(reference['sources'])
  frozen['diagnostic_selected_cases']=[{k:frozen['cases'][a*10+b][k]for k in ('level','case_index','flat_index','sampling_seed')}
   for a,b in e.READY_UPPER_CASES]
  task=e.profile_spec('xmove_pick')['task'];epdir=directory/'simulation'/task/'episode-0'
  rows=e.json_lines(epdir/'trace.jsonl');actual=result['episodes'][0]
  result.update(template);actual.update(template,fell=False,fell_before_first_policy_command=False,warmup_control=episode['warmup_control'],
   initial_pelvis_position_m=episode['initial_pelvis_position_m'],initial_solved_contacts=[],
   pick_interaction_observer=dict(contract='cvpr-pick-passive-interaction-observer-v1',observed_physics_substeps=3201,
    solved_interval_end_s=16.005,live_state_writes=False,physics_steps_added=0,controller_commands_modified=False,
    contact_phase='query existing solved contacts after the real mj_step; no mj_forward',
    force_count_semantics='positive-force contact points per physical substep; not unique collision events',counts={},first_contacts={}))
  names=episode['warmup_control']['initial_joint_names43'];hands=list(SIMPLE_HAND_JOINT_NAMES)
  for index,row in enumerate(rows):
   row['phase']='warmup_measured_pose_pd'if index<10 else 'warmup_ready_upper_v1'if index<50 else 'policy_latent'
   row['fell']=False
   row['pick_interaction_substeps']=[]
   row['pick_measured_control']=dict(sim_time_s=row['next_sim_time_s'],
    bodies={n:{'xyz_m':[0.,0.,.8],'quaternion_wxyz':[1.,0.,0.,0.]}for n in ('pelvis','left_wrist_yaw_link','right_wrist_yaw_link')},
    target_xyz_m=row['pick_native_state']['target_xyz_m'],joint_names=names,joint_qpos=[0.]*43,
    sent_body_token64=None if index<10 else [.25]*64 if index<50 else [0.]*64,sent_hand_targets={n:0. for n in hands})
  fixture.write(epdir.parent/'result.json',result);(epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps,rows)))
  with patch.object(e.subprocess,'check_output',return_value=json.dumps(probe).encode()):
   summary=e.validate_case(directory,case,frozen)
   self.assertFalse(summary['native_success']);self.assertEqual(summary['ticks'],800)
   self.assertFalse(summary['original_official_initialization']);self.assertEqual(summary['contract'],e.READY_UPPER_CONTRACT)
   self.assertEqual(summary['ready_upper_acceptance']['status'],'passed')
   rows[12]['pick_measured_control']['sent_body_token64'][0]=.75
   (epdir/'trace.jsonl').write_text('\n'.join(map(json.dumps,rows)))
   with self.assertRaisesRegex(ValueError,'Sent startup body64'):e.validate_case(directory,case,frozen)

 def test_diagnostic_selection_cannot_be_labeled_old_rtc_pilot(self):
  source=[dict(level=l,case_index=c,flat_index=l*10+c,sampling_seed=20261105+l*10+c)for l in range(3)for c in range(10)]
  chosen=[source[l*10+c]for l,c in e.READY_UPPER_CASES]
  frozen=dict(contract=e.READY_UPPER_CONTRACT,experiment_role=e.READY_UPPER_ROLE,warmup_profile='ready_upper_v1',
   original_official_initialization=False,native_saved_scene_initialization_preserved=True,full_official30_claim=False,
   cases=source,diagnostic_selected_cases=chosen,evaluation_denominator=4,planned_cases=4)
  self.assertEqual(e.execution_cases(frozen),chosen)
  frozen['original_official_initialization']=True
  with self.assertRaises(ValueError):e.execution_cases(frozen)

if __name__=='__main__':unittest.main()
