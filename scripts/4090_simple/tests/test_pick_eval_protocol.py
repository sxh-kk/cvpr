"""CPU contracts and all real official moving Pick saved reset constructors."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pick_eval_protocol as pick
import run_closed_loop as loop

DATA = Path('/home/zxc/CVPR_Evaluation_runtime/resources/simple')


def fixture(profile='xmove_pick'):
    pose = lambda x,y,z: {'position': [x,y,z], 'quaternion': [1.,0.,0.,0.]}
    top = .58 if profile == 'xmove_pick' else .45
    actors = {'table': {'size': [1.25,.79,.1], 'pose': pose(.3,0.,top-.05), 'material': {'unchanged':True}}}
    spatial = {'g1_sonic': pose(-.95,0.,0.), 'unused': pose(3.,4.,5.)}
    for index, uid in enumerate(('0','6','7','8')):
        name = 'target' if index == 0 else 'distractor_'+str(index-1)
        p = pose(-.28+index*.1,-.05,top+.1)
        actors[name] = {'uid':uid, 'asset': {'uid':uid,'res_id':'graspnet1b'}, 'pose':p, 'material':{'roughness':.3}}
        spatial[uid] = deepcopy(p)
    table = deepcopy(actors['table'])
    return {'uid': pick.PROFILES[profile]['uid'], 'metadata': {'physics_dt':.005,'render_hz':50,'control_hz':200,
        'max_episode_steps':800,'split':'train'}, 'robot_cfg': {'uid':'g1_sonic'},
        'sensor_cfgs': {'head_stereo': {'intrinsics':None,'depth_intrinsics':None,'depth_to_color_extrinsics':None}},
        'dr_cfgs': {'spatial':{'placement_attempts':100},'scene':{'randomize_scene_pose':True},'material':{'table_material':None}},
        'dr_state_dict': {'target':{'uid':'0','res_id':'graspnet1b'},'distractors':['6','7','8'],'spatial':spatial,
            'scene':{'table':table},'material':{'all':'saved'},'lighting':{'all':'saved'}},
        'layout': {'actors':actors,'scene':{'table':deepcopy(table)},'lights':[],'cameras':{}}}


def reverse(derived, receipt):
    restored = deepcopy(derived)
    restored['dr_state_dict']['spatial'].update(receipt['removed_unused_spatial_entries'])
    for key, values in receipt['removed_optional_sensor_fields'].items(): restored['sensor_cfgs'][key].update(values)
    for key, values in receipt['removed_optional_layout_camera_fields'].items(): restored['layout']['cameras'][key]['cam_cfg'].update(values)
    for key, values in receipt['removed_compatible_dr_fields'].items(): restored['dr_cfgs'][key].update(values)
    return restored


class MovingPickProtocolTests(unittest.TestCase):
    def test_clock_native_success_and_model_movement_are_explicit_and_not_collection_joysticks(self):
        for profile in pick.PROFILES:
            original = fixture(profile); before = deepcopy(original)
            protocol = pick.pick_official_protocol(original,profile)
            self.assertEqual((protocol['max_episode_steps'],protocol['maximum_simulated_seconds']), (800,16.))
            self.assertEqual(protocol['source_metadata_control_hz'],200)
            self.assertEqual(protocol['observation_hz'],50)
            self.assertEqual(protocol['world_translation_m'],[0.,0.,0.])
            self.assertIsNone(protocol['reset_dr_level'])
            self.assertEqual(protocol['effective_success_criteria'],.9)
            self.assertEqual(protocol['evaluation_control_policy'],pick.CONTROL_POLICY)
            self.assertEqual(original,before)
            self.assertIsNone(loop.episode_reset_options(original,2,recorded_pick=True)['dr_level'])

    def test_explicit_budget_is_strict_and_missing_uses_real_source_default_without_inserting_key(self):
        for profile in pick.PROFILES:
            source=fixture(profile);source['metadata'].pop('max_episode_steps');before=deepcopy(source)
            protocol=pick.pick_official_protocol(source,profile)
            self.assertEqual(protocol['budget_source'],'unchanged_native_task_default_metadata')
            self.assertIsNone(protocol['source_metadata_max_episode_steps'])
            self.assertEqual(source,before)
            for wrong in (True,800.,None,400,1200):
                bad=fixture(profile);bad['metadata']['max_episode_steps']=wrong
                with self.assertRaises(ValueError):pick.pick_official_protocol(bad,profile)

    def test_compatibility_is_reversible_and_does_not_change_real_poses_materials_lighting(self):
        for profile in pick.PROFILES:
            original=fixture(profile);before=deepcopy(original)
            derived,receipt=pick.derive_pick_eval_reset(original,profile)
            self.assertEqual(original,before);self.assertEqual(reverse(derived,receipt),original)
            self.assertEqual(derived['layout'],original['layout'])
            self.assertEqual(derived['dr_state_dict']['material'],original['dr_state_dict']['material'])
            self.assertEqual(derived['dr_state_dict']['lighting'],original['dr_state_dict']['lighting'])
            self.assertEqual(receipt['source_config_sha256'],pick.digest(original))
            adapted,adaptation=pick.adapt_pick_config(derived,profile)
            self.assertEqual(adaptation['world_translation_m'],[0.,0.,0.])
            self.assertEqual(adapted['layout'],original['layout'])

    def test_only_allowlisted_exact_empty_or_equivalent_values_are_removed(self):
        for change in (lambda s:s['sensor_cfgs']['head_stereo'].update(intrinsics=[1,2,3]),
            lambda s:s['dr_cfgs']['spatial'].update(placement_attempts=100.),
            lambda s:s['dr_cfgs']['scene'].update(randomize_scene_pose=1),
            lambda s:s['dr_cfgs']['material'].update(table_material='wood'),
            lambda s:s['dr_state_dict']['spatial'].pop('6'),
            lambda s:s['dr_state_dict']['spatial'].update(g1_other={})):
            source=fixture();change(source)
            with self.assertRaises(ValueError):pick.derive_pick_eval_reset(source,'xmove_pick')
        source=fixture();source['dr_cfgs']['scene']['unknown_new_field']=None
        derived,_=pick.derive_pick_eval_reset(source,'xmove_pick')
        self.assertIn('unknown_new_field',derived['dr_cfgs']['scene'])

    def test_task_identity_and_world_frame_cannot_be_changed(self):
        for mutate in (lambda s:s.update(uid=pick.PROFILES['xmove_bend_pick']['uid']),
            lambda s:s['robot_cfg'].update(uid='g1_wholebody'),
            lambda s:s['metadata'].update(physics_dt=.002),lambda s:s['metadata'].update(render_hz=200)):
            source=fixture();mutate(source)
            with self.assertRaises(ValueError):pick.pick_official_protocol(source,'xmove_pick')
        source=fixture();derived,_=pick.derive_pick_eval_reset(source,'xmove_pick')
        derived['layout']['actors']['target']['pose']['position'][0]+=.01
        with self.assertRaises(ValueError):pick.adapt_pick_config(derived,'xmove_pick')

    def test_private_cli_requires_matching_profile_sync_and_no_foreign_or_scripted_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            base=['run_closed_loop.py','--data-root',temp,'--sonic-resources',temp,'--output',temp+'/out',
                '--tasks',pick.PROFILES['xmove_pick']['task'],'--pick-official-protocol','--pick-profile','xmove_pick',
                '--seconds','16','--sync-observations','--isaac-observation-only','--sim-mode','mujoco_isaac']
            with patch.object(sys,'argv',base):self.assertTrue(loop.parse_args().pick_official_protocol)
            invalid=[base+['--stop-on-fall'],base+['--bend-official-protocol'],
                base+['--worker-task',pick.PROFILES['xmove_bend_pick']['task']]]
            wrong=base.copy();wrong[wrong.index('xmove_pick')]='xmove_bend_pick';invalid.append(wrong)
            missing=base.copy();missing.remove('--pick-official-protocol');invalid.append(missing)
            for argv in invalid:
                with self.subTest(argv=argv),patch.object(sys,'argv',argv),patch('sys.stderr'),self.assertRaises(SystemExit):loop.parse_args()

    def test_exact800_control_ticks_include_warmup_and_native_success_needs_no_navigation_event(self):
        import gymnasium as gym
        from test_mobile_eval_protocol import CounterEnv
        env=gym.wrappers.TimeLimit(CounterEnv(),max_episode_steps=800);env.reset()
        for tick in range(800):self.assertEqual(env.step(0)[3],tick==799)
        self.assertEqual(env.unwrapped.steps,800)
        episode={'task_success_observed':False,'fell':True,'policy_frames':0}
        self.assertEqual(loop.episode_termination(episode,True,False,.1),'task_terminated')
        self.assertTrue(episode['task_success_observed'])

    def test_redundant_layout_camera_none_fields_are_reversible_and_nonempty_not_discarded(self):
        source=fixture();source['layout']['cameras']['head_left']={'cam_cfg':{'intrinsics':None,
            'depth_intrinsics':None,'depth_to_color_extrinsics':None,'pose':{'position':[0.,0.,0.]}}}
        derived,receipt=pick.derive_pick_eval_reset(source,'xmove_pick')
        self.assertEqual(reverse(derived,receipt),source)
        self.assertEqual(derived['layout']['cameras']['head_left']['cam_cfg'],{'pose':{'position':[0.,0.,0.]}})
        source['layout']['cameras']['head_left']['cam_cfg']['intrinsics']=[[1.,0.,0.]]
        with self.assertRaisesRegex(ValueError,'nonempty redundant'):pick.derive_pick_eval_reset(source,'xmove_pick')

    def test_observer_binds_before_reset_step_and_stops_after201_without_policy_intervention(self):
        import mujoco
        from types import SimpleNamespace
        model=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005" gravity="0 0 0"/><worldbody>'
            '<body name="pelvis" pos="0 0 1"><geom type="sphere" size=".1"/></body>'
            '<body name="target" pos=".5 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>'
            '</worldbody></mujoco>')
        calls=[];engine=SimpleNamespace()
        def update():
            engine.mjModel=model;engine.mjData=mujoco.MjData(model);mujoco.mj_forward(model,engine.mjData)
            engine.mj_objects={'target':SimpleNamespace(id=model.body('target').id)}
        def step():calls.append(1);mujoco.mj_step(model,engine.mjData)
        engine.update_layout=update;engine.step=step
        raw=SimpleNamespace(mujoco=engine);raw.unwrapped=raw
        pick.install_pick_initialization_monitor(raw,'xmove_pick')
        with self.assertRaisesRegex(ValueError,'preceded'):engine.step()
        self.assertEqual(calls,[])
        engine.update_layout()
        for _ in range(220):engine.step()
        receipt=pick.initialization_receipt(raw)
        self.assertEqual((receipt['contract'],receipt['profile'],receipt['observed_physics_substeps']),
            (pick.INITIALIZATION_CONTRACT,'xmove_pick',201))
        self.assertEqual(len(calls),220)
        self.assertEqual(receipt['status'],'observation_complete')
        self.assertEqual(receipt['engineering_reference_status'],'passed')
        self.assertEqual(receipt['initialization_profile'],pick.INITIALIZATION_PROFILE)
        self.assertFalse(receipt['controller_commands_modified']or receipt['policy_phase_interventions'])

    def test_native_saved_scene_motion_is_warning_and_tail_metrics_are_read_only(self):
        import mujoco
        import numpy as np
        from types import SimpleNamespace
        from bend_initialization_monitor import BendInitializationMonitor
        model=mujoco.MjModel.from_xml_string('<mujoco><option timestep=".005" gravity="0 0 0"/><worldbody>'
            '<body name="pelvis" pos="0 0 3"><geom type="sphere" size=".1"/></body>'
            '<body name="target" pos=".5 0 1"><freejoint/><geom type="box" size=".05 .05 .05"/></body>'
            '</worldbody></mujoco>')
        data=mujoco.MjData(model);mujoco.mj_forward(model,data)
        observer=BendInitializationMonitor(model,data,model.body('target').id)
        data.qvel[:]=[.02,0.,0.,0.,0.,.2]
        for _ in range(201):mujoco.mj_step(model,data);observer.after_step(model,data)
        before=(data.qpos.copy(),data.qvel.copy(),data.time)
        raw=SimpleNamespace(_pick_initialization_monitor=observer,_pick_initialization_profile='xmove_bend_pick')
        receipt=pick.initialization_receipt(raw)
        self.assertEqual(receipt['status'],'observation_complete')
        self.assertEqual(receipt['engineering_reference_status'],'failed_or_incomplete')
        self.assertEqual(receipt['engineering_reference_warnings'],
            ['target_displacement_exceeds_2mm','target_rotation_exceeds_2deg'])
        self.assertEqual(observer.receipt()['status'],'failed_or_incomplete')
        self.assertEqual(receipt['initial_target_xyz_m'],[.5,0.,1.])
        tail=receipt['tail_settling_metrics']
        self.assertEqual(tail['sample_count'],41)
        self.assertAlmostEqual(tail['end_sim_time_s']-tail['start_sim_time_s'],.2)
        self.assertAlmostEqual(tail['maximum_pairwise_displacement_m'],.004)
        self.assertAlmostEqual(tail['endpoint_displacement_m'],.004)
        self.assertAlmostEqual(tail['endpoint_rotation_degrees'],np.degrees(.04),places=4)
        self.assertAlmostEqual(tail['maximum_pairwise_rotation_degrees'],np.degrees(.04),places=4)
        np.testing.assert_array_equal(data.qpos,before[0]);np.testing.assert_array_equal(data.qvel,before[1])
        self.assertEqual(data.time,before[2])
        complete=deepcopy(observer.receipt())
        for field,value in (('observed_physics_substeps',200),('finite_states',False),
            ('robot_target_contact_count',1),('contacts',[{'body':'robot'}]),
            ('live_physical_state_writes',True),('physics_steps_added',1),
            ('controller_commands_modified',True),('policy_phase_interventions',True),('private_FK_only',False)):
            corrupted=deepcopy(complete);corrupted[field]=value
            with self.subTest(field=field),patch.object(observer,'receipt',return_value=corrupted):
                self.assertEqual(pick.initialization_receipt(raw)['status'],'observation_failed_or_incomplete')

    @unittest.skipUnless(DATA.is_dir(),'Local official assets absent')
    def test_actual_all60_metadata_private_task_constructors_resets_and_complete_binding(self):
        from capture_simple import remap_episode_paths
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        import torch
        counts={};missing=[]
        with patch.dict(os.environ,{'SIMPLE_DATA_DIR':str(DATA),'CUDA_VISIBLE_DEVICES':''}):
            config=SimLoopConfig().load_wbc_yaml();config.update(ENV_NAME='simple',SIMULATE_DT=.005,ENABLE_ELASTIC_BAND=False)
            self.assertFalse(torch.cuda.is_initialized())
            for profile,spec in pick.PROFILES.items():
                count=0
                for level in range(3):
                    path=DATA/'eval'/spec['source_task']/f'dr-level-{level}/meta/episodes.jsonl'
                    rows=[json.loads(line) for line in path.read_text().splitlines()]
                    self.assertEqual([r['episode_index'] for r in rows],list(range(10)))
                    for row in rows:
                        source=remap_episode_paths(json.loads(row['environment_config']),DATA);before=deepcopy(source)
                        state,derivation,adaptation,protocol=pick.prepare_pick_reset(source,profile)
                        derived,_=pick.derive_pick_eval_reset(source,profile)
                        self.assertEqual(reverse(derived,derivation),source)
                        task=getattr(pick._adapter(profile),spec['factory'])(config,adapted_config=state)
                        task.reset(seed=0,options={'state_dict':state,'dr_level':None})
                        binding=pick.environment_binding_receipt(state,task.state_dict(),profile)
                        self.assertTrue(binding['matched'],binding)
                        self.assertEqual(task.success_criteria,.9)
                        self.assertEqual(source,before)
                        if protocol['source_metadata_max_episode_steps'] is None:missing.append((profile,level,row['episode_index']))
                        count+=1
                counts[profile]=count
            self.assertEqual(counts,{'xmove_pick':30,'xmove_bend_pick':30})
            self.assertFalse(torch.cuda.is_initialized())


if __name__=='__main__':unittest.main()
