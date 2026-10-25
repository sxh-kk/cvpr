"""CPU entry guards: no headset, SDK initialization, simulator, or rendering."""
import json
import copy
import ast
import inspect
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pico_simple_teleop as entry
import pico_sonic_reference as reference


def raw_packet(timestamp, kind=None):
    value = {'timeStampNs':timestamp,'Input':1,'Body':{'joints':[]}}
    if kind is not None:
        value['diagnostic_source_kind'] = kind
    return {'data':{'functionName':'Tracking','value':json.dumps(value)}}


def body_fixture():
    body = np.zeros((24,7))
    body[:,6] = 1.
    body[0,:3] = [0,1,0]
    for i in range(1,24):
        body[i,:3] = body[reference.BODY_PARENTS[i],:3]+[.02,-.1,.01]
    return body


def quality_fixture():
    vertices = [[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.25,.25)]
    cache = {'format':'rigid-container-collision-vertices-v1',
        'container_root_body_name':'basket','target_root_body_name':'box',
        'root_local_collision_vertices_m':vertices,'collision_geom_count':1,'collision_mesh_geom_count':0,
        'target_com_offset_local_m':[.01,.02,.03],
        'target_root_local_collision_vertices_m':np.asarray(vertices).__mul__(.1).tolist(),
        'upright_axis_local':[0,0,1],'upright_reference':'initial recorded upright'}
    support = {'support_measurement':'live_solved_mujoco_contact_force','target_mass_kg':1.,
        'target_weight_n':9.81,'container_target_contact_count':2,
        'container_force_on_target_world_n':[0,0,9.81],
        'container_upward_support_force_n':9.81,'container_support_weight_fraction':1.}
    raw = {'completed_steps':1,'sim_time_s':.02,'actual_target_xyz_m':[.1,.1,.1],
        'pelvis_z_m':.75,'contacts':{'left_hand':False,'right_hand':False,'container':True},
        'original_live_support':support,'target_pose7':[.1,.1,.1,1,0,0,0],
        'container_pose7':[0,0,0,1,0,0,0],'target_body_name':'box','container_body_name':'basket',
        'solver_stage':'post_mj_step_before_observation_refresh'}
    return cache,raw


def episode_metadata_fixture():
    session={name:None for name in entry.EPISODE_SESSION_FIELDS}
    session.update(task_id='simple/G1WholebodyHandoverTeleop-v0',source_kind='live_pico_human',
        accepted_new_dataset=False,physics_hz=200,decoder_hz=50,command_rgb_hz=30,
        render_mode='mujoco_deferred',raw_pico_va_wva_used=False,
        preview_engine='mujoco_egl',training_rgb_engine='pending_offline_isaac_rtx',
        preview_is_training_rgb=False,controller_identity={'decoder_sha256':'d'*64},
        encoder_identity={'source_sha256':{'encoder':'e'*64}},
        hand_control_profile={'profile_id':'official_middle_close'},
        source_code_sha256_at_session_start={'entry':'s'*64},calibration_source_asset_sha256={'camera':'c'*64})
    scene=entry.episode_scene_metadata(initialization={'object_restores_during_control':False,
        'external_assistance':False,'current_scene':'first'},standing_warmup_readiness={'passed':True},
        warmup={'events':[{'sim_time_s':i/50} for i in range(50)]},environment_config={'scene':'first'},
        physics_seed=1,instruction='handover',mujoco_object_body_names={'target':'box','container':'basket'},
        object_body_names={'target':'box','container':'basket'},mujoco_obj_names=['box','basket'],
        container_geometry_cache_sha256='g'*64,physics_model_sha256='m'*64)
    return session,scene


class EntryGuardTests(unittest.TestCase):
    def operator_loop(self, commands, episode):
        """Execute the actual production command loop with CPU-only callbacks."""
        tree=ast.parse(inspect.getsource(entry.run))
        loop=next(node for node in ast.walk(tree) if isinstance(node,ast.For)
            and isinstance(node.iter,ast.Call) and isinstance(node.iter.func,ast.Name)
            and node.iter.func.id=='scene_bound_commands')
        events=[]
        namespace=dict(vars(entry),commands=commands,episode=episode,following=True,
            reset_generation=0,reset_generation_at_loop_start=0,reset_this_batch=False,
            operator_message='',error_text=None,replay=None,history_status={},
            review_queue=None,latest_finished_uuid=None,result={'episodes':[]},
            reference=SimpleNamespace(calibration={'same_calibration':True}))
        def finalize(*args,**kwargs):
            events.append(('finalize',args,kwargs))
            namespace['episode']=None
            return {'collection':{'status':'pending','pending':True}}
        def reset_scene(*args,**kwargs):
            if namespace['episode'] is not None:
                raise ValueError('Finish the current episode before resetting the scene')
            events.append(('reset',args,kwargs))
            namespace['reset_generation']+=1
            namespace['following']=False
        def reject(reason):
            events.append(('reject',reason))
            namespace['following']=False
        namespace.update(finalize=finalize,reset_scene=reset_scene,reject_and_pause=reject)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(loop)],type_ignores=[])),
            entry.__file__,'exec'),namespace)
        return namespace,events

    def active_episode(self):
        episode=entry.RecordingState()
        episode.request_start()
        for tick in range(51):
            episode.on_tick(tick)
        return episode

    def replay(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/'packets.json'
        path.write_text(json.dumps(rows))
        return entry.RecordedInput(path)

    def test_replay_uses_only_past_packets_and_refuses_extrapolation(self):
        replay = self.replay([raw_packet(10_000_000_000+i*20_000_000) for i in range(3)])
        self.assertEqual(replay.at(.019).sequence,1)
        self.assertEqual(replay.at(.021).sequence,2)
        self.assertEqual(replay.at(.04).sequence,3)
        with self.assertRaises(EOFError):
            replay.at(.040001)
        for invalid in (-1,float('nan'),float('inf')):
            with self.assertRaises(ValueError):
                replay.at(invalid)

    def test_synthetic_diagnostic_is_never_labelled_human(self):
        kind = 'SYNTHETIC_NEUTRAL_NOT_HUMAN_NOT_TRAINING'
        replay = self.replay([raw_packet(100,kind),raw_packet(200,kind)])
        self.assertEqual(replay.source_kind,kind)
        actual = self.replay([raw_packet(100),raw_packet(200)])
        self.assertEqual(actual.source_kind,'recorded_pico_diagnostic')
        with self.assertRaisesRegex(ValueError,'Mixed diagnostic'):
            self.replay([raw_packet(100,kind),raw_packet(200)])

    def test_replay_rejects_bad_device_clock(self):
        for timestamps in ((100,100),(200,100)):
            with self.assertRaisesRegex(ValueError,'strictly increase'):
                self.replay([raw_packet(t) for t in timestamps])

    def test_recording_calibration_guard_only_blocks_calibration(self):
        self.assertIn('先保存后再校准',entry.recording_command_guard('calibrate',episode_active=True))
        self.assertIsNone(entry.recording_command_guard('calibrate',episode_active=False))
        for action in ('pause','save','discard','reset','record','follow'):
            self.assertIsNone(entry.recording_command_guard(action,episode_active=True))

    def test_actual_operator_loop_ignores_ui_and_x_calibration_without_rejecting_recording(self):
        for commands in (['calibrate'],entry.rising_operator_commands({'X':True},{})):
            episode=self.active_episode()
            state,events=self.operator_loop(commands,episode)
            self.assertEqual(events,[])
            self.assertIs(state['episode'],episode)
            self.assertEqual(episode.status,'recording')
            self.assertTrue(state['following'])
            self.assertEqual(state['reference'].calibration,{'same_calibration':True})
            self.assertIn('录制继续',state['operator_message'])

    def test_actual_operator_loop_keeps_pause_save_discard_and_reset_safety(self):
        state,events=self.operator_loop(['pause'],self.active_episode())
        self.assertEqual(events,[('reject','operator_pause')])
        for action,completed,reason in (('save',True,None),('discard',False,'operator_discard')):
            with self.subTest(action=action):
                state,events=self.operator_loop([action,'discard'],self.active_episode())
                self.assertEqual(events[0],('finalize',(),{'completed':completed,'reason':reason}))
                self.assertEqual([item[0] for item in events],['finalize','reset'])
                self.assertEqual(state['reset_generation'],1)
        state,events=self.operator_loop(['reset'],None)
        self.assertEqual([item[0] for item in events],['reset'])
        _,events=self.operator_loop(['reset'],self.active_episode())
        self.assertEqual(events[0][0],'reject')

    def test_episode_metadata_does_not_copy_growing_session_histories(self):
        session,scene=episode_metadata_fixture()
        class NoCopy:
            def __deepcopy__(self,memo):
                raise AssertionError('A growing session history was copied on record')
        for name in ('episodes','rejections','resets','prepared_deferred_models',
                     'prepared_quality_geometry_caches','background_review','unbounded_debug'):
            session[name]=NoCopy()
        fixed=entry.episode_session_identity(session)
        metadata=entry.bounded_episode_metadata(fixed,scene,human_demo_uuid='current',
            pico_calibration={'heading':.1},initial_target_z_m=.5)
        self.assertFalse(set(('episodes','rejections','resets','prepared_deferred_models',
            'prepared_quality_geometry_caches','background_review')) & set(metadata))
        before=json.dumps(metadata,sort_keys=True)
        session['resets']=[{'large_old_scene':'x'*10000} for _ in range(100)]
        again=entry.bounded_episode_metadata(entry.episode_session_identity(session),scene,
            human_demo_uuid='current',pico_calibration={'heading':.1},initial_target_z_m=.5)
        self.assertEqual(json.dumps(again,sort_keys=True),before)
        session['controller_identity']['decoder_sha256']='changed'
        self.assertEqual(fixed['controller_identity']['decoder_sha256'],'d'*64)

    def test_episode_metadata_keeps_all_audit_renderer_and_geometry_bindings(self):
        session,scene=episode_metadata_fixture()
        metadata=entry.bounded_episode_metadata(entry.episode_session_identity(session),scene,
            human_demo_uuid='current',pico_calibration={'heading':.1},initial_target_z_m=.5)
        consumed=set()
        for name in ('audit_pico_teleop_episode.py','render_deferred_pico.py',
                     'pico_episode_history.py','pico_episode_review.py','pico_deferred_capture.py'):
            tree=ast.parse((Path(entry.__file__).parent/name).read_text())
            for node in ast.walk(tree):
                if (isinstance(node,ast.Subscript) and isinstance(node.value,ast.Name)
                        and node.value.id=='metadata' and isinstance(node.slice,ast.Constant)
                        and isinstance(node.slice.value,str)):
                    consumed.add(node.slice.value)
                if (isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)
                        and isinstance(node.func.value,ast.Name) and node.func.value.id=='metadata'
                        and node.func.attr=='get' and node.args and isinstance(node.args[0],ast.Constant)
                        and isinstance(node.args[0].value,str)):
                    consumed.add(node.args[0].value)
        self.assertTrue(consumed <= set(metadata),consumed-set(metadata))
        self.assertEqual(metadata['environment_config'],{'scene':'first'})
        self.assertEqual(metadata['physics_model_sha256'],'m'*64)
        changed=copy.deepcopy(scene)
        changed.update(initialization={'current_scene':'second'},environment_config={'scene':'second'},physics_seed=2)
        next_metadata=entry.bounded_episode_metadata(entry.episode_session_identity(session),changed,
            human_demo_uuid='next',pico_calibration={'heading':.2},initial_target_z_m=.6)
        self.assertEqual(next_metadata['initialization']['current_scene'],'second')
        self.assertEqual(metadata['initialization']['current_scene'],'first')

    def test_full_loop_wall_timing_includes_startup_and_tail_but_excludes_clock_sleep(self):
        clock=SimpleNamespace(value=0)
        timer=entry.LoopWallTimer(clock_ns=lambda:clock.value,wall_clock_ns=lambda:123456789)
        for stage,elapsed_ns in (('input_validation',3_000_000),('operator_actions',2_000_000),
                ('control',20_000_000),('clock_sleep',10_000_000),('review_poll',4_000_000),
                ('preview_status',7_000_000),('status_disk',6_000_000)):
            timer.enter(stage)
            clock.value+=elapsed_ns
        result=timer.finish()
        self.assertTrue(result['complete'])
        self.assertEqual(result['loop_start_pc_wall_time_ns'],123456789)
        self.assertAlmostEqual(result['timing_s']['total'],.052)
        self.assertAlmostEqual(result['loop_critical_wall_s'],.042)
        self.assertAlmostEqual(result['tail_after_control_wall_s'],.017)
        self.assertAlmostEqual(sum(result['timing_s'][name] for name in timer.STAGES),.052)
        clock.value+=100_000_000  # Reading the previous complete loop cannot grow it.
        self.assertEqual(timer.snapshot(),result)
        result['timing_s']['total']=999
        self.assertAlmostEqual(timer.snapshot()['timing_s']['total'],.052)

    def test_loop_timing_partial_rejection_snapshot_does_not_close_or_reset_clock(self):
        clock=SimpleNamespace(value=0)
        timer=entry.LoopWallTimer(clock_ns=lambda:clock.value,wall_clock_ns=lambda:12)
        clock.value=2_000_000
        timer.enter('control')
        clock.value=102_000_000
        partial=timer.snapshot()
        self.assertFalse(partial['complete'])
        self.assertEqual(partial['active_stage_at_snapshot'],'control')
        self.assertAlmostEqual(partial['timing_s']['control'],.1)
        clock.value=112_000_000
        final=timer.finish()
        self.assertAlmostEqual(final['timing_s']['control'],.11)
        self.assertAlmostEqual(partial['timing_s']['control'],.1)
        self.assertEqual(clock.value,112_000_000)

    def test_hand_mapping_uses_official_close_by_name_in_entry(self):
        names = entry.MODEL_HAND_NAMES
        config = lambda values: SimpleNamespace(joint_names=values,
            init_qpos=np.zeros(len(values)),close_qpos=np.arange(1,len(values)+1))
        robot = SimpleNamespace(controller_cfg=SimpleNamespace(
            left_eef_cfg=config(names[:7]),right_eef_cfg=config(names[7:])))
        frame = SimpleNamespace(inputs={'left_trigger':.25,'right_trigger':.75,
                                        'left_grip':0.,'right_grip':0.})
        target = entry.named_hand_targets(robot,frame)
        self.assertEqual(target['left_hand_thumb_1_joint'],.7*.25)
        self.assertEqual(target['left_hand_index_0_joint'],-.25)
        self.assertEqual(target['right_hand_thumb_1_joint'],-.7*.75)
        self.assertEqual(target['right_hand_middle_1_joint'],1.5*.75)
        frame.inputs['right_trigger'] = float('nan')
        with self.assertRaises(ValueError):
            entry.named_hand_targets(robot,frame)

    def test_live_command_wait_uses_successor_without_repeating_source(self):
        clock = [0.]
        old = SimpleNamespace(device_timestamp_ns=100)
        new = SimpleNamespace(device_timestamp_ns=200)
        def quality(frame, **kwargs):
            return {'valid':True,'recordable':frame is new,'reasons':[]}
        with patch.object(entry,'read_live_frame',side_effect=[(old,None),(new,None)]), \
             patch.object(entry,'validate_frame',side_effect=quality):
            actual,q = entry.wait_fresh_command_frame(object(),old,100,.01,
                now=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))
        self.assertIs(actual,new)
        self.assertTrue(q['recordable'])
        self.assertGreater(clock[0],0.)
        self.assertLess(clock[0],.01)

    def test_live_command_wait_times_out_without_padding(self):
        clock = [0.]
        old = SimpleNamespace(device_timestamp_ns=100)
        with patch.object(entry,'read_live_frame',return_value=(old,None)), \
             patch.object(entry,'validate_frame',return_value={'valid':True,'recordable':False}):
            with self.assertRaisesRegex(RuntimeError,'不会重复'):
                entry.wait_fresh_command_frame(object(),old,100,.005,
                    now=lambda:clock[0],sleep=lambda dt:clock.__setitem__(0,clock[0]+dt))
        self.assertAlmostEqual(clock[0],.005)

    def test_live_command_wait_rejects_malformed_successor(self):
        with patch.object(entry,'read_live_frame',return_value=(None,'bad packet')):
            with self.assertRaisesRegex(ValueError,'bad packet'):
                entry.wait_fresh_command_frame(object(),None,100,1.,now=lambda:0.)

    def test_live_command_wait_does_not_accept_packet_after_deadline(self):
        clock=[0.]
        def slow_read(receiver):
            clock[0]=.011
            return SimpleNamespace(device_timestamp_ns=200),None
        with patch.object(entry,'read_live_frame',side_effect=slow_read):
            with self.assertRaisesRegex(RuntimeError,'不会重复'):
                entry.wait_fresh_command_frame(object(),None,100,.01,now=lambda:clock[0])

    def test_waited_button_press_stays_queued_after_release_without_duplicate(self):
        pressed={'X':False,'Y':True,'A':True}
        pending=entry.rising_operator_commands(pressed,{})
        self.assertEqual(pending,['toggle_follow','toggle_record'])
        self.assertEqual(entry.rising_operator_commands(pressed,pressed),[])
        released={'X':False,'Y':False,'A':False}
        pending.extend(entry.rising_operator_commands(released,pressed))
        self.assertEqual(pending,['toggle_follow','toggle_record'])

    def test_live_command_wait_rejects_validation_that_crossed_deadline(self):
        clock=[0.]
        def slow_validation(frame, **kwargs):
            clock[0]=.011
            return {'valid':True,'recordable':True}
        with patch.object(entry,'read_live_frame',return_value=(SimpleNamespace(device_timestamp_ns=200),None)), \
             patch.object(entry,'validate_frame',side_effect=slow_validation):
            with self.assertRaisesRegex(RuntimeError,'不会重复'):
                entry.wait_fresh_command_frame(object(),None,100,.01,now=lambda:clock[0])

    def test_live_command_wait_rejects_invalid_tracking_instead_of_holding(self):
        with patch.object(entry,'read_live_frame',return_value=(object(),None)), \
             patch.object(entry,'validate_frame',return_value={'valid':False,'reasons':['app_not_focused']}):
            with self.assertRaisesRegex(ValueError,'app_not_focused'):
                entry.wait_fresh_command_frame(object(),None,100,1.,now=lambda:0.)

    def test_standing_warmup_rejects_fall_tilt_or_nonfinite_state(self):
        before = {'pelvis_z_m':.78}
        after = {'pelvis_z_m':.76,'pelvis_tilt_deg':4.,'state_finite':True}
        receipt = entry.require_standing_warmup(before,after)
        self.assertTrue(receipt['passed'])
        for patch in ({'pelvis_z_m':.50},{'pelvis_tilt_deg':26.},{'state_finite':False}):
            with self.assertRaisesRegex(ValueError,'Standing warmup not ready'):
                entry.require_standing_warmup(before,{**after,**patch})

    def test_human_calibration_preserves_and_rejects_real_root_tilt(self):
        rest, parents = reference.load_human_info(reference.GEAR/'data/human/human_joints_info.pkl')
        fk = SimpleNamespace(rest_joints=rest,fk_parents=parents)
        neutral = body_fixture()
        receipt = entry.require_neutral_calibration(fk,neutral)
        self.assertLess(receipt['reference_root_tilt_deg'],.01)
        tilted = neutral.copy()
        tilted[:,3:] = Rotation.from_euler('x',50,degrees=True).as_quat()
        with self.assertRaisesRegex(ValueError,'humanTilt=.*要求≤15'):
            entry.require_neutral_calibration(fk,tilted)
        np.testing.assert_array_equal(tilted[:,3:],
            np.tile(Rotation.from_euler('x',50,degrees=True).as_quat(),(24,1)))

    def test_physical_snapshot_uses_actual_positions_and_named_state(self):
        names = ('left_knee_joint','right_knee_joint')
        model = SimpleNamespace(body=lambda name:SimpleNamespace(id=0))
        joints = {name:SimpleNamespace(qpos=np.array([i+.1]),qvel=np.array([i+.2]))
            for i,name in enumerate((*names,*entry.MODEL_HAND_NAMES))}
        data = SimpleNamespace(time=1.25,xpos=np.array([[1.,2.,.77]]),
            xquat=np.array([[1.,0,0,0]]),xmat=np.eye(3).reshape(1,9),
            qpos=np.zeros(50),qvel=np.zeros(49),qacc=np.zeros(49),ctrl=np.zeros(43),
            joint=lambda name:joints[name])
        state = entry.physical_snapshot(model,data,names)
        self.assertEqual(state['root_pose7'],[1.,2.,.77,1.,0,0,0])
        self.assertEqual(state['body_q29'],[.1,1.1])
        self.assertEqual(state['pelvis_tilt_deg'],0.)
        self.assertTrue(state['state_finite'])
        data.qvel[0] = np.nan
        self.assertFalse(entry.physical_snapshot(model,data,names)['state_finite'])

    def test_source_identity_covers_executed_entry_components(self):
        identities = entry.source_code_identities()
        self.assertEqual(set(identities),{'pico_simple_teleop.py','pico_live_input.py','pico_hand_control.py',
            'pico_sonic_reference.py','pico_teleop_recording.py','pico_teleop_ui.py',
            'pico_deferred_capture.py','render_deferred_pico.py','finish_pico_session.py',
            'audit_pico_teleop_episode.py','start_pico_teleop.sh',
            'handover_contact_quality.py','handover_geometry_quality.py','pico_episode_history.py',
            'pico_episode_review.py'})
        self.assertTrue(all(len(digest)==64 for digest in identities.values()))

    def test_instance_rate_limits_record_before_after_without_clock_changes(self):
        keys = [f'/app/runLoops/{loop}/rateLimitEnabled' for loop in ('main','present','rendering_0')]
        values = {keys[0]:True,keys[1]:True,keys[2]:False,'physics_dt':.005}
        settings = SimpleNamespace(get=lambda key:values.get(key),
            set_bool=lambda key,value:values.__setitem__(key,value))
        receipt = entry.disable_instance_render_rate_limits(settings)
        self.assertEqual(set(receipt['settings']),set(keys))
        self.assertTrue(receipt['settings'][keys[0]]['before'])
        self.assertTrue(all(values[key] is False for key in keys))
        self.assertEqual(values['physics_dt'],.005)

    def test_render_profiler_preserves_returns_exceptions_and_restores_methods(self):
        calls = []
        world = SimpleNamespace(render=lambda:calls.append('world'))
        isaac = SimpleNamespace(world=world,sync_states=lambda:calls.append('sync'),
            observation_pose_diagnostics=lambda:{'pose':'actual'})
        def render():
            isaac.world.render()
            return 'rgb'
        def fail():
            raise ValueError('original render error')
        isaac.render = render
        orchestrator = SimpleNamespace(step=fail)
        original_world_render = world.render
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.jsonl'
            profiler = entry.RenderProfiler(path).install(isaac,orchestrator)
            profiler.set_context('diagnostic',12,.24)
            self.assertEqual(isaac.render(),'rgb')
            self.assertEqual(calls,['world'])
            with self.assertRaisesRegex(ValueError,'original render error'):
                orchestrator.step()
            self.assertEqual(profiler.receipt()['statistics']['isaac.render']['calls'],1)
            profiler.close()
            self.assertIs(world.render,original_world_render)
            self.assertIs(isaac.render,render)
            self.assertIs(orchestrator.step,fail)
            rows = [json.loads(row) for row in path.read_text().splitlines()]
            by_name = {row['name']:row for row in rows}
            self.assertEqual(by_name['isaac.world.render']['parent_call_id'],by_name['isaac.render']['call_id'])
            self.assertIn('original render error',by_name['rep.orchestrator.step']['error'])
            self.assertTrue(all(row['phase']=='diagnostic' and row['tick50']==12 for row in rows))

    def test_render_profiler_rejects_pre_reset_world_without_partial_patches(self):
        isaac = SimpleNamespace(sync_states=lambda:None)
        original = isaac.sync_states
        with tempfile.TemporaryDirectory() as directory:
            profiler = entry.RenderProfiler(Path(directory)/'profile.jsonl')
            try:
                with self.assertRaisesRegex(ValueError,'env.reset'):
                    profiler.install(isaac,SimpleNamespace(step=lambda:None))
                self.assertEqual(profiler.patches,[])
                self.assertIs(isaac.sync_states,original)
            finally:
                profiler.close()

    def test_unrelated_viewport_is_read_only_by_default(self):
        viewport = SimpleNamespace(render_product_path='/Viewport',updates_enabled=True)
        cameras = {name:SimpleNamespace(get_render_product_path=lambda name=name:'/RP/'+name,
            get_resolution=lambda:(640,360)) for name in ('head_stereo_left','head_stereo_right')}
        receipt = entry.configure_unrelated_viewport(viewport,cameras)
        self.assertEqual(receipt['status'],'inspected_without_change')
        self.assertTrue(viewport.updates_enabled)
        self.assertFalse(receipt['disabled'])
        receipt = entry.configure_unrelated_viewport(viewport,cameras,disable=True)
        self.assertEqual(receipt['status'],'disabled')
        self.assertFalse(viewport.updates_enabled)
        self.assertTrue(all(camera.get_resolution()==(640,360) for camera in cameras.values()))

    def test_viewport_disable_refuses_shared_or_unknown_products(self):
        cameras = {name:SimpleNamespace(get_render_product_path=lambda name=name:'/RP/'+name,
            get_resolution=lambda:(640,360)) for name in ('head_stereo_left','head_stereo_right')}
        for product in (None,'','/RP/head_stereo_left','/RP/head_stereo_right'):
            viewport = SimpleNamespace(render_product_path=product,updates_enabled=True)
            receipt = entry.configure_unrelated_viewport(viewport,cameras,disable=True)
            self.assertEqual(receipt['status'],'refused')
            self.assertTrue(viewport.updates_enabled)
        self.assertEqual(entry.configure_unrelated_viewport(None,cameras,disable=True)['status'],'no_viewport')

    def test_viewport_disable_preserves_required_stereo_resolution(self):
        viewport = SimpleNamespace(render_product_path='/Viewport',updates_enabled=True)
        for cameras in ({}, {'head_stereo_left':SimpleNamespace(
            get_render_product_path=lambda:'/Left',get_resolution=lambda:(320,180))}):
            self.assertEqual(entry.configure_unrelated_viewport(viewport,cameras,disable=True)['status'],'refused')
            self.assertTrue(viewport.updates_enabled)

    def test_environment_config_binds_local_handover_assets_and_rejects_remote(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scene_dir = root/'scenes/hssd/room'
            scene_dir.mkdir(parents=True)
            (scene_dir/'room.usd').write_text('local fixture')
            (root/'object.usd').write_text('local fixture')
            state = {'uid':'g1_wholebody_handover_teleop','robot_cfg':{'uid':'g1_sonic'},
                'dr_state_dict':{'scene':{'uid':'hssd:scene0','name':'room',
                    'data_dir':'scenes/hssd/room','conf':{'uid':'scene0','name':'room'}},
                    'target':{'usd_path':str(root/'object.usd')}},'layout':{}}
            path = root/'environment_config.json'
            path.write_text(json.dumps(state))
            actual,receipt = entry.load_environment_config(path,root)
            self.assertEqual(actual,state)
            self.assertEqual(receipt['scene_name'],'room')
            self.assertEqual(receipt['local_resource_paths']['authoritative_scene_usd'],str(scene_dir/'room.usd'))
            remote = copy.deepcopy(state)
            remote['dr_state_dict']['target']['usd_path'] = 'https://example.invalid/object.usd'
            path.write_text(json.dumps(remote))
            with self.assertRaisesRegex(ValueError,'Remote resource'):
                entry.load_environment_config(path,root)
            state['uid'] = 'carry_box'
            path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError,'Handover task'):
                entry.load_environment_config(path,root)

    def test_environment_scene_manager_binding_is_instance_local(self):
        manager = SimpleNamespace(hssd_scenes=[{'uid':'scene0','name':'different_room'}])
        randomizer = SimpleNamespace(scene_manager=manager)
        task = SimpleNamespace(dr=SimpleNamespace(get_randomizer=lambda name:randomizer))
        state = {'dr_state_dict':{'scene':{'uid':'hssd:scene0','name':'saved_room',
            'conf':{'uid':'scene0','name':'saved_room'}}}}
        entry.bind_saved_scene_manager(task,state)
        self.assertIsNot(randomizer.scene_manager,manager)
        self.assertEqual(manager.hssd_scenes[0]['name'],'different_room')
        self.assertEqual(randomizer.scene_manager.hssd_scenes[0]['name'],'saved_room')
        randomizer.scene_manager.hssd_scenes[0]['name'] = 'modified'
        self.assertEqual(state['dr_state_dict']['scene']['conf']['name'],'saved_room')

    def test_environment_receipt_detects_room_and_material_changes(self):
        state = {'dr_state_dict':{'scene':{'name':'saved_room'},'material':{'roughness':.5}},
            'layout':{'actors':{'target':{'pose':[1.,2.,3.]}}}}
        self.assertTrue(entry.environment_binding_receipt(state,copy.deepcopy(state))['matched'])
        for key in ('scene','material'):
            changed = copy.deepcopy(state)
            changed['dr_state_dict'][key] = {'name':'changed'}
            self.assertFalse(entry.environment_binding_receipt(state,changed)['matched'])

    def test_egl_selection_rejects_changing_an_already_loaded_backend(self):
        environ = {'MUJOCO_GL':'glfw','unrelated':'keep'}
        with self.assertRaisesRegex(ValueError,'fresh EGL process'):
            entry.configure_mujoco_egl(environ,mujoco_already_loaded=True)
        self.assertEqual(environ['MUJOCO_GL'],'glfw')
        receipt = entry.configure_mujoco_egl(environ,mujoco_already_loaded=False)
        self.assertEqual(receipt['before'],'glfw')
        self.assertEqual(environ,{'MUJOCO_GL':'egl','unrelated':'keep'})
        self.assertEqual(entry.configure_mujoco_egl(environ,mujoco_already_loaded=True)['after'],'egl')

    def test_deferred_state_is_exactly_the_observed_pre_action_state(self):
        calls = []
        data = SimpleNamespace(time=.24)
        def get_state(model,actual,value,spec):
            calls.append((actual.time,spec))
            value[:] = [.24,1.,2.,3.]
        module = SimpleNamespace(mjtState=SimpleNamespace(mjSTATE_INTEGRATION=4095),
            mj_stateSize=lambda model,spec:4,mj_getState=get_state)
        value = entry.integration_state_snapshot(module,object(),data,.24)
        np.testing.assert_array_equal(value,[.24,1.,2.,3.])
        self.assertEqual(calls,[(.24,4095)])
        self.assertEqual(data.time,.24)
        with self.assertRaisesRegex(ValueError,'timestamps differ'):
            entry.integration_state_snapshot(module,object(),data,.22)
        self.assertEqual(len(calls),1)

    def test_malformed_live_packet_does_not_close_receiver_or_reuse_old_data(self):
        def malformed():
            raise ValueError('Tracking pose contains non-numeric values')
        receiver = SimpleNamespace(snapshot=malformed,close=lambda:self.fail('Receiver must remain open'))
        frame,error = entry.read_live_frame(receiver)
        self.assertIsNone(frame)
        self.assertIn('原始PICO数据包无效',error)
        self.assertIn('重新校准',error)
        current = object()
        receiver.snapshot = lambda:current
        self.assertEqual(entry.read_live_frame(receiver),(current,None))

    def test_deferred_cache_uses_actual_body_names_and_new_reset_directory(self):
        calls = []
        def prepare(directory,**kwargs):
            calls.append((directory,kwargs))
            return object()
        objects = {'target':SimpleNamespace(id=7),'container':SimpleNamespace(id=9)}
        model = SimpleNamespace(body=lambda body_id:SimpleNamespace(name={7:'actual_box',9:'actual_basket'}[body_id]))
        for index in (0,1):
            _,receipt = entry.prepare_deferred_cache(prepare,'/tmp/session',index,
                model=model,objects=objects,joint_names=('one','two'))
            self.assertEqual(receipt['object_body_names'],{'target':'actual_box','container':'actual_basket'})
            self.assertEqual(calls[-1][1]['joint_names'],['one','two'])
            self.assertIs(calls[-1][1]['model'],model)
        self.assertEqual(calls[0][0],Path('/tmp/session/prepared_models/0000'))
        self.assertEqual(calls[1][0],Path('/tmp/session/prepared_models/0001'))

    def test_deferred_quality_uses_exact_original_geometry_and_live_force(self):
        from handover_geometry_quality import container_footprint_from_poses
        cache,raw = quality_fixture()
        before = copy.deepcopy(raw)
        completed = entry.complete_quality_geometry([raw],cache)[0]
        expected = container_footprint_from_poses(cache,
            container_pose_wxyz=raw['container_pose7'],target_pose_wxyz=raw['target_pose7'])
        for key,value in expected.items():
            if key != 'limitations':
                self.assertEqual(completed['container_geometry'][key],value)
        for key,value in raw['original_live_support'].items():
            self.assertEqual(completed['container_geometry'][key],value)
        self.assertEqual(raw,before)
        self.assertEqual(completed['quality_geometry_mode'],'deferred_geometry_with_original_solved_support')
        self.assertNotIn('container_geometry',raw)
        self.assertIsNot(completed['container_geometry']['container_force_on_target_world_n'],
            raw['original_live_support']['container_force_on_target_world_n'])

    def test_deferred_quality_rejects_changed_stage_clock_or_force_provenance(self):
        cache,raw = quality_fixture()
        variants = []
        for change in ({'solver_stage':'post_forward'},{'completed_steps':True},
                       {'target_body_name':'other'},{'actual_target_xyz_m':[1,2,3]}):
            variants.append({**copy.deepcopy(raw),**change})
        force = copy.deepcopy(raw)
        force['original_live_support']['support_measurement'] = 'recomputed_from_poses'
        variants.append(force)
        for changed in variants:
            with self.assertRaises(ValueError):
                entry.complete_quality_geometry([changed],cache)
        second = {**copy.deepcopy(raw),'completed_steps':2,'sim_time_s':.05}
        with self.assertRaisesRegex(ValueError,'real50Hz'):
            entry.complete_quality_geometry([raw,second],cache)

    def test_live_quality_sampling_does_not_call_geometry_or_forward(self):
        import handover_geometry_quality as geometry
        _,fixture = quality_fixture()
        names = {1:'box',2:'basket',3:'pelvis'}
        def body(value):
            body_id = {name:index for index,name in names.items()}[value] if isinstance(value,str) else value
            return SimpleNamespace(id=body_id,name=names[body_id])
        model = SimpleNamespace(body=body,nbody=4)
        data = SimpleNamespace(time=.02,xpos=np.array([[0,0,0],[.1,.1,.1],[0,0,0],[0,0,.75]]),
            xquat=np.tile([1.,0,0,0],(4,1)))
        with patch.object(geometry,'target_contacts',return_value=fixture['contacts']) as contacts, \
             patch.object(geometry,'target_support_status',return_value=fixture['original_live_support']) as support, \
             patch.object(geometry,'container_footprint_from_poses',side_effect=AssertionError('No live geometry')):
            raw = entry.sample_original_solved_quality(model,data,completed_steps=1,
                target_body=1,container_body=2,solver_stage=fixture['solver_stage'])
        contacts.assert_called_once_with(model,data,'box',container_label='basket')
        support.assert_called_once_with(model,data,'box',container_label='basket')
        self.assertEqual(raw,fixture)

    def test_geometry_cache_is_prepared_once_and_can_be_hardlinked(self):
        import os
        cache,_ = quality_fixture()
        with tempfile.TemporaryDirectory() as directory:
            receipt = entry.prepare_quality_geometry(cache,directory,0)
            source = Path(receipt['path'])
            episode_cache = Path(directory)/'container_geometry_cache.json'
            os.link(source,episode_cache)
            self.assertEqual(os.stat(source).st_ino,os.stat(episode_cache).st_ino)
            with self.assertRaises(ValueError):
                entry.prepare_quality_geometry(cache,directory,0)
            entry.prepare_quality_geometry({**cache,'new_reset':True},directory,1)
            self.assertEqual(json.loads(episode_cache.read_text()),cache)

    def test_quality_geometry_source_receipt_binds_three_distinct_files(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = {'raw_samples_sha256':'raw_quality_samples.jsonl',
                'cache_sha256':'container_geometry_cache.json',
                'completed_samples_sha256':'quality_samples.jsonl'}
            for name in names.values():
                (root/name).write_bytes(name.encode())
            receipt = entry.quality_geometry_source_identities(root)
            self.assertEqual(receipt,{key:hashlib.sha256(name.encode()).hexdigest() for key,name in names.items()})

    def test_save_finishes_before_exactly_one_automatic_reset(self):
        calls=[]
        outcome={'capture':{'status':'raw_physics_capture_completed_pending_rtx'},
            'collection':{'number':1,'disposition':'kept'}}
        def finalize(**kwargs):
            calls.append(('finalize',kwargs))
            return outcome
        result,problem=entry.end_episode_and_reset(finalize,lambda:calls.append(('reset',{})),discard=False)
        self.assertIs(result,outcome)
        self.assertIsNone(problem)
        self.assertEqual(calls,[('finalize',{'completed':True,'reason':None}),('reset',{})])

    def test_stale_command_batch_cannot_discard_after_save_or_scene_reset(self):
        # These commands arrived together for the old recording. Saving it,
        # rejecting it, or failing a reset must not apply discard to its successor.
        for first,fail in (('save',False),('active_discard',False),
                           ('active_rejection',False),('reset',True)):
            with self.subTest(first=first,fail=fail):
                state={'transition':False,'generation':0}
                applied=[]
                for action in entry.scene_bound_commands([first,'discard','record'],lambda:
                        state['transition'] or state['generation'] != 0):
                    applied.append(action)
                    if action in ('save','active_discard'):
                        state['transition']=True  # before finalize can fail
                    else:
                        state['generation']+=1  # before reset can fail
                    if fail:
                        try:
                            raise RuntimeError('reset warmup failed')
                        except RuntimeError:
                            pass
                self.assertEqual(applied,[first])
        self.assertEqual(list(entry.scene_bound_commands(['discard'],lambda:True)),[])
        self.assertEqual(list(entry.scene_bound_commands(['calibrate','follow'],lambda:False)),
            ['calibrate','follow'])

    def test_active_discard_finishes_with_reason_before_reset(self):
        calls=[]
        def finalize(**kwargs):
            calls.append(('finalize',kwargs))
            return {'capture':{'status':'partial_capture_rejected'}}
        _,problem=entry.end_episode_and_reset(finalize,lambda:calls.append(('reset',{})),discard=True)
        self.assertIsNone(problem)
        self.assertEqual(calls,[('finalize',{'completed':False,'reason':'operator_discard'}),('reset',{})])

    def test_automatic_reset_does_not_hide_a_writer_or_history_failure(self):
        calls=[]
        outcome={'capture':{'status':'partial_capture_rejected','failure':'disk full'},
            'collection':{'status':'error','error':'index unavailable'}}
        result,problem=entry.end_episode_and_reset(lambda **kwargs:outcome,
            lambda:calls.append('reset'),discard=False)
        self.assertIs(result,outcome)
        self.assertEqual(calls,['reset'])
        self.assertIn('disk full',problem)
        self.assertIn('index unavailable',problem)

    def test_finalize_failure_does_not_reset_and_failed_reset_is_not_retried(self):
        calls=[]
        def failed_finalize(**kwargs):
            raise OSError('cannot preserve evidence')
        with self.assertRaisesRegex(OSError,'cannot preserve evidence'):
            entry.end_episode_and_reset(failed_finalize,lambda:calls.append('reset'),discard=False)
        self.assertEqual(calls,[])
        def failed_reset():
            calls.append('reset')
            raise ValueError('warmup not ready')
        with self.assertRaisesRegex(RuntimeError,'disk full.*warmup not ready'):
            entry.end_episode_and_reset(lambda **kwargs:{'capture':{'failure':'disk full'}},
                failed_reset,discard=False)
        self.assertEqual(calls,['reset'])

    def test_only_started_capture_is_queued_and_diagnostics_disable_collection(self):
        submitted=[]
        queue=SimpleNamespace(submit=lambda directory,**options:
            submitted.append((directory,options)) or {'status':'pending','uuid':Path(directory).name})
        episode=entry.RecordingState()
        episode.request_start()
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)
            path=output/'episodes'/episode.human_demo_uuid
            self.assertFalse(entry.episode_has_started_capture(episode,output))
            self.assertIsNone(entry.submit_finished_episode(queue,path,capture_started=False))
            episode.on_tick(0)
            self.assertFalse(entry.episode_has_started_capture(episode,output))
            path.mkdir(parents=True)
            episode.reject('failed tracking')
            self.assertTrue(entry.episode_has_started_capture(episode,output))
            entry.submit_finished_episode(queue,path,capture_started=True)
            self.assertEqual(submitted,[(path,{'collection_enabled':True})])
            self.assertIsNone(entry.submit_finished_episode(None,path,capture_started=True))
            entry.submit_finished_episode(queue,path,capture_started=True,collection_enabled=False)
            self.assertEqual(submitted[-1],(path,{'collection_enabled':False}))
            wrong_queue=SimpleNamespace(submit=lambda *args,**options:{'status':'pending','uuid':'older'})
            rejected=entry.submit_finished_episode(wrong_queue,path,capture_started=True)
            self.assertEqual(rejected['status'],'error')
            self.assertEqual(rejected['uuid'],episode.human_demo_uuid)

    def test_no_active_discard_targets_latest_pending_uuid_and_never_falls_back(self):
        calls=[]
        queue=SimpleNamespace(request_discard=lambda uuid:calls.append(uuid) or
            {'status':'pending_discard','uuid':uuid,'pending':True})
        status=entry.discard_finished_episode(queue,'latest-pending-attempt')
        self.assertEqual(status['uuid'],'latest-pending-attempt')
        self.assertEqual(calls,['latest-pending-attempt'])
        self.assertEqual(entry.discard_finished_episode(queue,None)['status'],'empty')
        self.assertEqual(calls,['latest-pending-attempt'])
        self.assertEqual(entry.discard_finished_episode(None,'diagnostic')['status'],'disabled_diagnostic')
        with self.assertRaisesRegex(ValueError,'different episode identity'):
            entry.discard_finished_episode(SimpleNamespace(request_discard=lambda uuid:{'uuid':'old'}),
                'latest-pending-attempt')

    def test_background_results_are_matched_to_latest_uuid_and_cannot_revive_discard(self):
        pending={'status':'pending','uuid':'new','pending':True,'available':False}
        snapshot={'latest_finished_uuid':'old','latest_episode':{'status':'ready','uuid':'old'},
            'latest_completed':{'status':'ready','uuid':'old'},'latest_results':[]}
        uuid,status=entry.latest_review_status(snapshot,'new',pending)
        self.assertEqual(uuid,'new')
        self.assertEqual(status,pending)
        snapshot['latest_results']=[{'status':'ready','uuid':'new','number':8,'disposition':'kept'}]
        uuid,status=entry.latest_review_status(snapshot,'new',pending)
        self.assertEqual(status['number'],8)
        discard={**pending,'status':'pending_discard'}
        _,status=entry.latest_review_status(snapshot,'new',discard)
        self.assertEqual(status,discard)
        snapshot['latest_results']=[{'status':'discarded','uuid':'new','number':8,'discarded':True}]
        _,status=entry.latest_review_status(snapshot,'new',discard)
        self.assertTrue(status['discarded'])

    def test_raw_writers_are_sealed_without_geometry_or_physical_review(self):
        state=entry.RecordingState()
        state.request_start()
        for tick in range(51):
            state.on_tick(tick)
        state.request_save()
        calls=[]
        capture=SimpleNamespace(close=lambda **options:calls.append(('capture',options)) or
            {'status':'raw_physics_capture_completed_pending_rtx'})
        ledger=SimpleNamespace(close=lambda:calls.append(('ledger',{})) or {'event_count':31})
        streams=[SimpleNamespace(close=lambda label=label:calls.append((label,{})))
            for label in ('raw_quality','original_forces')]
        with patch.object(entry,'complete_quality_geometry',side_effect=AssertionError('worker only')):
            outcome=entry.seal_episode_writers(state,capture,ledger,streams,
                completed=True,reason=None,source_kind='live_pico_human',capture_started=True,deferred=True)
        self.assertEqual([name for name,_ in calls],['capture','raw_quality','original_forces','ledger'])
        self.assertEqual(calls[0][1],{'completed':True})
        self.assertTrue(outcome['geometry_pending'])
        self.assertNotIn('quality_geometry_sources',outcome)
        self.assertNotIn('physical_gate',outcome)
        self.assertEqual(outcome['status'],'pending_rtx')
        self.assertFalse(outcome['accepted_new_dataset'])

    def test_raw_seal_failure_still_closes_other_writers_and_rejects_attempt(self):
        state=entry.RecordingState()
        calls=[]
        def failed_capture(**options):
            calls.append('capture')
            raise OSError('disk full')
        outcome=entry.seal_episode_writers(state,SimpleNamespace(close=failed_capture),
            SimpleNamespace(close=lambda:calls.append('ledger')),
            [SimpleNamespace(close=lambda:calls.append('raw_quality'))],completed=False,
            reason='tracking lost',source_kind='live_pico_human',capture_started=True,deferred=True)
        self.assertEqual(calls,['capture','raw_quality','ledger'])
        self.assertIn('disk full',outcome['finalization_errors']['capture'])
        self.assertEqual(outcome['state']['status'],'rejected')
        self.assertFalse(outcome['state']['pending_audit'])

    def test_collection_message_identifies_number_and_remaining_acceptance(self):
        message=entry.completed_episode_message('save',{'number':1,'disposition':'kept'})
        self.assertIn('第001条',message)
        self.assertIn('仍待图像和数据验收',message)
        rejected=entry.completed_episode_message('save',{'number':2,'disposition':'discarded','reasons':['未完成交接']})
        self.assertIn('第002条',rejected)
        self.assertIn('未完成交接',rejected)
        pending=entry.completed_episode_message('save',{'status':'pending','pending':True})
        self.assertIn('后台正在物理复核',pending)
        self.assertNotIn('已通过',pending)


if __name__=='__main__':
    unittest.main()
