"""CPU checks of deferred renderer's state and frame bridges, without RTX."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_deferred_pico import (assert_recorded_state_unchanged, canonical_event_sha256,
    restore_recorded_state, write_frame_bindings, recorded_task, recorded_environment,
    make_recorded_renderer, recorded_objects, recorded_environment_binding, recorded_grasp_frame)


class DeferredRendererTests(unittest.TestCase):
    def grasp_identity(self):
        return {'contract':'grasp-adapter-test-v1',
            'task_id':'cvpr/G1SonicTabletopGraspPico-v0',
            'source_official_task_id':'simple/G1WholebodyTabletopGraspMP-v0',
            'official_task_source_sha256':'a0f5d66a79e8afcc984bb20754639f2b157118aa8d31dc8886c2b952238975f4',
            'adapter_sha256':'b'*64}

    def grasp_metadata(self):
        identity=self.grasp_identity()
        return {'task_id':identity['task_id'],
            'source_official_task_id':identity['source_official_task_id'],
            'actual_task_source_sha256':identity['official_task_source_sha256'],
            'actual_native_success_criteria':.9,
            'grasp_adapter_identity':identity,
            'source_code_sha256_at_session_start':{'pico_grasp_task.py':identity['adapter_sha256']},
            'evidence_object_roles':['target'], 'object_body_names':{'target':'chips_can'}}

    def test_saved_environment_uses_the_recorded_task_profile(self):
        from pico_mobile_task import HANDOVER_TASK, MOBILE_TASK
        for task,profile in [(HANDOVER_TASK,'handover'),(MOBILE_TASK,'mobile_pp')]:
            with mock.patch('render_deferred_pico.load_environment_config',return_value=({},{})) as loader:
                recorded_environment('recorded.json','assets',task)
            loader.assert_called_once_with('recorded.json','assets',profile=profile)

    def test_recorded_task_dispatch_never_substitutes_handover_for_mobile(self):
        from pico_mobile_task import HANDOVER_TASK, MOBILE_TASK
        from pico_episode_history import MOBILE_OFFICIAL_SOURCE_SHA256
        self.assertEqual(recorded_task({}), HANDOVER_TASK)
        metadata={'task_id':MOBILE_TASK,'actual_task_source_sha256':MOBILE_OFFICIAL_SOURCE_SHA256,
                  'actual_native_success_criteria':.9}
        self.assertEqual(recorded_task(metadata), MOBILE_TASK)
        for changed in ({'task_id':'unknown'}, {'actual_task_source_sha256':'0'*64},
                        {'actual_native_success_criteria':True}, {'actual_native_success_criteria':.8}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                recorded_task({**metadata,**changed})

    def test_grasp_requires_official_predicate_adapter_bytes_and_target_only_roles(self):
        module=SimpleNamespace(adapter_identity=self.grasp_identity)
        metadata=self.grasp_metadata()
        with mock.patch.dict(sys.modules,{'pico_grasp_task':module}):
            self.assertEqual(recorded_task(metadata),metadata['task_id'])
            for changed in ({'source_official_task_id':'simple/G1WholebodyXMovePickTeleop-v0'},
                    {'actual_task_source_sha256':'0'*64}, {'actual_native_success_criteria':True},
                    {'actual_native_success_criteria':.4}, {'grasp_adapter_identity':None},
                    {'grasp_adapter_identity':{**self.grasp_identity(),'adapter_sha256':'c'*64}},
                    {'source_code_sha256_at_session_start':{'pico_grasp_task.py':'c'*64}},
                    {'source_code_sha256_at_session_start':None},
                    {'evidence_object_roles':['target','container']},
                    {'object_body_names':{'target':'chips_can','container':'table'}}):
                with self.subTest(changed=changed), self.assertRaises(ValueError):
                    recorded_task({**metadata,**changed})

    def test_grasp_archived_environment_is_not_adapted_twice(self):
        with mock.patch('render_deferred_pico.load_environment_config',return_value=({},{})) as loader:
            recorded_environment('recorded.json','assets',self.grasp_identity()['task_id'])
        loader.assert_called_once_with('recorded.json','assets',profile='grasp',grasp_config_is_adapted=True)

    def test_grasp_environment_uses_strict_uid_geometry_binding_helper(self):
        expected,actual={'saved':'environment'},{'reconstructed':'environment'}
        compare=mock.Mock(return_value={'matched':False,'reason':'material mismatch'})
        with mock.patch.dict(sys.modules,{'pico_grasp_task':SimpleNamespace(grasp_environment_binding_receipt=compare)}), \
                mock.patch('render_deferred_pico.environment_binding_receipt',side_effect=AssertionError('Wrong comparator')):
            result=recorded_environment_binding(self.grasp_identity()['task_id'],expected,actual)
        self.assertFalse(result['matched'])
        compare.assert_called_once_with(expected,actual)

    def test_grasp_frame_receipt_must_match_validated_scene_payload(self):
        frame={**self.grasp_identity(),'adapted_payload_sha256':'d'*64,
            'world_translation_m':[0.,0.,.785]}
        validate=mock.Mock(return_value=frame)
        environment={'pico_grasp_frame_adaptation':frame}
        with mock.patch.dict(sys.modules,{'pico_grasp_task':SimpleNamespace(validate_adapted_config=validate)}):
            self.assertEqual(recorded_grasp_frame({'grasp_frame_adaptation':copy.deepcopy(frame)},environment),frame)
            for changed in (None, {}, {**frame,'world_translation_m':[0.,0.,1.]},
                    {**frame,'adapted_payload_sha256':'e'*64}):
                with self.subTest(changed=changed), self.assertRaisesRegex(ValueError,'frame adaptation'):
                    recorded_grasp_frame({'grasp_frame_adaptation':changed},environment)
        validate.assert_called_with(environment)

    def test_changed_adapted_payload_is_rejected_before_renderer_construction(self):
        with mock.patch.dict(sys.modules,{'pico_grasp_task':SimpleNamespace(
                validate_adapted_config=mock.Mock(side_effect=ValueError('reset payload changed')))}):
            with self.assertRaisesRegex(ValueError,'payload changed'):
                recorded_grasp_frame({'grasp_frame_adaptation':{}},{})

    def test_grasp_constructs_own_task_and_installs_frame_hooks_before_reset(self):
        calls=[]
        task=object()
        env=SimpleNamespace(reset=lambda: calls.append('reset'), close=mock.Mock())
        def install(actual):
            self.assertIs(actual,env)
            calls.append('install')
            return {'installed':True}
        module=SimpleNamespace(make_grasp_task=mock.Mock(return_value=task),install_frame_adapter=install)
        constructor=mock.Mock(return_value=env)
        config={'INTERFACE':'lo'}
        environment={'uid':'g1_wholebody_tabletop_grasp_mp','already_translated':True}
        with mock.patch.dict(sys.modules,{'pico_grasp_task':module,
                'simple.envs.sonic_loco_manip':SimpleNamespace(SonicLocoManipEnv=constructor),
                'gymnasium':SimpleNamespace(make=mock.Mock(side_effect=AssertionError('No registered Mobile/Handover substitution')))}):
            actual,binding=make_recorded_renderer(self.grasp_identity()['task_id'],config,environment)
        self.assertIs(actual,env)
        self.assertEqual(binding,{'installed':True})
        self.assertEqual(calls,['install'])
        env.reset()
        self.assertEqual(calls,['install','reset'])
        module.make_grasp_task.assert_called_once_with(config,adapted_config=environment)
        constructor.assert_called_once_with(task=task,sonic_config=config,sim_mode='mujoco_isaac',
            headless=True,dr_level=0,sync_mujoco_observations=True,isaac_observation_only=True)

    def test_failed_grasp_hook_closes_only_its_new_environment(self):
        env=SimpleNamespace(close=mock.Mock())
        module=SimpleNamespace(make_grasp_task=mock.Mock(return_value=object()),
            install_frame_adapter=mock.Mock(side_effect=ValueError('frame mismatch')))
        with mock.patch.dict(sys.modules,{'pico_grasp_task':module,
                'simple.envs.sonic_loco_manip':SimpleNamespace(SonicLocoManipEnv=mock.Mock(return_value=env))}):
            with self.assertRaisesRegex(ValueError,'frame mismatch'):
                make_recorded_renderer(self.grasp_identity()['task_id'],{}, {})
        env.close.assert_called_once_with()

    def test_legacy_renderer_registry_arguments_are_preserved(self):
        from pico_mobile_task import HANDOVER_TASK, MOBILE_TASK
        for task in (HANDOVER_TASK,MOBILE_TASK):
            make=mock.Mock()
            with self.subTest(task=task), mock.patch.dict(sys.modules,{'gymnasium':SimpleNamespace(make=make)}):
                env,binding=make_recorded_renderer(task,{'SIMULATE_DT':.005},{})
                self.assertIs(env,make.return_value)
                self.assertIsNone(binding)
                make.assert_called_once_with(task,sonic_config={'SIMULATE_DT':.005},
                    sim_mode='mujoco_isaac',headless=True,dr_level=0,
                    sync_mujoco_observations=True,isaac_observation_only=True,disable_env_checker=True)

    def test_target_only_objects_do_not_remove_distractors_from_renderer_view(self):
        target=SimpleNamespace(xpos=np.array([.1,.2,.8]),xquat=np.array([1.,0.,0.,0.]))
        view=SimpleNamespace(mj_objects={'target':target,'distractor_1':object()})
        event={'objects':{'target':{'pose7':[.1,.2,.8,1.,0.,0.,0.]}}}
        objects=recorded_objects(view,event,('target',))
        self.assertEqual(set(objects),{'target'})
        self.assertEqual(set(view.mj_objects),{'target','distractor_1'})
        with self.assertRaisesRegex(ValueError,'roles'):
            recorded_objects(view,event,('target','container'))
        event['objects']['target']['pose7'][2]+=.001
        with self.assertRaisesRegex(ValueError,'target'):
            recorded_objects(view,event,('target',))

    def test_full_state_roundtrip_never_forwards_or_steps_and_detects_non_qpos_mutation(self):
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="pelvis"><freejoint/>'
            '<geom type="sphere" size=".1"/><camera name="head_stereo_left"/></body></worldbody></mujoco>')
        data=mujoco.MjData(model)
        data.time=1.25
        data.qvel[:]=np.arange(model.nv)*.1
        data.qacc_warmstart[:]=np.arange(model.nv)+10.
        data.qfrc_applied[:]=np.arange(model.nv)+20.
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        state=np.empty(mujoco.mj_stateSize(model,spec))
        mujoco.mj_getState(model,data,state,spec)
        destination=mujoco.MjData(model)
        with mock.patch('mujoco.mj_forward',side_effect=AssertionError('No forward/solver')), \
                mock.patch('mujoco.mj_step',side_effect=AssertionError('No integration')):
            before=restore_recorded_state(model,destination,state,spec)
            assert_recorded_state_unchanged(model,destination,before,spec)
        self.assertTrue(np.array_equal(before,state))
        destination.qacc_warmstart[0]+=.01
        with self.assertRaisesRegex(ValueError,'full integration'):
            assert_recorded_state_unchanged(model,destination,before,spec)

    def event(self):
        return {'event_index':0,'capture_index30':0,'physical_tick50':0,'sim_time_s':1.,
            'nominal_elapsed_s':0.,'actual_elapsed_s':0.,'phase':'warmup','action_valid':False,
            'source_command_index':None,'joint_names':['joint'],'state43':[.1],
            'physical_joint_q43_float64':[.1],'physical_joint_qvel43':[.2],
            'root_pose7':[0.,0.,.8,1.,0.,0.,0.],'root_qvel6':[0.]*6,
            'objects':{'target':{'pose7':[0.,0.,.3,1.,0.,0.,0.]},
                       'container':{'pose7':[.2,0.,.3,1.,0.,0.,0.]}},'action':None,
            'integration_state_sha256':'a'*64,'source_fields':{'human':'原始示范'}}

    def test_canonical_hash_and_each_frame_state_rgb_bindings(self):
        source=self.event()
        expected=hashlib.sha256(json.dumps(source,ensure_ascii=False,allow_nan=False,
            sort_keys=True,separators=(',',':')).encode('utf-8')).hexdigest()
        self.assertEqual(canonical_event_sha256(source),expected)
        self.assertEqual(canonical_event_sha256(dict(reversed(list(source.items())))),expected)
        with tempfile.TemporaryDirectory() as directory:
            event=copy.deepcopy(source)
            event.pop('integration_state_sha256'); event.pop('source_fields')
            event['rgb']={'pixel_sha256':'b'*64,'file_sha256':'c'*64}
            path=Path(directory)/'events.jsonl'
            path.write_text(json.dumps(event)+'\n')
            bridge=Path(directory)/'rtx_frame_bindings.jsonl'
            receipt=write_frame_bindings(bridge,[source],path)
            self.assertTrue(receipt['full_integration_state_unchanged'])
            self.assertEqual(receipt['frame_bindings_count'],1)
            self.assertEqual(receipt['frame_bindings_sha256'],hashlib.sha256(bridge.read_bytes()).hexdigest())
            row=json.loads(bridge.read_text())
            self.assertEqual(row['source_event_sha256'],expected)
            self.assertEqual(row['rtx_event_sha256'],canonical_event_sha256(event))
            self.assertEqual(row['integration_state_sha256'],'a'*64)
            self.assertEqual(row['rgb_pixel_sha256'],'b'*64)
            self.assertEqual(row['rgb_file_sha256'],'c'*64)

    def test_target_only_bridge_keeps_exact_original_object_evidence(self):
        source=self.event()
        source['objects'].pop('container')
        event=copy.deepcopy(source)
        event['rgb']={'pixel_sha256':'b'*64,'file_sha256':'c'*64}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.jsonl'
            path.write_text(json.dumps(event)+'\n')
            bridge=Path(directory)/'bridge.jsonl'
            receipt=write_frame_bindings(bridge,[source],path)
            self.assertTrue(receipt['full_integration_state_unchanged'])
            self.assertEqual(json.loads(path.read_text())['objects'],source['objects'])
            self.assertEqual(set(source['objects']),{'target'})

    def test_bridge_rejects_changed_original_action_state_or_count(self):
        for field in ('action','root_pose7','physical_tick50','missing'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                source=self.event(); event=copy.deepcopy(source)
                if field=='action':
                    event['action']={'body_token64':[.123456789012345]}
                elif field=='root_pose7':
                    event[field][0]+=.001
                elif field=='physical_tick50':
                    event[field]=2
                path=Path(directory)/'events.jsonl'
                path.write_text('' if field=='missing' else json.dumps(event)+'\n')
                with self.assertRaisesRegex(ValueError,'differ|changed'):
                    write_frame_bindings(Path(directory)/'bridge.jsonl',[source],path)


if __name__=='__main__':
    unittest.main()
