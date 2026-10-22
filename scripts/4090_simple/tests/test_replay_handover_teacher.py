"""CPU-only source/timing/name checks; no SIMPLE env or GPU is constructed."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import replay_handover_teacher as module
sys.path.insert(0, str(Path(__file__).resolve().parents[3]/"VLMExpert/third_party/SIMPLE/src"))
from simple.core.action import ActionCmd


def write(path, value):
    module.write_json(path, value)


def make_source(root):
    train = root / "training/Handover-sonic11-train"
    train.mkdir(parents=True)
    names = list(module.HAND_NAMES)
    training_names = names[:3] + names[5:7] + names[3:5] + names[7:10] + names[12:14] + names[10:12]
    all_joint_names = [f"body_{i}" for i in range(29)] + names
    object_names = [f"{role}.{term}" for role in ("container", "target")
                    for term in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
    info = {"fps":50,"sonic_version":"1.1","features":{"action":{"names":training_names},
            "observation.object_poses":{"names":object_names}}}
    write(train / "meta/info.json", info)
    write(root / "replay/meta/info.json", {"fps":50,"features":{"observation.state":{"names":all_joint_names}}})
    write(root / "training/selection.json", {"episodes":[{"source_scene_episode_index":0,"split":"train"}]})
    row = {"episode_index":0,"source_episode_index":0,"source_scene_episode_index":0,"lineage_complete":True,
        "hand_target_semantics":"collection-effective","lineage_group_sources":[{"source_episode_index":0}],
        "length":4,"environment_config":json.dumps({"uid":"g1_wholebody_handover_sonic"})}
    (train / "meta/episodes.jsonl").write_text(json.dumps(row)+"\n")
    tokens=np.arange(256,dtype=np.float32).reshape(4,64)/1000
    hands=np.arange(56,dtype=np.float32).reshape(4,14)/100
    natural=hands[:,[training_names.index(n) for n in module.HAND_NAMES]]
    replay=np.concatenate([tokens,natural],axis=1)
    data=train/'data/chunk-000/episode_000000.parquet'
    data.parent.mkdir(parents=True)
    objects=[0,0,.55,1,0,0,0, .2,0,.65,1,0,0,0]
    pq.write_table(pa.table({'action.body_token_v1_1':tokens.tolist(),'action':hands.tolist(),
        'replay_action78':replay.tolist(),'observation.base_pose':[[0,0,.75,1,0,0,0]]*4,
        'observation.base_vel':[[0]*6]*4,'observation.object_poses':[objects]*4,
        'timestamp':np.arange(4)/50}),data)
    p=root/'replay/validation';p.mkdir(parents=True)
    reset=np.arange(43,dtype=float)/100
    write(p/'source_episode_000000_physics.json',{'reset_joint_names':all_joint_names,'reset_joint_state43':reset.tolist(),
        'objects':{role:{'mass':.1} for role in ['target','container']}})
    write(p/'source_episode_000000_controller_reset.json',{'vector_sha256':hashlib.sha256(reset.astype('<f8').tobytes()).hexdigest()})
    np.save(p/'source_episode_000000_reset_joint_state43.npy',reset)
    write(p/'source_episode_000000_initial_objects.json',{'objects':{role:{'actual_pose7':v,'recorded_pose7':v}
        for role,v in [('container',objects[:7]),('target',objects[7:])]}})
    files=[p for p in root.rglob('*') if p.is_file()]
    write(root/'source0_transfer_audit.json',{'status':'source0_small_data_and_reset_sha256_verified','files':[
        {'path':str(p.relative_to(root)),'sha256':module.file_sha256(p),'bytes':p.stat().st_size} for p in files]})
    return replay


def make_generic_source(root):
    """Small receipt with24 registered families, real files for source0 and9."""
    expected = make_source(root)
    train=root/'training/Handover-sonic11-train'
    key='fixture/original'
    selection=[{'source_scene_episode_index':i,'source_dataset_key':key,
        'split':'train','data_category':'original_or_repaired'} for i in range(24)]
    write(root/'training/selection.json',{'episodes':selection})
    row=json.loads((train/'meta/episodes.jsonl').read_text())
    rows=[]
    for sid,dense in [(0,0),(9,2)]:
        item=dict(row,episode_index=dense,source_episode_index=sid,source_scene_episode_index=sid,
            source_dataset_key=key,data_category='original_or_repaired',split_restriction='unrestricted',
            lineage_group_sources=[{'source_dataset_key':key,'source_episode_index':sid}])
        rows.append(item)
        if sid:
            (train/f'data/chunk-000/episode_{dense:06d}.parquet').write_bytes(
                (train/'data/chunk-000/episode_000000.parquet').read_bytes())
            for path in (root/'replay/validation').glob('source_episode_000000_*'):
                path.with_name(path.name.replace('000000',f'{sid:06d}')).write_bytes(path.read_bytes())
    (train/'meta/episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    extract=root/'release_training_manifest_extract.json';write(extract,{'derived_not_original':True})
    def entry(path):return {'path':str(path.relative_to(root)),'sha256':module.file_sha256(path),'bytes':path.stat().st_size}
    files=[entry(path) for path in root.rglob('*') if path.is_file()
        and path.name not in ('source0_transfer_audit.json','release_training_manifest_extract.json')]
    episodes=[{'source_episode_index':sid,'split':'train'} for sid in range(24)]
    for sid,dense in [(0,0),(9,2)]:
        roles={'parquet':entry(train/f'data/chunk-000/episode_{dense:06d}.parquet')}
        roles.update({role:entry(root/f'replay/validation/source_episode_{sid:06d}_{suffix}')
            for role,suffix in [('physics','physics.json'),('controller_reset','controller_reset.json'),
                ('reset_joint_state43','reset_joint_state43.npy'),('initial_objects','initial_objects.json')]})
        episodes[sid].update(train_episode_index=dense,source_dataset_key=key,required_files=roles,
            parquet=roles['parquet'],source_rows=4,plan_declared_source_parquet_sha256='a'*64,
            lineage_group_sources=[{'source_dataset_key':key,'source_episode_index':sid}])
    write(root/'source_transfer_audit.json',{'schema':'cvpr-handover-train-source-transfer-v1',
        'status':'all24_source_files_sha256_verified','files':files,'episodes':episodes,
        'source_release_manifest_extract':entry(extract)})
    return expected


def make_plan(root, *, source=9):
    ancestor={'source_dataset_key':'fixture/original','source_episode_index':source,
        'source_original_category':'original_or_repaired','source_parquet_sha256':'a'*64,'source_video_sha256':'b'*64}
    spec={'ancestor':ancestor,'variant':0,'physics_seed':11,'material_seed':12,'generation_seed':13,
        'initial_object_deltas':{role:{'xy_delta_m':[.001,-.002],'world_z_yaw_delta_rad':.03}
            for role in ('target','container')}}
    candidate={**spec,'candidate_id':'registered-candidate','candidate_spec_sha256':module.canonical_sha256(spec),
        'split':'train','accepted':False}
    path=root/'candidate_plan.json'
    write(path,{'contract':'cvpr-handover-quality100-candidate-plan-v1','candidates':[candidate],
        'original_ancestors':[ancestor]})
    return path,candidate


def renderer_fixture(sim_time_s):
    """Synthetic pose protocol only; this never renders or certifies a camera."""
    from copy import deepcopy
    links={name:{'position_m':[0.,0.,.75],'quaternion_wxyz':[1.,0.,0.,0.]} for name in module.RENDERER_LINKS}
    actual={'links':deepcopy(links),'cameras':{'head_stereo_left':{
        'position_m':[0.,0.,1.2],'rotation_usd_axes':np.eye(3).reshape(-1).tolist()}}}
    # Keep a known mount offset in raw evidence, never assert USD/MJ equivalence.
    diagnostics={'mujoco_sim_time_s':sim_time_s,'isaac_world_time_s':.004,
        'links':{name:{'usd':{'path':'/World/robot/'+name,'valid':True,
            'position_m':[.01,0.,.75],'quaternion_wxyz':[1.,0.,0.,0.]},
            'mujoco_position_m':item['position_m'],'mujoco_quaternion_wxyz':item['quaternion_wxyz']}
            for name,item in links.items()},
        'cameras':{'head_stereo_left':{'usd':{'path':'/World/robot/d435_link/head_stereo_left','valid':True,
            'position_m':[.01,0.,1.2],'quaternion_wxyz':[1.,0.,0.,0.]},
            'sensor_position_m':[.01,0.,1.2],'sensor_quaternion_wxyz_usd_axes':[1.,0.,0.,0.],
            'mujoco_position_m':[0.,0.,1.2],'mujoco_rotation_usd_axes':np.eye(3).reshape(-1).tolist()}},
        'scope':'synthetic protocol fixture only, not actual rendered image evidence'}
    return diagnostics,actual


class ReplayHandoverTeacherTests(unittest.TestCase):
    def test_actual_calibration_asset_binding_rejects_missing_or_wrong_bytes_and_rechecks_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)/'data-root';root.mkdir()
            resource=Path(directory)/'actual-resource.urdf';resource.write_bytes(b'actual source asset bytes')
            relative='robots/model.urdf';(root/'robots').mkdir();(root/relative).symlink_to(resource)
            expected={relative:hashlib.sha256(resource.read_bytes()).hexdigest()}
            actual_root,observed=module.bound_calibration_assets(root,expected)
            self.assertEqual(actual_root,root.resolve())
            self.assertEqual(observed,expected)
            resource.write_bytes(b'changed actual source asset bytes')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):module.bound_calibration_assets(root,expected)
            resource.unlink()
            with self.assertRaisesRegex(ValueError,'Missing actual'):module.bound_calibration_assets(root,expected)

    def test_actual_teacher_hand_scalars_and_numpy_state_serialize_numerically(self):
        reference=np.arange(78,dtype=np.float32)/100
        # Reproduce the first real teacher command that previously failed after
        # fifty successful warmup rows: dict(zip(...)) retains np.float32 values.
        record={'actual_body_token64':reference[:64],
            'actual_hand_targets':dict(zip(module.HAND_NAMES,reference[64:78])),
            'state':np.array([.1,.2],np.float64),'step':np.int64(51),'finite':np.bool_(True)}
        decoded=json.loads(module.json_text(record))
        self.assertEqual(decoded['actual_hand_targets'][module.HAND_NAMES[0]],float(reference[64]))
        self.assertEqual(decoded['step'],51)
        self.assertIs(decoded['finite'],True)
        self.assertIsInstance(decoded['actual_hand_targets'][module.HAND_NAMES[0]],float)
        for invalid in [np.float32(np.nan),np.array([np.inf])]:
            with self.assertRaises(ValueError):module.json_text({'value':invalid})
        with self.assertRaises(TypeError):module.json_text({'value':object()})

    def test_true30hz_command_hold_index_and_no_future_or_tail(self):
        got=[module.command_source_index(j,30,650) for j in range(650)]
        self.assertEqual(got[:10],[0,0,2,2,4,5,5,7,7,9])
        self.assertTrue(all(0<=j-index<=1 for j,index in enumerate(got)))
        self.assertEqual([module.command_source_index(j,50,650) for j in range(650)],list(range(650)))
        with self.assertRaises(ValueError):module.command_source_index(650,30,650)

    def test_named_hand_permutation_equals_actual_replay_and_catches_double_swap(self):
        names=list(reversed(module.HAND_NAMES)); hands=np.arange(14,dtype=np.float32)[None]
        tokens=np.zeros((1,64),np.float32)
        expected=np.concatenate([tokens,hands[:,::-1]],axis=1)
        np.testing.assert_array_equal(module.repack_commands(tokens,hands,names,expected),expected)
        with self.assertRaisesRegex(ValueError,'actual replay'):
            module.repack_commands(tokens,hands,list(module.HAND_NAMES),expected)

    def test_positive_train_source_load_uses_parquet_recorded_rows_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expected=make_source(root)
            loaded=module.load_source0(root)
            np.testing.assert_array_equal(loaded['commands'],expected)
            self.assertEqual(loaded['source_split'],'train')
            self.assertEqual(len(loaded['commands']),4)

    def test_heldout_source_rejected_before_loading(self):
        for source in [39,65,80,92,94,-1,True]:
            with self.assertRaisesRegex(ValueError,'TRAIN ancestors'):
                module.load_source0('/nonexistent',source_episode=source)

    def test_generic_nonzero_train_dense_mapping_and_registered_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expected=make_generic_source(root)
            source=module.load_source(root,source_episode=9)
            self.assertEqual(source['row']['episode_index'],2)
            np.testing.assert_array_equal(source['commands'],expected)
            path,candidate=make_plan(root)
            binding=module.read_candidate(path,candidate['candidate_id'])
            prepared,receipt=module.apply_candidate(source,binding)
            self.assertEqual(receipt['apply_count'],1)
            self.assertEqual(receipt['phase'],'before_warmup')
            for role in ('target','container'):
                original=np.asarray(source['initial']['objects'][role]['pose7'])
                changed=np.asarray(prepared['initial']['objects'][role]['pose7'])
                np.testing.assert_allclose(changed[:2]-original[:2],[.001,-.002],rtol=0,atol=1e-16)
                np.testing.assert_allclose(changed[3:],[np.cos(.015),0,0,np.sin(.015)],atol=1e-15)
                np.testing.assert_array_equal(receipt['objects'][role]['source_pose7'],original)
            with self.assertRaisesRegex(ValueError,'only be applied once'):module.apply_candidate(prepared,binding)
            candidate['physics_seed']=999
            plan=json.loads(path.read_text());plan['candidates']=[candidate];write(path,plan)
            with self.assertRaisesRegex(ValueError,'hash'):module.read_candidate(path,candidate['candidate_id'])

    def test_cross_layer_parquet_hash_and_generated_or_heldout_ancestors_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_generic_source(root);path,candidate=make_plan(root)
            source=module.load_source(root,source_episode=9)
            binding=module.read_candidate(path,candidate['candidate_id'])
            binding['proposal']['ancestor']['source_parquet_sha256']=source['source_record']['parquet']['sha256']
            with self.assertRaisesRegex(ValueError,'layered'):module.apply_candidate(source,binding)
            transfer=json.loads((root/'source_transfer_audit.json').read_text())
            selection=root/'training/selection.json';value=json.loads(selection.read_text())
            value['episodes'][9]['data_category']='task_scene_regeneration';write(selection,value)
            for entry in transfer['files']:
                if entry['path']=='training/selection.json':entry.update(bytes=selection.stat().st_size,sha256=module.file_sha256(selection))
            write(root/'source_transfer_audit.json',transfer)
            with self.assertRaisesRegex(ValueError,'24 original TRAIN'):module.load_source(root,source_episode=9)

    def test_dense_real30hz_rgb_capture_warmup_no_labels_and_exact_named_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);names=[f'body_{i}' for i in range(29)]+list(module.MODEL_HAND_NAMES)
            writer=module.TrainingCapture(root/'capture',metadata={'accepted_new_dataset':False},
                joint_names=names,warmup_ticks=50,start_time=4.)
            rgb=np.arange(8*10*3,dtype=np.uint8).reshape(8,10,3)
            physical=np.arange(43,dtype=np.float64)/100
            command=ActionCmd('sonic_v1_1',controller=object(),body_token=np.arange(64,dtype=np.float32)/100,
                hand_targets=dict(zip(module.HAND_NAMES,np.arange(14,dtype=np.float32))))
            self.assertFalse(hasattr(command,'get'))  # Exact production interface that failed on RTX.
            for tick in range(58):
                if module.capture_tick_index(tick) is None:continue
                renderer,authoritative=renderer_fixture(4+tick/50)
                writer.append(tick50=tick,sim_time_s=4+tick/50,rgb=rgb,
                    observed_joint_names=names[::-1],observed_qpos=physical[::-1],
                    physical_qpos=physical,physical_qvel=np.zeros(43),root_pose7=[0,0,.75,1,0,0,0],
                    root_qvel6=np.zeros(6),objects={r:{'pose7':[0,0,.6,1,0,0,0]} for r in ('target','container')},
                    command=None if tick<50 else command,renderer_pose_diagnostics=renderer,
                    renderer_authoritative_poses=authoritative,observation_sim_time_s=4+tick/50)
            manifest=writer.close(completed=True)
            events=[json.loads(line) for line in (root/'capture/events.jsonl').read_text().splitlines()]
            self.assertEqual([e['physical_tick50'] for e in events[:7]],[0,2,4,5,7,9,10])
            self.assertEqual(manifest['warmup_event_count'],30)
            self.assertEqual(manifest['renderer_pose_contract'],module.RENDERER_POSE_CONTRACT)
            self.assertEqual(manifest['isaac_world_time_reference_s'],.004)
            self.assertFalse(manifest['render_mount_equivalence_verified'])
            self.assertFalse(manifest['accepted_new_dataset'])
            self.assertTrue(all(e['action'] is None and not e['action_valid'] for e in events[:30]))
            self.assertEqual(events[30]['source_command_index'],0)
            self.assertEqual(events[31]['source_command_index'],2)
            self.assertEqual(events[30]['action']['hand14'],[float(command['hand_targets'][n]) for n in module.MODEL_HAND_NAMES])
            from PIL import Image
            first=events[0]['rgb'];path=root/'capture'/first['path']
            np.testing.assert_array_equal(np.asarray(Image.open(path)),rgb)
            self.assertEqual(module.file_sha256(path),first['file_sha256'])
            self.assertEqual(hashlib.sha256(rgb.tobytes()).hexdigest(),first['pixel_sha256'])
            self.assertEqual(events[30]['renderer_camera'],'head_stereo_left')
            self.assertEqual(events[30]['renderer_pose_diagnostics']['mujoco_sim_time_s'],5.)
            self.assertFalse(events[30]['renderer_pose_validation']['render_mount_equivalence_verified'])

    def test_real_simple_actioncmd_parameters_and_wrong_dispatch_fail_closed(self):
        command=ActionCmd('sonic_v1_1',controller=object(),body_token=np.zeros(64,np.float32),
            hand_targets=dict.fromkeys(module.HAND_NAMES,0.))
        self.assertIs(module.sonic_command_parameters(command),command.parameters)
        self.assertFalse(hasattr(command,'get'))
        self.assertNotIn('simple.envs',sys.modules)
        for invalid in [command.parameters,ActionCmd('joint_position',**command.parameters),
                ActionCmd('sonic_v1_1',body_token=np.zeros(64),hand_targets={}),
                ActionCmd('sonic_v1_1',controller=None,body_token=np.zeros(64),hand_targets={})]:
            with self.assertRaises(ValueError):module.sonic_command_parameters(invalid)
        command.parameters=None
        with self.assertRaises(ValueError):module.sonic_command_parameters(command)

    def test_renderer_diagnostic_is_immutable_actual_clock_snapshot_with_mount_limit_explicit(self):
        diagnostics,actual=renderer_fixture(1.005)
        saved,receipt=module.renderer_pose_snapshot(diagnostics,sim_time_s=1.005,observation_sim_time_s=1.005,
            authoritative_poses=actual,world_time_reference_s=.004)
        self.assertEqual(saved,diagnostics)
        self.assertIsNot(saved,diagnostics)
        diagnostics['links']['pelvis']['usd']['position_m'][0]=9
        self.assertEqual(saved['links']['pelvis']['usd']['position_m'][0],.01)
        self.assertAlmostEqual(receipt['usd_mujoco_camera_position_l2_m'],.01)
        self.assertFalse(receipt['render_mount_equivalence_verified'])
        self.assertNotIn('simple.engines.isaacsim',sys.modules)

    def test_renderer_missing_stale_invalid_prim_or_camera_sensor_evidence_rejected(self):
        from copy import deepcopy
        diagnostics,actual=renderer_fixture(1.005)
        def validate(value,**kwargs):
            return module.renderer_pose_snapshot(value,sim_time_s=1.005,
                observation_sim_time_s=kwargs.get('observation_sim_time_s',1.005),
                authoritative_poses=actual,world_time_reference_s=.004)
        with self.assertRaisesRegex(ValueError,'required'):validate(None)
        with self.assertRaisesRegex(ValueError,'clocks'):validate(diagnostics,observation_sim_time_s=.985)
        for mutate,message in [
            (lambda d:d.update(mujoco_sim_time_s=.985),'clocks'),
            (lambda d:d.update(isaac_world_time_s=.00401),'world time advanced'),
            (lambda d:d['links'].pop('pelvis'),'lack required'),
            (lambda d:d['links']['pelvis']['usd'].update(valid=False),'Invalid actual USD'),
            (lambda d:d['links']['pelvis']['mujoco_position_m'].__setitem__(0,.001),'link evidence'),
            (lambda d:d['cameras']['head_stereo_left'].update(mujoco_camera_missing=True),'camera is missing'),
            (lambda d:d['cameras']['head_stereo_left']['sensor_position_m'].__setitem__(0,.012),'sensor and its USD'),
            (lambda d:d['cameras']['head_stereo_left']['mujoco_position_m'].__setitem__(0,.001),'camera evidence'),
            (lambda d:d['cameras']['head_stereo_left']['sensor_quaternion_wxyz_usd_axes'].__setitem__(0,np.nan),'finite'),
        ]:
            value=deepcopy(diagnostics);mutate(value)
            with self.subTest(message=message),self.assertRaisesRegex(ValueError,message):validate(value)

    def test_capture_rejects_stale_state_clock_and_applied_force_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            names=[f'body_{i}' for i in range(29)]+list(module.MODEL_HAND_NAMES)
            writer=module.TrainingCapture(Path(directory)/'capture',metadata={},joint_names=names,warmup_ticks=50,start_time=0.)
            kwargs=dict(tick50=0,sim_time_s=0.,rgb=np.zeros((2,3,3),np.uint8),observed_joint_names=names,
                observed_qpos=np.zeros(43),physical_qpos=np.ones(43),physical_qvel=np.zeros(43),
                root_pose7=[0,0,.75,1,0,0,0],root_qvel6=np.zeros(6),
                objects={r:{'pose7':[0,0,.6,1,0,0,0]} for r in ('target','container')},command=None)
            with self.assertRaisesRegex(ValueError,'same current frame'):writer.append(**kwargs)
            kwargs.update(physical_qpos=np.zeros(43))
            with self.assertRaisesRegex(ValueError,'renderer pose diagnostics'):writer.append(**kwargs)
            kwargs.update(physical_qpos=np.zeros(43),sim_time_s=.02)
            with self.assertRaisesRegex(ValueError,'actual sim clock'):writer.append(**kwargs)
            writer.close(completed=False)
        data=SimpleNamespace(qfrc_applied=np.zeros(4),xfrc_applied=np.zeros((2,6)))
        self.assertTrue(module.applied_force_state(data)['all_zero'])
        data.xfrc_applied[1,2]=1e-10
        self.assertFalse(module.applied_force_state(data)['all_zero'])
        data.xfrc_applied[1,2]=np.nan
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):module.applied_force_state(data)

    def test_sha_change_and_train_to_val_change_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_source(root)
            path=root/'training/selection.json'
            path.write_text(path.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'SHA/size'):
                module.load_source0(root)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_source(root)
            path=root/'training/selection.json'
            write(path,{'episodes':[{'source_scene_episode_index':0,'split':'val'}]})
            receipt=json.loads((root/'source0_transfer_audit.json').read_text())
            for entry in receipt['files']:
                if entry['path']=='training/selection.json':
                    entry.update(sha256=module.file_sha256(path),bytes=path.stat().st_size)
            write(root/'source0_transfer_audit.json',receipt)
            with self.assertRaisesRegex(ValueError,'TRAIN'):
                module.load_source0(root)

    def test_solver_hook_only_samples_after_real_steps_and_not_warmup(self):
        events=[];phase={'tick':None}
        def real_step():events.append('physics');return 7
        wrapped=module.solved_step_hook(real_step,lambda:phase['tick'],lambda step:events.append(('sample',step)))
        for _ in range(4):self.assertEqual(wrapped(),7)
        self.assertEqual(events,['physics']*4)
        phase['tick']=0
        for _ in range(4):wrapped()
        self.assertEqual(events[-2:],['physics',('sample',1)])
        phase['tick']=1
        for _ in range(4):wrapped()
        self.assertEqual(events[-2:],['physics',('sample',2)])
        all_steps=[];phase['tick']=None
        wrapped=module.solved_step_hook(lambda:all_steps.append('step'),lambda:None,lambda _:None,
            post_substep=lambda:all_steps.append('force_check'))
        for _ in range(8):wrapped()
        self.assertEqual(all_steps,['step','force_check']*8)


if __name__=='__main__':unittest.main()
