"""CPU-only source/timing/name checks; no SIMPLE env or GPU is constructed."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

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


def make_reference(root,source,binding):
    """Synthetic explicit reference bundle for producer/validator integration."""
    import prepare_handover_teacher_reference as reference
    bundle=root/'reference';bundle.mkdir()
    original=source['commands']
    commands=np.vstack([original,np.full((100,78),.25,np.float32)])
    applied=reference.held_commands(commands)
    def save(name,values):
        path=bundle/name;np.save(path,values,allow_pickle=False)
        return {'path':name,'sha256':reference.sha256(path),'rows':len(values),'columns':78,'dtype':'float32'}
    reference_descriptor=save('reference.npy',commands)
    applied_descriptor=save('held.npy',applied)
    repair=bundle/'result.json';write(repair,{'synthetic_fixture_only':True,
        'modified_command_schema':{'file':'held.npy','sha256':applied_descriptor['sha256']},
        'candidate_reset_identity':{key:binding[key] for key in ('candidate_id','candidate_spec_sha256','plan_sha256')}})
    parquet=next(path for path in source['sources'] if path.endswith('.parquet'))
    document={'contract':reference.CONTRACT,'scope':'TRAIN_teacher_only','source_episode_index':9,
        'parent_source_parquet_sha256':module.file_sha256(parquet),'parent_source_commands':len(original),
        'original_prefix_rows':len(original),'lineage_group_sources':source['row']['lineage_group_sources'],
        'command_rate_hz':50,'execution_command_rate_hz':30,'decoder_hz':50,'physics_hz':200,
        'body_token_version':'body_token_v1_1','hand_joint_names':list(module.HAND_NAMES),
        'commands':reference_descriptor,'actual_executed_commands':applied_descriptor,
        'prefix_verified':{'exact_native50_source_array_equal':True,'array_sha256':reference.array_sha256(original)},
        'startup_recipe':{'control':module.PADDED_WARMUP_MODE,'seconds':1.,'startup_padding_is_synthetic':True},
        'candidate_reset_binding':{key:binding[key] for key in ('candidate_id','candidate_spec_sha256','plan_sha256')},
        'fixed_encoder':{'sha256':reference.ENCODER_SHA256,'decoder_sha256':reference.DECODER_SHA256},
        'method':{'kind':'synthetic_fixture_no_physics_claim'},
        'repair_inputs_sha256':{str(repair):module.file_sha256(repair),str(bundle/'held.npy'):applied_descriptor['sha256']},
        'evaluation_oracle_allowed':False,'no_admission':True,'dataset_ready':False}
    path=bundle/'teacher_reference.json';write(path,document)
    return path,commands


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
    def test_teacher_reference_overlay_preserves_original_ancestry_reset_and_native_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_generic_source(root)
            source=module.load_source(root,source_episode=9)
            plan,_=make_plan(root);binding=module.read_candidate(plan,'registered-candidate')
            path,commands=make_reference(root,source,binding)
            original=deepcopy(source)
            overlaid=module.apply_teacher_reference(source,path,binding,warmup_control=module.PADDED_WARMUP_MODE,
                warmup_seconds=1.,command_rate=30)
            np.testing.assert_array_equal(source['commands'],original['commands'])
            np.testing.assert_array_equal(overlaid['commands'],commands)
            np.testing.assert_array_equal(overlaid['commands'][:4],source['commands'])
            self.assertEqual(overlaid['source_parent_rows'],4)
            self.assertEqual(overlaid['row'],source['row'])
            self.assertEqual(overlaid['initial'],source['initial'])
            self.assertEqual(overlaid['source_record'],source['source_record'])
            self.assertTrue(overlaid['teacher_reference']['no_admission'])
            self.assertEqual(overlaid['teacher_reference']['commands_rows'],104)
            self.assertTrue(overlaid['teacher_reference']['execution_requires_independent_admission'])
            for filename,identity in overlaid['teacher_reference_input_sha256'].items():
                self.assertEqual(overlaid['sources'][filename],identity)
                self.assertEqual(module.file_sha256(filename),identity)
            self.assertNotIn(str(path),source['sources'])
            # Read the native reference exactly once, then apply the real30Hz
            # hold; its separate applied table is evidence rather than input.
            actual=commands[[module.command_source_index(t,30,len(commands)) for t in range(len(commands))]]
            np.testing.assert_array_equal(actual,np.load(path.parent/'held.npy',allow_pickle=False))

    def test_teacher_reference_new_execution_is_explicit_and_rejects_wrong_source_startup_clock_or_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);make_generic_source(root)
            source=module.load_source(root,source_episode=9)
            plan,_=make_plan(root);binding=module.read_candidate(plan,'registered-candidate')
            path,commands=make_reference(root,source,binding)
            def apply(candidate=binding,**kwargs):
                args=dict(warmup_control=module.PADDED_WARMUP_MODE,warmup_seconds=1.,command_rate=30)
                args.update(kwargs)
                return module.apply_teacher_reference(source,path,candidate,**args)
            for kwargs in ({'command_rate':50},{'warmup_seconds':.5},{'warmup_control':'measured_pose_pd'}):
                with self.assertRaises(ValueError):apply(**kwargs)
            with self.assertRaisesRegex(ValueError,'registered candidate'):apply(None)
            wrong=deepcopy(binding);wrong['candidate_id']='another-reset'
            new_execution=apply(wrong)['teacher_reference']
            self.assertEqual(new_execution['repair_origin_candidate']['candidate_id'],binding['candidate_id'])
            self.assertEqual(new_execution['executed_candidate']['candidate_id'],'another-reset')
            self.assertTrue(new_execution['new_execution_differs_from_repair_origin'])
            self.assertTrue(new_execution['new_execution_explicitly_allowed'])
            wrong['proposal']['ancestor']['source_episode_index']=17
            with self.assertRaisesRegex(ValueError,'different source'):apply(wrong)
            commands[0,0]+=.01;np.save(path.parent/'reference.npy',commands,allow_pickle=False)
            document=json.loads(path.read_text());document['commands']['sha256']=module.file_sha256(path.parent/'reference.npy')
            write(path,document)
            with self.assertRaisesRegex(ValueError,'prefix was modified'):apply()

    def startup_history_fixture(self,tick):
        first=(.005,np.zeros(3),np.zeros(29),np.zeros(29),np.zeros(29),np.array([0.,0.,-1.]))
        def actual(index):
            return (first[0]+index/50,np.full(3,index/100),np.full(29,index/200),
                np.full(29,index/300),np.full(29,index/400),first[-1].copy())
        history=[deepcopy(first) for _ in range(max(0,9-tick))]
        history += [actual(index) for index in range(max(0,tick-9),tick+1)]
        return first,history

    def test_explicit_padding_fades_to_real_history_and_records_every994_input(self):
        from simple.baselines.sonic_v1_1 import decoder_input
        first,_=self.startup_history_fixture(0)
        adapter=module.RepeatFirstSnapshotDecoderHistory(first,decoder_input,{'sim_time_s':.005,'source':'unit-test live snapshot'})
        latent=np.full(64,.0625,np.float32)
        for tick in range(13):
            _,history=self.startup_history_fixture(tick)
            value=adapter(latent,history)
            event=adapter.receipt['decoder_inputs'][-1]
            self.assertEqual(value.shape,(1,994))
            self.assertEqual(event['padding_count'],max(0,9-tick))
            self.assertEqual(event['decoder_call_index'],tick)
            np.testing.assert_array_equal(np.asarray(event['input_values_float32'],np.float32),value[0])
            self.assertEqual(event['input_sha256'],hashlib.sha256(value.astype('<f4').tobytes()).hexdigest())
            self.assertTrue(all(t<=.005+tick/50+1e-9 for t in event['history_sample_times_s']))
            if tick>=9:np.testing.assert_array_equal(value,decoder_input(latent,history))
        self.assertEqual(adapter.receipt['padded_decoder_calls'],9)
        self.assertEqual(adapter.receipt['total_decoder_calls'],13)
        self.assertAlmostEqual(adapter.receipt['first_all_real_decoder_sim_time_s'],.185)
        self.assertTrue(adapter.receipt['history_injection'])
        self.assertEqual(adapter.receipt['decoder_inputs'][0]['history_sample_times_s'],[.005]*10)
        self.assertEqual(adapter.receipt['decoder_inputs'][9]['history_origin'][0]['actual_timestep_index'],0)

    def test_startup_padding_rejects_changed_initial_snapshot_and_future_or_delayed_slots(self):
        from simple.baselines.sonic_v1_1 import decoder_input
        first,history=self.startup_history_fixture(0)
        def adapter():return module.RepeatFirstSnapshotDecoderHistory(first,decoder_input,{'sim_time_s':.005})
        changed=deepcopy(history);changed[0][1][0]=1
        with self.assertRaisesRegex(ValueError,'first actual snapshot'):adapter()(np.zeros(64),changed)
        future=deepcopy(history);future[-1]=(.025,*future[-1][1:])
        with self.assertRaisesRegex(ValueError,'causal actual50Hz'):adapter()(np.zeros(64),future)
        value=adapter()
        for tick in range(10):value(np.zeros(64),self.startup_history_fixture(tick)[1])
        delayed=self.startup_history_fixture(10)[1];delayed[0]=deepcopy(first)
        with self.assertRaisesRegex(ValueError,'fade9..1'):value(np.zeros(64),delayed)
        self.assertEqual(value.receipt['total_decoder_calls'],10)

    def test_startup_adapter_is_instance_local_and_retains_exact_upstream_physics_code(self):
        import simple.baselines.sonic_v1_1 as sonic
        first,_=self.startup_history_fixture(0)
        adapter=module.RepeatFirstSnapshotDecoderHistory(first,sonic.decoder_input,{'sim_time_s':.005})
        one=object.__new__(sonic.SonicV11OnnxController)
        two=object.__new__(sonic.SonicV11OnnxController)
        original=two.apply_physics.__func__
        module.install_instance_startup_decoder(one,adapter)
        self.assertIs(one.apply_physics.__func__.__code__,original.__code__)
        self.assertIs(one.apply_physics.__func__.__globals__['decoder_input'],adapter)
        self.assertIs(two.apply_physics.__func__,original)
        self.assertIs(sonic.decoder_input,adapter.original_decoder_input)
        changed=[k for k,v in original.__globals__.items() if one.apply_physics.__func__.__globals__[k] is not v]
        self.assertEqual(changed,['decoder_input'])
        with self.assertRaisesRegex(ValueError,'unchanged upstream'):module.install_instance_startup_decoder(one,adapter)

    def test_new_warmup_metadata_does_not_change_legacy_live_receipt(self):
        from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
        old=object.__new__(LivePoseSonicWarmup);old.hand_policy='measured';old.reset()
        self.assertFalse(old.receipt['history_injection'])
        self.assertEqual(old.receipt['kind'],'live_initial_pose_sonic_v1_1')
        self.assertNotIn('seconds',old.receipt)
        def no_model_init(self,*args,**kwargs):self.hand_policy='measured';self.reset()
        with patch.object(LivePoseSonicWarmup,'__init__',no_model_init):
            new=module.make_repeat_first_snapshot_warmup('/unused',seconds=1.)
        self.assertEqual(new.receipt['kind'],module.PADDED_WARMUP_KIND)
        self.assertTrue(new.receipt['history_injection'])
        self.assertEqual(new.receipt['seconds'],1.)
        self.assertIn('tick0',new.receipt['hand_policy_scope'])
        self.assertIsNone(new.startup_decoder)

    def test_actual_post_step_outputs_and_per_event_padding_flags_are_not_inferred(self):
        from simple.baselines.sonic_v1_1 import decoder_input
        first,_=self.startup_history_fixture(0)
        adapter=module.RepeatFirstSnapshotDecoderHistory(first,decoder_input,{'sim_time_s':.005})
        warmup=SimpleNamespace(startup_decoder=adapter,receipt={'events':[]})
        controller=SimpleNamespace(_last_raw_action=np.arange(29,dtype=np.float32)/7,
            _body_target=np.arange(29,dtype=np.float64)/9)
        for tick in range(12):
            adapter(np.zeros(64),self.startup_history_fixture(tick)[1])
            warmup.receipt['events'].append({'sim_time_s':.005+tick/50,'history_injection':False})
            record={'tick50':tick,'sim_time_s':.005+tick/50,'decoder_calls_before':tick,'decoder_calls_after':tick+1}
            module.record_padded_decoder_step(warmup,controller,record,is_warmup=True)
            self.assertEqual(warmup.receipt['events'][-1]['history_injection'],tick<9)
            self.assertEqual(warmup.receipt['events'][-1]['startup_history_padding_slots'],max(0,9-tick))
            np.testing.assert_array_equal(record['controller_last_raw_policy_action29'],controller._last_raw_action)
            np.testing.assert_array_equal(record['body_pd_target29'],controller._body_target)
        # Teacher steps keep recording actual outputs, without relabelling or
        # appending warmup commands after the genuine warmup has ended.
        adapter(np.zeros(64),self.startup_history_fixture(12)[1])
        record={'tick50':12,'sim_time_s':.245,'decoder_calls_before':12,'decoder_calls_after':13}
        before=deepcopy(warmup.receipt['events'])
        module.record_padded_decoder_step(warmup,controller,record,is_warmup=False)
        self.assertEqual(warmup.receipt['events'],before)
        controller._last_raw_action[0]=np.nan
        with self.assertRaises(ValueError):module.record_padded_decoder_step(warmup,controller,record,is_warmup=False)
        controller._last_raw_action=np.zeros(29)
        controller._body_target=None
        with self.assertRaises(ValueError):module.record_padded_decoder_step(warmup,controller,record,is_warmup=False)
        controller._body_target=np.zeros(29)
        record['decoder_calls_after']=12
        with self.assertRaisesRegex(ValueError,'single physical decoder call'):
            module.record_padded_decoder_step(warmup,controller,record,is_warmup=False)

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
