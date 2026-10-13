"""Independent parameter oracle and full row-preservation negative controls."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec=importlib.util.spec_from_file_location('parameter_auditor',Path(__file__).parents[1]/'scripts/audit_smplh_parameters.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


@pytest.fixture
def model_asset(tmp_path):
    # Numerically valid full layout, explicitly synthetic; never authentic SMPL-H.
    parents=audit.PARENTS22.tolist()
    for wrist in (20,21):
        for _ in range(5):
            first=len(parents);parents.extend([wrist,first,first+1])
    template=np.zeros((6890,3),np.float32);template[0]=[.13,-.24,.81]
    for joint in range(1,52):template[joint]=template[parents[joint]]+[.01*(joint%3+1),.013,.021]
    reg=np.zeros((52,6890),np.float32);reg[np.arange(52),np.arange(52)]=1
    shapedirs=np.zeros((6890,3,16),np.float32)
    for joint in range(52):shapedirs[joint]=np.arange(48,dtype=np.float32).reshape(3,16)*1e-6*(joint+1)
    tree=np.array([[2**32-1,*parents[1:]],list(range(52))],dtype=np.uint32)
    path=tmp_path/'owned-model.npz';np.savez(path,v_template=template,J_regressor=reg,shapedirs=shapedirs,kintree_table=tree)
    return path,audit.load_model(path,audit.sha_file(path))


def params(count=4):
    return {'smplh.root_orient':np.zeros((count,3),np.float32),'smplh.pose_body':np.zeros((count,63),np.float32),
            'smplh.trans':np.tile([.4,.8,1.2],(count,1)).astype(np.float32),
            'smplh.betas':np.tile(np.linspace(-.3,.4,16),(count,1)).astype(np.float32)}


def masks(count=4):
    arrays={}
    for component in ('position','rotation'):
        arrays[f'estimated.valid.body_{component}']=np.ones((count,22),bool)
        arrays[f'source.observed.body_{component}']=np.ones((count,22),bool)
        arrays[f'source.interpolated.body_{component}']=np.zeros((count,22),bool)
        arrays[f'estimated.completed.body_{component}']=np.zeros((count,22),bool)
        arrays[f'quality.{component}_confidence']=np.full((count,22),.4,np.float32)
    arrays['source.feet_world']=np.zeros((count,6),np.float32)
    arrays['source.feet_valid']=np.ones((count,2),bool)
    arrays['source.feet_interpolated']=np.zeros((count,2),bool)
    return arrays


def test_full52_fk_checks_true16beta_rootpivot_and_body_axis_angle(model_asset):
    path,model=model_asset;a=params();a['smplh.root_orient'][:,2]=np.pi/2
    fk=audit.numpy_fk(model,a)
    rest=model['rest']+np.einsum('jcs,s->jc',model['shape'],a['smplh.betas'][0])
    Q=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    expected=(rest-rest[0])@Q.T+rest[0]+a['smplh.trans'][0]
    np.testing.assert_allclose(fk['positions'][0],expected,atol=2e-7)
    np.testing.assert_allclose(fk['root_world'],rest[0]+a['smplh.trans'],atol=1e-10)
    assert not np.allclose(fk['root_world'],a['smplh.trans'])
    a['smplh.pose_body'][:,3*19+1]=.37
    hand=np.zeros((4,90));hand[:,2]=.4
    rotated=audit.numpy_fk(model,a,pose_hand_axis_angle=hand)
    assert not np.allclose(rotated['global_rotations'][:,20],fk['global_rotations'][:,20])
    assert not np.allclose(rotated['global_rotations'][:,22],rotated['global_rotations'][:,20])
    assert rotated['positions'].shape==(4,52,3)


def test_model_true16_layout_and_fingerprint_cannot_be_faked(model_asset,tmp_path):
    path,model=model_asset
    with pytest.raises(ValueError,match='SHA'):audit.load_model(path,'0'*64)
    with np.load(path,allow_pickle=False) as source:raw={k:source[k].copy() for k in source.files}
    raw['shapedirs']=np.pad(raw['shapedirs'],((0,0),(0,0),(0,4)))
    other=tmp_path/'padded20.npz';np.savez(other,**raw)
    with pytest.raises(ValueError,match='true16'):audit.load_model(other,audit.sha_file(other))


def test_row6d_uses_matrix_rows_and_floor_inverse(model_asset):
    _,model=model_asset;a=params();a['smplh.root_orient'][:]=[.2,-.3,.4]
    fk=audit.numpy_fk(model,a);R=fk['global_rotations'][:,:22]
    six=R[...,:2,:].reshape(4,22,6)
    np.testing.assert_allclose(audit.row6_matrix(six),R,atol=1e-12)
    floor=audit.exp_so3(np.tile([0,0,.8],(4,1)));origin=np.tile([.3,.7,0.],(4,1))
    local=np.einsum('tji,tkj->tki',floor,fk['positions'][:,:22]-origin[:,None])
    world=np.einsum('tij,tkj->tki',floor,local)+origin[:,None]
    np.testing.assert_allclose(world,fk['positions'][:,:22],atol=1e-12)


def test_hand45_coefficients_require_components_and_mean_not_direct_axis_angle(tmp_path):
    basis=tmp_path/'hand-basis.npz';left=np.eye(45)*1.3;right=np.eye(45)*.7;mean=np.linspace(-.1,.2,45)
    np.savez(basis,hands_componentsl=left,hands_componentsr=right,hands_meanl=mean,hands_meanr=-mean)
    coefficient=np.ones((4,90))*.05
    converted=audit.hand_pca_to_axis_angle(coefficient,basis,audit.sha_file(basis))
    np.testing.assert_allclose(converted[:,:45],coefficient[:,:45]@left+mean)
    np.testing.assert_allclose(converted[:,45:],coefficient[:,45:]@right-mean)
    assert not np.allclose(converted,coefficient)


def _write(path,arrays,row_group_size):
    columns={}
    for key,value in arrays.items():
        if value.ndim==1:columns[key]=pa.array(value)
        else:columns[key]=pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)),value.shape[1])
    pq.write_table(pa.table(columns),path,row_group_size=row_group_size)


def test_full_dino_digest_survives_new_row_groups_but_rejects_one_changed_token(tmp_path):
    source=tmp_path/'source.parquet';output=tmp_path/'owned-vision.parquet'
    arrays={'frame_index':np.arange(9,dtype=np.int64),'episode_index':np.zeros(9,np.int64),
            'observation.features.dino':np.arange(9*5120,dtype=np.float32).reshape(9,5120),'observation.valid.dino':np.ones(9,bool)}
    _write(source,arrays,3);_write(output,arrays,5)
    mapping={k:k for k in arrays}
    assert audit.compare_streaming(source,output,mapping)['all_rows_equal']
    changed=deepcopy(arrays);changed['observation.features.dino'][8,5119]+=1;_write(output,changed,2)
    with pytest.raises(ValueError,match='streaming'):audit.compare_streaming(source,output,mapping)


def test_position_and_rotation_completion_are_independent_and_never_observed():
    arrays=masks();arrays['source.observed.body_rotation'][1,7]=False;arrays['estimated.completed.body_rotation'][1,7]=True
    audit.validate_parameter_masks(arrays,4)
    arrays['estimated.completed.body_rotation'][1,7]=False
    with pytest.raises(ValueError,match='completion'):audit.validate_parameter_masks(arrays,4)


def test_unavailable_masks_and_native_feet_cannot_gain_nonzero_labels():
    arrays=masks();arrays['estimated.valid.body_rotation'][1,7]=False
    with pytest.raises(ValueError,match='confidence'):audit.validate_parameter_masks(arrays,4)
    arrays['quality.rotation_confidence'][1,7]=0;audit.validate_parameter_masks(arrays,4)
    arrays['source.feet_valid'][2,0]=False;arrays['source.feet_world'][2,:3]=1
    with pytest.raises(ValueError,match='foot'):audit.validate_parameter_masks(arrays,4)


def test_full_population_requires_identity_frames_splits_and_no_operator_leak():
    expected=[dict(episode_index=0,source='egosuite',source_key='a',length=99,split='train',operator_id='p'),
              dict(episode_index=1,source='egosuite',source_key='b',length=100,split='train',operator_id='p'),
              dict(episode_index=2,source='egoexo4d_hm',source_key='h',length=102,split='val',operator_id='p')]
    assert audit.population_records(expected,expected)['frames']==301
    for key,value in [('source_key','wrong'),('length',98),('split','test')]:
        changed=deepcopy(expected);changed[0][key]=value
        with pytest.raises(ValueError,match='population'):audit.population_records(changed,expected)
    with pytest.raises(ValueError,match='population'):audit.population_records(expected[:-1],expected)
    leak=deepcopy(expected);leak[1]['split']='val'
    with pytest.raises(ValueError,match='leaks'):audit.population_records(leak,leak)


def test_owned_paths_cannot_escape_to_old_dataset(tmp_path):
    root=tmp_path/'new';root.mkdir();(root/'owned').write_bytes(b'ok');(tmp_path/'old').write_bytes(b'old')
    assert audit.owned_path(root,'owned').name=='owned'
    with pytest.raises(ValueError,match='escapes'):audit.owned_path(root,'../old')
    with pytest.raises(ValueError,match='relative'):audit.owned_path(root,tmp_path/'old')


def _standalone(tmp_path,model_asset,*,invalid_ground=False,quarantine=False):
    asset,_=model_asset;root=tmp_path/'old-dataset'/'new-standalone';root.mkdir(parents=True)
    model=root/'weights/model.npz';model.parent.mkdir();shutil.copyfile(asset,model);n=11
    arrays={key:np.zeros(n if width==1 else (n,width),dtype=dtype) for key,(dtype,width) in audit.independent_specs().items()}
    arrays.update(params(n));arrays.update(masks(n))
    arrays['frame_index']=arrays['index']=np.arange(n,dtype=np.int64);arrays['timestamp']=(np.arange(n)/30).astype(np.float32)
    arrays['observation.source_timestamp']=np.arange(n)/30;arrays['source.video_frame_index']=np.arange(n,dtype=np.int64)
    for key in arrays:
        if key.startswith('observation.valid.'):arrays[key][:]=True
        if key.endswith('_world_rotation'):arrays[key][:,0]=1
    for modality in ('head','camera','head_reference'):arrays['observation.state.'+modality+'_world'][:]=[5,4,1.8]
    arrays['observation.state.head_reference_world_rotation'][:]=[np.cos(.3),0,0,np.sin(.3)]
    arrays['smplh.root_orient'][:,2]=np.linspace(0,.3,n);arrays['smplh.pose_body']=np.linspace(-.2,.2,n*63,dtype=np.float32).reshape(n,63)
    arrays['source.feet_world'][:,[2,5]]=.52;arrays['observation.environment.floor_height'][:]=.5;arrays['observation.environment.floor_normal'][:,2]=1
    arrays['observation.state.hand_left_world'][:]=np.arange(63)*.01;arrays['observation.state.hand_right_world'][:]=np.arange(63)*.01+.3
    arrays['observation.video_time_error'][:]=.001;arrays['observation.video_timestamp']=np.arange(n)/30+.001
    if invalid_ground:
        arrays['observation.valid.floor'][:]=False;arrays['source.feet_world'][2:,[2,5]]=10
    # Finite parameter/source p does not create observed rotation.
    arrays['source.observed.body_rotation'][:,7]=False;arrays['estimated.completed.body_rotation'][:,7]=True
    metadata={'coordinate_verified':True,'head_reference':{'basis_verified':True},
              'ground':{'height_m':.5,'valid':not invalid_ground,'verified':False,'mode':'estimate','window_fallback':{'mode':'mean_valid_feet','offset_m':.02}},
              'source_data':str(tmp_path/'old-dataset'/'removed-source.parquet')}
    if quarantine:metadata.update(training_eligible=False,quarantine_reasons=['unverified-image-join'])
    record=dict(episode_index=0,source='egosuite',source_key='egosuite:fixture',length=n,dataset_from_index=0,dataset_to_index=n,
                split='train',data_path='episodes/00000000/motion.parquet',vision_path='episodes/00000000/vision.parquet',metadata=metadata)
    ep=root/'episodes/00000000';ep.mkdir(parents=True);_write(ep/'motion.parquet',arrays,3)
    vision={key:arrays[key] for key in ('frame_index','episode_index','timestamp','observation.source_timestamp','source.video_frame_index','observation.valid.dino','observation.video_timestamp','observation.video_time_error')}
    vision['timestamp']=vision['timestamp'].astype(np.float64);vision['observation.video_time_error']=vision['observation.video_time_error'].astype(np.float64)
    vision['observation.features.dino']=np.arange(n*5120,dtype=np.float32).reshape(n,5120);_write(ep/'vision.parquet',vision,4)
    (root/'meta').mkdir();(root/'meta/info.json').write_text(json.dumps({'schema_version':audit.PARAMETER_SCHEMA_VERSION,'fps':30,'model':{'path':'weights/model.npz','model_sha256':audit.sha_file(model)}}))
    (root/'.pipeline/episodes').mkdir(parents=True);(root/'.pipeline/episodes/00000000.json').write_text(json.dumps(record))
    return root,arrays,record


def _probe(root,output):
    return subprocess.run([sys.executable,str(Path(audit.__file__).resolve()),'--reader-probe','--root',str(root),
          '--output',str(output),'--blocked-roots',json.dumps([str(root.parent)]),'--allow-test-fixture'],capture_output=True,text=True)


def test_fresh_process_reader_all_interfaces_without_old_files(tmp_path,model_asset):
    root,arrays,record=_standalone(tmp_path,model_asset,invalid_ground=True)
    output=tmp_path/'probe.json';result=_probe(root,output)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text());assert report['passed'] and report['episodes']==1 and report['probes']==4
    assert report['old_source_open_attempts']==[]
    assert report['all_population_interfaces_checked'] and not report['motion_truth_accuracy_established']


def test_fresh_process_pyarrow_old_vision_path_is_denied(tmp_path,model_asset):
    root,_,record=_standalone(tmp_path,model_asset);old=root.parent/'old-vision.parquet';shutil.copyfile(root/record['vision_path'],old)
    record['vision_path']='../old-vision.parquet';(root/'.pipeline/episodes/00000000.json').write_text(json.dumps(record))
    result=_probe(root,tmp_path/'failed-probe.json')
    assert result.returncode!=0 and ('blocked' in result.stderr or 'owned relative' in result.stderr)


def test_fresh_process_quarantine_keeps_inspection_but_rejects_canonical(tmp_path,model_asset):
    root,_,_=_standalone(tmp_path,model_asset,quarantine=True);output=tmp_path/'quarantine-probe.json';result=_probe(root,output)
    assert result.returncode==0,result.stderr
    report=json.loads(output.read_text());assert report['passed'] and not report['rows'][0]['canonical_available']


def test_independent_physical_schema_rejects_stored_dense22_or_frame_gap(tmp_path,model_asset):
    root,a,r=_standalone(tmp_path,model_asset);audit.validate_physical(a,r)
    changed=deepcopy(a);changed['observation.state.body_world']=np.zeros((11,66),np.float32)
    with pytest.raises(ValueError,match='Physical'):audit.validate_physical(changed,r)
    changed=deepcopy(a);changed['frame_index'][8]+=1
    with pytest.raises(ValueError,match='frame grid'):audit.validate_physical(changed,r)


def test_independent_density_clusters_duplicate_support_and_border_not_bridge():
    groups=audit.density_height_clusters([0,0,0,.005,.014,.018,.018,.018],eps=.005,min_samples=3)
    assert len(groups)==2
    np.testing.assert_array_equal(groups[0],[0,0,0,.005]);np.testing.assert_array_equal(groups[1],[.014,.018,.018,.018])
    assert audit.density_height_clusters([0,.01,.02],eps=.005,min_samples=3)==[]


def test_selected_dino_full_join_uses_explicit_indices_and_exact_float_upcast(tmp_path):
    source=tmp_path/'long-source.parquet';a={'frame_index':np.arange(10,dtype=np.int64),'observation.video_time_error':np.arange(10,dtype=np.float32)*.001,
                                         'observation.features.dino':np.arange(10*5120,dtype=np.float32).reshape(10,5120)}
    _write(source,a,3);selected=np.array([0,2,5,9]);d=audit.selected_stream_digest(source,list(a),selected,casts={'observation.video_time_error':np.float64})
    assert d['observation.video_time_error']['dtype']==np.dtype(np.float64).str and d['frame_index']['rows']==4
    import hashlib
    assert d['observation.features.dino']['row_bytes_sha256']==hashlib.sha256(a['observation.features.dino'][selected].tobytes()).hexdigest()
    with pytest.raises(ValueError,match='join'):audit.selected_stream_digest(source,list(a),np.array([-1,3]))


def test_dense_cleanup_dry_run_never_includes_raw_weights_or_new_archive(tmp_path):
    old=tmp_path/'old';data=old/'data';data.mkdir(parents=True);s=data/'file.parquet';s.write_bytes(b'source')
    fitted=old/'derived/fit';fitted.mkdir(parents=True);f=(fitted/'episodes/0/motion.parquet');f.parent.mkdir(parents=True);f.write_bytes(b'fit')
    root=old/'new';root.mkdir();weights=old/'weights/model.npz';weights.parent.mkdir();weights.write_bytes(b'model');hm=tmp_path/'hm';hm.mkdir()
    plan=dict(egosuite_archive=str(old),egosuite_fitted_root=str(fitted),model_path=str(weights),hm_root=str(hm))
    row=dict(source='egosuite',source_key='s',source_path=str(s),fitted_path=str(f));report=audit.cleanup_dry_run(plan,[row],root)
    assert report['exact_dense_file_count']==2 and report['deletion_executed'] is False and s.exists() and f.exists()
    for path in (old,root,weights,hm):
        with pytest.raises(ValueError,match='scope'):audit.cleanup_dry_run(plan,[row],root,{'paths':[str(path)]})


def test_old_full_fit_gate_rejects_expanded_stale_and_incomplete_audits(tmp_path):
    root=tmp_path/'fitting';(root/'reports').mkdir(parents=True);sidecars=tmp_path/'sidecars';sidecars.mkdir()
    (root/'status.json').write_text(json.dumps({'job_signature':'frozen'}))
    ap=root/'reports/full-sidecar-audit.json';gp=root/'reports/full-acceptance.json'
    report={'passed':True,'episodes_checked':6614,'frames_checked':26841893,'selection_sha256':None,'sidecar_root':str(sidecars),
            'results':[{'episode_index':i,'passed':True} for i in range(6614)]};ap.write_text(json.dumps(report))
    gate={'passed':True,'scope':'full','episodes':6614,'frames':26841893,'full_candidate_quality_passed':True,'failures':[],
          'job_signature':'frozen','independent_audit_sha256':audit.sha_file(ap)};gp.write_text(json.dumps(gate))
    plan=dict(egosuite_fitting_job=str(root),egosuite_fitted_root=str(sidecars));assert audit.check_original_full_fit_gate(plan)[0]
    for key,value in [('scope','expanded'),('episodes',60),('frames',154473),('full_candidate_quality_passed',False),('job_signature','stale'),('independent_audit_sha256','0'*64)]:
        changed=deepcopy(gate);changed[key]=value;gp.write_text(json.dumps(changed));assert not audit.check_original_full_fit_gate(plan)[0]
    gp.write_text(json.dumps(gate));report['results']=report['results'][:-1];ap.write_text(json.dumps(report))
    gate['independent_audit_sha256']=audit.sha_file(ap);gp.write_text(json.dumps(gate));assert not audit.check_original_full_fit_gate(plan)[0]


def test_native_hm_invalid_chunk_finite_hand_values_remain_stored_without_availability(tmp_path,model_asset,monkeypatch):
    _,model=model_asset;n=4;valid=np.array([1,0,1,1],np.int8);a=params(n)
    raw={'trans':a['smplh.trans'][None],'root_orient':a['smplh.root_orient'][None],'pose_body':a['smplh.pose_body'][None],
         'betas_per_frame':a['smplh.betas'][None],'hand_pose':np.full((1,n,90),.07,np.float32),'valid':valid[None],
         'joints3d':np.arange(n*67*3,dtype=np.float32).reshape(1,n,67,3)*.0001,'chunk_ranges':np.array([[0,n]]),
         'cam_R':np.tile(np.eye(3,dtype=np.float32),(5,n,1,1)),'cam_t':np.zeros((5,n,3),np.float32)}
    path=tmp_path/'native.npz';np.savez(path,**raw);decoded=audit.load_hm_raw(path);pv,rv=audit.hm_masks(decoded);fk=audit.numpy_fk(model,a)
    a.update({'timestamp':(np.arange(n)/30).astype(np.float32),'smplh.pose_hand_pca':raw['hand_pose'][0],
              'smplh.valid.hand_pose':np.tile(valid[:,None].astype(bool),(1,2)),
              'source.feet_world':np.where(pv[:,[10,11],None],fk['positions'][:,[10,11]],0).reshape(n,6).astype(np.float32),
              'quality.observation_residual_m':np.zeros((n,22),np.float32),'quality.observation_residual_valid':np.zeros((n,22),bool)})
    for component,mask in [('position',pv),('rotation',rv)]:
        for prefix in ('source.observed','estimated.valid'):a[prefix+'.body_'+component]=mask
        for prefix in ('source.interpolated','estimated.completed'):a[prefix+'.body_'+component]=np.zeros((n,22),bool)
        a['quality.'+component+'_confidence']=mask.astype(np.float32)
    for side,s,e in [('left',25,46),('right',46,67)]:
        a['observation.state.hand_'+side+'_world']=raw['joints3d'][0,:,s:e].reshape(n,63).copy()
        a['observation.valid.hand_'+side+'_position']=np.tile(valid[:,None].astype(bool),(1,21))
    basis=tmp_path/'basis.npz';np.savez(basis,hands_componentsl=np.eye(45),hands_componentsr=np.eye(45),hands_meanl=np.zeros(45),hands_meanr=np.zeros(45))
    rec={'source_provenance':{'source_path':str(path),'source_identity':audit.file_identity(path),'source_npz_sha256':audit.sha_file(path)},'metadata':{'training_eligible':True}}
    monkeypatch.setattr(audit,'audit_hm_alignment',lambda *args:True);monkeypatch.setattr(audit,'audit_hm_camera',lambda *args:{'verified':True});monkeypatch.setattr(audit,'audit_ground',lambda *args:None)
    assert audit.audit_hm(a,rec,{'source_path':str(path)},model,{'path':str(basis),'sha256':audit.sha_file(basis)})['source_rows_preserved']==n
    assert a['observation.state.hand_left_world'][1].any() and not a['observation.valid.hand_left_position'][1].any()
    changed=deepcopy(a);changed['observation.state.hand_left_world'][1]=0
    with pytest.raises(ValueError,match='finite original'):audit.audit_hm(changed,rec,{'source_path':str(path)},model,{'path':str(basis),'sha256':audit.sha_file(basis)})


def test_runtime_hm_join_allows_only_measured_full_timesync_extension():
    original={'verified':True,'hm_rows':4,'video_rows':5,'hm_row_to_video_frame':{'offset':0,'step':1,'length':4}}
    runtime={**deepcopy(original),'actual_timesync_rows':5};audit.verify_runtime_join(runtime,original,5)
    for extra in (4,6,False,'5'):
        changed=deepcopy(runtime);changed['actual_timesync_rows']=extra
        with pytest.raises(ValueError,match='actual_timesync_rows'):audit.verify_runtime_join(changed,original,5)
    changed=deepcopy(runtime);changed['unknown_trusted_alignment']=True
    with pytest.raises(ValueError,match='Unknown'):audit.verify_runtime_join(changed,original,5)
    changed=deepcopy(runtime);changed['hm_row_to_video_frame']['offset']=1
    with pytest.raises(ValueError,match='source contract'):audit.verify_runtime_join(changed,original,5)
