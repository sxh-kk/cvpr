#!/usr/bin/env python3
"""Independent immutable sidecar audit: NumPy SMPL-H FK and fresh source joins.

This checker deliberately does not import the fitter, BodyModel, geometry or
sidecar validator. Passing certifies decoding/preservation and declared numeric
consistency. It does not certify motion truth, source twist, or model centers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for _thread_setting in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_thread_setting,'1')

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARENTS = np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
JOIN = {'timestamp','frame_index','episode_index','index','observation.source_timestamp',
        'observation.source_frame_left','observation.source_frame_right',
        'observation.source_interpolation_weight'}

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): h.update(block)
    return h.hexdigest()

def numeric_sha(arrays):
    h=hashlib.sha256()
    for name in sorted(arrays):
        x=np.ascontiguousarray(arrays[name])
        h.update(canonical([name,x.dtype.str,list(x.shape)]))
        h.update(x.tobytes())
    return h.hexdigest()

def job_digest(value):
    raw=(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    return hashlib.sha256(raw).hexdigest()

def job_motion_sha(arrays):
    h=hashlib.sha256()
    for name,x in sorted(arrays.items()):
        x=np.ascontiguousarray(x)
        h.update(name.encode());h.update(str(x.dtype).encode())
        h.update(json.dumps(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()

def audit_train_prior(job):
    """Freshly re-read train evidence; verify equal-group pooling without fitter."""
    failures=[]
    def check(value,message):
        if not bool(value):failures.append(message)
    plan=json.loads((job/'plan.json').read_text())
    unsigned={k:v for k,v in plan.items() if k!='job_signature'}
    check(job_digest(unsigned)==plan['job_signature'],'Frozen job signature mismatch')
    for path,sha in plan['implementation_sha256'].items():
        check(file_sha(path)==sha,'Frozen producer implementation changed: '+path)
    check(file_sha(plan['model_path'])==plan['model_sha256'],'Frozen job model changed')
    splits=json.loads(Path(plan['split_manifest_path']).read_text())
    check(file_sha(plan['split_manifest_path'])==plan['split_manifest_sha256'],'Frozen split manifest changed')
    membership={k:s for s,keys in splits['splits'].items() for k in keys}
    selected=plan['selection'];cal=selected['calibration_records'];val=selected['validation_records']
    check(job_digest(selected)==plan['selection_sha256'],'Frozen plan selection SHA mismatch')
    cal_ids={r['operator_id'] for r in cal};val_ids={r['operator_id'] for r in val}
    check(len(cal_ids)==len(cal) and not cal_ids&val_ids,'Calibration/evaluation operators duplicate or overlap')
    check(all(membership[r['source_key']]=='train' and r['split']=='train' for r in cal),'Shared calibration includes validation/test source')
    shared=json.loads((job/'shared-center-prior.json').read_text());prior=shared['prior']
    check(shared['job_signature']==plan['job_signature'],'Shared prior belongs to different job')
    group_ids=prior['metadata']['group_ids']
    check(set(group_ids)==cal_ids and len(group_ids)==len(cal_ids),'Shared prior groups differ from frozen train calibration')
    check(set(shared['train_group_ids'])==cal_ids,'Shared prior train list mismatch')
    centers=[];evidence=[]
    by_op={r['operator_id']:r for r in cal}
    for op in group_ids:
        envelope=json.loads((job/'calibration/source-prior'/f'{op}.json').read_text())
        row=by_op[op];source=envelope['source'];group=envelope['calibration'];center=group['center']
        check(envelope['phase']=='source_prior' and envelope['split']=='train','Train prior evidence has wrong phase/split')
        check(group['group_id']==op and envelope['group_id']==op and group['source_id']=='egosuite','Train prior group/source mismatch')
        check(source['episode_index']==row['episode_index'] and source['source_key']==row['source_key'],'Train calibration actual source identity mismatch')
        check(membership[source['source_key']]=='train','Actual calibration source belongs to heldout split')
        check(file_sha(source['record_path'])==source['record_sha256'],'Train calibration source record bytes changed')
        record=json.loads(Path(source['record_path']).read_text())
        check(record['metadata']['operator_id']==op,'Actual source metadata operator mismatch')
        path=Path(source['source_path']);stat=path.stat()
        sid={'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'ctime_ns':stat.st_ctime_ns,'inode':stat.st_ino,'device':stat.st_dev}
        check(source['source_identity']==sid,'Train calibration Parquet stat changed')
        arrays,unused=read_table(path,source['source_projection_columns'])
        check(job_motion_sha(arrays)==source['source_motion_sha256'],'Train calibration fresh motion SHA mismatch')
        check(not any(k=='observation.features.dino' for k in arrays),'Train calibration decoded DINO')
        check(center['model_sha256']==plan['model_sha256'] and len(center['betas'])==16,'Train calibration model/true16beta mismatch')
        expected_signature=job_digest({'job':plan['job_signature'],'phase':'source_prior','group_id':op,'source':source,'shared_sha256':None})
        check(envelope['signature']==expected_signature and envelope.get('shared_prior_sha256') is None,'Train bootstrap calibration provenance signature mismatch')
        centers.append(center)
        evidence.append({'operator_id':op,'episode_index':source['episode_index'],'frames':source['frames'],'source_motion_sha256':source['source_motion_sha256'],'calibration_input_sha256':center['input_sha256']})
    enabled=np.array([c['offset_valid'] for c in centers],bool)
    expected_valid=enabled.sum(0)>=2
    expected=np.zeros((22,3))
    anchors=np.array(centers[0]['anchors'])
    for center in centers:check(np.array_equal(center['anchors'],anchors),'Train groups use different model anchor axes')
    for j in np.flatnonzero(expected_valid):
        expected[j]=np.array([c['offsets'][j] for c in centers],float)[enabled[:,j]].mean(0)
    actual=np.array(prior['offsets'],float)
    error=float(np.max(np.abs(actual-expected)))
    check(error<1e-12,'Shared prior is not equal train-group masked mean')
    check(np.array_equal(prior['offset_valid'],expected_valid),'Shared prior joint support validity mismatch')
    check(np.array_equal(prior['anchors'],anchors),'Shared prior target anchors differ from train groups')
    check(prior['model_sha256']==plan['model_sha256'],'Shared prior model asset mismatch')
    check(prior['metadata'].get('native_axis_certified') is False,'Shared prior falsely certifies native axes')
    check(prior['metadata'].get('independent_center_accuracy_established') is False,'Shared prior falsely certifies source center truth')
    return {'passed':not failures,'failures':failures,'shared_prior_input_sha256':prior['input_sha256'],'calibration_groups_checked':len(centers),'calibration_frames_checked':sum(r['frames'] for r in evidence),'equal_group_mean_max_abs_error_m':error,'train_only_evidence':evidence,'heldout_operator_overlap':sorted(cal_ids&val_ids)}

def identity(path):
    p=Path(path).resolve(strict=True);s=p.stat()
    return {'path':str(p),'device':s.st_dev,'inode':s.st_ino,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def read_table(path,columns=None):
    t=pq.read_table(path,columns=columns)
    arrays={}
    for name in t.column_names:
        c=t[name].combine_chunks();typ=c.type
        if pa.types.is_fixed_size_list(typ):
            arrays[name]=c.values.to_numpy(zero_copy_only=False).reshape(len(c),typ.list_size)
        elif pa.types.is_list(typ):
            arrays[name]=np.asarray(c.to_pylist(),dtype=typ.value_type.to_pandas_dtype())
        else:
            arrays[name]=c.to_numpy(zero_copy_only=False)
    return arrays,t.schema.metadata or {}

def exp_so3(v):
    v=np.asarray(v,dtype=np.float64)
    theta=np.linalg.norm(v,axis=-1)
    K=np.zeros((*v.shape[:-1],3,3),float)
    K[...,0,1]=-v[...,2];K[...,0,2]=v[...,1]
    K[...,1,0]=v[...,2];K[...,1,2]=-v[...,0]
    K[...,2,0]=-v[...,1];K[...,2,1]=v[...,0]
    A=np.sinc(theta/np.pi)
    B=.5*np.sinc(theta/(2*np.pi))**2
    return np.eye(3)+A[...,None,None]*K+B[...,None,None]*(K@K)

def quat_matrix(q):
    q=np.asarray(q,dtype=float)
    q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    w,x,y,z=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)

def row6_matrix(x):
    x=np.asarray(x,dtype=float)
    a=x[...,:3];b=x[...,3:6]
    a=a/np.linalg.norm(a,axis=-1,keepdims=True)
    b=b-(a*b).sum(-1,keepdims=True)*a
    b=b/np.linalg.norm(b,axis=-1,keepdims=True)
    return np.stack([a,b,np.cross(a,b)],axis=-2)

def model_layout(path,sha):
    if file_sha(path)!=sha: raise ValueError('Model SHA differs from frozen contract')
    with np.load(path,allow_pickle=False) as a:
        template=np.asarray(a['v_template'],float)
        reg=np.asarray(a['J_regressor'],float)
        dirs=np.asarray(a['shapedirs'],float)
        tree=np.asarray(a['kintree_table'])
    if template.shape!=(6890,3) or reg.shape!=(52,6890) or tree.shape!=(2,52) or dirs.shape[:2]!=(6890,3):
        raise ValueError('Expected true 6890-vertex/52-joint SMPL-H')
    ids={int(v):i for i,v in enumerate(tree[1])}
    parents=np.array([-1]+[ids[int(v)] for v in tree[0,1:]])
    if not np.array_equal(parents[:22],PARENTS):raise ValueError('Unexpected SMPL-H body22 parent tree')
    expected=PARENTS.tolist()
    for wrist in [20,21]:
        for finger in range(5):
            first=len(expected);expected.extend([wrist,first,first+1])
    if not np.array_equal(parents,expected):raise ValueError('Unexpected full52 SMPL-H hand tree')
    if np.any(parents[1:]>=np.arange(1,52)):raise ValueError('Non-topological model tree')
    return reg@template,np.einsum('jv,vcs->jcs',reg,dirs),parents

def numpy_fk(rest,shape,parents,beta,aa,trans):
    J=rest+np.einsum('jcs,s->jc',shape,beta)
    T=len(aa);local=np.broadcast_to(np.eye(3),(T,52,3,3)).copy()
    local[:,:22]=exp_so3(aa)
    R=np.empty_like(local);p=np.empty((T,52,3))
    R[:,0]=local[:,0];p[:,0]=J[0]+trans
    for j in range(1,52):
        parent=parents[j]
        R[:,j]=R[:,parent]@local[:,j]
        p[:,j]=p[:,parent]+np.einsum('tij,j->ti',R[:,parent],J[j]-J[parent])
    return p[:,:22],R[:,:22],local[:,:22],J[0]

def summary(x):
    x=np.asarray(x,float)
    return {'count':int(x.size),'mean':float(x.mean()) if x.size else None,'p95':float(np.percentile(x,95)) if x.size else None,'max':float(x.max()) if x.size else None}

def find_center(mapping):
    if isinstance(mapping,dict):
        if {'offsets','anchors','offset_valid'}.issubset(mapping):return mapping
        for key in ['center_calibration','center','calibration']:
            if key in mapping:
                found=find_center(mapping[key])
                if found:return found
    return None

def audit_episode(folder,model_cache,selection=None,job=None):
    receipt=json.loads((folder/'receipt.json').read_text())
    declaration=receipt['declaration'];contract=declaration['contract']
    failures=[];metrics={}
    def check(condition,message):
        if not bool(condition):failures.append(message)
    def close(actual,expected,mask=None,tol=2e-5,name='numeric'):
        a=np.asarray(actual);b=np.asarray(expected)
        if mask is not None:a=a[mask];b=b[mask]
        error=float(np.max(np.abs(a-b))) if a.size else 0.
        metrics[name+'_max_abs_error']=error
        check(error<=tol,name+' disagrees with independent expected value')
    path=folder/'motion.parquet'
    check(receipt.get('state')=='complete','Receipt is not complete')
    check(receipt['contract_sha256']==digest(declaration),'Declaration SHA mismatch')
    check(receipt['parquet_sha256']==file_sha(path),'Parquet byte SHA mismatch')
    arrays,meta=read_table(path)
    check(receipt['numeric_sha256']==numeric_sha(arrays),'Parquet numeric SHA mismatch')
    check(receipt.get('execution_sha256')==digest(receipt.get('execution')),'Sequence execution metadata SHA mismatch')
    check(meta.get(b'contract_sha256',b'').decode()==receipt['contract_sha256'],'Parquet schema receipt SHA mismatch')
    check(contract.get('label_kind')=='model_fitted_estimate','Wrong estimated label declaration')
    check(contract.get('native_axis_certified') is False,'Native source axes falsely certified')
    check(declaration.get('source_axes_certified') is False,'Receipt source axes falsely certified')
    check(contract.get('target_model_axes_defined') is True,'Target model axes undeclared')
    check(contract.get('archive_fps')==30,'Archive FPS mismatch')
    check(contract['config_sha256']==digest(contract['config']),'Configuration SHA mismatch')
    check(contract['calibration_sha256']==digest(contract['calibration']),'Calibration SHA mismatch')
    check(contract.get('contact_policy')=='not_stored; recompute_from_fitted_feet_after_15Hz_selection','Contact policy mismatch')
    check('original_native_feet' in contract.get('ground_policy',''),'Ground fallback original-feet policy missing')
    check(not any('dino' in k.lower() or 'contact' in k.lower() for k in arrays),'Sidecar unexpectedly copied DINO or stored contacts')
    for key in ['source_identity','model_identity']:
        check(identity(contract[key]['path'])==contract[key],key+' changed stat/inode identity')
    source_path=Path(contract['source_identity']['path'])
    archive_root=Path(contract['dataset_root']).resolve()
    check(source_path.resolve().is_relative_to(archive_root/'data'),'Source Parquet outside declared archive')
    implementation=Path(__file__).resolve().parents[1]/'src/egomotion_pipeline/smplh_sidecar.py'
    check(file_sha(implementation)==contract['sidecar_implementation_sha256'],'Frozen writer implementation differs from present source')
    for filename,sha in contract.get('fitter_implementation_sha256',{}).items():
        check(file_sha(implementation.parent/filename)==sha,'Frozen fitter/decoder implementation changed: '+filename)
    execution=receipt.get('execution',{});execution_metadata=execution.get('metadata',{})
    if execution_metadata:
        check(execution_metadata.get('model',{}).get('model_sha256')==contract['model_sha256'],'Actual sequence model SHA mismatch')
        check(execution_metadata.get('config')==contract['config'],'Actual sequence config differs from frozen sidecar contract')
        check(execution_metadata.get('calibration_input_sha256')==contract['calibration']['center']['input_sha256'],'Actual sequence calibration fingerprint mismatch')
        check(execution_metadata.get('group_id')==contract['calibration']['group_id'] and execution_metadata.get('source_id')==contract['calibration']['source_id'],'Actual sequence source/group mismatches calibration')
        check(execution_metadata.get('native_axis_certified') is False and execution_metadata.get('absolute_rotation_accuracy_established') is False,'Actual execution falsely certifies source twist/orientation accuracy')
        check(execution_metadata.get('betas_frozen') is True and execution_metadata.get('center_offsets_frozen') is True,'Sequence changes fitted shape/center offsets')
        check(execution_metadata.get('camera_trajectory_modified') is False and execution_metadata.get('source_inputs_modified') is False,'Sequence altered source observations')
    elif contract.get('config',{}).get('fixture') is not True:
        check(False,'Production sequence execution metadata missing')
    source,unused=read_table(source_path,columns=contract['source_projection_columns'])
    source={k:v.reshape(-1) if v.ndim==2 and v.shape[1]==1 else v for k,v in source.items()}
    check(contract['source_motion_sha256']==numeric_sha(source),'Fresh original source motion SHA mismatch')
    for key,x in source.items():
        stored=key if key in JOIN else 'source.'+key
        check(stored in arrays and x.dtype==arrays[stored].dtype and np.array_equal(arrays[stored],x),'Original source data/mask/dtype changed: '+key)
    idx=int(declaration['episode_index']);T=int(declaration['frames'])
    check(len(arrays['frame_index'])==T,'Frame count mismatch')
    check(np.array_equal(arrays['frame_index'],np.arange(T)),'Frame join noncontiguous')
    check(np.all(arrays['episode_index']==idx),'Episode frame join mismatch')
    check(np.array_equal(arrays['timestamp'],(np.arange(T)/30).astype(np.float32)),'Archive exact30Hz grid mismatch')
    source_record_path=Path(contract['dataset_root'])/'.pipeline/episodes'/f'{idx:08d}.json'
    source_record=json.loads(source_record_path.read_text())
    check(digest(source_record)==declaration['source_record_sha256'],'Source record semantic SHA changed')
    check(source_record['source_key']==declaration['source_key'] and source_record['length']==T,'Source key/length metadata join mismatch')
    check(source_record.get('metadata',{})==declaration.get('source_metadata',{}),'Original source metadata changed')
    operator=source_record['metadata']['operator_id']
    group_id=contract['calibration'].get('group_id')
    qualified=operator+':record:'+declaration['source_key']
    check(group_id in {operator,qualified},'Fitted skeleton group not bound to source operator/recording')
    check(contract['calibration'].get('source_id')=='egosuite','Wrong group source namespace')
    if job:
        calibration_path=(job/'calibration/recording-specific'/f'{idx:08d}.json' if group_id==qualified else job/'calibration/groups'/f'{operator}.json')
        envelope=json.loads(calibration_path.read_text())
        check(envelope['calibration']==contract['calibration'],'Saved derivative calibration differs from frozen group/record file')
        check(envelope['operator_id']==operator and envelope['group_id']==group_id,'Group calibration routed to different source operator/recording')
        prior=json.loads((job/'shared-center-prior.json').read_text())
        check(envelope['shared_prior_sha256']==job_digest(prior['prior']),'Group calibration used different shared source prior')
        check(contract['calibration']['metadata']['source_prior_input_sha256']==prior['prior']['input_sha256'],'Per-group shared train prior identity mismatch')
        check(envelope['source']['source_archive']==contract['dataset_root'],'Group calibration belongs to different source archive')
        calibration_record=json.loads(Path(envelope['source']['record_path']).read_text())
        check(calibration_record['metadata']['operator_id']==operator,'Group calibration source metadata operator mismatch')
        check(file_sha(envelope['source']['record_path'])==envelope['source']['record_sha256'],'Frozen group calibration source record changed')
        expected_signature=job_digest({'job':prior['job_signature'],'phase':envelope['phase'],'group_id':group_id,'source':envelope['source'],'shared_sha256':envelope['shared_prior_sha256']})
        check(envelope['signature']==expected_signature,'Group calibration source/phase/group/shared signature mismatch')
    info_path=archive_root/'meta/info.json'
    if info_path.exists():
        info=json.loads(info_path.read_text());chunk_size=int(info['chunks_size'])
        expected_path=archive_root/info['data_path'].format(chunk_index=idx//chunk_size,file_index=idx%chunk_size,episode_index=idx)
        check(source_path.resolve()==expected_path.resolve(),'Source path does not match physical episode archive index')
    for key,x in arrays.items():check(np.isfinite(x).all(),'Nonfinite values in '+key)
    beta=arrays['fitted.betas'];aa=arrays['fitted.local_axis_angle'].reshape(T,22,3)
    check(beta.shape==(T,contract['shape_components']),'Shape component count mismatch')
    check(np.array_equal(beta,np.broadcast_to(beta[0],beta.shape)),'Body shape changes inside a skeleton group')
    key=(contract['model_identity']['path'],contract['model_sha256'])
    if key not in model_cache:model_cache[key]=model_layout(*key)
    rest,shapedirs,parents=model_cache[key]
    check(shapedirs.shape[-1]==beta.shape[-1],'Implicit model beta padding/truncation')
    expected_p,expected_R,local_R,J0=numpy_fk(rest,shapedirs,parents,beta[0],aa,arrays['fitted.smpl_trans'])
    pv=arrays['estimated.valid.body_world_position'];rv=arrays['estimated.valid.body_world_rotation']
    fpv=arrays['estimated.valid.body_floor_position'];frv=arrays['estimated.valid.body_floor_rotation']
    observed_p=arrays['source.observed.body_position'];observed_r=arrays['source.observed.body_rotation']
    for name,mask in [('position',pv),('rotation',rv),('floor_position',fpv),('floor_rotation',frv),('observed_position',observed_p),('observed_rotation',observed_r)]:
        check(mask.shape==(T,22) and mask.dtype==bool,'Bad22joint boolean mask: '+name)
    close(arrays['fitted.root_world'],expected_p[:,0],name='root_pivot')
    close(arrays['fitted.root_world'],arrays['fitted.smpl_trans']+J0,name='trans_plus_J0')
    close(arrays['fitted.local_rotation'].reshape(T,22,3,3),local_R,name='local_axis_angle_matrix')
    close(arrays['fitted.body_world_rotation'].reshape(T,22,4)[~rv],np.broadcast_to([1,0,0,0],(int((~rv).sum()),4)),tol=0,name='invalid_world_quaternion_sentinel')
    saved_R=quat_matrix(arrays['fitted.body_world_rotation'].reshape(T,22,4))
    close(saved_R,expected_R,rv,name='world_quaternion_FK')
    floor_q=source['observation.state.head_reference_floor_world_rotation'].reshape(T,4)
    floor_R=quat_matrix(floor_q);origin=source['observation.state.head_reference_floor_world'].reshape(T,3)
    floor_p=source['observation.valid.head_reference_floor_position'].reshape(T)
    floor_r=source['observation.valid.head_reference_floor_rotation'].reshape(T)
    check(np.array_equal(fpv,pv&floor_p[:,None]&floor_r[:,None]),'Floor estimated position validity differs from world/reference intersection')
    check(np.array_equal(frv,rv&floor_r[:,None]),'Floor estimated rotation validity differs from world/reference intersection')
    for frame,m_p,m_r in [('world',pv,rv),('floor',fpv,frv)]:
        payload=arrays[f'fitted.body_{frame}_rot6d_xyz'].reshape(T,22,9)
        xyz=payload[...,6:];r6=payload[...,:6]
        ep=expected_p if frame=='world' else np.einsum('tji,tkj->tki',floor_R,expected_p-origin[:,None])
        eR=expected_R if frame=='world' else np.swapaxes(floor_R,-1,-2)[:,None]@expected_R
        close(xyz,ep,m_p,name=frame+'_FK_position')
        close(r6,eR[...,:2,:].reshape(T,22,6),m_r,name=frame+'_row6D')
        check(np.all(xyz[~m_p]==0) and np.all(r6[~m_r]==0),frame+' invalid position/rotation blocks not zero')
        if m_r.any():
            decoded=row6_matrix(r6[m_r])
            close(decoded,eR[m_r],name=frame+'_row6D_SO3')
            close(np.linalg.det(decoded),np.ones(len(decoded)),tol=1e-10,name=frame+'_row6D_det')
        if frame=='floor':
            inverse=np.einsum('tij,tkj->tki',floor_R,xyz)+origin[:,None]
            close(inverse,expected_p,m_p,name='floor_inverse_world')
    check(np.array_equal(observed_p,source['observation.valid.body_position']),'Observed position mask substituted for estimated mask')
    check(np.array_equal(observed_r,source['observation.valid.body_rotation']),'Observed rotation mask substituted for estimated mask')
    inferred=arrays['estimated.inferred.body_position']
    check(inferred.shape==(T,22) and inferred.dtype==bool and not np.any(inferred&~pv),'Inferred completion flag outside estimated validity')
    check(np.array_equal(inferred,pv&~observed_p),'Inferred source-body completion mask mismatch')
    for part,valid in [('position',pv),('rotation',rv)]:
        weight=arrays[f'quality.{part}_confidence']
        check(weight.shape==(T,22) and np.all((weight>=0)&(weight<=1)) and np.all(weight[~valid]==0),'Invalid '+part+' quality proxy')
    residual_valid=arrays['quality.observation_residual_valid'];residual=arrays['quality.observation_residual_m']
    check(np.array_equal(residual_valid,observed_p&pv),'Observation residual validity mismatch')
    check(np.all(residual>=0) and np.all(residual[~residual_valid]==0),'Invalid residual value/sentinel')
    center=find_center(contract['calibration'])
    if center:
        d=np.asarray(center['offsets'],float);anchor=np.asarray(center['anchors'],int)
        check(d.shape==(22,3) and anchor.shape==(22,),'Bad frozen center calibration layout')
        reconstruction=expected_p+np.einsum('tjik,jk->tji',expected_R[:,anchor],d)
        close(arrays['fitted.reconstructed_source_landmarks_world'].reshape(T,22,3),reconstruction,name='stored_reconstructed_source_landmarks')
        e_residual=np.linalg.norm(reconstruction-source['observation.state.body_world'].reshape(T,22,3),axis=-1)
        close(residual,e_residual,residual_valid,name='frozen_offset_observation_residual')
        if center.get('model_sha256'):check(center['model_sha256']==contract['model_sha256'],'Center prior model SHA mismatch')
        if 'betas' in center:close(beta[0],np.asarray(center['betas'],float),name='frozen_group_beta')
    else:
        check(False,'Frozen center calibration unavailable for independent observation-residual reconstruction')
    native=source['observation.state.body_world'].reshape(T,22,3)
    check(np.array_equal(arrays['source.feet_world'],native[:,[10,11]].reshape(T,6)),'Ground fallback source feet changed')
    check(np.array_equal(arrays['source.feet_valid'],observed_p[:,[10,11]]),'Ground fallback source feet masks changed')
    for side,joint in [('left',20),('right',21)]:
        hand=source[f'observation.state.hand_{side}_world'].reshape(T,21,3)
        handmask=source[f'observation.valid.hand_{side}_position']
        good=handmask[:,0]&pv[:,joint]
        metrics[side+'_native_hand_to_fitted_wrist_m']=summary(np.linalg.norm(hand[:,0]-expected_p[:,joint],axis=-1)[good])
    # Contact diagnostics are independent 15Hz horizontal speed proxies only;
    # no stored ground-truth contact or unrequested sole/floor inference.
    selected=np.arange(0,T,2);feet=expected_p[selected][:,[10,11]];feet_valid=pv[selected][:,[10,11]]
    speed=np.linalg.norm(np.diff(feet[...,:2],axis=0),axis=-1)*15
    pairs=feet_valid[:-1]&feet_valid[1:]
    metrics['fitted_feet_horizontal_speed_15Hz_m_s']=summary(speed[pairs])
    ground_valid=source['observation.valid.floor'].reshape(T)
    height=source['observation.environment.floor_height'].reshape(T)
    floor_feet=pv[:,[10,11]]&ground_valid[:,None]
    native_floor_feet=observed_p[:,[10,11]]&ground_valid[:,None]
    metrics['fitted_foot_joint_height_over_archived_ground_m']=summary((expected_p[:,[10,11],2]-height[:,None])[floor_feet])
    metrics['native_foot_landmark_height_over_archived_ground_m']=summary((native[:,[10,11],2]-height[:,None])[native_floor_feet])
    times=source['observation.source_timestamp']
    dt=np.diff(times)
    source_R=quat_matrix(source['observation.state.body_world_rotation'].reshape(T,22,4))
    source_increment=source_R[1:]@np.swapaxes(source_R[:-1],-1,-2)
    fitted_increment=expected_R[1:]@np.swapaxes(expected_R[:-1],-1,-2)
    error_R=fitted_increment@np.swapaxes(source_increment,-1,-2)
    degrees=np.degrees(np.arccos(np.clip((np.trace(error_R,axis1=-2,axis2=-1)-1)/2,-1,1)))
    source_angle=np.degrees(np.arccos(np.clip((np.trace(source_increment,axis1=-2,axis2=-1)-1)/2,-1,1)))
    motion_pairs=observed_r[1:]&observed_r[:-1]&rv[1:]&rv[:-1]&(dt<=.05)[:,None]&(source_angle<=45)
    metrics['world_rotation_increment_residual_deg']=summary(degrees[motion_pairs])
    if execution_metadata:
        available=observed_p.any(1)
        available |= source['observation.valid.hand_left_position'][:,0] | source['observation.valid.hand_right_position'][:,0]
        root_jump=(np.linalg.norm(np.diff(native[:,0],axis=0),axis=-1)>.5)&observed_p[1:,0]&observed_p[:-1,0]
        transition=((dt>0)&(dt<=.05)&~root_jump)[:,None]&~((observed_r[1:]&observed_r[:-1])&(source_angle>45))
        continuous=transition.all(1)
        fill=available.copy();i=0
        while i<T:
            if available[i]:i+=1;continue
            j=i+1
            while j<T and not available[j]:j+=1
            if i>0 and j<T and j-i<=contract['config']['max_missing_frames'] and continuous[i-1:j].all():fill[i:j]=True
            i=j
        segments=[];i=0
        while i<T:
            if not fill[i]:i+=1;continue
            j=i+1
            while j<T and fill[j] and continuous[j-1]:j+=1
            segments.append([i,j]);i=j
        check(execution_metadata.get('segments')==segments,'Actual fitted sequence segment/reset/gap decomposition mismatch')
        check(not np.any(pv&~fill[:,None]) and not np.any(rv&~fill[:,None]),'Estimated labels crossed unavailable long gaps')
        quality=execution.get('quality',{})
        check(quality.get('unavailable_frames')==np.flatnonzero(~fill).tolist(),'Actual unavailable frame diagnostics mismatch')
        metrics['independent_segments']=len(segments)
        metrics['independent_unavailable_frames']=int((~fill).sum())
        metrics['filled_short_missing_frames']=int((fill&~available).sum())
        within_segment=np.zeros(max(T-1,0),bool)
        for a,b in segments:within_segment[a:max(a,b-1)]=True
        local_increment=local_R[1:]@np.swapaxes(local_R[:-1],-1,-2)
        local_change=np.degrees(np.arccos(np.clip((np.trace(local_increment,axis1=-2,axis2=-1)-1)/2,-1,1)))
        fitted_change=np.degrees(np.arccos(np.clip((np.trace(fitted_increment,axis1=-2,axis2=-1)-1)/2,-1,1)))
        local_pairs=within_segment[:,None]&rv[1:]&rv[:-1]
        metrics['continuous_local_rotation_change_deg']=summary(local_change[local_pairs])
        metrics['continuous_local_angular_speed_deg_s']=summary((local_change/np.maximum(dt[:,None],1e-12))[local_pairs])
        metrics['continuous_world_rotation_change_deg']=summary(fitted_change[local_pairs])
        metrics['continuous_local_step_above45deg_count']=int(((local_change>45)&local_pairs).sum())
        metrics['continuous_local_step_above90deg_count']=int(((local_change>90)&local_pairs).sum())
        metrics['continuous_rotation_pair_count']=int(local_pairs.sum())
        metrics['near_half_turn_local_pose_joint_count']=int((np.linalg.norm(aa,axis=-1)>np.deg2rad(175))[rv].sum())
        root_speed=np.linalg.norm(np.diff(expected_p[:,0],axis=0),axis=-1)/np.maximum(dt,1e-12)
        metrics['continuous_root_speed_m_s']=summary(root_speed[within_segment&pv[1:,0]&pv[:-1,0]])
        jumps=(local_change>45)&local_pairs
        metrics['rotation_confidence_at_above45deg_steps']=summary(np.minimum(arrays['quality.rotation_confidence'][1:],arrays['quality.rotation_confidence'][:-1])[jumps])
    metrics['15Hz_selected_frames']=len(selected)
    metrics['source_position_coverage']=float(observed_p.mean())
    metrics['estimated_position_coverage']=float(pv.mean())
    metrics['source_rotation_coverage']=float(observed_r.mean())
    metrics['estimated_rotation_coverage']=float(rv.mean())
    metrics['observation_reconstruction_residual_m']=summary(residual[residual_valid])
    if selection:
        matching=[r for r in selection['calibration_records']+selection['validation_records'] if r['episode_index']==idx]
        check(len(matching)==1,'Sidecar episode absent/duplicated in frozen selection')
        if matching:
            check(matching[0]['source_key']==declaration['source_key'],'Frozen selection source_key mismatch')
    return {'passed':not failures,'episode_index':idx,'source_key':declaration['source_key'],'frames':T,'failures':failures,'metrics':metrics,'model_sha256':contract['model_sha256'],'config_sha256':contract['config_sha256'],'calibration_sha256':contract['calibration_sha256'],'numeric_sha256':receipt['numeric_sha256'],'native_axis_certified':False,'motion_truth_certified':False}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sidecar-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',type=Path)
    parser.add_argument('--job',type=Path,help='Independently verify frozen source/train prior evidence and producer code')
    parser.add_argument('--episodes',type=int,nargs='*')
    args=parser.parse_args()
    selection=json.loads(args.selection.read_text()) if args.selection else None
    folders=sorted((args.sidecar_root/'episodes').glob('[0-9]'*8))
    if args.episodes:folders=[p for p in folders if int(p.name) in args.episodes]
    results=[];cache={}
    for folder in folders:
        try:r=audit_episode(folder,cache,selection=selection,job=args.job)
        except Exception as e:r={'passed':False,'episode_index':int(folder.name),'failures':[f'{type(e).__name__}: {e}']}
        results.append(r)
        print(json.dumps({'episode_index':r['episode_index'],'passed':r['passed'],'failures':r.get('failures',[])}),flush=True)
    completeness_failures=[]
    if selection and not args.episodes:
        required={int(r['episode_index']) for r in selection['validation_records']}
        found={r['episode_index'] for r in results}
        if found!=required:
            completeness_failures.append({'expected':'all_frozen_validation_episodes','missing':sorted(required-found),'unexpected':sorted(found-required)})
    prior_audit=None
    if args.job:
        try:prior_audit=audit_train_prior(args.job)
        except Exception as e:prior_audit={'passed':False,'failures':[f'{type(e).__name__}: {e}']}
    passed=bool(results) and all(r['passed'] for r in results) and not completeness_failures and (prior_audit is None or prior_audit['passed'])
    report={'format':'independent-smplh-sidecar-audit-v1','passed':passed,'completeness_failures':completeness_failures,'shared_train_prior_audit':prior_audit,'sidecar_root':str(args.sidecar_root.resolve()),'auditor_sha256':file_sha(__file__),'selection_sha256':file_sha(args.selection) if args.selection else None,'episodes_checked':len(results),'frames_checked':sum(r.get('frames',0) for r in results),'numeric_consistency_certified':passed,'motion_truth_certified':False,'native_axis_certified':False,'results':results}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
    return 0 if report['passed'] else 1

if __name__=='__main__':sys.exit(main())
