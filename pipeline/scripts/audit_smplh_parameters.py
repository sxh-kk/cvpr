#!/usr/bin/env python3
"""Independent full-population SMPL-H parameter dataset acceptance.

Numerics use NumPy and the immutable model arrays, never the producer decoder,
geometry, schema helpers, or fitted body caches. Source joins are checked before
cleanup; a separate fresh-process probe must prove reader independence from old
dataset paths. Storage consistency is not anatomical or motion-truth accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ.setdefault(_name,'1')

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

PARAMETER_SCHEMA_VERSION='egomotion-smplh16-bodyparams-hands21-camera-floor-dino5-v1'
PARENTS22=np.array([-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19])
PARAMETERS={'smplh.root_orient':3,'smplh.pose_body':63,'smplh.trans':3,'smplh.betas':16}
OFFICIAL_MODEL_SHA256='be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947'


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):digest.update(block)
    return digest.hexdigest()


def numeric_sha(arrays):
    digest=hashlib.sha256()
    for key in sorted(arrays):
        value=np.ascontiguousarray(arrays[key]);digest.update(canonical([key,value.dtype.str,list(value.shape)]));digest.update(value.tobytes())
    return digest.hexdigest()


def read_arrays(path,columns=None):
    table=pq.read_table(path,columns=columns);arrays={}
    for key in table.column_names:
        column=table[key].combine_chunks()
        if column.null_count:raise ValueError(key+': null source/output values')
        if pa.types.is_fixed_size_list(column.type):
            if column.values.null_count:raise ValueError(key+': null list values')
            arrays[key]=column.values.to_numpy(zero_copy_only=False).reshape(len(column),column.type.list_size)
        else:arrays[key]=column.to_numpy(zero_copy_only=False)
    return arrays


def owned_path(root,path):
    root=Path(root).resolve(strict=True);relative=Path(path)
    if relative.is_absolute():raise ValueError('Owned artifact/model paths must be relative')
    result=(root/relative).resolve(strict=True)
    if not result.is_relative_to(root):raise ValueError('Owned artifact/model escapes dataset root')
    if not result.is_file():raise ValueError('Owned artifact is not a regular file')
    return result


def scalar(value):
    a=np.asarray(value)
    return a.reshape(-1) if a.ndim==2 and a.shape[1]==1 else a


def exp_so3(vector):
    vector=np.asarray(vector,dtype=np.float64);theta=np.linalg.norm(vector,axis=-1)
    K=np.zeros((*vector.shape[:-1],3,3));x,y,z=np.moveaxis(vector,-1,0)
    K[...,0,1]=-z;K[...,0,2]=y;K[...,1,0]=z;K[...,1,2]=-x;K[...,2,0]=-y;K[...,2,1]=x
    return np.eye(3)+np.sinc(theta/np.pi)[...,None,None]*K+(.5*np.sinc(theta/(2*np.pi))**2)[...,None,None]*(K@K)


def quat_matrix(quaternion):
    q=np.asarray(quaternion,dtype=np.float64);norm=np.linalg.norm(q,axis=-1,keepdims=True)
    if np.any(~np.isfinite(q)) or np.any(norm<1e-12):raise ValueError('Invalid archived quaternion')
    w,x,y,z=np.moveaxis(q/norm,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(*q.shape[:-1],3,3)


def row6_matrix(rotation6):
    six=np.asarray(rotation6,dtype=np.float64).reshape(*np.asarray(rotation6).shape[:-1],2,3)
    a,b=six[...,0,:],six[...,1,:];norm=np.linalg.norm(a,axis=-1,keepdims=True)
    if np.any(norm<1e-10):raise ValueError('Degenerate row6D first row')
    a=a/norm;b=b-(a*b).sum(-1,keepdims=True)*a;norm=np.linalg.norm(b,axis=-1,keepdims=True)
    if np.any(norm<1e-10):raise ValueError('Degenerate row6D second row')
    b=b/norm
    return np.stack((a,b,np.cross(a,b)),axis=-2)


def load_model(path,expected_sha):
    if sha_file(path)!=expected_sha:raise ValueError('Immutable matching model SHA differs')
    with np.load(path,allow_pickle=False) as source:
        template=np.asarray(source['v_template'],dtype=np.float64)
        regressor=np.asarray(source['J_regressor'],dtype=np.float64)
        directions=np.asarray(source['shapedirs'],dtype=np.float64)
        tree=np.asarray(source['kintree_table'])
    if template.shape!=(6890,3) or regressor.shape!=(52,6890) or tree.shape!=(2,52) or directions.shape!=(6890,3,16):
        raise ValueError('Required real 6890/52/true16beta model layout differs')
    if any(not np.isfinite(x).all() for x in (template,regressor,directions)):raise ValueError('Nonfinite model')
    ids={int(value):index for index,value in enumerate(tree[1])}
    if len(ids)!=52:raise ValueError('Nonunique model joint IDs')
    parents=np.array([-1]+[ids[int(value)] for value in tree[0,1:]])
    expected=PARENTS22.tolist()
    for wrist in (20,21):
        for _ in range(5):
            start=len(expected);expected.extend([wrist,start,start+1])
    if not np.array_equal(parents,expected):raise ValueError('Full52 SMPL-H body and hand hierarchy differs')
    return dict(rest=regressor@template,shape=np.einsum('jv,vcs->jcs',regressor,directions),parents=parents,sha256=expected_sha)


def numpy_fk(model,arrays,*,pose_hand_axis_angle=None):
    """Full52 FK; unmeasured Suite fingers use rest, not generated hand21 labels."""
    root=np.asarray(arrays['smplh.root_orient'],dtype=np.float64)
    count=len(root);pose=np.asarray(arrays['smplh.pose_body'],dtype=np.float64)
    trans=np.asarray(arrays['smplh.trans'],dtype=np.float64);beta=np.asarray(arrays['smplh.betas'],dtype=np.float64)
    if root.shape!=(count,3) or pose.shape!=(count,63) or trans.shape!=(count,3) or beta.shape!=(count,16):raise ValueError('Parameter dimensions must be 3+63+3+16')
    if any(not np.isfinite(x).all() for x in (root,pose,trans,beta)):raise ValueError('Stored parameters must be finite, with explicit source validity')
    axis_angle=np.zeros((count,52,3));axis_angle[:,0]=root;axis_angle[:,1:22]=pose.reshape(count,21,3)
    if pose_hand_axis_angle is not None:
        hand=np.asarray(pose_hand_axis_angle,dtype=np.float64)
        if hand.shape!=(count,90) or not np.isfinite(hand).all():raise ValueError('Full52 finger local AA must be (T,90)')
        axis_angle[:,22:]=hand.reshape(count,30,3)
    local=exp_so3(axis_angle)
    J=model['rest'][None]+np.einsum('jcs,ts->tjc',model['shape'],beta)
    R=np.empty_like(local);p=np.empty((count,52,3));R[:,0]=local[:,0];p[:,0]=J[:,0]+trans
    for joint in range(1,52):
        parent=model['parents'][joint];R[:,joint]=R[:,parent]@local[:,joint]
        p[:,joint]=p[:,parent]+np.einsum('tij,tj->ti',R[:,parent],J[:,joint]-J[:,parent])
    return dict(positions=p,global_rotations=R,local_rotations=local,rest=J,root_world=J[:,0]+trans)


def hand_pca_to_axis_angle(coefficients,basis_path,expected_sha):
    """HM full45 PCA coefficients still require components and hand means."""
    if sha_file(basis_path)!=expected_sha:raise ValueError('Owned hand-PCA basis SHA differs')
    coefficient=np.asarray(coefficients,dtype=np.float64)
    if coefficient.ndim!=2 or coefficient.shape[1]!=90:raise ValueError('Expected HM45+45 hand PCA coefficients')
    with np.load(basis_path,allow_pickle=False) as model:
        outputs=[]
        for side,part in [('l',coefficient[:,:45]),('r',coefficient[:,45:])]:
            components=np.asarray(model['hands_components'+side],dtype=np.float64);mean=np.asarray(model['hands_mean'+side],dtype=np.float64)
            if components.shape!=(45,45) or mean.shape!=(45,):raise ValueError('Full45 hand-PCA/mean contract differs')
            outputs.append(part@components+mean)
    return np.concatenate(outputs,-1)


def close(actual,expected,mask=None,*,name,tolerance=2e-5):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape:raise ValueError(name+': shape differs')
    if mask is not None:a=a[mask];b=b[mask]
    error=float(np.max(np.abs(a-b))) if a.size else 0.
    if not np.isfinite(error) or error>tolerance:raise ValueError(f'{name}: independent mismatch {error:g}')
    return error


def exact(actual,expected,*,name):
    a=np.asarray(actual);b=np.asarray(expected)
    if a.shape!=b.shape or a.dtype!=b.dtype or not np.array_equal(a,b):raise ValueError(name+': source dtype/value/mask differs')


def streaming_digest(path,columns,batch_size=512):
    """Digest per-column row bytes, independent of Parquet row-group layout."""
    file=pq.ParquetFile(path);digests={key:hashlib.sha256() for key in columns};counts={key:0 for key in columns};specs={}
    for batch in file.iter_batches(batch_size=batch_size,columns=columns):
        table=pa.Table.from_batches([batch])
        for key in columns:
            c=table[key].combine_chunks()
            if c.null_count:raise ValueError(key+': null values during full streaming digest')
            if pa.types.is_fixed_size_list(c.type):
                if c.values.null_count:raise ValueError(key+': null list values')
                a=c.values.to_numpy(zero_copy_only=False).reshape(len(c),c.type.list_size)
            else:a=c.to_numpy(zero_copy_only=False)
            a=np.ascontiguousarray(a);spec=(a.dtype.str,a.shape[1:]);counts[key]+=len(a)
            if key in specs and specs[key]!=spec:raise ValueError('Streaming dtype/shape changed')
            specs[key]=spec;digests[key].update(a.tobytes())
    return {key:dict(dtype=specs[key][0],shape=[counts[key],*specs[key][1]],rows=counts[key],row_bytes_sha256=digests[key].hexdigest()) for key in columns}


def compare_streaming(source,output,mapping):
    """Every DINO row, frame ID, validity and timing value must survive exactly."""
    before=streaming_digest(source,list(mapping));after=streaming_digest(output,list(mapping.values()))
    for source_key,target_key in mapping.items():
        if before[source_key]!=after[target_key]:raise ValueError('Full streaming source preservation differs: '+source_key+' -> '+target_key)
    return dict(source=before,output=after,all_rows_equal=True)


def validate_parameter_masks(arrays,count):
    """Availability, original observations and completion are independent."""
    for component in ('position','rotation'):
        keys=[f'estimated.valid.body_{component}',f'source.observed.body_{component}',f'source.interpolated.body_{component}',f'estimated.completed.body_{component}']
        for key in keys:
            a=np.asarray(arrays[key])
            if a.dtype!=np.bool_ or a.shape!=(count,22):raise ValueError(key+': mask must be bool[T,22]')
        estimated,observed,interpolated,completed=[arrays[key] for key in keys]
        if not np.array_equal(completed,estimated&~observed):raise ValueError(component+': independent model completion differs')
        if np.any(interpolated&~observed):raise ValueError(component+': source interpolation exceeds original availability')
        confidence=np.asarray(arrays[f'quality.{component}_confidence'])
        if confidence.shape!=(count,22) or not np.isfinite(confidence).all() or np.any((confidence<0)|(confidence>1)) or np.any(confidence[~estimated]!=0):raise ValueError(component+': invalid confidence')
    feet=np.asarray(arrays['source.feet_world']);fv=np.asarray(arrays['source.feet_valid']);fi=np.asarray(arrays['source.feet_interpolated'])
    if feet.shape!=(count,6) or fv.shape!=(count,2) or fv.dtype!=np.bool_ or fi.shape!=(count,2) or fi.dtype!=np.bool_:raise ValueError('Original native feet contract differs')
    if not np.isfinite(feet).all() or np.any(feet.reshape(count,2,3)[~fv]!=0):raise ValueError('Invalid original native foot positions')
    if np.any(fi&~fv):raise ValueError('Native foot interpolation exceeds source availability')
    residual=np.asarray(arrays['quality.observation_residual_m']) if 'quality.observation_residual_m' in arrays else None
    if residual is not None:
        valid=np.asarray(arrays['quality.observation_residual_valid'])
        if residual.shape!=(count,22) or valid.shape!=(count,22) or valid.dtype!=np.bool_ or not np.isfinite(residual).all() or np.any(residual<0) or np.any(residual[~valid]!=0):raise ValueError('Invalid original observation residual values/masks')
        if np.any(valid&~(arrays['source.observed.body_position']&arrays['estimated.valid.body_position'])):raise ValueError('Residual validity exceeds original observation or model availability')


def population_records(records,inventory):
    """Exact original source membership, identity, split and frame population."""
    def key(row):return (row['source'],row['source_key'])
    if len({r['episode_index'] for r in records})!=len(records):raise ValueError('Duplicate output episode index')
    if len({key(r) for r in records})!=len(records):raise ValueError('Duplicate output source identity')
    if len({key(r) for r in inventory})!=len(inventory):raise ValueError('Duplicate expected source identity')
    actual={key(r):(r['length'],r['split']) for r in records};expected={key(r):(r['length'],r['split']) for r in inventory}
    if actual!=expected:raise ValueError('Full original source identity/frame/split population differs')
    for source in sorted({r['source'] for r in inventory}):
        operators={}
        for row in inventory:
            if row['source']!=source:continue
            operator=row.get('operator_id') or row.get('metadata',{}).get('operator_id') or row.get('source_record',{}).get('metadata',{}).get('operator_id')
            if operator:operators.setdefault(operator,set()).add(row['split'])
        if any(len(v)>1 for v in operators.values()):raise ValueError(source+': operator leaks across splits')
    groups={}
    for row in inventory:
        for group in row.get('split_group_ids',[]):groups.setdefault((row['source'],group),set()).add(row['split'])
    if any(len(v)>1 for v in groups.values()):raise ValueError('Participant/capture group leaks across splits')
    return dict(episodes=len(records),frames=sum(r['length'] for r in records),sources={source:dict(episodes=sum(r['source']==source for r in records),frames=sum(r['length'] for r in records if r['source']==source)) for source in sorted({r['source'] for r in records})},
                published_source_groups_disjoint=True,cross_source_person_identity_linked=False,physical_human_identity_independently_certified=False)


def independent_specs():
    """Written from the public storage contract, without importing its schema."""
    specs={'timestamp':('float32',1),'observation.source_timestamp':('float64',1),
           'observation.source_interpolation_weight':('float32',1)}
    for key in ('index','frame_index','episode_index','task_index','observation.source_frame_left','observation.source_frame_right'):
        specs[key]=('int64',1)
    for modality,width in (('hand_left',63),('hand_right',63),('head',3),('camera',3),('head_reference',3)):
        specs['observation.state.'+modality+'_world']=('float32',width)
        for kind in ('valid','interpolated'):
            specs['observation.'+kind+'.'+modality+'_position']=('bool',21 if modality.startswith('hand') else 1)
    for modality in ('head','camera','head_reference'):
        specs['observation.state.'+modality+'_world_rotation']=('float32',4)
        for kind in ('valid','interpolated'):specs['observation.'+kind+'.'+modality+'_rotation']=('bool',1)
    specs.update({'observation.valid.dino':('bool',1),'observation.video_timestamp':('float64',1),
                  'observation.video_time_error':('float64',1),'observation.environment.floor_height':('float32',1),
                  'observation.environment.floor_normal':('float32',3)})
    for key in ('floor','floor_verified','head_reference_basis'):specs['observation.valid.'+key]=('bool',1)
    for key,width in {**PARAMETERS,'source.feet_world':6}.items():specs[key]=('float32',width)
    for component in ('position','rotation'):
        for prefix in ('estimated.valid','source.observed','source.interpolated','estimated.completed'):
            specs[prefix+'.body_'+component]=('bool',22)
        specs['quality.'+component+'_confidence']=('float32',22)
    specs.update({'source.feet_valid':('bool',2),'source.feet_interpolated':('bool',2),
                  'quality.observation_residual_m':('float32',22),'quality.observation_residual_valid':('bool',22)})
    return specs


OPTIONAL={'smplh.pose_hand_pca':('float32',90),'smplh.valid.hand_pose':('bool',2),
          'source.capture_timestamp_ns':('int64',1),'source.video_frame_index':('int64',1),
          'observation.features.dino':('float32',5120)}


def validate_physical(arrays,record):
    specs=independent_specs();specs.update({k:v for k,v in OPTIONAL.items() if k in arrays})
    if set(arrays)!=set(specs):raise ValueError('Physical standalone columns differ: '+str(sorted(set(arrays)^set(specs))))
    n=record['length']
    for key,(dtype,width) in specs.items():
        a=np.asarray(arrays[key]);shape=(n,) if width==1 else (n,width)
        if a.shape!=shape or a.dtype!=np.dtype(dtype) or not np.isfinite(a).all():raise ValueError(key+': exact physical shape/dtype/finite contract differs')
    exact(arrays['frame_index'],np.arange(n,dtype=np.int64),name='Full zero-based frame grid')
    exact(arrays['timestamp'],(np.arange(n)/30).astype(np.float32),name='Exact 30Hz timestamp grid')
    exact(arrays['episode_index'],np.full(n,record['episode_index'],np.int64),name='Owned episode identity')
    exact(arrays['index'],record['dataset_from_index']+np.arange(n,dtype=np.int64),name='Contiguous global frame index')
    if record['dataset_to_index']!=record['dataset_from_index']+n:raise ValueError('Global episode interval differs')
    validate_parameter_masks(arrays,n)
    for component in ('position','rotation'):
        exact(arrays['source.feet_'+('valid' if component=='position' else 'interpolated')],
              arrays[('source.observed' if component=='position' else 'source.interpolated')+'.body_position'][:,[10,11]],name='Original native foot masks')
    for key in arrays:
        if key.startswith('observation.interpolated.'):
            valid=key.replace('observation.interpolated.','observation.valid.')
            if valid in arrays and np.any(arrays[key]&~arrays[valid]):raise ValueError('Independent modality interpolation exceeds source validity')
    if ('smplh.pose_hand_pca' in arrays)!=('smplh.valid.hand_pose' in arrays):raise ValueError('PCA parameters require separate source availability')


def file_identity(path):
    s=Path(path).stat()
    return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,device=s.st_dev,inode=s.st_ino)


def check_identity(path,expected):
    actual=file_identity(path)
    if any(actual.get(k)!=v for k,v in expected.items() if k in actual):raise ValueError('Original input identity changed: '+str(path))


def old_motion_digest(arrays):
    """Independent implementation of the frozen old numeric manifest encoding."""
    h=hashlib.sha256()
    for key in sorted(arrays):
        value=np.ascontiguousarray(arrays[key]);h.update(key.encode());h.update(str(value.dtype).encode())
        h.update(json.dumps(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def assert_source_record(record,row):
    for key in ('episode_index','source','source_key','source_episode_index','length','split','dataset_from_index','dataset_to_index'):
        if record.get(key)!=row.get(key):raise ValueError('Frozen source membership differs: '+key)
    if record.get('source_uid')!=row.get('source_uid',row['source_key'].split(':',1)[-1]):raise ValueError('Original source UID differs')
    if record.get('split_group_ids',[])!=row.get('split_group_ids',[]):raise ValueError('Original participant/capture split group differs')


def audit_ego(arrays,record,row,model):
    p=record['source_provenance'];source=Path(row['source_path']);fitted=Path(row['fitted_path'])
    if p['source_path']!=str(source) or p['fitted_path']!=str(fitted):raise ValueError('Original Ego source/fitted path identity differs')
    check_identity(source,p['source_identity']);check_identity(fitted,p['fitted_identity'])
    if sha_file(row['fitted_receipt_path'])!=p['fitted_receipt_sha256']:raise ValueError('Frozen fitted receipt SHA differs')
    for key in ('ground','head_reference','coordinate_verified','camera_pose_basis','world_registration','clock','bad_frame_masks'):
        if record['metadata'].get(key)!=row['source_record']['metadata'].get(key):raise ValueError('Original Ego source coordinate/floor/clock/mask provenance changed: '+key)
    columns=[k for k in pq.ParquetFile(source).schema_arrow.names if k!='observation.features.dino']
    raw=read_arrays(source,columns)
    if old_motion_digest(raw)!=p['source_motion_sha256']:raise ValueError('Full original motion numeric SHA differs')
    for key in independent_specs():
        if key in raw:
            expected=raw[key].astype(np.float64) if key=='observation.video_time_error' else raw[key]
            exact(arrays[key],expected,name='All-row Ego independent modality '+key)
    for component in ('position','rotation'):
        for prefix,oldprefix in (('source.observed','observation.valid'),('source.interpolated','observation.interpolated')):
            exact(arrays[prefix+'.body_'+component],raw[oldprefix+'.body_'+component],name='Original Ego body observation '+component)
    exact(arrays['source.feet_world'],raw['observation.state.body_world'].reshape(-1,22,3)[:,[10,11]].reshape(-1,6),name='Original Ego native feet')
    fit=read_arrays(fitted,['fitted.local_axis_angle','fitted.smpl_trans','fitted.betas','fitted.body_world_rotation',
                           'fitted.body_world_rot6d_xyz','fitted.body_floor_rot6d_xyz',
                           'estimated.valid.body_world_position','estimated.valid.body_world_rotation',
                           'estimated.valid.body_floor_position','estimated.valid.body_floor_rotation',
                           'quality.position_confidence','quality.rotation_confidence','quality.observation_residual_m','quality.observation_residual_valid'])
    exact(np.concatenate((arrays['smplh.root_orient'],arrays['smplh.pose_body']),-1),fit['fitted.local_axis_angle'],name='Frozen Ego local axis angles')
    for key,target in (('smplh.trans','fitted.smpl_trans'),('smplh.betas','fitted.betas'),
                       ('estimated.valid.body_position','estimated.valid.body_world_position'),('estimated.valid.body_rotation','estimated.valid.body_world_rotation'),
                       ('quality.position_confidence','quality.position_confidence'),('quality.rotation_confidence','quality.rotation_confidence'),
                       ('quality.observation_residual_m','quality.observation_residual_m'),('quality.observation_residual_valid','quality.observation_residual_valid')):
        exact(arrays[key],fit[target],name='Frozen Ego parameter/quality '+key)
    receipt=json.loads(Path(row['fitted_receipt_path']).read_text())
    if receipt['declaration']['contract']['model_sha256']!=model['sha256']:raise ValueError('Frozen fitted SMPL-H asset differs')
    fk=numpy_fk(model,arrays);P=fk['positions'][:,:22];R=fk['global_rotations'][:,:22]
    wpv=arrays['estimated.valid.body_position'];wrv=arrays['estimated.valid.body_rotation']
    world=fit['fitted.body_world_rot6d_xyz'].reshape(-1,22,9)
    xyzerr=close(world[...,6:],np.where(wpv[...,None],P,0),name='Actual full-row frozen FK world XYZ')
    rerr=close(world[...,:6],np.where(wrv[...,None],R[...,:2,:].reshape(-1,22,6),0),name='Actual full-row frozen row6D')
    close(quat_matrix(fit['fitted.body_world_rotation'].reshape(-1,22,4)),R,wrv,name='Frozen global quaternion FK')
    floorq=quat_matrix(raw['observation.state.head_reference_floor_world_rotation'])
    origin=raw['observation.state.head_reference_floor_world'];fpv=fit['estimated.valid.body_floor_position'];frv=fit['estimated.valid.body_floor_rotation']
    floor=fit['fitted.body_floor_rot6d_xyz'].reshape(-1,22,9)
    lp=np.einsum('tji,tkj->tki',floorq,P-origin[:,None]);lr=np.swapaxes(floorq,-1,-2)[:,None]@R
    close(floor[...,6:],np.where(fpv[...,None],lp,0),name='Actual frozen floor XYZ')
    close(floor[...,:6],np.where(frv[...,None],lr[...,:2,:].reshape(-1,22,6),0),name='Actual frozen floor row6D')
    if record['metadata'].get('parameter_provenance',{}).get('finger_pose_fitted') is not False:raise ValueError('EgoSuite unobserved SMPL finger pose must stay explicitly unfitted')
    if 'smplh.valid.hand_pose' in arrays and arrays['smplh.valid.hand_pose'].any():raise ValueError('EgoSuite must not invent source finger pose availability')
    check_identity(fitted,p['fitted_identity'])
    return dict(source_rows_preserved=len(P),independent_fk_xyz_max_error_m=xyzerr,independent_fk_rotation_max_element_error=rerr,
                full52_evaluated=True,finger_pose_semantics='unfitted_rest_chain_only; explicit_source_hand21_preserved',motion_truth_accuracy_established=False)


def load_hm_raw(path):
    with np.load(path,allow_pickle=False) as source:
        raw={key:source[key][0].copy() for key in ('trans','root_orient','pose_body','betas_per_frame','hand_pose','valid','joints3d')}
        raw.update({key:source[key].copy() for key in ('chunk_ranges','cam_R','cam_t')})
    n=len(raw['trans']);chunks=np.asarray(raw['chunk_ranges'])
    if chunks.ndim!=2 or chunks.shape[1]!=2 or chunks[0,0]!=0 or chunks[-1,1]!=n or np.any(chunks[:,1]<=chunks[:,0]) or np.any(chunks[1:,0]!=chunks[:-1,1]):raise ValueError('Original HM chunk cover differs from exact full take')
    for key,width in [('trans',3),('root_orient',3),('pose_body',63),('betas_per_frame',16),('hand_pose',90)]:
        if raw[key].shape!=(n,width):raise ValueError('Original HM real16/native parameter shape differs: '+key)
    if raw['joints3d'].shape!=(n,67,3) or raw['valid'].shape!=(n,):raise ValueError('Original HM released source shape differs')
    return raw


def hm_masks(raw):
    n=len(raw['trans']);aa=np.concatenate((raw['root_orient'][:,None],raw['pose_body'].reshape(n,21,3)),1)
    rv=np.isfinite(aa).all(-1)&raw['valid'].astype(bool)[:,None];pv=rv.copy()
    pv[:,0]&=np.isfinite(raw['betas_per_frame']).all(-1)&np.isfinite(raw['trans']).all(-1)
    for j in range(1,22):rv[:,j]&=rv[:,PARENTS22[j]];pv[:,j]&=pv[:,PARENTS22[j]]
    return pv,rv


def verify_runtime_join(join,inventory_join,actual_video_rows):
    """Only measured full timesync row count may extend the frozen join."""
    if any(key not in join or join[key]!=value for key,value in inventory_join.items()):raise ValueError('Frozen original frame-chain source contract differs')
    extra=set(join)-set(inventory_join)
    if extra-{'actual_timesync_rows'}:raise ValueError('Unknown runtime frame-chain field')
    if 'actual_timesync_rows' in join:
        rows=join['actual_timesync_rows']
        if isinstance(rows,bool) or not isinstance(rows,int) or rows!=actual_video_rows:raise ValueError('Runtime actual_timesync_rows differs from independently read full CSV slice')


def audit_hm_alignment(record,row,raw):
    meta=record['metadata'];join=meta['frame_alignment'];take=record['source_provenance']['take_metadata'];n=len(raw['trans'])
    if take!=row['take_metadata'] or join!=record['source_provenance']['frame_alignment']:raise ValueError('Frozen original take/frame-chain provenance differs')
    if join.get('hm_revision')!='6e1cd8627a0c5751b02493a87172dd3709c0c652' or meta.get('source_revision')!=join['hm_revision']:raise ValueError('Unpinned HM release')
    video_n=int(take['timesync_end_idx']-take['timesync_start_idx'])
    if 'actual_timesync_rows' in join:
        source_root=Path(row['source_path']).parents[2]
        sync=source_root/'EgoExo4D/captures'/take['capture']['capture_name']/'timesync.csv'
        cameras=[c['cam_id'] for c in take['capture']['cameras'] if c.get('is_ego') and c['cam_id'].startswith('aria')]
        if len(cameras)!=1:raise ValueError('Cannot independently prove runtime timesync extension without unique Aria source')
        stamps=timestamp_slice(sync,cameras[0]+'_214-1_capture_timestamp_ns',int(take['timesync_start_idx']),int(take['timesync_end_idx']))
        actual_video_rows=len(stamps)
    else:actual_video_rows=video_n
    verify_runtime_join(join,row['frame_alignment'],actual_video_rows)
    if join.get('hm_rows')!=n or join.get('video_rows')!=video_n or join.get('take_uid')!=take['take_uid']:raise ValueError('HM take/frame identity chain differs')
    if record['source_key'] not in ('egoexo4d_hm:'+take['take_name'],take['take_name']):raise ValueError('HM source UID differs from take metadata')
    for key,offset in [('hm_row_to_video_frame',0),('hm_row_to_extracted_image',1)]:
        if join.get(key)!={'offset':offset,'step':1,'length':n}:raise ValueError('HM image/video mapping differs')
    evidence=join['source_evidence']
    if evidence.get('commit')!='135d44ee5310c554fde1cbe33e02c7c204717588':raise ValueError('Unpinned HM generator/viewer implementation')
    required={'scripts/run_pipeline.py':'371461de4ea40f4f4a3ccc45ec90988dfa7590657a14999bfba55db3c3eb6f55',
              'slahmr/data/dataset.py':'bf2295037540f1034027489b5657881f610e473130190aa705e94e8260b97a37',
              'scripts/run_mesh_vis_hands_egoexo.py':'b0dde65e6c82a0c7d3aa46b51261ac04e0bda6ea71dc0aafe0e691b42b109b1f',
              'scripts/undistort_egoexo.py':'024b4cb51147e9c09b6577c5324744554eee822249acf1db9e0165dfc009b10d'}
    if {v['path']:v['sha256'] for v in evidence['files']}!=required:raise ValueError('Generator/viewer evidence hashes differ')
    vision=Path(row['vision_source_path']);dino=json.loads(vision.with_suffix('.json').read_text());video=dino.get('video',{})
    video_path=Path(video.get('video_path',''));s=video_path.stat()
    if (dino.get('contract')!='egoexo-video-dino5-30hz-v1' or dino.get('state')!='complete'
            or video.get('take_name')!=take['take_name'] or video.get('take_uid')!=take['take_uid']
            or video.get('length')!=video_n or dino.get('length')!=video_n or video.get('fps')!=30
            or video.get('clock')!='frame_aligned_take_seconds' or video.get('video_info',{}).get('frames')!=video_n
            or video.get('video_info',{}).get('average_rate')!='30' or video.get('video_info',{}).get('start_time_s')!=0.
            or video.get('video_fingerprint')!={'size_bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
            or video.get('relative_video_path')!=f"takes/{take['take_name']}/frame_aligned_videos/{video_path.name}"
            or pq.ParquetFile(vision).metadata.num_rows!=video_n):
        raise ValueError('Independent full HM take/video/DINO identity/length/rate/fingerprint chain differs')
    source_clock=read_arrays(vision,['frame_index','timestamp'])
    exact(source_clock['frame_index'],np.arange(video_n,dtype=np.int64),name='Original complete HM DINO frame grid')
    exact(source_clock['timestamp'],np.arange(video_n)/30,name='Original complete HM DINO source clock')
    if sha_file(vision)!=dino['parquet_sha256']:raise ValueError('Complete original HM DINO file SHA differs from source release extraction receipt')
    # The seven historical rounded takes may have separately pinned direct-viewer
    # evidence. Their generation length is not falsely asserted to be N-1.
    if join.get('verified'):
        if n!=video_n-1:
            registered={'iiith_cooking_110_2','iiith_cooking_120_4','iiith_cooking_23_1','iiith_cooking_24_1','iiith_cooking_43_1','iiith_cooking_62_2','sfu_cooking020_6'}
            if not (take['take_name'] in registered and n==(video_n//100)*100 and 0<video_n-n<100
                    and np.all(raw['chunk_ranges'][:,1]-raw['chunk_ranges'][:,0]==100)
                    and join.get('method')=='verified_by_pinned_release_viewer_row_contract_legacy_full100_tail'
                    and join.get('legacy_generation_flags') and join.get('legacy_rule_evidence')
                    and join.get('unused_video_frame_range_half_open')==[n,video_n]):
                raise ValueError('HM verified non-N-1 release lacks explicit rounded-generation evidence')
        return True
    if meta.get('training_eligible') or meta.get('canonical_training_eligible'):raise ValueError('Unverified HM mapping cannot be training eligible')
    return False


def audit_hm(arrays,record,row,model,hand_basis):
    p=record['source_provenance'];source=Path(row['source_path']);check_identity(source,p['source_identity'])
    if p['source_path']!=str(source) or sha_file(source)!=p['source_npz_sha256']:raise ValueError('HM raw NPZ provenance differs')
    raw=load_hm_raw(source);n=len(raw['trans']);pv,rv=hm_masks(raw)
    for key,oldkey in [('smplh.root_orient','root_orient'),('smplh.pose_body','pose_body'),('smplh.trans','trans'),('smplh.betas','betas_per_frame'),('smplh.pose_hand_pca','hand_pose')]:
        expected=np.nan_to_num(raw[oldkey],nan=0,posinf=0,neginf=0)
        exact(arrays[key],expected,name='Every native HM released parameter '+key)
    for component,mask in [('position',pv),('rotation',rv)]:
        exact(arrays['estimated.valid.body_'+component],mask,name='Original HM estimated mask')
        exact(arrays['source.observed.body_'+component],mask,name='Original HM optimizer numeric availability')
        if arrays['source.interpolated.body_'+component].any() or arrays['estimated.completed.body_'+component].any():raise ValueError('Native HM body must not gain interpolated/completed observations')
        exact(arrays['quality.'+component+'_confidence'],mask.astype(np.float32),name='HM binary availability proxy')
    exact(arrays['smplh.valid.hand_pose'],np.isfinite(raw['hand_pose'].reshape(n,2,45)).all(-1)&raw['valid'].astype(bool)[:,None],name='Source hand PCA availability')
    rejected_hand_counts={}
    for side,s,e in [('left',25,46),('right',46,67)]:
        valid=raw['valid'].astype(bool)[:,None]&np.isfinite(raw['joints3d'][:,s:e]).all(-1)
        exact(arrays['observation.valid.hand_'+side+'_position'],valid,name='Original HM explicit hand21 masks')
        exact(arrays['observation.state.hand_'+side+'_world'],np.nan_to_num(raw['joints3d'][:,s:e],nan=0,posinf=0,neginf=0).reshape(n,63),name='Every finite original HM hand21 point including invalid chunks')
        rejected_hand_counts[side]=int((~raw['valid'].astype(bool)[:,None,None]&np.isfinite(raw['joints3d'][:,s:e])).sum())
    if arrays['quality.observation_residual_valid'].any() or np.any(arrays['quality.observation_residual_m']!=0):raise ValueError('HM lacks independent capture residual ground truth')
    verified=audit_hm_alignment(record,row,raw)
    hand=hand_pca_to_axis_angle(arrays['smplh.pose_hand_pca'],hand_basis['path'],hand_basis['sha256'])
    fk=numpy_fk(model,arrays,pose_hand_axis_angle=hand);P=fk['positions'][:,:22]
    close(arrays['source.feet_world'].reshape(n,2,3),np.where(pv[:,[10,11],None],P[:,[10,11]],0),name='Native released HM model feet')
    camera=audit_hm_camera(arrays,record,row,raw,verified)
    audit_ground(arrays,record['metadata'],fk['positions'][:,:22],pv)
    return dict(source_rows_preserved=n,independent_full52_fk=True,hand_pca_components_and_means_owned=True,
                per_frame16beta_preserved=True,body_refitted=False,frame_generation_mapping_verified=verified,
                finite_native_hand_coordinate_values_in_rejected_chunks_preserved=rejected_hand_counts,
                camera=camera,quarantined=not record['metadata'].get('training_eligible',False),
                source_observation_semantics='optimizer_numeric_availability; not_capture_ground_truth',motion_truth_accuracy_established=False)


def timestamp_slice(path,column,start,end):
    result=[]
    with Path(path).open(newline='') as handle:
        rows=csv.DictReader(handle)
        if column not in (rows.fieldnames or []):raise ValueError('Missing original timesync column')
        for i,row in enumerate(rows):
            if i<start:continue
            if i>=end:break
            try:
                value=Decimal(row[column]);stamp=int(value) if value.is_finite() and value==int(value) else -1
            except (InvalidOperation,TypeError,ValueError,OverflowError):stamp=-1
            result.append(stamp)
    if len(result)!=end-start:raise ValueError('Original full timesync slice incomplete')
    return np.asarray(result,np.int64)


def audit_hm_camera(arrays,record,row,raw,aligned):
    """Reopen official SDK/MPS/VRS and rebuild every declared source pose."""
    meta=record['metadata'];provenance=meta.get('aria_provenance',{});n=len(arrays['timestamp'])
    present=scalar(arrays['observation.valid.camera_position'])
    if not aligned:
        if present.any() or scalar(arrays['observation.valid.head_position']).any():raise ValueError('Unmapped HM must not fabricate camera/head poses')
        return {'verified':False,'reason':'unverified source frame mapping; camera masks remain invalid'}
    if not provenance:
        if present.any() or meta.get('coordinate_verified'):raise ValueError('No camera provenance for claimed valid poses/world')
        return {'verified':False,'reason':'explicitly quarantined missing Aria assets'}
    from projectaria_tools.core import data_provider,mps
    for role in ('timesync','trajectory','gopro_calibrations','vrs'):
        original=provenance[role];check_identity(original['path'],original)
        if sha_file(original['path'])!=original['sha256']:raise ValueError('Original Aria source SHA differs: '+role)
    take=record['source_provenance']['take_metadata'];start,end=int(take['timesync_start_idx']),int(take['timesync_end_idx'])
    stamps=timestamp_slice(provenance['timesync']['path'],provenance['capture_timestamp_column'],start,end)[:n]
    exact(arrays['source.capture_timestamp_ns'],stamps,name='Every original HM/video capture timestamp')
    provider=data_provider.create_vrs_data_provider(provenance['vrs']['path']);device=provider.get_device_calibration()
    dc=device.get_camera_calib('camera-rgb').get_transform_device_camera().to_matrix();dh=device.get_transform_device_cpf().to_matrix()
    close(dc,provenance['device_camera_transform'],name='VRS true device-camera transform',tolerance=1e-10)
    close(dh,provenance['device_cpf_transform'],name='VRS true device-CPF transform',tolerance=1e-10)
    take_root=Path(provenance['vrs']['path']).parent
    poses=mps.MpsDataProvider(mps.MpsDataPathsProvider(str(take_root)).get_data_paths())
    world=np.broadcast_to(np.eye(4),(n,4,4)).copy();head=world.copy();valid=np.zeros(n,bool);graphs=set();max_error=0.
    bound=float(provenance['max_pose_error_s'])
    if bound!=.02:raise ValueError('Unexpected declared camera sampling error bound')
    for i,stamp in enumerate(stamps):
        if stamp<=0 or (i and stamp<=stamps[i-1]):continue
        pose=poses.get_closed_loop_pose(int(stamp))
        if pose is None:continue
        error=pose.tracking_timestamp.total_seconds()-int(stamp)/1e9
        if abs(error)>bound:continue
        wd=pose.transform_world_device.to_matrix();world[i]=wd@dc;head[i]=wd@dh;valid[i]=True
        graphs.add(str(pose.graph_uid));max_error=max(max_error,abs(error))
    for modality,pose in [('camera',world),('head',head)]:
        for component in ('position','rotation'):
            exact(scalar(arrays[f'observation.valid.{modality}_{component}']),valid,name='Every original Aria availability mask')
        close(arrays[f'observation.state.{modality}_world'],np.where(valid[:,None],pose[:,:3,3],0),name='Full original '+modality+' center')
        close(quat_matrix(arrays[f'observation.state.{modality}_world_rotation']),pose[:,:3,:3],valid,name='Full original '+modality+' rotation')
    B=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    close(arrays['observation.state.head_reference_world'],arrays['observation.state.camera_world'],name='Camera reference shares actual optical center',tolerance=0.)
    close(quat_matrix(arrays['observation.state.head_reference_world_rotation']),world[:,:3,:3]@B,valid,name='EE4D actual Aria post-multiplication')
    with Path(provenance['gopro_calibrations']['path']).open(newline='') as h:calibrations=[r for r in csv.DictReader(h) if float(r['quality'])==1.0]
    same_world=len(calibrations)>=4 and bool(graphs);errors=[];frame_valid=raw['valid'].astype(bool)
    for view,cal in enumerate(calibrations[:4],1):
        Q=quat_matrix(np.array([float(cal[f'q{a}_world_cam']) for a in 'wxyz']))
        p=np.array([float(cal[f't{a}_world_cam']) for a in 'xyz']);expected_R=Q.T;expected_t=-Q.T@p
        re=float(np.max(np.abs(raw['cam_R'][view,frame_valid]-expected_R))) if frame_valid.any() else float('inf')
        te=float(np.max(np.abs(raw['cam_t'][view,frame_valid]-expected_t))) if frame_valid.any() else float('inf')
        passed=max(re,te)<1e-5 and {str(cal['graph_uid'])}==graphs;same_world&=passed
        errors.append(dict(view=view,rotation_element_error=re,translation_m_error=te,passed=bool(passed)))
    with Path(provenance['trajectory']['path']).open(newline='') as h:
        gravity=np.array([[float(r[f'gravity_{a}_world']) for a in 'xyz'] for r in csv.DictReader(h)])
    gravity_ok=gravity.ndim==2 and gravity.shape[1]==3 and len(gravity)>0 and np.isfinite(gravity).all() and np.max(np.linalg.norm(gravity-[0,0,-9.81],axis=-1))<1e-4
    same_world&=gravity_ok
    if bool(meta['coordinate_verified'])!=bool(same_world):raise ValueError('Independent external-calibration/MPS graph/world evidence differs')
    if meta.get('training_eligible') and (not same_world or not valid.any()):raise ValueError('Unverified camera/world must stay training-ineligible')
    return dict(verified=bool(same_world),full_source_pose_rows=n,valid_source_camera_rows=int(valid.sum()),external_calibration_errors=errors,
                gravity_Z_up_verified=bool(gravity_ok),camera_max_nearest_time_error_s=max_error,physical_mount_accuracy_established=False)


def density_height_clusters(heights,eps=.005,min_samples=3):
    """Independent sorted 1D DBSCAN, with duplicate density preserved."""
    z=np.sort(np.asarray(heights,np.float64));n=len(z)
    if not n:return []
    lo=np.searchsorted(z,z-eps,'left');hi=np.searchsorted(z,z+eps,'right');core=hi-lo>=min_samples
    labels=np.full(n,-1,np.int64);core_ids=np.flatnonzero(core);label=-1;last=None
    for i in core_ids:
        if last is None or z[i]-z[last]>eps:label+=1
        labels[i]=label;last=i
    # sklearn visits sorted unique points: a border belongs to the first reachable
    # cluster, never merges two separate core components through border points.
    for i in np.flatnonzero(~core):
        neighbors=core_ids[(core_ids>=lo[i])&(core_ids<hi[i])]
        if len(neighbors):labels[i]=labels[neighbors[0]]
    return [z[labels==i] for i in range(label+1)]


def audit_ground(arrays,metadata,body,valid):
    ground=metadata['ground'];n=len(body)
    if ground.get('mode')!='estimate':raise ValueError('Unexpected native HM ground mode')
    expected_valid=False;expected_height=0.
    if metadata.get('coordinate_verified') and n>=2:
        params=ground.get('parameters',{});speed=float(params.get('static_speed_m_s',.15));eps=float(params.get('cluster_eps_m',.005))
        min_samples=int(params.get('cluster_min_samples',3));offset=float(params.get('toe_offset_m',.01));gap=float(params.get('max_gap_s',.075))
        feet=body[:,[10,11]];mask=valid[:,[10,11]];dt=np.full(n-1,1/30)
        edges=mask[:-1]&mask[1:]&(dt<=gap)[:,None];velocity=np.linalg.norm(np.diff(feet,axis=0),axis=-1)/dt[:,None]
        peak=np.zeros_like(mask,dtype=float);observable=np.zeros_like(mask)
        peak[:-1]=np.maximum(peak[:-1],np.where(edges,velocity,0));peak[1:]=np.maximum(peak[1:],np.where(edges,velocity,0))
        observable[:-1]|=edges;observable[1:]|=edges
        groups=density_height_clusters(feet[...,2][mask&observable&(peak<=speed)],eps,min_samples)
        if groups:expected_valid=True;expected_height=min(float(np.median(z)) for z in groups)-offset
    if bool(ground['valid'])!=expected_valid:raise ValueError('Independent native-foot whole-episode ground availability differs')
    if expected_valid:close(ground['height_m'],expected_height,name='Independent DBSCAN native HM ground height')
    exact(scalar(arrays['observation.valid.floor']),np.full(n,expected_valid,bool),name='Native ground mask')
    close(arrays['observation.environment.floor_height'],np.full(n,ground['height_m']),name='Stored original ground height')
    close(arrays['observation.environment.floor_normal'],np.broadcast_to([0,0,1],(n,3)),name='Native Z-up ground normal',tolerance=0.)


def selected_stream_digest(path,columns,selected,*,casts=None):
    selected=np.asarray(selected,np.int64);file=pq.ParquetFile(path);digests={key:hashlib.sha256() for key in columns};specs={};count=0;start=0
    if selected.ndim!=1 or len(selected)==0 or selected[0]<0 or np.any(np.diff(selected)<=0) or selected[-1]>=file.metadata.num_rows:raise ValueError('Explicit full source vision frame join invalid')
    for batch in file.iter_batches(batch_size=512,columns=columns,use_threads=False):
        stop=start+len(batch);rows=selected[(selected>=start)&(selected<stop)]-start
        if len(rows):
            raw=table_to_arrays(pa.Table.from_batches([batch]));count+=len(rows)
            for key in columns:
                value=np.ascontiguousarray(raw[key][rows]);value=value.astype(casts[key]) if casts and key in casts else value
                specs[key]=(value.dtype.str,value.shape[1:]);digests[key].update(value.tobytes())
        start=stop
        if start>selected[-1]:break
    if count!=len(selected):raise ValueError('Full selected source DINO coverage incomplete')
    return {key:dict(dtype=specs[key][0],shape=[count,*specs[key][1]],rows=count,row_bytes_sha256=digests[key].hexdigest()) for key in columns}


def table_to_arrays(table):
    arrays={}
    for key in table.column_names:
        c=table[key].combine_chunks()
        if c.null_count or (pa.types.is_fixed_size_list(c.type) and c.values.null_count):raise ValueError('Null full source/owned vision values')
        arrays[key]=(c.values.to_numpy(zero_copy_only=False).reshape(len(c),c.type.list_size) if pa.types.is_fixed_size_list(c.type) else c.to_numpy(zero_copy_only=False))
    return arrays


def audit_vision(root,arrays,record,row,receipt):
    source=row['source_path'] if row['source']=='egosuite' else row['vision_source_path'];owned=owned_path(root,record['vision_path'])
    proof=receipt['vision'];check_identity(source,proof['source_identity'])
    if proof['source_path']!=str(Path(source).resolve()) or proof.get('dense_body_columns_copied') is not False:raise ValueError('Owned DINO source/physical projection differs')
    extraction=row['source_record']['dino'] if row['source']=='egosuite' else read_json(Path(source).with_suffix('.json'))
    extraction_sha=(hashlib.sha256((json.dumps(extraction,sort_keys=True,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()).hexdigest()
                    if row['source']=='egosuite' else sha_file(Path(source).with_suffix('.json')))
    dino=record['dino'];tokens=['cls','register_0','register_1','register_2','register_3']
    if (dino.get('source_extraction')!=extraction or dino.get('source_extraction_metadata_sha256')!=extraction_sha
            or dino.get('source_signature')!=extraction.get('signature') or dino.get('encoder')!=extraction.get('encoder')
            or dino.get('token_order')!=tokens or dino.get('flatten_order')!='token_major'):
        raise ValueError('Standalone DINO extraction/weights/geometry/clock provenance changed')
    encoder=extraction['encoder']
    if (extraction.get('state')!='complete' or encoder.get('shape')!=[5,1024] or encoder.get('flat_shape')!=[5120]
            or encoder.get('token_order')!=tokens or encoder.get('flatten_order')!='token_major' or encoder.get('storage_dtype')!='float32'):
        raise ValueError('Original DINO encoder token/dtype contract differs')
    n=len(arrays['timestamp']);selected=np.asarray(arrays['source.video_frame_index'])
    if not np.array_equal(selected,np.arange(n,dtype=np.int64)):raise ValueError('Source video mapping cannot be replaced by implicit row prefix')
    preserved=['observation.features.dino','observation.valid.dino','observation.video_timestamp','observation.video_time_error']
    before=selected_stream_digest(source,preserved+['frame_index'],selected,casts={'observation.video_time_error':np.float64})
    after=streaming_digest(owned,pq.ParquetFile(owned).schema_arrow.names)
    required={'frame_index','episode_index','timestamp','observation.source_timestamp','source.video_frame_index',*preserved}
    if set(after)!=required:raise ValueError('Owned DINO must contain only independent projected vision/join columns')
    for key in preserved:
        if before[key]['shape']!=after[key]['shape'] or before[key]['row_bytes_sha256']!=after[key]['row_bytes_sha256']:raise ValueError('Every original DINO/token/mask/PTS row not exactly preserved: '+key)
    for key in required-{'observation.features.dino'}:
        expected=np.asarray(arrays[key]);expected=expected.astype(np.dtype(after[key]['dtype']))
        if after[key]['row_bytes_sha256']!=hashlib.sha256(np.ascontiguousarray(expected).tobytes()).hexdigest():raise ValueError('Full owned DINO-motion frame/timestamp join differs: '+key)
    if before['frame_index']['row_bytes_sha256']!=hashlib.sha256(selected.tobytes()).hexdigest():raise ValueError('Original source DINO explicit frame IDs differ')
    if proof['rows']!=n or proof['sha256']!=record['vision_sha256'] or sha_file(owned)!=proof['sha256']:raise ValueError('Complete owned DINO byte receipt differs')
    for key,value in after.items():
        if proof['column_byte_sha256'].get(key)!=value['row_bytes_sha256']:raise ValueError('Owned vision full numeric digest differs from write receipt')
    return dict(rows=n,all_source_token_values_preserved=True,shape=[5,1024],owned_projected_columns=True,
                source_selected_numeric_digest=before,owned_numeric_digest=after)


def window_ground(arrays,metadata,prefix):
    g=deepcopy(metadata['ground'])
    if not metadata.get('coordinate_verified'):g.update(valid=False,verified=False);return g
    if g.get('valid'):return g
    policy=g.get('window_fallback',{'mode':'mean_valid_feet','offset_m':.02})
    if g.get('mode')=='estimate' and policy['mode']=='mean_valid_feet':
        feet=arrays['source.feet_world'][:prefix].reshape(prefix,2,3);valid=arrays['source.feet_valid'][:prefix]&np.isfinite(feet).all(-1)
        if valid.any():g.update(height_m=float(feet[...,2][valid].mean()-policy['offset_m']),valid=True,verified=False)
    return g


def audit_window(window,original,metadata,model,selected,*,canonicalized,task,forecast_prefix):
    """Independently check returned 15Hz FK, reference and 18D trajectory."""
    a=window.arrays;n=len(selected);source={k:v[selected] for k,v in original.items()};expected=source.copy()
    pv=source['estimated.valid.body_position'];rv=source['estimated.valid.body_rotation']
    fk=numpy_fk(model,source);P=fk['positions'][:,:22];R=fk['global_rotations'][:,:22]
    A=np.eye(3);t=np.zeros(3)
    if canonicalized:
        r=quat_matrix(source['observation.state.head_reference_world_rotation'])[0];xy=r[:2,0];xy=xy/np.linalg.norm(xy)
        A=np.array([[xy[0],xy[1],0],[-xy[1],xy[0],0],[0,0,1.]])
        origin=source['observation.state.head_reference_world'][0].copy();origin[2]=0;t=-A@origin
    P=P@A.T+t;R=A@R
    close(a['observation.state.body_world'].reshape(n,22,3),np.where(pv[...,None],P,0),name='Fresh reader independently decoded body XYZ')
    close(quat_matrix(a['observation.state.body_world_rotation'].reshape(n,22,4)),R,rv,name='Fresh reader independent full body global rotations')
    world=a['observation.state.body_world_rot6d_xyz'].reshape(n,22,9)
    close(world[...,6:],np.where(pv[...,None],P,0),name='Fresh derived world XYZ cache')
    close(world[...,:6],np.where(rv[...,None],R[...,:2,:].reshape(n,22,6),0),name='Fresh independent row6D cache')
    returned_fk=numpy_fk(model,a)
    close(returned_fk['positions'][:,:22],P,name='Canonical returned parameters must decode in returned world')
    close(returned_fk['global_rotations'][:,:22],R,name='Canonical root local AA must left-multiply world transform')
    for key in ('smplh.pose_body','smplh.betas'):
        exact(a[key],source[key],name='Canonical unchanged local shape/body pose')
    for component,valid in [('position',pv),('rotation',rv)]:
        for prefix in ('source.observed','source.interpolated','estimated.completed'):
            key=prefix+'.body_'+component;exact(a[key],source[key],name='Fresh independent observation/completion mask')
        exact(a['observation.valid.body_'+component],valid,name='Fresh estimate masks')
    headpos=source['observation.state.head_reference_world']@A.T+t
    headrot=A@quat_matrix(source['observation.state.head_reference_world_rotation'])
    hp=scalar(source['observation.valid.head_reference_position']);hr=scalar(source['observation.valid.head_reference_rotation'])
    close(a['observation.state.head_reference_world'],np.where(hp[:,None],headpos,0) if canonicalized else source['observation.state.head_reference_world'],name='Fresh original camera head-reference position')
    close(quat_matrix(a['observation.state.head_reference_world_rotation']),headrot,hr,name='Fresh original camera reference rotation')
    g=window_ground(source,metadata,forecast_prefix if task=='fore' else n)
    if bool(window.metadata['ground']['valid'])!=bool(g['valid']):raise ValueError('Fresh native-foot ground fallback mask differs')
    close(window.metadata['ground']['height_m'],g['height_m'],name='Fresh prefix-only original-foot ground fallback',tolerance=2e-6)
    horizontal=headrot[:,:2,0];length=np.linalg.norm(horizontal,axis=-1);heading=np.zeros_like(horizontal)
    np.divide(horizontal,length[:,None],out=heading,where=length[:,None]>1e-6)
    floorR=np.broadcast_to(np.eye(3),(n,3,3)).copy();floorR[:,0,0]=heading[:,0];floorR[:,1,0]=heading[:,1]
    floorR[:,0,1]=-heading[:,1];floorR[:,1,1]=heading[:,0];floorR[length<=1e-6]=np.eye(3)
    floorvalid=hp&hr&(length>1e-6)&bool(g['valid'])&bool(metadata.get('head_reference',{}).get('basis_verified'))
    origin=headpos.copy();origin[:,2]=g['height_m'];origin=np.where(floorvalid[:,None],origin,0)
    floorR=np.where(floorvalid[:,None,None],floorR,np.eye(3))
    close(a['observation.state.head_reference_floor_world'],origin,name='Fresh camera-ground origin')
    close(quat_matrix(a['observation.state.head_reference_floor_world_rotation']),floorR,name='Fresh yaw-only ground frame')
    lp=np.einsum('tji,tkj->tki',floorR,P-origin[:,None]);lr=np.swapaxes(floorR,-1,-2)[:,None]@R
    f=a['observation.state.body_floor_rot6d_xyz'].reshape(n,22,9)
    close(f[...,6:],np.where((pv&floorvalid[:,None])[...,None],lp,0),name='Fresh floor XYZ')
    close(f[...,:6],np.where((rv&floorvalid[:,None])[...,None],lr[...,:2,:].reshape(n,22,6),0),name='Fresh floor global row6D')
    close(np.einsum('tij,tkj->tki',floorR,f[...,6:])+origin[:,None],P,pv&floorvalid[:,None],name='Fresh floor inverse back to world')
    for side in ('left','right'):
        hand=source['observation.state.hand_'+side+'_world'].reshape(n,21,3);valid=source['observation.valid.hand_'+side+'_position']
        hand=hand@A.T+t
        close(a['observation.state.hand_'+side+'_world'].reshape(n,21,3),hand,valid,name='Fresh preserved explicit hand21')
        local=np.einsum('tji,tkj->tki',floorR,hand-origin[:,None]);fv=valid&floorvalid[:,None]
        close(a['observation.state.hand_'+side+'_floor'].reshape(n,21,3),np.where(fv[...,None],local,0),name='Fresh explicit hands floor')
    localR=np.swapaxes(floorR,-1,-2)@headrot;localp=np.einsum('tji,tj->ti',floorR,headpos-origin)
    plane=origin.copy();plane[:,2]-=g['height_m'];deltaR=floorR.copy();deltap=plane.copy()
    deltaR[1:]=np.swapaxes(floorR[:-1],-1,-2)@floorR[1:];deltap[1:]=np.einsum('tji,tj->ti',floorR[:-1],np.diff(plane,axis=0))
    dv=floorvalid.copy();dv[1:]&=floorvalid[:-1]
    trajectory=np.concatenate((localR[:,:2].reshape(n,6),localp,deltaR[:,:2].reshape(n,6),deltap),-1)
    tm=np.concatenate((np.repeat(floorvalid[:,None],9,1),np.repeat(dv[:,None],9,1)),-1)
    exact(window.trajectory_valid,tm,name='Fresh 18D conditioning masks')
    close(window.trajectory,np.where(tm,trajectory,0),name='Fresh EE4D 18D pose/delta trajectory')
    if window.metadata['training_window'].get('requires_old_dense_archive') is not False:raise ValueError('Fresh standalone reader dependence not explicitly false')
    if window.metadata.get('label_kind')!='model_fitted_estimate' or window.metadata.get('absolute_rotation_accuracy_established') is not False:raise ValueError('Fresh FK estimates upgraded to motion truth')
    return dict(frames=n,canonicalized=canonicalized,task=task,independent_fk_and_floor_and_trajectory=True)


def install_old_path_guard(root,blocked_roots):
    """Block Python opens/stat/scandir and Arrow C++ path entrypoints.

    This is an interface dependency test, not an adversarial OS sandbox. Owned
    paths are allowed beneath a blocked parent; ancestor stat is allowed solely
    so Path.resolve can locate that owned child. Opening/scanning old data is
    denied, including through resolved symlinks caught by the reader itself.
    """
    allowed=os.path.abspath(os.fspath(root));blocked=[os.path.abspath(os.fspath(p)) for p in blocked_roots];denied=[]
    def check(path,kind='open'):
        if isinstance(path,(int,type(None))):return
        try:value=os.path.abspath(os.fsdecode(path))
        except TypeError:return
        if value==allowed or value.startswith(allowed+os.sep):return
        if kind=='stat' and (allowed.startswith(value+os.sep) or value==os.path.sep):return
        if any(value==p or value.startswith(p+os.sep) for p in blocked):
            denied.append({'operation':kind,'path':value});raise PermissionError('Fresh-reader old dataset path blocked: '+value)
    def hook(event,args):
        if event=='open':check(args[0])
        elif event=='os.scandir':check(args[0],'scandir')
    sys.addaudithook(hook)
    for name in ('stat','lstat','scandir','open'):
        original=getattr(os,name)
        def wrapper(path,*args,_original=original,_kind=('stat' if name in ('stat','lstat') else name),**kwargs):
            check(path,_kind);return _original(path,*args,**kwargs)
        setattr(os,name,wrapper)
    for name in ('ParquetFile','read_table'):
        original=getattr(pq,name)
        def wrapper(path,*args,_original=original,**kwargs):check(path,'Arrow C++ path');return _original(path,*args,**kwargs)
        setattr(pq,name,wrapper)
    return denied


def fresh_reader_probe(root,blocked_roots,*,allow_test_fixture=False):
    root=Path(root).resolve();info=json.loads((root/'meta/info.json').read_text());model=load_model(owned_path(root,info['model']['path']),info['model']['model_sha256'])
    if not allow_test_fixture and model['sha256']!=OFFICIAL_MODEL_SHA256:raise ValueError('Fresh reader requires official immutable model')
    denied=install_old_path_guard(root,blocked_roots)
    from egomotion_pipeline.smplh_parameters import SMPLHParameterArchive
    archive=SMPLHParameterArchive(root,allow_test_fixture=allow_test_fixture);records=archive.records();checked=[]
    for record in records:
        path=owned_path(root,record.get('data_path',f"data/chunk-{record['episode_index']//1000:03d}/file-{record['episode_index']%1000:03d}.parquet"))
        original=read_arrays(path,[k for k in pq.ParquetFile(path).schema_arrow.names if k!='observation.features.dino'])
        length=record['length']
        if length<3:raise ValueError('Cannot certify a 15Hz two-frame window for a <3 frame source episode')
        starts=sorted({0,((length-3)//2)*2});probes=[]
        for start in starts:
            selected=start+np.arange(2)*2
            window=archive.read_window(record['episode_index'],start_frame=start,num_frames=2,fps=15,canonicalize=False,load_dino=True)
            probes.append(audit_window(window,original,record['metadata'],model,selected,canonicalized=False,task='recon',forecast_prefix=1))
            if np.asarray(window.arrays['observation.features.dino']).shape!=(2,5120):raise ValueError('Fresh standalone owned DINO join not 5x1024')
        window=archive.read_window(record['episode_index'],start_frame=0,num_frames=2,fps=15,canonicalize=False,task='fore',forecast_prefix=1,load_dino=False)
        probes.append(audit_window(window,original,record['metadata'],model,np.array([0,2]),canonicalized=False,task='fore',forecast_prefix=1))
        meta=record['metadata'];reference=np.array([original['observation.valid.head_reference_position'][0],original['observation.valid.head_reference_rotation'][0]]).all()
        heading=np.linalg.norm(quat_matrix(original['observation.state.head_reference_world_rotation'][0])[:2,0])
        quarantined=bool(meta.get('training_eligible') is False or meta.get('canonical_training_eligible') is False or meta.get('quarantine_reasons'))
        canonical=bool(reference and meta.get('coordinate_verified') and meta.get('head_reference',{}).get('basis_verified') and heading>1e-6 and not quarantined)
        if canonical:
            window=archive.read_window(record['episode_index'],start_frame=0,num_frames=2,fps=15,canonicalize=True,load_dino=False)
            probes.append(audit_window(window,original,meta,model,np.array([0,2]),canonicalized=True,task='recon',forecast_prefix=1))
        if quarantined:
            try:archive.read_window(record['episode_index'],start_frame=0,num_frames=2,fps=15,canonicalize=True,load_dino=False)
            except ValueError:pass
            else:raise ValueError('Quarantined source must reject canonical training windows')
        checked.append(dict(episode_index=record['episode_index'],source_key=record.get('source_key'),length=record['length'],split=record.get('split'),probes=probes,canonical_available=canonical))
    if denied:raise ValueError('Fresh reader attempted old source access')
    return dict(passed=True,fresh_process=True,episodes=len(checked),probes=sum(len(r['probes']) for r in checked),all_population_interfaces_checked=True,
                guard='Python audit open + stat/lstat/scandir + Arrow ParquetFile/read_table path denial; owned subtree permitted',
                blocked_roots=[str(p) for p in blocked_roots],old_source_open_attempts=denied,rows=checked,
                coverage='Every episode first/last15Hz two-frame, forecastprefix1, plus valid canonical first window; all parameters/source rows audited separately',
                training_throughput_certified=False,motion_truth_accuracy_established=False)


def audit_episode(root,record,row,model,hand_basis):
    assert_source_record(record,row);path=owned_path(root,record['data_path']);vision=owned_path(root,record['vision_path'])
    receipt_path=owned_path(root,str(Path(record['data_path']).parent/'receipt.json'));receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema_version')!=PARAMETER_SCHEMA_VERSION or receipt.get('record')!=record or receipt.get('source_provenance')!=record['source_provenance']:raise ValueError('Atomic episode receipt/provenance differs')
    if receipt['motion']['sha256']!=record['motion_sha256'] or sha_file(path)!=record['motion_sha256']:raise ValueError('Complete owned motion byte SHA differs')
    parquet=pq.ParquetFile(path)
    if parquet.metadata.num_rows!=row['length'] or parquet.schema_arrow.metadata.get(b'parameter_schema_version')!=PARAMETER_SCHEMA_VERSION.encode():raise ValueError('Owned physical row count/schema metadata differs')
    arrays=read_arrays(path);validate_physical(arrays,record)
    if row['source']=='egosuite':body=audit_ego(arrays,record,row,model)
    elif row['source']=='egoexo4d_hm':body=audit_hm(arrays,record,row,model,hand_basis)
    else:raise ValueError('Unknown original source namespace')
    # Every modality/parameter row is independently checked above, while tokens
    # are streamed instead of materializing a multi-GiB longest episode.
    vision_report=audit_vision(root,arrays,record,row,receipt)
    check_identity(record['source_provenance']['source_path'],record['source_provenance']['source_identity'])
    fk=numpy_fk(model,arrays);hand_stats={}
    for side,joint in [('left',20),('right',21)]:
        hv=arrays['observation.valid.hand_'+side+'_position'][:,0]&arrays['estimated.valid.body_position'][:,joint]
        error=np.linalg.norm(arrays['observation.state.hand_'+side+'_world'].reshape(-1,21,3)[:,0]-fk['positions'][:,joint],axis=-1)[hv]
        hand_stats[side]=dict(paired_points=len(error),mean_m=float(error.mean()) if len(error) else None,p95_m=float(np.quantile(error,.95)) if len(error) else None,
                              interpretation='native explicit wrist versus shared-model FK wrist; not hand truth or anatomical center certification')
    return dict(passed=True,episode_index=record['episode_index'],source=record['source'],source_key=record['source_key'],frames=row['length'],split=row['split'],
                body=body,vision=vision_report,native_hand_wrist_residual=hand_stats,receipt_sha256=sha_file(receipt_path),
                native_axis_certified=False,absolute_rotation_accuracy_established=False)


def cleanup_dry_run(plan,rows,root,cleanup_plan=None):
    """Produce an exact dense-cache file list. This function never deletes."""
    owned=Path(root).resolve();old=Path(plan['egosuite_archive']).resolve();fit=Path(plan['egosuite_fitted_root']).resolve()
    preserved=[owned,Path(plan['model_path']).resolve(),Path(plan['hm_root']).resolve(),old/'weights',old/'.pipeline/jobs']
    candidates=[]
    for row in rows:
        if row['source']=='egosuite':
            for role,key in [('old_v5_motion_and_owned_copy_of_dino','source_path'),('derived_dense22_fitted_motion','fitted_path')]:
                path=Path(row[key]).resolve()
                if not (path.is_relative_to(old/'data') or path.is_relative_to(fit/'episodes')) or path.is_relative_to(owned):raise ValueError('Dense cleanup candidate escapes authorized source cache')
                candidates.append(dict(path=str(path),role=role,source_key=row['source_key'],identity=file_identity(path),bytes=path.stat().st_size))
    if cleanup_plan is not None:
        # Root may register directories as well as files. They must be strict
        # descendants of the old data/fitted cache, never a parent dataset root.
        for value in cleanup_plan.get('paths',cleanup_plan.get('delete_paths',[])):
            value=value['path'] if isinstance(value,dict) else value;p=Path(value).resolve()
            allowed=(p==old/'data' or p.is_relative_to(old/'data') or p==fit or p.is_relative_to(fit))
            if not allowed or p==owned or p.is_relative_to(owned) or any(x==p or x.is_relative_to(p) for x in preserved):raise ValueError('Registered cleanup scope includes preserved originals/weights/evidence/new archive')
    return dict(dry_run=True,deletion_executed=False,cleanup_requires_full_acceptance=True,exact_dense_file_count=len(candidates),
                bytes=sum(c['bytes'] for c in candidates),paths=candidates,preserved_roots=[str(p) for p in preserved],
                raw_egosuite_mcap_preserved=True,original_hm_npz_camera_video_dino_preserved=True,weights_and_evidence_preserved=True,
                source_record_and_receipt_provenance_must_be_retained=True)


def read_json(path):return json.loads(Path(path).read_text())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def check_original_full_fit_gate(plan):
    root=Path(plan['egosuite_fitting_job']);gate_path=root/'reports/full-acceptance.json';audit_path=root/'reports/full-sidecar-audit.json'
    if not gate_path.exists() or not audit_path.exists():return False,{'reason':'Original full acceptance and independent audit both required'}
    gate=read_json(gate_path);report=read_json(audit_path);status=read_json(root/'status.json')
    passed=bool(gate.get('passed') and gate.get('scope')=='full' and gate.get('episodes')==6614 and gate.get('frames')==26841893
       and gate.get('full_candidate_quality_passed') and not gate.get('failures') and gate.get('job_signature')==status.get('job_signature')
       and gate.get('independent_audit_sha256')==sha_file(audit_path) and report.get('passed') and report.get('episodes_checked')==6614
       and report.get('frames_checked')==26841893 and not report.get('selection_sha256')
       and report.get('sidecar_root')==str(Path(plan['egosuite_fitted_root']).resolve()))
    if passed:
        results=report.get('results',[])
        passed=bool(len(results)==6614 and {r['episode_index'] for r in results}==set(range(6614)) and all(r.get('passed') for r in results))
    return passed,dict(original_full_fit_acceptance_sha256=sha_file(gate_path),original_full_fit_audit_sha256=sha_file(audit_path),
                       original_full_fit_signature=status.get('job_signature'),full_source_population_verified=passed)


def full_audit(job,output,*,root=None,workers=1,limit=None,skip_fresh_reader=False):
    started=time.monotonic();job=Path(job).resolve();plan=read_json(job/'plan.json');root=Path(root or plan['output_root']).resolve()
    combined_path=job/'inventory/combined.json'
    if not combined_path.exists():raise ValueError('Frozen combined full original source population is required')
    inventory=read_json(combined_path);rows=inventory['rows'];records=[read_json(p) for p in sorted((root/'.pipeline/episodes').glob('*.json'))]
    for original in inventory.get('source_inventory_identities',{}).values():
        check_identity(original['path'],original)
        if sha_file(original['path'])!=original['sha256']:raise ValueError('Original source inventory/take/split identity changed')
    population=population_records(records,rows)
    if any(r['episode_index']!=i for i,r in enumerate(records)):raise ValueError('Owned global episode IDs are not contiguous')
    if any(rows[i]['dataset_to_index']!=rows[i+1]['dataset_from_index'] for i in range(len(rows)-1)):raise ValueError('Frozen full source global frame intervals overlap or gap')
    split_path=root/'meta/splits.json'
    if not split_path.exists():raise ValueError('Owned train/val/test split manifest required')
    splits=read_json(split_path);expected_splits={name:[r['source_key'] for r in records if r['split']==name] for name in ('train','val','test')}
    if splits.get('splits')!=expected_splits:raise ValueError('Owned split manifest does not match every original source membership')
    expected_groups={g:r['split'] for r in records for g in r.get('split_group_ids',[])}
    if splits.get('split_group_ids')!=expected_groups:raise ValueError('Owned split group manifest differs')
    episodes_path=owned_path(root,'meta/episodes.parquet');episode_table=read_arrays(episodes_path)
    for key in ('episode_index','source','source_key','source_uid','length','dataset_from_index','dataset_to_index','split','data_path','vision_path'):
        if key not in episode_table or list(episode_table[key])!=[r[key] for r in records]:raise ValueError('Owned episode lookup index differs: '+key)
    info=read_json(root/'meta/info.json')
    if info.get('schema_version')!=PARAMETER_SCHEMA_VERSION or info.get('fps')!=30 or info.get('requires_old_dense_archive') is not False:raise ValueError('Standalone native30Hz dataset ownership contract differs')
    if info['model'].get('model_sha256')!=OFFICIAL_MODEL_SHA256:raise ValueError('Only the immutable actual16 male SMPL-H asset can pass production acceptance')
    model=load_model(owned_path(root,info['model']['path']),OFFICIAL_MODEL_SHA256)
    if not info.get('hand_basis'):raise ValueError('Owned HM hand PCA basis and means required')
    hand_basis=dict(path=str(owned_path(root,info['hand_basis']['path'])),sha256=info['hand_basis']['sha256'])
    hand_pca_to_axis_angle(np.zeros((1,90)),hand_basis['path'],hand_basis['sha256'])
    original_hand_basis=Path(info['hand_basis']['source_asset'])
    with np.load(hand_basis['path'],allow_pickle=False) as target,np.load(original_hand_basis,allow_pickle=False) as original:
        for key in ('hands_componentsl','hands_componentsr','hands_meanl','hands_meanr'):exact(target[key],original[key],name='Owned official-HM hand basis and mean provenance')
    if info.get('dino',{}).get('checkpoint')!='facebook/dinov2-with-registers-large' or info['dino'].get('shape')!=[5,1024]:raise ValueError('Owned DINO checkpoint/token layout differs')
    # Validate source-specific inventories agree with the combined membership,
    # so a silently reduced combined list cannot stand in for the original set.
    original=[]
    for source in ('egosuite','egoexo4d_hm'):
        original.extend(read_json(job/'inventory'/f'{source}.json')['rows'])
    population_records(rows,original)
    hm_inventory=read_json(job/'inventory/hm.json')
    if {r['source_key'].split(':',1)[-1] for r in rows if r['source']=='egoexo4d_hm'}!={i['take_name'] for i in hm_inventory['items']}:raise ValueError('Original downloaded HM take population is incompletely represented')
    if population['sources']['egosuite']!={'episodes':6614,'frames':26841893} or population['sources']['egoexo4d_hm']!={'episodes':2649,'frames':11295991}:raise ValueError('Expected original full6614+2649 population differs from registered migration task')
    references=read_json(job/'references/hm/manifest.json')
    # Evidence filenames are local stable copies; require every entry's actual
    # SHA rather than accepting metadata's verified boolean.
    reference_items=references.get('files',references.get('items',[])) if isinstance(references,dict) else references
    for item in reference_items:
        value=item.get('local_path',item.get('saved_path',item.get('file')))
        if value:
            path=Path(value);path=path if path.is_absolute() else job/'references/hm'/path
            if sha_file(path)!=item['sha256']:raise ValueError('Pinned official local source evidence changed')
    bound=dict(plan_sha256=sha_file(job/'plan.json'),combined_inventory_sha256=sha_file(combined_path),
               model_sha256=model['sha256'],auditor_sha256=sha_file(__file__),reader_sha256=sha_file(Path(__file__).parents[1]/'src/egomotion_pipeline/smplh_parameters.py'),
               info_sha256=sha_file(root/'meta/info.json'),records_manifest_sha256=hashlib.sha256(canonical(records)).hexdigest(),
               episodes_metadata_sha256=sha_file(episodes_path),
               hand_basis_sha256=hand_basis['sha256'],original_hand_basis_asset_sha256=sha_file(original_hand_basis),
               split_metadata_sha256={str(p.relative_to(root)):sha_file(p) for p in sorted((root/'meta').glob('splits*.json'))})
    selected=records if limit is None else records[:limit];byindex={r['episode_index']:r for r in rows};reports=[];failures=[]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures={executor.submit(audit_episode,str(root),record,byindex[record['episode_index']],model,hand_basis):record for record in selected}
        for future in as_completed(futures):
            record=futures[future]
            try:reports.append(future.result())
            except Exception as error:failures.append(dict(episode_index=record['episode_index'],source_key=record['source_key'],error=repr(error)))
            if (len(reports)+len(failures))%100==0:print(json.dumps(dict(event='parameter_audit',checked=len(reports)+len(failures),expected=len(selected),failures=len(failures))),flush=True)
    probe=None
    if not failures and limit is None and not skip_fresh_reader:
        probe_path=Path(output).with_name('standalone-fresh-reader.json');command=[sys.executable,str(Path(__file__).resolve()),'--reader-probe','--root',str(root),
             '--blocked-roots',json.dumps([plan['egosuite_archive'],plan['egosuite_fitted_root'],plan['hm_root']]),'--output',str(probe_path)]
        result=subprocess.run(command,capture_output=True,text=True)
        if result.returncode:failures.append(dict(stage='fresh_reader_independence',error=result.stderr[-8000:] or result.stdout[-8000:]))
        else:
            probe=read_json(probe_path)
            bound['fresh_reader_sha256']=sha_file(probe_path)
            if not probe.get('passed') or probe.get('episodes')!=len(rows):failures.append(dict(stage='fresh_reader_independence',error='Not every original episode passed the blocked-old-path reader'))
    cleanup_plan=read_json(job/'cleanup-plan.json') if (job/'cleanup-plan.json').exists() else None
    dryrun=cleanup_dry_run(plan,rows,root,cleanup_plan);dryrun_path=Path(output).with_name('dense-cleanup-dry-run.json');write_json(dryrun_path,dryrun)
    complete=len(reports)==len(rows) and limit is None and probe is not None and not failures
    # Fitting numerical quality has a separate preregistered full-population
    # gate; parameter roundtrip integrity cannot replace it.
    fit_passed,fit_binding=check_original_full_fit_gate(plan);bound.update(fit_binding)
    if not fit_passed:failures.append(dict(stage='original_egosuite_full_fit_acceptance',error='Registered original full6614 fit acceptance absent or failed; expanded60 is insufficient'))
    quarantined=sum(r['body'].get('quarantined',False) for r in reports)
    report=dict(format='independent-standalone-smplh-parameter-full-audit-v1',passed=bool(complete and fit_passed),population=population,
                parameter_and_source_integrity_passed=bool(complete),original_egosuite_full_fit_acceptance_passed=fit_passed,
                cleanup_allowed=bool(complete and fit_passed),old_dense_dependency_absent=bool(probe and probe.get('passed')),
                quarantined_episodes=quarantined,all_episodes_training_eligible=quarantined==0,formal_motion_truth_certified=False,
                scope='full' if limit is None else 'partial_research',audited_episodes=len(reports),audited_frames=sum(r['frames'] for r in reports),
                binding=bound,failures=failures,episode_reports=sorted(reports,key=lambda r:r['episode_index']),fresh_reader_report=str(probe_path) if probe is not None else None,
                cleanup_dry_run=str(dryrun_path),deletion_executed=False,contacts='Not stored as capture truth; training15Hz must derive contacts after frame selection from model feet with estimate masks',
                completed_at=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    latest_records=[read_json(p) for p in sorted((root/'.pipeline/episodes').glob('*.json'))]
    if (sha_file(job/'plan.json')!=bound['plan_sha256'] or sha_file(combined_path)!=bound['combined_inventory_sha256']
            or sha_file(root/'meta/info.json')!=bound['info_sha256'] or hashlib.sha256(canonical(latest_records)).hexdigest()!=bound['records_manifest_sha256']
            or sha_file(__file__)!=bound['auditor_sha256'] or sha_file(Path(__file__).parents[1]/'src/egomotion_pipeline/smplh_parameters.py')!=bound['reader_sha256']
            or sha_file(episodes_path)!=bound['episodes_metadata_sha256'] or any(sha_file(root/p)!=s for p,s in bound['split_metadata_sha256'].items())):
        report['passed']=False;report['cleanup_allowed']=False;report['failures'].append({'stage':'frozen_inputs','error':'Plan/full population/owned records/schema/code/splits changed during audit'})
    write_json(output,report);return report


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--job');parser.add_argument('--root');parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=1);parser.add_argument('--limit',type=int);parser.add_argument('--skip-fresh-reader',action='store_true')
    parser.add_argument('--reader-probe',action='store_true');parser.add_argument('--blocked-roots',default='[]');parser.add_argument('--allow-test-fixture',action='store_true')
    args=parser.parse_args();sys.path.insert(0,str(Path(__file__).parents[1]/'src'))
    if args.reader_probe:
        if not args.root:parser.error('--reader-probe requires --root')
        # Open output before the source-path guard; the reader cannot acquire any
        # old path, but this already-open descriptor can record evidence there.
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('w') as handle:
            result=fresh_reader_probe(args.root,json.loads(args.blocked_roots),allow_test_fixture=args.allow_test_fixture)
            handle.write(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:result[k] for k in ('passed','episodes','probes')}));return
    if args.allow_test_fixture:parser.error('Synthetic assets cannot pass production full audit')
    if not args.job or args.workers<1 or (args.limit is not None and args.limit<1):parser.error('Use --job and positive worker/optional research limit counts')
    result=full_audit(args.job,args.output,root=args.root,workers=args.workers,limit=args.limit,skip_fresh_reader=args.skip_fresh_reader)
    print(json.dumps({k:result[k] for k in ('passed','scope','audited_episodes','audited_frames','cleanup_allowed','elapsed_seconds')}))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
