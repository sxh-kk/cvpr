"""Independent auditor detects consistently re-signed numerical corruption."""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline.schema import SCHEMA_VERSION, feature_specs_for_schema
from egomotion_pipeline import smplh_sidecar as writer

spec=spec_from_file_location('independent_smplh_audit',Path(__file__).parents[1]/'scripts/audit_smplh_sidecars.py')
audit=module_from_spec(spec);spec.loader.exec_module(audit)

def write_arrays(path,arrays,metadata=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    columns={k:pa.array(x) if x.ndim==1 else pa.FixedSizeListArray.from_arrays(pa.array(x.reshape(-1)),x.shape[1]) for k,x in arrays.items()}
    table=pa.table(columns)
    if metadata is not None:table=table.replace_schema_metadata(metadata)
    pq.write_table(table,path)

@pytest.fixture
def exported(tmp_path):
    T=5;root=tmp_path/'source';src=root/'data/chunk-000/file-000.parquet'
    parents=audit.PARENTS.tolist()
    for wrist in [20,21]:
        for finger in range(5):
            first=len(parents);parents.extend([wrist,first,first+1])
    parents=np.array(parents)
    rest=np.zeros((52,3));rest[0]=[.17,-.08,.24]
    for j in range(1,52):rest[j]=rest[parents[j]]+[.001*j,.01,.025]
    template=np.zeros((6890,3));template[:52]=rest
    reg=np.zeros((52,6890));reg[:, :52]=np.eye(52)
    directions=np.zeros((6890,3,16));directions[:52,0,:]=np.arange(52)[:,None]*np.arange(1,17)[None]*.0001
    model=tmp_path/'model.npz'
    np.savez(model,v_template=template,J_regressor=reg,shapedirs=directions,kintree_table=np.array([parents,np.arange(52)]))
    beta=np.arange(16)*.013;aa=np.zeros((T,22,3));aa[:,0,2]=np.arange(T)*.15;aa[:,16,0]=.4;aa[:,1,1]=-.22
    trans=np.zeros((T,3));trans[:,0]=np.arange(T)*.031;trans[:,2]=1.2
    pos,R,local,J0=audit.numpy_fk(rest,np.einsum('jv,vcs->jcs',reg,directions),parents,beta,aa,trans)
    source={}
    specs=feature_specs_for_schema(SCHEMA_VERSION)
    for key in writer.SOURCE_COLUMNS:
        shape=(T,) if specs[key]['shape']==[1] else (T,specs[key]['shape'][0])
        source[key]=np.zeros(shape,dtype=specs[key]['dtype'])
        if '.valid.' in key:source[key][...]=True
        if key.endswith('world_rotation'):source[key].reshape(T,-1,4)[...,0]=1
    source['frame_index']=np.arange(T,dtype=np.int64);source['index']=np.arange(T,dtype=np.int64)
    source['timestamp']=(np.arange(T)/30).astype(np.float32)
    source['observation.source_timestamp']=np.arange(T)/30+12.
    source['observation.source_frame_left']=np.arange(T,dtype=np.int64)
    source['observation.source_frame_right']=np.arange(T,dtype=np.int64)
    d=np.zeros((22,3));d[12]=[.023,-.006,.011];anchors=np.arange(22);anchors[12]=9
    reconstruction=pos+np.einsum('tjik,jk->tji',R[:,anchors],d)
    source['observation.state.body_world']=reconstruction.reshape(T,66).astype(np.float32)
    for side,joint in [('left',20),('right',21)]:
        hand=np.broadcast_to(reconstruction[:,joint,None],(T,21,3)).copy()
        source[f'observation.state.hand_{side}_world']=hand.reshape(T,63).astype(np.float32)
        source[f'observation.state.hand_{side}_floor']=hand.reshape(T,63).astype(np.float32)
    source['observation.state.head_reference_floor_world'][:]=[.2,-.1,0]
    source['observation.valid.body_position'][1,3]=False
    source['observation.valid.body_rotation'][2,5]=False
    write_arrays(src,source)
    record={'episode_index':0,'source':'egosuite','source_key':'egosuite:test','length':T,'metadata':{'operator_id':'test-operator','ground':{'valid':True,'height_m':0.,'verified':False}}}
    rp=root/'.pipeline/episodes/00000000.json';rp.parent.mkdir(parents=True);rp.write_text(json.dumps(record))
    center={'offsets':d.tolist(),'anchors':anchors.tolist(),'offset_valid':(np.linalg.norm(d,axis=-1)>0).tolist(),'model_sha256':audit.file_sha(model),'betas':beta.tolist()}
    fit={'body_world':pos,'global_rotations':R,'local_axis_angle':aa,'root_world':pos[:,0],'smpl_trans':trans,'betas':beta,
         'estimated_position_valid':np.ones((T,22),bool),'estimated_rotation_valid':np.ones((T,22),bool),
         'fit_quality':np.full((T,22),.7),'reconstructed_observations':reconstruction,
         'source_position_valid':source['observation.valid.body_position'].copy(),'source_rotation_valid':source['observation.valid.body_rotation'].copy()}
    contract=writer.build_sidecar_contract(root,src,model,{'fixture':True},{'center':center,'source_id':'egosuite','group_id':'test-operator'},source)
    out=tmp_path/'sidecar';writer.write_episode_sidecar(out,record,source,fit,contract)
    return out/'episodes/00000000'

def resign(folder,arrays,metadata):
    p=folder/'motion.parquet';write_arrays(p,arrays,metadata)
    receipt=json.loads((folder/'receipt.json').read_text())
    receipt['parquet_sha256']=audit.file_sha(p);receipt['numeric_sha256']=audit.numeric_sha(arrays)
    (folder/'receipt.json').write_text(json.dumps(receipt))

def test_independent_FK_floor_source_preservation_and_root_pivot(exported):
    r=audit.audit_episode(exported,{})
    assert r['passed'],r['failures']
    assert r['frames']==5
    assert r['metrics']['root_pivot_max_abs_error']<1e-6
    assert r['metrics']['floor_inverse_world_max_abs_error']<1e-6
    assert r['metrics']['frozen_offset_observation_residual_max_abs_error']<1e-6
    assert r['motion_truth_certified'] is False

@pytest.mark.parametrize('kind',['row6D','mask','root_pivot','offset_residual','source_hand','shape'])
def test_resigned_consistent_receipt_cannot_hide_corrupt_geometry_or_source(exported,kind):
    arrays,metadata=audit.read_table(exported/'motion.parquet')
    arrays={k:x.copy() for k,x in arrays.items()}
    if kind=='row6D':arrays['fitted.body_floor_rot6d_xyz'][0,0]+=.15
    elif kind=='mask':arrays['estimated.valid.body_floor_position'][0,0]=False
    elif kind=='root_pivot':arrays['fitted.root_world'][0,0]+=.17
    elif kind=='offset_residual':arrays['quality.observation_residual_m'][0,12]+=.03
    elif kind=='source_hand':arrays['source.observation.state.hand_left_world'][0,0]+=.02
    elif kind=='shape':arrays['fitted.betas'][2,15]+=.1
    resign(exported,arrays,metadata)
    r=audit.audit_episode(exported,{})
    assert not r['passed']
    assert r['failures']
