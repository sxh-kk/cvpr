import numpy as np
import pytest

from egomotion_pipeline.smplh_jobs import native_contact_proxy, sequence_inputs, split_memberships


def test_native_contact_does_not_bridge_missing_floor_or_time_gap():
    p = np.zeros((8, 22, 3))
    p[:, [10, 11], 2] = .01
    valid = np.ones((8, 22), bool)
    times = np.arange(8)/30
    times[4:] += 1
    floor = np.ones(8, bool)
    floor[2] = False
    contact, available = native_contact_proxy(p, valid, times, np.zeros(8), floor)
    assert not available[0].any()
    assert not available[2:5].any()
    assert not contact[2:5].any()
    valid[:, [10, 11]] = False
    contact, available = native_contact_proxy(p, valid, times, np.zeros(8), np.ones(8, bool))
    assert not contact.any() and not available.any()


def test_invalid_rotation_sentinel_and_pelvis_do_not_create_false_resets():
    count = 5
    p = np.zeros((count, 22, 3))
    p[..., 0] = 2
    p[2, 0] = 0
    q = np.zeros((count, 22, 4))
    q[..., 0] = 1
    q[:, 3] = [0, 0, 0, 1]
    q[2, 3] = [1, 0, 0, 0]
    pv = np.ones((count, 22), bool)
    rv = pv.copy()
    pv[2, 0] = False
    rv[2, 3] = False
    arrays = {
        'timestamp':np.arange(count,dtype=np.float32)/30,
        'observation.source_timestamp':np.arange(count)/30,
        'observation.state.body_world':p.reshape(count,66),
        'observation.state.body_world_rotation':q.reshape(count,88),
        'observation.valid.body_position':pv,
        'observation.valid.body_rotation':rv,
        'observation.environment.floor_height':np.zeros(count),
        'observation.valid.floor':np.ones(count,bool),
    }
    for side in ('left','right'):
        arrays[f'observation.state.hand_{side}_world']=np.zeros((count,63))
        arrays[f'observation.valid.hand_{side}_position']=np.zeros((count,21),bool)
    inputs=sequence_inputs(arrays)
    assert inputs['transition_mask'].all()
    arrays['observation.state.body_world'][4,:3]=[4,0,0]
    assert not sequence_inputs(arrays)['transition_mask'][3].any()


def test_split_operator_leakage_is_rejected(tmp_path):
    import json
    split=tmp_path/'splits.json'
    split.write_text(json.dumps({'splits':{'train':['egosuite:a'],'val':['egosuite:b'],'test':[]}}))
    records=[{'source_key':f'egosuite:{key}','metadata':{'operator_id':'same-person'}} for key in ('a','b')]
    with pytest.raises(ValueError,match='leaks'):
        split_memberships(split,records)


def test_split_original_recording_leakage_is_rejected(tmp_path):
    import json
    split=tmp_path/'splits.json'
    split.write_text(json.dumps({'splits':{'train':['egosuite:a'],'val':[],'test':['egosuite:b']}}))
    records=[{'source_key':f'egosuite:{key}','metadata':{'operator_id':key,'source_file_uid':'same-record'}} for key in ('a','b')]
    with pytest.raises(ValueError,match='leaks'):
        split_memberships(split,records)


def test_duplicate_split_membership_is_rejected(tmp_path):
    import json
    split=tmp_path/'splits.json'
    split.write_text(json.dumps({'splits':{'train':['a'],'val':['a'],'test':[]}}))
    with pytest.raises(ValueError,match='Duplicate'):
        split_memberships(split,[])


def test_edited_frozen_plan_is_rejected_before_any_external_reads():
    from egomotion_pipeline.smplh_jobs import assert_plan_current
    from egomotion_pipeline.storage import digest
    plan={'config':{'fps':30},'selection':{'calibration_records':[]}}
    plan['config_sha256']=digest(plan['config'])
    plan['selection_sha256']=digest(plan['selection'])
    plan['job_signature']=digest(plan)
    plan['config']['fps']=15
    with pytest.raises(ValueError,match='plan signature mismatch'):
        assert_plan_current(plan)
    unsigned={key:value for key,value in plan.items() if key!='job_signature'}
    plan['job_signature']=digest(unsigned)
    with pytest.raises(ValueError,match='config or selection digest mismatch'):
        assert_plan_current(plan)
