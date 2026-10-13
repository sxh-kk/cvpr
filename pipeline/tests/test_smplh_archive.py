import numpy as np

from egomotion_pipeline.smplh_archive import native_window_ground
from egomotion_pipeline.ground import resolve_window_ground
from test_smplh_sidecar import case


def _source():
    p=np.zeros((8,22,3),np.float32)
    p[:4,[10,11],2]=.12
    p[4:,[10,11],2]=.62
    return {'observation.state.body_world':p.reshape(8,66),
            'observation.valid.body_position':np.ones((8,22),bool)}


def _metadata():
    return {'coordinate_verified':True,'ground':{'height_m':0.,'valid':False,'mode':'estimate',
            'window_fallback':{'mode':'mean_valid_feet','offset_m':.02}}}


def test_forecast_native_floor_fallback_only_uses_observed_prefix():
    source=_source()
    selected=np.array([0,2,4,6])
    ground=native_window_ground(source,_metadata(),selected,task='fore',forecast_prefix=2)
    assert ground['valid']
    assert abs(ground['height_m']-.1)<1e-7
    assert ground['fallback_selected_frames']==2
    assert ground['fallback_body_source']=='native_source_feet'
    assert _metadata()['ground']['valid'] is False


def test_failed_native_prefix_cannot_be_recovered_from_fitted_or_future_feet():
    source=_source()
    source['observation.valid.body_position'][:4,[10,11]]=False
    selected=np.array([0,2,4,6])
    ground=native_window_ground(source,_metadata(),selected,task='fore',forecast_prefix=2)
    assert not ground['valid']
    # This is the second resolution done by prepare_window after injecting
    # fully available fitted feet. It must remain invalid.
    fitted=np.ones((4,22,3))
    resolved=resolve_window_ground(fitted,np.ones((4,22),bool),ground,True)
    assert not resolved['valid']


def test_source_missing_feet_never_become_ground_evidence():
    source=_source()
    source['observation.valid.body_position'][:,[10,11]]=False
    ground=native_window_ground(source,_metadata(),np.array([0,2,4,6]))
    assert not ground['valid']


def test_joined_reader_keeps_native_camera_and_exact_visual_tokens(case):
    import json
    from egomotion_pipeline.smplh_archive import SMPLHDerivedArchive
    from egomotion_pipeline.smplh_sidecar import write_episode_sidecar
    from egomotion_pipeline.storage import atomic_json
    from egomotion_pipeline.storage import Archive
    from egomotion_pipeline.schema import SCHEMA_VERSION
    from egomotion_pipeline.windows import prepare_window
    source,record,arrays,fit,contract,out=case
    cfg=json.loads((source.root/'.pipeline/config.json').read_text())
    cfg['schema_version']=SCHEMA_VERSION
    atomic_json(source.root/'.pipeline/config.json',cfg)
    source=Archive(source.root)
    before=source.data_path(0).read_bytes()
    write_episode_sidecar(out,record,arrays,fit,contract)
    reader=SMPLHDerivedArchive(out)
    window=reader.read_window(0,num_frames=4)
    expected=prepare_window(arrays,record['metadata'],start_frame=0,num_frames=4)
    np.testing.assert_array_equal(window.arrays['observation.features.dino'],arrays['observation.features.dino'][[0,2,4,6]])
    np.testing.assert_allclose(window.trajectory,expected.trajectory,atol=1e-7)
    for side in ('left','right'):
        np.testing.assert_array_equal(window.arrays[f'observation.state.hand_{side}_world'],expected.arrays[f'observation.state.hand_{side}_world'])
    assert window.metadata['target_joint_basis']=='smplh_body22_axes_v1'
    assert window.metadata['rotation_basis_verified'] is False
    assert window.metadata['label_kind']=='model_fitted_estimate'
    assert window.metadata['unified_dense22']['target_joint_basis']=='smplh_body22_axes_v1'
    assert window.metadata['unified_dense22']['label_kind']=='model_fitted_estimate'
    assert before==source.data_path(0).read_bytes()


def test_rotation_completion_is_independent_of_source_position_and_interpolation(case):
    import json
    from egomotion_pipeline.smplh_archive import SMPLHDerivedArchive
    from egomotion_pipeline.smplh_sidecar import build_sidecar_contract, write_episode_sidecar
    from egomotion_pipeline.storage import atomic_json, Archive, write_arrays
    from egomotion_pipeline.schema import SCHEMA_VERSION, feature_specs_for_schema
    from egomotion_pipeline.unified_dense22 import encode_unified_dense22
    source,record,arrays,fit,contract,out=case
    cfg=json.loads((source.root/'.pipeline/config.json').read_text())
    cfg['schema_version']=SCHEMA_VERSION
    atomic_json(source.root/'.pipeline/config.json',cfg)
    source=Archive(source.root)
    arrays['observation.valid.body_rotation'][2,6]=False
    arrays['observation.state.body_world_rotation'].reshape(8,22,4)[2,6]=[1,0,0,0]
    fit['source_rotation_valid']=arrays['observation.valid.body_rotation'].copy()
    arrays.update(encode_unified_dense22(arrays,record))
    write_arrays(source.data_path(0),arrays,feature_specs_for_schema(SCHEMA_VERSION))
    contract=build_sidecar_contract(source.root,source.data_path(0),contract['model_identity']['path'],
        contract['config'],contract['calibration'],arrays)
    write_episode_sidecar(out,record,arrays,fit,contract)
    window=SMPLHDerivedArchive(out).read_window(0,num_frames=4,load_dino=False)
    assert window.arrays['observation.source_observed.body_position'][1,6]
    assert not window.arrays['observation.source_observed.body_rotation'][1,6]
    assert not window.arrays['observation.estimated.completed.body_position'][1,6]
    assert window.arrays['observation.estimated.completed.body_rotation'][1,6]
    assert not window.arrays['observation.source_interpolated.body_rotation'][1,6]
    assert window.arrays['observation.interpolated.body_rotation'][1,6]
