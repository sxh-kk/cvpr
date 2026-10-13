import json
import numpy as np
import pyarrow.parquet as pq
import pytest
from egomotion_pipeline.cli import convert, main
from egomotion_pipeline.features import extract_features
from egomotion_pipeline.storage import Archive, read_arrays
from egomotion_pipeline.schema import SCHEMA_VERSION, dino_format
from egomotion_pipeline.validation import validate
from egomotion_pipeline.video import nearest_video_frames
from conftest import make_egosuite, write_video


def source_config(root):
    return {'fps':30, 'sources':[{'type':'egosuite_lerobot', 'name':'egosuite', 'root':str(root),
                                'coordinate_verified':True, 'rotation_basis_verified':True}]}


class FakeEncoder:
    metadata = {**dino_format(), 'checkpoint':'test_only_encoder'}
    def __call__(self, images):
        return np.stack([np.repeat((np.mean(image)+1+np.arange(5, dtype=np.float32)*100)[:, None],
                                   1024, axis=1) for image in images])


def test_pts_selection_and_video_holes(tmp_path):
    video = write_video(tmp_path/'gap.mp4', [0,1,3,4])
    frames = list(nearest_video_frames(video, [0,1/30,2/30,3/30,.4], 1/60+.001))
    assert [None if x[1] is None else round(x[1]*30) for x in frames] == [0,1,None,3,None]


def test_end_to_end_lerobot_resumption_features_and_validation(tmp_path):
    root = make_egosuite(tmp_path/'source')
    output = tmp_path/'output'; config = source_config(root)
    convert(config, output)
    archive = Archive(output)
    report = validate(archive, strict=True)
    assert report['episodes'] == 2 and report['frames'] == 10
    assert not read_arrays(archive.data_path(0), archive.specs)['observation.valid.dino'].any()
    with pytest.raises(ValueError, match='DINO is pending'):
        validate(archive, require_dino=True)
    convert(config, output)
    assert len(Archive(output).records()) == 2
    extract_features(archive, {'batch_size':2}, FakeEncoder())
    assert validate(archive, require_dino=True, strict=True)['frames'] == 10
    second = read_arrays(archive.data_path(1), archive.specs)
    assert np.allclose(second['observation.video_timestamp'], np.arange(5,10)/30)
    assert second['observation.valid.dino'].all()
    assert second['observation.features.dino'].shape == (5, 5120)
    tokens = second['observation.features.dino'].reshape(5, 5, 1024)
    assert np.allclose(tokens[:, :, 0]-tokens[:, :1, 0], np.arange(5)*100)
    extract_features(archive, {'batch_size':2}, FakeEncoder())
    with pytest.raises(ValueError, match='different model'):
        extract_features(archive, {'batch_size':3}, FakeEncoder())
    # No source annotation copied to either metadata or frame columns.
    assert 'task' not in pq.read_schema(archive.data_path(0)).names
    assert all('THIS_SOURCE_TEXT' not in p.read_text() for p in output.rglob('*.json'))
    metadata = pq.read_table(output/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert metadata[1]['dataset_from_index'] == 5 and metadata[1]['data/file_index'] == 1
    with pytest.raises(ValueError, match='different config'):
        Archive(output, {'fps':30, 'sources':[]})
    # Source content changing must not silently return a stale resumed episode.
    path = root/'data/chunk-000/file-000.parquet'
    table = pq.read_table(path).to_pylist(); table[0]['observation.state.body_world'][0][0] = 100
    import pyarrow as pa
    pq.write_table(pa.Table.from_pylist(table), path)
    with pytest.raises(ValueError, match='input changed'):
        convert(config, output)


def test_hf_and_pandas_lerobot_metadata_compatibility(tmp_path):
    datasets = pytest.importorskip('datasets')
    pandas = pytest.importorskip('pandas')
    root = make_egosuite(tmp_path/'source', episodes=1)
    output = tmp_path/'archive'; convert(source_config(root), output)
    info = json.loads((output/'meta/info.json').read_text())
    # Same scalar/Sequence mapping as the official v3 get_hf_features_from_features.
    features = datasets.Features({k: datasets.Value(v['dtype']) if v['shape'] == [1] else
                                 datasets.Sequence(datasets.Value(v['dtype']), length=v['shape'][0])
                                 for k, v in info['features'].items()})
    ds = datasets.load_dataset('parquet', data_files=str(output/'data/chunk-000/file-000.parquet'), features=features, split='train')
    assert len(ds) == 5 and len(ds[0]['observation.state.body_world']) == 66
    assert len(ds[0]['observation.features.dino']) == 5120
    tasks = pandas.read_parquet(output/'meta/tasks.parquet')
    assert tasks.index.name == 'task' and tasks.iloc[0].task_index == 0 and tasks.iloc[0].name == 'motion'


def test_cli_empty_14_body_input_is_not_success(tmp_path, capsys):
    root = make_egosuite(tmp_path/'source', body_joints=14)
    cfg = tmp_path/'config.json'; cfg.write_text(json.dumps(source_config(root)))
    assert main(['inspect','--config',str(cfg)]) == 0
    assert 'upper_body_14_episodes' in capsys.readouterr().out
    assert main(['convert','--config',str(cfg),'--output',str(tmp_path/'out')]) == 1
    assert 'No selected full-body' in capsys.readouterr().err


def test_strict_coordinates_and_missing_video_feature_mask(tmp_path):
    root = make_egosuite(tmp_path/'source', episodes=1)
    config = source_config(root); config['sources'][0]['rotation_basis_verified'] = False
    convert(config, tmp_path/'output')
    archive = Archive(tmp_path/'output')
    assert validate(archive)['warnings']
    with pytest.raises(ValueError, match='rotation_basis_verified'):
        validate(archive, strict=True)


def test_old_schema_is_rejected_before_reading_or_resuming(tmp_path):
    root = tmp_path/'old_archive'
    provenance = root/'.pipeline/config.json'
    provenance.parent.mkdir(parents=True)
    old = {'schema_version': 'egomotion-body22-ground-head-v2', 'config': {'fps': 30}}
    provenance.write_text(json.dumps(old))
    for config in [None, old['config']]:
        with pytest.raises(ValueError, match='schema.*camera-floor dense22'):
            Archive(root, config)
    assert json.loads(provenance.read_text()) == old
    new_archive = Archive(tmp_path/'new_archive', {'fps': 30})
    assert json.loads((new_archive.root/'.pipeline/config.json').read_text())['schema_version'] == SCHEMA_VERSION
