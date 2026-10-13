import numpy as np
import pyarrow.parquet as pq
import pytest
import importlib.util
import os
from pathlib import Path
from egomotion_pipeline import migration
from egomotion_pipeline.cli import convert, main
from egomotion_pipeline.features import extract_features
from egomotion_pipeline.schema import LEGACY_SCHEMA_VERSION, SCHEMA_VERSION, feature_specs
from egomotion_pipeline.storage import Archive, atomic_json, read_arrays, read_json, write_arrays
from egomotion_pipeline.validation import validate
from test_archive_and_video import source_config, FakeEncoder
from conftest import make_egosuite


def legacy_archive(tmp_path):
    source = make_egosuite(tmp_path/'source')
    root = tmp_path/'archive'
    config = source_config(source)
    config['sources'][0]['head_reference'] = {'basis_verified':True, 'camera_from_ee4d_camera':np.eye(3).tolist()}
    config['sources'][0]['ground'] = {'mode':'manual', 'height_m':0., 'verified':True}
    convert(config, root)
    archive = Archive(root)
    old_specs = feature_specs(1024)
    for record in archive.records():
        path = archive.data_path(record['episode_index'])
        arrays = read_arrays(path, archive.specs)
        arrays['observation.features.dino'] = arrays['observation.features.dino'][:, :1024]
        write_arrays(path, arrays, old_specs)
    provenance = read_json(root/'.pipeline/config.json')
    provenance['schema_version'] = LEGACY_SCHEMA_VERSION
    provenance['pipeline_version'] = '0.5.0'
    atomic_json(root/'.pipeline/config.json', provenance)
    archive = Archive(root)
    archive.finalize()
    return archive


def test_upgrade_preserves_every_motion_column_and_window(tmp_path):
    old = legacy_archive(tmp_path)
    before = [pq.read_table(old.data_path(i)) for i in range(2)]
    window_before = old.read_window(0, start_frame=0, num_frames=5, fps=30)
    assert validate(old)['frames'] == 10
    with pytest.raises(ValueError, match='upgrade-dino'):
        extract_features(old, {}, FakeEncoder())
    assert main(['upgrade-dino', '--dataset', str(old.root)]) == 0
    new = Archive(old.root)
    assert new.schema_version == SCHEMA_VERSION and new.specs['observation.features.dino']['shape'] == [5120]
    assert validate(new)['frames'] == 10
    for index, table in enumerate(before):
        after = pq.read_table(new.data_path(index))
        for name in table.column_names:
            if name != migration.FEATURE:
                assert after[name].equals(table[name]), name
        assert not read_arrays(new.data_path(index), new.specs)[migration.VALID].any()
    window_after = new.read_window(0, start_frame=0, num_frames=5, fps=30)
    np.testing.assert_array_equal(window_before.trajectory, window_after.trajectory)
    np.testing.assert_array_equal(window_before.arrays['observation.state.body_head_floor'],
                                  window_after.arrays['observation.state.body_head_floor'])
    assert read_json(new.root/'.pipeline/config.json')['motion_conversion_provenance']['pipeline_version'] == '0.5.0'
    # An upgrade does not claim old motion was generated with today's code.
    with pytest.raises(ValueError, match='different config'):
        Archive(new.root, source_config(tmp_path/'source'))
    assert migration.upgrade_dino(new.root)['already_current']
    extract_features(new, {'batch_size':2}, FakeEncoder())
    assert validate(new, require_dino=True)['frames'] == 10


def test_upgrade_preflights_all_episodes_and_rejects_cached_cls(tmp_path):
    archive = legacy_archive(tmp_path)
    hashes = [path.read_bytes() for path in [archive.data_path(0), archive.data_path(1)]]
    record = archive.records()[1]
    record['dino']['state'] = 'complete'
    atomic_json(archive.record_path(1), record)
    with pytest.raises(ValueError, match='pending features only'):
        migration.upgrade_dino(archive.root)
    assert not (archive.root/'.pipeline/dino_upgrade.json').exists()
    assert [archive.data_path(i).read_bytes() for i in range(2)] == hashes


def test_upgrade_resumes_after_partial_atomic_replacement(tmp_path, monkeypatch):
    archive = legacy_archive(tmp_path)
    widen = migration._widen_pending_file
    calls = []
    def fail_second(path):
        calls.append(path)
        if len(calls) == 2:
            raise RuntimeError('simulated interruption')
        widen(path)
    monkeypatch.setattr(migration, '_widen_pending_file', fail_second)
    with pytest.raises(RuntimeError, match='interruption'):
        migration.upgrade_dino(archive.root)
    with pytest.raises(ValueError, match='unfinished'):
        Archive(archive.root)
    assert pq.read_schema(archive.data_path(0)).field(migration.FEATURE).type.list_size == 5120
    assert pq.read_schema(archive.data_path(1)).field(migration.FEATURE).type.list_size == 1024
    calls.clear()
    monkeypatch.setattr(migration, '_widen_pending_file', lambda path: (calls.append(path), widen(path)))
    result = migration.upgrade_dino(archive.root)
    assert result['episodes'] == 2 and calls == [archive.data_path(1)]
    assert validate(Archive(archive.root))['frames'] == 10


def test_upgrade_respects_active_writer_lock(tmp_path):
    archive = legacy_archive(tmp_path)
    with archive.lock(), pytest.raises(RuntimeError, match='Another process'):
        migration.upgrade_dino(archive.root)
    assert Archive(archive.root).schema_version == LEGACY_SCHEMA_VERSION


@pytest.mark.parametrize('column', [migration.FEATURE, migration.VALID, 'observation.video_timestamp'])
def test_upgrade_rejects_nonzero_pending_data(tmp_path, column):
    archive = legacy_archive(tmp_path)
    arrays = read_arrays(archive.data_path(1), archive.specs)
    arrays[column][0] = 1
    write_arrays(archive.data_path(1), arrays, archive.specs)
    with pytest.raises(ValueError, match='Pending DINO'):
        migration.upgrade_dino(archive.root)
    assert pq.read_schema(archive.data_path(0)).field(migration.FEATURE).type.list_size == 1024


def upgrade_runner():
    path = Path(__file__).resolve().parents[1]/'scripts/process_dino_upgrade.py'
    spec = importlib.util.spec_from_file_location('upgrade_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('stage', ['completed_with_errors', 'failed'])
def test_upgrade_runner_propagates_conversion_errors(tmp_path, stage):
    archive = legacy_archive(tmp_path)
    conversion = tmp_path/'conversion'
    atomic_json(conversion/'status.json', {'stage':stage, 'output':str(archive.root),
        'pid':os.getpid(), 'processed_files':3, 'failed_files':1, 'validation_passed':True})
    job = tmp_path/'upgrade'
    result = upgrade_runner().main(['--dataset',str(archive.root), '--job-dir',str(job),
                                   '--wait-for-job',str(conversion)])
    status = read_json(job/'status.json')
    assert status['conversion_failed_files'] == 1
    if stage == 'failed':
        assert result == 1 and status['stage'] == 'failed'
        assert not (archive.root/'.pipeline/dino_upgrade.json').exists()
    else:
        assert result == 0 and status['stage'] == 'completed'
        assert read_json(job/'validation.json')['frames'] == 10
        assert Archive(archive.root).schema_version == SCHEMA_VERSION


def test_upgrade_runner_waits_without_touching_archive(tmp_path, monkeypatch):
    archive = legacy_archive(tmp_path)
    conversion = tmp_path/'conversion'
    state = {'stage':'converting', 'output':str(archive.root), 'pid':os.getpid(),
             'processed_files':1, 'failed_files':0}
    atomic_json(conversion/'status.json', state)
    runner = upgrade_runner()
    waits = []
    def finish_conversion(seconds):
        waits.append(seconds)
        assert not (archive.root/'.pipeline/dino_upgrade.json').exists()
        assert Archive(archive.root).schema_version == LEGACY_SCHEMA_VERSION
        state.update(stage='completed', processed_files=2, validation_passed=True)
        atomic_json(conversion/'status.json', state)
    monkeypatch.setattr(runner.time, 'sleep', finish_conversion)
    assert runner.main(['--dataset',str(archive.root), '--job-dir',str(tmp_path/'upgrade'),
                        '--wait-for-job',str(conversion)]) == 0
    assert waits == [15] and Archive(archive.root).schema_version == SCHEMA_VERSION
