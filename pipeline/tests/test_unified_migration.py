"""Production-like v4→v5 preservation and publication failure checks."""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline import unified_migration as migration
from egomotion_pipeline.schema import SCHEMA_VERSION, V4_SCHEMA_VERSION
from egomotion_pipeline.storage import Archive, atomic_json, read_arrays, read_json
from egomotion_pipeline.unified_dense22 import unified_specs
from egomotion_pipeline.validation import validate
from unified_v4_fixture import v4_archive


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _live_snapshot(root):
    """Public archive bytes, excluding the private migration journal/staging."""
    root = Path(root)
    paths = [root / '.pipeline/config.json']
    for relative in ('data', 'meta', '.pipeline/episodes'):
        paths.extend(path for path in (root / relative).rglob('*') if path.is_file())
    return {str(path.relative_to(root)): _sha(path) for path in paths}


def _job(root):
    journal = read_json(Path(root) / migration.JOURNAL)
    return Path(root) / journal['job_relative_path'], journal


def _fail_after_dir_rename(monkeypatch, root, which):
    """A real rename occurs, then the process fails before recording progress."""
    original = migration.os.replace
    root = Path(root)
    events = []

    def replace(source, target):
        source, target = Path(source), Path(target)
        original(source, target)
        is_move = ((which == 'backup' and source == root / 'data') or
                   (which == 'publish' and target == root / 'data'))
        if is_move:
            events.append((source, target))
            raise RuntimeError(f'crash after {which} data rename')

    monkeypatch.setattr(migration.os, 'replace', replace)
    return original, events


def test_unified_upgrade_preserves_all_numeric_columns_and_backups(tmp_path, monkeypatch):
    old = v4_archive(tmp_path / 'archive')
    monkeypatch.setattr(migration, 'BATCH_SIZE', 3)
    assert validate(old, require_dino=True)['frames'] == 16
    before = _live_snapshot(old.root)
    original_tables = [pq.read_table(old.data_path(i)) for i in range(2)]
    original_records = old.records()
    progress = []
    result = migration.upgrade_dense_v5(old.root, workers=1, progress=progress.append)
    assert result['state'] == 'complete' and result['completed_frames'] == 16
    assert progress[-1]['state'] == 'complete'
    new = Archive(old.root)
    assert new.schema_version == SCHEMA_VERSION
    assert validate(new, require_dino=True)['frames'] == 16
    backup = new.root / result['rollback_relative_path']
    for index, original in enumerate(original_tables):
        target = pq.read_table(new.data_path(index))
        assert pq.ParquetFile(new.data_path(index)).metadata.num_row_groups == 3
        assert set(target.column_names) == set(original.column_names) | set(unified_specs())
        for name in original.column_names:
            assert target[name].equals(original[name]), name
        assert _sha(migration._data_path(backup, index)) == before[str(new.data_path(index).relative_to(new.root))]
        record = new.records()[index]
        assert record['dino'] == original_records[index]['dino']
        assert record['metadata']['rotation_basis_verified'] is False
        assert record['metadata']['unified_dense22']['rotation_basis_verified'] is False
        assert record['motion_representation']['schema_version'] == SCHEMA_VERSION
        assert len(record['motion_representation']['motion_digest']) == 64
        receipt = read_json(new.root / result['job_relative_path'] / f'receipts/{index:08d}.json')
        assert receipt['validation']['all_old_columns_exact']
        assert len(receipt['output_column_sha256']) == len(new.specs)
        assert receipt['source_column_sha256']['observation.features.dino'] == receipt['output_column_sha256']['observation.features.dino']
    assert _sha(backup / 'config.json') == before['.pipeline/config.json']
    for relative, checksum in before.items():
        if relative.startswith('meta/'):
            assert _sha(backup / relative) == checksum
    arrays = read_arrays(new.data_path(0), new.specs)
    world = arrays['observation.state.body_world_rot6d_xyz'].reshape(8, 22, 9)
    assert np.all(world[0, 3, 6:] == 0) and np.any(world[0, 3, :6])
    assert np.all(world[0, 4, :6] == 0) and np.any(world[0, 4, 6:])
    assert arrays['observation.interpolated.body_world_rot6d_xyz_position'][0, 5]
    assert arrays['observation.valid.body_world_rot6d_xyz_position'][0, 5]
    hand = arrays['observation.state.hand_left_floor'].reshape(8, 21, 3)
    body = arrays['observation.state.body_floor_rot6d_xyz'].reshape(8, 22, 9)[..., 6:]
    assert np.linalg.norm(hand[0, 0] - body[0, 20]) > .005  # measured wrists stay separate
    assert arrays['observation.interpolated.hand_left_floor_position'][0, 7]
    assert not arrays['observation.valid.hand_left_floor_position'][1, 2]
    assert np.all(hand[1, 2] == 0)
    after = _live_snapshot(new.root)
    timestamps = [new.data_path(i).stat().st_mtime_ns for i in range(2)]
    assert migration.upgrade_dense_v5(new.root, workers=1)['already_current']
    assert _live_snapshot(new.root) == after
    assert [new.data_path(i).stat().st_mtime_ns for i in range(2)] == timestamps


def test_missing_floor_preserves_world_and_leaves_archived_local_masks_invalid(tmp_path):
    old = v4_archive(tmp_path / 'archive', missing_ground=True)
    assert validate(old, require_dino=True)['frames'] == 16
    migration.upgrade_dense_v5(old.root, workers=1)
    new = Archive(old.root)
    assert validate(new, require_dino=True)['frames'] == 16
    arrays = read_arrays(new.data_path(0), new.specs)
    assert arrays['observation.valid.body_world_rot6d_xyz_position'].any()
    assert arrays['observation.valid.body_world_rot6d_xyz_rotation'].any()
    for name in unified_specs():
        if 'floor' in name:
            assert not arrays[name].any(), name
    # Training-window fallback is a separate selected-window operation.
    window = new.read_window(0, start_frame=0, num_frames=4, fps=15)
    assert window.arrays['observation.valid.floor'].all()
    assert window.arrays['observation.valid.body_floor_rot6d_xyz_position'].any()
    assert not read_arrays(new.data_path(0), new.specs)['observation.valid.floor'].any()


def test_pilot_stages_only_selected_episode_without_publishing_gate(tmp_path):
    archive = v4_archive(tmp_path / 'archive')
    before = _live_snapshot(archive.root)
    result = migration.upgrade_dense_v5(archive.root, workers=1, pilot_indices=[1], commit=False)
    assert result['state'] == 'validated' and result['completed_episodes'] == 1
    assert result['pilot'] is True
    assert _live_snapshot(archive.root) == before
    assert not (archive.root / migration.JOURNAL).exists()
    assert Archive(archive.root).schema_version == V4_SCHEMA_VERSION
    job = archive.root / result['job_relative_path']
    assert migration._data_path(job, 1).is_file()
    assert not migration._data_path(job, 0).exists()
    with pytest.raises(ValueError, match='subset pilot'):
        migration.upgrade_dense_v5(archive.root, pilot_indices=[1])


def test_unified_upgrade_excludes_other_writers(tmp_path):
    archive = v4_archive(tmp_path / 'archive')
    before = _live_snapshot(archive.root)
    with archive.lock(), pytest.raises(RuntimeError, match='Another process'):
        migration.upgrade_dense_v5(archive.root, workers=1)
    assert _live_snapshot(archive.root) == before
    assert not (archive.root / migration.JOURNAL).exists()


def test_stage_failure_keeps_live_bytes_and_resumes_verified_receipts(tmp_path, monkeypatch):
    archive = v4_archive(tmp_path / 'archive')
    before = _live_snapshot(archive.root)
    original = migration._stage_episode
    calls = []

    def fail_second(task):
        calls.append(task[2]['episode_index'])
        if calls[-1] == 1:
            raise RuntimeError('worker interruption')
        return original(task)

    monkeypatch.setattr(migration, '_stage_episode', fail_second)
    with pytest.raises(RuntimeError, match='worker interruption'):
        migration.upgrade_dense_v5(archive.root, workers=1)
    assert _live_snapshot(archive.root) == before
    job, journal = _job(archive.root)
    assert journal['state'] == 'failed' and journal['phase'] == 'staging'
    first_stat = migration._stat(migration._data_path(job, 0))
    with pytest.raises(ValueError, match='unfinished'):
        Archive(archive.root)
    monkeypatch.setattr(migration, '_stage_episode', original)
    result = migration.upgrade_dense_v5(archive.root, workers=1)
    assert result['state'] == 'complete'
    assert migration._stat(migration._data_path(archive.root, 0)) == first_stat
    assert validate(Archive(archive.root), require_dino=True)['frames'] == 16


@pytest.mark.parametrize('which', ['backup', 'publish'])
def test_resume_after_real_directory_rename_never_reencodes(tmp_path, monkeypatch, which):
    archive = v4_archive(tmp_path / 'archive')
    source_sha = [_sha(archive.data_path(i)) for i in range(2)]
    original, events = _fail_after_dir_rename(monkeypatch, archive.root, which)
    with pytest.raises(RuntimeError, match=f'crash after {which}'):
        migration.upgrade_dense_v5(archive.root, workers=1)
    assert len(events) == 1
    job, journal = _job(archive.root)
    assert journal['phase'] == 'committing' and journal['state'] == 'failed'
    assert (archive.root / 'data').exists() == (which == 'publish')
    with pytest.raises(ValueError, match='unfinished'):
        Archive(archive.root)
    monkeypatch.setattr(migration.os, 'replace', original)

    def forbidden_reencode(task):
        raise AssertionError('commit recovery must reuse verified staging')

    monkeypatch.setattr(migration, '_stage_episode', forbidden_reencode)
    result = migration.upgrade_dense_v5(archive.root, workers=1)
    assert result['state'] == 'complete'
    assert [_sha(migration._data_path(job / 'backup', i)) for i in range(2)] == source_sha
    assert validate(Archive(archive.root), require_dino=True)['frames'] == 16


@pytest.mark.parametrize('tamper', ['source', 'staged', 'receipt'])
def test_validated_commit_detects_changed_inventory_before_publication(tmp_path, monkeypatch, tamper):
    archive = v4_archive(tmp_path / 'archive')
    original_commit = migration._commit_v5

    def interrupted_commit(*args):
        raise RuntimeError('pause before publication')

    monkeypatch.setattr(migration, '_commit_v5', interrupted_commit)
    with pytest.raises(RuntimeError, match='pause before publication'):
        migration.upgrade_dense_v5(archive.root, workers=1)
    job, journal = _job(archive.root)
    assert journal['phase'] == 'validated'
    if tamper == 'receipt':
        path = job / 'receipts/00000000.json'
        receipt = read_json(path)
        receipt['motion_digest'] = '0' * 64
        atomic_json(path, receipt)
        expected = 'receipts changed'
    else:
        path = migration._data_path(archive.root, 0) if tamper == 'source' else migration._data_path(job, 0)
        with path.open('ab') as stream:
            stream.write(b'tamper')
        expected = 'changed|mismatch'
    monkeypatch.setattr(migration, '_commit_v5', original_commit)
    if tamper == 'receipt':
        # Direct commit proves the validated receipt set is cryptographically locked.
        with pytest.raises(ValueError, match=expected):
            original_commit(Archive(archive.root, _allow_upgrade=True), journal, archive.root / migration.JOURNAL)
    else:
        with pytest.raises(ValueError, match=expected):
            migration.upgrade_dense_v5(archive.root, workers=1)
    assert (archive.root / 'data').is_dir()
    assert not (job / 'backup/data').exists()
    assert read_json(archive.root / '.pipeline/config.json')['schema_version'] == V4_SCHEMA_VERSION


def test_frame_identity_is_checked_before_any_live_publication(tmp_path):
    archive = v4_archive(tmp_path / 'archive')
    path = archive.data_path(1)
    table = pq.read_table(path)
    from egomotion_pipeline.storage import write_arrays
    arrays = read_arrays(path, archive.specs)
    arrays['frame_index'][3] = 2
    write_arrays(path, arrays, archive.specs)
    before = _live_snapshot(archive.root)
    with pytest.raises(ValueError, match='frame identity mismatch'):
        migration.upgrade_dense_v5(archive.root, workers=1)
    assert _live_snapshot(archive.root) == before
    assert pq.read_table(path)['observation.features.dino'].equals(table['observation.features.dino'])


def test_parallel_workers_emit_exact_preserved_parquet(tmp_path):
    archive = v4_archive(tmp_path / 'archive')
    result = migration.upgrade_dense_v5(archive.root, workers=2)
    assert result['state'] == 'complete' and result['completed_episodes'] == 2
    assert validate(Archive(archive.root), require_dino=True)['frames'] == 16


def test_resume_after_config_published_but_global_metadata_still_old(tmp_path, monkeypatch):
    archive = v4_archive(tmp_path / 'archive')
    original = migration.atomic_json
    config_path = archive.root / '.pipeline/config.json'

    def stop_after_config(path, payload):
        original(path, payload)
        if Path(path) == config_path and payload.get('schema_version') == SCHEMA_VERSION:
            raise RuntimeError('config published before meta')

    monkeypatch.setattr(migration, 'atomic_json', stop_after_config)
    with pytest.raises(RuntimeError, match='before meta'):
        migration.upgrade_dense_v5(archive.root, workers=1)
    assert read_json(config_path)['schema_version'] == SCHEMA_VERSION
    assert read_json(archive.root / 'meta/info.json')['schema_version'] == V4_SCHEMA_VERSION
    assert read_json(archive.root / migration.JOURNAL)['phase'] == 'committing'
    with pytest.raises(ValueError, match='unfinished'):
        Archive(archive.root)
    monkeypatch.setattr(migration, 'atomic_json', original)
    result = migration.upgrade_dense_v5(archive.root, workers=1)
    assert result['state'] == 'complete'
    assert validate(Archive(archive.root), require_dino=True)['frames'] == 16


@pytest.mark.parametrize('tamper', [False, True])
def test_native_sidecar_crosscheck_is_required_when_supplied(tmp_path, tamper):
    from egomotion_pipeline.native_dense22 import encode_native_dense22
    from egomotion_pipeline.storage import write_arrays
    archive = v4_archive(tmp_path / 'archive')
    native_root = tmp_path / 'native'
    body_specs = {name: spec for name, spec in unified_specs().items() if 'body_' in name}
    atomic_json(native_root / 'meta/info.json', {'fps': 30})
    for record in archive.records():
        index = record['episode_index']
        arrays = read_arrays(archive.data_path(index), archive.specs)
        native, _ = encode_native_dense22(arrays, record)
        if tamper and index == 1:
            native['observation.state.body_world_rot6d_xyz'][2, 6] += .01
        write_arrays(migration._data_path(native_root, index), native, body_specs)
    before = _live_snapshot(archive.root)
    if tamper:
        with pytest.raises(ValueError, match='Native dense22 value mismatch'):
            migration.upgrade_dense_v5(archive.root, workers=1, native_root=native_root)
        assert _live_snapshot(archive.root) == before
    else:
        result = migration.upgrade_dense_v5(archive.root, workers=1, native_root=native_root)
        assert result['state'] == 'complete'
        job, _ = _job(archive.root)
        assert read_json(job / 'receipts/00000001.json')['native_parity_max_abs'] == 0
