"""Fresh v5 decoding detects changes independently of the deleted v4 snapshot."""
import shutil

import pytest

from egomotion_pipeline.storage import Archive, atomic_json, read_arrays, read_json, write_arrays
from egomotion_pipeline.unified_migration import upgrade_dense_v5
from unified_v4_fixture import v4_archive


@pytest.fixture
def unified_archive(tmp_path):
    archive = v4_archive(tmp_path / 'archive')
    result = upgrade_dense_v5(archive.root, workers=1)
    archive = Archive(archive.root)
    return archive, archive.root / result['rollback_relative_path']


def _run(archive, job, **kwargs):
    from egomotion_pipeline.revalidation import revalidate_unified_dataset
    return revalidate_unified_dataset(archive.root, job, workers=1, batch_size=3, **kwargs)


def _failure(report):
    assert report['passed'] is False
    assert report.get('errors'), 'A rejected current archive must include an auditable cause'


def _change_array(archive, name, change):
    arrays = read_arrays(archive.data_path(0), archive.specs)
    change(arrays[name])
    write_arrays(archive.data_path(0), arrays, archive.specs)


def test_fresh_validation_succeeds_without_v4_backup(unified_archive, tmp_path):
    archive, backup = unified_archive
    # Only a local pytest fixture is removed; production deletion has separate guards.
    shutil.rmtree(backup)
    reopened = Archive(archive.root)
    window = reopened.read_window(0, start_frame=0, num_frames=4, fps=15)
    assert window.trajectory.shape == (4, 18)
    assert window.arrays['observation.state.body_world_rot6d_xyz'].shape == (4, 198)
    assert window.arrays['observation.valid.body_floor_rot6d_xyz_position'].any()
    report = _run(archive, tmp_path / 'audit')
    assert report['passed'] is True
    assert report['full_dataset_validation'] is True
    assert not backup.exists()
    assert len(list((tmp_path / 'audit/episodes').glob('*.json'))) == 2
    assert read_json(tmp_path / 'audit/validation.json')['passed'] is True


def test_subset_report_cannot_claim_full_dataset_validation(unified_archive, tmp_path):
    archive, _ = unified_archive
    report = _run(archive, tmp_path / 'pilot-audit', episode_indices=[1])
    assert report['passed'] is True
    assert report['full_dataset_validation'] is False
    assert [path.name for path in (tmp_path / 'pilot-audit/episodes').glob('*.json')] == ['00000001.json']


def test_finite_dino_corruption_is_rejected_by_full_numeric_digest(unified_archive, tmp_path):
    archive, _ = unified_archive
    _change_array(archive, 'observation.features.dino', lambda x: x.__setitem__((2, 1030), x[2, 1030] + .125))
    report = _run(archive, tmp_path / 'audit')
    _failure(report)
    assert any('numeric column hash differs' in error['error'] and 'observation.features.dino' in error['error']
               for error in report['errors'])


def test_new_audit_does_not_reuse_previous_success_when_dino_changes(unified_archive, tmp_path):
    archive, _ = unified_archive
    assert _run(archive, tmp_path / 'first')['passed'] is True
    _change_array(archive, 'observation.features.dino', lambda x: x.__setitem__((0, 5), x[0, 5] + 1))
    _failure(_run(archive, tmp_path / 'second'))


@pytest.mark.parametrize('column,mutate', [
    ('observation.valid.dino', lambda x: x.__setitem__(0, False)),
    ('observation.video_timestamp', lambda x: x.__setitem__(1, x[1] + .01)),
    ('observation.state.body_floor_rot6d_xyz', lambda x: x.__setitem__((1, 6), x[1, 6] + .02)),
    ('observation.state.body_world_rot6d_xyz', lambda x: x.__setitem__((1, 6), x[1, 6] + .02)),
    ('observation.state.hand_left_floor', lambda x: x.__setitem__((2, 3), x[2, 3] + .02)),
    ('observation.state.head_reference_height', lambda x: x.__setitem__(2, x[2] + .2)),
    ('observation.valid.body_world_rot6d_xyz_position', lambda x: x.__setitem__((2, 2), False)),
    ('observation.interpolated.hand_left_floor_position', lambda x: x.__setitem__((1, 2), True)),
    ('frame_index', lambda x: x.__setitem__(3, 2)),
    ('timestamp', lambda x: x.__setitem__(2, x[2] + .1)),
])
def test_fresh_decoding_rejects_corrupted_masks_geometry_and_identity(unified_archive, tmp_path, column, mutate):
    archive, _ = unified_archive
    _change_array(archive, column, mutate)
    _failure(_run(archive, tmp_path / 'audit'))


def test_body_world_6d_degeneracy_is_rejected(unified_archive, tmp_path):
    archive, _ = unified_archive

    def degenerate_rows(value):
        value.reshape(8, 22, 9)[1, 1, :6] = [1, 0, 0, 1, 0, 0]

    _change_array(archive, 'observation.state.body_world_rot6d_xyz', degenerate_rows)
    _failure(_run(archive, tmp_path / 'audit'))


@pytest.mark.parametrize('kind', ['global_frames', 'episode_length', 'missing_file', 'extra_file', 'unfinished_journal'])
def test_current_inventory_or_gate_mismatch_is_audited_and_rejected(unified_archive, tmp_path, kind):
    archive, _ = unified_archive
    if kind == 'global_frames':
        path = archive.root / 'meta/info.json'
        value = read_json(path); value['total_frames'] += 1
        atomic_json(path, value)
    elif kind == 'episode_length':
        path = archive.record_path(0)
        value = read_json(path); value['length'] += 1
        atomic_json(path, value)
    elif kind == 'missing_file':
        archive.data_path(0).unlink()
    elif kind == 'extra_file':
        extra = archive.root / 'data/chunk-777/file-000.parquet'
        extra.parent.mkdir(parents=True)
        shutil.copy2(archive.data_path(0), extra)
    else:
        path = archive.root / '.pipeline/dense_v5_upgrade.json'
        value = read_json(path); value.update(state='failed', phase='committing')
        atomic_json(path, value)
    _failure(_run(archive, tmp_path / 'audit'))
    assert read_json(tmp_path / 'audit/validation.json')['passed'] is False


def test_revalidation_rejects_wrong_receipt_numeric_identity(unified_archive, tmp_path):
    archive, _ = unified_archive
    journal = read_json(archive.root / '.pipeline/dense_v5_upgrade.json')
    path = archive.root / journal['job_relative_path'] / 'receipts/00000000.json'
    receipt = read_json(path)
    receipt['output_numeric_sha256'] = '0' * 64
    atomic_json(path, receipt)
    _failure(_run(archive, tmp_path / 'audit'))


def test_source_provenance_change_is_rejected_without_backup_dependency(unified_archive, tmp_path):
    archive, backup = unified_archive
    shutil.rmtree(backup)
    path = archive.record_path(0)
    record = read_json(path)
    record['metadata']['operator_id'] = 'changed-after-publication'
    atomic_json(path, record)
    _failure(_run(archive, tmp_path / 'audit'))


def test_unknown_floor_with_valid_world_is_allowed_missingness(tmp_path):
    archive = v4_archive(tmp_path / 'archive', missing_ground=True)
    result = upgrade_dense_v5(archive.root, workers=1)
    archive = Archive(archive.root)
    shutil.rmtree(archive.root / result['rollback_relative_path'])
    report = _run(archive, tmp_path / 'audit')
    assert report['passed'] is True and report['all_live_episodes_validated'] is True
    assert report['allowed_missingness']['unknown_archive_ground'] == 1
