"""Native final-FK audit evidence is immutable, bound, and source aware."""
from concurrent.futures import Future
from dataclasses import asdict
from types import SimpleNamespace
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from MotionExpert.audit_fit_quality import AUDIT_FORMAT, COLUMNS, _audit_tasks, audit_episode, code_binding
from MotionExpert.fit_quality import FitQualityPolicy, FitQualityReason


def fixture_episode(tmp_path, source='egosuite', index=0):
    root = tmp_path / 'archive'
    episode_dir = root / f'episodes/{index:08d}'
    episode_dir.mkdir(parents=True)
    (root / '.pipeline/episodes').mkdir(parents=True, exist_ok=True)
    output = tmp_path / 'audit'
    output.mkdir(exist_ok=True)
    count = 7
    arrays = {}
    for name, width in COLUMNS.items():
        boolean = 'valid' in name or 'observed' in name or 'completed' in name
        if name in ('frame_index', 'episode_index'):
            value = np.arange(count, dtype=np.int64) if name == 'frame_index' else np.full(count, index, np.int64)
        elif name == 'timestamp':
            value = (np.arange(count)/30).astype(np.float32)
        else:
            value = np.zeros((count, width), dtype=bool if boolean else np.float32)
            if 'estimated.valid.' in name or 'observation.valid.' in name or 'source.observed.' in name or name == 'source.feet_valid':
                value[:] = True
        arrays[name] = value
    # Native fit missing at last frame; all HM fit-linked landmarks are missing
    # with it. Raw observations are intentionally retained for EgoSuite.
    arrays['estimated.valid.body_position'][-1] = False
    arrays['estimated.valid.body_rotation'][-1] = False
    if source == 'egoexo4d_hm':
        arrays['source.feet_valid'][-1] = False
        for side in ('left', 'right'):
            arrays[f'observation.valid.hand_{side}_position'][-1] = False
    # Parameters put final head at 2.15 m for frame3, despite source head/CPF
    # and camera remaining together. Chunk size2 exercises inter-block edges.
    arrays['smplh.trans'][3, 0] = 2
    arrays['source.observed.body_position'][3, 15] = False
    arrays['estimated.completed.body_position'][3, 15] = True
    columns = {}
    for name, width in COLUMNS.items():
        value = arrays[name]
        kind = pa.bool_() if value.dtype == bool else pa.int64() if value.dtype.kind in 'iu' else pa.float32()
        columns[name] = pa.array(value.tolist(), type=pa.list_(kind, width)) if value.ndim == 2 else pa.array(value, type=kind)
    motion = episode_dir / 'motion.parquet'
    pq.write_table(pa.table(columns), motion, row_group_size=3)
    record = {'episode_index': index, 'source': source, 'source_key': f'{source}:sample', 'split': 'train',
              'length': count, 'data_path': str(motion.relative_to(root)),
              'motion_sha256': hashlib.sha256(motion.read_bytes()).hexdigest(),
              'metadata': {'coordinate_verified': True, 'source_identity': {'sha256': 'b'*64}}}
    path = root / f'.pipeline/episodes/{index:08d}.json'
    path.write_text(json.dumps(record))
    stat = motion.stat()
    committed = {'record': record, 'motion': {'sha256': record['motion_sha256']},
                 'output_identities': {'motion': {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}}}
    (episode_dir / 'receipt.json').write_text(json.dumps(committed))
    rest = np.zeros((22, 3))
    rest[15, 0] = .15
    model = SimpleNamespace(rest=rest, shape_basis=np.zeros((22, 3, 16)), model_sha256='c'*64)
    return root, path, output, model


@pytest.mark.parametrize('source', ['egosuite', 'egoexo4d_hm'])
def test_audit_reads_final_fk_and_binds_sidecar_without_modifying_parent(tmp_path, source):
    root, path, output, model = fixture_episode(tmp_path, source)
    original = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    receipt = audit_episode(root, path, output, asdict(FitQualityPolicy()), model=model, batch_size=2)
    assert receipt['format'] == AUDIT_FORMAT
    assert receipt['record_sha256'] == hashlib.sha256(original[path]).hexdigest()
    assert receipt['model_sha256'] == model.model_sha256
    assert receipt['code_binding'] == code_binding()
    assert receipt['source_policy_sha256']
    assert receipt['summary']['head_observation_strata']['head_completed']['newly_rejected_frames'] == 1
    with np.load(output / receipt['mask_path'], allow_pickle=False) as saved:
        assert saved['quality_valid'].dtype == bool
        assert saved['reasons'].dtype == np.uint16
        assert saved['warnings'].dtype == np.uint16
        assert saved['reasons'][3] & FitQualityReason.HEAD_CAMERA_FAR
        assert saved['head_completed'][3]
        assert saved['head_camera_distance_m'][3] == pytest.approx(2.15)
        assert not saved['quality_valid'][-1]
        assert not saved['newly_rejected'][-1]
        # Jump crossing an FK block boundary invalidates the before/after
        # neighbours too, using the same native episode scoring context.
        assert saved['newly_rejected'].tolist() == [False, False, True, True, True, False, False]
        assert receipt['native_valid_sha256'] == hashlib.sha256(saved['native_valid'].tobytes()).hexdigest()
    assert all(p.read_bytes() == value for p, value in original.items())
    with pytest.raises(FileExistsError, match='immutable'):
        audit_episode(root, path, output, asdict(FitQualityPolicy()), model=model, batch_size=2)


def test_audit_rejects_hm_landmarks_without_body_availability(tmp_path):
    root, path, output, model = fixture_episode(tmp_path, 'egoexo4d_hm')
    record = json.loads(path.read_text())
    motion = root / record['data_path']
    table = pq.read_table(motion)
    name = 'source.feet_valid'
    values = np.asarray(table[name].combine_chunks().values).reshape(7, 2)
    values[-1, 0] = True
    table = table.set_column(table.schema.get_field_index(name), name, pa.array(values.tolist(), type=pa.list_(pa.bool_(), 2)))
    pq.write_table(table, motion)
    # Simulate an accepted-but-contradictory source identity, so the explicit
    # new baseline invariant, rather than a stale-stat check, must catch it.
    record['motion_sha256'] = hashlib.sha256(motion.read_bytes()).hexdigest()
    path.write_text(json.dumps(record))
    committed_path = motion.parent / 'receipt.json'
    committed = json.loads(committed_path.read_text())
    committed.update(record=record, motion={'sha256': record['motion_sha256']})
    stat = motion.stat()
    committed['output_identities']['motion'] = {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    committed_path.write_text(json.dumps(committed))
    with pytest.raises(ValueError, match='HM fit-linked labels without body baseline: 1'):
        audit_episode(root, path, output, asdict(FitQualityPolicy()), model=model)
    assert not list(output.glob('*.npz'))


def test_audit_rejects_stale_motion_receipt_before_issuing_new_evidence(tmp_path):
    root, path, output, model = fixture_episode(tmp_path)
    motion = root / json.loads(path.read_text())['data_path']
    import os
    stamp = motion.stat()
    os.utime(motion, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    with pytest.raises(ValueError, match='Motion identity differs'):
        audit_episode(root, path, output, asdict(FitQualityPolicy()), model=model)


def test_bounded_scheduler_consumes_results_while_admitting_all_eligible_episodes():
    class CountingFuture(Future):
        consumed = False
        def result(self, timeout=None):
            if not self.consumed:
                pool.unconsumed -= 1
                self.consumed = True
            return super().result(timeout)
    class Pool:
        unconsumed = 0
        peak = 0
        def submit(self, fn, root, record, output, policy):
            self.unconsumed += 1
            self.peak = max(self.peak, self.unconsumed)
            assert self.unconsumed <= 16
            future = CountingFuture()
            future.set_result(record)
            return future
    pool = Pool()
    receipts = list(_audit_tasks(pool, range(9216), '/parent', '/audit', {}, 8))
    assert sorted(map(int, receipts)) == list(range(9216))
    assert pool.unconsumed == 0 and pool.peak == 16
