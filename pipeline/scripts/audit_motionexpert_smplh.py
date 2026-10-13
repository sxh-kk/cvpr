#!/usr/bin/env python3
"""Full-population, diagnostic-only real MotionExpert SMPL-H reader acceptance.

This checks the complete index and every window-eligible episode's first/last
samples. It does not train, recompute statistics, modify publication, or claim
to execute every indexed window. No pipeline modules or old source files are
available to the guarded reader.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import threading
import time

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')
sys.dont_write_bytecode = True

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

FORMAT = 'latest-motionexpert-full-population-reader-v1'
PROFILE = dict(fps=5, source_fps=30, frame_phase=0, sampling_step=6,
               window=11, forecast_prefix=6, segment_stride=5,
               min_valid_image_fraction=.5, motion_supervision='joint',
               require_full_window=True, train_task='fore', statistics_recomputed=False)
DATASETS = {}
GUARD = None


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def baseline_files(value, workspace):
    entries = value.get('files')
    if isinstance(entries, dict):
        entries = [dict(path=path, sha256=digest.get('sha256') if isinstance(digest, dict) else digest)
                   for path, digest in entries.items()]
    if not isinstance(entries, list) or not entries:
        raise ValueError('Baseline requires nonempty files with path and sha256')
    result = {}
    for item in entries:
        path = Path(item['path'])
        if not path.is_absolute():
            path = workspace / path
        path = path.resolve(strict=True)
        if not path.is_relative_to(workspace / 'MotionExpert'):
            raise ValueError('Latest ME baseline must contain only MotionExpert source files')
        expected = item['sha256']
        if not isinstance(expected, str) or len(expected) != 64 or path in result:
            raise ValueError('Invalid or duplicate ME baseline digest')
        result[path] = expected
    required = ['config.py', 'dataset/smplh_archive.py', 'dataset/smplh_motion_dataset.py',
                'dataset/smplh_representation.py', 'dataset/dense_representation.py',
                'utils/task_conditioning.py', 'configs/smplh_k12_global.yaml']
    if any((workspace / 'MotionExpert' / name).resolve() not in result for name in required):
        raise ValueError('Baseline omits a required current reader/config source')
    return result


def assert_baseline(files):
    for path, expected in files.items():
        if sha_file(path) != expected:
            raise ValueError('Current ME source differs from frozen latest baseline: ' + str(path))


def remove_pipeline_import_paths(script):
    """Do not offer the audit package directories to Python module discovery."""
    pipeline_root = Path(os.path.realpath(str(Path(script).parents[1])))
    retained, removed = [], []
    for entry in sys.path:
        resolved = Path(os.path.realpath(os.path.abspath(entry or os.getcwd())))
        (removed if resolved.is_relative_to(pipeline_root) else retained).append(entry)
    sys.path[:] = retained
    return removed


class ReaderGuard:
    """Dependency guard for Python/Arrow entry points, not an OS sandbox."""
    def __init__(self, root, blocked_roots, script):
        self.root = os.path.abspath(os.fspath(root))
        self.blocked = [os.path.abspath(os.fspath(path)) for path in blocked_roots]
        self.script = os.path.abspath(os.fspath(script))
        self.pipeline_root = os.path.abspath(str(Path(script).parents[1]))
        self.old_source_open_attempts, self.pipeline_imports = [], []
        self.local = threading.local()

    @staticmethod
    def within(path, root):
        return path == root or path.startswith(root + os.sep)

    def check(self, path, kind='open'):
        if isinstance(path, (int, type(None))) or getattr(self.local, 'checking', False):
            return
        if isinstance(path, (list, tuple)):
            for item in path:
                self.check(item, kind)
            return
        try:
            lexical = os.path.abspath(os.fsdecode(path))
        except TypeError:
            return
        if kind == 'stat' and (self.within(lexical, self.root) or self.root.startswith(lexical + os.sep) or lexical == os.sep):
            return
        self.local.checking = True
        try:
            resolved = os.path.realpath(lexical)
        finally:
            self.local.checking = False
        if self.within(resolved, self.root):
            return
        if resolved == self.script:
            return
        # Python may stat these two directories during module discovery. This
        # does not permit enumeration, opening a directory or inspecting files.
        if kind == 'stat' and resolved in (os.path.dirname(self.script), self.pipeline_root):
            return
        if any(self.within(resolved, prefix) for prefix in self.blocked):
            self.old_source_open_attempts.append(dict(operation=kind, path=resolved, requested=lexical))
            raise PermissionError('ME old/source dependency blocked: ' + resolved)
        if self.within(resolved, self.pipeline_root):
            self.pipeline_imports.append(dict(operation=kind, path=resolved))
            raise PermissionError('ME pipeline source dependency blocked: ' + resolved)

    def hook(self, event, args):
        if event == 'open':
            self.check(args[0])
        elif event == 'os.scandir':
            self.check(args[0], 'scandir')
        elif event == 'import':
            name = args[0]
            if name == 'pipeline' or name.startswith(('pipeline.', 'egomotion_pipeline')):
                self.pipeline_imports.append(dict(operation='import', module=name))
                raise ImportError('ME pipeline import blocked: ' + name)

    def install(self):
        existing = [name for name in sys.modules if name == 'pipeline' or name.startswith(('pipeline.', 'egomotion_pipeline'))]
        if existing:
            raise RuntimeError('Pipeline modules loaded before dependency guard: ' + ','.join(existing))
        sys.addaudithook(self.hook)
        for name in ('stat', 'lstat', 'scandir', 'open'):
            original = getattr(os, name)
            kind = 'stat' if name in ('stat', 'lstat') else name
            def wrapped(path, *args, _original=original, _kind=kind, **kwargs):
                self.check(path, _kind)
                return _original(path, *args, **kwargs)
            setattr(os, name, wrapped)
        for name in ('ParquetFile', 'read_table'):
            original = getattr(pq, name)
            def wrapped(path, *args, _original=original, **kwargs):
                self.check(path, 'Arrow read')
                return _original(path, *args, **kwargs)
            setattr(pq, name, wrapped)
        return self


def contacts_from_source(feet, valid, timestamps, ground):
    feet = np.asarray(feet, np.float64).reshape(11, 2, 3)
    valid = np.asarray(valid, bool) & np.isfinite(feet).all(-1)
    values, masks = np.zeros((11, 2), np.float32), np.zeros((11, 2), bool)
    if not ground.get('valid', False):
        return values, masks
    dt = np.diff(np.asarray(timestamps, np.float64))
    edge_time = np.isfinite(dt) & (dt > 0) & (dt <= 1.5 / 5)
    edges = valid[1:] & valid[:-1] & edge_time[:, None]
    speed = np.linalg.norm(np.diff(feet, axis=0), axis=-1) / np.where(edge_time, dt, 1)[:, None]
    masks[1:] = edges
    values[1:] = ((speed < .2) & (feet[1:, :, 2] - ground['height_m'] < .1)).astype(np.float32)
    masks[0], values[0] = edges[0], values[1]
    return np.where(masks, values, 0), masks


def sample_assertions(sample, record, direct, requests, requested_index):
    y, misc = sample['y'], sample['misc']
    if (misc['episode_index'] != record['episode_index'] or misc['seq_name'] != record['source_key']
            or misc['source'] != record['source'] or misc['idx'] != requested_index
            or not 0 <= misc['start_idx'] <= misc['end_idx'] < record['length']):
        raise ValueError('Actual ME sample does not match requested episode/source/index bounds')
    motion, mask = misc['motion'].numpy(), misc['feature_valid_mask'].numpy()
    traj, trajectory_mask = misc['traj'].numpy(), misc['trajectory_valid'].numpy()
    image, image_mask = y['img_embs'].numpy(), y['valid_img_embs'].numpy()
    if motion.shape != (11, 245) or traj.shape != (11, 18) or image.shape != (11, 5, 1024):
        raise ValueError('Actual ME sample dimensions differ from 245/18/DINO5')
    if not all(np.isfinite(value).all() for value in (motion, traj, image)):
        raise ValueError('Nonfinite actual ME sample')
    if np.any(motion[~mask] != 0) or np.any(traj[~trajectory_mask] != 0) or np.any(image[~image_mask] != 0):
        raise ValueError('Missing actual features are not zero with independent masks')
    if image_mask[6:].any() or image[6:].any() or image_mask[:6].mean() < .5:
        raise ValueError('Historical6 DINO threshold or future-image exclusion failed')
    metadata = json.loads(misc['metadata_json'])
    selected = np.arange(misc['start_idx'], misc['start_idx'] + 61, 6)
    if metadata['training_window']['source_frame_indices'] != selected.tolist() or misc['end_idx'] != int(selected[-1]):
        raise ValueError('Actual ME frame phase/step differs from 0/6')
    if misc['fps'] != 5 or misc['num_real_frames'] != 11 or not y['loss_mask'].bool().all():
        raise ValueError('5Hz complete joint history/future supervision failed')
    if not mask[6:, :108].any():
        raise ValueError('Actual sample has no valid future body target')
    if (metadata.get('label_kind') != 'model_fitted_estimate' or metadata.get('absolute_rotation_accuracy_established') is not False
            or metadata['training_window'].get('body_fk_joint_count') != 22):
        raise ValueError('Estimated SMPL-H provenance or full22 FK declaration lost')
    if 'betas' in y or 'pose_hand' in y or 'text' in y:
        raise ValueError('Shape, hand PCA or text unexpectedly became conditions')
    compared = ['smplh.betas', 'source.feet_world', 'source.feet_valid', 'source.feet_interpolated',
                'observation.state.hand_left_world', 'observation.state.hand_right_world',
                'observation.valid.hand_left_position', 'observation.valid.hand_right_position',
                'source.observed.body_position', 'source.observed.body_rotation',
                'estimated.valid.body_position', 'estimated.valid.body_rotation',
                'estimated.completed.body_position', 'estimated.completed.body_rotation']
    for key in compared:
        if not np.array_equal(misc[key].numpy(), direct[key]):
            raise ValueError('Actual ME discarded/replaced stored parameters/masks: ' + key)
    if direct['smplh.betas'].shape != (11, 16):
        raise ValueError('Actual source beta16 lost')
    source = misc['source.feet_world'].numpy()
    valid = misc['source.feet_valid'].numpy()
    contact, contact_mask = contacts_from_source(source, valid, direct['timestamp'], metadata['ground'])
    if not np.array_equal(mask[:, 243:245], contact_mask) or not np.array_equal(motion[:, 243:245], contact):
        raise ValueError('Actual contact targets differ from source feet at5Hz')
    if metadata['ground'].get('window_fallback_used'):
        usable = valid[:6] & np.isfinite(source[:6].reshape(6, 2, 3)).all(-1)
        expected = source[:6].reshape(6, 2, 3)[..., 2][usable].mean(dtype=np.float64) - .02
        if (not usable.any() or metadata['ground'].get('estimation_scope') != 'observed_prefix'
                or abs(metadata['ground']['height_m'] - expected) > 1e-7):
            raise ValueError('Floor fallback consumed inferred/future feet or changed its policy')
    if len(requests) != 1 or requests[0]['frames'] != selected[:6].tolist():
        raise ValueError('Actual ME visual reader requested frames beyond historical6')
    return dict(start_frame=int(selected[0]), end_frame=int(selected[-1]), source_frame_indices=selected.tolist(),
                motion_shape=[11, 245], trajectory_shape=[11, 18], dino_shape=[11, 5, 1024],
                beta_shape=[11, 16], history_dino_valid_count=int(image_mask[:6].sum()),
                future_dino_zero_and_invalid=True, missing_feature_zero_and_mask=True,
                body_valid_cells=int(mask[:, :108].sum()), hand_valid_cells=int(mask[:, 117:243].sum()),
                contact_valid_cells=int(contact_mask.sum()), source_contact_5Hz_verified=True,
                floor_fallback_used=bool(metadata['ground'].get('window_fallback_used', False)))


def worker_init():
    import torch
    from MotionExpert.dataset.dense_motion_dataset import ProjectedParquetReader
    torch.set_num_threads(1)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    for dataset in DATASETS.values():
        reader = ProjectedParquetReader(cache_mb=128, file_cache=4)
        dataset.reader = dataset.archive.reader = reader


def probe_episode(task):
    split, episode, indices = task
    dataset = DATASETS[split]
    record = dataset.archive_records[episode]
    result = dict(episode_index=episode, passed=False, first_last_passed=False, probes=[])
    try:
        for index in indices:
            requests = []
            reader = dataset.reader
            original = reader.read
            def tracked(path, frames, columns):
                if 'observation.features.dino' in columns:
                    requests.append(dict(path=str(path), frames=np.asarray(frames).tolist()))
                return original(path, frames, columns)
            reader.read = tracked
            try:
                sample = dataset.get_sample(index)
            finally:
                reader.read = original
            start = sample['misc']['start_idx']
            selected = np.arange(start, start + 61, 6)
            columns = ['timestamp', 'smplh.betas', 'source.feet_world', 'source.feet_valid', 'source.feet_interpolated',
                       'observation.state.hand_left_world', 'observation.state.hand_right_world',
                       'observation.valid.hand_left_position', 'observation.valid.hand_right_position',
                       'source.observed.body_position', 'source.observed.body_rotation',
                       'estimated.valid.body_position', 'estimated.valid.body_rotation',
                       'estimated.completed.body_position', 'estimated.completed.body_rotation']
            direct = reader.read(dataset.archive.data_path(episode), selected, columns)
            result['probes'].append(sample_assertions(sample, record, direct, requests, index))
        result.update(passed=True, first_last_passed=True)
    except Exception as error:
        result['error'] = repr(error)
    result['old_source_open_attempts'] = list(GUARD.old_source_open_attempts)
    result['pipeline_imports'] = list(GUARD.pipeline_imports)
    return result


def flush_json(handle, value):
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    handle.flush()
    os.fsync(handle.fileno())


def execute(args, report_handle, status_handle):
    global DATASETS, GUARD
    workspace = Path(args.workspace).resolve(strict=True)
    root = Path(args.root).resolve(strict=True)
    baseline_path = Path(args.baseline).resolve(strict=True)
    baseline_bytes = baseline_path.read_bytes()
    frozen = baseline_files(json.loads(baseline_bytes), workspace)
    assert_baseline(frozen)
    baseline_sha = hashlib.sha256(baseline_bytes).hexdigest()
    report = dict(format=FORMAT, scope='full', passed=False, formal_training_ready=False,
                  diagnostic=True, allow_unpublished=True, profile=PROFILE,
                  statistics_recomputed=False, all_indexed_windows_executed=False,
                  root=str(root), latest_me_baseline_sha256=baseline_sha, latest_baseline_sha256=baseline_sha,
                  latest_me_baseline_files={str(path.relative_to(workspace)): digest for path, digest in frozen.items()},
                  script_sha256=sha_file(__file__), started_at=datetime.now(timezone.utc).isoformat(),
                  episodes=args.expected_episodes, expected=args.expected_episodes, scanned=0,
                  eligible=0, window_eligible=0, legitimate_no_window=0, episode_reports=[], failures=[],
                  old_source_open_attempts=[], pipeline_imports=[])
    started = time.monotonic()
    report['removed_pipeline_import_paths'] = remove_pipeline_import_paths(__file__)
    sys.path.insert(0, str(workspace))
    GUARD = ReaderGuard(root, [root.parent, args.hm_root, *args.blocked_root], __file__).install()
    try:
        import torch
        from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
        torch.set_num_threads(1)
        pa.set_cpu_count(1)
        pa.set_io_thread_count(1)
        tasks, rows, binding = [], {}, None
        split_summary, source_summary = {}, {}
        for split in ('train', 'val', 'test'):
            flush_json(status_handle, dict(state='indexing', split=split, scanned=report['scanned'], updated_at=datetime.now(timezone.utc).isoformat()))
            dataset = SMPLHMotionDataset(archive_root=root, split=split, do_normalization=False,
                allow_unpublished=True, min_valid_image_fraction=.5, fps=5, window=11, forecast_prefix=6,
                segment_stride=5, train_task='fore', require_full_window=True, motion_supervision='joint')
            DATASETS[split] = dataset
            if binding is not None and dataset.archive.binding != binding:
                raise ValueError('Archive binding changed between split indices')
            binding = dataset.archive.binding
            all_records = dataset.archive_records
            if len(all_records) != args.expected_episodes:
                raise ValueError('Full archive population differs from required9263')
            by_sequence = {}
            for index, (sequence, _) in enumerate(dataset.idx_to_sidx_fidx):
                if sequence not in by_sequence:
                    by_sequence[sequence] = [index, index, 0]
                by_sequence[sequence][1] = index
                by_sequence[sequence][2] += 1
            accepted_sequence = {record['episode_index']: i for i, record in enumerate(dataset.records)}
            for record in all_records:
                if record['split'] != split:
                    continue
                episode = record['episode_index']
                row = dict(id=episode, episode_index=episode, source=record['source'], source_key=record['source_key'],
                           length=record['length'], split=split, eligible=episode in accepted_sequence,
                           windows=0, first_last_passed=None, exclusion_reason=None, passed=True)
                if not row['eligible']:
                    row.update(exclusion_reason='quarantine', quarantine_reasons=dataset.rejected_episodes[record['source_key']])
                else:
                    sequence = accepted_sequence[episode]
                    limits = by_sequence.get(sequence)
                    if limits:
                        row['windows'] = limits[2]
                        row['first_last_passed'] = False
                        tasks.append((split, episode, sorted(set(limits[:2]))))
                    else:
                        selected = np.arange(0, record['length'], 6)
                        values = dataset.reader.read(dataset.archive.data_path(episode), selected, ['source.feet_valid'])
                        ground = record['metadata']['ground']
                        if ground['valid'] or values['source.feet_valid'].any():
                            raise ValueError('Unexpected no-window episode outside original missing-source-feet policy: ' + record['source_key'])
                        row.update(exclusion_reason='no_usable_prefix_floor', no_window_evidence=dict(
                            archive_ground_valid=False, original_source_valid_feet=0, sampled_5Hz_frames=len(selected)))
                if episode in rows:
                    raise ValueError('Duplicate full episode coverage across splits')
                rows[episode] = row
            report['scanned'] = len(rows)
            split_summary[split] = dict(episodes=sum(r['split'] == split for r in all_records),
                eligible=len(dataset.records), windows=len(dataset), window_eligible=len(by_sequence),
                legitimate_no_window=len(dataset.records)-len(by_sequence), rejected_windows=dataset.rejected_windows,
                rejected_episodes=len(dataset.rejected_episodes))
            for reader in (dataset.reader,):
                for parquet, _ in reader.files.values():
                    parquet.close()
                reader.files.clear(); reader.groups.clear(); reader.cached_bytes = 0
        if set(rows) != set(range(args.expected_episodes)):
            raise ValueError('Exact complete episode identity set was not indexed')
        report.update(eligible=sum(r['eligible'] for r in rows.values()), window_eligible=len(tasks),
                      legitimate_no_window=sum(r['exclusion_reason'] == 'no_usable_prefix_floor' for r in rows.values()),
                      binding=binding, splits=split_summary, indexed_total=sum(r['windows'] for r in rows.values()))
        if (report['eligible'], report['window_eligible'], report['legitimate_no_window']) != (args.expected_eligible, args.expected_window_eligible, args.expected_no_window):
            raise ValueError('Actual eligibility/window population differs from registered full population')
        assert_baseline(frozen)
        flush_json(status_handle, dict(state='probing', expected=len(tasks), completed=0, indexed_total=report['indexed_total']))
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('fork'), initializer=worker_init) as pool:
            futures = {pool.submit(probe_episode, task): task[1] for task in tasks}
            completed = 0
            for future in as_completed(futures):
                episode = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = dict(passed=False, first_last_passed=False, error=repr(error), probes=[])
                row = rows[episode]
                row.update(result)
                if not result['passed']:
                    report['failures'].append(dict(episode_index=episode, source_key=row['source_key'], error=result.get('error')))
                    print(json.dumps(dict(event='worker_failed', **report['failures'][-1])), flush=True)
                report['old_source_open_attempts'].extend(result.get('old_source_open_attempts', []))
                report['pipeline_imports'].extend(result.get('pipeline_imports', []))
                completed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    progress = dict(state='probing', completed=completed, expected=len(tasks), failures=len(report['failures']),
                                    elapsed_s=time.monotonic()-started, updated_at=datetime.now(timezone.utc).isoformat())
                    flush_json(status_handle, progress)
                    print(json.dumps(dict(event='reader_progress', **progress)), flush=True)
        for source in ('egosuite', 'egoexo4d_hm'):
            members = [r for r in rows.values() if r['source'] == source]
            source_summary[source] = dict(episodes=len(members), native_frames=sum(r['length'] for r in members),
                eligible=sum(r['eligible'] for r in members), window_eligible=sum(r['windows'] > 0 for r in members),
                legitimate_no_window=sum(r['exclusion_reason'] == 'no_usable_prefix_floor' for r in members),
                quarantined=sum(not r['eligible'] for r in members), indexed_windows=sum(r['windows'] for r in members))
        assert_baseline(frozen)
        report.update(sources=source_summary, episode_reports=[rows[i] for i in sorted(rows)],
                      old_source_open_attempts=report['old_source_open_attempts'] + GUARD.old_source_open_attempts,
                      pipeline_imports=report['pipeline_imports'] + GUARD.pipeline_imports,
                      actual_first_last_samples=sum(len(row.get('probes', [])) for row in rows.values()),
                      original_missing_feet_policy_preserved=True, quarantine_exclusion_preserved=True)
        report['passed'] = (not report['failures'] and not report['old_source_open_attempts'] and not report['pipeline_imports']
                            and all(row['passed'] for row in rows.values()))
    except Exception as error:
        report['failures'].append(dict(error=repr(error)))
        report['passed'] = False
        report['old_source_open_attempts'].extend(GUARD.old_source_open_attempts)
        report['pipeline_imports'].extend(GUARD.pipeline_imports)
        print(json.dumps(dict(event='reader_failed', error=repr(error))), flush=True)
    report.update(expected_episodes=args.expected_episodes, scanned_episodes=report['scanned'],
                  eligible_episodes=report['eligible'], window_episodes=report['window_eligible'],
                  legitimate_no_window_episodes=report['legitimate_no_window'], rows=report['episode_reports'])
    report.update(completed_at=datetime.now(timezone.utc).isoformat(), elapsed_s=time.monotonic()-started,
                  limitations=['Complete population indexing and first/last real samples per window-eligible episode; not execution of every indexed window.',
                               'Diagnostic allow_unpublished=True does not publish the archive or upgrade original failed fit acceptance.',
                               'This is an ME interface/dependency proof. Independent full parameter/source integrity and policy qualification are separate.'])
    flush_json(report_handle, report)
    flush_json(status_handle, dict(state='passed' if report['passed'] else 'failed', scanned=report['scanned'],
        eligible=report['eligible'], window_eligible=report['window_eligible'], failures=len(report['failures']),
        formal_training_ready=False, completed_at=report['completed_at']))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--workspace', default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument('--hm-root', default='/gaozt-test1/datasets/EgoExo-4D-HM')
    parser.add_argument('--blocked-root', action='append', default=[])
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--expected-episodes', type=int, default=9263)
    parser.add_argument('--expected-eligible', type=int, default=9216)
    parser.add_argument('--expected-window-eligible', type=int, default=9205)
    parser.add_argument('--expected-no-window', type=int, default=11)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    # Existing evidence is never silently replaced. Only these preopened output
    # descriptors remain writable after the old-parent dependency guard starts.
    report_fd = os.open(args.report, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    status_fd = os.open(args.status, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(report_fd, 'w') as report, os.fdopen(status_fd, 'w') as status:
        raise SystemExit(execute(args, report, status))


if __name__ == '__main__':
    main()
