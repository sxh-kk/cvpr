"""Explicit frozen-UUID Mobile native reselection, with recoverable evidence.

snapshot reads only the named completed records. apply is the only command
that reclassifies directories; it preserves original data, numbers and old
decisions. Neither command starts a collector, SDK, model or GPU job.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import uuid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu', type=int, default=24, help='Own CPU affinity; must be in the inherited allowed set')
    commands = parser.add_subparsers(dest='command', required=True)
    snapshot = commands.add_parser('snapshot', help='Freeze exactly these already-numbered Mobile UUIDs')
    snapshot.add_argument('--index', type=Path, required=True)
    snapshot.add_argument('--uuid', action='append', required=True)
    snapshot.add_argument('--output', type=Path, required=True)
    apply = commands.add_parser('apply', help='Apply one explicit frozen snapshot; preserve old evidence')
    apply.add_argument('--index', type=Path, required=True)
    apply.add_argument('--snapshot', type=Path, required=True)
    apply.add_argument('--operation-uuid', required=True)
    apply.add_argument('--output', type=Path, required=True)
    apply.add_argument('--refresh-review-cache', action='store_true')
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error('Output already exists; choose a new receipt path')
    allowed = os.sched_getaffinity(0)
    if args.cpu not in allowed:
        parser.error('Requested diagnostic CPU is outside the inherited affinity')
    os.sched_setaffinity(0, {args.cpu})
    os.nice(max(0, 19 - os.getpriority(os.PRIO_PROCESS, 0)))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        os.environ[name] = '1'
    from pico_episode_history import EpisodeHistory
    history = EpisodeHistory(args.index)
    if args.command == 'snapshot':
        result = history.snapshot_mobile_native(args.uuid)
    else:
        operation = str(uuid.UUID(args.operation_uuid))
        if operation != args.operation_uuid:
            parser.error('Use a canonical operation UUID')
        result = history.reselect_mobile_native(json.loads(args.snapshot.read_text()), operation_uuid=operation)
        if args.refresh_review_cache and result.get('status') == 'committed':
            from pico_episode_review import ReviewQueue
            try:
                refresh = ReviewQueue(args.index).refresh_reselection_results(result['selected_uuids'], revision_id=operation)
            except Exception as error:
                refresh = {'status': 'error', 'error': str(error),
                           'scope': 'History reselection already committed; review cache refresh remains pending'}
            result = dict(result, review_cache_refresh=refresh)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({'output': str(args.output), 'status': result.get('status', 'snapshot_created')}, ensure_ascii=False))
    return 1 if result.get('status') == 'error' or result.get('review_cache_refresh', {}).get('status') == 'error' else 0


if __name__ == '__main__':
    raise SystemExit(main())
