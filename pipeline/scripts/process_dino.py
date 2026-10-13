"""Wait for an archive upgrade, then extract DINO with one process per GPU."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

from egomotion_pipeline.parallel_features import extract_features_parallel
from egomotion_pipeline.storage import Archive, atomic_json, read_json
from egomotion_pipeline.validation import validate
from egomotion_pipeline.dino_optimization import optimization_options


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--gpus', nargs='+', type=int, required=True)
    parser.add_argument('--wait-for-job', help='DINO format upgrade job directory')
    args = parser.parse_args(argv)
    job, root = Path(args.job_dir), Path(args.dataset).resolve()
    job.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    previous = read_json(job/'status.json') if (job/'status.json').exists() else {}
    state = {**previous, 'pid':os.getpid(), 'dataset':str(root), 'gpu_ids':args.gpus,
             'gpu_count':len(args.gpus), 'dino':'pending', 'workers':{}}

    def update(**fields):
        state.update(fields, updated_at=datetime.now(timezone.utc).isoformat(),
                     extraction_job_seconds=time.monotonic()-started)
        atomic_json(job/'status.json', state)

    def stop(signum, frame):
        raise InterruptedError('DINO extraction received a stop signal')

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        if args.wait_for_job:
            last = None
            while True:
                prerequisite = read_json(Path(args.wait_for_job)/'status.json')
                if Path(prerequisite['dataset']).resolve() != root:
                    raise ValueError('Upgrade job dataset differs from extraction dataset')
                stage = prerequisite['stage']
                update(stage='waiting_for_format_upgrade', upgrade_stage=stage,
                       upgraded_episodes=prerequisite.get('upgraded_episodes',0))
                if stage != last:
                    print('Waiting for DINO format upgrade:', stage, flush=True)
                    last = stage
                if stage == 'completed' and prerequisite.get('validation_passed'):
                    break
                if stage == 'failed':
                    raise RuntimeError('DINO format upgrade failed: '+str(prerequisite.get('fatal_error')))
                os.kill(prerequisite['pid'],0)
                time.sleep(15)
        config = read_json(args.config)
        update(batch_size=config.get('batch_size',32), optimization=optimization_options(config))
        archive = Archive(root)
        while True:
            try:
                with archive.lock():
                    records = archive.records()
                    frames = sum(r['length'] for r in records)
                    free = shutil.disk_usage(root).free
                    completed = [r for r in records if r['dino']['state']=='complete']
                    pending_frames = sum(r['length'] for r in records if r['dino']['state']!='complete')
                    if free < pending_frames*5120*4:
                        raise RuntimeError('Insufficient free space for float32 DINO features')
                    for record in records:
                        if not record.get('video') or not Path(record['video']['path']).is_file():
                            raise ValueError('Missing source video: '+record['source_key'])
                    atomic_json(job/'preflight.json', {'episodes':len(records), 'frames':frames,
                        'source_videos_exist':True, 'free_bytes':free,
                        'uncompressed_pending_dino_bytes':pending_frames*5120*4})
                    update(stage='loading_encoders', dino='running', total_episodes=len(records),
                           total_frames=frames, completed_episodes=len(completed),
                           completed_frames=sum(r['length'] for r in completed),
                           valid_frames=sum(r['dino']['valid_count'] for r in completed))
                    baseline = state['completed_frames']
                    extraction_started = time.monotonic()

                    def progress(event):
                        kind = event['kind']
                        if kind != 'heartbeat':
                            key = str(event['gpu_id'])
                            worker = state['workers'].setdefault(key,{})
                            if kind == 'worker_ready':
                                worker.update(pid=event['pid'], stage='ready')
                                print(f"GPU {key} encoder ready, pid={event['pid']}", flush=True)
                            elif kind == 'assigned':
                                worker.update(stage='extracting', episode_index=event['episode_index'],
                                              source_key=event['source_key'], length=event['length'],
                                              encoded_frames=0)
                            elif kind == 'batch':
                                worker.update(encoded_frames=event['encoded_frames'])
                            elif kind == 'committed':
                                worker.update(stage='ready', last_completed_episode=event['episode_index'])
                                state['completed_episodes'] += 1
                                state['completed_frames'] += event['length']
                                state['valid_frames'] += event['valid_count']
                        if len(state['workers']) == len(args.gpus):
                            state['stage']='extracting'
                        elapsed = time.monotonic()-extraction_started
                        rate=(state['completed_frames']-baseline)/max(elapsed,1e-6)
                        update(committed_frames_per_second=rate,
                               estimated_remaining_seconds=(frames-state['completed_frames'])/rate if rate else None)

                    print(f"Starting {len(args.gpus)} GPU workers: {args.gpus}, "
                          f"{len(records)} episodes, {frames} frames", flush=True)
                    extract_features_parallel(archive, config, args.gpus, progress=progress)
                    if any(r['dino']['state']!='complete' for r in archive.records()):
                        raise RuntimeError('DINO extraction left pending episodes')
                    update(stage='validating', dino='extracted')
                    report = validate(archive)
                    report.update(dino_completed_episodes=len(records),
                                  dino_valid_frames=state['valid_frames'],
                                  dino_missing_frames=frames-state['valid_frames'],
                                  dino_valid_fraction=state['valid_frames']/frames,
                                  allow_missing_video_frames=True)
                    atomic_json(job/'validation.json', report)
                break
            except RuntimeError as error:
                if str(error) != 'Another process is writing this archive':
                    raise
                update(stage='waiting_for_archive_lock')
                time.sleep(15)
        update(stage='completed', dino='complete', validation_passed=True,
               finished_at=datetime.now(timezone.utc).isoformat())
        print(f"Completed: {state['completed_episodes']} episodes, "
              f"{state['valid_frames']}/{frames} valid DINO frames", flush=True)
        return 0
    except BaseException as error:
        cancelled = isinstance(error,(InterruptedError,KeyboardInterrupt))
        update(stage='cancelled' if cancelled else 'failed', fatal_error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
