#!/usr/bin/env python3
"""Wait for the frozen conversion job, then upgrade pending DINO placeholders."""

import argparse
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from egomotion_pipeline.migration import upgrade_dino
from egomotion_pipeline.storage import Archive, atomic_json, read_json
from egomotion_pipeline.validation import validate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--wait-for-job', help='Conversion job directory containing status.json')
    args = parser.parse_args(argv)
    job = Path(args.job_dir)
    job.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    state = {'stage':'starting', 'pid':os.getpid(), 'dataset':str(Path(args.dataset).resolve()),
             'dino':'pending', 'wait_for_job':args.wait_for_job}
    def update(**fields):
        state.update(fields, updated_at=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=time.monotonic()-started)
        atomic_json(job/'status.json', state)
    try:
        if args.wait_for_job:
            status_path = Path(args.wait_for_job)/'status.json'
            while True:
                conversion = read_json(status_path)
                if Path(conversion['output']).resolve() != Path(args.dataset).resolve():
                    raise ValueError('Conversion job output differs from upgrade dataset')
                stage = conversion['stage']
                update(stage='waiting_for_conversion', conversion_stage=stage,
                       conversion_processed_files=conversion.get('processed_files'),
                       conversion_failed_files=conversion.get('failed_files'))
                if stage in {'completed', 'completed_with_errors'}:
                    if not conversion.get('validation_passed'):
                        raise ValueError('Conversion completed without successful validation')
                    break
                if stage == 'failed':
                    raise RuntimeError('Conversion failed; no DINO upgrade was started')
                try:
                    os.kill(conversion['pid'], 0)
                except ProcessLookupError:
                    raise RuntimeError('Conversion process exited before completion; resume it before upgrading')
                time.sleep(15)
        update(stage='upgrading')
        print('Upgrading pending DINO placeholders; motion columns are preserved', flush=True)
        def progress(journal):
            count = journal['completed_episodes']
            update(stage='upgrading', upgraded_episodes=count, total_episodes=journal['total_episodes'])
            if count == 1 or count % 25 == 0 or count == journal['total_episodes']:
                print(f"[{count}/{journal['total_episodes']}] DINO placeholder upgraded", flush=True)
        while True:
            try:
                result = upgrade_dino(args.dataset, progress=progress)
                break
            except RuntimeError as error:
                if not args.wait_for_job or str(error) != 'Another process is writing this archive':
                    raise
                update(stage='waiting_for_archive_lock')
                time.sleep(15)
        update(stage='validating', result=result)
        # Keep the lock through final validation so later writers cannot race it.
        archive = Archive(args.dataset)
        with archive.lock():
            report = validate(archive)
            atomic_json(job/'validation.json', report)
        update(stage='completed', validation_passed=True)
        print(f"completed: {report['episodes']} episodes, {report['frames']} frames; DINO pending", flush=True)
        return 0
    except BaseException as error:
        update(stage='failed', fatal_error=f'{type(error).__name__}: {error}')
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
