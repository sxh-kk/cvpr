"""Finish one frozen Grasp100 cohort locally; never upload or start training."""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

from process_pico_dataset import ProcessingDeferred, require_idle, reusable_admission, write

TASK = 'cvpr/G1SonicTabletopGraspPico-v0'
CONTRACT = 'cvpr-grasp100-local-processing-plan-v1'
SCRIPT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(value, message):
    if not value:
        raise ValueError(message)


def record(plan, paths, message):
    """Runtime derivatives also appear in the user's single project log."""
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S +08:00')
    project = Path(plan['project_log']).parent
    labels = '、'.join('`'+os.path.relpath(path, project)+'`' for path in paths)
    with Path(plan['project_log']).open('a') as stream:
        stream.write(f'\n- {stamp} | {labels}：{message}\n')


def frozen_cohort(plan):
    require(plan.get('contract') == CONTRACT and plan.get('task_id') == TASK,
            'Only the explicit local Grasp100 plan is supported')
    require(plan.get('training_started') is False and plan.get('upload_authorized') is False,
            'This entry point has no training or upload stage')
    path = Path(plan['frozen_manifest']).resolve(strict=True)
    require(sha(path) == plan['frozen_manifest_sha256'], 'Frozen cohort changed')
    frozen = read(path)
    require(frozen.get('contract') == 'cvpr-grasp100-rtx-admission-frozen-inputs-v1'
            and frozen.get('count') == 100 and len(frozen.get('episodes', [])) == 100,
            'Require the complete frozen100 manifest')
    rows = frozen['episodes']
    require({row['official_episode_index'] for row in rows} == set(range(100))
            and len({row['uuid'] for row in rows}) == 100
            and len({row['directory'] for row in rows}) == 100,
            'Frozen cohort must have one distinct human UUID per official initialization')
    require(all(row['number'] != 7 for row in rows)
            and next(row for row in rows if row['official_episode_index'] == 3)['number'] == 126,
            'The reviewed replacement126 must replace old007')
    return frozen


def verify_originals(frozen):
    checked = 0
    for row in frozen['episodes']:
        root = Path(row['directory'])
        require(root.is_dir() and not root.is_symlink(), 'Missing frozen original directory')
        for name, expected in row['source_files'].items():
            rel = Path(name)
            require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe original filename')
            path = root/rel
            require(path.is_file() and not path.is_symlink(), 'Original evidence missing: '+str(path))
            stat = path.stat()
            require(stat.st_size == expected['size_bytes'] and stat.st_mtime_ns == expected['mtime_ns'],
                    'Original evidence size/mtime changed: '+str(path))
            if expected.get('sha256'):
                require(sha(path) == expected['sha256'], 'Original evidence bytes changed: '+str(path))
            checked += 1
        metadata = read(root/'metadata.json')
        require(metadata.get('task_id') == TASK and metadata.get('human_demo_uuid') == row['uuid']
                and metadata['official_initialization']['official_episode_index'] == row['official_episode_index'],
                'Original UUID/task/initialization differs from the frozen cohort')
    return checked


def run_child(plan, argv, name, *, gpu=False):
    require_idle()
    release = Path(plan['release'])
    require(shutil.disk_usage(release).free >= 2 * 1024**3, 'Free disk below2GiB; preserve partial outputs')
    log = release/'logs'/(name+'.log')
    require(not log.exists(), 'Existing stage log is preserved; choose a new run identifier')
    environment = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if not gpu:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    with log.open('x') as output:
        child = subprocess.Popen(list(map(str, argv)), cwd=SCRIPT.parents[1], env=environment,
                                 stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        exit_code = child.wait()
    affected = [log, release]
    if any(Path(str(argument)).name == 'render_deferred_pico_batch.py' for argument in argv):
        arguments = list(map(str, argv))
        batch_report = Path(arguments[arguments.index('--report')+1]).resolve(strict=True)
        exit_path = batch_report.with_name(batch_report.stem+'_EXIT.json')
        require(not exit_path.exists(), 'Preserve the previous independently observed process-exit receipt')
        write(exit_path, {'contract': 'cvpr-grasp-rtx-observed-process-exit-v1', 'exit_code': exit_code,
            'child_pid': child.pid, 'finished_beijing': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'report_path': str(batch_report), 'report_sha256': sha(batch_report), 'argv': arguments,
            'parentdriver_path': str(Path(__file__).resolve()), 'parentdriver_sha256': sha(__file__)})
        affected.append(exit_path)
        affected.extend(sorted({Path(row['directory']).parent
            for row in read(plan['frozen_manifest'])['episodes']}))
    record(plan, affected, f'Grasp本地处理阶段{name}完成，退出码{exit_code}；保留阶段产物，未训练或上传。')
    require(exit_code == 0, f'{name} failed with exit{exit_code}; see{log}')


def process(plan_path):
    plan_path = Path(plan_path).resolve(strict=True)
    plan = read(plan_path)
    frozen = frozen_cohort(plan)
    require_idle()
    release = Path(plan['release']).absolute()
    require(not release.is_symlink(), 'Release must be independent')
    release.mkdir(exist_ok=True)
    (release/'logs').mkdir(exist_ok=True)
    with (release/'.processing.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = release/'PROCESSING_STATUS.json'
        status = read(status_path) if status_path.exists() else {'episodes': {}}
        run_id = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')

        def status_update(stage, **extra):
            status.update(status=stage, updated_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                          training_started=False, upload_started=False, **extra)
            write(status_path, status)
            record(plan, [status_path], '更新Grasp本地补图与准入进度：'+stage+'。')

        try:
            binding = release/'PROCESSING_PLAN.json'
            if binding.exists():
                require(read(binding) == plan, 'Existing release belongs to another plan')
            else:
                write(binding, plan)
                record(plan, [binding, release/'.processing.lock', release/'logs'], '保存Grasp100本地处理计划与独占处理锁。')
            count = verify_originals(frozen)
            status_update('rendering_rtx', original_files_verified=count)
            render_report = release/('RTX_BATCH_'+run_id+'.json')
            run_child(plan, [plan['simulation_python'], SCRIPT/'render_deferred_pico_batch.py',
                '--manifest', plan['frozen_manifest'], '--data-root', plan['render_data_root'],
                '--report', render_report], 'rtx_'+run_id, gpu=True)
            verify_originals(frozen)
            status_update('independent_admission')
            from pico_grasp_audit import audit_episode
            selection = {'contract': 'cvpr-pico-grasp-admissions-selection-v1', 'task_id': TASK,
                'selection_policy': 'native_only', 'candidate_cohort': {'candidate_count': 100,
                    'frozen_inventory_sha256': plan['frozen_manifest_sha256']}, 'episodes': []}
            for row in frozen['episodes']:
                directory = Path(row['directory'])
                admission = reusable_admission(directory)
                if admission is None:
                    previous = directory/'admission.json'
                    if previous.exists():
                        backup = release/'logs'/('prior_admission_'+row['uuid']+'_'+sha(previous)+'.json')
                        if not backup.exists():
                            shutil.copyfile(previous, backup)
                            record(plan, [backup], '保留原独立准入回执，按当前检查器重新核验。')
                    admission = audit_episode(directory, historical_source_manifest=plan['historical_sources'])
                    record(plan, [directory/'admission.json'], f'完成Grasp片段{row["number"]:03d}的独立CPU准入检查。')
                status['episodes'][row['uuid']] = {'number': row['number'],
                    'eligible_for_packing': admission.get('eligible_for_packing'),
                    'errors': admission.get('errors', [])}
                status_update('independent_admission', active_number=row['number'])
                require(admission.get('contract') == 'cvpr-live-pico-grasp-independent-admission-v1'
                        and admission.get('task_id') == TASK and admission.get('eligible_for_packing') is True,
                        'Grasp admission failed; preserve evidence and diagnose: '+str(admission.get('errors')))
                source_path = Path(read(directory/'collection_record.json')['source_episode_directory'])
                selection['episodes'].append({'human_demo_uuid': row['uuid'], 'episode_dir': str(directory),
                    'source_session': source_path.parent.parent.name, 'number': row['number'],
                    'admission_path': str(directory/'admission.json'),
                    'admission_sha256': sha(directory/'admission.json')})
            selection_path = release/'ADMISSIONS_SELECTION.json'
            if selection_path.exists():
                require(read(selection_path) == selection, 'Existing selection differs; preserve it for diagnosis')
            else:
                write(selection_path, selection)
                record(plan, [selection_path], '冻结全部100条通过独立准入的Grasp真人UUID，排除旧007、包含126。')
            status_update('packing_lossless')
            if not (release/'packed/PACK_READY.json').exists():
                run_child(plan, [plan['model_python'], SCRIPT/'pack_pico_handover.py', '--selection',
                    selection_path, '--template', plan['template'], '--output', release/'packed'], 'pack_'+run_id)
            require(read(release/'packed/PACK_READY.json')['selection']['sha256'] == sha(selection_path),
                    'Pack differs from frozen passed selection')
            status_update('extracting_clean_dino')
            if not (release/'dino/manifest.json').is_file() or read(release/'dino/manifest.json').get('state') != 'complete':
                run_child(plan, [plan['model_python'], SCRIPT.parents[1]/'VLMExpert/scripts/data/extract_unifolm_dino.py',
                    '--dataset-root', release/'packed/train', '--output', release/'dino', '--model-path', plan['dino_model'],
                    '--shared-image-profile', '256x320', '--device', 'cuda:0', '--batch-size', '32', '--workers', '1',
                    '--decoder-threads', '1', '--cpu-threads', '2', '--storage-dtype', 'float32',
                    '--inference-precision', 'bf16'], 'dino_'+run_id, gpu=True)
            status_update('checking_production_reader')
            ready_path = release/'PICO_DATASET_READY.json'
            if ready_path.exists():
                cached = read(ready_path)
                from verify_pico_handover import identity
                for relative, wanted in cached['payload_files'].items():
                    rel = Path(relative)
                    require(not rel.is_absolute() and '..' not in rel.parts,
                            'Unsafe cached readiness payload')
                    require(identity(release/rel) == wanted, 'Cached production payload changed: '+relative)
                for original, wanted in cached['external_reference_files'].items():
                    require(identity(Path(original)) == wanted, 'Cached production reference changed: '+original)
                require(cached['verifier_sha256'] == sha(SCRIPT/'verify_pico_handover.py'),
                        'Cached readiness checker changed; preserve and diagnose its receipt')
            else:
                run_child(plan, [plan['model_python'], SCRIPT/'verify_pico_handover.py', '--pack-root', release/'packed',
                    '--dino-root', release/'dino', '--reference-dino-root', plan['reference_dino'],
                    '--run-config', plan['run_config'], '--output', ready_path, '--expected-episodes', '100'],
                    'reader_'+run_id)
            ready = read(ready_path)
            require(ready.get('status') == 'passed' and ready.get('task_id') == TASK
                    and ready.get('train_episodes') == 100
                    and ready.get('production_ready_for_bound_pack_and_cache') is True
                    and ready.get('training_started') is False, 'Require actual local Grasp100 production readiness')
            count = verify_originals(frozen)
            status_update('local_training_data_ready_training_not_started', accepted_episodes=100,
                train_windows=ready['train_windows'], ready_sha256=sha(ready_path), original_files_verified=count)
            return status
        except ProcessingDeferred as error:
            status_update('waiting_for_collection_or_gpu', error=str(error), data_rejected=False)
            raise
        except BaseException as error:
            status_update('technical_failure_needs_diagnosis', error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = process(args.plan)
    except ProcessingDeferred as error:
        print(json.dumps({'status': 'waiting_for_collection_or_gpu', 'reason': str(error),
                          'data_rejected': False, 'training_started': False}, ensure_ascii=False))
        raise SystemExit(75)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
