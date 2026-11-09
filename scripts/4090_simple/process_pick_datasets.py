"""Resume the frozen299 pick cohort through local RTX, admission and readiness.

Run this driver with SIM310. Audit/compaction stay in that exact environment;
packing, DINO and production readers use the separately frozen model Python.
No upload, training, raw-data deletion or automatic overwrite is implemented.
"""
from __future__ import annotations

import argparse
from collections import Counter
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

from process_pico_dataset import ProcessingDeferred, require_idle, write

CONTRACT = 'cvpr-pico-pick299-local-processing-plan-v1'
FROZEN_CONTRACT = 'cvpr-pico-pick-rtx-frozen-inputs-v1'
FROZEN_SHA256 = 'c8991ce63cfcce24e5e2b357ad4b06df03214a339a46bc46551153ae94583f2d'
SCRIPT = Path(__file__).resolve().parent
DEFAULT_ROOT = Path('/home/zxc/CVPR_Evaluation_runtime/pico_latest300_20261109/processing_001')
RUNTIME = Path('/home/zxc/CVPR_Evaluation_runtime')
SIMULATION_PYTHON = Path('/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python')
MODEL_PYTHON = Path('/home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python')
RESERVE_BYTES = 2 * 1024**3
FRAME_BUDGET_BYTES = 704 * 1024  # 640x360 RGB plus lossless codec/container overhead.
EXPECTED_COUNTS = {'bend_pick': 100, 'xmove_pick': 99, 'xmove_bend_pick': 100}
TASKS = {'bend_pick': 'cvpr/G1SonicBendPickPico-v0', 'xmove_pick': 'cvpr/G1SonicXMovePickPico-v0',
    'xmove_bend_pick': 'cvpr/G1SonicXMoveBendPickPico-v0'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda word:
        (_ for _ in ()).throw(ValueError('Nonfinite JSON ' + word)))


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            value.update(block)
    return value.hexdigest()


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def record(plan, paths, message):
    log = Path(plan['project_log'])
    labels = '、'.join('`' + os.path.relpath(path, log.parent) + '`' for path in paths)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S +08:00')
    with log.open('a') as stream:
        stream.write(f'- {stamp} | {labels}：{message}\n')


def frozen_cohort(plan):
    require(plan.get('contract') == CONTRACT and plan.get('training_started') is False and
            plan.get('upload_authorized') is False, 'Only frozen local pick299 processing is supported')
    path = Path(plan['frozen_manifest']).resolve(strict=True)
    require(plan.get('frozen_manifest_sha256') == FROZEN_SHA256 and sha(path) == FROZEN_SHA256,
            'The explicitly reviewed299 manifest changed')
    frozen = read(path)
    rows = frozen.get('episodes', [])
    require(frozen.get('contract') == FROZEN_CONTRACT and frozen.get('candidate_count') == 300 and
            frozen.get('count') == len(rows) == 299 and frozen.get('expected_task_counts') == EXPECTED_COUNTS and
            dict(Counter(row['profile'] for row in rows)) == EXPECTED_COUNTS and
            len({row['uuid'] for row in rows}) == 299 and len({row['directory'] for row in rows}) == 299,
            'Processing task counts, UUIDs or directory identities differ')
    candidate_path = Path(frozen['candidate_manifest_path']).resolve(strict=True)
    require(sha(candidate_path) == frozen['candidate_manifest_sha256'], 'Original300 candidate manifest changed')
    candidates = read(candidate_path)
    original = {row['uuid']: row for row in candidates.get('episodes', [])}
    excluded = frozen.get('excluded_episodes', [])
    require(candidates.get('contract') == FROZEN_CONTRACT and candidates.get('count') == len(original) == 300 and
            len(excluded) == 1 and excluded[0].get('task_id') == TASKS['xmove_pick'] and excluded[0].get('number') == 4 and
            excluded[0].get('excluded_from_processing') is True and bool(excluded[0].get('reason')) and
            set(original) - {row['uuid'] for row in rows} == {excluded[0]['uuid']} and
            all(original.get(row['uuid']) == row for row in rows),
            'Only the explicit warmup-only XMove004 exclusion may differ from candidate300')
    for row in rows:
        require(TASKS.get(row['profile']) == row['task_id'] and
                not (row['profile'] == 'xmove_pick' and row['number'] == 4), 'Excluded or aliased task entered processing')
    require(sha(plan['historical_sources']) == plan['historical_sources_sha256'], 'Historical source identities changed')
    return frozen


def verify_originals(rows):
    checked = 0
    for row in rows:
        root = Path(row['directory']).resolve(strict=True)
        require(root.is_dir(), 'Missing frozen original directory')
        for name, expected in row['source_files'].items():
            relative = Path(name)
            require(relative.parts and not relative.is_absolute() and '..' not in relative.parts, 'Unsafe frozen source path')
            path = root/relative
            require(path.is_file() and not path.is_symlink(), 'Missing original evidence: ' + str(path))
            before = path.stat()
            require(before.st_size == expected['size_bytes'] and before.st_mtime_ns == expected['mtime_ns'] and
                    sha(path) == expected['sha256'], 'Frozen original bytes/mtime changed: ' + str(path))
            after = path.stat()
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Original changed while verifying')
            checked += 1
        metadata = read(root/'metadata.json')
        require(metadata.get('task_id') == row['task_id'] and metadata.get('human_demo_uuid') == row['uuid'],
                'Frozen original task/UUID differs')
    return checked


def code_identities():
    from pico_pick_audit import AUDITOR_FILES
    names = set(AUDITOR_FILES) | {'process_pick_datasets.py', 'run_pick_rtx.py', 'render_deferred_pick_batch.py',
        'render_deferred_pico.py', 'pico_pick_rgb_storage.py', 'pack_pico_handover.py', 'verify_pico_handover.py',
        'pack_handover_quality100.py', 'verify_handover_supplement.py', 'pico_episode_review.py',
        'process_pico_dataset.py', 'pico_simple_teleop.py', 'pico_xmove_pick_task.py',
        'pico_bend_pick_task.py', 'pico_xmove_bend_pick_task.py'}
    paths = [SCRIPT/name for name in sorted(names)]
    paths.append(SCRIPT.parents[1]/'VLMExpert/scripts/data/extract_unifolm_dino.py')
    return {str(path.resolve(strict=True)): sha(path) for path in paths}


def pack_dependency_identities(root):
    dependency = Path(root)/'dependencies/pillow123_model311'
    require((dependency/'PIL/__init__.py').is_file(), 'Missing private MODEL311 Pillow12.3 dependency')
    paths = [path for path in dependency.rglob('*') if path.is_file() and
        '__pycache__' not in path.parts and path.suffix != '.pyc']
    require(paths and not any(path.is_symlink() for path in paths), 'Unsafe private packing dependency')
    return {str(path.resolve(strict=True)): sha(path) for path in sorted(paths)}


def build_plan(processing_root=DEFAULT_ROOT, *, release_root=None, model_python=MODEL_PYTHON):
    root = Path(processing_root).resolve(strict=True)
    base = read(RUNTIME/'pico_grasp100_20261107_human01/PROCESSING_PLAN.json')
    plan = {key: base[key] for key in ('template', 'dino_model', 'reference_dino', 'run_config')}
    plan.update(contract=CONTRACT, processing_root=str(root),
        release_root=str(Path(release_root or root/'releases').absolute()),
        frozen_manifest=str(root/'FROZEN299_PROCESSING.json'), frozen_manifest_sha256=FROZEN_SHA256,
        historical_sources=str(root/'HISTORICAL_SOURCE_IDENTITIES.json'),
        historical_sources_sha256=sha(root/'HISTORICAL_SOURCE_IDENTITIES.json'),
        simulation_python=str(SIMULATION_PYTHON), model_python=str(Path(model_python).resolve(strict=True)),
        project_log='/home/zxc/CVPR2027/log.md', max_chunk=20,
        estimated_png_bytes_per_frame=FRAME_BUDGET_BYTES, reserve_free_bytes=RESERVE_BYTES,
        task_order=['bend_pick', 'xmove_pick', 'xmove_bend_pick'],
        expected_task_counts=EXPECTED_COUNTS, training_started=False, upload_authorized=False,
        pack_pythonpath=os.pathsep.join((str(root/'dependencies/pillow123_model311'), str(SCRIPT))),
        pack_dependency_sha256=pack_dependency_identities(root), code_sha256=code_identities())
    frozen_cohort(plan)
    return plan


def verify_plan(plan):
    require(Path(sys.executable).resolve() == Path(plan['simulation_python']).resolve(),
            'Execute this driver with its declared SIM310 Python; audit/storage must keep the original PNG environment')
    require(type(plan.get('max_chunk')) is int and 1 <= plan['max_chunk'] <= 20 and
            type(plan.get('reserve_free_bytes')) is int and plan['reserve_free_bytes'] >= RESERVE_BYTES and
            type(plan.get('estimated_png_bytes_per_frame')) is int and plan['estimated_png_bytes_per_frame'] >= FRAME_BUDGET_BYTES and
            plan.get('task_order') == ['bend_pick', 'xmove_pick', 'xmove_bend_pick'] and
            plan.get('expected_task_counts') == EXPECTED_COUNTS, 'Processing order/capacity bounds differ')
    require(plan.get('code_sha256') == code_identities(), 'Processing implementations changed; retain current derivatives and review before resume')
    require(plan.get('pack_pythonpath') == os.pathsep.join((str(Path(plan['processing_root'])/'dependencies/pillow123_model311'), str(SCRIPT))) and
            plan.get('pack_dependency_sha256') == pack_dependency_identities(plan['processing_root']),
            'Private MODEL311 packing PNG environment changed')
    for name in ('model_python', 'template', 'dino_model', 'reference_dino', 'run_config'):
        require(Path(plan[name]).exists(), 'Missing processing dependency: ' + name)


def choose_chunk(rows, offset, *, free_bytes, maximum=20, bytes_per_frame=FRAME_BUDGET_BYTES, reserve=RESERVE_BYTES):
    require(type(offset) is int and 0 <= offset < len(rows) and 1 <= maximum <= 20 and reserve >= RESERVE_BYTES,
            'Invalid adaptive chunk limits')
    used, selected = 0, []
    for row in rows[offset:offset+maximum]:
        root = Path(row['directory']).resolve(strict=True)
        if (root/'capture').exists() or (root/'rtx_completion.json').exists():
            break  # Existing/partial outputs are handled separately, never overwritten.
        frames = row.get('raw30_frames')
        require(type(frames) is int and frames > 0, 'Frozen original30Hz frame count missing')
        estimate = frames * bytes_per_frame
        # The first episode's master coexists with this chunk's PNGs.
        workspace = max([estimate] + [item['raw30_frames'] * bytes_per_frame for item in selected])
        if used + estimate + workspace + reserve > free_bytes:
            break
        selected.append(row); used += estimate
    if not selected:
        raise ProcessingDeferred('No safe RTX chunk fits with the2GiB reserve; existing files are preserved')
    return selected


def _storage_proof(directory):
    import pico_pick_rgb_storage as storage
    result = storage.verify_compacted_episode(directory)
    receipt = read(directory/'capture/rgb_storage.json')
    pngs = {str(directory/'capture'/entry['path']): entry['file_sha256'] for entry in receipt['png_files']}
    require(result.get('passed') is True and len(pngs) == result['frames'], 'Incomplete exact-PNG storage proof')
    return result, pngs


def reusable_admission(row):
    """Authenticate old admission plus exact missing PNGs, without re-auditing.

    Missing PNGs are legal only through a full current storage proof that
    recreates every originally admitted PNG byte. No authority/hash mismatch
    silently falls back to the auditor after compaction.
    """
    from pico_pick_audit import AUDITOR_FILES, BASE_CHECKS, CONTRACT as ADMISSION
    root = Path(row['directory']).resolve(strict=True)
    path = root/'admission.json'
    if not path.exists():
        require(not (root/'capture/rgb_storage.json').exists(), 'Compacted output has no original admission; preserve for diagnosis')
        return None
    require(path.is_file() and not path.is_symlink(), 'Unsafe previous admission')
    report = read(path)
    require(report.get('contract') == ADMISSION and report.get('task_id') == row['task_id'] and
            report.get('human_demo_uuid') == row['uuid'] and report.get('episode_directory') == str(root) and
            report.get('eligible_for_packing') is True and report.get('status') == 'independent_live_human_episode_audit_passed' and
            report.get('selection_policy') == 'native_only' and report.get('technical_blocked') is False and
            report.get('errors') == [] and report.get('accepted_new_dataset') is False,
            'Existing admission is rejected/stale; preserve it and diagnose instead of overwriting')
    checks = report.get('checks', {})
    require(BASE_CHECKS <= set(checks) and all(value.get('passed') is True for value in checks.values()) and
            report.get('pick_supervised_coverage', {}).get('supervised_native_success_observed') is True,
            'Previous admission lacks strict supervised success coverage')
    authorities = report.get('auditor_source_sha256', {})
    require({Path(name).name for name in authorities} == AUDITOR_FILES and
            all(sha(SCRIPT/Path(name).name) == expected for name, expected in authorities.items()),
            'Independent auditor changed; restore original PNGs explicitly before a new audit')
    pngs = {}
    if (root/'capture/rgb_storage.json').exists():
        _, pngs = _storage_proof(root)
    bindings = report.get('bindings', {})
    require(bool(bindings), 'Prior admission has no original evidence bindings')
    for name, expected in bindings.items():
        original = Path(name)
        require(original.is_absolute(), 'Admission binding must be an absolute original path')
        if original.is_file():
            require(sha(original) == expected, 'Prior admission input changed: ' + name)
        else:
            require(original.suffix == '.png' and pngs.get(name) == expected,
                    'Missing admitted evidence has no exact current PNG proof: ' + name)
    return report


def admit_and_compact(plan, row):
    import pico_pick_audit as auditor
    import pico_pick_rgb_storage as storage
    root = Path(row['directory']).resolve(strict=True)
    verify_originals([row])
    admission = reusable_admission(row)
    reused = admission is not None
    if admission is None:
        require((root/'rtx_completion.json').is_file() and (root/'capture/capture.json').is_file(),
                'Missing complete RTX; partial outputs are preserved')
        admission = auditor.audit_episode(root, historical_source_manifest=plan['historical_sources'])
        record(plan, [root/'admission.json'], f'独立核验{row["profile"]}原编号{row["number"]:03d}的RGB、物理及监督成功覆盖。')
        require(admission.get('eligible_for_packing') is True, 'Pick admission failed: ' + str(admission.get('errors')))
    complete = root/'capture/rgb_storage_complete.json'
    if complete.exists() and read(complete).get('derived_pngs_removed') is True:
        proof, _ = _storage_proof(root)
        frames = read(root/'capture/rgb_storage.json')['png_files']
        if any((root/'capture'/frame['path']).exists() for frame in frames):
            proof = storage.compact_episode(root, delete_pngs=True)
    else:
        if not (root/'capture/rgb_master.mp4').exists() and shutil.disk_usage(root).free < (
                plan['reserve_free_bytes'] + row['raw30_frames'] * plan['estimated_png_bytes_per_frame']):
            raise ProcessingDeferred('Insufficient master workspace above the2GiB reserve; preserve admitted PNGs')
        proof = storage.compact_episode(root, delete_pngs=True)
    record(plan, [root/'capture'], f'{row["profile"]}原编号{row["number"]:03d}通过独立准入并验证无损master可逐字节复现全部PNG，回收派生PNG；复用既有准入={reused}，原始物理未改。')
    storage.close_readers()
    verify_originals([row])
    return {'uuid': row['uuid'], 'profile': row['profile'], 'number': row['number'],
        'stage': 'admitted_and_compacted', 'admission_sha256': sha(root/'admission.json'),
        'storage_receipt_sha256': sha(root/'capture/rgb_storage.json'),
        'storage_complete_sha256': sha(complete), 'master_sha256': sha(root/'capture/rgb_master.mp4'),
        'reused_admission': reused, 'storage': proof}


def available_bytes(paths):
    """Check the actual destination devices, including not-yet-created outputs."""
    devices = {}
    for path in paths:
        parent = Path(path).absolute()
        while not parent.exists():
            parent = parent.parent
        parent = parent.resolve(strict=True)
        devices[parent.stat().st_dev] = shutil.disk_usage(parent).free
    return min(devices.values())


def run_child(plan, argv, name, *, gpu=False, packing=False, required_extra_bytes=0, capacity_paths=()):
    require_idle()
    root = Path(plan['processing_root'])
    arguments = list(map(str, argv))
    destinations = [root, *capacity_paths]
    for option in ('--output', '--report', '--log'):
        if option in arguments:
            destinations.append(Path(arguments[arguments.index(option)+1]))
    require(type(required_extra_bytes) is int and required_extra_bytes >= 0, 'Invalid stage capacity estimate')
    if available_bytes(destinations) < plan['reserve_free_bytes'] + required_extra_bytes:
        raise ProcessingDeferred('Insufficient stage capacity above the2GiB reserve; preserving all partial outputs')
    log = root/'logs'/(name+'.log')
    require(not log.exists(), 'Preserve previous stage log: ' + str(log))
    environment = dict(os.environ, OMNI_KIT_ACCEPT_EULA='YES', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TMPDIR='/dev/shm')
    if not gpu:
        environment['CUDA_VISIBLE_DEVICES'] = ''
    if packing:
        environment['PYTHONPATH'] = plan['pack_pythonpath']
    with log.open('x') as output:
        child = subprocess.Popen(list(map(str, argv)), cwd=SCRIPT.parents[1], env=environment,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        code = child.wait()
    affected = [log]
    for option in ('--output', '--report', '--log'):
        if option in arguments:
            affected.append(Path(arguments[arguments.index(option)+1]))
    record(plan, affected, f'完成pick处理阶段{name}，退出码{code}；未上传或启动训练，失败产物原样保留。')
    require(code == 0, f'{name} exited{code}; preserve outputs and inspect {log}')
    if available_bytes(destinations) < plan['reserve_free_bytes']:
        raise ProcessingDeferred('Stage finished below the2GiB reserve; retain outputs and wait for space')


def build_selection(plan, profile, rows):
    selected = {'contract': 'cvpr-pico-pick-admissions-selection-v1', 'task_id': TASKS[profile],
        'selection_policy': 'native_only', 'candidate_cohort': {'candidate_count': 300, 'processing_count': 299,
            'task_processing_count': len(rows), 'frozen_inventory_sha256': plan['frozen_manifest_sha256']}, 'episodes': []}
    for row in rows:
        root = Path(row['directory']).resolve(strict=True)
        reusable_admission(row)
        source = Path(read(root/'collection_record.json')['source_episode_directory'])
        selected['episodes'].append({'human_demo_uuid': row['uuid'], 'episode_dir': str(root),
            'source_session': source.parent.parent.name, 'number': row['number'],
            'admission_path': str(root/'admission.json'), 'admission_sha256': sha(root/'admission.json')})
    return selected


def pack_capacity_bytes(rows, output, template):
    """Budget actual copied authorities; count a hardlink only on its own device.

    Use the packer's frozen allowlist and shared destination identities. Source
    stat sizes cover all copied evidence, while parquet and per-episode JSON /
    directory overhead retain a conservative fixed allowance.
    """
    import pack_pico_handover as packer
    parent = Path(output).absolute()
    while not parent.exists():
        parent = parent.parent
    device = parent.stat().st_dev
    copied, destinations = 0, set()
    def add(path, destination, *, link=False):
        nonlocal copied
        if destination in destinations:
            return
        destinations.add(destination)
        source = Path(path)
        require(source.is_file() and not source.is_symlink(), 'Unsafe pack capacity source: ' + str(source))
        stat = source.stat()
        if not (link and stat.st_dev == device):
            copied += ((stat.st_size + 4095) // 4096) * 4096
    for name in packer.SOURCE_FILES:
        add(Path(template)/name, 'template/'+name)
    for row in rows:
        root = Path(row['directory']).resolve(strict=True)
        report = read(root/'admission.json')
        prefix = row['uuid']+'/'
        protocol = packer.task_protocol(row['task_id'])
        require(protocol['pick'], 'Capacity estimate only supports explicit three-task pick protocols')
        add(root/'admission.json', prefix+'admission.json')
        shared_sources = set()
        def shared(source, destination, *, link=False):
            shared_sources.add(source['path'])
            add(source['path'], destination, link=link)
        reference = report[protocol['reference_key']]
        trace = reference['control_trace']
        shared(trace, 'trace/'+trace['sha256'], link=True)
        historical = reference['historical_sources']
        shared({'path': historical['manifest_path']}, 'history/'+historical['manifest_sha256']+'/manifest.json')
        for source in historical['files']:
            shared(source, 'history/'+source['sha256']+'/'+source['filename'])
        for source in packer.pick_batch_sources(report).values():
            shared(source, 'batch/'+source['sha256']+'/'+source['filename'])
        for name, digest in read(root/'rtx_completion.json')['producer_source_sha256'].items():
            shared({'path': name}, 'producer/'+digest+'/'+Path(name).name)
        sources = dict(report['bindings'])
        sources[str(root/'collection_record.json')] = None
        for name in sources:
            if name in shared_sources:
                continue
            source = Path(name)
            if source.is_relative_to(root) and source.suffix.lower() != '.png':
                relative = str(source.relative_to(root))
                add(source, prefix+'run/'+relative,
                    link=packer.immutable_evidence_link(protocol, relative, True))
        for name in ('rgb_storage.json', 'rgb_storage_complete.json'):
            add(root/'capture'/name, prefix+'run/capture/'+name)
        for name in report['auditor_source_sha256']:
            add(name, prefix+'auditor_source/'+Path(name).name)
        add(root/'capture/rgb_master.mp4', prefix+'video', link=True)
    return copied + sum(row['raw30_frames'] for row in rows) * (16 << 10) + (len(rows)+1) * (1 << 20)


def verify_ready(path, release, plan, profile, expected_count):
    ready = read(path)
    release = Path(release).resolve(strict=True)
    pack, dino = release/'packed', release/'dino'
    require(ready.get('contract') == 'cvpr-pico-pick-production-verification-v1' and ready.get('status') == 'passed' and
            ready.get('task_id') == TASKS[profile] and ready.get('train_episodes') == expected_count and
            ready.get('production_ready_for_bound_pack_and_cache') is True and ready.get('training_started') is False and
            ready.get('verifier_sha256') == sha(SCRIPT/'verify_pico_handover.py'), 'Production readiness is missing or stale')
    require(ready.get('release_root') == str(release) and ready.get('pack_root') == str(pack) and
            ready.get('dino_root') == str(dino) and ready.get('pack_ready_sha256') == sha(pack/'PACK_READY.json') and
            ready.get('cache_manifest_sha256') == sha(dino/'manifest.json'), 'Readiness belongs to another release/cache')
    def check_identity(path, expected):
        stat = path.stat()
        require(path.is_file() and not path.is_symlink() and sha(path) == expected['sha256'] and
                stat.st_size == expected['size_bytes'] and stat.st_mtime_ns == expected['mtime_ns'],
                'Bound production input changed: ' + str(path))
    require(bool(ready.get('payload_files')) and bool(ready.get('external_reference_files')), 'Readiness input inventory missing')
    actual = {str(item.relative_to(release)) for directory in (pack, dino)
        for item in directory.rglob('*') if item.is_file()}
    require(actual == set(ready['payload_files']) and set(ready['external_reference_files']) == {
        str(Path(plan['run_config']).resolve(strict=True)),
        str(Path(plan['reference_dino']).resolve(strict=True)/'manifest.json')},
        'Readiness inventory differs from the selected production payload/configuration')
    for name, expected in ready['payload_files'].items():
        rel = Path(name)
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe cached production payload')
        check_identity(release/rel, expected)
    for name, expected in ready['external_reference_files'].items():
        require(Path(name).is_absolute(), 'External readiness authority must be absolute')
        check_identity(Path(name), expected)
    return ready


def finish_profile(plan, profile, rows, run_id):
    verify_plan(plan)
    release = Path(plan['release_root'])/profile
    require(not release.is_symlink(), 'Task release cannot be an alias')
    release.mkdir(parents=True, exist_ok=True)
    selection = build_selection(plan, profile, rows)
    selected_path = release/'ADMISSIONS_SELECTION.json'
    if selected_path.exists():
        require(read(selected_path) == selection, 'Preserve a release with a different accepted selection')
    else:
        write(selected_path, selection)
        record(plan, [selected_path], f'冻结{profile}全部{len(rows)}条通过准入的真人UUID，保留原采集编号。')
    packed = release/'packed'
    if not (packed/'PACK_READY.json').is_file():
        require(not list(release.glob('.packed.partial-*')), 'Previous partial pack preserved; diagnose before retrying')
        require(not packed.exists(), 'Partial published pack preserved; use explicit diagnosis instead of overwrite')
        run_child(plan, [plan['model_python'], SCRIPT/'pack_pico_handover.py', '--selection', selected_path,
            '--template', plan['template'], '--output', packed, '--link-immutable-raw'], profile+'_pack_'+run_id,
            packing=True, required_extra_bytes=pack_capacity_bytes(rows, packed, plan['template']))
    pack_receipt = read(packed/'PACK_READY.json')
    require(pack_receipt.get('contract') == 'cvpr-pico-pick-lerobot-pack-v1' and pack_receipt.get('task_id') == TASKS[profile] and
            pack_receipt.get('train_episodes') == len(rows) and pack_receipt.get('selection', {}).get('sha256') == sha(selected_path) and
            pack_receipt.get('packer_sha256') == sha(SCRIPT/'pack_pico_handover.py'), 'Pack changed or differs from accepted selection')
    dino = release/'dino'
    manifest = dino/'manifest.json'
    if not manifest.is_file() or read(manifest).get('state') != 'complete':
        run_child(plan, [plan['model_python'], SCRIPT.parents[1]/'VLMExpert/scripts/data/extract_unifolm_dino.py',
            '--dataset-root', packed/'train', '--output', dino, '--model-path', plan['dino_model'],
            '--shared-image-profile', '256x320', '--device', 'cuda:0', '--batch-size', '32', '--workers', '1',
            '--decoder-threads', '1', '--cpu-threads', '2', '--storage-dtype', 'float32', '--inference-precision', 'bf16'],
            profile+'_dino_'+run_id, gpu=True,
            required_extra_bytes=sum(row['raw30_frames'] for row in rows) * 5 * 1024 * 4 + (16 << 20))
    ready_path = release/'PICO_DATASET_READY.json'
    if not ready_path.exists():
        run_child(plan, [plan['model_python'], SCRIPT/'verify_pico_handover.py', '--pack-root', packed,
            '--dino-root', dino, '--reference-dino-root', plan['reference_dino'], '--run-config', plan['run_config'],
            '--output', ready_path, '--expected-episodes', str(len(rows))], profile+'_reader_'+run_id)
    ready = verify_ready(ready_path, release, plan, profile, len(rows))
    verify_plan(plan)
    verify_originals(rows)
    record(plan, [release], f'{profile}共{len(rows)}条完成打包、真实DINO和production reader准入；未上传、未训练。')
    return {'task_id': TASKS[profile], 'accepted_episodes': len(rows), 'train_windows': ready['train_windows'],
        'release': str(release), 'ready_sha256': sha(ready_path), 'status': 'local_training_data_ready_training_not_started'}


def process(plan_path):
    plan = read(Path(plan_path).resolve(strict=True))
    verify_plan(plan)
    frozen = frozen_cohort(plan)
    root = Path(plan['processing_root']).resolve(strict=True)
    (root/'logs').mkdir(exist_ok=True)
    (root/'rtx_chunks').mkdir(exist_ok=True)
    lock_path = root/'.pick_processing.lock'
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ProcessingDeferred('Another pick299 processor owns the exclusive lock') from error
        binding = root/'PICK_PROCESSING_PLAN.json'
        if binding.exists():
            require(read(binding) == plan, 'This processing root is bound to another plan')
        else:
            write(binding, plan)
            record(plan, [binding, lock_path, root/'logs'], '冻结pick299本地完整处理计划、代码身份和独占锁。')
        status_path = root/'PICK_PROCESSING_STATUS.json'
        status = read(status_path) if status_path.exists() else {'episodes': {}, 'tasks': {}}
        run_id = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S') + '_' + str(os.getpid())
        def update(stage, **extra):
            if 'error' not in extra:
                status.pop('error', None)
            status.update(status=stage, updated_beijing=now(), training_started=False, upload_started=False, **extra)
            write(status_path, status)
            record(plan, [status_path], '更新pick299本地处理状态：' + stage + '。')
        try:
            require_idle()
            checked = verify_originals(frozen['episodes'])
            update('originals_verified', original_files_verified=checked, accepted_target=299)
            for profile in plan['task_order']:
                rows = [row for row in frozen['episodes'] if row['profile'] == profile]
                offset = 0
                while offset < len(rows):
                    verify_plan(plan)
                    row = rows[offset]
                    directory = Path(row['directory']).resolve(strict=True)
                    if (directory/'capture').exists() or (directory/'rtx_completion.json').exists():
                        update('admitting_and_compacting', active_profile=profile, active_number=row['number'], active_offset=offset)
                        status['episodes'][row['uuid']] = admit_and_compact(plan, row)
                        update('episode_admitted_and_compacted', completed_episodes=len(status['episodes']))
                        offset += 1
                        continue
                    chunk = choose_chunk(rows, offset, free_bytes=shutil.disk_usage(directory).free,
                        maximum=plan['max_chunk'], bytes_per_frame=plan['estimated_png_bytes_per_frame'], reserve=plan['reserve_free_bytes'])
                    report = root/'rtx_chunks'/f'{profile}_{offset:03d}_{len(chunk):03d}_{run_id}.json'
                    render_log = report.with_suffix('.log')
                    update('rendering_rtx_chunk', active_profile=profile, active_offset=offset,
                        active_number=row['number'], chunk_count=len(chunk))
                    run_child(plan, [plan['simulation_python'], SCRIPT/'run_pick_rtx.py', '--python', plan['simulation_python'],
                        '--manifest', plan['frozen_manifest'], '--profile', profile, '--offset', str(offset), '--limit', str(len(chunk)),
                        '--data-root', frozen['groups'][profile]['render_data_root'], '--report', report, '--log', render_log],
                        profile+'_rtx_'+str(offset)+'_'+run_id, gpu=True,
                        required_extra_bytes=(sum(item['raw30_frames'] for item in chunk) +
                            max(item['raw30_frames'] for item in chunk)) * plan['estimated_png_bytes_per_frame'],
                        capacity_paths=[item['directory'] for item in chunk])
                    verify_originals(chunk)
                    for current in chunk:
                        update('admitting_and_compacting', active_profile=profile, active_number=current['number'])
                        status['episodes'][current['uuid']] = admit_and_compact(plan, current)
                        update('episode_admitted_and_compacted', completed_episodes=len(status['episodes']))
                    offset += len(chunk)
                update('packing_dino_production_reader', active_profile=profile)
                status['tasks'][profile] = finish_profile(plan, profile, rows, run_id)
                update('task_locally_ready', completed_tasks=len(status['tasks']))
            checked = verify_originals(frozen['episodes'])
            require(set(status['tasks']) == set(EXPECTED_COUNTS) and len(status['episodes']) == 299,
                    'Final per-task/episode readiness coverage differs')
            verify_plan(plan)
            update('all299_local_training_data_ready_training_not_started', original_files_verified=checked,
                accepted_episodes=299, excluded_episode='xmove_pick/004 remains unchanged and excluded for later recapture')
            return status
        except ProcessingDeferred as error:
            update('waiting_for_idle_or_space', error=str(error), data_rejected=False)
            raise
        except BaseException as error:
            update('technical_failure_needs_diagnosis', error=f'{type(error).__name__}: {error}')
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--plan', type=Path, help='Run an already frozen processing plan')
    modes.add_argument('--init-plan', type=Path, help='Write a fresh plan only; no GPU or processing starts')
    parser.add_argument('--processing-root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--release-root', type=Path)
    parser.add_argument('--model-python', type=Path, default=MODEL_PYTHON)
    args = parser.parse_args()
    if args.init_plan:
        require(not args.init_plan.exists() and not args.init_plan.is_symlink(), 'Preserve previous plan')
        plan = build_plan(args.processing_root, release_root=args.release_root, model_python=args.model_python)
        write(args.init_plan, plan)
        record(plan, [args.init_plan], '创建pick299本地分块处理计划，当前未启动补图、模型或训练。')
        print(json.dumps({'plan': str(args.init_plan), 'processing_started': False}, ensure_ascii=False))
        return
    try:
        result = process(args.plan)
    except ProcessingDeferred as error:
        print(json.dumps({'status': 'waiting_for_idle_or_space', 'reason': str(error), 'data_rejected': False}, ensure_ascii=False))
        raise SystemExit(75)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
