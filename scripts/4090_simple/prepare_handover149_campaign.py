"""Create an independent checkpoint-specific copy of the accepted 149/5k framework.

This is a CPU source/binding generator, not a GPU launcher. It never copies a
prior preflight, result, admission or namespace acceptance. Numerical runtime
sources are byte-identical except two explicitly checked checkpoint guards.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


STEPS = (4000, 6000, 7000, 8000, 9000, 10000)
BASE = Path('/home/zxc/quality100_4090_20261024/mirror/home/weisheng/psi0_runtime/quality100_milestone_eval_20261023')
TEMPLATE = BASE / 'handover149_5000_4090_eval_20261026_r2'
OLD_FRAMEWORK = TEMPLATE.name
OLD_RUN = 'handover149_experiment_5000_official30_4090_trt10133_20261026_r2'
OLD_MARKER = 'handover149-5000-4090-trt10133-private-home-net-tmp-v2-dino-source-bound'
SOURCE_RUN = ('/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/handover149_training_20261026/'
    'bs4_restart_001/formal10000/training/posttrain/'
    'handover_train149_joint38000_statekeep_bs4_8gpu.us.flow1000.cosine.lr1.0e-05.b32.gpus8.261026054254')
SCRIPTS = ('case_runner_template.py', 'cpu_dds_probe.py', 'environment_probe.py',
    'evaluation_4090.py', 'host_bootstrap.py', 'namespace_bootstrap.py',
    'namespace_common.py', 'namespace_launch.sh', 'prepare_checkpoint.py',
    'revision_resources.py', 'runtime_audit.py', 'seal_source_bundle.py',
    'strict_numerics_template.py', 'test_dino_source_binding_CPU.py')
POLICIES = ('host_target.json', 'native4090_resources.json', 'native_library_overlay.json')
ADAPTED = {'case_runner_template.py', 'environment_probe.py', 'evaluation_4090.py',
    'namespace_bootstrap.py', 'namespace_common.py', 'namespace_launch.sh',
    'prepare_checkpoint.py', 'runtime_audit.py', 'seal_source_bundle.py',
    'test_dino_source_binding_CPU.py'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def bind(path):
    path = Path(path)
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def file_identity(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def create(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def names(step, tag):
    require(type(step) is int and step in STEPS, 'Only explicitly requested campaign checkpoints')
    require(bool(re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]{0,39}', tag)), 'Use a short new campaign tag')
    return (f'handover149_{step}_4090_eval_20261026_{tag}',
            f'handover149_experiment_{step}_official30_4090_trt10133_20261026_{tag}',
            f'handover149-{step}-4090-trt10133-private-home-net-tmp-{tag}-dino-source-bound')


def one(text, before, after):
    require(text.count(before) == 1, 'Expected exactly one source identity anchor: ' + before)
    return text.replace(before, after)


def adapt(relative, original, *, step, tag, source):
    """Identity-only changes; no sampler, controller, scenario or numeric edits."""
    framework, run, marker = names(step, tag)
    if relative == 'runtime_template/protocol_core.py':
        return one(original, 'CHECKPOINT_STEPS=(5000,)', f'CHECKPOINT_STEPS=({step},)')
    if relative == 'runtime_template/rtc_acceptance.py':
        text = one(original, 'manifest.get("step")==5000', f'manifest.get("step")=={step}')
        return one(text, 'Only this149 task5000 deployment', f'Only this149 task{step} deployment')
    if relative not in ADAPTED:
        return original
    text = original.replace(OLD_FRAMEWORK, framework).replace(OLD_RUN, run).replace(OLD_MARKER, marker)
    if relative == 'test_dino_source_binding_CPU.py':
        # The old two real failed-case audits remain a regression fixture. They
        # are not a preflight for the new model and must never be relabelled.
        return one(text, "OLD=N.with_name(N.name.removesuffix('_r2'))",
                   "OLD=H.parent/'" + OLD_RUN.removesuffix('_r2') + "'")
    # These orchestration sources contain only checkpoint identity occurrences
    # of 5000. Preserve the historical reference's own checkpoint number.
    sentinel = 'REFERENCE_CHECKPOINT_KEEP_FIVE_THOUSAND'
    if relative == 'prepare_checkpoint.py':
        text = one(text, 'reference_model_step=5000', 'reference_model_step=' + sentinel)
        text = one(text, 'handover149_experiment_{step}_official30_4090_trt10133_20261026_r2',
                   'handover149_experiment_{step}_official30_4090_trt10133_20261026_' + tag)
    text = text.replace('5000', str(step))
    text = text.replace(sentinel, '5000')
    if relative == 'seal_source_bundle.py':
        text = one(text, "RAW=Path('/home/zxc/CVPR_Evaluation_runtime/checkpoints/handover149_bs4_20261026/ckpt_"
                   + str(step) + "')", 'RAW=Path(' + repr(str(source)) + ')')
    return text


def verify_runtime_bytes(relative, before, after, step):
    if not relative.startswith(('runtime_template/', 'helper_templates/')):
        return
    expected = before
    if relative == 'runtime_template/protocol_core.py':
        expected = one(before, 'CHECKPOINT_STEPS=(5000,)', f'CHECKPOINT_STEPS=({step},)')
    elif relative == 'runtime_template/rtc_acceptance.py':
        expected = one(before, 'manifest.get("step")==5000', f'manifest.get("step")=={step}')
        expected = one(expected, 'Only this149 task5000 deployment', f'Only this149 task{step} deployment')
    require(after == expected, 'Numerical/template bytes changed outside exact identity guards: ' + relative)


def validate_inputs(source, local, step, digest, *, hash_model=True):
    require(re.fullmatch('[0-9a-f]{64}', digest) is not None, 'Independent A100 model SHA256 required')
    require(source.name == f'ckpt_{step}', 'Backup directory must identify this checkpoint')
    backup, ready = read(source / 'backup_receipt.json'), read(local / 'LOCAL_RUN_READY.json')
    require(backup.get('contract') == 'cvpr-handover149-local-checkpoint-backup-v1'
        and backup.get('step') == step and backup.get('optimizer_copied') is False
        and backup['run_identity']['run_directory'] == SOURCE_RUN, 'Wrong saved BS4/149 checkpoint backup')
    model = backup['model']
    require({model.get(k) for k in ('source_sha256_before', 'source_sha256_after', 'local_sha256')} == {digest}
        and model.get('source_path') == SOURCE_RUN + f'/checkpoints/ckpt_{step}/model.safetensors',
        'Source-before/source-after/local SHA or actual A100 checkpoint path differs')
    retained = backup['completed_retention_marker']
    require(retained.get('all_models_preserved') is True and retained.get('saved_update', -1) >= step
        and step in retained.get('all_saved_model_steps', []), 'Checkpoint save completion is not proved')
    require(ready.get('contract') == 'cvpr-handover149-local-deployment-v1'
        and ready.get('status') == 'local_handover149_run_prepared'
        and ready.get('task_step') == ready.get('step') == step and ready.get('joint_sha256') == digest
        and ready.get('source_backup_receipt_sha256') == sha(source / 'backup_receipt.json')
        and ready.get('source_model_bytes_verified') is True and ready.get('checkpoint_save_completed') is True
        and ready.get('GPU_used') is False and ready.get('policy_forward_performed') is False,
        'A real checkpoint-specific CPU acceptance is required; never rename the 5k receipt')
    require((ready.get('train_episodes'), ready.get('train_windows'), ready.get('source_global_batch'),
        ready.get('num_blocks'), ready.get('state_drop_prob')) == (149, 59468, 32, 6, 0.),
        'CPU acceptance training/architecture identity differs')
    for name, expected in backup['metadata'].items():
        require(sha(source / name) == expected, 'Backup metadata changed: ' + name)
    raw = source / 'model.safetensors'
    deployed = local / 'checkpoints' / f'ckpt_{step}' / 'model.safetensors'
    a, b = raw.stat(), deployed.stat()
    require(not raw.is_symlink() and not deployed.is_symlink()
        and (a.st_dev, a.st_ino, a.st_size) == (b.st_dev, b.st_ino, b.st_size)
        and a.st_size == model['bytes'], 'Actual CPU-verified model must remain the downloaded hardlink')
    if hash_model:
        require(sha(raw) == digest and file_identity(raw.stat()) == file_identity(a),
                'Actual saved model bytes changed')
    require((local / 'motion/joint_motion.ckpt').is_file()
        and (local / 'provenance/runtime_metadata_validation.json').is_file(),
        'Checkpoint-specific Motion export and metadata bridge are missing')
    require(ready.get('strict_motion_load') == 'passed'
        and ready.get('full_saved_components_strict_CPU_load', {}).get('status') == 'passed'
        and ready.get('full_saved_components_strict_CPU_load', {}).get('CUDA_initialized') is False,
        'Real strict CPU component/Motion load acceptance required')
    require(read(local / 'dino_training_contract.json').get('motion_image_policy') == 'vlm_color_clean_motion',
        'Actual current DINO contract must bind the source training policy')
    return backup, ready


def clone_local(source, target):
    # Preserve original CPU receipts and weights, including their original path
    # provenance. prepare_checkpoint maps construction paths explicitly later.
    shutil.copytree(source, target, copy_function=os.link, symlinks=True)


def verify_template_runtime(template, mirror):
    """Bind numerical templates to the existing accepted 5k source manifest.

    Stored paths are namespace paths; compare the relative path below this
    exact template, never hash the same text after substituting output paths.
    """
    manifest = read(template.parent / OLD_RUN / 'checkpoint_manifest.json')
    preflight = read(template.parent / OLD_RUN / 'preflight_acceptance.json')
    require(manifest.get('step') == preflight.get('step') == 5000
        and preflight.get('status') == 'passed'
        and manifest.get('model_sha256') == preflight.get('model_sha256'),
        'Original admitted template checkpoint identity differs')
    prefix = str(Path('/home') / template.relative_to(mirror)) + '/'
    records = {row['path'][len(prefix):]: row for row in manifest['files']
               if row['path'].startswith(prefix)}
    checked = []
    for sub in ('runtime_template', 'helper_templates'):
        for path in sorted((template / sub).glob('*.py')):
            relative = str(path.relative_to(template))
            require(relative in records, 'Original accepted runtime source binding missing: ' + relative)
            expected = records[relative]
            require(path.stat().st_size == expected['bytes'] and sha(path) == expected['sha256'],
                    'Original accepted runtime template changed: ' + relative)
            checked.append({'relative_path': relative, 'accepted_binding': expected})
    return {'status': 'passed', 'accepted_manifest': bind(template.parent / OLD_RUN / 'checkpoint_manifest.json'),
            'accepted_preflight': bind(template.parent / OLD_RUN / 'preflight_acceptance.json'),
            'runtime_and_helper_bindings': checked}


def generate(args):
    step, tag = args.step, args.tag
    template = args.template.resolve(strict=True)
    source, local = args.source.resolve(strict=True), args.local_run.resolve(strict=True)
    framework_name, run_name, marker = names(step, tag)
    require(template.name == OLD_FRAMEWORK, 'Use the admitted 5k R2 template')
    root = template.parent
    framework, run = root / framework_name, root / run_name
    require(not framework.exists() and not run.exists(), 'Preserve all existing attempts; use a new tag')
    target = read(template / 'host_target.json')
    mirror = Path(target['mirror_home']).resolve(strict=True)
    require(template.is_relative_to(mirror), 'Template must be in the task mirror')
    template_acceptance = verify_template_runtime(template, mirror)
    validate_inputs(source, local, step, args.expected_model_sha256)
    files = [Path(name) for name in SCRIPTS + POLICIES]
    files += [path.relative_to(template) for sub in ('runtime_template', 'helper_templates')
              for path in sorted((template / sub).glob('*.py'))]
    # Read every input before writing. Only source files, never completed
    # acceptance/admission/result directories, enter the new framework.
    originals = {str(path): (template / path).read_bytes() for path in files}
    framework.mkdir();run.mkdir()
    try:
        clone_local(local, run / 'local_run')
        changed, source_bindings = [], []
        diffs = []
        for relative, blob in originals.items():
            before = blob.decode()
            after = adapt(relative, before, step=step, tag=tag, source=source)
            verify_runtime_bytes(relative, before, after, step)
            if relative.endswith('.py'):
                ast.parse(after, filename=relative)
            path = framework / relative;path.parent.mkdir(exist_ok=True)
            path.write_text(after)
            source_bindings.append({'relative_path': relative, 'template': bind(template / relative),
                                    'generated': bind(path), 'bytes_unchanged': before == after})
            if before != after:
                changed.append(relative)
                diffs.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                    fromfile='accepted5k/' + relative, tofile=f'checkpoint{step}/' + relative))
        policy = read(template / 'namespace_policy.json')
        policy['namespace_marker'] = marker
        policy['model_sha256'] = f'PENDING_VERIFIED_149_TASK{step}'
        create(framework / 'namespace_policy.json', policy)
        # Keep the accepted old 5k audits as explicit immutable CPU fixtures.
        # The nine tests still perform new deployment source-binding checks.
        create(framework / 'runtime_template_source_receipt.json', {
            'status': 'campaign_identity_only_source_adaptation', 'template': str(template), 'step': step,
            'source_bindings': source_bindings, 'numerical_runtime_bytes_unchanged': True,
            'original_template_acceptance': template_acceptance,
            'allowed_runtime_identity_edits': ['protocol_core.CHECKPOINT_STEPS',
                'rtc_acceptance.load_source_bound_dino_contract checkpoint guard'],
            'old_fixture_run': str(root / OLD_RUN.removesuffix('_r2')),
            'prior_preflight_or_results_copied': False})
        (framework / 'IDENTITY_SOURCE_DIFF.patch').write_text(''.join(diffs))
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1')
        with (framework / 'SOURCE_SEAL_CPU.log').open('x') as log:
            subprocess.run([sys.executable, str(framework / 'seal_source_bundle.py')],
                env=environment, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
        pin = read(framework / 'source_pin.json')
        require(pin['authorized_steps'] == [step] and pin['model_sha256'] == args.expected_model_sha256,
                'Fresh seal did not bind this exact checkpoint')
        creation = {
            'status': 'CPU_source_generated_waiting_fresh_namespace_qualification', 'step': step,
            'model_sha256': args.expected_model_sha256, 'framework_directory': str(framework),
            'run_directory': str(run), 'namespace_marker': marker,
            'source_backup': bind(source / 'backup_receipt.json'),
            'actual_CPU_acceptance': bind(local / 'LOCAL_RUN_READY.json'),
            'generator': bind(__file__), 'source_bindings': source_bindings,
            'identity_changed_files': changed, 'source_pin': bind(framework / 'source_pin.json'),
            'source_complete': bind(run / 'source_bundle/COMPLETE.json'),
            'numerical_and_helper_source_audit': bind(framework / 'runtime_template_source_receipt.json'),
            'GPU_started': False, 'GPU_admission_created': False, 'namespace_CPU_passed': False,
            'preflight_passed': False, 'formal_cases_completed': 0}
        create(framework / 'CAMPAIGN_GENERATION.json', creation)
        commands = [['bash', str(framework / 'namespace_launch.sh'), '--mirror-home', str(mirror),
                     '--phase', phase, '--attempt', 'attempt_001', '--expected-hostname', target['hostname']]
                    for phase in ('cpu-check', 'preflight', 'run')]
        namespace = lambda path: str(Path('/home') / path.relative_to(mirror))
        return {'framework_host': str(framework), 'framework_namespace': namespace(framework),
                'experiment_host': str(run), 'experiment_namespace': namespace(run), 'mirror_home': str(mirror),
                'creation_receipt': str(framework / 'CAMPAIGN_GENERATION.json'), 'phase_commands': commands,
                'next': 'Run fresh CPU qualification. GPU phases additionally require a source-bound ROOT_GPU_LAUNCH_ADMISSION.json and the preceding campaign completion gate.'}
    except BaseException as error:
        create(framework / 'CAMPAIGN_GENERATION_FAILED.json', {
            'status': 'CPU_generation_error', 'error': repr(error), 'GPU_started': False,
            'existing_attempts_preserved': True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--step', type=int, choices=STEPS, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--local-run', type=Path, required=True)
    parser.add_argument('--expected-model-sha256', required=True)
    parser.add_argument('--template-framework', '--template', dest='template', type=Path, default=TEMPLATE)
    print(json.dumps(generate(parser.parse_args()), separators=(',', ':')))


if __name__ == '__main__':
    main()
