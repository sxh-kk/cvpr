"""Recover the two unfinished 5k cases without changing frozen inference sources.

The original failed attempt and all prior evidence remain intact. This entry
explicitly repeats the original namespace/CPU/model/hardware/preflight gates,
then installs only an independently audited process-cleanup helper in memory.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import sys

STEP = 5000
MODEL_SHA = '57f689768b0c0edf9b4407eb441f701d1b35be9807d76f2397dda4083fb02cfd'
CASES = ('level_1_episode_09', 'level_2_episode_09')
CONTRACT = 'cvpr-handover149-5k-cleanup-recovery-v1'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def create(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def recovery_plan(protocol, state):
    require(state.get('step') == STEP and state.get('model_sha256') == MODEL_SHA
        and state.get('status') == 'execution_error' and state.get('pid') is None,
        'Require the actual stopped 5k execution error')
    require(state.get('current_case') == CASES[0] and len(state.get('failures', [])) == 1
        and 'Owned cleanup failure' in state['failures'][0]['error']
        and state['failures'][0].get('counted_as_policy_failure') is False,
        'Only the recorded cleanup failure is authorized for recovery')
    ordered = sorted(protocol['ordered_seed_pairs'],
                     key=lambda row: (row['official_episode_index'], row['official_level']))
    require(len(ordered) == 30 and len({row['case_id'] for row in ordered}) == 30,
            'Require thirty distinct original cases')
    completed = state.get('completed', [])
    require(completed == [row['case_id'] for row in ordered[:28]]
        and tuple(row['case_id'] for row in ordered[28:]) == CASES,
        'Only the last two cases may be executed; all original28 must remain')
    return ordered[:28], ordered[28:]


def prepare(framework, output):
    framework = Path(framework).resolve(strict=True)
    require(framework.name == 'handover149_5000_4090_eval_20261026_r2', 'Exact original5k framework required')
    output = Path(output).absolute()
    require(output.parent == framework and not output.exists(), 'Use a fresh child recovery directory')
    mirror = Path(read(framework / 'host_target.json')['mirror_home']).resolve(strict=True)
    namespace = lambda path: str(Path('/home') / Path(path).relative_to(mirror))
    experiment = framework.parent / 'handover149_experiment_5000_official30_4090_trt10133_20261026_r2'
    previous, pending = recovery_plan(read(experiment / 'protocol.json'), read(experiment / 'state.json'))
    require(not Path('/proc/2109022').exists(), 'Original evaluator still exists; leave it untouched')
    failed = experiment / 'cases' / CASES[0] / 'attempt_001'
    require(read(failed / 'owned_processes.json').get('cleanup_complete') is False,
            'Preserve the real failed cleanup evidence')
    groups = {row['process_group'] for row in read(failed / 'owned_processes.json')['processes']}
    pidfile = failed / 'owned_controller.pid'
    if pidfile.exists():
        groups.add(int(pidfile.read_text().strip()))
    for entry in Path('/proc').iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        require(not (int(fields[2]) in groups and fields[0] not in ('Z', 'X')),
                'An original owned process remains live; do not restart it')
    output.mkdir()
    for name in ('resume_handover149_eval.py', 'owned_process_cleanup.py'):
        shutil.copy2(Path(__file__).with_name(name), output / name)
    for name in ('state.json', 'results.json'):
        shutil.copy2(experiment / name, output / ('original_' + name))
    original = (framework / 'namespace_launch.sh').read_text()
    before = 'script_dir=$(cd -- "$(dirname -- "$0")" && pwd)'
    require(original.count(before) == 1, 'Native wrapper identity anchor differs')
    wrapper = original.replace(before, 'script_dir=' + shlex.quote(str(framework)))
    before = '[[ "$phase" == cpu-check || "$phase" == preflight || "$phase" == run ]]'
    require(wrapper.count(before) == 1, 'Native wrapper phase anchor differs')
    wrapper = wrapper.replace(before, '[[ "$phase" == run ]]')
    before = 'namespace_bootstrap.py "${args[@]}"'
    require(wrapper.count(before) == 1, 'Native wrapper execution anchor differs')
    command = shlex.join([namespace(output / 'resume_handover149_eval.py'), '--execute',
        '--framework', namespace(framework), '--recovery', namespace(output), '--cpu-attempt', 'attempt_002'])
    wrapper = wrapper.replace(before, command)
    (output / 'namespace_resume.sh').write_text(wrapper)
    bindings = [{ 'path': namespace(output / name), 'bytes': (output / name).stat().st_size,
        'sha256': sha(output / name)} for name in ('resume_handover149_eval.py',
        'owned_process_cleanup.py', 'namespace_resume.sh')]
    create(output / 'ROOT_RESUME_ADMISSION.json', dict(contract=CONTRACT, authorized=True,
        step=STEP, model_sha256=MODEL_SHA, completed_cases=[x['case_id'] for x in previous],
        pending_cases=[x['case_id'] for x in pending], source_bindings=bindings,
        original_state_sha256=sha(output / 'original_state.json'),
        original_results_sha256=sha(output / 'original_results.json'),
        actual_same_model_preflight_sha256=sha(experiment / 'preflight_acceptance.json'),
        original_wrapper_sha256=sha(framework / 'namespace_launch.sh'),
        reason='用户已授权完成当前5k和后续顺序测评；修复退出清理竞态，保留失败尝试并补跑最后两集',
        inference_sources_modified=False, preflight_rerun_claimed=False))
    return dict(recovery_host=str(output), recovery_namespace=namespace(output), mirror_home=str(mirror),
                launcher=str(output / 'namespace_resume.sh'))


def execute(framework, recovery, cpu_attempt):
    framework, recovery = Path(framework).resolve(strict=True), Path(recovery).resolve(strict=True)
    sys.path.insert(0, str(framework))
    evaluator = load('original5k_recovery_evaluator', framework / 'evaluation_4090.py')
    bootstrap = load('original5k_recovery_bootstrap', framework / 'namespace_bootstrap.py')
    from namespace_common import namespace_guard
    namespace_guard()
    evaluator.namespace_guard()
    evaluator.hardware()
    admission = read(recovery / 'ROOT_RESUME_ADMISSION.json')
    require(admission.get('authorized') is True and admission.get('contract') == CONTRACT
        and admission.get('step') == STEP and admission.get('model_sha256') == MODEL_SHA,
        'Explicit this-model recovery admission is required')
    for row in admission['source_bindings']:
        evaluator.check(row)
    require(sha(framework / 'namespace_launch.sh') == admission['original_wrapper_sha256'],
            'Native wrapper changed since recovery review')
    original_admission = read(framework / 'ROOT_GPU_LAUNCH_ADMISSION.json')
    require(original_admission.get('authorized') is True and original_admission.get('model_sha256') == MODEL_SHA
        and original_admission.get('namespace') == os.environ['HANDOVER149_4090_NAMESPACE']
        and 'run' in original_admission.get('allowed_phases', []), 'Original GPU admission differs')
    cpu_path, cpu = bootstrap.CPU_acceptance(cpu_attempt)
    run, base, strict, protocol, protocol_sha, manifest = evaluator.gate()
    signal.signal(signal.SIGTERM, run.ev.interrupted)
    signal.signal(signal.SIGINT, run.ev.interrupted)
    experiment = evaluator.N
    require(sha(experiment / 'state.json') == admission['original_state_sha256']
        and sha(experiment / 'results.json') == admission['original_results_sha256'],
        'Stopped evaluation changed since this recovery was authorized')
    state = read(experiment / 'state.json')
    previous, pending = recovery_plan(protocol, state)
    preflight = read(experiment / 'preflight_acceptance.json')
    require(sha(experiment / 'preflight_acceptance.json') == admission['actual_same_model_preflight_sha256']
        and preflight['status'] == 'passed' and preflight['hardware'] == evaluator.hardware()
        and preflight['model_sha256'] == MODEL_SHA and preflight['step'] == STEP
        and preflight['actual_cases'] == 3 and preflight['not_scored'] is True
        and preflight['contract_sha256'] == evaluator.u.unchanged_contract(protocol)
        and protocol['readiness']['receipt_sha256'] == sha(experiment / 'preflight_acceptance.json'),
        'Actual unchanged same-model preflight is required')
    cleanup = load('recovery_owned_process_cleanup', recovery / 'owned_process_cleanup.py')
    with (evaluator.L / 'worker_gpu.lock').open('a') as gpu, (evaluator.L / 'evaluation.lock').open('a') as lock:
        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluator.idle()
        checked = []
        for case in previous:
            saved = read(experiment / 'cases' / case['case_id'] / 'case_acceptance.json')
            for row in saved['raw_bindings']:
                evaluator.check(row)
            fresh = evaluator.verify_case(run, base, strict, protocol, protocol_sha,
                case, Path(saved['case_directory']))
            require(all(fresh[key] == saved[key] for key in ('native_success', 'strict_success',
                'model_sha256', 'step', 'case', 'case_directory')), 'Original case outcome or identity changed')
            checked.append(dict(case_id=case['case_id'], acceptance=evaluator.binding(
                experiment / 'cases' / case['case_id'] / 'case_acceptance.json')))
        create(recovery / 'original28_revalidated.json', dict(status='passed', actual_cases=28,
            cases=checked, CPU_qualification=evaluator.binding(cpu_path),
            recovery_admission=evaluator.binding(recovery / 'ROOT_RESUME_ADMISSION.json'),
            actual_same_model_preflight=evaluator.binding(experiment / 'preflight_acceptance.json'),
            process_cleanup=cleanup.install(run.ev), checked_utc=evaluator.now()))
        state['resolved_execution_errors'] = [dict(failures=state['failures'], error=state.get('error'),
            failed_utc=state.get('failed_utc'), recovery=str(recovery), status='retrying_preserved_attempt')]
        state.update(status='running', pid=os.getpid(), failures=[], recovery=str(recovery),
                     recovery_started_utc=evaluator.now())
        state.pop('error', None)
        state.pop('failed_utc', None)
        evaluator.write(experiment / 'state.json', state)
        try:
            for case in pending:
                case_root = experiment / 'cases' / case['case_id']
                target = case_root / ('attempt_002' if case['case_id'] == CASES[0] else 'attempt_001')
                require(not target.exists() and not (case_root / 'case_acceptance.json').exists(),
                        'Never overwrite prior attempts or acceptance')
                evaluator.idle()
                state.update(current_case=case['case_id'], current_path=str(target))
                evaluator.write(experiment / 'state.json', state)
                run.run_case(case, 'all_visible', target, replay_prefix=None)
                proof = evaluator.verify_case(run, base, strict, protocol, protocol_sha, case, target)
                evaluator.write(case_root / 'case_acceptance.json', proof)
                state['completed'].append(case['case_id'])
                evaluator.write(experiment / 'state.json', state)
                evaluator.summary(protocol)
            evaluator.gate()
            proofs = []
            for case in protocol['ordered_seed_pairs']:
                saved = read(experiment / 'cases' / case['case_id'] / 'case_acceptance.json')
                fresh = evaluator.verify_case(run, base, strict, protocol, protocol_sha,
                    case, Path(saved['case_directory']))
                require(all(fresh[key] == saved[key] for key in ('native_success', 'strict_success',
                    'model_sha256', 'step', 'case_directory')), 'Final recovered case changed')
                proofs.append(fresh)
            evaluator.write(experiment / 'final_raw_acceptance.json', dict(status='passed',
                hardware=evaluator.hardware(), step=STEP, model_sha256=MODEL_SHA, actual_cases=30,
                actual_controller_ACKs=45000, actual_sampling_requests=1740, actual_RTC_VJPs=17100,
                proofs=proofs, recovery=evaluator.binding(recovery / 'original28_revalidated.json'),
                checked_utc=evaluator.now()))
            summary = evaluator.summary(protocol, True)
            require(summary['accepted_episodes'] == 30, 'Incomplete recovered thirty')
            state['resolved_execution_errors'][0]['status'] = 'resolved_by_new_complete_attempt'
            state.update(status='complete', pid=None, current_case=None, current_path=None,
                final_raw_acceptance=evaluator.binding(experiment / 'final_raw_acceptance.json'),
                finished_utc=evaluator.now())
            evaluator.write(experiment / 'state.json', state)
            create(recovery / 'RECOVERY_COMPLETE.json', dict(status='complete', actual_new_cases=2,
                original_cases_revalidated=28, model_sha256=MODEL_SHA,
                final=evaluator.binding(experiment / 'final_raw_acceptance.json'),
                results=evaluator.binding(experiment / 'results.json'), completed_utc=evaluator.now()))
        except BaseException as error:
            state.update(status='execution_error', pid=None, error=repr(error), failed_utc=evaluator.now())
            state['failures'].append(dict(case=state.get('current_case'), error=repr(error), counted_as_policy_failure=False))
            evaluator.write(experiment / 'state.json', state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--framework', type=Path, required=True)
    parser.add_argument('--recovery', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--cpu-attempt', default='attempt_002')
    args = parser.parse_args()
    if args.execute:
        execute(args.framework, args.recovery, args.cpu_attempt)
    else:
        print(json.dumps(prepare(args.framework, args.recovery), separators=(',', ':')))


if __name__ == '__main__':
    main()
