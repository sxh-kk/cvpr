"""Durable serial continuation of the local 149-demo checkpoint evaluation.

Each checkpoint gets its own source backup, CPU export, private namespace CPU
qualification, real three-case GPU preflight and thirty-case evaluation. The
existing 5k process is only observed. Technical errors stop the queue and keep
all evidence; ordinary unsuccessful task episodes do not stop it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

STEPS = (4000, 6000, 7000, 8000, 9000, 10000)
SCRIPTS = Path(__file__).resolve().parent
CONTRACT = 'cvpr-handover149-serial-local-evaluation-v1'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def binding(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def process_identity(pid):
    try:
        # comm may contain spaces and parentheses. Fields after its last ')'
        # start at field3; Linux starttime is field22.
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=pid, start_ticks=int(value[19]), state=value[0])
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_live_process(identity):
    actual = process_identity(identity['pid'])
    return bool(actual and actual['state'] != 'Z'
                and actual['start_ticks'] == identity['start_ticks'])


def complete_result(directory, step, model_sha=None):
    """Only fully audited, final30 results allow the next GPU model to start."""
    directory = Path(directory)
    state_path = directory / 'state.json'
    if not state_path.exists():
        return None
    state = read(state_path)
    require(state.get('status') != 'execution_error' and not state.get('failures'),
            'Preceding evaluation has a technical error: ' + str(directory))
    require(state.get('step') == step, 'Preceding state checkpoint differs')
    if state.get('status') != 'complete':
        pid = state.get('pid')
        identity = process_identity(pid) if type(pid) is int else None
        require(identity is not None and identity['state'] != 'Z',
                'Incomplete preceding evaluation has no live evaluator: ' + str(directory))
        return None
    results = read(directory / 'results.json')
    final = read(directory / 'final_raw_acceptance.json')
    require(results.get('status') == 'complete'
        and results.get('expected_episodes') == results.get('accepted_episodes') == 30
        and final.get('status') == 'passed' and final.get('actual_cases') == 30,
        'Thirty final audited cases are required before continuing')
    rows = results.get('cases', [])
    expected = {f'level_{level}_episode_{episode:02d}'
                for level in range(3) for episode in range(10)}
    require(len(rows) == 30
        and {row['case']['case_id'] for row in rows} == expected
        and all(row.get('status') == 'passed' and row.get('not_scored') is False for row in rows)
        and set(state.get('completed', [])) == expected,
        'Final thirty unique scored cases or their execution acceptance differ')
    digests = {state.get('model_sha256'), results.get('model_sha256'), final.get('model_sha256')}
    digests.update(row.get('model_sha256') for row in rows)
    require(len(digests) == 1 and None not in digests
        and (model_sha is None or digests == {model_sha})
        and results.get('step') == final.get('step') == step
        and all(row.get('step') == step for row in rows), 'Final model identity differs')
    for key in ('native', 'strict'):
        expected_successes = sum(row[key + '_success'] is True for row in rows)
        require(results[key] == dict(successes=expected_successes, episodes=30),
                'Final result counts differ from individual cases')
    actual = binding(directory / 'final_raw_acceptance.json')
    recorded = state.get('final_raw_acceptance', {})
    require(recorded.get('sha256') == actual['sha256'] and recorded.get('bytes') == actual['bytes'],
            'Final state does not bind the current raw acceptance')
    return dict(step=step, model_sha256=next(iter(digests)),
                native=results['native'], strict=results['strict'],
                results=binding(directory / 'results.json'), final_raw_acceptance=actual)


def parse_last_json(text):
    for line in reversed(text.splitlines()):
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
    raise RuntimeError('Expected a final JSON object from the preparation tool')


class Queue:
    def __init__(self, config, path, lock):
        self.config, self.path, self.lock = config, Path(path), lock
        self.state_path = self.path / 'state.json'
        self.state = read(self.state_path) if self.state_path.exists() else dict(
            contract=CONTRACT, status='created', steps=list(STEPS), records={}, created_utc=now())

    def save(self, **fields):
        self.state.update(fields, updated_utc=now())
        write(self.state_path, self.state)

    def record(self, step, **fields):
        record = self.state['records'].setdefault(str(step), {})
        record.update(fields)
        self.save(current_step=step)
        return record

    def command(self, step, phase, command, *, env=None):
        log = self.path / 'logs' / f'{step}_{phase}.log'
        require(not log.exists(), 'Preserve the prior failed command log: ' + str(log))
        self.record(step, phase=phase, command=command, command_log=str(log))
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, env=env, pass_fds=(self.lock.fileno(),))
            identity = process_identity(child.pid)
            require(identity is not None, 'Cannot identify newly launched child')
            self.save(child=identity)
            code = child.wait()
        self.save(child=None)
        require(code == 0, f'{step} {phase} exited {code}; see {log}')
        return log

    def wait_complete(self, directory, step, digest=None):
        while True:
            report = complete_result(directory, step, digest)
            if report is not None:
                return report
            self.save(status='waiting_for_previous', waiting_step=step,
                      waiting_directory=str(directory))
            time.sleep(self.config.get('poll_seconds', 30))

    def wait_gpu(self, step):
        while True:
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
            owners = [line.strip() for line in query.stdout.splitlines()
                      if line.split(',')[0].strip() == self.config['gpu_uuid']]
            with Path('/tmp/quality100-4090-gpu.lock').open('a') as physical:
                try:
                    fcntl.flock(physical, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    free_lock = True
                except BlockingIOError:
                    free_lock = False
            if not owners and free_lock:
                return
            self.record(step, phase='waiting_for_idle_gpu', gpu_owners=owners)
            time.sleep(self.config.get('poll_seconds', 30))

    def stage(self, step):
        cfg = self.config
        record = self.state['records'].get(str(step), {})
        backup = Path(cfg['backup_root']) / f'ckpt_{step}'
        if 'backup' not in record:
            log = self.command(step, 'backup', [sys.executable,
                str(SCRIPTS / 'backup_handover149_checkpoint.py'), '--step', str(step),
                '--output-root', cfg['backup_root'], '--reference', cfg['reference_backup'],
                '--ssh-host', cfg['ssh_host'], '--minimum-free-gib', '12'])
            value = parse_last_json(log.read_text())
            require(Path(value['source']) == backup and len(value['expected_model_sha256']) == 64,
                    'Backup tool returned another source')
            record = self.record(step, backup=value)
        authority = record['backup']
        digest = authority['expected_model_sha256']
        require(read(backup / 'backup_receipt.json')['model']['local_sha256'] == digest,
                'Backup changed since source verification')
        local = self.path / f'checkpoint_{step}' / 'local_run'
        if not (local / 'LOCAL_RUN_READY.json').exists():
            local.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2',
                MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1',
                TRANSFORMERS_OFFLINE='1')
            self.command(step, 'cpu_export', [cfg['model_python'],
                str(SCRIPTS / 'prepare_handover149_eval.py'), '--step', str(step),
                '--expected-model-sha256', digest, '--source', str(backup),
                '--resources', cfg['resources'], '--output', str(local),
                '--repo', cfg['repo'], '--release', str(backup / 'release_metadata')], env=env)
        ready = read(local / 'LOCAL_RUN_READY.json')
        require(ready.get('step') == step and ready.get('joint_sha256') == digest
            and ready.get('GPU_used') is False, 'Wrong checkpoint CPU export')
        if 'framework' not in record:
            log = self.command(step, 'framework', [sys.executable,
                str(SCRIPTS / 'prepare_handover149_campaign.py'), '--step', str(step),
                '--expected-model-sha256', digest, '--source', str(backup),
                '--local-run', str(local), '--tag', cfg['tag'],
                '--template-framework', cfg['template_framework']])
            value = parse_last_json(log.read_text())
            require(all(key in value for key in ('framework_host', 'framework_namespace',
                'experiment_host', 'experiment_namespace', 'mirror_home')),
                'Framework generator returned an incomplete launch descriptor')
            record = self.record(step, framework=value)
        return record

    def namespace_phase(self, step, framework, phase):
        host = Path(framework['framework_host'])
        attempt = 'serial_001'
        output = host / 'namespace_attempts' / attempt / phase
        success = output / ('cpu_acceptance.json' if phase == 'cpu-check' else 'phase_complete.json')
        if success.exists():
            expected = 'passed' if phase == 'cpu-check' else 'complete'
            require(read(success).get('status') == expected, 'Prior namespace phase is incomplete')
            return success
        self.wait_gpu(step)
        self.save(status='running')
        self.command(step, phase, ['bash', str(host / 'namespace_launch.sh'),
            '--mirror-home', framework['mirror_home'], '--phase', phase,
            '--attempt', attempt, '--expected-hostname', self.config['hostname']])
        require(success.exists(), 'Namespace exited without actual phase acceptance')
        return success

    def execute(self):
        self.save(status='running', pid=os.getpid(), process=process_identity(os.getpid()),
                  config=binding(self.path / 'config.json'))
        previous = Path(self.config['predecessor'])
        previous_step, previous_sha = 5000, self.config['predecessor_sha256']
        for step in STEPS:
            # Source transfer/export is CPU-only and may overlap the preceding
            # model. All namespace phases wait for its complete final30 first.
            record = self.stage(step)
            preceding = self.wait_complete(previous, previous_step, previous_sha)
            self.record(step, preceding_final=preceding)
            framework = record['framework']
            host = Path(framework['framework_host'])
            digest = record['backup']['expected_model_sha256']
            cpu = self.namespace_phase(step, framework, 'cpu-check')
            admission = host / 'ROOT_GPU_LAUNCH_ADMISSION.json'
            if not admission.exists():
                pin = read(host / 'source_pin.json')
                require(pin['model_sha256'] == digest and pin['authorized_steps'] == [step],
                        'Only this downloaded model may receive GPU admission')
                marker = read(host / 'namespace_policy.json')['namespace_marker']
                write(admission, dict(authorized=True, namespace=marker,
                    model_sha256=digest, allowed_phases=['preflight', 'run'], coordinator='root',
                    reason='用户授权当前5k完成后按4000/6000/7000/8000/9000/10000逐一独立测评；实际CPU资格已通过',
                    CPU_acceptance=binding(cpu), checked_utc=now(), actual_GPU_preflight_passed=False))
            self.namespace_phase(step, framework, 'preflight')
            self.namespace_phase(step, framework, 'run')
            experiment = Path(framework['experiment_host'])
            final = complete_result(experiment, step, digest)
            require(final is not None, 'Namespace run did not finish thirty audited cases')
            self.record(step, phase='complete', result=final, finished_utc=now())
            previous, previous_step, previous_sha = experiment, step, digest
        self.save(status='complete', current_step=None, pid=None, finished_utc=now())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--resume', action='store_true', help='Reuse only verified complete phases; errors require review')
    args = parser.parse_args()
    path = args.campaign.resolve(strict=True)
    config = read(path / 'config.json')
    require(config.get('contract') == CONTRACT and config.get('steps') == list(STEPS),
            'Explicit fixed checkpoint order is required')
    require(config['hostname'] == socket.gethostname(), 'Wrong local evaluation host')
    (path / 'logs').mkdir(exist_ok=True)
    with (path / 'queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        queue = Queue(config, path, lock)
        if queue.state_path.exists():
            require(args.resume and queue.state['status'] != 'complete', 'Existing queue requires explicit reviewed --resume')
            require(queue.state.get('config', {}).get('sha256') == sha(path / 'config.json'),
                    'Do not change a persisted campaign configuration during resume')
            child = queue.state.get('child')
            require(not child or not same_live_process(child), 'Prior owned child still runs; leave it untouched')
        try:
            queue.execute()
        except BaseException as error:
            queue.save(status='execution_error', error=repr(error), pid=None,
                       failed_utc=now(), counted_as_policy_failure=False)
            raise


if __name__ == '__main__':
    main()
