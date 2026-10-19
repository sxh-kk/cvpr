#!/usr/bin/env python3
"""Detached CPU observer: refresh a local loss plot at completed update boundaries.

Never imports training code or signals the observed process. W&B extraction and
Matplotlib rendering use separate interpreters; neither modifies the training env.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def process_identity(pid):
    """Linux start ticks protect against a reused PID; zombies have already exited."""
    try:
        fields = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else fields[19]
    except (OSError, ValueError, IndexError):
        return None


def console_progress(path):
    with Path(path).open('rb') as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 65536))
        tail = handle.read().decode(errors='replace')
    rows = re.findall(r'Training steps:[^\r\n]*?\|\s*(\d+)/(\d+)', tail)
    return tuple(map(int, rows[-1])) if rows else (0, 0)


def observed_binding(state):
    return state['timestamp'], str(Path(state['run_directory']).resolve()), int(state['pid'])


def rendered_update(output, state, window=100):
    try:
        marker = read_json(output / 'loss_plot_metadata.json')
        if (marker['run_timestamp'] != state['timestamp']
                or Path(marker['run_directory']).resolve() != Path(state['run_directory']).resolve()
                or marker.get('window_size') != window
                or not (output / 'loss_100step.png').is_file()
                or not (output / 'loss_100step.svg').is_file()):
            return 0
        return int(marker['latest_complete_update'])
    except (OSError, ValueError, KeyError):
        return 0


def refresh(args, state, minimum):
    common = [str(args.curve_script), '--state', str(args.state), '--output-dir', str(args.output_dir),
              '--run-timestamp', state['timestamp'], '--window', str(args.window)]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    # Do not let a caller's plot-only Python packages enter the W&B runtime.
    env.pop('PYTHONPATH', None)
    commands = [(args.extract_python, [*common, '--mode', 'extract'], env)]
    plot_env = dict(env, MPLCONFIGDIR=str(args.output_dir / '.matplotlib'))
    if args.plot_library_dir:
        plot_env['PYTHONPATH'] = str(args.plot_library_dir)
    commands.append((args.plot_python, [*common, '--mode', 'plot',
                                      '--minimum-complete-update', str(minimum)], plot_env))
    for interpreter, command, child_env in commands:
        result = subprocess.run([interpreter, *command], env=child_env, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=60)
        if result.returncode == 75:
            return False  # Writer has not flushed a whole window; retry later.
        if result.returncode:
            raise RuntimeError(f'{command[-1]} failed ({result.returncode}): '
                               f'{result.stderr[-3000:]} {result.stdout[-1000:]}')
    return True


def watch(args):
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'monitor.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('Loss monitor already running for this output directory.', flush=True)
            return 0
        os.nice(10)
        state = read_json(args.state)
        binding = observed_binding(state)
        source_identity = process_identity(state['pid'])
        last = rendered_update(output, state, args.window)
        status = {'state': 'running', 'pid': os.getpid(), 'process_start_ticks': process_identity(os.getpid()),
                  'started_utc': now(), 'run_timestamp': state['timestamp'],
                  'run_directory': state['run_directory'], 'training_pid': state['pid'],
                  'training_process_start_ticks': source_identity,
                  'window': args.window, 'poll_seconds': args.poll_seconds,
                  'last_rendered_complete_update': last, 'refresh_count': 0, 'error_count': 0,
                  'last_error': None, 'cuda_visible_devices': ''}
        status['heartbeat_utc'] = now()
        write_json(output / 'monitor_status.json', status)
        while True:
            alive = source_identity is not None and process_identity(state['pid']) == source_identity
            status['training_alive'] = alive
            current, maximum = status.get('latest_console_update', 0), 0
            try:
                if observed_binding(read_json(args.state)) != binding:
                    status['state'] = 'stopped_run_changed'
                    break
                current, maximum = console_progress(state['log'])
                target = current // args.window * args.window
                status.update(latest_console_update=current, training_alive=alive)
                if target > last or last == 0 or not alive:
                    minimum = max(target, args.window) if alive else 0
                    if refresh(args, state, minimum):
                        last = rendered_update(output, state, args.window)
                        status.pop('waiting_for_flushed_window', None)
                        status.update(last_rendered_complete_update=last, last_refreshed_utc=now(),
                                      refresh_count=status['refresh_count'] + 1, last_error=None)
                        with (output / 'monitor_events.jsonl').open('a') as events:
                            events.write(json.dumps({'utc': now(), 'event': 'plot_refreshed',
                                                     'complete_update': last, 'console_update': current}) + '\n')
                        print(f'{now()} refreshed through complete update {last}', flush=True)
                    else:
                        status['waiting_for_flushed_window'] = minimum
                if not alive:
                    status['state'] = 'completed' if maximum and current >= maximum else 'training_stopped'
                    break
                status['next_refresh_update'] = (last // args.window + 1) * args.window
            except Exception as error:
                status.update(last_error=f'{type(error).__name__}: {error}',
                              error_count=status['error_count'] + 1)
                print(f'{now()} retry: {status["last_error"]}', flush=True)
            finally:
                if not alive and status['state'] == 'running':
                    # A missing log or failed final render must never keep a
                    # dead training run under observation indefinitely.
                    status['state'] = 'completed' if maximum and current >= maximum else 'training_stopped'
                status['heartbeat_utc'] = now()
                write_json(output / 'monitor_status.json', status)
            if args.mode == 'once' or not alive:
                break
            time.sleep(args.poll_seconds)
        status['heartbeat_utc'] = now()
        write_json(output / 'monitor_status.json', status)
        return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('start', 'watch', 'once', 'status'), default='start')
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--extract-python', default=sys.executable)
    parser.add_argument('--plot-python', default=sys.executable)
    parser.add_argument('--plot-library-dir', type=Path)
    parser.add_argument('--curve-script', type=Path, default=Path(__file__).with_name('training_loss_curve.py'))
    parser.add_argument('--window', type=int, default=100)
    parser.add_argument('--poll-seconds', type=float, default=10)
    args = parser.parse_args()
    if args.window <= 0 or args.poll_seconds < 1:
        parser.error('window must be positive and poll-seconds >= 1')
    args.state, args.output_dir = args.state.resolve(), args.output_dir.resolve()
    if args.mode == 'status':
        status = read_json(args.output_dir / 'monitor_status.json')
        status['monitor_alive'] = process_identity(status['pid']) == status['process_start_ticks']
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return 0
    if args.mode in ('watch', 'once'):
        return watch(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(Path(__file__).resolve()), '--mode', 'watch', '--state', str(args.state),
               '--output-dir', str(args.output_dir), '--extract-python', args.extract_python,
               '--plot-python', args.plot_python, '--curve-script', str(args.curve_script.resolve()),
               '--window', str(args.window), '--poll-seconds', str(args.poll_seconds)]
    if args.plot_library_dir:
        command += ['--plot-library-dir', str(args.plot_library_dir.resolve())]
    with (args.output_dir / 'monitor.log').open('ab', buffering=0) as log:
        child = subprocess.Popen(command, start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=log, stderr=subprocess.STDOUT, close_fds=True)
    for _ in range(50):
        try:
            status = read_json(args.output_dir / 'monitor_status.json')
            if (status['pid'] == child.pid or child.poll() == 0) and process_identity(status['pid']) == status['process_start_ticks']:
                print(json.dumps(status, ensure_ascii=False, indent=2))
                return 0
        except (OSError, ValueError, KeyError):
            pass
        if child.poll() is not None:
            raise RuntimeError(f'Monitor exited with {child.returncode}; see {args.output_dir / "monitor.log"}')
        time.sleep(0.1)
    raise RuntimeError('Monitor startup still pending; inspect monitor.log/status before starting another observer.')


if __name__ == '__main__':
    raise SystemExit(main())
