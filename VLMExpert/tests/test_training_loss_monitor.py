"""Exercise monitor step boundaries and retries with CPU-only child processes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).parents[1] / 'scripts/analysis/monitor_training_loss.py'
spec = importlib.util.spec_from_file_location('training_loss_monitor', SCRIPT)
monitor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(monitor)


def test_progress_uses_last_completed_count_and_render_marker_binding(tmp_path):
    log = tmp_path / 'train.log'
    log.write_text('noise\rTraining steps: 1%|x| 99/40000 [loss=3]\r'
                   'Training steps: 1%|x| 100/40000 [loss=2]\n')
    assert monitor.console_progress(log) == (100, 40000)
    state = {'timestamp': 'run', 'run_directory': str(tmp_path)}
    monitor.write_json(tmp_path / 'loss_plot_metadata.json', {
        'run_timestamp': 'other', 'run_directory': str(tmp_path), 'latest_complete_update': 900})
    for suffix in ('png', 'svg'):
        (tmp_path / f'loss_100step.{suffix}').touch()
    assert monitor.rendered_update(tmp_path, state) == 0
    marker = {'run_timestamp': 'run', 'run_directory': str(tmp_path), 'latest_complete_update': 100,
              'window_size': 100}
    monitor.write_json(tmp_path / 'loss_plot_metadata.json', marker)
    assert monitor.rendered_update(tmp_path, state) == 100
    assert monitor.rendered_update(tmp_path, state, 200) == 0
    assert monitor.process_identity(os.getpid())
    assert monitor.process_identity(999999999) is None


def _fixture(tmp_path):
    output = tmp_path / 'out'
    output.mkdir()
    state_path = tmp_path / 'state.json'
    log = tmp_path / 'train.log'
    state = {'timestamp': 'test', 'run_directory': str(tmp_path), 'pid': os.getpid(), 'log': str(log)}
    monitor.write_json(state_path, state)
    fake = tmp_path / 'fake_curve.py'
    fake.write_text('''import argparse, json, os
from pathlib import Path
p=argparse.ArgumentParser()
for name in ('mode','state','output-dir','run-timestamp','window','minimum-complete-update'):
 p.add_argument('--'+name)
a=p.parse_args(); out=Path(a.output_dir); state=json.loads(Path(a.state).read_text())
assert os.environ['CUDA_VISIBLE_DEVICES']==''
assert os.environ['OMP_NUM_THREADS']=='1'
with (out/'calls').open('a') as f: f.write(a.mode+'\\n')
if a.mode=='extract':
 assert 'PYTHONPATH' not in os.environ
else:
 n=int((out/'available').read_text())
 if n<int(a.minimum_complete_update): raise SystemExit(75)
 for suffix in ('png','svg'): (out/('loss_100step.'+suffix)).write_text(str(n))
 (out/'loss_plot_metadata.json').write_text(json.dumps({'run_timestamp':a.run_timestamp,
 'run_directory':state['run_directory'],'latest_complete_update':n,'window_size':int(a.window)}))
''')
    args = SimpleNamespace(mode='once', state=state_path, output_dir=output, curve_script=fake,
                           extract_python=sys.executable, plot_python=sys.executable,
                           plot_library_dir=None, window=100, poll_seconds=1)
    return args, state, log


def test_actual_child_refresh_only_on_new_complete_window_and_retry_not_flushed(tmp_path, monkeypatch):
    args, state, log = _fixture(tmp_path)
    monkeypatch.setattr(monitor.os, 'nice', lambda value: None)
    monkeypatch.setenv('PYTHONPATH', 'must-not-reach-wandb')
    log.write_text('Training steps: 1%|x| 100/40000 [loss=2]\r')
    (args.output_dir / 'available').write_text('100')
    assert monitor.watch(args) == 0
    assert (args.output_dir / 'calls').read_text().splitlines() == ['extract', 'plot']
    assert monitor.rendered_update(args.output_dir, state) == 100

    log.write_text('Training steps: 1%|x| 199/40000 [loss=2]\r')
    assert monitor.watch(args) == 0
    assert (args.output_dir / 'calls').read_text().splitlines() == ['extract', 'plot']

    log.write_text('Training steps: 1%|x| 200/40000 [loss=2]\r')
    assert monitor.watch(args) == 0  # Step200 has not been flushed by W&B yet.
    assert monitor.rendered_update(args.output_dir, state) == 100
    assert (args.output_dir / 'loss_100step.png').read_text() == '100'
    status = monitor.read_json(args.output_dir / 'monitor_status.json')
    assert status['waiting_for_flushed_window'] == 200

    (args.output_dir / 'available').write_text('200')
    assert monitor.watch(args) == 0
    assert monitor.rendered_update(args.output_dir, state) == 200
    events = [json.loads(line) for line in (args.output_dir / 'monitor_events.jsonl').read_text().splitlines()]
    assert [event['complete_update'] for event in events] == [100, 200]
    assert monitor.read_json(args.output_dir / 'monitor_status.json')['next_refresh_update'] == 300


def test_real_detached_start_duplicate_lock_and_training_end_exit(tmp_path):
    import time

    args, state, log = _fixture(tmp_path)
    # Bind the observer to an expendable CPU process, never an actual training PID.
    source = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    state['pid'] = source.pid
    monitor.write_json(args.state, state)
    (args.output_dir / 'available').write_text('100')
    log.write_text('Training steps: 1%|x| 100/40000 [loss=2]\r')
    command = [sys.executable, str(SCRIPT), '--state', str(args.state), '--output-dir', str(args.output_dir),
               '--curve-script', str(args.curve_script), '--poll-seconds', '1']
    observer_pid = None
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        observer_pid = json.loads(result.stdout)['pid']
        duplicate = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        assert json.loads(duplicate.stdout)['pid'] == observer_pid
        source.terminate()
        source.wait(timeout=5)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = monitor.read_json(args.output_dir / 'monitor_status.json')
            if status['state'] == 'training_stopped':
                break
            time.sleep(0.1)
        assert status['state'] == 'training_stopped'
        deadline = time.monotonic() + 5
        while monitor.process_identity(observer_pid) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert monitor.process_identity(observer_pid) is None
    finally:
        if source.poll() is None:
            source.terminate()
            source.wait(timeout=5)
        if observer_pid and monitor.process_identity(observer_pid):
            os.kill(observer_pid, 15)  # Owned test observer only.


@pytest.mark.parametrize('failure', ['missing_log', 'failed_final_render'])
def test_dead_training_exits_even_when_final_refresh_fails(tmp_path, failure):
    args, state, log = _fixture(tmp_path)
    state['pid'] = 999999999
    monitor.write_json(args.state, state)
    if failure == 'failed_final_render':
        log.write_text('Training steps: 1%|x| 100/40000 [loss=2]\r')
        args.curve_script.write_text('raise RuntimeError("controlled final failure")\n')
    result = subprocess.run([sys.executable, str(SCRIPT), '--mode', 'watch',
                             '--state', str(args.state), '--output-dir', str(args.output_dir),
                             '--curve-script', str(args.curve_script), '--poll-seconds', '1'],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    status = monitor.read_json(args.output_dir / 'monitor_status.json')
    assert status['state'] == 'training_stopped'
    assert status['error_count'] == 1 and status['last_error']
    assert not status['training_alive']
