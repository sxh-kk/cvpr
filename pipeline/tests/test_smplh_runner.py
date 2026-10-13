import importlib.util
import json
from pathlib import Path
import sys

import pytest

spec=importlib.util.spec_from_file_location("smplh_runner_test_module",Path(__file__).parents[1]/"scripts/process_smplh_dataset.py")
runner=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=runner
spec.loader.exec_module(runner)


def _initializer(*args):
    pass


def _worker(row):
    if row['episode_index']==0:
        raise ValueError('synthetic rejected episode')
    return row['episode_index']


def test_failed_worker_is_logged_and_other_process_results_are_consumed(monkeypatch,capsys):
    monkeypatch.setattr(runner,'worker_init',_initializer)
    completed=[]
    with pytest.raises(RuntimeError,match='1 worker tasks failed'):
        runner.parallel([({'episode_index':i},) for i in range(3)],_worker,None,None,2,completed.append)
    assert sorted(completed)==[1,2]
    failure=json.loads(capsys.readouterr().out.strip())
    assert failure['event']=='worker_failed'
    assert failure['episode_index']==0
    assert failure['error_type']=='ValueError'


def test_terminal_failure_updates_job_status_without_claiming_complete(tmp_path,monkeypatch):
    from types import SimpleNamespace
    job=tmp_path/'job';job.mkdir()
    args=SimpleNamespace(job=job,action='run')
    monkeypatch.setattr(runner,'parser',lambda:SimpleNamespace(parse_args=lambda:args))
    def fail(args):
        raise ValueError('synthetic source contract mismatch')
    monkeypatch.setattr(runner,'execute',fail)
    with pytest.raises(ValueError,match='source contract'):
        runner.main()
    status=json.loads((job/'status.json').read_text())
    assert status['state']=='failed'
    assert status['failed_action']=='run'
    assert status['error_type']=='ValueError'
    assert json.loads((tmp_path/'current_smplh_sequence.json').read_text())==status


def test_worker_rejects_producer_or_model_replacement_after_loading(tmp_path,monkeypatch):
    from hashlib import sha256
    producer=tmp_path/'producer.py';producer.write_text('frozen producer')
    model=tmp_path/'model.npz';model.write_bytes(b'frozen model')
    monkeypatch.setattr(runner,'_worker_plan',{
        'implementation_sha256':{str(producer):sha256(producer.read_bytes()).hexdigest()},
        'model_path':str(model),
    })
    monkeypatch.setattr(runner,'_worker_model_identity',runner.file_identity(model))
    runner.assert_worker_code_current()
    producer.write_text('edited while worker remains loaded')
    with pytest.raises(ValueError,match='Producer changed during processing'):
        runner.assert_worker_code_current()
    producer.write_text('frozen producer')
    replacement=tmp_path/'replacement';replacement.write_bytes(b'frozen model')
    replacement.replace(model)
    with pytest.raises(ValueError,match='Model asset changed during processing'):
        runner.assert_worker_code_current()
