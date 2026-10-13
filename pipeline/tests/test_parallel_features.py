import multiprocessing as mp
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from conftest import make_egosuite
from test_archive_and_video import FakeEncoder, source_config
from egomotion_pipeline.cli import convert
from egomotion_pipeline.features import extract_features
from egomotion_pipeline.parallel_features import extract_features_parallel
from egomotion_pipeline.storage import Archive, read_arrays
from egomotion_pipeline.validation import validate


class ParallelFakeEncoder(FakeEncoder):
    def __init__(self, config):
        pass

    def __call__(self, images):
        if np.mean(images[0]) < 4:
            time.sleep(.2)  # Make the first episode finish after later ones.
        return super().__call__(images)


class FailingEncoder(ParallelFakeEncoder):
    def __call__(self, images):
        if os.environ['CUDA_VISIBLE_DEVICES']=='0':
            time.sleep(1)
            raise ValueError('injected video/model failure')
        return FakeEncoder.__call__(self, images)


class MixedEncoder(ParallelFakeEncoder):
    def __init__(self, config):
        self.metadata = {**FakeEncoder.metadata, 'checkpoint':os.environ['CUDA_VISIBLE_DEVICES']}


class ExitingEncoder(ParallelFakeEncoder):
    def __init__(self, config):
        os._exit(27)


def make_archive(tmp_path, episodes=4):
    source = make_egosuite(tmp_path/'source', episodes=episodes)
    config = source_config(source)
    convert(config, tmp_path/'parallel')
    return Archive(tmp_path/'parallel'), config


def test_parallel_matches_serial_out_of_order_and_resumes_with_other_gpus(tmp_path):
    archive, source = make_archive(tmp_path)
    convert(source, tmp_path/'serial')
    serial = Archive(tmp_path/'serial')
    config = {'batch_size':2, 'device':'cuda'}
    before = {r['episode_index']:read_arrays(archive.data_path(r['episode_index']),archive.specs)
              for r in archive.records()}
    events = []
    with archive.lock():
        extract_features_parallel(archive,config,[0,1],encoder_factory=ParallelFakeEncoder,
                                  progress=events.append)
    extract_features(serial,config,FakeEncoder())
    committed = [e['episode_index'] for e in events if e['kind']=='committed']
    assert committed[0] != 0 and sorted(committed)==list(range(4))
    changed = {'observation.features.dino','observation.valid.dino',
               'observation.video_timestamp','observation.video_time_error'}
    for record in archive.records():
        index=record['episode_index']
        actual=read_arrays(archive.data_path(index),archive.specs)
        expected=read_arrays(serial.data_path(index),serial.specs)
        for key in actual:
            np.testing.assert_array_equal(actual[key],expected[key])
            if key not in changed:
                np.testing.assert_array_equal(actual[key],before[index][key])
    assert validate(Archive(archive.root),require_dino=True)['episodes']==4
    mtimes=[archive.data_path(i).stat().st_mtime_ns for i in range(4)]
    fresh=Archive(archive.root)
    with fresh.lock():
        extract_features_parallel(fresh,config,[3],encoder_factory=ParallelFakeEncoder)
    assert [archive.data_path(i).stat().st_mtime_ns for i in range(4)]==mtimes
    with pytest.raises(ValueError,match='different model'):
        extract_features_parallel(Archive(archive.root),{**config,'batch_size':3},[0,1],
                                  encoder_factory=ParallelFakeEncoder)
    assert [archive.data_path(i).stat().st_mtime_ns for i in range(4)]==mtimes


def test_worker_failure_keeps_commits_and_cleans_workers_before_resume(tmp_path):
    archive,_=make_archive(tmp_path,episodes=2)
    config={'batch_size':2}
    with archive.lock(),pytest.raises(RuntimeError,match='injected video/model failure'):
        extract_features_parallel(archive,config,[0,1],encoder_factory=FailingEncoder)
    fresh=Archive(archive.root)
    assert [r['dino']['state'] for r in fresh.records()]==['pending','complete']
    assert not read_arrays(fresh.data_path(0),fresh.specs)['observation.valid.dino'].any()
    assert not [p for p in mp.active_children() if p.name.startswith('dino-gpu-')]
    with fresh.lock():
        extract_features_parallel(fresh,config,[0,1],encoder_factory=ParallelFakeEncoder)
    assert validate(Archive(archive.root),require_dino=True)['episodes']==2


def test_mixed_worker_metadata_rejected_before_any_write(tmp_path):
    archive,_=make_archive(tmp_path,episodes=2)
    with archive.lock(),pytest.raises(ValueError,match='different DINO encoders'):
        extract_features_parallel(archive,{},[0,1],encoder_factory=MixedEncoder)
    assert all(r['dino']['state']=='pending' for r in Archive(archive.root).records())


def test_abrupt_worker_exit_does_not_hang_or_leave_workers(tmp_path):
    archive,_=make_archive(tmp_path,episodes=2)
    with archive.lock(),pytest.raises(RuntimeError,match='exited'):
        extract_features_parallel(archive,{},[0,1],encoder_factory=ExitingEncoder)
    assert not [p for p in mp.active_children() if p.name.startswith('dino-gpu-')]
    assert all(r['dino']['state']=='pending' for r in Archive(archive.root).records())


@pytest.mark.parametrize('gpu_ids',[[],[0,0],[-1],['0'],[True]])
def test_invalid_gpu_ids_rejected_without_starting_workers(tmp_path,gpu_ids):
    archive,_=make_archive(tmp_path,episodes=1)
    with pytest.raises(ValueError,match='GPU IDs'):
        extract_features_parallel(archive,{},gpu_ids,encoder_factory=ParallelFakeEncoder)


def test_cpu_device_rejected_for_parallel_gpu_dispatch(tmp_path):
    archive,_=make_archive(tmp_path,episodes=1)
    with pytest.raises(ValueError,match="device='cuda'"):
        extract_features_parallel(archive,{'device':'cpu'},[0],encoder_factory=ParallelFakeEncoder)


def load_runner(monkeypatch):
    path=Path(__file__).parents[1]/'scripts/process_dino.py'
    spec=importlib.util.spec_from_file_location('test_dino_job_runner',path)
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.signal,'signal',lambda *args:None)
    def extract(archive,config,gpus,**kwargs):
        return extract_features_parallel(archive,config,gpus,
            encoder_factory=ParallelFakeEncoder,**kwargs)
    monkeypatch.setattr(runner,'extract_features_parallel',extract)
    return runner


def test_runner_updates_out_of_order_progress_and_validates(tmp_path,monkeypatch):
    archive,_=make_archive(tmp_path,episodes=2)
    runner=load_runner(monkeypatch)
    config=tmp_path/'dino.json';config.write_text(json.dumps({'batch_size':2}))
    prerequisite=tmp_path/'upgrade';prerequisite.mkdir()
    (prerequisite/'status.json').write_text(json.dumps({'stage':'completed',
        'validation_passed':True,'dataset':str(archive.root)}))
    job=tmp_path/'job'
    assert runner.main(['--dataset',str(archive.root),'--config',str(config),
        '--job-dir',str(job),'--gpus','0','1','--wait-for-job',str(prerequisite)])==0
    status=json.loads((job/'status.json').read_text())
    report=json.loads((job/'validation.json').read_text())
    assert status['stage']=='completed' and status['completed_episodes']==2
    assert status['gpu_count']==2 and set(status['workers'])=={'0','1'}
    assert report['dino_completed_episodes']==2 and report['dino_missing_frames']==0


def test_runner_stops_before_extraction_on_failed_prerequisite(tmp_path,monkeypatch):
    archive,_=make_archive(tmp_path,episodes=1)
    runner=load_runner(monkeypatch)
    prerequisite=tmp_path/'upgrade';prerequisite.mkdir()
    (prerequisite/'status.json').write_text(json.dumps({'stage':'failed',
        'dataset':str(archive.root),'fatal_error':'injected upgrade error'}))
    job=tmp_path/'job'
    assert runner.main(['--dataset',str(archive.root),'--config',str(tmp_path/'unused.json'),
        '--job-dir',str(job),'--gpus','0','1','--wait-for-job',str(prerequisite)])==1
    status=json.loads((job/'status.json').read_text())
    assert status['stage']=='failed' and 'injected upgrade error' in status['fatal_error']
    assert Archive(archive.root).records()[0]['dino']['state']=='pending'
