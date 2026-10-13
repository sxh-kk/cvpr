"""Actual process contention, wait semantics and atomic episode commit regression.

The default module is a reviewed candidate copy outside the frozen producer.
Set SMPLH_SIDECAR_CANDIDATE_PATH to exercise the frozen v1 negative control.
When this test is applied to the repository, the production module is used.
"""
from copy import deepcopy
import fcntl
import importlib.util
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import sys
import time

import numpy as np
import pytest

from egomotion_pipeline.schema import SCHEMA_VERSION, feature_specs_for_schema
from egomotion_pipeline.storage import write_arrays
from test_smplh_sidecar import case as original_case

_candidate_default=Path(__file__).parent/"candidate/smplh_sidecar.py"
_candidate_path=os.environ.get("SMPLH_SIDECAR_CANDIDATE_PATH")
if _candidate_path or _candidate_default.exists():
    spec=importlib.util.spec_from_file_location("egomotion_pipeline._sidecar_process_candidate",_candidate_path or _candidate_default)
    sidecar=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=sidecar
    spec.loader.exec_module(sidecar)
else:
    from egomotion_pipeline import smplh_sidecar as sidecar

pytestmark=pytest.mark.skipif("fork" not in mp.get_all_start_methods(),reason="Production file locks require POSIX process semantics")


def _two_episodes(tmp_path):
    source,record,arrays,fit,contract,out=original_case.__wrapped__(tmp_path)
    cases=[]
    for index in range(2):
        row=deepcopy(record)
        row["episode_index"]=index
        row["source_key"]+=f":process-fixture-{index}"
        projected=deepcopy(arrays)
        projected["episode_index"][:]=index
        projected["index"][:]=np.arange(8)+8*index
        write_arrays(source.data_path(index),projected,feature_specs_for_schema(SCHEMA_VERSION))
        frozen=sidecar.build_sidecar_contract(source.root,source.data_path(index),contract["model_identity"]["path"],
                                              contract["config"],contract["calibration"],projected)
        cases.append((row,projected,deepcopy(fit),frozen))
    return source,out,cases


def _writer_child(output,case,inherited_parent_fd,queue,stage_release,fail_after_stage=False):
    # The child must not retain the parent's open-description reference to its
    # pre-held lock. Its own writer opens the root lock independently.
    os.close(inherited_parent_fd)
    row,arrays,fit,contract=case
    index=row["episode_index"]
    actual_flock=fcntl.flock
    original_write=sidecar._write_table
    def observed_flock(fd,operation):
        if operation & fcntl.LOCK_EX:
            queue.put(("waiting",index))
            started=time.monotonic()
            value=actual_flock(fd,operation)
            queue.put(("acquired",index,time.monotonic()-started))
            return value
        return actual_flock(fd,operation)
    def observed_write(path,arrays,fingerprint):
        original_write(path,arrays,fingerprint)
        queue.put(("parquet_staged",index,str(path)))
        if not stage_release.wait(15):
            raise RuntimeError("Parent did not release atomic staging observer")
        if fail_after_stage:
            raise OSError("injected process failure after Parquet staging")
    fcntl.flock=observed_flock
    sidecar._write_table=observed_write
    try:
        report=sidecar.write_episode_sidecar(output,row,arrays,fit,contract)
        queue.put(("done",index,report))
    except Exception as error:
        queue.put(("error",index,type(error).__name__,str(error)))
    finally:
        fcntl.flock=actual_flock
        sidecar._write_table=original_write


def _receive(queue,predicate,history,timeout=10):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        value=queue.get(timeout=max(.01,deadline-time.monotonic()))
        history.append(value)
        if predicate(value):
            return value
    raise AssertionError(f"Expected process event missing: {history}")


def _cleanup(processes,release,lock):
    release.set()
    fcntl.flock(lock,fcntl.LOCK_UN)
    for process in processes:
        process.join(10)
        if process.is_alive():
            process.terminate();process.join(5)


def test_same_root_processes_wait_then_commit_complete_distinct_episodes(tmp_path):
    source,out,cases=_two_episodes(tmp_path)
    original_bytes={i:source.data_path(i).read_bytes() for i in range(2)}
    context=mp.get_context("fork")
    queue,release=context.Queue(),context.Event()
    history=[];processes=[]
    out.mkdir()
    with (out/".writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            processes=[context.Process(target=_writer_child,args=(out,case,lock.fileno(),queue,release)) for case in cases]
            for process in processes:process.start()
            for _ in range(2):
                event=_receive(queue,lambda value:value[0] in ("waiting","error"),history)
                assert event[0]=="waiting",history
            # Both actual flock calls are contending with a separate process's
            # lock. A nonblocking implementation fails this regression here.
            with pytest.raises(Empty):queue.get(timeout=.3)
            assert all(process.is_alive() for process in processes)
            assert not (out/"episodes").exists()
            fcntl.flock(lock,fcntl.LOCK_UN)
            staged=_receive(queue,lambda value:value[0] in ("parquet_staged","error"),history)
            assert staged[0]=="parquet_staged",history
            stage_path=Path(staged[2])
            assert stage_path.exists() and not stage_path.with_name("receipt.json").exists()
            assert not (out/"episodes"/f"{staged[1]:08d}").exists()
            # A completed staged Parquet is invisible until the receipt and
            # directory commit are finished under the same lock.
            assert list(stage_path.parent.iterdir())==[stage_path]
            release.set()
            while len([v for v in history if v[0] in ("done","error")])<2:
                _receive(queue,lambda value:value[0] in ("done","error"),history)
            assert not [v for v in history if v[0]=="error"],history
            assert {v[1] for v in history if v[0]=="done"}=={0,1}
            waits=[v[2] for v in history if v[0]=="acquired"]
            assert len(waits)==2 and min(waits)>=.25
        finally:
            _cleanup(processes,release,lock)
    assert all(process.exitcode==0 for process in processes)
    for row,arrays,_,contract in cases:
        folder=out/"episodes"/f"{row['episode_index']:08d}"
        assert {p.name for p in folder.iterdir()}=={"motion.parquet","receipt.json"}
        assert sidecar.validate_episode_sidecar(out,row,arrays,contract)["passed"]
        assert source.data_path(row["episode_index"]).read_bytes()==original_bytes[row["episode_index"]]
    assert not list((out/".staging").iterdir())


def test_failed_process_staging_does_not_commit_or_block_other_episode(tmp_path):
    _,out,cases=_two_episodes(tmp_path)
    context=mp.get_context("fork")
    queue,release=context.Queue(),context.Event()
    history=[];processes=[]
    out.mkdir()
    with (out/".writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            processes=[context.Process(target=_writer_child,args=(out,case,lock.fileno(),queue,release,index==0))
                       for index,case in enumerate(cases)]
            for process in processes:process.start()
            for _ in range(2):
                event=_receive(queue,lambda value:value[0] in ("waiting","error"),history)
                assert event[0]=="waiting",history
            with pytest.raises(Empty):queue.get(timeout=.3)
            release.set()
            fcntl.flock(lock,fcntl.LOCK_UN)
            while len([v for v in history if v[0] in ("done","error")])<2:
                _receive(queue,lambda value:value[0] in ("done","error"),history)
        finally:
            _cleanup(processes,release,lock)
    errors=[v for v in history if v[0]=="error"]
    assert len(errors)==1 and errors[0][1:3]==(0,"OSError"),history
    assert "injected process failure" in errors[0][3]
    assert [v[1] for v in history if v[0]=="done"]==[1]
    assert not (out/"episodes/00000000").exists()
    row,arrays,_,contract=cases[1]
    assert sidecar.validate_episode_sidecar(out,row,arrays,contract)["passed"]
    assert not list((out/".staging").iterdir())
    assert all(process.exitcode==0 for process in processes)


def test_killed_writer_leaves_no_half_commit_and_peer_and_retry_succeed(tmp_path):
    _,out,cases=_two_episodes(tmp_path)
    context=mp.get_context("fork")
    queue=context.Queue()
    releases=[context.Event(),context.Event()]
    history=[];processes=[]
    out.mkdir()
    with (out/".writer.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            processes=[context.Process(target=_writer_child,args=(out,case,lock.fileno(),queue,releases[index]))
                       for index,case in enumerate(cases)]
            for process in processes:process.start()
            for _ in range(2):
                event=_receive(queue,lambda value:value[0] in ("waiting","error"),history)
                assert event[0]=="waiting",history
            with pytest.raises(Empty):queue.get(timeout=.3)
            fcntl.flock(lock,fcntl.LOCK_UN)
            staged=_receive(queue,lambda value:value[0] in ("parquet_staged","error"),history)
            assert staged[0]=="parquet_staged",history
            crashed=staged[1]
            stage_path=Path(staged[2])
            assert stage_path.exists() and not stage_path.with_name("receipt.json").exists()
            processes[crashed].kill();processes[crashed].join(5)
            assert processes[crashed].exitcode==-signal.SIGKILL
            assert not (out/"episodes"/f"{crashed:08d}").exists()
            # The OS releases the killed process's advisory lock. Its peer is
            # allowed to finish independently; the abandoned staging directory
            # remains outside the committed episode namespace.
            # Each process has its own observer event: SIGKILL while waiting
            # must not poison a shared Event's condition semaphore for a peer.
            releases[1-crashed].set()
            done=_receive(queue,lambda value:value[0] in ("done","error"),history)
            assert done[0]=="done" and done[1]==1-crashed,history
            assert not [v for v in history if v[0]=="error"],history
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)
            for index,process in enumerate(processes):
                if process.is_alive():
                    releases[index].set()
                process.join(10)
                if process.is_alive():
                    process.terminate();process.join(5)
    row,arrays,fit,contract=cases[crashed]
    assert stage_path.exists() and not stage_path.with_name("receipt.json").exists()
    assert sidecar.write_episode_sidecar(out,row,arrays,fit,contract)["written"]
    for row,arrays,_,contract in cases:
        assert sidecar.validate_episode_sidecar(out,row,arrays,contract)["passed"]
        folder=out/"episodes"/f"{row['episode_index']:08d}"
        assert {p.name for p in folder.iterdir()}=={"motion.parquet","receipt.json"}
