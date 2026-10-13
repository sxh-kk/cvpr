"""Real subprocess environment and interrupted lifecycle safety regressions."""
import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("parameter_lifecycle_test", SCRIPTS / "run_smplh_parameter_job.py")
lifecycle = importlib.util.module_from_spec(spec); spec.loader.exec_module(lifecycle)


@pytest.fixture
def job(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir(); (tmp_path / "reports").mkdir()
    plan = {"output_root": str(tmp_path / "standalone"), "egosuite_fitting_job": str(tmp_path / "oldfit")}
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    monkeypatch.setattr(lifecycle, "children", {})
    monkeypatch.setattr(lifecycle, "job_lock", None)
    monkeypatch.setattr(lifecycle, "assert_frozen_job", lambda path: {})
    yield tmp_path
    if lifecycle.job_lock is not None: lifecycle.job_lock.close()
    for child in lifecycle.children.values():
        if child.poll() is None: child.terminate(); child.wait(timeout=5)


def test_child_receives_explicit_pipeline_import_path(job, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/previous/import/path")
    child = lifecycle.launch(job, "environment", [sys.executable, "-c",
        "import os;print(os.environ['PYTHONPATH']);import egomotion_pipeline"])
    assert child.wait(timeout=10) == 0
    value = (job / "logs/environment.log").read_text()
    assert value.startswith(str(lifecycle.WORKSPACE / "pipeline/src"))
    assert "/previous/import/path" in value


def test_failure_in_second_child_is_seen_while_first_is_running(job, monkeypatch):
    class Child:
        def __init__(self, code): self.code = code; self.pid = 123
        def poll(self): return self.code
    lifecycle.children.update(first=Child(None), second=Child(1))
    monkeypatch.setattr(lifecycle.time, "sleep", lambda _: pytest.fail("Failed second child was ignored"))
    with pytest.raises(RuntimeError, match="second failed"):
        lifecycle.wait_children(job, ["first", "second"], "test")
    lifecycle.children.clear()


def test_registered_cleanup_resume_never_reopens_old_sources(job, monkeypatch):
    (job / "reports/dense-cleanup-manifest.json").write_text("{}")
    monkeypatch.setattr(lifecycle, "assert_no_orphan_jobs", lambda path: None)
    called = []
    monkeypatch.setattr(lifecycle, "finish_cleanup", lambda *args: called.append("verified_cleanup"))
    monkeypatch.setattr(lifecycle, "launch", lambda *args: pytest.fail("Resume attempted source conversion"))
    lifecycle.run(argparse.Namespace(job=str(job)))
    assert called == ["verified_cleanup"]


def test_duplicate_wrapper_cannot_overwrite_running_status(job, monkeypatch):
    status = job / "status.json"; status.write_text('{"state":"converting"}\n'); before = status.read_bytes()
    lock = (job / "lifecycle.lock").open("a+")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    monkeypatch.setattr(sys, "argv", ["lifecycle", "--job", str(job)])
    try:
        with pytest.raises(BlockingIOError): lifecycle.main()
    finally: lock.close()
    assert status.read_bytes() == before


def test_orphan_guard_precedes_cleanup_resume_and_initialization(job, monkeypatch):
    (job / "reports/dense-cleanup-manifest.json").write_text("{}")
    def reject(path): raise RuntimeError("Previous job child 123 is still running")
    monkeypatch.setattr(lifecycle, "assert_no_orphan_jobs", reject)
    monkeypatch.setattr(lifecycle, "finish_cleanup", lambda *args: pytest.fail("Orphan guard bypassed"))
    monkeypatch.setattr(lifecycle, "launch", lambda *args: pytest.fail("Duplicate writer started"))
    with pytest.raises(RuntimeError, match="still running"):
        lifecycle.run(argparse.Namespace(job=str(job)))
