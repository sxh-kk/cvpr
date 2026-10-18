"""Failed probe preparation must remain observable without hiding its cause."""
import json
import sys
from types import SimpleNamespace

import pytest

from MotionExpert import probe_prior as probe


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def status_fixture(tmp_path, owner="other"):
    job = tmp_path / "probe"
    status_path = tmp_path / ".pipeline/current_motionexpert_prior_probe.json"
    current = {"state": "extracting", "job_dir": str(job if owner == "this" else tmp_path / "other"),
               "pid": 12345}
    write_json(status_path, current)
    cfg = SimpleNamespace(DATA=SimpleNamespace(
        ARCHIVE_ROOT=str(tmp_path / "archive"), WINDOW_CACHE_ROOT=str(tmp_path / "cache/windows"),
        VISION_PROFILE_ROOT=str(tmp_path / "vision")))
    return job, status_path, current, cfg


@pytest.mark.parametrize("failure", ("cache", "reuse"))
def test_failed_preparation_records_job_without_overwriting_other_job(tmp_path, monkeypatch, failure):
    from MotionExpert import config
    from MotionExpert.dataset import probe_cache

    job, status_path, current, cfg = status_fixture(tmp_path)
    monkeypatch.setattr(config, "load_config", lambda path: cfg)
    original = ValueError("accepted cache changed") if failure == "cache" else FileNotFoundError("old targets missing")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr(probe_cache, "AcceptedProbeCache", fail if failure == "cache" else lambda *args: object())
    monkeypatch.setattr(probe, "prepare_reuse", fail)
    monkeypatch.setattr(sys, "argv", ["probe_prior", "--output", str(job), "--reuse-job", str(tmp_path / "baseline"),
                                      "--checkpoint160", "unused.ckpt", "--gpus", "0", "--stage", "extract"])
    with pytest.raises(type(original)) as raised:
        probe.main()
    assert raised.value is original
    progress = json.loads((job / "progress.json").read_text())
    assert progress["stage"] == "failed" and repr(original) == progress["error"]
    assert "Traceback" in progress["traceback"]
    assert json.loads(status_path.read_text()) == current


@pytest.mark.parametrize("failure", ("missing_plan", "missing_extraction", "rejected_extraction", "changed_checkpoint", "changed_reuse"))
@pytest.mark.parametrize("owner", ("this", "other"))
def test_fit_cli_precheck_failures_record_job_and_respect_global_owner(tmp_path, monkeypatch, failure, owner):
    from MotionExpert import prior_probe_fit

    job, status_path, current, cfg = status_fixture(tmp_path, owner)
    job.mkdir()
    checkpoint = tmp_path / "frozen.ckpt"
    checkpoint.write_bytes(b"frozen")
    plan = {"job_dir": str(job), "cache_root": cfg.DATA.WINDOW_CACHE_ROOT,
            "checkpoints": {"160": str(checkpoint)}, "checkpoint_stamps": {"160": probe.file_stamp(checkpoint)}}
    if failure != "missing_plan":
        write_json(job / "plan.json", plan)
    if failure not in ("missing_plan", "missing_extraction"):
        write_json(job / "extraction-complete.json", {"passed": failure != "rejected_extraction"})
    if failure == "changed_checkpoint":
        checkpoint.write_bytes(b"changed frozen checkpoint")
    if failure == "changed_reuse":
        def changed_reuse(plan):
            raise ValueError("Reused probe artifact changed")
        monkeypatch.setattr(probe, "assert_reuse_current", changed_reuse)
    monkeypatch.setattr(prior_probe_fit, "fit_job", lambda *args, **kwargs: pytest.fail("Precheck failure must not start fitting"))
    monkeypatch.setattr(sys, "argv", ["probe_prior", "--output", str(job), "--stage", "fit", "--gpus", "0"])
    expected_error = FileNotFoundError if failure in ("missing_plan", "missing_extraction") else ValueError
    with pytest.raises(expected_error):
        probe.main()
    assert json.loads((job / "progress.json").read_text())["stage"] == "failed"
    global_status = json.loads(status_path.read_text())
    if owner == "this" and failure != "missing_plan":
        assert global_status["state"] == "failed" and global_status["job_dir"] == str(job)
    else:
        assert global_status == current


def test_failure_recording_errors_preserve_original_precheck_exception(tmp_path, monkeypatch, capsys):
    job, status_path, _, cfg = status_fixture(tmp_path, "this")
    job.mkdir()
    write_json(job / "plan.json", {"job_dir": str(job), "cache_root": cfg.DATA.WINDOW_CACHE_ROOT})
    write_json(job / "extraction-complete.json", {"passed": False})

    def broken_writer(*args, **kwargs):
        raise OSError("status storage is unavailable")

    monkeypatch.setattr(probe, "event", broken_writer)
    monkeypatch.setattr(probe, "atomic_json", broken_writer)
    with pytest.raises(ValueError, match="requires completed GPU extraction"):
        probe.fit_and_record(job, device="cpu")
    assert capsys.readouterr().err.count("Unable to record probe failure") == 2
    assert json.loads(status_path.read_text())["state"] == "extracting"


def test_fit_cli_missing_job_records_failure_without_global_plan(tmp_path, monkeypatch):
    job, status_path, current, _ = status_fixture(tmp_path)
    monkeypatch.setattr(sys, "argv", ["probe_prior", "--output", str(job), "--stage", "fit", "--gpus", "0"])
    with pytest.raises(FileNotFoundError):
        probe.main()
    assert json.loads((job / "progress.json").read_text())["stage"] == "failed"
    assert json.loads(status_path.read_text()) == current


def test_foreign_plan_does_not_identify_global_status(tmp_path):
    job, status_path, current, cfg = status_fixture(tmp_path, "this")
    job.mkdir()
    write_json(job / "plan.json", {"job_dir": str(tmp_path / "different"), "cache_root": cfg.DATA.WINDOW_CACHE_ROOT})
    with pytest.raises(ValueError, match="belongs to another job"):
        probe.fit_and_record(job, device="cpu")
    assert json.loads((job / "progress.json").read_text())["stage"] == "failed"
    assert json.loads(status_path.read_text()) == current
