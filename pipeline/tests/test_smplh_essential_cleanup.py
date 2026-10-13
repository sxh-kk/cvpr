"""Essential cleanup recovery and mid-run gate protection, using tiny files."""
import importlib.util
from pathlib import Path

import pytest

from test_cleanup_smplh_dense import accepted, cleanup, put
from test_smplh_requalification import qualified, qualified_cleanup


EXECUTOR = Path("/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-qualified-20261012T084000Z/code/execute_essential_cleanup.py")
spec = importlib.util.spec_from_file_location("essential_cleanup_under_test", EXECUTOR)
executor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(executor)


@pytest.fixture
def essential(qualified, monkeypatch):
    monkeypatch.setattr(executor, "legacy", cleanup)
    manifest = qualified_cleanup.prepare_manifest(qualified["job"])
    old_manifest_sha = cleanup.sha(qualified_cleanup.manifest_path(qualified["job"]))
    executor.prepare_policy(qualified["job"])
    return {**qualified, "manifest": manifest, "old_manifest_sha": old_manifest_sha}


def test_pending_unlink_recovers_without_repeating_periodic_full_scans(essential, monkeypatch):
    job = essential["job"]
    original_unlink = cleanup.secure_unlink
    original_check = qualified_cleanup.check_manifest
    calls = []
    def counted_check(*args, **kwargs):
        calls.append(kwargs.get("context") is not None)
        return original_check(*args, **kwargs)
    monkeypatch.setattr(qualified_cleanup, "check_manifest", counted_check)
    def crash_after_unlink(mix, entry):
        original_unlink(mix, entry)
        raise RuntimeError("fixture crash after unlink before done receipt")
    monkeypatch.setattr(cleanup, "secure_unlink", crash_after_unlink)
    with pytest.raises(RuntimeError, match="fixture crash"):
        executor.execute(job)
    partial = cleanup.read(qualified_cleanup.execution_path(job))
    assert partial["pending"] == essential["manifest"]["files"][0]["path"]
    assert partial["deleted"] == [] and not Path(partial["pending"]).exists()
    monkeypatch.setattr(cleanup, "secure_unlink", original_unlink)
    result = executor.execute(job)
    assert result["state"] == "complete" and result["pending"] is None
    assert set(result["deleted"]) == {row["path"] for row in essential["manifest"]["files"]}
    assert result["bytes_deleted"] == essential["manifest"]["bytes"]
    assert calls == [False, False, True]  # Interrupted first; resumed first/final.
    assert cleanup.sha(qualified_cleanup.manifest_path(job)) == essential["old_manifest_sha"]
    assert all(path.exists() for path in essential["preserved"])


def test_changed_gate_between_unlinks_stops_before_next_file(essential, monkeypatch):
    job = essential["job"]
    monkeypatch.setattr(executor, "CHECK_INTERVAL", 1)
    # Register the explicit fixture-only interval before execution begins.
    path = executor.policy_path(job); path.unlink(); executor.prepare_policy(job)
    original_unlink = cleanup.secure_unlink
    def change_gate_after_unlink(mix, entry):
        original_unlink(mix, entry)
        authorization = job / "authorization.json"
        value = cleanup.read(authorization); value["user_request"] = "changed after preflight"
        put(authorization, value)
    monkeypatch.setattr(cleanup, "secure_unlink", change_gate_after_unlink)
    with pytest.raises(ValueError, match="Frozen cleanup evidence changed"):
        executor.execute(job)
    receipt = cleanup.read(qualified_cleanup.execution_path(job))
    assert len(receipt["deleted"]) == 1 and receipt["pending"] is None
    assert all(Path(row["path"]).exists() for row in essential["manifest"]["files"][1:])
    assert cleanup.sha(qualified_cleanup.manifest_path(job)) == essential["old_manifest_sha"]
