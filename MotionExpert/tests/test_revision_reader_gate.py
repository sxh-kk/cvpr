"""New revisions require owned bound acceptance; legacy publication stays valid."""
import json

import pytest

from MotionExpert.dataset import revision_acceptance
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, file_sha256


def reader(tmp_path, monkeypatch):
    archive = SMPLHParameterArchive.__new__(SMPLHParameterArchive)
    archive.root, archive.frames, archive._records = tmp_path, 0, []
    archive.info = {"quality_revision": {"format": revision_acceptance.REVISION_FORMAT}}
    archive.allow_unpublished = False
    path = tmp_path / ".pipeline/acceptance-report.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"format": revision_acceptance.ACCEPTANCE_FORMAT}))
    archive.state = {"episodes": 0, "frames": 0, "acceptance_report": str(path.relative_to(tmp_path)),
                     "acceptance_report_sha256": file_sha256(path)}
    checked = []
    monkeypatch.setattr(revision_acceptance, "check_revision_acceptance",
                        lambda obj, report, location: checked.append((obj, report, location)))
    return archive, path, checked


def test_revision_dispatches_its_owned_exact_acceptance(tmp_path, monkeypatch):
    archive, path, checked = reader(tmp_path, monkeypatch)
    archive._check_acceptance()
    assert len(checked) == 1
    assert checked[0][0] is archive
    assert checked[0][2] == path


@pytest.mark.parametrize("change", ["missing_path", "missing_sha", "changed_bytes", "missing_file", "external_path", "legacy_format", "empty_revision"])
def test_revision_cannot_inherit_legacy_missing_external_report_exemption(tmp_path, monkeypatch, change):
    archive, path, checked = reader(tmp_path, monkeypatch)
    if change == "missing_path":
        archive.state.pop("acceptance_report")
    elif change == "missing_sha":
        archive.state.pop("acceptance_report_sha256")
    elif change == "changed_bytes":
        path.write_text(path.read_text() + "\n")
    elif change == "missing_file":
        path.unlink()
    elif change == "external_path":
        archive.state["acceptance_report"] = str(path)
    elif change == "legacy_format":
        path.write_text(json.dumps({"format": "qualified-standalone-smplh-parameter-full-audit-v1"}))
        archive.state["acceptance_report_sha256"] = file_sha256(path)
    elif change == "empty_revision":
        archive.info["quality_revision"] = {}
    with pytest.raises((ValueError, FileNotFoundError)):
        archive._check_acceptance()
    assert not checked


def test_unpublished_candidate_needs_no_acceptance_until_publication(tmp_path, monkeypatch):
    archive, path, checked = reader(tmp_path, monkeypatch)
    archive.allow_unpublished = True
    path.unlink()
    archive.state.pop("acceptance_report")
    archive._check_acceptance()
    assert not checked


def test_legacy_moved_archive_keeps_optional_external_report_semantics(tmp_path, monkeypatch):
    archive, path, checked = reader(tmp_path, monkeypatch)
    archive.info = {}
    archive.state["acceptance_report"] = str(tmp_path.parent / "missing-old-external-proof.json")
    archive._check_acceptance()
    assert not checked
