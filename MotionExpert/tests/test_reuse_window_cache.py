"""Only immutable, computation-identical accepted shards may be reused."""
from copy import deepcopy
import hashlib
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest

from MotionExpert.dataset.window_cache import ARRAY_SPECS, CONTRACT, atomic_json, digest, file_sha256, stamp
from MotionExpert.dataset.vision_cache import CONTRACT as VISION_CONTRACT, storage_digest
from MotionExpert.reuse_window_cache import check_archive_info, computation_ast, previous_visual_source, reuse_episode, visual_reuse_proof
from MotionExpert import reuse_window_cache as reuse


def test_reviewed_camera_source_pair_is_pinned_to_current_implementation():
    from pathlib import Path
    from MotionExpert.dataset.window_cache import source_code_binding
    assert reuse.CAMERA_REUSE_SOURCE_PAIR["after"] == source_code_binding()
    assert reuse.CAMERA_REUSE_SOURCE_PAIR["validator_sha256"] == file_sha256(
        Path(reuse.__file__).with_name("validate_camera_revision.py"))


def test_publication_change_does_not_hide_fk_changes():
    base = '''class SMPLHParameterArchive:
    def _check_acceptance(self):
        return "old"
    def decode(self, values):
        return values + 1
'''
    assert computation_ast(base) == computation_ast(base.replace('"old"', '"new"'))
    assert computation_ast(base) != computation_ast(base.replace('values + 1', 'values + 2'))
    with pytest.raises(ValueError, match="one archive acceptance"):
        computation_ast(base.replace('_check_acceptance', '_wrong_method'))


def fixture_shard(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    episode = old / "episodes/00000007"
    episode.mkdir(parents=True)
    record = {"episode_index": 7, "split": "train", "source_key": "egosuite:unchanged", "length": 91,
              "data_path": "episodes/00000007/motion.parquet"}
    vision = {"manifest_sha256": "old", "encoder_sha256": "unchanged"}
    arrays = {}
    for key, (dtype, tail) in ARRAY_SPECS.items():
        value = np.zeros((2, *tail), dtype=dtype)
        if key == "start_frames":
            value[:] = [0, 30]
        path = episode / f"{key}.npy"
        np.save(path, value, allow_pickle=False)
        size, modified = stamp(path)
        arrays[key] = {"path": str(path.relative_to(old)), "dtype": dtype,
                       "shape": list(value.shape), "size_bytes": size, "mtime_ns": modified,
                       "sha256": file_sha256(path)}
    receipt = {"contract": CONTRACT, "state": "complete", "episode_index": 7,
               "source_record_sha256": digest(record), "split": "train", "windows": 2,
               "arrays": arrays, "start_frames_sha256": digest([0, 30]),
               "binding_sha256": "old_binding", "template_metadata": {"vision_profile": vision}}
    path = episode / "receipt.json"
    atomic_json(path, receipt)
    descriptor = {"receipt_path": str(path.relative_to(old)), "receipt_sha256": file_sha256(path),
                  "windows": 2}
    previous_vision, current_vision = tmp_path / "old-vision", tmp_path / "new-vision"
    prefix = "episodes/00000007"
    (previous_vision / prefix).mkdir(parents=True)
    (current_vision / prefix).mkdir(parents=True)
    visual_identities = {}
    for name in ("features.npy", "join.npz"):
        path = previous_vision / prefix / name
        if name == "features.npy":
            np.save(path, np.zeros((record["length"], 5, 1024), dtype=np.float32))
        else:
            np.savez(path, frame_index=np.arange(record["length"]), valid=np.ones(record["length"], bool))
        os.link(path, current_vision / prefix / name)
        size, modified = stamp(path)
        visual_identities[name] = {"size_bytes": size, "mtime_ns": modified}
    visual_manifest = {"contract": VISION_CONTRACT, "state": "complete", "scope": "all",
                       "validity_equal_source": True, "encoder": {"weights": "same"}}
    atomic_json(previous_vision / "manifest.json", visual_manifest)
    visual_receipt = {"contract": VISION_CONTRACT, "state": "complete", "episode_index": 7,
                      "source_key": record["source_key"], "source_record_sha256": storage_digest(record),
                      "frames": record["length"], "validity_equal_source": True, "valid_count": record["length"],
                      "encoder": visual_manifest["encoder"], "output_identities": visual_identities}
    old_visual_receipt = previous_vision / prefix / "receipt.json"
    atomic_json(old_visual_receipt, visual_receipt)
    visual_receipt["quality_revision_reuse"] = {
        "parent_receipt_sha256": file_sha256(old_visual_receipt),
        "parent_manifest_sha256": file_sha256(previous_vision / "manifest.json"),
        "join_all_native_rows_checked": True, "join_evidence": "changed_motion_all_native_rows_compared",
        "payload_reuse": "same_device_inode_hardlink"}
    atomic_json(current_vision / prefix / "receipt.json", visual_receipt)
    previous_archive, current_archive = tmp_path / "old-archive", tmp_path / "new-archive"
    (previous_archive / prefix).mkdir(parents=True)
    (current_archive / prefix).mkdir(parents=True)
    (previous_archive / record["data_path"]).write_bytes(b"already accepted frozen motion")
    os.link(previous_archive / record["data_path"], current_archive / record["data_path"])
    vision_sources = {"previous_root": previous_vision, "current_root": current_vision,
                      "previous_manifest": visual_manifest, "current_manifest": visual_manifest,
                      "previous_manifest_sha256": file_sha256(previous_vision / "manifest.json"),
                      "previous_archive_root": previous_archive, "current_archive_root": current_archive}
    return old, new, descriptor, record, dict(vision, manifest_sha256="new"), vision_sources


def test_raw_bytes_share_inode_and_receipts_remain_independent(tmp_path):
    old, new, descriptor, record, vision, vision_sources = fixture_shard(tmp_path)
    before = (old / descriptor["receipt_path"]).read_bytes()
    reuse_episode(old, new, descriptor, record, {"new": "binding"}, vision, vision_sources=vision_sources)
    receipt = json.loads((new / descriptor["receipt_path"]).read_text())
    assert receipt["binding_sha256"] == digest({"new": "binding"})
    assert receipt["template_metadata"]["vision_profile"] == vision
    assert receipt["reuse"]["visual_proof"]["payload_identity"] == "same_device_inode_size_mtime"
    assert receipt["reuse"]["visual_proof"]["feature_payload_read"] is False
    assert (old / descriptor["receipt_path"]).read_bytes() == before
    for key in ARRAY_SPECS:
        source = old / f"episodes/00000007/{key}.npy"
        target = new / receipt["arrays"][key]["path"]
        assert os.path.samefile(source, target)
        assert file_sha256(target) == receipt["arrays"][key]["sha256"]
    with pytest.raises(FileExistsError):
        reuse_episode(old, new, descriptor, record, {}, vision, vision_sources=vision_sources)


@pytest.mark.parametrize("previous_revision", [None, {"format": "prior-quality", "policy": {"distance": 1.}}])
def test_successive_quality_revisions_reuse_identical_shards(tmp_path, previous_revision):
    previous = {"fps": 30, "model": {"model_sha256": "same-model"}}
    if previous_revision is not None:
        previous["quality_revision"] = deepcopy(previous_revision)
    current = deepcopy(previous)
    current["quality_revision"] = {"format": "next-quality", "parent_revision": deepcopy(previous_revision)}
    frozen_previous, frozen_current = deepcopy(previous), deepcopy(current)
    check_archive_info(previous, current)
    assert previous == frozen_previous and current == frozen_current
    old, new, descriptor, record, vision, sources = fixture_shard(tmp_path)
    reuse_episode(old, new, descriptor, record, {"revision": "new"}, vision, vision_sources=sources)
    receipt = json.loads((new / descriptor["receipt_path"]).read_text())
    assert receipt["reuse"]["unchanged_complete_record"] is True
    for name in ARRAY_SPECS:
        assert os.path.samefile(old / f"episodes/00000007/{name}.npy", new / receipt["arrays"][name]["path"])


@pytest.mark.parametrize("mutation", ["fps", "model", "missing_parent", "different_parent"])
def test_revision_compatibility_rejects_other_info_or_wrong_lineage(mutation):
    previous = {"fps": 30, "model": {"model_sha256": "same-model"},
                "quality_revision": {"format": "prior-quality", "policy": {"distance": 1.}}}
    current = deepcopy(previous)
    current["quality_revision"] = {"format": "next-quality", "parent_revision": deepcopy(previous["quality_revision"])}
    if mutation == "fps":
        current["fps"] = 15
    elif mutation == "model":
        current["model"]["model_sha256"] = "different-model"
    elif mutation == "missing_parent":
        current["quality_revision"].pop("parent_revision")
    else:
        current["quality_revision"]["parent_revision"]["policy"]["distance"] = .5
    with pytest.raises(ValueError, match="archive information|parent revision declaration"):
        check_archive_info(previous, current)


def test_reject_changed_encoder_and_changed_array_identity(tmp_path):
    old, new, descriptor, record, vision, vision_sources = fixture_shard(tmp_path)
    with pytest.raises(ValueError, match="encoder"):
        reuse_episode(old, new, descriptor, record, {}, dict(vision, encoder_sha256="changed"), vision_sources=vision_sources)
    path = old / "episodes/00000007/motion.npy"
    original = path.stat()
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="array identity"):
        reuse_episode(old, new, descriptor, record, {}, vision, vision_sources=vision_sources)
    assert not (new / "episodes/00000007").exists()


def test_reject_changed_source_record(tmp_path):
    old, new, descriptor, record, vision, vision_sources = fixture_shard(tmp_path)
    with pytest.raises(ValueError, match="unchanged source"):
        reuse_episode(old, new, descriptor, dict(record, changed_quality=True), {}, vision, vision_sources=vision_sources)


def test_previous_visual_manifest_must_match_the_accepted_window_cache(tmp_path):
    *_, sources = fixture_shard(tmp_path)
    manifest = {"vision_profile_root": str(sources["previous_root"]),
                "binding": {"vision_manifest_sha256": sources["previous_manifest_sha256"]}}
    root, saved, sha = previous_visual_source(manifest)
    assert root == sources["previous_root"]
    assert saved == sources["previous_manifest"]
    assert sha == sources["previous_manifest_sha256"]
    with pytest.raises(ValueError, match="visual manifest identity"):
        previous_visual_source(dict(manifest, binding={"vision_manifest_sha256": "wrong"}))


@pytest.mark.parametrize("payload", ["features.npy", "join.npz"])
def test_same_values_and_encoder_without_hardlink_proof_are_rejected(tmp_path, payload):
    _, _, _, record, _, sources = fixture_shard(tmp_path)
    target = sources["current_root"] / "episodes/00000007" / payload
    stat, contents = target.stat(), target.read_bytes()
    target.unlink()
    target.write_bytes(contents)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="not the same frozen hardlinked payload"):
        visual_reuse_proof(record, **sources)


@pytest.mark.parametrize("mutation", ["source", "encoder", "parent", "join_check", "identity"])
def test_rebound_visual_receipt_cannot_change_source_or_proof(tmp_path, mutation):
    _, _, _, record, _, sources = fixture_shard(tmp_path)
    path = sources["current_root"] / "episodes/00000007/receipt.json"
    receipt = json.loads(path.read_text())
    if mutation == "source":
        receipt["source_record_sha256"] = "different_source"
    elif mutation == "encoder":
        receipt["encoder"] = {"weights": "other"}
    elif mutation == "parent":
        receipt["quality_revision_reuse"]["parent_receipt_sha256"] = "other"
    elif mutation == "join_check":
        receipt["quality_revision_reuse"]["join_all_native_rows_checked"] = False
    else:
        receipt["output_identities"]["join.npz"]["size_bytes"] += 1
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="Visual|visual"):
        visual_reuse_proof(record, **sources)


def test_immutable_motion_and_visual_can_inherit_the_existing_join_acceptance(tmp_path):
    _, _, _, record, _, sources = fixture_shard(tmp_path)
    path = sources["current_root"] / "episodes/00000007/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["quality_revision_reuse"].update(join_all_native_rows_checked=False,
                                             join_evidence="inherited_immutable_motion_and_visual")
    atomic_json(path, receipt)
    proof = visual_reuse_proof(record, **sources)
    assert proof["join_evidence"] == "inherited_immutable_motion_and_visual"
    target = sources["current_archive_root"] / record["data_path"]
    original, contents = target.stat(), target.read_bytes()
    target.unlink()
    target.write_bytes(contents)
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="Unchanged source motion"):
        visual_reuse_proof(record, **sources)


def _camera_proof_case(tmp_path, monkeypatch):
    parent, root = tmp_path / "parent-v3", tmp_path / "camera-v4"
    parent.mkdir(); root.mkdir()
    record = {"episode_index": 7, "source_key": "hm:unchanged", "motion_sha256": "f" * 64}
    old_source = b'class SMPLHParameterArchive:\n    def _check_acceptance(self):\n        pass\n'
    before_codes = {"MotionExpert.dataset.smplh_archive": hashlib.sha256(old_source).hexdigest(),
                    "motion": "1" * 64}
    after_codes = dict(before_codes, motion="2" * 64, camera="3" * 64)
    previous_binding = {"archive_binding": {"info_sha256": "old-info"}, "archive_fingerprint": "v3",
                        "protocol": {"fps": 5}, "split_fingerprint": "original-split",
                        "model_sha256": "model", "image_profile_sha256": "image",
                        "representation_sha256": before_codes}
    current_binding = dict(previous_binding, archive_binding={"info_sha256": "new-info"},
                           archive_fingerprint="v4", representation_sha256=after_codes)
    validator = tmp_path / "validate_camera_revision.py"
    validator.write_text("# fixed independently reviewed validation fixture\n")
    validator_sha = file_sha256(validator)
    monkeypatch.setattr(reuse, "__file__", str(tmp_path / "reuse_window_cache.py"))
    monkeypatch.setattr(reuse, "CAMERA_REUSE_SOURCE_PAIR", {
        "before": before_codes, "after": after_codes, "validator_sha256": validator_sha})
    parent_top = parent / ".pipeline/acceptance-report.json"
    atomic_json(parent_top, {"format": "accepted-parent-v3", "passed": True})
    parent_state = {"state": "accepted", "independent_acceptance_pending": False,
                    "acceptance_report": ".pipeline/acceptance-report.json",
                    "acceptance_report_sha256": file_sha256(parent_top)}
    atomic_json(parent / ".pipeline/dataset-state.json", parent_state)
    quality_root = root / ".pipeline/camera-quality"
    quality_root.mkdir(parents=True)
    mask = quality_root / "episode_00000007.npz"
    np.savez(mask, native_valid=np.ones(91, dtype=bool))
    qc = {"episode_index": 7, "mask_path": mask.name, "mask_sha256": file_sha256(mask),
          "summary": {"motion_revision_required": False, "newly_rejected_frames": 0,
                      "newly_rejected_native_edges": 0, "invalid_native_frames": 0,
                      "representation_changed_due_native_continuity": False}}
    atomic_json(quality_root / "episode_00000007.json", qc)
    quality = {"format": "smplh-camera-quality-audit-v1", "state": "complete", "diagnostic": False,
               "archive_root": str(parent), "episodes": [qc]}
    quality_path = quality_root / "report.json"
    atomic_json(quality_path, quality)
    population = {"episodes": 1, "frames": 91}
    build = {"parent_archive_binding": previous_binding["archive_binding"],
             "parent_archive_fingerprint": "v3", "quality_report_sha256": file_sha256(quality_path),
             "parent_acceptance_sha256": file_sha256(parent_top)}
    revision = {"format": reuse.CAMERA_REVISION_FORMAT, "scope": "full", "build_binding": build,
                "build_binding_sha256": digest(build), "binding": current_binding["archive_binding"],
                "archive_fingerprint": "v4", "source_root": str(parent), "output_root": str(root),
                "population": population}
    revision_path = root / ".pipeline/camera-quality-revision.json"
    atomic_json(revision_path, revision)
    certificate = {"source_record_sha256": digest(record), "motion_sha256": record["motion_sha256"],
                   "native_reference_all_valid": True, "native_transition_all_valid": True,
                   "camera_audit_receipt_sha256": digest(qc), "camera_mask_sha256": file_sha256(mask)}
    independent = {"format": "independent-smplh-camera-quality-revision-delta-audit-v1", "scope": "full",
                   "passed": True, "failures": [], "binding": current_binding["archive_binding"],
                   "parent_archive_binding": previous_binding["archive_binding"],
                   "parent_archive_fingerprint": "v3", "revision_manifest_sha256": file_sha256(revision_path),
                   "quality_audit_sha256": file_sha256(quality_path), "validator_sha256": validator_sha,
                   "population": population, "unchanged_reference_certificates": {"7": certificate}}
    independent_path = root / ".pipeline/independent-camera-audit.json"
    atomic_json(independent_path, independent)
    inherited = root / ".pipeline/inherited-parents/v3"
    inherited.mkdir(parents=True)
    parent_state_copy, parent_top_copy = inherited / "dataset-state.json", inherited / "acceptance-report.json"
    parent_state_copy.write_bytes((parent / ".pipeline/dataset-state.json").read_bytes())
    parent_top_copy.write_bytes(parent_top.read_bytes())
    proofs = {name: {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}
              for name, path in (("revision_manifest", revision_path), ("quality_audit", quality_path),
                                 ("independent_revision_audit", independent_path),
                                 ("parent_publication", parent_state_copy), ("parent_acceptance", parent_top_copy))}
    top = {"format": reuse.CAMERA_ACCEPTANCE_FORMAT, "scope": "full", "passed": True, "failures": [],
           "binding": current_binding["archive_binding"], "population": population, "proofs": proofs}
    top_path = root / ".pipeline/acceptance-report.json"
    atomic_json(top_path, top)
    atomic_json(root / ".pipeline/dataset-state.json", dict(parent_state, acceptance_report_sha256=file_sha256(top_path)))
    archive = SimpleNamespace(root=root, binding=current_binding["archive_binding"], archive_fingerprint="v4")
    return SimpleNamespace(root=root, parent=parent, record=record, qc=qc, certificate=certificate,
                           archive=archive, before=previous_binding, after=current_binding, old_source=old_source)


def test_reviewed_camera_pair_requires_published_per_episode_evidence(tmp_path, monkeypatch):
    case = _camera_proof_case(tmp_path, monkeypatch)
    evidence = reuse.camera_reuse_evidence(case.archive, case.before, case.after, case.parent)
    reuse.check_computation(case.before, case.after, case.old_source,
                            camera_quality_reuse=True, camera_evidence=evidence)
    assert evidence.episode(case.record) == case.certificate
    assert evidence.episode(dict(case.record, episode_index=8)) is None
    with pytest.raises(ValueError, match="published independent archive evidence"):
        reuse.check_computation(case.before, case.after, case.old_source, camera_quality_reuse=True)
    with pytest.raises(ValueError, match="published independent archive evidence"):
        reuse.check_computation(case.before, case.after, case.old_source,
                                camera_quality_reuse=True, camera_evidence={"unchanged": True})


def test_camera_source_exception_is_never_enabled_by_default(tmp_path, monkeypatch):
    from MotionExpert.dataset import smplh_archive

    case = _camera_proof_case(tmp_path, monkeypatch)
    source = tmp_path / "smplh_archive.py"
    source.write_bytes(case.old_source)
    monkeypatch.setattr(smplh_archive, "__file__", str(source))
    evidence = reuse.camera_reuse_evidence(case.archive, case.before, case.after, case.parent)
    with pytest.raises(ValueError, match="representation/indexing computation changed"):
        reuse.check_computation(case.before, case.after, case.old_source, camera_evidence=evidence)


@pytest.mark.parametrize("mutation", ["before", "after", "new_module", "unreviewed", "validator_unreviewed"])
def test_camera_reuse_rejects_any_unreviewed_complete_source_pair(tmp_path, monkeypatch, mutation):
    case = _camera_proof_case(tmp_path, monkeypatch)
    evidence = reuse.camera_reuse_evidence(case.archive, case.before, case.after, case.parent)
    before, after = deepcopy(case.before), deepcopy(case.after)
    if mutation == "before":
        before["representation_sha256"]["motion"] = "changed"
    elif mutation == "after":
        after["representation_sha256"]["motion"] = "changed"
    elif mutation == "new_module":
        after["representation_sha256"]["extra_computation"] = "changed"
    else:
        monkeypatch.setitem(reuse.CAMERA_REUSE_SOURCE_PAIR,
                            "after" if mutation == "unreviewed" else "validator_sha256", None)
    with pytest.raises(ValueError, match="exact reviewed complete source-code hash pair"):
        reuse.check_computation(before, after, case.old_source,
                                camera_quality_reuse=True, camera_evidence=evidence)


@pytest.mark.parametrize("mutation", ["record", "reference", "transition", "point_count", "edge_count", "native_gap", "continuity"])
def test_same_parquet_is_not_enough_when_native_reference_semantics_changed(tmp_path, monkeypatch, mutation):
    case = _camera_proof_case(tmp_path, monkeypatch)
    evidence = reuse.camera_reuse_evidence(case.archive, case.before, case.after, case.parent)
    if mutation == "record":
        changed = dict(case.record, split="different")
    else:
        changed = case.record
        if mutation in ("reference", "transition"):
            evidence.certificates["7"][f"native_{mutation}_all_valid"] = False
        else:
            summary = evidence.quality_episodes[7]["summary"]
            key = {"point_count": "newly_rejected_frames", "edge_count": "newly_rejected_native_edges",
                   "native_gap": "invalid_native_frames", "continuity": "representation_changed_due_native_continuity"}[mutation]
            summary[key] = True if mutation == "continuity" else 1
            evidence.certificates["7"]["camera_audit_receipt_sha256"] = digest(evidence.quality_episodes[7])
    with pytest.raises(ValueError, match="identical native reference computation"):
        evidence.episode(changed)


@pytest.mark.parametrize("artifact", ["camera-quality/report.json", "camera-quality/episode_00000007.npz", "independent-camera-audit.json"])
def test_camera_reuse_proofs_cannot_change_after_loading(tmp_path, monkeypatch, artifact):
    case = _camera_proof_case(tmp_path, monkeypatch)
    evidence = reuse.camera_reuse_evidence(case.archive, case.before, case.after, case.parent)
    path = case.root / ".pipeline" / artifact
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="evidence changed"):
        evidence.assert_current()


def test_camera_publication_requires_exact_schema_and_parent_declaration():
    old = {"fps": 30, "quality_revision": {"format": "smplh-fit-quality-revision-v1", "policy": {"exact": True}}}
    new = deepcopy(old)
    new["quality_revision"] = {"format": reuse.CAMERA_REVISION_FORMAT,
        "parent_revision": deepcopy(old["quality_revision"]), "parameter_values_preserved": True,
        "image_join_values_preserved": True, "world_observations_preserved": True,
        "ground_values_preserved": True}
    reuse.check_archive_info(old, new, camera_quality_reuse=True)
    new["quality_revision"]["ground_values_preserved"] = False
    with pytest.raises(ValueError, match="preserved values"):
        reuse.check_archive_info(old, new, camera_quality_reuse=True)
