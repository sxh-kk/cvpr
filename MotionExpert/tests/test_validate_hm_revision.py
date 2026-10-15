"""Independent delta acceptance: reject forged values without repeating old QA."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from MotionExpert import config as config_module
from MotionExpert import revise_hm_archive as builder
from MotionExpert import validate_hm_revision as validator
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
from MotionExpert.tests.fixtures import owned_archive
from MotionExpert.tests.test_revise_hm_archive import revision_source, build_small


@pytest.fixture
def candidate(revision_source, tmp_path, monkeypatch):
    root, vision, _ = revision_source
    archive = SMPLHParameterArchive(root)
    monkeypatch.setattr(config_module, "SMPLH_MODEL_SHA256", archive.model_sha256)
    records = archive.records()
    sources, splits = {}, {name: {"episodes": 0, "frames": 0} for name in ("train", "val", "test")}
    for record in records:
        count = sources.setdefault(record["source"], {"episodes": 0, "frames": 0})
        count["episodes"] += 1; count["frames"] += record["length"]
        splits[record["split"]]["episodes"] += 1; splits[record["split"]]["frames"] += record["length"]
    (root / "meta/summary.json").write_text(json.dumps({"episodes": len(records), "frames": archive.frames,
                                                      "sources": sources, "splits": splits}))
    (root / "meta/ontology.json").write_text(json.dumps({"joints": "SMPL-H52/body22"}))
    (root / ".pipeline/config.json").write_text(json.dumps({"inherited_source_configuration": True}))
    output = tmp_path / "candidate"
    build_small(revision_source, output)
    return root, vision, output


def test_delta_audit_and_publish_preserve_parent_and_do_not_repeat_unchanged_payloads(candidate, monkeypatch):
    parent, vision, root = candidate
    before = (parent / ".pipeline/dataset-state.json").read_bytes()
    original_parquet, original_table, original_numpy = pq.ParquetFile, pq.read_table, np.load
    def reject_unchanged(path):
        path = str(path)
        if any(f"episodes/{index:08d}/motion.parquet" in path for index in (0, 2, 3)):
            raise AssertionError("Unchanged motion numeric acceptance was repeated")
    def parquet(path, *args, **kwargs):
        reject_unchanged(path)
        return original_parquet(path, *args, **kwargs)
    def table(path, *args, **kwargs):
        reject_unchanged(path)
        return original_table(path, *args, **kwargs)
    def numpy(path, *args, **kwargs):
        path_text = str(path)
        if path_text.endswith("features.npy") or any(f"episodes/{index:08d}/join.npz" in path_text for index in (0, 2, 3)):
            raise AssertionError("Accepted DINO/unchanged visual join payload was reopened")
        return original_numpy(path, *args, **kwargs)
    monkeypatch.setattr(pq, "ParquetFile", parquet)
    monkeypatch.setattr(pq, "read_table", table)
    monkeypatch.setattr(np, "load", numpy)
    report = validator.audit_revision(parent, root, parent_vision_root=vision)
    assert report["passed"] and report["changed_episode_indices"] == [1]
    assert report["inherited_unchanged_motion_episodes"] == 3
    assert report["previous_physical_acceptance_repeated"] is False
    assert report["dino_payload_rehashed"] is False
    assert report["changed_motion_comparisons"][0]["ground_recomputed"] is True
    assert report["latest_motionexpert"]["windows"]
    # Dry audit itself cannot publish a candidate.
    assert json.loads((root / ".pipeline/dataset-state.json").read_text())["state"] == "candidate"
    proof = root / ".pipeline/independent-revision-audit.json"
    validator._write_new(proof, report)
    top = validator.publish_revision(parent, root, proof, parent_vision_root=vision)
    assert top["passed"] and top["parameter_values_preserved"]
    state = json.loads((root / ".pipeline/dataset-state.json").read_text())
    assert state["state"] == "accepted" and state["independent_acceptance_pending"] is False
    assert state["formal_training_ready"] is False
    assert (parent / ".pipeline/dataset-state.json").read_bytes() == before
    with pytest.raises(ValueError, match="published once"):
        validator.publish_revision(parent, root, proof, parent_vision_root=vision)


@pytest.mark.parametrize("poison", ["parameter", "hand_mask", "ground", "noop"])
def test_independent_audit_rejects_consistently_committed_builder_mistakes(revision_source, tmp_path, monkeypatch, poison):
    parent, vision, _ = revision_source
    monkeypatch.setattr(config_module, "SMPLH_MODEL_SHA256", SMPLHParameterArchive(parent).model_sha256)
    if poison == "ground":
        original = builder.estimate_revision_ground
        def bad_ground(*args, **kwargs):
            result = original(*args, **kwargs)
            result["height_m"] += .1
            return result
        monkeypatch.setattr(builder, "estimate_revision_ground", bad_ground)
    else:
        original = builder.apply_quality_masks
        def bad_masks(arrays, good):
            if poison == "noop":
                return {key: value.copy() for key, value in arrays.items()}
            result = original(arrays, good)
            if poison == "parameter":
                result["smplh.trans"][:, 0] += .1
            else:
                result["observation.valid.hand_left_position"] = arrays["observation.valid.hand_left_position"].copy()
            return result
        monkeypatch.setattr(builder, "apply_quality_masks", bad_masks)
    root = tmp_path / "candidate"
    build_small(revision_source, root)
    with pytest.raises(ValueError, match="value mismatch|gated-foot ground"):
        validator.audit_revision(parent, root, parent_vision_root=vision)
    assert json.loads((root / ".pipeline/dataset-state.json").read_text())["state"] == "candidate"


def test_publication_rejects_stale_audit_and_external_proof(candidate, tmp_path):
    parent, vision, root = candidate
    report = validator.audit_revision(parent, root, parent_vision_root=vision)
    external = tmp_path / "outside-audit.json"
    external.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="owned by the candidate"):
        validator.publish_revision(parent, root, external, parent_vision_root=vision)
    report["artifact_snapshot_sha256"] = "stale"
    proof = root / ".pipeline/independent-revision-audit.json"
    validator._write_new(proof, report)
    with pytest.raises(ValueError, match="stale"):
        validator.publish_revision(parent, root, proof, parent_vision_root=vision)
    assert json.loads((root / ".pipeline/dataset-state.json").read_text())["state"] == "candidate"
