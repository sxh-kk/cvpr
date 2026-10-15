"""Quality revision masks, immutable publication, joins, and resumable writes."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from MotionExpert import revise_hm_archive as revision
from MotionExpert.dataset.revision_acceptance import ACCEPTANCE_FORMAT, check_revision_acceptance
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, assert_smplh_archive_ready, file_sha256
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.dataset.window_cache import atomic_json, digest
from MotionExpert.tests.fixtures import owned_archive, parameter_arrays, rewrite_episode
from MotionExpert.tests.test_vision_profile_cache import write_cache


def write_quality(root, output, *, rejected=range(0, 31), selected=(1,)):
    output.mkdir()
    receipts = []
    policy = {"max_head_camera_distance_m": 1., "warning_head_camera_distance_m": .5}
    for index in selected:
        record_path = root / f".pipeline/episodes/{index:08d}.json"
        record = json.loads(record_path.read_text())
        count = record["length"]
        good = np.ones(count, dtype=bool)
        good[list(rejected)] = False
        native = np.ones(count, dtype=bool)
        arrays = {"quality_valid": good, "native_valid": native, "fit_finite": native.copy(),
                  "transition_valid": np.r_[False, good[1:] & good[:-1]],
                  "reasons": np.where(good, 0, 8).astype(np.uint16)}
        path = output / f"episode_{index:08d}.npz"
        np.savez(path, **arrays)
        receipt = {"format": revision.AUDIT_FORMAT, "episode_index": index,
                   "source_key": record["source_key"], "split": record["split"],
                   "record_sha256": hashlib.sha256(record_path.read_bytes()).hexdigest(),
                   "motion_sha256": record["motion_sha256"], "mask_path": path.name,
                   "mask_sha256": file_sha256(path), "native_valid_sha256": hashlib.sha256(native.tobytes()).hexdigest(),
                   "policy": policy, "summary": {"newly_rejected_frames": int((~good).sum())}}
        atomic_json(output / f"episode_{index:08d}.json", receipt)
        receipts.append(receipt)
    report = {"format": revision.AUDIT_FORMAT, "state": "complete", "archive_root": str(root),
              "production_modified": False, "episodes_completed": len(receipts), "episodes": receipts,
              "policy": policy, "scope": "all_archive_HM", "diagnostic": False}
    atomic_json(output / "report.json", report)
    return output / "report.json"


@pytest.fixture
def revision_source(owned_archive, tmp_path):
    # Original lower foot cluster is physically wrong and must disappear when
    # the corresponding fit frames are rejected. The native values stay stored.
    def poison_feet(arrays):
        arrays["source.feet_world"][:31, [2, 5]] = -4
    rewrite_episode(owned_archive, 1, motion_mutation=poison_feet)
    vision = write_cache(owned_archive, tmp_path / "original-vision")
    quality = write_quality(owned_archive, tmp_path / "quality")
    return owned_archive, vision, quality


def build_small(source, output, **kwargs):
    root, vision, quality = source
    return revision.build_revision(root, quality, output, source_vision_root=vision, workers=1, **kwargs)


def test_mask_gating_preserves_values_and_provenance_invariants():
    original = parameter_arrays(5)
    original["source.observed.body_rotation"][:, 13] = False
    original["estimated.completed.body_rotation"][:, 13] = True
    original["source.interpolated.body_position"][:, 10] = True
    original["source.feet_interpolated"][:, 0] = True
    original["quality.observation_residual_valid"][:] = True
    good = np.array([True, False, False, True, True])
    result = revision.apply_quality_masks(original, good)
    for component in ("position", "rotation"):
        assert not result[f"estimated.valid.body_{component}"][1:3].any()
        assert not result[f"source.observed.body_{component}"][1:3].any()
        assert not result[f"source.interpolated.body_{component}"][1:3].any()
        assert not result[f"quality.{component}_confidence"][1:3].any()
        np.testing.assert_array_equal(result[f"estimated.completed.body_{component}"],
                                      result[f"estimated.valid.body_{component}"] & ~result[f"source.observed.body_{component}"])
    assert result["estimated.completed.body_rotation"][0, 13]
    assert not result["quality.observation_residual_valid"][1:3].any()
    assert not result["source.feet_valid"][1:3].any()
    assert not result["source.feet_interpolated"][1:3].any()
    for side in ("left", "right"):
        assert not result[f"observation.valid.hand_{side}_position"][1:3].any()
        np.testing.assert_array_equal(result[f"observation.state.hand_{side}_world"], original[f"observation.state.hand_{side}_world"])
    for name in ("smplh.trans", "smplh.pose_body", "smplh.root_orient", "smplh.betas",
                 "source.feet_world", "observation.state.camera_world", "observation.valid.dino"):
        np.testing.assert_array_equal(result[name], original[name])
    assert original["estimated.valid.body_position"][1:3].all()


def test_ground_rejects_poisoned_cluster_and_preserves_prefix_fallback():
    arrays = parameter_arrays(91)
    arrays["source.feet_world"][:, [0, 3]] = 0
    arrays["source.feet_world"][:31, [2, 5]] = -4
    good = np.arange(91) >= 31
    revised = revision.apply_quality_masks(arrays, good)
    meta = {"coordinate_verified": True, "ground": {"mode": "estimate", "height_m": -4.01,
            "valid": True, "window_fallback": {"mode": "mean_valid_feet", "offset_m": .02}}}
    ground = revision.estimate_revision_ground(revised, meta, root_rest=np.zeros(3), root_shape_basis=np.zeros((3, 16)))
    assert ground["valid"] and ground["height_m"] == pytest.approx(.51, abs=1e-7)
    assert all(c["median_height_m"] > 0 for c in ground["clusters"])
    assert meta["ground"]["height_m"] == -4.01
    revised["source.feet_valid"][:] = False
    failed = revision.estimate_revision_ground(revised, meta, root_rest=np.zeros(3), root_shape_basis=np.zeros((3, 16)))
    assert not failed["valid"] and failed["height_m"] == 0
    assert failed["window_fallback"] == {"mode": "mean_valid_feet", "offset_m": .02}


def test_revision_is_candidate_original_records_and_dino_are_immutable(revision_source, tmp_path):
    root, vision, quality = revision_source
    before_record = (root / ".pipeline/episodes/00000001.json").read_bytes()
    before_motion_sha = file_sha256(root / "episodes/00000001/motion.parquet")
    original_profile = (vision / "manifest.json").read_bytes()
    output = tmp_path / "new-parameters"
    result = build_small(revision_source, output)
    assert result["state"] == "candidate" and result["changed_episode_indices"] == [1]
    assert result["parameter_values_preserved"] and result["image_join_values_preserved"]
    assert result["parent_model_sha256"] == result["model_sha256"]
    assert result["parent_split_fingerprint"] == result["split_fingerprint"]
    assert (root / ".pipeline/episodes/00000001.json").read_bytes() == before_record
    assert file_sha256(root / "episodes/00000001/motion.parquet") == before_motion_sha
    assert (vision / "manifest.json").read_bytes() == original_profile
    assert not os.path.samefile(root / "episodes/00000001/motion.parquet", output / "episodes/00000001/motion.parquet")
    assert os.path.samefile(root / "episodes/00000000/motion.parquet", output / "episodes/00000000/motion.parquet")
    for index in range(4):
        for name in ("features.npy", "join.npz"):
            assert os.path.samefile(vision / f"episodes/{index:08d}/{name}",
                                    output / f"vision_profiles/psi0_256x480_v1_30hz/episodes/{index:08d}/{name}")
    with pytest.raises(ValueError, match="not published"):
        assert_smplh_archive_ready(output)
    archive = SMPLHParameterArchive(output, allow_unpublished=True, vision_profile_root=result["vision_profile_root"])
    source = SMPLHParameterArchive(root)
    assert archive.records()[0] == source.records()[0]
    assert archive.archive_fingerprint != source.archive_fingerprint
    old = source._motion(source.records()[1], np.arange(91))
    new = archive._motion(archive.records()[1], np.arange(91))
    for name in old:
        if name.startswith("smplh.") or name.startswith("observation.state.") or name in revision.JOIN_COLUMNS:
            np.testing.assert_array_equal(new[name], old[name])
    assert not new["estimated.valid.body_position"][:31].any()
    assert new["estimated.valid.body_position"][31:].all()
    assert archive.records()[1]["metadata"]["ground"]["height_m"] == pytest.approx(.51, abs=1e-7)
    data = SMPLHMotionDataset(archive_root=output, allow_unpublished=True,
                             do_normalization=False, cond_img_feat=False,
                             vision_profile_root=result["vision_profile_root"])
    assert len(data) > 0
    assert data[2]["misc"]["motion"].shape == (11, 245)


def test_resume_reuses_successful_episodes_and_rejects_policy_drift(revision_source, tmp_path, monkeypatch):
    output = tmp_path / "new-parameters"
    first = build_small(revision_source, output)
    stamp = (output / "episodes/00000001/motion.parquet").stat().st_mtime_ns
    monkeypatch.setattr(revision, "apply_quality_masks", lambda *args: pytest.fail("Completed episode was recomputed"))
    second = build_small(revision_source, output, resume=True)
    assert second["archive_fingerprint"] == first["archive_fingerprint"]
    assert (output / "episodes/00000001/motion.parquet").stat().st_mtime_ns == stamp
    report = json.loads(revision_source[2].read_text())
    report["policy"]["max_head_camera_distance_m"] = .8
    # Replace the independent audit file instead of editing hard-linked bytes.
    atomic_json(revision_source[2], report)
    with pytest.raises(ValueError, match="policy differs"):
        build_small(revision_source, output, resume=True)


@pytest.mark.parametrize("relative", ["../escape", "/tmp/escape"])
def test_owned_artifacts_reject_escape(tmp_path, relative):
    with pytest.raises(ValueError, match="owned relative"):
        revision._owned(tmp_path, relative, exists=False)


def test_output_cannot_overlap_source_or_audit(revision_source):
    root, _, quality = revision_source
    for output in (root, root / "new", root.parent, quality.parent / "revision"):
        with pytest.raises(ValueError, match="separate|outside"):
            build_small(revision_source, output)


def test_visual_join_tampering_fails_before_rebinding(revision_source, tmp_path):
    root, vision, _ = revision_source
    path = vision / "episodes/00000001/join.npz"
    with np.load(path) as data:
        join = {key: data[key].copy() for key in data.files}
    join["observation.source_timestamp"][0] += 1
    np.savez(path, **join)
    receipt_path = vision / "episodes/00000001/receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["output_identities"]["join.npz"].update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="visual/source row join differs"):
        build_small(revision_source, tmp_path / "new")


def test_native_invalid_already_unavailable_does_not_rewrite_motion(owned_archive, tmp_path):
    def invalidate(arrays):
        result = revision.apply_quality_masks(arrays, np.arange(91) >= 31)
        arrays.update(result)
    rewrite_episode(owned_archive, 1, motion_mutation=invalidate)
    vision = write_cache(owned_archive, tmp_path / "vision")
    quality = write_quality(owned_archive, tmp_path / "quality")
    result = revision.build_revision(owned_archive, quality, tmp_path / "new", source_vision_root=vision, workers=1)
    assert result["motion_rewritten_episodes"] == []
    assert os.path.samefile(owned_archive / "episodes/00000001/motion.parquet",
                            tmp_path / "new/episodes/00000001/motion.parquet")


def test_quality_mask_identity_cannot_upgrade_source(owned_archive, tmp_path):
    quality = write_quality(owned_archive, tmp_path / "quality")
    receipt = json.loads(quality.read_text())["episodes"][0]
    index = receipt["episode_index"]
    raw = (owned_archive / f".pipeline/episodes/{index:08d}.json").read_bytes()
    record = json.loads(raw)
    receipt["native_valid_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="native-valid upstream SHA"):
        revision._load_quality(quality.parent, receipt, record, raw)


def test_acceptance_rejects_candidate_and_unbound_proofs(revision_source, tmp_path):
    result = build_small(revision_source, tmp_path / "new")
    archive = SMPLHParameterArchive(tmp_path / "new", allow_unpublished=True)
    with pytest.raises(ValueError, match="independent acceptance failed"):
        check_revision_acceptance(archive, result, tmp_path / "new/no.json")
    report = {"format": ACCEPTANCE_FORMAT, "scope": "full", "passed": True, "failures": [],
              "population": result["population"], "binding": result["binding"]}
    for name in ("parent_acceptance_inherited", "parameter_values_preserved", "camera_values_preserved",
                 "image_join_values_preserved", "quality_masks_accepted", "ground_recomputed_accepted", "latest_motionexpert_accepted"):
        report[name] = True
    with pytest.raises(ValueError, match="proof is missing"):
        check_revision_acceptance(archive, report, tmp_path / "new/no.json")


def test_worker_control_requires_positive_bounded_integer(tmp_path):
    control = tmp_path / "workers.json"
    atomic_json(control, {"active_workers": 2})
    assert revision._active_workers(control, 8) == 2
    atomic_json(control, {"active_workers": 8})
    assert revision._active_workers(control, 8) == 8
    atomic_json(control, {"active_workers": True})
    with pytest.raises(ValueError, match="active_workers"):
        revision._active_workers(control, 8)


def test_restart_after_incomplete_episode_does_not_redo_success(revision_source, tmp_path, monkeypatch):
    output = tmp_path / "new"
    original_function = revision._revise_episode
    def fail_second(task):
        if task["record"]["episode_index"] == 1:
            revision._prepare_episode_output(task)
            path = output / task["record"]["data_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"incomplete write")
            raise RuntimeError("simulated interruption")
        return original_function(task)
    monkeypatch.setattr(revision, "_revise_episode", fail_second)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_small(revision_source, output)
    saved = (output / "episodes/00000000/receipt.json").stat().st_mtime_ns
    monkeypatch.setattr(revision, "_revise_episode", original_function)
    result = build_small(revision_source, output, resume=True)
    assert result["state"] == "candidate"
    assert (output / "episodes/00000000/receipt.json").stat().st_mtime_ns == saved


def test_production_receipt_extra_identity_fields_are_allowed(revision_source, tmp_path):
    root, _, _ = revision_source
    for index in range(4):
        path = root / f"episodes/{index:08d}/receipt.json"
        receipt = json.loads(path.read_text())
        for identity in receipt["output_identities"].values():
            identity.update(device=123, inode=123, ctime_ns=123)
        atomic_json(path, receipt)
    assert build_small(revision_source, tmp_path / "new")["state"] == "candidate"


def complete_acceptance_proof(revision_source, tmp_path):
    output = tmp_path / "new"
    result = build_small(revision_source, output)
    archive = SMPLHParameterArchive(output, allow_unpublished=True)
    manifest_path = output / ".pipeline/hm-quality-revision.json"
    quality_path = output / ".pipeline/quality/report.json"
    independent_path = output / ".pipeline/independent-quality-audit.json"
    flags = ("parent_acceptance_inherited", "parameter_values_preserved", "camera_values_preserved",
             "image_join_values_preserved", "quality_masks_accepted", "ground_recomputed_accepted", "latest_motionexpert_accepted")
    independent = {"format": "independent-smplh-hm-quality-revision-delta-audit-v1",
                   "passed": True, "scope": "full", "failures": [], "binding": result["binding"],
                   "population": result["population"], "quality_audit_sha256": file_sha256(quality_path),
                   "revision_manifest_sha256": file_sha256(manifest_path), **{name: True for name in flags}}
    atomic_json(independent_path, independent)
    proof_paths = {"revision_manifest": manifest_path, "quality_audit": quality_path,
                   "independent_revision_audit": independent_path,
                   "parent_publication": output / ".pipeline/parent-publication/dataset-state.json"}
    report = {"format": ACCEPTANCE_FORMAT, "scope": "full", "passed": True, "failures": [],
              "population": result["population"], "binding": result["binding"],
              "proofs": {name: {"path": str(path.relative_to(output)), "sha256": file_sha256(path)}
                         for name, path in proof_paths.items()}}
    for name in flags:
        report[name] = True
    return output, archive, report, independent_path, independent, flags


def test_independent_acceptance_complete_proof_chain_and_tamper(revision_source, tmp_path):
    output, archive, report, independent_path, independent, _ = complete_acceptance_proof(revision_source, tmp_path)
    check_revision_acceptance(archive, report, output / "acceptance.json")
    independent["quality_audit_sha256"] = "0" * 64
    atomic_json(independent_path, independent)
    report["proofs"]["independent_revision_audit"]["sha256"] = file_sha256(independent_path)
    with pytest.raises(ValueError, match="independent evidence differs"):
        check_revision_acceptance(archive, report, output / "acceptance.json")


def test_independent_proof_requires_its_format_and_every_acceptance_flag(revision_source, tmp_path):
    output, archive, report, path, original, flags = complete_acceptance_proof(revision_source, tmp_path)
    # Rebind the digest deliberately: semantic evidence checks must reject a
    # different report format or any failed inner decision independently of SHA.
    for key, value in [("format", "unrelated-success-report")] + [(name, False) for name in flags]:
        changed = dict(original, **{key: value})
        atomic_json(path, changed)
        report["proofs"]["independent_revision_audit"]["sha256"] = file_sha256(path)
        with pytest.raises(ValueError, match="independent evidence differs"):
            check_revision_acceptance(archive, report, output / "acceptance.json")


def test_spawn_workers_build_without_reconstructing_source_archive(revision_source, tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    root, vision, quality = revision_source
    result = revision.build_revision(root, quality, tmp_path / "new", source_vision_root=vision, workers=2)
    assert result["state"] == "candidate" and result["changed_episode_indices"] == [1]


def test_unchanged_episode_joins_are_inherited_without_row_decode(owned_archive, tmp_path, monkeypatch):
    vision = write_cache(owned_archive, tmp_path / "vision")
    quality = write_quality(owned_archive, tmp_path / "quality", rejected=[])
    original = pq.read_table
    def no_episode_decode(path, *args, **kwargs):
        if Path(path).name == "motion.parquet":
            pytest.fail("Unchanged accepted episode was decoded again")
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pq, "read_table", no_episode_decode)
    result = revision.build_revision(owned_archive, quality, tmp_path / "new", source_vision_root=vision, workers=1)
    assert result["changed_episode_indices"] == []
    for index in range(4):
        receipt = json.loads((Path(result["vision_profile_root"]) / f"episodes/{index:08d}/receipt.json").read_text())
        assert receipt["quality_revision_reuse"]["join_evidence"] == "inherited_immutable_motion_and_visual"
        assert receipt["quality_revision_reuse"]["join_all_native_rows_checked"] is False


def test_revision_preserves_ontology_inventory_and_standalone_provenance(revision_source, tmp_path):
    root, vision, _ = revision_source
    original_visual_manifest = json.loads((vision / "manifest.json").read_text())
    original_visual_manifest["output_root"] = str(vision)
    atomic_json(vision / "manifest.json", original_visual_manifest)
    ontology = {"schema_version": revision.PARAMETER_SCHEMA_VERSION, "body_joints": ["pelvis", "head"],
                "position_unit": "meter", "target_world_axes": "X forward, Y left, Z up"}
    summary = {"schema_version": revision.PARAMETER_SCHEMA_VERSION, "episodes": 4, "frames": 364,
               "training_eligible_episodes": 3, "updated_at": "original_population_inventory"}
    original_config = {"schema_version": revision.PARAMETER_SCHEMA_VERSION, "standalone": True,
                       "job": "/historical/conversion/job"}
    atomic_json(root / "meta/ontology.json", ontology)
    atomic_json(root / "meta/summary.json", summary)
    atomic_json(root / ".pipeline/config.json", original_config)
    output = tmp_path / "new"
    result = build_small(revision_source, output)
    for relative, value in (("meta/ontology.json", ontology), ("meta/summary.json", summary)):
        assert os.path.samefile(root / relative, output / relative)
        assert json.loads((output / relative).read_text()) == value
        assert result["build_binding"]["preserved_metadata_sha256"][relative] == file_sha256(root / relative)
    assert os.path.samefile(root / ".pipeline/config.json", output / ".pipeline/parent-publication/config.json")
    assert json.loads((root / ".pipeline/config.json").read_text()) == original_config
    config = json.loads((output / ".pipeline/config.json").read_text())
    assert config["standalone"] and config["requires_old_dense_archive"] is False
    assert config["requires_parent_archive_at_training"] is False
    assert config["parent_configuration"] == ".pipeline/parent-publication/config.json"
    visual = json.loads((Path(result["vision_profile_root"]) / "manifest.json").read_text())
    assert visual["output_root"] == result["vision_profile_root"]
    assert visual["quality_revision_reuse"]["parent_profile_root"] == str(vision)
    assert visual["encoder"] == original_visual_manifest["encoder"]
    assert json.loads((vision / "manifest.json").read_text()) == original_visual_manifest


def test_inconsistent_original_foot_mask_cannot_be_upgraded():
    arrays = parameter_arrays(5)
    arrays["source.feet_valid"][0, 0] = False
    with pytest.raises(ValueError, match="foot availability differs"):
        revision.apply_quality_masks(arrays, np.ones(5, dtype=bool))
