"""Independent final-FK quality gates and source-specific revision contracts."""
from copy import deepcopy
import numpy as np
import pytest

from MotionExpert import validate_fit_revision as validator
from MotionExpert.tests.fixtures import parameter_arrays, owned_archive
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive


def test_independent_head_chain_matches_full_model_fk(owned_archive):
    archive = SMPLHParameterArchive(owned_archive)
    arrays = parameter_arrays(15)
    arrays["smplh.root_orient"][:] = [.13, -.2, .3]
    arrays["smplh.pose_body"][:, 3:6] = [-.4, .2, .1]
    arrays["smplh.pose_body"][:, 33:36] = [.2, .4, -.3]
    head, finite = validator.independent_head_fk(arrays, archive.model.rest, archive.model.shape_basis)
    full, _ = archive.model.decode(arrays)
    assert finite.all()
    np.testing.assert_allclose(head, full[:, 15], atol=1e-6, rtol=1e-6)
    arrays["smplh.pose_body"][2, 0] = np.nan
    head, finite = validator.independent_head_fk(arrays, archive.model.rest, archive.model.shape_basis)
    assert not finite[2] and np.isnan(head[2]).all()


@pytest.mark.parametrize("source", ["egosuite", "egoexo4d_hm"])
def test_source_specific_expected_mask_preserves_observations(source):
    old = parameter_arrays(3)
    old["quality.observation_residual_valid"][:] = True
    good = np.array([True, False, True])
    expected = validator._expected_masks(old, good, source)
    assert not expected["estimated.valid.body_position"][1].any()
    assert not expected["estimated.valid.body_rotation"][1].any()
    assert not expected["quality.observation_residual_valid"][1].any()
    for column in ("source.observed.body_position", "source.feet_valid", "observation.valid.hand_left_position"):
        if source == "egosuite":
            assert column not in expected
        else:
            assert not expected[column][1].any()
    assert not any(key.startswith("observation.state.") for key in expected)


def _proof_fixture(tmp_path):
    import json
    from types import SimpleNamespace
    from MotionExpert.dataset import fit_revision_acceptance as gate
    from MotionExpert.dataset.smplh_archive import file_sha256
    from MotionExpert.dataset.window_cache import digest
    records = [{"episode_index": i, "source": source, "length": 3,
                "metadata": ({"training_eligible": False, "quarantine_reasons": ["original"]} if i == 2 else {})}
               for i, source in enumerate(("egosuite", "egoexo4d_hm", "egoexo4d_hm"))]
    archive = SimpleNamespace(root=tmp_path.resolve(), _records=records, frames=9,
                              binding={"records_manifest_sha256": "x"}, archive_fingerprint="fingerprint")
    population = {"episodes": 3, "frames": 9}
    parent = {"state": "accepted", "independent_acceptance_pending": False, **population}
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {"path": name, "sha256": file_sha256(path)}
    parent_proof = write("parent.json", parent)
    quality = {"format": gate.AUDIT_FORMAT, "state": "complete", "diagnostic": False,
        "production_modified": False, "finished_at": "done", "episodes_planned": 2, "episodes_completed": 2,
        "episodes": [{"episode_index": 0}, {"episode_index": 1}],
        "source_population": {"scope": "all_eligible_archive", "total": 3, "eligible": 2,
                              "excluded": [{"episode_index": 2}]}}
    quality_proof = write("quality.json", quality)
    build = {"scope": "full", "quality_report_sha256": quality_proof["sha256"],
             "parent_publication_sha256": parent_proof["sha256"]}
    revision = {"format": gate.REVISION_FORMAT, "scope": "full", "binding": archive.binding,
        "archive_fingerprint": archive.archive_fingerprint, "build_binding": build,
        "build_binding_sha256": digest(build), "required_quality_episodes": [0, 1], "quality_episodes": [0, 1]}
    revision_proof = write("revision.json", revision)
    independent = {"format": gate.INDEPENDENT_AUDIT_FORMAT, "scope": "full", "passed": True,
        "failures": [], "binding": archive.binding, "population": population,
        "quality_audit_sha256": quality_proof["sha256"], "revision_manifest_sha256": revision_proof["sha256"],
        "new_head_specific_physical_gate": {"scope": "all_eligible_archive", "episodes_checked": 2, "episode_indices": [0, 1]},
        **{name: True for name in gate.ACCEPTANCE_FLAGS}}
    independent_proof = write("independent.json", independent)
    report = {"format": gate.ACCEPTANCE_FORMAT, "scope": "full", "passed": True, "failures": [],
        "population": population, "binding": archive.binding,
        "proofs": {"quality_audit": quality_proof, "revision_manifest": revision_proof,
                   "independent_revision_audit": independent_proof, "parent_publication": parent_proof},
        **{name: True for name in gate.ACCEPTANCE_FLAGS}}
    return archive, report


def test_strict_dual_source_proof_gate_accepts_complete_owned_evidence(tmp_path):
    from MotionExpert.dataset.fit_revision_acceptance import check_fit_revision_acceptance
    archive, report = _proof_fixture(tmp_path)
    check_fit_revision_acceptance(archive, report, tmp_path / "acceptance.json")


@pytest.mark.parametrize("poison", ["missing_source", "unchecked_fk", "external_proof", "changed_proof"])
def test_strict_dual_source_proof_gate_rejects_partial_or_forged_evidence(tmp_path, poison):
    import json
    from MotionExpert.dataset.fit_revision_acceptance import check_fit_revision_acceptance
    from MotionExpert.dataset.smplh_archive import file_sha256
    archive, report = _proof_fixture(tmp_path)
    if poison == "missing_source":
        path = tmp_path / "quality.json"
        data = json.loads(path.read_text()); data["episodes"] = data["episodes"][:1]
        path.write_text(json.dumps(data)); report["proofs"]["quality_audit"]["sha256"] = file_sha256(path)
    elif poison == "unchecked_fk":
        path = tmp_path / "independent.json"
        data = json.loads(path.read_text()); data["final_fk_quality_independently_checked"] = False
        path.write_text(json.dumps(data)); report["proofs"]["independent_revision_audit"]["sha256"] = file_sha256(path)
    elif poison == "external_proof":
        report["proofs"]["quality_audit"]["path"] = str(tmp_path / "quality.json")
    else:
        (tmp_path / "quality.json").write_text('{}')
    with pytest.raises(ValueError):
        check_fit_revision_acceptance(archive, report, tmp_path / "acceptance.json")


from MotionExpert.tests.test_revise_fit_archive import fit_revision_source


@pytest.fixture
def fit_source(fit_revision_source, monkeypatch):
    from MotionExpert import config as config_module
    archive = SMPLHParameterArchive(fit_revision_source[0])
    monkeypatch.setattr(config_module, "SMPLH_MODEL_SHA256", archive.model_sha256)
    return fit_revision_source


def _build_fit(source, root):
    from MotionExpert.revise_fit_archive import build_revision
    parent, vision, quality = source
    return build_revision(parent, quality, root, source_vision_root=vision, workers=1)


@pytest.mark.parametrize("workers", [1, 2])
def test_dual_source_delta_publish_and_owned_reader_gate(fit_source, tmp_path, workers):
    import json
    import os
    parent, vision, _ = fit_source
    root = tmp_path / "candidate"
    built = _build_fit(fit_source, root)
    assert built["changed_episode_indices"] == [0, 1]
    report = validator.audit_revision(parent, root, parent_vision_root=vision, workers=workers)
    assert report["new_head_specific_physical_gate"]["episode_indices"] == [0, 1, 3]
    assert report["new_head_specific_physical_gate"]["normal_controls"]["egosuite"]
    assert report["unchanged_motion_full_numeric_QA_repeated"] is False
    assert report["dino_payload_rehashed"] is False
    assert report["changed_motion_comparisons"][0]["ground_recomputed"] is False
    assert report["changed_motion_comparisons"][1]["ground_recomputed"] is True
    for index in (2, 3):
        assert os.path.samefile(parent / f"episodes/{index:08d}/motion.parquet", root / f"episodes/{index:08d}/motion.parquet")
    proof = root / ".pipeline/independent-fit-revision-audit.json"
    validator._write_new(proof, report)
    top = validator.publish_revision(parent, root, proof, parent_vision_root=vision)
    assert top["passed"] and top["final_fk_quality_independently_checked"]
    state = json.loads((root / ".pipeline/dataset-state.json").read_text())
    assert state["state"] == "accepted" and state["formal_training_ready"] is False
    SMPLHParameterArchive(root, vision_profile_root=root / validator.PROFILE)
    with pytest.raises(ValueError, match="published once"):
        validator.publish_revision(parent, root, proof, parent_vision_root=vision)


@pytest.mark.parametrize("poison", ["noop", "camera", "source_capture_mask"])
def test_independent_validator_rejects_builder_mistakes(fit_source, tmp_path, monkeypatch, poison):
    from MotionExpert import revise_fit_archive as builder
    original = builder.apply_fit_quality_masks
    def wrong(arrays, good, source):
        if poison == "noop":
            return dict(arrays)
        result = original(arrays, good, source)
        if poison == "camera":
            result["observation.state.camera_world"] = arrays["observation.state.camera_world"] + .1
        elif source == "egosuite":
            result["source.observed.body_position"] = arrays["source.observed.body_position"] & good[:, None]
        return result
    monkeypatch.setattr(builder, "apply_fit_quality_masks", wrong)
    root = tmp_path / "candidate"
    _build_fit(fit_source, root)
    with pytest.raises(ValueError, match="value mismatch"):
        validator.audit_revision(fit_source[0], root, parent_vision_root=fit_source[1], workers=1)


def test_independent_recomputation_rejects_consistently_bound_false_audit_mask(fit_source, tmp_path):
    import json
    from MotionExpert.dataset.window_cache import atomic_json
    from MotionExpert.dataset.smplh_archive import file_sha256
    parent, vision, report_path = fit_source
    report = json.loads(report_path.read_text())
    receipt = report["episodes"][0]
    path = report_path.parent / receipt["mask_path"]
    with np.load(path) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    arrays["quality_valid"][:] = arrays["native_valid"] & arrays["fit_finite"]
    arrays["newly_rejected"][:] = False
    np.savez(path, **arrays)
    receipt["mask_sha256"] = file_sha256(path)
    receipt["summary"]["newly_rejected_frames"] = 0
    atomic_json(report_path.parent / f"episode_{receipt['episode_index']:08d}.json", receipt)
    atomic_json(report_path, report)
    root = tmp_path / "candidate"
    _build_fit(fit_source, root)
    with pytest.raises(ValueError, match="Independent final-FK recomputation differs"):
        validator.audit_revision(parent, root, parent_vision_root=vision, workers=1)


def test_already_unavailable_fits_inherit_independent_observations_without_new_mask_audit(owned_archive, tmp_path):
    from MotionExpert.tests.fixtures import rewrite_episode
    from MotionExpert.tests.test_revise_fit_archive import write_fit_quality
    from MotionExpert.tests.test_vision_profile_cache import write_cache
    parent = SMPLHParameterArchive(owned_archive)
    for record in parent.records():
        def unavailable(arrays):
            for component in ("position", "rotation"):
                arrays[f"estimated.valid.body_{component}"][:] = False
                arrays[f"estimated.completed.body_{component}"][:] = False
                arrays[f"quality.{component}_confidence"][:] = 0
            arrays["quality.observation_residual_valid"][:] = False
            # Independent/source hand and foot observations remain present.
        rewrite_episode(owned_archive, record["episode_index"], motion_mutation=unavailable)
    vision = write_cache(owned_archive, tmp_path / "native-unavailable-visual")
    quality = write_fit_quality(owned_archive, tmp_path / "native-unavailable-quality", vision=vision)
    root = tmp_path / "candidate"
    result = _build_fit((owned_archive, vision, quality), root)
    assert result["changed_episode_indices"] == []
    report = validator.audit_revision(owned_archive, root, parent_vision_root=vision, workers=1)
    assert report["passed"] and report["inherited_new_rejected_frame_masks_checked"] == 0
    assert report["new_head_specific_physical_gate"]["newly_rejected_frames"] == 0


def _rewrite_profile_join(vision, index, mutation):
    import json
    path = vision / f"episodes/{index:08d}/join.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    mutation(arrays)
    np.savez(path, **arrays)
    receipt_path = path.with_name("receipt.json")
    receipt = json.loads(receipt_path.read_text())
    receipt["output_identities"]["join.npz"].update(size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
    receipt_path.write_text(json.dumps(receipt))


@pytest.mark.parametrize("derived", ["observation.video_time_error", "observation.video_timestamp"])
def test_native_dino_video_derivatives_may_differ_from_preserved_motion(fit_source, tmp_path, derived):
    import os
    parent, vision, _ = fit_source
    def recomputed(arrays):
        if derived.endswith("time_error"):
            arrays[derived] = np.nextafter(arrays[derived], np.inf)
        else:
            arrays[derived] += .001
    _rewrite_profile_join(vision, 0, recomputed)
    root = tmp_path / "candidate"
    _build_fit(fit_source, root)
    report = validator.audit_revision(parent, root, parent_vision_root=vision, workers=1)
    assert report["passed"] and report["image_join_values_preserved"]
    assert os.path.samefile(vision / "episodes/00000000/join.npz",
                           root / validator.PROFILE / "episodes/00000000/join.npz")


def test_native_source_join_mismatch_remains_exactly_rejected(fit_source, tmp_path, monkeypatch):
    from MotionExpert import revise_fit_archive as builder
    parent, vision, _ = fit_source
    def wrong_source(arrays):
        arrays["observation.source_timestamp"][1] = np.nextafter(arrays["observation.source_timestamp"][1], np.inf)
    _rewrite_profile_join(vision, 0, wrong_source)
    original = builder.shared._verify_visual_join
    def bypass_producer_rows(source_root, record, source_vision_root, **kwargs):
        return original(source_root, record, source_vision_root, compare_rows=False)
    monkeypatch.setattr(builder.shared, "_verify_visual_join", bypass_producer_rows)
    root = tmp_path / "candidate"
    _build_fit(fit_source, root)
    with pytest.raises(ValueError, match="visual join observation.source_timestamp"):
        validator.audit_revision(parent, root, parent_vision_root=vision, workers=1)
