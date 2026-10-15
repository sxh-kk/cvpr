"""Source-aware final-FK revisions preserve capture data and immutable parents."""
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from MotionExpert import revise_fit_archive as revision
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, assert_smplh_archive_ready, file_sha256
from MotionExpert.dataset.window_cache import atomic_json
from MotionExpert.fit_quality import (FitQualityPolicy, fk_head_world, native_fit_availability,
                                      score_fit_quality, summarize_fit_quality)
from MotionExpert.tests.fixtures import owned_archive, parameter_arrays, rewrite_episode
from MotionExpert.tests.test_vision_profile_cache import write_cache


def write_fit_quality(root, output, *, vision=None):
    """Real final-FK scorer, full eligible fixture population and bound arrays."""
    output.mkdir()
    archive = SMPLHParameterArchive(root, vision_profile_root=vision)
    receipts, excluded = [], []
    policy = FitQualityPolicy()
    for record in archive.records():
        index = record["episode_index"]
        if not revision.shared._source_eligible(record):
            excluded.append({"episode_index": index, "source_key": record["source_key"], "reasons": ["quarantined"]})
            continue
        arrays = revision.shared._columns(pq.read_table(archive.data_path(index)))
        head, finite = fk_head_world(archive.model, arrays)
        native = native_fit_availability(arrays["estimated.valid.body_position"], arrays["estimated.valid.body_rotation"])
        result = score_fit_quality(head, arrays["observation.state.camera_world"], arrays["observation.state.head_world"],
            native_valid=native, camera_valid=arrays["observation.valid.camera_position"],
            head_observation_valid=arrays["observation.valid.head_position"], timestamps_s=arrays["timestamp"],
            fit_finite=finite, policy=policy)
        path = output / f"episode_{index:08d}.npz"
        np.savez(path, **result.arrays())
        record_path = root / f".pipeline/episodes/{index:08d}.json"
        receipt = {"format": revision.AUDIT_FORMAT, "episode_index": index, "source_key": record["source_key"],
            "split": record["split"], "source": record["source"], "record_sha256": file_sha256(record_path),
            "motion_sha256": record["motion_sha256"], "mask_path": path.name, "mask_sha256": file_sha256(path),
            "native_valid_sha256": hashlib.sha256(native.tobytes()).hexdigest(), "policy": asdict(policy),
            "summary": summarize_fit_quality(result)}
        atomic_json(output / f"episode_{index:08d}.json", receipt)
        receipts.append(receipt)
    report = {"format": revision.AUDIT_FORMAT, "state": "complete", "archive_root": str(root),
        "production_modified": False, "diagnostic": False, "policy": asdict(policy), "episodes": receipts,
        "episodes_planned": len(receipts), "episodes_completed": len(receipts), "finished_at": "fixture_finished",
        "source_population": {"scope": "all_eligible_archive", "total": len(archive.records()), "episodes": len(archive.records()),
                              "eligible": len(receipts), "excluded": excluded}}
    atomic_json(output / "report.json", report)
    return output / "report.json"


@pytest.fixture
def fit_revision_source(owned_archive, tmp_path):
    parent = SMPLHParameterArchive(owned_archive)
    for record in parent.records():
        index = record["episode_index"]
        arrays = revision.shared._columns(pq.read_table(parent.data_path(index)))
        head, _ = fk_head_world(parent.model, arrays)
        def mutation(values, head=head.copy(), index=index):
            for name in ("head", "camera", "head_reference"):
                values[f"observation.state.{name}_world"] = head.copy()
            values["observation.state.camera_world"][:, 2] += .08
            values["observation.state.head_reference_world"] = values["observation.state.camera_world"].copy()
            if index in (0, 1):
                values["smplh.trans"][:31, 0] += 2
            if index == 0:
                values["source.observed.body_position"][:31, 15] = False
                values["estimated.completed.body_position"][:31, 15] = True
                values["quality.position_confidence"][:31, 15] = .15
            if index == 1:
                values["source.feet_world"][:31, [2, 5]] = -4
        rewrite_episode(owned_archive, index, motion_mutation=mutation)
    vision = write_cache(owned_archive, tmp_path / "vision")
    quality = write_fit_quality(owned_archive, tmp_path / "quality", vision=vision)
    return owned_archive, vision, quality


def build_small(source, output, **kwargs):
    root, vision, quality = source
    return revision.build_revision(root, quality, output, source_vision_root=vision, workers=1, **kwargs)


def test_egosuite_rejects_estimates_and_retains_independent_observations():
    arrays = parameter_arrays(5)
    arrays["source.observed.body_position"][:, 15] = False
    arrays["estimated.completed.body_position"][:, 15] = True
    arrays["quality.observation_residual_valid"][:] = True
    good = np.array([True, False, True, True, False])
    result = revision.apply_fit_quality_masks(arrays, good, "egosuite")
    for component in ("position", "rotation"):
        assert not result[f"estimated.valid.body_{component}"][~good].any()
        assert not result[f"estimated.completed.body_{component}"][~good].any()
        assert not result[f"quality.{component}_confidence"][~good].any()
        np.testing.assert_array_equal(result[f"source.observed.body_{component}"], arrays[f"source.observed.body_{component}"])
        np.testing.assert_array_equal(result[f"source.interpolated.body_{component}"], arrays[f"source.interpolated.body_{component}"])
    assert not result["quality.observation_residual_valid"][~good].any()
    for name in ("source.feet_world", "source.feet_valid", "source.feet_interpolated",
                 "observation.valid.hand_left_position", "observation.valid.hand_right_position",
                 "observation.state.hand_left_world", "observation.state.head_world", "observation.valid.camera_position"):
        np.testing.assert_array_equal(result[name], arrays[name])
    assert arrays["estimated.valid.body_position"].all()


def test_hm_rejects_fit_linked_source_masks_and_hands_feet():
    arrays = parameter_arrays(5)
    good = np.array([True, False, True, True, False])
    result = revision.apply_fit_quality_masks(arrays, good, "egoexo4d_hm")
    for name in ("source.observed.body_position", "source.observed.body_rotation", "source.feet_valid",
                 "observation.valid.hand_left_position", "observation.valid.hand_right_position"):
        assert not result[name][~good].any()
    for name in ("smplh.trans", "smplh.betas", "source.feet_world", "observation.state.head_world",
                 "observation.state.camera_world", "observation.valid.camera_position", "observation.valid.dino"):
        np.testing.assert_array_equal(result[name], arrays[name])


def test_candidate_end_to_end_preserves_unmodified_records_and_visual_payloads(fit_revision_source, tmp_path):
    root, vision, _ = fit_revision_source
    output = tmp_path / "new"
    result = build_small(fit_revision_source, output)
    assert result["state"] == "candidate" and result["changed_episode_indices"] == [0, 1]
    with pytest.raises(ValueError, match="not published"):
        assert_smplh_archive_ready(output)
    old = SMPLHParameterArchive(root)
    new = SMPLHParameterArchive(output, allow_unpublished=True, vision_profile_root=result["vision_profile_root"])
    for index in (2, 3):
        assert old.records()[index] == new.records()[index]
        assert os.path.samefile(old.data_path(index), new.data_path(index))
    for index in (0, 1):
        before = old._motion(old.records()[index], np.arange(91))
        after = new._motion(new.records()[index], np.arange(91))
        assert not after["estimated.valid.body_position"][:31].any()
        assert new.records()[index]["metadata"]["fit_quality_revision"]["native_valid_sha256"]
        for name in before:
            if name.startswith("smplh.") and not name.startswith("smplh.valid.") or name.startswith("observation.state."):
                np.testing.assert_array_equal(after[name], before[name])
    assert new.records()[0]["metadata"]["ground"] == old.records()[0]["metadata"]["ground"]
    assert not new.records()[0]["metadata"]["fit_quality_revision"]["ground_recomputed"]
    assert new.records()[1]["metadata"]["fit_quality_revision"]["ground_recomputed"]
    assert new.records()[1]["metadata"]["ground"]["height_m"] == pytest.approx(.51, abs=1e-7)
    for index in range(4):
        for name in ("features.npy", "join.npz"):
            assert os.path.samefile(vision / f"episodes/{index:08d}/{name}",
                                    Path(result["vision_profile_root"]) / f"episodes/{index:08d}/{name}")


def test_no_new_rejections_keep_records_exact_without_decoding(owned_archive, tmp_path, monkeypatch):
    root = owned_archive
    archive = SMPLHParameterArchive(root)
    # Remove all prior estimates: the new audit cannot certify an unavailable
    # fit but also has no newly rejected body labels to rewrite.
    for record in archive.records():
        def disable(arrays):
            arrays.update(revision.apply_fit_quality_masks(arrays, np.zeros(91, bool), record["source"]))
        rewrite_episode(root, record["episode_index"], motion_mutation=disable)
    vision = write_cache(root, tmp_path / "vision")
    quality = write_fit_quality(root, tmp_path / "quality", vision=vision)
    read_table = pq.read_table
    def reject_episode_decode(path, *args, **kwargs):
        if Path(path).name == "motion.parquet":
            pytest.fail("An unchanged accepted source was decoded")
        return read_table(path, *args, **kwargs)
    monkeypatch.setattr(pq, "read_table", reject_episode_decode)
    result = revision.build_revision(root, quality, tmp_path / "new", source_vision_root=vision, workers=1)
    assert result["changed_episode_indices"] == []


def test_resume_keeps_success_and_frozen_plan(fit_revision_source, tmp_path, monkeypatch):
    output = tmp_path / "new"
    result = build_small(fit_revision_source, output)
    stamp = (output / "episodes/00000000/motion.parquet").stat().st_mtime_ns
    monkeypatch.setattr(revision, "apply_fit_quality_masks", lambda *args: pytest.fail("Rewrote completed episode"))
    resumed = build_small(fit_revision_source, output, resume=True)
    assert result["archive_fingerprint"] == resumed["archive_fingerprint"]
    assert (output / "episodes/00000000/motion.parquet").stat().st_mtime_ns == stamp


def test_quality_new_rejection_count_cannot_hide_changed_labels(fit_revision_source, tmp_path):
    root, _, report_path = fit_revision_source
    receipt = json.loads(report_path.read_text())["episodes"][0]
    receipt["summary"]["newly_rejected_frames"] = 0
    record_path = root / ".pipeline/episodes/00000000.json"
    with pytest.raises(ValueError, match="newly rejected mask/count"):
        revision._load_quality(report_path.parent, receipt, json.loads(record_path.read_text()), record_path.read_bytes())


def test_audit_population_cannot_hide_original_quarantine(fit_revision_source, tmp_path):
    root, vision, report_path = fit_revision_source
    report = json.loads(report_path.read_text())
    report["source_population"]["excluded"] = []
    atomic_json(report_path, report)
    with pytest.raises(ValueError, match="source population"):
        revision.build_revision(root, report_path, tmp_path / "new", source_vision_root=vision, workers=1)


def test_quality_receipt_cannot_change_source_semantics(fit_revision_source):
    root, _, report_path = fit_revision_source
    receipt = json.loads(report_path.read_text())["episodes"][0]
    receipt["source"] = "egoexo4d_hm"
    record_path = root / ".pipeline/episodes/00000000.json"
    with pytest.raises(ValueError, match="accepted parent record"):
        revision._load_quality(report_path.parent, receipt, json.loads(record_path.read_text()), record_path.read_bytes())


def test_spawn_two_workers_preserves_source_specific_masks(fit_revision_source, tmp_path, monkeypatch):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    root, vision, report = fit_revision_source
    result = revision.build_revision(root, report, tmp_path / "new", source_vision_root=vision, workers=2)
    assert result["state"] == "candidate" and result["changed_episode_indices"] == [0, 1]


def test_rebound_profile_paths_and_parent_revision(fit_revision_source, tmp_path):
    root, vision, _ = fit_revision_source
    # An original v2 quality record is historical provenance, not an active
    # v3 decision; initialize it only after creating the frozen source vision.
    result = build_small(fit_revision_source, tmp_path / "new")
    new_info = json.loads((tmp_path / "new/meta/info.json").read_text())
    old_info = json.loads((root / "meta/info.json").read_text())
    assert new_info["quality_revision"]["parent_revision"] == old_info.get("quality_revision")
    for key in old_info:
        if key != "quality_revision":
            assert new_info[key] == old_info[key]
    visual = json.loads((Path(result["vision_profile_root"]) / "manifest.json").read_text())
    assert visual["output_root"] == result["vision_profile_root"]
    assert visual["quality_revision_reuse"]["parent_profile_root"] == str(vision)
    assert visual["encoder"] == json.loads((vision / "manifest.json").read_text())["encoder"]
