"""Quality gates cover source corruption without rejecting common camera motion."""
from dataclasses import asdict
import json

import numpy as np
import pytest

from MotionExpert.hm_quality import HMQualityPolicy, HMQualityReason as Reason, audit_camera_motion, score_hm_fit, summarize_hm_quality


def score(nose, camera=None, **kwargs):
    nose = np.asarray(nose, dtype=float)
    count = len(nose)
    defaults = dict(native_valid=np.ones(count, bool), camera_valid=np.ones(count, bool),
                    timestamps_s=np.arange(count) / 30.)
    defaults.update(kwargs)
    return score_hm_fit(nose, np.zeros_like(nose) if camera is None else camera, **defaults)


def test_common_translation_does_not_flag_camera_or_wearer_motion():
    camera = np.arange(12, dtype=float).reshape(4, 3) * 100
    result = score(camera + [.1, .1, .1], camera)
    assert result.quality_valid.all()
    assert result.transition_valid.tolist() == [False, True, True, True]
    np.testing.assert_allclose(result.relative_jump_m[1:], 0, atol=1e-12)


def test_multimeter_source_outlier_is_invalid_even_with_native_valid():
    result = score([[.1, 0, 0], [7, 0, 0], [7.1, 0, 0], [.1, 0, 0]])
    assert not result.quality_valid[1:3].any()
    assert ((result.reasons[1:3] & Reason.HEAD_CAMERA_FAR) != 0).all()
    assert result.native_valid.all()
    assert result.head_camera_distance_m[1] == 7


def test_finite_unknown_camera_is_reported_without_automatic_fit_rejection():
    result = score([[.1, 0, 0], [.1, 0, 0]], camera_valid=[True, False])
    assert result.quality_valid.all()
    assert result.distance_checked.tolist() == [True, False]
    assert np.isnan(result.head_camera_distance_m[1])
    assert result.reasons[1] & Reason.CAMERA_UNAVAILABLE
    assert not result.transition_checked.any()


def test_native_invalid_and_nonfinite_keep_independent_reasons():
    result = score([[.1, 0, 0], [float("nan"), 0, 0], [.1, 0, 0]],
                   native_valid=[False, True, True], fit_finite=[True, True, False])
    assert not result.quality_valid.any()
    assert result.reasons[0] & Reason.NATIVE_INVALID
    assert result.reasons[1] & Reason.FIT_NONFINITE
    assert result.reasons[2] & Reason.FIT_NONFINITE


def test_relative_jump_invalidates_both_endpoints_within_distance_limit():
    result = score([[.1, 0, 0], [.7, 0, 0], [.7, 0, 0]])
    assert result.quality_valid.tolist() == [False, False, True]
    assert result.transition_checked.tolist() == [False, True, True]
    assert not result.transition_valid.any()
    assert not ((result.reasons & Reason.HEAD_CAMERA_FAR) != 0).any()
    assert (result.reasons[:2] & Reason.RELATIVE_JUMP).all()


def test_invalid_source_does_not_invalidate_valid_neighbour_via_jump():
    result = score([[20, 0, 0], [.1, 0, 0]], native_valid=[False, True])
    assert result.quality_valid.tolist() == [False, True]
    assert not result.transition_checked.any()
    assert not (result.reasons[1] & Reason.RELATIVE_JUMP)


def test_large_timestamp_gap_disables_comparison_without_rejecting_frame():
    result = score([[.1, 0, 0], [.7, 0, 0]], timestamps_s=[0, 1])
    assert result.quality_valid.all()
    assert not result.transition_checked.any()
    assert result.reasons[1] & Reason.TIME_GAP


def test_warning_is_only_advisory_and_strict_threshold_boundary():
    result = score([[.5, 0, 0], [.6, 0, 0], [1., 0, 0]], timestamps_s=[0, 1, 2])
    assert result.quality_valid.all()
    assert not (result.reasons[0] & Reason.HEAD_CAMERA_WARNING)
    assert result.reasons[1] & Reason.HEAD_CAMERA_WARNING


def test_summary_serializes_and_separates_native_invalid_from_added_rejections():
    result = score([[.1, 0, 0], [7, 0, 0], [7, 0, 0]], native_valid=[True, False, True])
    summary = summarize_hm_quality(result)
    json.dumps(summary, allow_nan=False)
    assert summary["native_finite_frames"] == 2
    assert summary["newly_rejected_frames"] == 1
    assert summary["reason_counts"]["head_camera_far"] == 2
    assert sum(summary["distance_native_finite_histogram"]) == 2


@pytest.mark.parametrize("kwargs", [{"max_head_camera_distance_m": 0}, {"max_time_gap_s": float("nan")},
                                   {"warning_head_camera_distance_m": 2}])
def test_policy_rejects_ambiguous_thresholds(kwargs):
    with pytest.raises(ValueError):
        HMQualityPolicy(**kwargs)


@pytest.mark.parametrize("kwargs", [{"native_valid": [2]}, {"timestamps_s": []}, {"camera_valid": []}])
def test_score_rejects_unaligned_inputs(kwargs):
    with pytest.raises(ValueError):
        score([[.1, 0, 0]], **kwargs)


def test_empty_sequence_and_nonfinite_camera_are_explicit():
    empty = score(np.empty((0, 3)))
    assert len(empty.quality_valid) == 0
    result = score([[.1, 0, 0]], camera=[[float("inf"), 0, 0]])
    assert result.quality_valid.all()
    assert result.reasons[0] & Reason.CAMERA_UNAVAILABLE


def test_camera_speed_audit_reports_shared_world_correction_without_masking_body():
    camera = np.zeros((13, 3))
    camera[6:, 0] = 10
    times = np.arange(13) / 30
    before = camera.copy()
    body = score(camera + [.1, 0, 0], camera)
    audit = audit_camera_motion(camera, np.ones(13, bool), times)
    assert body.quality_valid.all()
    np.testing.assert_array_equal(camera, before)
    assert not audit["changes_masks"]
    assert audit["native30Hz"]["top_edges"][0]["native_frame"] == 6
    assert audit["native30Hz"]["top_edges"][0]["speed_m_s"] == pytest.approx(300)
    assert audit["training5Hz_phase0"]["top_edges"][0]["speed_m_s"] == pytest.approx(50)
    json.dumps(audit, allow_nan=False)


def test_camera_speed_audit_skips_unknown_or_gap_edges():
    audit = audit_camera_motion([[0, 0, 0], [100, 0, 0]], [True, False], [0, 1])
    assert audit["native30Hz"]["edges_checked"] == 0
    assert audit["native30Hz"]["speed_p50_p90_p95_p99_max_m_s"] is None


def test_read_only_audit_binds_output_and_exact_native_join(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from MotionExpert.audit_hm_quality import audit_episode

    root = tmp_path / "archive"
    root.mkdir()
    output = tmp_path / "audit"
    output.mkdir()
    source = tmp_path / "source.npz"
    joints = np.zeros((1, 3, 67, 3), np.float32)
    joints[0, :, 0, 0] = [.1, 7, .1]
    np.savez(source, joints3d=joints, valid=np.ones((1, 3), np.int8),
             trans=np.zeros((1, 3, 3)), root_orient=np.zeros((1, 3, 3)),
             pose_body=np.zeros((1, 3, 63)), betas_per_frame=np.zeros((1, 3, 16)))
    table = pa.table({"timestamp": [0., 1/30, 2/30],
                      "observation.state.camera_world": pa.array([[0., 0, 0]]*3, type=pa.list_(pa.float32(), 3)),
                      "observation.valid.camera_position": [True]*3,
                      "observation.source_frame_left": [0, 1, 2],
                      "observation.source_frame_right": [0, 1, 2],
                      "observation.source_interpolation_weight": [0., 0, 0]})
    motion = root / "motion.parquet"
    pq.write_table(table, motion)
    stat = source.stat()
    record = {"source": "egoexo4d_hm", "episode_index": 7169, "source_key": "egoexo4d_hm:example_cooking",
              "data_path": "motion.parquet", "length": 3, "split": "train", "motion_sha256": "a"*64,
              "metadata": {"coordinate_verified": True, "source_data": str(source),
                           "source_identity": {"path": str(source), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                                               "sha256": "b"*64}}}
    record_path = root / "record.json"
    record_path.write_text(json.dumps(record))
    source_before, motion_before, record_before = source.read_bytes(), motion.read_bytes(), record_path.read_bytes()
    receipt = audit_episode(root, record_path, output, asdict(HMQualityPolicy()))
    assert receipt["summary"]["newly_rejected_frames"] == 3
    assert receipt["frame_join"] == "native30Hz_identity_checked"
    assert receipt["native_valid_sha256_encoding"] == "bool_C_order_bytes"
    assert receipt["camera_motion_audit"]["native30Hz"]["edges_checked"] == 2
    with np.load(output / receipt["mask_path"]) as saved:
        assert saved["reasons"][1] & Reason.HEAD_CAMERA_FAR
        assert not saved["quality_valid"].any()
    assert source.read_bytes() == source_before
    assert motion.read_bytes() == motion_before
    assert record_path.read_bytes() == record_before

    # Fail rather than silently comparing different source/camera time rows.
    table = table.set_column(table.schema.get_field_index("observation.source_frame_right"),
                             "observation.source_frame_right", pa.array([0, 2, 2]))
    pq.write_table(table, motion)
    with pytest.raises(ValueError, match="exact native source frame joins"):
        audit_episode(root, record_path, output, asdict(HMQualityPolicy()))
def test_auditor_bounds_task_admission_before_consuming_results():
    from concurrent.futures import Future
    from MotionExpert.audit_hm_quality import _audit_tasks

    class CountingFuture(Future):
        consumed = False

        def result(self, timeout=None):
            if not self.consumed:
                pool.unconsumed -= 1
                self.consumed = True
            return super().result(timeout=timeout)

    class Pool:
        unconsumed = 0
        peak = 0

        def submit(self, fn, root, record, output, policy):
            self.unconsumed += 1
            self.peak = max(self.peak, self.unconsumed)
            assert self.unconsumed <= 4
            future = CountingFuture()
            future.set_result(record)
            return future

    pool = Pool()
    result = list(_audit_tasks(pool, range(2602), "/source", "/out", {}, 2))
    assert sorted(map(int, result)) == list(range(2602))
    assert pool.unconsumed == 0
    assert pool.peak == 4
