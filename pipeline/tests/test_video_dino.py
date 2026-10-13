"""Video-only sidecars retain the established encoder and PTS/mask contract."""

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline import features
from egomotion_pipeline.body_model import file_sha256
from egomotion_pipeline.schema import DINO_SHAPE, dino_format
from egomotion_pipeline.storage import read_json
from egomotion_pipeline.video_dino import (CONTRACT, ERROR, FEATURE, PTS, SCHEMA, VALID,
                                         extract_video, input_fingerprint, plan_videos,
                                         validate_plan, validate_video)


class Encoder:
    metadata = dino_format()

    def prepare(self, images):
        return images

    def encode_prepared(self, images):
        result = np.zeros((len(images), *DINO_SHAPE), np.float32)
        for index, image in enumerate(images):
            result[index] = image.mean() + np.arange(5, dtype=np.float32)[:, None]
        return result

    __call__ = encode_prepared


@pytest.fixture
def case(tmp_path):
    video = tmp_path / "input.mp4"
    with av.open(str(video), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            image = np.full((32, 32, 3), index * 20, np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            container.mux(stream.encode(frame))
        container.mux(stream.encode(None))
    item = {"video_id": "fixture/aria01_214-1", "video_path": str(video), "length": 10,
            "fps": 30, "output_path": str(tmp_path / "features.parquet"),
            "video_fingerprint": input_fingerprint(video),
            "preprocessing": {"mode": "raw", "rotate_ccw_quarters": 0}}
    config = {"batch_size": 3, "prefetch_batches": 2, "decoder_threads": 2,
              "optimized_storage": True, "max_video_error_s": 1 / 60 + .001}
    return item, config, Encoder()


def arrays(path):
    table = pq.read_table(path)
    result = {}
    for name in table.column_names:
        value = table[name].combine_chunks()
        result[name] = (value.values.to_numpy().reshape(len(table), -1) if name == FEATURE
                        else value.to_numpy(zero_copy_only=False)).copy()
    return result


def test_matches_existing_egosuite_extraction_with_missing_tail(case, monkeypatch):
    item, config, encoder = case
    extract_video(item, config, encoder)
    stored = arrays(item["output_path"])
    original = {FEATURE: np.zeros((10, 5120), np.float32), VALID: np.zeros(10, bool),
                PTS: np.zeros(10, np.float64), ERROR: np.zeros(10, np.float64),
                "observation.source_timestamp": np.arange(10, dtype=float) / 30}
    record = {"length": 10, "source_key": "fixture", "video": {
        "path": item["video_path"], "from_timestamp": 0., "to_timestamp": 9 / 30,
        "preprocessing": item["preprocessing"]}}
    monkeypatch.setattr(features, "quality_report", lambda _: {})
    features.extract_episode_arrays(record, original, config, encoder)
    for key in original:
        np.testing.assert_array_equal(stored[key], original[key])
    assert stored[VALID].sum() == 8
    assert not stored[FEATURE][8:].any()
    assert np.array_equal(stored[FEATURE][0].reshape(5, 1024)[:, 0], np.arange(5))


def test_same_job_resumes_without_encoding_or_rewriting(case):
    item, config, encoder = case
    first = extract_video(item, config, encoder)
    before = Path(item["output_path"]).stat().st_mtime_ns
    encoder.encode_prepared = lambda _: pytest.fail("Completed sidecar must be reused")
    second = extract_video(item, config, encoder)
    assert first["kind"] == "committed" and second["kind"] == "skipped"
    assert Path(item["output_path"]).stat().st_mtime_ns == before


def test_long_sidecar_crosses_row_groups_without_losing_invalid_rows(case):
    item, config, encoder = case
    item = {**item, "length": 3107}
    event = extract_video(item, {**config, "batch_size": 32}, encoder)
    assert pq.ParquetFile(item["output_path"]).metadata.num_row_groups >= 2
    assert event["record"]["length"] == 3107 and event["record"]["valid_count"] == 8
    assert validate_video(item["output_path"], event["record"])["missing_frames"] == 3099


def test_changed_config_or_encoder_is_rejected(case):
    item, config, encoder = case
    extract_video(item, config, encoder)
    with pytest.raises(ValueError, match="different input/model/config"):
        extract_video(item, {**config, "batch_size": 4}, encoder)
    encoder.metadata = {**encoder.metadata, "checkpoint": "changed"}
    with pytest.raises(ValueError, match="different input/model/config"):
        extract_video(item, config, encoder)


def test_changed_input_is_rejected(case):
    item, config, encoder = case
    Path(item["video_path"]).touch()
    with pytest.raises(ValueError, match="changed since preflight"):
        extract_video(item, config, encoder)


def test_interrupted_extraction_never_publishes_complete_output(case):
    item, config, encoder = case
    def interrupted(*_):
        raise RuntimeError("interrupted")
    encoder.encode_prepared = interrupted
    with pytest.raises(RuntimeError, match="interrupted"):
        extract_video(item, config, encoder)
    assert not Path(item["output_path"]).exists()
    assert not Path(item["output_path"]).with_suffix(".json").exists()
    assert not Path(item["output_path"]).with_suffix(".parquet.tmp").exists()


@pytest.mark.parametrize("bad_output", [lambda n: np.zeros((n, 1024), np.float32),
                                       lambda n: np.full((n, 5, 1024), np.nan, np.float32)])
def test_bad_encoder_output_is_rejected(case, bad_output):
    item, config, encoder = case
    encoder.encode_prepared = lambda images: bad_output(len(images))
    with pytest.raises(ValueError, match="finite"):
        extract_video(item, config, encoder)


def test_file_corruption_is_detected_on_resume(case):
    item, config, encoder = case
    extract_video(item, config, encoder)
    with Path(item["output_path"]).open("ab") as handle:
        handle.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        extract_video(item, config, encoder)


@pytest.mark.parametrize("field", [FEATURE, "frame_index", PTS, VALID])
def test_semantic_corruption_is_detected_even_with_new_checksum(case, field):
    item, config, encoder = case
    event = extract_video(item, config, encoder)
    data = arrays(item["output_path"])
    if field == FEATURE:
        data[field][-1, 0] = 1
    elif field == "frame_index":
        data[field][0] = 10
    elif field == PTS:
        data[field][0] = 1
    else:
        data[field][0] = False
    columns = [pa.FixedSizeListArray.from_arrays(pa.array(data[name].reshape(-1)), 5120)
               if name == FEATURE else pa.array(data[name], type=SCHEMA.field(name).type) for name in SCHEMA.names]
    pq.write_table(pa.Table.from_arrays(columns, schema=SCHEMA), item["output_path"])
    meta = {**event["record"], "parquet_sha256": file_sha256(item["output_path"])}
    with pytest.raises(ValueError):
        validate_video(item["output_path"], meta)


def test_full_report_counts_missing_outputs_as_failures(case):
    item, config, encoder = case
    extract_video(item, config, encoder)
    missing = {**item, "video_id": "missing", "output_path": str(Path(item["output_path"]).parent / "missing.parquet")}
    plan = {"items": [item, missing], "geometry_policy": "raw"}
    report = validate_plan(plan)
    assert not report["passed"] and report["completed_videos"] == 1
    assert report["frames"] == 10 and report["valid_frames"] == 8
    assert len(report["failed_videos"]) == 1


def test_plan_selects_ego_files_and_preserves_take_clock(case, tmp_path):
    item, _, _ = case
    root = tmp_path / "dataset"
    video = root / "EgoExo4D/takes/alpha/frame_aligned_videos/aria01_214-1.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(Path(item["video_path"]).read_bytes())
    take = {"take_name": "alpha", "take_uid": "take-id", "root_dir": "takes/alpha",
            "timesync_start_idx": 20, "timesync_end_idx": 28,
            "capture": {"cameras": [{"cam_id": "aria01", "is_ego": True, "device_type": "aria"},
                                     {"cam_id": "exo", "is_ego": False}]},
            "frame_aligned_videos": {"aria01": {"rgb": {"relative_path": "frame_aligned_videos/aria01_214-1.mp4"}}}}
    (root / "EgoExo4D/takes.json").write_text(json.dumps([take]))
    (root / "download_plan.json").write_text(json.dumps({"hm_revision": "pinned", "s3_files": [{
        "part": "takes", "relative_path": "takes/alpha/frame_aligned_videos/aria01_214-1.mp4",
        "size": video.stat().st_size}]}))
    plan = plan_videos(root, tmp_path / "output")
    assert plan["video_count"] == 1 and plan["total_frames"] == 8
    assert plan["items"][0]["video_info"]["frames"] == 8
    assert plan["items"][0]["clock"] == "frame_aligned_take_seconds"
    assert not plan["hm_row_alignment_established"]
    assert plan["contract"] == CONTRACT
    bad = deepcopy(take)
    bad["capture"]["cameras"][0]["is_ego"] = False
    (root / "EgoExo4D/takes.json").write_text(json.dumps([bad]))
    with pytest.raises(ValueError, match="one ego camera"):
        plan_videos(root, tmp_path / "output")
