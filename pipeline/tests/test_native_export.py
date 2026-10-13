"""Transactional export, source isolation, and resume tests on small archives."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from egomotion_pipeline.native_export import run_export, validate_export
from egomotion_pipeline.schema import Episode
from egomotion_pipeline.storage import Archive, read_arrays, write_arrays
from test_native_dense22 import make_native_input, WORLD, FLOOR, _mask


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(root):
    return {str(path.relative_to(root)): _sha(path) for path in sorted(root.rglob("*"))
            if path.is_file()}


def _json(path):
    return json.loads(Path(path).read_text())


def make_archive(root, episodes=2, *, missing_ground=False):
    archive = Archive(root, {"fps": 30, "native_test": True})
    # Production archives already have this writer lock.  Create it during
    # fixture setup so the exporter can acquire it read-only without mutation.
    with archive.lock():
        pass
    for index in range(episodes):
        arrays, record, _, _, _ = make_native_input()
        count = len(arrays["timestamp"])
        for key, spec in archive.specs.items():
            if key not in arrays:
                shape = (count,) if spec["shape"] == [1] else (count, *spec["shape"])
                arrays[key] = np.zeros(shape, dtype=spec["dtype"])
        # Nonzero DINO bytes help detect accidental source rewrites.
        arrays["observation.features.dino"][:] = index + 0.875
        arrays["observation.valid.dino"][:] = True
        if missing_ground and index == 0:
            record["metadata"]["ground"] = {"valid": False, "reason": "no_stationary_feet"}
            arrays["observation.valid.head_reference_floor_position"][:] = False
            arrays["observation.valid.head_reference_floor_rotation"][:] = False
        episode = Episode(f"native-{index}", "egosuite", np.arange(count) / 30,
                          {"body": arrays["observation.state.body_world"].reshape(count, 22, 3)},
                          {"body": arrays["observation.state.body_world_rotation"].reshape(count, 22, 4)},
                          metadata=deepcopy(record["metadata"]))
        archive.add(episode, arrays, f"fixture-input-{index}")
    archive.finalize()
    return Archive(root)


def _data_path(output, index):
    return Path(output) / f"data/chunk-{index // 1000:03d}/file-{index % 1000:03d}.parquet"


def _read_sidecar(path):
    table = pq.read_table(path)
    output = {}
    for key in table.column_names:
        column = table[key].combine_chunks()
        if hasattr(column.type, "list_size"):
            output[key] = column.values.to_numpy(zero_copy_only=False).reshape(len(table), column.type.list_size)
        else:
            output[key] = column.to_numpy(zero_copy_only=False)
    return output


def test_export_never_reads_dino_and_preserves_every_source_file(tmp_path, monkeypatch):
    archive = make_archive(tmp_path / "archive", missing_ground=True)
    source_before = _snapshot(archive.root)
    source_paths = {str(archive.data_path(index).resolve()) for index in range(2)}
    read_calls = []
    original_read_table = pq.read_table
    original_parquet_file = pq.ParquetFile

    def check_projection(source, columns):
        try:
            is_source = str(Path(source).resolve()) in source_paths
        except TypeError:
            is_source = False
        if is_source:
            assert columns is not None, "Source Parquet was read without a column projection"
            assert not any("dino" in key for key in columns), "DINO was read by motion export"
            read_calls.append(tuple(columns))

    def guarded_read_table(source, *args, **kwargs):
        check_projection(source, kwargs.get("columns"))
        return original_read_table(source, *args, **kwargs)

    class GuardedParquetFile:
        def __init__(self, source, *args, **kwargs):
            self.source = source
            self.delegate = original_parquet_file(source, *args, **kwargs)

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def read(self, *args, **kwargs):
            columns = kwargs.get("columns", args[0] if args else None)
            check_projection(self.source, columns)
            return self.delegate.read(*args, **kwargs)

        def read_row_group(self, index, *args, **kwargs):
            check_projection(self.source, kwargs.get("columns", args[0] if args else None))
            return self.delegate.read_row_group(index, *args, **kwargs)

        def iter_batches(self, *args, **kwargs):
            check_projection(self.source, kwargs.get("columns"))
            return self.delegate.iter_batches(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", guarded_read_table)
    monkeypatch.setattr(pq, "ParquetFile", GuardedParquetFile)
    output = tmp_path / "native"
    report = run_export(archive.root, output, workers=1)
    assert report["stage"] == "completed"
    assert report["failed_episodes"] == 0
    assert report["committed_episodes"] == 2
    assert report["committed_frames"] == 8
    assert read_calls
    assert _snapshot(archive.root) == source_before
    validation = validate_export(archive.root, output, workers=1)
    assert validation["stage"] == "completed"
    assert validation["validated_episodes"] == 2
    assert validation["validated_frames"] == 8
    assert validation["source_inventory_unchanged"]
    assert validation["code_unchanged"]
    first = _read_sidecar(_data_path(output, 0))
    assert not any("dino" in key for key in first)
    assert first[WORLD].shape == first[FLOOR].shape == (4, 198)
    assert first[_mask("world", "position")].all()
    assert first[_mask("world", "rotation")].all()
    assert not first[_mask("floor", "position")].any()
    assert not first[_mask("floor", "rotation")].any()
    assert _snapshot(archive.root) == source_before


def test_successful_resume_does_not_rewrite_committed_episode(tmp_path):
    archive = make_archive(tmp_path / "archive")
    output = tmp_path / "native"
    run_export(archive.root, output, workers=1)
    paths = [_data_path(output, index) for index in range(2)]
    before = [(path.stat().st_mtime_ns, _sha(path)) for path in paths]
    result = run_export(archive.root, output, workers=1, resume=True)
    assert result["stage"] == "completed"
    assert result["skipped_episodes"] == 2
    assert result["processed_episodes"] == 0
    assert [(path.stat().st_mtime_ns, _sha(path)) for path in paths] == before
    for index in range(2):
        record = _json(output / "episodes" / f"{index:08d}.json")
        for key in ("source_record_sha256", "source_motion_sha256", "output_motion_sha256",
                    "parquet_file_sha256"):
            assert isinstance(record[key], str) and len(record[key]) == 64
        assert record["source_input_fingerprint"] == f"fixture-input-{index}"


def test_partial_export_can_resume_complete_without_changing_scope(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=3)
    output = tmp_path / "native"
    partial = run_export(archive.root, output, workers=1, limit=1)
    assert partial["stage"] == "partial"
    assert partial["total_episodes"] == 3 and partial["selected_episodes"] == 1
    original = _sha(_data_path(output, 0))
    complete = run_export(archive.root, output, workers=1, resume=True)
    assert complete["stage"] == "completed"
    assert complete["committed_episodes"] == 3
    assert complete["skipped_episodes"] == 1
    assert _sha(_data_path(output, 0)) == original


@pytest.mark.parametrize("initial_limit", [None, 2])
def test_resume_cannot_shrink_committed_scope_or_rewrite_reports(tmp_path, initial_limit):
    archive = make_archive(tmp_path / "archive", episodes=3)
    output = tmp_path / "native"
    initial = run_export(archive.root, output, workers=1, limit=initial_limit)
    assert initial["stage"] == ("completed" if initial_limit is None else "partial")
    before = _snapshot(output)
    with pytest.raises(ValueError, match="committed|scope|limit|shrink|prefix|smaller"):
        run_export(archive.root, output, workers=1, limit=1, resume=True)
    # Reports and commit markers must describe every already committed file;
    # refusing the smaller request must not rewrite status/info/validation.
    assert _snapshot(output) == before


def test_source_record_change_rejects_stale_scope_before_writing(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=1)
    output = tmp_path / "native"
    run_export(archive.root, output, workers=1)
    before = _snapshot(output)
    path = archive.record_path(0)
    record = _json(path)
    record["metadata"]["operator_id"] = "changed-after-export"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        run_export(archive.root, output, workers=1, resume=True)
    assert _snapshot(output) == before


def test_source_motion_change_is_reported_and_never_overwrites_stale_output(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=1)
    output = tmp_path / "native"
    run_export(archive.root, output, workers=1)
    derived_path = _data_path(output, 0)
    derived_before = _sha(derived_path)
    arrays = read_arrays(archive.data_path(0), archive.specs)
    arrays["observation.state.body_world"][0, 0] += 0.25
    write_arrays(archive.data_path(0), arrays, archive.specs)
    result = run_export(archive.root, output, workers=1, resume=True)
    assert result["stage"] == "failed"
    assert result["failed_episodes"] == 1
    assert result["failures"]
    assert _sha(derived_path) == derived_before
    assert validate_export(archive.root, output, workers=1)["stage"] == "failed"


def test_corrupt_output_is_rejected_instead_of_silently_resumed(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=1)
    output = tmp_path / "native"
    run_export(archive.root, output, workers=1)
    path = _data_path(output, 0)
    original = path.read_bytes()
    path.write_bytes(original[:-12] + b"invalid-footer")
    corrupted_sha = _sha(path)
    result = run_export(archive.root, output, workers=1, resume=True)
    assert result["stage"] == "failed"
    assert result["failed_episodes"] == 1
    assert _sha(path) == corrupted_sha


def test_parallel_worker_outputs_match_serial(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=3)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    assert run_export(archive.root, serial, workers=1)["stage"] == "completed"
    assert run_export(archive.root, parallel, workers=2)["stage"] == "completed"
    for index in range(3):
        left = _read_sidecar(_data_path(serial, index))
        right = _read_sidecar(_data_path(parallel, index))
        assert set(left) == set(right)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize("kwargs", [
    {"workers": True}, {"workers": 0}, {"workers": -1}, {"workers": "1"},
    {"limit": True}, {"limit": 0}, {"limit": -1}, {"limit": 1.5},
    {"resume": "false"},
])
def test_invalid_export_options_fail_before_output_creation(tmp_path, kwargs):
    archive = make_archive(tmp_path / "archive", episodes=1)
    output = tmp_path / "native"
    with pytest.raises(ValueError):
        run_export(archive.root, output, **kwargs)
    assert not output.exists()


@pytest.mark.parametrize("relative", [".", "data", ".pipeline", "meta"])
def test_output_cannot_overwrite_archive_structure(tmp_path, relative):
    archive = make_archive(tmp_path / "archive", episodes=1)
    before = _snapshot(archive.root)
    with pytest.raises(ValueError):
        run_export(archive.root, archive.root / relative, workers=1)
    assert _snapshot(archive.root) == before


def test_wrong_native_provenance_is_episode_failure_not_false_success(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=1)
    path = archive.record_path(0)
    record = _json(path)
    record["metadata"]["body_basis_correction"] = np.broadcast_to(np.eye(3), (22, 3, 3)).tolist()
    path.write_text(json.dumps(record))
    output = tmp_path / "native"
    result = run_export(archive.root, output, workers=1)
    assert result["stage"] == "failed"
    assert result["failed_episodes"] == 1
    assert not _data_path(output, 0).exists()


@pytest.mark.parametrize("key", ["index", "frame_index", "episode_index", "task_index"])
def test_corrupted_source_frame_join_never_commits_labels(tmp_path, key):
    archive = make_archive(tmp_path / "archive", episodes=1)
    arrays = read_arrays(archive.data_path(0), archive.specs)
    arrays[key][1] += 9
    write_arrays(archive.data_path(0), arrays, archive.specs)
    source_before = _snapshot(archive.root)
    output = tmp_path / "native"
    report = run_export(archive.root, output, workers=1)
    assert report["stage"] == "failed"
    assert report["failed_episodes"] == 1
    assert not _data_path(output, 0).exists()
    assert _snapshot(archive.root) == source_before


def test_tampered_derived_semantics_is_not_silently_resumed(tmp_path):
    archive = make_archive(tmp_path / "archive", episodes=1)
    output = tmp_path / "native"
    run_export(archive.root, output, workers=1)
    path = output / "episodes/00000000.json"
    committed = _json(path)
    committed["derived_metadata"]["position_semantics"] = "smplx_joint_centers"
    path.write_text(json.dumps(committed))
    before = _sha(_data_path(output, 0))
    report = run_export(archive.root, output, workers=1, resume=True)
    assert report["stage"] == "failed" and report["failed_episodes"] == 1
    assert _sha(_data_path(output, 0)) == before
