"""Corruption, ownership and atomic recovery tests for parameter migration."""
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/process_smplh_parameters.py"
spec = importlib.util.spec_from_file_location("parameter_migration_test", SCRIPT)
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def vision_source(tmp_path, n=8):
    values = {"frame_index": np.arange(n, dtype=np.int64),
              "timestamp": np.arange(n) / 30,
              "observation.source_timestamp": np.arange(n) / 30,
              "observation.valid.dino": np.array([True, False] * (n // 2)),
              "observation.video_timestamp": np.arange(n) / 30 + .001,
              "observation.video_time_error": np.full(n, .001, dtype=np.float32),
              migration.FEATURE: np.arange(n * 5120, dtype=np.float32).reshape(n, 5120) * .001}
    columns = {k: (pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), 5120)
                   if k == migration.FEATURE else pa.array(v)) for k, v in values.items()}
    path = tmp_path / "raw.parquet"
    pq.write_table(pa.table(columns), path, row_group_size=3)
    return path, values


def test_streamed_visual_copy_has_exact_mapped_tokens_and_owned_identity(tmp_path):
    source, raw = vision_source(tmp_path)
    selected = np.array([0, 2, 5, 7])
    arrays = {k: raw[k][selected] for k in ("observation.valid.dino", "observation.video_timestamp", "observation.video_time_error")}
    arrays.update(frame_index=np.arange(4), episode_index=np.full(4, 31),
                  timestamp=(np.arange(4) / 30).astype(np.float32))
    arrays["observation.source_timestamp"] = selected / 30
    output = tmp_path / "owned/vision.parquet"
    receipt = migration.copy_vision(source, output, arrays, selected)
    actual = migration.table_arrays(pq.read_table(output))
    np.testing.assert_array_equal(actual[migration.FEATURE], raw[migration.FEATURE][selected])
    np.testing.assert_array_equal(actual["source.video_frame_index"], selected)
    np.testing.assert_array_equal(actual["episode_index"], np.full(4, 31))
    assert receipt["sha256"] == migration.file_sha(output)
    assert receipt["rows"] == 4
    for key, value in actual.items():
        assert receipt["column_byte_sha256"][key] == hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    assert migration.identity(source) == receipt["source_identity"]
    arrays["observation.video_timestamp"][1] += .0001
    with pytest.raises(ValueError, match="provenance"):
        migration.copy_vision(source, tmp_path / "bad.parquet", arrays, selected)


def committed(tmp_path):
    final = tmp_path / "episodes/00000000"; final.mkdir(parents=True)
    for name in ("motion", "vision"):
        pq.write_table(pa.table({"frame_index": [0, 1]}), final / (name + ".parquet"))
    source = tmp_path / "original.npz"; source.write_bytes(b"original immutable data")
    source_vision = tmp_path / "raw-vision.parquet"; source_vision.write_bytes(b"raw dino")
    row = dict(episode_index=0, source="egoexo4d_hm", source_key="egoexo4d_hm:fixture",
               length=2, split="train", dataset_from_index=0, dataset_to_index=2, source_path=str(source))
    plan = {"model_sha256": migration.MODEL_SHA256, "output_root": str(tmp_path / "production")}
    receipt = {"record": row.copy(), "conversion_binding": migration.conversion_binding(row, plan),
               "source_provenance": {"source_identity": migration.identity(source)},
               "output_identities": {k: migration.identity(final / (k + ".parquet")) for k in ("motion", "vision")}}
    for k in ("motion", "vision"):
        receipt[k] = {"sha256": migration.file_sha(final / (k + ".parquet"))}
    receipt["vision"].update(source_path=str(source_vision), source_identity=migration.identity(source_vision))
    migration.atomic_json(final / "receipt.json", receipt)
    return final, row, plan, receipt


def test_atomic_commit_recovers_missing_episode_index_without_reconversion(tmp_path):
    final, row, plan, receipt = committed(tmp_path)
    result = migration.convert_one(row, plan, str(tmp_path))
    assert result["resumed"] is True
    path = tmp_path / ".pipeline/episodes/00000000.json"
    assert json.loads(path.read_text()) == receipt["record"]
    assert migration.convert_one(row, plan, str(tmp_path))["resumed"] is True


@pytest.mark.parametrize("damage", ["output", "source", "vision_source", "row", "plan", "record"])
def test_resume_rejects_modified_data_job_inventory_and_index(tmp_path, damage):
    final, row, plan, receipt = committed(tmp_path)
    record = receipt["record"].copy()
    if damage == "output": (final / "vision.parquet").write_bytes(b"changed output")
    elif damage == "source": Path(row["source_path"]).write_bytes(b"changed source")
    elif damage == "vision_source": Path(receipt["vision"]["source_path"]).write_bytes(b"changed source vision")
    elif damage == "row": row["dataset_to_index"] = 3
    elif damage == "plan": plan["model_sha256"] = "changed"
    elif damage == "record": record["split"] = "test"
    with pytest.raises(ValueError): migration.validate_commit(final, receipt, row, plan, record)
