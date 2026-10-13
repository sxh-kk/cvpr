"""Upgrade pending v3 DINO placeholders without recomputing motion or video."""

import os
from datetime import datetime, timezone
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from .schema import (SCHEMA_VERSION, LEGACY_SCHEMA_VERSION, DINO_FLAT_DIM,
                     DINO_TOKEN_DIM, feature_specs)
from .storage import Archive, archive_provenance, atomic_json, read_json

FEATURE = "observation.features.dino"
VALID = "observation.valid.dino"


def _pending_width(path, record):
    """Inspect all placeholders before any dataset mutation, including on resume."""
    if record["dino"]["state"] != "pending":
        raise ValueError("upgrade-dino supports pending features only; completed CLS features need re-extraction in a new archive")
    source = pq.ParquetFile(path)
    field_type = source.schema_arrow.field(FEATURE).type
    if (not pa.types.is_fixed_size_list(field_type) or field_type.value_type != pa.float32()
            or field_type.list_size not in {DINO_TOKEN_DIM, DINO_FLAT_DIM}):
        raise ValueError(f"Unexpected pending DINO dtype/shape: {path}")
    if source.metadata.num_rows != record["length"]:
        raise ValueError(f"Episode length mismatch: {path}")
    columns = [FEATURE, VALID, "observation.video_timestamp", "observation.video_time_error"]
    for batch in source.iter_batches(batch_size=3000, columns=columns):
        feature = batch.column(0)
        if feature.null_count or feature.values.null_count or np.any(feature.values.to_numpy() != 0):
            raise ValueError(f"Pending DINO contains nonzero/null features: {path}")
        for index in range(1, len(columns)):
            values = batch.column(index)
            if values.null_count or np.any(values.to_numpy(zero_copy_only=False) != 0):
                raise ValueError(f"Pending DINO contains valid frames or video timestamps: {path}")
    return field_type.list_size


def _widen_pending_file(path):
    """Stream a single file; Arrow keeps every non-DINO column unchanged."""
    source = pq.ParquetFile(path)
    index = source.schema_arrow.get_field_index(FEATURE)
    schema = source.schema_arrow.set(index, pa.field(FEATURE, pa.list_(pa.float32(), DINO_FLAT_DIM)))
    tmp = path.with_suffix(".dino-upgrade.tmp")
    try:
        with pq.ParquetWriter(tmp, schema, compression="zstd") as writer:
            for batch in source.iter_batches(batch_size=3000):
                zero = pa.array(np.zeros(len(batch)*DINO_FLAT_DIM, dtype=np.float32))
                feature = pa.FixedSizeListArray.from_arrays(zero, DINO_FLAT_DIM)
                table = pa.Table.from_batches([batch]).set_column(index, schema.field(index), feature)
                writer.write_table(table, row_group_size=3000)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def upgrade_dino(root, progress=None):
    """Locked, resumable v3→v4 migration; completed features are never invented."""
    archive = Archive(root, _allow_upgrade=True)
    journal_path = archive.root/".pipeline/dino_upgrade.json"
    with archive.lock():
        provenance_path = archive.root/".pipeline/config.json"
        provenance = read_json(provenance_path)
        journal = read_json(journal_path) if journal_path.exists() else None
        version = provenance.get("schema_version")
        if version == SCHEMA_VERSION and (not journal or journal["state"] == "complete"):
            return {"state": "complete", "already_current": True, "schema_version": SCHEMA_VERSION}
        if version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
            raise ValueError("upgrade-dino requires a camera-floor v3 archive")
        if journal and (journal.get("source_schema") != LEGACY_SCHEMA_VERSION
                        or journal.get("target_schema") != SCHEMA_VERSION):
            raise ValueError("Unexpected DINO migration journal")
        records = archive.records()
        if not records:
            raise ValueError("Cannot upgrade an empty archive")
        widths = []
        for record in records:
            widths.append(_pending_width(archive.data_path(record["episode_index"]), record))
            if not journal and widths[-1] != DINO_TOKEN_DIM:
                raise ValueError("Mixed/unrecognized DINO layout without an upgrade journal")
        if journal is None:
            journal = {"state": "in_progress", "source_schema": LEGACY_SCHEMA_VERSION,
                       "target_schema": SCHEMA_VERSION, "started_at": datetime.now(timezone.utc).isoformat(),
                       "original_provenance": provenance, "total_episodes": len(records)}
        journal["state"] = "in_progress"
        atomic_json(journal_path, journal)
        for record, width in zip(records, widths):
            if width == DINO_TOKEN_DIM:
                _widen_pending_file(archive.data_path(record["episode_index"]))
            journal["completed_episodes"] = record["episode_index"]+1
            atomic_json(journal_path, journal)
            if progress:
                progress(dict(journal))
        # Readers are blocked by the journal until both shared metadata and
        # provenance agree with every file. A crash here can safely resume.
        archive.schema_version = SCHEMA_VERSION
        archive.specs = feature_specs()
        archive.finalize()
        updated = archive_provenance(journal["original_provenance"]["config"])
        updated["motion_conversion_provenance"] = journal["original_provenance"]
        atomic_json(provenance_path, updated)
        journal.update({"state": "complete", "completed_at": datetime.now(timezone.utc).isoformat(),
                        "frames": sum(r["length"] for r in records)})
        atomic_json(journal_path, journal)
        return {"state": "complete", "episodes": len(records), "frames": journal["frames"],
                "schema_version": SCHEMA_VERSION, "dino": "pending"}
