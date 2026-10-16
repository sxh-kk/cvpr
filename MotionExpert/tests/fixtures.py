"""Small SMPL-H forecasting fixtures independent of archives and SMPL assets."""

import torch

from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.dataset.normalization import MotionNormalizer, STATS_FORMAT
from MotionExpert.dataset.motion_layout import SMPLH_REPRE_TYPE as REPRE_TYPE
from MotionExpert.config import SMPLH_SCHEMA
from MotionExpert.motion_expert import MotionExpert
from MotionExpert.mydiffusion.flow_matching import FlowMatching


def smplh_stats(cfg=None):
    cfg = smplh_motion_expert_config() if cfg is None else cfg
    data = cfg.DATA
    task = "fore"
    supervision = "joint"
    scope = "observed_prefix"
    return {
        "format": STATS_FORMAT, "representation": REPRE_TYPE,
        "split": "train", "fps": data.FPS, "split_fingerprint": "unit-test-train-only",
        "archive_fingerprint": "unit-test-archive", "basis_mapping_fingerprint": None,
        "window": data.WINDOW, "forecast_prefix": data.FORECAST_PREFIX,
        "segment_stride": data.SEGMENT_STRIDE, "stats_schema_version": 4,
        "sampling_step": 30 // data.FPS, "train_task": task, "label_task": task,
        "complete_windows_only": getattr(data, "REQUIRE_FULL_WINDOW", False),
        "ground_estimation_scope": scope, "image_validity_scope": scope,
        "min_valid_image_fraction": getattr(data, "MIN_VALID_IMAGE_FRACTION", 0.),
        "motion_supervision": supervision,
        "statistics_scope": "all_valid_frames",
        "target_joint_basis": "smplh_body22_axes_v1", "rotation_basis_mode": "smplh_model_defined",
        "archive_schema_version": SMPLH_SCHEMA, "model_sha256": cfg.DATA.MODEL_SHA256,
        "joint_indices": list(cfg.SPARSE_JOINTS.INDICES),
        "contact_speed_m_s": .2, "contact_height_m": .1,
        "motion_mean": torch.zeros(245), "motion_std": torch.ones(245),
        "traj_mean": torch.zeros(18), "traj_std": torch.ones(18),
        "normalization_clip": None,
    }


def conditioning(batch=2, frames=6):
    generator = torch.Generator().manual_seed(3)
    return {"traj": torch.randn(batch, frames, 18, generator=generator),
            "img_embs": torch.randn(batch, frames, 5, 1024, generator=generator),
            "valid_frames": torch.ones(batch, frames, dtype=torch.bool),
            "valid_img_embs": torch.ones(batch, frames, dtype=torch.bool),
            "trajectory_valid": torch.ones(batch, frames, 18, dtype=torch.bool)}


class TinyFeatureModel(torch.nn.Module):
    """Deterministic condition-sensitive model to test the public flow API."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.level = torch.nn.Parameter(torch.tensor(.2))
        self.calls = 0
        self.last_y = None

    def forward(self, x, t, y, cond_scale=None, return_features=False, return_hidden=False):
        self.calls += 1
        self.last_y = y
        trajectory = y["traj"].masked_fill(y["traj_mask"].bool()[..., None], 0)
        image = y["img_embs"].masked_fill(y["img_mask"].bool()[..., None, None], 0)
        summary = trajectory.mean(-1) + image.mean((-1, -2))
        hidden = torch.nn.functional.pad(x, (0, 768-245)) + summary[..., None]
        prediction = self.level.expand_as(x)
        global_hidden = hidden.mean(1)
        if return_features:
            return prediction, hidden, global_hidden
        if return_hidden:
            return prediction, hidden
        return prediction


def tiny_expert():
    expert = MotionExpert.__new__(MotionExpert)
    expert.cfg = smplh_motion_expert_config()
    expert.device = torch.device("cpu")
    expert.dtype = torch.float32
    expert.model = TinyFeatureModel(expert.cfg).eval()
    expert.normalizer = MotionNormalizer(smplh_stats(expert.cfg))
    expert.flow = FlowMatching()
    expert.weights_path = "/unit-test/smplh.pt"
    expert.weights_kind = "model"
    return expert


from types import SimpleNamespace
from torch import nn

class InspectDecoder(nn.Module):
    """Exercise real condition assembly without allocating twelve large MoEs."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.ff = SimpleNamespace(last_aux_loss=None)
        self.call = None

    def forward(self, **kwargs):
        self.call = kwargs
        context, mask = kwargs['context'], kwargs['context_mask'][:, 0, 0]
        pooled = (context * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return kwargs['x'] + pooled[:, None] * 0.001


import json
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from MotionExpert.dataset import smplh_archive as archive_module
from MotionExpert.dataset.smplh_archive import (SMPLHParameterArchive, MOTION_SPECS, PARAMETER_SCHEMA_VERSION, SMPLHBodyModel, DINO_COLUMN, PARENTS, canonical_json, file_sha256)


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, sort_keys=True))


def _table(path, arrays, *, parameter=False):
    columns = {key: (pa.array(value) if value.ndim == 1 else
                    pa.FixedSizeListArray.from_arrays(pa.array(value.reshape(-1)), value.shape[1]))
               for key, value in arrays.items()}
    table = pa.table(columns)
    if parameter:
        table = table.replace_schema_metadata({b"parameter_schema_version": PARAMETER_SCHEMA_VERSION.encode()})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, row_group_size=13, compression="zstd")


def parameter_arrays(count=91, *, episode=0):
    arrays = {key: np.zeros((count,) if width == 1 else (count, width), dtype=dtype)
              for key, (dtype, width) in MOTION_SPECS.items()}
    frames = np.arange(count)
    arrays.update(frame_index=frames.astype(np.int64), episode_index=np.full(count, episode, np.int64),
                  index=frames.astype(np.int64) + episode * count, timestamp=(frames / 30).astype(np.float32))
    arrays["observation.source_timestamp"] = frames / 30
    arrays["observation.source_frame_left"] = frames.astype(np.int64)
    arrays["observation.source_frame_right"] = frames.astype(np.int64)
    arrays["observation.video_timestamp"] = frames / 30 + .001
    arrays["observation.video_time_error"][:] = .001
    arrays["smplh.trans"][:] = [2, 3, .5]
    arrays["smplh.trans"][:, 0] += frames * .002
    arrays["smplh.root_orient"][:, 2] = .2 + frames * .01
    arrays["smplh.pose_body"][:, 8] = frames * .003
    arrays["smplh.betas"][:, 0] = .2
    arrays["smplh.betas"][:, 15] = frames * .004
    arrays["source.feet_world"][:, [2, 5]] = .52
    arrays["source.feet_world"][:, [0, 3]] = 2 + frames[:, None] * .002
    arrays["source.feet_valid"][:] = True
    for component in ("position", "rotation"):
        arrays[f"estimated.valid.body_{component}"][:] = True
        arrays[f"source.observed.body_{component}"][:] = True
        arrays[f"quality.{component}_confidence"][:] = .6
    for key in arrays:
        if key.startswith("observation.valid."):
            arrays[key][:] = True
    arrays["observation.valid.floor_verified"][:] = False
    for name in ("head", "camera", "head_reference"):
        arrays[f"observation.state.{name}_world"][:] = [5, 4, 1.8]
        arrays[f"observation.state.{name}_world"][:, 0] += frames * .002
        yaw = .3 + frames * .01
        arrays[f"observation.state.{name}_world_rotation"] = np.stack(
            [np.cos(yaw / 2), np.zeros(count), np.zeros(count), np.sin(yaw / 2)], -1).astype(np.float32)
    for side in ("left", "right"):
        arrays[f"observation.state.hand_{side}_world"] = np.arange(count * 63, dtype=np.float32).reshape(count, 63) * .001
    arrays["observation.environment.floor_height"][:] = .5
    arrays["observation.environment.floor_normal"][:, 2] = 1
    return arrays


@pytest.fixture
def owned_archive(tmp_path, monkeypatch):
    root = tmp_path / "parameters"
    model_path = root / "weights/smplh/model.npz"
    model_path.parent.mkdir(parents=True)
    parents = PARENTS.tolist()
    for wrist in (20, 21):
        for _ in range(5):
            first = len(parents)
            parents.extend([wrist, first, first + 1])
    template = np.zeros((6890, 3), np.float32)
    template[0] = [.3, -.2, .4]
    for joint in range(1, 52):
        template[joint] = template[parents[joint]] + [.01, .02, .03]
    regressor = np.zeros((52, 6890), np.float32)
    regressor[np.arange(52), np.arange(52)] = 1
    shapes = np.zeros((6890, 3, 16), np.float32)
    for beta in range(16):
        shapes[:52, beta % 3, beta] = (.001 + np.arange(52) * .0001) * (beta + 1)
    np.savez_compressed(model_path, v_template=template, shapedirs=shapes,
                        J_regressor=regressor, kintree_table=np.array([parents, np.arange(52)]))
    model_sha = file_sha256(model_path)
    monkeypatch.setattr(archive_module, "MODEL_SHA256", model_sha)
    info = {"schema_version": PARAMETER_SCHEMA_VERSION, "fps": 30, "requires_old_dense_archive": False,
            "model": {"path": "weights/smplh/model.npz", "model_sha256": model_sha,
                      "body_joint_count": 22, "joint_count": 52, "shape_basis_count": 16},
            "dino": {"shape": [5, 1024], "storage_dtype": "float32", "flatten_order": "token_major"}}
    _json(root / "meta/info.json", info)
    records = []
    for idx in range(4):
        source = "egoexo4d_hm" if idx in (1, 2) else "egosuite"
        split = "val" if idx == 3 else "train"
        arrays = parameter_arrays(episode=idx)
        # Estimated completion stays separate from source observations.
        if idx == 0:
            arrays["source.observed.body_rotation"][:, 13] = False
            arrays["estimated.completed.body_rotation"][:, 13] = True
            arrays["observation.valid.hand_left_position"][:, 7] = False
        mp, vp = root / f"episodes/{idx:08d}/motion.parquet", root / f"episodes/{idx:08d}/vision.parquet"
        _table(mp, arrays, parameter=True)
        vision = {key: arrays[key].copy() for key in ("frame_index", "episode_index", "timestamp", "observation.source_timestamp",
                                                           "observation.valid.dino", "observation.video_timestamp", "observation.video_time_error")}
        # Published Suite promotes its source float32 grid; HM keeps float64.
        vision["timestamp"] = arrays["timestamp"].astype(np.float64) if source == "egosuite" else np.arange(91) / 30
        vision[DINO_COLUMN] = np.repeat(np.arange(91, dtype=np.float32)[:, None], 5120, axis=1)
        vision[DINO_COLUMN][36:] = np.nan  # Future visuals must never enter conditions.
        _table(vp, vision)
        metadata = {"coordinate_verified": idx != 2, "head_reference": {"basis_verified": True},
                    "rotation_basis_verified": source != "egosuite", "target_joint_basis": "smplh_body22_axes_v1",
                    "label_kind": "model_fitted_estimate", "body_basis_correction": None,
                    "ground": {"height_m": .5, "valid": True, "verified": False, "mode": "estimate",
                               "window_fallback": {"mode": "mean_valid_feet", "offset_m": .02}},
                    "training_eligible": idx != 2, "quarantine_reasons": ["camera_world_mismatch"] if idx == 2 else [],
                    "source_data": "/deleted-dense-source/motion.parquet"}
        if source == "egosuite":
            metadata["target_model_axes_defined"] = True
        record = {"episode_index": idx, "source": source, "source_key": f"{source}:{idx}", "source_uid": str(idx),
                  "length": 91, "dataset_from_index": idx * 91, "dataset_to_index": (idx + 1) * 91,
                  "split": split, "data_path": str(mp.relative_to(root)), "vision_path": str(vp.relative_to(root)),
                  "motion_sha256": file_sha256(mp), "vision_sha256": file_sha256(vp), "metadata": metadata,
                  "split_group_ids": [f"participant:{idx}", f"capture:{idx}"], "dino": {"state": "complete"}}
        _json(root / f".pipeline/episodes/{idx:08d}.json", record)
        receipt = {"schema_version": PARAMETER_SCHEMA_VERSION, "record": record, "completed_at": "2026-10-12",
                   "conversion_binding": "0" * 64, "motion": {"sha256": record["motion_sha256"]},
                   "vision": {"sha256": record["vision_sha256"]}, "output_identities": {
                       role: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                       for role, p in (("motion", mp), ("vision", vp))}}
        _json(root / f"episodes/{idx:08d}/receipt.json", receipt)
        records.append(record)
    splits = {name: [r["source_key"] for r in records if r["split"] == name] for name in ("train", "val", "test")}
    _json(root / "meta/splits.json", {"splits": splits, "split_group_ids": {g: r["split"] for r in records for g in r["split_group_ids"]}})
    fields = ("episode_index", "source", "source_key", "source_uid", "length", "dataset_from_index", "dataset_to_index", "split", "data_path", "vision_path")
    pq.write_table(pa.table({k: [r[k] for r in records] for k in fields}), root / "meta/episodes.parquet")
    _json(root / ".pipeline/dataset-state.json", {"state": "accepted", "independent_acceptance_pending": False,
                                                 "schema_version": PARAMETER_SCHEMA_VERSION, "episodes": 4, "frames": 364})
    return root


def rewrite_episode(root, index, *, motion_mutation=None, vision_mutation=None, metadata_mutation=None):
    record_path = root / f".pipeline/episodes/{index:08d}.json"
    receipt_path = root / f"episodes/{index:08d}/receipt.json"
    record, receipt = json.loads(record_path.read_text()), json.loads(receipt_path.read_text())
    if metadata_mutation:
        metadata_mutation(record["metadata"])
        original_motion_mutation = motion_mutation
        def sync_motion_metadata(arrays):
            if original_motion_mutation:
                original_motion_mutation(arrays)
            ground = record["metadata"]["ground"]
            arrays["observation.environment.floor_height"][:] = np.float32(ground["height_m"])
            arrays["observation.valid.floor"][:] = ground["valid"]
            arrays["observation.valid.floor_verified"][:] = ground["valid"] and ground.get("verified", False)
            arrays["observation.valid.head_reference_basis"][:] = record["metadata"]["head_reference"]["basis_verified"]
        motion_mutation = sync_motion_metadata
    for role, mutation in (("motion", motion_mutation), ("vision", vision_mutation)):
        if mutation is None:
            continue
        path = root / record["data_path" if role == "motion" else "vision_path"]
        table = pq.read_table(path)
        arrays = {}
        for key in table.column_names:
            column = table[key].combine_chunks()
            arrays[key] = (column.values.to_numpy(zero_copy_only=False).reshape(len(table), column.type.list_size).copy()
                           if pa.types.is_fixed_size_list(column.type) else column.to_numpy(zero_copy_only=False).copy())
        mutation(arrays)
        _table(path, arrays, parameter=role == "motion")
        record[f"{role}_sha256"] = file_sha256(path)
        receipt[role]["sha256"] = record[f"{role}_sha256"]
        receipt["output_identities"][role] = {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
    receipt["record"] = record
    _json(record_path, record); _json(receipt_path, receipt)
