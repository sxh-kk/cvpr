"""5Hz joint history reconstruction and forecasting from the owned SMPL-H archive."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.task_conditioning import apply_task_conditioning
from .representation import quaternion_matrix
from .motion_layout import K12_JOINT_INDICES, SMPLH_REPRE_TYPE, motion_layout
from .normalization import MotionNormalizer
from .smplh_archive import SMPLHParameterArchive, PARAMETER_SCHEMA_VERSION, DINO_COLUMN
from .smplh_representation import build_smplh_window, decode_smplh_motion


class SMPLHMotionDataset(Dataset):
    def __init__(self, *, archive_root=None, data_dir=None, split="train", split_manifest=None,
                 stats_path=None, do_normalization=True, cond_img_feat=True, cond_traj=True,
                 cond_betas=False, window=11, fps=5, forecast_prefix=6, segment_stride=5,
                 sparse_joint_indices=None, repre_type=SMPLH_REPRE_TYPE, img_feat_type="dinov2_reg",
                 row_group_cache_mb=128, parquet_file_cache=8, rotation_basis_path=None,
                 rotation_basis_mode="smplh_model_defined", target_joint_basis="smplh_body22_axes_v1",
                 require_verified_rotations=True, normalization_clip=None,
                 contact_speed_m_s=.2, contact_height_m=.1, min_valid_image_fraction=0.,
                 train_task="fore", require_full_window=True, motion_supervision="joint",
                 allow_unpublished=False, episode_indices=None, vision_profile_root=None,
                 image_profile=None, **legacy):
        if cond_betas or bool(legacy.get("cond_text", False)):
            raise ValueError("SMPL-H beta16 is used only for source FK and audit metadata")
        if repre_type != SMPLH_REPRE_TYPE or img_feat_type != "dinov2_reg":
            raise ValueError("SMPL-H archive requires K12/explicit-hands21 and DINO5")
        if sparse_joint_indices is not None and list(sparse_joint_indices) != list(K12_JOINT_INDICES):
            raise ValueError("SMPL-H245 requires the existing twelve sparse joints in model order")
        if rotation_basis_path or rotation_basis_mode != "smplh_model_defined" or target_joint_basis != "smplh_body22_axes_v1":
            raise ValueError("SMPL-H parameters use the common model-defined body axes without native corrections")
        if type(fps) is not int or fps != 5 or type(window) is not int or window != 11 or type(forecast_prefix) is not int or forecast_prefix != 6:
            raise ValueError("SMPL-H245 main profile is5Hz, complete11 frames and observed prefix6")
        if train_task != "fore" or require_full_window is not True:
            raise ValueError("SMPL-H245 requires forecast labels and complete windows")
        if motion_supervision != "joint":
            raise ValueError("SMPL-H motion supervision must be joint")
        if type(segment_stride) is not int or segment_stride < 1 or split not in ("train", "val", "test"):
            raise ValueError("Invalid SMPL-H segment stride or split")
        self.allow_unpublished = allow_unpublished
        self.diagnostic = bool(allow_unpublished or episode_indices is not None)
        self.archive = SMPLHParameterArchive(archive_root or data_dir or "", cache_mb=row_group_cache_mb,
                                             file_cache=parquet_file_cache, allow_unpublished=allow_unpublished,
                                             vision_profile_root=vision_profile_root, image_profile=image_profile)
        self.root, self.reader = self.archive.root, self.archive.reader
        self.vision_profile = self.archive.vision_cache.metadata if self.archive.vision_cache is not None else None
        self.image_profile = self.archive.vision_cache.image_profile if self.archive.vision_cache is not None else None
        self.archive_schema_version, self.model_sha256 = PARAMETER_SCHEMA_VERSION, self.archive.model_sha256
        if split_manifest and Path(split_manifest).resolve() != self.archive.manifest_path.resolve():
            raise ValueError("SMPL-H training must retain the owned original meta/splits.json")
        self.manifest = self.archive.manifest
        self.split_manifest_path = str(self.archive.manifest_path)
        self.split_fingerprint = self.archive.split_fingerprint
        self.archive_records = self.archive.records()
        selected = None
        if episode_indices is not None:
            selected = list(episode_indices)
            if (not selected or any(type(i) is not int or not 0 <= i < len(self.archive_records) for i in selected)
                    or len(set(selected)) != len(selected)):
                raise ValueError("Diagnostic episode selection must contain unique valid identities")
            selected = set(selected)
        self.rejected_episodes = {}
        self.records = []
        for record in self.archive_records:
            if record["split"] != split or (selected is not None and record["episode_index"] not in selected):
                continue
            meta = record["metadata"]
            eligible = (meta.get("training_eligible") is not False and meta.get("canonical_training_eligible") is not False
                        and not meta.get("quarantine_reasons", []))
            if not eligible:
                self.rejected_episodes[record["source_key"]] = list(meta.get("quarantine_reasons") or ["training_ineligible"])
            else:
                self.records.append(record)
        self.split, self.window, self.fps, self.forecast_prefix = split, window, fps, forecast_prefix
        self.sampling_step, self.segment_stride = 6, segment_stride
        self.train_task, self.require_full_window, self.motion_supervision = train_task, True, motion_supervision
        self.statistics_scope = "all_valid_frames"
        self.ground_estimation_scope = self.image_validity_scope = "observed_prefix"
        self.cond_img_feat, self.cond_traj, self.do_normalization = bool(cond_img_feat), bool(cond_traj), bool(do_normalization)
        self.repre_type, self.motion_feature_dim = SMPLH_REPRE_TYPE, 245
        self.sparse_joint_indices = list(K12_JOINT_INDICES)
        self.joint_indices = list(K12_JOINT_INDICES)
        self.img_feat_type, self.img_feat_dim, self.dataset_name = img_feat_type, 1024, "mixegomotion_smplh"
        self.rotation_basis_mode, self.target_joint_basis = rotation_basis_mode, target_joint_basis
        self.basis_mapping_fingerprint = "none"
        self.contact_speed_m_s, self.contact_height_m = float(contact_speed_m_s), float(contact_height_m)
        if not np.isfinite([self.contact_speed_m_s, self.contact_height_m]).all() or min(self.contact_speed_m_s, self.contact_height_m) <= 0:
            raise ValueError("Contact thresholds must be finite and positive")
        self.min_valid_image_fraction = float(min_valid_image_fraction)
        if not np.isfinite(self.min_valid_image_fraction) or not 0 <= self.min_valid_image_fraction <= 1:
            raise ValueError("MIN_VALID_IMAGE_FRACTION must lie in[0,1]")
        self.seq_names = [r["source_key"] for r in self.records]
        self.motion_data = {name: {"source": r["source"], "num_frames": r["length"], "episode_index": r["episode_index"]}
                            for name, r in zip(self.seq_names, self.records)}
        self._seq_indices = {name: i for i, name in enumerate(self.seq_names)}
        self.idx_to_sidx_fidx, self.rejected_windows = [], {}
        for sequence, record in enumerate(self.records):
            self._index_record(sequence, record)
        if not self.idx_to_sidx_fidx:
            raise ValueError(f"No usable complete SMPL-H windows in {split}; rejected={self.rejected_windows}")
        self.normalizer = MotionNormalizer(stats_path, clip=normalization_clip, allow_diagnostic=self.diagnostic) if self.do_normalization else None
        if self.normalizer:
            self._validate_statistics(self.normalizer.stats)

    def _validate_statistics(self, stats):
        if stats.get("diagnostic", False) and not self.diagnostic:
            raise ValueError("Diagnostic subset statistics cannot be used for formal SMPL-H training")
        expected = {"representation": self.repre_type, "split_fingerprint": self.split_fingerprint,
                    "joint_indices": self.joint_indices,
                    "archive_fingerprint": self.manifest["archive_fingerprint"], "archive_schema_version": self.archive_schema_version,
                    "model_sha256": self.model_sha256, "target_joint_basis": self.target_joint_basis,
                    "rotation_basis_mode": self.rotation_basis_mode, "basis_mapping_fingerprint": self.basis_mapping_fingerprint,
                    "fps": self.fps, "window": self.window, "forecast_prefix": self.forecast_prefix,
                    "segment_stride": self.segment_stride, "sampling_step": self.sampling_step,
                    "train_task": self.train_task, "label_task": self.train_task, "complete_windows_only": True,
                    "motion_supervision": self.motion_supervision, "statistics_scope": self.statistics_scope,
                    "ground_estimation_scope": self.ground_estimation_scope, "image_validity_scope": self.image_validity_scope,
                    "min_valid_image_fraction": self.min_valid_image_fraction,
                    "contact_speed_m_s": self.contact_speed_m_s, "contact_height_m": self.contact_height_m}
        for key, value in expected.items():
            if stats.get(key) != value:
                raise ValueError(f"SMPL-H training statistics {key} mismatch")

    def _assert_archive_current(self, record=None):
        self.archive.assert_current(record)

    def _index_record(self, sequence, record):
        self._assert_archive_current(record)
        selected = np.arange(0, record["length"], self.sampling_step, dtype=np.int64)
        if len(selected) < self.window:
            self.rejected_windows["incomplete_window"] = self.rejected_windows.get("incomplete_window", 0) + 1
            return
        columns = ["observation.state.head_reference_world", "observation.state.head_reference_world_rotation",
                   "observation.valid.head_reference_position", "observation.valid.head_reference_rotation",
                   "observation.valid.head_reference_basis", "estimated.valid.body_position", "estimated.valid.body_rotation",
                   "source.feet_valid", "frame_index", "episode_index", "index", "timestamp"]
        ground = record["metadata"].get("ground", {})
        if not ground.get("valid", False):
            columns.append("source.feet_world")
        if self.min_valid_image_fraction > 0 or self.archive.vision_cache is not None:
            columns.append("observation.valid.dino")
        path = self.archive.data_path(record["episode_index"])
        if self.archive.vision_cache is not None:
            columns.append("observation.source_timestamp")
            _, (parquet, _) = self.reader._file(path)
            if "source.video_frame_index" in parquet.schema_arrow.names:
                columns.append("source.video_frame_index")
        arrays = self.reader.read(path, selected, columns)
        if (not np.array_equal(arrays["frame_index"], selected) or not np.all(arrays["episode_index"] == record["episode_index"])
                or not np.array_equal(arrays["index"], record["dataset_from_index"] + selected)
                or not np.array_equal(arrays["timestamp"], (selected / 30).astype(np.float32))):
            raise ValueError("SMPL-H index frame/global-index/time join differs")
        if self.archive.vision_cache is not None:
            arrays["observation.valid.dino"] = self.archive.read_image_validity(record, selected, expected=arrays)
        q, p = arrays["observation.state.head_reference_world_rotation"], arrays["observation.state.head_reference_world"]
        head_valid = (arrays["observation.valid.head_reference_position"].astype(bool)
                      & arrays["observation.valid.head_reference_rotation"].astype(bool)
                      & arrays["observation.valid.head_reference_basis"].astype(bool)
                      & np.isfinite(p).all(-1) & np.isfinite(q).all(-1) & (np.linalg.norm(q, axis=-1) > 1e-8))
        heading_valid = np.linalg.norm(quaternion_matrix(q)[:, :2, 0], axis=-1) > 1e-6
        body_valid = (arrays["estimated.valid.body_position"] | arrays["estimated.valid.body_rotation"])[:, self.sparse_joint_indices]
        feet_valid = arrays["source.feet_valid"].astype(bool)
        if "source.feet_world" in arrays:
            feet_valid &= np.isfinite(arrays["source.feet_world"].reshape(-1, 2, 3)).all(-1)
        for start in range(0, len(selected), self.segment_stride):
            end, observed_end = start + self.window, start + self.forecast_prefix
            reason = None
            if end > len(selected):
                reason = "incomplete_window"
            elif not record["metadata"].get("coordinate_verified", False) or not record["metadata"].get("head_reference", {}).get("basis_verified", False):
                reason = "unverified_camera_world_axes"
            elif not (head_valid[start] and heading_valid[start]):
                reason = "invalid_initial_camera"
            elif not (body_valid[observed_end:end] & (head_valid & heading_valid)[observed_end:end, None]).any():
                reason = "no_valid_future_body_labels"
            elif ground.get("valid", False) and not np.isfinite(ground.get("height_m", np.nan)):
                reason = "invalid_floor_height"
            elif not ground.get("valid", False) and not (ground.get("mode") == "estimate"
                    and ground.get("window_fallback", {"mode": "mean_valid_feet"}).get("mode") == "mean_valid_feet"
                    and feet_valid[start:observed_end].any()):
                reason = "no_usable_prefix_floor"
            elif self.min_valid_image_fraction > 0 and arrays["observation.valid.dino"][start:observed_end].mean() < self.min_valid_image_fraction:
                reason = "insufficient_real_DINO"
            if reason:
                self.rejected_windows[reason] = self.rejected_windows.get(reason, 0) + 1
            else:
                self.idx_to_sidx_fidx.append((sequence, int(selected[start])))
        self._assert_archive_current(record)

    def __len__(self):
        return len(self.idx_to_sidx_fidx)

    def normalize(self, value, kind, valid=None):
        result = self.normalizer.normalize(value, kind, valid) if self.normalizer else torch.as_tensor(value).float()
        return result.masked_fill(~torch.as_tensor(valid).bool(), 0) if valid is not None else result

    def denormalize(self, value, kind="motion"):
        return self.normalizer.denormalize(value, kind) if self.normalizer else torch.as_tensor(value).float()

    def __getitem__(self, index):
        return self.get_sample(index)

    def get_sample(self, index, task=None):
        sequence, start = self.idx_to_sidx_fidx[int(index)]
        return self.get_from_seq_and_st(self.seq_names[sequence], start, idx=int(index), task=task)

    def get_from_seq_and_st(self, seq_name, st, idx=-1, task=None):
        if task not in (None, "fore"):
            raise ValueError("SMPL-H245 supports forecasting only")
        if seq_name not in self._seq_indices:
            raise KeyError(f"Unknown SMPL-H sequence {seq_name}")
        sequence = self._seq_indices[seq_name]
        record = self.records[sequence]
        arrays, metadata = self.archive.read_window(record["episode_index"], start_frame=st, num_frames=self.window,
                                                   fps=self.fps, forecast_prefix=self.forecast_prefix, load_dino=self.cond_img_feat)
        representation = build_smplh_window(arrays, metadata, task="fore", forecast_prefix=self.forecast_prefix,
                                             fps=self.fps, contact_speed_m_s=self.contact_speed_m_s, contact_height_m=self.contact_height_m)
        if not representation.motion_valid[self.forecast_prefix:, :108].any():
            raise ValueError("SMPL-H forecast has no valid future body labels under prefix-floor policy")
        motion_valid = torch.from_numpy(representation.motion_valid.copy()).bool()
        trajectory_valid = torch.from_numpy(representation.trajectory_valid.copy()).bool()
        motion = self.normalize(representation.motion, "motion", motion_valid)
        trajectory = self.normalize(representation.trajectory, "traj", trajectory_valid)
        y = {"traj": trajectory, "trajectory_valid": trajectory_valid.clone(), "feature_valid_mask": motion_valid,
             "valid_frames": torch.ones(self.window, dtype=torch.bool), "padding_mask": torch.ones(self.window, dtype=torch.bool)}
        if self.cond_img_feat:
            y.update(img_embs=torch.from_numpy(arrays[DINO_COLUMN].reshape(self.window, 5, 1024).copy()),
                     valid_img_embs=torch.from_numpy(arrays["observation.valid.dino"].copy()))
        else:
            y.update(img_embs=torch.zeros(self.window, 5, 1024), valid_img_embs=torch.zeros(self.window, dtype=torch.bool))
        if not self.cond_traj:
            y["trajectory_valid"] = torch.zeros_like(trajectory_valid)
        details = representation.metadata["training_window"]
        selected = arrays["frame_index"]
        details.update(source_frame_indices=selected.tolist(), diagnostic=self.diagnostic,
                       archive_schema_version=self.archive_schema_version, model_sha256=self.model_sha256)
        floor = np.tile(np.eye(4), (self.window, 1, 1))
        floor[:, :3, :3], floor[:, :3, 3] = details["floor_rotation"], details["floor_position"]
        misc = {"motion": motion, "traj": trajectory, "feature_valid_mask": motion_valid, "trajectory_valid": trajectory_valid,
                "seq_name": seq_name, "source": record["source"], "dataset": self.dataset_name, "idx": int(idx),
                "start_idx": int(st), "end_idx": int(selected[-1]), "episode_index": record["episode_index"],
                "num_real_frames": self.window, "fps": self.fps,
                "world_from_window": torch.tensor(details["archive_world_from_window"], dtype=torch.float32),
                "floor_transform": torch.tensor(floor, dtype=torch.float32), "floor_height": float(details["floor_height_m"]),
                "ground_fallback_used": bool(representation.metadata["ground"].get("window_fallback_used", False)),
                "metadata_json": json.dumps(representation.metadata), "motion_nonfinite_mask": torch.zeros_like(motion_valid),
                "traj_nonfinite_mask": torch.zeros(self.window, dtype=torch.bool), "discard_terrain": False}
        # Keep label provenance separate from model conditions and pseudo-truth.
        for key in ("estimated.valid.body_position", "estimated.valid.body_rotation", "source.observed.body_position",
                    "source.observed.body_rotation", "source.interpolated.body_position", "source.interpolated.body_rotation",
                    "estimated.completed.body_position", "estimated.completed.body_rotation", "quality.position_confidence",
                    "quality.rotation_confidence", "quality.observation_residual_m", "quality.observation_residual_valid"):
            misc[key] = torch.from_numpy(arrays[key].copy())
        for key in ("observation.state.hand_left_world", "observation.state.hand_right_world",
                    "observation.valid.hand_left_position", "observation.valid.hand_right_position",
                    "observation.interpolated.hand_left_position", "observation.interpolated.hand_right_position",
                    "source.feet_world", "source.feet_valid", "source.feet_interpolated",
                    "smplh.betas", "observation.state.body_world", "observation.state.body_world_rotation"):
            misc[key] = torch.from_numpy(arrays[key].copy())
        y = apply_task_conditioning(y, task or self.train_task, forecast_prefix=self.forecast_prefix,
                                   supervise_history=self.motion_supervision == "joint")
        self._assert_archive_current(record)
        return {"y": y, "misc": misc, "pred": {}}

    def process_sample_for_task(self, ret, task):
        return self.get_from_seq_and_st(ret["misc"]["seq_name"], int(ret["misc"]["start_idx"]), idx=int(ret["misc"]["idx"]), task=task)

    def ret_to_full_sequence(self, ret):
        has_prediction = "motion" in ret.get("pred", {})
        motion = ret["pred"]["motion"] if has_prediction else ret["misc"]["motion"]
        layout = motion_layout(self.repre_type)
        valid_delta = ret["misc"]["feature_valid_mask"][:, layout.global_slice].all(-1)
        options = {}
        if has_prediction:
            options.update(start_frame=self.forecast_prefix,
                           initial_floor_transform=ret["misc"]["floor_transform"][self.forecast_prefix - 1])
        return decode_smplh_motion(self.denormalize(motion),
                                  window_to_world=ret["misc"]["world_from_window"], floor_height=ret["misc"]["floor_height"],
                                  delta_valid=valid_delta, **options)
