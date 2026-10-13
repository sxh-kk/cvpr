"""Lazy, bounded-memory reader for immutable camera-floor dense22 archives."""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl

from ..utils.task_conditioning import apply_task_conditioning, TASK_TO_ID
from ..utils.torch_utils import careful_collate_fn
from .dense_representation import (MOTION_DIM, REPRE_TYPE, BODY_WORLD_6D_COLUMN,
                                  BODY_WORLD_6D_MASKS, build_dense_window, quaternion_matrix,
                                  body_world_components)
from .normalization import MotionNormalizer
from .split_manifest import (V5_SCHEMA, assert_archive_ready, motion_representation_identity,
                             read_records, load_split_manifest)

MOTION_COLUMNS = ["timestamp"]
for modality, count in (("body",22), ("head_reference",1), ("hand_left",21), ("hand_right",21)):
    MOTION_COLUMNS.extend([f"observation.state.{modality}_world", f"observation.valid.{modality}_position"])
    if modality in ("body", "head_reference"):
        MOTION_COLUMNS.extend([f"observation.state.{modality}_world_rotation", f"observation.valid.{modality}_rotation"])
IMAGE_COLUMNS = ["observation.features.dino", "observation.valid.dino"]
V5_MOTION_COLUMNS = [key for key in MOTION_COLUMNS if key not in {
    "observation.state.body_world", "observation.state.body_world_rotation",
    "observation.valid.body_position", "observation.valid.body_rotation",
}] + [BODY_WORLD_6D_COLUMN, *BODY_WORLD_6D_MASKS.values()]


class ProjectedParquetReader:
    """Cache only requested row groups/columns; each DataLoader worker owns a cache."""
    def __init__(self, *, cache_mb=128, file_cache=8):
        if cache_mb < 0 or file_cache < 1: raise ValueError("Invalid bounded Parquet cache settings")
        self.max_bytes = int(cache_mb*1024**2)
        self.max_files = int(file_cache)
        self._pid = os.getpid()
        self.files, self.groups = OrderedDict(), OrderedDict()
        self.cached_bytes = 0
        self.read_log = []

    def __getstate__(self):
        value = dict(self.__dict__)
        value.update(files=OrderedDict(), groups=OrderedDict(), cached_bytes=0, read_log=[])
        return value

    def _file(self, path):
        import pyarrow.parquet as pq
        if self._pid != os.getpid():
            self.files.clear(); self.groups.clear(); self.cached_bytes = 0; self.read_log.clear()
            self._pid = os.getpid()
        path = Path(path)
        stat = path.stat()
        key = (str(path), stat.st_mtime_ns, stat.st_size)
        if key in self.files:
            self.files.move_to_end(key)
            return key, self.files[key]
        value = pq.ParquetFile(path)
        offsets = np.cumsum([0]+[value.metadata.row_group(i).num_rows for i in range(value.num_row_groups)])
        self.files[key] = (value, offsets)
        while len(self.files) > self.max_files: self.files.popitem(last=False)
        return key, self.files[key]

    def read(self, path, frame_indices, columns):
        file_key, (parquet, offsets) = self._file(path)
        indices = np.asarray(frame_indices, dtype=np.int64)
        if indices.ndim != 1 or len(indices) == 0 or indices.min() < 0 or indices.max() >= offsets[-1]:
            raise ValueError("Selected archive frame indices outside episode")
        columns = tuple(columns)
        missing = set(columns)-set(parquet.schema_arrow.names)
        if missing: raise ValueError(f"Missing archive columns: {sorted(missing)}")
        selected_groups = np.searchsorted(offsets, indices, side="right")-1
        result = {}
        for group in np.unique(selected_groups):
            key = (file_key, int(group), columns)
            if key in self.groups:
                arrays = self.groups[key]; self.groups.move_to_end(key)
            else:
                table = parquet.read_row_group(int(group), columns=list(columns))
                arrays = {}
                for name in columns:
                    array = table[name].combine_chunks()
                    if array.null_count: raise ValueError(f"Null values in {name}")
                    if hasattr(array, "values"):
                        if array.values.null_count: raise ValueError(f"Null list elements in {name}")
                        width = array.type.list_size
                        value = array.values.to_numpy(zero_copy_only=False).reshape(len(array),width)
                    else: value = array.to_numpy(zero_copy_only=False)
                    arrays[name] = value
                size = sum(a.nbytes for a in arrays.values())
                self.read_log.append((str(path), int(group), columns))
                # Diagnostic log is bounded as well.
                del self.read_log[:-64]
                if size <= self.max_bytes:
                    while self.groups and self.cached_bytes+size > self.max_bytes:
                        _, old = self.groups.popitem(last=False)
                        self.cached_bytes -= sum(a.nbytes for a in old.values())
                    self.groups[key] = arrays; self.cached_bytes += size
            locations = np.flatnonzero(selected_groups == group)
            local = indices[locations]-offsets[group]
            for name, values in arrays.items():
                if name not in result: result[name] = np.empty((len(indices), *values.shape[1:]), dtype=values.dtype)
                result[name][locations] = values[local]
        return result


def _load_basis_mapping(path):
    if not path: return None
    value = json.loads(Path(path).read_text())
    if value.get("format") != "motionexpert-joint-basis-v1" or not value.get("target_basis") or not isinstance(value.get("sources"),dict):
        raise ValueError("Invalid documented source joint-basis mapping")
    for source, mapping in value["sources"].items():
        if mapping.get("verified") is not True or not mapping.get("provenance") or not mapping.get("source_rotation_semantics"):
            raise ValueError(f"Basis mapping {source} requires verified=true, provenance and source_rotation_semantics")
        matrices = np.asarray(mapping.get("corrections"),dtype=float)
        if matrices.shape != (22,3,3) or not np.isfinite(matrices).all() or not np.allclose(matrices.transpose(0,2,1)@matrices,np.eye(3),atol=1e-5) or not np.allclose(np.linalg.det(matrices),1,atol=1e-5):
            raise ValueError(f"Basis mapping {source} requires 22 proper rotation matrices")
    return value


class DenseMotionDataset(Dataset):
    def __init__(self, *, archive_root=None, data_dir=None, split="train", split_manifest=None,
                 stats_path=None, do_normalization=True, cond_img_feat=True, cond_traj=True,
                 cond_betas=False, window=120, fps=15, forecast_prefix=30, segment_stride=30,
                 sparse_joint_indices=None, repre_type=REPRE_TYPE, img_feat_type="dinov2_reg",
                 row_group_cache_mb=128, parquet_file_cache=8, rotation_basis_path=None,
                 rotation_basis_mode="unified", target_joint_basis="smpl22_global_joint_axes",
                 require_verified_rotations=True, normalization_clip=None,
                 contact_speed_m_s=.2, contact_height_m=.1, min_valid_image_fraction=0.,
                 train_task="recon", require_full_window=False, motion_supervision=None, **legacy):
        if cond_betas or bool(legacy.get("cond_text",False)):
            raise ValueError("Dense245 does not use SMPL betas, hand PCA or text conditions")
        if repre_type != REPRE_TYPE or img_feat_type != "dinov2_reg":
            raise ValueError("Dense archive requires dense12_rot6d_hands21_contacts_v1 and dinov2_reg")
        from ..config import K12_JOINTS
        if sparse_joint_indices is not None and list(sparse_joint_indices) != K12_JOINTS:
            raise ValueError("Dense12 joint ordering must match MotionExpert K12_JOINTS")
        if type(fps) is not int or fps < 1 or 30 % fps:
            raise ValueError("Training FPS must be a positive integer divisor of native30Hz")
        if type(window) is not int or window < 2 or type(segment_stride) is not int or segment_stride < 1:
            raise ValueError("Invalid window or stride")
        if type(forecast_prefix) is not int or not 0 < forecast_prefix < window: raise ValueError("Forecast prefix must leave a future region")
        if train_task is None:
            train_task = "recon"  #Legacy configs used model dropout rather than a fixed task.
        if train_task not in ("recon", "fore"):
            raise ValueError("TRAIN_TASK must be recon or fore")
        if motion_supervision is None and train_task == "fore":
            motion_supervision = "joint" if fps == 5 else "future_only"
        if motion_supervision not in (None, "joint", "future_only"):
            raise ValueError("MOTION_SUPERVISION must be joint, future_only or null for legacy data")
        if motion_supervision == "joint" and train_task != "fore":
            raise ValueError("Joint history/future supervision requires TRAIN_TASK=fore")
        if type(require_full_window) is not bool:
            raise ValueError("REQUIRE_FULL_WINDOW must be boolean")
        if fps == 5 and (train_task != "fore" or not require_full_window):
            raise ValueError("5Hz training requires TRAIN_TASK=fore and REQUIRE_FULL_WINDOW=True")
        if split not in ("train","val","test"): raise ValueError("Unsupported dataset split")
        self.root = Path(archive_root or data_dir or "")
        assert_archive_ready(self.root)
        info_path = self.root/"meta/info.json"
        if not info_path.is_file(): raise FileNotFoundError(f"Missing pipeline archive info: {info_path}")
        info = json.loads(info_path.read_text())
        if int(info.get("fps",-1)) != 30: raise ValueError("Archive FPS must be native30Hz")
        upgrade = self.root/".pipeline/dino_upgrade.json"
        if cond_img_feat:
            if upgrade.exists() and json.loads(upgrade.read_text()).get("state") != "complete":
                raise ValueError("DINO5 archive upgrade unfinished; motion-only statistics may use cond_img_feat=False")
            if info.get("features",{}).get("observation.features.dino",{}).get("shape") != [5120]:
                raise ValueError("Image conditioning requires upgraded CLS+4register DINO5 storage [5120]")
        provenance_path = self.root/".pipeline/config.json"
        if not provenance_path.is_file(): raise ValueError("Not a recognized pipeline archive")
        provenance = json.loads(provenance_path.read_text())
        self.archive_schema_version = provenance.get("schema_version")
        if self.archive_schema_version not in ("egomotion-body22-camera-floor-v3","egomotion-body22-camera-floor-dino5-v4",V5_SCHEMA):
            raise ValueError("Unsupported camera-floor archive schema")
        if self.archive_schema_version == V5_SCHEMA:
            if info.get("schema_version") != V5_SCHEMA:
                raise ValueError("V5 archive metadata schema versions disagree")
            for key, width in ((BODY_WORLD_6D_COLUMN,198), *[(key,22) for key in BODY_WORLD_6D_MASKS.values()]):
                if info.get("features",{}).get(key,{}).get("shape") != [width]:
                    raise ValueError(f"V5 archive requires authoritative {key} width {width}")
        self.motion_columns = V5_MOTION_COLUMNS if self.archive_schema_version == V5_SCHEMA else MOTION_COLUMNS
        self.body_position_valid_column = (BODY_WORLD_6D_MASKS["position"] if self.archive_schema_version == V5_SCHEMA
                                           else "observation.valid.body_position")
        self.archive_records = read_records(self.root)
        if rotation_basis_mode == "source_native" and any(r["source"] != "egosuite" for r in self.archive_records):
            raise ValueError("source_native forbids mixed sources anywhere in the archive; use unified mappings before adding EgoExo-HM")
        self.manifest = load_split_manifest(split_manifest, self.archive_records)
        self.split_manifest_path = str(Path(split_manifest).resolve())
        self.split_fingerprint = hashlib.sha256(json.dumps(self.manifest,sort_keys=True).encode()).hexdigest()
        accepted = set(self.manifest["splits"][split])
        self.records = [r for r in self.archive_records if r["source_key"] in accepted]
        self.split, self.window, self.fps, self.forecast_prefix = split, window, fps, forecast_prefix
        self.sampling_step = 30 // fps
        self.train_task, self.require_full_window = train_task, require_full_window
        self.motion_supervision = motion_supervision
        self.statistics_scope = ("all_valid_frames" if motion_supervision == "joint" else
                                 "future_only" if train_task == "fore" else "all_real_frames")
        self.ground_estimation_scope = "observed_prefix" if train_task == "fore" else "full_window"
        self.image_validity_scope = self.ground_estimation_scope
        self.segment_stride = segment_stride
        self.cond_img_feat, self.cond_traj = bool(cond_img_feat), bool(cond_traj)
        self.repre_type, self.motion_feature_dim = REPRE_TYPE, MOTION_DIM
        self.img_feat_type, self.img_feat_dim = img_feat_type, 1024
        self.dataset_name = "mixegomotion"
        self.do_normalization = bool(do_normalization)
        self.min_valid_image_fraction = float(min_valid_image_fraction)
        if not np.isfinite(self.min_valid_image_fraction) or not 0<=self.min_valid_image_fraction<=1:
            raise ValueError("MIN_VALID_IMAGE_FRACTION must lie in [0,1]")
        if self.cond_img_feat and self.min_valid_image_fraction>0 and all(r.get("dino",{}).get("state")=="pending" for r in self.records):
            raise ValueError("All selected archive records have pending DINO features; extract real DINO5 features before visual training")
        self.rotation_basis_mode, self.target_joint_basis = rotation_basis_mode, target_joint_basis
        self.basis_mapping = _load_basis_mapping(rotation_basis_path)
        self.basis_mapping_fingerprint = hashlib.sha256(Path(rotation_basis_path).read_bytes()).hexdigest() if rotation_basis_path else "none"
        self.contact_speed_m_s, self.contact_height_m = float(contact_speed_m_s),float(contact_height_m)
        if not np.isfinite([self.contact_speed_m_s,self.contact_height_m]).all() or min(self.contact_speed_m_s,self.contact_height_m)<=0:
            raise ValueError("Contact speed/height thresholds must be finite positive values")
        if rotation_basis_mode not in ("source_native","unified"): raise ValueError("ROTATION_BASIS_MODE must be source_native or unified")
        if rotation_basis_mode == "unified" and not require_verified_rotations:
            raise ValueError("Unified rotation supervision cannot bypass joint-basis verification")
        if self.basis_mapping and self.basis_mapping["target_basis"] != target_joint_basis:
            raise ValueError("Basis mapping target differs from TARGET_JOINT_BASIS")
        self._rotation_policies = [self._rotation_policy(r) for r in self.records]
        self.seq_names = [r["source_key"] for r in self.records]
        self.motion_data = {name:{"source":r["source"],"num_frames":int(r["length"]),"episode_index":int(r["episode_index"])} for name,r in zip(self.seq_names,self.records)}
        self._seq_indices = {name:i for i,name in enumerate(self.seq_names)}
        self.reader = ProjectedParquetReader(cache_mb=row_group_cache_mb,file_cache=parquet_file_cache)
        self.idx_to_sidx_fidx = []
        self.rejected_windows = {}
        for sequence, record in enumerate(self.records):
            self._index_record(sequence,record)
        if not self.idx_to_sidx_fidx: raise ValueError(f"No usable >=2-frame windows in {split}; rejected={self.rejected_windows}")
        self.normalizer = MotionNormalizer(stats_path,clip=normalization_clip) if self.do_normalization else None
        if self.normalizer:
            stats = self.normalizer.stats
            if stats.get("split_fingerprint") != self.split_fingerprint: raise ValueError("Training statistics split manifest mismatch")
            if stats.get("archive_fingerprint") != self.manifest["archive_fingerprint"]: raise ValueError("Training statistics archive fingerprint mismatch")
            if stats.get("target_joint_basis") != target_joint_basis or stats.get("rotation_basis_mode") != rotation_basis_mode:
                raise ValueError("Training statistics joint-axis policy mismatch")
            if stats.get("basis_mapping_fingerprint") != self.basis_mapping_fingerprint: raise ValueError("Training statistics source calibration mismatch")
            for key,actual in (("contact_speed_m_s",self.contact_speed_m_s),("contact_height_m",self.contact_height_m)):
                if float(stats.get(key,-1)) != actual: raise ValueError(f"Training statistics {key} mismatch")
            for key, actual in (("window",window),("forecast_prefix",forecast_prefix),("segment_stride",segment_stride)):
                if int(stats.get(key,-1)) != actual: raise ValueError(f"Training statistics {key} mismatch")
            if stats["fps"] != fps:
                raise ValueError("Training statistics FPS mismatch")
            policy = {"sampling_step": self.sampling_step, "train_task": self.train_task,
                      "label_task": self.train_task, "complete_windows_only": self.require_full_window,
                      "ground_estimation_scope": self.ground_estimation_scope,
                      "image_validity_scope": self.image_validity_scope,
                      "min_valid_image_fraction": self.min_valid_image_fraction,
                      "statistics_scope": self.statistics_scope}
            legacy_policy = fps == 15 and train_task == "recon" and not require_full_window
            for key, actual in policy.items():
                if key not in stats and legacy_policy:
                    continue
                if stats.get(key) != actual:
                    raise ValueError(f"Training statistics {key} mismatch")
            if self.train_task == "fore":
                declared = stats.get("motion_supervision") or "future_only"
                if declared != self.motion_supervision:
                    raise ValueError("Training statistics motion_supervision mismatch")

    def _assert_archive_current(self, record=None):
        """Reject a live migration and stale cached dataset instances.

        DINO-only record updates are allowed. Changed motion identity requires
        rebuilding the dataset and its split/statistics before another read.
        """
        assert_archive_ready(self.root)
        current_schema = json.loads((self.root/".pipeline/config.json").read_text()).get("schema_version")
        if current_schema != self.archive_schema_version:
            raise ValueError("Archive schema changed; rebuild Dataset, splits and training statistics")
        if record is not None:
            path = self.root/f".pipeline/episodes/{int(record['episode_index']):08d}.json"
            current = json.loads(path.read_text())
            required = current_schema == V5_SCHEMA
            if motion_representation_identity(current, required=required) != motion_representation_identity(record, required=required):
                raise ValueError("Archive motion identity changed; rebuild Dataset, splits and training statistics")

    def _index_record(self,sequence,record):
        self._assert_archive_current(record)
        selected = np.arange(0,int(record["length"]),self.sampling_step,dtype=np.int64)
        if len(selected)<2: return
        columns = ["observation.valid.head_reference_position", "observation.valid.head_reference_rotation",
                   "observation.state.head_reference_world", "observation.state.head_reference_world_rotation",
                   self.body_position_valid_column]
        ground = record["metadata"].get("ground",{})
        # V5's producer validates authoritative XYZ/6D before committing its
        # motion identity. Normal indexing only needs those independent masks;
        # failed-floor episodes also need finite prefix feet for the fallback.
        # Each actual sample still decodes and validates the complete labels.
        read_body_values = self.train_task == "fore" and (
            self.archive_schema_version != V5_SCHEMA or not ground.get("valid",False))
        if self.train_task == "fore":
            if self.archive_schema_version == V5_SCHEMA:
                columns.append(BODY_WORLD_6D_MASKS["rotation"])
                if read_body_values: columns.append(BODY_WORLD_6D_COLUMN)
            else:
                columns.extend(["observation.state.body_world", "observation.state.body_world_rotation",
                                "observation.valid.body_rotation"])
        check_images = self.min_valid_image_fraction > 0 and (self.cond_img_feat or self.train_task == "fore")
        if check_images: columns.append("observation.valid.dino")
        arrays = self.reader.read(self._path(record),selected,columns)
        self._assert_archive_current(record)
        q = arrays["observation.state.head_reference_world_rotation"].reshape(-1,4)
        p = arrays["observation.state.head_reference_world"].reshape(-1,3)
        head_valid = (arrays["observation.valid.head_reference_position"].reshape(-1).astype(bool)
                      & arrays["observation.valid.head_reference_rotation"].reshape(-1).astype(bool)
                      & np.isfinite(q).all(-1) & np.isfinite(p).all(-1) & (np.linalg.norm(q,axis=-1)>1e-8))
        heading_valid = np.linalg.norm(quaternion_matrix(q)[:,:2,0],axis=-1)>1e-6
        body_valid = arrays[self.body_position_valid_column].reshape(-1,22).astype(bool)
        rotation_valid = None
        if self.train_task == "fore":
            if read_body_values:
                body, body_valid, rotation, rotation_valid = body_world_components(arrays)
                body_valid &= np.isfinite(body).all(-1)
                rotation_valid &= np.isfinite(rotation).all((-1,-2))
            else:
                rotation_valid = arrays[BODY_WORLD_6D_MASKS["rotation"]].reshape(-1,22).astype(bool)
        from ..config import K12_JOINTS
        for start in range(0,len(selected)-1,self.segment_stride):
            end=min(start+self.window,len(selected)); reason=None
            observed_end = min(start+self.forecast_prefix, end)
            if self.require_full_window and end-start != self.window:
                reason="incomplete_window"
            elif not record["metadata"].get("coordinate_verified",False) or not record["metadata"].get("head_reference",{}).get("basis_verified",False):
                reason="unverified_camera_world_axes"
            elif not (head_valid[start] and heading_valid[start]): reason="invalid_initial_camera"
            elif not body_valid[start:end,K12_JOINTS].any(): reason="no_dense12_position_labels"
            elif self.train_task == "fore" and end <= observed_end:
                reason="no_future_frames"
            elif ground.get("valid",False) and not np.isfinite(ground.get("height_m",np.nan)):
                reason="invalid_floor_height"
            elif not ground.get("valid",False) and not (ground.get("mode")=="estimate" and ground.get("window_fallback",{"mode":"mean_valid_feet"}).get("mode")=="mean_valid_feet" and body_valid[start:(observed_end if self.train_task == "fore" else end),[10,11]].any()):
                reason="no_usable_floor"
            if reason is None and self.train_task == "fore":
                future_reference = (head_valid & heading_valid)[observed_end:end]
                future_body = (body_valid[observed_end:end,K12_JOINTS] | rotation_valid[observed_end:end,K12_JOINTS])
                if not (future_body & future_reference[:,None]).any():
                    reason="no_valid_future_body_labels"
            if reason is None and check_images:
                image_end = observed_end if self.train_task == "fore" else end
                fraction=float(arrays["observation.valid.dino"].reshape(-1)[start:image_end].astype(bool).mean())
                if fraction<self.min_valid_image_fraction: reason="insufficient_real_DINO"
            if reason:
                self.rejected_windows[reason]=self.rejected_windows.get(reason,0)+1
            else: self.idx_to_sidx_fidx.append((sequence,int(selected[start])))

    def _rotation_policy(self,record):
        meta = record["metadata"]
        if self.rotation_basis_mode == "source_native":
            if record["source"] != "egosuite" or meta.get("rotation_semantics") != "source_global_wxyz":
                raise ValueError("egosuite source_native rotation supervision requires only EgoSuite source_global_wxyz; mixed sources need documented joint-axis mappings")
            if self.target_joint_basis != "egosuite_native_v1": raise ValueError("source_native requires TARGET_JOINT_BASIS=egosuite_native_v1")
            if meta.get("body_basis_correction") is not None: raise ValueError("Native mode refuses already-corrected joint axes without an explicit target-basis declaration")
            if self.basis_mapping is not None: raise ValueError("Native mode must not apply a unified basis correction")
            return None
        if self.basis_mapping and record["source"] in self.basis_mapping["sources"]:
            mapping = self.basis_mapping["sources"][record["source"]]
            if mapping["source_rotation_semantics"] != meta.get("rotation_semantics"):
                raise ValueError("Joint mapping source rotation semantics mismatch")
            if meta.get("body_basis_correction") is not None: raise ValueError("Source already has a basis correction; refusing ambiguous double correction")
            return np.asarray(mapping["corrections"],dtype=float)
        if not meta.get("rotation_basis_verified",False):
            raise ValueError(f"{record['source_key']}: source joint local axes are unverified. Use explicit single-source EgoSuite source_native policy or supply documented ROTATION_BASIS_PATH for unified supervision")
        declared = meta.get("rotation_target_basis") or meta.get("target_joint_basis")
        if not declared: raise ValueError("Verified rotations must explicitly declare their target joint basis, or provide a documented mapping")
        if declared != self.target_joint_basis: raise ValueError("Verified archive target joint axes differ from configured target")
        return None

    def __len__(self): return len(self.idx_to_sidx_fidx)

    def _path(self,record):
        index = int(record["episode_index"])
        return self.root/f"data/chunk-{index//1000:03d}/file-{index%1000:03d}.parquet"

    def normalize(self,value,kind,valid=None):
        result = self.normalizer.normalize(value,kind,valid) if self.normalizer else torch.as_tensor(value).float()
        return result.masked_fill(~torch.as_tensor(valid).bool(),0) if valid is not None else result

    def denormalize(self,value,kind="motion"):
        return self.normalizer.denormalize(value,kind) if self.normalizer else torch.as_tensor(value).float()

    def __getitem__(self,index): return self.get_sample(index)

    def get_sample(self,index,task=None):
        sequence,start = self.idx_to_sidx_fidx[int(index)]
        return self.get_from_seq_and_st(self.seq_names[sequence],start,idx=int(index),task=task)

    def get_from_seq_and_st(self,seq_name,st,idx=-1,task=None):
        if seq_name not in self._seq_indices: raise KeyError(f"Unknown sequence {seq_name}")
        if type(st) is not int or st<0 or st%self.sampling_step:
            raise ValueError(f"Start frame must be a zero-based30Hz archive index divisible by {self.sampling_step}")
        sequence = self._seq_indices[seq_name]; record = self.records[sequence]
        self._assert_archive_current(record)
        selected = st+self.sampling_step*np.arange(self.window)
        selected = selected[selected<int(record["length"])]
        if self.require_full_window and len(selected) != self.window:
            raise ValueError("This training profile requires a complete window without tail padding")
        if len(selected)<2: raise ValueError("Window needs at least two real selected frames")
        conditioning_task = task if task is not None else self.train_task
        label_task = "fore" if self.train_task == "fore" else conditioning_task
        if label_task=="fore" and len(selected)<=self.forecast_prefix: raise ValueError("Forecast window must contain valid future frames after the observed prefix")
        # Small motion groups and large DINO groups are cached independently.
        arrays = self.reader.read(self._path(record),selected,self.motion_columns)
        metadata = deepcopy(record["metadata"])
        metadata.update(rotation_supervision_policy=self.rotation_basis_mode,rotation_target_basis=self.target_joint_basis)
        representation = build_dense_window(arrays,metadata,task=label_task,forecast_prefix=self.forecast_prefix,
                                             fps=self.fps,body_basis_correction=self._rotation_policies[sequence],
                                             rotation_basis_verified=True,contact_speed_m_s=self.contact_speed_m_s,
                                             contact_height_m=self.contact_height_m)
        count = len(selected)
        def pad(value):
            value = torch.as_tensor(np.asarray(value)).clone()
            if len(value)==self.window: return value
            return torch.cat([value,torch.zeros((self.window-len(value),*value.shape[1:]),dtype=value.dtype)],0)
        if label_task=="fore" and not representation.motion_valid[self.forecast_prefix:,:108].any():
            raise ValueError("Forecast window has no valid future body/trajectory labels under prefix-floor policy")
        motion_valid = pad(representation.motion_valid).bool()
        trajectory_valid = pad(representation.trajectory_valid).bool()
        motion = self.normalize(pad(representation.motion),"motion",motion_valid)
        trajectory = self.normalize(pad(representation.trajectory),"traj",trajectory_valid)
        valid_frames = torch.arange(self.window)<count
        y = {"traj":trajectory,"trajectory_valid":trajectory_valid,"feature_valid_mask":motion_valid,
             "valid_frames":valid_frames,"padding_mask":valid_frames.clone()}
        if self.cond_img_feat:
            observed_indices = selected[:self.forecast_prefix] if self.train_task == "fore" else selected
            image = self.reader.read(self._path(record),observed_indices,IMAGE_COLUMNS)
            value = np.asarray(image["observation.features.dino"])
            image_count = len(observed_indices)
            if value.shape!=(image_count,5120): raise ValueError("DINO features must have token-major width5120")
            valid = np.asarray(image["observation.valid.dino"]).reshape(image_count).astype(bool) & np.isfinite(value).all(-1)
            embeddings = np.where(valid[:,None],value,0).reshape(image_count,5,1024)
            y.update(img_embs=pad(embeddings).float(),valid_img_embs=pad(valid).bool())
        else:
            # Stable model/batch keys without ever projecting/reading DINO columns.
            y.update(img_embs=torch.zeros(self.window,5,1024),valid_img_embs=torch.zeros(self.window,dtype=torch.bool))
        if not self.cond_traj: y["trajectory_valid"] = torch.zeros_like(trajectory_valid)
        details = representation.metadata["training_window"]
        details["source_frame_indices"] = selected.tolist()
        floor = np.tile(np.eye(4),(count,1,1))
        floor[:,:3,:3] = np.asarray(details["floor_rotation"]); floor[:,:3,3] = np.asarray(details["floor_position"])
        misc = {"motion":motion,"traj":trajectory,"feature_valid_mask":motion_valid,"trajectory_valid":trajectory_valid,
                "seq_name":seq_name,"source":record["source"],"dataset":self.dataset_name,"idx":int(idx),
                "start_idx":int(st),"end_idx":int(selected[-1]),"episode_index":int(record["episode_index"]),
                "num_real_frames":count,"fps":self.fps,"world_from_window":torch.tensor(details["archive_world_from_window"],dtype=torch.float32),
                "floor_transform":pad(floor).float(),"floor_height":float(details["floor_height_m"]),
                "ground_fallback_used":bool(representation.metadata["ground"].get("window_fallback_used",False)),
                "metadata_json":json.dumps(representation.metadata),
                "motion_nonfinite_mask":torch.zeros_like(motion_valid),"traj_nonfinite_mask":torch.zeros(self.window,dtype=torch.bool),
                "discard_terrain":False}
        ret = {"y":y,"misc":misc,"pred":{}}
        if task is not None or self.train_task == "fore":
            ret["y"] = apply_task_conditioning(y,conditioning_task,forecast_prefix=self.forecast_prefix,
                                               supervise_history=self.motion_supervision == "joint")
        self._assert_archive_current(record)
        return ret

    def process_sample_for_task(self,ret,task):
        # Reload: fore's prefix-ground policy changes both labels and camera-height conditions.
        return self.get_from_seq_and_st(ret["misc"]["seq_name"],int(ret["misc"]["start_idx"]),idx=int(ret["misc"]["idx"]),task=task)

    def ret_to_full_sequence(self,ret):
        """Decode all GT frames, or forecast predictions from the observed anchor."""
        from .dense_representation import decode_dense_motion
        has_prediction = "motion" in ret.get("pred",{})
        value = ret["pred"]["motion"] if has_prediction else ret["misc"]["motion"]
        count = int(ret["misc"]["num_real_frames"])
        valid_delta = ret["misc"]["feature_valid_mask"][:count,108:117].all(-1)
        task_id = ret.get("y",{}).get("task_id")
        task_is_fore = task_id is not None and torch.as_tensor(task_id).numel() == 1 and int(task_id) == TASK_TO_ID["fore"]
        is_fore = self.train_task == "fore" or ret.get("task") == "fore" or task_is_fore
        decode_options = {}
        if has_prediction and is_fore:
            if self.forecast_prefix >= count:
                raise ValueError("Forecast prediction has no real frames after the observed prefix")
            decode_options = {"start_frame": self.forecast_prefix,
                              "initial_floor_transform": ret["misc"]["floor_transform"][self.forecast_prefix-1]}
        return decode_dense_motion(self.denormalize(value[:count],"motion"),window_to_world=ret["misc"]["world_from_window"],floor_height=ret["misc"]["floor_height"],delta_valid=valid_delta,**decode_options)


class DenseMotionDataModule(pl.LightningDataModule):
    def __init__(self,cfg): super().__init__(); self.cfg=cfg

    def setup(self,stage=None):
        data=self.cfg.DATA
        def make(split):
            dataset_class = dataset_class_for_representation(data.REPRE_TYPE)
            return dataset_class(archive_root=getattr(data,"ARCHIVE_ROOT",None) or getattr(data,"DATA_DIR",None),
                split=split,split_manifest=getattr(data,"SPLIT_MANIFEST",None),stats_path=getattr(data,"STATS_PATH",None),
                do_normalization=True,cond_img_feat=data.COND_IMG_FEAT,cond_traj=data.COND_TRAJ,cond_betas=data.COND_BETAS,
                window=data.WINDOW,fps=getattr(data,"FPS",15),forecast_prefix=getattr(data,"FORECAST_PREFIX",30),
                segment_stride=getattr(data,"SEGMENT_STRIDE",30),sparse_joint_indices=self.cfg.SPARSE_JOINTS.INDICES,
                repre_type=data.REPRE_TYPE,img_feat_type=data.IMG_FEAT_TYPE,
                row_group_cache_mb=getattr(data,"ROW_GROUP_CACHE_MB",128),parquet_file_cache=getattr(data,"PARQUET_FILE_CACHE",8),
                rotation_basis_path=getattr(data,"ROTATION_BASIS_PATH",None),rotation_basis_mode=getattr(data,"ROTATION_BASIS_MODE","unified"),
                target_joint_basis=getattr(data,"TARGET_JOINT_BASIS","smpl22_global_joint_axes"),
                require_verified_rotations=getattr(data,"REQUIRE_VERIFIED_ROTATIONS",True),normalization_clip=getattr(data,"NORMALIZATION_CLIP",None),
                contact_speed_m_s=getattr(data,"CONTACT_SPEED_M_S",.2),contact_height_m=getattr(data,"CONTACT_HEIGHT_M",.1),
                min_valid_image_fraction=getattr(data,"MIN_VALID_IMAGE_FRACTION",0.),
                train_task=getattr(data,"TRAIN_TASK",None) or "recon",require_full_window=getattr(data,"REQUIRE_FULL_WINDOW",False),
                motion_supervision=getattr(data,"MOTION_SUPERVISION",None))
        if stage in (None,"fit","train") and not hasattr(self,"train_dataset"): self.train_dataset=make("train")
        if stage in (None,"fit","validate","val") and not hasattr(self,"val_dataset"): self.val_dataset=make("val")
        if stage in ("test","predict") and not hasattr(self,"test_dataset"): self.test_dataset=make("test")

    def _loader(self,dataset,shuffle=False):
        data=self.cfg.DATA; workers=int(data.NUM_WORKERS)
        options={"pin_memory":getattr(data,"PIN_MEMORY",False),"persistent_workers":getattr(data,"PERSISTENT_WORKERS",False) and workers>0}
        if workers>0: options["prefetch_factor"]=getattr(data,"PREFETCH_FACTOR",2)
        sampler=None
        if shuffle and getattr(data,"SOURCE_BALANCE",False):
            counts={}
            for sequence,_ in dataset.idx_to_sidx_fidx:
                source=dataset.records[sequence]["source"]; counts[source]=counts.get(source,0)+1
            weights=getattr(data,"SOURCE_SAMPLE_WEIGHTS",{})
            weights=vars(weights) if hasattr(weights,"__dict__") else weights
            values=[]
            for sequence,_ in dataset.idx_to_sidx_fidx:
                source=dataset.records[sequence]["source"]; weight=float(weights.get(source,1.))
                if not np.isfinite(weight) or weight<=0: raise ValueError("Source weights must be finite positive")
                values.append(weight/counts[source])
            sampler=WeightedRandomSampler(values,num_samples=len(values),replacement=True); shuffle=False
        return DataLoader(dataset,batch_size=data.BATCH_SIZE,num_workers=workers,shuffle=shuffle,sampler=sampler,
                          collate_fn=careful_collate_fn,drop_last=getattr(data,"DROP_LAST",False) if dataset.split=="train" else False,**options)

    def train_dataloader(self): return self._loader(self.train_dataset,shuffle=True)
    def val_dataloader(self,shuffle=False): return self._loader(self.val_dataset,shuffle=shuffle)
    def test_dataloader(self): return self._loader(self.test_dataset)


# Keep the existing training import path stable while retiring the SMPL pack reader.
def dataset_class_for_representation(repre_type):
    from ..config import SMPLH_REPRESENTATION_VERSION
    if repre_type == SMPLH_REPRESENTATION_VERSION:
        from .smplh_motion_dataset import SMPLHMotionDataset
        return SMPLHMotionDataset
    if repre_type != REPRE_TYPE:
        raise ValueError(f"Unsupported motion representation: {repre_type}")
    return DenseMotionDataset


def EE4D_Motion_Dataset(*args, **kwargs):
    """Select an explicit representation; direct legacy calls retain dense245."""
    return dataset_class_for_representation(kwargs.get("repre_type", REPRE_TYPE))(*args, **kwargs)
EE4D_Motion_DataModule = DenseMotionDataModule
