import os
import numpy as np
import torch
import copy
import pytorch_lightning as pl
from tqdm.auto import tqdm


from loguru import logger
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .canonicalization import get_a_canonicalized_segment
from .representation_utils import (
    V4_BETA_FEATURES,
    full_to_sparse_motion,
    saved_sequence_to_repre,
    sparse_motion_feature_dim,
    sparse_motion_feature_indices,
    validate_sparse_body_joint_indices,
)
from .skeleton import get_smpl
from ..utils.torch_utils import careful_collate_fn
from ..utils.task_conditioning import apply_task_conditioning
from .feats import ImageFeats
from .text_conditions import TextFeatureStore


class EE4D_Motion_Dataset(Dataset):
    def __init__(
        self,
        *,
        data_dir,
        split,
        repre_type,
        cond_traj,
        cond_img_feat,
        cond_betas,
        window,
        img_feat_type,
        do_normalization=True,
        sparse_joint_indices=None,
        motion_pack_paths=None,
        dataset_name="ee4d",
        nymeria_img_feat_path=None,
        ee4d_img_feat_path=None,
        stats_path=None,
        stats_mode="ee4d_compat",
        cond_text=False,
        text_features_path=None,
        text_feature_dim=768,
        text_token_count=16,
        source_subsample=None,
        sequence_manifest_path=None,
        exclude_terrain=False,
        sanitize_nonfinite=True,
        normalization_clip=None,
        mmap_motion_packs=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.cond_img_feat = cond_img_feat
        self.cond_traj = cond_traj
        self.repre_type = repre_type
        if repre_type != "v4_beta":
            raise ValueError("MotionExpert uses the v4_beta representation")
        self.img_feat_type = img_feat_type
        self.cond_betas = cond_betas
        self.dataset_name = dataset_name
        self.motion_pack_paths = motion_pack_paths
        self.nymeria_img_feat_path = nymeria_img_feat_path
        self.ee4d_img_feat_path = ee4d_img_feat_path
        self.stats_path = stats_path
        self.stats_mode = stats_mode
        self.cond_text = bool(cond_text)
        self.text_features_path = text_features_path
        self.text_feature_dim = int(text_feature_dim)
        self.text_token_count = int(text_token_count)
        self.source_subsample = source_subsample or {}
        self.sequence_manifest_path = sequence_manifest_path
        self.exclude_terrain = bool(exclude_terrain)
        self.sanitize_nonfinite = bool(sanitize_nonfinite)
        if normalization_clip is None:
            self.normalization_clip = None
        else:
            self.normalization_clip = float(normalization_clip)
            if not np.isfinite(self.normalization_clip) or self.normalization_clip <= 0:
                raise ValueError(
                    "normalization_clip must be a finite positive number or None, "
                    f"got {normalization_clip!r}"
                )
        self.mmap_motion_packs = bool(mmap_motion_packs)
        self.text_features = None
        self.sparse_joint_indices = None
        if sparse_joint_indices is not None:
            self.sparse_joint_indices = validate_sparse_body_joint_indices(sparse_joint_indices)
        self.sparse_joints_enabled = self.sparse_joint_indices is not None
        self.motion_feature_dim = (
            sparse_motion_feature_dim(self.sparse_joint_indices)
            if self.sparse_joints_enabled
            else 243
        )
        if self.cond_betas:
            logger.warning("Conditioning on betas.")
        assert self.split in ["train", "val"]
        assert self.img_feat_type == "dinov2"

        if not self.cond_traj:
            logger.warning("NOT USING TRAJ AS CONDITION.")

        self.window = window
        self.segment_stride = 20
        self.do_normalization = do_normalization
        if not self.sanitize_nonfinite and self.do_normalization:
            raise ValueError("sanitize_nonfinite=False is only supported with do_normalization=False")
        if not self.do_normalization:
            logger.warning("NOT NORMALIZING. PLEASE CHECK.")

        # Load SMPL
        self.smpl = get_smpl()

        self.load_motion_data()
        if self.cond_text:
            if self.text_features_path:
                self.text_features = TextFeatureStore(
                    self.text_features_path,
                    token_count=self.text_token_count,
                    feature_dim=self.text_feature_dim,
                )
            else:
                raise ValueError("DATA.COND_TEXT=True requires DATA.TEXT_FEATURES_PATH")
        self.load_statistics()
        if self.cond_img_feat:
            self.img_feats = ImageFeats(
                self.data_dir,
                self.split,
                self.img_feat_type,
                nymeria_feat_path=self.nymeria_img_feat_path,
                ee4d_feat_path=self.ee4d_img_feat_path,
            )
            feat_values = self.img_feats.ego_img_feats.get("feats", {})
            if not feat_values and self.img_feats.nymeria_img_feats is not None:
                feat_values = self.img_feats.nymeria_img_feats.get("feats", {})
            if not feat_values:
                raise ValueError("no image feature sequences found in EE4D/Nymeria sidecars")
            first_feat = next(iter(feat_values.values()))
            self.img_feat_dim = int(first_feat.shape[-1])

    def load_motion_data(self):
        if self.motion_pack_paths is None:
            processed_paths = [f"{self.data_dir}/uniegomotion/ee_{self.split}.pt"]
        elif isinstance(self.motion_pack_paths, (str, os.PathLike)):
            processed_paths = [str(self.motion_pack_paths)]
        else:
            processed_paths = [str(path) for path in self.motion_pack_paths]

        self.motion_data = {}
        for processed_path in processed_paths:
            if not os.path.isfile(processed_path):
                raise FileNotFoundError(f"Motion pack not found: {processed_path}")
            logger.info(f"Loading {processed_path}.")
            try:
                pack = torch.load(
                    processed_path,
                    map_location="cpu",
                    weights_only=False,
                    mmap=self.mmap_motion_packs,
                )
            except (TypeError, RuntimeError) as exc:
                if not self.mmap_motion_packs:
                    raise
                logger.warning(
                    "Memory-mapped pack load failed for %s (%s); falling back to regular load.",
                    processed_path,
                    exc,
                )
                pack = torch.load(processed_path, weights_only=False)
            overlap = set(self.motion_data).intersection(pack)
            if overlap:
                raise ValueError(f"Duplicate sequence names across motion packs: {sorted(overlap)[:3]}")
            self.motion_data.update(pack)

        if self.sequence_manifest_path:
            manifest_path = str(self.sequence_manifest_path)
            if not os.path.isfile(manifest_path):
                raise FileNotFoundError(f"Motion QC manifest not found: {manifest_path}")
            manifest = torch.load(manifest_path, weights_only=False) if manifest_path.endswith(".pt") else None
            if manifest is None:
                import json

                with open(manifest_path, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            accepted = manifest.get("accepted_sequences") if isinstance(manifest, dict) else None
            if accepted is None and isinstance(manifest, dict):
                accepted = [
                    name
                    for name, status in manifest.get("sequence_status", {}).items()
                    if isinstance(status, dict) and bool(status.get("accepted", False))
                ]
            if not accepted:
                raise ValueError(f"Motion QC manifest has no accepted_sequences: {manifest_path}")
            accepted_set = {str(name) for name in accepted}
            available = set(self.motion_data)
            unknown = accepted_set - available
            if unknown:
                raise ValueError(
                    f"Motion QC manifest contains {len(unknown)} sequences absent from packs; "
                    f"first={sorted(unknown)[:3]}"
                )
            self.motion_data = {
                name: record for name, record in self.motion_data.items() if name in accepted_set
            }
            logger.info(
                f"Applied motion QC manifest {manifest_path}: "
                f"kept={len(self.motion_data)}/{len(available)} sequences"
            )

        if self.exclude_terrain:
            before = len(self.motion_data)
            self.motion_data = {
                name: record
                for name, record in self.motion_data.items()
                if not bool(record.get("discard_terrain", False))
            }
            logger.info(
                "Excluded terrain-interaction sequences: kept=%d/%d",
                len(self.motion_data),
                before,
            )

        self.seq_names = list(self.motion_data.keys())
        if not self.seq_names:
            raise ValueError("No motion sequences remain after QC filtering")
        logger.info(f"Loaded {len(self.motion_data)} sequences.")

        # Create a mapping from index to (seq_idx, frame_idx)
        # We use a stride of 20 frames to create segments.
        self.idx_to_sidx_fidx = []
        for seq_idx, seq_name in enumerate(self.seq_names):
            T = self.motion_data[seq_name]["num_frames"]
            for fidx in range(0, T - self.segment_stride, self.segment_stride):
                self.idx_to_sidx_fidx.append((seq_idx, fidx))
        self.idx_to_sidx_fidx = np.array(self.idx_to_sidx_fidx, dtype=int).reshape(-1, 2)
        self._subsample_train_windows()
        if not len(self.idx_to_sidx_fidx):
            raise ValueError("No motion windows remain; sequences must exceed 20 frames")

    def _subsample_train_windows(self):
        """Select a deterministic fraction independently for each source."""

        if self.split != "train" or not bool(self.source_subsample.get("enabled", False)):
            return
        ee_fraction = float(self.source_subsample.get("ee4d_fraction", 1.0))
        ny_fraction = float(self.source_subsample.get("nymeria_fraction", 1.0))
        seed = int(self.source_subsample.get("seed", 62))
        if not 0.0 < ee_fraction <= 1.0 or not 0.0 < ny_fraction <= 1.0:
            raise ValueError("source subsample fractions must lie in (0, 1]")

        source_indices = {"ee4d": [], "nymeriaplus": []}
        for index, (seq_idx, _frame_idx) in enumerate(self.idx_to_sidx_fidx):
            seq_name = self.seq_names[int(seq_idx)]
            source = self.motion_data[seq_name].get("source", "ee4d")
            source_indices.setdefault(source, []).append(index)

        selected: list[int] = []
        counts_before: dict[str, int] = {}
        counts_after: dict[str, int] = {}
        for source, indices in source_indices.items():
            if not indices:
                continue
            fraction = ny_fraction if source == "nymeriaplus" else ee_fraction
            count = max(1, int(round(len(indices) * fraction)))
            generator = np.random.default_rng(seed + (1 if source == "nymeriaplus" else 0))
            chosen = np.sort(generator.choice(np.asarray(indices), size=count, replace=False))
            selected.extend(int(item) for item in chosen)
            counts_before[source] = len(indices)
            counts_after[source] = len(chosen)
        self.idx_to_sidx_fidx = self.idx_to_sidx_fidx[np.sort(np.asarray(selected, dtype=int))]
        logger.info(
            f"Deterministic source subsample ({self.split}, seed={seed}): "
            f"before={counts_before} after={counts_after}"
        )

    def load_statistics(self):
        if not self.do_normalization:
            return
        logger.warning("Loading stats for the window of 80 frames.")
        if self.stats_mode not in {"ee4d_compat", "joint_masked"}:
            raise ValueError(
                "stats_mode must be 'ee4d_compat' or 'joint_masked', "
                f"got {self.stats_mode!r}"
            )
        default_name = (
            f"{self.repre_type}_ee_train_stats.pt"
            if self.stats_mode == "ee4d_compat"
            else f"{self.repre_type}_joint_masked_train_stats.pt"
        )
        stats_path = self.stats_path or f"{self.data_dir}/uniegomotion/{default_name}"
        loaded_stats = torch.load(stats_path, weights_only=False)
        stat_keys = (
            "motion_mean", "motion_std", "motion_min", "motion_max",
            "traj_mean", "traj_std", "traj_min", "traj_max",
        )

        if self.stats_mode == "joint_masked":
            schema_version = int(loaded_stats.get("stats_schema_version", -1))
            if schema_version < 2:
                raise ValueError(
                    f"Unsafe legacy joint statistics at {stats_path}: "
                    "recompute with run/compute_joint_motion_stats.py (schema >= 2)."
                )
            if loaded_stats.get("stats_mode") != "joint_masked":
                raise ValueError(
                    f"Statistics mode mismatch at {stats_path}: "
                    f"expected 'joint_masked', got {loaded_stats.get('stats_mode')!r}"
                )
            if loaded_stats.get("excluded_shards"):
                raise ValueError(
                    f"Joint statistics at {stats_path} omit shards "
                    f"{loaded_stats['excluded_shards']}; rebuild a complete QC pass."
                )
            sequence_qc = loaded_stats.get("sequence_qc")
            if not isinstance(sequence_qc, dict) or not sequence_qc:
                raise ValueError(f"Joint statistics at {stats_path} have no sequence QC records")

        for key in stat_keys:
            value = torch.as_tensor(loaded_stats[key])
            invalid = ~torch.isfinite(value)
            if invalid.any():
                if self.stats_mode == "joint_masked":
                    raise ValueError(
                        f"Joint statistics at {stats_path} contain "
                        f"{int(invalid.sum())} non-finite values in {key}; refusing silent repair."
                    )
                value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
                logger.warning(
                    f"Repaired {int(invalid.sum())} non-finite entries in {key} from {stats_path}."
                )
            if key.endswith("_std"):
                if self.stats_mode == "joint_masked" and (value <= 0).any():
                    raise ValueError(f"Joint statistics at {stats_path} contain non-positive values in {key}")
                value = torch.where(value.abs() < 1e-8, torch.ones_like(value), value)
            loaded_stats[key] = value

        if self.stats_mode == "joint_masked":
            motion_count = torch.as_tensor(loaded_stats.get("motion_count", []))
            if motion_count.numel() != 243 or not torch.all(motion_count > 0):
                raise ValueError(f"Joint statistics at {stats_path} have incomplete motion counts")
            motion_nonfinite = int(torch.as_tensor(loaded_stats.get("motion_nonfinite_count", 0)).sum())
            traj_nonfinite = int(torch.as_tensor(loaded_stats.get("traj_nonfinite_count", 0)).sum())
            bad_sequences = [
                name
                for name, record in sequence_qc.items()
                if int(record.get("nonfinite_motion_values", 0))
                or int(record.get("nonfinite_traj_frames", 0))
            ]
            if motion_nonfinite or traj_nonfinite or bad_sequences:
                raise ValueError(
                    f"Joint statistics at {stats_path} still report raw non-finite inputs: "
                    f"motion={motion_nonfinite}, trajectory={traj_nonfinite}, "
                    f"sequences={len(bad_sequences)}"
                )
        self.full_stats = loaded_stats
        self.stats = dict(loaded_stats)
        if self.sparse_joints_enabled:
            feature_indices = sparse_motion_feature_indices(self.sparse_joint_indices)
            for key in ("motion_mean", "motion_std", "motion_min", "motion_max"):
                if key in self.stats:
                    self.stats[key] = self.stats[key].index_select(0, feature_indices)

    def __len__(self):
        return len(self.idx_to_sidx_fidx)

    def normalize(self, a, k):
        if not self.do_normalization:
            return a
        na = (a - self.stats[k + "_mean"]) / (self.stats[k + "_std"] + 1e-6)
        if self.normalization_clip is not None:
            na = na.clamp(-self.normalization_clip, self.normalization_clip)
        return na

    def denormalize(self, a, k):
        if not self.do_normalization:
            return a
        na = a * (self.stats[k + "_std"] + 1e-6) + self.stats[k + "_mean"]
        return na

    def pad_to_window(self, ret):
        valid_frames = torch.ones(self.window).long()
        T = ret["misc"]["motion"].shape[0]
        if T != self.window:
            valid_frames[T:] = 0  # mark invalid frames
            pad_fn = lambda x: torch.cat([x, torch.zeros(self.window - T, *x.shape[1:], dtype=x.dtype)], dim=0)

            if "traj" in ret["y"]:
                ret["y"]["traj"] = pad_fn(ret["y"]["traj"])
            if "img_embs" in ret["y"]:
                ret["y"]["img_embs"] = pad_fn(ret["y"]["img_embs"])
                ret["y"]["valid_img_embs"] = pad_fn(ret["y"]["valid_img_embs"])
            if "traj" in ret["misc"]:
                ret["misc"]["traj"] = pad_fn(ret["misc"]["traj"])
            if "motion" in ret["misc"]:
                ret["misc"]["motion"] = pad_fn(ret["misc"]["motion"])
            if "feature_valid_mask" in ret["y"]:
                ret["y"]["feature_valid_mask"] = pad_fn(ret["y"]["feature_valid_mask"])
            if "feature_valid_mask" in ret["misc"]:
                ret["misc"]["feature_valid_mask"] = pad_fn(ret["misc"]["feature_valid_mask"])
            if "motion_nonfinite_mask" in ret["misc"]:
                ret["misc"]["motion_nonfinite_mask"] = pad_fn(ret["misc"]["motion_nonfinite_mask"])
            if "traj_nonfinite_mask" in ret["misc"]:
                ret["misc"]["traj_nonfinite_mask"] = pad_fn(ret["misc"]["traj_nonfinite_mask"])

        ret["y"]["valid_frames"] = valid_frames  # determines the number of frames
        return ret

    def __getitem__(self, idx):
        sidx, st = self.idx_to_sidx_fidx[idx]
        seq_name = self.seq_names[sidx]
        return self.get_from_seq_and_st(seq_name, st, idx)

    def get_from_seq_and_st(self, seq_name, st, idx):
        if seq_name not in self.motion_data:
            raise KeyError(f"Sequence is absent from the selected/QC-filtered split: {seq_name}")
        if not 0 <= st < self.motion_data[seq_name]["num_frames"] - 1:
            raise ValueError("Window start must leave at least two motion frames")
        en = min(st + self.window - 1, self.motion_data[seq_name]["num_frames"] - 1)
        floor_height = self.motion_data[seq_name]["floor_height"]
        # logger.warning(f"Sequence: {seq_name}, start index: {st}, end index: {en}")

        segment_data = get_a_canonicalized_segment(
            self.motion_data[seq_name]["smpl_params"],
            self.motion_data[seq_name]["aria_traj"],
            self.motion_data[seq_name]["kp3d"],
            self.smpl,
            st,
            en,
        )

        motion, traj = saved_sequence_to_repre(
            self.repre_type,
            segment_data["can_aria_traj"],
            segment_data["can_smpl_params"],
            segment_data["can_kp3d"],
            floor_height,
            self.smpl,
        )

        if self.sparse_joints_enabled:
            motion = full_to_sparse_motion(motion, self.sparse_joint_indices)

        feature_valid_mask = self.motion_data[seq_name].get("feature_valid_mask")
        if feature_valid_mask is None:
            feature_valid_mask = torch.ones(243, dtype=torch.bool)
        feature_valid_mask = torch.as_tensor(feature_valid_mask, dtype=torch.bool)
        if feature_valid_mask.ndim == 1:
            feature_valid_mask = feature_valid_mask[None].expand(motion.shape[0], -1)
        else:
            feature_valid_mask = feature_valid_mask[st : en + 1]
        if feature_valid_mask.shape[-1] != 243:
            raise ValueError(
                f"feature_valid_mask for {seq_name} must have 243 features, "
                f"got {tuple(feature_valid_mask.shape)}"
            )
        if self.sparse_joints_enabled:
            feature_valid_mask = feature_valid_mask.index_select(
                -1, sparse_motion_feature_indices(self.sparse_joint_indices)
            )

        # A malformed source window must not turn an entire distributed batch
        # into NaN. Mark non-finite motion dimensions invalid and replace them
        # with neutral values before normalization; trajectory NaNs are also
        # zeroed because trajectory has no per-feature validity mask.
        motion_finite = torch.isfinite(motion)
        motion_nonfinite_mask = (~motion_finite) & feature_valid_mask
        traj_nonfinite_mask = ~torch.isfinite(traj).all(dim=-1)
        feature_valid_mask = feature_valid_mask & motion_finite
        if self.sanitize_nonfinite:
            motion = torch.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
            traj = torch.nan_to_num(traj, nan=0.0, posinf=0.0, neginf=0.0)
        motion = self.normalize(motion, "motion")
        traj = self.normalize(traj, "traj")
        # Invalid source dimensions must not contribute to training, and their
        # normalized values are explicitly zero to avoid leaking arbitrary
        # Nymeria hand/beta estimates into conditioning or diagnostics.
        motion = motion.masked_fill(~feature_valid_mask, 0.0)
        if self.cond_img_feat:
            source = self.motion_data[seq_name].get("source", "ee4d")
            if source == "nymeriaplus":
                mapping = self.motion_data[seq_name].get("motion_to_dino_idx")
                # EE4D sequence names refer to 30 FPS source frames and the
                # legacy canonicalizer returns indices multiplied by three.
                # Nymeria packs are already sampled at 10 FPS, so use the
                # Dataset window indices directly for its 5 FPS sidecar.
                image_start_idx, image_end_idx = st, en
                if self.img_feats.nymeria_img_feats is None:
                    raise FileNotFoundError("NymeriaPlus image conditioning requires its DINO sidecar")
                else:
                    img_embs, valid_img_embs = self.img_feats.get_nymeria_img_feats(
                        seq_name,
                        image_start_idx,
                        image_end_idx,
                        motion_to_dino_idx=mapping,
                    )
            else:
                img_embs, valid_img_embs = self.img_feats.get_img_feats(
                    seq_name, segment_data["start_idx"], segment_data["end_idx"]
                )  # T x D

        if self.cond_betas:
            betas = segment_data["can_smpl_params"]["betas"][:1].view(10)

        text_features = None
        if self.cond_text:
            text_features = self.text_features.get(seq_name, st)

        ret = {}
        ret["y"] = {}  # holds conditioning inputs
        ret["pred"] = {}  # holds predicted outputs
        if self.cond_traj:
            ret["y"]["traj"] = traj
        if self.cond_img_feat:
            ret["y"]["img_embs"] = img_embs
            ret["y"]["valid_img_embs"] = valid_img_embs

        if self.cond_betas:
            ret["y"]["betas"] = betas
        if text_features is not None:
            text_pooled, text_tokens, text_mask, text_valid = text_features
            ret["y"]["text_pooled"] = text_pooled
            ret["y"]["text_tokens"] = text_tokens
            ret["y"]["text_mask"] = text_mask
            ret["y"]["text_valid"] = text_valid
        ret["misc"] = {  # holds ground truth and other metadata
            "seq_name": seq_name,
            "start_idx": segment_data["start_idx"],
            "end_idx": segment_data["end_idx"],
            "idx": idx,
            "dataset": self.dataset_name,
            "source": self.motion_data[seq_name].get("source", "ee4d"),
            "discard_terrain": bool(self.motion_data[seq_name].get("discard_terrain", False)),
            "traj": traj,
            "motion": motion,
            "feature_valid_mask": feature_valid_mask,
            "motion_nonfinite_mask": motion_nonfinite_mask,
            "traj_nonfinite_mask": traj_nonfinite_mask,
        }
        ret["y"]["feature_valid_mask"] = feature_valid_mask
        ret = self.pad_to_window(ret)
        return ret


    def process_sample_for_task(self, ret, task):
        ret = copy.deepcopy(ret)
        ret["y"] = apply_task_conditioning(
            ret["y"],
            task,
            forecast_prefix=self.window // 4,
        )
        return ret



class EE4D_Motion_DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str):
        assert stage in [None, "fit", "validate", "predict", "train", "val"]
        kwargs = dict(
            data_dir=self.cfg.DATA.DATA_DIR,
            repre_type=self.cfg.DATA.REPRE_TYPE,
            cond_img_feat=self.cfg.DATA.COND_IMG_FEAT,
            cond_traj=self.cfg.DATA.COND_TRAJ,
            window=self.cfg.DATA.WINDOW,
            img_feat_type=self.cfg.DATA.IMG_FEAT_TYPE,
            cond_betas=self.cfg.DATA.COND_BETAS,
            sparse_joint_indices=(
                list(self.cfg.SPARSE_JOINTS.INDICES)
                if getattr(self.cfg, "SPARSE_JOINTS", None) is not None
                and self.cfg.SPARSE_JOINTS.ENABLED
                else None
            ),
        )
        dataset_name = self.cfg.DATA.DATASET_NAME
        assert dataset_name in ["ee4d", "ee4d_nymeria"]

        def _pack_paths(split):
            ee_path = getattr(self.cfg.DATA, "EE4D_PACK", None)
            ny_path = getattr(self.cfg.DATA, "NYMERIA_PACK", None)
            ee_path = ee_path or f"{self.cfg.DATA.DATA_DIR}/uniegomotion/ee_{split}.pt"
            ny_path = ny_path or f"{self.cfg.DATA.DATA_DIR}/uniegomotion/nymeria_{split}.pt"
            ee_path = str(ee_path).format(split=split)
            ny_path = str(ny_path).format(split=split)
            return [ee_path] if dataset_name == "ee4d" else [ee_path, ny_path]

        def _dataset(split):
            nymeria_img_feat_path = getattr(self.cfg.DATA, "NYMERIA_DINO_PACK", None)
            if nymeria_img_feat_path:
                nymeria_img_feat_path = str(nymeria_img_feat_path).format(split=split)
            stats_path = getattr(self.cfg.DATA, "STATS_PATH", None)
            if stats_path:
                stats_path = str(stats_path).format(split=split)
            text_features_path = getattr(self.cfg.DATA, "TEXT_FEATURES_PATH", None)
            if text_features_path:
                text_features_path = str(text_features_path).format(split=split)
            sequence_manifest_path = getattr(self.cfg.DATA, "SEQUENCE_MANIFEST", None)
            if sequence_manifest_path:
                sequence_manifest_path = str(sequence_manifest_path).format(split=split)
            subsample_cfg = getattr(self.cfg.DATA, "SOURCE_SUBSAMPLE", None)
            source_subsample = {
                "enabled": bool(getattr(subsample_cfg, "ENABLED", False)) if subsample_cfg is not None else False,
                "ee4d_fraction": float(getattr(subsample_cfg, "EE4D_FRACTION", 1.0))
                if subsample_cfg is not None
                else 1.0,
                "nymeria_fraction": float(getattr(subsample_cfg, "NYMERIA_FRACTION", 1.0))
                if subsample_cfg is not None
                else 1.0,
                "seed": int(getattr(subsample_cfg, "SEED", 62)) if subsample_cfg is not None else 62,
            }
            return EE4D_Motion_Dataset(
                split=split,
                motion_pack_paths=_pack_paths(split),
                dataset_name=dataset_name,
                nymeria_img_feat_path=nymeria_img_feat_path,
                ee4d_img_feat_path=(str(self.cfg.DATA.EE4D_DINO_PACK).format(split=split)
                                    if getattr(self.cfg.DATA, "EE4D_DINO_PACK", None) else None),
                stats_path=stats_path,
                stats_mode=getattr(self.cfg.DATA, "STATS_MODE", "ee4d_compat"),
                cond_text=bool(getattr(self.cfg.DATA, "COND_TEXT", False)),
                text_features_path=text_features_path,
                text_feature_dim=int(getattr(self.cfg.DATA, "TEXT_FEATURE_DIM", 768)),
                text_token_count=int(getattr(self.cfg.DATA, "TEXT_TOKEN_COUNT", 16)),
                source_subsample=source_subsample,
                sequence_manifest_path=sequence_manifest_path,
                exclude_terrain=bool(getattr(self.cfg.DATA, "EXCLUDE_TERRAIN", False)),
                sanitize_nonfinite=bool(getattr(self.cfg.DATA, "SANITIZE_NONFINITE", True)),
                normalization_clip=getattr(self.cfg.DATA, "NORMALIZATION_CLIP", None),
                mmap_motion_packs=bool(getattr(self.cfg.DATA, "MMAP_MOTION_PACKS", True)),
                **kwargs,
            )

        if stage in [None, "fit", "train"] and not hasattr(self, "train_dataset"):
            self.train_dataset = _dataset("train")
            logger.info(f"Train dataset: {len(self.train_dataset)}")

        if stage != "train" and not hasattr(self, "val_dataset"):
            self.val_dataset = _dataset("val")
            logger.info(f"Val dataset: {len(self.val_dataset)}")

    def val_dataloader(self, shuffle=False):
        num_workers = self.cfg.DATA.NUM_WORKERS
        loader_kwargs = dict(
            pin_memory=getattr(self.cfg.DATA, "PIN_MEMORY", False),
            persistent_workers=getattr(self.cfg.DATA, "PERSISTENT_WORKERS", False) and num_workers > 0,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = getattr(self.cfg.DATA, "PREFETCH_FACTOR", 2)
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=careful_collate_fn,
            **loader_kwargs,
        )

    def train_dataloader(self):
        num_workers = self.cfg.DATA.NUM_WORKERS
        loader_kwargs = dict(
            pin_memory=getattr(self.cfg.DATA, "PIN_MEMORY", False),
            persistent_workers=getattr(self.cfg.DATA, "PERSISTENT_WORKERS", False) and num_workers > 0,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = getattr(self.cfg.DATA, "PREFETCH_FACTOR", 2)
        sampler = None
        shuffle = True
        if getattr(self.cfg.DATA, "SOURCE_BALANCE", False) and hasattr(self.train_dataset, "motion_data"):
            ee_weight = float(getattr(self.cfg.DATA, "EE4D_SAMPLE_WEIGHT", 1.0))
            ny_weight = float(getattr(self.cfg.DATA, "NYMERIA_SAMPLE_WEIGHT", 1.0))
            if not np.isfinite([ee_weight, ny_weight]).all() or min(ee_weight, ny_weight) <= 0:
                raise ValueError("Source sampling weights must be finite and positive")
            source_names = []
            for seq_idx, _frame_idx in self.train_dataset.idx_to_sidx_fidx:
                seq_name = self.train_dataset.seq_names[int(seq_idx)]
                source = self.train_dataset.motion_data[seq_name].get("source", "ee4d")
                source_names.append("nymeria" if source == "nymeriaplus" else "ee4d")
            source_counts = {
                source: source_names.count(source)
                for source in ("ee4d", "nymeria")
            }
            sample_weights = [
                (ny_weight if source == "nymeria" else ee_weight)
                / max(source_counts[source], 1)
                for source in source_names
            ]
            if sample_weights and all(source_counts.values()):
                sampler = WeightedRandomSampler(
                    torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False
                logger.info(
                    f"Balanced source sampler: windows={source_counts}, "
                    f"relative source weights=(ee4d={ee_weight:.3f}, nymeria={ny_weight:.3f})"
                )
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.DATA.BATCH_SIZE,
            num_workers=num_workers,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=careful_collate_fn,
            drop_last=getattr(self.cfg.DATA, "DROP_LAST", False),
            **loader_kwargs,
        )


