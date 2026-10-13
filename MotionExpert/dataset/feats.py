import os

import torch
from loguru import logger


class ImageFeats(object):
    def __init__(self, data_dir, split, img_feat_type, nymeria_feat_path=None, ee4d_feat_path=None):
        if img_feat_type != "dinov2":
            raise ValueError("E7 K12 400M consumes DINOv2 CLS features")
        self.data_dir = data_dir
        self.split = split
        self.img_feat_type = img_feat_type
        self.nymeria_feat_path = nymeria_feat_path
        self.ee4d_feat_path = ee4d_feat_path
        self.nymeria_img_feats = None
        self.load_img_feats()

    def load_img_feats(self):

        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            processed_path = self.ee4d_feat_path or f"{self.data_dir}/uniegomotion/egoview_dinov2_{self.split}.pt"
            logger.info(f"Loading dinov2 feats from {processed_path}.")
            self.ego_img_feats = torch.load(processed_path, weights_only=False, mmap=True)
            if self.nymeria_feat_path is None:
                default_nymeria_path = (
                    f"{self.data_dir}/uniegomotion/egoview_dinov2_nymeria_{self.split}.pt"
                )
                self.nymeria_feat_path = default_nymeria_path if os.path.isfile(default_nymeria_path) else None
            if self.nymeria_feat_path and not os.path.isfile(self.nymeria_feat_path):
                raise FileNotFoundError(f"Nymeria DINO sidecar not found: {self.nymeria_feat_path}")
            if self.nymeria_feat_path:
                logger.info(f"Loading Nymeria dinov2 feats from {self.nymeria_feat_path}.")
                self.nymeria_img_feats = torch.load(self.nymeria_feat_path, weights_only=False, mmap=True)
            return

    def get_nymeria_img_feats(self, seq_name, start_idx, end_idx, motion_to_dino_idx=None):
        """Return Nymeria RGB features aligned to the 10 FPS motion window."""
        if self.nymeria_img_feats is None:
            raise FileNotFoundError(
                "Nymeria DINO sidecar is unavailable; run "
                "run/extract_nymeriaplus_dinov2.py before image-conditioned training."
            )
        sidecar_feats = self.nymeria_img_feats.get("feats", {})
        if seq_name not in sidecar_feats:
            raise KeyError(f"Nymeria DINO sidecar has no sequence {seq_name!r}")
        if motion_to_dino_idx is None:
            sidecar_mapping = self.nymeria_img_feats.get("motion_to_dino_idx", {})
            motion_to_dino_idx = sidecar_mapping.get(seq_name)
        if motion_to_dino_idx is None:
            raise KeyError(f"Nymeria DINO sidecar has no motion mapping for {seq_name!r}")
        mapping = torch.as_tensor(motion_to_dino_idx, dtype=torch.long)
        if mapping.ndim != 1 or start_idx < 0 or end_idx < start_idx or end_idx >= mapping.numel():
            raise ValueError(
                f"invalid Nymeria motion-to-DINO mapping/window for {seq_name!r}: "
                f"mapping={tuple(mapping.shape)}, window=[{start_idx}, {end_idx}]"
            )
        mapping = mapping[start_idx : end_idx + 1]
        sidecar_tensor = torch.as_tensor(sidecar_feats[seq_name])
        if sidecar_tensor.ndim != 3 or tuple(sidecar_tensor.shape[1:]) != (5, 1024):
            raise ValueError(
                f"Nymeria DINO sidecar sequence {seq_name!r} must be [N,5,1024], "
                f"got {tuple(sidecar_tensor.shape)}"
            )
        if mapping.numel() and (mapping.min() < 0 or mapping.max() >= sidecar_tensor.shape[0]):
            raise ValueError(f"Nymeria DINO mapping is out of range for {seq_name!r}")
        feats = sidecar_tensor.index_select(0, mapping).float()

        valid_by_dino = self.nymeria_img_feats.get("valid", {}).get(seq_name)
        if valid_by_dino is None:
            valid_img_embs = torch.ones(feats.shape[0], dtype=torch.long)
        else:
            valid_img_embs = torch.as_tensor(valid_by_dino, dtype=torch.long).index_select(0, mapping)
        if self.img_feat_type == "dinov2":
            feats = feats[:, 0]
        elif self.img_feat_type != "dinov2_reg":
            raise ValueError(f"Nymeria DINO sidecar is incompatible with {self.img_feat_type!r}")
        return feats, valid_img_embs

    def get_img_feats(self, seq_name, start_idx, end_idx):
        take_name, st, en = seq_name.split("___")
        start_idx = start_idx + int(st)
        end_idx = end_idx + int(st)
        assert end_idx <= int(en)
        del seq_name

        if self.img_feat_type in ["dinov2", "dinov2_reg"]:
            start_idx = start_idx // 3
            end_idx = end_idx // 3
            embs = (
                self.ego_img_feats["feats"][take_name][start_idx // 2 : end_idx // 2 + 1].clone().float()
            )  # T/2 x 5 x 1024 because dino feats are at 5 fps

            # duplicate the embs to make it 10 fps
            # assert start_idx % 2 == 0
            embs = embs[:, None].repeat(1, 2, 1, 1).flatten(0, 1)  # T x 5 x 1024
            if start_idx % 2 != 0:
                embs = embs[1:]

            if embs.shape[0] == (end_idx - start_idx + 2):  # handle partial segment
                embs = embs[:-1]

            assert embs.shape[0] == (end_idx - start_idx + 1)

            valid_img_embs = torch.ones(embs.shape[0], dtype=torch.long)
            if self.img_feat_type == "dinov2":  # remove register tokens
                embs = embs[:, 0]  # T x 1024
            return embs, valid_img_embs
