"""Small preprocessed-format fixtures for exercising the complete data path."""

import copy
from pathlib import Path

import torch
import yaml

from MotionExpert.config import ROOT
from MotionExpert.dataset.text_conditions import TEXT_FEATURE_FORMAT


def write_fixture(root, *, global_token=False):
    root = Path(root)
    folder = root / "uniegomotion"
    folder.mkdir(parents=True, exist_ok=True)
    frames = 100
    rotation = torch.tensor([1., 0., 0., 0., 1., 0.])
    kp3d = torch.zeros(frames, 76, 3)
    kp3d[..., 2] = 1
    trajectory = torch.cat((rotation.repeat(frames, 1), torch.zeros(frames, 3)), dim=1)
    trajectory[:, -1] = 1.6
    record = {"num_frames": frames, "floor_height": 0.1, "kp3d": kp3d, "aria_traj": trajectory,
              "smpl_params": {"global_orient": rotation.repeat(frames, 1),
                              "body_pose": rotation.repeat(frames, 21, 1),
                              "transl": torch.zeros(frames, 3), "betas": torch.zeros(1, 10),
                              "left_hand_pose": torch.zeros(frames, 12), "right_hand_pose": torch.zeros(frames, 12)}}
    ee = "take___0___297"
    ny = "nymeria_fixture"
    ny_record = copy.deepcopy(record)
    ny_record["source"] = "nymeriaplus"
    ny_record["motion_to_dino_idx"] = torch.arange(frames) // 2
    valid_features = torch.ones(243, dtype=torch.bool)
    valid_features[207:231] = False
    valid_features[233:] = False
    ny_record["feature_valid_mask"] = valid_features
    features = torch.arange(50).float()[:, None, None].expand(-1, 5, 1024).clone()
    for split in ("train", "val"):
        torch.save({ee: record}, folder / f"ee_{split}.pt")
        torch.save({ny: ny_record}, folder / f"nymeria_{split}.pt")
        torch.save({"feats": {"take": features}}, folder / f"egoview_dinov2_{split}.pt")
        torch.save({"feats": {ny: features}}, folder / f"egoview_dinov2_nymeria_{split}.pt")
        torch.save({"format": TEXT_FEATURE_FORMAT, "token_count": 16, "feature_dim": 768,
                    "empty_id": 0, "pooled": torch.stack((torch.zeros(768), torch.ones(768))),
                    "tokens": torch.stack((torch.zeros(16, 768), torch.ones(16, 768))),
                    "token_mask": torch.tensor([[False] * 16, [True] * 16]), "valid": torch.tensor([False, True]),
                    "index": {name: {str(start): 1 for start in range(0, 80, 20)} for name in (ee, ny)}},
                   folder / f"clip_text_{split}_16.pt")
    stats = {f"{kind}_{key}": torch.full((dim,), 1. if key == "std" else 0.)
             for kind, dim in (("motion", 243), ("traj", 18)) for key in ("mean", "std", "min", "max")}
    torch.save(stats, folder / "stats.pt")
    filename = "e7_k12_400m_global_text.yaml" if global_token else "e7_k12_400m_text.yaml"
    cfg = yaml.safe_load((ROOT / "configs" / filename).read_text())
    cfg["DATA"].update(DATA_DIR=str(root.resolve()), STATS_PATH="uniegomotion/stats.pt",
                       STATS_MODE="ee4d_compat", SEQUENCE_MANIFEST=None, NUM_WORKERS=0, BATCH_SIZE=2)
    cfg["DATA"]["SOURCE_SUBSAMPLE"]["ENABLED"] = False
    cfg["TRAIN"].update(NUM_GPUS=1, NUM_EPOCHS=2, ACCUMULATE_GRAD_BATCHES=1,
                        CHECK_VAL_EVERY_N_EPOCHS=1, LOG_EVERY_N_STEPS=1)
    cfg_path = root / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return cfg_path
