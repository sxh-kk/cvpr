"""Small current-protocol checks; no production data or large model allocation."""
import hashlib
import json

import pytest
import torch

from ImageProcess import SharedImageProcessor
from MotionExpert.config import smplh_motion_expert_config, validate_architecture
from MotionExpert.dataset.motion_layout import motion_layout, SMPLH_REPRE_TYPE
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive, canonical_json, file_sha256
from MotionExpert.dataset.smplh_motion_dataset import SMPLHMotionDataset
from MotionExpert.dataset.smplh_representation import decode_smplh_motion
from MotionExpert.tests.fixtures import owned_archive, tiny_expert, conditioning, _json
from MotionExpert.visual_contract import checkpoint_visual_contract


def test_single_motion_layout_and_forecasting_protocol(owned_archive):
    layout = motion_layout(SMPLH_REPRE_TYPE)
    assert layout.motion_dim == 245 and len(layout.joint_indices) == 12
    assert layout.left_hand_slice == slice(117, 180) and layout.right_hand_slice == slice(180, 243)
    sample = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False, cond_img_feat=False)[0]
    assert sample["misc"]["motion"].shape == (11, 245)
    assert sample["misc"]["traj"].shape == (11, 18)
    assert sample["misc"]["observation.valid.hand_left_position"].shape == (11, 21)
    assert not sample["misc"]["feature_valid_mask"][:, 117 + 7 * 3:117 + 8 * 3].any()
    assert sample["y"]["loss_mask"].all()
    expert = tiny_expert()
    future = expert.predict_future({"img_embs": conditioning(batch=1)["img_embs"]}, seed=0)
    assert future["future_motion"].shape == (1, 5, 245)
    assert future["one_step_hidden"].shape == (1, 11, 768)
    assert future["one_step_global"].shape == (1, 768)
    for key, value in (("FPS", 15), ("REPRE_TYPE", "dense12_rot6d_hands21_contacts_v1")):
        cfg = smplh_motion_expert_config()
        setattr(cfg.DATA, key, value)
        with pytest.raises(ValueError):
            validate_architecture(cfg)


def test_retired_vision_preserves_motion_and_has_no_feature_fallback(owned_archive):
    original = SMPLHParameterArchive(owned_archive)
    entries = []
    for record in original.records():
        receipt = json.loads((owned_archive / f"episodes/{record['episode_index']:08d}/receipt.json").read_text())
        entries.append({"episode_index": record["episode_index"], "path": record["vision_path"],
                        "sha256": record["vision_sha256"], **receipt["output_identities"]["vision"]})
    _json(owned_archive / ".pipeline/retired-vision.json", {
        "format": "smplh-legacy-vision-retirement-v1", "episodes": entries,
        "info_sha256": file_sha256(owned_archive / "meta/info.json"),
        "records_manifest_sha256": hashlib.sha256(canonical_json(original.records())).hexdigest(),
    })
    for record in original.records():
        (owned_archive / record["vision_path"]).unlink()
    reopened = SMPLHParameterArchive(owned_archive)
    assert reopened.archive_fingerprint == original.archive_fingerprint
    arrays, _ = reopened.read_window(0, load_dino=False)
    assert arrays["smplh.betas"].shape == (11, 16)
    with pytest.raises(ValueError, match="VISION_PROFILE_ROOT"):
        reopened.read_window(0, load_dino=True)


def test_shared_checkpoint_contract_is_self_contained_and_required():
    process = SharedImageProcessor("256x480").metadata
    contract = {"format": "motionexpert-visual-v1", "mode": "shared_rgb", "image_process": process,
                "encoder": {"image_process": process}}
    checkpoint = {"motionexpert_visual_conditioning": contract, "hyper_parameters": {
        "DATA": {"VISION_PROFILE_ROOT": "/does-not-exist", "IMAGE_PROFILE": "256x480"}}}
    assert checkpoint_visual_contract(checkpoint) == contract
    with pytest.raises(ValueError, match="shared-image"):
        checkpoint_visual_contract({})
    checkpoint["motionexpert_visual_conditioning"] = {"format": "motionexpert-visual-v1", "mode": "legacy336"}
    with pytest.raises(ValueError, match="shared-image"):
        checkpoint_visual_contract(checkpoint)


def test_sparse_decode_recovers_observed_and_future_world_coordinates(owned_archive):
    dataset = SMPLHMotionDataset(archive_root=owned_archive, do_normalization=False, cond_img_feat=False)
    sample = dataset[0]
    observed = dataset.ret_to_full_sequence(sample)
    torch.testing.assert_close(observed["body_positions"],
        sample["misc"]["observation.state.body_world"].reshape(11, 22, 3)[:, dataset.joint_indices],
        rtol=1e-5, atol=1e-5)
    sample["pred"]["motion"] = sample["misc"]["motion"]
    future = dataset.ret_to_full_sequence(sample)
    torch.testing.assert_close(future["body_positions"], observed["body_positions"][6:], rtol=1e-5, atol=1e-5)
    assert future["left_hand"].shape == (5, 21, 3)
    assert future["right_hand"].shape == (5, 21, 3)


def test_public_forecast_decode_returns_five_frames():
    expert = tiny_expert()
    # A constant clean estimate with proper identity 6D floor rotations.
    motion = torch.zeros(1, 11, 245)
    motion[:, :, 108:114] = torch.tensor([1., 0., 0., 0., 1., 0.])
    from unittest.mock import patch
    with patch.object(expert.flow, "sample_loop", return_value={"motion": motion}):
        result = expert.predict_future(conditioning(batch=1), decode=True,
            decode_kwargs={"initial_floor_transform": torch.eye(4)})
    assert result["decoded"]["body_positions"].shape == (1, 5, 12, 3)
    assert result["decoded"]["left_hand"].shape == (1, 5, 21, 3)
    with pytest.raises(ValueError, match="T x 245"):
        decode_smplh_motion(torch.zeros(11, 244))
