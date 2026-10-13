"""SMPL-H configuration/statistics contracts independent of large assets."""
from types import SimpleNamespace
import torch
import pytest

from MotionExpert.config import (
    load_config, smplh_motion_expert_config, motion_expert_config,
    architecture_metadata, as_dict, config_from_dict, SMPLH_SCHEMA, SMPLH_MODEL_SHA256,
    K12_JOINTS, validate_architecture,
)
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.dataset.motion_layout import motion_layout
from MotionExpert.checkpoints import checkpoint_config, validate_checkpoint_architecture, validate_checkpoint_statistics
from MotionExpert.compute_joint_motion_stats import _accumulate, _finalize_accumulators
from MotionExpert.evaluate import compute_motion_metrics
from MotionExpert.merge_joint_motion_stats import _validate_partition
from MotionExpert.tests.fixtures import dense_stats


def smplh_stats(cfg=None):
    cfg = cfg or smplh_motion_expert_config()
    stats = dense_stats(cfg)
    stats.update(representation=cfg.DATA.REPRE_TYPE, motion_mean=torch.zeros(245), motion_std=torch.ones(245),
                 joint_indices=K12_JOINTS.copy(),
                 model_sha256=SMPLH_MODEL_SHA256, archive_schema_version=SMPLH_SCHEMA,
                 target_joint_basis=cfg.DATA.TARGET_JOINT_BASIS, rotation_basis_mode=cfg.DATA.ROTATION_BASIS_MODE)
    return stats


def test_default_is_smplh_and_legacy_configs_remain_explicit():
    cfg = load_config()
    assert cfg.MODEL.INPUT_FEATS == 245
    assert cfg.SPARSE_JOINTS.ENABLED
    assert cfg.SPARSE_JOINTS.INDICES == K12_JOINTS
    assert not cfg.DATA.COND_BETAS
    assert cfg.DATA.MOTION_SUPERVISION == "joint"
    assert cfg.DATA.TRAJECTORY_CONDITION_PROB == .5
    assert architecture_metadata(cfg)["forecast_protocol"]["trajectory_condition_prob"] == .5
    assert architecture_metadata(cfg)["feature_slices"] == architecture_metadata(motion_expert_config())["feature_slices"]
    assert architecture_metadata(cfg)["body_model"]["shape_usage"] == "source_fk_only"
    assert architecture_metadata(cfg)["body_model"]["shape_supervision"] is None
    assert architecture_metadata(config_from_dict(as_dict(cfg))) == architecture_metadata(cfg)
    assert load_config("MotionExpert/configs/dense_k12_global.yaml").MODEL.INPUT_FEATS == 245
    assert load_config(data_dir="/relocated/archive").DATA.SPLIT_MANIFEST == "/relocated/archive/meta/splits.json"


@pytest.mark.parametrize("field,value", [("input", 351), ("sparse", False), ("joints", list(range(22)))])
def test_smplh_cannot_change_the_sparse_network_contract(field, value):
    cfg = smplh_motion_expert_config()
    if field == "input":
        cfg.MODEL.INPUT_FEATS = value
    elif field == "sparse":
        cfg.SPARSE_JOINTS.ENABLED = value
    else:
        cfg.SPARSE_JOINTS.INDICES = value
    with pytest.raises(ValueError):
        validate_architecture(cfg)


def test_smplh_normalization_uses_new_slices_and_bound_model():
    stats = smplh_stats()
    stats["motion_mean"][108:117] = stats["traj_mean"][9:] = torch.arange(9.)
    normalizer = MotionNormalizer(stats,clip=1.)
    raw = torch.full((11,245),3.)
    raw[:,243:245] = .5
    assert torch.equal(normalizer.normalize(raw)[:,243:245],raw[:,243:245])
    unclipped = MotionNormalizer(stats)
    torch.testing.assert_close(unclipped.denormalize(unclipped.normalize(raw)),raw)
    bad = dict(stats,model_sha256="wrong")
    with pytest.raises(ValueError,match="model"):
        MotionNormalizer(bad)
    with pytest.raises(ValueError):
        validate_checkpoint_statistics(smplh_motion_expert_config(),{},MotionNormalizer(dense_stats()))
    with pytest.raises(ValueError,match="Diagnostic"):
        MotionNormalizer(dict(stats,diagnostic=True))
    MotionNormalizer(dict(stats,diagnostic=True),allow_diagnostic=True)


def test_checkpoint_cannot_cross_representation_even_with_same_backbone():
    cfg = smplh_motion_expert_config()
    state = {"input_process.weight":torch.zeros(768,245),"output_process.weight":torch.zeros(245,768),
             "global_token":torch.zeros(768)}
    checkpoint = {"state_dict":state,"hyper_parameters":as_dict(cfg),
                  "motionexpert_architecture":architecture_metadata(cfg)}
    assert checkpoint_config(checkpoint).MODEL.INPUT_FEATS == 245
    image_only_cfg = config_from_dict(as_dict(cfg))
    image_only_cfg.DATA.TRAJECTORY_CONDITION_PROB = 0.
    with pytest.raises(ValueError,match="architecture"):
        validate_checkpoint_architecture(image_only_cfg, checkpoint)
    with pytest.raises(ValueError,match="architecture"):
        validate_checkpoint_architecture(motion_expert_config(),checkpoint)
    state["output_process.weight"] = torch.zeros(246,768)
    with pytest.raises(ValueError,match="shape"):
        checkpoint_config(checkpoint)
    state["input_process.weight"] = torch.zeros(768,351)
    with pytest.raises(ValueError,match="incompatible"):
        checkpoint_config(checkpoint)


def test_smplh_shards_require_explicit_full_partition_metadata():
    shard = {"representation":smplh_motion_expert_config().DATA.REPRE_TYPE,
             "shard_index":0,"num_shards":2,"num_windows":10}
    with pytest.raises(ValueError,match="coverage"):
        _validate_partition(shard,expected_index=0,num_shards=2,path="synthetic")


def test_smplh_statistics_and_metrics_keep_k12_hands_and_future():
    cfg = smplh_motion_expert_config()
    layout = motion_layout(cfg.DATA.REPRE_TYPE)
    motion = torch.zeros(11,245)
    body = motion[:,:108].reshape(11,12,9)
    body[...,0] = body[...,4] = 1
    motion[:,108] = motion[:,112] = 1
    motion[:,117:243] = 2
    traj = torch.zeros(11,18); traj[:,9:] = motion[:,108:117]
    sample = {"y":{"valid_frames":torch.ones(11,dtype=torch.bool),"loss_mask":torch.ones(11,dtype=torch.bool)},
              "misc":{"motion":motion,"traj":traj,"feature_valid_mask":torch.ones_like(motion,dtype=torch.bool),
                      "trajectory_valid":torch.ones_like(traj,dtype=torch.bool),"source":"egoexo4d_hm","seq_name":"test"}}
    data = vars(cfg.DATA)
    dataset = SimpleNamespace(split="train",do_normalization=False,motion_feature_dim=245,repre_type=cfg.DATA.REPRE_TYPE,
        fps=5,window=11,forecast_prefix=6,sampling_step=6,train_task="fore",motion_supervision="joint",require_full_window=True,
        ground_estimation_scope="observed_prefix",image_validity_scope="observed_prefix",statistics_scope="all_valid_frames",
        min_valid_image_fraction=.5,segment_stride=5,split_fingerprint="unit",manifest={"archive_fingerprint":"unit"},
        root=__import__("pathlib").Path("/unit"),target_joint_basis=data["TARGET_JOINT_BASIS"],rotation_basis_mode=data["ROTATION_BASIS_MODE"],
        basis_mapping_fingerprint="none",contact_speed_m_s=.2,contact_height_m=.1,
        model_sha256=SMPLH_MODEL_SHA256,archive_schema_version=SMPLH_SCHEMA,get_sample=lambda index,task:sample)
    stats = _finalize_accumulators(_accumulate(dataset,[0],progress=False))
    assert stats["motion_count"].shape == (245,)
    assert (stats["motion_count"] == 11).all()
    assert (stats["motion_mean"][117:243] == 2).all()
    assert stats["joint_indices"] == K12_JOINTS
    assert stats["model_sha256"] == SMPLH_MODEL_SHA256
    MotionNormalizer(stats)
    pred = motion.clone(); pred[:6] = 500  # Evaluation must ignore reconstructed history.
    metrics = compute_motion_metrics(pred,motion,torch.ones_like(motion,dtype=torch.bool),task="fore",forecast_prefix=6,
        fps=5,floor_transform=torch.eye(4).repeat(11,1,1),representation=cfg.DATA.REPRE_TYPE)
    assert metrics["body_local_mpjpe_m"] == 0
    assert metrics["body_local_mpjpe_m_count"] == 5*12
    assert metrics["body_local_1s_mpjpe_m_count"] == 5*12
    assert "body_local_2s_mpjpe_m" not in metrics
    assert "betas_mae" not in metrics
    assert metrics["left_hand_local_mpjpe_m_count"] == 5*21
