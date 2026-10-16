"""Version the pooled motion readout and reject silent legacy restoration."""

from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from MotionExpert.checkpoints import checkpoint_config, validate_checkpoint_architecture
from MotionExpert.config import (MODEL_ARCHITECTURE_VERSION, architecture_metadata,
                                 as_dict, smplh_motion_expert_config, validate_architecture)
from MotionExpert.model import uniegomotion
from MotionExpert.tests.fixtures import InspectDecoder, conditioning


def checkpoint(cfg):
    return {"hyper_parameters": as_dict(cfg),
            "motionexpert_architecture": architecture_metadata(cfg),
            "state_dict": {"model.fixture": torch.ones(1)}}


def test_architecture_explicitly_records_pooling_and_normalization():
    cfg = smplh_motion_expert_config()
    metadata = architecture_metadata(cfg)
    assert metadata["architecture"] == MODEL_ARCHITECTURE_VERSION
    assert metadata["global_token_count"] == 0 and metadata["num_prefix_tokens"] == 1
    assert metadata["global_readout"] == "motion_hidden_mean"
    assert metadata["global_pooling"] == {
        "reduction": "mean", "token_axis": 1, "scope": "all_motion_hidden",
        "token_count": 11, "uses_label_validity": False,
    }
    assert metadata["attention_normalization"] == "qkv_prenorm_v2"
    assert checkpoint_config(checkpoint(cfg)).MODEL.GLOBAL_TOKEN_COUNT == 0


@pytest.mark.parametrize("kind", ["weights", "metadata", "both"])
def test_legacy_architecture_rejection_is_explicit(kind):
    cfg = smplh_motion_expert_config()
    source = checkpoint(cfg)
    if kind in ("weights", "both"):
        source["state_dict"]["model.global_token"] = torch.zeros(768)
    if kind in ("metadata", "both"):
        source["motionexpert_architecture"]["architecture"] = "e7_smplh12_hands21_global_v1"
    with pytest.raises(ValueError, match="legacy GLOBAL-token/attention"):
        checkpoint_config(source)
    with pytest.raises(ValueError, match="legacy GLOBAL-token/attention"):
        validate_checkpoint_architecture(cfg, source)


@pytest.mark.parametrize("field,value", [
    ("GLOBAL_TOKEN_COUNT", 1), ("GLOBAL_TOKEN_COUNT", False),
    ("GLOBAL_READOUT", "learnable_token"), ("ATTENTION_NORM_VERSION", "query_only"),
])
def test_old_or_ambiguous_model_options_are_rejected(field, value):
    cfg = smplh_motion_expert_config()
    setattr(cfg.MODEL, field, value)
    with pytest.raises(ValueError):
        validate_architecture(cfg)


@pytest.mark.parametrize("cond_scale", [None, 1.7])
def test_mean_pool_excludes_time_prefix_and_preserves_all_motion_gradients(cond_scale):
    cfg = smplh_motion_expert_config()
    with patch.object(uniegomotion, "MoEDecoderBlock", InspectDecoder):
        model = uniegomotion.UniEgoMotion(cfg, dropout=0).eval()
    state = torch.randn(2, 11, 245, requires_grad=True)
    y = conditioning(batch=2, frames=11)
    y["feature_valid_mask"] = torch.zeros(2, 11, 245, dtype=torch.bool)
    _, hidden, global_hidden = model(state, torch.ones(2), y, return_features=True,
                                     cond_scale=cond_scale)
    assert not any("global_token" in name for name, _ in model.named_parameters())
    assert model.num_prefix_tokens == 1
    assert model.tsfm[-1].call["x"].shape == (2, 12, 768)
    assert hidden.shape == (2, 11, 768)
    torch.testing.assert_close(global_hidden, hidden.mean(1), rtol=0, atol=0)
    global_hidden.square().mean().backward()
    assert torch.isfinite(state.grad).all()
    assert (state.grad.abs().sum(-1) > 0).all()


def test_activation_diagnostics_are_detached_token_maxima_and_reset_outside_training():
    cfg = smplh_motion_expert_config()
    with patch.object(uniegomotion, "MoEDecoderBlock", InspectDecoder):
        model = uniegomotion.UniEgoMotion(cfg, dropout=0)
    state = torch.randn(1, 11, 245, requires_grad=True)
    y = conditioning(batch=1, frames=11)
    model.train()
    _, hidden, global_hidden = model(state, torch.ones(1), y, return_features=True)
    rms = model._safety_activation_rms
    assert rms.shape == (12,) and not rms.requires_grad
    assert rms[-1] >= hidden.detach().float().square().mean(-1).sqrt().amax()
    assert global_hidden.requires_grad
    model.eval()
    model(state, torch.ones(1), y)
    assert model._safety_activation_rms is None
    model.train()
    cfg.TRAIN.SAFETY_ENABLED = False
    model(state, torch.ones(1), y)
    assert model._safety_activation_rms is None


@pytest.mark.parametrize("field,value", [
    ("SAFETY_ENABLED", 1), ("SAFETY_SAVE_BATCH", "true"),
    ("SAFETY_MAX_LOSS", float("nan")), ("SAFETY_MAX_GRAD_NORM", 0),
    ("SAFETY_MAX_ACTIVATION_RMS", float("inf")), ("SAFETY_ROLLING_MULTIPLIER", True),
    ("SAFETY_ROLLING_MIN_LOSS", -1), ("SAFETY_ROLLING_WINDOW", 0),
    ("SAFETY_ROLLING_WARMUP_STEPS", 129),
])
def test_invalid_training_guard_configuration_is_rejected(field, value):
    cfg = smplh_motion_expert_config()
    setattr(cfg.TRAIN, field, value)
    with pytest.raises(ValueError, match="SAFETY"):
        validate_architecture(cfg)


def test_metadata_cannot_hide_changed_context_normalization():
    cfg = smplh_motion_expert_config()
    source = checkpoint(cfg)
    source["motionexpert_architecture"] = deepcopy(source["motionexpert_architecture"])
    source["motionexpert_architecture"]["attention_normalization"] = "query_only"
    with pytest.raises(ValueError, match="architecture metadata"):
        checkpoint_config(source)
