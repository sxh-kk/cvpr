"""Strict restoration of the real MotionExpert positional-encoding alias."""

from collections import OrderedDict

import pytest
import torch
from torch import nn
from safetensors.torch import load_file, save_file

from MotionExpert.model.uniegomotion import PositionalEncoding, TimestepEmbedder
from psi.models.checkpoint_aliases import restore_shared_tensor_aliases


class _MotionPE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.pos_enc = PositionalEncoding(8, max_len=7)
        self.model.embed_timestep = TimestepEmbedder(8, self.model.pos_enc)


@pytest.mark.parametrize("saved_name", ["model.pos_enc.pe", "model.embed_timestep.pos_enc.pe"])
def test_real_shared_motion_pe_safetensors_roundtrip(tmp_path, saved_name):
    source, target = _MotionPE(), _MotionPE()
    # A nondefault value proves restoration uses the checkpoint, not a newly
    # generated sinusoidal buffer. The real Motion classes share this object.
    source.model.pos_enc.pe.add_(0.125)
    assert source.model.pos_enc is source.model.embed_timestep.pos_enc
    state = OrderedDict((name, value.clone()) for name, value in source.state_dict().items())
    missing = ({"model.pos_enc.pe", "model.embed_timestep.pos_enc.pe"} - {saved_name}).pop()
    del state[missing]
    path = tmp_path / "model.safetensors"
    save_file(state, path, metadata={"format": "pt"})
    loaded = load_file(path)
    with pytest.raises(RuntimeError, match="Missing key"):
        target.load_state_dict(loaded, strict=True)
    restored = restore_shared_tensor_aliases(target, loaded)
    assert missing not in loaded  # Do not mutate the checkpoint mapping.
    assert restored[missing] is restored[saved_name]
    target.load_state_dict(restored, strict=True)
    for name, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[name], value, rtol=0, atol=0)


def test_missing_real_independent_motion_weight_still_fails():
    model = _MotionPE()
    state = dict(model.state_dict())
    del state["model.embed_timestep.pos_enc.pe"]
    missing = "model.embed_timestep.time_embed.0.weight"
    del state[missing]
    restored = restore_shared_tensor_aliases(model, state)
    assert missing not in restored
    with pytest.raises(RuntimeError, match=missing):
        model.load_state_dict(restored, strict=True)


def test_conflicting_saved_aliases_are_rejected_before_loading():
    model = _MotionPE()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    state["model.embed_timestep.pos_enc.pe"].add_(1)
    before = model.model.pos_enc.pe.clone()
    with pytest.raises(RuntimeError, match="Conflicting checkpoint values"):
        restore_shared_tensor_aliases(model, state)
    torch.testing.assert_close(model.model.pos_enc.pe, before, rtol=0, atol=0)


@pytest.mark.parametrize("difference", ["storage", "offset", "stride", "dtype"])
def test_equal_shape_or_shared_storage_alone_does_not_prove_an_alias(difference):
    model = nn.Module()
    base = torch.zeros(8, 8)
    first = base[:4, :4]
    second = {
        "storage": base.clone()[:4, :4],
        "offset": base[1:5, :4],
        "stride": first.T,
        "dtype": first.view(torch.int32),
    }[difference]
    model.register_buffer("first", first)
    model.register_buffer("second", second)
    restored = restore_shared_tensor_aliases(model, {"first": first.clone()})
    assert "second" not in restored
    with pytest.raises(RuntimeError, match="second"):
        model.load_state_dict(restored, strict=True)
