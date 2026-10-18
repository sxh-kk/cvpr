"""Frozen, image-only MotionExpert features for the dual-source action expert."""

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class FrozenMotionFeatures:
    """All eleven generated motion positions, including five forecast positions."""

    tokens: torch.Tensor
    _pooled_tokens: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.tokens, torch.Tensor)
                or self.tokens.ndim != 3 or self.tokens.shape[0] < 1
                or self.tokens.shape[1:] != (11, 768)
                or not self.tokens.is_floating_point()):
            raise ValueError("Motion tokens must be floating-point [B,11,768]")
        if self.tokens.requires_grad or torch.is_inference(self.tokens):
            # Constants consumed by trainable K/V projections must be normal
            # tensors, not inference-mode tensors that cannot be saved by autograd.
            with torch.inference_mode(False):
                object.__setattr__(self, "tokens", self.tokens.detach().clone())
        if not torch.isfinite(self.tokens).all():
            raise ValueError("Motion tokens must be finite")
        with torch.inference_mode(False), torch.no_grad():
            dtype = torch.float64 if self.tokens.dtype == torch.float64 else torch.float32
            pooled = self.tokens.mean(dim=1, dtype=dtype).to(self.tokens.dtype)
        object.__setattr__(self, "_pooled_tokens", pooled)

    @property
    def p_m(self) -> torch.Tensor:
        return self._pooled_tokens


class FrozenMotionConditioner(nn.Module):
    """Register the frozen motion network so device moves/checkpoints include it.

    The MotionExpert facade is not an nn.Module. Its ``model`` is the registered
    child; facade device/dtype fields are refreshed before each feature call.
    Only six observed DINO frames are accepted. Future RGB, poses, trajectories
    and supervision masks are deliberately absent from this interface.
    """

    def __init__(self, expert, *, seed: int = 0):
        super().__init__()
        if not isinstance(getattr(expert, "model", None), nn.Module):
            raise TypeError("MotionExpert must expose its network as .model")
        if not callable(getattr(expert, "forward_features", None)):
            raise TypeError("MotionExpert must expose forward_features()")
        self.model = expert.model
        self._expert = expert
        self.seed = int(seed)
        cfg = getattr(expert, "cfg", None)
        if cfg is not None and (cfg.DATA.FPS, cfg.DATA.WINDOW, cfg.DATA.FORECAST_PREFIX) != (5, 11, 6):
            raise ValueError("Dual-source conditioning requires 5Hz/history6/window11 MotionExpert")
        self.model.requires_grad_(False)
        self.train(False)

    @classmethod
    def from_checkpoint(cls, path, *, use_ema=True, seed=0):
        if not path:
            raise ValueError("dual_source_v1 requires an explicit motion_checkpoint or injected MotionExpert")
        from MotionExpert import MotionExpert

        # Construction on CPU avoids allocating a second copy on a training GPU.
        # The owning Psi0Model is moved to the desired device by its caller.
        expert = MotionExpert(weights=path, device="cpu", dtype=torch.float32, use_ema=use_ema)
        return cls(expert, seed=seed)

    def train(self, mode: bool = True):
        super().train(False)
        self.model.requires_grad_(False)
        return self

    def forward(self, img_embs: torch.Tensor) -> FrozenMotionFeatures:
        if (not isinstance(img_embs, torch.Tensor) or not img_embs.is_floating_point()
                or img_embs.ndim != 4 or img_embs.shape[0] < 1
                or img_embs.shape[1:] != (6, 5, 1024)):
            raise ValueError("motion_img_embs must be floating-point [B,6,5,1024] observed DINO features")
        if not torch.isfinite(img_embs).all():
            raise ValueError("motion_img_embs must contain finite observed DINO features")
        self.train(False)
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            self._expert.device = parameter.device
            self._expert.dtype = parameter.dtype
        # Use no_grad rather than inference_mode: downstream trainable Linear
        # layers need to save these constants to compute their weight gradients.
        with torch.no_grad():
            output = self._expert.forward_features(
                {"img_embs": img_embs}, seed=self.seed, task="fore",
                forecast_prefix=6, output_frames=11, cond_scale=None,
            )
        features = FrozenMotionFeatures(output["one_step_hidden"])
        if features.tokens.shape[0] != img_embs.shape[0]:
            raise ValueError("MotionExpert returned an incorrect batch size")
        return features
