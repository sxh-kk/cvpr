"""Image-only MotionExpert features with explicit trainable and frozen modes."""

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass(frozen=True)
class MotionFeatures:
    """All eleven motion tokens and their differentiable mean.

    Caller grad/inference mode is preserved. A sampling call may return
    inference tensors, but they cannot subsequently be used as training
    constants: that requires a fresh ordinary forward pass. The owning model
    also checks reused features before feeding its trainable action head.
    """

    tokens: torch.Tensor
    _pooled_tokens: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self):
        if (not isinstance(self.tokens, torch.Tensor)
                or self.tokens.ndim != 3 or self.tokens.shape[0] < 1
                or self.tokens.shape[1:] != (11, 768)
                or not self.tokens.is_floating_point()):
            raise ValueError("Motion tokens must be floating-point [B,11,768]")
        if torch.is_grad_enabled() and torch.is_inference(self.tokens):
            raise ValueError("Inference-mode Motion tokens cannot be reused for training; run a fresh forward pass")
        if not torch.isfinite(self.tokens).all():
            raise ValueError("Motion tokens must be finite")
        dtype = torch.float64 if self.tokens.dtype == torch.float64 else torch.float32
        pooled = self.tokens.mean(dim=1, dtype=dtype).to(self.tokens.dtype)
        object.__setattr__(self, "_pooled_tokens", pooled)

    @property
    def p_m(self) -> torch.Tensor:
        return self._pooled_tokens


class FrozenMotionFeatures(MotionFeatures):
    """Explicit frozen-cache compatibility type, never a trainable source pass.

    The detached normal tensors can condition a trainable action head while
    MotionExpert itself is frozen. Joint MotionExpert training must instead
    use fresh ``MotionFeatures`` carrying its source graph.
    """

    def __post_init__(self):
        if not isinstance(self.tokens, torch.Tensor):
            raise ValueError("Motion tokens must be floating-point [B,11,768]")
        with torch.inference_mode(False), torch.no_grad():
            tokens = self.tokens.detach()
            if torch.is_inference(tokens):
                tokens = tokens.clone()
            object.__setattr__(self, "tokens", tokens)
            super().__post_init__()


class MotionConditioner(nn.Module):
    """Register the motion network and follow its explicit training policy.

    The MotionExpert facade is not an nn.Module. Its ``model`` is the registered
    child; facade device/dtype fields are refreshed before each feature call.
    Only six observed DINO frames are accepted. Future RGB, poses, trajectories
    and supervision masks are deliberately absent from this interface.
    """

    # Under this interface all trajectories are hidden and every input image
    # is one of the six valid historical observations. The trajectory embedding
    # is replaced by mask_tokens.traj, and mask_tokens.clip appears only in
    # future image slots excluded by both visual attention and the ego router.
    # output_process produces body parameters, not the hidden features consumed
    # by the action loss. Keep these known inactive parameters out of optimizers.
    _INACTIVE_PREFIXES = ("output_process.", "embed_traj_cond.", "embed_traj_valid.")
    _INACTIVE_NAMES = frozenset(("mask_tokens.clip",))

    def __init__(self, expert, *, seed: int = 0, trainable: bool = True):
        super().__init__()
        if type(trainable) is not bool:
            raise ValueError("trainable must be bool")
        if not isinstance(getattr(expert, "model", None), nn.Module):
            raise TypeError("MotionExpert must expose its network as .model")
        if not callable(getattr(expert, "forward_features", None)):
            raise TypeError("MotionExpert must expose forward_features()")
        self.model = expert.model
        # Only .model is registered even if an injected facade is an nn.Module;
        # preserve the historical motion_conditioner.model.* checkpoint layout.
        object.__setattr__(self, "_expert", expert)
        self.seed = int(seed)
        self.trainable = trainable
        cfg = getattr(expert, "cfg", None)
        if cfg is not None and (cfg.DATA.FPS, cfg.DATA.WINDOW, cfg.DATA.FORECAST_PREFIX) != (5, 11, 6):
            raise ValueError("Dual-source conditioning requires 5Hz/history6/window11 MotionExpert")
        self.uses_sparse_routing = any(
            hasattr(module, "routed_experts")
            and getattr(module, "top_k", 0) < getattr(module, "num_experts", 0)
            for module in self.model.modules()
        )
        self.inactive_parameter_names = tuple(
            name for name, _ in self.model.named_parameters() if self._inactive_parameter(name)
        )
        self.train(trainable)

    @classmethod
    def _inactive_parameter(cls, name):
        return name in cls._INACTIVE_NAMES or name.startswith(cls._INACTIVE_PREFIXES)

    @property
    def needs_find_unused_parameters(self) -> bool:
        """Sparse routed experts are selected per batch, not all on every rank."""
        return self.trainable and self.uses_sparse_routing

    def _apply_trainability(self):
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(self.trainable and not self._inactive_parameter(name))

    @classmethod
    def from_checkpoint(cls, path, *, use_ema=True, seed=0, trainable=True):
        if not path:
            raise ValueError("dual_source_v1 requires an explicit motion_checkpoint or injected MotionExpert")
        from MotionExpert import MotionExpert

        # Construction on CPU avoids allocating a second copy on a training GPU.
        # The owning Psi0Model is moved to the desired device by its caller.
        expert = MotionExpert(weights=path, device="cpu", dtype=torch.float32, use_ema=use_ema)
        return cls(expert, seed=seed, trainable=trainable)

    def train(self, mode: bool = True):
        super().train(mode if self.trainable else False)
        self._apply_trainability()
        return self

    def forward(self, img_embs: torch.Tensor) -> MotionFeatures:
        if (not isinstance(img_embs, torch.Tensor) or not img_embs.is_floating_point()
                or img_embs.ndim != 4 or img_embs.shape[0] < 1
                or img_embs.shape[1:] != (6, 5, 1024)):
            raise ValueError("motion_img_embs must be floating-point [B,6,5,1024] observed DINO features")
        if not torch.isfinite(img_embs).all():
            raise ValueError("motion_img_embs must contain finite observed DINO features")
        if not self.trainable:
            self.train(False)
        parameter = next(self.model.parameters(), None)
        if parameter is not None:
            self._expert.device = parameter.device
            self._expert.dtype = parameter.dtype
        # DINO stays frozen even when the motion backbone is trainable. This
        # detach affects only the supplied feature cache, never Motion hidden.
        image_features = img_embs.detach()
        if self.trainable and torch.is_grad_enabled() and torch.is_inference(image_features):
            # A frozen online DINO encoder may use inference_mode. Its output
            # must become an ordinary constant before trainable motion Linear
            # layers save that input for their weight gradients.
            with torch.inference_mode(False):
                image_features = image_features.clone()
        conditioning = {"img_embs": image_features}
        arguments = dict(seed=self.seed, task="fore", forecast_prefix=6,
                         output_frames=11, cond_scale=None)
        if self.trainable:
            # In particular, do not enable_grad inside sampling's no_grad or
            # inference_mode scopes. eval() itself does not disable autograd.
            output = self._expert.forward_features(conditioning, **arguments)
            features = MotionFeatures(output["one_step_hidden"])
        else:
            # Constants saved by trainable K/V projections must be normal
            # tensors. Disabling inference first must precede no_grad.
            with torch.inference_mode(False), torch.no_grad():
                output = self._expert.forward_features(
                    conditioning, **arguments,
                )
                features = FrozenMotionFeatures(output["one_step_hidden"])
        if features.tokens.shape[0] != img_embs.shape[0]:
            raise ValueError("MotionExpert returned an incorrect batch size")
        return features


class FrozenMotionConditioner(MotionConditioner):
    """Compatibility constructor whose default remains a frozen source."""

    def __init__(self, expert, *, seed: int = 0, trainable: bool = False):
        super().__init__(expert, seed=seed, trainable=trainable)

    @classmethod
    def from_checkpoint(cls, path, *, use_ema=True, seed=0, trainable=False):
        return super().from_checkpoint(path, use_ema=use_ema, seed=seed, trainable=trainable)
