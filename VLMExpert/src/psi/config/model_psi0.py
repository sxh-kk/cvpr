import os
from pathlib import Path
from tyro.conf import subcommand as cmd
from typing import Union, Annotated, Optional, List, Literal
from psi.config.config import ModelConfig
from pydantic import Field, model_validator

class Psi0ModelConfig(ModelConfig):
    ######################### hfm_action ############################
    action_architecture: Literal["psi0_joint", "dual_source_v1"] = "psi0_joint"
    # The motion source always supplies all 11 tokens of width 768.
    # Leave the checkpoint unset for dependency injection; actual use must resolve it.
    motion_checkpoint: Optional[str] = None
    motion_use_ema: bool = True
    motion_feature_seed: int = 0
    tune_motion: bool = True
    # Initial choice for our new branch; the original Psi0 has no MotionExpert LR.
    motion_expert_lr: float = Field(default=1e-5, gt=0, allow_inf_nan=False)
    # pretrained_model_name_or_path: str = "stabilityai/stable-diffusion-3-medium-diffusers"
    resnet_store_path: str | None = None # = "cache/checkpoints/resnet18/IN_1M_resnet18.pth"
    pretrained_action_header_path: str | None = None
    # precomputed_text_encodings: str = "cache/precomputed_text_encodings.pkl"
    # ckpt_step: Optional[int] = None

    rtc: bool = False
    max_delay: int = 8

    action_dim: int = 7 #36 # Da
    action_chunk_size: int = 6 #30 # Tp
    action_exec_horizon: int = 6 #30 # Ta
    observation_horizon: int = 1 # To past frames

    img_chunk: int = 1 # ?
    n_cams: int = 1 
    use_obs: str = "add_token"
    # DEPRECATED: kept only so legacy checkpoints load.
    dropout: float = 0.1
    noise_scheduler: str = "flow"
    train_diffusion_steps: int = 1000
    eval_diffusion_steps: int = 10
    share_cam_features: bool = False
    early_fusion: bool = False

    # observations
    odim: int = 15 # 32

    # conditions
    n_conditions: Optional[int] = 1
    token_fusion: Optional[str] = "concat"  # "concat", "cross", "perceiver"

    # tune_vision_backbone: bool = False
    # vision_backbone_lr: float = 1e-5

    """
        training loss weight for xyz, rpy, and gripper
        should all sumed to 1 in total, eg 0.1x3+0.2x3+0.1x1=1
    """
    loss_w: List[float] = Field(default_factory=lambda: [0.1, 0.2, 0.1])

    # noise nets
    time_dim: int = 256
    hidden_dim: int = 1536
    num_blocks: int = 6
    dim_feedforward: int = 2048
    nhead: int = 24
    activation: str = "gelu"

    view_feature_dim: int = 1920 # views feature dim for views and/ or vlm token dim
    use_film: bool = False
    combined_temb: bool = False

    # combined_temb consumes an SD3-style `pooled_projections` vector (global adaLN
    # conditioning). It is produced by a FROZEN text encoder over the instruction.
    pooled_text_encoder: Optional[str] = None            # None | "clip"
    pooled_text_encoder_path: str = "openai/clip-vit-large-patch14"
    pooled_projection_dim: int = 2048
    # Optional on-disk cache of {instruction: pooled_emb}. Loaded at startup if it
    # exists and re-saved after precompute, so the frozen embeddings are computed
    # once and reused across runs (offline precompute). None = in-memory only.
    pooled_cache_path: Optional[str] = None

    """
        Final action head: LayerNorm + (1+scale) identity path, zero-init modulation.
        Set False only to reproduce checkpoints trained with the old `x * scale`
        head, which could be unstable when multi-task finetuning.
    """
    final_layer_norm: bool = True

    """
        Query/key normalization inside the action-head attention ("rms_norm",
        "layer_norm", or None), important for stabilizing the multi-task finetuning.
    """
    qk_norm: Optional[str] = None

    use_dit: bool = False

    # layerwise  VLM conditioning fusion with action features
    vlm_layer_indices: Optional[List[int]] = None

    # Random state drop (train-only): with this probability the proprioceptive state is dropped.
    # Default mode (state token in the VLM context): the token is masked out of attention.
    # With state_as_action_token: the normalized state VECTOR is zeroed before projection, so the
    # token, its bias and its positional embedding stay in place whether or not it was dropped.
    state_drop_prob: float = 0.0

    # Prepend the proprio state as token 0 of the ACTION stream ([state | a_1..a_Tp]) so it joins
    # the action self-attention and AdaLN in every block, instead of being one token among the
    # VLM context. 
    state_as_action_token: bool = False

    # With state_as_action_token: a dropped state (state_drop_prob) is replaced by a LEARNED
    # null token instead of the zero vector's projection (Linear bias + state_pos). The zero
    # vector is a legitimate pose in normalized space, so "no state" and "neutral joints" were
    # aliased; a dedicated token lets the policy learn a distinct "state missing" embedding.
    # Fresh parameter (missing from older action-header checkpoints; strict=False loads it).
    state_null_token: bool = False

    # DEPRECATED: kept only so legacy checkpoints load; use state_drop_prob instead.
    state_feature_dropout: float = 0.2

    ################### qwen3vl ####################
    # Training Schedule
    weight_decay: float = 0.01 # L2 regularization strength

    # Core Arguments
    model_name_or_path: str = "Qwen/Qwen3-VL-2B-Instruct"  # First load, for initalization
    # vlm_run_dir: str | None = None
    vlm_ckpt_step: str | None = None
    tune_vlm: bool = False
    # Dual-source action loss consumes the complete outer Qwen hidden states,
    # not vocabulary logits. Keep that token path and project only one unused
    # logit position when explicitly enabled; checkpoint tensors stay unchanged.
    vlm_skip_unused_logits: bool = False


    tune_mm_llm: bool = False                   # [TrainingArguments] Train LLM or not
    tune_mm_vision: bool = False                # [TrainingArguments] Train VIT or not
    tune_mm_mlp: bool = False                   # [TrainingArguments] Train MLP or not
    # dataset_use: str = "egodex"               # [DataArguments] Dataset specification
    # output_dir: str = "./outputs/hfm_qwen3vl" # Output directory for checkpoints
    # cache_dir: str = "./cache/models"         # [TrainingArguments] Model cache location
    gradient_checkpointing: bool = False  # [TrainingArguments] Enable gradient checkpointing

    # Learning Rate Configuration.
    lang_backbone_lr: float = 1e-7  # [TrainingArguments] LLM-specific LR
    mm_projector_lr: float = 1e-5   # [TrainingArguments] Projector-specific LR
    vision_tower_lr: float = 1e-6   # [TrainingArguments] Vision encoder LR

    optim: str = "adamw_torch"      # [TrainingArguments] Optimizer selection

    # Sequence Configuration
    model_max_length: int = 4096 # [TrainingArguments] Max sequence length
    data_flatten: bool = True    # [DataArguments] Concatenate batch sequences
    data_packing: bool = True    # [DataArguments] Using packing data

    # Image Processing
    max_pixels: int = 576 * 28 * 28  # [DataArguments] Max image pixels (H*W) for image
    min_pixels: int = 16 * 28 * 28   # [DataArguments] Min image pixels for image


    # lora_r: int = 8                          # [TrainingArguments] LoRA r
    # lora_alpha: int= 16                      # [TrainingArguments] LoRA alpha
    # lora_dropout: float = 0.0                 # [TrainingArguments] LoRA dropout

    @model_validator(mode="after")
    def _validate_action_architecture(self) -> "Psi0ModelConfig":
        if self.vlm_skip_unused_logits and self.action_architecture != "dual_source_v1":
            raise ValueError("vlm_skip_unused_logits requires action_architecture=dual_source_v1")
        if self.action_architecture != "dual_source_v1":
            return self
        incompatible = [
            name for name in ("combined_temb", "use_dit", "use_film")
            if getattr(self, name)
        ]
        if self.vlm_layer_indices is not None:
            incompatible.append("vlm_layer_indices (must be None: last VLM layer only)")
        if self.pooled_text_encoder is not None:
            incompatible.append("pooled_text_encoder (must be None: no CLIP branch)")
        if self.n_conditions != 0:
            incompatible.append("n_conditions (must be 0)")
        if incompatible:
            raise ValueError("dual_source_v1 requires last-layer VLM and independent motion conditioning; "
                             "incompatible settings: " + ", ".join(incompatible))
        if not 0 <= self.state_drop_prob <= 1:
            raise ValueError("dual_source_v1 state_drop_prob must be between 0 and 1")
        if self.state_drop_prob > 0 and not self.state_null_token:
            raise ValueError("dual_source_v1 state_drop_prob > 0 requires state_null_token=True")
        return self

    @model_validator(mode="after")
    def _resolve_psi_home_paths(self) -> "Psi0ModelConfig":
        psi_home = os.environ.get("PSI_HOME", "/psi")
        if psi_home is None:
            return self
        def _resolve(p: str | None) -> str | None:
            if p is None or Path(p).is_absolute():
                return p
            candidate = Path(psi_home) / p
            return str(candidate) if candidate.exists() else p
        self.model_name_or_path = _resolve(self.model_name_or_path)
        self.pretrained_action_header_path = _resolve(self.pretrained_action_header_path)
        self.resnet_store_path = _resolve(self.resnet_store_path)
        self.motion_checkpoint = _resolve(self.motion_checkpoint)
        return self


def motion_model_kwargs(model_cfg, batch) -> dict:
    """Forward genuine six-frame DINO history only for the dual-source action model.

    Datasets must supply these features from aligned historical RGB. Current-only
    images and future label masks cannot substitute for the missing history.
    """
    if getattr(model_cfg, "action_architecture", "psi0_joint") != "dual_source_v1":
        return {}
    history = batch.get("motion_img_embs")
    if history is None:
        raise ValueError("dual_source_v1 requires batch['motion_img_embs'] [B,6,5,1024] from six "
                         "real 5Hz historical RGB frames ending at the VLM current frame; "
                         "the current dataset must provide this history, without repeating the "
                         "current image or using future labels")
    import torch
    if not isinstance(history, torch.Tensor) or not history.is_floating_point():
        raise ValueError("motion_img_embs must be a floating-point tensor [B,6,5,1024]")
    if history.ndim != 4 or tuple(history.shape[1:]) != (6, 5, 1024) or history.shape[0] < 1:
        raise ValueError("motion_img_embs must have shape [B,6,5,1024] with B >= 1")
    prefix = batch.get("input_ids")
    if prefix is not None and history.shape[0] != prefix.shape[0]:
        raise ValueError("motion_img_embs batch must match the VLM input_ids batch")
    return {"motion_img_embs": history}
