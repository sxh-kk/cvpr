"""Lightning precision options for fused AdamW with the existing clipping."""
from pytorch_lightning.plugins.precision import MixedPrecision, Precision
from pytorch_lightning.utilities import GradClipAlgorithmType


class BF16FusedMixedPrecision(MixedPrecision):
    """BF16 has no GradScaler; its gradients can be clipped before fused AdamW.

    Lightning 2.5's MixedPrecision rejects optimizers advertising internal AMP
    unscaling even on the BF16 path. Use its normal clipping implementation
    here, where gradients have never been scaled. FP16 keeps the stock plugin.
    """

    def __init__(self, device="cuda"):
        super().__init__(precision="bf16-mixed", device=device)

    def clip_gradients(self, optimizer, clip_val=0.0,
                       gradient_clip_algorithm=GradClipAlgorithmType.NORM):
        Precision.clip_gradients(self, optimizer, clip_val, gradient_clip_algorithm)


def trainer_precision_options(cfg, *, use_gpu=True):
    if not use_gpu:
        return {"precision": "32-true"}
    if cfg.TRAIN.PRECISION == "bf16-mixed" and getattr(cfg.TRAIN, "FUSED_ADAMW", False):
        return {"plugins": [BF16FusedMixedPrecision()]}
    return {"precision": cfg.TRAIN.PRECISION}
