"""Training loop for the retained text-conditioned E7 K12 400M model."""

import math
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR, StepLR

from ..config import as_dict, architecture_metadata
from ..checkpoints import validate_checkpoint_architecture
from ..model import UniEgoMotion
from ..mydiffusion.flow_matching import FlowMatching
from ..mydiffusion.mask_utils import apply_feature_mask


class UEM_Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters(as_dict(cfg))
        self.cfg = cfg
        self.window = cfg.DATA.WINDOW
        self.is_lstm = False
        self.model = UniEgoMotion(cfg)
        from ..dataset.normalization import MotionNormalizer
        self.normalizer = MotionNormalizer(cfg.DATA.STATS_PATH, clip=cfg.DATA.NORMALIZATION_CLIP)
        self.flow = FlowMatching(
            num_steps=cfg.FLOW.NUM_STEPS, solver=cfg.FLOW.SOLVER,
            beta_alpha=cfg.FLOW.BETA_ALPHA, beta_beta=cfg.FLOW.BETA_BETA,
            t_min=cfg.FLOW.T_MIN, prediction_type=cfg.FLOW.PREDICTION_TYPE,
            global_weight=cfg.FLOW.GLOBAL_WEIGHT,
            global_feature_start=cfg.FLOW.GLOBAL_FEATURE_START,
            global_feature_end=cfg.FLOW.GLOBAL_FEATURE_END,
        )

    def _set_motion_expert_train_phase(self, epoch):
        """Progressively expose routed, shared and backbone parameters."""
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        if self.is_lstm or expert_cfg is None or not bool(getattr(expert_cfg, "ENABLED", False)):
            return
        shared_epoch = int(getattr(expert_cfg, "SHARED_UNFREEZE_EPOCH", 30))
        for name, parameter in self.model.named_parameters():
            if name.startswith("embed_text_cond.") or name == "mask_tokens.text":
                parameter.requires_grad = False
                continue
            if name.startswith(("text_",)):
                # Language adapters are the only new trainable path and must
                # learn during the routed warm-up; delaying them until the
                # shared expert unfreezes would waste the text supervision.
                parameter.requires_grad = True
                continue
            if ".ff." not in name:
                # Keep the original input/attention/condition backbone frozen
                # until the routed branch has learned a useful partition.
                parameter.requires_grad = epoch >= shared_epoch
                continue
            is_router_or_routed = ".ff.router" in name or ".ff.routed_experts." in name or ".ff.routed_gate" in name
            is_shared = ".ff.shared." in name
            if is_router_or_routed:
                parameter.requires_grad = True
            elif is_shared:
                parameter.requires_grad = epoch >= shared_epoch
            else:
                parameter.requires_grad = epoch >= shared_epoch
        if epoch == 0:
            logger.warning(
                "Motion Expert shared/backbone/global parameters train from "
                f"epoch {shared_epoch}; routed and text parameters train immediately."
            )

    def on_train_epoch_start(self):
        self._set_motion_expert_train_phase(int(self.current_epoch))

    def configure_optimizers(self):
        fused = getattr(self.cfg.TRAIN, "FUSED_ADAMW", False) and torch.cuda.is_available()
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        motion_expert_enabled = bool(
            expert_cfg is not None and getattr(expert_cfg, "ENABLED", False)
        )

        def parameter_group(name):
            if not motion_expert_enabled:
                return "main"
            if name.startswith("text_"):
                return "text"
            if any(
                marker in name
                for marker in ("ff.router", "ff.routed_experts.", "ff.routed_gate")
            ):
                return "routed"
            return "unfrozen"

        grouped = {"main": [], "routed": [], "text": [], "unfrozen": []}
        for name, parameter in self.model.named_parameters():
            grouped[parameter_group(name)].append(parameter)
        if motion_expert_enabled:
            scales = {
                "routed": float(getattr(expert_cfg, "ROUTED_LR_SCALE", 1.0)),
                "text": float(getattr(expert_cfg, "TEXT_LR_SCALE", 1.0)),
                "unfrozen": float(getattr(expert_cfg, "UNFROZEN_LR_SCALE", 1.0)),
            }
            if any(scale < 0.0 for scale in scales.values()):
                raise ValueError(f"Motion Expert learning-rate scales must be non-negative, got {scales}")
            param_groups = [
                {
                    "params": grouped[name],
                    "lr": self.cfg.TRAIN.LR * scales[name],
                    "lr_scale": scales[name],
                    "phase": name,
                }
                for name in ("routed", "text", "unfrozen")
                if grouped[name]
            ]
        else:
            param_groups = [{"params": grouped["main"], "lr": self.cfg.TRAIN.LR, "lr_scale": 1.0, "phase": "main"}]
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.cfg.TRAIN.LR,
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY,
            fused=fused,
        )

        scheduler_name = getattr(self.cfg.TRAIN, "SCHEDULER", "step").lower()
        if scheduler_name == "cosine_warmup":
            warmup_epochs = max(0, getattr(self.cfg.TRAIN, "WARMUP_EPOCHS", 0))
            total_epochs = getattr(self.cfg.TRAIN, "SCHEDULER_TOTAL_EPOCHS", 0)
            if total_epochs <= 0:
                total_epochs = self.cfg.TRAIN.NUM_EPOCHS
            min_lr_ratio = getattr(self.cfg.TRAIN, "MIN_LR_RATIO", 0.1)

            def common_lr_lambda(epoch):
                if warmup_epochs > 0 and epoch < warmup_epochs:
                    return float(epoch + 1) / float(warmup_epochs)
                progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs - 1)
                cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            ramp_epochs = int(getattr(expert_cfg, "UNFREEZE_RAMP_EPOCHS", 0)) if motion_expert_enabled else 0
            shared_epoch = int(getattr(expert_cfg, "SHARED_UNFREEZE_EPOCH", 0)) if motion_expert_enabled else 0

            def phase_lr_lambda(epoch, phase):
                value = common_lr_lambda(epoch)
                if phase != "unfrozen" or ramp_epochs <= 0:
                    return value
                if epoch < shared_epoch:
                    return 0.0
                ramp = min(1.0, float(epoch - shared_epoch + 1) / float(ramp_epochs))
                return value * ramp

            scheduler = LambdaLR(
                optimizer,
                [lambda epoch, phase=group.get("phase", "main"): phase_lr_lambda(epoch, phase) for group in optimizer.param_groups],
            )
        else:
            scheduler = StepLR(optimizer, max(1, self.cfg.TRAIN.NUM_EPOCHS - 50), gamma=0.1)
        return [optimizer], [
            {
                "scheduler": scheduler,
                "interval": "epoch",
            }
        ]

    def on_train_start(self):
        self._set_motion_expert_train_phase(int(self.current_epoch))
        if self.cfg.MODEL.CKPT_PATH and not self.cfg.TRAIN.USE_CKPT_LR:
            for group in self.optimizers().param_groups:
                group["lr"] = self.cfg.TRAIN.LR * float(group.get("lr_scale", 1.0))
                group["weight_decay"] = self.cfg.TRAIN.WEIGHT_DECAY

    def _step(self, batch, mode):
        x = batch["misc"]["motion"]
        losses = self.flow.training_losses(self.model, x, model_kwargs={"y": batch["y"]})
        loss = losses["loss"].mean()
        aux = self.model.get_moe_auxiliary_loss()
        weight = float(self.cfg.MODEL.MOTION_EXPERT.LOAD_BALANCE_WEIGHT)
        metrics = {"flow_loss": loss, "local_mse": losses["local_mse"], "global_mse": losses["global_mse"]}
        if aux is not None:
            metrics["moe_aux_loss"] = aux
            loss = loss + weight * aux
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite {mode} loss in {batch['misc'].get('seq_name')}")
        metrics["loss"] = loss
        if "text_valid" in batch["y"]:
            metrics["text_valid_fraction"] = batch["y"]["text_valid"].float()
        for name, value in metrics.items():
            self.log(f"{mode}/{name}", value.mean(), on_step=mode == "train",
                     on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        if mode == "val":
            sources = batch["misc"].get("source", ["ee4d"] * x.shape[0])
            for index, source in enumerate(("ee4d", "nymeriaplus")):
                mask = torch.tensor([name == source for name in sources], device=x.device)
                self._validation_source_metrics[index, :3] += torch.stack([
                    losses[key][mask].sum() for key in ("loss", "local_mse", "global_mse")
                ]).double()
                self._validation_source_metrics[index, 3] += mask.sum()
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["motionexpert_architecture"] = architecture_metadata(self.cfg)
        checkpoint["motionexpert_normalization"] = self.normalizer.stats
        checkpoint["motionexpert_normalization_clip"] = self.normalizer.clip

    def on_load_checkpoint(self, checkpoint):
        validate_checkpoint_architecture(self.cfg, checkpoint)

    def on_validation_epoch_start(self):
        self._validation_source_metrics = torch.zeros(2, 4, device=self.device, dtype=torch.float64)

    def on_validation_epoch_end(self):
        totals = self._validation_source_metrics
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
        for index, source in enumerate(("ee4d", "nymeriaplus")):
            if totals[index, 3] > 0:
                for column, key in enumerate(("flow_loss", "local_mse", "global_mse")):
                    self.log(f"val/{source}/{key}", totals[index, column] / totals[index, 3], sync_dist=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, "val")
        if self.cfg.MODEL.GLOBAL_TOKEN_COUNT:
            x = batch["misc"]["motion"]
            generator = torch.Generator(device=x.device).manual_seed(int(self.cfg.TRAIN.SEED) + batch_idx)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            noise = apply_feature_mask(noise, batch["y"])
            t = torch.ones(x.shape[0], device=x.device)
            _, hidden, global_hidden = self.model(noise, t, batch["y"], return_features=True)
            valid = batch["y"]["valid_frames"].bool()
            diagnostics = {
                "global_t1_norm": global_hidden.float().norm(dim=-1).mean(),
                "global_t1_batch_std": global_hidden.float().std(dim=0, unbiased=False).mean(),
                "hidden_t1_norm": hidden.float().norm(dim=-1)[valid].mean(),
            }
            for name, value in diagnostics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True,
                         sync_dist=True, batch_size=x.shape[0])
        return loss

    def sample(self, y, B=None, cond_scale=None, return_all_pred_xstart=False,
               return_one_step_hidden=False, return_one_step_features=False):
        batch, frames = y["valid_frames"].shape
        if B is not None and B != batch:
            raise ValueError("B does not match the conditioning batch")
        return self.flow.sample_loop(
            self.model, (batch, frames, 153),
            model_kwargs={"y": y, "cond_scale": cond_scale}, progress=False,
            return_all_pred_xstart=return_all_pred_xstart,
            return_one_step_hidden=return_one_step_hidden,
            return_one_step_features=return_one_step_features,
        )
