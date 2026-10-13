"""Training and sampling for versioned body, explicit hands and shape targets."""

import math
from collections.abc import Mapping
import pytorch_lightning as pl
import torch
from loguru import logger
from torch.optim.lr_scheduler import LambdaLR, StepLR

from ..config import as_dict, architecture_metadata
from ..checkpoints import validate_checkpoint_architecture, validate_checkpoint_statistics
from ..model import UniEgoMotion
from ..mydiffusion.flow_matching import FlowMatching
from ..utils.task_conditioning import TASKS, apply_task_conditioning


TRAINING_PHASE_FORMAT = "motionexpert-training-phase-v1"
_MISSING = object()


def _config_value(config, name, default=None):
    return config.get(name, default) if isinstance(config, Mapping) else getattr(config, name, default)


def _validate_training_phase(value):
    if not isinstance(value, Mapping) or value.get("format") != TRAINING_PHASE_FORMAT:
        raise ValueError("Invalid MotionExpert training phase format")
    origin = value.get("origin")
    epoch = value.get("effective_shared_unfreeze_epoch")
    if origin not in ("from_scratch", "dense_warm_start"):
        raise ValueError("Invalid MotionExpert training origin")
    if type(epoch) is not int or epoch < 0 or (origin == "from_scratch" and epoch != 0):
        raise ValueError("Invalid effective shared unfreeze epoch for training origin")
    return {"format": TRAINING_PHASE_FORMAT, "origin": origin,
            "effective_shared_unfreeze_epoch": epoch}


def _training_phase_for_model_config(model_config):
    expert = _config_value(model_config, "MOTION_EXPERT", {})
    requested_epoch = _config_value(expert, "SHARED_UNFREEZE_EPOCH", 0)
    if type(requested_epoch) is not int or requested_epoch < 0:
        raise ValueError("SHARED_UNFREEZE_EPOCH must be a non-negative integer")
    origin = _config_value(model_config, "TRAINING_ORIGIN", _MISSING)
    epoch = _config_value(model_config, "EFFECTIVE_SHARED_UNFREEZE_EPOCH", _MISSING)
    if (origin is _MISSING) != (epoch is _MISSING):
        raise ValueError("Incomplete MotionExpert training phase configuration")
    if origin is _MISSING:
        # CKPT_PATH resumes a run; it says nothing about how that run started.
        origin = "dense_warm_start" if _config_value(model_config, "INIT_CKPT_PATH") else "from_scratch"
        epoch = requested_epoch if origin == "dense_warm_start" else 0
    phase = _validate_training_phase({"format": TRAINING_PHASE_FORMAT, "origin": origin,
                                      "effective_shared_unfreeze_epoch": epoch})
    if origin == "dense_warm_start" and epoch != requested_epoch:
        raise ValueError("Warm-start training phase disagrees with SHARED_UNFREEZE_EPOCH")
    return phase


def training_phase_from_checkpoint(checkpoint):
    """Keep a run's origin; legacy checkpoints use their original init path."""
    hyperparameters = checkpoint.get("hyper_parameters")
    if not isinstance(hyperparameters, Mapping) or not isinstance(hyperparameters.get("MODEL"), Mapping):
        raise ValueError("Checkpoint training phase requires original MODEL hyperparameters")
    configured = _training_phase_for_model_config(hyperparameters["MODEL"])
    if "motionexpert_training_phase" not in checkpoint:
        return configured
    recorded = _validate_training_phase(checkpoint["motionexpert_training_phase"])
    if recorded != configured:
        raise ValueError("Checkpoint training phase disagrees with its original hyperparameters")
    return recorded


def restore_training_phase(cfg, checkpoint):
    """Resolve resume phase before optimizers capture the schedule boundary."""
    phase = training_phase_from_checkpoint(checkpoint)
    requested_epoch = cfg.MODEL.MOTION_EXPERT.SHARED_UNFREEZE_EPOCH
    if type(requested_epoch) is not int or requested_epoch < 0:
        raise ValueError("SHARED_UNFREEZE_EPOCH must be a non-negative integer")
    if (phase["origin"] == "dense_warm_start"
            and requested_epoch != phase["effective_shared_unfreeze_epoch"]):
        raise ValueError("Resume cannot change the warm-start shared unfreeze epoch")
    original_model = checkpoint["hyper_parameters"]["MODEL"]
    original_expert = original_model.get("MOTION_EXPERT", {})
    if ("motionexpert_training_phase" not in checkpoint
            and "TRAINING_ORIGIN" not in original_model
            and original_model.get("CKPT_PATH") and not original_model.get("INIT_CKPT_PATH")
            and _config_value(original_expert, "SHARED_UNFREEZE_EPOCH", 0) > 0):
        logger.warning(
            "Legacy resumed checkpoint has no recorded training origin or init path; "
            "its original fresh/warm-start phase cannot be distinguished. Conservatively "
            "using from_scratch with shared/backbone training from epoch 0; the original "
            "warm-start schedule cannot be recovered from this checkpoint."
        )
    cfg.MODEL.TRAINING_ORIGIN = phase["origin"]
    cfg.MODEL.EFFECTIVE_SHARED_UNFREEZE_EPOCH = phase["effective_shared_unfreeze_epoch"]
    return phase


class UEM_Module(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        phase = _training_phase_for_model_config(cfg.MODEL)
        cfg.MODEL.TRAINING_ORIGIN = phase["origin"]
        cfg.MODEL.EFFECTIVE_SHARED_UNFREEZE_EPOCH = phase["effective_shared_unfreeze_epoch"]
        self.save_hyperparameters(as_dict(cfg))
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

    def _shared_unfreeze_epoch(self):
        # New dense input/output and condition projections have no pretrained
        # 153D counterpart. Fresh training must train the whole model at once.
        return _training_phase_for_model_config(self.cfg.MODEL)["effective_shared_unfreeze_epoch"]

    def _set_motion_expert_train_phase(self, epoch):
        """Progressively expose routed, shared and backbone parameters."""
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        if self.is_lstm or expert_cfg is None or not bool(getattr(expert_cfg, "ENABLED", False)):
            return
        shared_epoch = self._shared_unfreeze_epoch()
        for name, parameter in self.model.named_parameters():
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
                f"epoch {shared_epoch}; routed parameters train immediately."
            )

    def on_train_epoch_start(self):
        self._set_motion_expert_train_phase(int(self.current_epoch))
        self._train_metric_names = None
        self._train_metric_totals = None

    @staticmethod
    def _reduce_training_metrics(packed):
        """Reduce detached metric sums and their sample count in one collective."""
        result = packed.clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(result, op=torch.distributed.ReduceOp.SUM)
        return result[:-1] / result[-1].clamp_min(1)

    def _log_training_metrics(self, metrics, batch_size):
        names = tuple(metrics)
        values = torch.stack([value.detach().float().mean() for value in metrics.values()]).double()
        packed = torch.cat((values * batch_size, values.new_tensor([batch_size])))
        if getattr(self, "_train_metric_totals", None) is None:
            self._train_metric_names = names
            self._train_metric_totals = packed.clone()
        else:
            if names != self._train_metric_names:
                raise ValueError("Training metric names changed within an epoch")
            self._train_metric_totals.add_(packed)
        # Match Lightning's optimizer-step logging cadence, also across epochs
        # and resumed runs. All epoch batches are retained above, including the
        # tail after the last logged step. With accumulation, log the last
        # microbatch only, as the original step metrics did.
        trainer = getattr(self, "_trainer", None)
        if trainer is None:
            return
        interval = max(1, int(self.cfg.TRAIN.LOG_EVERY_N_STEPS))
        if (int(trainer.global_step) + 1) % interval:
            return
        if trainer.fit_loop._should_accumulate():
            return
        reduced = self._reduce_training_metrics(packed)
        for name, value in zip(names, reduced):
            self.log(f"train/{name}_step", value.float(), on_step=True,
                     on_epoch=False, sync_dist=False, batch_size=batch_size)

    def on_train_epoch_end(self):
        totals = getattr(self, "_train_metric_totals", None)
        if totals is None:
            return
        reduced = self._reduce_training_metrics(totals)
        for name, value in zip(self._train_metric_names, reduced):
            self.log(f"train/{name}_epoch", value.float(), on_step=False,
                     on_epoch=True, sync_dist=False, batch_size=1)
        self._train_metric_names = None
        self._train_metric_totals = None

    def configure_optimizers(self):
        fused = getattr(self.cfg.TRAIN, "FUSED_ADAMW", False) and torch.cuda.is_available()
        expert_cfg = getattr(self.cfg.MODEL, "MOTION_EXPERT", None)
        motion_expert_enabled = bool(
            expert_cfg is not None and getattr(expert_cfg, "ENABLED", False)
        )

        def parameter_group(name):
            if not motion_expert_enabled:
                return "main"
            if any(
                marker in name
                for marker in ("ff.router", "ff.routed_experts.", "ff.routed_gate")
            ):
                return "routed"
            return "unfrozen"

        grouped = {"main": [], "routed": [], "unfrozen": []}
        for name, parameter in self.model.named_parameters():
            grouped[parameter_group(name)].append(parameter)
        if motion_expert_enabled:
            scales = {
                "routed": float(getattr(expert_cfg, "ROUTED_LR_SCALE", 1.0)),
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
                for name in ("routed", "unfrozen")
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
            shared_epoch = self._shared_unfreeze_epoch() if motion_expert_enabled else 0

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

    def _image_forecast_profile(self):
        data = self.cfg.DATA
        return (getattr(data, "TRAIN_TASK", "recon") == "fore"
                and data.FPS == 5)

    def _joint_motion_supervision(self):
        mode = getattr(self.cfg.DATA, "MOTION_SUPERVISION", "future_only")
        if mode not in ("joint", "future_only"):
            raise ValueError("MOTION_SUPERVISION must be joint or future_only")
        return self._image_forecast_profile() and mode == "joint"

    def _task_conditioning(self, y, mode):
        """Match the supervised task to the image-only deployment inputs."""
        task = getattr(self.cfg.DATA, "TRAIN_TASK", "recon")
        if task is None:
            return dict(y)  # Explicit legacy checkpoint configuration.
        probability = getattr(self.cfg.DATA, "TRAJECTORY_CONDITION_PROB",
                              0.0 if self._image_forecast_profile() else 1.0)
        if (isinstance(probability, bool) or not isinstance(probability, (int, float))
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("TRAJECTORY_CONDITION_PROB must be a finite probability in [0,1]")
        # Validation uses the deployment condition even when a caller explicitly
        # experiments with privileged trajectory conditioning during training.
        if mode != "train" and self._image_forecast_profile():
            visible = False
        elif probability in (0, 1):
            visible = bool(probability)
        else:
            frames = y["valid_frames"]
            visible = torch.rand(frames.shape[0], device=frames.device) < probability
        return apply_task_conditioning(
            y, task, forecast_prefix=self.cfg.DATA.FORECAST_PREFIX,
            trajectory_condition=visible,
            noise_only_history=self._image_forecast_profile() and not self._joint_motion_supervision(),
            supervise_history=self._joint_motion_supervision())

    def _step(self, batch, mode, conditioning=None):
        x = batch["misc"]["motion"]
        y = self._task_conditioning(batch["y"], mode) if conditioning is None else conditioning
        losses = self.flow.training_losses(self.model, x, model_kwargs={"y": y})
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
        for name, value in losses.items():
            if name.endswith("_mse") or name.endswith("_valid_count"):
                metrics[name] = value.float()
        if mode == "train" and getattr(self.cfg.TRAIN, "BATCHED_METRIC_SYNC", False):
            self._log_training_metrics(metrics, x.shape[0])
        else:
            for name, value in metrics.items():
                self.log(f"{mode}/{name}", value.mean(), on_step=mode == "train",
                         on_epoch=True, sync_dist=True, batch_size=x.shape[0])
        if mode == "val":
            sources = batch["misc"].get("source", ["egosuite"] * x.shape[0])
            if isinstance(sources, str):
                sources = [sources] * x.shape[0]
            for source in set(sources):
                selection = torch.tensor([name == source for name in sources], device=x.device)
                totals = self._validation_source_metrics.setdefault(
                    str(source), torch.zeros(4, device=x.device, dtype=torch.float64))
                totals[:3] += torch.stack([
                    losses[key][selection].sum() for key in ("loss", "local_mse", "global_mse")
                ]).double()
                totals[3] += selection.sum()
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def on_save_checkpoint(self, checkpoint):
        checkpoint["motionexpert_architecture"] = architecture_metadata(self.cfg)
        checkpoint["motionexpert_normalization"] = self.normalizer.stats
        checkpoint["motionexpert_normalization_clip"] = self.normalizer.clip
        checkpoint["motionexpert_training_phase"] = _training_phase_for_model_config(self.cfg.MODEL)

    def on_load_checkpoint(self, checkpoint):
        validate_checkpoint_architecture(self.cfg, checkpoint)
        validate_checkpoint_statistics(self.cfg, checkpoint, self.normalizer)
        if training_phase_from_checkpoint(checkpoint) != _training_phase_for_model_config(self.cfg.MODEL):
            raise ValueError("Checkpoint training phase does not match the resolved resume phase")

    def on_validation_epoch_start(self):
        self._validation_source_metrics = {}

    def on_validation_epoch_end(self):
        local = {name: values.detach().cpu().tolist()
                 for name, values in self._validation_source_metrics.items()}
        gathered = [local]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, local)
        totals = {}
        for shard in gathered:
            for source, values in shard.items():
                totals[source] = totals.get(source, torch.zeros(4, dtype=torch.float64)) + torch.tensor(values)
        for source, values in totals.items():
            if values[3] > 0:
                for column, key in enumerate(("flow_loss", "local_mse", "global_mse")):
                    self.log(f"val/{source}/{key}", (values[column] / values[3]).to(self.device),
                             sync_dist=False)
        interval = int(getattr(self.cfg.TRAIN, "SAMPLE_VAL_EVERY_N_EPOCHS", 0))
        limit = int(getattr(self.cfg.TRAIN, "SAMPLE_VAL_MAX_SAMPLES", 16))
        if (interval > 0 and limit > 0 and not self.trainer.sanity_checking
                and (int(self.current_epoch) + 1) % interval == 0
                and self.trainer.is_global_zero):
            from ..evaluate import sampled_validation
            dataset = self.trainer.datamodule.val_dataset
            training = self.model.training
            self.model.eval()
            try:
                with torch.no_grad():
                    sampled = sampled_validation(
                        self, dataset, max_samples=limit,
                        forecast_prefix=int(self.cfg.DATA.FORECAST_PREFIX),
                        tasks=((self.cfg.DATA.TRAIN_TASK,) if getattr(self.cfg.DATA, "TRAIN_TASK", None)
                               is not None else TASKS))
                for name, value in sampled.items():
                    self.log(f"val_sample/{name}", torch.as_tensor(value, device=self.device, dtype=torch.float32),
                             sync_dist=False)
            finally:
                self.model.train(training)

    def validation_step(self, batch, batch_idx):
        y = self._task_conditioning(batch["y"], "val")
        loss = self._step(batch, "val", conditioning=y)
        if self.cfg.MODEL.GLOBAL_TOKEN_COUNT:
            x = batch["misc"]["motion"]
            generator = torch.Generator(device=x.device).manual_seed(int(self.cfg.TRAIN.SEED) + batch_idx)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            t = torch.ones(x.shape[0], device=x.device)
            _, hidden, global_hidden = self.model(noise, t, y, return_features=True)
            valid = y["valid_frames"].bool()
            diagnostics = {
                "global_t1_norm": global_hidden.float().norm(dim=-1).mean(),
                "global_t1_batch_std": global_hidden.float().std(dim=0, unbiased=False).mean(),
                "hidden_t1_norm": (hidden.float().norm(dim=-1) * valid).sum() / valid.sum().clamp_min(1),
            }
            for name, value in diagnostics.items():
                self.log(f"val/{name}", value, on_step=False, on_epoch=True,
                         sync_dist=True, batch_size=x.shape[0])
        return loss

    def sample(self, y, B=None, cond_scale=None, return_all_pred_xstart=False,
               return_one_step_hidden=False, return_one_step_features=False):
        if self._image_forecast_profile():
            y = apply_task_conditioning(
                y, y.get("task_id", getattr(self.cfg.DATA, "TRAIN_TASK", "fore")),
                forecast_prefix=self.cfg.DATA.FORECAST_PREFIX,
                trajectory_condition=False,
                noise_only_history=not self._joint_motion_supervision())
        batch, frames = y["valid_frames"].shape
        if B is not None and B != batch:
            raise ValueError("B does not match the conditioning batch")
        return self.flow.sample_loop(
            self.model, (batch, frames, self.cfg.MODEL.INPUT_FEATS),
            model_kwargs={"y": y, "cond_scale": cond_scale}, progress=False,
            return_all_pred_xstart=return_all_pred_xstart,
            return_one_step_hidden=return_one_step_hidden,
            return_one_step_features=return_one_step_features,
        )
