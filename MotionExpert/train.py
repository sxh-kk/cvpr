"""Train or resume sparse K12 motion with explicit hand supervision."""

import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from .config import load_config, as_dict, validate_architecture
from .check_data import validate_paths
from .checkpoints import read_checkpoint, model_state, checkpoint_config, validate_checkpoint_architecture, validate_checkpoint_statistics
from .dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
from .ema import EMA
from .module.uem_module import UEM_Module, restore_training_phase


class PeriodicModelCheckpoint(ModelCheckpoint):
    def _save_last_checkpoint(self, trainer, monitor_candidates):
        # Avoid serializing the 400M model on epochs without a periodic save.
        if self.every_n_epochs and (trainer.current_epoch + 1) % self.every_n_epochs == 0:
            super()._save_last_checkpoint(trainer, monitor_candidates)


def configure_initialization(cfg, *, resume=None, init_weights=None, from_scratch=False):
    """Explicit CLI initialization takes priority over configured checkpoints."""
    if sum((resume is not None, init_weights is not None, bool(from_scratch))) > 1:
        raise ValueError("Choose one of resume, init_weights and from_scratch")
    if from_scratch or init_weights is not None:
        cfg.MODEL.CKPT_PATH = None
        cfg.MODEL.INIT_CKPT_PATH = str(Path(init_weights).resolve()) if init_weights is not None else None
    else:
        resume = resume or cfg.MODEL.CKPT_PATH
        cfg.MODEL.CKPT_PATH = str(Path(resume).resolve()) if resume else None
        if resume:
            cfg.MODEL.INIT_CKPT_PATH = None
    # A new run must infer its own origin; resume restores the original below.
    for name in ("TRAINING_ORIGIN", "EFFECTIVE_SHARED_UNFREEZE_EPOCH"):
        if hasattr(cfg.MODEL, name):
            delattr(cfg.MODEL, name)
    return cfg.MODEL.CKPT_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-dir")
    parser.add_argument("--devices", type=int)
    parser.add_argument("--ddp-socket", action="store_true", help="Use single-node loopback sockets for NCCL")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--global-weight", type=float, help="Global delta loss weight for controlled ablations")
    parser.add_argument("--accumulate-grad-batches", type=int)
    parser.add_argument("--output-dir", type=Path)
    init = parser.add_mutually_exclusive_group()
    init.add_argument("--resume", type=Path)
    init.add_argument("--init-weights", type=Path)
    init.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--limit-train-batches", type=int)
    parser.add_argument("--limit-val-batches", type=int)
    args = parser.parse_args()
    if args.ddp_socket:
        for key in ("NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SHM_DISABLE"):
            os.environ[key] = "1"
        os.environ["NCCL_SOCKET_IFNAME"] = "lo"
    if args.resume and args.config is None:
        checkpoint = read_checkpoint(args.resume)
        cfg = checkpoint_config(checkpoint)
        del checkpoint
        if args.data_dir:
            parser.error("--data-dir with --resume requires --config to resolve all dataset paths")
    else:
        cfg = load_config(args.config, data_dir=args.data_dir)
    for section, key, value in (
        (cfg.TRAIN, "NUM_GPUS", args.devices), (cfg.TRAIN, "NUM_EPOCHS", args.max_epochs),
        (cfg.TRAIN, "MAX_STEPS", args.max_steps), (cfg.DATA, "BATCH_SIZE", args.batch_size),
        (cfg.DATA, "NUM_WORKERS", args.num_workers),
        (cfg.TRAIN, "ACCUMULATE_GRAD_BATCHES", args.accumulate_grad_batches),
    ):
        if value is not None:
            setattr(section, key, value)
    if args.output_dir:
        cfg.TRAIN.EXP_PATH = str(args.output_dir.resolve())
    if args.global_weight is not None:
        cfg.FLOW.GLOBAL_WEIGHT = args.global_weight
    resume = configure_initialization(cfg, resume=args.resume, init_weights=args.init_weights,
                                      from_scratch=args.from_scratch)
    validate_architecture(cfg)
    if resume:
        checkpoint = read_checkpoint(cfg.MODEL.CKPT_PATH)
        validate_checkpoint_architecture(cfg, checkpoint)
        restore_training_phase(cfg, checkpoint)
        del checkpoint
    validate_paths(cfg)
    pl.seed_everything(int(getattr(cfg.TRAIN, "SEED", 62)), workers=True)
    torch.set_float32_matmul_precision("high")
    torch.set_num_threads(4)
    output = Path(cfg.TRAIN.EXP_PATH)
    output.mkdir(parents=True, exist_ok=True)
    dm = EE4D_Motion_DataModule(cfg)
    # Validate the archive/splits/statistics and real DINO availability before
    # allocating the 400M-parameter model or optimizer.
    dm.setup("fit")
    model = UEM_Module(cfg)
    if cfg.MODEL.INIT_CKPT_PATH:
        checkpoint = read_checkpoint(cfg.MODEL.INIT_CKPT_PATH)
        validate_checkpoint_architecture(cfg, checkpoint)
        validate_checkpoint_statistics(cfg, checkpoint, model.normalizer)
        model.model.load_state_dict(model_state(model.model, checkpoint), strict=True)
        del checkpoint
    gpu_count = int(cfg.TRAIN.NUM_GPUS)
    if gpu_count < 0 or gpu_count > torch.cuda.device_count():
        raise ValueError(f"Requested {gpu_count} GPUs, available: {torch.cuda.device_count()}; use --devices 0 for CPU")
    use_gpu = gpu_count > 0
    devices = gpu_count if use_gpu else 1
    strategy = DDPStrategy(find_unused_parameters=True, static_graph=False) if devices > 1 else "auto"
    callbacks = [
        EMA(float(cfg.TRAIN.EMA_DECAY)),
        PeriodicModelCheckpoint(dirpath=str(output), every_n_epochs=int(cfg.TRAIN.SAVE_EVERY_N_EPOCHS),
                        save_last="link", save_top_k=-1, save_on_train_epoch_end=True),
        LearningRateMonitor("step"),
        TQDMProgressBar(refresh_rate=int(getattr(cfg.TRAIN, "PROGRESS_REFRESH_RATE", 10))),
    ]
    trainer = pl.Trainer(
        default_root_dir=str(output),
        logger=TensorBoardLogger(str(output), name="", version="", default_hp_metric=False),
        accelerator="gpu" if use_gpu else "cpu", devices=devices, strategy=strategy,
        precision=cfg.TRAIN.PRECISION if use_gpu else "32-true",
        max_epochs=int(cfg.TRAIN.NUM_EPOCHS), max_steps=int(cfg.TRAIN.MAX_STEPS),
        accumulate_grad_batches=int(cfg.TRAIN.ACCUMULATE_GRAD_BATCHES),
        gradient_clip_val=float(cfg.TRAIN.GRADIENT_CLIP_VAL),
        check_val_every_n_epoch=int(cfg.TRAIN.CHECK_VAL_EVERY_N_EPOCHS),
        num_sanity_val_steps=int(cfg.TRAIN.NUM_SANITY_VAL_STEPS),
        log_every_n_steps=int(cfg.TRAIN.LOG_EVERY_N_STEPS),
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        callbacks=callbacks,
    )
    if trainer.is_global_zero:
        (output / "config.yaml").write_text(yaml.safe_dump(as_dict(cfg), sort_keys=False), encoding="utf-8")
    trainer.fit(model, datamodule=dm, ckpt_path=cfg.MODEL.CKPT_PATH)
    if not args.fast_dev_run:
        final_path = output / "final.ckpt"
        trainer.save_checkpoint(str(final_path))
        if trainer.is_global_zero:
            temporary = output / ".last.ckpt.tmp"
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(final_path.name)
            temporary.replace(output / "last.ckpt")


if __name__ == "__main__":
    main()
