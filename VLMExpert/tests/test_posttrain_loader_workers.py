"""Exercise production loader construction and a real spawn-worker batch."""

import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from psi.config.config import TrainConfig
from psi.trainers.posttrain import PosttrainTrainer


class _Frames(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "input_ids": torch.tensor([1, 2, 3]),
            "pixel_values": torch.ones(4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "actions": np.zeros((2, 3), dtype=np.float32),
            "states": np.zeros((1, 3), dtype=np.float32),
            "motion_img_embs": torch.zeros(6, 5, 1024),
            "episode_index": index,
            "frame_index": os.getpid(),
        }


def _loaders(workers):
    dataset = _Frames()
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(seed=292285),
        train_cfg=TrainConfig(num_workers=workers, dataloader_start_method="spawn",
                              train_batch_size=2, val_batch_size=2),
        data_cfg=SimpleNamespace(num_workers=None, prefetch_factor=2),
        tokenizer=SimpleNamespace(model_max_length=128, pad_token_id=0),
        train_dataset=dataset,
    )
    return PosttrainTrainer.create_dataloaders(trainer, dataset, dataset)


def _shutdown(loader):
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None:
        iterator._shutdown_workers()


def test_train_and_validation_use_spawn_and_read_real_worker_batches():
    train, validation = _loaders(2)
    try:
        for loader in (train, validation):
            assert loader.multiprocessing_context.get_start_method() == "spawn"
            batch = next(iter(loader))
            assert batch["motion_img_embs"].shape == (2, 6, 5, 1024)
            assert batch["actions"].shape == (2, 2, 3)
            assert (batch["frame_index"] != os.getpid()).all()
            assert not torch.cuda.is_initialized()
    finally:
        _shutdown(train)
        _shutdown(validation)


def test_zero_train_workers_omit_process_only_options():
    train, validation = _loaders(0)
    assert train.multiprocessing_context is None
    assert not train.persistent_workers
    assert train.prefetch_factor is None
    batch = next(iter(train))
    assert (batch["frame_index"] == os.getpid()).all()
    # Validation keeps the original four-worker setting and the safe context.
    assert validation.num_workers == 4
    assert validation.multiprocessing_context.get_start_method() == "spawn"


def test_legacy_config_keeps_default_worker_context():
    assert TrainConfig().dataloader_start_method is None
