"""Fused BF16 updates retain clipping and EMA through Lightning's real loop."""
from types import SimpleNamespace

import pytorch_lightning as pl
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from MotionExpert.ema import EMA
from MotionExpert.precision import BF16FusedMixedPrecision, trainer_precision_options


def test_bf16_clips_unscaled_gradients_for_fused_optimizer():
    parameter = torch.nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([30., 40.])
    optimizer = torch.optim.AdamW([parameter], fused=True)
    plugin = BF16FusedMixedPrecision(device="cpu")
    assert plugin.scaler is None and optimizer._step_supports_amp_scaling
    plugin.clip_gradients(optimizer, clip_val=1.)
    torch.testing.assert_close(parameter.grad, torch.tensor([.6, .8]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA fused AdamW integration")
def test_lightning_fused_bf16_preserves_clipping_updates_and_ema(tmp_path):
    class TinyModule(pl.LightningModule):
        def __init__(self, fused):
            super().__init__()
            self.model = torch.nn.Linear(2, 2)
            self.fused = fused
            self.clipped_norms = []

        def training_step(self, batch, batch_idx):
            return self.model(batch[0]).float().square().mean()

        def configure_optimizers(self):
            return torch.optim.AdamW(self.parameters(), lr=5e-6, weight_decay=.01, fused=self.fused)

        def on_train_batch_end(self, outputs, batch, batch_idx):
            self.clipped_norms.append(float(torch.stack([p.grad.norm() for p in self.parameters()]).norm()))
            assert torch.isfinite(outputs["loss"])

    results = []
    for fused in (False, True):
        torch.manual_seed(67)
        model = TinyModule(fused)
        cfg = SimpleNamespace(TRAIN=SimpleNamespace(PRECISION="bf16-mixed", FUSED_ADAMW=fused))
        trainer = pl.Trainer(default_root_dir=tmp_path, accelerator="gpu", devices=[0],
                             **trainer_precision_options(cfg), max_steps=3, max_epochs=1,
                             gradient_clip_val=1., logger=False, enable_checkpointing=False,
                             enable_progress_bar=False, enable_model_summary=False, callbacks=[EMA(.992028)])
        trainer.fit(model, DataLoader(TensorDataset(torch.full((12, 2), 100.)), batch_size=4))
        optimizer = trainer.optimizers[0]
        optimizer.join()
        assert all(.99 <= norm <= 1.00001 for norm in model.clipped_norms)
        assert optimizer.optimizer.defaults["fused"] is fused
        results.append(([p.detach().cpu().clone() for p in model.parameters()],
                        [p.cpu().clone() for p in optimizer.ema_params]))
    for actual_group, expected_group in zip(results[1], results[0]):
        for actual, expected in zip(actual_group, expected_group):
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
