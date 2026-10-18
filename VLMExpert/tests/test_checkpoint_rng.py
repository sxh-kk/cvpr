"""Actual CPU Trainer/Accelerator checkpoint RNG, with production Rich logging."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_actual_trainer_save_and_accelerator_restore_do_not_advance_rng(tmp_path):
    if any(importlib.util.find_spec(name) is None for name in ("accelerate", "diffusers", "transformers")):
        pytest.skip("Requires the complete Psi0 runtime; no AST replacement")
    root = Path(__file__).parents[1]
    script = r'''
import importlib.util, json, logging, random, sys
from pathlib import Path
import numpy as np
import torch

script_path = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("checkpoint_gate_helpers", script_path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
# Configure the actual production Rich handler before importing Accelerate.
from psi.utils.overwatch import LOG_CONFIG
from psi.config.config import LaunchConfig, LoggingConfig, WandbConfig, TrainConfig, DataConfig
from psi.config.model_psi0 import Psi0ModelConfig
from psi.config.transform import DataTransform, FieldTransform, Psi0ModelTransform
from psi.config.transform_psi0_sonic import SonicRepackTransform
from psi.trainers.posttrain import PosttrainTrainer
from accelerate import Accelerator

assert LOG_CONFIG["handlers"]["console"]["enable_link_path"] is False
assert all(not handler.enable_link_path for handler in logging.getLogger().handlers
           if hasattr(handler, "enable_link_path"))
cfg = LaunchConfig(exp="cpu_rng", timestamp="bounded", log=LoggingConfig(), wandb=WandbConfig(),
                   train=TrainConfig(name="posttrain", output_dir=sys.argv[2], train_batch_size=1,
                                     max_training_steps=4, warmup_ratio=None),
                   data=DataConfig(transform=DataTransform(repack=SonicRepackTransform(),
                                                          field=FieldTransform(), model=Psi0ModelTransform())),
                   model=Psi0ModelConfig(action_dim=80, odim=45, action_chunk_size=30))
trainer = PosttrainTrainer(cfg, torch.device("cpu"))
accelerator = Accelerator(cpu=True, mixed_precision="no")
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
trainer.model, trainer.optimizer, trainer.lr_scheduler = accelerator.prepare(model, optimizer, scheduler)
trainer.accelerator = accelerator
random.seed(902)
np.random.seed(903)
torch.manual_seed(904)
checkpoint = Path(trainer.save_checkpoint(4))
rng_path = checkpoint / "random_states_0.pkl"
states = torch.load(rng_path, map_location="cpu", weights_only=False)
saved = {"python": states["random_state"], "numpy": states["numpy_random_seed"],
         "torch": states["torch_manual_seed"]}
def live():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
assert gate.tree_hash(live()) == gate.tree_hash(saved), "Post-save Rich logs consumed RNG"
# Accelerate1.7's CPU saver omits this key, but its loader reads it unconditionally.
# Only this newly-created CPU fixture gets an empty device list; no GPU state is mocked.
states["torch_cuda_manual_seed"] = []
torch.save(states, rng_path)
random.random()
np.random.rand()
torch.rand(1)
accelerator.load_state(str(checkpoint))
assert gate.tree_hash(live()) == gate.tree_hash(saved), "Post-load Rich logs consumed RNG"
print(json.dumps({"actual_trainer_accelerator_cpu_rng_passed": True}))
'''
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                       OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", script,
                             str(root / "scripts/validation/validate_unifolm_full_model.py"), str(tmp_path)],
                            env=environment, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"actual_trainer_accelerator_cpu_rng_passed": true' in result.stdout
