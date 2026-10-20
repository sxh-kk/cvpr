"""Production update cadence, config parsing, and bounded checkpoint retention."""

import ast
import copy
import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
import tyro

from psi.config.config import TrainConfig
from psi.config.train.posttrain_sonic_psi0_config import DynamicLaunchConfig
from psi.trainers.trainer import Trainer, should_save_training_checkpoint


ROOT = Path(__file__).parents[1]


def _actual_checkpoint_branch():
    tree = ast.parse((ROOT / "scripts/train.py").read_text())
    train = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "train")
    sync = next(node for node in ast.walk(train) if isinstance(node, ast.If)
                and isinstance(node.test, ast.Name) and node.test.id == "sync_gradients")
    completed = next(node for node in sync.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "completed_step" for target in node.targets))
    checkpoint = next(node for node in sync.body if isinstance(node, ast.If)
                      and isinstance(node.test, ast.Call)
                      and getattr(node.test.func, "id", None) == "should_save_training_checkpoint")
    branch = copy.deepcopy(sync)
    branch.body = [copy.deepcopy(completed), copy.deepcopy(checkpoint)]
    branch.orelse = []
    harness = ast.parse("def save_after_step(global_step, sync_gradients, trainer, config, accelerator):\n    pass\n")
    harness.body[0].body = [branch]
    namespace = {"should_save_training_checkpoint": should_save_training_checkpoint}
    exec(compile(ast.fix_missing_locations(harness), str(ROOT / "scripts/train.py"), "exec"), namespace)
    return namespace["save_after_step"]


def test_actual_loop_saves_only_completed_updates_and_overlap_saves_once():
    branch = _actual_checkpoint_branch()
    saved, barriers = [], []
    trainer = SimpleNamespace(max_training_steps=23, save_checkpoint=lambda step: saved.append(step))
    config = SimpleNamespace(train=TrainConfig(checkpointing_steps=7, max_training_steps=23,
                                               early_checkpoint_steps=(1, 7, 23, 23)))
    accelerator = SimpleNamespace(wait_for_everyone=lambda: barriers.append(True), is_rank_zero=lambda: False)
    # The extracted production branch queries overwatch, not Accelerator, for logging.
    branch.__globals__["overwatch"] = SimpleNamespace(is_rank_zero=lambda: False)
    for global_step in range(23):
        before = list(saved)
        branch(global_step, False, trainer, config, accelerator)
        assert saved == before  # first accumulation microbatch never saves
        branch(global_step, True, trainer, config, accelerator)
    assert saved == [1, 7, 14, 21, 23]
    assert len(barriers) == 2 * len(saved)


def test_formal_early_periodic_and_final_cadence_and_legacy_default():
    assert [step for step in range(1, 4501)
            if should_save_training_checkpoint(step, 2000, 4500, (100, 500, 1000))] == [
                100, 500, 1000, 2000, 4000, 4500]
    assert [step for step in range(1, 4501)
            if should_save_training_checkpoint(step, 2000, 4500)] == [2000, 4000, 4500]


@pytest.mark.parametrize("steps", [(0,), (-1,), (100, 0)])
def test_invalid_early_steps_fail_actual_config_and_helper(steps):
    with pytest.raises(ValidationError, match="positive optimizer update counts"):
        TrainConfig(early_checkpoint_steps=steps)
    with pytest.raises(ValueError, match="positive"):
        should_save_training_checkpoint(1, 2000, 40000, steps)


def test_checkpoint_helper_rejects_zero_completed_update():
    with pytest.raises(ValueError, match="positive"):
        should_save_training_checkpoint(0, 2000, 40000, (100,))
    assert TrainConfig().early_checkpoint_steps == ()


def test_actual_saver_strict_latest_three_has_at_most_four_before_prune(tmp_path):
    peaks = []

    def save_state(path):
        Path(path).mkdir()
        peaks.append(len(list((tmp_path / "checkpoints").iterdir())))

    trainer = SimpleNamespace(
        project_dir=str(tmp_path),
        train_cfg=TrainConfig(max_checkpoints_to_keep=3, preserve_checkpoint_every=None),
        accelerator=SimpleNamespace(save_state=save_state, wait_for_everyone=lambda: None, is_main_process=True),
    )
    updates = [step for step in range(1, 40001)
               if should_save_training_checkpoint(step, 2000, 40000, (100, 500, 1000))]
    for step in updates:
        Trainer.save_checkpoint(trainer, step)
        assert len(list((tmp_path / "checkpoints").iterdir())) <= 3
    assert {path.name for path in (tmp_path / "checkpoints").iterdir()} == {
        "ckpt_36000", "ckpt_38000", "ckpt_40000"}
    assert max(peaks) == 4


def _recipe_environment(tmp_path, early_steps, checkpointing_steps=None):
    env = dict(os.environ, EGO_DEX_VLM_PATH=str(tmp_path / "vlm"),
               MOTION_EXPERT_CHECKPOINT=str(tmp_path / "motion.ckpt"), MOTION_USE_EMA="1",
               UNIFOLM_DINO_ROOT=str(tmp_path), UNIFOLM_DINO_TRAIN_ID="train",
               UNIFOLM_DINO_VAL_ID="val", UNIFOLM_DINO_CACHE=str(tmp_path / "dino"),
               PSI_PYTHON=sys.executable, NPROC_PER_NODE="8", BATCH_SIZE="16",
               GRAD_ACCUM_STEPS="2", MAX_STEPS="40000", REPORT_TO="wandb")
    env.pop("UNIFOLM_EARLY_CHECKPOINT_STEPS", None)
    if early_steps is not None:
        env["UNIFOLM_EARLY_CHECKPOINT_STEPS"] = early_steps
    env.pop("UNIFOLM_CHECKPOINTING_STEPS", None)
    if checkpointing_steps is not None:
        env["UNIFOLM_CHECKPOINTING_STEPS"] = checkpointing_steps
    return env


@pytest.mark.parametrize("early_steps, checkpointing_steps, expected, expected_interval", [
    (None, None, (100, 500, 1000), 2000),
    (None, "1000", (100, 500, 1000), 1000),
    ("14100 14500 15000 15100 15500", None, (14100, 14500, 15000, 15100, 15500), 2000),
    (" \t14100\n14500\r 15000 ", None, (14100, 14500, 15000), 2000),
])
def test_actual_recipe_tuple_cli_parses_with_real_tyro_and_config(
    tmp_path, early_steps, checkpointing_steps, expected, expected_interval,
):
    env = _recipe_environment(tmp_path, early_steps, checkpointing_steps)
    result = subprocess.run(["bash", str(ROOT / "scripts/train/psi0/posttrain-dual-source-unifolm.sh"),
                             "--dry-run", "cadence_test"], env=env, check=True, text=True, capture_output=True)
    command = shlex.split(result.stdout.split("Command:", 1)[1])
    cfg = tyro.cli(DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,),
                   args=command[command.index("scripts/train.py") + 2:])
    assert cfg.train.early_checkpoint_steps == expected
    assert cfg.train.training_safety_checks is True
    assert (cfg.train.max_checkpoints_to_keep, cfg.train.preserve_checkpoint_every) == (3, None)
    assert cfg.train.checkpointing_steps == expected_interval
    assert (cfg.train.train_batch_size, cfg.train.gradient_accumulation_steps) == (16, 2)
    assert (cfg.train.max_training_steps, cfg.train.warmup_steps, cfg.train.lr_scheduler_type) == (40000, 1000, "cosine")
    assert cfg.train.learning_rate == 1e-4
    assert (cfg.model.lang_backbone_lr, cfg.model.vision_tower_lr,
            cfg.model.mm_projector_lr, cfg.model.motion_expert_lr) == (1e-6, 1e-5, 1e-4, 1e-5)


@pytest.mark.parametrize("early_steps", [
    "", " \t\n", "0", "-1", "+100", "01", "1.5", "1e3", "100,500",
    "14100 0 14500", "14100\ninvalid", "14100 --train.max_training_steps=1",
])
def test_actual_recipe_rejects_invalid_early_checkpoint_override_before_launch(tmp_path, early_steps):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train/psi0/posttrain-dual-source-unifolm.sh"),
         "--dry-run", "invalid_cadence_test"],
        env=_recipe_environment(tmp_path, early_steps), text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "UNIFOLM_EARLY_CHECKPOINT_STEPS must contain positive integers" in result.stderr
    assert "Command:" not in result.stdout


@pytest.mark.parametrize("checkpointing_steps", [
    "", " \t\n", "0", "-1", "+1000", "01000", "1.5", "1e3", "1000 2000",
    "1000\ninvalid", "1000 --train.max_training_steps=1",
])
def test_actual_recipe_rejects_invalid_checkpoint_interval_before_launch(tmp_path, checkpointing_steps):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/train/psi0/posttrain-dual-source-unifolm.sh"),
         "--dry-run", "invalid_interval_test"],
        env=_recipe_environment(tmp_path, None, checkpointing_steps), text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "UNIFOLM_CHECKPOINTING_STEPS must be a positive integer" in result.stderr
    assert "Command:" not in result.stdout
