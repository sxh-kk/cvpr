"""Parse the actual preparation recipe and prove its default never launches workers."""

import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest
import tyro

from psi.config.train.posttrain_sonic_psi0_config import DynamicLaunchConfig


ROOT = Path(__file__).parents[1]
RECIPES = ROOT / "scripts/train/psi0"


def _env(tmp_path):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("UNIFOLM_", "MOTION_"))}
    env.update(EGO_DEX_VLM_PATH=str(tmp_path / "vlm"),
               MOTION_EXPERT_CHECKPOINT=str(tmp_path / "motion.ckpt"),
               UNIFOLM_DINO_ROOT=str(tmp_path), UNIFOLM_DINO_TRAIN_ID="train",
               UNIFOLM_DINO_VAL_ID="val", UNIFOLM_DINO_CACHE=str(tmp_path / "dino"),
               PSI_PYTHON=sys.executable, NPROC_PER_NODE="8", BATCH_SIZE="16",
               GRAD_ACCUM_STEPS="2", MAX_STEPS="40000", REPORT_TO="wandb",
               UNIFOLM_RUN_TIMESTAMP="261019120000")
    return env


def _command(script, env):
    result = subprocess.run(["bash", str(RECIPES / script), "--dry-run", "same_experiment"],
                            env=env, capture_output=True, text=True, check=True)
    return shlex.split(result.stdout.split("Command:", 1)[1])


def _config(command):
    return tyro.cli(DynamicLaunchConfig, config=(tyro.conf.ConsolidateSubcommandArgs,),
                    args=command[command.index("scripts/train.py") + 2:])


@pytest.mark.parametrize("foreach", ["False", "True"])
def test_speedup_real_cli_only_changes_three_execution_flags(tmp_path, foreach):
    env = _env(tmp_path)
    env["UNIFOLM_OPTIMIZER_FOREACH"] = foreach
    baseline = _config(_command("posttrain-dual-source-unifolm.sh", env)).model_dump()
    command = _command("posttrain-dual-source-unifolm-speedup.sh", env)
    candidate = _config(command).model_dump()
    assert "--nproc_per_node=8" in command
    assert candidate["model"].pop("vlm_skip_unused_logits") is True
    assert baseline["model"].pop("vlm_skip_unused_logits") is False
    for flag in ("keep_train_batch_on_cpu", "sync_metrics_on_update_only"):
        assert candidate["train"].pop(flag) is True
        assert baseline["train"].pop(flag) is False
    assert candidate["train"]["optimizer_foreach"] is (foreach == "True")
    # Includes batch/accumulation/LRs, full tokens, pooling, state and safety settings.
    assert candidate == baseline


def test_default_is_metadata_check_without_training_subprocess(tmp_path):
    env = _env(tmp_path)
    Path(env["EGO_DEX_VLM_PATH"]).mkdir()
    Path(env["MOTION_EXPERT_CHECKPOINT"]).touch()
    for pack in ("train", "val"):
        metadata = tmp_path / pack / "meta"
        metadata.mkdir(parents=True)
        (metadata / "stats_psi0.json").write_text("{}")
        (metadata / "modality.json").write_text("{}")
        cache = tmp_path / "dino" / pack
        cache.mkdir(parents=True)
        (cache / "manifest.json").write_text('{"state":"complete"}')
    # A fake Python makes an accidental exec harmless and visible in this test.
    calls = tmp_path / "python_calls"
    stub = tmp_path / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$*" >> "$RECIPE_TEST_CALLS"\n')
    stub.chmod(0o755)
    env.update(PSI_PYTHON=str(stub), RECIPE_TEST_CALLS=str(calls))
    result = subprocess.run(["bash", str(RECIPES / "posttrain-dual-source-unifolm-speedup.sh")],
                            env=env, capture_output=True, text=True, check=True)
    assert "Command:" in result.stdout
    assert calls.read_text().splitlines() == [
        f"- {tmp_path}/dino/train/manifest.json", f"- {tmp_path}/dino/val/manifest.json"]
    assert "--train.optimizer-foreach=False" in result.stdout


def test_resume_is_explicit_and_missing_checkpoint_rejected_before_launch(tmp_path):
    env = _env(tmp_path)
    checkpoint = str(tmp_path / "checkpoints/ckpt_1000")
    env["UNIFOLM_RESUME_CHECKPOINT"] = checkpoint
    cfg = _config(_command("posttrain-dual-source-unifolm-speedup.sh", env))
    assert cfg.train.resume_from_checkpoint == checkpoint
    assert cfg.train.skip_resumed_steps is True
    result = subprocess.run(["bash", str(RECIPES / "posttrain-dual-source-unifolm-speedup.sh")],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "existing absolute ckpt_<step>" in result.stderr
