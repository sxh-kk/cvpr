"""Visual versions must match by default; explicit size adaptation keeps DINO fixed."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from ImageProcess import SharedImageProcessor
from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.visual_contract import (
    archive_visual_compatibility, cache_visual_contract, compare_visual_contracts,
)


def visual(profile="256x480", *, local=False):
    process = SharedImageProcessor(profile).metadata
    normalization = {
        "do_resize": False, "do_center_crop": False,
        "size": {"height": process["height"], "width": process["width"]},
        "do_rescale": True, "rescale_factor": 1 / 255,
        "do_normalize": True, "image_mean": [.485, .456, .406], "image_std": [.229, .224, .225],
    }
    encoder = {
        "checkpoint": "/unit-test/dino" if local else "facebook/dinov2-with-registers-large",
        "revision": "local" if local else "a" * 40,
        "shape": [5, 1024], "token_order": ["cls", "register_0", "register_1", "register_2", "register_3"],
        "storage_dtype": "float32", "pooling": "last_hidden_state[:, :5]", "normalization": "none_after_model",
        "processor": deepcopy(normalization), "preprocessing": deepcopy(normalization),
        "image_process": deepcopy(process), "optimization": {"preprocess_workers": 2, "preprocessor": "hf"},
    }
    if local:
        encoder["local_file_sha256"] = {"config.json": "1" * 64, "model.safetensors": "2" * 64}
    return {"format": "motionexpert-visual-v1", "mode": "shared_rgb", "image_process": process, "encoder": encoder}


def cache_for(contract):
    return SimpleNamespace(
        manifest={"contract": "smplh-dino-shared-rgb-cache-v1", "scope": "all", "state": "complete",
                  "image_process": deepcopy(contract["image_process"]), "encoder": deepcopy(contract["encoder"])},
        metadata={"manifest_sha256": "cache-manifest", "sampling_step": 1, "fps": 30},
        root=Path("/unit-test/already-validated-cache"),
    )


def test_exact_visual_profile_matches_and_optimization_settings_are_not_identity():
    trained, actual = visual(), visual()
    actual["encoder"]["optimization"] = {"preprocess_workers": 24, "preprocessor": "pil_parallel", "prefetch_batches": 2}
    result = compare_visual_contracts(trained, actual)
    assert result["mode"] == "matched" and result["encoder_identity_verified"]
    assert not result["requested"] and not result["profile_changed"]


def test_profile_mismatch_requires_explicit_size_adaptation():
    trained, actual = visual(), visual("256x320")
    with pytest.raises(ValueError, match="allow-visual-adaptation"):
        compare_visual_contracts(trained, actual)
    result = compare_visual_contracts(trained, actual, allow_visual_adaptation=True)
    assert result["mode"] == "explicit_shared_rgb_size_adaptation"
    assert result["trained_hw"] == [256, 480] and result["actual_hw"] == [256, 320]
    assert result["requested"] and result["profile_changed"]


@pytest.mark.parametrize("change", ["checkpoint", "revision", "normalization", "token_order", "local_sha"])
def test_adaptation_never_permits_different_dino_semantics(change):
    trained, actual = visual(local=change == "local_sha"), visual("256x320", local=change == "local_sha")
    if change == "normalization":
        actual["encoder"]["processor"]["image_mean"] = [.5, .5, .5]
    elif change == "token_order":
        actual["encoder"]["token_order"][0:2] = ["register_0", "cls"]
    elif change == "local_sha":
        actual["encoder"]["local_file_sha256"]["model.safetensors"] = "3" * 64
    else:
        actual["encoder"][change] = "b" * 40 if change == "revision" else "another-dino"
    with pytest.raises(ValueError, match="identity|token order"):
        compare_visual_contracts(trained, actual, allow_visual_adaptation=True)


def test_local_directory_can_move_when_weight_hashes_and_revision_match():
    trained, actual = visual(local=True), visual(local=True)
    actual["encoder"]["checkpoint"] = "/relocated/identical-dino"
    assert compare_visual_contracts(trained, actual)["mode"] == "matched"
    actual["encoder"].pop("local_file_sha256")
    with pytest.raises(ValueError, match="SHA256"):
        compare_visual_contracts(trained, actual, allow_visual_adaptation=True)


def test_size_permission_does_not_allow_different_augmentation_rules():
    trained, actual = visual(), visual("256x320")
    profile = deepcopy(actual["image_process"]["profile"])
    profile["augmentation"]["robust"]["min_scale"] = .9
    actual["image_process"] = SharedImageProcessor(profile).metadata
    actual["encoder"]["image_process"] = deepcopy(actual["image_process"])
    with pytest.raises(ValueError, match="size profiles only"):
        compare_visual_contracts(trained, actual, allow_visual_adaptation=True)


def test_manifest_is_reused_without_reopening_data_and_report_is_self_contained(monkeypatch):
    trained, actual = visual(), visual("256x320")
    cache = cache_for(actual)
    def unexpected_read(*args, **kwargs):
        raise AssertionError("Compatibility must reuse the validated manifest")
    monkeypatch.setattr(Path, "read_text", unexpected_read)
    monkeypatch.setattr(Path, "open", unexpected_read)
    report = archive_visual_compatibility(SimpleNamespace(archive=SimpleNamespace(vision_cache=cache)), trained,
                                          allow_visual_adaptation=True)
    assert report["trained_visual_conditioning"] == trained
    assert report["actual_visual_conditioning"] == actual
    assert report["visual_cache_metadata"] == cache.metadata
    assert report["visual_adaptation"]["requested"]
    json.dumps(report)
    report["actual_visual_conditioning"]["encoder"]["revision"] = "changed"
    assert cache.manifest["encoder"]["revision"] == "a" * 40


def test_missing_actual_cache_cannot_be_exempted():
    with pytest.raises(ValueError, match="VISION_PROFILE_ROOT"):
        cache_visual_contract(None)


@pytest.mark.parametrize("entrypoint", ["infer", "evaluate"])
@pytest.mark.parametrize("allow", [False, True])
def test_archive_cli_checks_visual_compatibility_and_records_explicit_adaptation(tmp_path, monkeypatch, entrypoint, allow):
    from MotionExpert import infer, evaluate, motion_expert
    from MotionExpert import config as config_module
    from MotionExpert.dataset import data_module, smplh_motion_dataset

    torch.set_num_threads(1)
    cfg = smplh_motion_expert_config()
    cfg.DATA.IMAGE_PROFILE = "256x320"
    trained, actual = visual(), visual("256x320")
    calls = []
    expert = SimpleNamespace(cfg=cfg, visual_conditioning=trained, normalizer=SimpleNamespace(stats={}, clip=None), model=None)
    def sample(*args, **kwargs):
        calls.append("sample")
        return {"motion": torch.zeros(1, 11, 245), "metadata": {}}
    expert.sample = sample
    dataset = SimpleNamespace(archive=SimpleNamespace(vision_cache=cache_for(actual)), normalizer=expert.normalizer,
                              split_fingerprint="unit-test-split")
    dataset.get_sample = lambda *a, **k: {"y": {"img_embs": torch.zeros(11, 5, 1024)}, "misc": {}}
    monkeypatch.setattr(config_module, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(evaluate, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(infer, "MotionExpert", lambda *a, **k: expert)
    monkeypatch.setattr(motion_expert, "MotionExpert", lambda *a, **k: expert)
    class DataModule:
        def __init__(self, *args):
            self.val_dataset = dataset
        def setup(self, *args):
            pass
    monkeypatch.setattr(data_module, "SMPLHMotionDataModule", DataModule)
    monkeypatch.setattr(smplh_motion_dataset, "SMPLHMotionDataset", lambda *a, **k: dataset)
    def validate(*args, **kwargs):
        calls.append("validation")
        return {"test_metric": 0.0}
    monkeypatch.setattr(evaluate, "sampled_validation", validate)
    output = tmp_path / ("result.pt" if entrypoint == "infer" else "result.json")
    argv = [entrypoint, "--config", "unit-test.yaml", "--output", str(output),
            "--weights" if entrypoint == "infer" else "--checkpoint", "unit-test.pt"]
    if allow:
        argv.append("--allow-visual-adaptation")
    monkeypatch.setattr(sys, "argv", argv)
    if not allow:
        with pytest.raises(ValueError, match="allow-visual-adaptation"):
            getattr({"infer": infer, "evaluate": evaluate}[entrypoint], "main")()
        assert not calls and not output.exists()
    else:
        getattr({"infer": infer, "evaluate": evaluate}[entrypoint], "main")()
        report = torch.load(output, weights_only=True)["metadata"] if entrypoint == "infer" else json.loads(output.read_text())
        assert report["trained_visual_conditioning"] == trained
        assert report["actual_visual_conditioning"] == actual
        assert report["visual_cache_metadata"] == dataset.archive.vision_cache.metadata
        assert report["visual_adaptation"]["mode"] == "explicit_shared_rgb_size_adaptation"
        assert len(calls) == 1
