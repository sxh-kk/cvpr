"""Cross-statistics initialization keeps physical contracts and resets run state."""

from copy import deepcopy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from MotionExpert import warm_start
from MotionExpert.checkpoints import parameter_phase, validate_checkpoint_statistics
from MotionExpert.config import as_dict, architecture_metadata, smplh_motion_expert_config
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.tests.fixtures import smplh_stats
from MotionExpert.tests.test_visual_compatibility import visual


class TinyGroupedModel(torch.nn.Module):
    """Real optimizer phase names without allocating the production network."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.global_token = torch.nn.Parameter(torch.zeros(768))
        self.backbone = torch.nn.Linear(2, 2)
        self.block = torch.nn.Module()
        self.block.ff = torch.nn.Module()
        self.block.ff.router = torch.nn.Linear(1, 1)
        self.block.ff.routed_experts = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(2)),
        ])
        self.block.ff.shared = torch.nn.Linear(1, 1)
        self.register_buffer("fixed_buffer", torch.tensor([3.0, 4.0]))


@pytest.fixture
def warm_case(tmp_path, monkeypatch):
    old_cfg = smplh_motion_expert_config()
    old_cfg.DATA.ARCHIVE_ROOT = "/old/v2/archive"
    old_cfg.DATA.STATS_PATH = "/old/unavailable/statistics.pt"
    old_stats = smplh_stats(old_cfg)
    old_stats["archive_fingerprint"] = "old-v2-archive"
    old_stats["archive_root"] = old_cfg.DATA.ARCHIVE_ROOT
    old_stats["num_windows"] = 100
    cfg = deepcopy(old_cfg)
    cfg.DATA.ARCHIVE_ROOT = "/new/v3/archive"
    cfg.DATA.STATS_PATH = str(tmp_path / "new-statistics.pt")
    cfg.MODEL.WARM_START_CKPT_PATH = str(tmp_path / "epoch59.ckpt")
    cfg.MODEL.WARM_START_WEIGHTS = "ema"
    cfg.MODEL.WARM_START_EXPECTED_COMPLETED_EPOCHS = 60
    cfg.TRAIN.NUM_EPOCHS = cfg.TRAIN.SCHEDULER_TOTAL_EPOCHS = 140
    cfg.TRAIN.WARMUP_EPOCHS = 20
    new_stats = deepcopy(old_stats)
    new_stats.update(archive_fingerprint="new-v3-archive", archive_root=cfg.DATA.ARCHIVE_ROOT,
                     num_windows=90)
    new_stats["motion_mean"][69] = .01
    new_stats["motion_std"][69] = .8
    new_stats["traj_mean"][6] = .02
    torch.save(new_stats, cfg.DATA.STATS_PATH)
    model = TinyGroupedModel(cfg)
    raw = {name: torch.full_like(value, .1 + index / 100)
           for index, (name, value) in enumerate(model.state_dict().items())}
    named = list(model.named_parameters())
    ordered, groups = [], []
    for phase in ("routed", "unfrozen"):
        members = [(name, parameter) for name, parameter in named if parameter_phase(name) == phase]
        offset = len(ordered)
        groups.append({"phase": phase, "params": list(range(offset, offset + len(members)))})
        ordered.extend(members)
    ema = tuple(torch.full_like(parameter, .5 + index / 100)
                for index, (_, parameter) in enumerate(ordered))
    contract = visual()
    checkpoint = {
        "state_dict": {f"model.{name}": value for name, value in raw.items()},
        "hyper_parameters": as_dict(old_cfg),
        "motionexpert_architecture": architecture_metadata(old_cfg),
        "motionexpert_normalization": old_stats,
        "motionexpert_normalization_clip": None,
        "motionexpert_visual_conditioning": deepcopy(contract),
        "ema_state_format": "original_model_with_ema_optimizer_v1",
        "epoch": 59, "global_step": 29280,
        "optimizer_states": [{"opt": {"state": {}, "param_groups": groups},
                              "ema": ema, "current_step": 29280,
                              "decay": .992028, "every_n_steps": 1}],
    }
    monkeypatch.setattr(warm_start, "training_visual_contract", lambda current: deepcopy(contract))
    module = SimpleNamespace(cfg=cfg, model=model,
                             normalizer=MotionNormalizer(new_stats))

    def publish():
        torch.save(checkpoint, cfg.MODEL.WARM_START_CKPT_PATH)
        return cfg.MODEL.WARM_START_CKPT_PATH

    return SimpleNamespace(module=module, checkpoint=checkpoint, cfg=cfg, old_cfg=old_cfg,
                           old_stats=old_stats, new_stats=new_stats, raw=raw,
                           ema_by_name=dict(zip((name for name, _ in ordered), ema)), publish=publish)


@pytest.mark.parametrize("kind", ["ema", "raw"])
def test_loads_selected_weights_preserves_new_statistics_and_does_not_modify_source(warm_case, kind):
    case = warm_case
    case.cfg.MODEL.WARM_START_WEIGHTS = kind
    path = case.publish()
    before_bytes = Path(path).read_bytes()
    normalizer = case.module.normalizer
    expected_stats = deepcopy(normalizer.stats)
    result = warm_start.apply_warm_start(case.module)
    assert isinstance(result, dict) and case.module.warm_start_provenance == result
    assert result["source_checkpoint_sha256"] == hashlib.sha256(before_bytes).hexdigest()
    assert result["source_epoch"] == 59 and result["source_completed_epochs"] == 60
    assert result["source_global_step"] == 29280 and result["weights_kind"] == kind
    assert result["source_statistics"]["archive_fingerprint"] == "old-v2-archive"
    assert result["target_statistics"]["archive_fingerprint"] == "new-v3-archive"
    assert result["new_epoch"] == result["new_global_step"] == 0
    for key in ("optimizer_state_restored", "scheduler_state_restored",
                "ema_training_state_restored", "rng_state_restored"):
        assert result[key] is False
    assert case.module.normalizer is normalizer
    for key, value in expected_stats.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(normalizer.stats[key], value)
        else:
            assert normalizer.stats[key] == value
    for name, value in case.module.model.state_dict().items():
        expected = case.ema_by_name.get(name, case.raw[name]) if kind == "ema" else case.raw[name]
        assert torch.equal(value, expected), name
    assert Path(path).read_bytes() == before_bytes
    assert case.cfg.MODEL.CKPT_PATH is None and case.cfg.MODEL.INIT_CKPT_PATH is None
    assert case.cfg.TRAIN.NUM_EPOCHS == case.cfg.TRAIN.SCHEDULER_TOTAL_EPOCHS == 140
    assert case.cfg.TRAIN.WARMUP_EPOCHS == 20
    assert case.cfg.DATA.ARCHIVE_ROOT == "/new/v3/archive"


def test_existing_statistics_validation_still_rejects_cross_version_initialization(warm_case):
    case = warm_case
    with pytest.raises(ValueError, match="differ|statistics"):
        validate_checkpoint_statistics(case.cfg, case.checkpoint, case.module.normalizer)


@pytest.mark.parametrize("change", ["split", "basis", "clip", "model", "missing_statistics", "source_profile"])
def test_rejects_physical_or_split_contract_changes(warm_case, change):
    case = warm_case
    stats = case.checkpoint["motionexpert_normalization"]
    if change == "split":
        stats["split_fingerprint"] = "different-held-out-participants"
    elif change == "basis":
        stats["basis_mapping_fingerprint"] = "different-axes"
    elif change == "clip":
        case.checkpoint["motionexpert_normalization_clip"] = 5.0
    elif change == "model":
        stats["model_sha256"] = "0" * 64
    elif change == "missing_statistics":
        del case.checkpoint["motionexpert_normalization"]
    else:
        stats["contact_speed_m_s"] = .4
    case.publish()
    with pytest.raises((ValueError, TypeError, KeyError)):
        warm_start.apply_warm_start(case.module)


@pytest.mark.parametrize("change", ["revision", "normalization", "profile"])
def test_rejects_visual_identity_and_geometry_changes(warm_case, change):
    case = warm_case
    old_visual = case.checkpoint["motionexpert_visual_conditioning"]
    if change == "profile":
        case.checkpoint["motionexpert_visual_conditioning"] = visual("256x320")
        case.checkpoint["hyper_parameters"]["DATA"]["IMAGE_PROFILE"] = "256x320"
    elif change == "revision":
        old_visual["encoder"]["revision"] = "b" * 40
    else:
        old_visual["encoder"]["processor"]["image_mean"] = [.5, .5, .5]
    case.publish()
    with pytest.raises(ValueError, match="visual|profile|DINO|identity|IMAGE_PROFILE"):
        warm_start.apply_warm_start(case.module)


@pytest.mark.parametrize("epoch", [39, 60, -1, 59.5, True])
def test_rejects_unexpected_or_invalid_completed_epoch(warm_case, epoch):
    case = warm_case
    case.checkpoint["epoch"] = epoch
    case.publish()
    with pytest.raises((ValueError, TypeError)):
        warm_start.apply_warm_start(case.module)


@pytest.mark.parametrize("change", [
    "missing", "format", "phase", "duplicate_id", "missing_id", "shape", "dtype", "nonfinite",
])
def test_ema_rejection_is_atomic_and_never_falls_back_to_raw(warm_case, change):
    case = warm_case
    optimizer = case.checkpoint["optimizer_states"][0]
    groups = optimizer["opt"]["param_groups"]
    if change == "missing":
        optimizer.pop("ema")
    elif change == "format":
        case.checkpoint["ema_state_format"] = "unknown-ema-format"
    elif change == "phase":
        groups[1]["phase"] = groups[0]["phase"]
    elif change == "duplicate_id":
        groups[1]["params"][0] = groups[0]["params"][0]
    elif change == "missing_id":
        groups[-1]["params"].pop()
    else:
        averages = list(optimizer["ema"])
        if change == "shape":
            averages[-1] = torch.zeros(19)
        elif change == "dtype":
            averages[-1] = averages[-1].double()
        else:
            averages[-1].fill_(float("nan"))
        optimizer["ema"] = tuple(averages)
    before = {key: value.clone() for key, value in case.module.model.state_dict().items()}
    case.publish()
    with pytest.raises((ValueError, TypeError, KeyError)):
        warm_start.apply_warm_start(case.module)
    assert all(torch.equal(value, before[name]) for name, value in case.module.model.state_dict().items())


def test_raw_weight_nonfinite_rejection_is_atomic(warm_case):
    case = warm_case
    case.cfg.MODEL.WARM_START_WEIGHTS = "raw"
    case.checkpoint["state_dict"]["model.backbone.weight"].fill_(float("inf"))
    before = {key: value.clone() for key, value in case.module.model.state_dict().items()}
    case.publish()
    with pytest.raises((ValueError, TypeError)):
        warm_start.apply_warm_start(case.module)
    assert all(torch.equal(value, before[name]) for name, value in case.module.model.state_dict().items())


def test_pinned_source_sha_is_checked_before_any_parameter_is_loaded(warm_case):
    case = warm_case
    case.cfg.MODEL.WARM_START_SHA256 = "0" * 64
    before = {key: value.clone() for key, value in case.module.model.state_dict().items()}
    case.publish()
    with pytest.raises(ValueError, match="SHA256"):
        warm_start.apply_warm_start(case.module)
    assert all(torch.equal(value, before[name]) for name, value in case.module.model.state_dict().items())


def test_new_checkpoint_preserves_warm_start_provenance_and_current_statistics(warm_case, monkeypatch):
    from MotionExpert.module import uem_module

    case = warm_case
    case.publish()
    report = warm_start.apply_warm_start(case.module)
    monkeypatch.setattr(uem_module, "training_visual_contract", lambda cfg: visual())
    monkeypatch.setattr(uem_module, "validate_visual_resume", lambda cfg, checkpoint: None)
    saved = {"hyper_parameters": as_dict(case.cfg),
             "state_dict": {f"model.{name}": value.clone()
                            for name, value in case.module.model.state_dict().items()}}
    uem_module.UEM_Module.on_save_checkpoint(case.module, saved)
    assert saved["motionexpert_warm_start"] == report
    assert saved["motionexpert_warm_start"] is not case.module.warm_start_provenance
    assert saved["motionexpert_training_phase"]["origin"] == "weights_init"
    assert saved["motionexpert_training_phase"]["effective_shared_unfreeze_epoch"] == 0
    assert torch.equal(saved["motionexpert_normalization"]["motion_std"], case.new_stats["motion_std"])
    assert saved["motionexpert_normalization"]["archive_fingerprint"] == "new-v3-archive"
    case.module.warm_start_provenance = None
    uem_module.UEM_Module.on_load_checkpoint(case.module, saved)
    assert case.module.warm_start_provenance == report
    assert case.module.warm_start_provenance is not saved["motionexpert_warm_start"]
    saved["motionexpert_warm_start"]["format"] = "unknown"
    with pytest.raises(ValueError, match="provenance"):
        uem_module.UEM_Module.on_load_checkpoint(case.module, saved)


def test_new_stage_builds_fresh_optimizer_ema_and_140_epoch_schedule(warm_case, monkeypatch):
    from MotionExpert.ema import EMAOptimizer
    from MotionExpert.module import uem_module

    case = warm_case
    case.publish()
    monkeypatch.setattr(uem_module, "UniEgoMotion", TinyGroupedModel)
    module = uem_module.UEM_Module(case.cfg)
    warm_start.apply_warm_start(module)
    assert module.cfg.MODEL.TRAINING_ORIGIN == "weights_init"
    assert module._shared_unfreeze_epoch() == 0
    module._set_motion_expert_train_phase(0)
    assert all(parameter.requires_grad for parameter in module.model.parameters())
    optimizers, schedulers = module.configure_optimizers()
    optimizer, scheduler = optimizers[0], schedulers[0]["scheduler"]
    assert not optimizer.state
    assert scheduler.last_epoch == 0
    assert scheduler.base_lrs == pytest.approx([1e-4, 1e-4])
    assert scheduler.get_last_lr() == pytest.approx([5e-6, 5e-6])
    assert scheduler.lr_lambdas[0](19) == pytest.approx(1.0)
    assert scheduler.lr_lambdas[0](139) == pytest.approx(.1)
    ema = EMAOptimizer(optimizer, device=torch.device("cpu"), decay=case.cfg.TRAIN.EMA_DECAY)
    assert ema.current_step == 0 and not ema.ema_params
    # Zero gradients keep the selected weights fixed while exercising first-step
    # EMA initialization. This is a small isolated optimizer test, not a data run.
    selected = [parameter.detach().clone() for parameter in ema.all_parameters()]
    ema.step()
    ema.join()
    assert ema.current_step == 1
    assert all(torch.equal(value, expected) for value, expected in zip(ema.ema_params, selected))


def test_source_configuration_must_agree_with_embedded_normalization_clip(warm_case):
    case = warm_case
    case.checkpoint["hyper_parameters"]["DATA"]["NORMALIZATION_CLIP"] = 5.0
    case.publish()
    with pytest.raises(ValueError, match="clipping"):
        warm_start.apply_warm_start(case.module)


@pytest.mark.parametrize("field,value", [
    ("ENABLED", False), ("NUM_EXPERTS", 13), ("NUM_ROUTED_EXPERTS", 12),
    ("TOP_K", 1), ("SHARED_EXPERT", False), ("CONDITIONED_ROUTER", False),
])
def test_rejects_changed_routed_architecture_even_when_tensor_shapes_match(warm_case, field, value):
    case = warm_case
    case.checkpoint["hyper_parameters"]["MODEL"]["MOTION_EXPERT"][field] = value
    case.publish()
    with pytest.raises(ValueError, match="architecture|requires"):
        warm_start.apply_warm_start(case.module)


@pytest.mark.parametrize("change", ["different_path", "unconfigured_path", "wrong_origin"])
def test_rejects_initialization_without_the_configured_warm_start_phase(warm_case, tmp_path, change):
    case = warm_case
    path = case.publish()
    override = None
    if change == "different_path":
        override = tmp_path / "different-source.ckpt"
        override.write_bytes(Path(path).read_bytes())
    elif change == "unconfigured_path":
        case.cfg.MODEL.WARM_START_CKPT_PATH = None
        override = path
    else:
        case.cfg.MODEL.TRAINING_ORIGIN = "from_scratch"
    with pytest.raises(ValueError, match="override|explicit|constructing"):
        warm_start.apply_warm_start(case.module, override)


def test_same_module_cannot_be_warm_started_twice(warm_case):
    case = warm_case
    case.publish()
    report = warm_start.apply_warm_start(case.module)
    selected = {name: value.clone() for name, value in case.module.model.state_dict().items()}
    with pytest.raises(ValueError, match="once"):
        warm_start.apply_warm_start(case.module)
    assert case.module.warm_start_provenance == report
    assert all(torch.equal(value, selected[name]) for name, value in case.module.model.state_dict().items())


def test_configured_warm_start_survives_default_cli_initialization(warm_case):
    from MotionExpert.train import configure_initialization

    cfg = warm_case.cfg
    cfg.MODEL.TRAINING_ORIGIN = "from_scratch"
    cfg.MODEL.EFFECTIVE_SHARED_UNFREEZE_EPOCH = 0
    source = cfg.MODEL.WARM_START_CKPT_PATH
    assert configure_initialization(cfg) is None
    assert cfg.MODEL.WARM_START_CKPT_PATH == source
    assert cfg.MODEL.CKPT_PATH is None and cfg.MODEL.INIT_CKPT_PATH is None
    assert not hasattr(cfg.MODEL, "TRAINING_ORIGIN")
    assert not hasattr(cfg.MODEL, "EFFECTIVE_SHARED_UNFREEZE_EPOCH")


@pytest.mark.parametrize("mode", ["resume", "from_scratch", "init_weights", "warm_start"])
def test_explicit_cli_mode_overrides_configured_warm_start(warm_case, tmp_path, mode):
    from MotionExpert.train import configure_initialization

    cfg = warm_case.cfg
    selected = tmp_path / "selected.ckpt"
    kwargs = {mode: True if mode == "from_scratch" else selected}
    result = configure_initialization(cfg, **kwargs)
    fields = {"resume": "CKPT_PATH", "init_weights": "INIT_CKPT_PATH", "warm_start": "WARM_START_CKPT_PATH"}
    for field in fields.values():
        assert getattr(cfg.MODEL, field) == (str(selected.resolve()) if fields.get(mode) == field else None)
    assert result == (str(selected.resolve()) if mode == "resume" else None)


@pytest.mark.parametrize("modes", [
    ("resume", "init_weights"), ("resume", "warm_start"), ("resume", "from_scratch"),
    ("init_weights", "warm_start"), ("init_weights", "from_scratch"), ("warm_start", "from_scratch"),
])
def test_mixed_explicit_initialization_modes_are_rejected(warm_case, tmp_path, modes):
    from MotionExpert.train import configure_initialization

    kwargs = {mode: True if mode == "from_scratch" else tmp_path / f"{mode}.ckpt" for mode in modes}
    with pytest.raises(ValueError, match="Choose one"):
        configure_initialization(warm_case.cfg, **kwargs)


@pytest.mark.parametrize("fields", [
    ("CKPT_PATH", "WARM_START_CKPT_PATH"), ("CKPT_PATH", "INIT_CKPT_PATH"),
    ("INIT_CKPT_PATH", "WARM_START_CKPT_PATH"),
])
def test_mixed_configured_initialization_modes_are_rejected(warm_case, tmp_path, fields):
    from MotionExpert.train import configure_initialization

    cfg = warm_case.cfg
    for field in ("CKPT_PATH", "INIT_CKPT_PATH", "WARM_START_CKPT_PATH"):
        setattr(cfg.MODEL, field, str(tmp_path / f"{field}.ckpt") if field in fields else None)
    with pytest.raises(ValueError, match="mutually exclusive"):
        configure_initialization(cfg)
