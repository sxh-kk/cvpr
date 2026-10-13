"""A new representation must not freeze randomly initialized projections."""
import unittest
from copy import deepcopy
from pathlib import Path
import tempfile
from unittest.mock import patch
import torch
from torch import nn
from MotionExpert.config import as_dict, config_from_dict, motion_expert_config
from MotionExpert.dataset.normalization import MotionNormalizer
from MotionExpert.module.uem_module import (
    UEM_Module, restore_training_phase, training_phase_from_checkpoint,
)
from MotionExpert.tests.fixtures import dense_stats
from MotionExpert.train import configure_initialization


class DenseTrainPhaseTests(unittest.TestCase):
    def module(self):
        module = UEM_Module.__new__(UEM_Module)
        nn.Module.__init__(module)
        module.cfg = motion_expert_config()
        module.cfg.MODEL.MOTION_EXPERT.SHARED_UNFREEZE_EPOCH = 30
        module.is_lstm = False
        module.model = nn.Module()
        module.model.input_process = nn.Linear(245, 768)
        module.model.output_process = nn.Linear(768, 245)
        module.model.embed_traj_cond = nn.Linear(18, 768)
        module.model.global_token = nn.Parameter(torch.zeros(768))
        module.normalizer = MotionNormalizer(dense_stats())
        return module

    def checkpoint(self, module, *, legacy=False):
        checkpoint = {"hyper_parameters": as_dict(module.cfg),
                      "state_dict": module.model.state_dict()}
        module.on_save_checkpoint(checkpoint)
        if legacy:
            checkpoint.pop("motionexpert_training_phase")
        return checkpoint

    def resumed(self, checkpoint):
        module = self.module()
        module.cfg = config_from_dict(checkpoint["hyper_parameters"])
        configure_initialization(module.cfg, resume=Path("/tmp/resumed-dense.ckpt"))
        restore_training_phase(module.cfg, checkpoint)
        module.on_load_checkpoint(checkpoint)
        return module

    def test_fresh_training_learns_all_new_projections_at_epoch_zero(self):
        module = self.module()
        module._set_motion_expert_train_phase(0)
        self.assertTrue(all(parameter.requires_grad for parameter in module.model.parameters()))
        output = module.model.output_process(module.model.input_process(torch.ones(1, 245)))
        output.square().mean().backward()
        self.assertGreater(module.model.output_process.weight.grad.abs().sum().item(), 0)
        self.assertEqual(module._shared_unfreeze_epoch(), 0)

    def test_dense_warm_start_can_use_requested_shared_phase(self):
        module = self.module()
        module.cfg.MODEL.INIT_CKPT_PATH = 'explicit_dense_weights.pt'
        module._set_motion_expert_train_phase(0)
        self.assertTrue(all(not parameter.requires_grad for parameter in module.model.parameters()))
        module._set_motion_expert_train_phase(30)
        self.assertTrue(all(parameter.requires_grad for parameter in module.model.parameters()))

    def test_fresh_checkpoint_resume_and_second_resume_keep_epoch_zero(self):
        source = self.module()
        checkpoint = self.checkpoint(source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "phase.ckpt"
            torch.save(checkpoint, path)
            checkpoint = torch.load(path, weights_only=True)
        self.assertEqual(checkpoint["motionexpert_training_phase"]["origin"], "from_scratch")
        resumed = self.resumed(checkpoint)
        resumed._set_motion_expert_train_phase(5)
        self.assertTrue(all(parameter.requires_grad for parameter in resumed.model.parameters()))
        self.assertEqual(resumed._shared_unfreeze_epoch(), 0)
        resumed_again = self.resumed(self.checkpoint(resumed))
        self.assertEqual(resumed_again._shared_unfreeze_epoch(), 0)

    def test_warm_checkpoint_resume_and_second_resume_keep_epoch_thirty(self):
        source = self.module()
        source.cfg.MODEL.INIT_CKPT_PATH = "explicit_dense_weights.pt"
        resumed = self.resumed(self.checkpoint(source))
        self.assertIsNone(resumed.cfg.MODEL.INIT_CKPT_PATH)
        self.assertEqual(resumed._shared_unfreeze_epoch(), 30)
        resumed._set_motion_expert_train_phase(29)
        self.assertTrue(all(not parameter.requires_grad for parameter in resumed.model.parameters()))
        resumed._set_motion_expert_train_phase(30)
        self.assertTrue(all(parameter.requires_grad for parameter in resumed.model.parameters()))
        resumed_again = self.resumed(self.checkpoint(resumed))
        self.assertEqual(resumed_again._shared_unfreeze_epoch(), 30)

    def test_legacy_uses_original_init_path_instead_of_resume_path(self):
        for warm_start in (False, True):
            with self.subTest(warm_start=warm_start):
                source = self.module()
                source.cfg.MODEL.CKPT_PATH = "an_earlier_resume.ckpt"
                source.cfg.MODEL.INIT_CKPT_PATH = "dense_init.pt" if warm_start else None
                with patch("MotionExpert.module.uem_module.logger.warning") as warning:
                    resumed = self.resumed(self.checkpoint(source, legacy=True))
                self.assertEqual(warning.call_count, 0 if warm_start else 1)
                self.assertEqual(resumed._shared_unfreeze_epoch(), 30 if warm_start else 0)

    def test_unambiguous_legacy_epoch_zero_does_not_warn(self):
        source = self.module()
        source.cfg.MODEL.CKPT_PATH = "an_earlier_resume.ckpt"
        source.cfg.MODEL.MOTION_EXPERT.SHARED_UNFREEZE_EPOCH = 0
        with patch("MotionExpert.module.uem_module.logger.warning") as warning:
            resumed = self.resumed(self.checkpoint(source, legacy=True))
        warning.assert_not_called()
        self.assertEqual(resumed._shared_unfreeze_epoch(), 0)

    def test_constructor_persists_training_origin_before_hyperparameters(self):
        for warm_start in (False, True):
            with self.subTest(warm_start=warm_start):
                source = self.module()
                cfg = source.cfg
                cfg.DATA.STATS_PATH = dense_stats()
                cfg.MODEL.INIT_CKPT_PATH = "dense_init.pt" if warm_start else None
                with patch("MotionExpert.module.uem_module.UniEgoMotion", return_value=source.model):
                    module = UEM_Module(cfg)
                self.assertEqual(module.hparams["MODEL"]["TRAINING_ORIGIN"],
                                 "dense_warm_start" if warm_start else "from_scratch")
                self.assertEqual(module.hparams["MODEL"]["EFFECTIVE_SHARED_UNFREEZE_EPOCH"],
                                 30 if warm_start else 0)

    def test_resume_phase_is_resolved_before_scheduler_creation(self):
        for warm_start in (False, True):
            with self.subTest(warm_start=warm_start):
                source = self.module()
                source.cfg.MODEL.INIT_CKPT_PATH = "dense_init.pt" if warm_start else None
                source.cfg.MODEL.MOTION_EXPERT.UNFREEZE_RAMP_EPOCHS = 4
                source.cfg.TRAIN.WARMUP_EPOCHS = 0
                source.cfg.TRAIN.MIN_LR_RATIO = 1.0
                resumed = self.resumed(self.checkpoint(source))
                optimizers, schedulers = resumed.configure_optimizers()
                self.assertEqual(optimizers[0].param_groups[0]["phase"], "unfrozen")
                schedule = schedulers[0]["scheduler"].lr_lambdas[0]
                self.assertEqual(schedule(5), 0.0 if warm_start else 1.0)
                self.assertEqual(schedule(30), .25 if warm_start else 1.0)

    def test_malformed_and_inconsistent_phase_metadata_is_rejected(self):
        source = self.module()
        checkpoint = self.checkpoint(source)
        bad_phases = [None, {},
            {"format": "future-version", "origin": "from_scratch", "effective_shared_unfreeze_epoch": 0},
            {"format": "motionexpert-training-phase-v1", "origin": "unknown", "effective_shared_unfreeze_epoch": 0},
            {"format": "motionexpert-training-phase-v1", "origin": "from_scratch", "effective_shared_unfreeze_epoch": True},
            {"format": "motionexpert-training-phase-v1", "origin": "from_scratch", "effective_shared_unfreeze_epoch": -1},
            {"format": "motionexpert-training-phase-v1", "origin": "from_scratch", "effective_shared_unfreeze_epoch": 30},
            {"format": "motionexpert-training-phase-v1", "origin": "dense_warm_start", "effective_shared_unfreeze_epoch": 30}]
        for phase in bad_phases:
            with self.subTest(phase=phase):
                bad = deepcopy(checkpoint)
                bad["motionexpert_training_phase"] = phase
                with self.assertRaises(ValueError):
                    training_phase_from_checkpoint(bad)
        with self.assertRaisesRegex(ValueError, "resolved resume phase"):
            other = self.module()
            other.cfg.MODEL.INIT_CKPT_PATH = "dense_init.pt"
            other.on_load_checkpoint(checkpoint)

    def test_resume_cannot_change_warm_shared_epoch(self):
        source = self.module()
        source.cfg.MODEL.INIT_CKPT_PATH = "dense_init.pt"
        cfg = motion_expert_config()
        cfg.MODEL.MOTION_EXPERT.SHARED_UNFREEZE_EPOCH = 20
        with self.assertRaisesRegex(ValueError, "cannot change"):
            restore_training_phase(cfg, self.checkpoint(source))
        self.assertFalse(hasattr(cfg.MODEL, "TRAINING_ORIGIN"))

    def test_explicit_fresh_and_init_override_configured_resume(self):
        for from_scratch in (False, True):
            with self.subTest(from_scratch=from_scratch):
                module = self.module()
                cfg = module.cfg
                cfg.MODEL.CKPT_PATH = "configured_resume.ckpt"
                cfg.MODEL.INIT_CKPT_PATH = "configured_init.pt"
                cfg.MODEL.TRAINING_ORIGIN = "dense_warm_start"
                cfg.MODEL.EFFECTIVE_SHARED_UNFREEZE_EPOCH = 30
                new_init = None if from_scratch else Path("explicit_init.pt")
                resume = configure_initialization(cfg, from_scratch=from_scratch, init_weights=new_init)
                self.assertIsNone(resume)
                self.assertIsNone(cfg.MODEL.CKPT_PATH)
                self.assertEqual(cfg.MODEL.INIT_CKPT_PATH, str(new_init.resolve()) if new_init else None)
                self.assertFalse(hasattr(cfg.MODEL, "TRAINING_ORIGIN"))
                self.assertEqual(module._shared_unfreeze_epoch(), 0 if from_scratch else 30)

    def test_explicit_resume_and_configured_initialization_still_work(self):
        cfg = motion_expert_config()
        cfg.MODEL.CKPT_PATH = "configured_resume.ckpt"
        cfg.MODEL.INIT_CKPT_PATH = "configured_init.pt"
        self.assertEqual(configure_initialization(cfg, resume=Path("explicit_resume.ckpt")),
                         str(Path("explicit_resume.ckpt").resolve()))
        self.assertIsNone(cfg.MODEL.INIT_CKPT_PATH)
        self.assertEqual(configure_initialization(cfg), cfg.MODEL.CKPT_PATH)
        cfg.MODEL.CKPT_PATH = None
        cfg.MODEL.INIT_CKPT_PATH = "configured_init.pt"
        self.assertIsNone(configure_initialization(cfg))
        self.assertEqual(cfg.MODEL.INIT_CKPT_PATH, "configured_init.pt")

    def test_initialization_choices_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "Choose one"):
            configure_initialization(motion_expert_config(), resume="resume.ckpt", from_scratch=True)


if __name__ == '__main__':
    unittest.main()
