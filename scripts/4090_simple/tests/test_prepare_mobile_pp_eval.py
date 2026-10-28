"""Independent Mobile deployment guards and tiny actual CPU tensor export."""
from copy import deepcopy
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import save_file

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("mobile_eval_test", SCRIPTS / "prepare_mobile_pp_eval.py")
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def encoder(version="old"):
    return {"checkpoint": "/metadata/dino", "local_file_sha256": {"model.safetensors": "weights"},
        "torch_version": version, "transformers_version": version,
        "image_process": {"numpy_version": "1.26.4", "pillow_version": "11.3.0", "height": 256, "width": 320},
        "shape": [5, 1024], "token_order": ["cls", "register_0", "register_1", "register_2", "register_3"],
        "storage_dtype": "float32", "optimization": {"compute_precision": "bf16"}}


def signature(model):
    return [0x44534143, 1, model["action_dim"], model["action_chunk_size"], model["hidden_dim"], model["num_blocks"],
        model["nhead"], model["view_feature_dim"], model["odim"], model["time_dim"], int(model["state_null_token"]), 2,
        struct.unpack("q", struct.pack("d", model["state_drop_prob"]))[0],
        struct.unpack("q", struct.pack("d", model["dropout"]))[0], 11, 768]


def tiny_weights(path, model, *, motion=True):
    weights = {"vlm_model.weight": torch.full((2, 2), 3.), "action_header.weight": torch.full((2, 2), 5.),
               "action_header._architecture_signature": torch.tensor(signature(model), dtype=torch.int64)}
    if motion:
        weights["motion_conditioner.model.weight"] = torch.full((2, 2), 9.)
    save_file(weights, path)


class TinyModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty((2, 2)))


class TinyHead(TinyModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.state_drop_prob = kwargs["state_drop_prob"]
        self.transformer_blocks = torch.nn.ModuleList([torch.nn.Identity() for _ in range(kwargs["action_num_blocks"])])
        self.register_buffer("_architecture_signature", torch.zeros(16, dtype=torch.int64))


def package(name, **attributes):
    value = ModuleType(name)
    value.__path__ = []
    value.__dict__.update(attributes)
    return value


class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source, self.release = self.root / "source", self.root / "local_release"
        self.resources, self.repo, self.output = self.root / "resources", self.root / "repo", self.root / "deployment"
        self.source.mkdir()
        self.release.mkdir()
        self.repo.mkdir()
        self.remote_release = "/remote/mobile100"
        self.config = {"model": {"action_architecture": "dual_source_v1", "num_blocks": 6,
            "action_dim": 80, "odim": 45, "action_chunk_size": 30, "state_drop_prob": 0., "state_null_token": True,
            "tune_motion": True, "tune_vlm": True, "rtc": False, "hidden_dim": 8, "nhead": 2,
            "view_feature_dim": 4, "time_dim": 4, "qk_norm": "layer_norm", "dropout": 0.,
            "pretrained_joint_checkpoint": "/source38/checkpoints/ckpt_38000/model.safetensors",
            "model_name_or_path": "/old/qwen", "motion_checkpoint": "/old/motion.ckpt"},
            "train": {"train_batch_size": 8, "gradient_accumulation_steps": 1, "max_training_steps": 10000,
                "resume_from_checkpoint": None, "skip_resumed_steps": False, "hf_token": ".hf_token",
                "output_dir": "/old/output", "deepspeed_config": "/old/zero3.json"},
            "data": {"root_dir": "/old/data", "dino_cache_root": "/old/dino", "transform": {
                "repack": {"action_keys": ["action.body_token_v1_1", "action[:14]"], "state_keys": ["observation.state"],
                    "action_mask_key": "action_mask", "pad_action_dim": 80, "pad_state_dim": 45},
                "model": {"shared_image_profile": "256x320", "motion_image_policy": "vlm_color_clean_motion"},
                "field": {"stat_path": self.remote_release + "/packed/source_dataset_statistics.json",
                    "action_min": [-1., 0.], "action_max": [1., 0.], "state_min": [-2.], "state_max": [2.]}}},
            "wandb": {key: None for key in deploy.EXPECTED_TRACKER}}
        self.base_config = deepcopy(self.config)
        self.write("prepared_run_config.json", self.config)
        self.config["train"]["hf_token"] = "PosixPath('.hf_token')"
        self.config["wandb"].update(deploy.EXPECTED_TRACKER)
        self.write("run_config.json", self.config)
        for name in ("argv.txt", "prepared_argv.txt"):
            (self.source / name).write_text("--model.motion_checkpoint=/old/motion.ckpt\n--model.action_architecture=dual_source_v1\n")
        self.write("dataset_statistics.json", {"inherited": "joint38000"})
        self.stat_sha = deploy.sha256(self.source / "dataset_statistics.json")
        self.preparation = {"contract": deploy.PREPARATION_CONTRACT, "task_id": deploy.TASK_ID,
            "source_step": 38000, "source_checkpoint_sha256": deploy.SOURCE_SHA,
            "source_checkpoint": self.config["model"]["pretrained_joint_checkpoint"], "fresh_run": True,
            "initial_global_step": 0, "optimizer_scheduler_rng_restored": False,
            "gpu_count": 8, "per_gpu_batch": 8, "gradient_accumulation_steps": 1, "effective_global_batch": 64,
            "train_episodes": 100, "total_frames": 61806, "train_windows": 55906, "max_optimizer_steps": 10000,
            "validation_role": deploy.VALIDATION_ROLE, "Mobile_validation_selected": False,
            "expected_validation_rows": list(range(40)), "release": self.remote_release,
            "bindings_sha256": {self.config["data"]["transform"]["field"]["stat_path"]: self.stat_sha},
            "config_sha256": deploy.sha256(self.source / "prepared_run_config.json"),
            "argv_sha256": deploy.sha256(self.source / "prepared_argv.txt"), "code_sha256": {}}
        for prefix in deploy.CRITICAL_PREFIXES:
            relative = prefix + "fixture.py"
            p = self.repo / relative
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("# isolated construction fixture\n")
            self.preparation["code_sha256"]["/frozen/632968d/" + relative] = deploy.sha256(p)
        self.make_release()
        self.write("preparation.json", self.preparation)
        self.initialization = {"source_checkpoint": self.preparation["source_checkpoint"], "source_step": 38000,
            "strict_joint_loader": True, "fresh_optimizer_scheduler_rng": True, "global_batch": 64,
            "backend": {"backend": "fused", "foreach": False, "initialized_parameter_states": 0}}
        self.write("task_initialization.json", self.initialization)
        self.write("task_window_selection.json", {"train_windows": 55906, "train_episode_indices": list(range(100)),
            "train_task_id": deploy.TASK_ID, "validation_role": deploy.VALIDATION_ROLE, "Mobile_validation_selected": False,
            "validation_windows": 40, "validation_rows": list(range(40))})
        self.state = {"state": "complete", "returncode": 0, "run_directory": deploy.EXPECTED_RUN,
            "task_id": deploy.TASK_ID, "fresh_run": True, "initial_global_step": 0,
            "optimizer_scheduler_rng_restored": False, "expected_ready_sha256": self.preparation["expected_ready_sha256"]}
        self.write("training_state.json", self.state)
        self.retention = {"saved_update": 10000, "all_models_preserved": True,
            "all_saved_model_steps": deploy.SAVE_STEPS, "full_recovery_steps": [9500, 10000]}
        self.write("checkpoint_retention.json", self.retention)
        self.write("gpu_check_update10000.json", {"status": "passed", "completed_updates": 10000})
        tiny_weights(self.source / "model.safetensors", self.config["model"])
        self.model_sha = deploy.sha256(self.source / "model.safetensors")
        self.model_bytes = (self.source / "model.safetensors").stat().st_size
        self.final = {"contract": "cvpr-mobile-pp100-final-training-artifact-acceptance-v1", "status": "passed",
            "project": deploy.EXPECTED_RUN, "configuration": {"sha256": self.preparation["config_sha256"]},
            "models": [{"step": 10000, "path": deploy.EXPECTED_RUN + "/checkpoints/ckpt_10000/model.safetensors",
                "sha256": self.model_sha, "size_bytes": self.model_bytes}]}
        for name, key in [("training_state.json", "state_file"), ("checkpoint_retention.json", "retention_file"),
                          ("gpu_check_update10000.json", "health_file")]:
            self.final[key] = {"sha256": deploy.sha256(self.source / name)}
        self.write("FINAL_TRAINING_ACCEPTANCE_001.json", self.final)
        self.receipt = {"contract": deploy.BACKUP_CONTRACT, "task_id": deploy.TASK_ID, "step": 10000, "optimizer_copied": False,
            "completed_retention_marker": self.retention,
            "run_identity": {"run_directory": deploy.EXPECTED_RUN, "run_config_sha256": deploy.sha256(self.source / "run_config.json"),
                "preparation_sha256": deploy.sha256(self.source / "preparation.json")},
            "model": {"source_path": deploy.EXPECTED_RUN + "/checkpoints/ckpt_10000/model.safetensors", "bytes": self.model_bytes,
                "source_sha256_before": self.model_sha, "source_sha256_after": self.model_sha, "local_sha256": self.model_sha},
            "metadata": {name: deploy.sha256(self.source / name) for name in deploy.METADATA}}
        self.write("backup_receipt.json", self.receipt)
        for path in [self.resources / "joint20k/qwen", self.resources / "dinov2-with-registers-large",
                     self.resources / "simple_local_usd_v1"]:
            path.mkdir(parents=True, exist_ok=True)
        self.reference_contract = {"encoder": encoder(), "shared_image_profile": "256x320",
            "history": {"fps": 5, "frames": 6, "shape": [6, 5, 1024], "augmentation": "none", "timestamps": "simulation_time"},
            "processor_overrides_required": {"use_fast": False, "do_resize": False, "do_center_crop": False}}
        deploy.write_json(self.resources / "joint20k/dino_training_contract.json", self.reference_contract)
        old = self.resources / "joint20k/motion/joint_motion.ckpt"
        old.parent.mkdir()
        old_payload = {"state_dict": {"weight": torch.full((2, 2), -7.)}, "motionexpert_normalization": {"mean": [0.]},
            "motionexpert_visual_conditioning": {"format": "motionexpert-visual-v1", "mode": "shared_rgb",
                "image_process": encoder()["image_process"], "encoder": encoder()}}
        torch.save(old_payload, old)
        metadata = {key: value for key, value in old_payload.items() if key != "state_dict"}
        self.motion_metadata_sha = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
        construction_files = []
        for name in deploy.QWEN_METADATA_NAMES:
            path = self.resources / "joint20k/qwen" / name
            path.write_text("accepted fixture " + name)
            construction_files.append(path)
        construction_files.append(self.resources / "joint20k/dino_training_contract.json")
        deploy.write_json(self.resources / "manifest.json", {"format": "cvpr-independent-4090-resources-v1", "step": 20000,
            "files": [{"path": str(p.relative_to(self.resources)), "bytes": p.stat().st_size, "sha256": deploy.sha256(p)}
                      for p in construction_files], "motion_validation": {"exported_metadata_sha256": self.motion_metadata_sha}})
        self.patchers = []
        self.addCleanup(lambda: [p.stop() for p in reversed(self.patchers)])
        self.start(patch.object(deploy, "EXPECTED_MODEL_SHA", self.model_sha))
        self.start(patch.object(deploy, "EXPECTED_MODEL_BYTES", self.model_bytes))
        self.start(patch.object(deploy, "EXPECTED_CONFIG_SHA", self.preparation["config_sha256"]))
        self.start(patch.object(deploy, "FINAL_ACCEPTANCE_SHA", deploy.sha256(self.source / "FINAL_TRAINING_ACCEPTANCE_001.json")))
        self.start(patch.object(deploy, "STAT_SHA", self.stat_sha))
        self.start(patch.object(deploy, "RESOURCES_MANIFEST_SHA", deploy.sha256(self.resources / "manifest.json")))
        self.start(patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}))
        self.saved_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), self.saved_path))

    def start(self, patcher):
        self.patchers.append(patcher)
        return patcher.start()

    def write(self, name, value):
        deploy.write_json(self.source / name, value)

    def rebind(self, name):
        self.receipt["metadata"][name] = deploy.sha256(self.source / name)
        self.write("backup_receipt.json", self.receipt)

    def make_release(self):
        field = self.config["data"]["transform"]["field"]
        self.pack = {"contract": deploy.PACK_CONTRACT, "task_id": deploy.TASK_ID, "selection_policy": "native_only",
            "normalization_refitted": False, "train_episodes": 100, "total_frames": 61806, "train_windows": 55906,
            "checkpoint_contract": {"source_step": 38000, "source_checkpoint_sha256": deploy.SOURCE_SHA,
                "bounds_refitted": False, "statistics": {"sha256": self.stat_sha},
                "bounds_sha256": {name: deploy.digest(field[name]) for name in ("action_min", "action_max", "state_min", "state_max")}}}
        self.write("release_metadata/packed/PACK_READY.json", self.pack)
        deploy.write_json(self.release / "packed/PACK_READY.json", self.pack)
        stat = self.release / "packed/source_dataset_statistics.json"
        stat.write_bytes((self.source / "dataset_statistics.json").read_bytes())
        self.manifest = {"contract": "unifolm-dino-native-frames-v1", "state": "complete", "episode_indices": list(range(100)),
            "source": {"total_episodes": 100, "total_frames": 61806, "fps": 30, "dataset_root": "/remote/train"},
            "encoder": encoder("mobile")}
        self.write("release_metadata/dino/manifest.json", self.manifest)
        original = deepcopy(self.manifest)
        original["source"]["dataset_root"] = "/original/train"
        deploy.write_json(self.release / "dino/manifest.json", original)
        self.relocation = {"contract": "cvpr-pico-mobile-pp-dino-host-relocation-v1", "task_id": deploy.TASK_ID, "status": "passed",
            "data_and_features_byte_identical": True, "PACK_READY_modified": False,
            "pack_ready_sha256": deploy.sha256(self.source / "release_metadata/packed/PACK_READY.json"),
            "manifest_sha256": deploy.sha256(self.source / "release_metadata/dino/manifest.json"),
            "original_manifest_sha256": deploy.sha256(self.release / "dino/manifest.json")}
        self.write("release_metadata/dino/REMOTE_DINO_RELOCATION.json", self.relocation)
        ready = {"contract": deploy.PRODUCTION_CONTRACT, "task_id": deploy.TASK_ID, "status": "passed",
            "remote_production_ready": True, "DINO_relocation_pending": False, "split_restriction": "train_only",
            "selection_policy": "native_only", "normalization_refitted": False,
            "train_episodes": 100, "total_frames": 61806, "train_windows": 55906,
            "pack_ready_sha256": self.relocation["pack_ready_sha256"], "cache_manifest_sha256": self.relocation["manifest_sha256"],
            "remote_relocation_binding": {"sha256": deploy.sha256(self.source / "release_metadata/dino/REMOTE_DINO_RELOCATION.json")}}
        self.write("release_metadata/A100_PICO_DATASET_READY.json", ready)
        self.preparation["expected_ready_sha256"] = deploy.sha256(self.source / "release_metadata/A100_PICO_DATASET_READY.json")

    def fake_model_modules(self):
        """Only constructors/metadata adapters are fake; safetensors and export are real."""
        read_checkpoint = lambda p: torch.load(p, map_location="cpu", weights_only=False)
        normalizer = lambda stats: SimpleNamespace(stats=stats, clip=stats.get("normalization_clip"))
        alias = lambda model, state: state
        return {
            "transformers": package("transformers", AutoConfig=SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace()),
                                    Qwen3VLForConditionalGeneration=TinyModel),
            "psi": package("psi"), "psi.models": package("psi.models"), "psi.deploy": package("psi.deploy"),
            "psi.models.dual_source_action": package("psi.models.dual_source_action", DualSourceActionTransformer=TinyHead),
            "psi.models.checkpoint_aliases": package("psi.models.checkpoint_aliases", restore_shared_tensor_aliases=alias),
            "psi.deploy.dual_source_runtime": package("psi.deploy.dual_source_runtime", load_dino_contract=lambda p: deploy.read(p)),
            "MotionExpert": package("MotionExpert"), "MotionExpert.dataset": package("MotionExpert.dataset"),
            "MotionExpert.checkpoints": package("MotionExpert.checkpoints", read_checkpoint=read_checkpoint,
                checkpoint_config=lambda value: SimpleNamespace(), validate_checkpoint_statistics=lambda *args: None),
            "MotionExpert.config": package("MotionExpert.config", as_dict=lambda cfg: {}, architecture_metadata=lambda cfg: {}),
            "MotionExpert.dataset.normalization": package("MotionExpert.dataset.normalization", MotionNormalizer=normalizer),
            "MotionExpert.model": package("MotionExpert.model", UniEgoMotion=TinyModel),
            "MotionExpert.visual_contract": package("MotionExpert.visual_contract",
                checkpoint_visual_contract=lambda value: value["motionexpert_visual_conditioning"],
                compare_visual_contracts=lambda *args, **kwargs: {}),
        }

    def test_exact_mobile_source_passes_without_old_handover_contracts(self):
        source, config, preparation, receipt = deploy.validate_source(self.source, self.release)
        self.assertEqual(source, self.source)
        self.assertEqual(preparation["effective_global_batch"], 64)
        self.assertEqual(config["model"]["state_drop_prob"], 0.)
        self.assertEqual(receipt["step"], 10000)

    def test_qwen_tokenizer_chat_and_config_metadata_must_match_accepted_inventory(self):
        report = deploy.validate_construction_metadata(self.resources)
        self.assertEqual(report["qwen_metadata_files_verified"], 11)
        self.assertFalse(report["old_joint_or_motion_parameter_hashes_recomputed"])
        path = self.resources / "joint20k/qwen/chat_template.jinja"
        path.write_text("different prompt with otherwise compatible model shapes")
        with self.assertRaisesRegex(ValueError, "construction metadata changed"):
            deploy.validate_construction_metadata(self.resources)

    def test_motion_template_metadata_is_bound_without_using_old_parameters(self):
        path = self.resources / "joint20k/motion/joint_motion.ckpt"
        with patch.dict(sys.modules, self.fake_model_modules()):
            report = deploy.validate_motion_template_metadata(path, self.motion_metadata_sha)
            self.assertFalse(report["old_parameter_values_used"])
            payload = torch.load(path, weights_only=False)
            payload["motionexpert_normalization"]["mean"] = [3.]
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "Motion construction metadata"):
                deploy.validate_motion_template_metadata(path, self.motion_metadata_sha)

    def test_receipt_is_required_before_consuming_streaming_model(self):
        (self.source / "backup_receipt.json").unlink()
        (self.source / "model.safetensors").write_bytes(b"unfinished download")
        with self.assertRaisesRegex(ValueError, "backup_receipt"):
            deploy.validate_source(self.source, self.release)

    def test_wrong_task_step_run_or_sha_rejected(self):
        changes = [("task_id", "simple/G1WholebodyHandoverTeleop-v0"), ("step", 9500),
                   ("contract", "cvpr-handover149-local-checkpoint-backup-v1")]
        for key, value in changes:
            bad = deepcopy(self.receipt)
            bad[key] = value
            with self.subTest(key=key):
                self.write("backup_receipt.json", bad)
                with self.assertRaises(ValueError):
                    deploy.validate_source(self.source, self.release)
        for section, key, value in [("model", "source_sha256_after", "0" * 64),
                ("run_identity", "run_directory", "/old_handover_run")]:
            bad = deepcopy(self.receipt)
            bad[section][key] = value
            self.write("backup_receipt.json", bad)
            with self.assertRaises(ValueError):
                deploy.validate_source(self.source, self.release)

    def test_actual_model_byte_change_cannot_reuse_backup_sha(self):
        path = self.source / "model.safetensors"
        old = path.read_bytes()
        path.write_bytes(old[:-1] + bytes([old[-1] ^ 1]))
        with self.assertRaisesRegex(ValueError, "model bytes"):
            deploy.validate_source(self.source, self.release)

    def test_runtime_cannot_change_state_retention_clean_policy_or_optimizer_resume(self):
        for group, key, value in [("model", "state_drop_prob", .5), ("model", "num_blocks", 12),
                                   ("train", "resume_from_checkpoint", "latest")]:
            bad = deepcopy(self.config)
            bad[group][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Runtime configuration"):
                deploy.validate_runtime_metadata(self.source, bad, self.preparation)
        bad = deepcopy(self.config)
        bad["data"]["transform"]["model"]["motion_image_policy"] = "shared_rgb"
        with self.assertRaises(ValueError):
            deploy.bind_training_motion_image_policy(self.reference_contract, bad)

    def test_statistics_and_inline_bounds_cannot_be_refit(self):
        bad = deepcopy(self.config)
        bad["data"]["transform"]["field"]["action_min"][0] = -3.
        with self.assertRaisesRegex(ValueError, "bound changed"):
            deploy.validate_release(self.source, self.release, bad, self.preparation)
        self.write("dataset_statistics.json", {"refitted": True})
        with self.assertRaisesRegex(ValueError, "statistics changed"):
            deploy.validate_release(self.source, self.release, self.config, self.preparation)

    def test_versions_declared_but_dino_weights_geometry_precision_must_match(self):
        result = deploy.recorded_encoder_transition(encoder(), encoder("mobile"))
        self.assertFalse(result["numerical_bitwise_equivalence_claimed"])
        self.assertEqual(result["versions"]["torch_version"]["mobile100_training"], "mobile")
        mutations = [("local_file_sha256", {"model.safetensors": "other"}), ("shape", [1, 1024]),
                     ("storage_dtype", "float16"), ("optimization", {"compute_precision": "fp32"})]
        for key, value in mutations:
            actual = encoder("mobile")
            actual[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "DINO transition"):
                deploy.recorded_encoder_transition(encoder(), actual)
        actual = encoder("mobile")
        actual["image_process"]["width"] = 480
        with self.assertRaises(ValueError):
            deploy.recorded_encoder_transition(encoder(), actual)

    def test_original_manifest_is_not_mistaken_for_actual_training_manifest(self):
        self.assertNotEqual(deploy.sha256(self.release / "dino/manifest.json"),
                            deploy.sha256(self.source / "release_metadata/dino/manifest.json"))
        result = deploy.validate_release(self.source, self.release, self.config, self.preparation)
        self.assertTrue(result["encoder_unchanged_by_host_relocation"])
        bad = deepcopy(self.relocation)
        bad["original_manifest_sha256"] = "0" * 64
        self.write("release_metadata/dino/REMOTE_DINO_RELOCATION.json", bad)
        with self.assertRaisesRegex(ValueError, "relocation proof"):
            deploy.validate_release(self.source, self.release, self.config, self.preparation)

    def test_strict_cpu_component_load_rejects_old_state_signature_or_missing_motion(self):
        with patch.dict(sys.modules, self.fake_model_modules()):
            report = deploy.strict_component_check(self.source / "model.safetensors", self.config, "/metadata/qwen")
            self.assertTrue(report["strict_VLM_load"] and report["strict_action_load"])
            self.assertFalse(report["forward_performed"])
            wrong = deepcopy(self.config["model"])
            wrong["state_drop_prob"] = .5
            tiny_weights(self.source / "bad.safetensors", wrong)
            with self.assertRaisesRegex(ValueError, "signature"):
                deploy.strict_component_check(self.source / "bad.safetensors", self.config, "/metadata/qwen")
            tiny_weights(self.source / "bad.safetensors", self.config["model"], motion=False)
            with self.assertRaisesRegex(ValueError, "saved VLM/action/Motion"):
                deploy.strict_component_check(self.source / "bad.safetensors", self.config, "/metadata/qwen")

    def test_cpu_prepare_exports_actual_new_motion_values_and_explicit_path_provenance(self):
        before = {name: deploy.sha256(self.source / name) for name in (*deploy.METADATA, "model.safetensors")}
        with patch.dict(sys.modules, self.fake_model_modules()), patch.object(deploy, "validate_config"):
            report = deploy.prepare(self.source, self.resources, self.output, self.repo, self.release)
        self.assertEqual(report["contract"], deploy.CONTRACT)
        self.assertFalse(report["previous_motion_parameters_used"])
        self.assertEqual(report["step"], 10000)
        self.assertFalse(report["GPU_used"] or report["policy_forward_performed"])
        exported = torch.load(self.output / "motion/joint_motion.ckpt", weights_only=False)
        old = torch.load(self.resources / "joint20k/motion/joint_motion.ckpt", weights_only=False)
        self.assertTrue(torch.equal(exported["state_dict"]["weight"], torch.full((2, 2), 9.)))
        self.assertTrue(torch.equal(old["state_dict"]["weight"], torch.full((2, 2), -7.)))
        config = deploy.read(self.output / "run_config.json")
        self.assertEqual(config["model"]["motion_checkpoint"], str(self.output / "motion/joint_motion.ckpt"))
        self.assertIsNone(config["model"]["pretrained_joint_checkpoint"])
        self.assertEqual(config["model"]["state_drop_prob"], 0.)
        contract = deploy.read(self.output / "dino_training_contract.json")
        self.assertEqual(contract["motion_image_policy"], "vlm_color_clean_motion")
        self.assertEqual(contract["encoder"]["torch_version"], "mobile")
        self.assertEqual(contract["history"]["shape"], [6, 5, 1024])
        self.assertEqual(before, {name: deploy.sha256(self.source / name) for name in before})
        self.assertTrue((self.output / "provenance/release_metadata/dino/REMOTE_DINO_RELOCATION.json").is_file())
        self.assertEqual(report["physical_task_success"], "not_measured")

    def test_cpu_gpu_visibility_guard_and_existing_output_preserve_files(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), self.assertRaisesRegex(ValueError, "CPU preparation"):
            deploy.prepare(self.source, self.resources, self.output, self.repo, self.release)
        self.assertFalse(self.output.exists())
        self.output.mkdir()
        (self.output / "keep").write_text("keep")
        with self.assertRaisesRegex(ValueError, "fresh independent"):
            deploy.prepare(self.source, self.resources, self.output, self.repo, self.release)
        self.assertEqual((self.output / "keep").read_text(), "keep")


if __name__ == "__main__":
    unittest.main()
