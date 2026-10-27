"""Isolated CPU checks: readiness, exact inherited config, real save/selection paths."""
import ast
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch, Mock

SCRIPTS = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mobile100_launcher_test", SCRIPTS / "a100_mobile_pp100.py")
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
spec = importlib.util.spec_from_file_location("mobile100_legacy_test", SCRIPTS / "a100_handover_merged114.py")
legacy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(legacy)


def base_config():
    # Small independent fixture; the production baseline is separately pinned by
    # SHA to original bytes. Extra fields prove whole-tree comparison, not only
    # a whitelist of LR/batch settings.
    return {"exp": "old_handover", "timestamp": "old", "seed": 20261025,
        "train": {"output_dir": "/old/training", "resume_from_checkpoint": None,
            "skip_resumed_steps": False, "max_training_steps": 10000, "warmup_steps": 1000,
            "train_batch_size": 8, "gradient_accumulation_steps": 1, "validation_steps": 1000,
            "max_checkpoints_to_keep": 100, "checkpointing_steps": 1000,
            "preserve_checkpoint_every": 1000, "early_checkpoint_steps": [],
            "learning_rate": 1e-5, "mixed_precision": "bf16", "hf_token": ".hf_token",
            "num_workers": 4, "optimizer_foreach": False, "max_grad_norm": 1.0,
            "lr_scheduler_kwargs": {"betas": [.95, .999], "weight_decay": 1e-6, "eps": 1e-8}},
        "data": {"root_dir": "/old/data", "dino_cache_root": "/old/data/dino",
            "train_repo_ids": ["train"], "val_repo_ids": ["val"],
            "transform": {"field": {"stat_path": "/old/stats", "action_min": [-1., 0.],
                                     "state_max": [1., 0.]},
                          "model": {"motion_image_policy": "vlm_color_clean_motion"}}},
        "model": {"pretrained_joint_checkpoint": "/source/model.safetensors", "state_drop_prob": 0.,
                  "num_blocks": 6, "action_architecture": "dual_source_v1"},
        "wandb": {"entity": None, "group": None, "id": None, "name": None}}


def plan_fixture():
    return {"contract": launch.CONTRACT, "task_id": launch.TASK_ID, "fresh_run": True,
        "initial_global_step": 0, "optimizer_scheduler_rng_restored": False,
        "gpus": list(range(8)), "gpu_count": 8, "effective_global_batch": 64,
        "max_optimizer_steps": 10000, "warmup_steps": 1000, "per_gpu_batch": 8, "gradient_accumulation_steps": 1,
        "peak_learning_rates": dict(launch.PEAK_LRS), "source_step": 38000,
        "source_checkpoint_sha256": launch.SOURCE_SHA, "source_checkpoint": "/source/model.safetensors",
        "train_episodes": 100, "total_frames": 61806, "train_windows": 55906,
        "validation_role": launch.VALIDATION_ROLE, "release": "/fixture/release", "timestamp": "new"}


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.reference = base_config()
        self.plan = plan_fixture()
        self.root = Path("/new/prepared")
        self.config = launch.expected_config(self.reference, self.plan, self.root)

    def test_authorized_only_dataset_output_and_save_changes(self):
        launch.validate_config(self.config, self.plan, self.reference, self.root)
        self.assertEqual(self.config["model"], self.reference["model"])
        self.assertEqual(self.config["train"]["validation_steps"], 1000)
        self.assertEqual(self.config["seed"], 20261025)
        self.assertEqual(self.config["train"]["lr_scheduler_kwargs"], self.reference["train"]["lr_scheduler_kwargs"])

    def test_lr_model_bounds_and_unrecognized_parameter_drift_rejected(self):
        mutations = [("train", "learning_rate", 2e-5), ("train", "mixed_precision", "fp16"),
                     ("train", "num_workers", 1), ("train", "new_training_option", True),
                     ("model", "state_drop_prob", .5), ("model", "num_blocks", 7)]
        for group, key, value in mutations:
            with self.subTest(key=key):
                config = copy.deepcopy(self.config)
                config[group][key] = value
                with self.assertRaisesRegex(ValueError, "drift"):
                    launch.validate_config(config, self.plan, self.reference, self.root)
        self.config["data"]["transform"]["field"]["action_min"][0] = -2.
        with self.assertRaises(ValueError):
            launch.validate_config(self.config, self.plan, self.reference, self.root)

    def test_resume_and_wrong_task_or_optimizer_restore_rejected(self):
        for key, value in [("resume_from_checkpoint", "latest"), ("skip_resumed_steps", True)]:
            config = copy.deepcopy(self.config)
            config["train"][key] = value
            with self.assertRaises(ValueError):
                launch.validate_config(config, self.plan, self.reference, self.root)
        for key, value in [("task_id", "simple/Handover"), ("fresh_run", False),
                           ("initial_global_step", 38000), ("optimizer_scheduler_rng_restored", True),
                           ("validation_role", "mobile_validation")]:
            with self.subTest(key=key):
                bad = dict(self.plan, **{key: value})
                with self.assertRaises(ValueError):
                    launch.validate_config(self.config, bad, self.reference, self.root)

    def test_real_production_checkpoint_function_saves_exact_thirteen_once(self):
        path = SCRIPTS.parents[1] / "VLMExpert/src/psi/trainers/trainer.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == "should_save_training_checkpoint")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
        train = self.config["train"]
        saved = [step for step in range(1, 10001) if namespace[node.name](step,
            train["checkpointing_steps"], train["max_training_steps"], tuple(train["early_checkpoint_steps"]))]
        self.assertEqual(saved, list(range(4000, 10001, 500)))
        self.assertEqual(len(saved), 13)
        self.assertFalse(any(step < 4000 for step in saved))


class ReadinessTests(unittest.TestCase):
    def make_release(self, root):
        episodes = [{"episode_index": i, "length": 619 if i < 6 else 618,
            "eligible_current_rows": list(range(30, (619 if i < 6 else 618) - 29)),
            "task_id": launch.TASK_ID, "split": "train", "split_restriction": "train_only"} for i in range(100)]
        files = {"packed/PACK_READY.json": {"contract": launch.PACK_CONTRACT, "task_id": launch.TASK_ID,
            "selection_policy": "native_only", "train_episodes": 100, "total_frames": 61806, "train_windows": 55906},
            "packed/train/meta/info.json": {}, "packed/source_dataset_statistics.json": {}, "dino/manifest.json": {}}
        for name, value in files.items():
            launch.write(root / name, value)
        p = root / "packed/train/meta/episodes.jsonl"
        p.write_text("".join(json.dumps(e) + "\n" for e in episodes))
        identities = {}
        for name in [*files, "packed/train/meta/episodes.jsonl"]:
            p = root / name
            identities[name] = {"sha256": launch.sha(p), "size_bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
        ready = {"contract": launch.READY_CONTRACT, "task_id": launch.TASK_ID, "selection_policy": "native_only",
            "status": "passed", "remote_production_ready": True, "production_ready_for_bound_pack_and_cache": True,
            "DINO_relocation_pending": False, "training_started": False, "CUDA_initialized": False,
            "GPU_used": False, "normalization_refitted": False, "split_restriction": "train_only",
            "train_episodes": 100, "total_frames": 61806, "train_windows": 55906,
            "release_root": str(root), "pack_root": str(root / "packed"), "dino_root": str(root / "dino"),
            "payload_files": identities, "pack_ready_sha256": launch.sha(root / "packed/PACK_READY.json"),
            "cache_manifest_sha256": launch.sha(root / "dino/manifest.json")}
        launch.write(root / launch.READY_NAME, ready)
        return ready

    def test_actual_mobile_ready_binds_all_payload_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_release(root)
            digest = launch.sha(root / launch.READY_NAME)
            with patch.object(launch, "RELEASE", root):
                self.assertEqual(launch.validate_ready(root, digest)["train_windows"], 55906)
                (root / "packed/source_dataset_statistics.json").write_text('{"drift":true}')
                with self.assertRaisesRegex(ValueError, "changed"):
                    launch.validate_ready(root, digest)

    def test_missing_or_local_or_wrong_task_ready_cannot_prepare_or_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ready = self.make_release(root)
            for key, value in [("task_id", "simple/G1WholebodyHandoverTeleop-v0"),
                    ("remote_production_ready", False), ("DINO_relocation_pending", True),
                    ("status", "pending"), ("train_episodes", 99), ("train_windows", 55905)]:
                with self.subTest(key=key), patch.object(launch, "RELEASE", root):
                    launch.write(root / launch.READY_NAME, dict(ready, **{key: value}))
                    with self.assertRaisesRegex(ValueError, "incomplete"):
                        launch.validate_ready(root, launch.sha(root / launch.READY_NAME))
            (root / launch.READY_NAME).unlink()
            args = SimpleNamespace(release=root, expected_ready_sha256="0" * 64)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}), patch.object(launch, "RELEASE", root), \
                 patch.object(launch, "external") as external:
                with self.assertRaises(FileNotFoundError):
                    launch.prepare(args)
                external.assert_not_called()

    def test_expected_sha_cannot_be_guessed_or_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_release(root)
            with patch.object(launch, "RELEASE", root):
                for digest in [None, "", "0" * 64]:
                    with self.subTest(digest=digest), self.assertRaises(ValueError):
                        launch.validate_ready(root, digest)

    def test_failed_ready_blocks_train_before_importing_external_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            release = root / "release"
            release.mkdir()
            self.make_release(release)
            launch.write(root / "preparation.json", {"contract": launch.CONTRACT,
                "runner_sha256": launch.sha(launch.__file__), "release": str(release), "expected_ready_sha256": "0" * 64})
            with patch.object(launch, "RELEASE", release), patch.object(launch, "external") as external:
                with self.assertRaisesRegex(ValueError, "READY SHA"):
                    launch.train(root)
                external.assert_not_called()


class RuntimeAndSelectionTests(unittest.TestCase):
    def test_worker_requires_this_launch_byte_receipt_and_rejects_completed_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = dict(plan_fixture(), config_sha256="config", expected_ready_sha256="ready", runner_sha256="runner")
            receipt = {"contract": launch.CONTRACT, "task_id": launch.TASK_ID, "status": "passed",
                "config_sha256": "config", "expected_ready_sha256": "ready", "runner_sha256": "runner",
                "full_payload_sha256_verified": True, "source_checkpoint_sha256": launch.SOURCE_SHA, "launcher_pid": 123}
            with self.assertRaisesRegex(ValueError, "through launch.sh"):
                launch.require_launch_acceptance(root, plan)
            launch.write(root / "launch_acceptance.json", receipt)
            launch.write(root / "training_state.json", {"state": "running", "launcher_pid": 123, "expected_ready_sha256": "ready"})
            digest = launch.sha(root / "launch_acceptance.json")
            with patch.dict(os.environ, {"CVPR_MOBILE100_LAUNCH_RECEIPT_SHA256": digest}):
                launch.require_launch_acceptance(root, plan)
                launch.write(root / "training_state.json", {"state": "complete", "launcher_pid": 123, "expected_ready_sha256": "ready"})
                with self.assertRaisesRegex(ValueError, "already finished"):
                    launch.require_launch_acceptance(root, plan)
            with patch.dict(os.environ, {"CVPR_MOBILE100_LAUNCH_RECEIPT_SHA256": "0" * 64}):
                with self.assertRaisesRegex(ValueError, "through launch.sh"):
                    launch.require_launch_acceptance(root, plan)

    def test_public_train_entry_cannot_bypass_affinity_priority_guard(self):
        plan = dict(plan_fixture(), training_cpus=list(range(128)), training_nice=0)
        with patch.object(launch, "load", return_value=(None, plan, None, None)), \
             patch.object(launch, "require_cpu_check"), patch.dict(os.environ, {"WORLD_SIZE": "8"}), \
             patch.object(os, "sched_getaffinity", return_value={127}), \
             patch.object(os, "getpriority", return_value=19), patch.object(launch, "trainer_class") as factory:
            with self.assertRaisesRegex(ValueError, "separate ordinary SSH"):
                launch.train(Path("/unused-fixture"))
            factory.assert_not_called()

    def test_restricted_cpu_or_nice19_inspection_context_cannot_launch(self):
        plan = {"training_cpus": list(range(128)), "training_nice": 0}
        for cpus, nice in [({127}, 19), (set(range(128)), 19), ({127}, 0)]:
            with patch.object(os, "sched_getaffinity", return_value=cpus), \
                 patch.object(os, "getpriority", return_value=nice), self.assertRaises(ValueError):
                launch.require_launch_runtime(plan)
        with patch.object(os, "sched_getaffinity", return_value=set(range(128))), \
             patch.object(os, "getpriority", return_value=0):
            self.assertEqual(launch.require_launch_runtime(plan)["cpus"], list(range(128)))
        self.assertEqual(launch.cpu_list("0-127"), list(range(128)))
        for value in ["127", "0-31,31", "31-0", "0-31,NaN"]:
            with self.assertRaises(ValueError):
                launch.cpu_list(value)

    def test_historical_val_requires_exact_same_rows_not_only_count(self):
        candidates = {i: list(range(i * 10, i * 10 + 8)) for i in range(5)}
        expected = [v for values in candidates.values() for v in values]
        self.assertEqual(launch.select_validation_rows(candidates, expected), expected)
        candidates[0][0] = 99
        with self.assertRaisesRegex(ValueError, "previous40"):
            launch.select_validation_rows(candidates, expected)

    def test_actual_mobile_mapper_injection_never_uses_upstream_handover_dataset(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ReadinessTests().make_release(root)
            episodes = [json.loads(line) for line in (root / "packed/train/meta/episodes.jsonl").read_text().splitlines()]
            ranges, originals, candidates, offset, filtered = {}, [], {}, 0, 0
            for episode in episodes:
                i, length = episode["episode_index"], episode["length"]
                ranges[i] = (offset, offset + length)
                originals.extend(offset + row for row in episode["eligible_current_rows"])
                count = len(episode["eligible_current_rows"])
                candidates[i] = list(range(filtered, filtered + count))
                filtered += count
                offset += length
            history = SimpleNamespace(pack_info=[{}], source_episode_ranges=[ranges], original_indices=originals)
            train = SimpleNamespace(raw_dataset=SimpleNamespace(motion_history=history))
            val = object()
            val_candidates = {i: list(range(i * 8, i * 8 + 8)) for i in range(5)}
            def eligible(data, expected_episodes, *, source_frame_rule):
                self.assertEqual(source_frame_rule, "ceil")
                self.assertEqual(expected_episodes, 100 if data is train else 5)
                return (candidates if data is train else val_candidates), None, None
            config = SimpleNamespace(data=lambda split, **kwargs: train if split == "train" else val)
            plan = dict(plan_fixture(), release=str(root), expected_validation_rows=list(range(40)))
            actual_train, actual_val, counts = launch.datasets(config, None, plan, legacy,
                                                               SimpleNamespace(eligible_dataset_rows=eligible))
            self.assertIsInstance(actual_train, torch.utils.data.Subset)
            self.assertEqual(len(actual_train), 55906)
            self.assertEqual(len(actual_val), 40)
            self.assertEqual(counts["train_task_id"], launch.TASK_ID)
            self.assertEqual(counts["validation_role"], launch.VALIDATION_ROLE)

    def test_real_retention_wrapper_keeps13_models_and_last_two_full_states(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "checkpoints").mkdir()
            class Base:
                def __init__(self):
                    self.project_dir = root
                    self.accelerator = SimpleNamespace(is_main_process=True)
                def save_checkpoint(self, step):
                    path = root / "checkpoints" / f"ckpt_{step}"
                    path.mkdir()
                    for name in ["model.safetensors", "optimizer.bin", "scheduler.bin", "random_states_0.pkl"]:
                        (path / name).write_bytes(f"{step}-{name}".encode())
                    return step
            up = SimpleNamespace(trainer_class=lambda plan: Base)
            plan = dict(plan_fixture(), migration_helper="unused-test-helper")
            with patch.object(launch, "module", return_value=SimpleNamespace()), \
                 patch.object(torch.distributed, "broadcast_object_list", side_effect=lambda *a, **k: None):
                cls = launch.trainer_class(plan, legacy, up)
                instance = cls()
                with self.assertRaisesRegex(ValueError, "before4000"):
                    instance.save_checkpoint(3999)
                self.assertEqual(list((root / "checkpoints").iterdir()), [])
                for step in launch.SAVE_STEPS:
                    self.assertEqual(instance.save_checkpoint(step), step)
            record = launch.read(root / "checkpoint_retention.json")
            self.assertEqual(record["all_saved_model_steps"], list(launch.SAVE_STEPS))
            self.assertEqual(record["full_recovery_steps"], [9500, 10000])
            for step in launch.SAVE_STEPS:
                path = root / "checkpoints" / f"ckpt_{step}"
                self.assertEqual((path / "model.safetensors").read_bytes(), f"{step}-model.safetensors".encode())
                self.assertEqual((path / "optimizer.bin").exists(), step >= 9500)
            self.assertTrue(callable(up.datasets))


class PreparationRoundTripTests(unittest.TestCase):
    """Run actual prepare/load on tiny input files and an isolated parser stub.

    The stub replaces only the heavyweight external frozen runtime. Actual SHA,
    reference/config, package, symlink and initialization guards still execute.
    """
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.release = self.root / "release"
        self.release.mkdir()
        ReadinessTests().make_release(self.release)
        self.reference = self.root / "reference"
        self.reference.mkdir()
        source = self.root / "source/model.safetensors"
        source.parent.mkdir()
        source.write_bytes(b"tiny fixture weight, never loaded")
        self.source_sha = launch.sha(source)
        self.baseline = base_config()
        self.baseline["model"]["pretrained_joint_checkpoint"] = str(source)
        self.baseline["train"]["output_dir"] = str(self.root / "old_run/training")
        val, val_dino, code = self.root / "old_val", self.root / "old_val_dino", self.root / "frozen_code"
        for path in [val / "meta/info.json", val / "meta/episodes.jsonl", val_dino / "manifest.json", code / "toy.py"]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}")
        self.prior = dict(plan_fixture(), source_checkpoint=str(source), source_checkpoint_sha256=self.source_sha,
            parent_prepared=str(self.root / "old_prepared"), frozen_code=str(code), val=str(val), val_dino=str(val_dino),
            code_sha256={str(code / "toy.py"): launch.sha(code / "toy.py")},
            bindings_sha256={str(path): launch.sha(path) for path in
                [val / "meta/info.json", val / "meta/episodes.jsonl", val_dino / "manifest.json"]})
        self.prior["bindings_sha256"]["/old/packed/source_dataset_statistics.json"] = launch.sha(
            self.release / "packed/source_dataset_statistics.json")
        # Actual pinned previous-run metadata, not a synthetic current config,
        # is the authority; here both are intentionally tiny fixture bytes.
        actual = copy.deepcopy(self.baseline)
        actual["wandb"]["id"] = "runtime-run-id"
        actual["train"]["hf_token"] = "PosixPath('.hf_token')"
        for name, value in [("run_config.json", self.baseline), ("actual_run_config.json", actual),
                            ("preparation.json", self.prior),
                            ("task_window_selection.json", {"validation_rows": list(range(40))})]:
            launch.write(self.reference / name, value)
        (self.reference / "argv.txt").write_text("fixture\n")
        (self.reference / "handover_val_episodes.jsonl").write_bytes((val / "meta/episodes.jsonl").read_bytes())
        helpers = {}
        for index, name in enumerate(["a100_handover_merged114.py", "quality100_bs8_sync_retry.py", "state_dropout_migration.py"]):
            (self.reference / name).write_text(f"# isolated external fixture {index}\n")
            helpers[name] = launch.sha(self.reference / name)
        summary = {"contract": launch.REFERENCE_CONTRACT, "reference_files": [
            {"local_path": str(p), "remote_path": "/fixture/" + p.name,
             "sha256": launch.sha(p), "size_bytes": p.stat().st_size} for p in self.reference.iterdir()]}
        launch.write(self.reference / "REFERENCE_SUMMARY.json", summary)
        self.output = self.root / "mobile_prepared"
        self.args = SimpleNamespace(release=self.release, expected_ready_sha256=launch.sha(self.release / launch.READY_NAME),
            reference=self.reference, output=self.output, training_cpus="0-127")
        self.addCleanup(patch.stopall)
        patch.object(launch, "RELEASE", self.release).start()
        patch.object(launch, "SOURCE_SHA", self.source_sha).start()
        patch.object(launch, "SUMMARY_SHA", launch.sha(self.reference / "REFERENCE_SUMMARY.json")).start()
        patch.object(launch, "LEGACY_SHA", helpers["a100_handover_merged114.py"]).start()
        patch.object(launch, "UPSTREAM_SHA", helpers["quality100_bs8_sync_retry.py"]).start()
        patch.object(launch, "MIGRATION_SHA", helpers["state_dropout_migration.py"]).start()
        patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": ""}).start()
        # Minimal independent parser, with the same call boundary as the frozen
        # shared implementation. Nothing imports/models/weights in this fixture.
        def rewrite(original, overrides):
            self.assertEqual(original, ["fixture"])
            config = copy.deepcopy(self.baseline)
            for name, value in overrides.items():
                cursor = config
                parts = name.split(".")
                for part in parts[:-1]:
                    cursor = cursor[part]
                cursor[parts[-1]] = value
            return [json.dumps(config, separators=(",", ":"))]
        shared = SimpleNamespace(rewrite_argv=rewrite,
            parse_training_argv=lambda path: SimpleNamespace(model_dump=lambda **kwargs: json.loads(path.read_text())))
        guarded_legacy = SimpleNamespace(sealed_code=lambda code: self.prior["code_sha256"])
        self.up = SimpleNamespace(shared=shared, selected_gpu_idle=Mock())
        patch.object(launch, "external", return_value=(guarded_legacy, self.up)).start()

    def prepare(self):
        with patch.object(sys, "stdout", io.StringIO()):
            launch.prepare(self.args)

    def test_real_prepare_load_preserves_independent_links_and_full_reference(self):
        self.prepare()
        config, plan, _, _ = launch.load(self.output)
        self.assertEqual(config.model_dump()["train"]["validation_steps"], 1000)
        self.assertEqual(plan["source_step"], 38000)
        self.assertFalse(plan["optimizer_scheduler_rng_restored"])
        self.assertEqual(plan["expected_validation_rows"], list(range(40)))
        self.assertEqual((self.output / "data/train").resolve(), self.release / "packed/train")
        self.assertEqual((self.output / "data/val").resolve(), Path(self.prior["val"]))
        self.assertEqual((self.output / "reference/actual_run_config.json").read_bytes(),
                         (self.reference / "actual_run_config.json").read_bytes())
        self.assertFalse((self.output / "training_state.json").exists())
        self.assertIn("run --prepared", (self.output / "launch.sh").read_text())

    def test_load_rejects_changed_ready_and_actual_reference_file(self):
        self.prepare()
        p = self.output / "reference/actual_run_config.json"
        original = p.read_bytes()
        p.write_bytes(original + b" ")
        with self.assertRaisesRegex(ValueError, "reference file identity"):
            launch.load(self.output)
        p.write_bytes(original)
        p = self.release / launch.READY_NAME
        p.write_bytes(p.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "READY SHA"):
            launch.load(self.output)

    def test_even_resealed_argv_and_config_cannot_change_lr_or_resume(self):
        self.prepare()
        config = launch.read(self.output / "run_config.json")
        config["train"]["resume_from_checkpoint"] = "latest"
        config["train"]["learning_rate"] = .123
        launch.write(self.output / "run_config.json", config)
        (self.output / "argv.txt").write_text(json.dumps(config, separators=(",", ":")))
        plan = launch.read(self.output / "preparation.json")
        plan.update(config_sha256=launch.sha(self.output / "run_config.json"), argv_sha256=launch.sha(self.output / "argv.txt"))
        launch.write(self.output / "preparation.json", plan)
        with self.assertRaisesRegex(ValueError, "Configuration drift"):
            launch.load(self.output)

    def test_launch_rehash_catches_same_size_mtime_payload_or_weight_drift(self):
        self.prepare()
        for path, expected_message in [(self.release / "packed/train/meta/info.json", "payload changed"),
                                      (Path(self.prior["source_checkpoint"]), "source model bytes")]:
            with self.subTest(path=path.name):
                before = path.stat()
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
                # Fast worker identity path passes; the one launch-wide SHA
                # scan must still reject before any GPU query or process start.
                launch.load(self.output)
                with patch.object(launch, "require_cpu_check"), \
                     patch.object(os, "sched_getaffinity", return_value=set(range(128))), \
                     patch.object(os, "getpriority", return_value=0), \
                     patch.object(launch.subprocess, "Popen") as popen:
                    with self.assertRaisesRegex(ValueError, expected_message):
                        launch.run(self.output)
                    popen.assert_not_called()
                    self.up.selected_gpu_idle.assert_not_called()
                path.write_bytes(original)
                os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))

    def test_bad_prepared_to_actual_reference_diff_is_not_runtime_metadata(self):
        actual = launch.read(self.reference / "actual_run_config.json")
        actual["train"]["learning_rate"] = .5
        launch.write(self.reference / "actual_run_config.json", actual)
        # Rebind fixture manifest to test semantic reference comparison itself.
        summary = launch.read(self.reference / "REFERENCE_SUMMARY.json")
        for item in summary["reference_files"]:
            p = Path(item["local_path"])
            item.update(sha256=launch.sha(p), size_bytes=p.stat().st_size)
        launch.write(self.reference / "REFERENCE_SUMMARY.json", summary)
        with patch.object(launch, "SUMMARY_SHA", launch.sha(self.reference / "REFERENCE_SUMMARY.json")):
            with self.assertRaisesRegex(ValueError, "Actual Handover configuration"):
                self.prepare()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
