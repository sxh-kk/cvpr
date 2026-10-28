"""CPU preparation of the completed Mobile P&P100 task10000 model.

Independent Mobile contract: no task50/149 admission, no old Motion parameter
fallback, no policy forward, simulation, training, or GPU initialization. Older
joint20k resources supply construction metadata only. All three trained modules
come from the explicitly accepted new complete joint checkpoint.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import sys

from export_bundle import canonical_visual_identity, export_motion, sha256, write_json
from prepare_task_snapshot import remap_config, validate_config

CONTRACT = "cvpr-mobile-pp100-local-deployment-v1"
BACKUP_CONTRACT = "cvpr-mobile-pp100-local-checkpoint-backup-v1"
PREPARATION_CONTRACT = "cvpr-mobile-pp100-a100-fresh-v1"
TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
PACK_CONTRACT = "cvpr-pico-mobile-pp-lerobot-pack-v1"
PRODUCTION_CONTRACT = "cvpr-pico-mobile-pp-production-verification-v1"
EXPECTED_STEP = 10000
EXPECTED_MODEL_SHA = "9b1ea759ef3aedcc7697d62f6ff36deb12dcc2e379cce61c6ca904e80826b390"
EXPECTED_MODEL_BYTES = 12066487572
EXPECTED_CONFIG_SHA = "854042b54d4b31d8b30c495539c4b220cf9dfa31b5570d8e052111a9137683b8"
FINAL_ACCEPTANCE_SHA = "d1fea82b176d5ef4eab4df28cdec2ab2b150fb7eabf4cbfe76112616f8702e50"
RESOURCES_MANIFEST_SHA = "bd78327a481d71643a1ebe295f10224f6fdeee76f5480e76c58fce66031e246b"
QWEN_METADATA_NAMES = ("added_tokens.json", "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
    "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
    "video_preprocessor_config.json", "vocab.json")
EXPECTED_RUN = ("/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp100_training_20261027/"
    "run_001/training/posttrain/mobile_pp100_joint38000_statekeep_8gpu.us.flow1000.cosine.lr1.0e-05.b64.gpus8.261027164426")
EXPECTED_TRACKER = {"entity": "", "group": "posttrain", "id": "nn7ebojc", "name": Path(EXPECTED_RUN).name}
SOURCE_SHA = "12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7"
STAT_SHA = "931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4"
VALIDATION_ROLE = "historical_handover_regression_only_not_mobile_validation"
SAVE_STEPS = list(range(4000, 10001, 500))
ROOT_METADATA = ("run_config.json", "argv.txt", "dataset_statistics.json", "task_initialization.json",
    "task_window_selection.json", "prepared_run_config.json", "prepared_argv.txt", "preparation.json",
    "training_state.json", "checkpoint_retention.json", "gpu_check_update10000.json", "FINAL_TRAINING_ACCEPTANCE_001.json")
RELEASE_METADATA = ("packed/PACK_READY.json", "dino/manifest.json", "A100_PICO_DATASET_READY.json",
    "dino/REMOTE_DINO_RELOCATION.json")
METADATA = ROOT_METADATA + tuple("release_metadata/" + name for name in RELEASE_METADATA)
REPO = Path(__file__).resolve().parents[2]
CRITICAL_PREFIXES = ("MotionExpert/", "ImageProcess/", "VLMExpert/src/psi/models/",
    "VLMExpert/src/psi/deploy/", "VLMExpert/src/psi/config/", "VLMExpert/src/psi/utils/")


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def regular(root, relative):
    name = Path(relative)
    require(not name.is_absolute() and name.parts and ".." not in name.parts, "Unsafe source metadata path")
    path = root
    for part in name.parts:
        path /= part
        require(not path.is_symlink(), "Source metadata may not be a symlink: " + relative)
    require(path.is_file() and path.resolve().is_relative_to(root), "Missing source metadata: " + relative)
    return path


def validate_runtime_metadata(source, config, preparation):
    prepared_path, argv_path = source / "prepared_run_config.json", source / "prepared_argv.txt"
    require(sha256(prepared_path) == preparation["config_sha256"] == EXPECTED_CONFIG_SHA
            and sha256(argv_path) == preparation["argv_sha256"], "Exact prepared Mobile100 config/argv identity differs")
    expected = read(prepared_path)
    require(expected["train"].get("hf_token") == ".hf_token"
            and all(expected["wandb"].get(k) is None for k in EXPECTED_TRACKER),
            "Require the original pre-tracker Mobile configuration")
    expected["train"]["hf_token"] = "PosixPath('.hf_token')"
    expected["wandb"].update(EXPECTED_TRACKER)
    require(config == expected, "Runtime configuration differs beyond the exact recorded Path/tracker serialization")
    tokens = shlex.split(argv_path.read_text())
    require(tokens and tokens == shlex.split((source / "argv.txt").read_text()),
            "Runtime argv semantic tokens differ from prepared argv")
    return {"status": "passed", "prepared_config_sha256": sha256(prepared_path),
        "prepared_argv_sha256": sha256(argv_path), "runtime_config_sha256": sha256(source / "run_config.json"),
        "runtime_argv_sha256": sha256(source / "argv.txt"), "all_other_config_values_exact": True,
        "argv_tokens_exact": True, "argv_token_count": len(tokens),
        "allowed_runtime_fields": ["train.hf_token", *("wandb." + k for k in EXPECTED_TRACKER)]}


def validate_release(source, release, config, preparation):
    """Read already accepted metadata only; do not re-audit frames or features."""
    remote = source / "release_metadata"
    ready = read(remote / "A100_PICO_DATASET_READY.json")
    pack = read(remote / "packed/PACK_READY.json")
    manifest = read(remote / "dino/manifest.json")
    relocation = read(remote / "dino/REMOTE_DINO_RELOCATION.json")
    require(sha256(remote / "A100_PICO_DATASET_READY.json") == preparation["expected_ready_sha256"]
            and ready.get("contract") == PRODUCTION_CONTRACT and ready.get("task_id") == TASK_ID
            and ready.get("status") == "passed" and ready.get("remote_production_ready") is True
            and ready.get("DINO_relocation_pending") is False
            and ready.get("split_restriction") == "train_only" and ready.get("selection_policy") == "native_only"
            and ready.get("normalization_refitted") is False
            and (ready.get("train_episodes"), ready.get("total_frames"), ready.get("train_windows")) == (100, 61806, 55906),
            "Actual Mobile100 A100 production readiness differs from the trained input")
    require(pack.get("contract") == PACK_CONTRACT and pack.get("task_id") == TASK_ID
            and pack.get("selection_policy") == "native_only" and pack.get("normalization_refitted") is False
            and (pack.get("train_episodes"), pack.get("total_frames"), pack.get("train_windows")) == (100, 61806, 55906)
            and sha256(remote / "packed/PACK_READY.json") == ready["pack_ready_sha256"]
                == sha256(regular(release, "packed/PACK_READY.json")), "Local and actual trained Mobile pack identity differs")
    require(sha256(remote / "dino/manifest.json") == ready["cache_manifest_sha256"]
            and manifest.get("contract") == "unifolm-dino-native-frames-v1" and manifest.get("state") == "complete"
            and manifest.get("episode_indices") == list(range(100))
            and (manifest.get("source", {}).get("total_episodes"), manifest.get("source", {}).get("total_frames"),
                 manifest.get("source", {}).get("fps")) == (100, 61806, 30), "Actual trained clean DINO source identity differs")
    require(relocation.get("contract") == "cvpr-pico-mobile-pp-dino-host-relocation-v1"
            and relocation.get("task_id") == TASK_ID and relocation.get("status") == "passed"
            and relocation.get("data_and_features_byte_identical") is True and relocation.get("PACK_READY_modified") is False
            and sha256(remote / "dino/REMOTE_DINO_RELOCATION.json") == ready["remote_relocation_binding"]["sha256"]
            and relocation.get("pack_ready_sha256") == ready["pack_ready_sha256"]
            and relocation.get("manifest_sha256") == sha256(remote / "dino/manifest.json")
            and relocation.get("original_manifest_sha256") == sha256(regular(release, "dino/manifest.json")),
            "Explicit original-local to trained-A100 DINO relocation proof differs")
    local_manifest = read(release / "dino/manifest.json")
    require(local_manifest["encoder"] == manifest["encoder"], "DINO relocation changed encoder identity")
    checkpoint = pack["checkpoint_contract"]
    field = config["data"]["transform"]["field"]
    require(checkpoint.get("source_step") == 38000 and checkpoint.get("source_checkpoint_sha256") == SOURCE_SHA
            and checkpoint.get("bounds_refitted") is False
            and sha256(source / "dataset_statistics.json") == STAT_SHA
                == checkpoint["statistics"]["sha256"] == preparation["bindings_sha256"][field["stat_path"]]
                == sha256(regular(release, "packed/source_dataset_statistics.json")), "Inherited joint38000 statistics changed")
    require(set(checkpoint["bounds_sha256"]) == {"action_min", "action_max", "state_min", "state_max"},
            "All four action/state bound identities are required")
    for name, expected in checkpoint["bounds_sha256"].items():
        require(digest(field[name]) == expected, "Inherited normalization bound changed: " + name)
    return {"status": "passed", "metadata_only": True, "frame_or_feature_audit_repeated": False,
        "trained_ready_sha256": sha256(remote / "A100_PICO_DATASET_READY.json"),
        "trained_manifest_sha256": sha256(remote / "dino/manifest.json"),
        "original_local_manifest_sha256": sha256(release / "dino/manifest.json"),
        "encoder_unchanged_by_host_relocation": True}


def validate_source(source, release, *, verify_weight=True):
    source, release = (Path(p).resolve(strict=True) for p in (source, release))
    # Publication of the model-only transfer receipt is mandatory before opening
    # model bytes. A streaming or unfinished download cannot be consumed.
    receipt = read(regular(source, "backup_receipt.json"))
    require(receipt.get("contract") == BACKUP_CONTRACT and receipt.get("task_id") == TASK_ID
            and type(receipt.get("step")) is int and receipt["step"] == EXPECTED_STEP
            and receipt.get("optimizer_copied") is False, "Require the exact completed Mobile task10000 model-only backup")
    require(set(METADATA) <= set(receipt["metadata"]), "Missing Mobile source metadata SHA bindings")
    for name in METADATA:
        require(sha256(regular(source, name)) == receipt["metadata"][name], "Saved Mobile metadata changed: " + name)
    final = read(source / "FINAL_TRAINING_ACCEPTANCE_001.json")
    require(sha256(source / "FINAL_TRAINING_ACCEPTANCE_001.json") == FINAL_ACCEPTANCE_SHA
            and final.get("contract") == "cvpr-mobile-pp100-final-training-artifact-acceptance-v1"
            and final.get("status") == "passed" and final.get("project") == EXPECTED_RUN
            and final["configuration"]["sha256"] == sha256(source / "prepared_run_config.json") == EXPECTED_CONFIG_SHA,
            "Independent final Mobile100 training acceptance/configuration identity differs")
    models = [m for m in final["models"] if m["step"] == EXPECTED_STEP]
    expected_model = str(Path(EXPECTED_RUN) / "checkpoints/ckpt_10000/model.safetensors")
    require(len(models) == 1 and models[0]["path"] == expected_model
            and models[0]["sha256"] == EXPECTED_MODEL_SHA and models[0]["size_bytes"] == EXPECTED_MODEL_BYTES,
            "Accepted final task10000 model authority differs")
    identity, model_record = receipt["run_identity"], receipt["model"]
    require(identity.get("run_directory") == EXPECTED_RUN
            and identity.get("run_config_sha256") == receipt["metadata"]["run_config.json"]
            and identity.get("preparation_sha256") == receipt["metadata"]["preparation.json"]
            and model_record.get("source_path") == expected_model and model_record.get("bytes") == EXPECTED_MODEL_BYTES
            and all(model_record.get(key) == EXPECTED_MODEL_SHA for key in
                    ("source_sha256_before", "source_sha256_after", "local_sha256")),
            "Mobile checkpoint source/run/before-after-local SHA identity differs")
    weights = regular(source, "model.safetensors")
    require(weights.stat().st_size == EXPECTED_MODEL_BYTES
            and (not verify_weight or sha256(weights) == EXPECTED_MODEL_SHA), "Actual local task10000 model bytes differ")
    for name, key in (("training_state.json", "state_file"), ("checkpoint_retention.json", "retention_file"),
                      ("gpu_check_update10000.json", "health_file")):
        require(sha256(source / name) == final[key]["sha256"], "Final acceptance metadata binding differs: " + name)
    state, retention = read(source / "training_state.json"), read(source / "checkpoint_retention.json")
    require(state.get("state") == "complete" and state.get("returncode") == 0
            and state.get("run_directory") == EXPECTED_RUN and state.get("task_id") == TASK_ID
            and state.get("fresh_run") is True and state.get("initial_global_step") == 0
            and state.get("optimizer_scheduler_rng_restored") is False, "Require completed fresh Mobile100 training")
    require(receipt.get("completed_retention_marker") == retention
            and retention.get("saved_update") == EXPECTED_STEP and retention.get("all_models_preserved") is True
            and retention.get("all_saved_model_steps") == SAVE_STEPS
            and retention.get("full_recovery_steps") == [9500, 10000], "Task10000 completed-save marker differs")
    config, preparation = read(source / "run_config.json"), read(source / "preparation.json")
    require(preparation.get("contract") == PREPARATION_CONTRACT and preparation.get("task_id") == TASK_ID
            and preparation.get("source_step") == 38000 and preparation.get("source_checkpoint_sha256") == SOURCE_SHA
            and preparation.get("fresh_run") is True and preparation.get("initial_global_step") == 0
            and preparation.get("optimizer_scheduler_rng_restored") is False
            and (preparation.get("gpu_count"), preparation.get("per_gpu_batch"), preparation.get("gradient_accumulation_steps"),
                 preparation.get("effective_global_batch")) == (8, 8, 1, 64)
            and (preparation.get("train_episodes"), preparation.get("total_frames"), preparation.get("train_windows"),
                 preparation.get("max_optimizer_steps")) == (100, 61806, 55906, 10000)
            and preparation.get("validation_role") == VALIDATION_ROLE and preparation.get("Mobile_validation_selected") is False,
            "Require the true all100 train-only Mobile fresh38k task0 provenance")
    validate_runtime_metadata(source, config, preparation)
    initialization = read(source / "task_initialization.json")
    require(initialization.get("source_checkpoint") == preparation["source_checkpoint"]
            and initialization.get("source_step") == 38000 and initialization.get("strict_joint_loader") is True
            and initialization.get("fresh_optimizer_scheduler_rng") is True and initialization.get("global_batch") == 64
            and initialization.get("backend", {}).get("backend") == "fused"
            and initialization.get("backend", {}).get("foreach") is False
            and initialization.get("backend", {}).get("initialized_parameter_states") == 0,
            "Strict new optimizer/full joint38000 initialization proof differs")
    windows = read(source / "task_window_selection.json")
    require(windows.get("train_windows") == 55906 and windows.get("train_episode_indices") == list(range(100))
            and windows.get("train_task_id") == TASK_ID and windows.get("validation_role") == VALIDATION_ROLE
            and windows.get("Mobile_validation_selected") is False
            and windows.get("validation_windows") == 40
            and windows.get("validation_rows") == preparation["expected_validation_rows"],
            "All100 Mobile TRAIN or historical Handover-only validation identity differs")
    require(state.get("expected_ready_sha256") == preparation["expected_ready_sha256"],
            "Completed training state and prepared input READY differ")
    model, train = config["model"], config["train"]
    require((model.get("action_architecture"), model.get("num_blocks"), model.get("action_dim"), model.get("odim"),
             model.get("action_chunk_size"), model.get("state_drop_prob"), model.get("state_null_token"))
                == ("dual_source_v1", 6, 80, 45, 30, 0., True)
            and model.get("tune_motion") is True and model.get("tune_vlm") is True and model.get("rtc") is False
            and model.get("pretrained_joint_checkpoint") == preparation["source_checkpoint"],
            "Require the complete six-block state-retention Mobile dual-source model")
    require((train.get("train_batch_size"), train.get("gradient_accumulation_steps"), train.get("max_training_steps")) == (8, 1, 10000)
            and train.get("resume_from_checkpoint") is None and train.get("skip_resumed_steps") is False,
            "Require Mobile task10000 fresh optimization, not resumed task progress")
    repack, image = (config["data"]["transform"][k] for k in ("repack", "model"))
    require(repack.get("action_keys") == ["action.body_token_v1_1", "action[:14]"]
            and repack.get("state_keys") == ["observation.state"] and repack.get("action_mask_key") == "action_mask"
            and (repack.get("pad_action_dim"), repack.get("pad_state_dim")) == (80, 45)
            and (image.get("shared_image_profile"), image.get("motion_image_policy")) == ("256x320", "vlm_color_clean_motion"),
            "Body token/state/mask or actual clean visual policy differs")
    validate_release(source, release, config, preparation)
    return source, config, preparation, receipt


def validate_frozen_model_code(repo, preparation):
    records = {}
    for original, expected in preparation["code_sha256"].items():
        require("/632968d/" in original, "Unexpected frozen training source identity")
        relative = original.split("/632968d/", 1)[1]
        if any(relative.startswith(prefix) for prefix in CRITICAL_PREFIXES):
            require(sha256(regular(repo, relative)) == expected, "Inference model source changed: " + relative)
            records[relative] = expected
    require(all(any(n.startswith(prefix) for n in records) for prefix in CRITICAL_PREFIXES),
            "Missing critical frozen inference package proof")
    return {"status": "passed", "training_frozen_revision": "632968d", "matched_files": len(records), "sha256": records}


def validate_construction_metadata(resources):
    """Check accepted tokenizer/config semantics, without hashing old parameters."""
    manifest_path = regular(resources, "manifest.json")
    require(sha256(manifest_path) == RESOURCES_MANIFEST_SHA, "Accepted construction resource manifest SHA differs")
    manifest = read(manifest_path)
    require(manifest.get("format") == "cvpr-independent-4090-resources-v1" and manifest.get("step") == 20000,
            "Wrong accepted construction resource bundle")
    wanted = {"joint20k/qwen/" + name for name in QWEN_METADATA_NAMES} | {"joint20k/dino_training_contract.json"}
    records = {item["path"]: item for item in manifest["files"] if item["path"] in wanted}
    require(set(records) == wanted, "Accepted Qwen/DINO construction metadata inventory is incomplete")
    actual_qwen = {str(p.relative_to(resources)) for p in (resources / "joint20k/qwen").rglob("*") if p.is_file()}
    require(actual_qwen == wanted - {"joint20k/dino_training_contract.json"}, "Unbound Qwen tokenizer/config file")
    for name, item in records.items():
        path = regular(resources, name)
        require(path.stat().st_size == item["bytes"] and sha256(path) == item["sha256"],
                "Qwen/DINO construction metadata changed: " + name)
    return {"status": "passed", "resource_manifest_sha256": RESOURCES_MANIFEST_SHA,
        "qwen_metadata_files_verified": len(QWEN_METADATA_NAMES), "metadata_file_sha256": {name: item["sha256"] for name, item in records.items()},
        "motion_template_metadata_sha256": manifest["motion_validation"]["exported_metadata_sha256"],
        "old_joint_or_motion_parameter_hashes_recomputed": False}


def validate_motion_template_metadata(path, expected):
    from MotionExpert.checkpoints import read_checkpoint
    # read_checkpoint uses CPU mmap. Inspect only non-tensor metadata, never
    # copy or compare the old parameter values into the new exported model.
    checkpoint = read_checkpoint(path)
    metadata = {key: value for key, value in checkpoint.items() if key != "state_dict"}
    encoded = json.dumps(metadata, sort_keys=True, default=lambda value: value.tolist())
    actual = hashlib.sha256(encoded.encode()).hexdigest()
    require(actual == expected, "Accepted Motion construction metadata changed")
    return {"status": "passed", "metadata_sha256": actual, "old_parameter_values_used": False}


def recorded_encoder_transition(reference, actual):
    before = canonical_visual_identity({"encoder": deepcopy(reference)})["encoder"]
    after = canonical_visual_identity({"encoder": deepcopy(actual)})["encoder"]
    versions = {}
    for key in ("torch_version", "transformers_version"):
        require(isinstance(before.get(key), str) and isinstance(after.get(key), str), "Recorded DINO runtime version missing: " + key)
        versions[key] = {"accepted_reference": before.pop(key), "mobile100_training": after.pop(key)}
    before.pop("checkpoint", None)
    after.pop("checkpoint", None)
    require(before == after, "DINO transition changed weight/token/processor/precision/geometry identity")
    return {"status": "passed", "versions": versions, "all_other_encoder_fields_equal": True,
        "DINO_file_bytes_equal": True, "numerical_bitwise_equivalence_claimed": False,
        "actual_online_contract": "dino_training_contract.json retains the actual Mobile encoder runtime identity"}


def bind_training_motion_image_policy(contract, config):
    require(config["data"]["transform"]["model"].get("motion_image_policy") == "vlm_color_clean_motion",
            "Explicit vlm_color_clean_motion source policy is required")
    result = deepcopy(contract)
    result["motion_image_policy"] = "vlm_color_clean_motion"
    return result


def strict_component_check(weights, config, qwen):
    """Generic strict meta/CPU load copied from committed f4487c2, not its task gate."""
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration
    from psi.models.dual_source_action import DualSourceActionTransformer
    from psi.models.checkpoint_aliases import restore_shared_tensor_aliases
    torch.set_num_threads(1)
    m = config["model"]
    with safe_open(weights, framework="pt", device="cpu") as saved:
        keys = set(saved.keys())
        require(all(n.startswith(("vlm_model.", "action_header.", "motion_conditioner.")) for n in keys),
                "Unexpected tensor namespace in the complete model")
        signature = [0x44534143, 1, m["action_dim"], m["action_chunk_size"], m["hidden_dim"], m["num_blocks"],
            m["nhead"], m["view_feature_dim"], m["odim"], m["time_dim"], int(m["state_null_token"]),
            {None: 0, "rms_norm": 1, "layer_norm": 2}[m["qk_norm"]],
            struct.unpack("q", struct.pack("d", float(m["state_drop_prob"])))[0],
            struct.unpack("q", struct.pack("d", float(m["dropout"])))[0], 11, 768]
        require(saved.get_tensor("action_header._architecture_signature").tolist() == signature,
                "Saved architecture/state-drop signature differs from deployment config")
        qcfg = AutoConfig.from_pretrained(qwen, local_files_only=True)
        qcfg._attn_implementation = "sdpa"
        with torch.device("meta"):
            vlm = Qwen3VLForConditionalGeneration(qcfg)
            head = DualSourceActionTransformer(action_dim=m["action_dim"], action_pred_horizon=m["action_chunk_size"],
                action_hidden_dim=m["hidden_dim"], action_num_blocks=m["num_blocks"], action_nheads=m["nhead"],
                view_feature_dim=m["view_feature_dim"], odim=m["odim"], time_dim=m["time_dim"], qk_norm=m["qk_norm"],
                state_drop_prob=m["state_drop_prob"], state_null_token=m["state_null_token"], dropout=m["dropout"])
        head._buffers["_architecture_signature"] = torch.tensor(signature, dtype=torch.int64)
        vlm_state = {n.removeprefix("vlm_model."): saved.get_tensor(n) for n in keys if n.startswith("vlm_model.")}
        original_vlm_keys = set(vlm_state)
        vlm_state = restore_shared_tensor_aliases(vlm, vlm_state)
        vlm.load_state_dict(vlm_state, strict=True, assign=True)
        head_state = {n.removeprefix("action_header."): saved.get_tensor(n) for n in keys if n.startswith("action_header.")}
        head.load_state_dict(head_state, strict=True, assign=True)
        require(head.state_drop_prob == 0. and len(head.transformer_blocks) == 6, "Loaded state0/six-block semantics differ")
        counts = {p: sum(n.startswith(p + ".") for n in keys) for p in ("vlm_model", "action_header", "motion_conditioner")}
        require(all(counts.values()), "Complete saved VLM/action/Motion tensors are required")
    require(not torch.cuda.is_initialized(), "CPU structural load initialized CUDA")
    return {"status": "passed", "strict_VLM_load": True, "strict_action_load": True, "strict": True, "assign": True,
        "target_construction": "meta", "saved_values": "CPU mmap; no weight conversion", "forward_performed": False,
        "source_tensor_counts": counts, "restored_proven_VLM_aliases": sorted(set(vlm_state) - original_vlm_keys),
        "action_signature": signature, "state_drop_prob": 0., "num_blocks": 6, "CUDA_initialized": False}


def prepare(source, resources, output, repo, release):
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU preparation requires CUDA_VISIBLE_DEVICES='' explicitly")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    source, config, preparation, receipt = validate_source(source, release)
    original_stat = (source / "model.safetensors").stat()
    resources, repo, release = (Path(p).resolve(strict=True) for p in (resources, repo, release))
    output = Path(output).absolute()
    require(not output.exists() and not output.is_symlink()
            and not any(output.resolve().is_relative_to(p) or p.is_relative_to(output.resolve())
                        for p in (repo, source, resources, release)), "Use a fresh independent deployment outside source/repository/resources/release")
    code = validate_frozen_model_code(repo, preparation)
    construction = validate_construction_metadata(resources)
    sys.path[:0] = [str(repo / "VLMExpert/src"), str(repo), str(repo / "pipeline/src"), str(repo / "VLMExpert")]
    base = resources / "joint20k"
    for path in (base / "qwen", base / "motion/joint_motion.ckpt", resources / "dinov2-with-registers-large",
                 resources / "simple_local_usd_v1"):
        require(path.exists(), "Missing pinned construction resource: " + str(path))
    motion_metadata = validate_motion_template_metadata(base / "motion/joint_motion.ckpt",
                                                        construction["motion_template_metadata_sha256"])
    manifest = read(source / "release_metadata/dino/manifest.json")
    reference_contract = read(base / "dino_training_contract.json")
    transition = recorded_encoder_transition(reference_contract["encoder"], manifest["encoder"])
    contract = bind_training_motion_image_policy(reference_contract, config)
    contract["encoder"] = canonical_visual_identity({"encoder": deepcopy(manifest["encoder"])})["encoder"]
    contract.update(online_checkpoint=str(resources / "dinov2-with-registers-large"),
        source_manifest_sha256=sha256(source / "release_metadata/dino/manifest.json"),
        source_dataset="Mobile P&P100 train_only native30Hz clean cache; historical Handover VAL is regression-only")
    replacements = {"model.model_name_or_path": str(base / "qwen"),
        "model.motion_checkpoint": str(output / "motion/joint_motion.ckpt"), "model.pretrained_joint_checkpoint": None,
        "data.root_dir": str(resources / "simple_local_usd_v1"), "data.dino_cache_root": None,
        "data.transform.field.stat_path": str(output / "dataset_statistics.json"),
        "train.output_dir": str(output / "unused_training"), "train.resume_from_checkpoint": None,
        "train.deepspeed_config": str(repo / "VLMExpert/scripts/deepspeed/zero3.json")}
    deployed, argv = remap_config(config, (source / "argv.txt").read_text(), replacements)
    output.mkdir(parents=True)
    try:
        provenance = output / "provenance"
        for name in (*METADATA, "backup_receipt.json"):
            copied = provenance / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, copied)
        write_json(provenance / "runtime_metadata_validation.json", validate_runtime_metadata(source, config, preparation))
        write_json(provenance / "model_code_validation.json", code)
        write_json(provenance / "construction_metadata_validation.json", construction)
        write_json(provenance / "motion_template_metadata_validation.json", motion_metadata)
        shutil.copy2(resources / "manifest.json", provenance / "construction_resources_manifest.json")
        write_json(output / "run_config.json", deployed)
        (output / "argv.txt").write_text(argv)
        shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
        target = output / "checkpoints/ckpt_10000/model.safetensors"
        target.parent.mkdir(parents=True)
        os.link(source / "model.safetensors", target)
        write_json(output / "dino_training_contract.json", contract)
        from psi.deploy.dual_source_runtime import load_dino_contract
        load_dino_contract(output / "dino_training_contract.json")
        structure = strict_component_check(target, deployed, base / "qwen")
        write_json(output / "component_load_validation.json", structure)
        motion = export_motion(base / "motion/joint_motion.ckpt", target, output / "motion/joint_motion.ckpt",
                               "38000_then_mobile_pp100_task_10000", reference_contract["encoder"])
        require(motion.get("strict_meta_load") == "passed" and motion.get("exact_serialization_comparison") == "passed",
                "New complete-joint Motion export did not pass strict exact CPU serialization")
        motion.update(parameter_source="new_complete_mobile_pp100_task10000_joint_checkpoint",
            parameter_source_sha256=EXPECTED_MODEL_SHA, previous_motion_parameters_used=False,
            construction_metadata_only_source=str(base / "motion/joint_motion.ckpt"),
            mobile100_actual_encoder_runtime_transition=transition, actual_encoder_sha256=digest(contract["encoder"]))
        write_json(output / "motion/export_validation.json", motion)
        validate_config(deployed, output / "argv.txt", repo)
        require(deployed["model"]["state_drop_prob"] == 0., "Deployment must preserve measured state retention")
        validate_source(source, release, verify_weight=False)
        current = (source / "model.safetensors").stat()
        require((original_stat.st_dev, original_stat.st_ino, original_stat.st_size, original_stat.st_mtime_ns)
                == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), "Source model changed during CPU preparation")
        import torch
        require(not torch.cuda.is_initialized(), "CPU preparation initialized CUDA")
        report = {"contract": CONTRACT, "task_id": TASK_ID, "status": "local_mobile_pp100_run_prepared",
            "step": EXPECTED_STEP, "task_step": EXPECTED_STEP, "source_unifolm_step": 38000, "initial_task_step": 0,
            "step_semantics": "mobile_pp100_task_optimizer_updates", "source_global_batch": 64,
            "train_episodes": 100, "train_windows": 55906, "train_split": "train_only",
            "validation_role": VALIDATION_ROLE, "Mobile_validation_selected": False,
            "joint_sha256": EXPECTED_MODEL_SHA, "run_dir": str(output), "source_snapshot": str(source),
            "state_drop_prob": 0., "num_blocks": 6, "source_backup_receipt_sha256": sha256(source / "backup_receipt.json"),
            "full_saved_components_strict_CPU_load": structure, "strict_motion_load": motion["strict_meta_load"],
            "motion_weights": motion["weights"], "previous_motion_parameters_used": False,
            "normalization_refitted": False, "statistics_sha256": STAT_SHA, "actual_DINO_runtime_transition": transition,
            "motion_image_policy": "vlm_color_clean_motion", "deployment_path_changes": replacements,
            "optimizer_rng_restored": False, "GPU_used": False, "CUDA_initialized": False,
            "policy_forward_performed": False, "gpu_validation": "not_run", "physical_task_success": "not_measured",
            "checkpoint_save_completed": True, "training_completed": True,
            "source_metadata_sha256": {name: sha256(provenance / name) for name in (*METADATA, "backup_receipt.json")},
            "verifier_sha256": sha256(Path(__file__)),
            "construction_metadata_validation": construction, "motion_template_metadata_validation": motion_metadata,
            "shared_preparation_helpers_sha256": {name: sha256(Path(__file__).with_name(name))
                for name in ("export_bundle.py", "prepare_task_snapshot.py")},
            "strict_component_helper_origin": "committed f4487c2 prepare_handover149_eval.py generic meta-load only; Mobile gate is independent",
            "source_model_bytes_verified": True, "source_model_identity_unchanged_during_prepare": True}
        write_json(output / "LOCAL_RUN_READY.json", report)
        return report
    except BaseException as error:
        write_json(output / "PREPARATION_FAILED.json", {"contract": CONTRACT, "status": "failed_incomplete", "error": repr(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "resources", "output", "release"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=REPO)
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.resources, args.output, args.repo, args.release), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
