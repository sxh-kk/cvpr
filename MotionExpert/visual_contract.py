"""Keep visual versions separate from unchanged physical motion statistics."""
from copy import deepcopy
from collections.abc import Mapping
import json
from pathlib import Path
import re


def _validate_shared(value, configured_profile=None):
    from ImageProcess import load_profile
    process, encoder = value.get("image_process"), value.get("encoder")
    if not isinstance(process, Mapping) or not isinstance(encoder, Mapping) or "profile" not in process:
        raise ValueError("Shared-image checkpoint requires complete image_process and encoder metadata")
    profile = load_profile(process["profile"])
    if (process.get("profile_sha256") != profile.fingerprint or process.get("profile_id") != profile.profile_id
            or (process.get("height"), process.get("width")) != profile.output_hw
            or encoder.get("image_process") != process):
        raise ValueError("Shared-image checkpoint profile hash/dimensions or encoder identity differ")
    # Built-in names/config mappings are portable. Do not reopen an old config-file
    # path when exporting a checkpoint whose entire profile is already embedded.
    if configured_profile is not None and (isinstance(configured_profile, Mapping) or configured_profile in ("256x480", "256x320")):
        if load_profile(configured_profile).fingerprint != profile.fingerprint:
            raise ValueError("Checkpoint IMAGE_PROFILE conflicts with its embedded visual contract")


def training_visual_contract(cfg):
    root = getattr(cfg.DATA, "VISION_PROFILE_ROOT", None)
    profile = getattr(cfg.DATA, "IMAGE_PROFILE", None)
    if root is None:
        raise ValueError("Training requires a completed native30Hz shared VISION_PROFILE_ROOT")
    manifest = json.loads((Path(root) / "manifest.json").read_text())
    if (manifest.get("contract") != "smplh-dino-shared-rgb-cache-v1"
            or manifest.get("state") != "complete" or manifest.get("scope") != "all"):
        raise ValueError("Training visual profile must be a completed full cache")
    from ImageProcess import load_profile
    extracted = load_profile(manifest["image_process"]["profile"])
    if profile is not None and load_profile(profile).fingerprint != extracted.fingerprint:
        raise ValueError("Training IMAGE_PROFILE does not match the extracted visual version")
    contract = {"format": "motionexpert-visual-v1", "mode": "shared_rgb",
                "image_process": deepcopy(manifest["image_process"]),
                "encoder": deepcopy(manifest["encoder"])}
    _validate_shared(contract, profile)
    return contract


def checkpoint_visual_contract(checkpoint):
    value = checkpoint.get("motionexpert_visual_conditioning")
    data = checkpoint.get("hyper_parameters", {}).get("DATA", {})
    if not isinstance(value, Mapping) or value.get("format") != "motionexpert-visual-v1" or value.get("mode") != "shared_rgb":
        raise ValueError("Checkpoint requires the shared-image visual conditioning contract")
    _validate_shared(value, data.get("IMAGE_PROFILE"))
    return deepcopy(value)


def cache_visual_contract(cache):
    """Use the already validated cache manifest, without reopening archive files."""
    if cache is None:
        raise ValueError("Archive inference/evaluation requires a validated VISION_PROFILE_ROOT")
    manifest = cache.manifest
    if (manifest.get("contract") != "smplh-dino-shared-rgb-cache-v1"
            or manifest.get("state") != "complete" or manifest.get("scope") != "all"):
        raise ValueError("Actual visual cache must be a completed full shared-RGB cache")
    contract = {"format": "motionexpert-visual-v1", "mode": "shared_rgb",
                "image_process": deepcopy(manifest["image_process"]),
                "encoder": deepcopy(manifest["encoder"])}
    _validate_shared(contract)
    return contract


def _encoder_semantics(contract):
    """Separate DINO identity/normalization from geometry and execution settings."""
    encoder = deepcopy(contract["encoder"])
    required = {"checkpoint", "revision", "shape", "token_order", "storage_dtype",
                "pooling", "normalization", "processor", "preprocessing"}
    if not required.issubset(encoder):
        raise ValueError("Visual compatibility requires complete DINO identity/token/normalization metadata")
    if (encoder["shape"] != [5, 1024] or encoder["storage_dtype"] != "float32"
            or encoder["token_order"] != ["cls", "register_0", "register_1", "register_2", "register_3"]):
        raise ValueError("DINO token order/shape/storage contract differs from CLS + 4 registers")
    revision = encoder["revision"]
    if not isinstance(revision, str) or (revision != "local" and not re.fullmatch(r"[0-9a-fA-F]{40}", revision)):
        raise ValueError("DINO weight identity requires a pinned revision")
    for key in ("processor", "preprocessing"):
        if not isinstance(encoder[key], Mapping):
            raise ValueError("DINO normalization/preprocessing metadata must be a mapping")
        if not {"do_rescale", "rescale_factor", "do_normalize", "image_mean", "image_std"}.issubset(encoder[key]):
            raise ValueError("DINO normalization metadata is incomplete")
        if encoder[key].get("do_resize") is not False or encoder[key].get("do_center_crop") is not False:
            raise ValueError("Shared-RGB DINO must disable its second resize and center crop")
        # The shared processor is the authority for H/W; these disabled resize
        # size fields merely describe the current profile in the HF processor.
        encoder[key].pop("size", None)
    encoder.pop("image_process", None)
    encoder.pop("optimization", None)
    if encoder["revision"] == "local":
        files = encoder.get("local_file_sha256")
        if (not isinstance(files, Mapping) or not files
                or not any(str(name).endswith((".bin", ".safetensors")) for name in files)
                or any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value) for value in files.values())):
            raise ValueError("Local DINO weights require their file SHA256 identities")
        # A checkpoint directory may move while every model/config byte is fixed.
        encoder["checkpoint"] = "local_files_by_sha256"
    return encoder


def _size_adaptation_invariants(process):
    """Remove only size fields; keep interpolation, implementation and augmentation."""
    process = deepcopy(process)
    for key in ("profile_id", "profile_sha256", "height", "width"):
        process.pop(key, None)
    profile = process["profile"]
    profile.pop("profile_id", None)
    for key in ("resize_hw", "center_crop_hw", "output_hw"):
        profile["geometry"].pop(key, None)
    return process


def compare_visual_contracts(trained, actual, *, allow_visual_adaptation=False):
    """Default to the trained RGB profile; explicitly allow size-only adaptation.

    DINO weights, revision, normalization and token ordering always remain fixed.
    Optimization parameters such as preprocessing workers are execution details.
    This declaration permits different conditioning geometry; it does not train
    or alter model weights.
    """
    if not isinstance(allow_visual_adaptation, bool):
        raise TypeError("allow_visual_adaptation must be a bool")
    for value in (trained, actual):
        if not isinstance(value, Mapping) or value.get("format") != "motionexpert-visual-v1" or value.get("mode") != "shared_rgb":
            raise ValueError("Visual compatibility requires shared-image conditioning contracts")
        _validate_shared(value)
    if _encoder_semantics(trained) != _encoder_semantics(actual):
        raise ValueError("Actual DINO weight/revision/normalization/token identity differs from the checkpoint; visual adaptation cannot permit this")
    source, target = trained["image_process"], actual["image_process"]
    changed = source != target
    if changed:
        if not allow_visual_adaptation:
            raise ValueError("Actual archive visual profile differs from the checkpoint; pass --allow-visual-adaptation only for an explicit shared-RGB size adaptation")
        if _size_adaptation_invariants(source) != _size_adaptation_invariants(target):
            raise ValueError("Visual adaptation permits shared-RGB size profiles only; other preprocessing/augmentation identity differs")
    return {
        "requested": allow_visual_adaptation,
        "profile_changed": changed,
        "mode": "explicit_shared_rgb_size_adaptation" if changed else "matched",
        "scope": "image_geometry_only",
        "encoder_identity_verified": True,
        "trained_profile_id": source["profile_id"],
        "actual_profile_id": target["profile_id"],
        "trained_hw": [source["height"], source["width"]],
        "actual_hw": [target["height"], target["width"]],
    }


def archive_visual_compatibility(dataset, trained, *, allow_visual_adaptation=False):
    """Audit the cache already opened by the dataset and return report metadata."""
    cache = dataset.archive.vision_cache
    actual = cache_visual_contract(cache)
    adaptation = compare_visual_contracts(trained, actual, allow_visual_adaptation=allow_visual_adaptation)
    return {
        "trained_visual_conditioning": deepcopy(trained),
        "actual_visual_conditioning": actual,
        "visual_cache_metadata": deepcopy(cache.metadata),
        "visual_cache_root": str(cache.root),
        "visual_adaptation": adaptation,
    }


def validate_visual_resume(cfg, checkpoint):
    if training_visual_contract(cfg) != checkpoint_visual_contract(checkpoint):
        raise ValueError("Resume visual version differs from the checkpoint; use --init-weights for an explicit new adaptation run")
