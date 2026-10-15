"""Seed a new quality revision with unchanged, previously accepted raw shards.

Only episodes whose complete committed records are identical can be reused.
Changed episodes, indices and packed arrays must go through the normal builder.
This command never publishes a cache or copies an old acceptance decision.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np

from .dataset.window_cache import ARRAY_SPECS, CONTRACT, atomic_json, cache_binding, digest, file_sha256, protocol_from_config, stamp
from .dataset.vision_cache import CONTRACT as VISION_CONTRACT, storage_digest


def computation_ast(source):
    """Ignore only the archive publication gate, never decoding/indexing logic."""
    tree = ast.parse(source)
    classes = [item for item in tree.body if isinstance(item, ast.ClassDef)
               and item.name == "SMPLHParameterArchive"]
    if len(classes) != 1:
        raise ValueError("Expected the original SMPL-H archive class")
    methods = classes[0].body
    if sum(isinstance(item, ast.FunctionDef) and item.name == "_check_acceptance" for item in methods) != 1:
        raise ValueError("Expected one archive acceptance method")
    classes[0].body = [item for item in methods
                       if not (isinstance(item, ast.FunctionDef) and item.name == "_check_acceptance")]
    return ast.dump(tree, include_attributes=False)


def check_computation(previous_binding, current_binding, previous_source):
    """A publication-only reader extension cannot alter the cached values."""
    before = deepcopy(previous_binding["representation_sha256"])
    after = deepcopy(current_binding["representation_sha256"])
    key = "MotionExpert.dataset.smplh_archive"
    if hashlib.sha256(previous_source).hexdigest() != before.get(key):
        raise ValueError("Previous implementation snapshot does not match the accepted cache")
    from .dataset import smplh_archive
    current_source = Path(smplh_archive.__file__).read_bytes()
    if computation_ast(previous_source) != computation_ast(current_source):
        raise ValueError("Archive computation changed beyond the publication gate; rebuild all shards")
    before.pop(key)
    after.pop(key)
    if before != after:
        raise ValueError("Window representation/indexing computation changed; rebuild all shards")
    for key in ("protocol", "split_fingerprint", "model_sha256", "image_profile_sha256"):
        if previous_binding.get(key) != current_binding.get(key):
            raise ValueError(f"Unchanged-shard reuse requires identical {key}")


def check_archive_info(previous_info, current_info):
    """Allow only a new revision declaration over the same archive schema."""
    before, after = deepcopy(previous_info), deepcopy(current_info)
    parent_revision = before.pop("quality_revision", None)
    revision = after.pop("quality_revision", None)
    if before != after:
        raise ValueError("Only the declared quality revision may change archive information")
    if parent_revision is not None and (not isinstance(revision, dict)
            or revision.get("parent_revision") != parent_revision):
        raise ValueError("Revised archive does not preserve its exact parent revision declaration")


def _owned_file(root, relative):
    root, relative = Path(root).resolve(strict=True), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Visual reuse artifacts require owned relative paths")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Visual reuse artifact escapes its profile root")
    return path


def previous_visual_source(cache_manifest):
    """Bind the old visual version to the already accepted window manifest."""
    root = Path(cache_manifest["vision_profile_root"]).resolve(strict=True)
    path = _owned_file(root, "manifest.json")
    expected = cache_manifest["binding"]["vision_manifest_sha256"]
    if file_sha256(path) != expected:
        raise ValueError("Previously accepted visual manifest identity changed")
    manifest = json.loads(path.read_text())
    if (manifest.get("contract") != VISION_CONTRACT or manifest.get("state") != "complete"
            or manifest.get("scope") != "all" or manifest.get("validity_equal_source") is not True):
        raise ValueError("Previously accepted visual profile is not a complete native source")
    return root, manifest, expected


def visual_reuse_proof(record, *, previous_root, current_root, previous_manifest,
                       current_manifest, previous_manifest_sha256,
                       previous_archive_root, current_archive_root):
    """Prove embeddings and all join/validity bytes are the same frozen files.

    Equal encoder/profile declarations cannot prove image-validity ordering.
    Checking both receipts and device/inode/stat establishes exact payload reuse
    without rereading the feature payloads.
    """
    receipts, paths = [], []
    prefix = f"episodes/{record['episode_index']:08d}"
    for root, manifest in ((previous_root, previous_manifest), (current_root, current_manifest)):
        path = _owned_file(root, f"{prefix}/receipt.json")
        receipt = json.loads(path.read_text())
        if (receipt.get("contract") != VISION_CONTRACT or receipt.get("state") != "complete"
                or receipt.get("episode_index") != record["episode_index"]
                or receipt.get("source_key") != record["source_key"]
                or receipt.get("source_record_sha256") != storage_digest(record)
                or receipt.get("frames") != record["length"]
                or receipt.get("validity_equal_source") is not True
                or receipt.get("encoder") != manifest.get("encoder")
                or not isinstance(receipt.get("encoder"), dict) or not receipt["encoder"]
                or set(receipt.get("output_identities", {})) != {"features.npy", "join.npz"}):
            raise ValueError("Visual reuse receipt differs from the unchanged source/encoder")
        receipts.append(receipt)
        paths.append(path)
    if receipts[0].get("valid_count") != receipts[1].get("valid_count"):
        raise ValueError("Visual reuse validity count changed")
    old_receipt_sha256, new_receipt_sha256 = map(file_sha256, paths)
    inherited = receipts[1].get("quality_revision_reuse", {})
    join_evidence = inherited.get("join_evidence")
    join_proven = ((inherited.get("join_all_native_rows_checked") is True
                   and join_evidence == "changed_motion_all_native_rows_compared")
                  or (inherited.get("join_all_native_rows_checked") is False
                      and join_evidence == "inherited_immutable_motion_and_visual"))
    if (inherited.get("parent_receipt_sha256") != old_receipt_sha256
            or inherited.get("parent_manifest_sha256") != previous_manifest_sha256
            or not join_proven
            or inherited.get("payload_reuse") != "same_device_inode_hardlink"):
        raise ValueError("Rebound visual receipt lacks its exact parent/join reuse proof")
    motion_stats = [_owned_file(root, record["data_path"]).stat()
                    for root in (previous_archive_root, current_archive_root)]
    motion_identity = [(stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns) for stat in motion_stats]
    if motion_identity[0] != motion_identity[1]:
        raise ValueError("Unchanged source motion is not the same frozen hardlinked payload")
    identities = {}
    for name in ("features.npy", "join.npz"):
        stats = []
        for root, receipt in zip((previous_root, current_root), receipts):
            path = _owned_file(root, f"{prefix}/{name}")
            stat = path.stat()
            identity = receipt["output_identities"][name]
            if (stat.st_size, stat.st_mtime_ns) != (identity.get("size_bytes"), identity.get("mtime_ns")):
                raise ValueError("Visual reuse payload differs from its committed file identity")
            stats.append((stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
        if stats[0] != stats[1]:
            raise ValueError("Visual features/join are not the same frozen hardlinked payload")
        identities[name] = dict(zip(("device", "inode", "size_bytes", "mtime_ns"), stats[0]))
    if tuple(map(file_sha256, paths)) != (old_receipt_sha256, new_receipt_sha256):
        raise ValueError("Visual receipt changed during reuse verification")
    return {"previous_receipt_sha256": old_receipt_sha256, "current_receipt_sha256": new_receipt_sha256,
            "previous_manifest_sha256": previous_manifest_sha256, "source_record_sha256": storage_digest(record),
            "payload_identity": "same_device_inode_size_mtime", "payloads": identities,
            "motion_identity": dict(zip(("device", "inode", "size_bytes", "mtime_ns"), motion_identity[0])),
            "feature_payload_read": False, "join_bytes_preserved": True, "join_evidence": join_evidence}


def reuse_episode(old_root, new_root, descriptor, record, binding, vision_metadata, *, vision_sources):
    """Hardlink raw arrays; commit a fresh receipt only after identity checks."""
    old_root, new_root = Path(old_root).resolve(), Path(new_root).resolve()
    receipt_path = old_root / descriptor["receipt_path"]
    if file_sha256(receipt_path) != descriptor["receipt_sha256"]:
        raise ValueError("Accepted episode receipt identity changed")
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get("contract") != CONTRACT or receipt.get("state") != "complete"
            or receipt.get("episode_index") != record["episode_index"]
            or receipt.get("source_record_sha256") != digest(record)
            or receipt.get("split") != record["split"]
            or receipt.get("windows") != descriptor["windows"]
            or set(receipt.get("arrays", {})) != set(ARRAY_SPECS)):
        raise ValueError("Accepted episode receipt differs from the unchanged source")
    before_vision = deepcopy(receipt["template_metadata"].get("vision_profile"))
    after_vision = deepcopy(vision_metadata)
    for value in (before_vision, after_vision):
        if not isinstance(value, dict):
            raise ValueError("Unchanged shard requires committed visual metadata")
        value.pop("manifest_sha256", None)
    if before_vision != after_vision:
        raise ValueError("Visual encoder/geometry/normalization changed")
    visual_proof = visual_reuse_proof(record, **vision_sources)
    destination = new_root / f"episodes/{record['episode_index']:08d}"
    if destination.exists():
        raise FileExistsError("Reuse never replaces a committed or unknown destination")
    temporary = new_root / f"episodes/.{record['episode_index']:08d}.reusing"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    identities = {}
    try:
        for name, (dtype, tail) in ARRAY_SPECS.items():
            identity = receipt["arrays"][name]
            relative = Path(identity["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Accepted shard array path is not owned")
            source = (old_root / relative).resolve(strict=True)
            if not source.is_relative_to(old_root):
                raise ValueError("Accepted shard array escapes its root")
            if (identity.get("dtype") != dtype
                    or identity.get("shape") != [receipt["windows"], *tail]
                    or stamp(source) != (identity["size_bytes"], identity["mtime_ns"])):
                raise ValueError("Accepted shard array identity changed")
            target = temporary / f"{name}.npy"
            os.link(source, target)
            if source.stat().st_ino != target.stat().st_ino or stamp(target) != stamp(source):
                raise ValueError("Hardlink failed to preserve accepted array identity")
            identities[name] = dict(identity, path=str((destination / target.name).relative_to(new_root)))
        starts = np.load(temporary / "start_frames.npy", allow_pickle=False)
        if (starts.dtype != np.int64 or starts.shape != (receipt["windows"],)
                or digest(starts.tolist()) != receipt["start_frames_sha256"]):
            raise ValueError("Accepted shard window keys changed")
        result = deepcopy(receipt)
        result.update(binding_sha256=digest(binding), arrays=identities)
        result["template_metadata"]["vision_profile"] = deepcopy(vision_metadata)
        result["reuse"] = {"source_receipt_sha256": descriptor["receipt_sha256"],
                           "source_cache": str(old_root), "unchanged_complete_record": True,
                           "array_identity": "same_filesystem_hardlinks", "visual_proof": visual_proof}
        atomic_json(temporary / "receipt.json", result)
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--previous-cache", required=True, type=Path)
    parser.add_argument("--previous-acceptance", required=True, type=Path)
    parser.add_argument("--previous-archive-implementation", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    from .build_window_cache import _archive
    from .config import load_config

    old, root = args.previous_cache.resolve(strict=True), args.output_root.resolve()
    if old == root or root.is_relative_to(old) or old.is_relative_to(root):
        raise ValueError("New cache must be independent of the previous cache")
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (root / "manifest.json").exists() or (root / "reuse.json").exists():
            raise FileExistsError("Seed only a new cache before running the normal builder")
        manifest_path = old / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        acceptance_bytes = args.previous_acceptance.read_bytes()
        acceptance = json.loads(acceptance_bytes)
        acceptance_sha256 = hashlib.sha256(acceptance_bytes).hexdigest()
        if (manifest.get("contract") != CONTRACT or manifest.get("state") != "complete"
                or manifest.get("scope") != "all" or manifest.get("diagnostic") is not False
                or manifest.get("binding_sha256") != digest(manifest["binding"])
                or acceptance.get("format") != "motionexpert-window-cache-acceptance-v1"
                or acceptance.get("passed") is not True or acceptance.get("state") != "complete"
                or acceptance.get("cache_manifest_sha256") != file_sha256(manifest_path)
                or Path(acceptance.get("cache_root", "")).resolve() != old):
            raise ValueError("Previous complete cache lacks its matching passed acceptance")
        records_path = old / manifest["records_path"]
        if file_sha256(records_path) != manifest["records_sha256"]:
            raise ValueError("Previous cache record identity changed")
        before_records = {record["episode_index"]: record for record in json.loads(records_path.read_text())}
        cfg = load_config(args.config)
        archive = _archive(cfg, False)
        binding = cache_binding(archive, protocol_from_config(cfg))
        check_computation(manifest["binding"], binding, args.previous_archive_implementation.read_bytes())
        old_archive = Path(manifest["archive_root"]).resolve(strict=True)
        old_info = old_archive / "meta/info.json"
        if file_sha256(old_info) != manifest["binding"]["archive_binding"]["info_sha256"]:
            raise ValueError("Previous archive information changed")
        check_archive_info(json.loads(old_info.read_text()), archive.info)
        for key in ("model_sha256", "hand_basis_sha256", "split_metadata_sha256"):
            if manifest["binding"]["archive_binding"][key] != binding["archive_binding"][key]:
                raise ValueError(f"Unchanged-shard reuse requires identical archive {key}")
        if file_sha256(old_archive / "meta/ontology.json") != file_sha256(archive.root / "meta/ontology.json"):
            raise ValueError("Archive ontology changed; rebuild all shards")
        descriptors = {item["episode_index"]: item for item in manifest["episodes"]}
        previous_vision_root, previous_vision_manifest, previous_vision_manifest_sha256 = previous_visual_source(manifest)
        vision_sources = {"previous_root": previous_vision_root, "current_root": archive.vision_cache.root,
                          "previous_manifest": previous_vision_manifest,
                          "current_manifest": archive.vision_cache.manifest,
                          "previous_manifest_sha256": previous_vision_manifest_sha256,
                          "previous_archive_root": old_archive, "current_archive_root": archive.root}
        reused = []
        for record in archive.records():
            index = record["episode_index"]
            if before_records.get(index) != record:
                continue
            descriptor = descriptors[index]
            # Zero-window shards have no template metadata and cost no FK.
            if descriptor["windows"] == 0:
                continue
            reuse_episode(old, root, descriptor, record, binding, archive.vision_cache.metadata, vision_sources=vision_sources)
            reused.append(index)
            if len(reused) % 250 == 0:
                print(f"Reused {len(reused)} unchanged episode shards", flush=True)
        archive.assert_current()
        if file_sha256(manifest_path) != acceptance["cache_manifest_sha256"]:
            raise ValueError("Previous accepted manifest changed during reuse")
        if (file_sha256(args.previous_acceptance) != acceptance_sha256
                or file_sha256(records_path) != manifest["records_sha256"]
                or file_sha256(previous_vision_root / "manifest.json") != previous_vision_manifest_sha256):
            raise ValueError("Previously accepted source/visual evidence changed during reuse")
        atomic_json(root / "reuse.json", {"state": "seeded_only_not_accepted", "episodes": reused,
                    "binding_sha256": digest(binding), "source_manifest_sha256": file_sha256(manifest_path),
                    "previous_acceptance_sha256": acceptance_sha256,
                    "full_builder_and_independent_acceptance_required": True})
        print(f"Seeded {len(reused)} shards; run the full builder and independent acceptance", flush=True)


if __name__ == "__main__":
    main()
