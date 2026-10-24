"""Strict post-collection transfer and real A100 CPU reader verification.

Prepare requires all100 collection/diversity, pack, DINO and local production
gates. It never collects, extracts features, constructs a model or trains.
Upload is explicit; it owns a fresh release, preserves original DINO metadata,
relocates only declared metadata fields, and writes RELEASE_READY last.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import uuid

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CONTRACT = "cvpr-handover-random100-final-transfer-v1"
REMOTE_ROOT = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_random100_20261024"
REMOTE_PYTHON = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126/bin/python"
REMOTE_FACTORY = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/quality100_compare_20261023/quality100_bs8_fresh.py"
REMOTE_REPO = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_38k_preparation/fused_adamw_validation/code/632968d"
REMOTE_CONFIG = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/quality100_compare_20261023/experiment_bs8_state_keep_fresh_20261023/run_config.json"
REMOTE_REFERENCE = "/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_quality100_20261023/dino-quality100-v1"
HELPERS = ("verify_handover_random100.py", "verify_handover_supplement.py", "pack_handover_random100.py",
    "pack_handover_quality100.py", "handover_random100_sources.py", "prepare_carrybox_v11.py",
    "prepare_handover_eval.py", "finetune_carrybox.py", "finalize_handover_random100.py")
MANIFEST_KEYS = ("contract", "source", "image_process", "encoder", "extraction", "storage_dtype", "shape", "episode_indices")
SSH = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12"]


def require(value, message):
    if not value: raise ValueError(message)


def now(): return datetime.now(timezone.utc).isoformat()
def read(path): return json.loads(Path(path).read_text())
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""): result.update(block)
    return result.hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream: stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def owned(root, relative):
    root, name = Path(root).resolve(strict=True), Path(relative)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe release-relative path")
    raw = root / name
    require(raw.is_file() and not raw.is_symlink() and raw.resolve().is_relative_to(root), "Missing or aliased release payload")
    return raw


def identity(path):
    path = Path(path); before = path.stat()
    require(path.is_file() and not path.is_symlink(), "Use regular payload files")
    result = {"size_bytes": before.st_size, "mtime_ns": before.st_mtime_ns, "sha256": sha(path)}
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), "Payload changed while hashing")
    return result


def verify_shas(bindings):
    require(bindings and all(Path(path).is_file() and sha(path) == expected for path, expected in bindings.items()),
            "Bound gate/source bytes changed")


def relocate_metadata(manifest, receipts, dataset_root):
    """Pure whitelisted relocation; no fake roots or source identity rewrites."""
    require(manifest.get("contract") == "unifolm-dino-native-frames-v1" and manifest.get("state") == "complete"
            and manifest.get("signature") == digest({key: manifest[key] for key in MANIFEST_KEYS}),
            "Require a completed identity-bound original DINO manifest")
    require(Path(dataset_root).is_absolute() and ".." not in Path(dataset_root).parts,
            "Use an explicit canonical destination dataset root")
    indices = manifest["episode_indices"]
    require(len(indices) == len(set(indices)) and set(receipts) == set(indices), "DINO receipt coverage differs")
    after = deepcopy(manifest); after["source"]["dataset_root"] = str(dataset_root)
    after["signature"] = digest({key: after[key] for key in MANIFEST_KEYS})
    restored = deepcopy(after); restored["source"]["dataset_root"] = manifest["source"]["dataset_root"]
    restored["signature"] = manifest["signature"]
    require(restored == manifest, "Relocation changed a non-whitelisted manifest field")
    changed = {}
    for index, before in receipts.items():
        require(before.get("contract") == manifest["contract"] and before.get("state") == "complete"
                and before.get("episode_index") == index and before.get("manifest_signature") == manifest["signature"]
                and before.get("signature") == digest({"manifest_signature": manifest["signature"], "source": before["source"]}),
                "Original receipt identity differs")
        receipt = deepcopy(before); receipt["manifest_signature"] = after["signature"]
        receipt["signature"] = digest({"manifest_signature": after["signature"], "source": receipt["source"]})
        check = deepcopy(receipt); check["manifest_signature"] = before["manifest_signature"]; check["signature"] = before["signature"]
        require(check == before, "Relocation changed non-whitelisted receipt fields")
        changed[index] = receipt
    return after, changed


def validate_gate_reports(collection, ready, production):
    require(collection.get("contract") == "cvpr-handover-random100-collection-ready-v1"
            and collection.get("quality_and_diversity_passed") is True and collection.get("episodes") == 100
            and collection.get("ready_for_pack") is True, "Exactly100 strict collection/diversity must pass first")
    require(ready.get("contract") == "cvpr-handover-random100-lerobot-pack-v1"
            and ready.get("train_episodes") == 100 and ready.get("train_payload_packed") is True
            and ready.get("train_target_satisfied") is True and ready.get("minimum_distinct_sources_satisfied") is True
            and ready.get("old_captures_merged") is False and ready.get("normalization_refitted") is False
            and ready.get("final_ready") is False, "Require exact new100 immutable pack; no old-data merge")
    require(production.get("contract") == "cvpr-handover-random100-reader-verification-v1"
            and production.get("status") == "real_DINO_and_production_reader_passed"
            and production.get("train_episodes") == 100 and production.get("train_windows") == ready["train_windows"]
            and production.get("production_ready_for_bound_pack_and_cache") is True
            and production.get("formal100_quality_and_diversity_passed") is True
            and production.get("GPU_used") is False and production.get("CUDA_initialized") is False
            and production.get("model_constructed") is False and production.get("training_started") is False
            and production.get("old_packages_modified") is False and production.get("VAL_or_TEST_loaded") is False
            and production.get("PACK_READY_modified") is False
            and production.get("final_ready_automatically_promoted") is False,
            "A real complete local DINO/production CPU reader pass is mandatory before any upload")
    require(len(production.get("cache_episodes", [])) == 100
            and {row["episode_index"] for row in production["cache_episodes"]} == set(range(100))
            and all(row.get("every_feature_finite") is True and row.get("every_frame_valid") is True
                    for row in production["cache_episodes"])
            and {row["episode_index"] for row in production.get("actual_reader_queries", [])} == set(range(100))
            and all(row.get("actual_current_RGB_and_production_collator_passed") is True
                    for row in production["actual_reader_queries"]),
            "Local verification did not inspect all100 real DINO episodes/production samples")


def local_gates(collection_root, pack_root, dino_root, local_reader_report):
    import verify_handover_random100 as verifier
    import pack_handover_random100 as packer
    production_report = local_reader_report
    collection_root, pack_root, dino_root = [Path(path).resolve(strict=True) for path in (collection_root, pack_root, dino_root)]
    marker = collection_root / "COLLECTION_READY.json"
    collection, ready, production = read(marker), read(pack_root / "PACK_READY.json"), read(production_report)
    validate_gate_reports(collection, ready, production)
    pack, ready, _, _, bindings = verifier.inspect_pack(pack_root)
    require(Path(production["pack_root"]).resolve(strict=True) == pack
            and Path(production["dino_root"]).resolve(strict=True) == dino_root
            and production["verifier_sha256"] == sha(HERE / "verify_handover_random100.py"),
            "Local production receipt belongs to another pack/cache/verifier")
    registry, diversity = Path(collection["registry_path"]).resolve(strict=True), Path(collection["diversity_path"]).resolve(strict=True)
    require(registry.is_relative_to(collection_root / "registry_versions") and diversity.is_relative_to(collection_root / "registry_versions")
            and sha(registry) == collection["registry_sha256"] and sha(diversity) == collection["diversity_sha256"],
            "Final strict registry/diversity binding differs")
    requirements = collection_root / "REBUILD_REQUIREMENTS.json"
    r, d = read(registry), read(diversity)
    packer.validate_selection(r["admissions"], read(requirements))
    packer.validate_diversity(d, sha(requirements), sha(registry), r["admissions"])
    for name, original in (("registry.json", registry), ("diversity.json", diversity), ("REBUILD_REQUIREMENTS.json", requirements)):
        require(sha(pack / "rebuild_evidence" / name) == sha(original), "Pack is not the completed collection snapshot")
    manifest = read(dino_root / "manifest.json")
    require(Path(manifest["source"]["dataset_root"]).resolve(strict=True) == pack / "train"
            and set(manifest["episode_indices"]) == set(range(100)), "Local DINO root/frame provenance differs")
    source_receipts = production["source_sha256"]
    require(all(source_receipts.get(path) == expected for path, expected in bindings.items()), "Production receipt omits packed payload bindings")
    for index in range(100):
        for name in ("features.npy", "frames.npz", "receipt.json"):
            path = dino_root / "episodes" / f"{index:08d}" / name
            require(source_receipts.get(str(path)) == sha(path), "DINO actual episode bytes differ from local production pass")
    require(source_receipts.get(str(dino_root / "manifest.json")) == sha(dino_root / "manifest.json"), "DINO manifest changed since local verification")
    verify_shas(source_receipts); verify_shas(d["bindings"])
    return {**bindings, str(marker): sha(marker), str(registry): sha(registry), str(diversity): sha(diversity),
            str(requirements): sha(requirements), str(Path(production_report).resolve()): sha(production_report)}, manifest


def remote_python(host, python, code, value):
    require(re.fullmatch(r"[A-Za-z0-9_.@:-]+", host) and not host.startswith("-"), "Invalid explicit SSH host")
    command = shlex.join([python, "-B", "-c", code])
    result = subprocess.run([*SSH, host, command], input=json.dumps(value, allow_nan=False),
                            text=True, capture_output=True, check=True)
    return json.loads(result.stdout.strip().splitlines()[-1])


READONLY_REMOTE_PROBE = r'''
from pathlib import Path
import hashlib,json,sys
v=json.load(sys.stdin)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
repo=Path(v['repo']).resolve(strict=True)
paths=[Path(v['factory']),Path(v['run_config']),Path(v['reference_dino'])/'manifest.json',repo/'scripts/4090_simple/finetune_carrybox.py',repo/'VLMExpert/src/psi/data/lerobot/motion_history.py']
assert all(p.is_file() for p in paths)
config=json.loads(paths[1].read_text());processor=Path(config['model']['model_name_or_path']).resolve(strict=True)
assert processor.is_dir()
print(json.dumps({'bindings':{str(p.resolve()):sha(p) for p in paths},'repo':str(repo),'processor':str(processor),'destination_exists':Path(v['root']).exists(),'read_only':True}))
'''


def inventory_trees(trees):
    result = {}
    for name, root in trees.items():
        root = Path(root).resolve(strict=True)
        require(root.is_dir() and name in ("pack", "dino", "code", "evidence"), "Unexpected transferable tree")
        for path in sorted(root.rglob("*")):
            require(not path.is_symlink(), "Transfer payload may not contain path aliases")
            if path.is_file(): result[f"{name}/{path.relative_to(root)}"] = identity(path)
    return result


def prepare(args):
    bindings, _ = local_gates(args.collection_root, args.pack_root, args.dino_root, args.local_reader_report)
    bundle = Path(args.bundle_root).absolute()
    require(not bundle.exists() and not bundle.is_symlink() and not bundle.resolve().is_relative_to(REPO)
            and all(not bundle.resolve().is_relative_to(Path(path).resolve()) for path in
                    (args.collection_root, args.pack_root, args.dino_root)), "Use a fresh external independent transfer bundle")
    require(str(args.remote_root) == REMOTE_ROOT, "This release owns only the requested new random100 remote destination")
    remote = {"root": str(args.remote_root), "python": str(args.remote_python), "repo": str(args.remote_repo),
              "factory": str(args.remote_factory), "run_config": str(args.remote_run_config),
              "reference_dino": str(args.remote_reference_dino), "host": args.host}
    probe = remote_python(args.host, remote["python"], READONLY_REMOTE_PROBE, remote)
    require(probe.get("destination_exists") is False, "Preserve existing remote data; require a fresh destination")
    require(probe.get("repo") == remote["repo"], "Use the actual canonical production repository, not a path alias")
    shared = str(Path(remote["repo"]) / "scripts/4090_simple/finetune_carrybox.py")
    require(probe["bindings"][shared] == sha(HERE / "finetune_carrybox.py"), "Actual A100 production metadata helpers differ")
    bundle.mkdir(parents=True); code = bundle / "code"; code.mkdir()
    for name in HELPERS:
        shutil.copyfile(HERE / name, code / name)
        require(sha(HERE / name) == sha(code / name), "Helper changed while freezing transfer code")
    evidence = bundle / "evidence"; evidence.mkdir()
    collection = read(Path(args.collection_root) / "COLLECTION_READY.json")
    evidence_files = {"COLLECTION_READY.json": Path(args.collection_root) / "COLLECTION_READY.json",
        "registry.json": Path(collection["registry_path"]), "diversity.json": Path(collection["diversity_path"]),
        "REBUILD_REQUIREMENTS.json": Path(args.collection_root) / "REBUILD_REQUIREMENTS.json",
        "local_production_reader.json": Path(args.local_reader_report)}
    for name, path in evidence_files.items():
        shutil.copyfile(path, evidence / name); require(sha(path) == sha(evidence / name), "Local gate snapshot changed")
    trees = {"pack": str(Path(args.pack_root).resolve()), "dino": str(Path(args.dino_root).resolve()),
             "code": str(code.resolve()), "evidence": str(evidence.resolve())}
    value = {"contract": CONTRACT, "state": "all_local_gates_passed_explicit_upload_pending", "created_utc": now(),
        "release_id": uuid.uuid4().hex, "remote": remote, "remote_readonly_preflight": probe, "trees": trees,
        "local_inputs": {key: str(Path(getattr(args, key)).resolve()) for key in
                         ("collection_root", "pack_root", "dino_root", "local_reader_report")},
        "gate_sha256": bindings, "files": inventory_trees(trees), "finalizer_source_sha256": sha(__file__),
        "ready_for_training": False, "upload_performed": False, "training_started": False}
    verify_shas(bindings); write_new(bundle / "TRANSFER_PLAN.json", value)
    return value


REMOTE_CREATE = r'''
from pathlib import Path
import hashlib,json,sys
v=json.load(sys.stdin);p=v['plan'];root=Path(p['remote']['root'])
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert root.is_absolute() and root.parent.is_dir() and str(root)==v['allowed_root']
assert root.parent.resolve()/root.name==root
owner={'contract':p['contract'],'release_id':p['release_id'],'plan_digest':digest(p)}
if root.exists():
 assert v['resume'] and not root.is_symlink() and not (root/'RELEASE_READY.json').exists()
 assert all((root/name).is_file() and not (root/name).is_symlink() for name in ('TRANSFER_OWNER.json','TRANSFER_PLAN.json'))
 assert json.loads((root/'TRANSFER_OWNER.json').read_text())==owner
 assert json.loads((root/'TRANSFER_PLAN.json').read_text())==p
 assert all(not item.is_symlink() and item.resolve().is_relative_to(root) and (item.is_file() or item.is_dir()) for item in root.rglob('*'))
 for name in ('pack','dino','code','evidence'):
  directory=root/name
  assert directory.is_dir() and not directory.is_symlink()
  assert all(not item.is_symlink() and (item.is_file() or item.is_dir()) for item in directory.rglob('*'))
else:
 root.mkdir()
 for name,value in [('TRANSFER_OWNER.json',owner),('TRANSFER_PLAN.json',p)]:
  with (root/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 for name in ('pack','dino','code','evidence'):(root/name).mkdir()
print(json.dumps({'owned_root':str(root),'release_id':p['release_id']}))
'''


def check_release_aliases(root):
    require(root.is_dir() and not root.is_symlink() and root.resolve() == root, "Use the actual owned release directory")
    for path in root.rglob("*"):
        require(not path.is_symlink() and path.resolve().is_relative_to(root)
                and (path.is_file() or path.is_dir()), "Owned release contains a path alias or special file")


def check_remote_payload(root, plan, *, restore_mtimes=False, changed_metadata=()):
    check_release_aliases(root)
    changed = set(changed_metadata)
    actual = set()
    for name in plan["trees"]:
        directory = root / name
        require(directory.is_dir() and not directory.is_symlink(), "Copied roots may not be path aliases")
        for path in directory.rglob("*"):
            require(not path.is_symlink(), "Copied payload contains a path alias")
            if path.is_file(): actual.add(str(path.relative_to(root)))
    require(actual == set(plan["files"]), "Remote payload has missing/extra files")
    for relative, expected in plan["files"].items():
        path = owned(root, relative)
        if relative in changed: continue
        stat = path.stat()
        require(stat.st_size == expected["size_bytes"] and sha(path) == expected["sha256"], "Copied payload SHA/size differs: " + relative)
        if restore_mtimes and stat.st_mtime_ns != expected["mtime_ns"]:
            os.utime(path, ns=(stat.st_atime_ns, expected["mtime_ns"]))
        require(path.stat().st_mtime_ns == expected["mtime_ns"], "Native source/cache mtime identity differs: " + relative)


def relocate_remote_dino(root, plan):
    check_release_aliases(root)
    cache = root / "dino"; manifest = read(cache / "manifest.json")
    originals = root / "dino-original-metadata"
    relative = ["manifest.json", *[f"episodes/{index:08d}/receipt.json" for index in manifest["episode_indices"]]]
    for name in relative:
        source, target = owned(cache, name), originals / name
        require(sha(source) == plan["files"]["dino/" + name]["sha256"], "Original metadata differs before relocation")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        require(sha(target) == sha(source), "Preserved original DINO metadata changed")
    receipts = {index: read(cache / f"episodes/{index:08d}/receipt.json") for index in manifest["episode_indices"]}
    actual_root = (root / "pack/train").resolve(strict=True)
    after, changed = relocate_metadata(manifest, receipts, actual_root)
    changes = {}
    replacements = {"manifest.json": after, **{f"episodes/{index:08d}/receipt.json": value for index, value in changed.items()}}
    for name, value in replacements.items():
        path = owned(cache, name)
        temporary = root / "metadata-write-temp" / (uuid.uuid4().hex + ".json")
        write_new(temporary, value); temporary.replace(path)
        changes["dino/" + name] = {"original_sha256": sha(originals / name), "relocated_sha256": sha(path),
            "changed_fields": ["source.dataset_root", "signature"] if name == "manifest.json" else ["manifest_signature", "signature"]}
    receipt = {"contract": CONTRACT, "state": "copied_SHA_verified_DINO_metadata_explicitly_relocated",
        "release_id": plan["release_id"], "original_dataset_root": manifest["source"]["dataset_root"],
        "actual_remote_dataset_root": str(actual_root), "original_metadata_archive": str(originals),
        "metadata_changes": changes, "features_and_frames_identical_to_local_extraction": True,
        "PACK_READY_modified": False, "A100_feature_extraction_performed": False,
        "ready_for_training": False, "GPU_used": False, "training_started": False}
    destination = root / "DINO_RELOCATION_RECEIPT.json"
    if destination.exists(): require(read(destination) == receipt, "Existing relocation receipt differs")
    else: write_new(destination, receipt)
    return receipt


def real_remote_reader(root, plan):
    """Use the actual existing A100 Psi code; never the old32794-window factory."""
    os.environ.update(CUDA_VISIBLE_DEVICES="", PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1")
    sys.dont_write_bytecode = True
    remote = plan["remote"]
    verify_shas(plan["remote_readonly_preflight"]["bindings"])
    spec = importlib.util.spec_from_file_location("random100_readonly_frozen_factory", remote["factory"])
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    actual_repo = Path(base.shared.REPO).resolve(strict=True)
    require(actual_repo == Path(remote["repo"]).resolve(strict=True), "Actual production code root differs")
    snapshot = root / "code"
    require(sha(snapshot / "finetune_carrybox.py") == sha(base.shared.__file__), "Snapshot differs from actual production metadata helper")
    # Use genuine production infrastructure and pin the new portable helpers.
    sys.modules["finetune_carrybox"] = base.shared
    for name in HELPERS:
        module = Path(name).stem
        if module not in ("finetune_carrybox", "finalize_handover_random100"):
            sys.modules.pop(module, None)
    sys.path.insert(0, str(snapshot))
    import verify_handover_random100 as verifier
    verifier.REPO = actual_repo
    verifier.shared.REPO = actual_repo
    verifier.packer.legacy.REPO = actual_repo
    base.shared.REPO = actual_repo
    require((actual_repo / "VLMExpert/src/psi/data/lerobot/motion_history.py").is_file(), "Real production motion history is absent")
    config_path = root / "run_config_remote_reader.json"
    config = read(remote["run_config"])
    config["data"]["transform"]["field"]["stat_path"] = str(root / "pack/source_dataset_statistics.json")
    if config_path.exists(): require(read(config_path) == config, "Existing localized CPU configuration differs")
    else: write_new(config_path, config)
    report_path = root / "A100_PRODUCTION_READER.json"
    if report_path.exists():
        report = read(report_path)
        validate_gate_reports(read(root / "evidence/COLLECTION_READY.json"), read(root / "pack/PACK_READY.json"), report)
        verify_shas(report["source_sha256"])
    else:
        report = verifier.verify(root / "pack", root / "dino", Path(remote["reference_dino"]), config_path, report_path)
    import torch
    import psi.data.lerobot.motion_history as history_module
    require(not torch.cuda.is_initialized(), "Actual CPU reader initialized CUDA")
    require(Path(history_module.__file__).resolve().is_relative_to(actual_repo), "Reader imported a nonproduction history module")
    validate_gate_reports(read(root / "evidence/COLLECTION_READY.json"), read(root / "pack/PACK_READY.json"), report)
    return {"path": str(report_path), "sha256": sha(report_path), "actual_repo": str(actual_repo),
        "actual_finetune_helper": str(Path(base.shared.__file__).resolve()),
        "actual_motion_history_module": str(Path(history_module.__file__).resolve()),
        "CUDA_initialized": False, "model_constructed": False}


def remote_finish(root):
    root = Path(root).resolve(strict=True); plan = read(root / "TRANSFER_PLAN.json")
    require(plan["contract"] == CONTRACT and str(root) == REMOTE_ROOT == plan["remote"]["root"], "Unexpected owned release")
    require(read(root / "TRANSFER_OWNER.json") == {"contract": CONTRACT, "release_id": plan["release_id"], "plan_digest": digest(plan)},
            "Owned transfer identity differs")
    require(not (root / "RELEASE_READY.json").exists(), "Preserve existing final release readiness")
    require(sha(__file__) == plan["finalizer_source_sha256"], "Uploaded finalizer differs")
    check_remote_payload(root, plan, restore_mtimes=True)
    relocation = relocate_remote_dino(root, plan)
    check_remote_payload(root, plan, changed_metadata=relocation["metadata_changes"])
    try:
        reader = real_remote_reader(root, plan)
        check_remote_payload(root, plan, changed_metadata=relocation["metadata_changes"])
        for relative, values in relocation["metadata_changes"].items():
            require(sha(owned(root, relative)) == values["relocated_sha256"], "Relocated metadata changed during actual reading")
        verify_shas(plan["remote_readonly_preflight"]["bindings"])
        report = {"contract": CONTRACT, "state": "all100_local_and_A100_quality_DINO_production_gates_passed",
            "created_utc": now(), "release_id": plan["release_id"], "episodes": 100,
            "remote_root": str(root), "ready_for_new_training_review": True,
            "A100_actual_reader": reader, "transfer_plan_sha256": sha(root / "TRANSFER_PLAN.json"),
            "source_pack_ready_sha256": sha(root / "pack/PACK_READY.json"),
            "relocation_receipt_sha256": sha(root / "DINO_RELOCATION_RECEIPT.json"),
            "local_production_receipt_sha256": sha(root / "evidence/local_production_reader.json"),
            "features_and_frames_byte_identical_to_local_extraction": True, "home_alias_created": False,
            "old_data_or_VAL_modified": False, "PACK_READY_modified": False,
            "GPU_used": False, "model_constructed": False, "training_started": False,
            "independent_physical_audit_rerun_on_A100": False,
            "scope": "Transferred immutable strict local collection/pack and checked real A100 RGB/state/action/history/collator; no new simulation, policy score or training launch"}
        write_new(root / "RELEASE_READY.json", report)
        return report
    except Exception as error:
        write_new(root / ("REMOTE_VERIFICATION_FAILED-" + uuid.uuid4().hex + ".json"),
                  {"error": repr(error), "ready_for_training": False, "release_id": plan["release_id"]})
        raise


def upload(bundle_root, *, resume=False):
    bundle = Path(bundle_root).resolve(strict=True); plan = read(bundle / "TRANSFER_PLAN.json")
    require(plan.get("contract") == CONTRACT and plan.get("finalizer_source_sha256") == sha(__file__), "Transfer plan/finalizer identity differs")
    require(plan["remote"]["root"] == REMOTE_ROOT, "Refuse another or existing-data destination")
    local_gates(**plan["local_inputs"])
    verify_shas(plan["gate_sha256"])
    require(inventory_trees(plan["trees"]) == plan["files"], "Frozen local payload changed before upload")
    remote = plan["remote"]
    remote_python(remote["host"], remote["python"], REMOTE_CREATE,
                  {"plan": plan, "allowed_root": REMOTE_ROOT, "resume": resume})
    transport = shlex.join(SSH)
    for name, source in plan["trees"].items():
        subprocess.run(["rsync", "-a", "--checksum", "--protect-args", "-e", transport,
            str(Path(source)) + "/", f"{remote['host']}:{REMOTE_ROOT}/{name}/"], check=True)
    command = shlex.join([remote["python"], "-B", REMOTE_ROOT + "/code/finalize_handover_random100.py", "_remote-finish", "--root", REMOTE_ROOT])
    subprocess.run([*SSH, remote["host"], command], check=True)
    for name in ("RELEASE_READY.json", "DINO_RELOCATION_RECEIPT.json", "A100_PRODUCTION_READER.json"):
        destination = bundle / ("A100_" + name)
        require(not destination.exists(), "Preserve existing returned transfer evidence")
        subprocess.run(["rsync", "-a", "--checksum", "--protect-args", "-e", transport,
            f"{remote['host']}:{REMOTE_ROOT}/{name}", str(destination)], check=True)
    require(inventory_trees(plan["trees"]) == plan["files"], "Immutable local payload changed during transfer")
    final = read(bundle / "A100_RELEASE_READY.json")
    require(final.get("release_id") == plan["release_id"] and final.get("ready_for_new_training_review") is True
            and final.get("source_pack_ready_sha256") == sha(Path(plan["trees"]["pack"]) / "PACK_READY.json"),
            "Returned remote ready receipt does not bind the uploaded source")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__); modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("prepare")
    for name in ("collection-root", "pack-root", "dino-root", "local-reader-report", "bundle-root"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--host", default="115.191.40.142")
    for name, value in (("remote-root", REMOTE_ROOT), ("remote-python", REMOTE_PYTHON), ("remote-repo", REMOTE_REPO),
                        ("remote-factory", REMOTE_FACTORY), ("remote-run-config", REMOTE_CONFIG), ("remote-reference-dino", REMOTE_REFERENCE)):
        p.add_argument("--" + name, type=Path, default=Path(value))
    p = modes.add_parser("upload"); p.add_argument("--bundle-root", required=True, type=Path); p.add_argument("--resume", action="store_true")
    p = modes.add_parser("_remote-finish"); p.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.mode == "prepare": report = prepare(args)
    elif args.mode == "upload": report = upload(args.bundle_root, resume=args.resume)
    else: report = remote_finish(args.root)
    print(json.dumps({key: report[key] for key in ("contract", "state", "release_id")}, ensure_ascii=False))


if __name__ == "__main__": main()
