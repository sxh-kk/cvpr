"""Only immutable, computation-identical accepted shards may be reused."""
from copy import deepcopy
import json
import os

import numpy as np
import pytest

from MotionExpert.dataset.window_cache import ARRAY_SPECS, CONTRACT, atomic_json, digest, file_sha256, stamp
from MotionExpert.dataset.vision_cache import CONTRACT as VISION_CONTRACT, storage_digest
from MotionExpert.reuse_window_cache import check_archive_info, computation_ast, previous_visual_source, reuse_episode, visual_reuse_proof


def test_publication_change_does_not_hide_fk_changes():
    base = '''class SMPLHParameterArchive:
    def _check_acceptance(self):
        return "old"
    def decode(self, values):
        return values + 1
'''
    assert computation_ast(base) == computation_ast(base.replace('"old"', '"new"'))
    assert computation_ast(base) != computation_ast(base.replace('values + 1', 'values + 2'))
    with pytest.raises(ValueError, match="one archive acceptance"):
        computation_ast(base.replace('_check_acceptance', '_wrong_method'))


def fixture_shard(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    episode = old / "episodes/00000007"
    episode.mkdir(parents=True)
    record = {"episode_index": 7, "split": "train", "source_key": "egosuite:unchanged", "length": 91,
              "data_path": "episodes/00000007/motion.parquet"}
    vision = {"manifest_sha256": "old", "encoder_sha256": "unchanged"}
    arrays = {}
    for key, (dtype, tail) in ARRAY_SPECS.items():
        value = np.zeros((2, *tail), dtype=dtype)
        if key == "start_frames":
            value[:] = [0, 30]
        path = episode / f"{key}.npy"
        np.save(path, value, allow_pickle=False)
        size, modified = stamp(path)
        arrays[key] = {"path": str(path.relative_to(old)), "dtype": dtype,
                       "shape": list(value.shape), "size_bytes": size, "mtime_ns": modified,
                       "sha256": file_sha256(path)}
    receipt = {"contract": CONTRACT, "state": "complete", "episode_index": 7,
               "source_record_sha256": digest(record), "split": "train", "windows": 2,
               "arrays": arrays, "start_frames_sha256": digest([0, 30]),
               "binding_sha256": "old_binding", "template_metadata": {"vision_profile": vision}}
    path = episode / "receipt.json"
    atomic_json(path, receipt)
    descriptor = {"receipt_path": str(path.relative_to(old)), "receipt_sha256": file_sha256(path),
                  "windows": 2}
    previous_vision, current_vision = tmp_path / "old-vision", tmp_path / "new-vision"
    prefix = "episodes/00000007"
    (previous_vision / prefix).mkdir(parents=True)
    (current_vision / prefix).mkdir(parents=True)
    visual_identities = {}
    for name in ("features.npy", "join.npz"):
        path = previous_vision / prefix / name
        if name == "features.npy":
            np.save(path, np.zeros((record["length"], 5, 1024), dtype=np.float32))
        else:
            np.savez(path, frame_index=np.arange(record["length"]), valid=np.ones(record["length"], bool))
        os.link(path, current_vision / prefix / name)
        size, modified = stamp(path)
        visual_identities[name] = {"size_bytes": size, "mtime_ns": modified}
    visual_manifest = {"contract": VISION_CONTRACT, "state": "complete", "scope": "all",
                       "validity_equal_source": True, "encoder": {"weights": "same"}}
    atomic_json(previous_vision / "manifest.json", visual_manifest)
    visual_receipt = {"contract": VISION_CONTRACT, "state": "complete", "episode_index": 7,
                      "source_key": record["source_key"], "source_record_sha256": storage_digest(record),
                      "frames": record["length"], "validity_equal_source": True, "valid_count": record["length"],
                      "encoder": visual_manifest["encoder"], "output_identities": visual_identities}
    old_visual_receipt = previous_vision / prefix / "receipt.json"
    atomic_json(old_visual_receipt, visual_receipt)
    visual_receipt["quality_revision_reuse"] = {
        "parent_receipt_sha256": file_sha256(old_visual_receipt),
        "parent_manifest_sha256": file_sha256(previous_vision / "manifest.json"),
        "join_all_native_rows_checked": True, "join_evidence": "changed_motion_all_native_rows_compared",
        "payload_reuse": "same_device_inode_hardlink"}
    atomic_json(current_vision / prefix / "receipt.json", visual_receipt)
    previous_archive, current_archive = tmp_path / "old-archive", tmp_path / "new-archive"
    (previous_archive / prefix).mkdir(parents=True)
    (current_archive / prefix).mkdir(parents=True)
    (previous_archive / record["data_path"]).write_bytes(b"already accepted frozen motion")
    os.link(previous_archive / record["data_path"], current_archive / record["data_path"])
    vision_sources = {"previous_root": previous_vision, "current_root": current_vision,
                      "previous_manifest": visual_manifest, "current_manifest": visual_manifest,
                      "previous_manifest_sha256": file_sha256(previous_vision / "manifest.json"),
                      "previous_archive_root": previous_archive, "current_archive_root": current_archive}
    return old, new, descriptor, record, dict(vision, manifest_sha256="new"), vision_sources


def test_raw_bytes_share_inode_and_receipts_remain_independent(tmp_path):
    old, new, descriptor, record, vision, vision_sources = fixture_shard(tmp_path)
    before = (old / descriptor["receipt_path"]).read_bytes()
    reuse_episode(old, new, descriptor, record, {"new": "binding"}, vision, vision_sources=vision_sources)
    receipt = json.loads((new / descriptor["receipt_path"]).read_text())
    assert receipt["binding_sha256"] == digest({"new": "binding"})
    assert receipt["template_metadata"]["vision_profile"] == vision
    assert receipt["reuse"]["visual_proof"]["payload_identity"] == "same_device_inode_size_mtime"
    assert receipt["reuse"]["visual_proof"]["feature_payload_read"] is False
    assert (old / descriptor["receipt_path"]).read_bytes() == before
    for key in ARRAY_SPECS:
        source = old / f"episodes/00000007/{key}.npy"
        target = new / receipt["arrays"][key]["path"]
        assert os.path.samefile(source, target)
        assert file_sha256(target) == receipt["arrays"][key]["sha256"]
    with pytest.raises(FileExistsError):
        reuse_episode(old, new, descriptor, record, {}, vision, vision_sources=vision_sources)


@pytest.mark.parametrize("previous_revision", [None, {"format": "prior-quality", "policy": {"distance": 1.}}])
def test_successive_quality_revisions_reuse_identical_shards(tmp_path, previous_revision):
    previous = {"fps": 30, "model": {"model_sha256": "same-model"}}
    if previous_revision is not None:
        previous["quality_revision"] = deepcopy(previous_revision)
    current = deepcopy(previous)
    current["quality_revision"] = {"format": "next-quality", "parent_revision": deepcopy(previous_revision)}
    frozen_previous, frozen_current = deepcopy(previous), deepcopy(current)
    check_archive_info(previous, current)
    assert previous == frozen_previous and current == frozen_current
    old, new, descriptor, record, vision, sources = fixture_shard(tmp_path)
    reuse_episode(old, new, descriptor, record, {"revision": "new"}, vision, vision_sources=sources)
    receipt = json.loads((new / descriptor["receipt_path"]).read_text())
    assert receipt["reuse"]["unchanged_complete_record"] is True
    for name in ARRAY_SPECS:
        assert os.path.samefile(old / f"episodes/00000007/{name}.npy", new / receipt["arrays"][name]["path"])


@pytest.mark.parametrize("mutation", ["fps", "model", "missing_parent", "different_parent"])
def test_revision_compatibility_rejects_other_info_or_wrong_lineage(mutation):
    previous = {"fps": 30, "model": {"model_sha256": "same-model"},
                "quality_revision": {"format": "prior-quality", "policy": {"distance": 1.}}}
    current = deepcopy(previous)
    current["quality_revision"] = {"format": "next-quality", "parent_revision": deepcopy(previous["quality_revision"])}
    if mutation == "fps":
        current["fps"] = 15
    elif mutation == "model":
        current["model"]["model_sha256"] = "different-model"
    elif mutation == "missing_parent":
        current["quality_revision"].pop("parent_revision")
    else:
        current["quality_revision"]["parent_revision"]["policy"]["distance"] = .5
    with pytest.raises(ValueError, match="archive information|parent revision declaration"):
        check_archive_info(previous, current)


def test_reject_changed_encoder_and_changed_array_identity(tmp_path):
    old, new, descriptor, record, vision, vision_sources = fixture_shard(tmp_path)
    with pytest.raises(ValueError, match="encoder"):
        reuse_episode(old, new, descriptor, record, {}, dict(vision, encoder_sha256="changed"), vision_sources=vision_sources)
    path = old / "episodes/00000007/motion.npy"
    original = path.stat()
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns + 1_000_000_000))
    with pytest.raises(ValueError, match="array identity"):
        reuse_episode(old, new, descriptor, record, {}, vision, vision_sources=vision_sources)
    assert not (new / "episodes/00000007").exists()


def test_reject_changed_source_record(tmp_path):
    old, new, descriptor, record, vision, vision_sources = fixture_shard(tmp_path)
    with pytest.raises(ValueError, match="unchanged source"):
        reuse_episode(old, new, descriptor, dict(record, changed_quality=True), {}, vision, vision_sources=vision_sources)


def test_previous_visual_manifest_must_match_the_accepted_window_cache(tmp_path):
    *_, sources = fixture_shard(tmp_path)
    manifest = {"vision_profile_root": str(sources["previous_root"]),
                "binding": {"vision_manifest_sha256": sources["previous_manifest_sha256"]}}
    root, saved, sha = previous_visual_source(manifest)
    assert root == sources["previous_root"]
    assert saved == sources["previous_manifest"]
    assert sha == sources["previous_manifest_sha256"]
    with pytest.raises(ValueError, match="visual manifest identity"):
        previous_visual_source(dict(manifest, binding={"vision_manifest_sha256": "wrong"}))


@pytest.mark.parametrize("payload", ["features.npy", "join.npz"])
def test_same_values_and_encoder_without_hardlink_proof_are_rejected(tmp_path, payload):
    _, _, _, record, _, sources = fixture_shard(tmp_path)
    target = sources["current_root"] / "episodes/00000007" / payload
    stat, contents = target.stat(), target.read_bytes()
    target.unlink()
    target.write_bytes(contents)
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="not the same frozen hardlinked payload"):
        visual_reuse_proof(record, **sources)


@pytest.mark.parametrize("mutation", ["source", "encoder", "parent", "join_check", "identity"])
def test_rebound_visual_receipt_cannot_change_source_or_proof(tmp_path, mutation):
    _, _, _, record, _, sources = fixture_shard(tmp_path)
    path = sources["current_root"] / "episodes/00000007/receipt.json"
    receipt = json.loads(path.read_text())
    if mutation == "source":
        receipt["source_record_sha256"] = "different_source"
    elif mutation == "encoder":
        receipt["encoder"] = {"weights": "other"}
    elif mutation == "parent":
        receipt["quality_revision_reuse"]["parent_receipt_sha256"] = "other"
    elif mutation == "join_check":
        receipt["quality_revision_reuse"]["join_all_native_rows_checked"] = False
    else:
        receipt["output_identities"]["join.npz"]["size_bytes"] += 1
    atomic_json(path, receipt)
    with pytest.raises(ValueError, match="Visual|visual"):
        visual_reuse_proof(record, **sources)


def test_immutable_motion_and_visual_can_inherit_the_existing_join_acceptance(tmp_path):
    _, _, _, record, _, sources = fixture_shard(tmp_path)
    path = sources["current_root"] / "episodes/00000007/receipt.json"
    receipt = json.loads(path.read_text())
    receipt["quality_revision_reuse"].update(join_all_native_rows_checked=False,
                                             join_evidence="inherited_immutable_motion_and_visual")
    atomic_json(path, receipt)
    proof = visual_reuse_proof(record, **sources)
    assert proof["join_evidence"] == "inherited_immutable_motion_and_visual"
    target = sources["current_archive_root"] / record["data_path"]
    original, contents = target.stat(), target.read_bytes()
    target.unlink()
    target.write_bytes(contents)
    os.utime(target, ns=(original.st_atime_ns, original.st_mtime_ns))
    with pytest.raises(ValueError, match="Unchanged source motion"):
        visual_reuse_proof(record, **sources)
