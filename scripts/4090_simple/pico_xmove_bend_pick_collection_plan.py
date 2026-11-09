"""Freeze the official100 XMoveBendPick TRAIN initial layouts without using a simulator.

SIMPLE SpatialDR restores target/distractors by their saved asset UID. Official metadata's layout poses can reflect
the completed demonstration; its dr_state_dict.spatial retains the reset poses.
Only those four actor poses are restored. Saved layout lights remain intact:
the source omitted their DR state. Source records, DR, geometry,
materials, robot spawn and camera definitions are preserved for provenance.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import uuid

CONTRACT = "cvpr-pico-xmove-bend-pick-official-train100-collection-plan-v1"
TASK_ID = "cvpr/G1SonicXMoveBendPickPico-v0"
SOURCE_TASK_ID = "simple/G1WholebodyXMoveBendPickTeleop-v0"
TASK_UID = "g1_wholebody_xmove_bend_pick_teleop"
SOURCE_SHA256 = "4dc984a6819cad361b35a6b811f57d9afad0118f354db7137b497276fa5f0ed0"
EPISODE_INDICES = list(range(100))
REPO = Path(__file__).resolve().parents[2]
POSE_ROLES = ("target", "distractor_0", "distractor_1", "distractor_2")
DERIVATION = {
    "pose_source": "environment_config.dr_state_dict.spatial",
    "target_and_distractor_key": "layout.actors[role].asset.uid; never role ordinal",
    "modified_fields": ["layout.actors." + role + ".pose" for role in POSE_ROLES],
    "lighting_source": "saved layout.lights preserved; source DR lighting empty, explicit private task restoration",
    "all_other_config_values_preserved": True,
    "dr_state_dict_preserved": True,
    "resource_paths_rebound": False,
    "official_source_split": "train",
    "teacher_actions_or_states_read": False,
    "physics_or_GPU_used": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON object key: " + key)
        result[key] = value
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=_unique_object, parse_constant=lambda value:
        (_ for _ in ()).throw(ValueError("Nonfinite JSON value: " + value)))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def semantic_sha(value):
    return digest(canonical(value).encode("utf-8"))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def regular(path):
    path = Path(path).absolute()
    require(not any(part.is_symlink() for part in (path, *path.parents)), "Plan/source symlinks are not accepted")
    require(path.is_file(), "Missing plan/source file: " + str(path))
    return path


def owned(root, relative):
    require(isinstance(relative, str), "Plan file paths must be relative strings")
    name = Path(relative)
    require(name.parts and not name.is_absolute() and ".." not in name.parts, "Unsafe plan file path")
    path = regular(root / name)
    require(path.resolve().is_relative_to(root.resolve()), "Plan file escapes the frozen directory")
    return path


def pose(value, label):
    require(isinstance(value, dict) and set(value) == {"position", "quaternion"}, "Incomplete pose: " + label)
    for field, count in (("position", 3), ("quaternion", 4)):
        items = value[field]
        require(isinstance(items, list) and len(items) == count
            and all(type(item) in (int, float) and math.isfinite(item) for item in items), "Invalid pose vector: " + label)
    require(any(value["quaternion"]), "Zero quaternion: " + label)


def derive_reset(config):
    """Return a separate reset config plus UID-to-spatial provenance mappings."""
    require(isinstance(config, dict) and config.get("uid") == TASK_UID
        and config.get("metadata", {}).get("split") == "train"
        and config.get("robot_cfg", {}).get("uid") == "g1_sonic", "Require the official XMoveBendPick G1 TRAIN task")
    dr = config.get("dr_state_dict", {})
    actors = config.get("layout", {}).get("actors", {})
    spatial = dr.get("spatial", {})
    require(isinstance(spatial, dict) and set(actors) == {"table", *POSE_ROLES}, "Official XMoveBendPick actor/DR inventory differs")
    pose(spatial.get("g1_sonic"), "robot/g1_sonic")
    require(isinstance(dr.get("distractors"), dict) and len(dr["distractors"]) == 3, "Require the three original distractor identities")
    result, mapping, used_keys, distractor_uids = deepcopy(config), [], set(), set()
    for role in POSE_ROLES:
        actor = actors[role]
        asset = actor.get("asset", {})
        uid, namespace = asset.get("uid"), asset.get("res_id")
        require(isinstance(uid, str) and uid and actor.get("uid") == uid and isinstance(namespace, str) and namespace,
                "Actor and asset UID/namespace differ: " + role)
        selected = dr.get(role) if role in ("target", "container") else dr["distractors"].get(uid)
        require(isinstance(selected, dict) and selected.get("uid") == uid and selected.get("res_id") == namespace,
                "DR-selected asset does not match actor UID: " + role)
        key = "container_" + uid if role == "container" else uid
        require(key not in used_keys and key in spatial, "Missing or ambiguous initial DR spatial key: " + key)
        pose(actor.get("pose"), "source layout/" + role)
        pose(spatial[key], "initial spatial/" + key)
        result["layout"]["actors"][role]["pose"] = deepcopy(spatial[key])
        mapping.append({"layout_role": role, "asset_uid": uid, "asset_namespace": namespace,
            "initial_dr_spatial_key": key, "original_layout_pose": deepcopy(actor["pose"]),
            "initial_dr_spatial_pose": deepcopy(spatial[key]),
            "pose_changed": canonical(actor["pose"]) != canonical(spatial[key])})
        used_keys.add(key)
        if role.startswith("distractor_"):
            distractor_uids.add(uid)
    require(distractor_uids == set(dr["distractors"]), "Distractor roles do not cover the original selected assets")
    require(set(spatial) == used_keys | {"g1_sonic"}, "Unmapped original DR spatial identity")
    lights = config["layout"].get("lights")
    require(dr.get("lighting") == {} and isinstance(lights, list) and lights
        and all(isinstance(light, dict) and isinstance(light.get("uid"), str) and light["uid"] for light in lights)
        and len({light["uid"] for light in lights}) == len(lights), "Require source-omitted DR lighting and unique saved layout lights")
    # Reverse every explicit derivation to prove no other config field changed.
    reversed_config = deepcopy(result)
    for item in mapping:
        reversed_config["layout"]["actors"][item["layout_role"]]["pose"] = deepcopy(item["original_layout_pose"])
    reversed_config["layout"]["lights"] = deepcopy(config["layout"]["lights"])
    require(canonical(reversed_config) == canonical(config), "Derivation changed more than original actor poses")
    return result, mapping


def source_rows(payload):
    require(digest(payload) == SOURCE_SHA256, "Source fingerprint is not the frozen official100 XMoveBendPick TRAIN archive")
    records = [parse_json(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    require(len(records) == 100 and all(isinstance(row, dict) and type(row.get("episode_index")) is int for row in records)
        and sorted(row["episode_index"] for row in records) == EPISODE_INDICES, "Require exactly100 unique official episodes0..99")
    results, dr_hashes = [], set()
    for source_row, record in enumerate(records):
        original_text = record.get("environment_config")
        require(isinstance(original_text, str) and original_text, "Preserve the original environment_config JSON string bytes")
        config = parse_json(original_text)
        derived, mapping = derive_reset(config)
        index = record["episode_index"]
        dr_sha = semantic_sha(config["dr_state_dict"])
        require(dr_sha not in dr_hashes, "Official plan contains a repeated DR initialization")
        dr_hashes.add(dr_sha)
        original = original_text.encode("utf-8")
        reset = json_bytes(derived)
        results.append({"row": {"official_episode_index": index, "source_row_number": source_row,
            "original_config_path": f"original/episode_{index:03d}.json", "original_config_sha256": digest(original),
            "reset_config_path": f"reset/episode_{index:03d}.json", "reset_config_sha256": digest(reset),
            "dr_state_sha256": dr_sha, "spatial_state_sha256": semantic_sha(config["dr_state_dict"]["spatial"]),
            "pose_mapping": mapping}, "original": original, "reset": reset})
    return sorted(results, key=lambda item: item["row"]["official_episode_index"])


def build_plan(source_jsonl, output_dir):
    """Publish a fresh outside-repository manifest after validating all100 rows."""
    source = regular(source_jsonl)
    payload = source.read_bytes()
    prepared = source_rows(payload)
    output = Path(output_dir).absolute()
    require(not output.exists() and not output.is_symlink()
        and not output.resolve().is_relative_to(REPO.resolve()), "Use a fresh collection-plan directory outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not any(part.is_symlink() for part in (output.parent, *output.parent.parents)), "Collection-plan directory may not traverse symlinks")
    staging = output.with_name("." + output.name + "." + uuid.uuid4().hex + ".partial")
    staging.mkdir(mode=0o700)
    (staging / "source_episodes.jsonl").write_bytes(payload)
    for item in prepared:
        for key in ("original", "reset"):
            path = staging / item["row"][key + "_config_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(item[key])
    manifest = {"contract": CONTRACT, "task_id": TASK_ID, "source_official_task_id": SOURCE_TASK_ID, "source": {"original_path": str(source),
        "frozen_path": "source_episodes.jsonl", "sha256": SOURCE_SHA256, "size_bytes": len(payload)},
        "episode_count": 100, "official_episode_indices": EPISODE_INDICES, "unique_dr_states": 100,
        "derivation": DERIVATION, "episodes": [item["row"] for item in prepared]}
    (staging / "manifest.json").write_bytes(json_bytes(manifest))
    require(source.read_bytes() == payload, "Official source changed while freezing collection plan")
    # Do not overwrite a concurrently created directory, including an empty one.
    require(not output.exists() and not output.is_symlink(), "Collection-plan output appeared while freezing")
    staging.rename(output)
    return output / "manifest.json"


def load_plan(path):
    """Return100 independently validated rows with absolute paths and plan SHA.

    External original metadata may have been moved; the exact source snapshot
    inside the plan must still have the fixed official archive fingerprint.
    """
    path = regular(path)
    payload = path.read_bytes()
    manifest = parse_json(payload.decode("utf-8"))
    require(isinstance(manifest, dict) and set(manifest) == {"contract", "task_id", "source_official_task_id", "source", "episode_count",
        "official_episode_indices", "unique_dr_states", "derivation", "episodes"}
        and manifest.get("contract") == CONTRACT and manifest.get("task_id") == TASK_ID and manifest.get("source_official_task_id") == SOURCE_TASK_ID
        and type(manifest.get("episode_count")) is int and manifest["episode_count"] == 100
        and manifest.get("official_episode_indices") == EPISODE_INDICES and manifest.get("unique_dr_states") == 100
        and canonical(manifest.get("derivation")) == canonical(DERIVATION), "Wrong frozen XMoveBendPick100 collection-plan contract")
    source = manifest["source"]
    require(isinstance(source, dict) and set(source) == {"original_path", "frozen_path", "sha256", "size_bytes"}
        and isinstance(source.get("original_path"), str) and Path(source["original_path"]).is_absolute()
        and source.get("frozen_path") == "source_episodes.jsonl" and source.get("sha256") == SOURCE_SHA256,
        "Wrong official source provenance/fingerprint")
    frozen = owned(path.parent, source["frozen_path"]).read_bytes()
    require(type(source["size_bytes"]) is int and len(frozen) == source["size_bytes"], "Frozen source size changed")
    expected = source_rows(frozen)
    require(canonical(manifest["episodes"]) == canonical([item["row"] for item in expected]),
            "Frozen plan rows/mapping differ from independent official initial-state derivation")
    results, plan_sha = [], digest(payload)
    for item in expected:
        row = deepcopy(item["row"])
        for key in ("original", "reset"):
            actual = owned(path.parent, row[key + "_config_path"])
            require(actual.read_bytes() == item[key], "Frozen " + key + " config changed: " + str(row["official_episode_index"]))
            row[key + "_config_path"] = str(actual)
        row.update(plan_sha256=plan_sha, manifest_path=str(path), task_id=TASK_ID)
        results.append(row)
    require(path.read_bytes() == payload, "Collection-plan manifest changed during validation")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    build = modes.add_parser("build")
    build.add_argument("--source-jsonl", type=Path, required=True)
    build.add_argument("--output-dir", type=Path, required=True)
    check = modes.add_parser("check")
    check.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    path = build_plan(args.source_jsonl, args.output_dir) if args.mode == "build" else args.plan
    rows = load_plan(path)
    print(json.dumps({"manifest": str(Path(path).absolute()), "plan_sha256": rows[0]["plan_sha256"],
        "official_episodes": len(rows), "source_sha256": SOURCE_SHA256, "physics_or_GPU_used": False}))


if __name__ == "__main__":
    main()
