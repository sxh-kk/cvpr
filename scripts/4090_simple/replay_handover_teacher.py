"""Bounded original-TRAIN teacher diagnostic/capture with local SONIC v1.1.

This is never an accepted new dataset or a learned-policy success rate. Source
commands are already v1.1 and are not re-encoded. The independent 50/30Hz runs
share a documented cold start and one second of measured/live-pose warmup;
neither reconstructs the original C++ controller history or warmup primer.
Module import does not load SIMPLE, MuJoCo, ONNX or rendering packages. Capture
validation lazily imports only SIMPLE's dependency-free ActionCmd class.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import traceback
from types import FunctionType, MethodType

import numpy as np

from prepare_handover_eval import TASK, VAL_IDS, remap_paths, pose, vector


HAND_NAMES = tuple(f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
                   for finger, n in (("thumb", 3), ("index", 2), ("middle", 2)) for j in range(n))
MODEL_HAND_NAMES = tuple(f"{side}_hand_{finger}_{j}_joint" for side in ("left", "right")
                   for finger, n in (("thumb", 3), ("middle", 2), ("index", 2)) for j in range(n))
CAPTURE_CONTRACT = "cvpr-handover-teacher-rtx-capture-v1"
RENDERER_POSE_CONTRACT = "cvpr-handover-renderer-pose-evidence-v1"
RENDERER_LINKS = ("pelvis", "torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
CANDIDATE_SPEC_KEYS = ("ancestor", "variant", "physics_seed", "material_seed", "generation_seed", "initial_object_deltas")
CALIBRATION_HELPER = Path(__file__).resolve().parents[2]/"VLMExpert/third_party/SIMPLE/src/simple/sensors/head_camera_mount.py"
PADDED_WARMUP_MODE = "live_pose_sonic_v1_1_repeat_first_snapshot"
PADDED_WARMUP_KIND = "live_initial_pose_repeat_first_snapshot_sonic_v1_1"
STARTUP_HISTORY_CONTRACT = "cvpr-sonic-v11-startup-repeat-first-snapshot-v1"


def json_default(value):
    """Convert actual NumPy control/state scalars without stringifying values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported diagnostic JSON value: {type(value).__name__}")


def json_text(value, *, indent=None):
    return json.dumps(value, default=json_default, allow_nan=False, ensure_ascii=False, indent=indent)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json_text(value, indent=2)+"\n")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calibration_source_identities():
    """Load only the NumPy calibration helper, never a sensor/Isaac package."""
    spec = importlib.util.spec_from_file_location("handover_capture_camera_calibration",CALIBRATION_HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return dict(helper.SOURCE_IDENTITIES)


def bound_calibration_assets(data_root, expected):
    """Hash the actual resources selected by SIMPLE_DATA_DIR, including symlinks.

    The local resource tree intentionally symlinks its robot asset directory;
    validity therefore follows the actual bytes rather than forbidding symlinks.
    """
    root = Path(data_root).resolve(strict=True)
    if not root.is_dir() or not isinstance(expected,dict) or not expected:
        raise ValueError("Actual calibration data-root and source identities are required")
    observed = {}
    for relative, identity in expected.items():
        path = Path(relative)
        if (not isinstance(relative,str) or path.is_absolute() or ".." in path.parts
                or not isinstance(identity,str) or len(identity)!=64
                or any(c not in "0123456789abcdef" for c in identity)):
            raise ValueError("Invalid relative calibration source identity")
        actual = root/path
        if not actual.is_file():
            raise ValueError("Missing actual calibration source asset: " + relative)
        actual_sha = file_sha256(actual)
        if actual_sha != identity:
            raise ValueError("Actual calibration source asset SHA mismatch: " + relative)
        observed[relative] = actual_sha
    return root,observed


def bound_sources(root, required, *, receipt_name="source0_transfer_audit.json"):
    root = Path(root).resolve(strict=True)
    receipt_path = root / receipt_name
    receipt = json.loads(receipt_path.read_text())
    if receipt_name == "source_transfer_audit.json":
        accepted = (receipt.get("status") == "all24_source_files_sha256_verified"
            and receipt.get("schema") == "cvpr-handover-train-source-transfer-v1")
    else:
        accepted = receipt.get("status") == "source0_small_data_and_reset_sha256_verified"
    if not accepted:
        raise ValueError("TRAIN source transfer receipt is not accepted")
    entries = list(receipt["files"])
    if receipt_name == "source_transfer_audit.json":
        entries.append(receipt["source_release_manifest_extract"])
    if len({entry["path"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate source identity entries")
    expected = {entry["path"]: entry for entry in entries}
    result = {str(receipt_path): file_sha256(receipt_path)}
    for relative in required:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("Unsafe source path")
        path = root / relative_path
        if path.is_symlink() or not path.resolve().is_relative_to(root) or relative not in expected:
            raise ValueError("Source file is not locally identity-bound")
        if path.stat().st_size != expected[relative]["bytes"] or file_sha256(path) != expected[relative]["sha256"]:
            raise ValueError(f"Source SHA/size mismatch: {relative}")
        result[str(path)] = expected[relative]["sha256"]
    return result


def repack_commands(tokens, hands, source_hand_names, replay_action78):
    tokens, hands = np.asarray(tokens), np.asarray(hands)
    if (tokens.ndim != 2 or tokens.shape[1] != 64 or hands.shape != (len(tokens), 14)
            or len(source_hand_names) != 14 or set(source_hand_names) != set(HAND_NAMES)
            or len(set(source_hand_names)) != 14):
        raise ValueError("Expected body64 and fourteen unique named hand channels")
    perm = [source_hand_names.index(name) for name in HAND_NAMES]
    commands = np.concatenate([tokens, hands[:, perm]], axis=1).astype(np.float32)
    replay = np.asarray(replay_action78, dtype=np.float32)
    if (not np.isfinite(commands).all() or commands.shape != replay.shape
            or not np.array_equal(commands, replay)):
        raise ValueError("Named source targets differ from actual replay_action78")
    return commands


def command_source_index(tick, rate, length):
    if type(tick) is not int or tick < 0 or type(length) is not int or length <= 0 or rate not in (30, 50):
        raise ValueError("Invalid 50Hz tick, source length or command rate")
    if tick >= length:
        raise ValueError("Do not execute unrecorded source tail rows")
    if rate == 50:
        return tick
    # Exact training ceil(k*50/30), consumed with floor(t*30) at true50Hz.
    index30 = tick * 30 // 50
    return (index30 * 50 + 29) // 30


def load_source(snapshot_root, *, source_episode=0):
    import pyarrow.parquet as pq

    if type(source_episode) is not int or source_episode < 0 or source_episode in VAL_IDS:
        raise ValueError("Only explicit original TRAIN ancestors; heldout sources are forbidden")
    root = Path(snapshot_root).resolve(strict=True)
    train = "training/Handover-sonic11-train"
    shared = ["training/selection.json", f"{train}/meta/info.json", f"{train}/meta/episodes.jsonl", "replay/meta/info.json"]
    generic = (root / "source_transfer_audit.json").is_file()
    source_record = None
    if generic:
        receipt_name = "source_transfer_audit.json"
        transfer = json.loads((root / receipt_name).read_text())
        matches = [item for item in transfer["episodes"] if item["source_episode_index"] == source_episode]
        if len(matches) != 1 or matches[0].get("split") != "train":
            raise ValueError("Source is not one of the 24 explicit original TRAIN ancestors")
        source_record = matches[0]
        train_episode = source_record["train_episode_index"]
        roles = {role:item["path"] for role,item in source_record["required_files"].items()}
        required = shared + list(roles.values()) + [transfer["source_release_manifest_extract"]["path"]]
    else:
        if source_episode != 0:
            raise ValueError("Nonzero TRAIN ancestor requires the complete 24-source transfer receipt")
        receipt_name, train_episode = "source0_transfer_audit.json", 0
        roles = {"parquet": f"{train}/data/chunk-000/episode_000000.parquet"}
        roles.update({key: f"replay/validation/source_episode_000000_{suffix}" for key,suffix in (
            ("physics","physics.json"),("controller_reset","controller_reset.json"),
            ("reset_joint_state43","reset_joint_state43.npy"),("initial_objects","initial_objects.json"))})
        required = shared + list(roles.values())
    sources = bound_sources(root, required, receipt_name=receipt_name)
    selection = json.loads((root / shared[0]).read_text())
    selected = [row for row in selection["episodes"] if row.get("source_scene_episode_index") == source_episode]
    if len(selected) != 1 or selected[0].get("split") != "train":
        raise ValueError("Source is not unambiguously TRAIN")
    if generic:
        eligible = {item["source_scene_episode_index"] for item in selection["episodes"]
            if item.get("split") == "train" and item.get("data_category") == "original_or_repaired"}
        receipt_ids = [item["source_episode_index"] for item in transfer["episodes"]]
        if (len(eligible) != 24 or len(receipt_ids) != 24 or len(set(receipt_ids)) != 24
                or set(receipt_ids) != eligible or source_episode not in eligible or eligible.intersection(VAL_IDS)):
            raise ValueError("Only the 24 original TRAIN ancestors are eligible, no regenerated/heldout sources")
    info = json.loads((root / f"{train}/meta/info.json").read_text())
    replay_info = json.loads((root / "replay/meta/info.json").read_text())
    rows = [json.loads(line) for line in (root / f"{train}/meta/episodes.jsonl").read_text().splitlines()]
    matching_rows = [item for item in rows if item["episode_index"] == train_episode]
    if len(matching_rows) != 1:
        raise ValueError("TRAIN dense episode index must identify exactly one row")
    row = matching_rows[0]
    if (row.get("source_scene_episode_index") != source_episode or row.get("source_episode_index") != source_episode
            or row.get("lineage_complete") is not True or row.get("hand_target_semantics") != "collection-effective"
            or info.get("fps") != 50 or replay_info.get("fps") != 50 or str(info.get("sonic_version")) != "1.1"
            or any(ancestor.get("source_episode_index") in VAL_IDS for ancestor in row["lineage_group_sources"])):
        raise ValueError("Wrong TRAIN source identity, rate or heldout lineage")
    if generic and (row.get("data_category") != "original_or_repaired"
            or row.get("split_restriction") != "unrestricted"
            or row["lineage_group_sources"] != source_record["lineage_group_sources"]
            or len(row["lineage_group_sources"]) != 1
            or row.get("source_dataset_key") != source_record["source_dataset_key"]
            or selected[0].get("source_dataset_key") != source_record["source_dataset_key"]):
        raise ValueError("TRAIN ancestor lineage/category identity mismatch")
    table = pq.read_table(root / roles["parquet"], columns=[
        "action.body_token_v1_1", "action", "replay_action78", "observation.base_pose",
        "observation.base_vel", "observation.object_poses", "timestamp"])
    data = table.to_pydict()
    commands = repack_commands(data["action.body_token_v1_1"], data["action"],
        info["features"]["action"]["names"], data["replay_action78"])
    if len(commands) != row["length"] or not np.allclose(data["timestamp"], np.arange(len(commands))/50, rtol=0, atol=2e-5):
        raise ValueError("Source executed row count or actual50Hz clock mismatch")
    physics = json.loads((root / roles["physics"]).read_text())
    initial_objects = json.loads((root / roles["initial_objects"]).read_text())
    reset = vector(np.load(root / roles["reset_joint_state43"], allow_pickle=False), 43, "reset")
    reset_proof = json.loads((root / roles["controller_reset"]).read_text())
    if (not np.array_equal(reset, physics["reset_joint_state43"])
            or reset_proof["vector_sha256"] != hashlib.sha256(reset.astype("<f8").tobytes()).hexdigest()
            or physics["reset_joint_names"] != replay_info["features"]["observation.state"]["names"]):
        raise ValueError("Source physical reset vector/name identity mismatch")
    root_pose, root_velocity = pose(data["observation.base_pose"][0], "root"), vector(data["observation.base_vel"][0], 6, "root qvel")
    object_names = info["features"]["observation.object_poses"]["names"]
    objects = {}
    for role in ("target", "container"):
        terms = [f"{role}.{term}" for term in ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")]
        initial_pose = pose([data["observation.object_poses"][0][object_names.index(term)] for term in terms], role)
        proof = initial_objects["objects"][role]
        if not np.array_equal(initial_pose, proof["actual_pose7"]) or not np.array_equal(initial_pose, proof["recorded_pose7"]):
            raise ValueError("Initial object pose is not the actual source row0")
        objects[role] = {"pose7": initial_pose.tolist(), "mass": float(physics["objects"][role]["mass"])}
    initial = {"source_episode_index": source_episode, "joint_names": physics["reset_joint_names"], "joint_qpos": reset.tolist(),
        "root_pose7": root_pose.tolist(), "root_qvel6": root_velocity.tolist(), "objects": objects, "sources": sources}
    if generic and (source_record["source_rows"] != len(commands)
            or source_record["parquet"] != source_record["required_files"]["parquet"]
            or source_record["parquet"]["sha256"] != sources[str(root / roles["parquet"])]):
        raise ValueError("Converted TRAIN parquet identity/row count mismatch")
    return {"commands": commands, "row": row, "initial": initial, "sources": sources, "source_split": "train",
        "source_record": source_record, "model_joint_names": physics["reset_joint_names"][:29] + list(MODEL_HAND_NAMES)}


def load_source0(snapshot_root, *, source_episode=0):
    """Backward-compatible entry point; new ancestors require the full receipt."""
    return load_source(snapshot_root, source_episode=source_episode)


def apply_teacher_reference(source, path, binding, *, warmup_control, warmup_seconds, command_rate):
    """Overlay a preregistered TRAIN tail while preserving all original evidence.

    The reference array is native50Hz; the regular command_source_index still
    performs the actual30Hz hold. Never execute its separate held-table as a
    second native reference or silently replace source ancestry/reset data.
    """
    if binding is None or command_rate != 30 or warmup_seconds != 1.:
        raise ValueError("Modified teacher requires a registered candidate, true30Hz and exactly1s startup")
    from prepare_handover_teacher_reference import validate_reference
    expected = {key:binding[key] for key in ("candidate_id","candidate_spec_sha256","plan_sha256")}
    expected["source_episode_index"] = binding["proposal"]["ancestor"]["source_episode_index"]
    commands, receipt = validate_reference(path,source,expected_candidate=expected,expected_warmup=warmup_control,
        allow_new_execution_candidate=True)
    if len(commands)>len(source["commands"])+250:
        raise ValueError("Modified TRAIN teacher may append at most5s of commands")
    document = json.loads(Path(receipt["path"]).read_text())
    identities = {receipt["path"]:receipt["sha256"],receipt["commands_path"]:receipt["commands_sha256"],
        receipt["actual_executed_commands_path"]:receipt["actual_executed_commands_sha256"],
        **document["repair_inputs_sha256"]}
    for filename,identity in identities.items():
        if file_sha256(filename)!=identity or (filename in source["sources"] and source["sources"][filename]!=identity):
            raise ValueError("Modified teacher inputs changed or conflict with original source identity")
    overlaid = deepcopy(source)
    overlaid.update(commands=commands.copy(),source_parent_rows=len(source["commands"]),
        teacher_reference=receipt,teacher_reference_input_sha256=identities)
    # load_source intentionally shares the reset's source-proof dict. Keep that
    # original reset proof untouched when extending command provenance.
    overlaid["sources"] = {**source["sources"],**identities}
    return overlaid


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=True).encode()).hexdigest()


def read_candidate(plan_path, candidate_id):
    """Bind one immutable proposal, never treating the plan as acceptance."""
    plan_path = Path(plan_path).resolve(strict=True)
    plan = json.loads(plan_path.read_text())
    if plan.get("contract") != "cvpr-handover-quality100-candidate-plan-v1":
        raise ValueError("Unrecognized candidate plan contract")
    matches = [item for item in plan["candidates"] if item["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise ValueError("Candidate ID must identify exactly one registered proposal")
    candidate = matches[0]
    proposal = {key:candidate[key] for key in CANDIDATE_SPEC_KEYS}
    if (candidate.get("candidate_spec_sha256") != canonical_sha256(proposal)
            or candidate.get("split") != "train" or candidate.get("accepted") is not False
            or proposal["ancestor"] not in plan["original_ancestors"]
            or proposal["ancestor"]["source_episode_index"] in VAL_IDS):
        raise ValueError("Candidate hash/split/original TRAIN ancestor binding failed")
    for name in ("physics_seed", "material_seed", "generation_seed"):
        if type(proposal.get(name)) is not int or not 0 < proposal[name] < 2**30:
            raise ValueError("Candidate seeds must be distinct from reserved heldout seed range")
    # These are deliberately small initial-condition proposals, not control-time corrections.
    deltas = proposal["initial_object_deltas"]
    if set(deltas) != {"target", "container"}:
        raise ValueError("Candidate must explicitly specify both initial object deltas")
    for role, delta in deltas.items():
        xy = vector(delta["xy_delta_m"], 2, role + " XY delta")
        yaw = delta["world_z_yaw_delta_rad"]
        if (not isinstance(yaw, (int, float)) or not math.isfinite(yaw)
                or np.max(np.abs(xy)) > .01 + 1e-12 or abs(yaw) > .05 + 1e-12):
            raise ValueError("Candidate initial pose proposal exceeds registered small bounds")
    return {"plan_path": str(plan_path), "plan_sha256": file_sha256(plan_path),
        "candidate_id": candidate_id, "candidate_spec_sha256": candidate["candidate_spec_sha256"],
        "candidate": candidate, "proposal": proposal}


def apply_candidate(source, binding):
    """Apply XY/world-Z yaw once to the reset input, before ANY warmup/history."""
    proposal = binding["proposal"]
    ancestor, record = proposal["ancestor"], source.get("source_record")
    if (record is None or ancestor["source_episode_index"] != source["initial"]["source_episode_index"]
            or ancestor["source_dataset_key"] != record["source_dataset_key"]
            or ancestor["source_original_category"] != "original_or_repaired"
            or ancestor["source_parquet_sha256"] != record["plan_declared_source_parquet_sha256"]):
        raise ValueError("Candidate must match original TRAIN ancestry and layered source SHA identity")
    if source.get("perturbation_receipt") is not None:
        raise ValueError("Initial candidate pose may only be applied once")
    prepared = deepcopy(source)
    receipt = {"apply_count": 1, "phase": "before_warmup", "object_restores_during_control": False,
        "initial_object_deltas": deepcopy(proposal["initial_object_deltas"]), "objects": {},
        "material_seed_applied": False, "generation_seed_applied": False,
        "seed_scope": "physics_seed is env.reset seed; material/generation seeds are proposal identities only"}
    for role, delta in proposal["initial_object_deltas"].items():
        original = pose(source["initial"]["objects"][role]["pose7"], role)
        applied = original.copy()
        applied[:2] += delta["xy_delta_m"]
        half = delta["world_z_yaw_delta_rad"] / 2
        c, s = math.cos(half), math.sin(half)
        w, x, y, z = original[3:]
        applied[3:] = [c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w]
        applied = pose(applied, role + " perturbed")
        prepared["initial"]["objects"][role]["pose7"] = applied.tolist()
        receipt["objects"][role] = {"source_pose7": original.tolist(), "applied_pose7": applied.tolist(),
            "xy_delta_m": list(delta["xy_delta_m"]), "world_z_yaw_delta_rad": delta["world_z_yaw_delta_rad"]}
    prepared["perturbation_receipt"] = receipt
    prepared["sources"][binding["plan_path"]] = binding["plan_sha256"]
    return prepared, receipt


def capture_tick_index(tick50):
    """30Hz sample k exists only on its actual ceil(k*50/30) physical tick."""
    if type(tick50) is not int or tick50 < 0:
        raise ValueError("Physical capture tick must be a nonnegative integer")
    k = tick50 * 30 // 50
    return k if (k * 50 + 29) // 30 == tick50 else None


def applied_force_state(data):
    arrays = [np.asarray(data.qfrc_applied), np.asarray(data.xfrc_applied)]
    if any(not np.isfinite(array).all() for array in arrays):
        raise RuntimeError("Nonfinite externally applied force")
    return {"all_zero": all(not np.any(array != 0) for array in arrays),
        "qfrc_applied_all_zero": not np.any(arrays[0] != 0),
        "xfrc_applied_all_zero": not np.any(arrays[1] != 0),
        "max_abs_qfrc_applied": float(np.max(np.abs(arrays[0]), initial=0)),
        "max_abs_xfrc_applied": float(np.max(np.abs(arrays[1]), initial=0)),
        "qfrc_applied_max_abs": float(np.max(np.abs(arrays[0]), initial=0)),
        "xfrc_applied_max_abs": float(np.max(np.abs(arrays[1]), initial=0)),
        "qfrc_applied_nonzero_count": int(np.count_nonzero(arrays[0])),
        "xfrc_applied_nonzero_count": int(np.count_nonzero(arrays[1]))}


def sonic_command_parameters(command):
    """Validate the actual SIMPLE object, whose API deliberately has no .get()."""
    from simple.core.action import ActionCmd

    if not isinstance(command, ActionCmd) or command.type != "sonic_v1_1":
        raise ValueError("Capture requires actual SIMPLE ActionCmd(type='sonic_v1_1')")
    parameters = command.parameters
    if (not isinstance(parameters, dict)
            or not {"controller", "body_token", "hand_targets"}.issubset(parameters)
            or parameters["controller"] is None):
        raise ValueError("SONIC ActionCmd must retain its actual controller/token/hand parameters")
    return parameters


class RepeatFirstSnapshotDecoderHistory:
    """Instance-local decoder adapter for an explicit causal startup prefix.

    A real sample is appended by the unchanged physical controller before each
    call. The first9 inputs therefore contain9..1 copies of reset t0. Every
    later input must contain exactly ten real chronological50Hz samples. No
    synthetic past timestamps, body-state forcing or future references exist.
    """
    def __init__(self, first_sample, original_decoder_input, first_snapshot):
        self.first_sample = deepcopy(first_sample)
        self.original_decoder_input = original_decoder_input
        self.receipt = {"contract":STARTUP_HISTORY_CONTRACT,"policy":"repeat_first_snapshot",
            "seeded_history_slots":10,"initialization_count":1,"same_timestamp":True,
            "timestamp_policy":"All startup copies retain actual reset t0; never fabricated past timestamps",
            "padding_scope":"Decoder proprioception only; never RGB/state observations or training action labels",
            "first_snapshot":deepcopy(first_snapshot),"first_snapshot_sha256":canonical_sha256(first_snapshot),
            "decoder_inputs":[],"padded_decoder_calls":0,"total_decoder_calls":0,
            "first_all_real_decoder_sim_time_s":None,"state_forcing":False,
            "source_future_used":False,"history_injection":True,
            "adapter_source_sha256":file_sha256(__file__)}

    def __call__(self, latent64, history):
        latent = vector(latent64,64,"startup actual token64").astype(np.float32)
        if len(history)!=10:
            raise ValueError("Startup decoder requires exactly ten explicitly described slots")
        tick = self.receipt["total_decoder_calls"]
        pad = max(0,9-tick)
        actual_indices = list(range(max(0,tick-9),tick+1))
        origins = ([{"kind":"startup_repeat_first_snapshot","padded_from_initial":True,
            "padded_from_actual_timestep_index":0} for _ in range(pad)]
            + [{"kind":"actual_physical_sample","actual_timestep_index":index} for index in actual_indices])
        expected_times = [self.first_sample[0]]*pad + [self.first_sample[0]+index/50 for index in actual_indices]
        times = [float(sample[0]) for sample in history]
        if len(origins)!=10 or not np.allclose(times,expected_times,rtol=0,atol=1e-6):
            raise ValueError("Startup padding must fade9..1 then use only causal actual50Hz samples")
        widths=(3,29,29,29,3)
        for index,sample in enumerate(history):
            if len(sample)!=6 or any(np.shape(v)!=(w,) or not np.isfinite(v).all()
                    for v,w in zip(sample[1:],widths)):
                raise ValueError("Invalid actual/startup decoder sample shape or finite values")
            if index<pad and any(not np.array_equal(a,b) for a,b in zip(sample[1:],self.first_sample[1:])):
                raise ValueError("Startup padding must repeat only the first actual snapshot")
        if pad:
            value=np.concatenate([latent]+[np.stack([sample[index] for sample in history]).reshape(-1)
                for index in range(1,6)]).astype(np.float32)[None]
        else:
            value=self.original_decoder_input(latent,history)
        if value.shape!=(1,994) or not np.isfinite(value).all():
            raise ValueError("Invalid official decoder994 input")
        self.receipt["decoder_inputs"].append({"decoder_call_index":tick,
            "sim_time_s":times[-1],"history_sample_times_s":times,"padding_count":pad,
            "history_origin":origins,"all_ten_samples_real":pad==0,
            "input_values_float32":value[0].tolist(),
            "input_sha256":hashlib.sha256(value.astype('<f4',copy=False).tobytes()).hexdigest()})
        self.receipt["padded_decoder_calls"] += int(pad>0)
        self.receipt["total_decoder_calls"] += 1
        if not pad and self.receipt["first_all_real_decoder_sim_time_s"] is None:
            self.receipt["first_all_real_decoder_sim_time_s"] = times[-1]
        return value


def install_instance_startup_decoder(controller, adapter):
    """Keep the upstream physical method, replacing only this instance's input builder."""
    original=controller.apply_physics.__func__
    if ("decoder_input" not in original.__code__.co_names
            or original.__globals__.get("decoder_input") is not adapter.original_decoder_input
            or "apply_physics" in controller.__dict__):
        raise ValueError("Require the unchanged upstream controller before startup adaptation")
    globals_copy=dict(original.__globals__)
    globals_copy["decoder_input"]=adapter
    adapted=FunctionType(original.__code__,globals_copy,original.__name__,original.__defaults__,original.__closure__)
    adapted.__kwdefaults__=original.__kwdefaults__
    controller.apply_physics=MethodType(adapted,controller)
    adapter.receipt["physical_method_adapter"]={"scope":"this controller instance only; no module-global monkeypatch",
        "unchanged_function_code":True,"globals_replaced":["decoder_input"],
        "upstream_source_path":original.__globals__["__file__"],
        "upstream_source_sha256":file_sha256(original.__globals__["__file__"])}


def record_padded_decoder_step(warmup, controller, record, *, is_warmup):
    """Bind actual post-step decoder output and label the startup prefix truthfully.

    These values are read from the physical controller after env.step, never
    inferred from a desired reference or a later previous-action input.
    """
    decoder = warmup.startup_decoder.receipt["decoder_inputs"][-1]
    if (record["decoder_calls_after"] != record["decoder_calls_before"]+1
            or decoder["decoder_call_index"] != record["tick50"]
            or not math.isclose(decoder["sim_time_s"],record["sim_time_s"],abs_tol=1e-6)):
        raise ValueError("Padded decoder step is not the current single physical decoder call")
    raw_action = vector(controller._last_raw_action,29,"actual controller raw policy action29")
    body_target = vector(controller._body_target,29,"actual controller physical PD target29")
    event = None
    if is_warmup:
        event = warmup.receipt["events"][-1]
        if not math.isclose(event["sim_time_s"],record["sim_time_s"],abs_tol=1e-6):
            raise ValueError("Padded warmup event is not the current actual decoder command")
    record.update(startup_decoder_input_index=decoder["decoder_call_index"],
        startup_history_padding_slots=decoder["padding_count"],
        controller_last_raw_policy_action29=raw_action.tolist(),
        body_pd_target29=body_target.tolist())
    if event is not None:
        event.update(history_injection=decoder["padding_count"]>0,
            startup_history_padding_slots=decoder["padding_count"])


def make_repeat_first_snapshot_warmup(resource_dir, *, seconds):
    """Use the fixed live-pose encoder from tick0 with declared startup padding."""
    from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
    import simple.baselines.sonic_v1_1 as sonic

    class PaddedLivePoseWarmup(LivePoseSonicWarmup):
        def reset(self):
            super().reset()
            self.startup_decoder=None
            self.receipt.update(kind=PADDED_WARMUP_KIND,seconds=seconds,history_injection=True,
                startup_adapter_source_sha256=file_sha256(__file__),
                proprioception="Explicit first-reset-snapshot padding during first9 decoder calls; then ten real50Hz samples",
                hand_policy_scope="Measured first live hands from tick0; no teacher references or future labels")

        def command(self,robot,controller,observation):
            if self.startup_decoder is None:
                import mujoco
                model,data=robot.mjModel,robot.mjData
                if (controller._history or controller.decoder_calls or controller._last_tick is not None
                        or controller._body_target is not None
                        or not np.isclose(data.time,observation['sim_time_s'],rtol=0,atol=1e-9)):
                    raise ValueError("Startup padding requires the first actual post-reset snapshot and fresh controller")
                q=np.asarray([data.joint(n).qpos[0] for n in sonic.BODY_JOINT_NAMES],float)
                dq=np.asarray([data.joint(n).qvel[0] for n in sonic.BODY_JOINT_NAMES],float)
                pelvis=model.body('pelvis').id
                velocity=np.zeros(6)
                mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,pelvis,velocity,1)
                gravity=data.xmat[pelvis].reshape(3,3).T@np.array([0.,0.,-1.])
                previous=np.asarray(((q-sonic.DEFAULT_ANGLES)/sonic.ACTION_SCALE)[sonic.POLICY_TO_BODY],np.float32)
                root_joint=int(model.body_jntadr[pelvis]);root_dof=int(model.jnt_dofadr[root_joint])
                snapshot={"sim_time_s":float(data.time),"body_joint_names":list(sonic.BODY_JOINT_NAMES),
                    "physical_body_qpos29":q.tolist(),"physical_body_qvel29":dq.tolist(),
                    "observed_joint_names":list(observation['joint_names']),
                    "observed_joint_qpos":np.asarray(observation['joint_qpos']).tolist(),
                    "root_pose7":[*data.xpos[pelvis],*data.xquat[pelvis]],
                    "root_qvel6":data.qvel[root_dof:root_dof+6].tolist(),"actual_qvel":data.qvel.tolist(),
                    "body_local_angular_velocity3":velocity[:3].tolist(),"projected_gravity3":gravity.tolist(),
                    "controller_reset_previous_raw_action29":controller._last_raw_action.tolist(),
                    "effective_initial_previous_raw_action29":previous.tolist(),
                    "effective_previous_action_rule":"Unchanged upstream apply_physics initializes measured-pose target before its first sample",
                    "body_default_angles29":sonic.DEFAULT_ANGLES.tolist(),"body_action_scale29":sonic.ACTION_SCALE.tolist(),
                    "source":"Only the first actual post-reset observation/current physical sensors; no teacher trajectory"}
                sample=(float(data.time),velocity[:3].copy(),(q-sonic.DEFAULT_ANGLES)[sonic.POLICY_TO_BODY].copy(),
                    dq[sonic.POLICY_TO_BODY].copy(),previous.copy(),gravity.copy())
                adapter=RepeatFirstSnapshotDecoderHistory(sample,sonic.decoder_input,snapshot)
                install_instance_startup_decoder(controller,adapter)
                for _ in range(10):controller._history.append(deepcopy(sample))
                self.startup_decoder=adapter
                self.receipt['startup_history_padding']=adapter.receipt
            command=super().command(robot,controller,observation)
            self.phase='warmup_'+PADDED_WARMUP_MODE
            return command
    return PaddedLivePoseWarmup(resource_dir,hand_policy='measured')


def renderer_pose_snapshot(diagnostics, *, sim_time_s, observation_sim_time_s,
                           authoritative_poses, world_time_reference_s=None):
    """Copy the just-rendered evidence; never refresh, render or infer a pose.

    USD-to-MuJoCo mount offsets are recorded, not certified away. Only the
    renderer's reported MuJoCo state and the sensor's own USD prim are compared
    here. The external auditor independently checks the serialized integration
    state. Isaac's observation-only clock must remain fixed, unlike MuJoCo time.
    """
    if not isinstance(diagnostics, dict) or not isinstance(authoritative_poses, dict):
        raise ValueError("Fresh actual renderer pose diagnostics and physical pose evidence are required")
    saved = deepcopy(diagnostics)
    clock = saved.get("mujoco_sim_time_s")
    world_time = saved.get("isaac_world_time_s")
    times = (clock, world_time, sim_time_s, observation_sim_time_s)
    if (any(not isinstance(value, (int,float,np.number)) or not math.isfinite(value) for value in times)
            or not math.isclose(clock, sim_time_s, rel_tol=0, abs_tol=1e-6)
            or not math.isclose(observation_sim_time_s, sim_time_s, rel_tol=0, abs_tol=1e-6)):
        raise ValueError("Renderer diagnostic, current observation and actual MuJoCo clocks differ")
    if (world_time_reference_s is not None
            and not math.isclose(world_time, world_time_reference_s, rel_tol=0, abs_tol=1e-9)):
        raise ValueError("Observation-only Isaac world time advanced during capture")
    links, cameras = saved.get("links"), saved.get("cameras")
    if (not isinstance(links, dict) or not set(RENDERER_LINKS).issubset(links)
            or not isinstance(cameras, dict) or "head_stereo_left" not in cameras):
        raise ValueError("Renderer diagnostics lack required actual body links or head_stereo_left camera")
    def quaternion(value, label):
        q = vector(value, 4, label)
        if not np.isclose(np.linalg.norm(q), 1, rtol=0, atol=1e-5):
            raise ValueError("Nonunit renderer quaternion: " + label)
        return q
    def qdistance(a, b):
        return min(float(np.max(np.abs(a-b))),float(np.max(np.abs(a+b))))
    def usd_pose(entry, label):
        if (not isinstance(entry, dict) or entry.get("valid") is not True
                or not isinstance(entry.get("path"), str) or not entry["path"].startswith("/")):
            raise ValueError("Invalid actual USD prim pose: " + label)
        return vector(entry["position_m"],3,label+" USD position"),quaternion(entry["quaternion_wxyz"],label+" USD quaternion")
    link_residuals = {}
    for name, entry in links.items():
        usd_position, usd_quaternion = usd_pose(entry.get("usd"),name)
        if name not in RENDERER_LINKS:
            continue  # Extra real camera-mount USD prim has no reported MuJoCo link.
        actual = authoritative_poses["links"][name]
        mj_position = vector(entry["mujoco_position_m"],3,name+" reported MuJoCo position")
        mj_quaternion = quaternion(entry["mujoco_quaternion_wxyz"],name+" reported MuJoCo quaternion")
        actual_position = vector(actual["position_m"],3,name+" actual MuJoCo position")
        actual_quaternion = quaternion(actual["quaternion_wxyz"],name+" actual MuJoCo quaternion")
        if (not np.allclose(mj_position,actual_position,rtol=0,atol=1e-9)
                or qdistance(mj_quaternion,actual_quaternion)>1e-9):
            raise ValueError("Renderer MuJoCo link evidence differs from actual current state: " + name)
        link_residuals[name] = {"usd_mujoco_position_l2_m":float(np.linalg.norm(usd_position-mj_position)),
            "usd_mujoco_quaternion_sign_invariant_max_abs":qdistance(usd_quaternion,mj_quaternion)}
    camera = cameras["head_stereo_left"]
    if not isinstance(camera, dict) or camera.get("mujoco_camera_missing"):
        raise ValueError("The actual rendered head_stereo_left MuJoCo camera is missing")
    usd_position, usd_quaternion = usd_pose(camera.get("usd"),"head_stereo_left")
    sensor_position = vector(camera["sensor_position_m"],3,"sensor position")
    sensor_quaternion = quaternion(camera["sensor_quaternion_wxyz_usd_axes"],"sensor quaternion in USD axes")
    sensor_position_error = float(np.max(np.abs(sensor_position-usd_position)))
    sensor_quaternion_error = qdistance(sensor_quaternion,usd_quaternion)
    if sensor_position_error>1e-5 or sensor_quaternion_error>1e-5:
        raise ValueError("Actual camera sensor and its USD prim pose differ")
    mj_position = vector(camera["mujoco_position_m"],3,"reported MuJoCo camera position")
    mj_rotation = vector(camera["mujoco_rotation_usd_axes"],9,"reported MuJoCo camera rotation")
    actual = authoritative_poses["cameras"]["head_stereo_left"]
    if (not np.allclose(mj_position,vector(actual["position_m"],3,"actual camera position"),rtol=0,atol=1e-9)
            or not np.allclose(mj_rotation,vector(actual["rotation_usd_axes"],9,"actual camera rotation"),rtol=0,atol=1e-9)):
        raise ValueError("Renderer MuJoCo camera evidence differs from actual current state")
    if (not np.allclose(mj_rotation.reshape(3,3).T @ mj_rotation.reshape(3,3),np.eye(3),rtol=0,atol=1e-5)
            or not np.isclose(np.linalg.det(mj_rotation.reshape(3,3)),1,rtol=0,atol=1e-5)):
        raise ValueError("Invalid actual MuJoCo camera rotation")
    return saved, {"contract":RENDERER_POSE_CONTRACT,"camera":"head_stereo_left",
        "source":"last_observation_pose_diagnostics from the actual preceding get_obs; never recomputed",
        "mujoco_clock_matches_observation_and_event":True,"actual_mujoco_links_and_camera_verified":True,
        "isaac_world_time_s":float(world_time),"isaac_world_time_fixed":True,
        "sensor_usd_position_max_abs_m":sensor_position_error,
        "sensor_usd_quaternion_sign_invariant_max_abs":sensor_quaternion_error,
        "usd_mujoco_link_residuals":link_residuals,
        "usd_mujoco_camera_position_l2_m":float(np.linalg.norm(usd_position-mj_position)),
        "render_mount_equivalence_verified":False}


class TrainingCapture:
    """Write raw real-time RGB/state/command events; independent audit accepts."""
    def __init__(self, directory, *, metadata, joint_names, warmup_ticks, start_time):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        if (len(joint_names) != 43 or len(set(joint_names)) != 43
                or list(joint_names[-14:]) != list(MODEL_HAND_NAMES)
                or type(warmup_ticks) is not int or warmup_ticks < 50 or warmup_ticks % 5):
            raise ValueError("Need named model43 order and >=1s warmup aligned to both clocks")
        self.metadata = deepcopy(metadata)
        self.names = list(joint_names)
        self.warmup_ticks, self.start_time = warmup_ticks, start_time
        self.events_path = self.directory / "events.jsonl"
        self.stream = self.events_path.open("w", buffering=1)
        self.count = self.warmup_count = self.command_count = 0
        self.renderer_world_time_reference_s = None
        self.closed = False

    def append(self, *, tick50, sim_time_s, rgb, observed_joint_names, observed_qpos,
               physical_qpos, physical_qvel, root_pose7, root_qvel6, objects, command,
               renderer_pose_diagnostics=None, renderer_authoritative_poses=None, observation_sim_time_s=None):
        from PIL import Image

        k = capture_tick_index(tick50)
        if k is None or k != self.count or not np.isclose(sim_time_s-self.start_time, tick50/50, rtol=0, atol=1e-6):
            raise ValueError("Capture requires each dense real ceil30Hz frame and matching actual sim clock")
        names = list(observed_joint_names)
        if len(names) != 43 or len(set(names)) != 43 or set(names) != set(self.names):
            raise ValueError("Observation must contain exactly the named actual43 joints")
        observed = vector(observed_qpos, 43, "observed state43")
        state = np.asarray([observed[names.index(name)] for name in self.names], dtype=np.float32)
        physical = vector(physical_qpos, 43, "physical qpos43")
        velocity = vector(physical_qvel, 43, "physical qvel43")
        if not np.array_equal(state, physical.astype(np.float32)):
            raise ValueError("RGB observation and physical qpos are not the same current frame")
        renderer_diagnostics, renderer_validation = renderer_pose_snapshot(renderer_pose_diagnostics,
            sim_time_s=sim_time_s, observation_sim_time_s=observation_sim_time_s,
            authoritative_poses=renderer_authoritative_poses,world_time_reference_s=self.renderer_world_time_reference_s)
        pixels = np.asarray(rgb)
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
            raise ValueError("Expected actual uint8 RGB HxWx3, no color conversion or synthetic image")
        pixels = np.ascontiguousarray(pixels)
        warmup = tick50 < self.warmup_ticks
        action = None
        source_index = None
        if not warmup:
            parameters = sonic_command_parameters(command)
            if parameters["body_token"] is None:
                raise ValueError("A teacher event requires the actual applied v1.1 token and named hand targets")
            token = vector(parameters["body_token"], 64, "actual body token64")
            hands = parameters["hand_targets"]
            if set(hands) != set(MODEL_HAND_NAMES):
                raise ValueError("Actual hand command must have exactly fourteen named channels")
            hand = vector([hands[name] for name in MODEL_HAND_NAMES], 14, "actual model hand14")
            action = {"body_token64": token.tolist(), "hand14": hand.tolist(), "hand_joint_names": list(MODEL_HAND_NAMES)}
            source_index = tick50 - self.warmup_ticks
            if command_source_index(source_index, 30, source_index+1) != source_index:
                raise ValueError("Training event must precede a genuinely new held30Hz command")
        image_path = Path("rgb") / f"frame_{k:06d}.png"
        (self.directory / image_path).parent.mkdir(exist_ok=True)
        Image.fromarray(pixels).save(self.directory / image_path, format="PNG", compress_level=3)
        event = {"event_index": k, "capture_index30": k, "physical_tick50": tick50, "sim_time_s": float(sim_time_s),
            "nominal_elapsed_s": k/30, "actual_elapsed_s": tick50/50,
            "phase": "warmup" if warmup else "teacher_command", "action_valid": not warmup,
            "source_command_index": source_index, "joint_names": self.names, "state43": state.tolist(),
            "physical_joint_q43_float64": physical.tolist(), "physical_joint_qvel43": velocity.tolist(),
            "root_pose7": pose(root_pose7, "actual root").tolist(),
            "root_qvel6": vector(root_qvel6, 6, "actual root velocity").tolist(),
            "objects": {role: {"pose7": pose(objects[role]["pose7"], role).tolist()} for role in ("target", "container")},
            "renderer_camera":"head_stereo_left","renderer_pose_diagnostics":renderer_diagnostics,
            "renderer_pose_validation":renderer_validation,
            "rgb": {"path": str(image_path), "file_sha256": file_sha256(self.directory / image_path),
                "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "width": pixels.shape[1], "height": pixels.shape[0], "channels": 3, "dtype": "uint8"},
            "action": action}
        self.stream.write(json_text(event)+"\n")
        self.count += 1
        if self.renderer_world_time_reference_s is None:
            self.renderer_world_time_reference_s = float(renderer_diagnostics["isaac_world_time_s"])
        self.warmup_count += int(warmup)
        self.command_count += int(not warmup)

    def close(self, *, completed):
        if self.closed:
            return self.manifest
        self.stream.close()
        self.manifest = {**self.metadata, "contract": CAPTURE_CONTRACT, "version": 1,
            "status": "raw_capture_completed_pending_independent_audit" if completed else "partial_capture_rejected",
            "events_path": "events.jsonl", "events_sha256": file_sha256(self.events_path),
            "event_count": self.count, "warmup_event_count": self.warmup_count,
            "command_event_count": self.command_count, "joint_names": self.names,
            "hand_joint_names": list(MODEL_HAND_NAMES), "command_rate_hz": 30, "physics_hz": 200,
            "decoder_hz": 50, "render_engine": "Isaac RTX", "warmup_seconds": self.warmup_ticks/50,
            "renderer_pose_contract":RENDERER_POSE_CONTRACT,
            "isaac_world_time_reference_s":self.renderer_world_time_reference_s,
            "render_mount_equivalence_verified":False,
            "object_restores_during_control": False, "accepted_new_dataset": False,
            "history": "Only preceding real warmup/current RGB; no teacher future image or state input",
            "alignment": "RGB/state/actual command sampled before the same real50Hz step at ceil(k*50/30); warmup action=null and invalid"}
        write_json(self.directory / "capture.json", self.manifest)
        self.closed = True
        return self.manifest


def local_environment_config(source, asset_root):
    asset_root = Path(asset_root).resolve(strict=True)
    config = json.loads(source["row"]["environment_config"])
    original_uid = config["uid"]
    if original_uid not in ("g1_wholebody_handover_sonic", "g1_wholebody_handover_teleop"):
        raise ValueError("Wrong source Handover task")
    config["uid"] = "g1_wholebody_handover_teleop"
    for role, item in source["initial"]["objects"].items():
        value = item["pose7"]
        actor = config["layout"]["actors"][role]
        pose_dict = {"position": value[:3], "quaternion": value[3:]}
        actor["pose"] = deepcopy(pose_dict)
        key = f"container_{actor['uid']}" if role == "container" else str(actor["uid"])
        config["dr_state_dict"]["spatial"][key] = deepcopy(pose_dict)
    return remap_paths(config, asset_root, asset_root), original_uid


def solved_step_hook(original_step, current_teacher_tick, capture_quality, *, physics_substeps=4, post_substep=None):
    """Observe the real solved50Hz boundary before any observation mj_forward."""
    calls = 0
    def step(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        if post_substep is not None:
            post_substep()
        tick = current_teacher_tick()
        if tick is not None:
            calls += 1
            if calls % physics_substeps == 0:
                if calls // physics_substeps != tick + 1:
                    raise RuntimeError("Teacher solver-boundary sampling clock drifted")
                capture_quality(tick + 1)
        return result
    return step


def run(args):
    binding = None
    if bool(args.candidate_plan) != bool(args.candidate_id):
        raise ValueError("Use --candidate-plan and --candidate-id together")
    if args.candidate_plan:
        binding = read_candidate(args.candidate_plan, args.candidate_id)
        source_id = binding["proposal"]["ancestor"]["source_episode_index"]
        seed = binding["proposal"]["physics_seed"]
        if (args.source_episode is not None and args.source_episode != source_id) or (args.seed is not None and args.seed != seed):
            raise ValueError("Explicit source/seed contradicts registered candidate")
    else:
        source_id, seed = args.source_episode or 0, args.seed if args.seed is not None else 20261022
    source = load_source(args.snapshot_root, source_episode=source_id)
    perturbation = None
    if binding:
        source, perturbation = apply_candidate(source, binding)
    if getattr(args,"teacher_reference",None) is not None:
        source = apply_teacher_reference(source,args.teacher_reference,binding,warmup_control=args.warmup_control,
            warmup_seconds=args.warmup_seconds,command_rate=args.command_rate)
    calibration_identities = calibration_source_identities()
    resource_data_root,calibration_asset_sha = bound_calibration_assets(args.data_root,calibration_identities)
    output = args.output.resolve()
    if (output.exists() or output.is_relative_to(Path(__file__).resolve().parents[2])
            or output.is_relative_to(args.snapshot_root.resolve())
            or output.is_relative_to(args.data_root.resolve()) or output.is_relative_to(args.sonic_resources.resolve())):
        raise ValueError("Use a fresh output outside repository/source snapshot")
    capture_path = None
    if args.capture_training is not None:
        capture_path = args.capture_training
        if not capture_path.is_absolute():
            capture_path = output / capture_path
        capture_path = capture_path.resolve()
        if (args.render_mode != "isaac" or args.command_rate != 30 or args.warmup_seconds < 1
                or int(round(args.warmup_seconds*50)) % 5 or binding is None
                or capture_path.exists() or capture_path == output
                or capture_path.is_relative_to(Path(__file__).resolve().parents[2])
                or capture_path.is_relative_to(args.snapshot_root.resolve())
                or capture_path.is_relative_to(args.data_root.resolve())
                or capture_path.is_relative_to(args.sonic_resources.resolve())):
            raise ValueError("Training capture needs fresh external path, registered candidate, RTX, true30Hz and >=1s aligned warmup")
    config, original_uid = local_environment_config(source, args.data_root)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "environment_config.json", config)
    result = {"status": "initializing", "task_id": "simple/"+TASK, "source_episode_index": source_id,
        "source_split": "train", "source_commands": len(source["commands"]), "command_rate_hz": args.command_rate,
        "source_task_uid": original_uid, "source_sha256": source["sources"], "source_tokens_reencoded": False,
        "teacher_replay_diagnostic": True, "learned_policy_used": False, "accepted_new_dataset": False,
        "object_restores_during_control": False,
        "resource_data_root":str(resource_data_root),"calibration_source_asset_sha256":calibration_asset_sha,
        "render_mode": args.render_mode, "rgb_requested": args.render_mode == "isaac", "rgb_collected": False,
        "scope": "Local original-TRAIN teacher cold-start diagnostic/capture, not exact original C++ controller history restoration. No object/state restores during control. Independent physical/data acceptance is required before training.",
        "physics_seed": seed, "pd_profile": args.pd_profile, "hand_torque_profile": args.hand_torque_profile,
        "warmup_control": {"kind":args.warmup_control, "seconds":args.warmup_seconds},
        "perturbation_receipt": perturbation,
        "initial_object_deltas": binding["proposal"]["initial_object_deltas"] if binding else None,
        "source_provenance": {"source_record": source["source_record"], "source_row": source["row"]},
        "started_utc": datetime.now(timezone.utc).isoformat()}
    if binding:
        result.update({key:binding[key] for key in ("plan_path","plan_sha256","candidate_id","candidate_spec_sha256")})
        result["candidate"] = binding["candidate"]
    if "teacher_reference" in source:
        result.update(teacher_reference=source["teacher_reference"],source_parent_rows=source["source_parent_rows"],
            teacher_reference_input_sha256=source["teacher_reference_input_sha256"],
            teacher_command_policy="original native50Hz prefix plus explicit TRAIN-only modified-teacher tail")
    write_json(output / "result.json", result)
    os.environ["SIMPLE_DATA_DIR"] = str(resource_data_root)
    import gymnasium as gym
    import mujoco
    import simple.envs
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.baselines.sonic_v1_1 import SonicV11OnnxController
    from handover_eval_runtime import initialize, measured_task_state
    from handover_contact_quality import sample_live_state, evaluate_capture
    from handover_geometry_quality import build_container_geometry_cache
    import simple.baselines.sonic_v1_1 as sonic_module
    script_directory = Path(__file__).resolve().parent
    simple_directory = Path(sonic_module.__file__).resolve().parents[1]
    code_paths = (
        Path(__file__).resolve(), script_directory/"handover_eval_runtime.py",
        script_directory/"prepare_handover_eval.py", script_directory/"handover_contact_quality.py",
        script_directory/"handover_geometry_quality.py", Path(sonic_module.__file__).resolve(),
        simple_directory/"core/action.py",simple_directory/"engines/isaacsim.py",
        simple_directory/"engines/mujoco.py",simple_directory/"sensors/head_camera_mount.py",
        simple_directory/"envs/sonic_loco_manip.py")
    if args.warmup_control in ("live_pose_sonic_v1_1",PADDED_WARMUP_MODE):
        code_paths += (simple_directory/"baselines/live_pose_sonic_warmup.py",)
    if "teacher_reference" in source:
        code_paths += (script_directory/"prepare_handover_teacher_reference.py",)
    result["capture_code_sha256"] = {str(path):file_sha256(path) for path in code_paths}
    code_directory = output / "provenance/code"
    code_directory.mkdir(parents=True)
    snapshots = []
    for index, path in enumerate(code_paths):
        target = code_directory / f"{index:02d}-{path.name}"
        shutil.copyfile(path, target)
        if file_sha256(target) != result["capture_code_sha256"][str(path)]:
            raise RuntimeError("Capture code changed during source snapshot")
        snapshots.append({"original_path":str(path),"path":str(target.relative_to(output)),"sha256":file_sha256(target)})
    write_json(output / "provenance/code_manifest.json", snapshots)
    result["code_snapshots"] = snapshots
    result["force_sampling_contract"] = "Only the solved last of four real5ms MuJoCo substeps per50Hz tick, before observation mj_forward; no synthetic/offline force estimates"

    env = video = training_capture = None
    states, times, qs, roots, velocities = [], [], [], [], []
    try:
        controller = SonicV11OnnxController(args.sonic_resources, allow_unverified_compatibility=True,
            pd_profile=args.pd_profile, hand_torque_profile=args.hand_torque_profile, sync_mujoco_observations=True)
        result["controller_identity"] = controller.identity
        sonic_config = SimLoopConfig().load_wbc_yaml()
        sonic_config.update(ENV_NAME="simple", SIMULATE_DT=.005)
        rendered = args.render_mode == "isaac"
        env = gym.make("simple/"+TASK, sonic_config=sonic_config, sim_mode="mujoco_isaac" if rendered else "mujoco",
            headless=True, dr_level=0, sync_mujoco_observations=True, isaac_observation_only=rendered,
            disable_env_checker=not rendered)
        raw = env.unwrapped
        if not rendered:
            # Both hooks are instance-only. Removing camera sensors before reset
            # prevents creation of any MuJoCo GL renderer; physics stays genuine.
            raw.task.sensor_cfgs = {}
            raw._render_frame = lambda: {}
        obs, info = env.reset(seed=seed, options={"state_dict": config, "dr_level": None})
        obs, info, initial = initialize(raw, source["initial"], {
            "scope": "Original TRAIN ancestor one-time cold start before warmup; not original controller/history restoration",
            "teacher_actions_used": True, "teacher_primer_used": False, "restores_during_control": False})
        result["initialization"] = initial
        result["initialization"]["object_restores_during_control"] = False
        result["instruction"] = raw.task.instruction
        if not isinstance(result["instruction"], str) or not result["instruction"].strip():
            raise ValueError("Actual task instruction is required")
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        if (not np.isclose(model.opt.timestep, .005, atol=1e-12)
                or not np.array_equal(model.opt.gravity, [0, 0, -9.81])
                or not np.isclose(raw.task.render_hz, 50)):
            raise ValueError("Require real200Hz gravity physics and50Hz environment steps")
        mujoco.mj_saveModel(model, str(output / "physics_model.mjb"))
        result["physics_model"] = {"sha256": file_sha256(output / "physics_model.mjb")}
        def in_body_tree(body, ancestor):
            while body:
                if body == ancestor:
                    return True
                body = int(model.body_parentid[body])
            return False
        equality = []
        for index in range(model.neq):
            entry = {"id":index,"name":model.equality(index).name,"type":int(model.eq_type[index]),
                "obj1_id":int(model.eq_obj1id[index]),"obj2_id":int(model.eq_obj2id[index]),
                "active":bool(data.eq_active[index])}
            if entry["type"] in (int(mujoco.mjtEq.mjEQ_CONNECT),int(mujoco.mjtEq.mjEQ_WELD)):
                entry["obj1_body_name"] = model.body(entry["obj1_id"]).name
                entry["obj2_body_name"] = model.body(entry["obj2_id"]).name
                target_id, pelvis_id = raw.mujoco.mj_objects["target"].id, model.body("pelvis").id
                if entry["active"] and any(in_body_tree(a,target_id) and in_body_tree(b,pelvis_id)
                        for a,b in ((entry["obj1_id"],entry["obj2_id"]),(entry["obj2_id"],entry["obj1_id"]))):
                    raise ValueError("Externally constrained target-to-robot attachment is forbidden")
            equality.append(entry)
        result["physics_model"]["equalities"] = equality
        result["physics_model"]["target_robot_external_attachment"] = False
        cache = build_container_geometry_cache(model, data,
            container_body=raw.mujoco.mj_objects["container"].id,
            target_body=raw.mujoco.mj_objects["target"].id)
        write_json(output / "container_geometry_cache.json", cache)
        quality_rows, quality_phase = [], {"teacher_tick": None}
        initial_target_z = float(raw.mujoco.mj_objects["target"].xpos[2])
        result["initial_target_z_m"] = initial_target_z
        force_stream = (output / "applied_force_samples.jsonl").open("w", buffering=1)
        force_count = 0
        def check_applied_force():
            nonlocal force_count
            check = applied_force_state(data)
            force_count += 1
            force_stream.write(json_text({"substep_index":force_count,"physics_step":force_count,
                "sim_time_s":float(data.time),**check})+"\n")
            if not check["all_zero"]:
                raise RuntimeError("Nonzero external qfrc_applied/xfrc_applied; reject assisted physical capture")
        quality_stream = (output / "quality_samples.jsonl").open("w", buffering=1)
        def capture_quality(completed):
            row = sample_live_state(model, data, cache=cache, completed_steps=completed,
                target_body=raw.mujoco.mj_objects["target"].id,
                container_body=raw.mujoco.mj_objects["container"].id,
                solver_stage="post_mj_step_before_observation_refresh")
            quality_rows.append(row)
            row["applied_forces"] = applied_force_state(data)
            quality_stream.write(json_text(row)+"\n")
        raw.mujoco.step = solved_step_hook(raw.mujoco.step,
            lambda: quality_phase["teacher_tick"], capture_quality, post_substep=check_applied_force)
        controller.reset()
        warmup = None
        if args.warmup_control == "live_pose_sonic_v1_1":
            from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
            warmup = LivePoseSonicWarmup(args.sonic_resources, hand_policy="measured")
        elif args.warmup_control == PADDED_WARMUP_MODE:
            warmup = make_repeat_first_snapshot_warmup(args.sonic_resources,seconds=args.warmup_seconds)
        if rendered:
            from video_capture import RawVideoRecorder
            video = RawVideoRecorder(output / "rollout.mp4", fps=50, shape=np.asarray(obs["head_stereo_left"]).shape)
        spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        start_time = float(data.time)
        warmup_ticks = int(round(args.warmup_seconds * 50))
        total = warmup_ticks + len(source["commands"])
        if capture_path:
            metadata = {key:deepcopy(result[key]) for key in ("plan_sha256","candidate_id","candidate_spec_sha256",
                "source_episode_index","source_sha256","source_provenance","initial_target_z_m",
                "perturbation_receipt","initial_object_deltas","warmup_control","controller_identity",
                "capture_code_sha256","code_snapshots","instruction","resource_data_root",
                "calibration_source_asset_sha256")}
            if "teacher_reference" in result:
                metadata.update({key:deepcopy(result[key]) for key in ("teacher_reference","source_parent_rows",
                    "teacher_reference_input_sha256","teacher_command_policy")})
            training_capture = TrainingCapture(capture_path,metadata=metadata,joint_names=source["model_joint_names"],
                warmup_ticks=warmup_ticks,start_time=start_time)
            result["training_capture_path"] = str(capture_path)
        result.update(status="running", warmup_ticks=warmup_ticks, requested_control_ticks=total,
            physics_hz=200, decoder_hz=50, rgb_hz=50 if rendered else None)
        result["official_success_ever"] = False
        result["fell"] = False
        result["finite_physics"] = True
        write_json(output / "result.json", result)
        def capture():
            state = np.empty(mujoco.mj_stateSize(model, spec), dtype=np.float64)
            mujoco.mj_getState(model, data, state, spec)
            states.append(state)
            times.append(float(data.time))
            qs.append([float(data.joint(name).qpos[0]) for name in source["initial"]["joint_names"]])
            pelvis = model.body("pelvis").id
            roots.append([*data.xpos[pelvis].tolist(), *data.xquat[pelvis].tolist()])
            velocities.append(data.qvel.copy())
            if video:
                video.write(obs["head_stereo_left"], sim_time_s=float(data.time))
        capture()
        with (output / "trace.jsonl").open("w", buffering=1) as trace:
            for tick in range(total):
                if not np.isclose(float(data.time)-start_time, tick/50, rtol=0, atol=1e-6):
                    raise RuntimeError("Physical50Hz tick drifted from actual simulator clock")
                reference = None
                force_before = applied_force_state(data)
                if not force_before["all_zero"]:
                    raise RuntimeError("Nonzero pre-control externally applied force")
                if tick < warmup_ticks:
                    quality_phase["teacher_tick"] = None
                    command = (warmup.command(raw.task.robot, controller, obs) if warmup else controller.hold(obs))
                    phase = warmup.phase if warmup else "warmup_measured_pose_pd"
                else:
                    quality_phase["teacher_tick"] = tick-warmup_ticks
                    source_index = command_source_index(tick-warmup_ticks, args.command_rate, len(source["commands"]))
                    reference = source["commands"][source_index]
                    command = controller.step(reference[:64],
                        {name:float(value) for name,value in zip(HAND_NAMES, reference[64:78])}, obs)
                    phase = "teacher_command"
                if training_capture and capture_tick_index(tick) is not None:
                    pelvis = model.body("pelvis").id
                    root_joint = int(model.body_jntadr[pelvis])
                    root_dof = int(model.jnt_dofadr[root_joint])
                    # Consume the actual preceding get_obs diagnostic. Do not
                    # call renderer observation_pose_diagnostics/render here.
                    renderer_diagnostic = getattr(raw.isaac,"last_observation_pose_diagnostics",None)
                    actual_renderer_poses = {"links":{name:{"position_m":data.xpos[model.body(name).id].copy(),
                        "quaternion_wxyz":data.xquat[model.body(name).id].copy()} for name in RENDERER_LINKS},
                        "cameras":{"head_stereo_left":{"position_m":data.cam_xpos[model.camera("head_stereo_left").id].copy(),
                            "rotation_usd_axes":data.cam_xmat[model.camera("head_stereo_left").id].copy()}}}
                    training_capture.append(tick50=tick,sim_time_s=float(data.time),rgb=obs["head_stereo_left"],
                        observed_joint_names=obs["joint_names"],observed_qpos=obs["joint_qpos"],
                        physical_qpos=[float(data.joint(name).qpos[0]) for name in source["model_joint_names"]],
                        physical_qvel=[float(data.joint(name).qvel[0]) for name in source["model_joint_names"]],
                        root_pose7=[*data.xpos[pelvis],*data.xquat[pelvis]],root_qvel6=data.qvel[root_dof:root_dof+6],
                        objects={role:{"pose7":[*raw.mujoco.mj_objects[role].xpos,*raw.mujoco.mj_objects[role].xquat]}
                            for role in ("target","container")},command=command if reference is not None else None,
                        renderer_pose_diagnostics=renderer_diagnostic,renderer_authoritative_poses=actual_renderer_poses,
                        observation_sim_time_s=obs.get("sim_time_s"))
                record = {"tick50": tick, "phase": phase, "sim_time_s": float(data.time),
                    "elapsed_s": float(data.time)-start_time,
                    "source_command_index": source_index if reference is not None else None,
                    "actual_body_token64": command["body_token"].tolist() if command["body_token"] is not None else None,
                    "actual_hand_targets": command["hand_targets"], "decoder_calls_before": controller.decoder_calls,
                    "measured_before": measured_task_state(raw), "physics_state_row_before": len(states)-1,
                    "applied_forces_before": force_before}
                obs, reward, terminated, truncated, info = env.step(command)
                capture()
                pelvis = model.body("pelvis").id
                height = float(data.xpos[pelvis, 2])
                tilt = float(np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3,3)[2,2], -1, 1))))
                finite = all(np.isfinite(x).all() for x in (data.qpos, data.qvel, data.qacc, data.ctrl))
                result["finite_physics"] = bool(result["finite_physics"] and finite)
                result["official_success_ever"] = bool(result["official_success_ever"] or terminated)
                result["fell"] = bool(result["fell"] or height < .35 or tilt > 75)
                record.update(next_sim_time_s=float(data.time), measured_after=measured_task_state(raw),
                    reward=float(reward), official_success=bool(terminated), truncated=bool(truncated),
                    decoder_calls_after=controller.decoder_calls, pelvis_z_m=height, pelvis_tilt_degrees=tilt,
                    finite_physics=bool(finite), physics_state_row_after=len(states)-1)
                if args.warmup_control == PADDED_WARMUP_MODE:
                    record_padded_decoder_step(warmup,controller,record,is_warmup=reference is None)
                record["applied_forces_after"] = applied_force_state(data)
                if not record["applied_forces_after"]["all_zero"]:
                    raise RuntimeError("Nonzero post-control externally applied force")
                trace.write(json_text(record)+"\n")
                result["completed_control_ticks"] = tick+1
                if (tick+1) % 100 == 0:
                    print(json.dumps({"completed_ticks":tick+1,"requested_ticks":total,
                        "sim_time_s":float(data.time),"phase":phase,"official_success_ever":result["official_success_ever"]}), flush=True)
                if not finite:
                    raise RuntimeError("Nonfinite physical state")
                if truncated:
                    raise RuntimeError("Unexpected environment truncation")
        result.update(status="teacher_diagnostic_completed", physical_outcome_status="source_end",
            decoder_calls=controller.decoder_calls, clipped_torque_elements=controller.clipped_torque_elements,
            final_measured_state=measured_task_state(raw), simulated_seconds=float(data.time)-start_time,
            source_commands_executed=len(source["commands"]),
            warmup_control=warmup.receipt if warmup else {"kind":"measured_pose_pd", "seconds":args.warmup_seconds})
        quality = evaluate_capture(quality_rows, initial_target_z_m=initial_target_z)
        write_json(output / "physical_quality.json", quality)
        result["physical_quality"] = quality
    except Exception as error:
        result.update(status="error", error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if "quality_stream" in locals():
                quality_stream.close()
            if "force_stream" in locals():
                force_stream.close()
                result["applied_force_checks"] = {"path":"applied_force_samples.jsonl",
                    "sha256":file_sha256(output / "applied_force_samples.jsonl"),"sample_count":force_count,
                    "expected_sample_count":4*result.get("completed_control_ticks",0),
                    "sampling_hz":200,"all_zero_verified":result["status"]=="teacher_diagnostic_completed"
                        and force_count==4*result.get("completed_control_ticks",0),
                    "force_values_actively_cleared_during_control":False}
                result["force_guard"] = {**result["applied_force_checks"],
                    "steps":force_count,"target_robot_attachment":False,
                    "equality_count":len(result["physics_model"]["equalities"]),
                    "all_applied_forces_zero":result["applied_force_checks"]["all_zero_verified"]}
            result["code_unchanged_during_run"] = all(file_sha256(path)==sha
                for path,sha in result["capture_code_sha256"].items())
            if not result["code_unchanged_during_run"]:
                result.update(status="error",error="Capture source changed during run; raw capture rejected")
            if "teacher_reference" in source:
                try:
                    result["teacher_reference_inputs_unchanged"] = all(file_sha256(path)==identity
                        for path,identity in source["teacher_reference_input_sha256"].items())
                except OSError as error:
                    result["teacher_reference_inputs_unchanged"] = False
                    result["teacher_reference_input_error"] = repr(error)
                if not result["teacher_reference_inputs_unchanged"]:
                    result.update(status="error",error="Modified teacher inputs changed during run; raw capture rejected")
            try:
                _,end_asset_sha = bound_calibration_assets(resource_data_root,calibration_identities)
                result["calibration_source_assets_unchanged"] = end_asset_sha==calibration_asset_sha
            except (OSError,ValueError) as error:
                result["calibration_source_assets_unchanged"] = False
                result["calibration_source_asset_error"] = repr(error)
            if not result["calibration_source_assets_unchanged"]:
                result.update(status="error",error="Actual calibration source assets changed during run; raw capture rejected")
            if training_capture:
                training_capture.metadata.update(code_unchanged_during_run=result["code_unchanged_during_run"],
                    applied_force_checks=result.get("applied_force_checks"),force_guard=result.get("force_guard"),
                    physics_model=result["physics_model"],
                    calibration_source_assets_unchanged=result["calibration_source_assets_unchanged"])
                if "teacher_reference" in result:
                    training_capture.metadata["teacher_reference_inputs_unchanged"] = result["teacher_reference_inputs_unchanged"]
                if args.warmup_control == PADDED_WARMUP_MODE and warmup is not None and warmup.startup_decoder is not None:
                    training_capture.metadata.update(startup_history_padding_receipt=deepcopy(warmup.startup_decoder.receipt),
                        warmup_control_actual_kind=PADDED_WARMUP_KIND)
                result["training_capture"] = training_capture.close(completed=result["status"]=="teacher_diagnostic_completed")
            if states:
                np.savez_compressed(output / "trajectory.npz", physics_state_spec=np.asarray(spec),
                    physics_state_integration=np.stack(states), sim_time_s=np.asarray(times),
                    joint_names=np.asarray(source["initial"]["joint_names"]), actual_joint_q43=np.asarray(qs),
                    root_pose7=np.asarray(roots), actual_qvel=np.asarray(velocities))
                result["trajectory"] = {"rows":len(states),"sha256":file_sha256(output / "trajectory.npz"),
                    "scope":"Every actual pre/post50Hz state including real warmup; mjSTATE_INTEGRATION permits independent replay checks"}
            if video:
                result["video"] = video.close()
                result["rgb_collected"] = bool(result["video"]["frames"])
        except Exception as error:
            result.update(status="error", artifact_finalization_error=repr(error))
            raise
        finally:
            result["ended_utc"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "result.json", result)
            if env is not None:
                env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sonic-resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-rate", type=int, choices=(30,50), required=True)
    parser.add_argument("--source-episode", type=int, default=None, help="Original TRAIN source ID; default0 without candidate")
    parser.add_argument("--candidate-plan", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--teacher-reference",type=Path,
        help="Explicit candidate-bound TRAIN modified-teacher JSON; original prefix/reset/ancestry retained")
    parser.add_argument("--capture-training", nargs="?", const=Path("capture"), type=Path,
        help="Isaac-only raw training capture directory; flag alone uses OUTPUT/capture")
    parser.add_argument("--seed", type=int, default=None, help="Candidate physics_seed or default20261022")
    parser.add_argument("--render-mode", choices=("none", "isaac"), default="none")
    parser.add_argument("--warmup-control", choices=("measured_pose_pd","live_pose_sonic_v1_1",PADDED_WARMUP_MODE), default="measured_pose_pd")
    parser.add_argument("--warmup-seconds", type=float, default=1.)
    parser.add_argument("--pd-profile", choices=("fixed_cpp_baseline","official_v1_1_ankle_pitch_1_5"), default="fixed_cpp_baseline")
    parser.add_argument("--hand-torque-profile", choices=("mjcf_limits","recording_0p7_nm"), default="mjcf_limits")
    args = parser.parse_args()
    if (not args.output.is_absolute() or not math.isfinite(args.warmup_seconds) or args.warmup_seconds < .2
            or abs(args.warmup_seconds*50-round(args.warmup_seconds*50))>1e-6):
        parser.error("Need absolute output and at least ten true50Hz warmup ticks")
    run(args)


if __name__ == "__main__":
    main()
