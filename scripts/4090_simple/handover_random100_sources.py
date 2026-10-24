"""Identity-bound raw Handover references for new, independently audited runs.

These are raw trajectory candidates, not accepted demonstrations. The official
rich ``simple-teleop`` archive is distinct from the older ``simple`` archive.
No recorded EEF field is treated as a measured wrist pose: both EEF columns in
this archive contain the same delayed teleoperation command. Use named FK.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

from prepare_carrybox_v11 import (BODY_NAMES, ENCODER_SHA, FUTURE_OFFSETS, SIMPLE_HAND_NAMES,
    TRAIN_HAND_NAMES, TRAIN_STATE_NAMES, derived_velocity, encoder_input, matrix_wxyz)
from prepare_handover_eval import remap_paths

REPO = Path(__file__).resolve().parents[2]
SIMPLE = REPO / "VLMExpert/third_party/SIMPLE"
RESOURCES = Path("/home/zxc/CVPR_Evaluation_runtime/resources")
DEFAULT_ARCHIVE = RESOURCES / "simple_finetune_sources/rich_teleop/G1WholebodyHandoverTeleop-v0.zip"
DEFAULT_PROVENANCE = DEFAULT_ARCHIVE.with_name("G1WholebodyHandoverTeleop-v0.receipt.json")
PREFIX = "G1WholebodyHandoverTeleop-v0/level-0"
DEFAULT_RAW_ROOT = DEFAULT_ARCHIVE.with_suffix("") / PREFIX
DEFAULT_ASSET_ROOT = RESOURCES / "simple_local_usd_v1"
ARCHIVE_SHA256 = "3c5d65c3ed9508132c6614e00293a2371099130ec11a994add18d0bb9606fb1c"
REVISION = "e78fb93cc28912a3031a10b8656d32d7f0a2b867"
SOURCE_KEY = f"USC-PSI-Lab/psi-data@{REVISION}/simple-teleop/{PREFIX}"
CONTRACT = "cvpr-handover-independent-raw-reference-v1"
HELDOUT = frozenset((39, 65, 80, 92, 94))
INCOMPLETE = frozenset((38,))
STATE_SOURCE_NAMES = tuple(BODY_NAMES) + tuple(SIMPLE_HAND_NAMES)
OBJECT_TERMS = ("pos_x", "pos_y", "pos_z", "quat_w", "quat_x", "quat_y", "quat_z")
CODE_PATHS = (
    SIMPLE / "src/simple/cli/teleop_decoupled_wbc.py",
    SIMPLE / "src/simple/agents/pico_decoupled_agent.py",
    SIMPLE / "src/simple/robots/g1_sonic.py",
    SIMPLE / "third_party/decoupled_wbc/control/robot_model/robot_model.py",
    SIMPLE / "src/simple/engines/mujoco.py",
    SIMPLE / "src/simple/assets/graspnet.py",
    SIMPLE / "src/simple/assets/objects.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def finite_array(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == shape and np.isfinite(result).all(), f"Require finite {name}{shape}")
    return result


class RawSourceCatalog:
    """Verify the pinned archive and extracted metadata before selecting sources."""
    def __init__(self, raw_root=DEFAULT_RAW_ROOT, *, archive_path=DEFAULT_ARCHIVE,
                 provenance_path=DEFAULT_PROVENANCE):
        self.root = Path(raw_root).resolve(strict=True)
        self.archive = Path(archive_path).resolve(strict=True)
        self.provenance = Path(provenance_path).resolve(strict=True)
        receipt = read(self.provenance)
        expected_url = f"https://huggingface.co/datasets/USC-PSI-Lab/psi-data/resolve/{REVISION}/simple-teleop/G1WholebodyHandoverTeleop-v0.zip"
        require(receipt.get("source_url") == expected_url and receipt.get("source_revision") == REVISION
                and receipt.get("archive_sha256") == receipt.get("hf_lfs_sha256") == ARCHIVE_SHA256,
                "Wrong archive namespace, revision or official LFS identity")
        require(sha256(self.archive) == ARCHIVE_SHA256, "Official rich archive bytes changed")
        self.sources = {str(self.archive): ARCHIVE_SHA256, str(self.provenance): sha256(self.provenance)}
        self.bind_member("meta/info.json"); self.bind_member("meta/episodes.jsonl")
        self.metadata_sources = dict(self.sources)
        self.info = read(self.root / "meta/info.json")
        self.episodes = [json.loads(line) for line in (self.root / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        require(self.info.get("fps") == 50 and self.info.get("total_episodes") == len(self.episodes) == 100,
                "Require the complete native50Hz rich100 source metadata")
        self.by_id = {episode["episode_index"]: episode for episode in self.episodes}
        require(set(self.by_id) == set(range(100)), "Raw100 episode identities differ")
        state = self.info["features"]["observation.state"]
        action = self.info["features"]["action"]
        require(state.get("shape") == action.get("shape") == [43]
                and tuple(state["names"]) == STATE_SOURCE_NAMES
                and len(action["names"]) == len(set(action["names"])) == 43
                and set(action["names"]) == set(STATE_SOURCE_NAMES), "Named raw state/action43 schema differs")

    def bind_member(self, relative):
        name = Path(relative)
        require(not name.is_absolute() and ".." not in name.parts, "Unsafe archive member")
        path = (self.root / name).resolve(strict=True)
        require(path.is_relative_to(self.root) and path.is_file(), "Raw member escapes source root")
        with zipfile.ZipFile(self.archive) as archive:
            content = archive.read(f"{PREFIX}/{name.as_posix()}")
        expected = hashlib.sha256(content).hexdigest()
        require(sha256(path) == expected, f"Extracted raw bytes differ from official archive: {name}")
        self.sources[str(path)] = expected
        return path

    def parquet(self, source_id):
        relative = self.info["data_path"].format(episode_index=source_id,
            episode_chunk=source_id // self.info.get("chunks_size", 1000))
        return self.bind_member(relative)


def hand_commands(actions, action_names):
    """Keep declared targets and the explicitly inferred upstream physical mapping.

    The legacy PICO path writes natural thumb/index/middle arrays against MJCF
    thumb/middle/index traversal. This is a bound-code inference, not proof of
    the historical per-episode runtime. No angle clipping/scaling is applied.
    """
    actions = np.asarray(actions, np.float64)
    require(actions.ndim == 2 and actions.shape[1] == 43 and np.isfinite(actions).all()
            and len(action_names) == len(set(action_names)) == 43
            and set(action_names) == set(STATE_SOURCE_NAMES), "Named finite raw action43 required")
    indices = [action_names.index(name) for name in SIMPLE_HAND_NAMES]
    declared = actions[:, indices].copy()
    swap = np.array([0, 1, 2, 5, 6, 3, 4, 7, 8, 9, 12, 13, 10, 11])
    effective = declared[:, swap].copy()
    return declared, effective, {"output_joint_names": list(SIMPLE_HAND_NAMES),
        "declared_raw_indices": indices, "effective_execution_raw_indices": np.asarray(indices)[swap].tolist(),
        "semantics": "upstream_collection_effective_index_middle_swap_inferred_from_bound_code",
        "historical_per_episode_recipe_binding_verified": False,
        "target_angles_scaled_or_clipped": False, "new_execution_hand_torque_cap_nm": .7,
        "torque_limit_semantics": "0.7Nm clips post-PD torque, never multiplies target angles",
        "measured_hand_state_used_as_command": False}


def _current_object_mass_policy():
    path = SIMPLE / "src/simple/engines/mujoco.py"
    functions = [node for node in ast.walk(ast.parse(path.read_text()))
                 if isinstance(node, ast.FunctionDef) and node.name == "_build_object"]
    require(len(functions) == 1, "Unknown MuJoCo object construction policy")
    expressions = [kw.value for node in ast.walk(functions[0]) if isinstance(node, ast.Call)
                   for kw in node.keywords if kw.arg == "mass"]
    require(len(expressions) == 1 and ast.unparse(expressions[0]) == "0.1 / num_convex",
            "Object mass policy changed; explicit new mass evidence required")
    return .1, {"value_kg": .1, "origin": "bound_current_MujocoEngine_object_builder_not_recorded_source_mass",
                "engine_path": str(path), "engine_sha256": sha256(path),
                "actual_compiled_model_mass_must_be_verified_by_initialize": True}


def _bind_assets(config, asset_root):
    asset_root = Path(asset_root).resolve(strict=True)
    bound = {str(path): sha256(path) for path in CODE_PATHS}
    xml = asset_root / "robots/g1_sonic/g1_29dof_with_hand.xml"
    tree = ET.parse(xml)
    model_hands = [[node.get("name") for node in tree.findall(".//joint")
                    if (node.get("name") or "").startswith(side + "_hand_")]
                   for side in ("left", "right")]
    require([name for side in model_hands for name in side] == list(TRAIN_HAND_NAMES),
            "Actual MJCF hand traversal differs from the documented execution mapping")
    bound[str(xml)] = bound[str(xml.resolve(strict=True))] = sha256(xml)
    for role in ("target", "container"):
        actor = config["layout"]["actors"][role]
        asset = actor["asset"]
        family = {"graspnet1b": "graspnet", "objects": "objects"}.get(asset["res_id"])
        require(family is not None and str(asset["uid"]).isdigit(), "Unsupported raw asset family/identity")
        usd = Path(asset["usd_path"])
        require(usd.is_absolute() and ".." not in usd.parts and usd.is_relative_to(asset_root),
                "Object USD escaped the selected resource tree")
        # The registered local v2 resource tree deliberately aliases immutable
        # source resources. Bind both the selected alias and resolved payload.
        bound[str(usd)] = bound[str(usd.resolve(strict=True))] = sha256(usd)
        directory = asset_root / f"assets/{family}/models/{int(asset['uid']):03d}/coacd_0.05"
        meshes = sorted(directory.rglob("convex_piece*")) if directory.is_dir() else []
        require(meshes and all(path.is_file() for path in meshes), "Missing actual object collision meshes")
        for path in meshes:
            bound[str(path)] = bound[str(path.resolve(strict=True))] = sha256(path)
    def visit(value):
        if isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, list):
            for item in value: visit(item)
        elif isinstance(value, str) and Path(value).is_absolute() and Path(value).is_file():
            path = Path(value)
            require(".." not in path.parts and path.is_relative_to(asset_root), "Remapped configuration file escaped assets")
            bound[str(path)] = bound[str(path.resolve(strict=True))] = sha256(path)
    visit(config)
    return bound


def load_raw_source(source_id, *, raw_root=DEFAULT_RAW_ROOT, asset_root=DEFAULT_ASSET_ROOT,
                    archive_path=DEFAULT_ARCHIVE, provenance_path=DEFAULT_PROVENANCE, catalog=None):
    """Return a complete named real reference, not old-final50/source24 data."""
    require(type(source_id) is int and 0 <= source_id < 100 and source_id not in HELDOUT,
            "Select an explicit original TRAIN source, never the five protected VAL identities")
    require(source_id not in INCOMPLETE, "Source38 is a one-frame incomplete record, not a demonstration")
    catalog = catalog or RawSourceCatalog(raw_root, archive_path=archive_path, provenance_path=provenance_path)
    import pyarrow.parquet as pq
    episode = catalog.by_id[source_id]
    n = episode["length"]
    require(type(n) is int and n >= 60, "Raw source lacks complete native references")
    path = catalog.parquet(source_id)
    columns = ["observation.state", "action", "observation.base_pose", "observation.base_vel",
               "observation.object_poses", "frame_index", "episode_index", "timestamp"]
    table = pq.read_table(path, columns=columns).to_pydict()
    states = finite_array(table["observation.state"], (n, 43), "state43")
    actions = finite_array(table["action"], (n, 43), "action43")
    roots = finite_array(table["observation.base_pose"], (n, 7), "base pose")
    root_vel = finite_array(table["observation.base_vel"], (n, 6), "recorded root qvel")
    object_values = finite_array(table["observation.object_poses"], (n, 14), "object poses")
    times = finite_array(table["timestamp"], (n,), "timestamps")
    require(np.array_equal(table["frame_index"], np.arange(n))
            and np.array_equal(table["episode_index"], np.full(n, source_id))
            and np.allclose(times, np.arange(n) / 50, rtol=0, atol=2e-5), "Native frame/source/time identity differs")
    matrix_wxyz(roots[:, 3:])
    state_names = catalog.info["features"]["observation.state"]["names"]
    permutation = [state_names.index(name) for name in TRAIN_STATE_NAMES]
    states = states[:, permutation]
    velocities = derived_velocity(states, times)
    declared, effective, hand_proof = hand_commands(actions, catalog.info["features"]["action"]["names"])
    object_names = catalog.info["features"]["observation.object_poses"]["names"]
    require(len(object_names) == len(set(object_names)) == 14, "Ambiguous named object pose fields")
    objects = {role: object_values[:, [object_names.index(f"{role}.{term}") for term in OBJECT_TERMS]].copy()
               for role in ("target", "container")}
    for value in objects.values(): matrix_wxyz(value[:, 3:])
    config = json.loads(episode["environment_config"])
    require(config.get("uid") in ("g1_wholebody_handover_sonic", "g1_wholebody_handover_teleop")
            and config["robot_cfg"]["uid"] == "g1_sonic", "Wrong raw task/robot environment")
    original_uid = config["uid"]
    config["uid"] = "g1_wholebody_handover_teleop"
    for role, values in objects.items():
        actor = config["layout"]["actors"][role]
        initial_pose = {"position": values[0, :3].tolist(), "quaternion": values[0, 3:].tolist()}
        actor["pose"] = deepcopy(initial_pose)
        key = f"container_{actor['uid']}" if role == "container" else str(actor["uid"])
        config["dr_state_dict"]["spatial"][key] = deepcopy(initial_pose)
    asset_root = Path(asset_root).resolve(strict=True)
    config = remap_paths(config, asset_root, asset_root)
    assets = _bind_assets(config, asset_root)
    mass, mass_proof = _current_object_mass_policy()
    sources = {**catalog.metadata_sources, str(path): catalog.sources[str(path)],
               **assets, str(Path(__file__).resolve()): sha256(__file__),
               str(Path(__file__).with_name("prepare_carrybox_v11.py")): sha256(Path(__file__).with_name("prepare_carrybox_v11.py"))}
    initial = {"source_episode_index": source_id, "joint_names": list(TRAIN_STATE_NAMES),
        "joint_qpos": states[0].tolist(), "joint_qvel": velocities[0].tolist(),
        "root_pose7": roots[0].tolist(), "root_qvel6": root_vel[0].tolist(),
        "objects": {role: {"pose7": values[0].tolist(), "mass": mass} for role, values in objects.items()},
        "sources": sources}
    values = {"reference_full_state43": states, "reference_full_body_dq": velocities[:, :29],
        "reference_full_root_xyz": roots[:, :3].copy(), "reference_full_root_wxyz": roots[:, 3:].copy(),
        "reference_full_root_qvel6": root_vel, "reference_full_object_poses": objects,
        "reference_full_hand_commands14": effective, "reference_full_declared_hand_commands14": declared,
        "reference_full_hand_command_names": list(SIMPLE_HAND_NAMES),
        "reference_full_state_joint_names": list(TRAIN_STATE_NAMES)}
    evidence = {"contract": CONTRACT, "source_dataset_key": SOURCE_KEY, "source_id": source_id,
        "split": "train", "protected_VAL_excluded": sorted(HELDOUT), "source_frames": n,
        "archive_sha256": ARCHIVE_SHA256, "parquet_sha256": sources[str(path)],
        "environment_source": "official archive-bound meta/episodes.jsonl.environment_config",
        "initial_object_pose_source": "actual named Parquet row0, never final layout.actor pose",
        "original_task_uid": original_uid, "object_mass_policy": mass_proof, "hand_commands": hand_proof,
        "source_joint_velocity": "central/one-sided finite differences of real measured native q/t; not stored qvel",
        "root_velocity": "recorded MuJoCo qvel[:6] from upstream proprio serializer",
        "EEF_policy": "observation.eef_state and action.eef are delayed command proxies; neither used as measured FK",
        "complete_original_mode0_current_range_inclusive": [0, n - 46],
        "original_future_offsets50": FUTURE_OFFSETS.tolist(), "original_tail_padding_allowed": False,
        "historical_controller_rng_integration_state_restored": False,
        "new_successful_demonstration": False, "independent_human_session_proven": False,
        "source_scope": "Distinct raw trajectory candidate; new object-adaptive physical execution and full acceptance still required",
        "asset_binding_scope": "Bound actor USD, collision pieces, configuration-referenced files and current source implementation; complete rendered model/assets verified separately during collection"}
    return {"source_id": source_id, "source_episode_index": source_id, "config": config,
        "initial": initial, "values": values, "hand_commands14": effective,
        "hand_joint_names": list(SIMPLE_HAND_NAMES), "native_frame_indices": np.arange(n, dtype=np.int64),
        "native_timestamps": times, "row": deepcopy(episode), "sources": sources, "evidence": evidence}


def reference_wrist_fk(source, model, *, frames=None):
    """CPU FK from named measured q/root, on private MjData; never steps physics."""
    import mujoco
    values = source["values"]
    n = len(values["reference_full_state43"])
    frames = np.arange(n, dtype=np.int64) if frames is None else np.asarray(frames)
    require(frames.ndim == 1 and frames.dtype.kind in "iu" and len(frames)
            and np.all((frames >= 0) & (frames < n)), "FK requires existing original frame indices")
    data = mujoco.MjData(model)
    joints = [model.joint(name) for name in values["reference_full_state_joint_names"]]
    require(all(joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE for joint in joints), "FK needs named scalar hinges")
    qadr = [int(joint.qposadr[0]) for joint in joints]
    root = model.joint("floating_base_joint")
    require(root.type[0] == mujoco.mjtJoint.mjJNT_FREE, "FK needs the real floating root")
    root_adr = int(root.qposadr[0])
    bodies = [model.body(side + "_wrist_yaw_link").id for side in ("left", "right")]
    poses = np.empty((len(frames), 2, 7), np.float64)
    for output, frame in enumerate(frames):
        data.qpos[:] = model.qpos0
        data.qpos[qadr] = values["reference_full_state43"][frame]
        data.qpos[root_adr:root_adr + 7] = np.r_[values["reference_full_root_xyz"][frame], values["reference_full_root_wxyz"][frame]]
        mujoco.mj_forward(model, data)
        for side, body in enumerate(bodies): poses[output, side] = np.r_[data.xpos[body], data.xquat[body]]
    return {"frame_indices": frames, "left_pose7": poses[:, 0], "right_pose7": poses[:, 1],
            "measurement": "named measured q/root forwarded on private MjData, not stored EEF command fields"}


class OfficialMode0Encoder:
    """CPU-only official encoder sanity tool; requires real current+45 future."""
    def __init__(self, sonic_resources):
        import onnxruntime as ort
        path = Path(sonic_resources).resolve(strict=True) / "model_encoder.onnx"
        require(sha256(path) == ENCODER_SHA, "Official SONIC1.1 encoder SHA differs")
        options = ort.SessionOptions(); options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        require(self.session.get_providers() == ["CPUExecutionProvider"]
                and list(self.session.get_inputs()[0].shape) == [1, 1751]
                and list(self.session.get_outputs()[0].shape) == [1, 64], "Official CPU mode0 encoder contract differs")
        self.path = path

    def encode(self, source, frame):
        require(type(frame) is int, "Original encoder frame must be an integer")
        values = source["values"]
        inputs = encoder_input(values["reference_full_state43"][:, :29], values["reference_full_body_dq"],
                               values["reference_full_root_wxyz"], frame)
        token = self.session.run(None, {self.session.get_inputs()[0].name: inputs})[0]
        require(token.shape == (1, 64) and np.isfinite(token).all(), "Nonfinite official encoder output")
        return token[0], {"mode": "original_measured_reference_mode0", "source_id": source["source_id"],
            "current_native_frame": frame, "real_reference_frames": (frame + FUTURE_OFFSETS).tolist(),
            "encoder_sha256": ENCODER_SHA, "encoder_input_sha256": hashlib.sha256(inputs.tobytes()).hexdigest(),
            "encoder_provider": "CPUExecutionProvider", "reference_tail_synthesized": False}


def inventory(catalog=None):
    catalog = catalog or RawSourceCatalog()
    records = []
    for episode in catalog.episodes:
        source_id, n = episode["episode_index"], episode["length"]
        path = catalog.parquet(source_id)
        status = "protected_VAL" if source_id in HELDOUT else "incomplete_raw_record" if source_id in INCOMPLETE or n < 60 else "TRAIN_raw_reference_candidate"
        records.append({"source_id": source_id, "frames": n, "status": status,
                        "parquet_sha256": catalog.sources[str(path)], "new_demonstration_accepted": False})
    return {"contract": CONTRACT, "source_dataset_key": SOURCE_KEY, "archive_sha256": ARCHIVE_SHA256,
        "raw_episodes": len(records), "raw_frames": sum(item["frames"] for item in records),
        "TRAIN_candidate_count": sum(item["status"] == "TRAIN_raw_reference_candidate" for item in records),
        "protected_VAL": sorted(HELDOUT), "incomplete_raw_sources": sorted(INCOMPLETE),
        "independent_human_sessions_proven": False, "new_successes_accepted": 0,
        "records": records, "sources": dict(catalog.sources)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--asset-root", type=Path, default=DEFAULT_ASSET_ROOT)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--source-ids", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--sonic-resources", type=Path, help="Optional actual CPU ONNX sanity, never loads a GPU provider")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--output", type=Path, help="Fresh external JSON evidence; no output rewrites")
    args = parser.parse_args()
    catalog = RawSourceCatalog(args.raw_root, archive_path=args.archive, provenance_path=args.provenance)
    encoder = OfficialMode0Encoder(args.sonic_resources) if args.sonic_resources else None
    report = {"contract": CONTRACT, "inventory": inventory(catalog) if args.inventory else None, "references": []}
    for source_id in args.source_ids:
        source = load_raw_source(source_id, catalog=catalog, asset_root=args.asset_root)
        evidence = deepcopy(source["evidence"])
        if encoder:
            token, diagnostic = encoder.encode(source, 0)
            evidence["official_mode0_CPU_sanity"] = {**diagnostic, "token64": token.tolist()}
        report["references"].append(evidence)
    if args.output:
        output = args.output.resolve()
        require(not output.exists() and not output.is_relative_to(REPO)
                and not output.is_relative_to(catalog.root) and not output.is_relative_to(args.asset_root.resolve()),
                "Use a fresh external report outside code, raw source and assets")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream: stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    else:
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
