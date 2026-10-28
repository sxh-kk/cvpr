"""One-time Mobile P&P initialization from an admitted human capture's row zero.

No commands, images, event rows or later integration states are opened. The
original admission and frozen capture-start receipt authenticate the first row;
the whole states payload is not re-read to repeat the completed dataset audit.
This module never constructs an environment, advances physics or replays actions.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from initial_physics_state import _model_fields


CONTRACT = "cvpr-mobile-pp-initialization-v1"
TASK_ID = "simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
TASK_UID = "g1_wholebody_locomotion_pick_between_tables_teleop"
SOURCE_PACK_SHA256 = "5208307c097baf7bae6553cc7fdc5db51d0a6ef21efe9ab4c9b9e6e2599e6957"
PACK_CONTRACT = "cvpr-pico-mobile-pp-lerobot-pack-v1"
ADMISSION_CONTRACT = "cvpr-live-pico-mobile-pp-independent-admission-v1"
CAPTURE_CONTRACT = "cvpr-live-pico-deferred-physical-capture-v1"
START_CONTRACT = "cvpr-pico-frozen-capture-start-v1"
SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)
PATH_FIELDS = {"model.mesh_pathadr", "model.npaths", "model.paths", "model.nbuffer"}


def require(value, message):
    if not value:
        raise ValueError(message)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _json_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _mesh_paths(model):
    """Account for the entire path table in mesh/reference order, fail closed."""
    raw, cursor, paths = bytes(model.paths), 0, []
    require(len(raw) == model.npaths, "Model path byte count differs")
    for address in model.mesh_pathadr:
        address = int(address)
        if address == -1:
            paths.append(None)
            continue
        require(address == cursor, "Unaccounted, aliased or reordered mesh path table")
        end = raw.find(b"\0", address)
        require(end >= 0, "Unterminated mesh resource path")
        paths.append(raw[address:end].decode("utf-8", errors="strict"))
        cursor = end + 1
    require(cursor == len(raw), "Only complete mesh-path tables can be relocated")
    return paths


def path_buffer_proof(source_npaths, live_npaths, source_nbuffer, live_nbuffer):
    # MuJoCo3.3.6 aligns each MJMODEL_POINTERS allocation to64B. The paths
    # allocation is followed by B_rownnz; all other dimensions are required
    # equal before this proof. Its padding is therefore the only buffer delta.
    # https://github.com/google-deepmind/mujoco/blob/3.3.6/src/engine/engine_io.c#L320-L358
    require(mujoco.__version__ == "3.3.6", "Mesh-path buffer normalization requires verified MuJoCo3.3.6 layout")
    expected = ((live_npaths + 63) // 64 - (source_npaths + 63) // 64) * 64
    require(live_nbuffer - source_nbuffer == expected, "Model buffer difference is not exclusively aligned path storage")
    return {"alignment_bytes": 64, "source_npaths": source_npaths, "live_npaths": live_npaths,
        "source_nbuffer": source_nbuffer, "live_nbuffer": live_nbuffer, "expected_buffer_delta": expected,
        "actual_buffer_delta": live_nbuffer - source_nbuffer,
        "basis": "MuJoCo3.3.6 MJMODEL_POINTERS: identical other dimensions,64B alignment, paths followed by B_rownnz"}


def _asset_identity(path, root, cache):
    resource = Path(path)
    if not resource.is_absolute():
        require(root is not None and ".." not in resource.parts, "Relative mesh path needs its explicit compiler mesh root")
        resource = Path(root) / resource
    resolved = resource.resolve(strict=True)
    require(resolved.is_file(), "Mesh resource is not a file")
    before = file_identity(resolved)
    key = tuple(before.values())
    if key not in cache:
        cache[key] = _sha(resolved)
    require(file_identity(resolved) == before, "Mesh resource changed during identity proof")
    return {"path": str(resource), "resolved_path": str(resolved), "sha256": cache[key], **before}


def compare_mobile_models(source, live, *, source_mesh_root=None, live_mesh_root=None):
    """Permit path relocation only after exact physical/geometry and file proof.

    The shared initial_physics_state comparator remains unchanged. Every
    non-path static field, including embedded mesh arrays and name/order bytes,
    is still compared exactly. Unknown path-table layouts or versions fail.
    """
    original, actual = _model_fields(source), _model_fields(live)
    differences = [k for k in sorted(set(original) | set(actual)) if original.get(k) != actual.get(k)]
    forbidden = set(differences) - PATH_FIELDS
    require(not forbidden, "Initial Mobile physics/geometry model is incompatible: " + ", ".join(sorted(forbidden)[:24]))
    normalized_source = {k: v for k, v in original.items() if k not in PATH_FIELDS}
    normalized_live = {k: v for k, v in actual.items() if k not in PATH_FIELDS}
    receipt = {"compatible": True, "comparison": "exact static physics/geometry/name fields; conditional verified mesh-path relocation",
        "mujoco_version": mujoco.__version__, "fields_compared": len(original),
        "source_static_sha256": _json_sha(original), "target_static_sha256": _json_sha(actual),
        "source_normalized_sha256": _json_sha(normalized_source), "target_normalized_sha256": _json_sha(normalized_live),
        "different_fields": differences, "path_relocation_used": bool(differences),
        "excluded_compilation_signature": {"source": int(source.signature), "target": int(live.signature),
            "reason": "mjSpec compilation linkage; MJB reload sets it to zero; all physical fields compared"}}
    if not differences:
        return receipt
    require(source.nbody > 0 and live.nbody > 0, "Path buffer proof needs the following nonempty body array")
    receipt["buffer_path_storage_proof"] = path_buffer_proof(source.npaths, live.npaths, source.nbuffer, live.nbuffer)
    original_paths, actual_paths = _mesh_paths(source), _mesh_paths(live)
    require(len(original_paths) == len(actual_paths), "Mesh resource order differs")
    assets, cache = [], {}
    for i, (a, b) in enumerate(zip(original_paths, actual_paths)):
        require((a is None) == (b is None), "Inline/file mesh source kind differs")
        if a is None:
            continue
        left, right = _asset_identity(a, source_mesh_root, cache), _asset_identity(b, live_mesh_root, cache)
        require(left["sha256"] == right["sha256"] and left["size_bytes"] == right["size_bytes"],
                "Mesh resource bytes differ at index " + str(i))
        assets.append({"mesh_index": i, "mesh_name": source.mesh(i).name,
            "source_model_path": a, "live_model_path": b, "source": left, "live": right})
    receipt.update(mesh_resource_identities=assets, mesh_resource_identity_sha256=_json_sha(assets),
        file_backed_meshes_verified=len(assets), source_mesh_root=str(source_mesh_root) if source_mesh_root else None,
        live_mesh_root=str(live_mesh_root) if live_mesh_root else None,
        embedded_mesh_geometry_exact=True, all_other_static_fields_exact=True)
    return receipt


def _source_robot_mesh_root(data_root):
    xml = Path(data_root) / "robots/g1_sonic/g1_29dof_with_hand.xml"
    compilers = ET.parse(xml).getroot().findall("compiler")
    require(len(compilers) == 1 and "meshdir" in compilers[0].attrib, "Missing source robot compiler mesh root")
    root = (xml.parent / compilers[0].attrib["meshdir"]).resolve(strict=True)
    return root, {"path": str(xml), "resolved_path": str(xml.resolve()), "sha256": _sha(xml), "mesh_root": str(root)}


def _regular(path):
    path = Path(path)
    require(path.is_absolute(), "Initialization paths must be absolute")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "Symlink source is not allowed")
    require(stat.S_ISREG(path.stat().st_mode), "Initialization source must be a regular file")
    return path


def _owned(root, relative):
    name = Path(relative)
    require(not name.is_absolute() and name.parts and ".." not in name.parts, "Unsafe packed evidence path")
    path = _regular(root / name)
    require(path.resolve().is_relative_to(root.resolve()), "Evidence escapes source pack")
    return path


def file_identity(path):
    """Local frozen-file identity; does not read the states payload."""
    s = Path(path).stat()
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "device": s.st_dev, "inode": s.st_ino}


def _fd_identity(stream):
    s = os.fstat(stream.fileno())
    return {"size_bytes": s.st_size, "mtime_ns": s.st_mtime_ns, "device": s.st_dev, "inode": s.st_ino}


def read_first_state(path, *, event_count, width, identity):
    """Read only the NPY header and first C-order float64 row, with no np.load."""
    path = _regular(path)
    require(set(identity) == {"size_bytes", "mtime_ns", "device", "inode"}
            and all(type(v) is int and v >= 0 for v in identity.values()), "Invalid frozen states identity")
    require(file_identity(path) == identity, "Frozen states file identity changed")
    with path.open("rb") as stream:
        require(_fd_identity(stream) == identity, "Opened states file identity changed")
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported states NPY version")
        require(shape == (event_count, width) and not fortran and dtype.str == "<f8",
                "Require the complete C-order little-endian float64 integration matrix")
        require(stream.tell() + event_count * width * 8 == identity["size_bytes"], "Truncated or trailing states payload")
        payload = stream.read(width * 8)
        require(len(payload) == width * 8, "Truncated first integration state")
        require(_fd_identity(stream) == identity and file_identity(path) == identity,
                "States identity changed during first-row read")
    value = np.frombuffer(payload, dtype="<f8").copy()
    require(np.isfinite(value).all(), "Initial integration state contains NaN/Inf")
    return value, hashlib.sha256(payload).hexdigest()


def _split_membership(split, pack):
    by_index = {e["episode_index"]: e["human_demo_uuid"] for e in pack["episodes"]}
    groups = {}
    for name, count in (("train", 95), ("eval", 5), ("train_probe", 5)):
        indices, uuids = split[name + "_episode_indices"], split[name + "_uuids"]
        require(isinstance(indices, list) and len(indices) == count and len(set(indices)) == count
                and all(type(i) is int and i in by_index for i in indices), "Invalid frozen " + name + " indices")
        require(uuids == [by_index[i] for i in indices], "Frozen split UUID/order differs: " + name)
        groups[name] = indices
    require(set(groups["train"]).isdisjoint(groups["eval"])
            and set(groups["train"]) | set(groups["eval"]) == set(range(100))
            and set(groups["train_probe"]) <= set(groups["train"]), "Frozen train/eval/probe overlap or coverage differs")
    return groups


def load_initialization(manifest, episode_index):
    """Return an initializer for one of frozen five train probes/five heldouts.

    Manifest schema is ``CONTRACT`` with task_id, source_pack_root,
    source_pack_ready_sha256, split_manifest={path,sha256}, and episodes0..9.
    Each episode has role, source_episode_index, human_demo_uuid, states_identity.
    The split JSON supplies train/eval/train_probe_episode_indices and *_uuids.
    """
    manifest = _regular(Path(manifest))
    doc = _read(manifest)
    require(doc.get("contract") == CONTRACT and doc.get("task_id") == TASK_ID,
            "Wrong Mobile initialization manifest contract/task")
    require(type(episode_index) is int and 0 <= episode_index < 10, "Require frozen initialization case0..9")
    root = Path(doc["source_pack_root"])
    require(root.is_absolute() and not any(p.is_symlink() for p in (root, *root.parents)), "Unsafe source pack root")
    pack_path = _owned(root, "PACK_READY.json")
    require(doc.get("source_pack_ready_sha256") == SOURCE_PACK_SHA256 == _sha(pack_path),
            "Require the unchanged admitted original100 PACK_READY identity")
    pack = _read(pack_path)
    require(pack.get("contract") == PACK_CONTRACT and pack.get("task_id") == TASK_ID
            and pack.get("selection_policy") == "native_only" and len(pack.get("episodes", [])) == 100
            and {e["episode_index"] for e in pack["episodes"]} == set(range(100))
            and len({e["human_demo_uuid"] for e in pack["episodes"]}) == 100,
            "Wrong original admitted Mobile100 pack")
    split_path = _regular(Path(doc["split_manifest"]["path"]))
    require(_sha(split_path) == doc["split_manifest"]["sha256"], "Frozen split manifest identity changed")
    split = _read(split_path)
    require(split.get("task_id") == TASK_ID, "Wrong split task")
    groups = _split_membership(split, pack)
    rows = doc["episodes"]
    require(isinstance(rows, dict) and set(rows) == {str(i) for i in range(10)}, "Require all ten frozen initialization cases")
    by_index = {e["episode_index"]: e for e in pack["episodes"]}
    for i in range(10):
        row = rows[str(i)]
        role, source = ("train_probe", groups["train_probe"][i]) if i < 5 else ("heldout", groups["eval"][i - 5])
        require(row.get("role") == role and row.get("source_episode_index") == source
                and row.get("human_demo_uuid") == by_index[source]["human_demo_uuid"],
                "Initialization case role/order/UUID differs from frozen split")
    row = deepcopy(rows[str(episode_index)])
    entry = by_index[row["source_episode_index"]]
    admission_path = _owned(root, entry["admission_path"])
    require(_sha(admission_path) == entry["admission_sha256"], "Original admitted evidence identity changed")
    audit = _read(admission_path)
    require(audit.get("contract") == ADMISSION_CONTRACT and audit.get("task_id") == TASK_ID
            and audit.get("human_demo_uuid") == row["human_demo_uuid"] and audit.get("selection_policy") == "native_only"
            and audit.get("status") == "independent_live_human_episode_audit_passed"
            and audit.get("eligible_for_packing") is True and audit.get("technical_blocked") is False
            and audit.get("official_task_success") is True and audit.get("selection_status") == "selected",
            "Require original admitted native-successful Mobile human source")
    for name in ("independent_native_mobile_task_success", "all_original200hz_external_forces_zero"):
        require(audit.get("checks", {}).get(name, {}).get("passed") is True, "Missing original native/zero-force admission proof")
    original = Path(entry["original_episode_directory"])
    sources = {}
    for name in ("deferred_capture/capture.json", "deferred_capture/model.mjb", "deferred_capture/states.npy", "capture_start_receipt.json"):
        source = _owned(root, entry["evidence_files"][name])
        expected = audit["bindings"].get(str(original / name))
        require(isinstance(expected, str) and len(expected) == 64, "Missing admitted source binding: " + name)
        if name != "deferred_capture/states.npy":
            require(_sha(source) == expected, "Admitted initialization evidence changed: " + name)
        sources[name] = {"path": str(source), "sha256": expected}
    capture = _read(sources["deferred_capture/capture.json"]["path"])
    metadata = capture["metadata"]
    require(capture.get("contract") == CAPTURE_CONTRACT and capture.get("requested_completed") is True
            and capture.get("failure") is None and capture.get("state_spec") == SPEC
            and capture.get("state_spec_name") == "mjSTATE_INTEGRATION" and capture.get("states_dtype") == "float64"
            and capture.get("states_path") == "states.npy" and capture.get("model_path") == "model.mjb"
            and metadata.get("human_demo_uuid") == row["human_demo_uuid"] and metadata.get("task_id") == TASK_ID
            and metadata.get("source_kind") == "live_pico_human" and metadata.get("replay_source_sha256") is None
            and metadata.get("actual_native_success_criteria") == .9,
            "Wrong complete Mobile capture/state/UUID/task identity")
    start = _read(sources["capture_start_receipt.json"]["path"])
    require(start.get("contract") == START_CONTRACT and start.get("human_demo_uuid") == row["human_demo_uuid"]
            and start.get("frozen_before_first_capture") is True and start.get("single_use") is True
            and start.get("captured_grid_samples_before_commit") == 0
            and start.get("integration_state_width") == capture["state_width"]
            and start.get("integration_state_encoding") == "little-endian float64 contiguous bytes"
            and start.get("integration_state_sha256_before") == start.get("integration_state_sha256_after")
            and start.get("sim_time_s_before") == start.get("sim_time_s_after"), "Wrong frozen original capture boundary")
    require(capture.get("states_sha256") == sources["deferred_capture/states.npy"]["sha256"]
            and capture.get("model_sha256") == sources["deferred_capture/model.mjb"]["sha256"], "Capture/admission source SHA differs")
    state, state_sha = read_first_state(sources["deferred_capture/states.npy"]["path"],
        event_count=capture["event_count"], width=capture["state_width"], identity=row["states_identity"])
    require(state_sha == start.get("integration_state_sha256_before")
            and float(state[0]) == start.get("sim_time_s_before") == capture.get("start_time"),
            "First integration state differs from admitted frozen capture start")
    model = mujoco.MjModel.from_binary_path(sources["deferred_capture/model.mjb"]["path"])
    require(state.size == mujoco.mj_stateSize(model, SPEC), "Initial state width differs from actual MJB")
    scratch = mujoco.MjData(model)
    mujoco.mj_setState(model, scratch, state, SPEC)
    require(not np.any(scratch.qfrc_applied) and not np.any(scratch.xfrc_applied), "Initial state has external forces")
    receipt = {"contract": CONTRACT, "task_id": TASK_ID, "case_index": episode_index, **row,
        "manifest_path": str(manifest), "manifest_sha256": _sha(manifest), "source_pack_ready_sha256": SOURCE_PACK_SHA256,
        "split_manifest": deepcopy(doc["split_manifest"]), "admission_sha256": entry["admission_sha256"],
        "sources": sources, "state_spec": SPEC, "state_size": int(state.size), "source_row": 0,
        "initial_state_sha256": state_sha, "source_sim_time_s": float(state[0]),
        "states_payload_read_scope": "NPY header and first row only; accepted full SHA reused with frozen file identity",
        "teacher_actions_used": False, "source_images_used": False, "future_states_read": False,
        "controller_history_injected": False, "restores_during_control": False}
    return MobilePPInitialState(model, state, capture, receipt)


class MobilePPInitialState:
    def __init__(self, model, state, capture, receipt):
        self.model, self.state = model, state.copy()
        self.state.setflags(write=False)
        # Do not carry the source's calibration/reference/warmup histories into
        # the consumer. Only static scene construction and joint/actor names
        # survive after validating the bounded provenance metadata.
        self.capture = {"joint_names": list(capture["joint_names"]), "metadata": {
            "mujoco_object_body_names": deepcopy(capture["metadata"]["mujoco_object_body_names"])}}
        self.receipt, self.applied = deepcopy(receipt), False
        self.environment_config = deepcopy(capture["metadata"]["environment_config"])
        self.seed = capture["metadata"]["physics_seed"]
        self.source_data_root = capture["metadata"].get("resource_data_root")

    def restore(self, raw):
        """Call once immediately after env.reset, before agent/server history.

        Uses SIMPLE's zero-time Isaac refresh and render APIs, never _get_obs
        (which calls mj_forward). It preserves the live model/data and views.
        """
        if self.applied or getattr(raw, "_mobile_pp_initial_state_restored", False):
            raise RuntimeError("Mobile initial state may only be restored once")
        require(getattr(raw, "step_count", None) == 0, "Restore only to a newly reset environment before control")
        require(getattr(raw.task, "uid", None) == TASK_UID and float(raw.task.success_criteria) == .9,
                "Actual environment must be the native Mobile task with threshold .9")
        require(getattr(raw, "sim_thread", None) is None, "Cannot initialize a background simulation")
        require(hashlib.sha256(self.state.astype("<f8", copy=False).tobytes()).hexdigest()
                == self.receipt["initial_state_sha256"], "Validated in-memory initial state changed")
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        require(math.isclose(float(model.opt.timestep), .005, rel_tol=0, abs_tol=0), "Mobile physics timestep differs")
        source_mesh_root = live_mesh_root = None
        source_xml = None
        if bytes(self.model.paths) != bytes(model.paths):
            require(self.source_data_root is not None, "Source resource root is required for mesh relocation")
            source_mesh_root, source_xml = _source_robot_mesh_root(self.source_data_root)
            robot_spec = getattr(raw.mujoco, "robot_mjcf", None)
            require(robot_spec is not None, "Live robot compiler spec is required for mesh relocation")
            live_mesh_root = (Path(robot_spec.modelfiledir) / robot_spec.meshdir).resolve(strict=True)
        compatibility = compare_mobile_models(self.model, model,
            source_mesh_root=source_mesh_root, live_mesh_root=live_mesh_root)
        if source_xml is not None:
            compatibility["source_robot_xml_resolution"] = source_xml
        names = self.capture["joint_names"]
        actual_named = raw.mujoco.get_robot_qpos()
        require(len(names) == 43 and len(set(names)) == 43
                and isinstance(actual_named, dict) and len(actual_named) == 43
                and set(actual_named) == set(names), "Actual robot named joint membership differs")
        # Captures use canonical training43 (middle before index); SIMPLE's
        # observation order places index before middle. Never treat either
        # vector as a positional match: authenticate the same named physical
        # addresses, then retain SIMPLE's actual order for state43_to45.
        actual_names = tuple(actual_named)
        joint_bindings = []
        for name in names:
            left, right = self.model.joint(name), model.joint(name)
            a = (int(left.id), int(left.qposadr[0]), int(left.dofadr[0]))
            b = (int(right.id), int(right.qposadr[0]), int(right.dofadr[0]))
            require(a == b, "Named robot physical joint addresses differ: " + name)
            require(float(actual_named[name]) == float(data.qpos[b[1]]),
                    "Robot observation is not the actual named physical joint: " + name)
            joint_bindings.append({"name": name, "joint_id": a[0], "qpos_address": a[1],
                "dof_address": a[2], "observation_index": actual_names.index(name)})
        bindings = self.capture["metadata"]["mujoco_object_body_names"]
        require(set(raw.mujoco.mj_objects) == set(bindings)
                and all(raw.mujoco.mj_objects[k].name == v for k, v in bindings.items()), "Actual actor role/body order differs")
        for name in ("reward", "_init_target_height", "_contact_started"):
            require(hasattr(raw.task, name), "Missing native Mobile progress field: " + name)
        isaac = getattr(raw, "isaac", None)
        require(isaac is None or (getattr(raw, "sync_mujoco_observations", False)
                and getattr(raw, "isaac_observation_only", False)), "Require zero-time observation-only RTX refresh")
        require(callable(raw._render_frame) and callable(raw._get_info), "Missing fresh SIMPLE observation interfaces")
        before_time = float(data.time)
        self.applied = True
        raw._mobile_pp_initial_state_restored = True
        # MuJoCo's Python binding requires a writeable argument even though the
        # C API consumes it. Keep the authenticated source copy immutable.
        mujoco.mj_setState(model, data, self.state.copy(), SPEC)

        def unchanged(label):
            current = np.empty_like(self.state)
            mujoco.mj_getState(model, data, current, SPEC)
            require(np.array_equal(current, self.state), label + " changed the complete initial integration state")
            return hashlib.sha256(current.astype("<f8", copy=False).tobytes()).hexdigest()

        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        mujoco.mj_comVel(model, data)
        mujoco.mj_camlight(model, data)
        unchanged("Derived kinematics/velocity refresh")
        raw.task._init_target_height, raw.task._contact_started, raw.task.reward = None, False, 0.
        raw._success = False
        if hasattr(raw, "last_reward"):
            raw.last_reward = 0.
        if isaac is not None:
            isaac.refresh_from_mujoco(raw.mujoco, update_kinematics=True)
        unchanged("Zero-time RTX refresh")
        named = raw.mujoco.get_robot_qpos()
        require(tuple(named) == actual_names and all(float(named[x["name"]]) == float(data.qpos[x["qpos_address"]])
                for x in joint_bindings), "Fresh observation named joint binding changed")
        frames = raw._render_frame()
        info = raw._get_info()
        restored_sha = unchanged("Fresh observation/info")
        require(isinstance(frames, dict) and not {"joint_qpos", "joint_names", "sim_time_s"} & frames.keys(),
                "Fresh observation renderer cannot override the physical clock/state")
        image = np.asarray(frames.get("head_stereo_left"))
        require(image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3
                and image.shape[0] > 0 and image.shape[1] > 0, "Missing fresh uint8 RGB head_stereo_left observation")
        observation = {"joint_qpos": np.asarray(list(named.values()), dtype=np.float32),
            "joint_names": tuple(named), "sim_time_s": float(data.time), **frames}
        require(observation["sim_time_s"] == self.receipt["source_sim_time_s"], "Initial server-history clock differs")
        if isinstance(info, dict) and "time" in info.get("proprio", {}):
            require(float(info["proprio"]["time"]) == observation["sim_time_s"], "Fresh proprioception clock differs")
        receipt = {**deepcopy(self.receipt), "initialization_count": 1, "model_compatibility": compatibility,
            "joint_names": list(names), "actor_body_bindings": deepcopy(bindings),
            "source_capture_joint_names": list(names), "actual_observation_joint_names": list(actual_names),
            "source_to_observation_indices": [x["observation_index"] for x in joint_bindings],
            "named_joint_addresses_equal": True, "named_joint_bindings": joint_bindings,
            "restored_state_sha256": restored_sha, "previous_reset_sim_time_s": before_time,
            "actual_initial_sim_time_s": float(data.time), "applied_before_new_history": True,
            "fresh_rgb_rendered": True, "isaac_zero_time_refresh": isaac is not None,
            "live_model_data_objects_preserved": True, "external_forces_zero": True, "physics_steps": 0,
            "progress_reset": ["task.reward", "task._init_target_height", "task._contact_started", "env._success"],
            "observation_refresh": "kinematics/comPos/comVel/camlight plus zero-time render; no mj_forward",
            "initial_info_boundary": "qpos/qvel-derived kinematics are refreshed; qacc/actuator_force are not recorded integration fields and are not certified restored measurements"}
        return observation, info, receipt
