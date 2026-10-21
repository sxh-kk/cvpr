"""One-time, model-checked initialization from an actual rollout's first state.

This client helper never reads teacher commands or subsequent trajectory rows.
The source MJB is only a compatibility reference; live model/data objects and
their cached views remain owned by the newly reset evaluation environment.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import mujoco
import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _digest_array(value):
    array = np.ascontiguousarray(value)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True)
    digest = hashlib.sha256(header.encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _model_fields(obj, prefix="model"):
    """Include every exposed static numeric field, names, and option structures.

    Comparing only model dimensions would miss inertia, contacts, meshes and
    actuator changes. New unsupported public data structures fail closed.
    """
    values = {}
    for name in sorted(dir(obj)):
        if name.startswith("_"):
            continue
        if prefix == "model" and name == "signature":
            # mjmodel.h: compilation signature shared with its originating
            # mjSpec. MuJoCo resets it to zero when loading an MJB; it is not
            # a physical parameter and cannot compare a reload to a compile.
            continue
        value = getattr(obj, name)
        key = f"{prefix}.{name}"
        if callable(value):
            continue
        if isinstance(value, np.ndarray):
            values[key] = {"kind": "array", "dtype": value.dtype.str,
                           "shape": list(value.shape), "sha256": _digest_array(value)}
        elif isinstance(value, bytes):
            values[key] = {"kind": "bytes", "length": len(value),
                           "sha256": hashlib.sha256(value).hexdigest()}
        elif isinstance(value, (bool, int, float, str, np.generic)):
            values[key] = {"kind": "scalar", "value": value.item() if isinstance(value, np.generic) else value}
        elif type(value).__module__.startswith("mujoco"):
            values.update(_model_fields(value, key))
        else:
            raise ValueError(f"Unsupported static model field {key}: {type(value).__name__}")
    return values


def compare_models(source, target):
    source_fields, target_fields = _model_fields(source), _model_fields(target)
    mismatches = [key for key in sorted(set(source_fields) | set(target_fields))
                  if source_fields.get(key) != target_fields.get(key)]
    if mismatches:
        raise ValueError("Initial physics model is incompatible: " + ", ".join(mismatches[:24]))
    serialized = json.dumps(source_fields, sort_keys=True, separators=(",", ":"), allow_nan=False)
    target_serialized = json.dumps(target_fields, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"compatible": True, "comparison": "exact static field values and array bytes; no tolerance",
            "mujoco_version": mujoco.__version__, "fields_compared": len(source_fields),
            "source_static_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "target_static_sha256": hashlib.sha256(target_serialized.encode()).hexdigest(),
            "excluded_compilation_signature": {"source": int(source.signature), "target": int(target.signature),
                "reason": "mjSpec compilation linkage; MJB reload sets it to zero; all physical fields are compared"},
            "field_receipts": source_fields}


def _npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError(f"Unsupported NPY format {version}")


def read_initial_integration_state(path):
    """Decode only the scalar spec and first C-order row, never future rows."""
    with zipfile.ZipFile(path) as archive:
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise ValueError("Duplicate NPZ entries are ambiguous")
        with archive.open("physics_state_spec.npy") as stream:
            shape, fortran, dtype = _npy_header(stream)
            if shape != () or fortran or dtype.kind not in "iu":
                raise ValueError("Physics state specification must be an integer scalar")
            value = stream.read(dtype.itemsize)
            if len(value) != dtype.itemsize:
                raise ValueError("Truncated physics state specification")
            spec = int(np.frombuffer(value, dtype=dtype)[0])
        if spec != int(mujoco.mjtState.mjSTATE_INTEGRATION):
            raise ValueError("Initial state must use complete mjSTATE_INTEGRATION")
        with archive.open("physics_state_integration.npy") as stream:
            shape, fortran, dtype = _npy_header(stream)
            if len(shape) != 2 or shape[0] < 1 or shape[1] < 1 or fortran or dtype.kind != "f" or dtype.itemsize != 8:
                raise ValueError("Integration states must be a nonempty C-order float64 matrix")
            payload = stream.read(shape[1] * dtype.itemsize)
            if len(payload) != shape[1] * dtype.itemsize:
                raise ValueError("Truncated first integration state")
            first = np.frombuffer(payload, dtype=dtype).astype(np.float64, copy=True)
    if not np.isfinite(first).all():
        raise ValueError("Initial integration state contains NaN/Inf")
    return first, spec


class InitialPhysicsState:
    def __init__(self, run_directory, *, task_id, seed):
        directory = Path(run_directory)
        if not directory.is_absolute():
            raise ValueError("Initial physics run must be an absolute directory")
        self.directory = directory.resolve(strict=True)
        result_path = self.directory / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("task_id") != task_id or result.get("seed") != seed:
            raise ValueError("Initial physics run must have the same task and seed as this evaluation")
        if result.get("physical_outcome_status") not in ("source_end", "task_terminated", "sim_time_limit", "stopped_on_fall"):
            raise ValueError("Initial physics source is not a completed physical run")
        model_path, trajectory_path = self.directory / "physics_model.mjb", self.directory / "trajectory.npz"
        model_receipt = result.get("physics_model", {})
        model_sha = file_sha256(model_path)
        if model_receipt.get("sha256") != model_sha:
            raise ValueError("Initial physics source MJB SHA256 disagrees with result.json")
        self.model = mujoco.MjModel.from_binary_path(str(model_path))
        self.state, self.spec = read_initial_integration_state(trajectory_path)
        if self.state.size != mujoco.mj_stateSize(self.model, self.spec):
            raise ValueError("Initial integration state length does not match the source model")
        self.applied = False
        self.receipt = {"source_run": str(self.directory), "source_result_sha256": file_sha256(result_path),
            "source_mjb_sha256": model_sha, "source_trajectory_sha256": file_sha256(trajectory_path),
            "source_physical_outcome_status": result["physical_outcome_status"],
            "source_task_id": task_id, "source_seed": seed,
            "state_spec": self.spec, "state_size": self.state.size, "source_row": 0,
            "initial_state_sha256": _digest_array(self.state), "source_sim_time_s": float(self.state[0]),
            "scope": "first actual physics state only; no teacher action, future row or RGB replay"}

    def restore(self, raw):
        if self.applied:
            raise RuntimeError("Initial physics state may only be applied once")
        model, data = raw.mujoco.mjModel, raw.mujoco.mjData
        compatibility = compare_models(self.model, model)
        before_time = float(data.time)
        mujoco.mj_setState(model, data, self.state, self.spec)
        self.applied = True
        mujoco.mj_forward(model, data)
        after = np.empty_like(self.state)
        mujoco.mj_getState(model, data, after, self.spec)
        if not np.array_equal(after, self.state):
            raise ValueError("mj_forward changed the restored integration state")
        raw.task._reset_progress()
        raw._success = False
        observation, info = raw._get_obs(), raw._get_info()
        observed = np.empty_like(self.state)
        mujoco.mj_getState(model, data, observed, self.spec)
        if not np.array_equal(observed, self.state):
            raise ValueError("Fresh initial observation advanced or changed restored physics")
        receipt = {**self.receipt, "initialization_count": 1, "model_compatibility": compatibility,
            "previous_reset_sim_time_s": before_time, "actual_initial_sim_time_s": float(data.time),
            "restored_state_sha256": _digest_array(observed), "live_model_data_objects_preserved": True,
            "new_rgb_after_restore": True, "task_progress_reset": True}
        return observation, info, receipt
