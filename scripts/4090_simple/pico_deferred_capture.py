"""Archive real PICO physics for later, zero-step RTX image acquisition.

Only CPU MuJoCo state/FK operations are used here. EGL preview pixels are never
training observations. A completed archive remains pending RTX acquisition and
independent audit; this module cannot admit an episode to a training dataset.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from pico_teleop_recording import CaptureRejected, _copy_command, _json_value, _snapshot
from replay_handover_teacher import MODEL_HAND_NAMES, capture_tick_index


CONTRACT = "cvpr-live-pico-deferred-physical-capture-v1"
PENDING = "raw_physics_capture_completed_pending_rtx"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _vector(value, size, label):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{label} must be a finite vector of length {size}")
    return result.copy()


def _pose(value, label):
    result = _vector(value, 7, label)
    if not np.isclose(np.linalg.norm(result[3:]), 1., rtol=0., atol=1e-6):
        raise ValueError(f"{label} needs a unit wxyz quaternion")
    return result


def _equal(actual, expected, label, *, pose=False):
    actual, expected = np.asarray(actual), np.asarray(expected)
    equal = actual.shape == expected.shape and np.allclose(actual, expected, rtol=0., atol=1e-9)
    if pose and not equal and actual.shape == expected.shape == (7,):
        equal = (np.allclose(actual[:3], expected[:3], rtol=0., atol=1e-9)
                 and np.allclose(actual[3:], -expected[3:], rtol=0., atol=1e-9))
    if not equal:
        raise ValueError("Integration state disagrees with measured " + label)


def _action(command):
    copied = _copy_command(command)
    if copied is None:
        return None
    parameters = copied.parameters
    token = _vector(parameters["body_token"], 64, "Actual SONIC v1.1 token")
    hands = parameters["hand_targets"]
    if set(hands) != set(MODEL_HAND_NAMES):
        raise ValueError("Actual command requires all fourteen named hand channels")
    hand = _vector([hands[name] for name in MODEL_HAND_NAMES], 14, "Actual hand command")
    return {"body_token64": token.tolist(), "hand14": hand.tolist(),
            "hand_joint_names": list(MODEL_HAND_NAMES)}


class _PhysicalVerifier:
    """Check exported observations against exactly the saved integration state."""

    def __init__(self, model, names, metadata, state_spec):
        import mujoco
        self.mujoco = mujoco
        self.model = model
        self.names = list(names)
        self.spec = mujoco.mjtState(state_spec)
        self.width = mujoco.mj_stateSize(model, self.spec)
        self.data = mujoco.MjData(model)
        self.qadr, self.vadr = [], []
        for name in self.names:
            joint = model.joint(name)
            if int(joint.type[0]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                raise ValueError("All named state43 joints must have scalar coordinates")
            self.qadr.append(int(joint.qposadr[0]))
            self.vadr.append(int(joint.dofadr[0]))
        self.pelvis = model.body("pelvis").id
        root_joint = int(model.body_jntadr[self.pelvis])
        if (int(model.body_jntnum[self.pelvis]) != 1 or root_joint < 0
                or int(model.jnt_type[root_joint]) != int(mujoco.mjtJoint.mjJNT_FREE)):
            raise ValueError("The G1 pelvis must have its actual free root joint")
        self.root_dof = int(model.jnt_dofadr[root_joint])
        bindings = metadata.get("object_body_names", metadata.get("mujoco_object_body_names", {}))
        if not isinstance(bindings, dict):
            raise ValueError("Object body bindings must be a dictionary")
        self.objects = {role: model.body(bindings.get(role, role)).id for role in ("target", "container")}
        if len(set(self.objects.values())) != 2:
            raise ValueError("Target and container must be distinct physical bodies")

    def check(self, event, state):
        state = _vector(state, self.width, "mjSTATE_INTEGRATION")
        self.mujoco.mj_setState(self.model, self.data, state, self.spec)
        # Derived transforms only. mj_forward would overwrite integration's
        # solver warmstart, making an apparently zero-step state roundtrip lossy.
        self.mujoco.mj_kinematics(self.model, self.data)
        self.mujoco.mj_camlight(self.model, self.data)
        _equal([self.data.time], [event["sim_time_s"]], "sim time")
        _equal(self.data.qpos[self.qadr], event["physical_joint_q43_float64"], "q43")
        _equal(self.data.qvel[self.vadr], event["physical_joint_qvel43"], "qvel43")
        _equal(np.r_[self.data.xpos[self.pelvis], self.data.xquat[self.pelvis]], event["root_pose7"], "root pose", pose=True)
        _equal(self.data.qvel[self.root_dof:self.root_dof + 6], event["root_qvel6"], "root velocity")
        for role, body in self.objects.items():
            _equal(np.r_[self.data.xpos[body], self.data.xquat[body]], event["objects"][role]["pose7"], role + " pose", pose=True)
        roundtrip = np.empty(self.width, dtype=np.float64)
        self.mujoco.mj_getState(self.model, self.data, roundtrip, self.spec)
        if not np.array_equal(state, roundtrip):
            raise ValueError("CPU FK unexpectedly changed archived integration state")
        return state


class PreparedDeferredModel:
    """Model archive and scratch FK state allocated outside real-time control.

    Prepare once after each actual scene reset/initialization. Consecutive
    episodes may reuse it after close; simultaneous captures may not. Model
    mutation during control is unsupported. The reader independently rehashes
    the linked model rather than trusting this preparation receipt.
    """

    def __init__(self, directory, *, model, joint_names, metadata):
        import mujoco
        self.names = list(joint_names)
        _validate_header(self.names, 50, 0.)
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        copied_metadata = _json_value(_snapshot(metadata))
        self.verifier = _PhysicalVerifier(model, self.names, copied_metadata, self.spec)
        self.model = model
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.model_path = self.directory / "model.mjb"
        mujoco.mj_saveModel(model, str(self.model_path), None)
        self.model_sha256 = _sha(self.model_path)
        self.model_path.chmod(0o444)
        self._stat = self._file_identity()
        self.in_use = False
        receipt = {"contract": "cvpr-pico-deferred-prepared-model-v1",
            "model_path": "model.mjb", "model_sha256": self.model_sha256,
            "state_spec": self.spec, "state_width": self.verifier.width,
            "joint_names": self.names,
            "object_body_names": {role: model.body(body).name for role, body in self.verifier.objects.items()},
            "mujoco_version": mujoco.__version__, "accepted_new_dataset": False}
        (self.directory / "prepared_model.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")

    def _file_identity(self):
        stat = self.model_path.stat()
        return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)

    def acquire(self, *, model, names, metadata):
        if self.in_use:
            raise ValueError("Prepared model already belongs to an unclosed capture")
        if model is not self.model or list(names) != self.names:
            raise ValueError("Prepared model must match the exact initialized physical model and joint order")
        bindings = metadata.get("object_body_names", metadata.get("mujoco_object_body_names", {}))
        if {role: model.body(bindings.get(role, role)).id for role in ("target", "container")} != self.verifier.objects:
            raise ValueError("Prepared model object bindings changed; prepare again after scene reset")
        if self.model_path.is_symlink() or self._file_identity() != self._stat:
            raise ValueError("Prepared compiled model archive changed")
        self.in_use = True


def prepare_deferred_model(directory, *, model, joint_names, metadata):
    """Perform MJB write/hash and scratch allocation before the live loop."""
    return PreparedDeferredModel(directory, model=model, joint_names=joint_names, metadata=metadata)


def _validate_header(names, warmup_ticks, start_time):
    if (len(names) != 43 or any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != 43 or list(names[-14:]) != list(MODEL_HAND_NAMES)):
        raise ValueError("Need the actual named model43 order, ending with canonical hand14")
    if type(warmup_ticks) is not int or warmup_ticks < 50 or warmup_ticks % 5:
        raise ValueError("Warmup requires >=50 real ticks aligned to the 50/30Hz grid")
    if isinstance(start_time, bool) or not isinstance(start_time, (int, float, np.number)) or not math.isfinite(start_time):
        raise ValueError("A finite actual simulation start time is required")


def _validate_event(event, index, names, warmup_ticks, start_time):
    tick = event["physical_tick50"]
    if capture_tick_index(tick) != index or event["event_index"] != index or event["capture_index30"] != index:
        raise ValueError("Capture requires every dense real ceil30Hz frame, without skips")
    clock = float(event["sim_time_s"])
    if not math.isfinite(clock) or not np.isclose(clock, start_time + tick / 50, rtol=0., atol=1e-6):
        raise ValueError("Actual integration time does not match the 50/30Hz capture grid")
    if event["joint_names"] != names:
        raise ValueError("Event joint order differs from the archived model43 order")
    observed = _vector(event["state43"], 43, "Observed state43").astype(np.float32)
    physical = _vector(event["physical_joint_q43_float64"], 43, "Physical q43")
    _vector(event["physical_joint_qvel43"], 43, "Physical qvel43")
    _pose(event["root_pose7"], "Actual root")
    _vector(event["root_qvel6"], 6, "Actual root velocity")
    if not np.array_equal(observed, physical.astype(np.float32)):
        raise ValueError("Observed state and actual integration q43 are not the same frame")
    for role in ("target", "container"):
        _pose(event["objects"][role]["pose7"], role)
    warmup = tick < warmup_ticks
    if (event["phase"] != ("warmup" if warmup else "teacher_command")
            or event["action_valid"] is not (not warmup)
            or event["source_command_index"] != (None if warmup else tick - warmup_ticks)):
        raise ValueError("Warmup/action validity or command source does not match actual time")
    if warmup:
        if event["action"] is not None:
            raise ValueError("Warmup observations cannot acquire supervised action labels")
    else:
        action = event["action"]
        if not isinstance(action, dict) or action.get("hand_joint_names") != list(MODEL_HAND_NAMES):
            raise ValueError("Actual SONIC command must retain canonical fourteen named hands")
        _vector(action["body_token64"], 64, "Actual body token64")
        _vector(action["hand14"], 14, "Actual hand14")
    source = event["source_fields"]
    if not isinstance(source, dict):
        raise ValueError("source_fields must be a copied dictionary")
    if "observation_sim_time_s" in source:
        _equal([float(source["observation_sim_time_s"])], [clock], "observation clock")
    if (event["integration_state_index"] != index
            or event["nominal_elapsed_s"] != index / 30 or event["actual_elapsed_s"] != tick / 50):
        raise ValueError("Event indexes and elapsed times must describe the same actual state")
    if event.get("preview") is not None and event["preview"].get("training_observation") is not False:
        raise ValueError("EGL preview must never be marked as training RGB")
    if "rgb" in event:
        raise ValueError("Deferred physical archives cannot claim training RGB acquisition")


class DeferredStateCapture:
    """Synchronous, fail-closed physical archive; no renderer or SDK is started.

    ``integration_state`` and all measured fields must describe the same real
    pre-action state at ``start_time + tick50 / 50``. Call only on every
    ``ceil(k*50/30)`` tick, starting with zero. ``source_fields`` retains original
    human input evidence without interpreting raw va/wva as valid controls.
    """

    def __init__(self, directory, *, metadata, joint_names, warmup_ticks, start_time, model,
                 prepared_model=None):
        import mujoco
        self.names = list(joint_names)
        _validate_header(self.names, warmup_ticks, start_time)
        self.metadata = _json_value(_snapshot(metadata))
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a copied JSON dictionary")
        self.spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
        self._prepared = prepared_model
        if prepared_model is not None and not isinstance(prepared_model, PreparedDeferredModel):
            raise TypeError("prepared_model must be returned by prepare_deferred_model")
        if prepared_model is None:
            self._verifier = _PhysicalVerifier(model, self.names, self.metadata, self.spec)
        else:
            self._verifier = prepared_model.verifier
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.warmup_ticks, self.start_time = warmup_ticks, float(start_time)
        self.count = self.warmup_count = self.command_count = 0
        self.closed, self._error, self.manifest = False, None, None
        self._states = []
        self.events_path = self.directory / "events.jsonl"
        model_path = self.directory / "model.mjb"
        acquired = False
        try:
            if prepared_model is None:
                mujoco.mj_saveModel(model, str(model_path), None)
                self.model_sha256 = _sha(model_path)
            else:
                # Cache and episode are on the same filesystem. No large-file
                # copy/hash or MjData allocation belongs in the live loop.
                prepared_model.acquire(model=model, names=self.names, metadata=self.metadata)
                acquired = True
                os.link(prepared_model.model_path, model_path, follow_symlinks=False)
                self.model_sha256 = prepared_model.model_sha256
            self.stream = self.events_path.open("x", buffering=1)
        except BaseException:
            if acquired:
                prepared_model.in_use = False
            raise

    @property
    def failed(self):
        return self._error is not None

    @property
    def error(self):
        return self._error

    def append(self, *, tick50, sim_time_s, observed_joint_names, observed_qpos,
               physical_qpos, physical_qvel, root_pose7, root_qvel6, objects,
               command, integration_state, source_fields=None, preview_rgb=None):
        if self.closed or self.failed:
            raise CaptureRejected(self._error or "Capture is already closed")
        try:
            observed_names = list(observed_joint_names)
            if len(observed_names) != 43 or len(set(observed_names)) != 43 or set(observed_names) != set(self.names):
                raise ValueError("Observation must contain exactly the actual named43 joints")
            observed = _vector(observed_qpos, 43, "Observed q43")
            state43 = np.array([observed[observed_names.index(name)] for name in self.names], dtype=np.float32)
            k = self.count
            warmup = type(tick50) is int and tick50 < self.warmup_ticks
            actual_action = _action(command)
            if not warmup and actual_action is None:
                raise ValueError("A command frame requires the actual applied SONIC v1.1 token and hand14")
            state = _vector(integration_state, self._verifier.width, "mjSTATE_INTEGRATION")
            event = {"event_index": k, "capture_index30": k, "physical_tick50": tick50,
                "sim_time_s": float(sim_time_s), "nominal_elapsed_s": k / 30,
                "actual_elapsed_s": tick50 / 50, "phase": "warmup" if warmup else "teacher_command",
                "action_valid": not warmup, "source_command_index": None if warmup else tick50 - self.warmup_ticks,
                "joint_names": self.names.copy(), "state43": state43.tolist(),
                "physical_joint_q43_float64": _vector(physical_qpos, 43, "Physical q43").tolist(),
                "physical_joint_qvel43": _vector(physical_qvel, 43, "Physical qvel43").tolist(),
                "root_pose7": _pose(root_pose7, "Actual root").tolist(),
                "root_qvel6": _vector(root_qvel6, 6, "Actual root velocity").tolist(),
                "objects": {role: {"pose7": _pose(objects[role]["pose7"], role).tolist()} for role in ("target", "container")},
                "action": None if warmup else actual_action,
                "integration_state_index": k,
                "integration_state_sha256": hashlib.sha256(state.tobytes(order="C")).hexdigest(),
                "source_fields": _json_value(_snapshot(source_fields if source_fields is not None else {})),
                "preview": None}
            if warmup and actual_action is not None:
                event["executed_warmup_command"] = actual_action
            if preview_rgb is not None:
                pixels = np.asarray(preview_rgb)
                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) <= 0:
                    raise ValueError("Optional preview must be actual uint8 RGB HxWx3")
                pixels = pixels.copy(order="C")
                event["preview"] = {"engine": "MuJoCo EGL", "training_observation": False,
                    "saved_as_training_rgb": False, "pixels_saved": False,
                    "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "width": pixels.shape[1], "height": pixels.shape[0], "channels": 3, "dtype": "uint8"}
            _validate_event(event, k, self.names, self.warmup_ticks, self.start_time)
            self._verifier.check(event, state)
            encoded = json.dumps(_json_value(event), ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            self.stream.write(encoded + "\n")
            self._states.append(state)
            self.count += 1
            self.warmup_count += int(warmup)
            self.command_count += int(not warmup)
        except Exception as error:
            self._error = f"{type(error).__name__}: {error}"
            raise CaptureRejected(self._error) from error

    enqueue = append

    def close(self, *, completed):
        if type(completed) is not bool:
            raise TypeError("completed must be a boolean")
        if self.closed:
            return _snapshot(self.manifest)
        self.closed = True
        if completed and not self.command_count and not self.failed:
            self._error = "A completed archive requires a real warmup and actual command observations"
        manifest = {"contract": CONTRACT, "version": 1, "metadata": self.metadata,
            "status": PENDING if completed and not self.failed else "partial_capture_rejected",
            "failure": self.error, "requested_completed": completed,
            "accepted_new_dataset": False, "eligible_for_packing": False, "rtx_ready": False,
            "joint_names": self.names, "hand_joint_names": list(MODEL_HAND_NAMES),
            "warmup_ticks": self.warmup_ticks, "warmup_seconds": self.warmup_ticks / 50,
            "start_time": self.start_time, "event_count": self.count,
            "warmup_event_count": self.warmup_count, "command_event_count": self.command_count,
            "physics_hz": 200, "decoder_hz": 50, "command_rate_hz": 30,
            "state_spec": self.spec, "state_spec_name": "mjSTATE_INTEGRATION",
            "state_width": self._verifier.width, "states_dtype": "float64",
            "states_path": "states.npy", "events_path": "events.jsonl", "model_path": "model.mjb",
            "model_sha256": self.model_sha256,
            "online_operator_preview_engine": "MuJoCo EGL", "training_rgb_acquired": False,
            "rgb_acquisition": "pending_deferred_render_from_archived_state",
            "alignment": "Pre-action mjSTATE_INTEGRATION/state/actual-command on dense ceil(k*50/30) ticks; warmup labels are null",
            "limitation": "The human observed EGL preview; later RTX images are not live operator feedback or a learned-policy evaluation"}
        try:
            self.stream.close()
            matrix = np.stack(self._states) if self._states else np.empty((0, self._verifier.width), dtype=np.float64)
            with (self.directory / "states.npy").open("xb") as stream:
                np.save(stream, matrix, allow_pickle=False)
            if _sha(self.directory / "model.mjb") != self.model_sha256:
                raise ValueError("Compiled model archive changed during capture")
            manifest["events_sha256"] = _sha(self.events_path)
            manifest["states_sha256"] = _sha(self.directory / "states.npy")
        except Exception as error:
            self._error = self._error or f"{type(error).__name__}: {error}"
            manifest.update(status="partial_capture_rejected", failure=self.error)
        finally:
            if not self.stream.closed:
                self.stream.close()
            self._states.clear()
            if self._prepared is not None:
                self._prepared.in_use = False
        self.manifest = _json_value(manifest)
        try:
            (self.directory / "capture.json").write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        except Exception as error:
            self._error = self._error or f"{type(error).__name__}: {error}"
            self.manifest.update(status="partial_capture_rejected", failure=self.error)
        return _snapshot(self.manifest)


def _owned_file(directory, relative):
    if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("Archive files must be owned relative paths")
    candidate = directory / relative
    if candidate.is_symlink() or candidate.resolve(strict=True).parent != directory:
        raise ValueError("Archive file must remain inside its physical capture directory")
    if not candidate.is_file():
        raise ValueError("Archive path must be a regular file")
    return candidate


def load_deferred(directory, *, verify=True):
    """Read an immutable archive for offline rendering, without simulating.

    Returns ``manifest`` (including nested ``metadata``), ``events`` (standard
    measured state/action fields plus original ``source_fields``), read-only
    mmap ``states``, loaded ``model``, integer ``state_spec`` and ``joint_names``.
    Full verification binds each event to the saved state and compiled model.
    Even a verified completed archive is not training-ready or accepted.
    """
    import mujoco
    directory = Path(directory).resolve(strict=True)
    manifest = json.loads(_owned_file(directory, "capture.json").read_text())
    _json_value(manifest)
    if manifest.get("contract") != CONTRACT or manifest.get("version") != 1:
        raise ValueError("Unsupported deferred physical archive contract")
    if (manifest.get("accepted_new_dataset") is not False or manifest.get("rtx_ready") is not False
            or manifest.get("eligible_for_packing") is not False
            or manifest.get("training_rgb_acquired") is not False):
        raise ValueError("A raw deferred archive cannot claim training acquisition or acceptance")
    if (manifest.get("status") not in (PENDING, "partial_capture_rejected")
            or any(type(manifest.get(key)) is not int or manifest[key] < 0
                   for key in ("event_count", "warmup_event_count", "command_event_count"))
            or not isinstance(manifest.get("metadata"), dict)
            or any(manifest.get(key) != value for key, value in
                   (("physics_hz", 200), ("decoder_hz", 50), ("command_rate_hz", 30),
                    ("state_spec_name", "mjSTATE_INTEGRATION"), ("states_dtype", "float64")))):
        raise ValueError("Invalid physical capture completion, clocks or archive schema")
    names = manifest["joint_names"]
    _validate_header(names, manifest["warmup_ticks"], manifest["start_time"])
    paths = {kind: _owned_file(directory, manifest[kind + "_path"]) for kind in ("model", "events", "states")}
    if verify:
        for kind, path in paths.items():
            if _sha(path) != manifest[kind + "_sha256"]:
                raise ValueError(kind + " archive SHA256 mismatch")
    if manifest["state_spec"] != int(mujoco.mjtState.mjSTATE_INTEGRATION):
        raise ValueError("Archive must retain the complete mjSTATE_INTEGRATION state")
    model = mujoco.MjModel.from_binary_path(str(paths["model"]))
    states = np.load(paths["states"], mmap_mode="r", allow_pickle=False)
    width = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    if (states.dtype != np.dtype(np.float64) or states.shape != (manifest["event_count"], width)
            or manifest["state_width"] != width or not np.isfinite(states).all()):
        raise ValueError("Archived integration matrix shape, dtype or finiteness mismatch")
    events = [json.loads(line) for line in paths["events"].read_text().splitlines()]
    if len(events) != manifest["event_count"]:
        raise ValueError("State/event counts disagree")
    verifier = _PhysicalVerifier(model, names, manifest["metadata"], manifest["state_spec"])
    warmup_count = 0
    for index, event in enumerate(events):
        _json_value(event)
        _validate_event(event, index, names, manifest["warmup_ticks"], manifest["start_time"])
        warmup_count += int(event["phase"] == "warmup")
        if verify:
            state = np.ascontiguousarray(states[index])
            if hashlib.sha256(state.tobytes()).hexdigest() != event["integration_state_sha256"]:
                raise ValueError("Event/state row SHA256 mismatch")
            verifier.check(event, state)
    if (warmup_count != manifest["warmup_event_count"]
            or len(events) - warmup_count != manifest["command_event_count"]):
        raise ValueError("Warmup and actual command counts disagree")
    if manifest["status"] == PENDING and (not manifest["command_event_count"] or manifest.get("failure") is not None):
        raise ValueError("Failed or commandless archive cannot be completed")
    return {"manifest": manifest, "events": events, "states": states, "model": model,
            "state_spec": manifest["state_spec"], "joint_names": list(names)}
