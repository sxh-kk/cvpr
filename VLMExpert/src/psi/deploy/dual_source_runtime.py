"""Local online DINO and episode-local, simulation-time observation history.

This module is CPU-only until ``LocalDinoEncoder`` is constructed. Observation
collection never acquires the policy inference lock, so clients can keep posting
camera frames while actions execute or the shared GPU runs a policy forward.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import threading
from typing import Any

import numpy as np


class HistoryError(ValueError):
    def __init__(self, message: str, *, status_code: int = 422, **details: Any):
        super().__init__(message)
        self.status_code = status_code
        self.detail = {"status": "invalid_observation", "message": message, **details}


class HistoryWarmingUp(HistoryError):
    def __init__(self, targets: list[float], missing: list[float], available: list[float]):
        super().__init__("Collect real episode observations for the full past second", status_code=425)
        self.detail = {"status": "warming_up", "target_sim_times_s": targets,
                       "missing_sim_times_s": missing, "available_sim_times_s": available}


@dataclass(frozen=True)
class Observation:
    sim_time_s: float
    rgb: np.ndarray
    source_sha256: str


@dataclass
class EpisodeSession:
    episode_id: str
    generation: int
    frames: list[Observation] = field(default_factory=list)
    previous_action: np.ndarray | None = None
    last_action_time_s: float | None = None


@dataclass(frozen=True)
class HistorySnapshot:
    session_id: str
    episode_id: str
    generation: int
    sim_time_s: float
    target_times_s: tuple[float, ...]
    observations: tuple[Observation, ...]
    previous_action: np.ndarray | None

    @property
    def rgb(self) -> list[np.ndarray]:
        return [frame.rgb for frame in self.observations]


class SimulationHistory:
    """Select six distinct causal frames at t-1.0, t-0.8, ..., t.

    A frame may lag its target by at most one native 30 Hz period (+1 ms).
    Repeated HTTP requests at the same simulator time never advance history.
    Episode changes require an explicit reset; delayed old-episode messages are
    rejected instead of silently resetting the new episode back to the old one.
    """
    def __init__(self, profile="256x320", *, tolerance_s=1 / 30 + .001,
                 max_sessions=8, max_frames=180, retention_s=2.0):
        from ImageProcess import SharedImageProcessor
        self.processor = SharedImageProcessor(profile)
        if self.processor.profile.output_hw != (256, 320):
            raise ValueError("dual_source_v1 deployment requires the 256x320 RGB profile")
        self.tolerance_s = float(tolerance_s)
        if not 0 < self.tolerance_s < .1 or max_frames < 6 or max_sessions < 1 or retention_s < 1.1:
            raise ValueError("Invalid simulation history limits")
        self.max_sessions, self.max_frames, self.retention_s = max_sessions, max_frames, retention_s
        self._lock = threading.RLock()
        self._sessions: dict[str, EpisodeSession] = {}
        self._generation = 0

    @staticmethod
    def identity(history: dict) -> tuple[str, str]:
        if not isinstance(history, dict):
            raise HistoryError("history must contain session_id and episode_id")
        values = (history.get("session_id"), history.get("episode_id"))
        if any(not isinstance(value, str) or not value or len(value) > 256 for value in values):
            raise HistoryError("session_id and episode_id must be nonempty strings up to 256 characters")
        return values

    @staticmethod
    def simulation_time(history: dict) -> float:
        value = history.get("sim_time_s")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise HistoryError("history.sim_time_s must be a finite nonnegative simulator time in seconds")
        return float(value)

    def reset(self, history: dict) -> dict:
        session_id, episode_id = self.identity(history)
        with self._lock:
            if session_id not in self._sessions and len(self._sessions) >= self.max_sessions:
                raise HistoryError("Maximum active history sessions reached", status_code=409)
            self._generation += 1
            self._sessions[session_id] = EpisodeSession(episode_id, self._generation)
        return {"status": "reset", "session_id": session_id, "episode_id": episode_id,
                "history_frames": 0, "rtc_cleared": True}

    def observe(self, history: dict, image: Any) -> dict:
        session_id, episode_id = self.identity(history)
        timestamp = self.simulation_time(history)
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3 or min(array.shape[:2]) < 1:
            raise HistoryError("Observation must be nonempty RGB uint8 HWC, without implicit clipping/conversion")
        source_digest = hashlib.sha256(json.dumps(array.shape).encode("ascii"))
        source_digest.update(np.ascontiguousarray(array).tobytes())
        source_sha256 = source_digest.hexdigest()
        # Geometry is computed once here, before either DINO or VLM normalization.
        rgb = self.processor.process_window([array], stage="inference", training=False).images[0]
        rgb.setflags(write=False)
        with self._lock:
            if history.get("reset") is True:
                self.reset(history)
            elif "reset" in history and history["reset"] is not False:
                raise HistoryError("history.reset must be a boolean")
            if session_id not in self._sessions:
                self.reset(history)
            session = self._sessions[session_id]
            if session.episode_id != episode_id:
                raise HistoryError("Episode changed without /reset or history.reset=true", status_code=409)
            if session.frames and timestamp < session.frames[-1].sim_time_s - 1e-8:
                raise HistoryError("Simulator time moved backwards; reset the episode", status_code=409)
            duplicate = bool(session.frames and abs(timestamp - session.frames[-1].sim_time_s) <= 1e-8)
            if duplicate:
                if session.frames[-1].source_sha256 != source_sha256:
                    raise HistoryError("Two different RGB observations share one simulator timestamp", status_code=409)
            else:
                session.frames.append(Observation(timestamp, rgb, source_sha256))
                session.frames[:] = [frame for frame in session.frames if timestamp - frame.sim_time_s <= self.retention_s]
                session.frames[:] = session.frames[-self.max_frames:]
            return {"status": "observed", "session_id": session_id, "episode_id": episode_id,
                    "sim_time_s": timestamp, "history_frames": len(session.frames), "duplicate": duplicate}

    def snapshot(self, history: dict) -> HistorySnapshot:
        session_id, episode_id = self.identity(history)
        current = self.simulation_time(history)
        targets = [current - offset / 5 for offset in range(5, -1, -1)]
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.episode_id != episode_id:
                raise HistoryError("Unknown session/episode; submit observations first", status_code=409)
            if session.last_action_time_s is not None and current <= session.last_action_time_s + 1e-8:
                raise HistoryError("Each action request needs a new simulator timestamp", status_code=409)
            selected, missing = [], []
            for target in targets:
                candidates = [frame for frame in session.frames if frame.sim_time_s <= target + 1e-8]
                frame = candidates[-1] if candidates else None
                if frame is None or target - frame.sim_time_s > self.tolerance_s:
                    missing.append(target)
                else:
                    selected.append(frame)
            if missing or len({frame.sim_time_s for frame in selected}) != 6:
                raise HistoryWarmingUp(targets, missing, [frame.sim_time_s for frame in session.frames])
            if abs(selected[-1].sim_time_s - current) > 1e-8:
                raise HistoryError("Current policy RGB must exactly match current simulator time")
            previous = None if session.previous_action is None else session.previous_action.copy()
            return HistorySnapshot(session_id, episode_id, session.generation, current,
                                   tuple(targets), tuple(selected), previous)

    def commit_action(self, snapshot: HistorySnapshot, raw_action: np.ndarray) -> None:
        with self._lock:
            session = self._sessions.get(snapshot.session_id)
            if session is None or session.generation != snapshot.generation:
                raise HistoryError("Episode reset during policy inference; discard this action", status_code=409)
            if session.last_action_time_s is not None and snapshot.sim_time_s <= session.last_action_time_s:
                raise HistoryError("A newer action already committed; discard this action", status_code=409)
            session.previous_action = np.asarray(raw_action, dtype=np.float32).copy()
            session.last_action_time_s = snapshot.sim_time_s


def load_dino_contract(contract_path: str | Path) -> dict:
    """Verify every local DINO file and training preprocessing before loading it."""
    from ImageProcess import SharedImageProcessor
    from egomotion_pipeline.features import DINO_PREPROCESSING
    from egomotion_pipeline.contracts import dino_format
    from egomotion_pipeline.io import file_sha256

    contract = json.loads(Path(contract_path).read_text())
    checkpoint = Path(contract["online_checkpoint"])
    if not checkpoint.is_absolute() or not checkpoint.is_dir():
        raise ValueError("Online DINO checkpoint must be an existing absolute local directory")
    encoder = contract["encoder"]
    hashes = encoder.get("local_file_sha256")
    if (encoder.get("model_id") != "facebook/dinov2-with-registers-large" or encoder.get("revision") != "local"
            or not isinstance(hashes, dict) or not hashes
            or not any(name.endswith((".safetensors", ".bin")) for name in hashes)
            or encoder.get("weights_sha256") != hashlib.sha256(json.dumps(
                hashes, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()):
        raise ValueError("DINO must have the exact pinned local training weight identity")
    for key, value in dino_format().items():
        if key != "checkpoint" and encoder.get(key) != value:
            raise ValueError(f"DINO training token contract differs: {key}")
    shared = SharedImageProcessor(contract["shared_image_profile"])
    if shared.profile.output_hw != (256, 320) or shared.metadata != encoder.get("image_process"):
        raise ValueError("Online ImageProcess profile or NumPy/Pillow implementation differs from training")
    if contract.get("history") != {"fps": 5, "frames": 6, "shape": [6, 5, 1024],
                                   "augmentation": "none", "timestamps": "simulation_time"}:
        raise ValueError("DINO contract must describe six real 5 Hz simulator-time observations")
    if contract.get("processor_overrides_required") != {"use_fast": False, "do_resize": False, "do_center_crop": False}:
        raise ValueError("Online DINO must use the slow processor without second resize/crop")
    preprocessing = encoder.get("preprocessing", {})
    for key, value in DINO_PREPROCESSING.items():
        if preprocessing.get(key) != value:
            raise ValueError(f"DINO training preprocessing differs: {key}")
    for relative, expected in hashes.items():
        # Keep this inference-only path independent of LeRobot/PyArrow imports.
        name = Path(relative)
        resolved = (checkpoint / name).resolve(strict=True)
        if (name.is_absolute() or ".." in name.parts or not resolved.is_file()
                or not resolved.is_relative_to(checkpoint.resolve(strict=True))):
            raise ValueError(f"DINO local resource escapes checkpoint directory: {relative}")
        if file_sha256(resolved) != expected:
            raise ValueError(f"DINO local resource SHA256 differs: {relative}")
    actual_names = {str(path.relative_to(checkpoint)) for path in checkpoint.rglob("*")
                    if path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}}
    if actual_names != set(hashes):
        raise ValueError("DINO checkpoint contains model/config files outside the pinned identity")
    return contract


class LocalDinoEncoder:
    def __init__(self, contract_path: str | Path, *, device: str):
        from egomotion_pipeline.features import DinoEncoder
        self.contract = load_dino_contract(contract_path)
        expected = self.contract["encoder"]
        # Preserve the cache extraction precision/attention policy when declared.
        optimization = expected.get("optimization", {})
        options = {key: optimization[key] for key in
                   ("inference_precision", "attention_implementation") if key in optimization}
        self.encoder = DinoEncoder({"checkpoint": self.contract["online_checkpoint"], "revision": "local",
                                    "device": device, "image_profile": self.contract["shared_image_profile"], **options})
        if self.encoder.metadata["local_file_sha256"] != expected["local_file_sha256"]:
            raise ValueError("Loaded DINO identity differs from the verified training contract")

    def encode(self, snapshot: HistorySnapshot) -> np.ndarray:
        # Already-clean geometry goes to BOTH branches, with no second geometry pass.
        values = self.encoder.encode_prepared(self.encoder.prepare(snapshot.rgb, processed=True))
        if values.shape != (6, 5, 1024) or not np.isfinite(values).all():
            raise ValueError("Online DINO must return finite [6,5,1024] CLS + four register tokens")
        return np.asarray(values, dtype=np.float32)
