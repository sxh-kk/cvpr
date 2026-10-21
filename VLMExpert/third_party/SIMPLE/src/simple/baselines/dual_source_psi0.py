"""UnifoLM dual_source_v1 client and explicit SIMPLE/SONIC v1.1 contracts.

This adapter requires an independently identified body_token_v1_1 controller.
It never sends the latent to the old 36-dimensional joint/navigation WBC.
All observation times are simulator seconds, independent of HTTP wall time.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
import math
import uuid
from typing import Protocol

import numpy as np
import requests

from simple.baselines.client import RequestMessage, ResponseMessage
from simple.core.action import ActionCmd


# Official USC-PSI-Lab/psi-data revision e78fb93cc28912a3031a10b8656d32d7f0a2b867,
# unifolm_sonic_lerobot_val.zip SHA256:
# 3d264d6454d59e83be5dadaa17f122b732a8c6cb86a2429c33c1f4fc912fc4b7.
# info.json SHA256 3b58f2cb791fc0e82bcc6c4dc08a6ced44cd5729e9531bd741f2e392dff1d467.
# observation.state.names and action.names[0:14] are copied below; fps is 30.
BODY_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
TRAINING_HAND_JOINT_NAMES = tuple(
    f"{side}_hand_{finger}_{joint}_joint"
    for side in ("left", "right")
    for finger, count in (("thumb", 3), ("middle", 2), ("index", 2))
    for joint in range(count)
)
TRAINING_STATE_JOINT_NAMES = BODY_JOINT_NAMES + TRAINING_HAND_JOINT_NAMES
SIMPLE_HAND_JOINT_NAMES = tuple(
    f"{side}_hand_{finger}_{joint}_joint"
    for side in ("left", "right")
    for finger, count in (("thumb", 3), ("index", 2), ("middle", 2))
    for joint in range(count)
)
ACTION_FPS = 30.0


def state43_to45(joint_qpos, joint_names: Sequence[str]) -> np.ndarray:
    """Reorder named radian joint positions into the training state and pad 2.

    Both hands differ from SIMPLE's order: training puts middle before index.
    Missing, duplicate, additional or unnamed joints are rejected.
    """
    names = tuple(joint_names)
    values = np.asarray(joint_qpos)
    if (len(names) != 43 or len(set(names)) != 43 or
            set(names) != set(TRAINING_STATE_JOINT_NAMES)):
        raise ValueError("Expected exactly the 43 named training joints, with no duplicates")
    if (values.shape != (43,) or not np.issubdtype(values.dtype, np.number) or
            np.iscomplexobj(values) or not np.isfinite(values).all()):
        raise ValueError("joint_qpos must be a finite numeric (43,) vector in radians")
    lookup = dict(zip(names, values))
    result = np.zeros((1, 45), dtype=np.float32)
    result[0, :43] = [lookup[name] for name in TRAINING_STATE_JOINT_NAMES]
    if not np.isfinite(result).all():
        raise ValueError("joint_qpos is outside float32 range")
    return result


def split_action80(action) -> tuple[np.ndarray, dict[str, float]]:
    """Return the v1.1 body latent and 14 named radian hand targets.

    The final 2 channels are training padding; they never control neck joints.
    """
    action = np.asarray(action)
    if (action.shape != (80,) or not np.issubdtype(action.dtype, np.number) or
            np.iscomplexobj(action) or not np.isfinite(action).all()):
        raise ValueError("Expected one finite (80,) dual-source action, never a legacy 36/78-vector")
    body = np.asarray(action[:64], dtype=np.float32).copy()
    hands = {name: float(value) for name, value in zip(TRAINING_HAND_JOINT_NAMES, action[64:78])}
    return body, hands


def simple_hand_targets(hand_targets: Mapping[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """Convert named targets to G1Sonic's thumb/index/middle hand actuator order."""
    if set(hand_targets) != set(SIMPLE_HAND_JOINT_NAMES):
        raise ValueError("Expected exactly the 14 named Dex3 hand joints")
    ordered = np.asarray([hand_targets[name] for name in SIMPLE_HAND_JOINT_NAMES], dtype=np.float32)
    if not np.isfinite(ordered).all():
        raise ValueError("Hand targets must be finite float32 radians")
    return ordered[:7].copy(), ordered[7:].copy()


class HistoryWarmingUp(RuntimeError):
    def __init__(self, detail: dict):
        super().__init__("Collect one second of real simulator history")
        self.detail = detail


class DualSourceHttpActionClient:
    """Local HTTP transport with explicit observe/reset and warming-up handling."""
    def __init__(self, server_ip="127.0.0.1", server_port=22085, *, timeout_s=120.0, session=None):
        if server_ip not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Independent SIMPLE evaluation requires a localhost model service")
        if not 0 < int(server_port) < 65536 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Invalid localhost port or HTTP timeout")
        host = "[::1]" if server_ip == "::1" else server_ip
        self.base_url = f"http://{host}:{int(server_port)}"
        self.timeout_s = float(timeout_s)
        self.session = session or requests.Session()
        # Local traffic must not leave the machine through HTTP(S)_PROXY.
        self.session.trust_env = False

    def _post(self, endpoint: str, payload: dict):
        response = self.session.post(self.base_url + endpoint, json=payload, timeout=self.timeout_s)
        if response.status_code == 425:
            detail = response.json().get("detail", {})
            if isinstance(detail, dict) and detail.get("status") == "warming_up":
                raise HistoryWarmingUp(detail)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _request(image_dict, instruction, state_dict, history):
        return RequestMessage(image_dict, instruction, history, state_dict, {}, [], "simple",
                              datetime.now().isoformat()).serialize()

    def reset(self, session_id: str, episode_id: str):
        return self._post("/reset", {"session_id": session_id, "episode_id": episode_id})

    def observe(self, image_dict: dict, *, history: dict):
        return self._post("/observe", self._request(image_dict, "", {}, history))

    def query_action(self, image_dict: dict, instruction: str, state_dict: dict, *, history: dict):
        result = ResponseMessage.deserialize(self._post(
            "/act", self._request(image_dict, instruction, state_dict, history)))
        action = np.asarray(result.action)
        if action.shape != (30, 80) or not np.isfinite(action).all():
            raise ValueError("Dual-source server must return finite actions with shape (30, 80)")
        if result.err != 0.0:
            raise RuntimeError(f"Dual-source service reported an error: {result.err}")
        return action

    def close(self):
        self.session.close()


class BodyTokenV11Controller(Protocol):
    """Supplied by the separately verified SONIC v1.1 decoder/controller adapter.

    identity binds decoder_sha256 and a nonempty provenance to body_token_v1_1.
    hold must apply real physical control while history accumulates; it must
    document whether it provides balance or only holds measured joint poses.
    step consumes denormalized latents and named absolute hand radians.
    """
    identity: Mapping[str, str]

    def reset(self) -> None: ...
    def hold(self, observation: dict) -> ActionCmd: ...
    def step(self, body_token: np.ndarray, hand_targets: Mapping[str, float], observation: dict) -> ActionCmd: ...


class DualSourcePsi0Agent:
    """Collect every frame, then execute v1.1 chunks on the simulation clock.

    Controller injection is deliberate: the old navigation/joint-space SONIC
    path is incompatible. Construction fails until a v1.1 adapter is supplied.
    A 30 Hz or faster rendered observation stream is required. No duplicate
    image or wall-clock timestamp is used to fill missing history.
    """
    def __init__(self, robot, host="127.0.0.1", port=22085, *, controller=None,
                 client=None, execute_steps=30, session_id=None, warmup_control=None):
        if controller is None:
            raise ValueError("Supply a verified body_token_v1_1 SONIC controller; legacy WBC is incompatible")
        identity = getattr(controller, "identity", {})
        decoder_sha = identity.get("decoder_sha256", "")
        if (identity.get("body_token_version") != "body_token_v1_1" or
                len(decoder_sha) != 64 or any(c not in "0123456789abcdef" for c in decoder_sha) or
                not identity.get("provenance")):
            raise ValueError("Controller must bind body_token_v1_1 to decoder SHA256 and compatibility provenance")
        if isinstance(execute_steps, bool) or not isinstance(execute_steps, int) or not 1 <= execute_steps <= 30:
            raise ValueError("execute_steps must be an integer in [1, 30]")
        self.robot, self.controller = robot, controller
        self.warmup_control = warmup_control
        self.client = client or DualSourceHttpActionClient(host, port)
        self.execute_steps = execute_steps
        self.session_id = session_id or f"dual-source-{uuid.uuid4().hex}"
        self.episode_id = None
        self._clear_local()

    def _clear_local(self):
        self._chunk = None
        self._chunk_time = None
        self._last_time = None
        self.warming_up = True
        self.warmup_phase = "warmup_measured_pose_pd"

    def reset(self, *, episode_id=None, **kwargs):
        self._clear_local()
        self.episode_id = str(episode_id) if episode_id is not None else uuid.uuid4().hex
        self.client.reset(self.session_id, self.episode_id)
        self.controller.reset()
        if self.warmup_control is not None:
            self.warmup_control.reset()

    def get_action(self, observation, instruction=None, info=None, **kwargs):
        if self.episode_id is None:
            self.reset()
        t = observation.get("sim_time_s")
        if isinstance(t, bool) or not isinstance(t, (int, float)) or not math.isfinite(t) or t < 0:
            raise ValueError("Observation requires finite nonnegative sim_time_s from mjData.time")
        if self._last_time is not None:
            dt = t - self._last_time
            if dt <= 1e-8:
                raise ValueError("Simulator time must advance; reset the agent on a new episode")
            if dt > 1 / ACTION_FPS + .001:
                raise ValueError("Dual-source action execution requires rendered observations at >=30 Hz")
        states = state43_to45(observation["joint_qpos"], observation["joint_names"])
        rgb = np.asarray(observation["head_stereo_left"])
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("head_stereo_left must be a real uint8 HWC RGB frame")
        history = {"session_id": self.session_id, "episode_id": self.episode_id, "sim_time_s": float(t)}
        images = {"rgb_head_stereo_left": rgb}
        # Every camera frame is observed, including during chunk execution.
        self.client.observe(images, history=history)
        self._last_time = float(t)
        idx = 0 if self._chunk_time is None else int(math.floor((t - self._chunk_time) * ACTION_FPS + 1e-7))
        if self._chunk is None or idx >= self.execute_steps:
            try:
                chunk = np.asarray(self.client.query_action(
                    images, instruction or "", {"states": states}, history=history))
            except HistoryWarmingUp:
                self.warming_up = True
                if self.warmup_control is not None:
                    command = self.warmup_control.command(self.robot, self.controller, observation)
                    self.warmup_phase = self.warmup_control.phase
                    return command
                return self.controller.hold(observation)
            if chunk.shape != (30, 80) or not np.isfinite(chunk).all():
                raise ValueError("Expected a finite (30, 80) denormalized policy action chunk")
            self._chunk, self._chunk_time, idx = chunk.copy(), float(t), 0
            self.warming_up = False
        latent, hands = split_action80(self._chunk[idx])
        command = self.controller.step(latent, hands, observation)
        if not isinstance(command, ActionCmd):
            raise TypeError("SONIC controller must return a SIMPLE ActionCmd")
        return command
