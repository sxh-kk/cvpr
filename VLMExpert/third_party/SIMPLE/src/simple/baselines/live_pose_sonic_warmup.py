"""Optional deployment warmup using a frozen live-pose G1 mode-0 reference.

The repeated frames below describe a desired constant reference, not measured
proprioception or camera history. All decoder/history samples remain real.
"""
from pathlib import Path
import hashlib
import json

import numpy as np

from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, SIMPLE_HAND_JOINT_NAMES, state43_to45
from simple.baselines.sonic_v1_1 import POLICY_TO_BODY, OFFICIAL_OBSERVATION_SHA256


ENCODER_SHA256 = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference_input(body_q, reference_wxyz, actual_wxyz):
    """Official mode0 layout; q absolute, dq zero, heading from live robot."""
    import mujoco
    body_q = np.asarray(body_q, dtype=float)
    reference_wxyz, actual_wxyz = np.asarray(reference_wxyz, float), np.asarray(actual_wxyz, float)
    if (body_q.shape != (29,) or reference_wxyz.shape != (4,) or actual_wxyz.shape != (4,) or
            not np.isfinite(np.r_[body_q, reference_wxyz, actual_wxyz]).all() or
            not np.allclose([np.linalg.norm(reference_wxyz), np.linalg.norm(actual_wxyz)], 1, atol=1e-5)):
        raise ValueError("Warmup requires finite live absolute body29 and unit root quaternions")
    w, x, y, z = actual_wxyz
    yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    c, s = np.cos(yaw), np.sin(yaw)
    remove_heading = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, reference_wxyz)
    six = (remove_heading @ rotation.reshape(3,3))[:, :2].reshape(6)
    value = np.zeros((1, 1751), np.float32)
    value[0,4:294] = np.tile(body_q[POLICY_TO_BODY], 10)
    value[0,584:644] = np.tile(six, 10)
    return value


class LivePoseSonicWarmup:
    def __init__(self, resource_dir):
        import onnxruntime as ort
        root = Path(resource_dir).resolve(strict=True)
        path = root / "model_encoder.onnx"
        if _sha(path) != ENCODER_SHA256 or _sha(root/"observation_config.yaml") != OFFICIAL_OBSERVATION_SHA256:
            raise ValueError("Live-pose warmup needs the fixed official SONIC v1.1 encoder/config")
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs)!=1 or list(inputs[0].shape)!=[1,1751] or inputs[0].type!="tensor(float)" or
                len(outputs)!=1 or list(outputs[0].shape)!=[1,64]):
            raise ValueError("Unexpected v1.1 warmup encoder contract")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name
        self.reset()

    def reset(self):
        self.body_q = self.root_wxyz = self.hands = None
        self.phase = "warmup_measured_pose_pd"
        self.receipt = {"kind":"live_initial_pose_sonic_v1_1", "encoder_sha256":ENCODER_SHA256,
            "helper_source_sha256":_sha(__file__),
            "encoder_mode":0, "reference":"frozen first live absolute joint pose and world root orientation; zero desired joint velocity",
            "heading":"reference world orientation minus current actual robot yaw",
            "initialization_count":0, "encoder_calls":0, "events":[],
            "proprioception":"ten real 50Hz frames before latent; no invented measurements",
            "camera_history":"real observations only; no source RGB, teacher future or state forcing"}

    def command(self, robot, controller, observation):
        state43_to45(observation["joint_qpos"], observation["joint_names"])
        named = dict(zip(observation["joint_names"], observation["joint_qpos"]))
        actual_root = np.asarray(robot.mjData.body("pelvis").xquat, dtype=float).copy()
        if self.body_q is None:
            self.body_q = np.array([named[name] for name in BODY_JOINT_NAMES], dtype=float)
            self.root_wxyz = actual_root.copy()
            self.hands = {name:float(named[name]) for name in SIMPLE_HAND_JOINT_NAMES}
            self.receipt.update(initialization_count=1, reference_body_q29=self.body_q.tolist(),
                reference_root_wxyz=self.root_wxyz.tolist(), reference_hands=dict(self.hands),
                first_sim_time_s=float(observation["sim_time_s"]))
            reference = {"body_joint_names":BODY_JOINT_NAMES,"body_q29":self.body_q.tolist(),
                "root_wxyz":self.root_wxyz.tolist(),"hands":self.hands}
            self.receipt["initial_live_reference_sha256"] = hashlib.sha256(json.dumps(
                reference,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        if len(controller._history) < 10:
            self.phase = "warmup_measured_pose_pd"
            return controller.hold(observation)
        value = reference_input(self.body_q, self.root_wxyz, actual_root)
        token = np.asarray(self.session.run([self.output_name], {self.input_name:value})[0], np.float32)
        if token.shape != (1,64) or not np.isfinite(token).all():
            raise ValueError("Invalid live-pose warmup token")
        self.phase = "warmup_live_pose_sonic_v1_1"
        self.receipt["encoder_calls"] += 1
        self.receipt["events"].append({"sim_time_s":float(observation["sim_time_s"]),
            "actual_root_wxyz":actual_root.tolist(), "token64":token[0].tolist()})
        return controller.step(token[0], self.hands, observation)
