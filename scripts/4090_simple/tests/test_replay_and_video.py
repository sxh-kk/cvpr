"""CPU checks of real ffmpeg output and one-time source state initialization."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from replay_carrybox_v11 import initialize_source_state, save_physics_model, finalize_replay_report
from video_capture import RawVideoRecorder
from simple.baselines.dual_source_psi0 import BODY_JOINT_NAMES, TRAINING_STATE_JOINT_NAMES


class ReplayVideoTests(unittest.TestCase):
    def test_capture_is_sealed_before_isaac_close_exits_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Recorder:
                def close(self):
                    (root / "capture.json").write_text('{"state":"capture_closed"}')
            def isaac_close():
                self.assertTrue((root / "capture.json").is_file())
                self.assertEqual(json.loads((root / "result.json").read_text())["status"], "source_end")
                raise SystemExit(0)
            with self.assertRaises(SystemExit):
                finalize_replay_report(root / "result.json", {"status":"source_end"}, Recorder(), isaac_close)

    def test_failed_capture_close_records_error_before_isaac_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Recorder:
                def close(self):
                    raise OSError("capture disk error")
            def isaac_close():
                report = json.loads((root / "result.json").read_text())
                self.assertEqual(report["status"], "error")
                self.assertEqual(report["physical_outcome_status"], "source_end")
                self.assertIn("capture disk error", report["command_capture_close_error"])
                raise SystemExit(0)
            with self.assertRaises(SystemExit):
                finalize_replay_report(root / "result.json", {"status":"source_end","physical_outcome_status":"source_end"}, Recorder(), isaac_close)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe are required")
    def test_actual_h264_contains_all_real_camera_frames_at_50fps(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.mp4"
            recorder = RawVideoRecorder(path, fps=50, shape=(24, 32, 3))
            for index in range(10):
                recorder.write(np.full((24, 32, 3), index*20, np.uint8), sim_time_s=.005+index*.02)
            recorder.write(np.full((24, 32, 3), 180, np.uint8), sim_time_s=.185)
            summary = recorder.close()
            self.assertEqual(summary["frames"], 10)
            self.assertEqual(summary["ffmpeg_returncode"], 0)
            result = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
                                     "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json", str(path)],
                                    capture_output=True, text=True, check=True)
            stream = json.loads(result.stdout)["streams"][0]
            self.assertEqual(stream["nb_read_frames"], "10")
            self.assertEqual(stream["r_frame_rate"], "50/1")

    @unittest.skipUnless(shutil.which("ffmpeg"), "ffmpeg is required")
    def test_missing_camera_frames_and_failed_ffmpeg_never_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = RawVideoRecorder(Path(directory) / "frames.mp4", fps=50, shape=(24, 32, 3))
            frame = np.zeros((24, 32, 3), np.uint8)
            recorder.write(frame, sim_time_s=0)
            with self.assertRaises(ValueError):
                recorder.write(frame, sim_time_s=.04)
            recorder.close()
            recorder = RawVideoRecorder(Path(directory) / "cannot_encode.invalid", fps=50, shape=frame.shape)
            with self.assertRaises(RuntimeError):
                try:
                    recorder.write(frame, sim_time_s=0)
                finally:
                    recorder.close()

    def test_initialize_source_q_root_object_and_velocities_once_without_advancing_time(self):
        spec = mujoco.MjSpec()
        root = spec.worldbody.add_body(name="pelvis")
        root.add_freejoint(name="floating_base_joint")
        root.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.03, 0, 0], mass=1)
        for name in TRAINING_STATE_JOINT_NAMES:
            body = root.add_body(name="body_" + name)
            body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1])
            body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[.01, 0, 0], mass=.01)
        cube = spec.worldbody.add_body(name="cube")
        cube.add_freejoint(name="cube_joint")
        cube.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[.115, .19, .217], mass=.1)
        model = spec.compile()
        data = mujoco.MjData(model)
        data.time = 3.2
        mujoco.mj_forward(model, data)
        teacher = {"observation.state": np.arange(43)[None]/100,
                   "observation.base_pose": np.array([[-1.2, 0, .8, 1, 0, 0, 0]]),
                   "observation.base_vel": np.arange(6)[None]/1000,
                   "observation.body_joint_velocities": np.arange(29)[None]/200,
                   "observation.object_poses": np.array([[-.55, .01, .217, 1, 0, 0, 0]])}
        pose = SimpleNamespace(position=None, quaternion=None)
        task = SimpleNamespace(_reset_progress=lambda: None, layout=SimpleNamespace(actors={"target": SimpleNamespace(pose=pose)}))
        raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=model, mjData=data), task=task, isaac=None)
        receipt = initialize_source_state(raw, teacher)
        np.testing.assert_allclose([data.joint(name).qpos[0] for name in TRAINING_STATE_JOINT_NAMES], teacher["observation.state"][0])
        np.testing.assert_allclose([data.joint(name).qvel[0] for name in BODY_JOINT_NAMES], teacher["observation.body_joint_velocities"][0])
        np.testing.assert_allclose(data.qpos[:7], teacher["observation.base_pose"][0])
        np.testing.assert_allclose(data.qvel[:6], teacher["observation.base_vel"][0])
        np.testing.assert_allclose(data.joint("cube_joint").qpos, teacher["observation.object_poses"][0])
        self.assertEqual(data.time, 3.2)
        self.assertEqual(receipt["initialization_count"], 1)
        with tempfile.TemporaryDirectory() as directory:
            physics = save_physics_model(model, spec, Path(directory))
            self.assertTrue(physics["cpu_reload_verified"])
            self.assertEqual(mujoco.MjModel.from_binary_path(physics["path"]).nq, model.nq)


if __name__ == "__main__":
    unittest.main()
