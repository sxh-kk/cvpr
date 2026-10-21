"""Checked CPU ffmpeg raw-RGB recording at the simulator's actual camera rate."""
from __future__ import annotations

from pathlib import Path
import math
import shutil
import subprocess


class RawVideoRecorder:
    def __init__(self, path, *, fps, shape):
        self.path = Path(path)
        self.fps, self.shape = float(fps), tuple(shape)
        if len(self.shape) != 3 or self.shape[2] != 3 or any(size <= 0 for size in self.shape):
            raise ValueError("Video frames must be HxWx3 RGB")
        if not math.isfinite(self.fps) or self.fps <= 0 or self.shape[0] % 2 or self.shape[1] % 2:
            raise ValueError("H.264 yuv420p requires positive fps and even frame dimensions")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("ffmpeg is required when --video is selected")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path = self.path.with_suffix(".ffmpeg.log")
        self._log = self.log_path.open("wb")
        self.frames, self.first_time, self.last_time, self.closed = 0, None, None, False
        height, width, _ = self.shape
        self.command = [executable, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                        "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{width}x{height}",
                        "-framerate", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                        "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", str(self.path)]
        try:
            self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._log)
        except Exception:
            self._log.close()
            raise

    def write(self, frame, *, sim_time_s):
        import numpy as np

        if self.closed:
            raise RuntimeError("Video recorder is closed")
        frame, timestamp = np.asarray(frame), float(sim_time_s)
        if frame.shape != self.shape or frame.dtype != np.uint8 or not np.isfinite(timestamp):
            raise ValueError("Video RGB shape/dtype or simulator timestamp mismatch")
        if self.last_time is not None:
            if abs(timestamp - self.last_time) <= 1e-8:
                return
            if abs(timestamp - self.last_time - 1/self.fps) > 1e-5:
                raise ValueError("Video requires every genuine camera frame at the declared simulation rate")
        if self.process.poll() is not None:
            raise RuntimeError(f"ffmpeg exited while recording; inspect {self.log_path}")
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(f"ffmpeg pipe failed; inspect {self.log_path}") from error
        self.frames += 1
        self.first_time = timestamp if self.first_time is None else self.first_time
        self.last_time = timestamp

    def close(self):
        if not self.closed:
            self.closed = True
            try:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
                    raise RuntimeError(f"ffmpeg did not finalize the video: {self.log_path}")
            finally:
                self._log.close()
            if self.process.returncode != 0 or not self.frames or not self.path.is_file() or not self.path.stat().st_size:
                raise RuntimeError(f"ffmpeg failed or produced an empty video: {self.log_path}")
        return {"path": str(self.path), "fps": self.fps, "frames": self.frames,
                "first_sim_time_s": self.first_time, "last_sim_time_s": self.last_time,
                "ffmpeg_returncode": self.process.returncode, "log": str(self.log_path)}
