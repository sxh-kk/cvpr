"""Stream video by presentation timestamps, without loading an episode into RAM."""

from pathlib import Path
from fractions import Fraction
import numpy as np
import av


def _nearest_frames(iterator, targets, max_error_s):
    targets = np.asarray(targets, dtype=float)
    if len(targets) and (not np.isfinite(targets).all() or np.any(np.diff(targets) < 0)):
        raise ValueError("Video targets must be finite, non-decreasing")
    if not np.isfinite(max_error_s) or max_error_s <= 0:
        raise ValueError("Video tolerance must be positive")
    previous, last_pts = None, -np.inf
    def next_frame():
        nonlocal last_pts
        frame = next(iterator, None)
        if frame is not None:
            if frame[0] <= last_pts:
                raise ValueError("Decoded video PTS must be strictly increasing")
            last_pts = frame[0]
        return frame
    try:
        current = next_frame() if len(targets) else None
        for target in targets:
            while current is not None and current[0] < target:
                previous, current = current, next_frame()
            candidates = [x for x in [previous, current] if x is not None]
            closest = min(candidates, key=lambda x: abs(x[0]-target)) if candidates else None
            if closest is None or abs(closest[0]-target) > max_error_s:
                yield None, None
            else:
                yield closest[1], closest[0]
    finally:
        iterator.close()


def _mp4_frames(path, decoder_threads=0):
    if not Path(path).exists():
        raise FileNotFoundError(path)
    with av.open(str(path)) as container:
        if decoder_threads:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = decoder_threads
            stream.thread_type = "AUTO"
        for frame in container.decode(video=0):
            if frame.pts is not None:
                yield float(frame.pts*frame.time_base), frame.to_ndarray(format="rgb24")


def nearest_video_frames(path, targets, max_error_s, *, decoder_threads=0):
    yield from _nearest_frames(_mp4_frames(path, decoder_threads), targets, max_error_s)


def _mcap_frames(video, decoder_threads=0):
    from .egosuite_mcap import decoded_messages, timestamp_ns
    codecs = {"h264": "h264", "h265": "hevc", "vp9": "vp9", "av1": "av1"}
    codec, current_format, last_stamp = None, None, None
    origin = int(video["clock_origin_ns"])
    packet_time_base = Fraction(1, 1_000_000_000)
    bad_ns = {round(t*1e9) for t in video.get("bad_timestamps_s", [])}
    def output(frames):
        for frame in frames:
            if frame.pts is None:
                raise ValueError("MCAP decoder lost capture timestamps")
            # PyAV can omit time_base on delayed frames returned by flush().
            # Every packet has the explicit capture-nanosecond time base below.
            pts = float(frame.pts*(frame.time_base or packet_time_base))
            yield pts, None if round(pts*1e9) in bad_ns else frame.to_ndarray(format="rgb24")
    messages = decoded_messages(video["path"], [video["topic"]])
    drained = False
    try:
        for _, message, decoded in messages:
            if decoded.format not in codecs:
                raise ValueError(f"Unsupported MCAP CompressedVideo format: {decoded.format}")
            if current_format is not None and current_format != decoded.format:
                raise ValueError("MCAP video codec changes within an episode")
            if codec is None:
                current_format = decoded.format
                codec = av.CodecContext.create(codecs[current_format], "r")
                if decoder_threads:
                    codec.thread_count = decoder_threads
                    codec.thread_type = "AUTO"
            stamp, _ = timestamp_ns(decoded, message)
            if last_stamp is not None and stamp <= last_stamp:
                raise ValueError("MCAP video capture timestamps must be strictly increasing in file order")
            last_stamp = stamp
            # Each Foxglove message is a complete access unit, not an MP4 container.
            # Keep codec state across delta frames and assign the message capture PTS.
            packet = av.Packet(decoded.data)
            packet.pts = packet.dts = stamp-origin
            packet.time_base = packet_time_base
            yield from output(codec.decode(packet))
        if codec is not None:
            drained = True
            yield from output(codec.decode(None))
    finally:
        # Closing a partial MCAP iterator can otherwise leave frame-threaded
        # decoding in flight and deadlock avcodec_free_context on some streams.
        if codec is not None and not drained:
            try:
                codec.decode(None)
            except av.FFmpegError:
                pass  # Cleanup must preserve the original exception/GeneratorExit.
        messages.close()


def nearest_mcap_video_frames(video, targets, max_error_s, *, decoder_threads=0):
    yield from _nearest_frames(_mcap_frames(video, decoder_threads), targets, max_error_s)


def image_preprocessor(config):
    """Explicit image geometry policy, recorded in feature provenance."""
    mode = config.get("mode", "raw")
    rotate = int(config.get("rotate_ccw_quarters", 0))
    if mode in {"raw", "preundistorted"}:
        return lambda image: np.ascontiguousarray(np.rot90(image, rotate))
    if mode != "aria":
        raise ValueError("video_preprocessing.mode must be raw, preundistorted, or aria")
    from projectaria_tools.core import calibration
    from projectaria_tools.core import data_provider
    provider = data_provider.create_vrs_data_provider(config["vrs_path"])
    device = provider.get_device_calibration()
    source_calib = device.get_camera_calib(config.get("camera_label", "camera-rgb"))
    size = int(config.get("output_size", 512))
    target_calib = calibration.get_linear_camera_calibration(size, size, float(config.get("focal_length", 150)))
    native_size = tuple(int(x) for x in source_calib.get_image_size())
    def process(image):
        # Native Aria image orientation must be specified, never guessed from dimensions.
        native = np.ascontiguousarray(np.rot90(image, rotate))
        if native.shape[1::-1] != native_size:
            raise ValueError(f"Video resolution/orientation {native.shape[1::-1]} does not match Aria calibration {native_size}; supply a correctly calibrated or preundistorted video")
        undistorted = calibration.distort_by_calibration(native, target_calib, source_calib)
        return np.ascontiguousarray(np.rot90(undistorted, int(config.get("output_rotate_ccw_quarters", 0))))
    return process
