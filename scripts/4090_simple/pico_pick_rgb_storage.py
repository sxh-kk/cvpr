"""Compact derived RTX PNGs with exact, independently reproducible PNG bytes.

Only ``episode/capture/rgb/*.png`` can be removed. Original deferred physics,
commands, metadata and render receipts are never changed. Callers own the
explicit cohort; this helper does not discover episodes or start a renderer.
"""
from collections import OrderedDict
from contextlib import contextmanager
from fractions import Fraction
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import uuid
import zlib

import av
import numpy as np
import PIL
from PIL import Image


CONTRACT = "cvpr-pico-pick-exact-derived-rgb-storage-v1"
MASTER_NAME = "rgb_master.mp4"
RECEIPT_NAME = "rgb_storage.json"
COMPLETE_NAME = "rgb_storage_complete.json"
_READERS = OrderedDict()


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _signature(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), "Missing or linked RGB authority: " + str(path))
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _episode(directory):
    root = Path(directory).absolute()
    require(root.is_dir() and not any(p.is_symlink() for p in (root, *root.parents)),
            "Use a real canonical episode directory")
    capture = root / "capture"
    require(capture.is_dir() and not capture.is_symlink(), "Missing real derived capture directory")
    rgb = capture / "rgb"
    require(not rgb.is_symlink(), "Derived RGB directory cannot be a symlink")
    return root, capture


def _json(path):
    _signature(path)
    return json.loads(Path(path).read_text())


def _encoder():
    return {"format": "PNG", "mode": "RGB", "compress_level": 3,
            "PIL_version": PIL.__version__, "zlib_version": zlib.ZLIB_VERSION,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION}


def _png_bytes(pixels):
    require(pixels.dtype == np.uint8 and pixels.ndim == 3 and pixels.shape[2] == 3,
            "Require original uint8 RGB geometry")
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", compress_level=3)
    return stream.getvalue()


def _source(directory):
    root, capture = _episode(directory)
    authorities = [capture / "capture.json", capture / "events.jsonl",
                   root / "rtx_completion.json", root / "deferred_capture/capture.json"]
    signatures = {str(p): _signature(p) for p in authorities}
    manifest, completion, raw = _json(authorities[0]), _json(authorities[2]), _json(authorities[3])
    require(manifest.get("status") == "raw_capture_completed_pending_independent_audit"
            and completion.get("status") == "deferred_rtx_completed_pending_independent_audit"
            and completion.get("archived_physics_steps") == 0
            and raw.get("status") == "raw_physics_capture_completed_pending_rtx"
            and manifest.get("command_rate_hz") == raw.get("command_rate_hz") == 30,
            "Only completed derived RTX RGB may be compacted")
    hashes = {str(p.relative_to(root)): sha(p) for p in authorities}
    require(manifest.get("events_sha256") == hashes["capture/events.jsonl"]
            and completion.get("capture_manifest_sha256") == hashes["capture/capture.json"]
            and completion.get("source_manifest_sha256") == hashes["deferred_capture/capture.json"],
            "RTX/source/event SHA authority differs")
    frames, paths = [], set()
    with authorities[1].open() as stream:
        for index, line in enumerate(stream):
            event = json.loads(line)
            rgb = event["rgb"]
            name = Path(rgb["path"])
            require(event.get("event_index") == index and event.get("capture_index30") == index
                    and name.parts == ("rgb", f"frame_{index:06d}.png") and str(name) not in paths,
                    "Only the exact dense derived PNG frame paths are supported")
            paths.add(str(name))
            require(rgb.get("channels") == 3 and rgb.get("dtype") == "uint8"
                    and type(rgb.get("width")) is int and rgb["width"] > 0
                    and type(rgb.get("height")) is int and rgb["height"] > 0,
                    "Original RGB geometry is missing")
            require(all(isinstance(rgb.get(k), str) and len(rgb[k]) == 64
                        and all(c in "0123456789abcdef" for c in rgb[k])
                        for k in ("file_sha256", "pixel_sha256")), "Invalid original RGB SHA")
            frames.append({"index": index, **{key: rgb[key] for key in
                ("path", "file_sha256", "pixel_sha256", "width", "height", "channels", "dtype")}})
    require(len(frames) == manifest.get("event_count") == completion.get("completed_frames") > 0,
            "Completed derived RGB frame counts differ")
    require(len({(f["height"], f["width"]) for f in frames}) == 1, "RGB geometry changed mid-episode")
    require(all(_signature(Path(p)) == stat for p, stat in signatures.items()),
            "Source authorities changed while reading RGB")
    return root, capture, frames, hashes, signatures


def _publish(path, value, *, replace=False):
    require(not path.is_symlink(), "RGB storage authority cannot be a symlink")
    data = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        elif path.exists():
            require(path.read_bytes() == data, "Existing RGB authority differs: " + str(path))
        else:
            os.link(temporary, path)  # Publish without overwriting a concurrent file.
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _locked(capture):
    descriptor = os.open(capture / ".rgb_storage.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _pixels(path, frame):
    _signature(path)
    require(sha(path) == frame["file_sha256"], "Original PNG file SHA differs: " + str(path))
    with Image.open(path) as image:
        require(image.mode == "RGB", "Original PNG must be RGB")
        pixels = np.asarray(image).copy()
    require(list(pixels.shape) == [frame["height"], frame["width"], 3]
            and hashlib.sha256(pixels.tobytes()).hexdigest() == frame["pixel_sha256"],
            "Original PNG pixels/geometry differ")
    return pixels


def _encode(capture, frames, output):
    with av.open(str(output), "w", format="mp4") as container:
        stream = container.add_stream("libx264rgb", rate=30)
        stream.options = {"crf": "0", "preset": "veryslow"}
        stream.pix_fmt = "rgb24"
        stream.width, stream.height = frames[0]["width"], frames[0]["height"]
        stream.time_base = Fraction(1, 30)
        stream.codec_context.thread_count = 1
        for item in frames:
            pixels = _pixels(capture / item["path"], item)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = item["index"], Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decoded(master):
    with av.open(str(master)) as container:
        require(len(container.streams.video) == 1 and len(container.streams) == 1,
                "RGB master must contain exactly one video stream")
        stream = container.streams.video[0]
        require(stream.average_rate == 30 and stream.codec_context.name == "h264",
                "RGB master must retain the original 30fps lossless H264 stream")
        stream.codec_context.thread_count = 1
        for index, frame in enumerate(container.decode(video=0)):
            require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - index / 30) < 1e-7,
                    "RGB master frame PTS differs")
            yield frame.to_ndarray(format="rgb24")


def _verify_master(master, frames):
    count = 0
    for count, pixels in enumerate(_decoded(master), 1):
        require(count <= len(frames), "RGB master has extra frames")
        item = frames[count - 1]
        require(list(pixels.shape) == [item["height"], item["width"], 3]
                and hashlib.sha256(pixels.tobytes()).hexdigest() == item["pixel_sha256"],
                "Decoded RGB master pixels differ at frame " + str(count - 1))
        require(hashlib.sha256(_png_bytes(pixels)).hexdigest() == item["file_sha256"],
                "Exact original PNG bytes cannot be reproduced at frame " + str(count - 1))
    require(count == len(frames), "RGB master is missing frames")


def compact_episode(directory, delete_pngs=True):
    """Verify all exact PNG reproductions before removing any derived PNG.

    Interrupted removal resumes from the immutable verified receipt. Existing
    or corrupt masters/receipts are preserved and rejected, never overwritten.
    ``delete_pngs=False`` creates a verified master while retaining all PNGs.
    """
    require(type(delete_pngs) is bool, "Explicit boolean delete_pngs required")
    root, capture = _episode(directory)
    with _locked(capture):
        root, capture, frames, sources, signatures = _source(root)
        master, receipt = capture / MASTER_NAME, capture / RECEIPT_NAME
        require(not any((capture / name).is_symlink() for name in (MASTER_NAME, RECEIPT_NAME, COMPLETE_NAME)),
                "RGB storage authorities cannot be symlinks")
        if not master.exists():
            require(not receipt.exists(), "RGB receipt exists without its master; preserve for diagnosis")
            partial = capture / ("." + MASTER_NAME + "." + uuid.uuid4().hex + ".partial")
            try:
                _encode(capture, frames, partial)
                _verify_master(partial, frames)
                with partial.open("rb") as stream:
                    os.fsync(stream.fileno())
                os.link(partial, master)
            finally:
                partial.unlink(missing_ok=True)
        _signature(master)
        master_stat = _signature(master)
        _verify_master(master, frames)
        master_sha = sha(master)
        value = {"contract": CONTRACT, "status": "all_original_png_bytes_verified_reproducible",
            "source_files_sha256": sources, "module_sha256": sha(Path(__file__)),
            "png_encoder": _encoder(), "master": {"path": MASTER_NAME, "sha256": master_sha,
                "bytes": master.stat().st_size, "fps": 30, "codec": "libx264rgb", "crf": 0,
                "preset": "veryslow", "width": frames[0]["width"], "height": frames[0]["height"]},
            "frames": len(frames), "png_files": frames,
            "all_decoded_pixel_sha256_verified": True, "all_reencoded_png_file_sha256_verified": True,
            "original_physics_modified": False, "scope": "derived RTX RGB only; no rescale, resample or dynamics"}
        _publish(receipt, value)
        # Recheck every still-existing source before beginning any removal.
        for item in frames:
            path = capture / item["path"]
            if path.exists() or path.is_symlink():
                _pixels(path, item)
            else:
                require(receipt.exists(), "Original PNG missing before verified storage receipt")
        require(all(_signature(Path(p)) == stat for p, stat in signatures.items())
                and _signature(master) == master_stat, "RGB authorities changed before deletion")
        if delete_pngs:
            for item in frames:
                path = capture / item["path"]
                if path.exists():
                    require(not path.is_symlink() and sha(path) == item["file_sha256"],
                            "PNG changed immediately before unlink")
                    path.unlink()
            descriptor = os.open(capture / "rgb", os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        marker = {"contract": CONTRACT, "status": "complete", "receipt_sha256": sha(receipt),
            "master_sha256": master_sha, "frames": len(frames), "derived_pngs_removed": delete_pngs,
            "original_physics_modified": False}
        _publish(capture / COMPLETE_NAME, marker, replace=True)
        return {**marker, "master_path": str(master), "receipt_path": str(receipt)}


def verify_compacted_episode(directory):
    """Independently decode every frame and recreate every original PNG SHA."""
    root, capture, frames, sources, signatures = _source(directory)
    receipt_path, master = capture / RECEIPT_NAME, capture / MASTER_NAME
    receipt, marker = _json(receipt_path), _json(capture / COMPLETE_NAME)
    require(receipt.get("contract") == marker.get("contract") == CONTRACT
            and receipt.get("status") == "all_original_png_bytes_verified_reproducible"
            and marker.get("status") == "complete" and receipt.get("png_files") == frames
            and receipt.get("source_files_sha256") == sources
            and receipt.get("frames") == marker.get("frames") == len(frames)
            and receipt.get("module_sha256") == sha(Path(__file__))
            and receipt.get("png_encoder") == _encoder()
            and marker.get("receipt_sha256") == sha(receipt_path), "RGB storage receipt/implementation differs")
    master_stat = _signature(master)
    require(receipt.get("master", {}).get("path") == MASTER_NAME
            and receipt["master"].get("sha256") == marker.get("master_sha256") == sha(master),
            "RGB master SHA differs")
    _verify_master(master, frames)
    require(_signature(master) == master_stat and all(_signature(Path(p)) == stat for p, stat in signatures.items()),
            "RGB authority changed during independent verification")
    return {"contract": CONTRACT, "passed": True, "frames": len(frames),
            "master_path": str(master), "master_sha256": receipt["master"]["sha256"],
            "receipt_sha256": sha(receipt_path), "exact_png_file_sha256_verified": True}


class _Reader:
    def __init__(self, root):
        verify_compacted_episode(root)
        self.root, self.capture, self.frames, _, signatures = _source(root)
        self.signatures = {**signatures, **{str(self.capture / name): _signature(self.capture / name)
            for name in (MASTER_NAME, RECEIPT_NAME, COMPLETE_NAME)}}
        self.by_path = {item["path"]: item for item in self.frames}
        self.generator = None
        self.cursor = -1
        self.last_png = None

    def unchanged(self):
        return all(_signature(Path(path)) == before for path, before in self.signatures.items())

    def close(self):
        if self.generator is not None:
            self.generator.close()
        self.generator = None

    def read(self, relative, expected):
        require(self.unchanged(), "Cached RGB master or authority changed")
        item = self.by_path.get(relative)
        require(item is not None and (expected is None or expected == item["file_sha256"]),
                "Missing PNG path/SHA does not match frozen events")
        index = item["index"]
        if self.generator is None or index < self.cursor:
            self.close()
            self.generator = _decoded(self.capture / MASTER_NAME)
            self.cursor = -1
        while self.cursor < index:
            pixels = next(self.generator)
            self.cursor += 1
            self.last_png = _png_bytes(pixels)
        require(hashlib.sha256(self.last_png).hexdigest() == item["file_sha256"],
                "Restored PNG no longer matches original bytes")
        return self.last_png


def read_png_bytes(path, expected_sha256=None):
    """Read existing PNG or exact missing PNG from a bounded cached decoder.

    The first missing-PNG read performs a full independent master verification;
    subsequent ascending frame reads reuse one decoder and check authority stat
    signatures. Only four episode decoders and one PNG per decoder are retained.
    """
    path = Path(path).absolute()
    if path.exists() or path.is_symlink():
        _signature(path)
        data = path.read_bytes()
        require(expected_sha256 is None or hashlib.sha256(data).hexdigest() == expected_sha256,
                "Existing PNG SHA differs")
        return data
    require(path.parent.name == "rgb" and path.parent.parent.name == "capture",
            "Missing PNG must be inside canonical derived capture/rgb")
    root, capture = _episode(path.parent.parent.parent)
    key = str(root)
    reader = _READERS.get(key)
    if reader is not None and not reader.unchanged():
        reader.close()
        del _READERS[key]
        reader = None
    if reader is None:
        reader = _Reader(root)
        _READERS[key] = reader
    _READERS.move_to_end(key)
    while len(_READERS) > 4:
        _, old = _READERS.popitem(last=False)
        old.close()
    return reader.read(str(path.relative_to(capture)), expected_sha256)


def verify_missing_png(path, expected_sha256):
    """Fail closed unless a missing PNG reproduces the caller's admitted SHA."""
    require(not Path(path).exists(), "verify_missing_png requires an absent PNG")
    read_png_bytes(path, expected_sha256)
    return True


def restore_episode(directory):
    """Restore original PNG bytes at their exact original paths, without overwrite."""
    root, capture = _episode(directory)
    with _locked(capture):
        verify_compacted_episode(root)
        frames = _json(capture / RECEIPT_NAME)["png_files"]
        restored = 0
        for item in frames:
            path = capture / item["path"]
            if path.exists() or path.is_symlink():
                _pixels(path, item)
                continue
            data = read_png_bytes(path, item["file_sha256"])
            path.parent.mkdir(exist_ok=True)
            temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".partial")
            try:
                with temporary.open("xb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
                restored += 1
            finally:
                temporary.unlink(missing_ok=True)
        return {"contract": CONTRACT, "passed": True, "restored_pngs": restored, "frames": len(frames)}


def close_readers():
    """Release cached containers at a caller's batch boundary."""
    for reader in _READERS.values():
        reader.close()
    _READERS.clear()
