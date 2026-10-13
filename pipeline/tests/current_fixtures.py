"""Tiny current-contract inputs; no old dense archives or model weights."""

from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ImageProcess import SharedImageProcessor
from egomotion_pipeline.contracts import PARAMETER_SCHEMA_VERSION
from egomotion_pipeline.io import atomic_json


def write_video(path, indices):
    path = Path(path)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 32, 24, "yuv420p"
        for index in indices:
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index % 255, np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def write_mcap_video(path, count):
    """Foxglove compressed H264, with intentionally different capture/log clocks."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, timestamp_pb2
    from mcap_protobuf.writer import Writer

    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    descriptor = descriptor_pb2.FileDescriptorProto(name="camera.proto", package="foxglove", syntax="proto3")
    descriptor.dependency.append(timestamp_pb2.DESCRIPTOR.name)
    message = descriptor.message_type.add(name="CompressedVideo")
    for name, number, kind, type_name in (
        ("timestamp", 1, 11, ".google.protobuf.Timestamp"),
        ("data", 2, 12, None), ("format", 3, 9, None),
    ):
        field = message.field.add(name=name, number=number, type=kind, label=1)
        if type_name:
            field.type_name = type_name
    pool.Add(descriptor)
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName("foxglove.CompressedVideo"))
    origin = 1_776_320_366_552_000_123
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 32, 24, "yuv420p"
    codec.time_base, codec.framerate = Fraction(1, 30), Fraction(30)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "keyint=30:bframes=0"}
    with Path(path).open("wb") as stream, Writer(stream) as writer:
        def emit(packet):
            stamp = origin + round(packet.pts * 1e9 / 30)
            value = cls()
            value.timestamp.seconds, value.timestamp.nanos = divmod(stamp, 1_000_000_000)
            value.data, value.format = bytes(packet), "h264"
            writer.write_message("/sensor/camera/head_left/video", value, log_time=stamp + 5_000_000_000)
        for index in range(count):
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index * 17 % 255, np.uint8), format="rgb24")
            frame.pts = index
            for packet in codec.encode(frame):
                emit(packet)
        for packet in codec.encode():
            emit(packet)
    return {
        "kind": "mcap", "path": str(Path(path).resolve()),
        "topic": "/sensor/camera/head_left/video", "clock_origin_ns": origin,
        "timestamp_origin": 0.0, "from_timestamp": 0.0, "to_timestamp": (count - 1) / 30,
        "bad_timestamps_s": [6 / 30],
        "preprocessing": {"mode": "preundistorted", "rotate_ccw_quarters": 0},
    }


def make_source(tmp_path, kind="egoexo4d_hm"):
    root = tmp_path / "archive"
    (root / "meta").mkdir(parents=True)
    atomic_json(root / "meta/info.json", {"schema_version": PARAMETER_SCHEMA_VERSION, "fps": 30})
    atomic_json(root / "meta/splits.json", {"train": [0], "val": [], "test": []})
    atomic_json(root / ".pipeline/dataset-state.json", {"state": "accepted", "independent_acceptance_pending": False})
    pq.write_table(pa.table({"episode_index": [0]}), root / "meta/episodes.parquet")
    count = 13
    ids = np.arange(count, dtype=np.int64) + (2 if kind == "egoexo4d_hm" else 0)
    if kind == "egosuite":
        path = tmp_path / "original.mcap"
        video = write_mcap_video(path, count)
    else:
        path = write_video(tmp_path / "original.mp4", range(count + 2))
        video = {"video_path": str(path), "preprocessing": {"mode": "raw", "rotate_ccw_quarters": 0},
                 "video_fingerprint": {"size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}}
    valid = np.ones(count, dtype=bool)
    valid[6] = False
    data = {
        "episode_index": np.zeros(count, np.int64), "frame_index": np.arange(count, dtype=np.int64),
        "index": np.arange(count, dtype=np.int64), "timestamp": (np.arange(count) / 30).astype(np.float32),
        "observation.source_timestamp": np.arange(count) / 30,
        "source.video_frame_index": ids, "observation.valid.dino": valid,
    }
    motion = root / "episodes/00000000/motion.parquet"
    motion.parent.mkdir(parents=True)
    pq.write_table(pa.table(data), motion)
    record = {
        "episode_index": 0, "source": kind, "source_key": kind + ":test", "length": count,
        "dataset_from_index": 0, "dataset_to_index": count, "data_path": str(motion.relative_to(root)),
        "vision_path": "episodes/00000000/vision.parquet",
        "dino": {"source_extraction": {"video": video, "video_fingerprint": {
            "size_bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}}},
        "metadata": {"frame_alignment": {"verified": True}},
    }
    atomic_json(root / ".pipeline/episodes/00000000.json", record)
    atomic_json(root / "episodes/00000000/receipt.json", {
        "schema_version": PARAMETER_SCHEMA_VERSION, "record": record,
        "output_identities": {"motion": {"bytes": motion.stat().st_size, "mtime_ns": motion.stat().st_mtime_ns}}})
    return root, record, motion


class Encoder:
    def __init__(self, profile="256x480"):
        self.processor = SharedImageProcessor(profile)
        self.metadata = {"image_process": self.processor.metadata, "shape": [5, 1024]}
        self.seen = []

    def prepare(self, images):
        return [self.processor.process_image(image) for image in images]

    def encode_prepared(self, images):
        self.seen.extend(images)
        return np.stack([np.full((5, 1024), float(image.mean()), np.float32) for image in images])


def make_encoder(tmp_path, monkeypatch, *, image_profile="256x480", preprocessor="hf", **model_overrides):
    from egomotion_pipeline.features import DinoEncoder
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.config = SimpleNamespace(model_type="dinov2_with_registers", hidden_size=1024,
                                          num_register_tokens=4, patch_size=14, num_hidden_layers=24)
            for key, value in model_overrides.items():
                setattr(self.config, key, value)
        def forward(self, pixel_values):
            hidden = torch.arange(7, device=pixel_values.device).view(1, 7, 1)
            return SimpleNamespace(last_hidden_state=hidden.expand(len(pixel_values), 7, 1024))
    processor = transformers.BitImageProcessor(size={"shortest_edge": 256}, crop_size={"height": 224, "width": 224})
    monkeypatch.setattr(transformers.AutoImageProcessor, "from_pretrained", lambda *a, **k: processor)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", lambda *a, **k: Model())
    (tmp_path / "config.json").write_text("{}")
    return DinoEncoder({"checkpoint": str(tmp_path), "revision": "local", "device": "cpu",
                        "image_profile": image_profile, "preprocessor": preprocessor})
