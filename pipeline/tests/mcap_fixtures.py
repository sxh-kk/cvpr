"""Small native EgoSuite schemas and inter-frame H.264 for MCAP integration tests."""

from fractions import Fraction
from pathlib import Path
import av
import numpy as np
from google.protobuf import descriptor_pb2 as dp, descriptor_pool, message_factory, timestamp_pb2
from mcap_protobuf.writer import Writer

BASE_NS = 1_776_320_366_552_000_123


def message_classes():
    pool = descriptor_pool.DescriptorPool()
    pool.AddSerializedFile(timestamp_pb2.DESCRIPTOR.serialized_pb)
    def file(name, package, dependencies=()):
        f = dp.FileDescriptorProto(name=name, package=package, syntax="proto3")
        f.dependency.extend(dependencies)
        return f
    def message(f, name, fields):
        m = f.message_type.add(name=name)
        for name, number, kind, *options in fields:
            v = m.field.add(name=name, number=number, type=kind, label=dp.FieldDescriptorProto.LABEL_OPTIONAL)
            if options and options[0]:
                v.type_name = options[0]
            if len(options) > 1:
                v.label = dp.FieldDescriptorProto.LABEL_REPEATED
    double, string, msg = dp.FieldDescriptorProto.TYPE_DOUBLE, dp.FieldDescriptorProto.TYPE_STRING, dp.FieldDescriptorProto.TYPE_MESSAGE
    header = file("common/header.proto", "common.header")
    message(header, "Header", [("count", 1, 4), ("ts", 2, 3), ("sender", 3, string)])
    pool.Add(header)
    pose_common = file("pose/pose_common.proto", "pose.common")
    message(pose_common, "Position", [(c, i+1, double) for i, c in enumerate("xyz")])
    message(pose_common, "Quaternion", [(c, i+1, double) for i, c in enumerate("wxyz")])
    message(pose_common, "Transform", [("pos", 1, msg, ".pose.common.Position"), ("quat", 2, msg, ".pose.common.Quaternion")])
    pool.Add(pose_common)
    pose = file("pose/pose.proto", "pose", [header.name, pose_common.name])
    for name in ["BodyFrame", "UpperBodyFrame", "HeadFrame", "LeftHandFrame", "RightHandFrame"]:
        message(pose, name, [("header", 1, msg, ".common.header.Header"), ("transforms", 2, msg, ".pose.common.Transform", True)])
    pool.Add(pose)
    fox = file("foxglove/camera.proto", "foxglove", [timestamp_pb2.DESCRIPTOR.name])
    message(fox, "Vector3", [(c, i+1, double) for i, c in enumerate("xyz")])
    message(fox, "Quaternion", [(c, i+1, double) for i, c in enumerate("xyzw")])
    message(fox, "FrameTransform", [("timestamp", 1, msg, ".google.protobuf.Timestamp"),
                                    ("parent_frame_id", 2, string), ("child_frame_id", 3, string),
                                    ("translation", 4, msg, ".foxglove.Vector3"), ("rotation", 5, msg, ".foxglove.Quaternion")])
    message(fox, "FrameTransforms", [("transforms", 1, msg, ".foxglove.FrameTransform", True)])
    message(fox, "CompressedVideo", [("timestamp", 1, msg, ".google.protobuf.Timestamp"),
                                     ("frame_id", 2, string), ("data", 3, 12), ("format", 4, string)])
    message(fox, "CameraCalibration", [("timestamp", 1, msg, ".google.protobuf.Timestamp"),
                                       ("frame_id", 2, string), ("width", 3, 7), ("height", 4, 7),
                                       ("distortion_model", 5, string)]+
            [(c, i+6, double, "", True) for i, c in enumerate(["D", "K", "R", "P"])])
    pool.Add(fox)
    annotation = file("annotation/bad_frame.proto", "annotation", [header.name])
    for name in ["BadFrameBody", "BadFrameHand", "BadFrameImage"]:
        message(annotation, name, [("header", 1, msg, ".common.header.Header"), ("is_bad", 2, 8), ("problem_type", 3, string)])
    pool.Add(annotation)
    session = file("session/metadata.proto", "session", [header.name])
    message(session, "TaskInfo", [(c, i+1, string) for i, c in enumerate(["episode_uuid", "task_name", "task_description", "environment_id", "scene_id"])])
    message(session, "Operator", [("operator_id", 2, string)])
    message(session, "SessionMetadata", [("header", 1, msg, ".common.header.Header"), ("batch_version", 2, string),
                                         ("task_info", 6, msg, ".session.TaskInfo"), ("operator", 7, msg, ".session.Operator")])
    pool.Add(session)
    return {name: message_factory.GetMessageClass(pool.FindMessageTypeByName(name))
            for name in ["pose.BodyFrame", "pose.UpperBodyFrame", "pose.HeadFrame", "pose.LeftHandFrame", "pose.RightHandFrame",
                         "foxglove.FrameTransforms", "foxglove.CompressedVideo", "foxglove.CameraCalibration",
                         "annotation.BadFrameBody", "annotation.BadFrameHand", "annotation.BadFrameImage", "session.SessionMetadata"]}


def h264_packets(indices):
    codec = av.CodecContext.create("libx264", "w")
    codec.width, codec.height, codec.pix_fmt = 32, 24, "yuv420p"
    codec.time_base, codec.framerate = Fraction(1, 30), Fraction(30)
    codec.options = {"preset": "ultrafast", "tune": "zerolatency", "x264-params": "keyint=30:bframes=0"}
    for index in indices:
        frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index*30, np.uint8), format="rgb24")
        frame.pts = index
        for packet in codec.encode(frame):
            yield packet.pts, bytes(packet)
    for packet in codec.encode():
        yield packet.pts, bytes(packet)


def make_mcap(path, count=5, body_joints=22, upper_only=False, camera="head_left",
              missing_camera=(), camera_offsets=None, missing_hand=(), video_indices=None,
              bad=None, omit_camera=False, raw_video=False, duplicate_body=False, reverse_body=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = message_classes()
    def stamp(index):
        return BASE_NS+round(index*1e9/30)
    def timestamp(value, ns):
        value.seconds, value.nanos = divmod(ns, 1_000_000_000)
    with path.open("wb") as stream, Writer(stream) as writer:
        def write(topic, value, ns):
            # Logger time intentionally differs from the sensor capture clock.
            writer.write_message(topic, value, log_time=ns+5_000_000_000)
        meta = classes["session.SessionMetadata"]()
        meta.task_info.episode_uuid = "internal-session-uuid"
        meta.task_info.task_name = meta.task_info.task_description = "THIS_SOURCE_TEXT_MUST_NOT_BE_EXPORTED"
        meta.task_info.scene_id, meta.operator.operator_id = "scene-1", "operator-1"
        write("/session/metadata", meta, BASE_NS)
        indices = list(reversed(range(count))) if reverse_body else list(range(count))
        if duplicate_body:
            indices.append(0)
        body_topic = "/pose/upper_body" if upper_only else "/pose/body"
        for key, topic, joints in [("UpperBodyFrame" if upper_only else "BodyFrame", body_topic, body_joints),
                                   ("HeadFrame", "/pose/head", 1), ("LeftHandFrame", "/pose/left_hand", 21),
                                   ("RightHandFrame", "/pose/right_hand", 21)]:
            for index in indices if topic == body_topic else range(count):
                if "hand" in topic and index in missing_hand:
                    continue
                value = classes["pose."+key]()
                value.header.ts = stamp(index)
                for joint in range(joints):
                    t = value.transforms.add()
                    t.pos.x, t.pos.y, t.pos.z = index/30, joint*.01, 1.
                    t.quat.w, t.quat.z = .5, np.sqrt(3)/2
                write(topic, value, stamp(index))
        prefix = f"/sensor/camera/{camera}"
        if not omit_camera:
            for index in range(count):
                if index in missing_camera:
                    continue
                value = classes["foxglove.FrameTransforms"]()
                t = value.transforms.add()
                ns = stamp(index)+(camera_offsets or {}).get(index, 0)
                timestamp(t.timestamp, ns)
                t.parent_frame_id, t.child_frame_id = "world", camera+"_camera"
                t.translation.x, t.translation.z, t.rotation.w = index/30, 1.5, 1.
                write(prefix+"/extrinsic", value, ns)
        calib = classes["foxglove.CameraCalibration"]()
        calib.width, calib.height, calib.frame_id = 32, 24, camera+"_camera"
        calib.K.extend([100., 0., 16., 0., 100., 12., 0., 0., 1.])
        write(prefix+"/intrinsic", calib, BASE_NS)
        video_topic = f"/sensor/camera_raw/{camera}/video" if raw_video else prefix+"/video"
        for index, packet in h264_packets(range(count) if video_indices is None else video_indices):
            value = classes["foxglove.CompressedVideo"]()
            timestamp(value.timestamp, stamp(index))
            value.format, value.frame_id, value.data = "h264", camera+"_camera", packet
            write(video_topic, value, stamp(index))
        for part, bad_indices in (bad or {}).items():
            topic = f"/annotation/bad_frame/camera/{camera}" if part == "image" else f"/annotation/bad_frame/pose/{part}"
            name = "BadFrameImage" if part == "image" else "BadFrameHand" if part == "hand" else "BadFrameBody"
            for index in range(count):
                value = classes["annotation."+name]()
                value.header.ts, value.is_bad = stamp(index), index in bad_indices
                write(topic, value, stamp(index))
    return path
