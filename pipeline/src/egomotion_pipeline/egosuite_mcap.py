"""Read native EgoSuite Protobuf topics using the schemas embedded in MCAP.

Pose timestamps remain integer nanoseconds until subtraction of the body clock
origin. Frame joins use time, never message order or a guessed nominal FPS.
"""

from pathlib import Path
import numpy as np
from .adapters import selected
from .schema import Episode

POSE_TOPICS = {"body": "/pose/body", "head": "/pose/head",
               "hand_left": "/pose/left_hand", "hand_right": "/pose/right_hand"}
UPPER_JOINTS = [0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
LOWER_JOINTS = [1, 2, 4, 5, 7, 8, 10, 11]
BAD_TOPICS = {"hand": "/annotation/bad_frame/pose/hand",
              "upper_body": "/annotation/bad_frame/pose/upper_body",
              "lower_body": "/annotation/bad_frame/pose/lower_body"}


class UpperBodyOnly(ValueError):
    """A documented 14-joint source cannot supply full-body supervision."""


def discover_egosuite_mcap(config):
    root = Path(config["root"])
    paths = [root] if root.is_file() and root.suffix == ".mcap" else sorted(root.rglob("*.mcap"))
    if not paths:
        raise FileNotFoundError(f"No EgoSuite .mcap files under {root}")
    emitted, seen = 0, set()
    for path in paths:
        # Match downloaded filenames, even if session metadata has a different UUID.
        uid = path.stem
        if not selected(uid, config):
            continue
        if uid in seen:
            raise ValueError(f"Duplicate MCAP filename UUID {uid}; select one recording root")
        seen.add(uid)
        yield path, uid
        emitted += 1
        if config.get("limit") and emitted >= int(config["limit"]):
            return


def make_proto_reader(stream):
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory
    return make_reader(stream, decoder_factories=[DecoderFactory()])


def decoded_messages(path, topics):
    with Path(path).open("rb") as stream:
        reader = make_proto_reader(stream)
        # File order avoids a heap of large video payloads. Each consumer handles
        # the payload timestamp; it must not assume log_time is the capture clock.
        for schema, channel, message, decoded in reader.iter_decoded_messages(
                topics=topics, log_time_order=False):
            if channel.message_encoding != "protobuf" or schema.encoding != "protobuf":
                raise ValueError(f"{channel.topic}: expected embedded Protobuf schema")
            yield channel.topic, message, decoded


def timestamp_ns(decoded, message):
    if "header" in decoded.DESCRIPTOR.fields_by_name and decoded.HasField("header"):
        return int(decoded.header.ts), "header.ts"
    if "timestamp" in decoded.DESCRIPTOR.fields_by_name and decoded.HasField("timestamp"):
        stamp = decoded.timestamp
        return int(stamp.seconds)*1_000_000_000+int(stamp.nanos), "timestamp"
    return int(message.log_time), "log_time_fallback"


def pose_arrays(transforms, count, topic, camera=False):
    if len(transforms) != count:
        if topic == "/pose/body" and len(transforms) == 14:
            raise UpperBodyOnly(f"{topic}: upper-body 14 joints")
        raise ValueError(f"{topic}: expected {count} transforms, got {len(transforms)}; body must have 22-joint layout")
    position, rotation = [], []
    for transform in transforms:
        pkey, qkey = ("translation", "rotation") if camera else ("pos", "quat")
        p, q = getattr(transform, pkey), getattr(transform, qkey)
        position.append([p.x, p.y, p.z] if transform.HasField(pkey) else [np.nan]*3)
        rotation.append([q.w, q.x, q.y, q.z] if transform.HasField(qkey) else [0.]*4)
    return np.asarray(position), np.asarray(rotation)


def ordered(samples, topic):
    samples.sort(key=lambda row: row[0])
    stamps = np.asarray([row[0] for row in samples], dtype=np.int64)
    if len(stamps) > 1 and np.any(np.diff(stamps) <= 0):
        raise ValueError(f"{topic}: duplicate capture timestamps")
    return stamps


def nearest_indices(stamps, targets, tolerance_ns):
    right = np.searchsorted(stamps, targets).clip(0, len(stamps)-1)
    left = np.maximum(right-1, 0)
    indices = np.where(np.abs(stamps[left]-targets) <= np.abs(stamps[right]-targets), left, right)
    errors = stamps[indices]-targets
    return indices, np.abs(errors) <= tolerance_ns, errors


def inspect_mcap(path, uid):
    with Path(path).open("rb") as stream:
        reader = make_proto_reader(stream)
        summary = reader.get_summary()
        if summary and summary.channels and summary.statistics:
            stats = summary.statistics
            topics = {c.topic: {"schema": summary.schemas[c.schema_id].name,
                               "messages": stats.channel_message_counts.get(c.id, 0)}
                      for c in summary.channels.values()}
            duration = (int(stats.message_end_time)-int(stats.message_start_time))/1e9
        else:
            # Also support valid, unindexed MCAP files; inspection then scans data.
            topics, start, end = {}, None, None
            for schema, channel, message in reader.iter_messages(log_time_order=False):
                info = topics.setdefault(channel.topic, {"schema": schema.name if schema else None, "messages": 0})
                info["messages"] += 1
                start = message.log_time if start is None else min(start, message.log_time)
                end = message.log_time if end is None else max(end, message.log_time)
            duration = (end-start)/1e9 if start is not None else 0.
    body_joints = 0
    for _, _, decoded in decoded_messages(path, ["/pose/body", "/pose/upper_body"]):
        body_joints = len(decoded.transforms)
        if body_joints == 22 or "/pose/body" not in topics:
            break
    return {"uid": uid, "path": str(Path(path).resolve()), "duration_s": duration,
            "body_joints": body_joints, "topics": topics}


def read_egosuite_mcap(path, uid, config):
    camera = config.get("camera", "head_left")
    if camera not in {"head_left", "head_right"}:
        raise ValueError("EgoSuite ego camera must be head_left or head_right")
    tolerance = float(config.get("max_pose_error_s", .002))
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("max_pose_error_s must be finite and non-negative")
    tolerance_ns = round(tolerance*1e9)
    prefix = f"/sensor/camera/{camera}"
    video_topic = config.get("video_topic", prefix+"/video")
    allowed_video = {prefix+"/video", f"/sensor/camera_raw/{camera}/video"}
    if video_topic not in allowed_video:
        raise ValueError("video_topic must select the configured camera's regular or raw stream")
    image_bad_topic = f"/annotation/bad_frame/camera/{camera}"
    topics = list(POSE_TOPICS.values())+list(BAD_TOPICS.values())+[
        "/pose/upper_body", "/session/metadata", prefix+"/extrinsic", prefix+"/intrinsic",
        video_topic, image_bad_topic]
    samples = {key: [] for key in [*POSE_TOPICS, "camera"]}
    bad_samples = {key: [] for key in BAD_TOPICS}
    image_bad, clock_fields, counts = [], {}, {}
    metadata, calibration = {}, None
    video_start, video_end, video_format = None, None, None
    upper_body_only = False
    reverse_pose = {topic: key for key, topic in POSE_TOPICS.items()}
    reverse_bad = {topic: key for key, topic in BAD_TOPICS.items()}
    for topic, message, decoded in decoded_messages(path, topics):
        counts[topic] = counts.get(topic, 0)+1
        if topic == "/session/metadata":
            # Deliberate allowlist: task names/descriptions and semantic topics are
            # never accessed or copied into training data or provenance.
            for key in ["episode_uuid", "scene_id", "environment_id"]:
                value = getattr(decoded.task_info, key)
                if value:
                    metadata["source_episode_uuid" if key == "episode_uuid" else key] = value
            if decoded.operator.operator_id:
                metadata["operator_id"] = decoded.operator.operator_id
            if decoded.batch_version:
                metadata["batch_version"] = decoded.batch_version
            continue
        if topic == "/pose/upper_body":
            upper_body_only = len(decoded.transforms) == 14
            continue
        if topic == prefix+"/intrinsic":
            if calibration is None:
                calibration = {key: getattr(decoded, key) for key in ["width", "height", "frame_id", "distortion_model"]}
                calibration.update({key: list(getattr(decoded, key)) for key in ["K", "D", "R", "P"]})
            continue
        if topic == prefix+"/extrinsic":
            transforms = [t for t in decoded.transforms if t.child_frame_id == camera+"_camera"]
            if len(transforms) != 1 or transforms[0].parent_frame_id != "world":
                raise ValueError(f"{uid}: camera extrinsic must contain world -> {camera}_camera")
            stamp, field = timestamp_ns(transforms[0], message)
            p, q = pose_arrays(transforms, 1, topic, camera=True)
            samples["camera"].append((stamp, p, q))
        else:
            stamp, field = timestamp_ns(decoded, message)
            if topic == video_topic:
                if video_format is not None and decoded.format != video_format:
                    raise ValueError(f"{uid}: video codec changes within an episode")
                video_format = decoded.format
                video_start = stamp if video_start is None else min(video_start, stamp)
                video_end = stamp if video_end is None else max(video_end, stamp)
            elif topic == image_bad_topic:
                image_bad.append((stamp, bool(decoded.is_bad)))
            elif topic in reverse_bad:
                bad_samples[reverse_bad[topic]].append((stamp, bool(decoded.is_bad)))
            else:
                key = reverse_pose[topic]
                p, q = pose_arrays(decoded.transforms, 22 if key == "body" else 1 if key == "head" else 21, topic)
                samples[key].append((stamp, p, q))
        fields = clock_fields.setdefault(topic, {})
        fields[field] = fields.get(field, 0)+1
    if not samples["body"]:
        if upper_body_only:
            raise UpperBodyOnly(f"{uid}: upper-body 14 joints without /pose/body")
        raise ValueError(f"{uid}: no full 22-joint /pose/body messages")
    if config.get("require_camera", True) and not samples["camera"]:
        raise ValueError(f"{uid}: missing {prefix}/extrinsic; set require_camera=false for motion-only conversion")
    body_stamps = ordered(samples["body"], POSE_TOPICS["body"])
    origin = int(body_stamps[0])
    timestamps = (body_stamps-origin).astype(np.float64)/1e9
    positions, rotations, position_valid, rotation_valid, joins = {}, {}, {}, {}, {}
    for key, rows in samples.items():
        if not rows:
            joins[key] = {"messages": 0, "matched": 0}
            continue
        stamps = body_stamps if key == "body" else ordered(rows, prefix+"/extrinsic" if key == "camera" else POSE_TOPICS[key])
        indices, matched, errors = nearest_indices(stamps, body_stamps, tolerance_ns)
        positions[key] = np.stack([row[1] for row in rows])[indices]
        if key in {"body", "head", "camera"}:
            rotations[key] = np.stack([row[2] for row in rows])[indices]
            rotation_valid[key] = np.broadcast_to(matched[:, None], positions[key].shape[:-1]).copy()
        position_valid[key] = np.broadcast_to(matched[:, None], positions[key].shape[:-1]).copy()
        joins[key] = {"messages": len(rows), "matched": int(matched.sum()),
                      "max_matched_error_s": float(np.max(np.abs(errors[matched]))/1e9) if matched.any() else None}
    quality = {}
    for key, rows in bad_samples.items():
        if not rows:
            continue
        stamps = ordered(rows, BAD_TOPICS[key])
        indices, matched, _ = nearest_indices(stamps, body_stamps, tolerance_ns)
        bad = np.asarray([row[1] for row in rows])[indices] & matched
        quality[key] = {"messages": len(rows), "matched": int(matched.sum()), "bad_body_frames": int(bad.sum())}
        for modality in ["hand_left", "hand_right"] if key == "hand" else ["body"]:
            if modality not in positions:
                continue
            joints = list(range(21)) if key == "hand" else UPPER_JOINTS if key == "upper_body" else LOWER_JOINTS
            position_valid[modality][np.ix_(bad, joints)] = False
            if modality in rotation_valid:
                rotation_valid[modality][np.ix_(bad, joints)] = False
    video = None
    if video_start is not None:
        raw_video = "/camera_raw/" in video_topic
        start, end = (video_start-origin)/1e9, (video_end-origin)/1e9
        video = {"kind": "mcap", "path": str(Path(path).resolve()), "topic": video_topic,
                 "format": video_format, "clock_origin_ns": origin, "from_timestamp": start,
                 "to_timestamp": end, "timestamp_origin": start, "camera": camera,
                 "bad_timestamps_s": sorted((stamp-origin)/1e9 for stamp, bad in image_bad if bad),
                 "preprocessing": config.get("video_preprocessing", {"mode": "raw" if raw_video else "preundistorted"})}
    metadata.update({"source_format": "egosuite_mcap", "source_data": str(Path(path).resolve()),
                     "source_root": str(Path(config["root"]).resolve()), "source_file_uid": uid,
                     "source_revision": config.get("revision", "unspecified"),
                     "source_fps": float(1e9/np.median(np.diff(body_stamps))) if len(body_stamps) > 1 else None,
                     "body_position_source": "egosuite_mocap", "hand_position_source": "egosuite_mocap",
                     "rotation_semantics": "source_global_wxyz", "head_semantics": "egosuite_head_world",
                     "camera_pose_basis": "egosuite_opencv_camera_to_world",
                     "hand_joint_convention": "wrist_then_thumb_index_middle_ring_pinky_4_each",
                     "clock": {"origin_ns": origin, "units": "seconds_relative_to_first_body_capture",
                               "topic_timestamp_fields": clock_fields},
                     "pose_alignment": {"method": "nearest_capture_timestamp", "max_pose_error_s": tolerance, "modalities": joins},
                     "topic_message_counts": counts, "bad_frame_masks": quality,
                     "quality": "finite_and_zero_quaternion_checks_plus_available_vendor_frame_flags; not_joint_confidence"})
    if calibration is not None:
        metadata[f"camera_intrinsics/{camera}"] = calibration
    return Episode(uid, config.get("name", "egosuite"), timestamps, positions, rotations,
                   position_valid, rotation_valid, metadata, video).check()
