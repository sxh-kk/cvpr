import json
from fractions import Fraction
from pathlib import Path
import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from egomotion_pipeline.schema import PARENTS


@pytest.fixture
def model_path(tmp_path):
    # A non-zero root pivot and a joint-specific shape basis make FK tests meaningful.
    rest = np.zeros((22, 3))
    rest[0] = [2, 3, 4]
    for joint in range(1, 22):
        rest[joint] = rest[PARENTS[joint]] + [joint*.01, .1, .05]
    shapedirs = np.zeros((22, 3, 2))
    shapedirs[:, 0, 0] = np.arange(22)*.02
    shapedirs[:, 1, 1] = .1
    path = tmp_path/"model.npz"
    np.savez(path, v_template=rest, shapedirs=shapedirs, J_regressor=np.eye(22),
             kintree_table=np.array([PARENTS, np.arange(22)]))
    return path


def write_video(path, pts_indices, fps=30):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width, stream.height = 32, 24
        stream.pix_fmt = "yuv420p"
        for index in pts_indices:
            frame = av.VideoFrame.from_ndarray(np.full((24, 32, 3), index % 255, np.uint8), format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def make_egosuite(root, episodes=2, count=5, fps=30, body_joints=22):
    root = Path(root)
    (root/"meta/episodes/chunk-000").mkdir(parents=True)
    (root/"data/chunk-000").mkdir(parents=True)
    feature_shapes = {"body_world": [body_joints, 3], "body_world_rotation": [body_joints, 4],
                      "hand_left_world": [21, 3], "hand_right_world": [21, 3],
                      "head_world": [1, 3], "head_world_rotation": [1, 4],
                      "head_left_camera_position": [3], "head_left_camera_rotation": [4]}
    features = {"observation.state."+k: {"dtype": "float32", "shape": shape, "names": None} for k, shape in feature_shapes.items()}
    video_key = "observation.images.head_left"
    features[video_key] = {"dtype": "video", "shape": [24, 32, 3]}
    info = {"codebase_version": "v3.0", "fps": fps, "features": features,
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"}
    (root/"meta/info.json").write_text(json.dumps(info))
    rows, metadata = [], []
    for episode in range(episodes):
        for frame in range(count):
            row = {"index": episode*count+frame, "episode_index": episode, "frame_index": frame,
                   "timestamp": float(frame/fps)}
            for key, shape in feature_shapes.items():
                value = np.zeros(shape)
                if "rotation" in key:
                    value[..., 0] = 1
                else:
                    value[..., 0] = episode+frame/fps
                row["observation.state."+key] = value.tolist()
            row["task"] = "THIS_SOURCE_TEXT_MUST_NOT_BE_EXPORTED"
            rows.append(row)
        prefix = f"videos/{video_key}"
        metadata.append({"episode_index": episode, "episode_uuid": f"uuid-{episode}", "length": count,
                         "data/chunk_index": 0, "data/file_index": 0,
                         "dataset_from_index": episode*count, "dataset_to_index": (episode+1)*count,
                         "tasks": ["THIS_SOURCE_TEXT_MUST_NOT_BE_EXPORTED"], "operator_id": f"operator-{episode}",
                         prefix+"/chunk_index": 0, prefix+"/file_index": 0,
                         prefix+"/from_timestamp": episode*count/fps,
                         prefix+"/to_timestamp": (episode*count+count-1)/fps,
                         "camera_intrinsics/head_left": [100, 100, 16, 12, 0, 0, 0, 0]})
    pq.write_table(pa.Table.from_pylist(rows), root/"data/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist(metadata), root/"meta/episodes/chunk-000/file-000.parquet")
    write_video(root/"videos"/video_key/"chunk-000/file-000.mp4", range(episodes*count), fps)
    return root
