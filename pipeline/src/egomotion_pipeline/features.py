import numpy as np
import re
from pathlib import Path
from .body_model import file_sha256
from .storage import read_arrays, write_arrays, atomic_json, quality_report, digest
from .video import nearest_video_frames, nearest_mcap_video_frames, image_preprocessor
from .schema import DINO_CHECKPOINT, DINO_SHAPE, DINO_FLAT_DIM, SCHEMA_VERSION, dino_format


DINO_PREPROCESSING = {"do_resize": True, "size": {"shortest_edge": 336},
                      "resample": 3, "do_center_crop": False,
                      "do_rescale": True, "rescale_factor": 1/255,
                      "do_normalize": True, "image_mean": [0.485, 0.456, 0.406],
                      "image_std": [0.229, 0.224, 0.225]}


class DinoEncoder:
    def __init__(self, config):
        import torch
        import transformers
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = config.get("checkpoint", DINO_CHECKPOINT)
        revision = config.get("revision")
        if not revision:
            raise ValueError("Pin DINO revision to a commit hash, or provide a local checkpoint with revision='local'")
        if revision != "local" and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("DINO revision must be a 40-character commit hash, or 'local'")
        kwargs = {"revision": revision} if revision != "local" else {"local_files_only": True}
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False, **kwargs)
        # Match UniEgoMotion's Resize(336, bicubic) + ImageNet normalization.
        # Override HF's default 256 resize / 224 center crop explicitly.
        for key, value in DINO_PREPROCESSING.items():
            setattr(self.processor, key, value)
        self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, **kwargs).to(self.device).eval()
        self.model.requires_grad_(False)
        model_config = self.model.config
        if (model_config.model_type != "dinov2_with_registers"
                or model_config.hidden_size != DINO_SHAPE[1]
                or getattr(model_config, "num_register_tokens", 0) != 4
                or model_config.patch_size != 14 or model_config.num_hidden_layers != 24):
            raise ValueError("The archive requires dinov2-with-registers-large (ViT-L/14, CLS + 4 registers, 5x1024)")
        self.metadata = {**dino_format(), "checkpoint": checkpoint, "revision": revision,
                         "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                         "processor": self.processor.to_dict(),
                         "preprocessing": {**DINO_PREPROCESSING, "input": "RGB uint8 after source-specific geometry",
                                           "resize": "shortest_edge_keep_aspect", "interpolation": "bicubic"},
                         "inference_dtype": str(next(self.model.parameters()).dtype),
                         "torch_version": torch.__version__, "transformers_version": transformers.__version__}
        if revision == "local":
            paths = sorted(p for p in Path(checkpoint).rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"})
            if not paths:
                raise ValueError("Local DINO checkpoint has no fingerprintable model/config files")
            self.metadata["local_file_sha256"] = {str(p.relative_to(checkpoint)): file_sha256(p) for p in paths}

    def __call__(self, images):
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            result = self.model(**inputs).last_hidden_state[:, :DINO_SHAPE[0]]
        return result.float().cpu().numpy()


def extract_features(archive, config, encoder=None):
    if archive.schema_version != SCHEMA_VERSION:
        raise ValueError("Legacy CLS-only archive: run upgrade-dino before extracting 5x1024 features")
    batch_size = int(config.get("batch_size", 32))
    tolerance = float(config.get("max_video_error_s", 1/60+.001))
    if batch_size <= 0:
        raise ValueError("DINO batch_size must be positive")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("DINO max_video_error_s must be finite and nonnegative")
    encoder = encoder or DinoEncoder(config)
    for record in archive.records():
        video = record["video"]
        if not video:
            raise ValueError(f"{record['source_key']}: no ego video in provenance")
        video_path = Path(video["path"])
        stat = video_path.stat()
        video_fingerprint = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        signature = digest({"config": config, "video": video, "video_fingerprint": video_fingerprint,
                            "encoder": encoder.metadata})
        if record["dino"]["state"] == "complete":
            if record["dino"].get("signature") != signature:
                raise ValueError("Existing DINO features use a different model/preprocessing; create a new archive")
            continue
        arrays = read_arrays(archive.data_path(record["episode_index"]), archive.specs)
        count = record["length"]
        arrays["observation.features.dino"][:] = 0
        arrays["observation.valid.dino"][:] = False
        arrays["observation.video_timestamp"][:] = 0
        arrays["observation.video_time_error"][:] = 0
        target = (arrays["observation.source_timestamp"] - float(video.get("timestamp_origin", 0))
                  + float(video["from_timestamp"]))
        process = image_preprocessor(video.get("preprocessing", {"mode": "raw"}))
        images, indices, pts_values = [], [], []
        def flush():
            if not images:
                return
            feature = np.asarray(encoder(images), dtype=np.float32)
            if feature.shape != (len(images), *DINO_SHAPE) or not np.isfinite(feature).all():
                raise ValueError("DINO encoder must return finite (batch, 5, 1024) CLS + register tokens")
            arrays["observation.features.dino"][indices] = feature.reshape(len(images), DINO_FLAT_DIM)
            arrays["observation.valid.dino"][indices] = True
            arrays["observation.video_timestamp"][indices] = pts_values
            arrays["observation.video_time_error"][indices] = np.asarray(pts_values)-target[indices]
            images.clear(); indices.clear(); pts_values.clear()
        frames = (nearest_mcap_video_frames(video, target, tolerance) if video.get("kind") == "mcap"
                  else nearest_video_frames(video["path"], target, tolerance))
        for index, (image, pts) in enumerate(frames):
            if image is None or pts < float(video["from_timestamp"])-1e-6 or pts > float(video["to_timestamp"])+1e-6:
                continue
            images.append(process(image)); indices.append(index); pts_values.append(pts)
            if len(images) >= batch_size:
                flush()
        flush()
        if not arrays["observation.valid.dino"].any():
            raise ValueError(f"{record['source_key']}: no aligned video frames")
        write_arrays(archive.data_path(record["episode_index"]), arrays, archive.specs)
        record["dino"] = {"state": "complete", "signature": signature,
                          "encoder": encoder.metadata, "video": video, "target_fps": 30,
                          "max_video_error_s": tolerance, "valid_count": int(arrays["observation.valid.dino"].sum()),
                          "video_fingerprint": video_fingerprint}
        record["quality"] = quality_report(arrays)
        atomic_json(archive.record_path(record["episode_index"]), record)
        print(f"DINO {record['source_key']}: {record['dino']['valid_count']}/{count}", flush=True)
    archive.finalize()
