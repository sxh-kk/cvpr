import numpy as np
import re
from pathlib import Path
from .io import file_sha256
from .contracts import DINO_CHECKPOINT, DINO_SHAPE, dino_format
from .dino_optimization import optimization_options, ParallelPILPreprocessor


DINO_PREPROCESSING = {"do_resize": False, "resample": 3, "do_center_crop": False,
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
        self.options = optimization_options(config)
        profile = config.get("image_profile", "256x480")
        from ImageProcess import SharedImageProcessor, load_profile
        self.shared_processor = SharedImageProcessor(load_profile(profile))
        cuda = str(self.device).startswith("cuda")
        if not cuda and (self.options["inference_precision"] != "fp32"
                         or self.options["pin_memory"] or self.options["compile_model"]):
            raise ValueError("DINO mixed precision, pinned memory and compilation require CUDA")
        if cuda:
            torch.backends.cuda.matmul.allow_tf32 = self.options["inference_precision"] == "tf32"
            if self.options["inference_precision"] == "bf16" and not torch.cuda.is_bf16_supported():
                raise ValueError("Selected CUDA device does not support BF16")
        checkpoint = config.get("checkpoint", DINO_CHECKPOINT)
        revision = config.get("revision")
        if not revision:
            raise ValueError("Pin DINO revision to a commit hash, or provide a local checkpoint with revision='local'")
        if revision != "local" and not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("DINO revision must be a 40-character commit hash, or 'local'")
        kwargs = {"revision": revision} if revision != "local" else {"local_files_only": True}
        self.processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=False, **kwargs)
        preprocessing = dict(DINO_PREPROCESSING)
        preprocessing.update(size={"height": self.shared_processor.profile.output_hw[0],
                                   "width": self.shared_processor.profile.output_hw[1]})
        # Geometry is shared with Psi0; branch-specific ImageNet normalization follows it.
        for key, value in preprocessing.items():
            setattr(self.processor, key, value)
        model_kwargs = dict(kwargs)
        if self.options["attention_implementation"] != "auto":
            model_kwargs["attn_implementation"] = self.options["attention_implementation"]
        self.model = AutoModel.from_pretrained(checkpoint, torch_dtype=torch.float32, **model_kwargs).to(self.device).eval()
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
                         "preprocessing": {**preprocessing, "input": "RGB uint8 after source-specific geometry",
                                           "resize": "shared_image_profile",
                                           "interpolation": "profile_defined"},
                         "inference_dtype": str(next(self.model.parameters()).dtype),
                         "torch_version": torch.__version__, "transformers_version": transformers.__version__}
        self.metadata["image_process"] = self.shared_processor.metadata
        if any(key in config for key in self.options):
            self.metadata["optimization"] = {**self.options, "implementation": "pil-prefetch-v1",
                "attention_resolved": getattr(model_config, "_attn_implementation", "unknown"),
                "compute_precision": self.options["inference_precision"],
                "output_storage_dtype": "float32", "pillow_version": __import__("PIL").__version__}
        if revision == "local":
            paths = sorted(p for p in Path(checkpoint).rglob("*") if p.is_file() and p.suffix in {".safetensors", ".bin", ".json"})
            if not paths:
                raise ValueError("Local DINO checkpoint has no fingerprintable model/config files")
            self.metadata["local_file_sha256"] = {str(p.relative_to(checkpoint)): file_sha256(p) for p in paths}
        self.parallel_processor = (ParallelPILPreprocessor(self.options["preprocess_workers"], self.shared_processor)
            if self.options["preprocessor"] == "pil_parallel" else None)
        if self.options["compile_model"]:
            self.model = torch.compile(self.model, mode="reduce-overhead")

    def prepare(self, images, *, processed=False, image_processor=None):
        processor = getattr(self, "parallel_processor", None)
        shared = image_processor or getattr(self, "shared_processor", None)
        if processor is not None:
            inputs = {"pixel_values": self.torch.from_numpy(processor(images, processed=processed, image_processor=shared))}
        else:
            if shared is not None:
                if processed:
                    if any(np.asarray(im).shape != (*shared.profile.output_hw, 3) or np.asarray(im).dtype != np.uint8 for im in images):
                        raise ValueError("Shared RGB must be uint8 HWC and match the DINO image profile")
                else:
                    images = [shared.process_image(im) for im in images]
            elif processed:
                raise ValueError("Already processed RGB requires an explicit shared image profile")
            inputs = self.processor(images=images, return_tensors="pt")
        if getattr(self, "options", {}).get("pin_memory", False):
            inputs = {key: value.pin_memory() for key, value in inputs.items()}
        return inputs

    def encode_prepared(self, inputs):
        options = getattr(self, "options", {})
        inputs = {k: v.to(self.device, non_blocking=options.get("pin_memory", False)) for k, v in inputs.items()}
        from contextlib import nullcontext
        cast = (self.torch.autocast(device_type="cuda", dtype=self.torch.bfloat16)
                if options.get("inference_precision") == "bf16" else nullcontext())
        with self.torch.inference_mode(), cast:
            result = self.model(**inputs).last_hidden_state[:, :DINO_SHAPE[0]]
        return result.float().cpu().numpy()

    def __call__(self, images):
        return self.encode_prepared(self.prepare(images))

    def encode_window(self, images, *, stage="posttrain", training=True, seed=None, augmentation=None):
        """Encode one history with shared augmentation; return the same RGB for the other branch.

        DINO remains frozen. Convert img_embs to a tensor and use MotionExpert.forward_features
        to train MotionExpert. Future frames are never passed to this encoder.
        """
        from ImageProcess import SharedImageProcessor
        # One frozen DINO supports all three stages without reloading its weights.
        target = "256x320" if stage == "posttrain" else "256x480" if stage in {"pretrain", "robust"} else None
        expected_hw = (256, 320) if stage == "posttrain" else (256, 480)
        shared = (SharedImageProcessor(target) if target is not None and
                  self.shared_processor.profile.output_hw != expected_hw else self.shared_processor)
        window = shared.process_window(images, stage=stage, training=training,
                                       seed=seed, augmentation=augmentation)
        features = self.encode_prepared(self.prepare(window.images, processed=True, image_processor=shared))
        if features.shape != (len(window.images), *DINO_SHAPE) or not np.isfinite(features).all():
            raise ValueError("DINO window must contain finite CLS + register features")
        return {"img_embs": features, "rgb": window.images, "augmentation": window.augmentation,
                "image_process": shared.metadata, "stage": stage}

    def close(self):
        processor = getattr(self, "parallel_processor", None)
        if processor is not None:
            processor.close()


def extraction_options(config):
    batch_size = config.get("batch_size", 32)
    tolerance = float(config.get("max_video_error_s", 1/60+.001))
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("DINO batch_size must be a positive integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("DINO max_video_error_s must be finite and positive")
    return batch_size, tolerance
