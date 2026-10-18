# Shared image processing

This package owns common image geometry and window-synchronized augmentation.
In the general `shared_rgb` mode, MotionExpert/DINO and Ψ₀ receive the same
processed current RGB before each encoder performs its own normalization.
The current UnifoLM recipe explicitly uses a separate VLM color policy described
below. This package does not change motion coordinates, perform
camera undistortion, normalize image channels, load model weights, or start jobs.
Input camera geometry must already follow the source dataset's established policy.

| Profile | First resize, H×W | Final RGB, H×W | Intended stages |
| --- | --- | --- | --- |
| `256x480` | 270×480 | 256×480 | MotionExpert pretraining; Robust fine-tuning |
| `256x320` | 240×320 | 256×320 | SONIC-style post-training |

The geometry follows the pinned Ψ₀ revision
`2830f9367fc12ca49d9b4df4348726daae9db15d`: fixed NEAREST resize, same-size center
crop, training augmentation, then PIL BICUBIC resize to the Qwen 32-pixel grid.
The configured minimum area is 65,536 pixels. Geometry preserves the complete
source image without an additional deterministic crop; fixed rectangular resize
can change its aspect ratio. These profiles are not separate model sizes.

## Install and deterministic extraction

```sh
python -m pip install -e ./ImageProcess
```

The repository root also supports `from ImageProcess import ...` directly.
Only NumPy and Pillow are required; no Qwen or torchvision dependency is needed.

```python
from ImageProcess import SharedImageProcessor, load_profile

profile = load_profile("256x480")  # Also accepts a JSON path or mapping.
processor = SharedImageProcessor(profile)
rgb = processor.process_image(source_rgb)  # HWC uint8, (256, 480, 3).
metadata = processor.metadata              # JSON-serializable provenance.
```

DINO must disable its own second resize/crop and perform only its encoder-specific
rescale/normalization on this RGB. Qwen uses its own normalization. Offline
pretraining extraction uses `process_image`, or `process_window(...,
stage="pretrain")`, with no random augmentation. Metadata binds the complete
profile hash, source revision, processing implementation, Pillow/NumPy versions,
and final height/width. Persist it beside extracted features and in checkpoints.

## Shared training augmentation

```python
processor = SharedImageProcessor("256x320")
window = processor.process_window(history_rgb, stage="posttrain", seed=123)

# Apply each encoder's normalization separately after this shared result.
dino_history_rgb = window.images
qwen_current_rgb = window.images[-1]

# Or replay the exact draw in a separate branch:
same_current = processor.process_window(
    [history_rgb[-1]], stage="posttrain", augmentation=window.augmentation
).images[0]
```

Every frame of the window shares one sampled augmentation, so temporal crop
changes do not create artificial motion. Randomness uses a local NumPy generator;
it does not consume the process-wide random state. Passing the same seed and
inputs reproduces a result. `augmentation` is JSON-serializable and can be replayed
only with the same profile, stage and `training` setting. Use either a seed or a
replay record. Reusing the returned RGB images is preferred to processing twice.

- `pretrain`: deterministic geometry, no random augmentation.
- `posttrain`: brightness/contrast/saturation factors 0.8–1.2 and hue ±0.05,
  in a randomly sampled order.
- `robust`: the same color augmentation plus fixed-aspect random resized crop
  with area fraction 0.85²–1.0, random position and antialiased bilinear resize.
  The RGB output remains the profile's final size.
- `validation`, `inference`, or `training=False`: deterministic geometry only.

Cropping operates at the intermediate 270×480 or 240×320 resolution; final Qwen
grid alignment happens afterwards. These operations perturb image appearance and
viewport, not physical camera pose or 3D labels. There is no mirroring or camera
rotation in this implementation. In `shared_rgb` mode, the sampled crop and color
factors are identical for both encoders, even though their patch layouts and
normalization differ. Sample again for each training window; a single fixed draw
for an entire episode is not required.

Augmenting the shared Motion window requires online DINO encoding of those
augmented images. Unaugmented precomputed DINO embeddings cannot substitute for
such features. This requirement does not apply to VLM-only color augmentation.
Historical image caches must include profile/augmentation identity if reused.

## UnifoLM: VLM color, clean Motion history

The current UnifoLM recipe explicitly sets
`Psi0ModelTransform.motion_image_policy="vlm_color_clean_motion"` and `img_aug=True`.
MotionExpert reads all six historical inputs from the existing deterministic
`256x320` DINO cache. The VLM current frame has the same source, timestamp and
geometry profile, but receives its own Ψ₀ color augmentation during training.
`shared_rgb` remains the compatible default for other callers.

The LeRobot wrapper restores the raw current RGB as uint8 without resizing it.
The model transform then uses this package once:

```python
processor = SharedImageProcessor("256x320")
vlm_current_rgb = processor.process_window(
    [raw_current_rgb], stage="posttrain", training=img_aug and not no_aug
).images[0]
```

The order is NEAREST resize to240×320, sampled color operations, then BICUBIC
alignment to256×320. Applying color after a separate final resize would change
this order. Do not preprocess the raw input with `process_image` first.
No random crop or Robust augmentation is enabled by this policy. Each training
sample gets a fresh color draw; validation, `no_aug`, and inference use clean
deterministic geometry. Motion history remains clean in all cases.

This branch policy reuses the existing cache bytes and metadata without
re-extraction. DINO stays frozen; MotionExpert may still be trained from those
features. It deliberately differs from sharing one augmented RGB across both
encoders and does not add augmentation to cached DINO features.

## MotionExpert and current-frame alignment

MotionExpert reads six real timestamps at 5Hz: previous five frames plus the
current frame, spanning one second. The last timestamp always matches the Ψ₀
current frame. In `shared_rgb` mode, `window.images[-1]` is also the exact processed
RGB supplied to Ψ₀, with separate encoder normalization. Under the UnifoLM branch
policy, only the VLM copy receives training color augmentation. The motion model predicts
the next five frames and exposes one-call hidden tokens for later joint training.

Offline DINO features preserve every native30Hz frame; MotionExpert selects
six 5Hz timestamps relative to the current frame when reading. The pretraining
cache uses deterministic `256x480` geometry, while the UnifoLM post-training cache
uses deterministic `256x320` geometry. Shared Motion-window post-training or Robust
augmentation uses online encoding; the VLM-only UnifoLM color policy keeps its
clean cache. The same frozen DINO weights support
both image sizes and always return CLS plus four register tokens, `[5,1024]`.

For robot startup, the recommended protocol is to collect one second of actual
history before enabling the motion prior. History buffering and readiness are
managed by the caller; this image processor does not create missing history.

Current data and extraction commands are documented in
[pipeline](../pipeline/README.md); fresh training and differentiable hidden-token
interfaces are documented in [MotionExpert](../MotionExpert/README.md).

## CPU checks

```sh
python -m unittest discover -s ImageProcess/tests -v
```

Tests cover both profiles, reference pixel geometry, synchronized historical and
current images, JSON replay, local RNG, augmentation bounds, and invalid inputs.
When torchvision is installed, a pixel-level reference test also compares the
crop and color operations to its PIL implementation. No model download is used.
