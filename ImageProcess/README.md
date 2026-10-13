# Shared image processing

MotionExpert/DINO and Ψ₀ receive the same RGB image before each encoder performs
its own rescaling and normalization. This package owns image geometry and
window-synchronized augmentation. It does not change motion coordinates, perform
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
rotation in this implementation. The sampled crop and color factors are identical
for both encoders, even though their patch layouts and normalization differ.

Randomly augmented training requires online DINO encoding of the augmented images.
Unaugmented precomputed DINO embeddings cannot substitute for those features.
Historical image caches must include profile/augmentation identity if reused.

## MotionExpert and current-frame alignment

MotionExpert reads six real timestamps at 5Hz: previous five frames plus the
current frame, spanning one second. The last image, `window.images[-1]`, is the
same current image supplied to the Ψ₀ backbone. Both branches use that processed
RGB directly, with separate encoder normalization. The motion model predicts
the next five frames and exposes one-call hidden tokens for later joint training.

Offline DINO features preserve every native30Hz frame; MotionExpert selects
indices 0, 6, 12, … when reading. This offline cache uses deterministic
`256x480` pretraining geometry. Post-training and Robust augmentation use online
encoding of the shared six-frame window. The same frozen DINO weights support
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
