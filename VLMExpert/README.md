# VLM Expert

VLM Expert implements the Σ₀ dual-source policy on top of [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05). See the [project README](../README.md) for data, weights and the training workflow, and [data protocols](../docs/protocols/README.md) for representations.

## Architectures

Set `model.action_architecture` when selecting a model:

| Value | Action expert | Use |
| --- | --- | --- |
| `dual_source_v1` | Serial cross-attention to VLM and Motion features | Σ₀ dual-source training |
| `psi0_joint` | Upstream Ψ₀ action expert | Upstream models and configurations; the configuration default |

Each dual-source block applies action self-attention, VLM cross-attention, Motion cross-attention and an FFN. Each branch has AdaLN modulation and a gated residual. The current proprioceptive state is prepended as a token and removed before action output.

The VLM supplies its complete final-layer multimodal sequence `H_V [B,1,N,Dv]`. MotionExpert supplies `H_M [B,11,768]` from one forward pass at `t_M=1` with Gaussian motion input. Its 11 tokens cover six historical and five future positions; future motion labels are not inputs. The valid-token mean `p_V` and the 11-token mean `p_M` condition the action blocks together with the action timestep.

Dual-source models use the final VLM layer and independent Motion conditioning. They require `n_conditions=0` and disable the legacy DiT, CLIP, FiLM, `traj2ds` and layerwise VLM options. A positive `state_drop_prob` requires `state_null_token=True`. The two action architectures have different checkpoint layouts.

## Inputs and image alignment

The UnifoLM recipe uses the following shapes:

| Input or output | Shape | Contents |
| --- | --- | --- |
| `motion_img_embs` | `[B,6,5,1024]` | DINO CLS and four register tokens at six real 5 Hz timestamps |
| `states` | `[B,1,45]` | Normalized, padded proprioceptive state |
| `action_samples` / action output | `[B,30,80]` | 64 body tokens, 14 hand coordinates and two padding dimensions |
| VLM prefix | Qwen processor output | Current image, instruction and chat template |

Motion observations occur at `t-1.0, t-0.8, t-0.6, t-0.4, t-0.2, t`. The last observation and the VLM image must come from the same camera and timestamp. Missing history cannot be replaced by repeated current images or future frames. VLM inputs contain the observation prefix only, without teacher-forced FAST action answers.

The image geometry is defined by [`ImageProcess/configs/256x320.json`](../ImageProcess/configs/256x320.json). DINO weights, normalization and image profiles must match the Motion checkpoint; adapting Motion pretraining from 256×480 to 256×320 is explicit in the UnifoLM configuration.

| `data.transform.model.motion_image_policy` | RGB handling |
| --- | --- |
| `vlm_color_clean_motion` | UnifoLM recipe: VLM receives the current RGB with shared geometry and training-time color augmentation; Motion reads clean DINO history. `img_aug=True`, `view_aug=False`, random crop disabled. |
| `shared_rgb` | The caller supplies the processed current RGB shared with Motion. Local `img_aug` and `view_aug` are disabled. Any Motion augmentation must be applied consistently to all six RGB frames before online DINO encoding. |

Validation and inference use deterministic geometry with random color augmentation disabled. Color parameters are defined in the image profile; incompatible legacy `color_jitter` settings are rejected.

## UnifoLM feature cache

The [data preparation script](scripts/data/prepare_unifolm.py) preserves the official train/validation packs, actions, states, SONIC 1.1 tokens and shared statistics. Download and extraction commands are in the [project README](../README.md).

[`extract_unifolm_dino.py`](scripts/data/extract_unifolm_dino.py) stores all native 30 Hz DINO features as `[T,5,1024]`, with BF16 encoder inference and FP16 storage in the default recipe. It stores DINO features rather than Motion hidden states so MotionExpert can continue training.

The LeRobot wrapper selects six aligned rows from these caches and supplies `motion_img_embs` to the collator and trainers. Windows without a complete one-second history, with timestamp gaps, or crossing episode boundaries are excluded. Features are read through bounded per-worker mmap/LRU caches.

The reader uses these settings:

```python
dino_cache_root = "~/sigma0/data/UnifoLM/dino/256x320"
shared_image_profile = "256x320"
dino_max_open_episodes = 8
```

Feature manifests record source data, image profile, DINO weights and extraction versions. Use compatible package versions from [`uv.lock`](uv.lock) for extraction and training. Training requires completed manifests for both packs and matching `stats_psi0.json`. Re-running extraction with the same settings reuses completed episodes; changing sources or settings requires a separate cache.

For parallel extraction, set `--num-shards` and `--shard-index` on each GPU process. All shards must share the same output root, image profile, weights, precision and episode selection.

## Joint training

The [dual-source launcher](scripts/train/psi0/posttrain-dual-source-unifolm.sh) loads [`configs/unifolm.env.sh`](../configs/unifolm.env.sh) automatically. It initializes a new action expert and jointly trains the EgoDex VLM and MotionExpert with the action flow loss. DINO stays frozen.

| Setting | Value |
| --- | --- |
| Training steps / warmup | 40,000 / 1,000 |
| Batch size per GPU / gradient accumulation | 16 / 2 |
| Precision / seed | BF16 / 292285 |
| Action chunk / action width / state width | 30 / 80 / 45 |
| Action blocks | 6 |
| Image policy | `vlm_color_clean_motion` |

| Parameter group | Initial learning rate |
| --- | --- |
| Action expert | `1e-4` |
| VLM language backbone | `1e-6` |
| VLM vision tower | `1e-5` |
| VLM multimodal projector | `1e-4` |
| MotionExpert | `1e-5` |

`NPROC_PER_NODE` controls the number of GPU workers; the base launcher defaults to one. `BATCH_SIZE`, `GRAD_ACCUM_STEPS` and `MAX_STEPS` override the corresponding defaults. `MOTION_USE_EMA=1` selects EMA initialization when available; set it to `0` for raw Motion weights.

Run these commands from the repository root:

```bash
bash VLMExpert/scripts/train/psi0/posttrain-dual-source-unifolm.sh --dry-run
bash VLMExpert/scripts/train/psi0/posttrain-dual-source-unifolm.sh --check-only
bash VLMExpert/scripts/train/psi0/posttrain-dual-source-unifolm.sh
```

`--dry-run` prints the command. `--check-only` also checks required files and metadata. Neither starts workers.

Set `tune_vlm=False` with all `tune_mm_*` flags disabled to freeze the VLM, or `tune_motion=False` to freeze MotionExpert. Frozen sources stay in `eval/no_grad`. Joint training preserves gradients through both token sequences and their pooled summaries. The trainer freezes output layers unused by the action loss and uses DDP with `find_unused_parameters=True` for the Motion MoE.

### Resume and task adaptation

Set `UNIFOLM_RESUME_CHECKPOINT` to an existing absolute `checkpoints/ckpt_N` directory to continue training. The launcher restores model weights, optimizer, scheduler and rank RNG state, skips completed updates and retains the learning-rate schedule. DataLoader worker augmentation state is not saved, so augmentation is not guaranteed to replay bit for bit.

For a new task adaptation run, set `model.pretrained_joint_checkpoint` to a trained joint model's `checkpoints/ckpt_N/model.safetensors`. This loads the VLM, MotionExpert and action expert while starting a new optimizer, scheduler and step count. Do not combine it with `train.resume_from_checkpoint`, a separate action-header initialization or LoRA.

Keep the original `model.motion_checkpoint` available when loading a joint checkpoint: the joint weights include MotionExpert parameters, while the original checkpoint supplies its configuration and normalization protocol. Architecture and tensor signatures are checked when loading, including when `strict=False` is requested.

The [optimized launcher](scripts/train/psi0/posttrain-dual-source-unifolm-speedup.sh) enables optional logits, batch-transfer and metric-sync optimizations. It defaults to `--check-only` and eight GPU workers; use `--run` to start training. It uses the same model, loss and learning-rate settings as the base recipe.

## Model API

The model implementation is [`Psi0Model`](src/psi/models/psi0.py), configured by [`Psi0ModelConfig`](src/psi/config/model_psi0.py). When importing from the repository root, include `VLMExpert/src` and the repository root on `PYTHONPATH`.

Load a saved run with `Psi0Model.from_pretrained(run_dir, ckpt_step, launch_config, device)`. It loads `checkpoints/ckpt_<step>/model.safetensors` and initializes the processor and action scheduler from the run configuration.

With a loaded model and a prepared batch, the training forward call is:

```python
output = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    pixel_values=batch["pixel_values"],
    image_grid_thw=batch["image_grid_thw"],
    action_samples=noisy_actions,
    states=states,
    timestep=action_t,
    motion_img_embs=batch["motion_img_embs"],
    return_vlm_features=True,
)
actions = output.action
p_v = output.vlm_features.p_v
p_m = output.motion_features.p_m
```

`extract_vlm_features(input_ids, attention_mask, pixel_values, image_grid_thw)` returns `tokens [B,1,N,Dv]`, `pooled_tokens [B,1,Dv]` and `p_v [B,Dv]` for the dual-source model. Padding is excluded from pooling, and each sample must contain at least one valid token. FP16/BF16 pooling accumulates in FP32 and returns the input dtype. `psi0_joint` additionally supports explicitly selected VLM layers, returning `[B,V,N,Dv]` tokens and `[B,V,Dv]` pooled features.

`extract_motion_features(motion_img_embs=...)` returns `tokens [B,11,768]` and `p_m [B,768]`. `motion_feature_seed` controls the initial motion noise (default `0`); training dropout still follows the training RNG. A fixed seed is reproducible for the same batch layout.

`forward` accepts reusable `vlm_features` and `motion_features` from the same observation. With trainable sources, keep the feature computation graph and re-extract after each optimizer update; detached or stale features cannot replace fresh source forwards.

Sampling is available through `predict_action`, `predict_action_with_training_rtc_flow`, `predict_action_with_rtc_flow` and `predict_action_with_rtc_flow_naive_inpaint`. All accept `motion_img_embs` or reusable `motion_features`, extract observation features once and reuse them through the action denoising loop. Set `return_vlm_features=True` to return actions and feature summaries together. The UnifoLM training recipe disables RTC.

## Further reading

- [Training launchers](scripts/train/README.md)
- [SIMPLE serving and evaluation tools](../scripts/4090_simple/README.md)
- [Teleoperation](../docs/teleoperation.md)
- [Upstream Ψ₀ documentation](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05)
- [Third-party sources and licenses](../docs/dependencies/README.md)
