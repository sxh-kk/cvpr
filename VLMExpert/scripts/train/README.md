# Training launchers

Data, weights and the full workflow are described in the [project README](../../../README.md). The [VLM Expert README](../../README.md) documents model inputs, image alignment and the dual-source API.

## Dual-source training

Run the launcher from the repository root after preparing the data and weights:

```bash
bash VLMExpert/scripts/train/psi0/posttrain-dual-source-unifolm.sh
```

It automatically loads [`configs/unifolm.env.sh`](../../../configs/unifolm.env.sh). Edit that file for your storage paths, or set `UNIFOLM_ENV_FILE` to a separate configuration file.

| Entry point | Purpose |
| --- | --- |
| [`posttrain-dual-source-unifolm.sh`](psi0/posttrain-dual-source-unifolm.sh) | Train a new dual-source action expert while jointly updating the VLM and MotionExpert |
| [`posttrain-dual-source-unifolm-speedup.sh`](psi0/posttrain-dual-source-unifolm-speedup.sh) | Enable optional training optimizations; defaults to input checks, requires `--run` to train |

The base launcher supports `--dry-run` to print the command and `--check-only` to check files and metadata before printing it. Both return without starting training.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `NPROC_PER_NODE` | `1` for the base launcher; `8` for the optimized launcher | GPU workers on one node |
| `BATCH_SIZE` | `16` | Batch size per GPU |
| `GRAD_ACCUM_STEPS` | `2` | Gradient accumulation steps |
| `MAX_STEPS` | `40000` | Optimizer updates |
| `MOTION_USE_EMA` | `1` | Select EMA (`1`) or raw (`0`) Motion initialization |
| `UNIFOLM_RESUME_CHECKPOINT` | Unset | Absolute `checkpoints/ckpt_N` directory for continuation |

For a new task adaptation, use `model.pretrained_joint_checkpoint` to initialize from a trained joint model and start a new optimizer. This differs from continuing a run with `UNIFOLM_RESUME_CHECKPOINT`. See [resume and task adaptation](../../README.md#resume-and-task-adaptation).

## Upstream recipes

The other recipes under [`psi0/`](psi0/) and [`openpi/`](openpi/) are retained from Ψ₀ or implement task-specific adaptations. Their datasets, action representations and initialization weights can differ from the dual-source recipe. Refer to the [pinned Ψ₀ documentation](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05) for the upstream workflows.

## Optional container and cluster tools

| Tool | Purpose |
| --- | --- |
| [`Dockerfile`](Dockerfile) | Package VLM Expert dependencies from [`pyproject.toml`](../../pyproject.toml) and [`uv.lock`](../../uv.lock) |
| [`build-sqsh.sh`](build-sqsh.sh), [`inner.sh`](inner.sh) | Build an Enroot image |
| [`slurm_job.sh`](slurm_job.sh) | Submit an upstream training recipe through Slurm/Pyxis |
| [`ddp.launch.yaml`](ddp.launch.yaml), [`deepspeed.launch.yaml`](deepspeed.launch.yaml) | Accelerate launcher configurations |

Container builds use `VLMExpert/` as their build context. Cluster helpers retain upstream scheduler settings; configure resource requests, container locations, dataset mounts and cache paths for your cluster. The dual-source launcher above uses a single-node, standalone distributed launch.
