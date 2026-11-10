# Σ₀: From Human Interaction and Motion to Humanoid Loco-Manipulation

**English** | [简体中文](README.zh-CN.md)

## Abstract

Humanoid loco-manipulation requires a robot to preserve a hand-object interaction while adapting its whole-body motion across varying initial states. Collecting synchronized whole-body teleoperation demonstrations is costly, yet egocentric videos offer scalable supervision at different action granularities: fine-grained wrist and fingertip trajectories capture object interaction, whereas sparse skeletal motion describes whole-body coordination. This supervision--execution asymmetry motivates learning the two priors separately and coupling them during robot control. We present Σ₀, a vision-language-action framework that learns an interaction prior through hand-action prediction and a coordination prior by jointly reconstructing observed skeletal motion and forecasting its future evolution. During robot-supervised post-training, the policy integrates both priors through proprioception-conditioned serial cross-attention, directly predicting robot-native body and hand actions without retargeting human motion. Across six SIMPLE tasks, Σ₀ achieves 92.8% success, surpassing the strongest baseline by 8.1 percentage points. With 10 demonstrations per task, Σ₀ achieves 83.3% success, whereas the variant without human-prior pretraining achieves 75.0% with 100 demonstrations per task. These results support data-efficient humanoid control from complementary human priors.

## Installation

## Configuration

Default paths and training settings are provided in these files:

| Component | Configuration |
| --- | --- |
| MotionExpert | [Pretraining YAML](MotionExpert/configs/smplh_k12_psi0_pretrain.yaml) |
| DINO feature extraction | [Encoder configuration](pipeline/configs/dino.json) |
| Dual-source training | [Paths and initialization](configs/unifolm.env.sh) |

Data, models and outputs default to `~/sigma0/data`, `~/sigma0/models` and `~/sigma0/runs`. Edit the configurations to use your own storage. The joint training launcher loads `configs/unifolm.env.sh` automatically.

## Pretrained Weights

| Model | Source | Use |
| --- | --- | --- |
| EgoDex390K VLM | [Ψ₀ pretrained backbone](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/pre.fast.2605160748.ckpt.ego390k) | Available VLM initialization for dual-source training |
| DINOv2 ViT-L/14 with registers | [DINOv2 model](https://huggingface.co/facebook/dinov2-with-registers-large/tree/e4c89a4e05589de9b3e188688a303d0f3c04d0f3) | Frozen visual encoder for MotionExpert and UnifoLM |
| MotionExpert | Output of [Motion pretraining](#motionexpert) | `final.ckpt`; joint training selects EMA weights by default |

Download the upstream VLM backbone and the DINO encoder to the default model directories:

```bash
python VLMExpert/scripts/data/download.py \
  --repo-id USC-PSI-Lab/psi-model \
  --remote-dir psi0/pre.fast.2605160748.ckpt.ego390k \
  --local-dir "$HOME/sigma0/models/egodex-vlm" \
  --repo-type model

python scripts/download_dino.py
```

The VLM downloader places the selected model directly in `egodex-vlm/`, including its configuration, tokenizer and weights. The DINO downloader reads its model and version from the [encoder configuration](pipeline/configs/dino.json); use `--output` to choose another model directory.

For VLM pretraining from scratch, see [EgoDex](https://github.com/apple/ml-egodex), [Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) and the [upstream pretraining instructions](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05#pre-train-vlm). Additional [Ψ₀ checkpoints](https://huggingface.co/USC-PSI-Lab/psi-model) are upstream models; Σ₀ result checkpoints are not linked in this repository.

## Data Preparation

| Data | Purpose | Source / preparation |
| --- | --- | --- |
| EgoSuite / EgoStandard | Human motion source data | [Official bucket](https://huggingface.co/buckets/LightwheelAI/EgoStandard), [downloader](datasets/download_egosuite.py) |
| Ego-Exo4D-HM | Human motion and camera source data | [HM dataset](https://huggingface.co/datasets/Ego-Exo4D-HM/npz-datasets), [downloader](datasets/download_egoexo4d_hm.py) |
| EgoDex | Human hand-action pretraining | [Dataset and preparation](https://github.com/apple/ml-egodex) |
| UnifoLM SONIC | Robot-supervised joint training | [Fixed data revision](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/e78fb93cc28912a3031a10b8656d32d7f0a2b867/sonic), [preparation script](VLMExpert/scripts/data/prepare_unifolm.py) |
| SIMPLE | Task demonstrations and evaluation initial states | [Training data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple), [evaluation data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval) |

### Motion pretraining data

Download the source data:

```bash
python datasets/download_egosuite.py run \
  --output-dir "$HOME/sigma0/data/EgoSuite"

python datasets/download_egoexo4d_hm.py \
  --output-dir "$HOME/sigma0/data/EgoExo-4D-HM"
```

The HM downloader uses configured AWS credentials for the associated videos and camera resources. Dataset selection and download options are listed by each script's `--help`.

MotionExpert reads a prepared SMPL-H archive with dataset splits and body-model resources. The complete conversion from raw sources to a trainable archive is not included. See [pipeline](pipeline/README.md#数据格式) for the expected layout.

With a compatible archive available, extract DINO features from its source camera videos:

```bash
ARCHIVE_ROOT="$HOME/sigma0/data/MixEgoMotion/smplh-parameters-v4-camera-quality"
VISION_ROOT="$ARCHIVE_ROOT/vision_profiles/psi0_256x480_v1_30hz"

python -m egomotion_pipeline.cli extract-smplh-features \
  --dataset "$ARCHIVE_ROOT" \
  --output "$VISION_ROOT" \
  --config pipeline/configs/dino.json
```

Then build the window cache and normalization statistics using the [MotionExpert preparation steps](MotionExpert/README.md#数据与缓存). The output paths match the default pretraining YAML. Image and motion conventions are documented in [data protocols](docs/protocols/README.md).

### UnifoLM data and features

The preparation script downloads the pinned SONIC data and preserves the official train/validation packs and shared statistics. Extract DINO features for both packs:

```bash
python VLMExpert/scripts/data/prepare_unifolm.py \
  --root "$HOME/sigma0/data/UnifoLM"

for PACK in unifolm_sonic_lerobot_train unifolm_sonic_lerobot_val; do
  python VLMExpert/scripts/data/extract_unifolm_dino.py \
    --dataset-root "$HOME/sigma0/data/UnifoLM/$PACK" \
    --output "$HOME/sigma0/data/UnifoLM/dino/256x320/$PACK" \
    --model-path "$HOME/sigma0/models/dinov2-with-registers-large" \
    --shared-image-profile 256x320 --device cuda:0 \
    --batch-size 32 --workers 4 \
    --inference-precision bf16 --storage-dtype float16
done
```

The cache stores all native 30 Hz frames. Training selects six real observations at 5 Hz, ending at the VLM image timestamp. See [VLM Expert](VLMExpert/README.md#unifolm-feature-cache) for cache loading and alignment.

### SIMPLE task data

Download an upstream demonstration pack and its evaluation initial states. This example selects Handover; change `TASK` to select another published task:

```bash
TASK=G1WholebodyHandoverTeleop-v0
SIMPLE_ROOT="$HOME/sigma0/data/SIMPLE"

hf download USC-PSI-Lab/psi-data \
  "simple/$TASK.zip" "simple-eval/$TASK.zip" \
  --repo-type dataset --local-dir "$SIMPLE_ROOT"

unzip "$SIMPLE_ROOT/simple/$TASK.zip" -d "$SIMPLE_ROOT/simple"
unzip "$SIMPLE_ROOT/simple-eval/$TASK.zip" -d "$SIMPLE_ROOT/simple-eval"
```

These task packs retain their upstream action and controller conventions. Use a task adapter compatible with the selected data; the dual-source policy interface is documented in [SIMPLE tools](scripts/4090_simple/README.md) and the [control interface](scripts/4090_simple/CONTROL.md).

## Training

### MotionExpert

After preparing the archive, features, window cache and statistics:

```bash
python -m MotionExpert.train --from-scratch
```

| Setting | Default |
| --- | --- |
| GPU workers / batch size per GPU | 8 / 256 |
| Epochs / warmup epochs | 200 / 10 |
| Learning rate / precision | `1e-4` / BF16 mixed |
| Motion rate / window | 5 Hz / 6 observed + 5 future frames |

Adjust `--devices`, `--batch-size` and `--num-workers` for available resources. Resume a run with its saved configuration:

```bash
MOTION_RUN="$HOME/sigma0/runs/motion/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch"

python -m MotionExpert.train --resume "$MOTION_RUN/last.ckpt"
```

### Dual-source action expert

The launcher initializes a new action expert from the downloaded EgoDex VLM and the MotionExpert training output, and jointly updates both priors on UnifoLM:

```bash
bash VLMExpert/scripts/train/psi0/posttrain-dual-source-unifolm.sh
```

| Setting | Default |
| --- | --- |
| GPU workers / batch size per GPU / accumulation | 1 / 16 / 2 |
| Optimizer updates / warmup updates | 40,000 / 1,000 |
| Action-expert learning rate / precision / seed | `1e-4` / BF16 / 292285 |
| Action chunk / action width / state width | 30 / 80 / 45 |

Set `NPROC_PER_NODE`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS` and `MAX_STEPS` to adjust the run. Paths and initial weights are configured in `configs/unifolm.env.sh`. The [VLM Expert guide](VLMExpert/README.md#joint-training) lists parameter-group learning rates, continuation and task-adaptation options. Other launchers are indexed in [training scripts](VLMExpert/scripts/train/README.md); baseline methods are described in [baselines](VLMExpert/baselines/README.md).

## Evaluation

### Motion forecasting

Evaluate motion prediction on 16 validation windows:

```bash
MOTION_RUN="$HOME/sigma0/runs/motion/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch"

python -m MotionExpert.evaluate \
  --checkpoint "$MOTION_RUN/final.ckpt" \
  --config MotionExpert/configs/smplh_k12_psi0_pretrain.yaml \
  --split val --max-samples 16 \
  --output "$HOME/sigma0/runs/evaluation/motion-val.json"
```

Increase the positive `--max-samples` value for a larger selection, or change `--split` and `--checkpoint`. This command reports motion metrics such as position and rotation error. The [MotionExpert API](MotionExpert/README.md#一步hidden与在线共享图像) also provides prediction and feature extraction.

### SIMPLE policy evaluation

Use the dual-source [serving and closed-loop evaluation tools](scripts/4090_simple/README.md). They require a trained run directory, [SIMPLE scene assets](https://github.com/physical-superintelligence-lab/SIMPLE), [SONIC 1.1 resources](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl) and matching controller compatibility metadata. Choose the task, checkpoint, episode count and seed in the evaluation command.

The exact per-result checkpoints, 10/100-demonstration subsets and evaluation settings for the success rates reported in the abstract are not specified in the released configurations.

For demonstration recording, see [PICO teleoperation](docs/teleoperation.md).

## Code Structure

| Component | Description |
| --- | --- |
| [MotionExpert](MotionExpert/README.md) | Human motion prior, forecasting and feature extraction |
| [ImageProcess](ImageProcess/README.md) | Shared image geometry and augmentation |
| [pipeline](pipeline/README.md) | DINO extraction from prepared motion archives |
| [VLM Expert](VLMExpert/README.md) | VLM backbone and dual-source action expert |
| [SIMPLE tools](scripts/4090_simple/README.md) | Policy serving, closed-loop evaluation and demonstration collection |

## Acknowledgements

This project builds on [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0). See [third-party code](docs/dependencies/README.md) for sources and licenses.
