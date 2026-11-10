# Σ₀：从人类交互与运动到人形机器人移动操作

[English](README.md) | **简体中文**

## 摘要

人形机器人移动操作要求机器人在不同初始状态下调整全身运动，同时保持手与物体之间的交互。采集同步的全身遥操作示范成本高昂，而第一视角视频能够在不同动作粒度上提供可扩展的监督：精细的手腕和指尖轨迹刻画物体交互，稀疏的骨骼运动则描述全身协调。这种监督与执行之间的不对称性促使我们分别学习两类先验，并在机器人控制中将其耦合。我们提出视觉—语言—动作框架 Σ₀，通过手部动作预测学习交互先验，并通过联合重建已观测的骨骼运动与预测其未来演化来学习协调先验。在机器人数据监督的后训练阶段，策略通过以本体感知为条件的串行交叉注意力融合两类先验，直接预测机器人原生的身体和手部动作，无需对人体运动进行重定向。在 SIMPLE 的六项任务上，Σ₀ 的成功率达到 92.8%，较最强基线高出 8.1 个百分点。每项任务使用 10 条示范时，Σ₀ 的成功率达到 83.3%；未进行人类先验预训练的变体在每项任务使用 100 条示范时，成功率为 75.0%。这些结果支持通过互补的人类先验实现高数据效率的人形机器人控制。

## 安装

## 路径配置

默认路径和训练参数已包含在以下配置中：

| 组件 | 配置文件 |
| --- | --- |
| MotionExpert | [预训练 YAML](MotionExpert/configs/smplh_k12_psi0_pretrain.yaml) |
| DINO 特征提取 | [编码器配置](pipeline/configs/dino.json) |
| 双来源联合训练 | [路径与初始化配置](configs/unifolm.env.sh) |

数据、模型和输出默认放在 `~/sigma0/data`、`~/sigma0/models` 和 `~/sigma0/runs`。可直接修改配置以适配自己的存储位置。联合训练入口会自动加载 `configs/unifolm.env.sh`。

## 预训练权重

| 模型 | 来源 | 用途 |
| --- | --- | --- |
| EgoDex390K VLM | [Ψ₀ 预训练骨干](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/pre.fast.2605160748.ckpt.ego390k) | 可用于双来源训练的 VLM 初始化 |
| DINOv2 ViT-L/14 with registers | [DINOv2 模型](https://huggingface.co/facebook/dinov2-with-registers-large/tree/e4c89a4e05589de9b3e188688a303d0f3c04d0f3) | MotionExpert 和 UnifoLM 使用的冻结视觉编码器 |
| MotionExpert | [运动预训练](#motionexpert)的输出 | `final.ckpt`；联合训练默认使用 EMA 权重初始化 |

将上游 VLM 骨干和 DINO 编码器下载到默认模型目录：

```bash
python VLMExpert/scripts/data/download.py \
  --repo-id USC-PSI-Lab/psi-model \
  --remote-dir psi0/pre.fast.2605160748.ckpt.ego390k \
  --local-dir "$HOME/sigma0/models/egodex-vlm" \
  --repo-type model

python scripts/download_dino.py
```

VLM 下载器会将所选模型的配置、分词器和权重直接放在 `egodex-vlm/` 下。DINO 下载器从[编码器配置](pipeline/configs/dino.json)读取模型及版本，可通过 `--output` 更换模型目录。

需要自行进行 VLM 预训练时，可参考 [EgoDex](https://github.com/apple/ml-egodex)、[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) 和[上游预训练说明](https://github.com/physical-superintelligence-lab/Psi0/tree/cda3fdfafe47cb6fa75400d013de1bd36e3cea05#pre-train-vlm)。其他 [Ψ₀ 权重](https://huggingface.co/USC-PSI-Lab/psi-model)属于上游模型；本仓库尚未提供 Σ₀ 论文结果权重的下载链接。

## 数据准备

| 数据 | 用途 | 来源与准备入口 |
| --- | --- | --- |
| EgoSuite / EgoStandard | 人类运动原始数据 | [官方 bucket](https://huggingface.co/buckets/LightwheelAI/EgoStandard)、[下载器](datasets/download_egosuite.py) |
| Ego-Exo4D-HM | 人类运动及相机原始数据 | [HM 数据集](https://huggingface.co/datasets/Ego-Exo4D-HM/npz-datasets)、[下载器](datasets/download_egoexo4d_hm.py) |
| EgoDex | 人类手部动作预训练 | [数据与准备说明](https://github.com/apple/ml-egodex) |
| UnifoLM SONIC | 机器人数据监督的联合后训练 | [固定数据版本](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/e78fb93cc28912a3031a10b8656d32d7f0a2b867/sonic)、[准备脚本](VLMExpert/scripts/data/prepare_unifolm.py) |
| SIMPLE | 任务示范与评估初始状态 | [训练数据](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple)、[评估数据](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval) |

### 运动预训练数据

下载原始数据：

```bash
python datasets/download_egosuite.py run \
  --output-dir "$HOME/sigma0/data/EgoSuite"

python datasets/download_egoexo4d_hm.py \
  --output-dir "$HOME/sigma0/data/EgoExo-4D-HM"
```

HM 下载器使用已配置的 AWS 凭据获取配套视频和相机资源。数据选择和下载选项见各脚本的 `--help`。

MotionExpert 读取处理后的 SMPL-H 归档、数据分割和人体模型资源。仓库尚未包含原始数据到可训练归档的完整转换流程。归档布局见 [pipeline](pipeline/README.md#数据格式)。

准备好兼容归档后，使用对应的原始相机视频提取 DINO 特征：

```bash
ARCHIVE_ROOT="$HOME/sigma0/data/MixEgoMotion/smplh-parameters-v4-camera-quality"
VISION_ROOT="$ARCHIVE_ROOT/vision_profiles/psi0_256x480_v1_30hz"

python -m egomotion_pipeline.cli extract-smplh-features \
  --dataset "$ARCHIVE_ROOT" \
  --output "$VISION_ROOT" \
  --config pipeline/configs/dino.json
```

随后按照 [MotionExpert 数据准备步骤](MotionExpert/README.md#数据与缓存)构建窗口缓存和归一化统计。输出路径与默认预训练 YAML 一致。图像和运动约定见[数据协议](docs/protocols/README.md)。

### UnifoLM 数据与特征

准备脚本下载固定版本的 SONIC 数据，保留官方训练集、验证集和共享统计。为两套数据提取 DINO 特征：

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

缓存保存全部原生 30Hz 帧，训练时选取 6 个真实的 5Hz 观测，末帧与 VLM 图像时间一致。缓存读取与对齐要求见 [VLM Expert](VLMExpert/README.md#unifolm-feature-cache)。

### SIMPLE 任务数据

下载上游任务示范及评估初始状态。以下默认使用 Handover，更换 `TASK` 可选择其他已发布任务：

```bash
TASK=G1WholebodyHandoverTeleop-v0
SIMPLE_ROOT="$HOME/sigma0/data/SIMPLE"

hf download USC-PSI-Lab/psi-data \
  "simple/$TASK.zip" "simple-eval/$TASK.zip" \
  --repo-type dataset --local-dir "$SIMPLE_ROOT"

unzip "$SIMPLE_ROOT/simple/$TASK.zip" -d "$SIMPLE_ROOT/simple"
unzip "$SIMPLE_ROOT/simple-eval/$TASK.zip" -d "$SIMPLE_ROOT/simple-eval"
```

这些数据保留上游的动作表示和控制器约定，需要使用与所选数据兼容的任务适配器。双来源策略接口见 [SIMPLE 工具](scripts/4090_simple/README.md)和[控制接口](scripts/4090_simple/CONTROL.md)。

## 训练

### MotionExpert

完成归档、特征、窗口缓存和统计准备后：

```bash
python -m MotionExpert.train --from-scratch
```

| 参数 | 默认值 |
| --- | --- |
| GPU 进程数／每卡 batch | 8 / 256 |
| 训练轮数／warmup 轮数 | 200 / 10 |
| 学习率／精度 | `1e-4` / BF16 mixed |
| 运动频率／窗口 | 5Hz / 6 帧观测＋5 帧未来 |

可通过 `--devices`、`--batch-size` 和 `--num-workers` 调整资源。使用已保存的配置恢复训练：

```bash
MOTION_RUN="$HOME/sigma0/runs/motion/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch"

python -m MotionExpert.train --resume "$MOTION_RUN/last.ckpt"
```

### 双来源动作专家

训练入口使用下载的 EgoDex VLM 和 MotionExpert 训练输出初始化，创建新的动作专家，并在 UnifoLM 上联合更新两类先验：

```bash
bash VLMExpert/scripts/train/psi0/posttrain-dual-source-unifolm.sh
```

| 参数 | 默认值 |
| --- | --- |
| GPU 进程数／每卡 batch／梯度累积 | 1 / 16 / 2 |
| 优化步数／warmup 步数 | 40,000 / 1,000 |
| 动作专家学习率／精度／seed | `1e-4` / BF16 / 292285 |
| 动作片段长度／动作维数／状态维数 | 30 / 80 / 45 |

可通过 `NPROC_PER_NODE`、`BATCH_SIZE`、`GRAD_ACCUM_STEPS` 和 `MAX_STEPS` 调整训练。路径和初始化权重由 `configs/unifolm.env.sh` 配置。分组学习率、恢复训练和任务适配见 [VLM Expert](VLMExpert/README.md#joint-training)。其他入口见[训练脚本索引](VLMExpert/scripts/train/README.md)，基线方法见 [baselines](VLMExpert/baselines/README.md)。

## 评估

### 运动预测

在 16 个验证窗口上评估运动预测：

```bash
MOTION_RUN="$HOME/sigma0/runs/motion/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch"

python -m MotionExpert.evaluate \
  --checkpoint "$MOTION_RUN/final.ckpt" \
  --config MotionExpert/configs/smplh_k12_psi0_pretrain.yaml \
  --split val --max-samples 16 \
  --output "$HOME/sigma0/runs/evaluation/motion-val.json"
```

增大正整数 `--max-samples` 可评估更多窗口，也可通过 `--split` 和 `--checkpoint` 更换评估对象。该命令输出位置、旋转等运动误差。[MotionExpert API](MotionExpert/README.md#一步hidden与在线共享图像)还提供运动预测和特征提取。

### SIMPLE 策略评估

使用双来源[模型服务与闭环评估工具](scripts/4090_simple/README.md)。需要已训练的运行目录、[SIMPLE 场景资产](https://github.com/physical-superintelligence-lab/SIMPLE)、[SONIC 1.1 资源](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl)和匹配的控制器兼容元数据，并在评估命令中指定任务、checkpoint、episode 数和 seed。

摘要中各组成功率所用的具体 checkpoint、10／100 条示范子集和评估设置，尚未在已发布配置中逐项指定。

示范录制见 [PICO 遥操作](docs/teleoperation.md)。

## 代码结构

| 组件 | 说明 |
| --- | --- |
| [MotionExpert](MotionExpert/README.md) | 人类运动先验、运动预测与特征提取 |
| [ImageProcess](ImageProcess/README.md) | 共享图像几何与数据增强 |
| [pipeline](pipeline/README.md) | 从运动归档提取 DINO 特征 |
| [VLM Expert](VLMExpert/README.md) | VLM 骨干与双来源动作专家 |
| [SIMPLE 工具](scripts/4090_simple/README.md) | 模型服务、闭环评估与示范采集 |

## 致谢

本项目基于 [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0) 构建。第三方代码来源与许可证见[第三方代码](docs/dependencies/README.md)。
