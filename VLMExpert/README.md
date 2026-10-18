## 本仓库扩展：双来源 Action Expert

`action_architecture="dual_source_v1"` 实现双来源串行结构。当前 UnifoLM 配方设置
`tune_vlm=True, tune_motion=True`，联合更新 VLM、MotionExpert 与新 action expert；DINO 保持冻结。
两套 expert 也可分别通过 `tune_vlm=False`（不设置 `tune_mm_*`）、`tune_motion=False` 固定为 `eval/no_grad`。
原官方架构仍用 `action_architecture="psi0_joint"`，默认值保留以便读取旧模型。

每个 block 执行 `Action Self-Attention → Cross-Attention VLM → Cross-Attention Motion → FFN`。
四个分支各有 AdaLN 和门控残差；每层独立 `ModHead` 输出四组 `gamma/beta/g`，共 `12d`。
调制采用 `(1 + gamma) * LayerNorm(x) + beta`。仅调制头零初始化，attention／FFN 的输出投影
和最终动作 linear 保留常规初始化，避免与零 gate 形成永久无梯度分支。

- `H_V [B,1,N,Dv]`：固定使用 VLM 最后一层的完整多模态序列，padding 同时从均值和注意力排除。
- `H_M [B,11,768]`：MotionExpert 在自身 `t_M=1` 完整高斯输入上前向一次，包含全部历史6／未来5位置。
- `p_V` 为有效 VLM tokens 均值，`p_M` 为全部11个 motion tokens 均值，不使用未来标签 mask。
- 两路摘要分别归一化、投影，与 action 时间 embedding 拼接，经 `f_fuse` 得到调制条件；旧 CLIP 分支不参与。
- 两路完整 memory 独立归一化、投影后供 cross-attention 读取，各 block 不更新 memory。
- 当前 proprio state 投影为 action 序列首 token，输出前移除。RTC 保留逐 action token 时间，
  已知动作前缀及 state 使用 `time_embed(0)`。

首次新建模型示例（先将 `vlm` 加载为需要的 EgoDex 微调权重）：

```python
from psi.config.model_psi0 import Psi0ModelConfig
from psi.models.psi0 import Psi0Model

cfg = Psi0ModelConfig(
    action_architecture="dual_source_v1",
    model_name_or_path="/absolute/path/to/egodex-vlm",
    motion_checkpoint="/absolute/path/to/motion.ckpt",
    motion_use_ema=True,  # 初始化采用所选权重身份；之后该网络参与联合训练
    tune_vlm=True,
    tune_motion=True,
    lang_backbone_lr=1e-6,
    vision_tower_lr=1e-5,
    mm_projector_lr=1e-4,
    motion_expert_lr=1e-5,
    view_feature_dim=vlm.config.text_config.hidden_size,
    n_conditions=0,
)
model = Psi0Model(cfg, vlm).to("cuda").train()
output = model(
    input_ids=input_ids, attention_mask=attention_mask,
    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
    action_samples=noisy_actions, states=states, timestep=action_t,
    motion_img_embs=motion_img_embs,  # [B,6,5,1024]
    return_vlm_features=True,
)
# output.action, output.vlm_features.p_v, output.motion_features.p_m
```

`motion_img_embs` 必须是六个真实5Hz历史时刻的 DINO CLS＋4 register 特征，最后时刻对应 VLM 当前帧。
两路复用同一份处理后的当前 RGB，历史窗口共享增强；DINO 权重身份、归一化和视觉尺寸必须遵守
Motion checkpoint 的视觉协议，跨尺寸适配须显式声明。接口不会从张量形状推断这些来源关系。
标准 transform／collator 会保留并堆叠该字段，三个 action trainer 会转交给模型。
UnifoLM 的 LeRobot wrapper 在配置 `dino_cache_root` 后，会从原生缓存读取历史并处理共享当前 RGB，
无需把整份特征写入 Parquet；其他数据源仍需提供真实历史。
携带历史特征时，`observations` 必须已经是同一共享窗口处理后的当前 RGB，模型 transform 不再单独
裁切或改变颜色；该路径要求关闭本地 `img_aug/view_aug`，窗口增强应在上游统一完成。
现有单帧数据源不会自动得到历史，缺失时明确报错，不复制当前图或读取未来标签。

训练每个 batch、四种 `predict_action*` 采样及三个 trainer 的评估入口均只提取一次两路特征；
去噪循环复用它们。`motion_feature_seed=0` 仅用独立 RNG 生成固定初始运动噪声；
联合训练中的 dropout／router jitter 仍正常消耗训练 RNG；
初始运动噪声在相同 batch 布局下可复现，重排 batch 会改变样本对应的噪声位置。
需要复用时可先调用 `extract_motion_features(motion_img_embs=...)`，再传 `motion_features=...`；
`forward` 也接收本模型 `extract_vlm_features()` 产生的 `vlm_features`。
复用特征必须属于同一观察窗口。联合训练时 tokens 与均值都保留 source graph，禁止用
`detach/no_grad/inference_mode` 的缓存替代可训练来源；每次 optimizer 更新后重新提取，不能复用旧图。
采样遵守外部 `no_grad/inference_mode`，各去噪步仍共用同一次观察提取。

新模式拒绝 layerwise VLM、旧 DiT／CLIP／FiLM／traj2ds 条件。
state dropout 大于零时需启用 `state_null_token=True`，缺失 state 使用专门的 null token。
新 head 的 tensor 架构签名可随 safetensors 保存；旧 joint head、缺参数或配置不匹配的权重会被拒绝，
包括 `strict=False`。新任务从头训练新 head，不恢复旧 optimizer。
完整 checkpoint 包含联合更新后的 Motion 网络权重；恢复时仍需显式原 Motion checkpoint 来重建其配置和统计协议。
从本目录直接运行需 `PYTHONPATH=src:..`，或在专用环境安装仓库根目录的 MotionExpert 包。

### UnifoLM 下载、DINO 提取与读取

数据根目录为 `/gaozt-test1/datasets/UnifoLM`，与 EgoSuite、EgoExo-4D-HM 等并列。
使用 [官方 SONIC 数据包](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/e78fb93cc28912a3031a10b8656d32d7f0a2b867/sonic)，
固定 revision `e78fb93cc28912a3031a10b8656d32d7f0a2b867`。训练 ZIP 为38,855,909,476字节，
下载准备脚本按8MiB持久分块保存，校验每次 HTTP 206 的范围与完整长度，失败只重试当前块；
重跑复用已完成且本地 SHA 匹配的块。最终以官方整包 SHA256、解压 CRC 和 train/val 共享统计验收，
安全拒绝越界 ZIP 或符号链接残留。本地分块 SHA 用于恢复校验，不是官方分块校验值。
脚本只准备数据，不启动训练；已存在的数据需通过原成员 CRC 才能复用。

```bash
# 从仓库根目录运行；下载仅使用 Python 标准库，数据与缓存始终在仓库外。
python VLMExpert/scripts/data/prepare_unifolm.py --root /gaozt-test1/datasets/UnifoLM
# 默认64个并发请求，可用 --download-workers 调整；完成块保存在 .pipeline/downloads。
# 已完整下载时可用 --skip-download，只校验与解压。
```

两套 pack 保留官方 action/state/SONIC1.1 tokens、split 与元数据。目录布局：

```text
/gaozt-test1/datasets/UnifoLM/
  archives/official/sonic/           # 固定版本 ZIP
  unifolm_sonic_lerobot_train/      # 原始 LeRobot pack
  unifolm_sonic_lerobot_val/
  dino/256x320/
    unifolm_sonic_lerobot_train/
      manifest.json
      episodes/00000000/{features.npy,frames.npz,receipt.json}
    unifolm_sonic_lerobot_val/
  .pipeline/                       # 下载／提取日志、收据及本机验证记录
```

**DINO 缓存与 Ψ₀ 读取端共用 `data.shared_image_profile`，当前为 `256x320`。**
唯一几何来源是 [ImageProcess 配置](../ImageProcess/configs/256x320.json)：
共享中间处理后输出 H256×W320；DINO 只再做自己的 rescale／ImageNet normalization，关闭二次 resize/crop。
模型 transform 的 profile 与最终 resize/crop 配置自动同步，缓存分支校验输入 RGB 的尺寸与 uint8 类型。
完整 profile/hash/实现、NumPy/Pillow 版本、DINO 本地权重/config SHA 和原始视频／Parquet 身份随缓存保存。
提取与读取需匹配 Ψ₀ `uv.lock` 的 NumPy1.26.4／Pillow11.3.0；不一致会明确拒绝。
这次后训练使用与 Motion 预训练 H256×W480 相同的 DINO 权重，并显式适配到 H256×W320；
原 Motion 预训练的图像配置和缓存不修改。

```bash
python VLMExpert/scripts/data/extract_unifolm_dino.py \
  --dataset-root /gaozt-test1/datasets/UnifoLM/unifolm_sonic_lerobot_val \
  --output /gaozt-test1/datasets/UnifoLM/dino/256x320/unifolm_sonic_lerobot_val \
  --model-path /gaozt-test1/datasets/MixEgoMotion/weights/dinov2-with-registers-large/e4c89a4e05589de9b3e188688a303d0f3c04d0f3 \
  --shared-image-profile 256x320 --device cuda:0 \
  --batch-size 32 --workers 4 --inference-precision bf16 --storage-dtype float16
# 提取训练集时把 dataset-root/output 的 pack ID 换为 unifolm_sonic_lerobot_train。
```

每个 episode 保存**所有原生30Hz**特征 `[T,5,1024]`，不保存 Motion hidden，后续 Motion 可继续联合训练。
当前配置使用 BF16 encoder inference／FP16 存储，读取时每次复制六行；DINO 参数冻结。
同一批参数可以通过 `--num-shards N --shard-index I` 分配到多个独立 GPU 进程共写同一输出根目录，
所有 shard 必须使用同一 profile、权重、精度与 episode 选择。每个 episode 有独立锁，
写入完成后才提交 receipt；只有所有 episode 齐全才把 manifest 标为 complete。
原参数重跑会复用已提交 episode，未完成 episode 重新提取；来源或配置改变时拒绝复用。

训练读取设置：

```python
# LerobotDataConfig 字段；动作、state 与原来的 delta_timestamps 查询保持一致。
dino_cache_root = "/gaozt-test1/datasets/UnifoLM/dino/256x320"
shared_image_profile = "256x320"
dino_max_open_episodes = 8
```

wrapper 按原始 row/episode/frame/timestamp 建立过滤映射，逐时刻取
`t-1.0, t-0.8, t-0.6, t-0.4, t-0.2, t`，返回 `[6,5,1024]` 的 `motion_img_embs`；
VLM 当前帧保持原动作查询的30Hz时刻。缺少真实一秒历史、跨 episode、缺帧、时间断点、
任一中间原生帧无效的窗口都排除，不重复首帧、不借未来图像。
过滤后的任务采样权重及 episode 迭代同步调整；官方 split 元数据遗留的旧全量范围不用于本地行映射。
特征以每 worker 有界 LRU/mmap 按需读取，不预载整份缓存。
当前离线缓存关闭随机颜色／裁切增强，也拒绝独立 `img_aug/view_aug/adaptive_resize`；
若要随机增强，需要两路共用增强窗口并在线运行冻结 DINO，不能复用这份确定性缓存。
数据与 GPU 验证范围见 [UnifoLM 验证记录](../docs/unifolm_dino_validation.json)。

### UnifoLM 联合训练配方

使用 [posttrain-dual-source-unifolm.sh](scripts/train/psi0/posttrain-dual-source-unifolm.sh)，
必须显式提供 EgoDex VLM、Motion checkpoint、UnifoLM 数据路径／ID，以及
`UNIFOLM_DINO_CACHE=/gaozt-test1/datasets/UnifoLM/dino/256x320`；
脚本的 `--dry-run` 可先打印命令，不启动训练；`MOTION_USE_EMA=1/0` 指定初始化权重选择。
数据 wrapper 自动把已完成的原生 DINO 缓存接入 `motion_img_embs`，并生成同 profile 的当前 RGB；
缺少或未完成的缓存明确报错，下载 ZIP 本身不会自动产生 DINO。

| 参数组 | 初始学习率 | 来源 |
| --- | --- | --- |
| 新 action expert | `1e-4` | Ψ₀ UnifoLM |
| VLM 语言主干 | `1e-6` | Ψ₀ UnifoLM |
| VLM 视觉主干 | `1e-5` | Ψ₀ UnifoLM |
| VLM 多模态投影 | `1e-4` | Ψ₀ UnifoLM |
| MotionExpert | `1e-5` | 本仓库新增的初始设定，可调 |

参考 [Ψ₀ 官方 UnifoLM 配方](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/scripts/train/psi0/posttrain-psix-unifolm-g1-sonic1.1.sh)，
沿用 posttrain、40,000 步、1,000 步 warmup、每卡 batch 16、累积2步和 BF16，
动作80维、state45维、horizon30；action expert 保留本图的6个双来源 block、VLM最后一层、无CLIP。
官方单帧随机增强在这里关闭，增强需在共享历史窗口上完成。
只使用 action flow loss 联合反传，没有额外增加 Motion 重建损失。
VLM 最后 norm 参与所选最后 hidden 的计算，保持可训；独立词表输出头不读 logits，保持冻结。
Motion 仅冻结当前 hidden 接口不使用的运动输出头、被完整遮蔽的轨迹输入投影和未来图像 mask token；
注意力、图像投影、时间编码、路由器、共享／路由专家均可训练。MoE 并非每批命中全部专家，
该模式 DDP 使用 `find_unused_parameters=True, static_graph=False`。
新 head 从头初始化，不能沿用冻结阶段或原 joint head 的 optimizer 状态。

数据准备与验证不启动训练。现有 HTTP／机器人服务还没有历史 DINO 请求字段；
双来源服务部署仍需接入真实历史缓冲，与已实现的离线 LeRobot 读取分开验收。

```bash
cd VLMExpert
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:..:. \
  python -m pytest -p no:cacheprovider -q tests/test_vlm_features.py \
    tests/test_dual_source_action.py tests/test_dual_source_integration.py tests/test_dual_source_data.py \
    tests/test_dual_source_joint_training.py
```

原冻结基线见 [历史验证](../docs/dual_source_action_validation.json)；本轮联合训练验证见
[联合训练验证记录](../docs/dual_source_joint_validation.json)。

## VLM tokens 池化接口

`Psi0Model` 在原来传入 action expert 的完整 hidden tokens 上增加 masked mean pooling，
得到 `p_V`。输入包含图像、指令及聊天模板的有效 tokens，padding 不参与均值；不重新选择层、
不增加归一化或可学习参数。FP16／BF16 输入使用 FP32 累加，输出保持输入 dtype；
两种架构均保留可训练来源的梯度；`dual_source_v1` 根据显式冻结开关决定是否禁用来源梯度。

```python
features = model.extract_vlm_features(
    input_ids=input_ids,
    attention_mask=attention_mask,
    pixel_values=pixel_values,
    image_grid_thw=image_grid_thw,
)
full_tokens = features.tokens  # [B, V, N, D]，原 action expert 的输入
p_v = features.p_v            # 默认 [B, D]
```

`N` 是完整多模态序列长度，`D` 由所加载的 VLM 决定。默认取原来的最后一层，`V=1`；
仅 `psi0_joint` 可显式配置 `vlm_layer_indices`，此时 `V` 是选中的层数，逐层池化，不混合各层。
`dual_source_v1` 固定 `V=1`。
`features.pooled_tokens` 始终是 `[B,V,D]`；`p_v` 在 `V>1` 时也是 `[B,V,D]`。
`attention_mask` 为 `[B,N]` 的 0/1 或 bool；传 `None` 表示全部有效，每个样本至少需一个有效 token。

`forward` 和四个 `predict_action*` 方法共用该提取接口，每个观察输入只提取／池化一次。
默认输出保持原样；传入 `return_vlm_features=True` 可通过返回对象的 `.action`、
`.vlm_features.p_v` 同时取得动作与摘要。完整 tokens 仍进入原注意力，特征对象另存于
`joint_attention_kwargs["vlm_features"]`，供双来源 action expert 读取。
`psi0_joint` 保留原 CLIP `pooled_projections` 路径；`dual_source_v1` 使用 `p_V`、`p_M` 和 action 时间调制。

提取使用当前实例加载的 VLM 权重；使用 EgoDex 微调结果时需先加载对应权重。
输入必须是观察 prefix，不能包含 teacher forcing 的真实 FAST 动作答案。
池化接口不增加 GLOBAL token 或可学习参数；双来源 action head 是单独的新参数结构。

轻量回归验证（PyTorch、pytest；接线检查用实际方法和模型替身，无需下载权重）：

```bash
cd VLMExpert
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
  python -m pytest -p no:cacheprovider -q tests/test_vlm_features.py
```

初始池化接口的 CPU／单卡 BF16 验证见 [池化验证记录](../docs/vlm_pooling_validation.json)。

以下保留官方上游说明；源码快照来源见 [版本记录](../docs/psi0_upstream.json)。

---

<h1 align="center">[RSS'26] Ψ₀: An Open Foundation Model <br/> Towards Universal Humanoid Loco-Manipulation
</h1>

<p align="center">
  <img src="assets/media/teaser.jpg" alt="Psi0 teaser image" />
</p>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.12263-df2a2a.svg)](https://arxiv.org/abs/2603.12263)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://psi-lab.ai/Psi0)
[![Model](https://img.shields.io/badge/Hugging%20Face-Model-yellow)](https://huggingface.co/USC-PSI-Lab/psi-model)
[![Data](https://img.shields.io/badge/Hugging%20Face-Data-pink)](https://huggingface.co/datasets/USC-PSI-Lab/psi-data)
[![License](https://img.shields.io/badge/License-Apache2.0-blue.svg)](./LICENSE)

</div>

Contributors: [Songlin Wei](https://songlin.github.io/), [Hongyi Jing](https://hongyijing.me/), [Boqian Li](https://boqian-li.github.io/), [Zhenyu Zhao](https://zhenyuzhao.com/), [Jiageng Mao](https://pointscoder.github.io/), [Zhenhao Ni](https://nizhenhao-3.github.io/) , [Sicheng He](https://hesicheng.net/), [Jie Liu](https://jie0530.github.io/), [Xiawei Liu](https://www.xiaweiliu.com/), Kaidi Kang,  Sheng Zang,[Weiduo Yuan](https://weiduoyuan.com/), [Marco Pavone](https://profiles.stanford.edu/marco-pavone), Di Huang, [Yue Wang](https://yuewang.xyz/)

-------

$\Psi_0$ is an open vision-language-action (VLA) model for dexterous humanoid loco-manipulation. Our model first learns task semantics and visual representation from large-scale human egocentic videos, and then is post-trained on a smaller amount of real-world teleoperated robot data, to learn general dynamics of the embodiment. 

<details>
<summary>[Optional] Expand to know more about Ψ₀.</summary>

Our foundation model is capable of acquiring new long-horizontal dexterous loco-manipulation skill by fine-tuning using as few as 80 trajectories. ***Our key finding is that scaling the right data in the right way.***

At the top, the $\Psi_0$ model consists of two end-to-end trained components: a vision–language backbone (System-2) and a multimodal diffusion transformer (System-1) action expert. The backbone is based on Qwen’s Qwen3-VL-2B-Instruct, which extracts vision–language features from observations and instructions. These features condition a flow-based multimodal diffusion transformer inspired by Stable **Diffusion** 3. The action expert (≈500M parameters) predicts future whole-body action chunks, enabling efficient fusion of visual, linguistic, and action representations. At the lowest level (System-0), an RL-based tracking controller executes the predicted lower-body action commands, ensuring stable and precise physical control.

<p align="center">
  <img src="assets/media/arch.png" alt="Psi0 model" />
</p>
</details>

<p></p>

## 📢 News & Updates

- [x] [2026-09-13] Released new Psi-0 [checkpoints](#checkpoints) specifically trained for [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) and its training recipes: [post-train](scripts/train/psi0/posttrain-psix-unifolm-g1-sonic1.0.sh) and [fine-tune](scripts/train/psi0/finetune-sonic-psi-dream-baseline.sh). Checkout out this [release node](examples/psi0_for_sonic.md).
- [x] [2026-08-30] Released [SIMPLE SONIC wholebody training recipe](scripts/train/psi0/finetune-sonic-simple-psi0.sh).
- [x] [2026-08-30] Released [docker support](#docker-support).
- [x] [2026-07-14] Released DreamZero baseline for SIMPLE.
- [x] [2026-06-13] Released SONIC integration for Psi-0.
- [x] [2026-06-03] 🎉🎉🎉 Psi-0 won the Best Paper Award at the 2nd 3D-LLM/VLA Workshop at CVPR 2026.


## Table of Contents
<!-- - [Installation](#-environment-setup) -->
<!-- - [Pre- & Post- Training](#-) -->
<!-- - [Data Pre-Processing](#-) -->
- [Finetune Ψ₀ on Unitree G1 Humanoid Robot](#finetune-psi0)
  - [Installation](#installation)
  - [Data Collection](#data-collection)
  - [Fine-Tuning](#training-real)
  - [Open-Loop Evaluation](#open-loop-evaluation)
  - [Deployment](#deployment)
  - [Ψ₀ with SONIC](#psi0-sonic)
- [Baselines](#baselines)
  - [GR00T N1.6](#groot-n16)
  - [OpenPi π0.5](#openpi-05)
  - [InternVLA-M1](#internvla-m1)
  - [H-RDT](#h-rdt)
  - [EgoVLA](#egovla)
  - [Diffusion Policy](#diffusion-policy)
  - [ACT](#act) 
- [Simulation 🚀🚀🚀](#simulation)
  - [Install SIMPLE](#install-simple)
  - [Data Generation](#data-generation)
  - [Fine-Tuning](#training-sim)
  - [Evaluation in SIMPLE](#evaluation-in-simple)
- [Reproduce Ψ₀: Pre-Training and Post-Training](#pre-post-train)
- [Checkpoints](#checkpoints)
- [Troubleshootings](#troubleshootings)
- [Citation](#️-citation)

<a id="finetune-psi0"></a>
## Finetune Ψ₀ on Unitree G1 Humanoid Robot

### Installation

Clone the project and change directory to the project root:
```bash
git clone git@github.com:physical-superintelligence-lab/Psi0.git 
cd Psi0
```
We use [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python dependencies. Install `uv` if not already installed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Set up the $\Psi_0$ environment:

> ℹ️ We manage the $\Psi_0$ environment and all the baselines through `uv` and they all share the same `src/` code.  See [Environment Management](baselines/README.md) for more details.

```
uv venv .venv-psi --python 3.11
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync \
  --group serve \
  --group viz \
  --group psi \
  --index-strategy unsafe-best-match \
  --active
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
```


> **Intel GPU (XPU) inference:** Psi0 supports inference on Intel GPUs. After `uv sync` above, install the XPU-enabled PyTorch wheels:
> ```bash
> uv pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/xpu
> ```
> Skip the `flash_attn` installation on XPU; FlashAttention is CUDA-only and Psi0 uses PyTorch SDPA when it is unavailable. The serving CLI defaults to `cuda:0`, so explicitly pass `--device xpu:0` when serving on XPU.

Test installation, a version number should be displayed.
```bash
python -c "import psi;print(psi.__version__);"
```

Verify the shared `lerobot` stack is importable.
```bash
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

#### Docker Support

Pull the prebuilt image from the GitHub Container Registry:

```bash
docker pull ghcr.io/physical-superintelligence-lab/psi0:latest
docker tag ghcr.io/physical-superintelligence-lab/psi0:latest psi:train
```

The retag is what lets `docker-compose.yml` find it — it refers to the image as
`psi:${PSI_TAG:-train}`, and `PSI_TAG` overrides only the tag, not the registry
path. Use a dated tag such as `:260830` to pin an exact build.

The image holds **dependencies only** — python 3.11, torch 2.7 cu12 and
flash-attn in `/workspace/.venv-psi`, on a plain CUDA base. Your checkout stays
outside it and is bind-mounted, so `psi` is installed editable and source edits
take effect on a container restart rather than a rebuild.

Serve a checkpoint with it:

```bash
docker compose up serve-psi0-sonic-http
```

Every argument is overridable from the shell or `.env` (`PORT`, `RUN`,
`CKPT_STEP`, `ACTION_EXEC_HORIZON`, `RTC_FLAG`, `GPUS`, ...), and
`docker compose run --rm serve-psi0-sonic-http --help` lists the server's own
options.

If you want to build it locally:

```bash
docker compose build psi     # or: docker build -f scripts/train/Dockerfile -t psi:train .
```

See [scripts/train/README.md](scripts/train/README.md) for the enroot/cluster
recipe and for how dependencies are pinned.

<details>
<summary> [Deprecated.] Psi-0 with AMO</summary>

### Data Collection
> 📂 We open-sourced all the 9 real-world tasks. You can directly download the data and jump to the [Fine-Tuning](#training-real).

See the detailed teleoperation guide here:  
[Real-World Deployment Guide](real/README.md#real-world-deployment)


#### Pre-Processing: Convert Raw Data to LeRobot Format

```
export task=Hug_box_and_move

hf download USC-PSI-Lab/psi-data \
  g1_real_raw/$task.zip \
  --local-dir=$PSI_HOME/data/real_teleop_g1 \
  --repo-type=dataset

unzip $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task.zip -d $PSI_HOME/data/real_teleop_g1/g1_real_raw/$task
```
You should observe similar folder structure:

```
g1_real_raw
└── Hug_box_and_move
    ├── episode_0
    │   ├── color
    │   │   ├── frame_000000.jpg
    │   │   └── ...
    │   └── data.json
    └── ...
```

Edit the task description file with the following format, eg.,
```
vim scripts/data/task_description_dict.json
```
```
{
  "Hug_box_and_move": "Hug box and move."
}
```

Run conversion script
```
python scripts/data/raw_to_lerobot.py \
  --data-root=$PWD/data/real_teleop_g1/g1_real_raw \
  --work-dir=$PWD/data/real \
  --repo-id=psi0-real-g1 \
  --robot-type=g1 \
  --task=$task
```

Calculate stats
```
python scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/real \
  --task=$task
```

Create **$\Psi_0$** format stats (simply a copy for now)
```
cp $PSI_HOME/data/real/$task/meta/stats.json $PSI_HOME/data/real/$task/meta/stats_psi0.json
```

Now it's ready to finetune $\Psi_0$.

> ✈️ If training env is already configured, directly launch training via `scripts/train/psi0/finetune-real-psi0.sh $task`


<a id="training-real"></a>
### Fine-Tuning

> ✔️ Suppose the data is already collected and processed. Now we can proceed to fine-tune the $\Psi_0$ model.

>  There is a [known issue](https://github.com/physical-superintelligence-lab/Psi0/issues/3) of loading our real data, apply this fix first `python scripts/data/patch_lerobot_meta.py $PSI_HOME/data/real/$task`

> 📝 Here we illustrate by using the pre-collected data from [Huggingface psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/real).

Set up the environment variables following `.env.sample`. The environment variables will be loaded by the `dotenv.load_dotenv()` in python.
```
cp .env.sample .env
# and edit the following env variables 
# HF_TOKEN=<YOUR HF READ TOKEN>
# WANDB_API_KEY=<API KEY for wandb logging>
# WANDB_ENTITY=<wandb entity>
# PSI_HOME=<Path where PSI cache/checkpoint/data are located by convention>

source .env
echo $PSI_HOME
```

Download the collected real-world data and extract it:
```
export task=Pick_bottle_and_turn_and_pour_into_cup

hf download USC-PSI-Lab/psi-data \
  real/$task.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip $PSI_HOME/data/real/$task.zip -d $PSI_HOME/data/real
```
> 👀 If you want to visualize the episode please refer to the [Data Visualization](examples/visualize.md) in the examples.

Launch the training script:
```
scripts/train/psi0/finetune-real-psi0.sh $task
```

> 🖥️ You can always change the GPUs, e.g., `CUDA_VISIBLE_DEVICES=0,1,2,3 scripts/train/...`.  

> ⚠️ Please try to maintain a reasonable global batch size = device batch size x number of GPUs x gradient accumulation step. We use global batch size 128 throughout all the real-world and simulation experiments.


### Open-Loop Evaluation
> Follow the steps in `examples/simple/openloop_eval.ipynb`

Load the training dataset, and run model inference to see how model fits the training data.

### Deployment

#### Serve $\Psi_0$ (RTC mode)

```bash
bash ./scripts/deploy/serve_psi0-rtc.sh
```

#### Start $\Psi_0$ Client (RTC mode)

```bash
bash ./real/scripts/deploy_psi0-rtc.sh
```

For detailed real-world deployment environment setup, please also refer to the dedicated documentation:

[Real-World Teleoperation Guide](real/README.md)
</details>

<a id="psi0-sonic"></a>
### Ψ₀ with SONIC

[SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) is a powerful whole-body controller for humanoid robots. $\Psi_0$ now supports data collection, fine-tuning, and deployment with SONIC. Please use [our fork](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl/tree/main) to avoid any compatibility issues.

Initialize the SONIC submodule first:

```bash
git submodule update --init --recursive third_party/GR00T-WholeBodyControl
```

For the full environment setup — workstation venvs, TensorRT + C++ build, PICO/XRoboToolkit, and the robot-side camera server — see the **[SONIC real-world teleoperation guide](real/SONIC/README.md)**.

#### Data collection

Please follow the [SONIC real-world teleoperation guide](real/SONIC/README.md#data-collection) to record demonstrations.

Datasets are saved locally under `third_party/GR00T-WholeBodyControl/outputs/<dataset-name>/` in LeRobot format.

#### Pre-Processing: Convert to $\Psi_0$ LeRobot Format

Convert the SONIC-collected dataset into the $\Psi_0$ LeRobot format:

```bash
export task=<dataset-name>

python scripts/data/raw_sonic_to_psi_lerobot.py \
  --data-root=third_party/GR00T-WholeBodyControl/outputs/$task \
  --work-dir=$PSI_HOME/data/sonic/lerobot \
  --repo-id=$task \
  --robot-type=g1
```

Calculate stats
```bash
python scripts/data/calc_modality_stats.py \
  --work-dir=$PSI_HOME/data/sonic/lerobot \
  --task=$task
```

Create **$\Psi_0$** format stats (simply a copy for now)
```bash
cp $PSI_HOME/data/sonic/lerobot/$task/meta/stats.json $PSI_HOME/data/sonic/lerobot/$task/meta/stats_psi0.json
```

Now it's ready to fine-tune.

#### Finetune $\Psi_0$ with SONIC

```bash
bash ./scripts/train/psi0/finetune-real-sonic-psi0.sh $task
```

#### Deploy $\Psi_0$ with SONIC

Please follow the [SONIC real-world deployment guide](real/SONIC/DEPLOYMENT.md) for detailed instructions.

##### Serve Policy Server of $\Psi_0$ with SONIC (RTC mode)

```bash
bash ./scripts/deploy/serve_psi0-rtc-sonic.sh
```

##### Start whole-body controller on robot for $\Psi_0$ with SONIC (RTC mode)

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-robot.sh
```

##### Start Policy Client of $\Psi_0$ with SONIC (RTC mode)

```bash
bash ./real/scripts/deploy_psi0-sonic-rtc-client.sh
```

<details>
<summary>Baselines </summary>

## Baselines
<a id="groot-n16"></a>

### GR00T
Install the env 
```bash
cd src/gr00t; uv sync
```
1. training
```bash
cd src/gr00t
./scripts/train_gr00t.sh --dataset-path /your/lerobot/dataset
```
2. serving a checkpoint
```bash
cd src/gr00t
./scripts/deploy_gr00t.sh
```

3. openloop eval on trained checkpoint using gt
```bash
cd src/gr00t
./scripts/openloop_eval.sh
```

<a id="openpi-05"></a>

### OpenPI $\pi_{0.5}$

Please see more detailed instructions here: [baselines/pi05](baselines/pi05/README.md).

### InternVLA-M1
Install the env 
```bash
cd src/InternVLA-M1; uv sync --python 3.10
```
1. training
```bash
cd src/InternVLA-M1
bash scripts/train_internvla.sh
```
2. serving a checkpoint
```bash
cd src/InternVLA-M1
./scripts/deploy_internvla.sh
```

### H-RDT

See quick-start doc for [baseline/hrdt](examples/quick_start/hrdt.md).

### EgoVLA

See quick-start doc for [baseline/egovla](examples/quick_start/egovla.md).

### Diffusion Policy
See dedicated doc here [baseline/dp](baselines/dp/README.md)

### ACT
See dedicated doc here [baseline/act](baselines/act/README.md)
</details>

## Simulation

We use [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) to benchmark $\Psi_0$ and all the baselines.

> 📢 SIMPLE is an easy-to-use humanoid benchmarking simulator built on the MuJoCo physics engine and Isaac Sim rendering.

### Install SIMPLE

Currently, there are two options to integrate SIMPLE and Psi-0.

#### [Option 1] Install stand-alone SIMPLE (Best for collecting data through teleoperation)

> We recommend to install [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) on stand alone desktop with a NVIDIA GPU (3090/4090/5090). 

Please refer to the SIMPLE repo [here](https://github.com/physical-superintelligence-lab/SIMPLE)

#### [Option 2] Install SIMPLE as third-party dependency (Best for evaluting Psi-0 and all baselines)

Please refer the more details steps [here](examples/quick_start/psi.md).

### Data Generation
> 📂 We also provide 6 pre-collected whole-body humanoid loco-manipulation tasks at [Huggingface psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple). If you want to use the existing simulation data, jump to the [Fine-Tuning](#training-sim)

#### Motion-Planning Based Data Generation
Please refert to the SIMPLE docs.

#### Teleoperation in Simulator
Please refert to the SIMPLE docs.

<a id="training-sim"></a>
### Fine-Tuning

> 👉 You can skip fine-tuning and download our released [checkpoints for SIMPLE](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/simple-checkpoints).

Download [SIMPLE task data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple) and extract it:

> 💡 Dont forget `source .env` first before following below commands.

```
export task=G1WholebodyXMovePickTeleop-v0

hf download USC-PSI-Lab/psi-data \
  simple/$task.zip \
  --local-dir=$PSI_HOME/data \
  --repo-type=dataset

unzip $PSI_HOME/data/simple/$task.zip -d $PSI_HOME/data/simple
```

> 👀 If you want to visualize the episode please refer to the [Data Visualization](examples/visualize.md) in the examples.

Start training:

> Please [set up the envrionment variables](#training-real) if not done so yet.

```
bash scripts/train/psi0/finetune-simple-psi0.sh $task
```
The training will create a run dir which is located under `.runs` in the project root.
If your GPU has limited VRAM, set `--train.optimizer-foreach=false` to reduce optimizer-step memory usage at the cost of some speed.

### Evaluation in SIMPLE

#### Serve $\Psi_0$
```
export run_dir=<the run dir here under folder .runs>
export ckpt_step=<checkpoint step>
uv run --active --group psi --group serve serve_psi0 \
  --host 0.0.0.0 \
  --port 22085 \
  --run-dir=$run_dir \
  --ckpt-step=$ckpt_step \
  --action-exec-horizon=24 \
  --rtc
```

For Intel XPU, add `--device xpu:0` to the command above.

Run open-loop evaluation (offline)

[examples/simple/openloop_eval.ipynb](examples/simple/openloop_eval.ipynb)

#### Run the Evaluation in SIMPLE

This `quick-start` guide assumes running SIMPLE on a Stand-alone workstation with NVIDIA GPU.

> We recommend serving the VLA models on a remote server other than locally as IsaacSim is also resource demanding. 

> If the server is started on a remote server, run ssh port forward. eg., `ssh -L 22086:localhost:22086 songlin@nebula100`.

> Once port forward is done, open a new terminal to test if server is up `curl -i http://localhost:22085/health`

Download eval tasks from [USC-PSI-Lab/psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/simple-eval).


```
cd /path/to/SIMPLE
export task=G1WholebodyXMovePickTeleop-v0
```

Download eval data and extract it:
```
hf download USC-PSI-Lab/psi-data \
	simple-eval/$task.zip \
	--local-dir=data/evals \
	--repo-type=dataset

unzip data/evals/simple-eval/$task.zip -d data/evals/simple-eval
```

Now start SIMPLE eval in the SIMPLE environment:

> We provide three domain randomization levels: `level-0`, `level-1`, `level-2` for each task

```
export dr=level-0
```
We use two different entrypoints for evaluating different tasks:

set entrypoint and agent to `eval_decoupled_wbc.py` and `psi0_decoupled_wbc` if the evaluating task ends with `Teleop`, which means the task data is collected using teleoperation:
```
export entry=eval_decoupled_wbc.py
export agent=psi0_decoupled_wbc
```

and set entrypoint and agent to `eval.py` and `psi0` if the evaluating task ends with `MP`, which means the task data is generated using CuRobo Motion planning:
```
export entry=eval.py
export agent=psi0
```

Launch the evaluation script:
```
python src/simple/cli/$entry \
	simple/$task \
	$agent \
	$dr \
	--host=localhost \
	--port=9000 \
	--sim-mode=mujoco_isaac \
	--no-headless \
	--data-format=lerobot \
	--data-dir=data/evals/simple-eval/$task/$dr
```

The policy rollout videos will be found in folder `third_party/SIMPLE/data/evals/psi0`.

> The evaluation for a single episode could take up to 6~10 minutes because SIMPLE use a synchronous rendering API in IsaacSim. See here for [more explanation](#).

<a id="pre-post-train"></a>
## Reproduce Ψ₀: Pre-Training and Post-Training


### Pre-Train VLM

Download and cache the official `Qwen/Qwen3-VL-2B-Instruct` weights.
```
python scripts/predownload_qwen3vl.py
```

Pre-train on the [EgoDex dataset](https://github.com/apple/ml-egodex)

Pre-compute `48 DoF EgoDex action`:

> We re-use the pre-process code from [H-RDT EgoDex Pre-Processing](https://github.com/HongzheBi/H_RDT?tab=readme-ov-file#data-preprocessing).
> 1. Change the paths in `src/h_rdt/datasets/pretrain/setup_pretrain.sh`.
> 2. Tweak the `NUM_PROCESSES` if on a powerful server, i tried max 64.
> 3. set `FORCE_OVERWRITE=True` if the processing script is disrupted.

```
source src/h_rdt/datasets/pretrain/setup_pretrain.sh
source .venv-psi/bin/activate
bash src/h_rdt/datasets/pretrain/run_pretrain_pipeline.sh
```

> [Optinal] If you also want to train `FAST` tokenizer, please refer to [traing FAST](src/fast/README.md).

```
bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh 
```

Pre-train on [humanoid everyday dataset](https://huggingface.co/datasets/USC-GVL/humanoid-everyday)

> Please download the pre-processed HE data here:  `hf download USC-PSI-Lab/psi-data HE_RAW.zip --repo-type=dataset`

```
bash scripts/train/psi0/pretrain-he-psi0-fast.sh
```

Save the pretrained checkpoints once training is done:
```
python scripts/save_pretrain_qwen3vl_backbone.py
```

### Post-Train Action Expert

Download pre-trained `psi-0` VLM backbone
```
python scripts/data/download.py \
  --repo-id=USC-PSI-Lab/psi-model \
  --remote-dir=psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --local-dir=$PSI_HOME/cache/checkpoints/psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k \
  --repo-type=model
```

Post-train on [humanoid everyday (HE) dataset](https://huggingface.co/datasets/USC-GVL/humanoid-everyday)
```
bash scripts/train/psi0/posttrain-he-psi0.sh
```

Save post-trained action header once training is over
```
python scripts/save_posttrain_action_expert.py
```

## Checkpoints

The released checkpoints on [HuggingFace Psi-Model](https://huggingface.co/USC-PSI-Lab/psi-model) is listed

***Newest checkpoints for Psi-0 + SONIC whoblebody control***:

| Checkpoint | Description | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(EgoDex) | Pre-trained VLM backbone (EgoDex 390K steps) | `psi0/pre.fast.2605160748.ckpt.ego390k` |
| $\Psi_0$ VLM<br/>(Sonic-v1.0) | Post-trained VLM On [50H UnifoLM](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/sonic)  | `psi0/postpre.sonic1.0.unifolm.2609092156.40k` |
| $\Psi_0$ Action Expert<br/>(Sonic-v1.0) | Post-trained Action Expert On [50H UnifoLM](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/sonic) | `psi0/postpre.sonic1.0.unifolm.2609092156.40k` |
| $\Psi_0$ VLM + Action Expert <br/>(Sonic) | Fine-tuned VLM On `5h Psi-Dream data`  | `psi0/sonic-checkpoints/multi-task.psi-dream.2609092156` |
|---|---|---|
| $\Psi_0$ VLM<br/>(Sonic-v1.1) | Post-trained VLM On [50H UnifoLM](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/sonic)  | `psi0/postpre.sonic1.1.unifolm.2609181726.40k` |
| $\Psi_0$ Action Expert<br/>(Sonic-v1.1) | Post-trained Action Expert On [50H UnifoLM](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/main/sonic) | `psi0/postpre.sonic1.1.unifolm.2609181726.40k` |

`Old` Psi-0 + AMO decoupled wholebody control:

| Checkpoint | Description | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(Baseline) | Pre-trained VLM backbone (EgoDex 200K steps + HE 30K steps) | `psi0/pre.fast.1by1.2601091803.ckpt.ego200k.he30k` |
| $\Psi_0$ Action Expert<br/>(Baseline) | Post-trained Action Expert On HE | `psi0/postpre.1by1.pad36.2601131206.ckpt.he30k` |

and more variants for ablation studies:
| Checkpoint | Description | Remote Directory |
|---|---|---|
| $\Psi_0$ VLM<br/>(Ablation Study) | Pre-trained VLM backbone only on EgoDex 200K steps | `psi0/pre.fast.egodex.2512241941.ckpt200k` |
| $\Psi_0$ VLM<br/>(Ablation Study) | Pre-trained VLM backbone only on HE 48K steps  | `psi0/pre.abl.only.he.2512311516.48k` |
| $\Psi_0$ VLM<br/>(Ablation Study) | Pre-trained VLM backbone only on 10% EgoDex  | `psi0/pre.abl.ego.10per.2602021632.46k` |
| $\Psi_0$ Action Expert<br/>(Ablation Study) | Post-train on HE by picking pre-trained variant `psi0/pre.abl.only.he.2512311516.48k` | `psi0/postpre.abl.only.he.2602050012` |
| $\Psi_0$ Action Expert<br/>(Ablation Study) | Post-train on HE by picking pre-trained variant `psi0/pre.abl.ego.10per.2602021632.46k` | `psi0/postpre.abl.ego.10per.2602050006` |


Download the selected models

> Edit `.env` to use `HF_ENDPOINT=https://hf-mirror.com` if needed.

```
export CKPT_DIR=<Remote Directory>

python scripts/data/download.py \
  --repo-id=USC-PSI-Lab/psi-model \
  --remote-dir=$CKPT_DIR \
  --local-dir=$PSI_HOME/cache/checkpoints/$CKPT_DIR \
  --repo-type=model
```

## Troubleshootings

1. Lerobot dataset issues: `stack(): argument 'tensors' (position 1) must be tuple of Tensors, not Column`

This usually means the environment is still on the legacy PSI `lerobot` stack. Resync the PSI env so it uses the
same `lerobot` and `datasets` versions as SIMPLE, then verify the import layout:

```bash
source .venv-psi/bin/activate
uv sync --group psi --active
python -c "from psi.data.lerobot.compat import LEROBOT_LAYOUT; print(LEROBOT_LAYOUT)"
```

2. Fail to install `evdev`, `src/evdev/input.c:10:10: fatal error: Python.h: No such file or directory`

```
sudo apt update
sudo apt install -y python3-dev python3-venv build-essential \
    linux-headers-$(uname -r)
```

3. RuntimeError: Could not load libtorchcodec. Likely causes ...
```
sudo apt-get install ffmpeg
```

4. ImportError: cannot import name 'Deprecated' from 'wandb.proto.wandb_telemetry_pb2' 

re-install `wandb`
```
source .venv-pusht/bin/activate
uv pip uninstall wandb
uv pip install wandb==0.18.0
```

5. support `sm_120` on newer GPUs like `5090` or `RTX 6000`, UserWarning: Ignoring invalid value for boolean flag CUDA_LAUNCH_BLOCKING: truevalid values are 0 or 1.

update `torch` and `flash-attn`
```
uv pip uninstall torch torchvision
uv pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install flash-attn --no-build-isolation
```

6. Failed to download and build `lerobot ... `, Use `git lfs logs last` to view the log.

```
GIT_LFS_SKIP_SMUDGE=1 uv ...
```
## Citation

```
@article{wei2026psi0,
  title={{$\Psi_0$}: An Open Foundation Model Towards Universal Humanoid Loco-Manipulation},
  author={Wei, Songlin and Jing, Hongyi and Li, Boqian and Zhao, Zhenyu and Mao, Jiageng and Ni, Zhenhao and He, Sicheng and Liu, Jie and Liu, Xiawei and Kang, Kaidi and others},
  journal={arXiv preprint arXiv:2603.12263},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0.

See the [LICENSE](LICENSE) file for details.
