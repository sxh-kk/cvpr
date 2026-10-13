# MotionExpert / VLMExpert 上传与迁移文档

清点日期：2026-10-06。目标仓库：<https://github.com/sxh-kk/cvpr>，公开，默认分支 `main`。

## 1. 本次迁移范围

两个项目保留原来的目录名称与包布局。上传副本独立于原工作目录，原代码和本地权重继续保留。

```text
cvpr/
├── MotionExpert/             # motionexpert 包，训练 / 推理 / GLOBAL
├── VLMExpert/                # psi 包，VLM / WBT / baselines / real
│   └── third_party/          # 展开的第三方源码和静态资源快照
├── docs/
│   ├── MIGRATION.md
│   └── inventory/
│       ├── summary.json
│       ├── files.csv
│       └── upstream_dependencies.json
└── scripts/verify_inventory.py
```

| 项目 | 项目文件数 | 项目文件原始字节数 | 主要内容 |
| --- | ---: | ---: | --- |
| MotionExpert | 51 | 364,412 | 训练、推理、EMA、Flow Matching、模型、数据读取、两份 YAML、测试、统计资源和项目文档 |
| VLMExpert | 5,926 | 约 2.45 GB | PSI0 / GLOBAL / WBT、本地 baselines、部署与 teleop、FAST tokenizer、统计资源、机器人 assets 和第三方依赖 |

表中的字节数以原项目快照为参考；最终发布字节数、SHA-256 和修改记录以清单为准。
仓库新增的 README、迁移文档、清点报告及校验脚本不计入这两个项目的文件数。
Git LFS 使用 368 个文件路径，内容以机器人网格、演示媒体、第三方原生库和安装包为主；checkpoint 不在其中。

### 上传内容

- 两个项目的 Python / C++ 源码、训练脚本、配置、测试、文档与依赖声明。
- MotionExpert 的 `assets/e7_normalization.pt` 和 `assets/smplx_representation.pt`，它们是统计 / 表示资源。
- VLMExpert 的 `assets/stats/` 和 `src/fast/egodex-rel-50w-1x48-v2048-s100/` tokenizer 配置与词表。
- 机器人 URDF、网格、SDK 资源、示例和第三方许可证；静态 SMPL 关节回归矩阵及 adapter 归一化统计继续保留。

### 排除内容

- 所有 `weights/`、`checkpoints/` 中的文件及 `.ckpt`、`.safetensors`、`.onnx` 等模型权重。
- `.pt` / `.pth` 模型与离线张量产物，明确识别的统计和静态表示资源除外。
- MotionExpert 的 `weights/e7_k12_400m_text_ema_fp32.pt`（1,630,872,294 字节）。
- SONIC / AMO / adapter 等控制模型及 EgoVLA 的手部网络权重。
- `.env`、私有机器配置、虚拟环境、缓存、`*.egg-info`、构建产物、日志及实验输出。
- 原 checkout 的 `.git` 指针、`.gitmodules` 和嵌套 `.gitattributes`；第三方内容已展开，仓库采用统一的 Git LFS 规则。
- 工作目录外的数据集、HF / Torch 缓存和另行训练的模型目录，它们不属于这两个项目的上传副本。

`files.csv` 的 `uploaded` / `excluded` 状态和 `reason` 列可用于逐项追溯。被排除项仅记录路径、类别与大小，不记录文件内容。
上传文件记录最终 `sha256` 与原始 `source_sha256`；公开发布修改列为 `publication_change`。

公开副本移除了 `VLMExpert/src/psi/utils/utils.py` 中一条含 Hugging Face token 的注释，改为说明从 `HF_TOKEN` 环境变量读取认证。
两个项目的 README 增加了本仓库不提供权重的迁移说明。模型计算、训练流程及配置数值沿用原代码。

## 2. 获取仓库与校验

准备 Git 和 Git LFS 后执行：

```bash
git lfs install
git clone https://github.com/sxh-kk/cvpr.git
cd cvpr
git lfs pull
python scripts/verify_inventory.py
```

Git LFS 跟踪的是非 checkpoint 资源。GitHub 的 ZIP 下载可能不能直接提供全部资源实体；需要完整目录时使用 Git clone 与 `git lfs pull`。
代码浏览可跳过 LFS 下载；机器人资产或媒体的完整验证需要补齐 LFS。
校验脚本检查清单中的文件大小、SHA-256、符号链接与排除项是否意外出现在快照中。

## 3. MotionExpert 环境与路径

MotionExpert 使用 Python >= 3.10，`requirements.txt` 固定 Transformers 4.53.2，并要求 `huggingface-hub >= 0.30.0, < 1.0`。
建议为它建立独立环境；GPU 训练需按机器 CUDA 环境选择 PyTorch。

```bash
cd MotionExpert
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m unittest tests.test_pipeline tests.test_global -v
```

本包安装后的入口为 `motionexpert-train`、`motionexpert-infer`、`motionexpert-export`、`motionexpert-text` 和 `motionexpert-check-data`。
源码测试在 `MotionExpert/` 内运行，避免 `tests.fixtures` 被其他项目的 `tests` 包遮蔽。

### 数据准备

默认数据路径为项目相对路径 `../datasets/ee4d_motion_uniegomotion`，迁移后等价于 `cvpr/datasets/ee4d_motion_uniegomotion`。
也可使用 `MOTIONEXPERT_DATA_DIR` 或 `--data-dir` 指向独立的数据盘。

每个 split 需要以下离线文件；验证同样需要自己的 motion / image / text / QC 文件，归一化统计使用训练集统计：

| 相对数据根目录的文件 | 用途 |
| --- | --- |
| `uniegomotion/ee_{split}.pt` | EE4D motion pack |
| `uniegomotion/nymeria_{split}.pt` | NymeriaPlus motion pack |
| `uniegomotion/egoview_dinov2_{split}.pt` | EE4D DINOv2 特征 |
| `uniegomotion/egoview_dinov2_nymeria_{split}.pt` | NymeriaPlus DINOv2 特征 |
| `uniegomotion/clip_text_{split}_16.pt` | 16-token CLIP 文本特征 |
| `uniegomotion/motion_qc_{split}.json` | QC 序列清单 |
| `uniegomotion/v4_beta_joint_masked_train_stats_clean.pt` | 训练集联合统计 |

```bash
export MOTIONEXPERT_DATA_DIR=/data/ee4d_motion_uniegomotion
motionexpert-check-data --config configs/e7_k12_400m_global_text.yaml
```

数据不在仓库内。缺少 CLIP sidecar 或统计时，使用原 README 中的 `motionexpert-text`、`compute_joint_motion_stats`、`build_motion_qc_manifest` 等流程重建。

### 训练与外部权重

GLOBAL 配置默认随机初始化；数据备齐后可使用：

```bash
motionexpert-train --config configs/e7_k12_400m_global_text.yaml \
  --from-scratch --devices 8 --ddp-socket
```

实际卡数由 `--devices` 调整。`--ddp-socket` 是原工程的单机通信兼容选项，正常通信环境可省略。

需要继续旧 E7 训练时，另行准备 `e7_k12_400m_text_ema_fp32.pt`，放到 `MotionExpert/weights/` 或使用 `--init-weights` 指定路径。
旧 E7 权重不含 GLOBAL token，不能当作 GLOBAL 检查点完整恢复。
同阶段恢复使用完整 `.ckpt`；推理与特征提取使用明确的外部权重路径：

```bash
motionexpert-train --resume /models/motion-global/last.ckpt --ddp-socket
motionexpert-export /models/motion-global/last.ckpt /models/motion-global/export.pt
motionexpert-infer --input /data/conditioning.pt \
  --weights /models/motion-global/export.pt --features-only --output /data/features.pt
```

`one_step_hidden` 为 `[B,T,768]`，GLOBAL 版本还输出 `[B,768]` 的 `one_step_global`。
配套统计优先随训练检查点 / 导出文件迁移；不要把另一套数据统计与已有权重随意混用。

## 4. VLMExpert 环境与路径

目录名称是 `VLMExpert`，安装包和导入名仍为 `psi`。它依赖 Python 3.10、PyTorch 2.7.0、Transformers 4.57.0 等，完整声明见 `pyproject.toml` 与 `uv.lock`。
与 MotionExpert 分开安装，避免不同 Transformers 版本互相覆盖。

按原工程使用 uv：

```bash
cd VLMExpert
uv venv .venv-psi --python 3.10
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --group psi \
  --index-strategy unsafe-best-match --active
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
cp .env.sample .env
```

`flash_attn` 的安装需要兼容 CUDA / 编译环境。CPU 单元测试与代码中的 SDPA 提取路径不需要执行 Flash Attention GPU 内核。
`.env` 根据机器编辑；原训练入口要求它能够被 `load_dotenv()` 读取。

| 变量 | 迁移配置 |
| --- | --- |
| `PSI_HOME` | 本机独立的缓存 / 数据 / 模型根目录，例如 `/data/psi` |
| `DATA_HOME`、`HF_LEROBOT_HOME` | 数据与 LeRobot 数据根目录 |
| `HF_HOME`、`TORCH_HOME`、`UV_CACHE_DIR` | 模型与依赖缓存目录 |
| `HF_TOKEN` | 按模型 / 数据访问需求设置，保存在本机环境中 |
| `WANDB_API_KEY`、`WANDB_ENTITY` | 按实际日志账号设置；默认 GLOBAL 启动脚本使用 wandb |

`.env.sample` 中的 `/hfm` 是示例值，迁移时应替换。只复制 `.env.sample`；本次上传没有包含原 `.env`。

### GLOBAL 预训练

外部准备 EgoDex、HE 数据，以及 `Qwen/Qwen3-VL-2B-Instruct` 模型。仓库内保留 FAST tokenizer 的词表与处理配置。
两个阶段分别进行约 200K / 30K optimizer steps，正式启动前确认数据根目录、GPU 数、有效 batch size 与日志配置。

```bash
export PSI_PYTHON="$PWD/.venv-psi/bin/python"
export CUDA_VISIBLE_DEVICES=0,1
export EGODEX_DATA_ROOT=/data/egodex
export HE_DATA_ROOT=/data/HE_RAW
export QWEN_CHECKPOINT=/models/Qwen3-VL-2B-Instruct
export FAST_CHECKPOINT="$PWD/src/fast/egodex-rel-50w-1x48-v2048-s100"

bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh
```

EgoDex 完成后，将实际产物导出到外部目录，再启动 HE：

```bash
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/save_pretrain_qwen3vl_backbone.py /models/egodex-run \
  --output /models/egodex-global-export --dtype bfloat16
bash scripts/train/psi0/pretrain-he-psi0-fast.sh /models/egodex-global-export
```

HE 是新阶段，加载 EgoDex 模型 / processor 后重新建立优化器、scheduler 和阶段步数；同阶段 resume 才恢复完整训练状态。
模型、processor、tokenizer、特殊 token 配置与 `pretraining_metadata.json` 必须来自同一份产物。
checkpoint 不在此仓库内，文档中的 `/models/...` 均由使用者另行提供。

### 下游特征接口

在 VLMExpert 环境中可按以下方式加载另行提供的 GLOBAL backbone：

```python
from PIL import Image
import torch
from psi.models.global_vlm import load_global_backbone, extract_global_features

model, processor = load_global_backbone(
    "/models/he-global-export", device="cuda:0", dtype=torch.bfloat16
)
features = extract_global_features(
    model, processor, [[Image.open("/data/observation.jpg").convert("RGB")]],
    ["Pick up the object."]
)
global_hidden = features.global_hidden
ordinary_mask = features.ordinary_mask
```

输入不含真实动作答案；GLOBAL 特征通过一次前向得到。详细输入布局、mask、保存恢复与表示定义见 `VLMExpert/project.md`。
独立评估入口是 `scripts/eval_global_vlm.py`，其输入 JSONL 需要预处理图像、instruction 和归一化动作。

### WBT / 仿真 / 机器人依赖

WBT 使用 `src/psi/config/train/posttrain_wbt_psi0_config.py`，默认数据根目录 `/hfm/data/WBT` 应改成实际路径。
数据和动作协议说明见 `dataset_WBT.md`；仓库保留 `assets/stats/wbt_native_stats.json`。

`third_party/SIMPLE`、`third_party/GR00T-WholeBodyControl` 与其中的依赖源码均已展开。
此快照不需要执行 `git submodule update --init --recursive` 来补齐这些源码；上游仓库 URL 记录在 `upstream_dependencies.json`。
原 checkout 缺少可用 Git 元数据，因此本次不声称这些目录对应某个已验证的上游 commit；当前仓库 commit 与文件 SHA-256 定义这份快照。

需要 SIMPLE 仿真时，继续按原 README / `examples/quick_start/simple.md` 安装 SIMPLE、cuRobo、Isaac Sim 等运行依赖。
源码与 SDK 资源不等于已安装的运行环境。AMO、SONIC、adapter、EgoVLA 网络等权重均须从原项目文档指向的位置另行准备，按原相对路径放回。
第三方源码的本地新增文件可能被其 `.gitignore` 忽略，提交前可用 `git check-ignore` 检查；明确需要的新增源码可逐文件添加。

## 5. 清点与验证结果

验证日期：2026-10-06，使用现有兼容环境，在 CPU 上执行项目已有回归测试：

| 验证 | 结果 | 内容 |
| --- | --- | --- |
| MotionExpert | 18 / 18 通过 | 两种数据来源、对齐、mask、归一化、GLOBAL / EMA、特征与采样一致性 |
| VLMExpert | 26 / 26 通过 | GLOBAL 梯度、causal 前缀、词表、保存 / 导出 / HE 衔接和 WBT 协议 |
| 发布副本完整性 | 以 `verify_inventory.py` 输出为准 | 上传文件大小、SHA-256、符号链接、排除项、checkpoint 边界 |

MotionExpert：

```bash
cd MotionExpert
source .venv/bin/activate
python -m unittest tests.test_pipeline tests.test_global -v
```

VLMExpert：

```bash
cd VLMExpert
source .venv-psi/bin/activate
PYTHONPATH=src OMP_NUM_THREADS=2 python -m unittest discover -s tests -p 'test_*.py' -v
```

这里的 unittest discovery 执行 GLOBAL / WBT 测试类；`test_simple_datagen_curobo_missing.py` 的 pytest 风格函数不属于这 26 项。
迁移验证未执行完整数据集训练、GPU 多卡训练、SIMPLE 仿真或机器人部署，也未验证被排除的外部 checkpoint。
原根 Python 环境曾因 Transformers 与 huggingface-hub 版本不匹配无法导入；本次使用两个项目对应的兼容环境完成了测试，没有更改共享环境。

## 6. 上传后再次迁移

1. Git clone，并补齐 Git LFS 的非 checkpoint 资源。
2. 执行 `python scripts/verify_inventory.py`，确认快照完整。
3. 为 MotionExpert / VLMExpert 分别建立 Python 环境。
4. 准备训练数据、模型缓存、实际 `.env` 和外部权重，修改数据 / 输出路径。
5. 执行两组回归测试，再按项目说明进行数据检查、训练或特征提取。

后续产生的 checkpoint、日志和私有配置应保留在本机或另行管理；仓库根 `.gitignore` 已加入这些排除规则。
第三方文件继续遵循各自的许可与版权声明，VLMExpert 的原 PSI0 来源与引用信息保留在原 README 中。
