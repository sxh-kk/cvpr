# 环境与数据迁移

MotionExpert、pipeline和ImageProcess的当前接口以各自README为准。
旧运动实验、旧dense归档、旧视觉特征和旧Motion checkpoint已清理。
Git仓库不包含训练数据、模型权重或运行环境。

```bash
git lfs install
git clone https://github.com/sxh-kk/cvpr.git
cd cvpr
git lfs pull
```

MotionExpert和DINO提取使用独立环境，分别安装所需GPU版PyTorch，再安装项目：

```bash
python -m pip install -e ./ImageProcess
python -m pip install -e './pipeline[dino]'
# 在训练环境中：
python -m pip install -e ./ImageProcess
python -m pip install -e ./MotionExpert
```

迁移运动归档时需完整复制 `smplh-parameters-v1` 的motion、records、receipt、metadata、
自有模型、验收证据和旧vision退休清单，并单独复制 `statistics/smplh_k12_5hz_joint.pt`。
DINO提取还需原始MCAP／MP4及DINO预训练权重，修改record中的原视频路径须同步重建来源绑定。
新视觉缓存提取后与数据共同迁移。训练配置中的归档、分割、统计、视觉和输出路径须指向新位置。

当前数据／提取格式见 [pipeline](../pipeline/README.md)，
从头训练与一步hidden接口见 [MotionExpert](../MotionExpert/README.md)，
两路同步RGB处理见 [ImageProcess](../ImageProcess/README.md)。

仓库 `docs/inventory` 与 `scripts/verify_inventory.py` 是2026-10-06初次迁移快照，
不作为当前MotionExpert文件或训练协议的验证入口。第三方许可与VLM资产来源继续保留。

## VLMExpert 环境与路径

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
