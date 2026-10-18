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

`VLMExpert` 于2026-10-18整体替换为 [官方Psi0](https://github.com/physical-superintelligence-lab/Psi0)
`main` 的固定版本 `cda3fdfafe47cb6fa75400d013de1bd36e3cea05`，Python包名仍为 `psi`。
初始快照除移除上游遗留的一行凭据注释外保持该版本内容；此前GLOBAL token／WBT定制及其专用入口已退役，
可从替换前的Git提交恢复。当前在该快照上增加了 VLM tokens 的 masked mean pooling 接口，以及
`dual_source_v1` 双来源 action expert。新模式支持联合训练 VLM／Motion 与新 action expert，使用最后层 VLM tokens 和完整11个 motion hidden，
由两路池化摘要及 action 时间调制串行 SA／CA-VLM／CA-Motion／FFN；旧 `psi0_joint` 路径保留。
模块README说明了本地接口，后附官方说明。新模式接收真实历史 DINO 特征；UnifoLM 已新增原生30Hz
提取器与 LeRobot 历史读取，数据位于 `/gaozt-test1/datasets/UnifoLM`，缓存以episode保存mmap。
DINO 与当前 VLM RGB 共用 `data.shared_image_profile=256x320`，绑定完整配置、NumPy/Pillow版本与权重身份；
迁移时源路径或文件身份改变会拒绝旧缓存，需按新路径重新验收。现有服务入口的历史缓冲仍需接线。
不能直接把单帧官方部署命令当作双来源入口；数据准备和训练是分开的命令。
完整恢复仍需原 Motion checkpoint／统计来重建其协议，联合更新后的 Motion 网络权重同时随完整模型保存。
新任务使用模块 README 中的双来源 UnifoLM 配方，不恢复冻结阶段的 optimizer；两套 expert 各自也可显式冻结。
版本、展开依赖和排除文件记录见 [psi0_upstream.json](psi0_upstream.json)。
`docs/inventory` 仍是初次迁移的历史快照，不用于核对本次更新。

### 独立安装

新版要求 **Python 3.11**、PyTorch 2.7.0、Transformers 4.57.1，LeRobot固定到
`songlin/lerobot@09929d8057b044b53aecaf5c6d7eb71f99e8beb9`；完整版本以
`VLMExpert/pyproject.toml` 和 `uv.lock` 为准。与MotionExpert、DINO提取环境分开安装，
不要在正在运行的训练环境中升级依赖。

```bash
cd VLMExpert
uv venv .venv-psi --python 3.11
source .venv-psi/bin/activate
GIT_LFS_SKIP_SMUDGE=1 uv sync --group serve --group viz --group psi \
  --index-strategy unsafe-best-match --active
# CUDA环境按官方说明另装Flash Attention：
uv pip install flash_attn==2.7.4.post1 --no-build-isolation
cp .env.sample .env
```

`.env` 中的 `PSI_HOME`、数据、缓存和日志路径按本机设置，数据、权重、checkpoint和运行环境放在仓库外。
CUDA、XPU、Docker、训练和部署的具体要求见 [官方README](../VLMExpert/README.md)。
本次只同步源代码，没有安装或替换任何现有环境。

官方README仍有引用已移除脚本的旧示例；启动前以实际 `scripts/train/psi0/` 脚本和
`pyproject.toml` 注册入口为准。当前 `serve_psi0_simple` 仍硬性检查CUDA，未验收其XPU路径。

### 展开的仿真与控制依赖

本仓库继续将 `third_party/SIMPLE`、`third_party/GR00T-WholeBodyControl` 及嵌套依赖展开为普通文件，
按官方Psi0固定的子模块commit同步，而非跟随各依赖的最新分支。
`VLMExpert/.gitmodules` 等上游文件保留来源信息，但当前Git仓库没有相应gitlink，
因此不要使用官方README中的 `git submodule update --init --recursive` 补齐本仓库依赖。
SIMPLE、cuRobo、Isaac Sim、机器人SDK等运行环境仍需按相应上游说明另行安装。

模型权重和数据不随源码同步；例如AMO、SONIC、EgoVLA网络与checkpoint须从官方文档指定位置准备。
大体积演示视频、Isaac USD／USDA以及本仓无法复用的新增仿真二进制资源亦不随本次源码同步，
具体排除路径见版本记录，使用相应场景或机器人模型前从固定上游版本另行准备。
展开依赖内的 `.gitattributes` 不复制，由根仓库统一管理Git LFS资源。
保留原有必要的归一化及人体模型资源，不把训练权重重新提交到Git。
仿真、真机部署及模型训练尚需在专用环境中验收；源码一致性与CPU测试不代表已通过这些运行验收。
