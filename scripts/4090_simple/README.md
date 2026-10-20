# 4090 独立 SIMPLE 环境工具

遵守 [交接文档](../../docs/4090_SIMPLE_HANDOFF.md)。本目录仅提供环境约束、
本机路径配置与不依赖权重的冒烟检查；没有接通生产双来源服务或控制策略。
实际安装／验证记录在本机仓库外 `~/CVPR_Evaluation_runtime/reports/`。

## Conda 与依赖

本机创建了两个新环境，未复用用户其他机器人环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -n cvpr-simple-model311 python=3.11 pip setuptools wheel
conda create -n cvpr-simple-sim310 python=3.10 pip setuptools wheel
```

Conda 管理解释器和环境；安装 Python 包时使用目标环境内的 pip，或
`uv pip --python /目标conda环境/bin/python`，不使用 `uv sync` 创建第三套 Python 环境。
所有安装都传对应的 `constraints-*.txt`，避免本地包升级 Torch／NumPy。
两个新环境的 Conda 配置已保存本机缓存／资源变量和 `PYTHONNOUSERSITE=1`；
后者防止 `~/.local/lib/python3.10/site-packages` 中的旧机器人包污染新环境。
直接调用绝对路径 Python 时也须加载下述配置，或显式传此变量。

模型侧先从 `https://download.pytorch.org/whl/cu126` 安装 Torch 2.7.0、
Torchvision 0.22.0，再安装 `VLMExpert/pyproject.toml` 的 **project dependencies、
psi 和 serve dependency-groups**。pip 新版本可用 `--group /绝对路径/pyproject.toml:psi`
和 `:serve`；普通 editable 安装不会启用这些组。另安装：

- `ImageProcess`、`MotionExpert`、`pipeline[dino]`、`VLMExpert` 本地 editable 包。
- Diffusers 0.37.0；LeRobot 固定提交
  `09929d8057b044b53aecaf5c6d7eb71f99e8beb9`（本机源码在仓库外 runtime/lerobot）。
- FlashAttention 2.7.4.post1 的官方 Python3.11／Torch2.7／CUDA12／C++11 ABI TRUE wheel；
  先查询 `torch._C._GLIBCXX_USE_CXX11_ABI`，不能跨环境照搬扩展。
  不启用 DeepSpeed 预编译训练算子；本任务不启动训练。

仿真侧先从 `https://download.pytorch.org/whl/cu128` 安装同版本 Torch／Torchvision，再安装：

- 本地 `SIMPLE`，含其声明的 Isaac Sim `[all,extscache]==4.5.0`、MuJoCo 3.3.6；
  Isaac wheel 的附加索引为 `https://pypi.nvidia.com`。
- 本地 `SIMPLE/third_party/openpi-client`、`gear_sonic[sim]`、`decoupled_wbc`、
  `unitree_sdk2_python`；显式使用本地路径，pip 不读取 `tool.uv.sources`。
- ONNX Runtime、pyzmq、msgpack、requests、huggingface-hub、Pillow、PyYAML、tyro；
  pin 及其 cmeel 依赖版本见约束文件。无需安装 `[full]`、XRobo 遥操作、所有基线。
- 当前 SIMPLE 实际机器人／渲染路径还强制依赖固定快照的 CuRobo。
  将 `SIMPLE/third_party/curobo` 复制到仓库外 build 目录，设置 `CUDA_HOME`、
  `TORCH_CUDA_ARCH_LIST=8.9`、`MAX_JOBS=2`，在仿真环境中构建；不可使用脚本默认的 SM80。
  展开源码没有自身 `.git`，本机显式设置
  `SETUPTOOLS_SCM_PRETEND_VERSION_FOR_NVIDIA_CUROBO=0.0.0+cvpr.5810bc9` 以生成本地包 metadata；
  这是构建标签，源码身份仍绑定交接文档记录的完整上游提交。
  Warp 固定 1.7.0。构建与实际导入结果以本机报告为准。

`pip freeze`、`conda env export`、`conda list --explicit` 的本机快照在 reports 中；
约束文件不是全量锁文件。宿主驱动不因安装不同 CUDA runtime 而升级。

## 本机路径与冒烟检查

在仓库根目录运行。激活环境后加载本机配置：

```bash
source scripts/4090_simple/local.env.example.sh
mkdir -p "$CVPR_SIMPLE_RUNTIME/reports"
python -m pip check
```

`SIMPLE_DATA_DIR` 现在可覆盖 SIMPLE 数据根目录；为空时保持旧默认。
其余 `CVPR_*` 变量是部署约定，**现有生产模型服务不会自动读取这些变量或补齐接口**。
资源包到达后仍需重映射 run_config 中的路径，严格加载与校验。

模型环境：

```bash
python scripts/4090_simple/probe.py model --output "$CVPR_SIMPLE_RUNTIME/reports"
python scripts/4090_simple/resources.py --root "$CVPR_SIMPLE_RUNTIME/resources" \
  --output "$CVPR_SIMPLE_RUNTIME/reports/resources.json"
```

仿真环境，GPU 检查串行执行：

```bash
python scripts/4090_simple/make_isaac_experience.py --output "$SIMPLE_ISAAC_EXPERIENCE"
python scripts/4090_simple/probe.py mujoco --output "$CVPR_SIMPLE_RUNTIME/reports"
python scripts/4090_simple/probe.py curobo --output "$CVPR_SIMPLE_RUNTIME/reports"
python scripts/4090_simple/probe.py isaac --output "$CVPR_SIMPLE_RUNTIME/reports"
```

Isaac 首次启动按安装器提示接受许可；自动化启动可在已授权安装时设置
`OMNI_KIT_ACCEPT_EULA=YES`。Isaac probe 先创建 SimulationApp，再导入 Omniverse APIs，
使用本机几何体和 RTX 相机，不以无窗口启动代替真实 RGB 检查。
本次环境通过 `SIMPLE_ISAAC_PRELOAD_PINOCCHIO=1` 显式预加载 Pinocchio，避免 Kit 的旧
Assimp 导致随后导入 SIMPLE／G1 时出现 HPP-FCL 未定义符号。默认 Kit 的 Lula 又要求旧
URDF ABI，因此同时使用生成的专用 experience：从已安装4.5配置保留 core／physics／RTX／
camera／Replicator 等依赖，仅移除11个引入 Lula 的 Cortex／旧机械臂示例根依赖。
生成文件及来源 SHA／排除清单位于外部 runtime/apps；该配置用于 SIMPLE 的 CuRobo／SONIC 路径，
不提供 Lula。未设置本次 opt-in 变量时，共享 helper 保持通用默认行为。
Omniverse APIs 仍在 app 创建后导入，检查完成还须审计整个启动日志中的 Error／Traceback。
CuRobo probe 使用其自带 UR5e，在 batch1 的两个姿态上对照 CPU Pinocchio 检查 CUDA FK，
不验收 G1 控制策略、碰撞查询或 IK。Isaac 结果在 Kit 关闭前落盘，须同时查看 JSON 状态和日志。
probe 的 JSON 记录 running／failed／passed；图片、日志和 JSON 都写入外部 reports。

跨环境通信：模型环境先运行下述 server，仿真环境再运行 client：

```bash
python scripts/4090_simple/transport_probe.py server --port "$CVPR_MODEL_PORT"
python scripts/4090_simple/transport_probe.py client --port "$CVPR_MODEL_PORT" \
  --image "$CVPR_SIMPLE_RUNTIME/reports/mujoco-rgb.png"
```

server 只绑定 `127.0.0.1`，复用 Psi0 的 NumPy wire helpers；client 使用 SIMPLE 的
`HttpActionClient`。检查真实相机图像往返、两 session 隔离及 reset。
返回的 `30×80` 数组是计数标记，**不是模型预测，严禁用于控制机器人**。
此 fixture 没有实现真实历史、在线 DINO、RTC 或生产服务的 session 生命周期。
检查后停止 fixture server，不将它作为正式推理服务保留。

## 尚须验收

固定 20k 联合 checkpoint、构造配置、统计、Qwen 小文件、DINO 身份及匹配的
`body_token_v1_1` SONIC decoder／controller、SIMPLE 任务／USD／纹理／初始条件全部到齐后，
才继续真实六帧历史、43→45 state、30×80 action／反归一化、单 episode 闭环、
多 episode reset，以及模型／仿真各自与同时运行的显存和延迟测量。
不得以默认官方权重、旧 SONIC 或额外 SIMPLE 微调替代 UnifoLM-20k 零样本测评。
