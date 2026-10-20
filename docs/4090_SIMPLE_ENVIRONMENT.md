# 4090 独立 SIMPLE：本机环境记录

日期：2026-10-20，北京时间。仓库位置 `/home/zxc/CVPR_Evaluation`，初始提交
`50c54253bf62678e10952acb3ddd6dc0f757a5c5`；首次克隆到空目录，439 个 LFS 文件的
`git lfs fsck` 通过。安装约束、可复现命令和检查脚本见
[环境工具](../scripts/4090_simple/README.md)，需求与协议仍以
[交接文档](4090_SIMPLE_HANDOFF.md)为准。

**当前结论：两套独立 Conda 环境的基础冒烟检查通过；20k 模型推理、G1 控制闭环及正式
SIMPLE 测评均未验收。** 本次没有登录或修改 A100，没有启动训练、微调或正式测评。

## 本机与环境

- Ubuntu 22.04.5 LTS，x86_64，内核 `6.8.0-138-generic`；系统内存约 187 GiB。
- NVIDIA 驱动 `570.211.01`；`nvidia-smi` 报告 RTX 4090、49,140 MiB 显存，
  Torch 报告 SM89、device total memory 50,851,282,944 bytes。
  这台机器报告的容量不同于交接文档的 24 GB 假设；Vulkan 另报告 81,898 MB，
  各 API 的容量读数不一致，不能据此宣称完整模型与渲染能同时装入显存。
- 驱动显示的最高 CUDA 版本为 12.8；现有编译工具链实际是 `/usr/local/cuda-12.0`。
  没有升级驱动、系统 CUDA 或系统 Python。
- 使用现有 Conda 25.7.0 创建 `cvpr-simple-model311` 与 `cvpr-simple-sim310`；
  路径分别为 `/home/zxc/miniconda3/envs/` 下的同名目录。包通过明确指定这些解释器的
  `uv pip` 安装，Conda 管理环境与 Python；没有创建替代 venv。
- 缓存、构建源码、权重目录、日志及图片位于仓库外
  `/home/zxc/CVPR_Evaluation_runtime/`。本次安装后 `/home` 仍约有 98 GiB 可用，
  尚未计入待迁移权重与任务资产。
- 原有 `sonic_isaaclab` 等环境未修改；现有 `/home/zxc/isaacsim` 为 6.0.1 预发布版本，
  本任务独立安装 4.5，不使用该旧目录。

模型环境实际主要版本：

```text
Python 3.11.17
torch 2.7.0+cu126 / torchvision 0.22.0+cu126 / triton 3.3.0
transformers 4.57.1 / accelerate 1.7.0 / deepspeed 0.17.1
flash-attn 2.7.4.post1 / qwen-vl-utils 0.0.14 / diffusers 0.37.0
numpy 1.26.4 / pillow 11.3.0 / einops 0.8.1 / omegaconf 2.3.0
tyro 0.9.32 / safetensors 0.6.2 / lerobot 0.3.3
fastapi 0.152.0 / uvicorn 0.54.0
```

安装了 Psi0 的 project dependencies、`psi` 和 `serve` 两个依赖组，以及当前仓库
`ImageProcess`、`MotionExpert`、`pipeline[dino]` 的本地包。LeRobot 源码固定到
`09929d8057b044b53aecaf5c6d7eb71f99e8beb9`，保存在 runtime/lerobot。
FlashAttention 使用官方匹配 Python3.11／Torch2.7／CUDA12／C++11 ABI TRUE 的 wheel，
其 SHA256 为 `22013b8c74a63fc70e69be1e10ff02e4ad8fec84a43600bdca67b434ed417113`。

仿真环境实际主要版本：

```text
Python 3.10.22
torch 2.7.0+cu128 / torchvision 0.22.0+cu128
isaacsim 4.5.0.0 / extscache-kit 4.5.0.0 / extscache-physics 4.5.0.0
mujoco 3.3.6 / dm-control 1.0.21 / numpy 1.26.4 / scipy 1.15.3
gear-sonic 0.1.0 / decoupled-wbc 0.1.0 / unitree-sdk2py 1.0.1
cyclonedds 0.10.2 / pin 2.7.0 / onnxruntime 1.23.2 / warp-lang 1.7.0
nvidia-curobo 0.0.0+cvpr.5810bc9 / tyro 1.0.16
```

SIMPLE、openpi-client、gear_sonic、decoupled_wbc、Unitree SDK 使用仓库固定展开源码。
没有安装遥操作全家桶、所有其他基线或全部训练依赖。CuRobo 构建源码复制到
runtime/build/curobo，使用宿主 CUDA12.0、Torch cu128、`TORCH_CUDA_ARCH_LIST=8.9`、
`MAX_JOBS=2` 编译。展开源码没有自己的 Git metadata，显式设置 SCM 本地构建标签；
该标签不冒充上游正式版本，真实源码提交为
`5810bc9cb69129a0e1a8aad47704bd6da35370c0`。
CuRobo 与 LeRobot 的外部 editable 源码是环境的一部分，不可作为临时文件删除。

## 已实际验证

1. 两个环境最终 `python -m pip check` 均为 `No broken requirements found.`。
   Conda 激活后解释器版本、环境变量和禁用 user-site 均检查通过；Psi0、MotionExpert、
   ImageProcess、pipeline 均导入当前 checkout，未落入同名旧包。
2. 两个 CUDA runtime 均能使用本机 SM89 GPU，BF16 矩阵算子有限。
   模型环境 FlashAttention 在无梯度 BF16、`[1,128,8,64]` 输入上执行真实前向，
   对照 float32 SDPA 最大绝对误差 `0.0037746429`。没有加载预训练 DINO／VLM／联合权重。
3. MuJoCo 原生场景执行 100 次 step、累计 0.2 秒仿真时间，下落位置有限且符合重力方向；
   EGL 相机输出 `240×320×3 uint8` RGB，标准差约 21.64，并已查看真实渲染图片。
4. gear_sonic 和 decoupled_wbc 的本地 G1 URDF／mesh 可构造，均为 43 关节，
   中立姿态 Pinocchio FK 有限。SDK／控制配置／策略工厂模块导入通过，未创建 DDS channel。
5. CuRobo 五个编译扩展均可导入。进一步在自带 UR5e 的两个姿态、batch1 上执行真实 CUDA FK，
   与 CPU Pinocchio 末端位置对照，最大误差 `1.6718e-7 m`。未验证 G1 IK、碰撞查询或控制策略。
6. Isaac Sim4.5 headless 实际启动，先创建 SimulationApp，再导入 Omniverse 和当前
   `simple.engines.isaacsim`，执行 90 次 render step；RTX 相机输出 `240×320×3` RGB，
   标准差约 44.56，已查看图片。几何体为本机 USD primitives，没有加载 SIMPLE 任务资产。
   首次初始化约 186 秒；缓存后含实际 SIMPLE 引擎导入的整次冒烟约 15 秒。
   最终使用下述专用 experience，完整启动日志无 Error／Traceback。
   这些是环境检查耗时，不是策略推理延迟或评测吞吐；未清理其他占卡进程以制造独占条件。
7. Python3.11 FastAPI fixture 绑定 `127.0.0.1:18765`，复用 Psi0 NumPy wire helpers；
   Python3.10 使用 SIMPLE `HttpActionClient`，以真实 MuJoCo RGB 验证请求／响应、
   图像无损往返、两个 session 隔离和 reset。`30×80` 返回值为人工计数标记，
   不是 ActionExpert 输出；生产历史缓冲、在线 DINO 和 RTC 没有被该测试验收。
   测试服务已停止，18765 端口已释放。

## 安装过程中发现并处理的问题

- Python3.10 默认 user-site 包污染新 Conda 环境，导致旧 Gymnasium、JAX 等包进入
  `pip check`。在这两个新环境保存 `PYTHONNOUSERSITE=1` 后检查通过；没有卸载或修改原用户包。
- SIMPLE 依赖文件未启用 CuRobo，但机器人及渲染源码强制导入它。本次从固定源码补装，
  而不是将 `import simple` 成功当成完整依赖通过。
- Kit 先启动后才导入 Pinocchio 时，HPP-FCL 报 Assimp C++11 符号未定义。
  预加载 Pinocchio 后，默认 Kit 的 Lula 又因旧 URDF ABI 出现扩展启动错误，
  因此不能仅凭当时的 RGB 成功就将默认 Kit 判为通过。
  本次通过 `SIMPLE_ISAAC_PRELOAD_PINOCCHIO=1` 显式预加载，同时生成专用
  `runtime/apps/simple4090.kit`，从已安装4.5 base experience 仅移除11个会带入
  Lula 的 Cortex／旧机械臂示例根依赖，保留 SIMPLE 使用的 core／physics／RTX／camera 等。
  来源 SHA、排除清单见 `reports/isaac-experience.json`；路径已保存到本机 Conda 配置。
  当前 SIMPLE 的 CuRobo 路径不导入 Lula，不宣称这两组 ABI 已能通用混用。
  未设置本次 opt-in 时保持共享 helper 的原默认行为。
  已移除 probe 自身的 Pinocchio 预加载，用实际共享 helper、专用 experience 重测
  引擎和 RGB，并审计完整日志无 Error／Traceback 后才通过。
- SIMPLE 新增 `SIMPLE_DATA_DIR` 覆盖，支持仓库外本机数据目录，原默认路径和空白配置
  回退行为已检查。模型与仿真相关缓存路径也已保存到新环境的 Conda 配置。
- Kit 原生 shutdown 可能在返回 Python 前结束进程。probe 改为先保存 JSON，再在
  interpreter cleanup 关闭 app；失败也写入 JSON。应同时看 JSON status 和日志，
  不把单独的退出码0当作成功。关闭流程本身未作独立生命周期验收。

## 缺失资源与尚未解决的接口

当前本机资源清单中的九类路径均缺少有效载荷，见外部 `reports/resources.json`：
固定20k联合 `model.safetensors`、run_config、argv、dataset_statistics、Qwen 小文件目录、
Motion 构造／归一化目录、匹配的 DINO 目录、`body_token_v1_1` SONIC 目录、SIMPLE 数据目录。
原有仓库 URDF／mesh 不等于完整任务／USD／纹理／初始条件。
本次没有从 A100 搬运资源，也不能确认源端20k是否完成或受到快照保护。

现有生产代码仍有交接文档列出的可移植 Motion 构造、双来源 HTTP／WebSocket 真实历史和
在线 DINO 接线、SIMPLE 专用43→45 state及64 latent＋14手动作的语义映射缺口。
SONIC 包版本、默认文件名和64维 shape 不能证明 v1.1 decoder 身份。
不能用旧32维state／36维action客户端、默认官方模型或不匹配的SONIC替代这次20k测评。

因此尚未进行指定20k权重严格载入与SHA绑定、真实6帧历史推理、完整tokens／pooling、
动作反归一化、G1单episode闭环、多episode reset、任务成功率／视频统计，
也没有测量模型／仿真独立及共同运行的峰值显存和端到端延迟。
完整离线运行的 Isaac 任务资产／材质缓存亦待资源到齐后检查。

## 本机证据与后续入口

外部 `/home/zxc/CVPR_Evaluation_runtime/reports/` 保存：

- `hardware.txt`、`git-lfs-fsck.txt`、各安装和历史失败日志。
- `model.json`、`mujoco.json`、`curobo.json`、`isaac.json` 与各自 probe 日志和 RGB 图片。
- `model-pip-check.txt`、`sim-pip-check.txt`、`conda-activation-check.txt`。
- `sim-imports-minimal.log`、`sonic-urdf-fk.log`、`curobo-imports.log`、`transport-client.log`。
- 两环境的 `*-versions.json`、`*-pip-freeze.txt`、`*-conda-environment.yml`、
  `*-conda-explicit.txt`；完整版本以这些本机快照为准。
- `resources.json`，其中 presence/hash 只表示本机文件盘点，不表示加载与语义验收。

激活对应 Conda 环境后，按[工具说明](../scripts/4090_simple/README.md)加载
runtime/local.env.sh 或仓库的 example 配置。资源包到达后先严格核对身份、重映射本机路径，
再按交接文档继续单4090、batch1、单仿真实例的分阶段验收。
