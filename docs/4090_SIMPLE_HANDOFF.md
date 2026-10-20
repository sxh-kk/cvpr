# 4090 独立 SIMPLE 测评：需求与环境交接

本文件供 **4090 机器上的 Codex／开发者**读取并执行。它记录本次用户需求、已核实的代码边界、环境配置顺序和验收标准；不是“现有代码已经通过4090全流程验收”的声明。

## 1. 用户目标与当前任务

用户要求：A100继续训练当前UnifoLM双来源模型；使用**完成20,000次优化器更新时的联合checkpoint**，在4090机器独立完成SIMPLE闭环评测。

- DINO、VLM、MotionExpert、ActionExpert、SONIC控制器、MuJoCo物理仿真、Isaac Sim渲染、评测统计及视频输出全部位于4090机器。
- 模型服务和仿真客户端通过本机`127.0.0.1`通信。资源迁移完成后，推理、历史缓冲与仿真不依赖A100服务或A100文件系统。
- 先按**单张RTX 4090、batch1、单个仿真实例**配置和验收。模型与渲染共享24GB显存的实际峰值／延迟尚未测量；显存不足要如实报告，不能静默改成A100远端推理。
- 首次正式结果是**UnifoLM-20k向SIMPLE的零样本迁移评测**。额外SIMPLE微调会产生另一模型版本，不能替换本次20k结果。

**现在先执行4090本机环境配置**：检查硬件和已有环境，建立隔离环境，安装依赖并做分阶段冒烟验证，留下实际环境报告。权重未到时继续完成不依赖权重的工作。后续模型／控制接口接线按下述缺口推进，不能把“安装成功”报告成“模型测评成功”。

A100正在运行的训练、依赖、进程和参数保持原样。不要启动新训练、SIMPLE微调或大规模评测来代替环境验收；不要清理用户已有环境、权重或任务资源。安装、缓存、权重及日志放到仓库外，代码／必要配置／文档按根`AGENTS.md`提交。

## 2. 获取代码与本次需求

仓库：<https://github.com/sxh-kk/cvpr>。第三方代码以[固定上游记录](psi0_upstream.json)为准，`VLMExpert/third_party/SIMPLE`等目录已经展开为普通文件。

新机器在用户选定的工作目录执行：

```bash
git clone https://github.com/sxh-kk/cvpr.git
cd cvpr
git rev-parse HEAD
cat AGENTS.md
cat docs/4090_SIMPLE_HANDOFF.md
```

已有checkout时，先检查当前分支、未提交改动及本地提交，再在该分支执行安全的fast-forward同步。不要覆盖已有工作或强制重置。

本仓库**不是**上游Psi0的子模块checkout，不执行旧文档中的`git submodule update --init --recursive`，也不另拉上游最新源码覆盖展开目录。Git LFS可取回仓库追踪的资源，但Git／LFS不会提供被排除的训练权重、SONIC模型及全部仿真资产。

## 3. 必须保持的模型与数据协议

| 项目 | 当前20k模型的协议 |
| --- | --- |
| 架构 | `dual_source_v1`，6个action blocks，使用当前仓库实现 |
| VLM | 完整最后层prefix hidden tokens、原mask、masked mean `p_V` |
| Motion | 5Hz历史5帧＋当前帧；共6帧，覆盖过去1秒；输出完整11个hidden和mean `p_M` |
| 在线DINO | 冻结`dinov2-with-registers-large`；每帧CLS＋4 registers，即`5×1024`；输入`[B,6,5,1024]` |
| 图像 | `ImageProcess`显式`256x320` profile；VLM与Motion当前帧同源、同时间、同几何；推理关闭随机增强 |
| state | 原43维关节状态按训练顺序／单位映射，再补到45维；保留当前双来源state token接法 |
| action | 每段`30×80`；64维`body_token_v1_1`＋14维手部动作＋2维padding；padding不作为学好的neck控制 |
| 推理 | `eval()`／无梯度；初期保持既有混合精度策略，不先做量化或删减模型以绕过验收 |

历史图像必须来自本次episode的真实观测，按**仿真时间戳**取`t−1.0,t−0.8,t−0.6,t−0.4,t−0.2,t`。执行动作期间也应收集图像；不能把最近6次HTTP请求当成历史6帧。episode切换要清空历史及RTC状态，先积累足够真实历史；不重复单帧伪造窗口，不使用未来图像或未来标签。

## 4. 两个独立运行环境

使用两个venv／Conda环境或两个容器。根据已有安装选择可复用的隔离环境，先核对版本，不修改系统Python或A100环境。

| 环境 | Python | 基线与用途 |
| --- | --- | --- |
| 模型服务 | **3.11** | 当前Psi0＋MotionExpert＋ImageProcess＋在线DINO；本机HTTP服务 |
| 仿真控制 | **3.10** | 当前SIMPLE＋gear_sonic／相关机器人SDK＋MuJoCo＋Isaac Sim |

当前源码的版本依据：

- [Psi0 pyproject](../VLMExpert/pyproject.toml)：Python3.11；核心依赖在`psi`依赖组，HTTP服务还需`serve`组。仅`pip install -e VLMExpert`不会自动安装这两个依赖组。
- [SIMPLE pyproject](../VLMExpert/third_party/SIMPLE/pyproject.toml)：Python3.10、Isaac Sim4.5.0、MuJoCo3.3.6、Torch2.7.0、Torchvision0.22.0、NumPy1.26.4。SONIC／SDK是单独依赖组及本地源码依赖。
- [gear_sonic pyproject](../VLMExpert/third_party/SIMPLE/third_party/gear_sonic/pyproject.toml)也要求Python3.10。

A100模型环境已验收的主要版本，可作为4090模型环境的起点；**这不是4090实测结果**：

```text
torch                 2.7.0（A100基线CUDA 12.6 / cu126）
torchvision           0.22.0
transformers          4.57.1
accelerate            1.7.0
flash-attn            2.7.4.post1
qwen-vl-utils         0.0.14
diffusers             0.37.0
numpy                 1.26.4
pillow                11.3.0
einops                0.8.1
omegaconf             2.3.0
tyro                  0.9.32
safetensors           0.6.2
lerobot               0.3.3，源码固定songlin/lerobot提交
                      09929d8057b044b53aecaf5c6d7eb71f99e8beb9
```

模型环境还需安装本仓库`ImageProcess`、`MotionExpert`和`pipeline[dino]`的本地包及其必要依赖。用安装约束保护已选择的Torch／Torchvision／NumPy版本；不要让本地包安装意外升级核心栈。

SIMPLE的`uv`索引包含`cu128`，与A100模型环境的`cu126`基线不同。两环境可以各自使用兼容的CUDA runtime，但**都受同一个宿主驱动约束**；分别记录实际`torch.version.cuda`和驱动兼容性。先使用现有驱动制定匹配安装方案，不机械升级系统驱动。

FlashAttention必须匹配4090的Ada／SM89、Python、Torch、CUDA和C++ ABI；不复制A100环境或已编译扩展。安装后做真实小规模无梯度前向验证。Isaac安装遵循**4.5版本**要求，不因为当前在线文档展示6.x就直接升级；许可交互按安装器实际要求处理。

本机无需使用A100侧`VLMExpert/scripts/train/psi0/unifolm-local.env.sh`：它包含A100绝对路径。创建4090本地配置，数据／权重路径全部指向本机资源。

注意：`VLMExpert/examples/quick_start/psi.md`及部分上游示例仍有Python3.10单环境、`--all-groups`或子模块命令；它们不构成本次双来源部署的已验收安装配方。也不要在同一个Python进程里混装两边的包。

## 5. 环境配置执行顺序

1. **检查4090机器。** 记录OS／架构、GPU型号与显存、驱动、现有CUDA工具链、系统内存、磁盘、已有环境和占卡进程。Linux Ubuntu22.04 x86_64是当前SIMPLE上游基线；其他系统先核兼容性。不要停止用户其他GPU任务来做冒烟验证。
2. **准备模型环境。** 安装Python3.11和版本约束内的依赖，启用`psi`／`serve`组，安装本地三个模块；检查`pip check`、本地源码导入、CUDA小算子和BF16。确认不会落到上游安装的同名旧`psi`包。
3. **准备仿真环境。** 安装Python3.10、SIMPLE和所需SONIC／SDK依赖，核对缺失的编译依赖；记录Torch CUDA wheel和驱动兼容情况。不要为单次评测无目的地安装所有基线、训练和文档依赖。
4. **检查物理与渲染。** 分别检查MuJoCo基本step和Isaac Sim headless启动、相机生成真实RGB；按Isaac生命周期先启动`SimulationApp`再导入相关Omniverse模块。没有窗口不等于不渲染，相机仍需RTX。
5. **检查本机通信。** 在不加载大模型时先验证两个环境之间的localhost请求／响应和session reset；绑定本机地址即可，使用端口可配置。
6. **资源到齐后检查模型。** 加载固定20k联合模型，检查一次batch1、真实历史输入的输出形状、有限性、完整tokens／pooling和动作反归一化；然后与控制器联调。
7. **整套4090验收。** 单个仿真实例、单episode开始，记录模型独立／仿真独立／同时运行三种峰值显存及端到端延迟，随后验证多episode reset。达到此阶段才报告“整条链路可运行”，再按固定任务／seed统计成功率。

可把硬件检查、依赖检查与源码review分给subagent并行；GPU检查先核占卡情况，串行或显式分配资源，防止相互抢显存。

建议报告保存在仓库外的`<4090工作目录>/reports/`。依赖检查可使用目标环境的`python -m pip check`；uv新建的venv可能没有pip模块，此时使用`uv pip check --python /该环境/bin/python`。检查命令例如：

```bash
nvidia-smi
cat /etc/os-release
ldd --version
df -h
# 分别激活模型环境和仿真环境后执行；此处示例使用uv检查：
uv pip check --python "$(command -v python)"
python -c 'import sys, torch; print(sys.version); print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
```

上面的CUDA查询仅在确认GPU可用于检查后执行。没有4090访问权限或硬件时，不把CPU导入检查写成4090实测。

## 6. 资源包与路径迁移

**权重和仿真资产不在GitHub代码里。** 4090侧可以先配环境，等待资源到齐后继续，不自动改用官方其他checkpoint。

| 本机所需资源 | 处理要求 |
| --- | --- |
| `checkpoints/ckpt_20000/model.safetensors` | 约12.066GB；包含训练后的VLM／Motion／ActionExpert，严格载入并记录SHA256 |
| `run_config.json`、`argv.txt` | 当前服务恢复配置使用；将绝对路径重映射到4090资源目录 |
| Qwen config／tokenizer／processor小文件 | 包括词表、chat template和图像processor；联合模型已包含VLM权重，无需原初始Qwen大权重 |
| Motion构造配置／normalization／视觉协议 | 当前加载器先用原Motion格式checkpoint建模，再以联合Motion权重覆盖；见下节。不能缺少构造依赖，也不能拿旧epoch100权重替代训练后的20k Motion |
| `dataset_statistics.json`及原bounds | run配置内已有state/action bounds，仍随包保存统计并核对字段顺序；不重新用SIMPLE测试集拟合归一化 |
| DINO权重／config／processor及身份记录 | `facebook/dinov2-with-registers-large`，使用与训练缓存一致的权重和预处理；本机加载 |
| SONIC控制器／decoder及配置 | 必须与`body_token_v1_1`配套，记录文件身份、关节与动作协议 |
| SIMPLE任务及机器人资产、评测初始条件 | 大型USD／纹理／模型／任务数据等部分资源不在本仓库，按固定版本准备并核对LFS指针文件 |

不需要搬运optimizer、scheduler、训练RNG、W&B、完整UnifoLM训练集或其离线DINO缓存。

A100源任务目录仅用于识别待迁移资源，**不是4090运行路径**：

```text
/gaozt-test1/datasets/UnifoLM/.pipeline/training/posttrain/
dual_source_unifolm_joint_motion100_8gpu_recovery.us.flow1000.cosine.lr1.0e-04.b256.gpus8.261020043942/
```

`checkpoints/ckpt_20000`是否已生成，以源端实时状态和保存完成记录为准。当前训练每1000步保存、保留3份；20k在23k保存时可能轮转删除。A100已启动独立CPU等待任务，保存成功后保护完整快照，再导出和校验；实时状态、4090拉取命令及资源边界见[资源交接](4090_SIMPLE_TRANSFER.md)。只有资源目录发布READY后才拉取，不使用未完成的目录。

## 7. 当前代码缺口：安装依赖不会自动解决

1. **可移植加载。** [Psi0Model.from_pretrained](../VLMExpert/src/psi/models/psi0.py)先构造`Psi0Model`，经`MotionConditioner.from_checkpoint`读取原Motion checkpoint，随后才严格加载联合模型中的Motion子权重。需要本机可用的构造资源；可导出“20k Motion子权重＋必要metadata”的便携文件，或从配置空构造再严格加载联合权重。修改不能丢失Motion统计和架构／视觉契约校验。
2. **双来源服务。** [HTTP服务](../VLMExpert/src/psi/deploy/serve_psi0_simple.py)目前未向`predict_action`传`motion_img_embs`；双来源模型会拒绝缺少历史的输入。SONIC HTTP复用此服务，WebSocket入口也未完成该接线。需补按session隔离的真实历史缓冲、在线DINO、仿真时间戳和reset。
3. **专用SIMPLE agent。** 当前[通用psi0客户端](../VLMExpert/third_party/SIMPLE/src/simple/baselines/psi0.py)及`psi0_decoupled_wbc`按旧32维state／36维动作工作；需要正确映射原43维state，并将64 latent＋14手动作交给对应SONIC链路，忽略2维padding。单纯补零或按80维截断不能修正语义。
4. **SONIC v1.1身份。** 内置gear_sonic的包版本／`sonic_model12`标签、64维shape不能证明其decoder与训练数据v1.1编码器配套。无法取得配套身份时列为实质阻塞，不能用不兼容控制器产生的失败率评价20k模型。
5. **旧配方边界。** 现有`finetune-sonic-simple-psi0.sh`使用旧`action.body_token`、78动作／43状态及旧初始化，不能原样用于这份80／45双来源模型；本次目标也不包含额外SIMPLE训练。

## 8. 4090端交付与判定

阶段结果分开报告：

- **环境完成**：两环境解释器与包版本、`pip check`、本地模块导入、CUDA／BF16、MuJoCo step、Isaac相机RGB、本机通信都有实际命令和结果；资源缺失及失败也逐项记录。
- **模型可推理**：指定20k权重的SHA和配置已绑定；真实历史6帧、完整tokens／pooling、state及`30×80`输出契约检查通过。
- **控制链路可执行**：v1.1身份及state／action映射通过，单episode能闭环执行，多个episode不串历史；不以环境导入成功替代此项。
- **正式测评**：保存任务、seed、随机化等级、执行频率／窗口、采样步数、RTC模式和同步方式；统计成功率、视频、端到端延迟及峰值显存。结果标为UnifoLM-20k零样本迁移。

任何阶段尚未验收时，报告该阶段实际边界。环境和资源在4090本地配齐后，完整运行应不再访问A100文件路径／服务；离线复现还需把Isaac扩展、纹理等首次运行会下载的资源完整缓存。

参考：[迁移说明](MIGRATION.md)、[当前双来源实现说明](../VLMExpert/README.md)、[PyTorch版本安装矩阵](https://pytorch.org/get-started/previous-versions/)、[SIMPLE官方SONIC示例](https://github.com/physical-superintelligence-lab/SIMPLE#quick-start-for-sonic-wholebody-vla)。外部最新示例与本仓库固定快照可能不同，冲突时先核对实际源码和依赖版本。
