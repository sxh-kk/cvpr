# 4090 独立 SIMPLE 环境工具

遵守 [交接文档](../../docs/4090_SIMPLE_HANDOFF.md)。本目录提供双Conda环境约束、
本机路径、资源核验、真实历史／联合模型／HTTP检查及有界SIMPLE控制闭环工具。
实际结果与未通过的边界见[GPU验收记录](../../docs/4090_SIMPLE_GPU_ACCEPTANCE.md)。
已独立固定的28k权重的CarryBox适配、训练/保留测试划分与实际结果见
[本地任务适配](../../docs/4090_SIMPLE_CARRYBOX_FINETUNE.md)。
实际安装／验证记录在本机仓库外 `~/CVPR_Evaluation_runtime/reports/`。

A100已提供CPU导出与保存完成等待工具`export_bundle.py`，4090可通过SSH／rsync拉取资源。
具体源路径、READY判定、SHA校验及当前缺项见[资源交接](../../docs/4090_SIMPLE_TRANSFER.md)。

## Handover任务微调权重的本机测评

`prepare_task_snapshot.py`专用于从UnifoLM38k初始化的task50联合微调checkpoint。
源`SNAPSHOT_READY.json`须绑定保存完成、模型SHA、source38k/task0/当前task_step及已验收30Hz数据契约，
同时提供两份实际初始化收据；不将任务6000步误标为UnifoLM6000步。
它严格核对架构、token、RGB、历史及原归一化，仅接受已验收任务的路径/划分/LR差异。
新Motion权重从这次完整联合模型导出，旧本机包只提供Qwen/DINO及Motion构造元信息。
旧`prepare_joint_snapshot.py`完整model/data相等的默认限制保持不变。

```bash
python scripts/4090_simple/prepare_task_snapshot.py \
  --source /本机独立目录/source --resources "$CVPR_SIMPLE_RUNTIME/resources" \
  --output /同一文件系统的新目录/deployment
```

`prepare_handover_eval.py`只准备交付包的五条原VAL场景39/65/80/92/94，不更新模型。
输入是`heldout_reset`的meta、data、validation和provenance小文件，不需要源视频。
它从row0及独立物体初态日志构造一次cold start，避免使用environment_config残留的最终物体姿态；
具名43关节、root姿态/速度和目标/篮子质量均绑定来源，资产复用既有本机资源。
记录源任务uid的显式映射，不认证两套任务/控制代码完全相同。

```bash
python scripts/4090_simple/prepare_handover_eval.py \
  --heldout-root /本机独立目录/release/heldout_reset \
  --asset-root "$SIMPLE_DATA_DIR" --output /本机新目录/heldout_eval_data
```

`run_closed_loop.py --handover-initialization /绝对路径/handover_initialization.json`仅适用Handover。
它在真实历史采集前一次设置记录的初态，不使用教师动作/primer或后续状态进行控制，不在控制中重置物体。
缺失的关节速度明确置零，MuJoCo软限位的记录穿透最多允许0.001rad且保留原值，不能据此放宽动作限制。
该模式使用`dr_level=None`恢复所有记录randomizer；普通官方DR0会重采材质和干扰物，两者必须分开统计。
源collection增益/扭矩档案未完全绑定时继续保留诊断边界，不伪造controller compatibility证据。

`--continue-after-task-success`保留首次官方success标志并继续观察到`--seconds`上限，
仅用于Handover；不新增reward调用或修改成功判据。测评可用16秒、每段执行30步、RTC off、
10步采样、同步Isaac RTX观测和`--video`；每个物理episode使用独立模型服务进程和明确采样seed。
trace额外保存目标/篮子姿态和具名接触，帮助区分仅碰篮、完整交接和释放；
这些50Hz观测不冒充交付包完整的`handover-physics-v2`重新验收或官方独立TEST成绩。

2026-10-22的任务6000步实际导出、五条16秒GPU结果及排除的前期DR0诊断见
[测评记录](../../docs/A100_SIMPLE_TASK50_FINETUNE.md#任务6000步导出与本机handover物理测评)和
[逐条证据索引](../../docs/a100_task6000_local_handover_evaluation.json)。
五条中1条触发接触成功标志、均未跌倒；触发标志的末帧仍有左手接触，不能称为完整释放验收通过。

后续[数据质量与100条重新采集](../../docs/HANDOVER_QUALITY100.md)单独审计原轨迹祖先、
模板权重与物理时钟差异。`plan_handover_quality100.py`只生成候选计划，不能将计划算作数据。
`replay_handover_teacher.py`是已编码教师命令的真实本机诊断/重采入口；
`handover_contact_quality.py`和`handover_geometry_quality.py`保留源50Hz严格规则，
包括原解算接触力、掌根、交接顺序与容器支撑，物理通过仍不自动授权数据入库。

用户要求进一步扩充原始动作来源并提高场景与动作多样性后，使用独立的
[随机化100条采集流程](README_handover_random100.md)。该流程重新加载完整原始TRAIN候选，
进行真实场景随机化、物体反馈动作适配、完整碰撞投影和多样性验收，再打包、提取DINO并同步A100；
不将旧数据、失败试采或仅修改seed的候选计为新的100条。

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
  不启用 DeepSpeed 预编译训练算子；Action-only本地适配使用普通 AdamW，
  联合适配的独立入口使用CPU FP32 master和原生CPU AdamW，保留生产梯度检查与逐组裁剪。
  CarryBox 的 CPU v1.1 重编码另需 `onnxruntime==1.23.2`，仿真侧读取示范需
  `pyarrow==20.0.0`，均已加入对应环境约束。

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
其余 `CVPR_*` 变量是部署约定，模型服务的资源目录和端口仍须显式传参。
资源包换目录后须同步重映射 run_config 和argv，严格加载与校验。

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

## 真实20k模型与有界闭环

以下从仓库根运行，产物均在仓库外。两个终端分别激活对应的Conda环境并source本机配置。
先完整核验包。官方scene0的props路径有误，首次使用仿真Conda环境创建独立修复目录，
保留源USD／纹理，记录源／派生SHA与大小写纠错；不覆盖已有派生目录：

```bash
conda activate cvpr-simple-sim310
source scripts/4090_simple/local.env.example.sh
python scripts/4090_simple/prepare_simple_assets.py \
  --source-root "$CVPR_SIMPLE_SOURCE_DATA_DIR" --output-root "$SIMPLE_DATA_DIR"
```

仅此步骤的文件与CPU USD组合检查不代表MDL编译或RTX渲染通过。
然后用真实任务采集图像和状态；采集使用测量姿态PD，不是模型控制：

```bash
conda activate cvpr-simple-sim310
source scripts/4090_simple/local.env.example.sh
python scripts/4090_simple/capture_simple.py --data-root "$SIMPLE_DATA_DIR" \
  --sonic-resources "$CVPR_SONIC_DIR" --sim-mode mujoco \
  --output "$CVPR_SIMPLE_RUNTIME/reports/joint20k/simple-history.npz"
```

在模型环境使用严格加载和真实在线DINO，检测完整tokens／pooling及有限动作：

```bash
conda activate cvpr-simple-model311
source scripts/4090_simple/local.env.example.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/4090_simple/probe_joint.py --resources "$CVPR_SIMPLE_RUNTIME/resources" \
  --history "$CVPR_SIMPLE_RUNTIME/reports/joint20k/simple-history.npz" \
  --instruction 'bend to pick up the cracker box' \
  --output "$CVPR_SIMPLE_RUNTIME/reports/joint20k"
python -m psi.deploy.serve_psi0_simple --policy psi0 --run-dir "$CVPR_JOINT_20K_DIR" \
  --ckpt-step 20000 --device cuda:0 --host 127.0.0.1 --port "$CVPR_MODEL_PORT" \
  --action-exec-horizon 30 --rtc-mode off --num-inference-steps 10
```

回到仿真环境。生产服务ready后依次执行；HTTP replay校验425/409及跨episode reset，
不把replay当作真实控制：

```bash
python scripts/4090_simple/probe_http.py \
  --history "$CVPR_SIMPLE_RUNTIME/reports/joint20k/simple-history.npz" \
  --port "$CVPR_MODEL_PORT" --output "$CVPR_SIMPLE_RUNTIME/reports/joint20k/http-replay-new"
OMNI_KIT_ACCEPT_EULA=YES python scripts/4090_simple/run_closed_loop.py \
  --data-root "$SIMPLE_DATA_DIR" --sonic-resources "$CVPR_SONIC_DIR" \
  --port "$CVPR_MODEL_PORT" --sim-mode mujoco_isaac --seconds 5 \
  --pd-profile official_v1_1_ankle_pitch_1_5 --diagnostic-unverified-sonic \
  --output "$CVPR_SIMPLE_RUNTIME/reports/joint20k/loop-new"
```

闭环默认3个官方任务，各dr0 episode0。`--tasks`可筛选任务，`--episodes-per-task 2`
验证同一session重置；`--sim-mode mujoco`用于不带RTX的对照。每次使用新输出目录保留证据。
50Hz真实本体历史和30Hz高层动作均用仿真时钟；同步HTTP期间仿真暂停，不能据此声称实时。
正式任务得分还需完整episode、成功判定、视频与足够样本；有界5秒通过仅表示集成链路可执行。

SONIC 官方 v1.1 版本已经通过固定发布身份与官方 Psi0 训练脚本确认；
`--diagnostic-unverified-sonic`记录原 UnifoLM token 未逐值复算这一追溯边界。
不设置该选项时，须提供真实可追溯的 `--compatibility-evidence`，不得自行把未验证项置真。
原 UnifoLM-20k 基线保留；用户后续授权的本地 SIMPLE 适配使用独立训练目录，
与未适配的 A100 权重分别记录，不能混合报告。

CPU回归：模型环境运行 `python -m unittest discover -s VLMExpert/tests -p test_dual_source_serving.py -v`；
仿真环境运行 `python -m unittest discover -s scripts/4090_simple/tests -p 'test_*control.py' -v`
及 `-p test_smoke_reporting.py`。CPU测试覆盖契约与报告，不代替上述GPU／物理检查。

独立的举箱保持任务、RTX同域成功示范和本地微调流程见 [LiftBoxHold记录](../../docs/4090_SIMPLE_LIFTBOX_FINETUNE.md)。旧CarryBox派生包不能用于训练。

`compare_liftbox_eval.py` 在训练和source4/8前后四次物理运行完成后汇总实际证据：核对完整初态、
预热/控制/渲染、checkpoint及逐次采样RNG，独立重算保存的原始/归一化动作误差与50Hz日志中的
连续保持条件。它只读取本机文件，不加载模型、不启动仿真；完成运行、站稳和任务成功分别报告。
命令及本地pilot边界见上述记录，不能把两例结果写成官方SIMPLE成功率。

`check_liftbox_joint_gpu.py`用于独立两次联合更新资格检查：读取已验收准备目录，核对实际数据和
DINO文件身份，对齐A100五组参数及峰值LR，使用生产反传/安全检查与CPU原生AdamW状态，
不保存模型或optimizer。实际两步结果见LiftBoxHold记录；短检查不等于完整联合微调通过。

```bash
source scripts/4090_simple/local.env.example.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/4090_simple/check_liftbox_joint_gpu.py \
  --prepared-run "$CVPR_SIMPLE_RUNTIME/finetune/lift_box_hold_28000_action300_rtx_v1" \
  --alignment "$CVPR_SIMPLE_RUNTIME/reports/a100_unifolm_actual_training_alignment.json" \
  --output "$CVPR_SIMPLE_RUNTIME/reports/liftbox-joint-check-new" --run-gpu
```

这里的alignment报告为本机SSH只读核对结果，不包含认证信息；无需挂载或调用A100运行模型。

本地完整联合pilot使用独立入口，保持五组峰值LR和生产安全检查。`--base-run`可显式选择
已保存完成、SHA核验及Motion提取通过的更新快照；新旧模型结构、数据、统计与DINO契约必须相同。
成功示范的原28k质量锚点继续单独保留，不改写为新权重来源。只保存最终模型，不能恢复optimizer续训。

```bash
python scripts/4090_simple/finetune_lift_box_hold_joint.py \
  --prepared-run "$CVPR_SIMPLE_RUNTIME/finetune/lift_box_hold_28000_action300_rtx_v1" \
  --base-run "$CVPR_SIMPLE_RUNTIME/resources/latest31000/run" \
  --alignment "$CVPR_SIMPLE_RUNTIME/reports/a100_unifolm_actual_training_alignment.json" \
  --output "$CVPR_SIMPLE_RUNTIME/finetune/lift_box_hold_31000_joint300_rtx_new" \
  --batch-size 4 --grad-accum 4 --max-steps 300 --warmup-steps 20 --train
```

需先完成新快照迁移和`prepare_joint_snapshot.py`，并使用已source本机配置的模型Conda环境。
可在另一新输出目录以`--prepare-only`检查真实JSON/argv及资源身份；准备完成不代表GPU训练或物理成功。
300步、有效batch16是本地小包适配预算，不等于A100有效batch256、40k更新的完整日程。

真实热身前缀v2另用`collect_liftbox_prefix_v2.py`重新采集完整轨迹，再由
`prepare_liftbox_prefix_v2.py`验收和打包；不能向旧视频或DINO缓存拼接历史。
前30行只有真实图像/状态，`action_valid=false`，不参与动作统计和监督；首个teacher当前帧30
可读取完整六帧真实历史。已通过的原始数值验收只重绑封存证据和文件身份，缺证据即拒绝。
本机v2成功专家包为TRAIN0/1/6、VAL3、TEST9，失败2/4/5及候选顺序另行保留；
这是有记录的成功示范筛选，不是盲测。原source4/8仍须物理对照，另加TEST9。

v2数据、特征和训练产物放在`$CVPR_SIMPLE_RUNTIME/prefix_v2`，该本机路径链接到容量充足的
`/var/tmp/cvpr-evaluation-zxc-prefix-v2`。以下命令在模型Conda环境运行；DINO已按原提取命令
对三个split的全部新图像重新提取，并在准备阶段核对逐集SHA和实际历史索引。

```bash
python scripts/4090_simple/prepare_liftbox_prefix_training_v2.py \
  --base-run "$CVPR_SIMPLE_RUNTIME/resources/latest31000/run" \
  --dataset-root "$CVPR_SIMPLE_RUNTIME/prefix_v2/resources/liftbox_prefix_v2_train016_val3_test9" \
  --dino-cache-root "$CVPR_SIMPLE_RUNTIME/prefix_v2/dino/liftbox_prefix_v2_train016_val3_test9" \
  --output "$CVPR_SIMPLE_RUNTIME/prefix_v2/finetune/prepared_prefix_v2" \
  --batch-size 4 --grad-accum 4 --max-steps 300 --warmup-steps 20 --prepare-only
python scripts/4090_simple/finetune_lift_box_hold_joint.py \
  --prepared-run "$CVPR_SIMPLE_RUNTIME/prefix_v2/finetune/prepared_prefix_v2" \
  --base-run "$CVPR_SIMPLE_RUNTIME/resources/latest31000/run" \
  --alignment "$CVPR_SIMPLE_RUNTIME/reports/a100_unifolm_actual_training_alignment.json" \
  --output "$CVPR_SIMPLE_RUNTIME/prefix_v2/finetune/joint300_prefix_v2" \
  --batch-size 4 --grad-accum 4 --max-steps 300 --warmup-steps 20 --train
```

使用全新输出目录保存每次尝试。v2从已核验的A100 31k初始化，五组联合更新，DINO冻结，
不继承前一次本地300步模型或optimizer。实际两步GPU资格、完整300步及独立CPU审计通过；
三初态物理结果为原source4/8成功、新source9失败，总计2/3，细节与追溯限制见LiftBoxHold记录。
`compare_liftbox_eval.py`按准备记录核对TEST9，并要求source4/8/9前后结果齐全，不能只报告成功初态。


## 停止采集后的已有数据精选合并

`select_handover_curated.py` 仅读取已验收记录：保留全部新14条，原100先按相同的
真实完整碰撞投影final50门筛选，再按质量优先和实际运动近重复去重；不启动仿真。
`pack_handover_curated.py` 创建可变数量的独立LeRobot包，新视频无损编码、旧视频
原字节复用，只重排旧Parquet的三项索引，38k统计和动作标签保持。
`curate_handover_dino_cache.py` 将已真实提取的新特征与旧特征显式重编号、绑定新视频和元数据，
保留原manifest/receipt，复制数组字节；`verify_handover_curated.py` 验证真实DINO、
生产Dataset/MotionHistoryIndex/collator和所有标签/history/future。它们使用独立curated合同，
不放宽原exact100/至少50来源工具。

本次实例目录：`/home/zxc/CVPR_Evaluation_runtime/handover_curated_merge_20261024`。
筛选结果28条（新14+旧14），原100中25条满足更严格完整投影门，另去11条近重复。
正式可用状态以该目录的`CURATED_DATASET_READY.json`与生产读取记录为准；不自动训练或同步A100。

## 新14条与完整原100条合并并在A100微调

`merge_handover_114.py` 使用独立 exact114 合同，完整保留新14条和原100条；
0–13来自前述28包的新14，14–113依次来自原100的0–99。
视频、DINO数组、归一化和动作保持原值，Parquet只重排episode/global/task索引。
原100保持原physics-v2验收，新14保持各自原准入，不声称114条全部通过额外5mm投影门。
本次本地目录为 `/home/zxc/CVPR_Evaluation_runtime/handover_merged114_20261025`，
训练数据位于 `packed/train`、真实特征位于 `dino`；共49,089帧、42,363完整训练窗。
`pack`、`cache`、`verify`依次构建并验收；`MERGED114_DATASET_READY.json`绑定本地就绪结果。

`upload_handover_merged114.py --local-root LOCAL --remote-root REMOTE` 复制到独立A100目录，
原100远端视频/特征经SHA匹配后可直接复制到新编号，随后全量校验文件内容并恢复纳秒mtime。
远端用 `merge_handover_114.py relocate --pack-root REMOTE/packed --cache-root REMOTE/dino`
在保留传入元数据和原缓存备份后重绑定DINO根路径，再运行 `verify` 实际读取全部114集。
远端 `A100_DATASET_READY.json` 必须绑定实际重定位后的manifest、reindex、PACK_READY和生产报告，
不能沿用本地manifest哈希充当远端验收。

`a100_handover_merged114.py prepare/check/run` 复用SHA绑定的原A100冻结训练代码及状态迁移helper，
从已验收joint38k模型重新开始optimizer/scheduler与步数；继承最近融合模型、state_drop_prob=0、
256×320 clean DINO历史及VLM颜色增强，保持独立5集/40窗VAL，不使用TRAIN作验证。
正式配置固定8卡、每卡8、累积1、全局64、10000次optimizer更新、1000步warmup。
每1000步保存并保留全部模型权重；仅本新run超过最近两份的optimizer/scheduler/RNG会移除，
旧run和旧数据保留。实际更新2/20/最终步检查各卡有限性、参数探针一致性及内存，
单次短检查不代表完整训练或策略成功率已通过。

## PICO 仿真遥操作采集

`bash scripts/4090_simple/start_pico_teleop.sh` 使用已有 Conda 仿真环境，接入
PICO 原始单包全身输入、固定 SONIC v1.1 SMPL mode2、真实200Hz MuJoCo、
50Hz decoder、30Hz实际状态和命令录制。默认实时MuJoCo EGL双目预览，退出后
以同一归档状态补渲染RTX训练图像并独立审计，不重跑物理。电脑面板只监听127.0.0.1，
`pico_desktop_preview.py` 在本机43101提供只读相机大画面，复用43100连续MJPEG视频流及状态，
不额外订阅PICO。双目 H.264 可另回传指定 PICO，实际头显显示仍待验收。
设备设置、按键与校准、实时失败处理、独立数据验收
及实际验证边界见[采集说明](../../docs/PICO_SIMPLE_TELEOP.md)。
手指使用显式 SONIC v1.1 middle-close 对握姿态，前扳机或侧握键控制；界面区分原始输入、
下发目标和实际关节闭合程度。闭合不等同于成功抓取，仍须检查实际接触和任务结果。
控制页面也使用连续视频流；状态每500ms刷新不再限制画面帧率。暂停时不重复发布同一仿真帧，
短暂等新PICO包仍受原100ms限额约束，不用重复包充当新采集数据。
新版本录制中忽略重新标定请求，只提示先保存；起录只绑定当前示例必要身份和场景，
不复制全会话历史，并记录完整循环耗时以定位实时中断。正在运行的旧进程不热更新。
每条结束只等待原始写入封存，随后重置场景并暂停，下一条重新标定、开始跟随。
独立CPU后台进程完成几何复算和物理预验收，按集合内顺序编号为001等，分到 `kept/编号/` 或
`discarded/编号/` 并记录中文失败原因；不占用GPU、SDK或采集线程的归档锁。
页面单独显示后台状态和最近结果，待审时可按UUID申请丢弃。队列与独立补算回执可跨重启恢复；
退出采集后，补图入口先等待本次后台任务完成，待审和已申请丢弃的记录不得进入RTX／训练准入。
没有活动录制时“丢弃上一条”只移动上一条到丢弃目录，保留同一编号及原始证据；
集合索引与 `collection_records.jsonl` 跨重启保存。保留仅代表物理预验收通过，
RTX补图与完整独立审计仍待完成；丢弃记录被补图和审计入口拦截。
`audit_pico_teleop_episode.py EPISODE_DIR` 独立审计原始录制；回放／固定姿态诊断
不增加人类示范数量，通过审计后仍需特征与训练协议打包。

独立新增人体骨盆平移→planner意图、异步CPU planner和零SDK物理诊断入口，默认不接当前采集。
`replay_pico_locomotion_intent.py` 仅读封存动作包，`diagnose_pico_locomotion.py` 仅在独立进程
执行真实步行检查；当前仅前向通过本地物理门，后退／横移仍有失败，不能视为四方向跟随就绪。
资源、命令、坐标及验证限制见[独立行走说明](../../docs/PICO_SIMPLE_TELEOP.md#独立人体平移与行走诊断默认关闭)。
