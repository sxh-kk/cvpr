# SIMPLE 双来源控制与真实历史

`simple.baselines.dual_source_psi0`是20k双来源专用客户端，不沿用旧32-state／36-action路径。
`/observe`和`/act`使用原RequestMessage格式，history明确携带`session_id`、`episode_id`、
`sim_time_s`；`/reset`使用前两项的平坦JSON。425只表示真实历史未满一秒，其他错误中断执行。
客户端仅接受localhost，忽略系统HTTP代理；每个渲染帧都上报，包括动作chunk执行期间。

关节契约依据官方`USC-PSI-Lab/psi-data`固定revision
`e78fb93cc28912a3031a10b8656d32d7f0a2b867`的val ZIP，SHA256为
`3d264d6454d59e83be5dadaa17f122b732a8c6cb86a2429c33c1f4fc912fc4b7`。
其`info.json` SHA256为`3b58f2cb791fc0e82bcc6c4dc08a6ced44cd5729e9531bd741f2e392dff1d467`。
原43维state单位为弧度，body29后两手都是thumb3／middle2／index2；SIMPLE手顺序是
thumb3／index2／middle2，故按关节名映射，再在末尾补2维。动作严格为30×80：
64维body_token_v1_1、14维训练手顺序、2维padding；padding不控制颈部。

环境的`joint_names`和`sim_time_s`已纳入Gymnasium observation_space。
时间直接来自`mjData.time`。高层动作依训练30Hz按仿真时间取样；相机须至少30Hz，
不能按HTTP请求次数或渲染次数加速动作。首任务原配置为physics_dt=0.005、render_hz=50。

## 真实历史采集

以下由仿真Conda环境执行；Isaac与真实评测应由主进程串行调度，避免争抢4090显存。

```bash
source scripts/4090_simple/local.env.example.sh
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python scripts/4090_simple/capture_simple.py \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --sonic-resources /home/zxc/CVPR_Evaluation_runtime/resources/sonic_v1_1 \
  --output /home/zxc/CVPR_Evaluation_runtime/reports/simple-history.npz
```

默认使用Psi0的XMovePick任务、DR0、episode0的官方固定environment_config，
将旧机器SIMPLE/data路径重映射到本机数据根目录。输出真实RGB、sim_time_s、按训练顺序的
state43、episode_id及完整来源metadata，同时保存首末PNG和JSON。
这里采用实测关节姿态PD保持，不运行20k策略或平衡策略；即使采集完成，也不证明任务闭环。

## SONIC v1.1适配与身份边界

`simple.baselines.sonic_v1_1.SonicV11OnnxController`核对固定官方decoder和observation_config
SHA后构建本机ONNX Runtime会话。适配依据仓库固定`GR00T-WholeBodyControl/gear_sonic_deploy`
的`policy_parameters.hpp`及`GatherHis*`／`CreatePolicyCommand`实现：
每个MuJoCo物理tick执行PD，在50Hz控制点采集真实10帧角速度、相对默认角度、关节速度、
上一raw action和重力方向，按term拼接为64＋930维decoder输入。先按IsaacLab顺序解码，
再按关节名输出43个扭矩；增益、default angles、action scales均与固定C++逐值对照。
不重复帧补齐proprioception，不借用旧SONIC standing token，不覆盖第三方快照。
实测姿态PD温启时，上一raw action按实际保持目标反解，保证decoder的last-action历史与
已发指令一致；不把保持姿态的目标误记为零residual。

默认`pd_profile="fixed_cpp_baseline"`保留固定源码增益。显式选择
`official_v1_1_ankle_pitch_1_5`时，核对固定官方README SHA
`0a69d0a418fddc2d96e9a70244876c32d40a44bee2b20c70c21747807c2972fd`，
对body顺序4／10的踝pitch KP和KD额外乘1.5，并将实际增益写入identity。
这是官方文档的v1.1调参配方，是否在本机稳定仍需实测，不能因采用该配方就声明通过。

官方v1.1文件身份与UnifoLM训练body_token_v1_1所用encoder的身份是两项独立证据。
缺少独立兼容证据时，构造器默认拒绝运行；只有显式
`allow_unverified_compatibility=True`才允许诊断，identity记录
`unverified_diagnostic_only`。这类结果不能作为20k正式任务失败率或完整闭环验收。
正式兼容证据JSON需绑定`body_token_version`、`decoder_sha256`、
`compatibility_verified=true`与可复核的`evidence`；调用方必须确认其证据真实成立。

控制器先构造，再显式传给agent：

```python
controller = SonicV11OnnxController(sonic_dir, compatibility_evidence=evidence_json)
agent = DualSourcePsi0Agent(env.unwrapped.task.robot, host="127.0.0.1", port=21074,
                           controller=controller, execute_steps=30)
agent.reset(episode_id="fixed-task-seed-episode")
action = agent.get_action(observation, instruction=env.unwrapped.task.instruction)
observation, reward, terminated, truncated, info = env.step(action)
```

每个episode同时reset环境和agent，清空图像、chunk、RTC和SONIC proprioception历史。
是否站稳、能否操作、任务成功率、GPU峰值与延迟，都必须由真实执行另行验收。

## CarryBox teacher 回放与独立模型评估

新增任务的箱体、场景、初态与本地成功指标见
[SIMPLE CarryBox 说明](../../VLMExpert/third_party/SIMPLE/docs/CARRYBOX_LOCAL.md)。
训练示范先从真实 50 Hz q/root 重新编码为官方 v1.1，再构造独立的 30 Hz 数据。
旧 `action.body_token` 不可重命名后使用。先运行 teacher 物理回放检查标签与控制器：

```bash
source scripts/4090_simple/local.env.example.sh
PYTHONNOUSERSITE=1 conda run --no-capture-output -n cvpr-simple-sim310 \
  python scripts/4090_simple/replay_carrybox_v11.py \
  --dataset-root "$CVPR_SIMPLE_RUNTIME/resources/carrybox_v11_smoke/train" \
  --conversion-provenance "$CVPR_SIMPLE_RUNTIME/resources/carrybox_v11_smoke/train/meta/conversion_provenance.json" \
  --state-action-contract "$CVPR_SIMPLE_RUNTIME/resources/carrybox_v11_smoke/STATE_ACTION_CONTRACT.json" \
  --data-root "$SIMPLE_DATA_DIR" --sonic-resources "$CVPR_SONIC_DIR" \
  --output "$CVPR_SIMPLE_RUNTIME/reports/carrybox-teacher-v11" \
  --episode 0 --seconds 20 --sim-mode mujoco --video
```

仿真 Conda 环境需要 `pyarrow` 读取 Parquet；视频需要本机 `ffmpeg`。
脚本核对具名 state43、action14、body qvel29、encoder SHA、独立 token 复算记录和合同快照 SHA。
仅在开始时设置一次示范的真实 q/root/object 与已记录/推导的速度，随后全程真实物理；
手速度和箱体速度因源数据未记录而初始置零，并留痕。先以实测姿态 PD 采集至少 0.2 秒真实
proprioception，再从 source clock 0 开始按 30 Hz 取 teacher token，decoder 50 Hz、PD 200 Hz。
这不是模型的一秒相机历史温启，也不是模型效果。到示范有效尾部立即停止，不补未来运动。
报告包含 source episode/frame/time、关节 RMSE、根/箱体姿态误差、跌倒、任务进度、
`trajectory.npz`、实际接触、CPU 重新载入验证的物理模型 MJB、PNG 和完整 MP4。
MJB 包含编译后的网格和物理参数；时变状态保存在 NPZ。
附带 XML 是 MjSpec 诊断 dump，当前 MuJoCo 3.3 attach 导出的 XML 可能丢失 meshdir/默认类，
不能直接作为可复现模型使用。训练归档没有场景/光照/材质状态，回放该部分采用本地官方任务配方，
这个 domain 差异也记录在初始化报告。回放脚本完成不自动代表标签已通过控制质量验收。
可用 `audit_carrybox_replay.py --trajectory <外部trajectory.npz> --robot-mjcf
<SIMPLE_DATA_DIR>/robots/g1_sonic/g1_29dof_with_hand.xml --output <外部audit目录>`
在 CPU 重建两条 q/root 的手掌 FK，分别检查根位姿与关节误差贡献。
重建接触是同一几何下的一致性证据，不能当成源记录的接触力或独立 FK 真值。
该 MJCF 将 palm 网格融合在 wrist_yaw body 中，接触统计必须包含它。

原生 50 Hz 控制诊断使用 `replay_carrybox_native50.py` 直接读取锁定的原始 Parquet，
不经过旧 30 Hz 转换包。`recorded-legacy` 只诊断身份未知的原 `action.body_token`，
必须显式 `--acknowledge-unverified-legacy`，不得当作 v1.1 标签。
`reencoded-v11` 用记录的当前航向离线编码；`feedback-heading-v11` 每次新命令以真实
MuJoCo pelvis yaw 消去未来参考 world root 的航向，仅改变官方 encoder 输入 584:644。
q、dq、未来采样偏移和 world reference 全部保持原记录，参考 anchor A=I；这不是空间重定向。
离线 actual-q 复算不自动等价于原遥操作 reference command，也不自动通过控制质量验收。
独立 `feedback-vr-fk-v11` 使用官方 mode 1：以真实 source q/root 的独立 MuJoCo FK
构造固定 world 腕部与 torso 派生头部参考，再变换到当前真实 pelvis 的完整局部坐标系；
保留源未来 12 腿关节参考。这是显式构造的新 world-FK reference expert，
不是从归档恢复原遥操作 head/wrist command；同样必须完整物理回放后才能接受为新示范。
另一个独立变体 `feedback-wrist-ik-v11` 保持 mode 0，用实际／source root 的当前
SE(3) 偏差预测本次 0.9 秒参考窗的 root，仅对两臂 14 个参考关节做 world-wrist DLS IK。
目标为原 source 固定 world 腕 pose；每腕目标位置修正最多 6 cm、旋转最多 10°，
每关节修正最多 0.20 rad。腿／waist 的 q 和 dq 原值保留；arm dq 加入同一修正曲线的差分。
报告包含目标、修正、残差与饱和。此变体没有实际箱体反馈或 phase gate，不改 gains/caps；
它是显式新 reference expert，FK 误差下降不能代替完整抓持／放置验收。

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n cvpr-simple-sim310 \
  python scripts/4090_simple/replay_carrybox_native50.py \
  --source "$CVPR_SIMPLE_RUNTIME/resources/simple_finetune_sources/G1WholebodyXMoveBendCarryBoxSonic-v0" \
  --contract "$CVPR_SIMPLE_RUNTIME/resources/simple_finetune_sources/CARRYBOX_STATE_ACTION_CONTRACT.json" \
  --data-root "$SIMPLE_DATA_DIR" --sonic-resources "$CVPR_SONIC_DIR" \
  --output "$CVPR_SIMPLE_RUNTIME/reports/carrybox-native30-feedback-source0" \
  --token-mode feedback-heading-v11 --command-hz 30 \
  --sync-observations \
  --episode 0 --seconds 60 --warmup-seconds .2 --sim-mode mujoco \
  --pd-profile official_v1_1_ankle_pitch_1_5 --video
```

`--command-hz` 默认 50；选择 30 时，只有 `floor(source_elapsed*30)` 变化才更新
encoder token 与手目标，其余 50 Hz tick 零阶保持，decoder 仍为 50 Hz、PD 200 Hz。
事件 n 在第 `(n*50+29)//30` 个实际相机 tick 执行，间隔为 40/40/20 ms，不能宣称
是均匀 30 Hz 物理观测。`command_events.jsonl` 与 `command_rgb/event_*.png` 保存同一
step 前的真实 RGB/state43/base/object、实际 token/hand14、源帧与真实时间。
RGB SHA 绑定原始 uint8 像素；MP4 只作预览，不能替代 lossless 训练图像。
可选 `--record-commands <新外部目录>` 接入独立 lossless recorder，关闭归档不自动授予数据验收。
完整回放不因首次成功提前结束，保留所有失败阶段与示范有效尾部；验收需另核任务 predicate。

`--sync-observations` 显式刷新 MuJoCo 积分后 qpos 对应的派生 transforms、接触和速度；
不改变 qpos/qvel 或仿真时钟。Isaac 随后重新同步 bridge，以 Replicator `delta_time=0.0`
重新渲染，不额外将 root/joint 做十进制 round，Isaac/USD 内部表示精度仍适用；
state43 始终从具名 MuJoCo 关节读取。controller 在真实 50 Hz proprioception
采样边界同样刷新后读取 gravity/angular velocity。报告 phase 为
`post_step_mj_forward_same_qpos`；旧默认保留 `legacy_mj_step_derived_phase`，
其 qpos 与 xpos/渲染可能相差一个 5 ms 物理步，旧成功回放不能替代新相位验收。
`run_closed_loop.py` 支持同名选项，后续模型配对评估须与数据采集保持一致。

`--hand-torque-profile recording_0p7_nm` 是独立诊断变量，按固定官方 model12 YAML 与
deploy MJCF 的静态限矩对照将两侧 thumb0 限制为 2.45 Nm，其余手关节限制为 0.7 Nm。
归档缺少逐集采集运行配置 stamp，这个兼容参数名不能证明原采集必经某条 body PD 路径。
默认 `mjcf_limits` 保持现有模型限制，
不得把这个选项与航向或命令节奏实验同时改动后归因。手指 KP/KD、body gains 和模型均不随该选项改变。
`run_closed_loop.py` 接受同名 profile 参数，模型配对测试应显式匹配获验收数据的实际设置。

同一组独立初态比较微调前后模型，保留服务 `/info` 身份并明确检查真实 checkpoint step，例如：

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n cvpr-simple-sim310 \
  python scripts/4090_simple/run_closed_loop.py \
  --data-root "$SIMPLE_DATA_DIR" --sonic-resources "$CVPR_SONIC_DIR" \
  --output "$CVPR_SIMPLE_RUNTIME/reports/carrybox-model-before" \
  --tasks G1WholebodyXMoveBendCarryBoxSonic-v0 \
  --seed 10001 --episodes-per-task 5 --seconds 60 \
  --expected-checkpoint-step 28000 --port 21074 \
  --sim-mode mujoco --pd-profile official_v1_1_ankle_pitch_1_5 \
  --diagnostic-unverified-sonic --video
```

示例 step 28000 必须与所启动服务的真实身份一致；微调后更换对应 step 和独立输出目录。
CarryBox 每集 seed 依次递增，保存完整 reset config；默认三个旧任务列表和 20k 检查不变。
`--video` 以每个实际 50 Hz 相机帧写入 ffmpeg rawvideo stdin，包含温启与终止帧，
不抽帧或复制缺失帧。MP4 编码/收尾错误、非零退出、空视频都会导致结果失败。
推理慢于仿真时暂停仿真等待 localhost HTTP，视频显示仿真时间；wall time 单独记录，不能据视频宣称实时运行。

## 独立 LiftBoxHold 任务

`simple/G1WholebodyLiftBoxHoldSonic-v0` / `g1_wholebody_lift_box_hold_sonic` 复用 CarryBox 的
场景、箱体和控制，单独使用 `lift_box_airborne_hand_contact_v1`：旋转后箱体几何最低点
至少高于固定地板 10 cm，至少一个真实 hand／融合 wrist 掌接触的 normal force >0.01 N，
无 floor/table 接触，在 50 Hz 真实物理观测中连续保持 1 秒。重复查询不计时，漏采间隔或
条件中断重置计时。允许膝部等机器人部位辅助，不要求双手或静止悬停，不要求放桌。
完整 episode 无跌倒和有限状态仍是数据／评估验收的独立条件，任务 predicate 本身不替代它们。
原 CarryBox 的放桌标准、指令与默认入口不变。

新任务统一指令为：
“Lift the box so its lowest point is at least 10 cm above the floor, and maintain hand contact without floor or table support for 1 second.”

```bash
PYTHONNOUSERSITE=1 conda run --no-capture-output -n cvpr-simple-sim310 \
  python scripts/4090_simple/replay_carrybox_native50.py \
  --source "$CVPR_SIMPLE_RUNTIME/resources/simple_finetune_sources/G1WholebodyXMoveBendCarryBoxSonic-v0" \
  --contract "$CVPR_SIMPLE_RUNTIME/resources/simple_finetune_sources/CARRYBOX_STATE_ACTION_CONTRACT.json" \
  --data-root "$SIMPLE_DATA_DIR" --sonic-resources "$CVPR_SONIC_DIR" \
  --task G1WholebodyLiftBoxHoldSonic-v0 --episode 0 --seed 10001 \
  --token-mode feedback-heading-v11 --command-hz 30 --sync-observations \
  --sim-mode isaacsim --isaac-observation-only \
  --pd-profile official_v1_1_ankle_pitch_1_5 --hand-torque-profile mjcf_limits \
  --seconds 60 --video \
  --output "$CVPR_SIMPLE_RUNTIME/reports/liftbox-source0-rtx" \
  --record-commands "$CVPR_SIMPLE_RUNTIME/resources/liftbox_candidates/source0-rtx"
```

`--isaac-observation-only` 是显式可选 RTX 路径：MuJoCo 保持 200 Hz 物理和 50 Hz decoder，
Isaac 只在 50 Hz observation 边界同步并渲染，不调用其物理 step。
按本机 Isaac API 刷新 articulation kinematics、PhysX→USD transformations、无物理积分的
`World.render()` 后以 Replicator `delta_time=0.0` 采集。实际 USD XformCache links／camera、
sensor pose 与 MuJoCo FK 随 trace/events 留痕，不能仅凭非静态 RGB SHA 证明刷新有效。
是否更快、相机是否正确随动，须用实际 GPU 回放比较；旧默认仍保留每物理子步 Isaac step。
固定 G1 USD／MJCF 存在既有 attachment 差：USD 腰链到 torso 为 44 mm，MJCF 为
35+19=54 mm；shoulder 相对 torso 又分别为 247.78／237.78 mm。
因此 torso／相机约差 10 mm，腕部差随腰部旋转变化，不能直接称为时间相位错误。
真实 USD pose 与该静态关节帧差的预测残差已另存运行报告；本任务保留原资产和镜头校准。

EGL 仅作快速物理资格检查。正式新训练采集统一 RTX renderer 合同；不能混用两种渲染域宣称同域效果。
基础设施／渲染检查须在启动前显式 `--qualification-only`，例如 `--seconds 2`，
其结果与正式 `pilot_collection` 尝试分别记录，短检查不能转成训练示范。
Lift 成功后正常 `task_terminated`，原 source 剩余长度与真实 future 边界不改短；
所有实际命令、lossless PNG、完整 video、NPZ 和终止状态保存。NPZ 的
`physics_state_integration`／`physics_state_spec` 保存每个真实观测及 `task_terminal`，
用于独立重建接触力与成功条件。capture 在 Isaac `env.close()` 前封存。

报告绑定新任务指令、task/controller/render 冻结声明 SHA 和 append-only attempt ledger。
`physical_outcome_status` 在清理前固化，录像／封存出错不能把已发生的物理失败改称基础设施失败。
同冻结合同的基础设施重试保留全部失败记录；真实失败不得通过择优重跑或调整标准隐藏。
`run_closed_loop.py --tasks G1WholebodyLiftBoxHoldSonic-v0` 可作微调前后配对，
controller／renderer／任务合同与独立初态必须一致。

失败 source8 仍须保留，并在微调前后覆盖其真实初态。评估客户端可额外指定：

```bash
--tasks G1WholebodyLiftBoxHoldSonic-v0 --episodes-per-task 1 --seed 10009 \
--sync-observations \
--initial-physics-run /home/zxc/CVPR_Evaluation_runtime/reports/liftbox-source8-rtx-v1
```

此选项要求单任务、单 episode、DR0、相同 task／seed。`initial_physics_state.py` 先核对源 MJB SHA，
再逐字段精确比较新环境与 MJB 的静态模型：名字／顺序、关节与 actuator、几何／mesh／碰撞、
质量／惯性、相机、solver 与 timestep 等均须一致；仅排除 MJB reload 会清零的 mjSpec compilation signature。
随后只读取 NPZ 的首个完整 `mjSTATE_INTEGRATION`，在 reset 后向原有 mjData 一次恢复位置、速度、
ctrl、外力、warmstart 和真实时间；保留 model／data 及缓存 view，`mj_forward` 后重新获取当前真实 RGB。
agent／controller／HTTP history 随后清空，继续实际物理与策略控制。
报告保存源 result／MJB／NPZ SHA、首状态与恢复后 SHA、双方静态模型摘要、初始化次数 1。
不会读取后续 teacher 动作或轨迹行，也不复用历史 RGB。未传此选项时仍为标准 reset。

部署需积累一秒真实 RGB 历史。默认 `measured_pose_pd` 在这一秒锁定首帧身体关节目标，
不运行平衡 decoder；它不保证自由浮动机器人站稳。可显式加
`--warmup-control live_pose_sonic_v1_1 --sync-observations`：先采十帧真实 50 Hz proprioception，
随后用固定官方 v1.1 encoder 的 mode0 编码首个 live 关节姿态／世界根朝向保持参考，
零期望关节速度、当前实际 yaw 归一化，继续原有 50 Hz decoder／200 Hz PD，直至图像历史满。
重复的 encoder reference 表示期望保持姿态，不能当作重复采样的相机或 proprioception。
此 consumer 选项不读取 teacher 未来、不改模型、增益、任务标准或冻结的示范采集器；默认行为保留。
报告区分前十帧 PD 与随后 SONIC 预热，保存 helper／encoder SHA、首次 live reference SHA、
实际预热 decoder 次数及每次 token；交接策略时保留真实 SONIC 历史。
CPU MJB／ORT 的 1.5 秒物理检查支持它修复首秒失稳，实际 RTX／学习策略效果仍须单独验证。
它也不自动弥补训练首窗口已开始开手／伸手、而部署仍在站立的状态分布差异。

CPU契约检查：

```bash
PYTHONNOUSERSITE=1 /home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  -m unittest discover -s scripts/4090_simple/tests -p test_simple_control.py -v
```

8项覆盖双手state/action顺序、padding排除、逐帧上报与episode reset、HTTP／425、
真实Gymnasium检查器、真实MuJoCo物理tick与50Hz历史，以及编译固定C++常量的数值对照。
其中确定性decoder仅为CPU测试fixture，绝不作为策略闭环结果。

## 有界真实闭环诊断

先在模型Conda环境单独启动真实20k服务，绑定`127.0.0.1:21074`，
`--action-exec-horizon 30`，首轮关闭RTC。确认服务就绪后，在仿真Conda环境运行：

```bash
source scripts/4090_simple/local.env.example.sh
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python scripts/4090_simple/run_closed_loop.py \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --sonic-resources /home/zxc/CVPR_Evaluation_runtime/resources/sonic_v1_1 \
  --output /home/zxc/CVPR_Evaluation_runtime/reports/closed-loop-mujoco \
  --port 21074 --seconds 5 --sim-mode mujoco --diagnostic-unverified-sonic
```

默认三个Psi0任务各一个固定episode。可用`--tasks`、`--episodes-per-task`和
`--episode-start`缩小或重复，`--sim-mode isaacsim`使用MuJoCo物理加Isaac真实渲染。
`--pd-profile official_v1_1_ankle_pitch_1_5`是显式可选配方。每个任务独立进程串行运行，
防止关闭Isaac SimulationApp后复用失效进程。`--stop-on-fall`可在首个摔倒诊断处结束；
默认继续到有界时间，以保留服务和decoder调用证据，同时将摔倒阶段明确记录为失败诊断。

输出包括每次HTTP／425／reset、每帧30Hz动作索引、50Hz decoder实际调用、状态有限性、
pelvis初末位置／四元数、最小高度／最大倾斜、首次失衡和首次策略命令时刻，以及真实RGB PNG。
如果温启期间已失衡，报告明确排除策略质量归因。任务predicate、真实loop执行与正式评测资格
分开记录；有界诊断不生成正式成功率。
终止步也重新检查物理有限性和摔倒，避免把最后一步倒地错误归因为任务成功。
由于Isaac关闭可能使出错的Python进程返回0，supervisor同时审核保存的完整状态、episode数量、
实际decoder和同步时钟证据；只有进程与报告都满足条件才标记完成。报告审核的两个CPU回归位于
`test_smoke_reporting.py`。

所有HTTP采用同步等待。推理期间**暂停MuJoCo和Isaac推进**，并逐请求核对前后mjData.time相同；
因此慢于仿真帧周期的GPU预测不会偷用旧状态追赶墙钟。报告分别保留仿真秒、墙钟秒和HTTP延迟，
50Hz控制／30Hz动作只表示仿真时间频率，不宣称实现物理实时吞吐。
