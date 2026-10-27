# PICO 遥操作采集：SIMPLE Mobile P&P

本入口单独采集 SIMPLE Benchmark 的 **Mobile P&P**：从第一张桌子抓起 cracker box，
持物移动到第二张桌子，放进其容器后松手。官方任务 ID 为
`simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`，任务 UID 为
`g1_wholebody_locomotion_pick_between_tables_teleop`。
[官方论文表 1 的 Mobile P&P 与任务名对应关系](https://arxiv.org/html/2606.08278v1#S4.T1)。

截至 2026-10-27，资源、真实 CPU 冷启动／站立及旧版移动控制已有独立验证，旧版也收到过有效 PICO 全身追踪。
本次按用户要求把 Mobile 默认控制改为 **`continuous_vr_mode1`：左杆前进、右杆转向，双臂与颈部参考持续跟随人体**，
取消 B 切换、移动时上身锁定和 3 厘米返回姿态门。旧策略以 `locked_upper_mode0` 显式保留。
**本轮 CPU 分配与诊断修订已通过 294 项 PICO CPU 回归及生产几何复核并行的 18 秒空手入口检查。
旧 `live_human_003_continuous_vr1` 的超时证据与原数据保留，已正常退出；
新 `live_human_004_cpu_isolated` 已实际加载并核对源码、planner CPU29 和预览 CPU30。
页面检查时无错误、暂停且未录制，等待头显 Send／X 校准／Y 跟随。真人持物、完整抓放任务与训练数据准入仍未验收。**
下面把新操作方式与旧版证据分开说明。旧 Handover 默认 mode2、原固定 90 条候选及其交付范围保持独立。

## 独立入口与资源

在包含本入口的仓库根目录运行；本次准备工作树为
`/home/zxc/.codex/worktrees/pico-locomotion/CVPR_Evaluation`：

```bash
bash scripts/4090_simple/start_pico_mobile_pp.sh
```

这是后续现场验收与采集的操作入口；启动脚本不安装依赖，也不停止其他采集、测评或 GPU 进程。
它读取已有 `/home/zxc/CVPR_Evaluation_runtime/local.env.sh`，使用
`/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python`，显式选择 `--task mobile_pp`。
默认 `mujoco_deferred` 由独立 MuJoCo EGL 进程提供操作反馈；真实物理与归档状态由控制进程推进。
操作预览与后续同状态 RTX 训练图像分别留证，预览画面不能代替训练 RGB 验收。

本次固定资源根为：

```text
/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027/resources/
  data_root/
  official_train_episode000_reset_environment_config.json
  mobile_pp_cold_start_profile.json
```

使用真实官方 `hssd:scene6 = 102344022` 双桌场景，不替换为 Handover 房间。
`data_root` 通过独立覆盖目录修正官方 USD 资源路径，复用已有机器人、对象和材质字节；
原官方场景、纹理、几何与材质参数保留。资源固定官方
`USC-PSI-Lab/SIMPLE@f5b58ab75210367c71ff7347155a656d403169a8`，
场景归档 SHA 为 `da90862e486eb3f8dff25291923be4d03023afe57ed8aae9c824f20f4fd66338`。

先前下载的官方 Mobile 数据共 99 条、62,764 帧、50 Hz，全部标记为 TRAIN，全部使用该 scene6。
这些旧示范不计入本次真人数量，也不使用其动作替代操作者。
其 metadata 的 `dr_state_dict` 保存初始随机化状态，`layout` 中动态对象却已是录制结束位置。
因此 `official_train_episode000_reset_environment_config.json` 显式从原始 DR 初始位姿重建 reset layout，
保留源文件与派生说明；实际 reset 后必须严格匹配整个 DR＋layout。
该配置 SHA 为 `d4c43a1fe1219f31e2b0bef3f9a2356a00a48b9a694d7f0a34430beff7c71f79`。

冷启动是明确的 Mobile 专用姿态：保留 SONIC 默认腿／腰 15 关节，手臂 14 关节取官方 TRAIN0 第 0 帧
`observation.arm_joints` 的实际观测，手指张开。只允许在首次物理步进前初始化一次，不在运行中改写状态。
它不是 SIMPLE／SONIC 原生默认姿态；未使用动作或后续示范帧，也未移动桌子、物体或修改碰撞／重力／PD。
profile SHA 为 `a865ff08984acc6f43d4a81cceb9c67588e695d9f9d69472b49e563a9034dd43`。

planner 使用既有固定文件：
`/home/zxc/CVPR_Evaluation_runtime/handover_random100_20261024/official-planner-v1/planner_sonic.onnx`。
目录名属于历史资源位置，不代表 Mobile 复用 Handover 任务或数据质量门。
planner 在独立 CPU worker 运行，50 Hz 控制循环只提交／读取有界结果。

## 连续控制与操作顺序

默认控制策略为 `continuous_vr_mode1`，也可以在命令中显式写明：

```bash
bash scripts/4090_simple/start_pico_mobile_pp.sh \
  --mobile-control-policy continuous_vr_mode1
```

这是已通过连续 mode1 短时入口检查的操作方式；本轮 planner CPU 修订的验收范围见下方。
旧锁上身 `collection/live_human_002_v2` 已正常退出。后续 `collection/live_human_003_continuous_vr1`
曾由 `continuous_mode1_001/RELOAD_COMPLETE.json` 核对实际源码身份、mode1 和页面，
之后出现 `planner_request_expired_no_reference`；确认暂停且无活动录制后已请求 quit 并正常停止，原始数据保留。
新 `collection/live_human_004_cpu_isolated` 已加载，`control_timeout_fix_001/RELOAD_COMPLETE.json` 的 `passed=true`，
逐项核对启动源码 SHA 与实际 worker CPU；浏览器只读检查无错误、暂停且未录制，追踪原因是 `no_tracking_packet`。
需要操作者在头显开启 Send，再按 X 校准、Y 跟随；磁盘源码、诊断与现场身份分别留证。

PICO 的 XRoboToolkit 设备设置、单接收器限制、有效全身追踪和电脑预览方式沿用
[基础 PICO 采集说明](PICO_SIMPLE_TELEOP.md#启动与操作)。电脑控制页默认
`http://127.0.0.1:43100/`，只读大画面默认 `http://127.0.0.1:43101/`；端口以实际启动配置为准。
连续策略不绑定 B 的模式切换，页面隐藏该切换按钮。

1. 站直、双脚落地、松开两个摇杆，确认全身追踪有效。按 **X 校准**，再按 **Y 开始跟随**。
   标定绑定此刻机器人实际腕部和躯干 FK，以及同包人体 Body24 数据；标定不会改写物理状态。
2. 双臂参考持续来自同一个有效新包中的 **Body22／23 左右腕和 Body12 颈部**，经 SONIC mode1 跟随。
   移动和停止时都更新这些参考；可边前进／转向边调整双臂。两手的前扳机或同侧侧握键仍分别控制手指闭合，
   取同侧两者的较大闭合量；手指保持独立。先小幅检查实际方向和抓持，关节闭合量不能证明已经夹住物体。
3. 按 **A 开始录制**。起始一秒 warmup 结束后，再从第一张桌子抓起并抬高目标。
   Mobile 不要求右手→双手→左手交接；可使用任意一只手或双手搬运。
4. **左杆向前**控制前进，速度上限 0.3 米／秒；**右杆左右**控制转向。直接使用摇杆，不需要先按 B。
   机器人腿部接受 planner 参考，双臂与颈部参考继续实时更新；松杆发出停止意图，机器人仍需真实减速。
   横移、倒退和右杆纵向未开放，按原逐轴规则屏蔽并保存原始四轴、实际采用四轴及忽略原因；
   有效前进和转向部分分别保留。缺轴、非法幅值、非有限值、过期／逆序输入等保护仍适用。
5. 到第二张桌子后松杆，确认机器人实际停稳，再把目标放入容器，双手彻底松开并保持物体稳定至少一秒。
   **不需要 B 返回、恢复移动前的手臂姿态或满足旧 3 厘米姿态门**；连续策略中双臂一直在跟随。
   停稳仍是实际操作要求，不把松杆等同于已停止或夹持稳定。
6. 按 **A 保存结束**。原始写入封存后重置并暂停，下一条重新 X 校准、Y 跟随。
   **Y 暂停**会中断正在录制的条目并保留原因。页面“丢弃当前／上一条”沿用 UUID 与编号归档，保留原始证据。

mode1 的三点是控制参考，不是未来观测或直接 PD 关节目标。标定时读取真实 `left_wrist_yaw_link`、
`right_wrist_yaw_link` 与 `torso_link` 的世界位姿，按固定官方局部 offset 得到左右腕 tip 和躯干上方参考点；
offset 只应用一次。腕部位置／朝向与颈部朝向都绑定当前实测 FK，颈部采用实测 torso 原点／朝向适配，
不假设机器人初始躯干恒为单位朝向或固定高度。保存源文件 SHA、标定输入、offset、实际 FK 和派生参考。
该适配不修改官方物理、模型或 PD，也不声称三点参考必然在实际机器人上无误差实现。

每个新人体包先去掉**当前人体骨盆**的位置和朝向，再把校准后的局部三点，通过同 tick 的
**当前实际机器人完整 root 平移和旋转**运输到世界坐标，交给 mode1 时使用同一 root 求逆。
因此移动／转向后参考随机器人当前 root 更新，不锚定第一次标定的世界位置，也不把人体房间中的绝对平移当步行命令。
左右腕和颈部来自 Body24，不能用独立 Head／Controller 包替代。

真实物理仍为 **200 Hz**；控制／planner 参考按 **50 Hz** 推进，token 和手指新命令按 **30 Hz** 网格生成与归档。
保留真实积分状态、时间对齐和原始包，不以 planner 未来参考填充真实状态。
异步 planner 启动准备时物理时钟保持暂停，不生成跟随动作；等就绪、收到新的有效人体包且摇杆归中后，再启动连续控制时钟。
追踪无效、时间倒退、planner 结果过期或超过实时延迟预算等仍会暂停并留证，不补造物理帧或复用旧包冒充新输入。
取消旧模式门不放宽身体物理、实时、零外力或数据质量门。

## 显式保留的旧锁上身策略

仅在需要重现旧控制时选择：

```bash
bash scripts/4090_simple/start_pico_mobile_pp.sh \
  --mobile-control-policy locked_upper_mode0
```

该 legacy 策略先以 mode2 跟随当前人体，按 B 进入 mode0 移动，锁定切换时腰／双臂 17 关节参考，
手指仍独立。进入和返回都要求松杆、真实连续站稳至少 0.5 秒（26 个连续 50 Hz 端点）；
返回还要求人体恢复入场姿态，root-local SMPL24 任一点偏差不超过 3 厘米，且人体根倾角不超过 15°。
不匹配时保持移动和录制并提示；通过后按当前机器人朝向重新绑定参考。
这些 B、上身锁与 3 厘米条件只属于 legacy，不应用于默认连续策略，也不等于 decoder 连续性或持物验收。
旧 v1／v2 的诊断与现场中断证据保留在下方历史记录。

## 输出和物理预检查

默认输出与集合目录为：

```text
/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027/collection/
  live_YYYYMMDD_HHMMSS/
  pico_episode_history.json
  collection_records.jsonl
  .pico_episode_review/
  kept/编号/
  discarded/编号/
```

可用 `--output /绝对新目录` 覆盖本次运行路径；集合索引位于该目录的父目录，
因此指定新集合时也应选择独立的父目录，避免与旧 Handover 编号混放。已有输出目录拒绝覆盖。
每条记录保留原始 PICO 包、token／具名手指命令、真实关节和根／对象状态、50 Hz 已解算接触、
200 Hz 外力、实际积分状态以及本次任务、场景、冷启动和 planner 身份。
控制策略、encoder mode、人体参考来源、实测 FK 标定与当前 root 运输证据单独绑定；legacy 的模式切换也留证。

`pico_episode_history.py` 按明确的任务 ID 分派到 `pico_mobile_quality.py`，不会套用 Handover 的手间交接顺序。
Mobile 的新增采集预检查要求：目标被手接触且抬升至少 0.10 米，在同一段连续持物区间内，
目标与机器人根的水平净位移均至少 0.75 米；最终目标水平净位移也至少 0.75 米。
最后 50 个真实 50 Hz 状态中，双手全部释放，至少 49 帧有容器接触，目标最大两点位移小于 5 毫米，
并通过既有承接几何／支撑必要条件、完整时钟、归档和 200 Hz 零外力检查。
这些是本次采集额外质量门，不是官方 benchmark 成功规则；原生成功标志单独保存，不改写。

**`kept` 只表示原始物理预检查通过，`eligible_for_packing` 仍为 false。**
它不表示 RTX 图像、完整独立审计、clean DINO、训练格式或生产读取已经通过。
独立 CPU 后台复核完成后，launcher 只封存并报告 raw 证据，**不自动补图、不上传、不启动训练**。
旧 `audit_pico_teleop_episode.py`、`finish_pico_session.py` 和固定 90 条 Handover 的准入／打包入口
尚不支持本次 Mobile 任务，不能直接套用。完整 Mobile 独立 RTX／DINO 准入仍待实现和验收。

## 连续 mode1 初版的实测结果与范围

CPU 隔离修订前的连续策略证据位于运行根 `continuous_mode1_001/`，本轮新修订证据见下一节。
`cpu_tests_002.log` 记录 **291 项 PICO CPU 回归全部通过，无 skip**；
`CONTINUOUS_MODE1_ENTRY_ACCEPTANCE.json` 的 `passed=true`，绑定两次真实 Mobile 场景、
SONIC encoder／decoder、异步 CPU planner、MuJoCo 物理和 EGL 预览的空手诊断：

- `actual_entry_diagnostic_002`：14 秒，689 个真实 50 Hz tick、413 个新 30 Hz token，
  最大累计 lag 为 **3.8955 毫秒**。面向桌子前进受到场景空间限制，根水平净位移约 **0.0843 米**，
  不能将该结果称为自由行走通过。
- `actual_entry_diagnostic_003`：16 秒，789 个真实 50 Hz tick、473 个新 30 Hz token，
  最大累计 lag 为 **3.8082 毫秒**。先转再前进，实际朝向变化约 **-186.10°**、根水平净位移约 **0.54531 米**；
  最低骨盆高 **0.76586 米**、最大骨盆倾角 **6.4783°**。左右臂真实关节都有运动，
  单关节峰峰变化最大分别约 0.6814／0.6387 rad，覆盖了腿部移动时双臂继续更新的短时接线。

两次实际状态均有限，200 Hz 原始外力全为零，新 token 全部使用 mode1，模式切换事件为零。
从原始包和实测 FK 标定独立重建世界 VR 参考的最大误差均为 **0 米**，token SHA 与命令证据一致。
这验证参考与录制链路的一致性，不代表实际机器人腕部始终零跟踪误差。
两次输入均明确标记为 `SYNTHETIC_MOBILE_CONTINUOUS_MODE1_NOT_HUMAN_NOT_TRAINING`，
是合成空手诊断，不计真人、持物搬运、任务成功或 RTX／DINO 训练准入。

第一次 `actual_entry_diagnostic_001` 在开始跟随之前，因诊断 replay 对重复包重新计算接收主机时间而触发
`host_timestamp_regressed`。已修为一个原始包只绑定一次接收时间，失败目录与修复说明保留；
live 输入验证门没有更改。没有把诊断工具技术失败当作数据质量失败。

旧现场 `live_human_002_v2` 已正常停止，原数据保留；后续 `live_human_003_continuous_vr1` 曾通过 `RELOAD_COMPLETE.json` 核对源码与控制策略身份。
当时浏览器显示“双臂持续跟随，无需B切换”，B 按钮隐藏，桌面与双手预览可见；该加载快照不代表后续无超时，
也不是当前运行状态。003 的后续故障与本轮修订见下节。
本轮短时空手结果与旧 v2 的 268 项 CPU 回归／9 秒 IDLE 诊断分开，后者不验证连续 mode1。
真人的抓取、持物行走／转向、完整放置及独立 RTX／DINO 准入仍需各自实际验收，不能把短时入口检查称作完整任务就绪。

## 异步 planner 超时与 CPU 分配修订

`live_human_003_continuous_vr1` 出现 `MobileControlError: planner_request_expired_no_reference`。
只读诊断保存在该会话的
`fault_diagnosis/async_timeout_readonly_20261027T090346.306376Z/ASYNC_TIMEOUT_DIAGNOSIS.json`。
当时 planner 与后台 review 同用逻辑 CPU31、nice19，预览使用 CPU30；四次超时均落在北京时间
**16:57:51–16:58:36** 的 review 工作区间。已接受计划的 ORT 墙钟耗时中位数从约 **71.56ms** 升至 **133.11ms**。
旧版本没有保存失败请求的完整排队／推理／后处理／调度耗时拆分，因此这些是共享 CPU 和时间重叠的证据，
不能声称已经精确确定每次失败原因、planner 卡死或必然复现 deadline 失败。过期输出仍被正确拒绝。

本轮只对 Mobile 的 planner worker 显式分配不同逻辑 CPU：从本进程可用 CPU 列表选择倒数第三个，
当前机器为 **CPU29**，避开本入口预览的倒数第二个 CPU30 和后台 review 的最后一个 CPU31。
worker 仍是 nice19、ORT 单线程；只修改本入口自己的 worker，不调整其他任务的 affinity 或优先级。
这不保留全机独占 CPU，也不保证物理核心或其他进程没有竞争。可用 CPU 少于三个时记录
`isolation_available=false`，不能声称三个角色已经隔离。

`AsyncPlanner` 新增可选 `cpu_affinity`，未传入时保持原默认行为，其他调用方不因本轮 Mobile 策略自动改绑。
新增排队时间、完整请求墙钟、完整推理墙钟和进程 CPU 时间等诊断，Mobile 在暂停前保存
`last_fault`，包含 pending request、worker 状态、tick 和 CPU 分配快照，避免只留下短暂停止原因。
**Mobile 的 0.20 秒请求 deadline、完整未来参考要求和拒绝过期结果规则保持不变**，不放宽迟延门或复用过期参考。

运行根 `control_timeout_fix_001/CPU_AFFINITY_CONTENTION_COMPARISON.json` 记录真实 planner 的受控 CPU 对照：
在 CPU31 上运行一个受控 busy 负载，planner 同用 CPU31 与改用 CPU29 分别完成 **12／12** 次，
均在 200ms 内交付；完整请求最大墙钟分别约 **141.28ms／68.20ms**。
这支持把本入口 planner 与该竞争负载分开可减少抢占耗时，**未复现现场 deadline 失败**，
该受控 busy 对照没有重放历史生产几何任务，也不能证明全部实时场景都不会超时。

### 生产几何负载并行检查与现场重载

`control_timeout_fix_001/CPU_ISOLATION_ENTRY_ACCEPTANCE.json` 的 `passed=true`。
真实生产 `complete_quality_geometry` 在 CPU31 上只读同一份已封存 **1,700 行 raw**，运行两遍，
分别耗时 **23.56／34.94 秒**，原 raw 与 cache SHA 未改。它与 CPU29 上 18 秒真实 Mobile／SONIC／
异步 planner／EGL 空手控制的实测时间重叠 **99.8855%**；控制记录包含 **890 个实际 50 Hz tick、
534 个新 30 Hz token、103 个不同的已接受规划请求**。这些请求的实际 planner worker 均在 CPU29。

本次无 rejection，实际状态全部有限，新 token 全部 mode1，原始外力全零，ledger 中 token SHA 全部一致；
最大控制 lag 为 **4.431888ms**，最大完整 planner 请求墙钟为 **87.124684ms**，最大排队时间为 **14.607986ms**。
请求期限仍为 **200ms**，没有放宽原门或复用过期结果。本轮 **294 项 PICO CPU 回归全部通过，无 skip**；
独立 review 另运行 19 项 planner 测试（包含真实 spawn 的排队过期与清除旧指标）和 28 项 control 测试，全部通过。
这是同一生产几何算法负载下的短时空手并发检查，仍不证明全部调度条件永不超时，也不计真人或持物任务验收。

原 003 会话在暂停且无录制时正常退出，原记录和故障证据保留。
后续 `control_timeout_fix_001/RELOAD_COMPLETE.json` 单独确认 `passed=true`：
新会话为 `collection/live_human_004_cpu_isolated`，启动源码 SHA 逐项匹配，
实际 planner 子进程 PID3812334 使用 CPU29、预览 PID3812414 使用 CPU30，后台 review 使用 CPU31。
该分配只约束本入口 worker，仍不声称全机或物理核心独占。
浏览器只读核对时无 error、`no_tracking_packet`、暂停、未录制，显示连续双臂说明且无 B 切换按钮；需要头显 Send 后 X／Y；
加载通过不等于操作者已经开始采集，更不代表真人完整持物任务或独立 RTX／DINO 准入完成。

## 共享资源与旧策略的已有验证

下列资源和冷启动证据继续绑定相同文件；移动、切换及入口诊断覆盖当时的旧控制版本，均不验证本次连续 mode1。

资源回执位于运行根 `resources/STATIC_RESOURCES_READY.json`、`RESOURCE_CLOSURE_AUDIT.json`：
场景 CPU 组合 5,144 prim、0 错误，75 个入口及 214 层依赖核对通过，810 个文件身份留证。
这不包含实际 RTX／MDL 运行验收。

`MOBILE_PROFILE_CPU_ACCEPTANCE.json` 与 `cpu_cold_start_006/CPU_COLD_START_RECEIPT.json` 记录上述专用初始姿态：
在真实 Mobile TRAIN0 双桌场景中，首次步进前的初始化次数为 1，初始没有手－桌／目标／容器接触；
1 秒、50 帧站立检查通过，200 个物理子步外力全零，目标最大漂移约 0.509 毫米。
先前默认姿态的手－桌穿入及技术失败回执仍保留，没有放宽物理门或覆盖旧证据。

`control_preflight_attempt002/INDEPENDENT_AUDIT.json` 记录独立的短时空手控制检查：
655 个控制间隔、2,620 个物理子步，实际前进约 0.808 米、右转约 177.6°，随后停止，原有实时与零外力门通过。
该检查没有 PICO SDK、物体、完整任务场景或渲染，没有证明持物稳定或真人操作。
原程序末尾的控制台 JSON 序列化曾报错，已安全写出的原始结果及后续独立核查、说明均保留在同目录，
不把退出码或单次短测直接当成任务成功／完整实时吞吐验收。

`gpu_entry_diagnostic_001/` 另记录 8 秒新完整入口诊断：实际加载 Mobile 场景、专用冷启动、
SONIC 预热、真实异步 CPU planner、MuJoCo EGL 预览，以及 30／50／200 Hz 原始采集链路；
最大累计 lag 约 4.4 毫秒，没有技术拒绝或异常。输入是明确的合成中立诊断，未连接真人 PICO；
未完成抓放任务，结束时也未保存为成功示例，因此不计入真人数据或训练准入。
该结果覆盖完整入口短时接线，不能证明人体模式切换、持物稳定或独立 RTX 训练图像质量。

正式采集就绪仍需本次连续策略的实际 PICO 输入、真人抓取／持物转向／放置，以及独立 RTX 图像等验收。
本入口不改动或启动 A100 训练，后续训练数据交付另行审核。

`gpu_mode_switch_diagnostic_004` 又完成 9 秒合成输入的实际入口往返切换：
第 100 个 50 Hz tick 请求进入移动，第 300 个 tick 经站稳和人体姿态门返回操作模式，
最大累计 lag 约 5.41 毫秒，无技术异常／拒绝。此前 003 的录制输入回放因 NumPy 时间标量
不满足严格输入类型而停止；已将回放主机时间规范为 Python float，并保留失败与修复后回执。
这是 **v1 的空手、松杆 IDLE 往返切换检查**，未覆盖本轮 v2 的轴屏蔽和 mode0 参考范围诊断修订；
未把合成输入、预览或原生成功标志当作真人完整任务通过。
该目录的 `AUDIT004.json` 独立核对了 443 个真实 20ms 控制间隔、266 次新 token、
263 份 capture／ledger／积分状态、438 个质量样本及 1,752 条零外力记录，
并从原始包纯 FK 重算返回姿态门；该 scope 通过，原生任务成功标志全部为 false。

旧 v1 现场入口曾在 `collection/live_human_001` 启动，控制页为 `http://127.0.0.1:43100/`。
已通过实际浏览器核对桌面、目标物和双手预览。检查时页面处于暂停／等待标定，
`tracking.valid=false`、原因为 `no_tracking_packet`，没有开始录制。
需要操作者在头显启用 XRoboToolkit 全身追踪与数据发送；有效追踪后再按 X／Y／A。
该状态仅记录本次启动时的检查，不代表头显反馈或真人任务已经通过。
随后同一接收器已收到有效全身追踪，实际包序号超过 2,000 且状态年龄持续低于 100ms；
当时的启动快照仍是暂停、未标定、零条录制，等待操作者主动按 X／Y／A；它不是后续现场操作的当前状态。


## 历史：2026-10-27 锁上身 v1 中断、v2 修订与重启

北京时间约 15:53，`collection/live_human_001` 的 v1 现场记录发生两次中断，原始证据保留：

- tick 926：左杆出现尚未开放的横移／倒退输入，v1 将整条控制暂停。这是控制入口的方向拒绝，不能当作任务质量失败。
- tick 1775：mode0 planner 的未来关节参考超过 MuJoCo `jnt.range`，采集器将参考错误地当成实际关节／直接控制目标硬拒绝。
  上一 tick 的右踝 roll 实测值在范围内、根高约 0.766 米且状态有限；`POLICY_TO_BODY` 重排已核对正确。
  该中断不能解释成已经测得机器人关节越界或跌倒。

`live_human_001/fault_diagnosis/LIMIT_GUARD_1775_CPU_REPRODUCTION.json` 从绑定输入链精确复现
82 个已接受计划的输入／输出 SHA；重建后续候选计划时，未来 0.1 秒的右踝 roll 参考为
`-0.263297856 rad`，仅比下界 `-0.2618 rad` 超出约 `0.001498 rad`（0.08582°），
而当时实关节约为 `-0.083187145 rad`，处于范围内。原失败 1751 输入没有归档，
因此这是确定性重建候选，不能冒称后续候选原输出 SHA 已由现场核定。

mode0 的 1751 维输入是 encoder 条件参考，经 SONIC encoder／decoder 后才生成实际控制，
未来参考超出 MuJoCo 软关节范围与实际机器人状态越界是不同事件；官方链路也没有对该参考施加这项硬拒绝。
当时的 v2 修订保留 mode0 身份、输入形状和有限性校验，把 `jnt.range` 比较改为完整诊断证据：
记录每个参考帧、具名关节、原参考值、实际模型范围和超出量，不裁切参考、不因该项超出直接拒绝整条，
也不把参考数组直接送成 PD 目标。真实物理、时钟、追踪、有限性和数据质量门保持原样。
相关证据使用 `cvpr-sonic-mode0-reference-range-diagnostics-v1`，它与移动控制 contract 的版本号分别记录。

旧锁上身移动控制的 contract 升为 `cvpr-pico-mobile-frozen-upperbody-async-reference-v2`，与新连续 mode1 contract 分开。
除前述参考诊断外，v2 对有效、及时且范围合法的摇杆包逐轴屏蔽未开放方向：`left_axis_x=0`、
`left_axis_y=max(0, 原值)`，保留右杆横向转向，未绑定的右杆纵向不参与控制；
纯横移／倒退意图成为停止／IDLE 意图，混合前进和横移仍保留有效前进部分，不再因方向未开放整条暂停。
原四轴、屏蔽后四轴、被忽略的轴及原因都保存；`applied_axes_left_x_y_right_x_y` 是方向屏蔽后、
deadzone 判定前的数值，最终控制意图见 `desired_command`。缺轴、非有限值、非法幅值、过期／逆序输入等检查不放宽。
移动准备完成但摇杆尚未归中时继续静态保持并提示松杆，避免首移动 tick 因预先推杆中断录制。
真正暂停时，中文状态显示暂停及具体原因，避免页面继续停留在“正在移动”的旧提示。

上述旧锁上身 v2 修订当时已按用户“重启”的明确授权替换现场进程；代码身份为 `711d2d0`。
`gpu_mode_switch_diagnostic_004` 仍只证明旧 v1 的 IDLE 切换 scope，不作为新版本证据。
当时已有 268 项 PICO CPU 回归全部通过，包含实际页面 JavaScript 状态测试、
越界参考原样送入 encoder、非有限参考拒绝、方向屏蔽和准备阶段提前推杆等待；没有跳过项目。
完整真人持物、任务成功和 RTX／DINO 训练准入结论仍保持待验收。

该次旧策略重启回执位于运行根 `restart_v2_001/RESTART_COMPLETE.json`：旧 `live_human_001`
通过控制接口正常退出，原日志／故障证据保留，没有修改 PC 接收服务或 A100 任务。
新会话为 `collection/live_human_002_v2`，启动源码 SHA 与上述提交逐项一致，
planner 预热及实际浏览器预览通过，检查时无错误、未录制、暂停等待标定，追踪为 `no_tracking_packet`。

`restart_v2_001/actual_entry_diagnostic_001/V2_ENTRY_ACCEPTANCE.json` 记录新版本的实际场景、
EGL 和 SONIC 入口检查：9 秒、443 个真实控制间隔、266 次新 token，最大累计 lag 为 6.43ms；
提前推杆期间仍保持实测入场姿态，横移／倒退分别在12个新命令中被屏蔽为新的 IDLE 意图，
之后经站稳／人体姿态门返回操作模式，未出现中断。所有 mode0 命令均携带参考范围诊断且未裁切参考。
输入明确标记为合成诊断，不订阅真人 SDK，也不计入真人记录；它只覆盖空手 IDLE 往返和输入恢复，
没有证明场景内持物行走／转向、任务成功或训练图像准入。
