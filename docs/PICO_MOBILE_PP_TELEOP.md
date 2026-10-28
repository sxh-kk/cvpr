# PICO 遥操作采集：SIMPLE Mobile P&P

本入口单独采集 SIMPLE Benchmark 的 **Mobile P&P**：从第一张桌子抓起 cracker box，
持物移动到第二张桌子，放进其容器后松手。官方任务 ID 为
`simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`，任务 UID 为
`g1_wholebody_locomotion_pick_between_tables_teleop`。
[官方论文表 1 的 Mobile P&P 与任务名对应关系](https://arxiv.org/html/2606.08278v1#S4.T1)。

截至 2026-10-27，资源、真实 CPU 冷启动／站立及移动控制已有独立验证，并已完成本轮真人采集。
本次按用户要求把 Mobile 默认控制改为 **`continuous_vr_mode1`：左杆前进、右杆转向，双臂与颈部参考持续跟随人体**，
取消 B 切换、移动时上身锁定和 3 厘米返回姿态门。旧策略以 `locked_upper_mode0` 显式保留。
用户在采够后要求处理当前 100 条并上传 A100，已从 121 条编号记录冻结 100 个 selected UUID；
`live_human_005_native_only` 已正常退出后才开始 RTX，固定清单不追加或替换样本。
固定 100／100 条现已通过 RTX、10 阶段独立审计、无损转换、61,806 帧真实 clean DINO 及 300 次本地 CPU 生产查询，
实际训练窗口为 55,906；无拒收或替换。A100全部字节／文件身份、路径重绑及300次实际CPU生产查询已通过，数据交付完成，
见[固定 100 条核定与交付](PICO_MOBILE_PP_DATASET_ADMISSION.md)。
用户随后授权：待实际 A100 READY 后用独立新 run 沿用上次 Handover114 配置，joint38000 仅模型初始化、
fresh step0、8卡×batch8、10,000步；4,000起每500步保存13份模型，保留2份完整恢复状态，验证间隔仍1,000。
历史 Handover 的5条／40固定窗口仅作回归监控，Mobile100全部 train_only，没有 Mobile 验证集成绩。
新训练入口已通过独立review及38项CPU回归并单独部署；首轮root盘容量门在GPU启动前拦截，旧证据保留，不计数据拒收或GPU训练失败。
数据盘独立`run_001`于10月28日00:46:39启动8卡训练，08:02:16完成10,000步／returncode0，13份模型全部保存、末步GPU健康检查通过。
末训练batch loss为0.4521014392375946；历史Handover回归loss为10.945638656616211，不是Mobile验证成绩。
13份模型身份及9500／10000完整恢复状态CPU最终验收已通过；未做GPU恢复训练或任务成功率验收，详见[完整训练结果与验收](PICO_MOBILE_PP_DATASET_ADMISSION.md#用户后续授权的独立训练完整训练结果与验收)。

本轮 CPU 分配与诊断修订已通过 294 项 PICO CPU 回归及生产几何复核并行的 18 秒空手入口检查。
旧 `live_human_003_continuous_vr1` 的超时证据与原数据保留，已正常退出；
历史会话 `live_human_004_cpu_isolated` 曾实际加载并核对源码、planner CPU29 和预览 CPU30。
当时加载页面快照无错误、暂停且未录制；操作者随后继续录制，之后 collector 因磁盘 ENOSPC 自行退出，随后按用户授权恢复为 `live_human_005_native_only`。
早期仅官方 native 筛选已应用到冻结的 001–026：18 条保留、3 条官方未成功、4 条技术阻塞、1 条人工丢弃意图保留；26 个完成 job 缓存已刷新。
未封存的第 27 次尝试另保留为技术故障。这些早期回执不代表后来固定 100 条的独立 RTX／DINO 或训练数据就绪。
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
5. 到第二张桌子后松杆，确认机器人实际停稳，再把目标放入容器并松手，观察原生成功标志。
   **不需要 B 返回、恢复移动前的手臂姿态或满足旧 3 厘米姿态门**；连续策略中双臂一直在跟随。
   这些是操作建议，任务成功只依下文绑定的官方 native 结果，不增加位移、抬升或稳定时长门。
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
取消旧模式门不放宽实时、有限性和零外力等技术保护；任务成功筛选遵循下文的 native_only 策略。

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

## 输出与官方成功筛选

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

`pico_episode_history.py` 按明确任务 ID 分派到 `pico_mobile_quality.py`。
按用户最新要求，Mobile 的筛选策略改为 **`selection_policy=native_only`**，版本为
**`mobile-native-task-success-v2`**：技术证据完整时，只判断录制中是否曾有原始布尔
`official_task_success=true`。该字段来自实际环境返回的 `terminated`，不重算或改写官方标志。
原 0.75 米位移、10 厘米抬升、持续持物、额外末段释放／稳定窗口、容器几何／承重等条件
全部退出任务成功筛选；Handover 交接顺序也不适用。旧测量可保留为诊断，不能继续否决官方成功。

### 绑定的实际官方语义

实际使用的任务源码为：

```text
/home/zxc/CVPR_Evaluation/VLMExpert/third_party/SIMPLE/src/simple/tasks/g1_wholebody_locomotion_pick_between_tables_teleop.py
```

本批元数据中的 `actual_task_source_sha256` 为
`fee537d56e134b0399ff147013438168601498ab9b3ee068658dcad5fb8ca521`，
`actual_native_success_criteria=0.9`；筛选必须核对本条实际绑定，不能替换为镜像中另一阈值或推测的任务版本。
任务依据原始容器接触判定和无手接触条件累积 reward，每次满足加 0.02，否则归零，
`check_success` 比较累计 reward 是否达到 `.9`。容器判定沿用该源码的原始布尔表达式，
不把更严格的理想容器几何另行冒充官方实现。

当前实际 `SonicLocoManipEnv.step` 先调用 `compute_reward`，再调用 `check_success`，而后者又调用一次
`compute_reward`，因此同一 50 Hz 步实际执行两次奖励更新；条件持续满足时每步累计 0.04，
从零开始约 23 步就可达到 `.9`。**`.9` 是累计阈值，不能解释为额外要求持续 0.9 秒。**
本轮不修改 SIMPLE 任务、奖励或环境源码，保留它们实际运行的语义。

环境每步直接令 `_success=terminated`，并非一旦成功永久为真的 latch。
官方 `cli/eval.py` 在 `terminated or truncated` 时退出，再读取 `_success`。
遥操作为方便继续操作和保存，在某步 terminated 后仍可继续物理步进；因此
`any(official_task_success)` 表示 **“曾达到官方终止成功”**，既不要求最终一帧仍为 true，
也不声称官方环境具有 latch。每个样本必须是原始 bool，缺失或无效时不能用 truthy 值或补造 false 代替。

### 技术完整性与旧记录重筛

原始 capture 封存、文件身份与 SHA、30／50／200 Hz 时钟、状态有限性、原始外力等技术完整性继续验证。
完整性异常须记录为技术证据不完整／技术失败并保留原件，不能冒充“官方任务未成功”；
技术证据未知时，核定后的官方成功结果也保持未知。只有证据完整且原始 native 从未为 true，
才能按官方任务未成功筛除。人工丢弃标记仍保留，不因改规则自动复活人工丢弃记录。

本轮重筛保持原 **UUID 和编号**，将旧集合回执、旧索引／复核结果保存为 revision archive，
然后以新版本记录重筛结果；原始录制、native 布尔、metadata 和旧证据不覆盖。
已完成的后台 job 与 `EpisodeHistory.remember` 会返回旧缓存，不能仅重新 submit 就声称重筛完成；
需显式更新新旧规则的回执和队列 completed／latest 结果。后台 Python worker 的导入也不会因磁盘改代码自动更新。

规则更新不重启或改动 collector／SDK／控制循环；review worker 只允许在 **job 边界**轮换，不中断正在处理的 job。
本次重筛与缓存刷新已完成，范围和异常分类见下面的实际回执；后续 collector 的 ENOSPC 自行退出单独留证，不能解释为规则更新停止了采集。

**在新策略下，`kept` 表示官方 native 曾成功且技术完整性通过，`eligible_for_packing` 仍为 false。**
它不表示 RTX 图像、完整独立审计、clean DINO、训练格式或生产读取已经通过。
独立 CPU 后台复核完成后，launcher 只封存并报告 raw 证据，**不自动补图、不上传、不启动训练**。
本轮 `audit_pico_teleop_episode.py` 已增加显式 Mobile 派发和 `pico_mobile_audit.py` 独立检查，
转换、DINO 生产读取和上传入口也使用 Mobile 专属合同。`finish_pico_session.py` 及旧固定 90 条 Handover 计划
仍不能作为本批 Mobile 的自动后处理入口。用户另行授权的固定 100 条处理、实际验证及就绪门见
[Mobile 数据核定与交付](PICO_MOBILE_PP_DATASET_ADMISSION.md)；单凭集合 `kept` 不会跳过这些阶段。

### 001–026 的实际重筛结果

本次显式冻结原编号 **001–026**，revision 为 `f35b782f-0bab-4910-85ce-c84de56e8d43`。
CLI 重筛回执 `native_reselection_001/applied_receipt.json` 的 `status=committed`，
`review_cache_refresh.refreshed=true` 且 `updated_results=26`。原 UUID 和编号保持，结果为：

- **18 条 `selected`／`kept`**。其中 002、004、006、010、011、020、023 共 **7 条**由旧附加门的丢弃恢复为保留。
- **003、015、017**：完整原始证据中官方 native 从未成功，状态 `official_task_not_successful`。
- **001、016、025、026**：录制中断或原始 capture 不完整，状态 `technical_evidence_incomplete`／`technical_blocked`，官方核定结果未知，不能算任务失败。
- **021**：原生成功，但已有独立的操作者丢弃意图，保留为 `operator_discarded`；不计质量失败，也不自动恢复。

旧集合索引、journal 和每条旧回执保存在：

```text
/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027/collection/
  reclassification_archive/f35b782f-0bab-4910-85ce-c84de56e8d43/
```

重筛运行回执为
`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027/native_reselection_001/applied_receipt.json`；
完成 job、队列和页面摘要旧字节另在 `.pico_episode_review/reselections/同revision/` 留存，26 条新摘要已显式刷新。
新 review worker PID3909051 使用 CPU31、nice19、隐藏 GPU，三个筛选源码 SHA 与 `mobile-native-task-success-v2` 匹配并已就绪；
空闲后会按原机制正常退出，不把持久常驻当就绪要求。本轮 330 项 PICO CPU 与真实 Node 页面 JavaScript 路径验证通过，另有 2 项 CLI 验证通过。
独立 `native_reselection_001/post_apply_verification.json` 已通过：
**364 个原始文件、共 5,235,085,470 字节的 SHA 保持不变**，26 份旧集合回执 SHA 核对通过，
原编号、UUID alias 与26个队列缓存的新分类／路径逐条对应，无 pending 事务。
未封存第27次尝试的全部10个原有文件 SHA 也保持不变。CLI父侧2项实测通过（0.544秒），总验证为330＋2项。
这些结果不把 `kept` 升为训练数据就绪，仍没有自动 RTX、DINO、上传或训练动作。

### 磁盘写满与未封存尝试

`live_human_004_cpu_isolated` 后续因 `/home` 写满而发生 `OSError: ENOSPC`，collector **自行退出**，
43100 连接拒绝；退出由磁盘满引起，用户随后授权恢复前未重启。第 27 次尝试 UUID
`f0cdd5fa-3c2f-4791-9c25-79c96ae11d9e` 没有封存 `episode.json`，
`capture_start_receipt.json`、`deferred_capture/capture.json` 和 `deferred_capture/states.npy` 为 0 字节。
该原件完整保留在原 UUID 目录，**未正式编号，也不纳入冻结 001–026 或官方任务失败统计**；
不能用已有零散 raw 行补造封存或宣称成功数据已可用。

为恢复写入空间，只将 13 个既有 uv 旧 `.tmp` 缓存保留迁移到 `/tmp`，在原路径建立符号链接，释放约 **2.1GB**；
没有删除采集数据。磁盘恢复不等于未封存尝试已恢复完整，也不代表 collector 已恢复运行。

### 用户授权恢复采集

用户随后明确要求恢复，已在独立新目录 `collection/live_human_005_native_only` 重新启动同一官方场景、连续 mode1 与原有 CPU 分配，
未覆盖旧目录或沿用旧标定。`restore_collection_005/RESTORE_COMPLETE.json` 的 `passed=true`：
实际启动源码 SHA 全部匹配，planner CPU29、preview CPU30、native v2 review CPU31／nice19，
页面和相机预览已恢复，全身追踪有效且操作者已重新进入 following，无 error。
随后新保存的 **027–029** 已由新后台使用 `native_only`／`mobile-native-task-success-v2` 判为 selected/kept，
这是新增现场记录，不属于之前冻结重筛的 001–026。

恢复时另清理约3.2GB可再生成的Omniverse编译纹理缓存，原始纹理资源与采集数据均未删除。
恢复检查观测到 `/home` 可用约338GB；不把全部容量变化归因于本次缓存清理。
旧未封存尝试仍保留为技术故障，未补造记录，也未占用新编号027。

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
真人示例的任务成功现按下文 native_only 新回执核定；上述合成空手诊断不能替代真人 native 记录，独立 RTX／DINO 准入另行验收。

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

这些旧入口检查本身不包含真人任务成功；本批真人的任务选择按 native_only 重筛，独立 RTX 图像及训练数据准入仍需另行验收。
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
当时尚未核定真人任务成功或 RTX／DINO 训练准入；当前真人任务选择以 native_only 新回执为准。

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
