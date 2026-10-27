# PICO 遥操作采集：SIMPLE Mobile P&P

本入口单独采集 SIMPLE Benchmark 的 **Mobile P&P**：从第一张桌子抓起 cracker box，
持物移动到第二张桌子，放进其容器后松手。官方任务 ID 为
`simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`，任务 UID 为
`g1_wholebody_locomotion_pick_between_tables_teleop`。
[官方论文表 1 的 Mobile P&P 与任务名对应关系](https://arxiv.org/html/2606.08278v1#S4.T1)。

截至 2026-10-27，已完成资源、真实 CPU 冷启动／站立、短时空手移动、完整入口诊断和实际预览启动；
**已收到本次 PICO 有效全身追踪并进行现场操作，v1 出现两次中断；持物移动、完整抓放任务及训练数据准入仍未验收。**
本轮 v2 源码修订尚未通过实际 GPU 入口验收，也未重启到现场进程，不能视为已生效或完整修复。
本页记录新入口与已有证据，后续实际验收另行补充。旧 Handover 入口、原固定 90 条候选及其交付范围保持独立。

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

## 操作顺序

PICO 的 XRoboToolkit 设备设置、单接收器限制、有效全身追踪和电脑预览方式沿用
[基础 PICO 采集说明](PICO_SIMPLE_TELEOP.md#启动与操作)。本入口的 **B 专门切换移动模式**，
不能照搬旧 Handover 文档中“B 未绑定动作”的说明。
电脑控制页默认 `http://127.0.0.1:43100/`，只读大画面默认 `http://127.0.0.1:43101/`；
端口以本次实际启动配置为准。

1. 站直、双脚落地、松开两个摇杆；先确认全身追踪有效。按 **X 校准**，再按 **Y 开始当前姿态跟随**。
   姿态模式使用当前人体参考，经固定 SONIC v1.1 SMPL mode 2 跟随手臂／身体姿态。
2. 左右前扳机或同侧侧握键分别控制左右手指闭合，取两者较大闭合量；两只手保持独立。
   先小幅检查方向和真实抓持。关节闭合量不能证明已经夹住物体。
3. 按 **A 开始录制**，看到起始一秒 warmup 结束后，再从第一张桌子抓起并抬高目标。
   Mobile 不要求右手→双手→左手交接；可使用任意一只手或双手完成搬运。
4. 保持合适的持物姿态，松杆并站稳后按 **B**。等页面显示移动模式准备完成，再操作摇杆。
   移动期间，上身的腰／手臂 17 关节参考保持切换时的实际姿态，身体上身不继续跟随人的新动作；
   双手手指仍分别接受实时扳机／侧握输入。保持参考不保证实际夹持不滑落。
5. **左杆向前**控制前进，速度上限为 0.3 米／秒；**右杆左右**控制转向。
   松杆发出停止意图，机器人仍需真实减速。倒退与横移仍未开放；现场 v1 对相应输入会整条暂停。
   待验收 v2 改为逐轴屏蔽：左杆横向置零、向后的纵向置零，前进和右杆转向分别保留，
   原始四轴、实际采用四轴及忽略原因均留证。这不表示已支持倒退／横移，也不表示当前现场已更新。
6. 到达第二张桌子后松开两个摇杆，等待机器人实际连续站稳至少 **0.5 秒**，
   同时把身体和手臂恢复为进入移动时的姿态，再按 **B** 返回姿态跟随。
   该条件在进入和退出移动模式时都检查：26 个连续 50 Hz 端点覆盖整半秒，追踪有效、身体直立且速度受限。
   半秒从满足真实停止条件开始计算，不是从松杆那刻简单倒计时；准备中或起录 warmup 内不能切换。
   返回门还比较进入移动时与当前人体的 root-local SMPL 24 点，任一点位置偏差都必须不超过 **3 厘米**。
   同时要求当前人体根倾角不超过 15°，防止整体倾斜绕过上述已去根旋转的点云检查。
   不匹配时只提示恢复姿态，继续保持移动模式和当前录制，不会擅自切回或因此丢弃示例。
   这是保守的切换门，不代表 decoder 输出连续性或物理持物稳定已经通过验收。
   通过后按当前机器人朝向和当前人体／手腕重新绑定参考，保留走动后的朝向。
7. 在姿态模式把目标放进第二张桌子的容器，双手彻底松开，让物体稳定至少一秒，然后按 **A 保存结束**。
   原始写入封存后重置并暂停；下一条重新按 X 校准、Y 跟随。**Y 暂停**会中断正在录制的条目并保留原因。
   页面“丢弃当前／上一条”沿用 UUID 与编号归档，不删除原始证据。

模式切换条件未满足时，页面会说明原因，不能据此认为 B 已经执行成功。
追踪无效、时间倒退、planner 结果过期或超过实时延迟预算会暂停并留证，不补造物理帧或旧输入。
移动模式准备期间仍推进真实物理，使用进入时实测姿态的保持参考，不把新的身体动作混入该参考；
原始状态、命令和时钟继续留证，等待 planner 也不能绕过原有延迟检查。
待验收 v2 在准备中提前推杆时继续上述静态参考并提示松杆，不进入首个移动控制点，也不因此结束当前录制。

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
200 Hz 外力、实际积分状态以及本次任务、场景、冷启动和 planner 身份。模式切换也独立留证。

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

## 已有验证与范围

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

正式采集就绪仍需本次实际 PICO 输入与模式切换、真人抓取／持物转向／放置，以及独立 RTX 图像等验收。
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

现场入口已在 `collection/live_human_001` 启动，控制页为 `http://127.0.0.1:43100/`。
已通过实际浏览器核对桌面、目标物和双手预览。检查时页面处于暂停／等待标定，
`tracking.valid=false`、原因为 `no_tracking_packet`，没有开始录制。
需要操作者在头显启用 XRoboToolkit 全身追踪与数据发送；有效追踪后再按 X／Y／A。
该状态仅记录本次启动时的检查，不代表头显反馈或真人任务已经通过。
随后同一接收器已收到有效全身追踪，实际包序号超过 2,000 且状态年龄持续低于 100ms；
当时的启动快照仍是暂停、未标定、零条录制，等待操作者主动按 X／Y／A；它不是后续现场操作的当前状态。


## 2026-10-27 现场中断与待验收 v2 修订

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
待验收修订保留 mode0 身份、输入形状和有限性校验，把 `jnt.range` 比较改为完整诊断证据：
记录每个参考帧、具名关节、原参考值、实际模型范围和超出量，不裁切参考、不因该项超出直接拒绝整条，
也不把参考数组直接送成 PD 目标。真实物理、时钟、追踪、有限性和数据质量门保持原样。
相关证据使用 `cvpr-sonic-mode0-reference-range-diagnostics-v1`，它与移动控制 contract 的版本号分别记录。

本轮移动控制 contract 升为 `cvpr-pico-mobile-frozen-upperbody-async-reference-v2`。
除前述参考诊断外，v2 对有效、及时且范围合法的摇杆包逐轴屏蔽未开放方向：`left_axis_x=0`、
`left_axis_y=max(0, 原值)`，保留右杆横向转向，未绑定的右杆纵向不参与控制；
纯横移／倒退意图成为停止／IDLE 意图，混合前进和横移仍保留有效前进部分，不再因方向未开放整条暂停。
原四轴、屏蔽后四轴、被忽略的轴及原因都保存；`applied_axes_left_x_y_right_x_y` 是方向屏蔽后、
deadzone 判定前的数值，最终控制意图见 `desired_command`。缺轴、非有限值、非法幅值、过期／逆序输入等检查不放宽。
移动准备完成但摇杆尚未归中时继续静态保持并提示松杆，避免首移动 tick 因预先推杆中断录制。
真正暂停时，中文状态显示暂停及具体原因，避免页面继续停留在“正在移动”的旧提示。

上述均为 **待实际验收、未替换现场进程的 v2 源码修订**。`gpu_mode_switch_diagnostic_004` 只证明旧 v1 的
IDLE 切换 scope，不作为新版本已经通过的证据。当前没有完成 v2 实际 GPU 入口检查或现场重启；
只能在用户明确同意重启当前任务后替换进程，运行中的旧进程不会热更新。
本轮已有 268 项 PICO CPU 回归全部通过，包含实际页面 JavaScript 状态测试、
越界参考原样送入 encoder、非有限参考拒绝、方向屏蔽和准备阶段提前推杆等待；没有跳过项目。
完整真人持物、任务成功和 RTX／DINO 训练准入结论仍保持待验收。
