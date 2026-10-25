# PICO 4 → SONIC v1.1 → SIMPLE 仿真采集

本入口用于人在 PICO 中操控本机 SIMPLE G1 Handover 场景，采集独立人类示范。
不启动模型推理服务、不访问 A100，也不使用现有数据集的动作来替代操作者。
运行环境是已有 Conda `cvpr-simple-sim310`，Python 3.10.22、MuJoCo 3.3.6、
Isaac Sim 4.5、ONNX Runtime 1.23.2。所有模型、资产与录制文件留在本机。

## 启动与操作

电脑 `172.16.20.42` 与 PICO `172.16.20.83` 位于同一局域网。
PICO XRoboToolkit 1.1.1 中开启 Head、Controller、Full-body、Send；采用两个手柄，
不切换到手部追踪。两个脚踝定位器需在 PICO 系统完成标定。Hand 不需要开启。

```bash
cd /home/zxc/CVPR_Evaluation
bash scripts/4090_simple/start_pico_teleop.sh
```

默认资源读取 `/home/zxc/CVPR_Evaluation_runtime/local.env.sh`，不修改 Conda 环境。
输出为 `/home/zxc/CVPR_Evaluation_runtime/pico_human_demos/时间戳/`。
电脑控制面板为 `http://127.0.0.1:43100/`，只监听本机；端口冲突时传 `--port`。
可用 `--output /绝对新目录` 指定录制位置，拒绝覆盖已有目录或写入 Git。

需要在4090主机显示大画面时，另开终端启动只读预览：

```bash
cd /home/zxc/CVPR_Evaluation
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/pico_desktop_preview.py
```

打开 `http://127.0.0.1:43101/`，可点击“全屏画面”。它复用采集器实际左眼 JPEG，
显示追踪、标定、跟随与录制状态；不创建新的 SDK 订阅、仿真或录制器。
默认画面为 MuJoCo EGL 操作反馈，训练用 RTX 图像仍在退出后补渲染。
采集器使用其他端口时传 `--collector-port`，预览自身端口可用 `--port` 修改。
PICO 仍须保持 XRoboToolkit 前台及 Send 开启，电脑看画面不需要打开头显 Camera。

1. 戴好头显、站直、双脚落地，双手处于自然可跟踪姿态。先检查控制面板全身数据有效。
2. 按左手柄 **X：校准**。校准使用实际机器人朝向和六个腕部关节；倾斜起始姿态会拒绝。
3. 按左手柄 **Y：开始跟随／暂停**。先小幅左右移动双臂、屈伸腿，检查左右、方向及抓持。
4. 左右**前扳机分别控制左右手抓紧程度**，同侧侧握键也可控制；本机闭合量取两者最大值。
   使用固定 SONIC v1.1 `middle-close` 对握姿态，按具名关节从全零张开姿态插值；
   每次下发仍检查真实关节范围。该映射与旧 SIMPLE `close_qpos` 不同，单独记录版本与源码SHA。
   电脑页面显示原始扳机／侧握、已下发闭合量和实际关节闭合量。实际闭合量只是关节投影，
   不代表手已经接触或夹住物体；暂停时“下发”显示未下发，不能把输入变化当作已执行。
5. 按右手柄 **A：开始录制**。等进入录制后先保持姿势一秒，看到 warmup 结束再开始抓取。
   完成右手抓起 → 双手交接 → 左手持物 → 放到容器并松手，至少稳定保持一秒。
6. 再按 **A：保存待补图／验收示例**。保存不代表成功。电脑面板可丢弃、重置场景及退出。
   重置只在示例之间进行，并清空控制器历史与标定；下一条需要重新校准。
7. 在面板选择退出，实时采集结束后，启动脚本自动串行补渲染 RTX 图像并独立验收。
   不在下一条实时操作过程中运行昂贵的 RTX 补图。中断或重启后，可单独恢复后处理：

```bash
source /home/zxc/CVPR_Evaluation_runtime/local.env.sh
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/finish_pico_session.py \
  /home/zxc/CVPR_Evaluation_runtime/pico_human_demos/某次运行
```

**B 未绑定仿真动作，也不能作为 XRoboToolkit 1.1.1 的相机打开入口。**
启动时传 `--headset-video` 可启用 TCP 13579 双目 H.264 服务；默认只在电脑查看，
不启动头显回传。按已核对的官方
1.1.1 客户端，Camera → Remote Vision → Listen 会请求 Camera Source IP，
填写电脑局域网地址 `172.16.20.42`，关闭 MV-HEVC 模式；此窗口不填写端口。
只有实际在头显中看到正确左右眼画面后，才算头显反馈验收完成。
当前可直接使用上述电脑大画面进行显示，实际头显回传仍未验收。

XRoboToolkit 的 PC SDK 只支持一个反馈消费者；不要同时启动上游 PicoManager、
Unity Demo 的 SDK 订阅或其他接收诊断。本入口不结束这些外部进程，只取得一个本入口
锁；外部消费者仍可能抢走数据。断流会暂停并拒收当前示例。若 Working 却没有新数据，
已观察到完全退出并重新打开头显应用可以恢复注册；不能把界面 Working 当成输入验收。

## 控制与时间协议

- 原始单包回调约 90Hz；每包检查 Full-body 24 点、Controller 模式、focus、Head 状态、
  单位四元数、连续时间及异常跳变。未验证的 PICO `va/wva` 不用于控制或监督。
- 固定官方 SONIC v1.1 **SMPL encoder mode 2**，输入 1751、输出 64。
  身体参考采用官方 SMPL FK；十帧是明确的当前期望姿态保持，不是实测未来。
  编码器、解码器、配置、人体骨架文件及对应源码均绑定 SHA。
- 当前模式编码腿部姿态与根方向，不编码人的绝对根平移。行走、转身、够取范围及
  抓持稳定性需要实际操作者验证，不能从接口测试推断任意步行能力。
- MuJoCo 为 200Hz 真实重力物理；decoder 从真实仿真状态以 50Hz 构建自身历史。
  30Hz 更新 token／手指，其余控制点保持命令，decoder 仍每 20ms 执行。
- 控制页与电脑大画面均通过 `/stream.mjpg` 连续显示最新真实左眼帧；状态／按钮文字独立每500ms刷新，
  不再用500ms状态轮询更新画面。暂停时不重复发布同一仿真时刻的图像；浏览器断流会重连。
  每个JPEG后立即补齐下一MIME段的分界与图像头，保证暂停只有一帧时也能显示；
  已用实际浏览器确认首帧640×360，不只检查HTTP能收到JPEG字节。
  20秒真实EGL／SONIC合成开合诊断中，HTTP实收289个不同帧／10.01秒，约28.77fps；
  同时完成1000个控制步，最大累计延迟19.94ms。此为预览吞吐实测，不是任务成功率。
  控制过程每秒只写有界 `live_status.json`；完整 `result.json` 在暂停和退出时刷新，
  避免随历史示例增长的整份结果序列化挤占实时预算。原始命令、状态与外力记录仍完整保留。
- 相邻控制步偶尔读到同一个PICO包时，在原100ms累计迟延限额内等新包；坏包／无效追踪仍立即拒收，
  超时仍暂停。不会把旧包或重复时间戳当成新的30Hz样本；等待期间读到的X／Y／A短按也会锁存。
- 默认 `mujoco_deferred`：操作者看到真实 MuJoCo EGL 双目预览。具名关节状态、
  完整 `mjSTATE_INTEGRATION` 与实际下发命令在同一 pre-step 状态记录；50Hz 网格的
  `ceil(k*50/30)` 对应 20／40ms 实际间隔，保留实际时钟及名义 30Hz 时间。
  不把环境 render_hz 改为 30（200/30 的整数截断会造成频率错误）。
- 保存后重建完全相同的房间、资产、灯光和材质，按归档状态只做 FK／相机计算及
  RTX 渲染，逐帧核对原状态、关节、根和物体。此过程不重跑物理、不重新执行 SONIC，
  完整积分状态在渲染前后必须逐值不变。RTX 图像是延后生成的训练观测，
  **不是操作者当时看到的反馈画面**；这项视觉域差异保留在元数据中。
- 开始前有真实控制 warmup；每条示范另存一秒真实历史状态，并从这些过去状态生成
  对应 RGB，warmup 动作标签为无效。
  之后必须记录全新的实际30Hz命令。追踪失效、编码失败、外力辅助、物理失稳、录制掉帧
  或超出实时容忍时延均不得计为合格数据。

## 保存与独立验收

每条新示范有独立 UUID，保存 `metadata.json`、`commands.jsonl`、
`quality_samples.jsonl`、`applied_force_samples.jsonl`、`episode.json`，以及
`deferred_capture/` 内的实际 MJB 模型、积分状态和事件。补图后 `capture/` 内是 RTX PNG
及其同状态事件，`rtx_completion.json` 绑定两阶段。控制 ledger 保留原始 PICO 包和实际 token、
具名14维手指目标；状态为模型需要的具名43关节，后续按既有接口补齐45维。
实时阶段不写昂贵的 RTX PNG；写入失败会拒收整条，不静默丢帧。
诊断用 `--render-mode isaac` 仍保留原在线 RTX 有界异步写盘，但当前实测不能持续实时运行。

```bash
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/audit_pico_teleop_episode.py \
  /home/zxc/CVPR_Evaluation_runtime/pico_human_demos/某次运行/episodes/某个UUID
```

独立审计重读实际文件并重算物理门禁：检查真实人类来源、完整原始输入与控制身份、
图像字节和像素 SHA、RGB／状态／动作对齐、监督覆盖、真实50Hz接触与放置质量，
以及完整200Hz零外力证据。官方弱 task success 标志不能代替这项验收。
仅 `eligible_for_packing=true` 的示例可进入后续打包；仍需生成 clean DINO 特征、
训练元数据和模型协议检查，才可交给当前联合 checkpoint 微调。
录制完成、审计完成和训练数据包完成是三个不同状态。

`--pico-replay`／`--diagnostic-record` 只用于软件诊断；固定姿态或已有 PICO 包回放
永远不能冒充新的人类示范，不增加高质量数据条数。

## 当前验收边界

2026-10-25 已完成原始输入、SMPL FK／encoder、录制状态机、HTTP 面板及真实
loopback H.264 编解码测试。固定中立姿态诊断通过真实 MuJoCo／SONIC／EGL 四秒、
200个控制步，最大累计延迟9.44ms，控制步间隔中位19.28ms、p99 27.75ms。
这只是短时控制接口检查，不能代表长时吞吐或 Handover 成功。
最终二十秒诊断完成1000个控制步，最大累计延迟89.11ms；597帧RTX补图及
完整积分状态不变检查、995条原始solved支撑力／几何复算、实际命令和零外力检查通过。
合成站立诊断仅“真人来源”和“任务成功”两关按预期拒绝，不增加训练示范。
本次103项相关CPU回归覆盖已通过。
完整同场景在线 RTX 即使关闭独立 viewport，仍在14步后累计落后559ms，被拒收；
因此默认采用上述实时物理与延后 RTX 两阶段方案。
已有600包真实 PICO 诊断的初始参考躯干倾角约50°，从站姿直接跟随会跌倒，故不能
作为站直校准样本。此现象尚不能单独区分当时佩戴姿态与系统标定问题。
本次机器可读记录：`/home/zxc/CVPR_Evaluation_runtime/pico_teleop_20261025/VALIDATION.json`。
实际佩戴后已接收24点全身骨架，站直标定通过；4090网页大画面已显示实际场景。
双手跟随动作、真实录制和成功Handover仍须逐项验收；不能用上述诊断代替。
当前真人接收端保留待机；原始输入仅由同一个SDK订阅读取，暂停时的最近原始包保存在
运行目录`latest_pico_input.json`，避免启动第二个诊断客户端抢占订阅。
真人 live_human_03 的首条录制753e752c仅完成25个控制点、15帧起始预录，观测跨度0.48秒，
累计延迟106ms后拒收。该段左右扳机／侧握与下发手目标均为零，25条原始solved接触均无手目标接触。
独立编译几何检查显示碰撞开启，最近指端仍距目标约2.21cm；这段没有覆盖此前完整抓握尝试。
不能把该短片当成成功示范，或据此断言所有抓握输入均无效。诊断证据保存在本机
`/home/zxc/CVPR_Evaluation_runtime/pico_teleop_20261025/grasp_diagnosis_20261025/`。
握姿和PD路径优化后，20秒明确标记为合成手开合的诊断完成1000个真实控制步、597帧归档
（30帧起始预录、567帧命令），最大累计延迟17.63ms，未放宽100ms保护门槛。
该检查使用原 MuJoCo 碰撞、力矩限制和 SONIC 网络，仅说明此段控制／录制链路可运行；
不是新的人类示范、抓取成功或已补齐RTX训练图像的证据。
诊断目录为 `/home/zxc/CVPR_Evaluation_runtime/pico_teleop_20261025/grasp_fix_diagnostic_01/`。
随后真人 live_human_04 的 dace6584 录制完成2538个50Hz采样、1523帧归档，观测跨度50.74秒；
按主机输入到达时钟计算平均29.995Hz（不表示每帧墙钟间隔严格相同）。右手实际闭合及
右手持物→双手→左手持物已观测到，目标最大抬升19.69cm。末段篮筐倾至约73.49°，
目标重心仍在外轮廓之外约12.24mm，因此完整物理放置验收失败，仍不计高质量训练示范。
原录制保留待RTX补图，另已导出归档状态重建视频；该视频不是原始预览像素或训练RGB。
live_human_04 为切换流畅预览而在暂停、无活动示例时受控退出；重启说明与待补图命令保存在
该运行目录的 `maintenance_restart.json`，不会与下一轮实时采集并行执行RTX后处理。
