# Mobile P&P：本机官方30份初始化评测

## 范围与当前状态

2026-11-05用户要求在本机完成Mobile P&P官方30次评测。
官方口径为Level 0、1、2各10份保存的初始化，各自episode 0–9，
不是生成30个随机seed，也不是只对Level 0重复三次。

本次默认选择最新官方99＋历史1条数据训练的最后已保存checkpoint6000。
该训练于2026-11-04按用户要求停止，未完成计划10000步；
不套用旧Mobile100的10000步终验，也不把旧checkpoint替代本次模型。

运行根为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_20261105`。
`EVALUATION_REQUEST.json`已冻结30份metadata及原始environment_config字符串的SHA、
等级、case编号、环境seed与策略采样seed。
准备时A100原SSH端点超时；用户随后提供新端口42720，已实际连接并取回本次6000步权重。
源前后及本机完整SHA均为`ffe51673a13da83be6b3596a7ebee9a730f47cd61c9e69bc97e3c267d4d6f4a2`，
模型共12,066,487,572字节，源文件身份和23份远端metadata在复制前后稳定；
第24份停止记录保留本机原件。`source/backup_receipt.json`和`SOURCE_BACKUP_REVIEW_003.json`
记录实际备份及独立复核；没有复制optimizer，也没有启动或重启训练。

前两次复制分别因远端CPU参数格式、缺少rsync在传输载荷前失败，
`TRANSFER_FAILURE_001/002.json`及原日志均保留。
最终通过SSH只读流传输完成，未向服务器安装软件，未改变全局SSH配置或已有信任记录；
新端口的首次信任及后续严格校验记录仅保存在本次运行目录。

真实CPU准备于2026-11-05 14:17（北京时间）完成，`CPU_DEPLOYMENT_PROCESS_001.json`返回0。
`deployment/LOCAL_RUN_READY.json`核准VLM 625项、Action 259项、Motion 1060项源张量的严格加载，
新Motion的精确序列化比较通过，SHA为`1a67712be665862c9f673353722b21b55f186e8874e2bea0aef5941b39ec6572`；
未使用旧Motion参数，也未在CPU准备期间初始化CUDA。
`official30_001`已在本机单张RTX 4090完成30例，2026-11-05 14:18:37–15:57:41（北京时间），
进程返回0。`state.json`及`FINAL_EVALUATION_ACCEPTANCE.json`均为`complete`，
30个不同官方case均通过技术验收，实际结果如下。

## 实际30例结果与证据

- Level 0：原生成功3/10（30%），成功case为0、5、8。
- Level 1：原生成功4/10（40%），成功case为1、5、7、8。
- Level 2：原生成功3/10（30%），成功case为2、3、5。
- 合计：原生成功10/30（33.3%），20例未成功，技术异常0例、未尝试0例。
- 机器人跌倒0例、首次策略前跌倒0例；10次成功均在策略执行后发生，没有仅预热阶段的成功。

每等级的技术有效分母均为10。20例未成功均跑满1200步／24秒；
没有因失败而延长时间、改变成功标准或重跑，`automatic_retries=0`。
全部30例保留真实640×360、50fps视频，已逐例解码核准帧数等于控制tick数加首帧；
HTTP、实际在线DINO身份、原生终止标志、物理时钟与完整因果历史也已逐例验收。

运行根中的主要交付：

- `official30_001/FINAL_EVALUATION_ACCEPTANCE.json`：完整30例验收，
  SHA256为`e5a21505edbd1c19492031859adbc10406a5a2a3c16db99d41b7216e93a049a8`。
- `OFFICIAL30_PROCESS_001.json`：实际启动命令、PID／session身份、最终返回0。
- `OFFICIAL30_RESULT_SUMMARY.json`及`official30_cases.csv`：各等级与逐例结果、模型SHA、视频及原记录路径。
- `official30_001/level{0,1,2}/case_00..09/CASE_SUMMARY.json`：逐例原生结果及证据身份。
  视频位于各case的`simulation/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/episode-{case}/rollout.mp4`。
- `INDEPENDENT_CASE_REVIEW_PREFIX_001.json`：固定前22例独立只读复核；不冒充全30终验。
- `FINAL_RESULT_INDEPENDENT_REVIEW_001.json`：在前22例收据和证据身份未变的前提下，
  增量复核最后8例并核对全30唯一覆盖、原生计数、完整验收和实际退出码0，结果通过；
  SHA256为`4d544620db6af6489ba09003a36be0334f432c651defa103f58fae8d2d029c21`。
- `FIRST5_QUALITY_OBSERVATION_001.json`及`review/level0_cases00_04_observation.png`：固定前5例的只读质量观察。

前5例观察中，case01已完成带物搬运但末端落在篮外；case02–04未形成持续带物搬运，
随后空手走向第二桌。这些观察结合50Hz trace和少量已保存RGB，
不将稀疏接触记录冒充完整200Hz抓持证明，也不从5例推断过拟合或泛化。

本轮完成后，评测自身模型服务和仿真进程已退出；没有停止或重启Grasp采集器及其它任务。

## 模型与数据绑定

训练来源为`mobile_pp_official100_training_20261104/run_001`，
数据为`pico_mobile_pp_20261101_official99_plus1_human100_01`：
100条真人轨迹、99种官方TRAIN初始化、56,423帧、50,523训练窗。
保持真实43／78维至生产45／80维协议、既有统计及clean Motion DINO；不重新拟合统计。

独立部署必须绑定实际选中模型的源／本机SHA、稳定文件身份、
新鲜远端训练停止状态与已完成保存标记、真实配置及数据就绪收据。
远端DINO manifest已重绑路径，不能用本机原manifest冒充实际训练metadata。
VLM、ActionExpert、MotionExpert参数均来自同一个本次joint checkpoint；
Motion必须重新导出，旧Motion只可提供非张量构造metadata。
保留state dropout为0、null token及六层ActionExpert，不能在部署时重做迁移。
实际完成CPU严格加载与Motion精确导出后，才可发布本次部署就绪标记。

`prepare_mobile_pp_official_eval.py`是本批停止点的独立CPU入口；
`evaluate_mobile_pp30.py`负责逐例启动及实际结果验收。
源快照的24份metadata原路径见入口的`METADATA_SOURCES`及运行根的`SOURCE_COPY_PLAN.json`；
其中STOPPED_BY_USER保留本机原件，其余须实际取回A100文件。
下载未完成时不生成`backup_receipt.json`，也不发布部署READY。

本次真实备份核准后，在仓库根目录执行了以下两步。
保留已完成运行；复现实验必须使用新的部署和评测输出目录，不能覆盖原证据。

```bash
MOBILE_EVAL_ROOT=/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_20261105
MOBILE_MODEL_SHA256=ffe51673a13da83be6b3596a7ebee9a730f47cd61c9e69bc97e3c267d4d6f4a2
CUDA_VISIBLE_DEVICES='' /home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  scripts/4090_simple/prepare_mobile_pp_official_eval.py \
  --source "$MOBILE_EVAL_ROOT/source" \
  --resources /home/zxc/CVPR_Evaluation_runtime/resources \
  --release /home/zxc/CVPR_Evaluation_runtime/pico_mobile_pp_20261101_official99_plus1_human100_01 \
  --output "$MOBILE_EVAL_ROOT/deployment" \
  --step 6000 --expected-model-sha256 "$MOBILE_MODEL_SHA256"

/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/evaluate_mobile_pp30.py \
  --deployment "$MOBILE_EVAL_ROOT/deployment" --data-root "$MOBILE_EVAL_ROOT/data_root" \
  --training-manifest "$MOBILE_EVAL_ROOT/source/release_metadata/dino/manifest.json" \
  --output "$MOBILE_EVAL_ROOT/official30_001" \
  --runtime-root /home/zxc/CVPR_Evaluation_runtime \
  --checkpoint-step 6000 --expected-model-sha256 "$MOBILE_MODEL_SHA256" \
  --port 21075 --sampling-seed 20261105
```

上述两步已实际执行，启动命令与进程身份另见`CPU_DEPLOYMENT_PROCESS_001.json`及`OFFICIAL30_PROCESS_001.json`。
每次campaign必须使用新目录，`state.json`记录实际进度，
`FINAL_EVALUATION_ACCEPTANCE.json`只有`status=complete`且30个不同case均有效时才代表完整完成。

新增两个入口的28项CPU回归已通过，覆盖停止／保存证据、远端DINO不能被本机原manifest替代、
实际小型safetensors到新Motion导出及两入口合同衔接、30份固定case、部署文件变更拒绝、
技术异常继续其余case、原生warmup成功、HTTP／视频验收和PID复用拒绝清理。
最终版本记录位于运行根的`CPU_REGRESSION_002.json`及日志，首次001仍保留；
因补充SIMPLE与视频源码绑定而重新检查，检查只限制自身CPU26／nice19并隐藏GPU。
这些CPU回归只验证工具逻辑；本次12GB真实权重的严格加载另有上述实际CPU准备收据，
实际GPU闭环则由`official30_001`逐例保存视频、HTTP、trace和`CASE_SUMMARY.json`验收。

## 固定评测协议

使用原始官方state_dict恢复整份初始化，reset的DR参数为None，
不注入PICO冷姿态、示范状态行、teacher动作、未来GT或matched训练初始化。
环境seed固定0；独立策略采样seed为`20261105 + 10 * level + case`，两者分别记录。

每例1200个50Hz控制步，即24秒，包含既有1秒live-pose SONIC预热；
30Hz动作、每次执行30步chunk、10步采样、RTC关闭。
MuJoCo负责物理，Isaac RTX只同步呈现同一状态，HTTP等待不推进仿真时间。
原生成功以实际任务constructor的0.9及native terminated为准，
归档metadata的0.4只保留来源记录，不擅自赋给运行任务。
不因跌倒提前结束、不增加几何成功门、不在成功后延长时间。

每例启动独立策略服务和仿真进程，保存真实视频、逐步trace、
客户端／服务端HTTP与RNG证据、实际在线DINO身份和最终结果。
技术异常原件保留，单独计数，不计作任务失败，也不静默重跑覆盖。
只有30个不同官方case均完成技术验收，才报告完整30次结果；
分别报告各等级的原生成功数与技术有效分母。

现有双来源SONIC控制器适配的`formal_evaluation_eligible=false`继续保留。
这次使用官方初始化、时间和成功语义，但不是原官方控制器的认证benchmark；
Level 0仍为训练同一场景，不能将全部30例概括为独立新场景泛化。
Grasp采集器及其它任务不停止、不重启；实际启动前核对本机GPU计算任务。

## 同一6000步模型的test-time RTC版本

用户另行授权开启test-time RTC。复用上述已经严格加载、源／本机SHA核准的新6000步部署，
不重训、不重导Motion、不改数据、统计或官方初始化。新运行根为
`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_testtimertc_20261105`，
与原`official30_001`分开保存；首次请求及基线验收SHA见新根的`EVALUATION_REQUEST.json`。

该模型训练配置为`model.rtc=false`，只使用现有推理期`rtc_mode=test_time`，不冒充训练期RTC。
完整预测仍为30×80，每次实际执行前15帧（0.5秒，恰好25个50Hz控制tick）后重新规划；
服务保留归一化完整预测，左移15行、补零15行，客户端只接收反归一化后的15×80。
固定`d=0`、`s=15`、`mask_schedule=exponential`、`guidance_alpha=0.9`及10步采样。
软mask前15行非零、后15行严格为零，不约束补零尾部；现有VJP覆盖全部80维，
包括最终不执行的末2个padding通道，本轮不另改采样算法。

仍采用同步HTTP等待时暂停仿真时钟的协议，RTC用于相邻计划的连续性guidance，
不宣称异步实时延迟补偿。相较原RTC off／执行30帧，本版同时更改了执行窗口，
分数差不能独立归因于RTC。物理、控制器、因果历史、1秒预热、24秒总预算与原生0.9标准均保持。

生产启动收据绑定实际RTC参数；逐请求记录实际plain／test_time模型方法调用和响应shape。
验收除原30例门外，还核对首次plain、后续test_time、425不采样、每25tick整窗消费、
请求间隔0.5秒以及动作索引和chunk起点，拒绝只开flag或客户端仍执行旧30帧的情况。
运行前需完成新增CPU回归和batch1、单张4090、连续两次真实采样的GPU健康检查；
GPU健康检查使用旧视频RGB回放和零state，只核对数值、VJP和冻结参数，不是物理评测或成功率证据。

新增关键路径CPU回归已通过52项（运行时33项、campaign19项），实际命令和范围分别保存于
`RTC_RUNTIME_CPU_REGRESSION_001.json`、`CPU_CAMPAIGN_REGRESSION_002.json`及关联日志。
单卡4090、batch1的两次真实GPU采样已返回0，`gpu_health_001/GPU_RTC_ACCEPTANCE.json`
记录首plain／次test_time、10次实际VJP调用及后9步非零修正依据，两份15×80输出均有限。
1941个参数张量全部冻结，前后参数grad均为None；GPU峰值allocated为9,593,193,984字节。
首次外部launcher缺少`data_root`参数，在模型／GPU启动前失败，原因和原日志已另存，
修正launcher后才执行上述GPU检查，不将该准备失败混入30例任务分母。
检查日志含attention backward非确定性内核提示；固定seed不代表RTC跨次逐bit一致，
本轮保留现有attention实现，不擅自切换内核。

```bash
CUDA_VISIBLE_DEVICES=0 /home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/evaluate_mobile_pp30.py \
  --deployment /home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_20261105/deployment \
  --data-root /home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_20261105/data_root \
  --training-manifest /home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_20261105/source/release_metadata/dino/manifest.json \
  --output /home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_testtimertc_20261105/official30_rtc_001 \
  --runtime-root /home/zxc/CVPR_Evaluation_runtime \
  --expected-model-sha256 ffe51673a13da83be6b3596a7ebee9a730f47cd61c9e69bc97e3c267d4d6f4a2 \
  --checkpoint-step 6000 --port 21076 --sampling-seed 20261105 \
  --rtc-mode test_time --execute-steps 15
```

该命令为本版入口，实际启动和完成以新根的进程收据、逐例证据及最终`status=complete`为准。

### RTC版实际30例结果（已完成）

`official30_rtc_001`于2026-11-05 19:50:59–21:32:37（北京时间）实际完成，
`OFFICIAL30_RTC_PROCESS_001.json`记录源提交`c1efe3b02ad37f42c7fa6b5792a85c4a9c0a0ab5`和退出码0。
`state.json`与`FINAL_EVALUATION_ACCEPTANCE.json`均为`complete`，30例全部技术有效。

- Level 0：6/10（60%），成功case为0、1、5、6、7、8；原基线3/10。
- Level 1：5/10（50%），成功case为0、1、5、6、7；原基线4/10。
- Level 2：4/10（40%），成功case为1、2、3、8；原基线3/10。
- 合计：15/30（50.0%），原基线10/30（33.3%），本轮观察到提高16.7个百分点。
- 技术异常0例、未尝试0例、跌倒0例、策略前跌倒0例；15例未成功均完整跑满24秒。

配对结果为：两版都成功8例、失败转成功7例、成功转失败2例、两版都失败13例。
新增成功为L0/case01、06、07，L1/case00、06，以及L2/case01、08；
退步为L1/case08和L2/case05。无自动重跑、延时或成功标准改动。
三等级全部10例均完成独立增量原证据复核，`LEVEL{0,1,2}_PREFIX_10_INDEPENDENT_REVIEW_001.json`
分别核准6／5／4次成功及技术有效性。

全30例实际记录共1171次`test_time`采样；首段plain与后续RTC路由、15×80响应、
25tick整窗消费、真实逐tick观察、HTTP不推进仿真钟、参数grad为None，
以及640×360／50fps视频逐例通过验收。评测自身服务、仿真与监督器均已退出；没有停止其他任务。

新运行根中：

- `official30_rtc_001/FINAL_EVALUATION_ACCEPTANCE.json`：完整30例实际验收，
  SHA256为`64e45508ff51fa59753efab1952652c2bc350d38e74dfff60bf98f47889f3704`。
- `OFFICIAL30_RTC_COMPARISON.json`：同模型、同固定初始化的配对汇总，
  SHA256为`c75bf83d9ab1b9dc075dece7c60057279096174f18d3cc4090752e9fff523d74`。
- `official30_rtc_paired_cases.csv`：全30例的两版成功标志、时长、视频及原JSON路径。
- `official30_rtc_001/level{0,1,2}/case_00..09/CASE_SUMMARY.json`：逐例RTC执行与原生结果。
- `FINAL_RESULT_INDEPENDENT_REVIEW_001.json`：复用三个等级的独立前缀复核并确认证据未变，
  全30唯一覆盖、两版模型／初始化／seed身份、state／final逐字节一致及实际退出码0均通过；
  SHA256为`3cf1b5f28eb8b73e59100457941b379361d553c8d203f202548c7d517ae9c151`。

该差异是本轮固定30例的观察值，既包含RTC也包含执行窗口从30帧缩短到15帧的影响，
不作为纯RTC因果效应、跨seed显著性或新场景泛化证明；原SONIC适配及非确定性内核边界保持。

6000步RTC版15次未成功的逐例阶段分类、原视频打包，以及后续同配置5000步权重测评，
见[RTC失败阶段复核与5000步测评](PICO_MOBILE_PP_RTC_FAILURES_AND_5000.md)。

## 15例逐例复核与预热关联（补充分析002）

本节只分析上述6000步RTC原运行的15例未成功，不启动新测评、不改动作或判据。
逐例查看原始RGB／视频并结合50Hz轨迹后，主要阶段为：首桌取物／离桌未完成11例
（其中1例取物时掉落），到篮后放置掉落2例，搬运迟滞、24秒结束仍持物2例。
15例均跑满24秒；技术异常0、跌倒0。这里的“失败阶段”不是已通过干预证明的因果根因。

以下时间均为相对首帧的仿真时间；物体pose／接触使用step后的`next_sim_time_s`，
图像使用动作前时刻。50Hz采样接触不能证明200Hz物理全程持续抓持，物体原点高度或
靠近篮原点也不能替代原生成功判据。

1. **L0/case02**：约4–6秒接触盒子，约6–8秒空手离桌，目标仍留在首桌；
   后续走到空篮处执行动作。卡在抓取到离桌衔接。
2. **L0/case03**：3–4秒未形成持续带物抓持，8秒空手离开；
   盒子仍由第一桌／干扰物支撑。
3. **L0/case04**：4秒附近短暂接触后空手走向第二桌；目标未被带走。
4. **L0/case09**：虽有连续手接触采样（首末样本跨度约0.96秒），目标仍接触桌上玩具枪；
   随后空手离桌，不能把这段接触直接称为稳定抓持。
5. **L1/case02**：反复在首桌抓取，约16秒再次伸手；23秒目标仍在第一桌。
6. **L1/case03**：2–4秒抓取未成，8秒空手转向第二桌；16／23秒篮子空。
7. **L1/case04**：盒子仍在首桌，8秒空手转身，之后在空篮附近执行动作。
8. **L1/case08**：已右手夹持搬运，但8–16秒目标距篮约1.63→1.73米，
   推进停滞；22.82秒才进入篮原点XY距离15厘米内，24秒仍有右手接触、无篮接触。
9. **L1/case09**：2–8秒在首桌尝试，16秒空手离桌；
   目标最终仍接触第一桌与香蕉。
10. **L2/case00**：在首桌反复伸手／推盒后停留，没有完成有效带物离桌。
11. **L2/case04**：已搬到篮边；约17–18秒目标与篮子一起离开第二桌、随后落地。
12. **L2/case05**：8–14秒搬运推进不足，23.48秒才首次测到篮接触；
    24秒仍有右手接触，未完成释放。
13. **L2/case06**：取物阶段未形成持续带物离桌；约5.14秒后无采样手接触，
    约7.28秒起出现地面接触。不能称为“稳定搬运中才掉落”。
14. **L2/case07**：已搬到篮处，释放附近目标从篮沿落地，篮子仍在第二桌上。
15. **L2/case09**：预热期目标已平移／翻转，后续策略段无采样手接触，
    目标留在首桌，机器人空手离开。

### 首个策略动作前的目标扰动

首个学习策略动作在仿真绝对时间1.005秒（相对首帧1.0秒）。前1秒预热中，
L0/case02、03、04、09，L1/case02、03、04、09和L2/case09这9例，
目标从首个post-step样本到预热末已水平偏移**9.44–14.88厘米**，且9例全部失败；
15例成功案例在相同时段最大偏移仅**0.551厘米**。这9例baseline与RTC的前50个
预热样本在目标／篮pose、接触、reward、骨盆姿态和控制阶段等核对字段完全一致。
所以该前置扰动不是高层RTC动作造成的，且不能删除这9例或据此推算修复后的成功率。

预热并非冻结物理：前10tick／0.2秒为测量姿态PD保持，后40tick／0.8秒为
live-pose SONIC平衡。9例中6例在预热已有采样手接触，其余3例未采到；
初始几何／重力接触与预热控制均应列为优先排查候选，不能说9例全由手推，
也不能在没有干预对照及力／冲量记录时认定根因。

### RTC改善、退步与Ψ₀差距的解释范围

RTC版本由10/30提高到15/30，7例改善、2例退步。两例退步均实际检查原baseline视频：
L1/case08进入篮原点XY距离15厘米内由17.42秒延后到22.82秒，baseline20.46秒成功；
L2/case05首次测量篮接触由15.72秒延后到23.48秒，baseline18.68秒成功。
RTC两例均在24秒结束时仍有手接触，说明中途停滞和晚到影响了预算内完成释放。
这不是技术超时，也不证明延长时间必成或RTC引导是唯一原因。

与公开Ψ₀的7／5／6相比，本轮6／5／4在总量上少1／0／2例，即15/30对18/30。
公开表没有逐case成功身份，无法把少的3例对应到本轮具体失败。
本地SONIC适配、模型／训练数据及6000步未完成计划训练的停止点，与公开版本不同；
现有证据不证明欠拟合、过拟合或某个参数造成10个百分点差距。
RTC仅改善计划衔接，不保证实际抓住、失败重试或释放成功；本轮还同时缩短了执行窗口，
不能把全部改善或退步归因单一RTC因素。

原生实现另有判据细节：每tick会经reward和success检查调用两次`compute_reward`，
实测成功末段为23个连续正reward控制步，不能把阈值0.9解释为连续0.9秒；
接触判断还受`or`／`and`优先级与geom顺序影响，trace测量接触不能直接重建判分。
本轮未改动或重新评分。14个失败的原生reward始终0，仅L2/case04短暂达0.06，
因此多数失败发生在形成有效入篮／脱手条件之前，不能仅归咎于计数阈值。

优先排查顺序为：隔离预热目标扰动，检查未抓住却继续离桌的阶段推进，
再对搬运停滞做RTC／执行窗口单因素对照，最后检查篮沿碰撞与释放。
这些是尚未执行的实验建议；任何协议变更须另立版本，保留本轮原始15/30结果。

完整观察与对照记录位于原RTC运行根的`failure_analysis_002/`：
`level0_cases.json`、`level1_cases.json`、`level2_cases.json`、
`protocol_and_comparison.json`、`all30_measured_metrics.json`、
`nine_large_warmup_baseline_comparison.json`、`FAILURE_REVIEW.json`及
`failure_review_cases.csv`。每例均保留原视频／trace路径与文件身份。
用户要求直接文字说明，不交付Canvas。

## 起步手碰物的独立控制优化（wrist_retreat_v1）

本节是用户随后授权的部署预热优化，原6000步15/30及进行中的5000步测评保持。
由于正式评测逐例校验源码SHA，实现位于独立工作树
`/home/zxc/.codex/worktrees/mobile-warmup-safe/CVPR_Evaluation`、分支`codex/mobile-warmup-safe`，
没有覆盖正在使用的`codex/pico-locomotion`源文件。默认仍为`legacy`；
新测评必须使用新源码、独立输出目录，并显式传`--warmup-arm-profile wrist_retreat_v1`。

### 固定控制规则与边界

首个真实post-reset状态不变；首腕world位置／朝向、首骨盆水平heading定义统一参考：
两手后退5厘米、分别向外3厘米、上抬5厘米，0.2秒cubic smoothstep后保持world目标。
仅依据当前机器人本体状态，在私有MjData求14臂关节IK；不读物体位置／接触或teacher，
不写live qpos/qvel，不关闭碰撞，不重置或冻结物体，不延长原24秒预算或补造RGB历史。
从首个受控5毫秒子步起增加14臂关节纯重力前馈，再经原PD／actuator限幅；
腿／腰和手指gain、原力矩上限、物理模型、solver及原生判据均不改。
SONIC期望臂reference与实际位置PD目标一致，真实whole-body平衡仍可因臂运动改变。

首个学习策略命令到来后，用固定0.2秒cubic混合交还decoder臂目标并衰减前馈至零。
previous-action history从实际施加body29位置目标反算；不记录未执行的raw decoder输出。
详细CLI、receipt字段及控制范围见[CONTROL.md](../scripts/4090_simple/CONTROL.md)。
helper中每迭代0.04rad是IK求解步长，不能称为实际关节速度上限；有限IK残差也不能用来
事后删除案例或改判分母。

### 实际CPU物理对照：官方全30份初始化

记录根`/home/zxc/CVPR_Evaluation_runtime/mobile_warmup_safe_20261105`。
`paired30_002/PAIRED30_CPU_ACCEPTANCE.json`实际完成30份原初始化×两版、每版1秒，
每例50个真实控制帧／200个物理子步，CPU28／nice19／GPU隐藏，无渲染或学习策略调用。
全部30个原版的50Hz目标pose7与接触列表严格复现原RTC记录，差为0；
每对的MJB与完整post-reset INTEGRATION状态SHA相同。
首个reset内0→0.005秒原样保留，新控制从0.005秒后的首受控子步生效。

- 窗口内目标最大XY漂移的最坏值：原版159.919毫米，新版10.593毫米；
  中位值4.476毫米→0.703毫米。
- 1秒末目标XY漂移的最坏值：148.747毫米→4.730毫米。
- 出现负距离手指／腕掌接触目标的初始化：16/30→3/30，
  对应200Hz接触样本总数361→31；手／腕碰桌两版均0/30。
- 目标最大旋转131.53度→5.09度，最大垂向变化90.92毫米→0.886毫米。
- 最低骨盆高度0.788376米→0.787552米；最大骨盆倾斜8.272度→11.429度，
  末帧倾斜最坏值0.582度→1.724度；两版均未跌倒。

剩余3例为L2/case00、06、08，原始reset几何已存在手／腕与目标相交，
不是把物体移开后宣称修复。新版其目标最大XY漂移分别5.938、2.637、2.062毫米。
L1/case03的10.593毫米最大漂移没有手／腕碰物或碰桌，来自原始目标与洗护瓶的相交解算；
不通过修改物体位置、摩擦或碰撞来消除此限制。50Hz未采到手接触也不等于物理上没有：
本次200Hz原版复现确认此前L0/case02、03及L1/case02均有被50Hz漏掉的短接触。

此结果只证明预热阶段物体扰动明显减少，不等于绝对无碰撞、姿态完全无影响或
学习策略成功率已提高。最大倾斜增加的代价须随结果保留；此CPU对照本身不含完整学习策略／RTX闭环，
不能用它直接替换原15/30。后续独立完整测评结果见下节；现有采集器及其它GPU任务未停止或重启。

实际候选排查也保留原件：单独重力补偿未改善L0/case04；完整SONIC DEFAULT臂姿态的
静态FK显示此站位下存在指端与桌面穿透风险，因此未采用。最终固定腕轨迹是新控制协议，不冒充官方原控制器。
全30验收JSON SHA256：`c9476a8ff947a6f39bbf627a206902f362bd2999f054c6a3652beac15905999b`。

另外实际执行一次局部CPU接管诊断：1秒预热后仅由当前机器人本体生成一个固定mode0 token，
继续0.4秒／80物理子步，不使用学习策略、未来teacher或物体反馈。
从1.005秒启动handoff，1.205秒前馈降至0且解除臂override，之后保持原控制。
完整qpos/qvel/qacc/ctrl/xpos/xmat均finite，实际arm位置目标最大步差0.048498rad，
body目标最大步差0.050375rad；此段骨盆最低0.786272米、最大倾斜1.942度，未碰目标或桌面。
`handoff_001/HANDOFF_CPU_ACCEPTANCE.json`的SHA256为
`34277185de1a47c1e2979539043feec27fbbb4285274b54bb460eb5baa43c3ba`。
它验证有限混合与纯重力前馈退出的实际物理路径，不冒充新学习策略闭环验收。
相关51项CPU回归通过、零skip，含旧版逐步数值一致、真实G1模型IK、物体状态不影响参考、
实际previous-action history、首步前馈、原限幅、reset、CLI与独立控制协议绑定。
正式入口的臂收据验收器还实际读取上述50帧预热＋20帧接管证据通过，记录为
`ACTUAL_ARM_RECEIPT_VALIDATION.json`；未伪造原生成功或完整策略CASE_SUMMARY。

### 完整RTC测评的资源枚举故障与保留记录（2026-11-06）

用户随后授权完整30例测评，仍用同一6000步权重、sampling seed基数20261105、
test-time RTC／15帧执行窗、24秒含预热、原生0.9判据，只增加显式腕参考profile。
运行根为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_wristretreat_rtc_20261106`。
`official30_rtc_001`首例L0/case00原生成功，805个50Hz tick／16.1秒；
独立复核确认50次真实预热参考、240次物理前馈调用及1.005→1.205秒有限交接，
实际1次plain／30次RTC调用、因果图像历史和806帧原视频均通过。

随后L0/case01、02在reset时编译失败，错误为`brasket`质量／惯量小于MuJoCo下限，
两例均尚未发起策略HTTP调用。这是技术错误，不能作为策略失败计分。
发现重复故障后仅中断本轮自有监督器，保留全部日志、首例原视频及两份错误原件；
当时正在启动的case03记录为中断，未启动的26例不补造结果。
该尝试的完整30例验收为`incomplete`，不发布成功率，也不覆盖原15/30。

直接原因是共享篮子模型目录新增了`textures/`子目录，其创建时间落在首例RTX加载期间：
旧Objects／GraspNet资源加载器每经过一层`os.walk`都会覆盖碰撞网格列表，
最后进入仅含PNG的子目录后清空了已发现的convex pieces；因此编译出的自由物体没有geom。
新PNG与原材质贴图SHA相同，USD→OBJ→MTL引用链符合导入器复制纹理的解释，
但未记录文件写入syscall，具体写入进程／函数仍属推断。
修复为沿原遍历顺序累计convex pieces、无网格时明确报错，保留所有原网格、
质量、摩擦、惯量计算和碰撞参数，不删除共享纹理或修改旧run。
正式入口同时冻结物体加载器及MuJoCo构建器源码身份。
实际新增6项网格回归和20项campaign回归均通过。76个资源目录、1945个凸片的数量／顺序不变；
CPU隐藏GPU、CPU28／nice19实际重建全部30份reset，30份MJB文件字节、各626项静态字段、
709维完整post-reset INTEGRATION均与原归档精确一致，物理时间仍0.005秒。
记录为`asset_mesh_recovery_cpu_001/ACCEPTANCE.json`，SHA256为
`b1002a2c56a0b9cf46343b6589723beeb86d26da104d67c4aacc32c765c9d373`。
独立复核后使用独立目录重做完整30例，原故障尝试始终保留。

### 修复后完整30例RTX／RTC结果（2026-11-06，已完成）

`official30_rtc_002`于北京时间2026-11-06 09:32:23–11:09:38实际完成全30例，
源提交为`e99a15259216d1d78a575ef9119a6d2f1389baf7`。
`OFFICIAL30_WRIST_RTC_PROCESS_002.json`记录真实退出码0；`state.json`与
`FINAL_EVALUATION_ACCEPTANCE.json`逐字节相同、均为`complete`，后者SHA256为
`4be825538b6ad4adc296d9ff4531703b9df3fdd3f864fc6c46987d542b16436a`。
正式计分只包含本轮统一源码的30例，之前001技术中断尝试单独保留、不混入或择优复用。

与原6000步RTC同模型SHA、同官方初始化／seed、同RTC执行窗、同24秒预算和原生0.9判据相比：

- Level 0：6/10→9/10；新版成功case00、01、02、03、04、06、07、08、09。
- Level 1：5/10→7/10；新版成功case01、02、03、06、07、08、09。
- Level 2：4/10→6/10；新版成功case00、01、02、03、07、09。
- 合计15/30（50.0%）→22/30（73.3%），本轮多成功7例，提高23.3个百分点。
- 本轮30例全部技术有效，技术错误0、未尝试0、跌倒0、策略前跌倒0。
  8个未成功案例均实际跑满1200tick／24秒，未因质量失败重跑。

配对分类为：两版均成功11例、旧失败转新成功11例、旧成功转新失败4例、两版均未成功4例。
改善案例为L0/case02、03、04、09，L1/case02、03、08、09，L2/case00、07、09；
退步案例为L0/case05、L1/case00、05、L2/case08，原视频全部保留，不能只报告改善案例。
新版全部未成功为L0/case05，L1/case00、04、05，L2/case04、05、06、08。

实际完整策略运行中，新旧各30例前50帧目标pose7均与各自已验收CPU预热轨迹逐值精确一致。
按实际50Hz首秒poststep观测及相同post-reset锚点计，目标最大XY漂移的最坏值
159.919毫米→10.574毫米，中位4.462毫米→0.703毫米，首秒末最坏148.747毫米→4.730毫米。
观测到非正距离（≤0）手指／腕掌接触目标的案例13/30→3/30；新版剩余L2/case00、06、08仍有原始初态相交。
50Hz采样不能证明子步无碰撞；它与上节独立200Hz对照的16/30→3/30、10.593毫米峰值分别记录，
不混用采样口径。L1/case03的最大漂移仍来自原有物体间干涉，未改物体位置消除。

全30例共实际30次plain首段采样、1095次test-time RTC采样；全部保留真实历史、
15×80执行窗／25tick消费、HTTP停仿真钟、冻结参数无grad及有限臂交接的验收收据。
30段原视频均为640×360／50Hz，逐例实际解码验收通过。
部署准备／模型／DINO及所有30份初始化身份继承原6000步版本；修复只恢复原碰撞网格枚举，
30份物理模型和完整初态的精确一致证明见上节，不改变质量、solver或成功判据。

运行记录及配对结果位于同一新运行根：

- `official30_rtc_002/level{0,1,2}/case_00..09/CASE_SUMMARY.json`和原`rollout.mp4`；
- `paired_initialization_analysis_002/FINAL_PAIRED_COMPARISON.json`及`final_paired_comparison.csv`：
  全30唯一配对、四类结果、首秒指标和原文件身份；
- `paired_initialization_analysis_002/FINAL_PAIRING_ACCEPTANCE.json`：旧新60份summary与
  420份证据记录（含60段原视频）的当前身份核准，SHA256为
  `e54c9d90d53df3cbe57823878e62c8980b89a3fdb27005916e339244d66ae670`；
  同目录`FAILURE_VIDEO_PATHS.json`与`new_failure_video_paths.txt`列出新版8段失败原视频。
  原片交付包为`Mobile_PP_6000_RTC_wristretreat_8_failure_videos.zip`，31,602,824字节，
  SHA256为`c59dfb7ee6de802f91d8c40f43f5da1babdfc995fd3ac160faaad8600d7cb1ba`；
  `PACK_FAILURE_VIDEOS_RECEIPT.json`核准仅新8例、ZIP CRC及解包字节SHA与原片一致，未转码。
- `INDEPENDENT_CASE{0,1}_REVIEW_002.json`及六份`INDEPENDENT_INCREMENT_*_REVIEW_002.json`：
  首两例与后续新增批次的独立实际证据复核。
- `INDEPENDENT_FINAL_ACCEPTANCE_002.json`：真实rc0、全30唯一覆盖、全部视频与证据身份、
  29321个控制tick／29351视频帧、RTC调用及自有进程退出的独立终验，SHA256为
  `2f0260463f61c6ce4afe15be0ebe8bd9bc5769235e042af9000e91761d694d03`。

系统根分区在本次运行前已无可用空间；只将本轮自有子进程的`TMPDIR`／`SQLITE_TMPDIR`
设在运行根`tmp`，未删除其它数据，未声称OptiX固定缓存位置已迁移。
这不改变仿真时间或模型参数；非致命缓存告警保留在原日志。
采集器一直保持原暂停会话，没有停止／重启；测评只清理自己的模型和仿真进程。

该结果是固定30份官方初始化、SONIC适配控制协议下的单次配对观察，
不是官方原控制器认证成绩或独立新场景泛化结果；相同seed也不保证GPU逐bit确定。
启动扰动下降与整体成功率增加并存，但仍有4例退步，不能据此保证每次任务均改善。

### 新版8例未成功的逐例复核（2026-11-06）

本节只分析上述已完成的`official30_rtc_002`，不混入旧版15例失败、5000步版本或001技术中断尝试。
将8段原视频的1Hz全程帧、关键段加密帧，与原始50Hz物体／篮筐pose、接触和reward对齐。
复核结论为：**5例放置接触失稳并将篮筐带离桌面，2例搬运／放置未完成，1例放置过晚且接触不连续**。
因此按任务阶段计，6/8（75%）集中在放置，2/8（25%）集中在搬运未完成。
8例均完整运行1200tick／24仿真秒、没有跌倒或技术错误；“到时终止”是共同终止方式，不是共同根因。

逐例观察（以下为约数，精确poststep仿真时刻保存在复核JSON；视频PTS比仿真时钟少0.005秒）：

- **L0/case05：侧面接近篮沿后碰筐，后续放到裸桌面。** 13.805秒盒子首次接触篮筐，
  原视频13.6–14.2秒可见仍被右手持住的横向盒子接近篮沿、篮筐随后离开桌缘；
  14.245秒篮筐参考点已低于0.2米。目标与手的最后接触直到21.945秒，22秒后盒子才落到桌面，
  此时篮筐已在地面；全程reward为0。此例并非只差更多时间或最后一次开手。
- **L1/case00：放置时盒／筐接触不稳，最终一起落地。** 16.525秒首次碰筐，
  19.345–19.505秒短暂积累reward、最高0.34，之后反复清零；
  19.785秒篮筐参考点低于0.2米，最终盒子与篮筐分离在地面。
- **L1/case04：篮沿接触后盒／筐掉下桌。** 15.665秒首次碰筐，
  16.445秒最后观测到目标与手接触，16.705秒篮筐参考点低于0.2米；最高reward仅0.10，最终0。
- **L1/case05：持续持盒碰筐后失稳。** 14.325秒首次碰筐，
  15.825秒最后观测到目标与手接触，15.965秒篮筐参考点低于0.2米；
  多次短暂正reward均中断，最高0.10，最终盒／筐落地分离。
- **L2/case04：放置接触失稳，跌落中短暂满足接触条件也没有形成成功。**
  16.545秒首次碰筐，17.645秒最后观测到目标与手接触，17.805秒篮筐参考点低于0.2米；
  最高reward0.26出现在18.045秒，最终盒子落地、reward为0。
- **L2/case05：持物后的推进长时间停滞，24秒仍未完成放置。** 视频可见约5–6秒已持物离开源桌，
  没有回桌重复抓取；8–18秒目标距篮筐的XY距离约1.62→1.66米，几乎没有向篮筐推进。
  之后才明显靠近，24秒末仍有手接触，XY距离约0.237米，全程没有盒／筐接触、reward为0。
- **L2/case06：同样是运输推进不足。** 8–18秒目标距篮筐的XY距离约1.75→1.64米，
  仅靠近约0.11米；24秒末仍有手接触、XY距离约0.419米，从未接触篮筐、reward为0。
  这两例是持物目标的接近进度停滞，日志不足以断言机器人base完全静止或某个步态模块失效。
- **L2/case08：放置过晚，并发生接触中断。** 21.305秒开始碰筐，但首次正reward直到23.605秒，
  距离24.005秒终点仅剩21个tick，已经少于当前原生门所需的23个连续合格tick。
  23.685秒最后观测到手接触；23.705–23.925秒连续12tick正reward、最高0.46，
  23.945秒目标与篮筐接触丢失一帧，reward清零；随后恢复接触，末值仅0.10。
  该例不能归为“完全没放进去”或单纯“没松手”，也不能保证多给0.2秒一定成功。

**直接原因与合理机制应分开。** 5例失稳的视频共同特征是持盒从篮筐侧面／篮沿附近接近，
尚未形成稳定承托时仍有盒／手或盒／筐相对运动，之后篮筐离开桌面。
这些案例13.805–16.545秒就已碰筐，仍剩约7.5–10.2秒预算，延长时限无法直接修复已被移走的接收容器。
原生任务要求的也不只是“盒子最后落在第二张桌子上”。

当前实现对这些错误缺少显式的任务阶段纠错：`dual_source_psi0.py`的策略输入为当前图像、
当前具名关节位置及因果图像历史，物体／篮筐pose和上述接触仅用于诊断，不是策略输入。
每次执行15个30Hz动作，即25个50Hz控制tick／0.5仿真秒后才重新采样；
期间虽逐帧收集图像，仍继续执行该段已有计划，没有基于盒／筐接触的提前中断或“对齐—下降—承托—脱手—撤手”确认流程。
因此终端对齐误差和接触失稳可能在下一次重规划前扩大；RTC约束前后计划连续性，
不等同于几何避碰、放稳验证或释放时机保证。这是代码支持的机制解释，尚无消融量化各因素的因果贡献。

实际官方指令为`pick up the cracker box from table1,locomotion to table2,and place on table2`
（原记录有额外空格），没有明确要求篮内放置；而计分函数检查盒／筐接触与无手接触。
语言约束比计分条件宽是可核实差异，但不能仅凭这一点认定模型受此驱动而撞筐，
也未在本次分析中改指令。Level 2两例停滞视频后半程明显偏暗、视角有摆动，
可能增加视觉定位难度；没有亮度／相机／RTC对照，不能据此认定光照、RTC或SONIC中的某一个是唯一根因。
首秒初始化修复没有提供终端放置能力；新旧闭环有4例退步也已完整保留，不能宣称所有后续行为均改善。

**计分与证据边界。** `g1_wholebody_locomotion_pick_between_tables_teleop.py`的reward在
盒／筐条件成立且没有含`hand`的body接触目标时每次加0.02，否则清零。
`sonic_loco_manip.py`每个50Hz tick先算reward，再经`check_success`对同一状态再算一次；
本轮22个成功案例的最终正reward段全部为23tick，约覆盖0.46秒控制区间（首尾poststep采样点差0.44秒）。
所以原生0.9是累计门限，不能解释成必须稳定0.9物理秒；本次未改变计分实现。
容器条件实际表达式为`A or (B and XY)`，存在几何接触顺序相关的优先级，
并非对称的`(A or B) and XY`；诊断没有保存geom1／geom2顺序，不能声称只用pose／无序接触完整重演官方判定。
也没有额外的篮筐高度／直立硬门，不能将“落地”当成新增拒收标准；
这里的5例均以实际reward未达门、最终盒／筐分离判失败。

trace的物体pose／接触／reward来自`next_sim_time_s`，比同一行写入视频的prestep图像晚0.02秒，
复核按这一时序对齐。50Hz快照不能证明全部5ms子步接触，未保存原始手指动作／接触力，
故“最后手接触”不等于主动张手命令，视频和接触时序也不足以唯一识别推篮身体部位／冲量。
同步HTTP等待期间仿真钟暂停，全部8例在技术正常的策略运行中耗尽24秒任务预算，
与PICO断连、动作包超时、推理墙钟慢是不同问题。

改进优先级应为：先检查末端入筐轨迹、速度和承托后撤手的时序，再分析Level 2约10秒的持物推进停滞。
如测试更短末端执行窗、不同指令或接触确认，应作为单独标明协议的对照，不改这轮24秒预算、
成功标准或已有22/30结果，也不凭当前视频宣称存在已经验证的收益。
新增确认机制应使用部署可用的当前相机／已有本体反馈，不将诊断用物体真值、未来轨迹或teacher动作接入策略。

原始证据与新复核产物均在上述运行根的`failure_analysis_003`：

- `RECORDED_TRACE_METRICS.json`：只读复算全30例，含8失败时序、接触段、最后1秒样本、
  运输快照、输入summary／trace／result的SHA256；配套`analyze_recorded_traces.py`可复算。
- `level01/LEVEL01_FAILURE_ANALYSIS.json`、`LEVEL2_FAILURE_ANALYSIS.json`：8例视频观察及测量报告；
  `level01/`、`level2/`保存原视频抽帧和关键事件清单，保留原片路径和时间映射。
- `INDEPENDENT_TRACE_REVIEW_003.json`及其独立脚本：从原30份trace／result／请求日志核对
  分类、23tick计分段、0.5秒采样节奏和HTTP停仿真钟；视频因果解释另受上述可观测性限制。

本节没有新启动仿真、重跑失败案例或修改权重／控制器／评分逻辑。

### 5000步、30秒与更短执行窗的独立版本（2026-11-06）

用户在上述8例复核后授权优化、将任务时限改为30秒并改用5000步权重。
本次采用单一预先冻结的组合配置，不根据各case成败调参或自动重跑：

- 复用已完成严格CPU加载及独立Motion导出的5000步部署：
  `/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_ckpt5000_testtimertc_20261105/deployment`；
  模型SHA256为`50d12beded25fa0dce5be9153bf4ce9807fe89fb952ab21f7733011d04828ba4`。
  不使用6000步joint或其Motion，不启动／恢复训练，也不重复已完成的数据数值验收。
- 保留已验收`wrist_retreat_v1`初始化及有限臂交接、具名映射、原SONIC／DINO／统计／物理资源。
  仍恢复原Level 0／1／2各10份完整官方初始化，环境seed0，策略seed基数20261105。
- 最多1500个50Hz控制tick，即30仿真秒，包含原50tick／1秒真实历史预热；
  外层运行时限和实际Gym TimeLimit统一延长。归档metadata仍是原1200步／`.4`，
  不改写源文件、场景、物理参数或原生`.9`成功函数。默认24秒／1200步和matched协议保持。
- `test_time` RTC，预测30个30Hz动作、10步采样、d0／exponential／alpha0.9；
  实际执行6行动作／10个50Hz控制tick，即每0.2仿真秒重规划，旧计划实际左移6行。
  s仍显式为15，保留原逐位置软mask定义，尾补6行全部落在自由区。
  这避免默认s6扩大软mask覆盖，但不代表实际闭环引导作用不变：请求频率、shift及每次执行的前段均改变。
  新配置允许更频繁使用当前图像，是否改善放置和运输停滞必须由本轮真实结果判断。
- 不加入诊断物体真值、未来轨迹、teacher动作或特权接触纠错；原生成功即终止，
  无额外几何／脱手门，失败完整跑满1500tick；每例保留原视频和请求／控制时钟证据。

campaign的显式参数为`--checkpoint-step 5000 --seconds 30 --rtc-mode test_time
--execute-steps 6 --rtc-min-exec-horizon 15 --warmup-arm-profile wrist_retreat_v1`。
底层另有`--mobile-time-limit-seconds`，默认24；campaign会同步传入30，
避免仅修改外层`--seconds`、实际仍在1200tick截断。实际receipt同时保存source与effective预算。

独立运行根为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp5000_30s_fastRTC_20261106`，
正式输出为`official30_001`。已在2026-11-06 06:02:23–08:02:54 UTC实际完成全30例，
监督器真实返回0，运行源提交为`77dcba133ac6b49ea40e1f7ec6469ff392cb6272`。
30份独立逐例验收及最终验收均通过；state/final字节一致，
本次61个所属leader及进程组全部退出，没有停止原采集器或其他任务。

- 30秒原生成功为**17/30（56.7%）**，Level 0／1／2分别**8/10、6/10、3/10**。
  技术异常、未尝试、自动重试、机器人跌倒及策略前跌倒均为0。
- 同一条30秒轨迹的前1200tick内成功为**15/30（50.0%）**，分别7/10、6/10、2/10。
  加时新增2例：L0/case02在第1214tick／24.28秒成功，L2/case09在第1244tick／24.88秒成功。
  这是同轨迹24秒前缀，不是独立24秒重跑，也不按reward另行发明成功判定。
- 共实际36703个控制tick、36733个视频帧，1500个真实预热tick；
  sampler记录为30次plain初始化、3498次`test_time`、0次`train`。
  全部13个未成功case均完整运行1500tick，证明底层实际时限也已延长。
- 复用已通过的资源、模型严格加载及物理验收，不重做源数据数值审核。
  新增预算／RTC入口52项CPU回归通过；只读比较工具32项小fixture及独立review通过。
  比较本轮与6000步`wrist002`实际保存的各case前50帧，目标pose7及仿真时钟30/30完全相同；
  这是新保存记录之间的比较，不是重跑物理验收或使用目标真值控制策略。

真实逐例配对与旧5000步legacy／24秒版8/30相比：6例双方成功、11例双方未成功，
新增成功11例、退步2例，净增加9例。与6000步`wrist_retreat_v1`／24秒版22/30相比：
13例双方成功、4例双方未成功，新增成功4例、退步9例，净减少5例。
因此本轮组合没有超过此前6000步优化版本。
该组合同时改变权重相对6000版、初始化相对旧5000版、实际执行窗和预算，
不能将差异归因于单一因素；缩短执行窗并未由这组多因素结果证明必然改善放置。
30秒成绩不混作原24秒协议成绩，`formal_evaluation_eligible=false`保持。

13例原生未成功视频的代表帧与50Hz接触记录均另行查看，按观察阶段分组：

- 5例抓起后持续运输未完成：L1/case07、09，L2/case01、02、05。
  末段仍距篮约1.6–1.8米、保持手部接触，不能归为仅差一点放手时间。
- 2例运输停滞后晚接近、结束仍持物：L2/case00、08。
  case00直到29.725秒才首次碰篮，剩14tick且仍有右手接触；case08未记录到篮接触。
- 6例已到篮处的放置／撤手问题：L0/case01、04盒篮落离目标桌；L1/case00篮倾倒后
  盒子贴在髋部、无已证实落地；L1/case06盒子偏在篮沿后滑落，篮留在桌上；
  L2/case03晚放手并反复再接触清零，末18个保存点脱手有篮接触、累计仍仅0.70；
  L2/case06从17.545秒起连续624个保存点同时有篮及右手接触，未完成有效脱手。

这些是视频和保存接触点支持的失败阶段，不能唯一证明手指命令、速度命令或视觉模块的根因；
没有保存相应动作命令、接触力及连续base轨迹，不推断“只要再加时必成功”。
原生成功函数没有额外桌面支撑／姿态门，因此成功点的篮高度异常另行保留，
不据几何诊断重打分，也不将原生成功一律表述为已经验收的桌上放置。

其中L1/case05是**原生成功中的物理放置异常**：原视频确认盒篮跌离目标桌后才触发成功。
最后一次手部目标接触在第1090tick／elapsed21.80秒；篮体reference z在21.945绝对仿真秒
已降至约0.040米。第1124tick／elapsed22.48秒原生成功时，篮z约0.05679米，
比第一保存帧低约0.58396米；盒z约0.2406米，同时接触`left_knee_link`及`brasket`，双手已脱离。
这不是桌上放置。没有盒子直接`world`接触记录，不能进一步声称盒子自身必然直接触地。
应报告“17次原生通过中已复核发现1例物理放置异常”，保留原函数17/30，
不悄悄扣成16/30的另一套官方成绩。两个旧参考的原生成功保存点均没有篮reference z低于0.5米，
此高度仅作诊断线索，不构成新评分门，也不能仅凭高度验收其余成功的桌面支撑。

实际证据均在上述独立运行根，数据及视频不入Git：

- `INDEPENDENT_FINAL_ACCEPTANCE_001.json`：SHA256
  `c13a08ca4f07ada0a1a9a677fc1f55389aea9ac2cbf64e5111c02a3a8b659bcf`，
  全30覆盖、首次成功时点、HTTP／采样调用、视频／时钟及61所属进程退出核准。
- `FINAL_COMPARISON_001.json`：SHA256
  `83fe6a36d5037e04f222990aa5c416e3f6446da0c3e90034498c6ee7e26d8cf6`，
  两份旧结果的真实逐例配对、24秒前缀／加时、原视频索引及单列物理诊断。
- `failure_observation_001`至`005`及`failure_observation_level2_001`：
  13例原视频代表帧、接触时点、文件身份及观察限制；末者的`LEVEL2_FAILURE_GROUPING.json`
  SHA256为`bec5e53d12fe877fd95c8348a820b84c25b602699b63d156aa2795d706804bcc`。
- `native_success_observation_001/L1_CASE05_NATIVE_SUCCESS_PHYSICAL_OBSERVATION_001.json`：
  SHA256为`d4a3f8eb29e9bdd4e671e47348b57ae4b72a82621803ec4720f4896178b26d1f`，
  单列原生成功异常的视频观察、落桌时点及原件身份，不改变成功布尔值。
- `FINAL_COMPARISON_REVIEW_001.json`：独立比较review通过且突出上述物理异常，
  SHA256为`222365a4e8070805fd7799d622361706780c0741f8f509ce87c157e8dd934cde`。
- `Mobile_PP_5000_30s_videos_001.zip`：14段未裁剪、未转码原视频，13段原生失败位于
  `native_failures`，L1/case05异常成功另放`native_success_physical_anomaly`；该例仍保持原生成功。
  实际大小81123644字节，SHA256为
  `cbc6171539fb31b81ec5b1f2b2ce5f9ca91eaf90d2be58aed2774a4e3a74f789`。
  `VIDEO_DELIVERY_CHECK_001.json`记录新包CRC、14条唯一视频覆盖、大小及manifest一致检查通过；
  原视频沿用已独立验收SHA及复制前后stat，不重复原件hash／解码或改写文件。

## 相关工作的Mobile P&P报告值（2026-11-05核对）

这里记录原作者公开值，不将其当成本机统一配置复现。每等级10次的公开表中，
Ψ₀为7／5／6，即18/30（60.0%）；λ₀为9／7／7，即23/30（76.7%）；
StarVLA为9／8／8，即25/30（83.3%）。来源为
[SIMPLE官方当前表](https://github.com/physical-superintelligence-lab/SIMPLE#-simulation-benchmarking-results)和
[λ₀ v2 Table 2](https://arxiv.org/html/2610.18950v2)。
λ₀的90.0%是六任务180次的整体值；其Appendix D.2明确采用逐任务最佳记录配置，
未确立仅由验证集选配置，不能视为统一checkpoint／推理设置下的对照。

ACT还存在来源差异：SIMPLE当前表为6／5／7，λ₀表2引用为5／5／5，分别保留，不合并。
新运行根的`MOBILE_PP_RELATED_WORK_REPORTED_METRICS.json`、
`mobile_pp_related_work_reported_metrics.csv`保存12条主表记录及来源，含ACT两条版本。

另有[WholeBodyWAM（WBC-Grounded Coordination）论文](https://arxiv.org/html/2609.16644v1)
使用不同SONIC后训练与任务微调设置。其Table II的PickBetweenTables百分比和其他五个方法
保存为独立组`additional_SONIC_paper`及`mobile_pp_related_work_sonic_paper_metrics.csv`，
只列原百分比与三等级等权均值，不换算成官方30例计数，不将其六任务Overall混作本任务指标。
公开值的模型、训练数据、控制器与配置选择口径应随引用保留，不能据此声称本机结果是公平排名。
