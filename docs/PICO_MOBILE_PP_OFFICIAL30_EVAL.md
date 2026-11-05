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
学习策略成功率已提高。最大倾斜增加的代价须随结果保留；新完整30例学习策略／RTX闭环尚未执行，
不能将原15/30替换成新成功率。现有采集器及其它GPU任务未停止或重启。

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
