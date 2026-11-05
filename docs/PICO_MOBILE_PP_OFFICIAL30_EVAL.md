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
