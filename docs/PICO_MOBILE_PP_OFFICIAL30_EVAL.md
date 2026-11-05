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
准备时本机没有该新6000步权重；A100原SSH端点连续超时，已请求用户确认开机或新端口。
**当前尚未下载本次模型、未启动策略或物理评测，没有30例成功率。**

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

取得真实备份后，在仓库根目录运行以下两步。
`MOBILE_MODEL_SHA256`必须填入源／下载后已实际核准的SHA，不能使用旧模型或占位值。

```bash
MOBILE_EVAL_ROOT=/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_20261105
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

上述命令仅为已经准备的操作入口，不是模型已下载或评测已执行的记录。
每次campaign必须使用新目录，`state.json`记录实际进度，
`FINAL_EVALUATION_ACCEPTANCE.json`只有`status=complete`且30个不同case均有效时才代表完整完成。

新增两个入口的28项CPU回归已通过，覆盖停止／保存证据、远端DINO不能被本机原manifest替代、
实际小型safetensors到新Motion导出及两入口合同衔接、30份固定case、部署文件变更拒绝、
技术异常继续其余case、原生warmup成功、HTTP／视频验收和PID复用拒绝清理。
最终版本记录位于运行根的`CPU_REGRESSION_002.json`及日志，首次001仍保留；
因补充SIMPLE与视频源码绑定而重新检查，检查只限制自身CPU26／nice19并隐藏GPU。
这些是新工具的CPU回归，不是本次12GB真实权重严格加载或GPU闭环通过收据。

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
