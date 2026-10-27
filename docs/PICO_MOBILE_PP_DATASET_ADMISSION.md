# PICO Mobile P&P：固定 100 条核定与 A100 交付

2026-10-27，用户在采够后要求处理当前 **100 条 Mobile P&P 真人保留记录并上传 A100**。
本批从当时 121 条已编号记录中冻结 100 个 `selected` UUID；不追加后续录制，不用新记录替换失败候选。
采集器已按这次要求正常退出，`COLLECTION_FINISHED.json` 确认 collector、planner 和 preview 均已退出、无 active capture。
此后才开始 RTX 阶段；SDK 接收服务及其他任务没有因此停止或重启。

**本地阶段已通过：固定 100／100 条完成原积分状态 RTX、10 阶段独立审计、无损转换、真实 clean DINO
及 300 次 CPU 生产查询，共 61,806 帧、55,906 个训练窗口。A100 上传正在进行，远端字节核对、路径重绑和生产读取仍待验收。**
本地通过不代表 A100 已收到全部文件或已完成交付，最终状态必须以后文远端收据为准。
旧 Handover 固定 90 条的交付见[原交付记录](PICO_DATASET_ADMISSION.md)，其清单、合同和远端目录不用于本批。
采集操作、场景与官方成功语义见[Mobile P&P 采集说明](PICO_MOBILE_PP_TELEOP.md)。

## 固定清单与任务身份

本机运行根为 `/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027`，
交付准备目录为该根下的 `delivery100_001/`：

- `FROZEN_100.json`：用户范围冻结原件，记录 UUID、编号、原路径及三份原始摘要。
- `FROZEN100_INVENTORY.json`：流程使用的固定 100 条清单，合同为 `cvpr-pico-mobile-pp-collection-admission-inventory-v1`。
- `source_index.json`：冻结时集合索引副本，不在处理过程中重新扫描增长中的 history。
- `HISTORICAL_SOURCE_IDENTITIES.json` 与 `source_snapshots/`：实际录制各版本的 producer 源码摘要及对应原字节。
- `COLLECTION_FINISHED.json`、`A100_PREFLIGHT.json` 和阶段日志：退出、容量及验证证据。

冻结原件 SHA 为 `2f6356904e4d3d5912de9365e13d7e417685a8a2be3ed9b3a47c6830d0c41061`；
原集合索引 SHA 为 `4369b6a379e85cd4a70513b70dabd49358684e8f8b7b15dc70b0d102596ef690`。
处理计划 `delivery100_001/PROCESSING_PLAN.json` 绑定 `FROZEN100_INVENTORY.json` 的实际 SHA
`9b42cd4a6fcccb614ff7b01a310c414072281b662d8642c3112eee9da795c24e`，
固定处理代码提交 `69d306352760ed26cdee8e41af8d13f49cc273cb`、资源及独立输出目录。
无论新生成还是复用 `ADMISSIONS_SELECTION.json`，进入打包前都要重读并校验清单合同、冻结 SHA、
UUID 唯一且属于本批、原编号／session／路径／三份原 SHA、当前人工丢弃意图及当前检查器绑定。
本批流程要求固定 100 条全部独立通过才进入整包上传；不能静默换样本、扩充范围或上传不完整的 100 条包。

100 条均为 `continuous_vr_mode1`、实际 encoder mode1；任务必须逐条绑定：

```text
simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0
actual_task_source_sha256 = fee537d56e134b0399ff147013438168601498ab9b3ee068658dcad5fb8ca521
actual_native_success_criteria = 0.9
selection_policy = native_only
```

任务和资源使用官方 Mobile scene6（`102344022`）双桌布局及每条已封存的 reset 配置，
本地资源根为 `mobile_pp_teleop_20261027/resources/data_root/`。
实际 Mobile 任务类、源码、完整 reset 环境和 RTX manifest 均须一致；不能加载 Handover 房间补图。

## 成功判定与技术故障

任务成功仅按 `mobile-native-task-success-v2`：原始布尔 `official_task_success` 曾为 true，
表示曾达到实际官方环境的终止成功。实际环境每个 50 Hz 步两次更新 reward，阈值 `.9` 不是持续 0.9 秒；
环境不 latch，官方 eval 遇 terminated 退出，遥操作继续录制后的末帧不必仍为 true。
不增加 0.75 米位移、10 厘米抬升、持续持物、交接顺序或额外末段释放／稳定／几何成功门。

真人来源、原 capture 封存、SHA、30／50／200 Hz 时钟、有限性、原始外力和真实执行证据仍须完整。
只有技术证据完整且原始 native 从未成功，才可记为官方任务未成功。
缺文件、渲染或解码异常、源码身份不匹配等记录为技术失败，保留原件、旧收据和失败产物，停止当前流程并诊断修复；
不能把这些错误算成质量拒收，不能补录替换，也不能宣称交付完成。
独立审计复核已记录的人体参考及实际命令，不重跑 planner 推理；原始记录只保存了 planner 预测摘要，
不能宣称已从证据独立重现完整未来预测。

## 处理阶段与数据协议

1. `render_deferred_pico.py` 按每帧原 `mjSTATE_INTEGRATION` 恢复，再更新 FK／RTX；不重跑物理、策略或 SONIC。
   实际任务、完整场景 reset 和状态字节绑定；部分失败产物保留，不覆盖后假称首次成功。
2. `audit_pico_teleop_episode.py` 显式派发到 `pico_mobile_audit.py`，独立核对真人来源、封存边界、
   SONIC 固定身份、原 PICO 包与实际命令 ledger、dense RGB／状态／动作、原始 200 Hz 零外力、
   原积分状态与 RTX 桥接、官方 native，以及连续 mode1 人体／当前 root 参考运输。
   按角色核对实际历史 producer 源码与当前 auditor，不能因 checkout 前缀不同误拒，也不接受只换摘要的缺失原件。
3. `pack_pico_handover.py` 保留历史文件名，但按 Mobile 专属合同转换。新包为真人 `train_only`，
   不混入旧 TRAIN／VAL／TEST，不重新拟合统计，不重新编码实际执行的 64 维 SONIC token。
4. 43 维实际关节状态保留，生产 reader 按继承协议补到 45；实际 64＋14 维动作补到 80，末两维监督关闭。
   前 30 行真实 warmup RGB／状态保留作历史，动作标签为 NaN、`action_valid=false`、80 维 mask 全 false。
   第 30 行起只保留未来 30 步完整的 current，六帧历史为 `[-30,-24,-18,-12,-6,0]`。
5. 视频无损编码并逐像素解码比较，Parquet 逐值回读。名义时间为 `row/30`，物理来源索引为
   `ceil(row*50/30)`；DINO sidecar 的 `source_frame_index` 是连续视频解码序号 row，不能混用。
6. 真实 clean DINO 使用 `dinov2-with-registers-large`、`256x320`、CLS＋4 registers、`[N,5,1024]`、
   float32 存储，沿用已验收的权重／预处理／bf16 推理身份。新 RGB 必须重新提取真实特征。
7. 本机实际 CPU 生产 Dataset／collator 核对完整历史、未来动作、当前 RGB、DINO 和归一化数值，
   通过后写 `PICO_DATASET_READY.json`。`PACK_READY.json` 仅代表无损包，不代表生产就绪。
8. 上传前重读本地所有绑定，远端核对全部 SHA、文件大小及纳秒 mtime。字节上传通过仍标记路径重绑与生产读取待完成。
9. A100 在独立已提交代码快照中重绑 DINO 路径、保留 `dino.before-remote-relocation`，
   训练文件及特征数组字节不改。再做真实 CPU 生产读取并核对最终收据及所有当前文件身份。

这些入口只在采集器、自动后处理和其他 GPU 计算退出后进入 GPU 阶段；资源等待不算数据失败。
不停止、重启或修改其他进程。数据交付流程本身不修改或启动 A100 训练；
用户后续授权的独立训练另见[训练准备说明](#用户后续授权的独立训练尚未启动)。

## 独立合同与远端就绪

Mobile 使用明确的合同和 exact task ID；旧 Handover 合同仅保持旧入口兼容：

- 处理计划：`cvpr-pico-mobile-pp-admission-processing-plan-v1`。
- 通过清单：`cvpr-pico-mobile-pp-admissions-selection-v1`。
- 独立审计：`cvpr-live-pico-mobile-pp-independent-admission-v1`。
- 无损包：`cvpr-pico-mobile-pp-lerobot-pack-v1`。
- 本机／A100 生产读取：`cvpr-pico-mobile-pp-production-verification-v1`。
- 上传及字节核对：`cvpr-pico-mobile-pp-a100-byte-transfer-v1`／`cvpr-pico-mobile-pp-a100-byte-verification-v1`。
- 远端 DINO 路径重绑：`cvpr-pico-mobile-pp-dino-host-relocation-v1`。

本批远端目标是全新独立目录：

```text
/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261027_human100_01
```

其中 `packed/train/` 为训练数据，`dino/` 为特征；不能复用旧 Handover90 或现有训练目录。
2026-10-27 的只读 preflight 观测该目标尚不存在、父目录可用 `7,323,607,105,536` 字节，
所需 Python、DINO 参考和读取配置存在。该检查隐藏 GPU、nice19，仅约束自身 CPU127，未修改训练。
容量记录是当时快照，不等于已上传或未来可用量承诺。

上传与远端检查均隐藏 GPU、nice19，且只限定自身一个允许 CPU；自动解码线程继承亲和性，
不声称整机独占或总 OS 线程数为 1。代码通过 `git archive` 固定提交保存到目标下 `tools/code_提交/`，
不切换训练 checkout。最终必须实际获得 `A100_PICO_DATASET_READY.json` 的
`status=passed`、`remote_production_ready=true`、`DINO_relocation_pending=false`、`training_started=false`，
并回读验证其任务合同、代码、计数和当前文件身份，才可报告 A100 数据交付完成。

## 准备验证与保留的技术失败

准备阶段 **114 项 CPU 回归通过**：Mobile helper／独立审计 54 项、渲染派发 5 项、
固定清单流程 22 项、转换／就绪／上传 33 项。覆盖跨任务合同拒绝、缓存清单禁止替换 UUID、
技术失败不冒充 native 失败、真实无损视频回读、DINO 路径重绑、远端字节脚本本地执行及子进程 CPU 约束。
`delivery100_001/delivery_cpu_tests_final.log` 记录 114 项、17.093 秒、`OK`；未因后续本地通过删除旧测试记录。

首个实际样本 002 的早期 RTX 尝试曾被旧 Handover 环境配置检查错误拦住；这是技术派发错误，未算任务失败。
修复后该样本的 1,020 帧 RTX 与 10 阶段独立审计通过。
早期 `first_mobile_rtx_002.log`、`first_mobile_rtx_002_retry01.log` 与原始录制证据保留，
不能把最终通过改写成从未发生过技术失败。

## 2026-10-27 实际本地验收

独立本地发布目录为：

```text
/home/zxc/CVPR_Evaluation_runtime/pico_mobile_pp100_20261027_human01
```

其 `PROCESSING_STATUS.json` 记录 `status=local_verified_uploading`，100 个 UUID 全部
`eligible_for_packing=true`，与冻结清单完全一致，没有拒收、追加或替换。
包内 100 份独立 admission 各有 10 个检查阶段且全部通过；RTX、无损 MP4 逐像素回读与 Parquet 逐值回读已完成。
`packed/PACK_READY.json` 实际记录 100 条、**61,806 帧和 55,906 个完整训练窗口**；
该窗口数已由本地生产读取验收，不再只是 `61,806 - 59×100` 的协议推导。
真人来源保持 `train_only`，统计继承原合同，实际 SONIC token 未重新编码。

`dino/manifest.json` 的 `state=complete`，100 份 episode cache receipt 全部完成，
帧数合计 61,806。真实 clean DINO 沿用 `256x320`、CLS＋4 registers、每帧 `5×1024`、
float32 存储和 bf16 推理身份，`extraction.training=false`，没有在线增强。
清单保存图像处理 profile 的完整定义，并不表示本轮提取启用了其中训练增强。

`PICO_DATASET_READY.json` 的 `status=passed`、`production_ready_for_bound_pack_and_cache=true`。
本地真实 CPU Dataset／collator 对每条首／中／末查询，共 **300 次生产查询**，
核对完整 30 步未来动作、六帧 DINO 历史、当前 RGB、mask 与归一化数值；
另核对全部缓存的形状、有限性及文件绑定。该 CPU 验收回执中的
`GPU_used`、`CUDA_initialized`、`policy_constructed`、`training_started` 均为 false；
这些标志描述 CPU 验收过程，不否认此前 RTX 和 DINO 阶段使用了 GPU。
该检查没有训练模型，也不是闭环任务成功率测评。

已读取实际本地回执并核对的 SHA：

- `packed/PACK_READY.json`：`5208307c097baf7bae6553cc7fdc5db51d0a6ef21efe9ab4c9b9e6e2599e6957`。
- `dino/manifest.json`：`a19b709473e8c884eb24034469fc8bf5fa890f315b1f380c1bd556b4d4dca199`。
- `PICO_DATASET_READY.json`：`81e4fe7550f028940abd82884bef854573b57b2c05db6f28557cc638b2615b3c`。

## A100 上传进行中与待验收阶段

`A100_TRANSFER_MANIFEST.json` 已固定 **3,070 个文件、37,307,298,981 字节**，
绑定上述本地就绪 SHA 和独立 Mobile 远端目标；此计数表示待传输清单，不表示远端已完成校验。
本地流程已进入上传。北京时间 22:40:40–22:40:50 的只读网络抽样中，
上传专属 SSH 连接约 10.109 秒获 TCP ACK 50,983,680 字节，约 5.04MB/s，rsync 读写计数同步增长。
这是短时实际传输进展，不是全量吞吐验收、完成百分比或剩余时间承诺。
`rsync --info=stats2` 主要在结束时输出统计，日志暂不增长不能单独作为停滞证据。

**目前不能宣称 A100 上传完成、远端生产读取通过或本批交付完成。**
本地就绪回执仍为 `remote_production_ready=false`、`DINO_relocation_pending=true`。
剩余阶段是全部文件远端 SHA／大小／纳秒 mtime 核对、固定提交独立快照中的 DINO 路径重绑，
以及 A100 实际 CPU 生产读取和最终 `A100_PICO_DATASET_READY.json` 回读验收。
期间不修改或启动 A100 训练。所有运行数据、日志、资源与权重均保存在仓库外。

## 用户后续授权的独立训练：尚未启动

用户最新要求：**等本批上传和 A100 实际生产就绪验收全部通过后，沿用上次 Handover114 的实际配置
训练一次，从 4,000 步开始每 500 步保存模型。** 当前仍在上传，新训练入口
[`a100_mobile_pp100.py`](../scripts/4090_simple/a100_mobile_pp100.py) 已通过独立 review 和 38 项 CPU 回归
（19 项新入口＋19 项旧入口），但尚未准备或启动；实际 GPU、完整训练及13次保存仍待验收。
远端 READY 也尚未通过。新的 prepared／run 拟用 `/root/workspace/mobile_pp_human100_train_20261027`，
只读检查时该目录尚不存在；不能复用旧 Handover run 或向交付清单中加入训练产物。

本地已只读归档上次配置、实际 resolved config、启动记录及源码原字节，位于：

```text
/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027/training100_001/reference/
```

`REFERENCE_SUMMARY.json` 与 `source_identity_001–004.json` 记录远端原路径和 SHA。
旧实际 resolved config SHA 为 `2b912d75731767e080f864a1bf3f61496793781cb3f2efd60229f7222016ab2a`；
训练源码冻结提交为 `632968de7fb1557a0c52b299ad1da07d8d10c64b`。
新入口以这些原字节核对参考配置，允许新数据／输出路径、实验标识和本次保存节点等明确改动，
其余训练配置不得静默漂移。

沿用的实际配置为：

- 严格载入原联合预训练 `ckpt_38000/model.safetensors` 的**模型权重**，新任务从 optimizer step0 开始；
  `fresh_run=true`、`resume_from_checkpoint=null`，不恢复旧 optimizer、scheduler 或 RNG。
  初始化权重的原记录 SHA 为 `12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7`。
  4,000 是首个保存点，不是从旧任务第 4,000 步恢复。
- 8 卡 DDP，每卡 batch8、梯度累积1，全局 batch64；10,000 个 optimizer update、warmup1,000、cosine、bf16，
  seed=`20261025`，每 rank 4 个 spawn 数据 worker，梯度裁剪1.0，原训练安全检查保持。
- 实际 fused AdamW、`foreach=false`，betas=`[0.95,0.999]`、eps=`1e-8`、weight decay=`1e-6`。
  Action／MM projector 峰值 LR=`1e-5`，Motion／vision=`1e-6`，language=`1e-7`；
  VLM、Motion 和 Action 按旧实际配置联合更新，状态保持策略不变。
- 保存 **4,000、4,500、5,000、…、10,000，共 13 份模型权重**，4,000 前不保存模型。
  保留所有这些模型，仅在这个新 run 内保留最近 2 份完整恢复状态，不清理旧 run。
  当前实现用 `checkpointing_steps=10000`、显式 `early_checkpoint_steps=4000:500:9500`、
  `preserve_checkpoint_every=500` 达成，不改变共享 trainer 的保存语义。
- `validation_steps=1000`、val batch1、`val_num_batches=-1` 沿用旧配置；
  实际触发保持旧零基 global_step 逻辑及最后一步检查，不因模型保存改成500步验证。

**验证只是历史 Handover 回归监控，没有 Mobile 验证集成绩。**
旧 `data/val` 软链实际指向 `handover_task50_30hz/Handover-sonic11-val`，
它含 5 条官方 Handover level0 episode（源编号 39／65／80／92／94）、1,562 帧，
旧 reader 每条选 8 个固定窗口，共 40 个；`val_num_batches=-1` 遍历的是这 40 个已选窗口。
对应 DINO 为 `dino/handover_task50_30hz/Handover-sonic11-val`。
新入口明确标记 `historical_handover_regression_only_not_mobile_validation`；
本批 Mobile 100 条继续全部为 `train_only`，不拆出人工 VAL，不把 Handover loss 报成 Mobile 验证损失。

下面是已落盘入口的命令模板，**只有最终 A100 READY 通过后才适用**。在远端包含该入口的独立代码根下运行；
`MOBILE_TRAIN_REFERENCE` 和 `MOBILE_A100_READY_SHA256` 分别使用已部署原字节参考目录和实际最终远端收据 SHA。
`training100_001/REMOTE_CPU_SCOPE_001.json` 记录普通训练会话完整允许 CPU 为 `0-127`，
此次只读检查自身则为 CPU127／nice19／GPU隐藏；命令中的 `--training-cpus 0-127` 声明以后实际训练的范围，
不是让 CPU 准备进程占用全部 CPU。

```bash
MOBILE_TRAIN_OUTPUT=/root/workspace/mobile_pp_human100_train_20261027

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 127 nice -n 19 \
  /gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126/bin/python -B \
  scripts/4090_simple/a100_mobile_pp100.py prepare \
  --release /gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261027_human100_01 \
  --reference "$MOBILE_TRAIN_REFERENCE" --output "$MOBILE_TRAIN_OUTPUT" \
  --expected-ready-sha256 "$MOBILE_A100_READY_SHA256" --training-cpus 0-127

env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 127 nice -n 19 \
  /gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126/bin/python -B \
  scripts/4090_simple/a100_mobile_pp100.py check --prepared "$MOBILE_TRAIN_OUTPUT"
```

准备与 CPU check 显式隐藏 GPU、nice19、仅绑定自身 CPU127，不启动 GPU 训练。
实际启动须在这些检查通过后，使用**另开的普通 SSH 会话**，
保持已声明的 CPU0-127 完整训练亲和性及 nice0，再执行 `bash "$MOBILE_TRAIN_OUTPUT/launch.sh"`。
不能从 GPU 隐藏、nice19、仅 CPU127 的只读检查包装器继承启动训练。
新入口在准备前要求实际远端 READY，并在启动前检查独立输出容量和所选 GPU 状态；
这些启动检查及13次保存的真实结果仍待执行，不能用本地 CPU 测试代替。
