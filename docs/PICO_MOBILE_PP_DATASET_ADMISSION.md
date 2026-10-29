# PICO Mobile P&P：固定 100 条核定与 A100 交付

2026-10-27，用户在采够后要求处理当前 **100 条 Mobile P&P 真人保留记录并上传 A100**。
本批从当时 121 条已编号记录中冻结 100 个 `selected` UUID；不追加后续录制，不用新记录替换失败候选。
采集器已按这次要求正常退出，`COLLECTION_FINISHED.json` 确认 collector、planner 和 preview 均已退出、无 active capture。
此后才开始 RTX 阶段；SDK 接收服务及其他任务没有因此停止或重启。

**本批 A100 数据交付已完成：固定 100／100 条完成原积分状态 RTX、10 阶段独立审计、无损转换、真实 clean DINO，
共 61,806 帧、55,906 个训练窗口；本机和 A100 分别通过 300 次实际 CPU 生产查询。**
初次上传的全部文件身份核对、远端 DINO 路径重绑及最终就绪收据回读均已通过，详见下文实际 A100 交付记录。
用户随后授权的独立8卡训练已完成10,000步并保存全部13份模型，末步GPU健康检查通过；
最终9500／10000完整恢复状态的CPU验收和精确离线指标解析也已实际通过。
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
用户后续授权的独立训练另见[完整训练结果与验收](#用户后续授权的独立训练完整训练结果与验收)。

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

100 个 UUID 全部 `eligible_for_packing=true`，与冻结清单完全一致，没有拒收、追加或替换。
本地发布目录保留本机就绪回执，并在远端实际验收后回存最终 A100 回执。
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

## 2026-10-28 实际 A100 交付完成

独立目标 `/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261027_human100_01` 已完成交付。
`A100_TRANSFER_MANIFEST.json` 固定的 **3,070 个文件、37,307,298,981 字节**全部通过远端 SHA、大小及纳秒 mtime 核对。
`A100_BYTE_VERIFICATION.json` 的 `status=all_copied_bytes_and_identities_verified`；
它保留字节阶段的 `remote_production_ready=false`／`DINO_relocation_pending=true`，属于该阶段原始收据，
后续最终验收不回写这些历史状态。

远端使用数据代码提交 `69d306352760ed26cdee8e41af8d13f49cc273cb` 的独立快照，
完成 DINO 本机路径重绑与签名更新，保留 `dino.before-remote-relocation` 原缓存。
训练数据与特征数组保持原字节，`PACK_READY.json` 没有改写；重绑后的 DINO manifest 使用新的实际摘要。
实际 A100 CPU Dataset／collator 通过 **300 次首／中／末生产查询**，覆盖全部100条，
最终收据绑定 **3,171 个当前 payload 文件**以及100份cache记录；这与初次传输清单属于不同阶段，计数不可混用。
验收进程隐藏 GPU、nice19、仅限定自身 CPU127，没有启动策略或训练。

`A100_PICO_DATASET_READY.json` 于北京时间 **2026-10-28 00:37:28**生成，
`status=passed`、`remote_production_ready=true`、`DINO_relocation_pending=false`，
记录100条／61,806帧／55,906窗口。回读时再次核对当前文件身份，
`PROCESSING_STATUS.json` 于 **00:40:55**记录 `A100_ready_for_training_not_started`，流水线正常退出。

最终实际摘要为：

- `A100_BYTE_VERIFICATION.json`：`20c90142d2950ed99ef2ffaa410379b07ea51cf8924eb6ee2a410ec743de7cea`。
- 重绑后的 `dino/manifest.json`：`8369929e3e6c89c97192a107a2ab18b86d206d4e79084be2d16bca154ba46c27`。
- 未改写的 `packed/PACK_READY.json`：`5208307c097baf7bae6553cc7fdc5db51d0a6ef21efe9ab4c9b9e6e2599e6957`。
- 最终 `A100_PICO_DATASET_READY.json`：`38b15799606518761eb1e45929be6b189c1bf79a83c690c1a6e781dea2663d21`。

本机 `PICO_DATASET_READY.json` 仍保存当时本地验收的 pending 标志，远端最终状态以独立 A100 回执为准。
最终数据回执中的 `GPU_used=false`、`CUDA_initialized=false`、`policy_constructed=false`、`training_started=false`
描述**这次 CPU 数据验收事件**；不否认之前的 RTX／DINO GPU 计算，也不禁止用户随后授权的新训练。
本次数据交付没有修改旧训练或旧数据。所有运行数据、日志、资源与权重均保存在仓库外。

## 用户后续授权的独立训练：完整训练结果与验收

用户最新要求：**等本批上传和 A100 实际生产就绪验收全部通过后，沿用上次 Handover114 的实际配置
训练一次，从 4,000 步开始每 500 步保存模型。** 数据交付与实际 A100 READY 现已完成；新训练入口
[`a100_mobile_pp100.py`](../scripts/4090_simple/a100_mobile_pp100.py) 已通过独立 review 和 38 项 CPU 回归
（19 项新入口＋19 项旧入口）。训练入口提交 `3e7aa6ce507a0103afd88caf107ce71dbc881e02` 已单独部署到
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp100_training_20261027/code/3e7aa6ce507a0103afd88caf107ce71dbc881e02`，
22个部署文件的SHA已核准。首轮 `/root/workspace/mobile_pp_human100_train_20261027` 的启动被容量门拦住，
该次未启动 GPU；第二次准备改用数据盘独立目录
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp100_training_20261027/run_001`，
已通过 prepare、CPU check，并于北京时间 **2026-10-28 00:46:39**实际启动8卡训练。
该唯一run已于北京时间 **10月28日08:02:16**完成10,000步并正常退出，13份模型均已保存，10000步GPU健康检查通过；
最终完整恢复状态CPU验收及离线指标回执均为 `passed`，详见本节末的结果与验收范围。
不复用旧 Handover run，也不向数据交付清单中加入训练产物。

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

### 输出盘容量门与第二次准备

首轮拟输出所在 root 盘，`supervisor.log` 拦截时记录可用 `61,002,874,880` 字节，
稍后只读 stat 为 `61,002,878,976` 字节，均小于启动门要求的 `250,339,174,916` 字节。
因此该次在 GPU 启动前拒绝继续，没有生成 `training_state.json`。
旧 `/root/workspace/mobile_pp_human100_train_20261027` prepared、首轮 `DISPATCH_RECEIPT` 及
`supervisor.log` 中的容量错误保留，不覆盖、不清理旧权重。
这属于输出容量技术尝试，不计为数据拒收或 GPU 训练失败，已通过的 A100 数据就绪结果不变。

随后只读检查数据盘可用 `6,998,150,086,656` 字节，新 `run_001` 当时尚不存在。
第二次 prepare／CPU check 已在上述数据盘新目录通过，仍为入口提交 `3e7aa6ce507a0103afd88caf107ce71dbc881e02`，
仅改变独立输出及其 data 绑定路径，初始化、训练数值配置与13个保存节点不变。
这次数据盘训练随后已完成10,000步，全部13份模型已保存；容量门的首轮技术尝试记录继续保留。

下面是本次已执行入口及绑定的**等价历史命令记录，不是待操作模板**。
**`run_001` 已完成训练，不对这个既有 run 再次 prepare、run 或 launch。**
原字节参考位于已部署入口快照的 `reference/`，最终远端收据 SHA 已明确写入。
`training100_001/REMOTE_CPU_SCOPE_001.json` 记录普通训练会话完整允许 CPU 为 `0-127`，
当时只读检查自身则为 CPU127／nice19／GPU隐藏；命令中的 `--training-cpus 0-127` 声明实际训练的范围，
不是让 CPU 准备进程占用全部 CPU。

```bash
# 已执行入口的等价历史记录；不要对当前 run 再次执行。
MOBILE_TRAIN_SOURCE=/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp100_training_20261027/code/3e7aa6ce507a0103afd88caf107ce71dbc881e02
MOBILE_TRAIN_REFERENCE="$MOBILE_TRAIN_SOURCE/reference"
MOBILE_TRAIN_OUTPUT=/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp100_training_20261027/run_001
MOBILE_A100_READY_SHA256=38b15799606518761eb1e45929be6b189c1bf79a83c690c1a6e781dea2663d21
cd "$MOBILE_TRAIN_SOURCE"

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
本次实际启动在这些检查通过后，使用**另开的普通 SSH 会话**，
保持已声明的 CPU0-127 完整训练亲和性及 nice0，执行独立 run 的启动入口
（等价于 `bash "$MOBILE_TRAIN_OUTPUT/launch.sh"`，已执行，不再重跑）。
不能从 GPU 隐藏、nice19、仅 CPU127 的只读检查包装器继承启动训练。
新入口在准备前要求实际远端 READY，并在启动前检查独立输出容量和所选 GPU 状态；
这次实际启动已经通过这些门，不能把CPU测试或短时健康检查当作完整训练与保存验收。

### 2026-10-28 实际启动与前20步GPU健康检查

`training100_001/PREPARE_CHECK_RECEIPT_002.json` 的 `status=prepared_and_cpu_check_passed`，
绑定最终 A100 READY、100条／55,906个Mobile训练窗口与40个历史Handover回归窗口。
实际 `run_config.json` SHA 为 `854042b54d4b31d8b30c495539c4b220cf9dfa31b5570d8e052111a9137683b8`，
`argv.txt` SHA 为 `a8d73892ea5449d9852dea28a399e1ca91c40056b30ca9e4cbc0f9e5800b1559`。
启动前 `launch_acceptance.json` 核对3,171个当前payload及初始化权重SHA后通过。

实际 project 目录为：

```text
/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp100_training_20261027/run_001/training/posttrain/mobile_pp100_joint38000_statekeep_8gpu.us.flow1000.cosine.lr1.0e-05.b64.gpus8.261027164426
```

训练于 UTC `2026-10-27T16:46:39.202102Z`（北京时间10月28日00:46:39）启动，
torchrun PID为 `1581879`，launcher PID为 `1581769`；实际使用CPU0-127、nice0，未修改其他进程。
原数据验收时的 `training_started=false` 继续保留为历史事件，当前训练状态由新run单独记录。
实际初始化回执确认 joint38000 严格模型加载、fresh optimizer／scheduler／RNG、global batch64、
fused AdamW／`foreach=false`；8个rank的状态策略迁移回执均为 `passed`。

历史 `TRAINING_HEALTH_ACCEPTANCE_002.json` 及其绑定的当时状态快照记录：

- 2步和20步GPU健康检查均通过，每卡实际动作形状 `8×30×80`，8个rank每个batch均同步梯度更新。
- 各卡loss及clip前梯度有限，fused优化器及内存健康检查通过；
  每个可训练张量抽4值、每rank共7,700个探针，跨rank一致。
- 该证据是**前20步的有界探针检查**，不表示全部参数逐位一致、全程训练通过或任务准确率。
- 健康回执生成时为北京时间 **00:54:06**，尚无checkpoint，符合首个模型在4,000步保存的策略；
  `all13_checkpoints_verified=false`、完整训练与13份保存尚未完成，均是**当时**的pending事实，不能继续当作当前状态。

健康回执 SHA 为 `b24f3657e1ee20b01a7e67b30f5ee82a49771adf7ea76d4eb1e69a07ee63c292`。
这些是已发生的启动与短时健康记录，不将其作为之后训练状态持续不变的保证。

### 完整训练、13份保存与精确离线指标

`FINAL_STATUS_PRECHECK_001.json` 和更新后的 `TRAINING_STATUS_LATEST.json` 已确认唯一run为
`state=complete`、`returncode=0`。UTC启动时间为 `2026-10-27T16:46:39.202102Z`，
结束时间为 `2026-10-28T00:02:16.369598Z`，即北京时间00:46:39至08:02:16，耗时约7小时15分37秒。
所有4,000／4,500／…／10,000共13份模型已保存，保留记录中的完整恢复点为9,500和10,000。
10000步GPU健康回执为 `passed`，末步各优化器组LR均为0；
loss、clip前梯度、fused优化器及内存健康和跨rank的7,700个有界探针均通过。
这些检查不表示全参数跨rank逐位相等，也不等同于任务成功率或GPU恢复训练测试。

`offline_monitor/METRICS_ACCEPTANCE_001.json` 的 `status=passed`，只读解析
`run-nn7ebojc.wandb` 的10,000条history、11次eval和exit0记录，原文件26,439,494字节、CRC及读取前后身份检查通过。
W&B step采用零基：`step=9999`对应已完成10,000次更新，准确指标为：

- 最后一个训练batch的 `train/loss=0.4521014392375946`，不是整轮平均loss。
- 历史Handover固定5条／40窗最终 `eval/loss=10.945638656616211`，
  normalized body-token L1=`0.12158753722906113`、hand-joint L1=`0.23109500110149384`。
- 首次回归eval为 `8.88845157623291`（完成1次更新）；最低为 `8.308838844299316`（完成1,001次更新），
  此后最终回归loss升高。eval记录在W&B step0、1000、…、9000及9999，不能把step1000写成仅完成1000次更新。

以上eval始终是**历史Handover回归监控**，不是Mobile验证集、Mobile泛化成绩或物理成功率。
指标回执SHA为 `a74f99e79e8c65a95fab708fa19625fda35766dc04c3a226f3c630f9e0011437`，
原离线文件SHA为 `fec18eb14a6b5f136b49b220ecc86d1cb12d215070a409142b6b803b6573e062`。

### 最终模型与完整恢复状态CPU验收通过

`training100_001/FINAL_TRAINING_ACCEPTANCE_001.json` 已实际生成，`status=passed`，
验收时间为北京时间 **10月28日08:09:32**（UTC `2026-10-28T00:09:32.049760Z`），
SHA为 `d1fea82b176d5ef4eab4df28cdec2ab2b150fb7eabf4cbfe76112616f8702e50`。
该回执绑定已结束的唯一run、配置、保留策略、末步GPU健康记录及下列文件验收：

- **13份模型身份全部通过。** 前12份复用先前已核SHA且stat身份未变的收据，仅补算10,000步模型SHA；
  验收各模型头部、大小及稳定文件身份，不把复用收据写成重新扫描了全部13份权重。
- **9500／10000两个完整恢复点通过。** 两个model的所有张量、Adam一阶／二阶moment全部有限，
  二阶moment非负，5组优化器recipe核对通过；各恢复点的optimizer、scheduler及8个rank RNG文件均核对完整SHA和稳定身份。
- Adam按参数实际梯度参与保留稀疏计数，9500／10000的已初始化state均为1,784份，
  counter范围分别为1–9500和1–10000，合法性及跨恢复点转移通过；不错误要求每个参数的Adam step都等于全局N。
- 8卡实际scheduler采用 `last_epoch=8N`：9500步为76,000、10000步为80,000，
  `_step_count` 分别为76,001和80,001；这不是80,000次optimizer更新，末恢复点5组LR均为0。
- 两点均具备rank0–7共8份RNG文件，CPU／CUDA RNG保存结构检查通过。
  验收进程隐藏GPU、nice19、仅CPU127，`CUDA_initialized=false`；没有恢复CUDA RNG或重启GPU训练来证明恢复运行。

最新权重位于上述project的 `checkpoints/ckpt_10000/model.safetensors`，
大小为 **12,066,487,572字节**，SHA为
`9b1ea759ef3aedcc7697d62f6ff36deb12dcc2e379cce61c6ca904e80826b390`。
最终结论为10,000步正常完成、13份模型及最近2份完整恢复状态通过已声明范围的验收；
它不表示GPU恢复训练已实测、任务成功率已通过或全参数跨rank逐位相等。
历史Handover回归loss后期升高的事实和“没有Mobile验证集成绩”的边界仍保留。

独立最终review无阻断，记录为 `training100_001/FINAL_REVIEW_001.json`，
SHA为 `020ec95da0dfd4adb2eb6402bf40b0c42d8119e206ba70d2079054f85e54c567`。
本批最终验收已完成，该run后续检查即将停止；执行状态见 `CONTINUATION_MONITOR.json`。

## 最新task-step10000的本机闭环诊断

本轮使用已经完成Mobile100训练的 **task-step10000**，在本机运行官方level0固定case 0–9。
有效campaign `official_level0_003` 的10个case均已完成，运行返回码为0，最终CPU验收通过。
技术有效10/10、原生成功0/10、机器人跌倒0/10、首次策略命令前机器人跌倒0/10。
评测证据保存在仓库外：

```text
/home/zxc/CVPR_Evaluation_runtime/mobile_pp100_eval_20261028
```

下文将该目录记作 `E`。本轮模型SHA为
`9b1ea759ef3aedcc7697d62f6ff36deb12dcc2e379cce61c6ca904e80826b390`，
大小12,066,487,572字节；task-step10000指本次Mobile任务完成10,000次更新，
源模型初始化为UnifoLM joint38000，不能将其写成Mobile训练48,000步。

### 备份、CPU准备与实际在线DINO身份

最终评测绑定的当前备份记录为 `E/source/backup_receipt.json`，
SHA为 `9367ec0a07c84f0873b8e22c80818bf3a50d549f861b392c4c0999a6c7ccefe7`。
原 `E/BACKUP_TRANSFER_RECEIPT_001.json`（SHA
`0a90df75858457e9f64a653f39e71afb223e5628f7c7342c8801f10a942178cc`）仍保留作为早期传输收据。
备份未复制optimizer，未修改训练；源模型复用既有最终SHA并核对前后stat身份，
不将此记录描述为重新扫描全部训练恢复状态。

`E/deployment/LOCAL_RUN_READY.json` 的状态为 `local_mobile_pp100_run_prepared`，
SHA为 `49590746afab48626e40db55f08e1cd1a3c5d92ed6a10a54a84619b7c5bf75e5`。
保存的VLM、ActionExpert及MotionExpert组件均通过严格CPU加载；Motion参数实际取自同一份
Mobile task-step10000联合权重，没有替换为先前Motion参数。准备过程只重绑本机路径，
沿用统计SHA `931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4`，
未重拟合归一化，未恢复optimizer／RNG，未执行策略forward或初始化CUDA。
这份CPU准备回执本身不证明物理任务成功。
本轮CPU检查共48项通过：准备检查28项由 `test_prepare_mobile_pp_eval.py` 的13项新测试、
`test_prepare_task_snapshot.py` 的9项共享回归和 `test_export_bundle.py` 的6项共享回归组成；
另有Mobile协议6项、serve4项、smoke5项、Handover兼容5项。
该总数包含共享／兼容回归，不代表48项全部新增；之后另完成下述真实10例GPU闭环。

十例各自的 `ONLINE_DINO_RUNTIME_ACCEPTANCE.json` 均实际 `passed`，
相同内容SHA为 `0a5bad271d199a80af3e4a87e734a6d397691ae0fda4b52f309eb79efac3fbdb`；
例如 `E/official_level0_003/case_00/ONLINE_DINO_RUNTIME_ACCEPTANCE.json`。
这些回执将每个服务实际启动的DINO身份与Mobile训练manifest
`8369929e3e6c89c97192a107a2ab18b86d206d4e79084be2d16bca154ba46c27` 对齐：
`dinov2-with-registers-large`、CLS＋4个register（5×1024）、clean `psi0_256x320_v1`、
bf16计算及float32输出，运行时为Torch `2.7.0+cu126`／Transformers `4.57.1`。
实际DINO权重文件SHA为 `edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b`，
在线派生运行身份SHA为 `38ec7e9a6bff2cc86bf363a1d87c4ada1160e72b2e27de6e09d468fa412dc3b9`。
旧接受参考的Torch `2.6.0+cu118`／Transformers `4.49.0` 与本次实际训练运行时的差异已显式记录；
编码器文件字节及其余协议字段相同，不宣称不同运行时或设备的数值逐位相等。

### 官方固定case、成功语义与诊断边界

`E/official_eval/METADATA_PROTOCOL_AUDIT_001.json` 已完成元数据与实际任务调用路径核对，
SHA为 `4507056eb5242bb9d44a4e1fcd06d9a15a61e7c6145728512e0c5698a7028833`。
官方case归档来自 `USC-PSI-Lab/psi-data` 的固定revision
`e78fb93cc28912a3031a10b8656d32d7f0a2b867`，归档SHA为
`8a4b0ff7d068967dac8baec2c4a4deca4f644215a1ee15e921739f3dfe90e7e0`。
本轮仅选择level0的case 0–9，每个case在独立进程中恢复官方原始 `state_dict`，
reset使用 `dr_level=None`，不重新按level0抽样，也不注入示范行或PICO冷启动profile。

实际任务为 `simple/G1WholebodyLocomotionPickBetweenTablesTeleop-v0`。
当前官方任务构造器与CLI的有效 `success_criteria=0.9`；归档metadata中的 `0.4` 原样保留。
reset恢复DR状态不会将metadata中的0.4赋给实际任务阈值，运行结果明确记录实际0.9。
成功按上游原生 `task.check_success`／环境 `terminated` 判定，并在原生成功时结束；
不增加Handover式继续运行或额外几何成功门，也不将0.9误写成0.9秒。
时限为 **1200步／50Hz＝24秒，包含warmup**，本次1秒warmup之后约有23秒策略执行时间；
不能按24秒策略执行后再追加warmup。

MuJoCo负责实际物理，Isaac RTX只同步呈现同一物理状态的观测，等待HTTP策略响应期间仿真时间暂停。
本轮双来源SONIC控制器适配明确为诊断模式，
`formal_evaluation_eligible=false`，尚未认证为原控制器的正式benchmark结果。
所有case仍使用HSSD `scene6=102344022`，与Mobile100训练为同一场景，不能称为新场景泛化。
尤其case 0的机器人、目标盒与篮子初始位置／朝向与固定human TRAIN0重合，
桌面与场景几何也相同；材质和干扰物不同。这一外观变化不构成未见布局或未见场景验证。

### 十例最终结果与技术验收

`official_level0_001` 在Isaac环境启动时发生EOF，尚未执行策略；
`official_level0_002` 在TIME_WAIT相关端口预检时被拦截，没有launch。
两次技术尝试的原始目录和日志完整保留，不计为任务失败，也不替代有效case。
有效campaign的十例均达到时限 `env_truncated`：每例1,200个50Hz tick／24秒，
其中50帧warmup、1,150帧策略执行，首次策略命令在约1.005秒。
每例 `/act` 实际返回23次200及50次warmup阶段的425，`/observe` 为1,200次；
425记录中的RNG前后未变，未将等待历史的响应当作策略动作或技术错误。
最终验收核对了客户端／服务端请求顺序、因果观测时间、有限状态、等待HTTP时仿真时间不推进、
原生terminated与结果一致性，以及每例实际模型和DINO身份。
十段视频均由ffprobe实读确认640×360、50fps、1,201帧，ffmpeg均正常结束。

下面结合原始测量字段与独立逐例视频复核，按case编号记录已观测阶段；
十例的 `native_success` 均为false。
高度、接触和距离只作行为描述，不增加成功门，也不把单次接触当作稳定抓持：

- **case 0：** 抓起盒子并搬运到篮子处，目标最大抬高约27.8厘米、XY位移约2.215米，
  首次篮接触约21.485秒；直到约23.965秒才出现最后3帧无手接触，最大返回reward为0.1，
  未达到原生0.9终止。末端目标与篮中心XY距离约5.0厘米不能替代原生判定。
- **case 1：** 视频确认持续抓持，但未送到第二桌；目标最大抬高约8.5厘米、XY位移约49.8厘米，
  未记录篮接触，末端目标距篮约2.007米，未完成原生成功。
- **case 2：** 首次策略前目标已移动约14.1厘米、倾斜约75.4°，之后未记录手或篮接触，
  未完成原生成功。
- **case 3：** 首次策略前目标已移动约13.3厘米、倾斜约81.0°，之后未记录手或篮接触，
  未完成原生成功。
- **case 4：** warmup阶段约0.065秒已有右手接触记录，首次策略前目标已移动约14.6厘米、
  倾斜约91.0°；视频显示后续空手走向第二桌，未记录篮接触，未完成原生成功。
- **case 5：** 视频确认持续抓持，但未送到第二桌；目标最大抬高约11.1厘米、XY位移约67.2厘米，
  未记录篮接触，末端目标距篮约1.490米，未完成原生成功。
- **case 6：** 抓取尝试将目标碰倒；约12.905秒首次记录右手接触，
  目标高度相对首帧最大增加不足1毫米、XY位移约14.8厘米，未记录篮接触，未完成原生成功。
- **case 7：** 抓取未建立；约3.665秒首次记录右手接触，目标未抬高，首末XY位移不足0.1毫米，
  未记录篮接触，未完成原生成功。
- **case 8：** 抓取未建立；未记录手或篮接触，目标未抬高，首末XY位移约0.4毫米，未完成原生成功。
- **case 9：** 首次策略前目标已平移约9.44厘米、倾斜约95.87°并靠在toy_gun旁；
  只有warmup约0.305秒的一帧右手接触，学习策略阶段未抓回、未提起搬运，未完成原生成功。

case 2／3／4／9共四例目标在首次策略命令前已翻倒或移动，不能将其初始变化全部归因于模型输出。
这些case仍按同一原生规则计为false，未因初始目标变化过滤、重置或追加排除门。
这与十例均无机器人跌倒是不同观测，不能把目标翻倒写成机器人跌倒。

最终收据为 `E/FINAL_EVALUATION_ACCEPTANCE_001.json`，`status=complete`，
完成时间为UTC `2026-10-28T06:45:43.236422+00:00`（北京时间14:45:43），
SHA为 `4fda8a18fc6a1361de712627803bb360bc2be8d3265291cf37290534189b3674`。
该收据绑定10份 `CASE_SUMMARY.json`、命令、DINO运行回执、服务启动和请求日志、
原始 `result.json`、`trace.jsonl`、HTTP账本及视频的SHA／大小。
每例完整原件保留在 `E/official_level0_003/case_00` 至 `case_09`，
视频位于各自 `simulation/G1WholebodyLocomotionPickBetweenTablesTeleop-v0/episode-<编号>/rollout.mp4`。

最终结论是本轮10例技术有效的闭环诊断中原生成功0/10、机器人跌倒0/10；
48项CPU检查与真实GPU运行均已完成，未重复已经验收的数据数值审计。
这份结果仍受上述控制器适配与同场景／重合初态边界约束，不宣称正式原控制器benchmark成绩、
新场景泛化或Mobile验证集loss；也不改变既有训练验收和数据准入记录。

## 固定100条的95训练／5验证与20k过拟合诊断

用户随后授权把同一冻结100条Mobile按完整episode改作本次实验的95 TRAIN／5 EVAL，训练20,000个优化步。
原归档的`train_only`元数据、READY、原始数据、clean DINO、继承统计及旧100条训练全部保留；
这是新run的显式角色覆盖，不重写旧准入合同、不追加后来采集的样本。
旧Mobile100 task-step10000已见过全部100条，不能用它初始化新的5条验证；
本实验仅从原始joint38000模型权重开始，task step0与optimizer／scheduler／RNG均fresh。

`mobile_pp_holdout_data.py`按`sha256(UTF8('42:'+UUID))`升序冻结顺序，前5条EVAL、次5条作为TRAIN内的clean probe；
所有划分按完整来源UUID／lineage与原始窗口核对无交叉，不参考模型成绩选择episode。
EVAL原索引为48、56、60、64、80，共2,859帧／2,564完整窗口；
TRAIN为其余95条，共58,947帧／53,342窗口。
TRAIN probe原索引31、66、67、74、98，共2,435窗口，属于TRAIN，不能当成独立验证。
冻结split SHA为`0d87f6ae5634d07e312cfef51fa9e23e37ba70500b9ad69c4a577011457f8ac9`；
本地实验记录根目录为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp95_eval5_20k_20261028`。

新入口`a100_mobile_pp95_eval5.py`与旧100条入口隔离，仅复用原实配的只读纯helper／固定生产源码。
保持8卡×每卡8／全局64、bf16、fused AdamW、原LR组、warmup1000、statekeep与VLM颜色增强／Motion clean DINO策略；
把最大优化步改为20,000，余弦scheduler随新总预算计算。
从4,000到20,000每500步保存模型，共33份；仅此新run内旧optimizer／scheduler／RNG可裁到最新两份，末次应为19,500与20,000。
prepare/check必须隐藏GPU；训练必须单独普通SSH、全CPU0–127／nice0，不能从CPU127／nice19检查器继承训练亲和性。
复用已经通过的数据数值验收，只检验当前文件身份、小metadata SHA、新划分及代表性生产读取；不声称重新完整扫描大payload。

过拟合诊断在真实fresh step0及完成1,000、2,000、…、20,000步后执行，关闭旧生产loop的偏移验证触发。
EVAL5与TRAIN probe5使用同一个`val/no_aug` clean生产reader，固定与checkpoint无关的评估随机数；
FM loss覆盖各自全部2,564／2,435窗口，按窗口数给micro、按episode给macro；
采样生成的归一化body64／hand14 L1各取每episode固定8窗、每组40窗，明确排除末两维padding。
不把40窗采样L1写成完整验证集L1，也不将不同长度episode的无权batch均值称为全窗口均值。
8rank按`rows[rank::world]`无填充分片，末尾统一SUM，保存并恢复Py／NumPy／Torch CPU及本rank GPU随机状态和所有子模块train/eval标志。
每个诊断保存独立JSON及曲线JSONL，绑定split SHA和已完成优化步。

首轮0/10不足以单独证明过拟合：case2／3／4／9在首次学习策略命令前已发生目标显著偏移，case0则到时限后段才放手。
本次应结合clean训练probe与Mobile验证曲线、早晚权重的同口径闭环表现判断；
5条规模很小且来自同场景，若据其选择checkpoint，它们是validation，不是独立test或新场景泛化证明。
训练启动／完成及实际过拟合结论须以新的运行回执为准，本节描述冻结协议，不宣称已经完成20k。

### 本次实际启动与评测入口验收

训练入口提交`3e412c68596eefb694e24bf869a9ba83039e9a11`已推送；
实际A100根为`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pico_mobile_pp95_eval5_20k_20261028/run_001`，
其`training/posttrain/mobile_pp95_eval5_joint38000_statekeep_8gpu.us.flow1000.cosine.lr1.0e-05.b64.gpus8.261028073258`
于UTC2026-10-28 07:36:26（北京时间15:36:26）启动。
配置SHA为`8464d90a6a7fa4c3b303f7132ad5b6f9738dc171f43a9fb8d6980a0f39012468`。
部署的训练wrapper／数据／指标helper封存在同父目录`launcher_001`，不随本地后续评测修改而变更。
新95／5分区实际A100 CPU生产读取通过，完整原100的READY及数值审计复用，未重复验帧或提特征。

实际fresh0诊断通过：EVAL完整2,564窗FM micro为5.494083136413101、episode macro为5.521463940963139；
clean TRAIN probe完整2,435窗FM micro为5.484604628815544、episode macro为5.481355528428457。
这些是训练前基线，不能拿它们与旧run末batch随机训练loss直接比较并判断过拟合。
实际完成2、20优化步的8卡健康门均通过，每卡actions为8×30×80、每batch同步梯度，
fused后端、全rank有限性与每个可训练tensor前4值的跨卡探针通过（每rank7,700值，不是全参数逐位一致性证明）。
20步训练batch loss为5.131587028503418；训练进程均全CPU0–127／nice0，检查器自身CPU127／nice19且GPU隐藏。
本地`STARTUP_AND_UPDATE20_ACCEPTANCE_001.json`的SHA为
`df317137308dedc25c71d6684f4f0dbb2add9b32ba18d4a4b12a177c367f0e6a`，明确`training_completed=false`。
此时没有4k后的checkpoint，20k完整训练、恢复状态终验与闭环成功率均未完成；后续状态不覆盖这份启动快照。

匹配初态对照固定TRAIN probe case0–4／heldout case5–9，来源是本节已冻结的10个episode。
`mobile_pp_initial_state.py`只读取已准入capture的NPY头及row0完整`mjSTATE_INTEGRATION`，
首行SHA绑定原始admission→capture_start冻结回执；不读后续状态、真人动作、图像或旧controller历史。
`run_closed_loop.py`使用显式`--mobile-initialization-manifest`、`--mobile-initialization-case`、
`--expected-mobile-split-sha256`三参数，expected SHA必须来自当前模型本次95／5训练／部署收据，
在创建环境前核对；新scene reset→一次恢复row0→先写回执→清空agent/server/RTC/controller→建立新历史。
仍为native `.9`、1200步／24秒含warmup、不因跌倒提前停；matched诊断与旧官方固定case入口互斥，
不将source初态对照称为原控制器正式benchmark。

本地`matched_initialization.json` SHA为`20b0127bec5779b473fee82f86c464ee6d56c022fb7c5a940c356dc86a549eb1`。
10个实际来源首行与MJB加载已通过；真实RTX零步检查001发现资源目录重定位，002发现采集canonical43与SIMPLE43观测顺序不同，
两份技术失败原件保留，不计任务失败、不放宽物理门。
Mobile专用模型比较保持所有非路径物理／几何／名称字段严格相等；仅在逐项112个mesh文件SHA与大小、引用顺序都一致后，
允许4项路径存储字段变化，并限定MuJoCo3.3.6按64B对齐推导nbuffer差：源／新npaths9,721／9,163、buffer差−512。
实际变化来自采集和评测data_root目录，非模型质量变化；共享`initial_physics_state.compare_models`不修改。
具名关节要求43个唯一同名集合、逐名字joint id／qposadr／dofadr一致及实际值对应；
返回names/value保持同序，沿用既有`state43_to45`具名转换，不改变训练canonical表示。

真实4090／SIMPLE／Isaac观测同步零步003已通过：**仅matched case0／TRAIN来源31**，
完整709维积分首态与来源逐值相等，时间均2.1049999999999773秒、advance0，fresh RGB为360×640×3；
112个mesh身份和43关节映射均已持久化核准。
回执`MATCHED_REAL_RTX_ZERO_STEP_003.json` SHA为
`9ab0f7751c65b60165b3f75f9dbf2d1604c1add2ccd049845c2b4d029da06439`。
这是一个环境的零步初始化验收，未运行策略，不代表10例完整闭环成功率。

20k终验后，从有完整FM诊断且已保存的17份4k／5k／…／20k模型中，按heldout episode-macro FM最低、同分较早步数选择best；
另外对照4k与20k，相同checkpoint仅测一次。保存的其它500步模型未经该FM评估，不声称33份全部已评分。
新`prepare_mobile_pp95_eval5_eval.py`独立核验本次FINAL／BACKUP／TRAINING合同、真实终验SHA、
33模型／21诊断／最新两份恢复状态、split和原始准备配置；不能套旧10000合同或使用已见过全部数据的旧参数。
模型备份必须原子发布收据并核源before／after／local SHA，才可消费12GB权重；
完整VLM／action严格CPU加载，Motion全部从选中的新联合权重导出，旧资源仅提供构造metadata。
入口已通过15项隔离CPU测试与独立review；真实新权重部署仍须等待训练终验与备份，不能把这些测试称为已部署。

后续CPU准备命令（`STEP`与`FINAL_SHA`须取实际终验）：

```bash
N=/home/zxc/CVPR_Evaluation_runtime/mobile_pp95_eval5_20k_20261028
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  taskset -c 24 nice -n 19 /home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  scripts/4090_simple/prepare_mobile_pp95_eval5_eval.py --repo "$PWD" \
  --source "$N/source/ckpt_$STEP" --output "$N/deployments/ckpt_$STEP" \
  --resources /home/zxc/CVPR_Evaluation_runtime/resources \
  --release /home/zxc/CVPR_Evaluation_runtime/pico_mobile_pp100_20261027_human01 \
  --step "$STEP" --expected-final-acceptance-sha256 "$FINAL_SHA"
```

### 20k实际完成与闭环等待（2026-10-29）

本次独立run于UTC2026-10-29 03:12:40（北京时间11:12:40）正常结束，
`training_state.json`为`complete`且`returncode=0`，实际完成20,000优化步。
末次八卡生产健康门通过，每卡actions为8×30×80，最后训练batch loss为0.5012839436531067；
该batch值不能作为Mobile验证loss。

实际执行已冻结的`final_cpu_acceptance_program.py`，检查进程隐藏GPU、nice19且仅自身CPU127，
未改训练或采集。33份4k..20k／每500步模型均完成真实SHA、读取前后stat身份与1944张量header检查，
文件大小均为12,066,487,572字节、SHA彼此不同。
19,500与20,000两份恢复状态分别核验optimizer全部矩的有限性、二阶矩非负性、稀疏参数step合法范围、
调度器步数／LR、8rank CPU／CUDA RNG保存结构及全部模型张量有限性；没有恢复CUDA随机数或运行模型。
远端实际`FINAL_TRAINING_ACCEPTANCE_001.json`的SHA为
`188140fd6e21df5f0c738b96a3944d90cc8e9f0ccebab9497aff45f5abc63daa`。
这份训练终验只执行一次，后续按稳定文件身份复用。

38份原字节metadata已复制到本地实验根的`source_metadata_001`。
21份诊断0／1k／…／20k的完整EVAL5／TRAIN probe5窗数、原始行SHA、每episode固定8个采样行及macro重新计算均通过新部署器核验；
不重复原始帧、DINO或已通过的数据数值审计。
`METADATA_ACCEPTANCE_001.json` SHA为
`99395a2ad7981f30436dc7a25740c62c4f93318b08b21a68ab4bda94f7b6c2cc`。
首次本地metadata验收因部署器错误地要求W&B只能有4个运行期字段而拦截；
实际固定配置还保留`project`与`resume`，原字节与失败证据均保留。
修复只允许本次final绑定的`entity/group/id/name`变化，其余W&B字段仍与固定prepared配置逐值相等；
16项CPU回归通过，包括静态字段漂移拒绝，修复后使用同一份已复制metadata通过验收。

在17份已保存且具完整FM评分的候选中，最低heldout episode-macro为4,000步的1.1517715271896758，
因此early与best合并一次，后续仅评测4,000及20,000两份不同模型。
两份模型SHA分别为`0be905e739ae6b75c8bd1427308244512f1c4a3600e09bdf0bd1ece06c8ccf42`与
`6280909689225a7b86cadc0e03c4d0a0e2f337348b4dd4eb865a8d73a71cfe09`。
4k／20k完整EVAL5 FM micro分别为1.150268201092941／1.4320999190440857，
clean TRAIN probe5分别为0.8672590875313512／0.7308843961922601；
20k EVAL episode-macro为1.4284265553208957。
训练probe继续下降而heldout FM回升是本划分的离线过拟合信号，仍需同初态闭环比较其实际影响；
5条来自相同场景且参与选模，只作为validation。

检查时官方99初始化采集器仍在运行，保持原进程和PICO接收服务；
当前尚未执行所选模型的本地model-only备份、完整CPU部署及matched闭环评测。
按既定资源约束等待采集空闲后继续，不能把训练终验通过写成整个过拟合评测完成。

## 新一轮官方99初始化真人数采（2026-10-28）

用户确认新一轮按官方不同TRAIN初始化采集。旧100条均绑定同一TRAIN0 reset SHA
`d4c43a1fe1219f31e2b0bef3f9a2356a00a48b9a694d7f0a34430beff7c71f79`；它们是独立真人轨迹，但不具备99种初始化覆盖。
已有95／5的20k实验继续独立运行，不追加新样本、不改其split／配置／统计，不能把同初始化验证称为跨初始化泛化。
实际官方源只有99条，index0..98，metadata SHA
`f71ebef96f817fb4068b0b6ec333638ab3af88116ccbf106f8195f38a3ce0bbd`。

新根为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official99_20261028/round_001`，
`plan/manifest.json` SHA为`c9ee488637fb24a36893334dca6f6bff471e76aced6213d5c35eb63d812b89d5`。
独立冻结99份源配置、从DR空间状态重建的初始actor.pose、各文件SHA及UID映射；保存源原件，不使用示范动作、未来状态或终态布局。
实际每次reset重新核当前配置字节／路径，完整DR＋layout硬匹配；预览epoch、物理模型、几何cache、seed和每条metadata同步更新。
后台按同UUID／native-v2完整技术证据及selected结果推进；失败重试同配置，主动中断只释放同配置重试，技术分类和原件保留。
writer／初始化永久技术错误不能被晚到复核自动清除。后续只能按`progress.accepted`的index→UUID映射冻结交付，未登记sealed-UUID的极端崩溃可产生额外尝试但不能误推进。

初始CPU检查99份都能重建并站立，但只查站立不足以核准起录：旧TRAIN0专用cold profile在43号预热0.330s时由右拇指碰盒，随后位移0.170094m。
这是准备兼容故障，没有当作数据质量拒收；完整201点200Hz接触／外力与原profile证据保留在`target_warmup_diagnosis_001`。
独立候选仅把原profile两肘改至1.0rad，其它27身体关节、手指、环境DR、物体、碰撞、重力、PD、控制与native .9条件保持。
仍只在首次物理步前初始化一次。新profile实际完成全部99份50tick／1秒CPU SONIC预热，
每份201点200Hz记录：机器人pelvis全子树与target接触总数0、外力全零、target仅与table接触；
最大漂移0.000509392309m、最大pelvis倾角3.308670898°。这是准备检查，不能当真人遥操作、持物移动或任务成功验收。

全99通过后才原子复制发布`resources/mobile_pp_cold_start_profile.json`，SHA
`8f50ae58876cc2bac34037f8819301754fcb1d37178d162a49c7c16c91570530`；发布收据
`resources/MOBILE_COLD_PROFILE_PUBLICATION_001.json` SHA
`32705640e72fd995a42cb2a48f7c3033861e181380c67f06639c2e093eac453b`，绑定全部case trace及结果SHA。
373项相关PICO CPU回归全部通过，无skip，包含真实Node页面执行、手柄命令门、进度／中断／技术故障恢复；
日志`collection_cpu_regression_001.log`。两名独立review检查启动、元信息、按UUID推进和主动中断边界后无剩余阻断。

启动入口`start_pico_mobile_pp_official99.sh`要求本轮已核准profile存在，临时文件写本轮`tmp`，只进行实时预览／原始采集和独立CPU复核。
初次空启动未收到追踪、未标定、未产生episode，已正常退出并保留日志；修正后`LAUNCH_002.json`于UTC10:37:33启动新会话。
`collection/live_20261028_183733`实际入口已核定：当前配置DR＋layout匹配、站立预热通过、EGL真实相机ready、启用头显H.264服务，
加载的全部源码SHA与工作树逐项匹配。`STARTUP_ACCEPTANCE_002.json` SHA
`7cfc53d831e256e2ad7ff428341fcb3a6a887a573af8de83905445a0c751e634`；检查时第1/99份、0条完成、暂停且未标定，
PICO已被USB识别且IP仍172.16.20.83，但尚无XRobo追踪包／头显视频连接，不能声称真人输入或头显显示已验收。
此轮不在数采期间启动RTX／DINO／上传；操作与恢复步骤见[新轮次操作说明](PICO_MOBILE_PP_TELEOP.md#2026-10-28-新一轮官方99份train初始化)。
