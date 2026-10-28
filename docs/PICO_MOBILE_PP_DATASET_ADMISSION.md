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
