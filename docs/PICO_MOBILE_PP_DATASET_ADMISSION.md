# PICO Mobile P&P：固定 100 条核定与 A100 交付

2026-10-27，用户在采够后要求处理当前 **100 条 Mobile P&P 真人保留记录并上传 A100**。
本批从当时 121 条已编号记录中冻结 100 个 `selected` UUID；不追加后续录制，不用新记录替换失败候选。
采集器已按这次要求正常退出，`COLLECTION_FINISHED.json` 确认 collector、planner 和 preview 均已退出、无 active capture。
此后才开始 RTX 阶段；SDK 接收服务及其他任务没有因此停止或重启。

**当前是准备与首条实测通过记录，尚未宣称全 100 条、clean DINO 或 A100 生产读取完成。**
首个实际样本 002 已完成 1,020 帧原积分状态 RTX 和 10 阶段独立审计；不能用这一结果代替全量收据。
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
后续处理计划另绑定 `FROZEN100_INVENTORY.json` 的实际 SHA、代码提交、资源及独立输出目录。
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
不停止、重启或修改其他进程，也不修改或启动 A100 训练。

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

## 本轮已完成的验证与剩余范围

准备阶段 **114 项 CPU 回归通过**：Mobile helper／独立审计 54 项、渲染派发 5 项、
固定清单流程 22 项、转换／就绪／上传 33 项。覆盖跨任务合同拒绝、缓存清单禁止替换 UUID、
技术失败不冒充 native 失败、真实无损视频回读、DINO 路径重绑、远端字节脚本本地执行及子进程 CPU 约束。
实际样本 **002 的 1,020 帧 RTX 与 10 阶段独立审计通过**；全 100 条处理继续按各阶段真实收据核定。

固定 100 条原始 capture 合计 **61,806 个 30 Hz 帧**。按每条去除 30 行 history-only 前缀及末 29 个
未来不完整 current，推导为 `61,806 - 59×100 = 55,906` 个训练窗口；
**55,906 此时只是协议推导，未当作全量生产 reader 已通过的实测值。**
全 100 条独立审计、无损打包、clean DINO、本机生产读取、远端全部字节与 A100 生产读取，
必须分别取得实际结果后更新本节。所有运行数据、日志、资源与权重均保存在仓库外。
