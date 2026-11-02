# PICO Mobile P&P：官方99初始化＋历史TRAIN0一条的100条交付

> 本轮数据交付已完成：固定100条、56,423帧、50,523窗通过RTX／独立审计、无损打包、clean DINO、本机及A100实际生产读取，主driver正常退出0，补充来源包已独立核准。A100最终就绪收据生成于北京时间2026-11-02 06:34:32；训练未启动。

## 固定范围与来源

用户授权将本轮已完成的99条官方初始化真人演示，加上此前TRAIN0初始化的一条成功演示，独立处理、打包并上传A100。补充样本按已向用户说明的确定规则选取：旧TRAIN0保留记录中，编号最早且已完成完整独立验收的一条。

新99条来自 `/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official99_20261028/round_002`。冻结时进度为99 accepted、127次尝试、revision272、无pending/error，覆盖官方 `episode_index=0..98`。官方源只有99份保存的初始化，没有第100份保存的seed列表。

历史补充为旧编号 **002**，UUID `136fd169-df19-494c-a820-9b92cfb645bc`，session `live_human_003_continuous_vr1`，原目录 `/home/zxc/CVPR_Evaluation_runtime/mobile_pp_teleop_20261027/collection/kept/002`。它与新99条UUID均不同；其原DR spatial与新官方index0完全相同，spatial SHA为 `c2d7614fd182f84f885ea90c2564023e4e65c77fda5519a63bf5237930e28f8e`。

因此本批是 **100条独立真人轨迹、99种官方TRAIN初始化，index0有两条轨迹**。不能写成100个官方seed或100种不同初始化。`round_001`的5次旧试采均为技术不完整记录，并非本次历史补充来源。

最终冻结由以下仓库外证据绑定，处理时不再扫描增长的集合替换UUID：

- `preparation_001/FROZEN_100.json`：SHA `79d77a9f8e259b9c51411c768228ebf276aa2962c13696576bd3fc35740bfc7c`，exact100 lineage合同，历史002在冻结清单首项，其余99条属于本轮官方99集合。
- `preparation_001/FROZEN100_INVENTORY.json`：SHA `3ca551fc24fe233145e70d959565089c23782847986c21a69c801f28f10a2e4f`，固定100个待交付候选；`replacement_allowed=false`、不追加后续录制。
- `preparation_001/HISTORICAL_SOURCE_IDENTITIES.json`：SHA `e886106b3bc5cec98b4c3b292063641d5c88a5aee7ba07d46290f82298a97faf`，保留实际录制各版本源码身份。
- 官方99初始化manifest：SHA `c9ee488637fb24a36893334dca6f6bff471e76aced6213d5c35eb63d812b89d5`；官方源metadata SHA `f71ebef96f817fb4068b0b6ec333638ab3af88116ccbf106f8195f38a3ce0bbd`。
- `preparation_001/COLLECTION_FINISHED.json`：确认采集器已在GPU处理前正常退出、无active capture或pending review；原PICO PC接收服务和其他任务保持运行。

100条的RTX与独立admission均已通过，共1000项阶段检查，实际event_count合计56,423帧（旧002为1,020，新99为55,403）。`ADMISSIONS_SELECTION.json`的SHA为 `6617a511157017d30ad7845d5325577d17d0f204c5955d69cf3350c1543642d7`；独立集合／顺序／300份core文件身份复核见记录根的 `ADMISSION_COHORT_REVIEW_001.json`，SHA `a874f0ba55c93081a3e56a3c6659cc9ffa9424ea116a53fc29acab54d165c320`。该复核只读取现成收据及小文件身份，没有重复已通过的物理／数值审计。

冻结头部的50,523窗最初为估算，字段保留 `counts_are_not_production_acceptance=true`。本轮实际 `packed/PACK_READY.json`现已确认100条、56,423帧、50,523窗，SHA为 `f12d8d00ce0ec75e110f7bb7682ff9adb3b7d2006c84758b61d903e7a6f441b6`；100条各首尾共200次实际CPU LeRobot查询通过，当前RTX像素精确匹配。打包收据仍明确DINO及联合生产读取待完成，不能将它代替最终A100就绪证明。

## 历史差异与可复用验收

全部记录沿用同一Mobile任务、continuous VR mode1、200Hz物理／50Hz控制／30Hz采集、相同SONIC encoder／decoder／planner及PD、具名手指协议和官方native `.9`判定。历史002与新index0的controller identity仅对应session兼容回执路径不同。

旧002冷姿态profile SHA为 `a865ff08984acc6f43d4a81cceb9c67588e695d9f9d69472b49e563a9034dd43`，新99为 `8f50ae58876cc2bac34037f8819301754fcb1d37178d162a49c7c16c91570530`。实际身体差异仅左右肘：旧约0.079805／0.079961rad，新均1.0rad；其余身体值和张开手指相同。两者均仅在首个物理步前初始化一次。保留各自真实metadata，不把旧记录改写为新冷姿态。

旧002已有1020帧完整RTX与10阶段独立admission，`native-v2 selected`、官方成功为true且无技术阻断。其admission SHA为 `5caf03553de78ff64d53d69b9a5eed18e0548060df71be95911fcfcf36cb3d27`，RTX completion SHA为 `53b6fb784abd31c94ef909e24513e7087b5a54e2f141b62b410b338123663566`。本轮复用既有RTX／audit证据，不重复已通过的物理／数值审计。

旧002的无损视频、Parquet、1020帧clean DINO及frames侧车已有完整来源绑定；本次只读身份复核确认其SHA、大小、mtime仍匹配。本轮采用现有流水线，在新聚合包中实际提取全部100条的clean DINO并绑定新source／cache signature；旧002复用的是原RTX和独立admission，旧缓存及旧回执原字节保留。新99条的新RGB仍需真实RTX和DINO，已有旧002缓存不能被当作新99特征。本轮整包已完成实际A100验收，详情见文末最终交付记录；旧缓存与旧回执继续保留。

来源和四份可复用payload的实际SHA见 `provenance_review_001/REPORT.json`，SHA `838a091e6fe0c5467804213d33ea9443a5afbc7a2c237db5a39919e1072f8e14`。

## 独立目录与处理协议

本批本机发布根为：

```text
/home/zxc/CVPR_Evaluation_runtime/pico_mobile_pp_20261101_official99_plus1_human100_01
```

独立A100目标为：

```text
/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_mobile_pp_20261101_official99_plus1_human100_01
```

实际处理计划为本机发布根的 `PROCESSING_PLAN.json`，固定源码提交 `3a8775cff89c7d796e17ac164bed86f7f18b03f7`。只处理并交付数据，不启动训练，不改旧Mobile固定100、95／5实验、Handover数据或其他运行任务。容量／资源检查属于当时快照，不能作为最终就绪收据。

RTX按归档的逐帧完整 `mjSTATE_INTEGRATION` 恢复渲染，不重跑物理、SONIC或planner，不使用未来GT或teacher动作。全部100条必须沿用 `mobile-native-task-success-v2`；技术错误保留原件及失败产物，诊断后恢复同一固定范围，不以替换候选凑数，也不当作任务质量拒收。

保留实际 **43维关节状态与78维动作（64 SONIC token＋14手指）**；生产reader按继承协议补到45维state／80维action，动作末两维padding监督关闭。前30行真实warmup RGB／状态仅作历史，动作标签NaN、`action_valid=false`、80维mask全false。六帧历史索引为 `[-30,-24,-18,-12,-6,0]`，只允许未来30步完整的current；不重新拟合统计或重编码真人实际执行token。

名义时间为 `row/30`，原物理采样tick为 `ceil(row*50/30)`；DINO侧车的 `source_frame_index`保留连续视频解码序号row。RGB视频需无损回读，Parquet需逐值验证。真实clean DINO沿用 `dinov2-with-registers-large`、256×320、CLS＋4register、每帧5×1024、float32存储及已绑定bf16推理／clean预处理身份。

A100上传与最终CPU检查隐藏GPU、nice19，仅限制检查器自身允许CPU，不改变训练／采集进程亲和性或优先级。上传完成后仍需独立路径重绑和实际生产读取；字节上传通过不能冒称训练数据就绪。

## 处理中实际修订的缓存存储位置

本轮启动收据中的`all_caches_under_home=true`过于宽泛：虽然`XDG_CACHE_HOME`位于`/home`，
其`ov/texturecache`历史软链接实际指向根分区的`/var/tmp/cvpr-evaluation-zxc-cache/texturecache-20261021`。
本轮RTX生成的`.ovtex`材质缓存继续占用系统盘。原启动收据保留，另以`preparation_001/TEXTURE_CACHE_RELOCATION_001.json`追加更正；
没有据此删除旧数据、系统日志、原缓存或停止其他任务。

北京时间2026-11-01 20:45:54，在实际CPU审计间隙完成原子软链接切换。切换前复制并逐项核对
**6,249个文件、10,208,043,484字节**的SHA，源和目标清单在核对前后稳定。
新物理目录为`/home/zxc/CVPR_Evaluation_runtime/cache/omni_texturecache_20261101_mobile_official100_001`，
原`/var/tmp`缓存完整保留，可按收据原子恢复链接；主数据流程从未暂停或重启。

独立实际确认收据`TEXTURE_CACHE_NEW_RENDERER_CONFIRMATION_001.json`的SHA为
`1ac034613d99017a9f3c46a5a87e7cae056030a70e8dae70e58fddac5d2641f0`。
新目录实测出现20个切换后新增文件，旧目录的集合、逐项大小和mtime保持不变。
迁移前后实际Kit启动日志的25项参数完全相同，base参数仅episode路径不同，data root相同。
未捕捉到短生命周期缓存FD／maps，不能声称对此留有直接句柄证明；本次也没有新增同帧像素等价或GPU短测。
调整仅涉及派生材质缓存存储位置，未改变模型、资产、场景、seed或渲染参数；缓存命中／编译可能改变启动耗时。
小型OptiX缓存仍在根分区，不能把修订后状态写成“所有缓存都在数据盘”。

## 本机已通过的处理结果

本轮 `packed/PACK_READY.json`确认100条、56,423帧、50,523窗，2,660个包内payload文件；逐条Parquet原值回读及无损视频像素回读通过。原43／78维值按继承协议供45／80维reader读取，warmup与padding mask保留，`tokens_reencoded=false`、`physics_steps_performed=0`、`normalization_refitted=false`。

`dino/manifest.json`已为`complete`，SHA `fac4cecf2d18529796ba4da5aa3f77a711b08a2d1124e7d9ee88a9ec47c9fc14`，本机signature为 `cf37b8f9ed4a3b6bebb9bb11e6c3410fb79fbbacbbd079cbc39dda575d753ba4`。全部100条实际新提取，特征为`[N,5,1024]`、float32存储且全值有限。encoder参数描述的`inference_dtype=torch.float32`与实际计算优化字段分开保留；`optimization.compute_precision`及`inference_precision`为bf16，不将参数dtype误称为计算精度。权重SHA为 `38ec7e9a6bff2cc86bf363a1d87c4ada1160e72b2e27de6e09d468fa412dc3b9`，`psi0_256x320_v1` profile SHA为 `c0e155d336f3db4cc947944ff0d9b7afa3db283a088da0227579952fa24b45af`。

本机 `PICO_DATASET_READY.json`的SHA为 `fcf308b2820da1853e4492be3550f3bb7b6150507d2b95a7c71bf2aa017c491c`，实际`status=passed`、`production_ready_for_bound_pack_and_cache=true`，100条cache读取和每条首／中／尾共300次实际Dataset／collator查询通过，真实窗口数50,523。动作与mask形状`[30,80]`、状态`[1,45]`、历史特征`[6,5,1024]`；不是把50,523个窗口全部物化查询。继承统计SHA为 `931f2b536b2fd47e8d0964ad966202ccbda852d05ea88074be321987200a6ac4`，未重拟合；检查`GPU_used=false`、`CUDA_initialized=false`、`policy_constructed=false`、`training_started=false`。

本机收据按其实际阶段保留`remote_production_ready=false`及`DINO_relocation_pending=true`，不能替代远端最终就绪。上传清单 `A100_TRANSFER_MANIFEST.json`的SHA为 `f31d9a1a37fc4f61844742f990864f3b6bb3503294ab5a0883c9fa7c7eabac68`，实际3,064个文件、36,373,589,231字节；这是初次传输payload统计，不含后续工具归档、路径重绑备份和补充来源包。

上传中的被动连接观察见 `preparation_001/UPLOAD_NETWORK_OBSERVATION_001.json`：同一SSH进程143.215秒写出88,209,961字节，约0.616MB/s；当时TCP delivery约4.47Mb/s。Wi-Fi信号为−45dBm、节能关闭，不能仅据这些快照断言是Wi-Fi或服务端故障。按该段速率估算全清单约16.4小时，仅为当时估计，不是完成收据或时限承诺；该次观察时主driver和上传进程继续运行；此后主driver实际退出0，最终完成事件见下节，未为提速重启或修改主上传。

## 实际A100交付完成

最终数据汇总见`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_release_20261101/FINAL_DELIVERY_SUMMARY_001.json`。本批实际完成100条、56,423帧、50,523窗；
100条真人轨迹仍只覆盖99种官方TRAIN初始化，不改写旧002与新99的冷姿态差异。

- 主driver的`DRIVER_EXIT_001.json`实际`exit_code=0`，SHA `f927fb13cd92877226f529d4e20fbbec26360acb3bdd3f434217e2f513c45043`；
  `PROCESSING_STATUS.json`为`A100_ready_for_training_not_started`，SHA `d93a219a4d66ffcd11b01695c0810815b42aaba98b5834fe81a1e0683ff8212c`。
- 初次传输实际3,064文件、36,373,589,231字节全部通过SHA／size／纳秒mtime核对；
  `A100_BYTE_VERIFICATION.json` SHA `6c06e56c4ef7e9f52fac6536dfe260afdae4b573d83258e74b86dad7d572bb37`。该字节阶段原有pending字段保留历史语义。
- 最终`A100_PICO_DATASET_READY.json` SHA `672079bec26432011049f6a08939e17721814954dc3f99d63a51f46d88769426`，实际`status=passed`、
  `remote_production_ready=true`、`DINO_relocation_pending=false`，绑定3,165个当前payload和100份cache。
  A100实际Dataset／collator查询300次，覆盖每条首／中／尾；并非逐一物化全部50,523窗。
- 远端重绑后的DINO manifest SHA `f891fe0860814f89a436f81959c1a86a3acc9510a184e27e4baec1b3e0eefd1d`；路径重绑收据SHA
  `66d4d1b3fcd5ee3ea812ccdee0c061be609284f24ade6127cf1b1d7341cb688c`。原cache备份保留，训练数据与特征数组字节不变，
  `PACK_READY.json`仍为 `f12d8d00ce0ec75e110f7bb7682ff9adb3b7d2006c84758b61d903e7a6f441b6`；本机READY的历史远端pending标志未重写。
- 独立补充来源包已核对254个archive members／253份原文件；`SUPPLEMENTAL_VERIFICATION.json` SHA
  `8dc6f31b01afd1fcd718669e1d319e4d993a6dfcd73e672bc17de85801fc5fc5`。该补充回执不替代Dataset READY，也没有重复生产数值验收。

这些完成事实来自本轮实际收据及其当前身份绑定；不沿用旧批次结果，不重复RTX／DINO／物理／生产数值验收。
远端CPU验收隐藏GPU、nice19并仅限制自身CPU；本轮没有启动训练，也不报告策略成功率。

## 文档收尾的技术拦截

北京时间2026-11-02 06:38，有限收尾脚本在文档validate阶段发现仓库现有可执行Git LFS hook，按预设保护退出。此时主driver及独立补充来源包已通过，工作树尚未被该脚本修改；这不是数据质量拒收或A100验收失败。

`DOCS_FINALIZE_FAILURE_001.json`、`COMPLETION_SEQUENCE_EXIT_001.json`和原日志继续保留，不覆盖为成功。随后复核本轮真实完成收据并直接更新文档，正常执行已查看的Git LFS hooks提交；没有禁用hook、重启上传、重跑已通过的物理／数值验收或启动训练。

本次授权仅处理与交付数据，不启动训练，不报告训练loss或策略闭环成功率。
