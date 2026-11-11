# BendPick 第5000步 RTC 测评

本入口测评新采集100条 BendPick 训练的 `ckpt_5000`。它属于已完成10000步的独立8卡训练；不使用Handover或Mobile权重、不复制优化器、不停止服务器后续任务。

准备入口为 `prepare_bend_pick_eval.py`，测评入口为 `evaluate_bend_pick30.py`。复用Mobile P&P已运行的完整模型严格加载、当前5000步Motion导出、在线DINO、HTTP和RTC审计；Bend具有独立任务、备份及部署合同，不使用Mobile“用户已停止训练”证明。

## 固定配置

- 使用原始联合模型的第5000次任务优化器更新，部署和source在同一磁盘分区用硬链接保留一份12.07GB模型。Motion参数从这份5000步joint逐张量严格导出；原joint20k目录仅提供架构、归一化及构造元数据。
- 训练配置仍为 `rtc=false`。服务端显式启用 `--rtc-mode test_time`，预测30帧、执行15帧、10步采样、delay0、min horizon15、exponential mask、guidance alpha0.9。50Hz下每25个控制tick重规划；HTTP等待期间暂停仿真。
- 保留训练 `motion_image_policy=vlm_color_clean_motion`，配置与DINO合同一致；当前图像无在线训练增强，Motion使用实际因果6×5Hz历史。没有未来历史、FSQ重投影或teacher动作注入。
- 使用官方BendMP三个级别各10个固定场景。每例从新服务、新仿真开始，环境seed0；本次策略seed沿Mobile参考为 `20261105+10*level+case`。
- 每例原生总预算400个50Hz控制tick，即8秒，含50tick预热。原生奖励为目标相对首次奖励基线抬高量除以0.1米；成功阈值0.9，即约9厘米。保留原生终止和时限，不追加接触、保持、落地或末态几何门。
- 保存每例视频、HTTP双方账本、控制轨迹、RTC实际调用、因果历史和原生成功标记。技术错误单独计数，不当作任务失败；没有自动重试。

## 官方初始化与适配边界

官方来源为 `USC-PSI-Lab/psi-data` 的固定revision `e78fb93cc28912a3031a10b8656d32d7f0a2b867`，文件 `simple-eval/G1WholebodyBendPickMP-v0.zip`，SHA256为 `6bcdb9ddfa1d3f851144f585bda19507ad9efb53e71514dad98836fcefadc5e2`。只取所需metadata和既有资产闭包，不使用teacher视频、parquet状态或动作。

源任务 `simple/G1WholebodyBendPickMP-v0` 使用AMO/500Hz机器人。本次控制沿用训练时的私有 `cvpr/G1SonicBendPickPico-v0`、SONIC/200Hz、一次 `+0.405m` 世界坐标变换及pelvis补偿。帧钩子在首次reset前安装；每例 `dr_level=None` 恢复固定场景，避免重新随机材质和布局。因此结果属于官方固定场景的SONIC适配诊断，不是原生AMO benchmark。

原官方eval包的配置仍写 `split=train`、内部dr_level0，这些原始字节保持不变，级别取官方目录。不能据此声称独立heldout或新场景泛化。20份配置有不对应任何真实actor的多余spatial记录：只在派生reset中移除这些unused UID，记录原配置、派生配置SHA及精确删除条目，可逆恢复原件；所有真实物体、机器人、材质和奖励保持。训练adapter本身的严格验证不放宽。

原件还含当前SIM310相机配置类不接受的 `intrinsics`、`depth_intrinsics`、`depth_to_color_extrinsics` 三项空字段。派生reset仅在值为 `None` 时去除这三个明确字段；非空标定拒绝，未知字段不静默删除。逐相机删除路径、原值及原件/派生SHA写入同一可逆收据，不改变相机几何。全部60个官方相机已通过实际配置类构造检查，真实渲染仍由场景测评验证。

## 准备与启动

本次实际运行根为 `/var/tmp/cvpr-bend-pick-ckpt5000-20261111/`，使用本机系统盘空间。位置引用保存在 `CVPR_Evaluation_runtime/bendpick_ckpt5000_testtimertc_20261111/LOCATION.json`；服务器训练目录及原数据不变。

`source/backup_receipt.json` 需要固定Bend run、24份metadata、已完成训练/13节点保存证明、源文件stat前后相同，以及远端实际传输流SHA和本地落盘重读SHA一致。备份不使用伪造的stopped标记。

```bash
CUDA_VISIBLE_DEVICES='' python -B scripts/4090_simple/prepare_bend_pick_eval.py \
  --source /path/to/source --resources /path/to/resources \
  --release /path/to/original/local/bend_pick \
  --output /path/to/new/deployment --repo /path/to/this/repo \
  --step 5000 --expected-model-sha256 <实际备份SHA256>

python -B scripts/4090_simple/evaluate_bend_pick30.py \
  --deployment /path/to/deployment --data-root /path/to/official_eval/data_root \
  --training-manifest /path/to/source/release_metadata/dino/manifest.json \
  --output /path/to/new/official30_rtc_001 --runtime-root /path/to/CVPR_Evaluation_runtime \
  --expected-model-sha256 <实际备份SHA256> --checkpoint-step 5000 \
  --sampling-seed 20261105 --rtc-mode test_time --execute-steps 15
```

准备进程使用MODEL311、隐藏GPU、单核低优先级。正式campaign使用SIM310及完整CPU affinity，服务端用MODEL311；确认本机无其他GPU计算任务后运行，仅清理本次创建且身份匹配的进程组。首次真实场景仍必须检查画面、低层控制、HTTP、RTC及视频；CPU测试和静态资产闭包不等于物理或渲染验收。

运行记录为 `FETCH_STATE.json`、`EVALUATION_STAGE.json`、`CPU_PREPARATION_EXIT.json` 和 `OFFICIAL30_RTC_EXIT.json`。完整30例结果为 `official30_rtc_001/FINAL_EVALUATION_ACCEPTANCE.json`，每例结果在对应 `level*/case_*/CASE_SUMMARY.json`；失败保留 `TECHNICAL_FAILURE.json`，不覆盖旧尝试。

本次001尝试在首例相机初始化时因上述空字段报错，尚无物理步或模型动作；已精确停止其所有自有进程，保留记录，不能记作抓取失败。修复后使用新的 `official30_rtc_002/`，复用已严格验收的同一5000步部署，30个场景、种子、RTC及原生成功条件不变。当前运行位置与状态读取 `ACTIVE_EVALUATION.json`，启动/退出分别读取 `OFFICIAL30_RTC_002_LAUNCH.json`、`RETRY002_EXIT.json`；不能把001的中止状态当作002结果。
