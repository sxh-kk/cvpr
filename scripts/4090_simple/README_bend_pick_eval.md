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


当前eval源还带有SIM310配置构造器不接受的三项DR配置。reset派生v3只接受并记录以下精确类型和值：

- `spatial.placement_attempts` 为整数100：当前SpatialDR本来硬编码100次放置尝试，保存的实际位姿直接恢复。
- `scene.randomize_scene_pose` 为布尔true：保存场景的center_offset、orientation和桌体恢复到inner state后直接返回，未运行姿态采样。
- `material.table_material` 为None：没有配置材质覆盖；保存的桌体、机器人及物体材质状态完整恢复。

非精确类型或值一律拒绝，未知参数不静默丢弃。逐路径原值、兼容理由及原件/派生SHA记录在收据中；
相机字段、DR字段及unused spatial均能逆向补回，逐字段恢复原配置完全相等。
这只证明完整保存状态且 `dr_level=None` 的reset兼容，不声称新的随机采样分布等价。
CPU实测已完成60个相机和240个DR配置构造，以及全部30个真实私有任务的完整构造、
`task.reset(seed=0, options={'state_dict': adapted, 'dr_level': None})` 和实际环境binding匹配；
共享scene manager未改、原件未改、CUDA未初始化。此检查不创建仿真env、不进行物理步或RTX渲染。

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


002尝试随后在私有任务构造时遇到 `SpatialDRCfg` 不支持 `placement_attempts`，同样尚无任何物理步或RGB。
其自有进程已核实全部退出，原记录保留，不计任务失败。上述v3全面审计修复后，下一次使用新的
`official30_rtc_003/`，继续复用相同已验收5000步部署。003现已完成30个技术有效场景，成功28/30；原结果完整保留。
当前实际位置/状态仍以 `ACTIVE_EVALUATION.json` 为准，003启动和退出分别核
`OFFICIAL30_RTC_003_LAUNCH.json`、`RETRY003_EXIT.json`（尚不存在时表示尚未发布该阶段凭据）；
不得读取001/002的失败凭据充当003结果，也不得覆盖旧尝试。

## 初始化桌沿支撑修复：stable_support_v1

003的两例失败L2/03、L2/08在预热期间发生目标滑落。逐5ms真实MuJoCo接触复现发现：两例初态没有机器人与目标接触，真实碰撞网格编译得到的目标质心投影先超出桌左沿约4.42mm、2.97mm；盒子自行外翻后，才在0.350s、0.420s首次碰到机器人髋部。纯PD或无控制也会滑落，不能通过更换预热手臂控制修复这个支撑问题。

新增显式可选 `--initialization-profile stable_support_v1`，默认仍为 `official_original`。新规则对全部30例统一计算，不读取旧成功标签：从同一生产对象构造器和真实凸碰撞网格编译目标局部COM，将其投影到水平桌面矩形内缩5mm的区域，取最短XY平移。只同步修改派生配置中目标layout和DR的XY；Z、四元数、机器人、其他物体、材质及原官方文件不变。每例保存原件/派生SHA、精确改动字段、移动量、桌面边界、真实COM及网格身份；内部28例严格不动，另外两例分别向内平移9.419mm、7.974mm。

几何清单需先在CPU上通过 `bend_support_initialization.py --prepare` 生成；加载时重新编译同一小型对象并核来源、MuJoCo版本和网格字节，不信任手填COM。worker在真实完整模型首次物理步前，再只读核对目标COM/质量/姿态和桌面几何，校核原坐标到SONIC的+.405m变换。此检查不向运行中的qpos写入修正。

新profile还在reset的1个物理步及50个真实预热控制tick（总201个5ms子步）被动记录目标运动与机器人接触。FK在私有MjData中计算，不改变真实动力学或增加步数。初始化有效性要求目标最大漂移不超过2mm、转角不超过2度、没有机器人接触目标且状态有限；这是单独的初始化技术检查，不替代原生抓取成功条件。策略阶段没有避障shield、物体固定或控制改写。

真实CPU预热测试覆盖全部30例，最大目标漂移不足1mm、没有机器人接触目标；两修复例及成功对照的生产监视器有/无对照，在全部201子步中15个真实状态数组和时间逐元素一致。后续真实RTX/策略闭环仍需单独验收，不把这项预热检查算作抓取成功。

```bash
CUDA_VISIBLE_DEVICES='' python -B scripts/4090_simple/bend_support_initialization.py \
  --prepare --data-root /path/to/official_eval/data_root \
  --output /path/to/new/support_geometry.json

python -B scripts/4090_simple/evaluate_bend_pick30.py \
  --deployment /path/to/deployment --data-root /path/to/official_eval/data_root \
  --training-manifest /path/to/source/release_metadata/dino/manifest.json \
  --runtime-root /path/to/CVPR_Evaluation_runtime --output /path/to/new/stable_support_eval \
  --expected-model-sha256 <实际备份SHA256> --checkpoint-step 5000 \
  --sampling-seed 20261105 --rtc-mode test_time --execute-steps 15 \
  --initialization-profile stable_support_v1 \
  --support-geometry-manifest /path/to/new/support_geometry.json \
  --prioritize-support-adjusted
```

`--prioritize-support-adjusted`只按统一几何规则的applied字段把受修订的场景前置，所有30例、各自seed和独立服务重置保持；不根据模型结果选择或重试。新campaign使用独立 `cvpr-bend-pick-fixed30-stable-support-v1-sonic-rtc-diagnostic-v1` 合同，结果明确称修订初始化的SONIC诊断，不能冒充官方原始30例原样成绩，也不能改写003的28/30。

本次修复的几何、物理诊断和新测评入口位于 `/var/tmp/cvpr-bendpick-init-contact-20261111/`；新输出为 `eval_stable_support_v1_001/`，当前状态由该目录外的 `STABLE_SUPPORT_ACTIVE.json` 与campaign `state.json`记录。原权重和部署继续复用 `/var/tmp/cvpr-bend-pick-ckpt5000-20261111/deployment/`。
