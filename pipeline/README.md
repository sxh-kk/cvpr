# EgoMotion 30Hz 统一数据管线

转换 EgoExo4D-HM 发布 NPZ 和 EgoSuite **原生 MCAP** 数据。独立 SMPL-H 参数归档已完成全量迁移、最新 MotionExpert 接口验收和正式发布。验收通过后已删除旧处理后 dense22 Parquet；原始输入、模型权重和任务证据保留，原拟合任务的失败报告保持原值。

## 本轮最新 MotionExpert 验收

独立任务：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-qualified-20261012T084000Z/`。最终结果以此任务的 `status.json`、`reports/full-qualified-acceptance.json` 与清理收据为准，原迁移／拟合任务及失败报告不覆盖。

**2026-10-12 10:36 UTC：关键验收完成，任务 `complete`，归档 `accepted`。** 最新 MotionExpert 默认正式读取已通过；清理后全部9,263段的自有文件、Parquet字段／行数和mask结构可访问，禁止旧来源路径。已删除审核清单内19,855个旧dense Parquet，共835,186,426,720字节（约835.19GB）；原MCAP、HM／Aria／原全视频DINO、权重、JSON和job证据保留。清理保留逐文件身份核验与持久化恢复，仅在开始／结束做整体核对，期间检查冻结证据身份。简明结论见任务 `reports/critical-acceptance-summary.json`；当前归档入口为 `/gaozt-test1/datasets/MixEgoMotion/current_dataset.json`。

已完成全量5Hz窗口索引：train 1,001,223、val 125,262、test 125,234，共 **1,251,719** 个完整窗口；9,205段的18,410个首尾样本全部通过数值、双手、来源脚接触、地面兜底与DINO历史mask检查。47段HM继续隔离，11段EgoSuite因原脚点／地面均无效而无训练窗口，原数据与mask保留。

按用户最新要求，已通过的数值、索引与完整模型前反向结果直接复用。原审计器将两个启动目录的 `stat` 误判为pipeline依赖，故原整体失败报告仍保留；新报告通过精确源码差异及独立11项guard测试核对该误判，明确 `numeric_checks_rerun=false`、`full_population_guard_v2_rerun=false`，不冒称用新guard再次执行全量样本。正式发布准入和清理后自有文件访问能力均已通过；后者只读元数据／文件头，不读取数据页、不执行FK、窗口索引或模型。正式全量train的245维motion＋18维轨迹归一化统计尚未生成，`formal_training_ready=false`，归档接口验收不代表训练已开始。

首次全量来源审计及独立15Hz reader已完成，reader覆盖全部9,263段。原审计进程在最终JSON序列化时因无有效帧诊断值 `inf` 退出1；其日志和失败执行收据保持不变。另存 `reports/parameter-integrity-execution-witness.json` 与恢复报告：用冻结源码控制流、当次完整reader、日志和任务启动前未变更的元数据身份证明来源断言全部完成。逐样本诊断数值未落盘，不能恢复；恢复报告逐段注明 `diagnostics_available=false`，没有重跑数值。恢复证据已通过独立审核。正式读取器成功加载归档后的脚本自身查询误判也单独记录，原失败报告保留，没有重跑已成功的读取器。

## 统一 SMPL-H 参数归档

发布路径为 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1/`；原迁移任务为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-parameters-v1-20261012T032915Z/`，其历史失败状态不覆盖。EgoSuite 6,614 段／26,841,893 帧与 EgoExo4D-HM 2,649 take／11,295,991 帧，共 **9,263 段／38,137,884 帧**，保存原30Hz参数。当前发布状态以归档 `.pipeline/dataset-state.json` 和前节最新MotionExpert独立任务的 `reports/full-qualified-acceptance.json` 为准。

迁移阶段历史状态：2026-10-12 04:51 UTC完成全部参数与视觉文件迁移，当时为 `converted_pending_acceptance`。合并split为train 7,409段／30,510,543帧，val 915段／3,812,804帧，test 939段／3,814,537帧。9,216段满足归档训练准入标记，47个HM take／61,743帧保留并隔离：44个标定数值表示差异、2个轨迹实际跨world graph、1个官方valid全零。段准入不代替逐帧／逐关节mask；当前正式发布和旧缓存清理已完成，见前节。

**原任务历史状态（2026-10-12 06:13 UTC）：生命周期已停止，状态为 `failed`。** 原EgoSuite全部6,614段／26,841,893帧独立数值审计与四项数值门槛均通过：身体重建mean0.688cm／P951.847cm，手腕P950.329cm，连续local>45°比例0.00457%。原数据13,231项文件身份保留检查通过，但22个MotionExpert文件相对原拟合任务的源码快照发生SHA变化，导致原整体门禁失败。新参数归档最终全量独立审计尚未执行，cleanup manifest／execution均未创建，旧dense22仍保留；当前任务不是继续运行或全量验收通过。未修改原失败报告、质量阈值或快照基线。证据见任务 `reports/original-full-fit-gate-observation.json`。

| 内容 | 存储形式与语义 |
|---|---|
| 身体 | `smplh.root_orient[3]`、`pose_body[63]`、`trans[3]`、真实 `betas[16]`；轴角为局部参数，世界位置和全局6D由同一个52关节 SMPL-H 模型 FK 导出 |
| EgoSuite 来源 | 已完成的 dense22→SMPL-H 拟合参数；shape按校准组冻结。拟合标签是模型估计，源观测 mask 与估计 mask 分开 |
| HM 来源 | 发布的 SMPL-H 参数原值，保留逐帧16维 beta；不重新拟合、不套用 EgoSuite 中心偏移 |
| 显式双手 | 左右各21×3世界点及逐点有效／插值 mask 独立保留。无效但有限的原始手坐标仍保存，不升级 mask |
| 可选手指参数 | HM `smplh.pose_hand_pca[90]` 为左右各45维 PCA系数，加原手部均值；独立 `smplh.valid.hand_pose[2]`。EgoSuite 未拟合手指 pose，显式手21仍可监督 |
| 相机／头戴 | 世界位置＋`wxyz`、有效／插值 mask；头戴原点为实际 RGB 光心，统一到 EE4D 轴约定。HM 使用 `T_world_device @ T_device_camera_RGB`，CPF头部参考另存 |
| 地面／脚 | 原地面高度、法向、有效／verified mask；保存来源脚点和 mask 供窗口兜底。缺地面时15Hz所选有效脚点平均高度兜底，fore仅用观察前缀 |
| DINO | 自有 `vision.parquet`：float32 `[5120]`＝CLS＋4 register，`dinov2-with-registers-large`；特征、PTS、mask逐值迁移，不重新提取。时间误差统一float64，保留HM源精度 |
| 时钟／关联 | 全局episode／frame／index、30Hz名义时钟、原来源时钟／左右帧／插值权重，独立 `source.video_frame_index`；HM另有实际capture纳秒时间 |
| 质量与来源 | 估计可用、源观测、源插值、模型补全、残差与confidence分别保存；confidence是数值拟合代理。隔离数据拒绝canonical训练窗口 |

每段为 `episodes/<全局八位id>/{motion.parquet,vision.parquet,receipt.json}`；索引在 `.pipeline/episodes/`，汇总与划分在 `meta/{summary.json,episodes.parquet,splits.json}`。模型与手部PCA基底在归档自己的 `weights/smplh/`。归档不依赖旧 dense22 文件、原HM资产或旧主归档才能读取。

DINO两来源共用同一权重、最短边336的bicubic resize、无crop及ImageNet归一化，但源图像几何保留差异：EgoSuite来自已去畸变的head-left，HM来自未去畸变的Aria RGB视频。`source.video_frame_index` 在EgoSuite标识原30Hz特征表的行，真实MCAP图像由最近PTS关联；在HM才直接标识发布视频的零起点帧号。图像PTS与关联误差均保留。DINO mask表示视觉可用性，不能替代身体监督或坐标注册mask；EgoSuite另排除了原bad-image flags，HM未提供同等图像质量标记。完整对照见任务 `reports/independent-cross-source-dino-contract.json`。

EgoSuite 沿用原训练划分；HM 按 participant／capture 连通分量划分，未知participant保守共同分组，整组只能进入一个split。精确全量成员在任务 `inventory/combined.json`；每段索引同时保存 `split`、`split_group_ids`，不能随机拆相邻窗口。

HM row `i` 对应视频／DINO零起点 `i` 的依据冻结于 `references/hm/`：官方合并结果viewer取 `{i+1:06d}.jpg`、帧抽取从1编号。普通take按 `T=N−1`；7个登记的历史take另需零起点、连续100帧chunk及 `T=floor(N/100)*100`。未消费的视频尾帧范围单独记录，原HM DINO全视频文件保留。该映射复现发布代码约定，不等于独立图像标注认证。每个take再核对四个exo相机的世界注册与Aria重力方向。

全量处理中出现的隔离段仍保存参数、双手、相机、DINO及原mask；`training_eligible=false` 的段不开放canonical训练窗口。当前已核查的 `fair_bike_06` 五个take使用同一world graph，源外部相机矩阵能由CSV的未归一化四元数、矩阵求逆及float32保存逐值复现；与SDK归一化表示的最大平移差约11微米，略超本任务预登记10微米门槛。该数值诊断不会在运行中放宽门槛或升级mask。另有官方 `valid` 全零的take，有限参数继续保留但监督保持无效。最终数量以全量验收报告为准；诊断证据位于任务 `reports/hm-world-quarantine-diagnosis.json`、`independent-hm-calibration-numeric-bounds.json` 与 `hm-source-invalid-take-diagnosis.json`。

```python
from egomotion_pipeline.smplh_parameters import SMPLHParameterArchive

archive = SMPLHParameterArchive('/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1')
window = archive.read_window(0, start_frame=0, num_frames=80, fps=15)
# 30Hz偶数相位0,2,4,...；选帧后FK、地面与增量计算
body = window.arrays['observation.state.body_floor_rot6d_xyz'].reshape(80, 22, 9)
left = window.arrays['observation.state.hand_left_floor'].reshape(80, 21, 3)
trajectory = window.trajectory                       # [80,18]
trajectory_mask = window.trajectory_valid
dino = window.arrays['observation.features.dino'].reshape(80, 5, 1024)
```

身体XYZ与6D是窗口中的临时张量，不重复保存dense22 Parquet。SMPL根中心满足 `root=J0(beta)+trans`；规范化时同步改变root orientation及trans，保留Z-up并将首帧相机XY与水平朝向统一。轨迹前9维为相机相对地面位姿，后9维为相邻地面参考增量；旋转取前两行6D，增量在上一帧参考系表达。读取允许原坐标检查 `canonicalize=False`，但mask继续生效。

MotionExpert使用独立SMPL-H参数reader，按30→5Hz选11帧，用真实16维beta做完整body22 FK后选择既有12稀疏关节，**保持245维模型表示**：K12各6D＋XYZ108、参考增量9、左右显式hands21各63、contacts2。beta仅用于源FK与审计，不作为模型条件或预测目标。联合重建历史6帧／预测未来5帧，图像条件只读取历史6帧；SMPL-H训练逐样本以0.5概率开放历史6帧轨迹，未来轨迹始终隐藏，验证／部署只用历史图像。该概率是起始实验值，仅改变概率无需重新处理归档或重算同表示统计；地面兜底及接触使用来源脚点。默认策略为模型定义轴，旧原生轴245统计和权重不能因同维度直接复用。正式训练仍要求归档accepted且全量新train统计完成；诊断小样本验证不替代发布验收。EgoSuite原生轴和绝对旋转认证继续为false。接口见 [MotionExpert README](../MotionExpert/README.md)，此适配不改变本pipeline参数reader的15／30Hz接口。

处理入口为 [run_smplh_parameter_job.py](scripts/run_smplh_parameter_job.py)：迁移已提交拟合与HM，补齐剩余EgoSuite；等待原拟合全量质量验收；独立验证全部参数／手／相机／地面／DINO／split；禁用旧路径进行完整population reader检查；再执行已授权的精确文件清理，并做删除后读取检查。任一阶段失败，状态与原因留在任务目录，未验收旧数据保留。

```bash
JOB=/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-parameters-v1-20261012T032915Z
cat "$JOB/status.json"
tail -f "$JOB/logs/lifecycle.log"
# 两套来源的迁移日志
tail -f "$JOB/logs/egosuite.log" "$JOB/logs/egoexo4d_hm.log"
```

## SMPL-H 拟合与旧 sidecar 验收

新增 [smplh_sequence.py](src/egomotion_pipeline/smplh_sequence.py)、[smplh_sidecar.py](src/egomotion_pipeline/smplh_sidecar.py) 和 [process_smplh_dataset.py](scripts/process_smplh_dataset.py)，在独立目录拟合完整原30Hz序列。模型使用与 HM 匹配的 `weights/smplh/male/model.npz`，真实16维 beta；身体22点的位置和全局6D均由同一次目标模型 FK 导出。这里的 SMPL-H 表示是拟合估计，并不包含从源手21点恢复的完整 SMPL-H 手指 pose。

20位 train operator 用于等人物权重的共享中心偏移先验。每个 operator 用有效肢体骨长估计一次 shape，并校准 neck／collar／head 的局部中心偏移；序列拟合时二者冻结。同人物的另一录制若骨长签名不一致，则使用带 source_key 的 recording-specific 校准，避免误复用 shape。16维 shape 的可观测信息仍来自肢体骨长，并非身体表面观测。局部偏移与 pose 也不是独立测量的解剖真值。

拟合包含稳健XYZ、源世界旋转增量、局部旋转／活动范围／twist／模型静态弯曲平面的软先验、native手腕／手掌方向。v2初始化用世界旋转增量或父关节运动传递已有姿态，几何方向作弱修正，缺观测部分在同跟踪段内用SO(3)桥接；这是离线估计，不改变源mask。时序项用局部旋转对数平方，默认250次迭代。脚接触使用有效原脚点的速度／地面代理，仅以弱项匹配观测到的水平脚点运动，不将低于速度阈值的脚点强制设为静止。姿态先验是显式启发式，没有加载 VPoser 或其他学习姿态先验。长序列按连续跟踪段分块重叠，局部旋转在 SO(3) 上融合后重新 FK；长缺失和 tracking reset 不跨段约束。脚关节点不强制压到地面。

拟合阶段原 v5 保持不变；派生目录按 `(episode_index, frame_index)` 关联来源。motion-only sidecar 不重复保存 RGB 或 5120维 DINO。源双手21点、相机／地面和源 mask 在 `source.*` 中逐值保留，原视觉特征按同一归档帧读取。

| 派生字段 | 每帧形状 | 语义 |
|---|---:|---|
| `fitted.body_{world,floor}_rot6d_xyz` | 198 | 目标 SMPL-H body22，前两行6D＋XYZ |
| `fitted.local_axis_angle` / `local_rotation` | 66 / 198 | SMPL-H局部pose参数／同一pose的局部3×3旋转矩阵 |
| `fitted.root_world` / `smpl_trans` | 各3 | root世界中心／SMPL transl，满足 `trans=root−J0(beta)` |
| `fitted.betas` | 16 | 同校准组冻结的真实模型shape系数 |
| `fitted.reconstructed_source_landmarks_world` | 66 | FK中心加校准偏移得到的源观测重建，区别于目标中心 |
| `estimated.valid.body_{world,floor}_{position,rotation}` | 各22 | 数值模型估计可用；不代表原始观测或已认证监督 |
| `source.observed.body_{position,rotation}` | 各22 | 未改变的源观测mask |
| `estimated.inferred.body_position` | 22 | 可用模型估计中缺少源XYZ观测的点 |
| `quality.{position,rotation}_confidence` | 各22 | 残差代理与保守补全权重；不是准确率概率 |

旋转目标轴由 SMPL-H 定义，但 `native_axis_certified=false`、`absolute_rotation_accuracy_established=false`。源旋转增量和XYZ只能约束部分 twist，缺观测段可能产生不稳定模型旋转。必须检查时序和质量，不能通过更改认证标记接入训练。

当前任务：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/`。20位train共享校准人物与60位验证人物不相交；验证完整154,473帧已通过独立审计及预登记门槛。2026-10-11 14:07 UTC已启动64进程全量6,614段／26,841,893帧候选处理，先校准1,829个operator，再拟合完整30Hz序列。候选目录为 `/gaozt-test1/datasets/MixEgoMotion/derived/smplh-body22-sequence-v2/egosuite/`；`status.json` 与 `logs/full.log` 保存当前阶段及进度，`logs/full-lifecycle.log` 记录阶段切换；这是后台运行任务，尚未完成全量或替换主归档。

v2扩大验收body观测重建mean0.714cm／P951.875cm，native手腕与FK身体腕P950.372cm；连续local>45°边236／3,395,480（0.00695%）。仍有87个>90°边，最大163.51°，旋转继续标记模型估计；不能把 pooled门槛通过写成全部帧精确。详见 [expanded-acceptance.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/expanded-acceptance.json)、[sidecar-audit.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/sidecar-audit.json)。

第一轮v1完整60段的独立数值审计通过，但身体平均残差1.192cm、局部旋转>45°边比例0.642%，未通过1cm／0.1%的预登记门槛，未启动全量。历史结果保留于 [v1 expanded-acceptance.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v1-20261011T123135Z/reports/expanded-acceptance.json)。v2修订依据train受控实验，沿用相同门槛；单片段改善不代替60段验收。

```bash
PY=/tmp/egomotion-pipeline-venv/bin/python
JOB=/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z
$PY pipeline/scripts/process_smplh_dataset.py status --job "$JOB"
$PY pipeline/scripts/audit_smplh_sidecars.py --job "$JOB" \
  --selection "$JOB/selection.json" --sidecar-root "$JOB/expanded-sidecars" \
  --output "$JOB/reports/sidecar-audit.json"
$PY pipeline/scripts/accept_smplh_job.py --job "$JOB"
tail -f "$JOB/logs/full.log"
```

plan冻结模型、config、代码、split与selection；改动后使用新job，旧候选不混写。worker在任务前后核对冻结代码和模型，plan自身内容也校验签名。提交采用阻塞进程锁和目录原子rename，竞争／异常／进程被终止不会留下半个已提交episode；合法提交可验证后续跑。独立审计从模型NPZ重新用NumPy FK，并重新读取源数值、mask、分组及 train-only 先验。扩大门槛预先登记于 `quality-policy.json`：所有有效源XYZ重建mean≤1cm／P95≤3cm，native手腕与FK身体腕P95≤1cm，同连续段内局部旋转单步>45°的比例≤0.1%，并要求完整数据与源码保留检查通过。未通过则不启动全量；通过仅允许全量候选计算，不能直接发布为主归档或宣称运动真值认证。

当前全量入口为job内 [run_full_candidate.py](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/code/run_full_candidate.py)：先重新执行扩大accept，并核对当前job／policy／audit与精确60段；不直接依赖runner的旧布尔报告。全量完成后自动使用独立审计和 `accept_smplh_job.py --scope full` 检查全部6,614段，结果为 `reports/full-sidecar-audit.json` 与 `reports/full-acceptance.json`。未通过全量门槛时候选仍保持质量失败状态，不替换v5。

[SMPLHDerivedArchive](src/egomotion_pipeline/smplh_archive.py) 是显式选择拟合标签的验证级联合reader。15Hz仅读取30Hz偶数相位，随后构建18维相机地面轨迹；缺地面兜底只用原生脚点，fore只用观察前缀；DINO与native手点继续从原帧关联。当前会验证并读取完整序列，不承诺训练吞吐；现有 MotionExpert reader、split和统计仍对应源 v5，未在本次切换为候选标签。该研究reader不承担HM全量参数归档；HM统一流程与帧对应合同见上文。

## 旧v5输出与表示（迁移来源）

| LeRobot feature | 每帧形状 | 含义 |
|---|---:|---|
| `observation.state.body_world_rot6d_xyz` | 198 → 22×9 | v5 身体世界表示，逐关节前两行6D旋转＋XYZ；训练读取的权威身体列 |
| `observation.state.body_floor_rot6d_xyz` | 198 → 22×9 | v5 相机地面参考表示；world／地面改变后须重新计算的缓存 |
| `observation.valid.body_{world,floor}_rot6d_xyz_{rotation,position}` | 各22 | 旋转与位置分别使用逐关节mask，不能按9维整块合并有效性 |
| `observation.interpolated.body_{world,floor}_rot6d_xyz_{rotation,position}` | 各22 | 插值来源标记；有效插值仍参与监督 |
| `observation.state.hand_left_floor` / `hand_right_floor` | 各63 → 21×3 | 与身体相同相机地面参考系的显式手部XYZ |
| `observation.valid.hand_{left,right}_floor_position` / `observation.interpolated.hand_{left,right}_floor_position` | 各21 | 手部逐点有效性／插值来源；身体wrist与手部根点分别保留 |
| `observation.state.body_world` | 66 → 22×3 | SMPL 风格顺序的 22 个身体关节点，世界坐标位置，米；EgoSuite 为源观测，HM 为模型 FK |
| `observation.state.body_head_floor` | 66 → 22×3 | dense22 在相机地面投影参考系中的位置，米；Z 为离地高度 |
| `observation.valid.body_head_floor_position` / `observation.interpolated.body_head_floor_position` | 22 / 22 | 逐关节可用性／是否依赖插值，包含身体点和相机参考的状态 |
| `observation.state.body_world_rotation` | 88 → 22×4 | **全局**四元数，`wxyz`，不是父关节局部旋转 |
| `observation.state.hand_left_world` / `hand_right_world` | 各 63 → 21×3 | wrist、thumb、index、middle、ring、pinky |
| `observation.state.head_world` / `_rotation` | 3 / 4 | 源头部参考轨迹，参考点类型见每段 provenance |
| `observation.state.camera_world` / `_rotation` | 3 / 4 | 实际第一视角相机轨迹 |
| `observation.state.head_reference_world` / `_rotation` | 3 / 4 | EE4D/EgoEgo 风格的统一头戴参考轨迹，原点为所选相机光心 |
| `observation.state.head_reference_floor_world` / `_rotation` | 3 / 4 | 头戴参考系在地面的绝对投影，只保留水平朝向 |
| `observation.state.head_reference_height` | 1 | 头戴参考点离地高度，米 |
| `observation.environment.floor_height` / `floor_normal` | 1 / 3 | 输出世界坐标中的水平地面；法向量 `[0,0,1]` |
| `observation.valid.floor` / `floor_verified` | 1 / 1 | 有地面标签／地面标签经标定确认；二者含义不同 |
| `observation.valid.head_reference_basis` | 1 | 原生相机轴到统一参考轴的变换是否核对 |
| `observation.features.dino` | 5120 → 5×1024 | `dinov2-with-registers-large`：CLS＋4个 register，按 token 顺序展平，float32 |
| `observation.valid.*` | 每关节或每模态 | 有效性；初始化的 DINO 特征全部无效 |
| `observation.interpolated.*` | 每关节或每模态 | 是否通过插值生成 |
| `observation.source_timestamp` | 1 | 源 episode 时间；float64 |
| `observation.source_frame_left/right` | 1 | 插值端点在源段中的行号 |
| `observation.video_timestamp` / `video_time_error` | 1 | 真正提取特征的视频 PTS 及 PTS−目标时间 |

数组以一维 feature 存入 Parquet，身体按 joint-major、DINO 按 token-major 展平；恢复时按表中形状 reshape。`meta/ontology.json` 保存关节及 DINO token 顺序，`meta/info.json` 保存 feature dtype、shape 和逐维名称。状态位置、四元数、6D旋转、地面和 DINO 为 float32；源时钟、视频 PTS 为 float64；索引为 int64；mask 为 bool。世界旋转将关节局部列向量映射到世界坐标。无效XYZ、6D子块和DINO写零，保留的原四元数列以单位四元数表示无效旋转，读取时必须使用mask。新16列与原XYZ、全局wxyz、相机、时钟、DINO各列物理共存于同一主Parquet。

两个来源共用以下存储约定：EgoSuite 保留完整身体22点与原生全局旋转；EgoExo4D-HM 用匹配的SMPL-H模型解码为同一22点顺序。身体为 dense22×(6D＋XYZ)，完整头戴参考原点为所选相机光心，身体local参考为该相机的地面投影。归档保存30Hz绝对轨迹；窗口接口默认选15Hz并生成EE4D风格18维头戴条件。视觉特征按归档帧对齐，完整保存5个token。相同存储形状不代表两来源关节轴已认证；当前原生轴训练模式仅允许EgoSuite，混合HM须另行验证配准与关节轴。

| 窗口返回值 | 形状 | 约定 |
|---|---|---|
| `window.arrays["observation.state.body_{world,floor}_rot6d_xyz"]` | T×198 → T×22×9 | 规范化／地面解析后重新编码的6D＋XYZ |
| `window.arrays["observation.state.hand_{left,right}_floor"]` | T×63 → T×21×3 | 根据同一窗口地面参考重新派生的显式手部位置 |
| `window.arrays["observation.state.body_head_floor"]` | T×66 → T×22×3 | 身体位置，米；地面投影参考，仅水平朝向 |
| `window.arrays["observation.valid.body_head_floor_position"]` | T×22 | 每个身体点独立mask |
| `window.trajectory` / `trajectory_valid` | T×18 / T×18 | 完整头戴相对地面9维＋地面参考增量9维，选帧后计算 |
| `window.arrays["observation.features.dino"]` | T×5120 → T×5×1024 | 顺序为 `cls, register_0, register_1, register_2, register_3` |
| `window.arrays["observation.valid.dino"]` | T | 同一帧5个token共用一个mask，未提取时为false |

其中 T 为所选训练帧数。18维包含两组「旋转矩阵前两行的6D旋转＋3D位置」，仅在窗口读取时派生，不是额外的30Hz存档列。DINO不执行时间插值；选帧时取对应归档帧的特征和mask，不重新计算。

dense22固定顺序如下（索引从0开始）：

```text
pelvis, left_hip, right_hip, spine1, left_knee, right_knee,
spine2, left_ankle, right_ankle, spine3, left_foot, right_foot,
neck, left_collar, right_collar, head, left_shoulder, right_shoulder,
left_elbow, right_elbow, left_wrist, right_wrist
```

目标坐标约定为 X 向前、Y 向左、Z 向上，米。**只有经核对的 `world_from_source` 和 `coordinate_verified=true` 才表示已经完成这一对齐。** EgoSuite 文档给出这一坐标约定；HM 的具体发布版本仍需在真实样本上核对。示例保留 HM 的原坐标、标记待验证，`validate --strict` 会拒绝未确认的结果。

全局四元数的“全局”只确定目标坐标系，不能保证两个来源的关节局部轴相同。`body_basis_correction[j]` 定义从统一关节局部坐标到源关节局部坐标的旋转：

```text
p_out = R_world_from_source @ (scale * p_source) + t_world_from_source
R_out[j] = R_world_from_source @ R_source[j] @ body_basis_correction[j]
```

源局部轴尚未核对时，旋转仍可保存用于检查，但 `rotation_basis_verified=false`，不能据此声称统一了旋转监督。骨架拓扑相同也不意味着两来源骨长、拟合误差、标注质量相同；不会把 HM 的拟合位置称为光学 mocap 真值。双手根点与身体 wrist 不被强制粘合，quality 中保留其偏差统计。thumb 四级点的解剖命名在发布文档间存在差异，当前使用有序点名，不擅自宣称完成解剖对齐。

`body_head_floor` 只使用关节位置和相机参考旋转，**不要求 `rotation_basis_verified=true`**。v5已保存数值可逆的 `22×(6D+XYZ)`，其旋转轴认证仍沿用源provenance。将源全局wxyz编码成6D不会补齐未知的关节局部轴；原生单来源监督与跨来源统一轴认证是独立事项。

## 主归档 v5 整合与恢复

当前schema为 `egomotion-dense22-hands21-camera-floor-dino5-v5`。新转换直接生成全部列；既有v4通过 [unified_migration.py](src/egomotion_pipeline/unified_migration.py) 整合到原 `/gaozt-test1/datasets/MixEgoMotion`。不重新解码MCAP，不重提DINO，不拟合SMPL或重估关节中心。世界旋转由原全局wxyz归一化得到，6D取矩阵前两行；身体／双手局部位置均为 `R_floor.T @ (p_world − p_floor)`，身体局部旋转为 `R_floor.T @ R_world_joint`。

先在独立staging中编码全部episode，逐列记录dtype、shape及数值SHA256，再从磁盘独立解码验证所有旧列和16个新列；已完成DINO的token、mask、PTS、误差、checkpoint签名和complete状态全部保留。可选 `--native-root` 与历史原生dense22派生文件逐值对照。全部验收后才提交主 `data/` 目录、逐段provenance和全局meta。

```bash
# 小规模试验不发布主归档，不创建阻塞主读取的全量journal。
python -m egomotion_pipeline.unified_migration \
  --dataset /gaozt-test1/datasets/MixEgoMotion --workers 4 \
  --pilot-episodes 0 1 --no-commit
# 正式整合；同一命令恢复中断。开始前停止现有训练读取者。
python -m egomotion_pipeline.unified_migration \
  --dataset /gaozt-test1/datasets/MixEgoMotion --workers 16 \
  --native-root /gaozt-test1/datasets/MixEgoMotion/derived/egosuite-native-dense22-v1
watch -n 5 cat /gaozt-test1/datasets/MixEgoMotion/.pipeline/dense_v5_upgrade.json
```

整合持有归档独占写锁；全量journal状态经过 `staging → validated → committing → complete`，失败记录 `failed` 并保留原phase。未complete时pipeline及MotionExpert拒绝读取；v4对象在迁移后需要重新创建，旧split／stats／checkpoint不能用于v5继续训练。提交保留原data、config、meta和episode记录至journal中的 `rollback_relative_path`，不自动删除v4备份。两个目录rename之后或metadata部分发布后中断都可重跑恢复；`complete`最后发布。

每段 `motion_representation` 保存schema、producer SHA256和motion数值摘要。MotionExpert的split与统计指纹包含这一运动身份，DINO列与其状态不进入motion摘要；更新视觉特征不会冒充运动标签改变，运动表示变更则必须重建split与训练统计。历史native sidecar继续作为审计产物保留，v5训练直接读取主归档，不再依赖其存在。

缺失地面的段保留世界表示，归档floor／hand-floor保持无效；本次整合不在整段上添加脚点均值兜底。窗口未知地面可以用选帧后有效脚点兜底，`fore`仅使用观测前缀估计；已存档地面作为固定离线环境标注使用。有效插值仍有效，并显式记录来源；手wrist与身体wrist的偏差仅报告，不强制粘合。原生局部轴认证仍为false，数值回环通过不等于解剖轴通过认证。

**2026-10-11 v5正式整合及统计验收完成：6,614段、26,841,893帧。** 主归档仍为 `/gaozt-test1/datasets/MixEgoMotion`；16进程迁移约1,189.67秒，主Parquet共565,873,546,076字节。逐段独立磁盘验证确认全部旧数值列精确保留，与历史native身体表示最大数值差为0。6,614段DINO均complete，其中26,539,780帧有效；其余302,113帧继续由原mask标为无效。DINO原始来源provenance精确保留；地面失败9,403帧及全部段 `rotation_basis_verified=false` 保持原状。

首次发布汇总见 [production_summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/production_summary.json)，记录发布、统计、完整模型及真实边界四项验收、训练资产和当时保留的回退快照。该历史报告保持原样；快照现已在下述独立复验通过后删除。

首次发布报告位于 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/`，`publication-validation.json` 与 `statistics-validation.json` 均 `passed=true`，任务 `status.json` 及全量journal均complete。

**2026-10-11当前v5全量独立复验及v4备份清理已完成。** 最新汇总见 [production_summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/production_summary.json)。[validation.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/validation.json) 重新从当前主Parquet解码全部6,614段、26,841,893帧、75列，包含全部DINO数值，并核对逐列摘要、mask、几何和来源元数据；失败0段，耗时429.16秒。物理Parquet共565,873,546,076字节，解码数值共661,840,555,701字节。此流程不读取v4备份，不复用历史逐段通过结果；`passed`、`full_dataset_validation` 与 `all_live_episodes_validated` 均为true。11段未知地面与原生轴尚未认证继续作为显式缺失／provenance保留。

[motionexpert-validation.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/motionexpert-validation.json) 在148.72秒内完成当前train/val/test读取、forecast屏蔽、split与统计资产的只读验收，确认训练读取不依赖v4备份；没有执行本次模型前向／反向或训练，也没有重算统计。[revalidation-tests.xml](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/revalidation-tests.xml) 为23项独立故障回归全部通过，覆盖DINO值篡改、mask／几何／inventory错误及临时备份删除后正常读取。

[cleanup-result.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/cleanup-result.json) 记录仅删除 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/migrations/dense-v5-full-20261011T080945Z/backup/`：13,233个文件，删除前逻辑498,734,566,282字节（其中原Parquet 498,649,658,599字节），分配498,785,428,992字节。删除在独占写锁及最终身份核对后执行，当前v5文件和episode记录身份未变，删除后15Hz窗口读取通过。该快照已不存在，**不能再从它回退v4**；归档journal与migration status保持 `state=phase=complete`，`rollback_available=false`。主归档、native sidecar、DINO权重、训练资产与历史报告／receipt／inventory继续保留。

已生成 [splits.json](../MotionExpert/exp/dense_k12/splits.json) 和 [stats.pt](../MotionExpert/exp/dense_k12/stats.pt)，划分按operator／原录制分组且集合互不相交：

| split | episode数 | 原30Hz帧数 |
|---|---:|---:|
| train | 5,323 | 21,473,273 |
| val | 647 | 2,684,309 |
| test | 644 | 2,684,311 |

训练统计使用15Hz、120帧窗口，共360,208个train窗口，16个连续分片合并；245＋18维均有有效观测且统计全部有限，非有限观测0，重复的地面参考增量共享统计，脚接触保持identity归一化。窗口存在重叠，统计通道计数不等于独立归档帧数。统计SHA256为 `54457b581a63daf978fee99e70436ace5ad5c634fca0c26ebac59603816279a1`；train/val/test reader检查通过。

首次发布时的完整模型smoke已通过：[training-smoke.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/training-smoke.json)。399,807,365参数主干使用 `B=1,T=120` 的真实归档窗口前向／反向，身体6D、XYZ、全局增量、双手和接触各组loss有限；输入／输出／相机／视觉投影梯度非零且有限，700个有梯度的参数张量全部有限。train/val/test读取均为245维运动、18维相机、5×1024视觉；fore未来条件屏蔽及历史loss排除检查通过。未执行optimizer step、未写checkpoint，此检查不代表已经训练出可用权重。本次备份清理验收未重新执行该模型smoke，当前源码身份与历史验收范围见新的MotionExpert只读报告。

同一正式任务目录保留 [pipeline-tests.xml](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/pipeline-tests.xml)、[audit-publication.py](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/audit-publication.py)、[validate-full-model.py](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/validate-full-model.py)；[validation-artifacts.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/validation-artifacts.json) 记录其SHA256和字节数，已逐文件核对一致。

真实边界验收 [boundary-validation.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/boundary-validation.json) 通过，覆盖13段、17个窗口、646项检查。11个地面失败episode共9,403帧，其中10段完全没有有效脚点，无法通过当前规则获得地面／local监督。唯一episode894的双脚仅在原30Hz帧956有效：从原帧840开始的窗口，脚点位于15Hz窗口索引58（零起点）的未来区，recon可兜底，fore仍无效且Dataset明确拒绝；从原帧900开始，脚点位于索引28的观测前缀，两任务均可兜底，但地面仍未认证。

边界检查同时通过缺失身体／手部点的mask与零填充、世界逆变换、15Hz增量和相机参考统一验证；身体／双手逆变换最大误差均不超过 `1.6e-6m`，这是表示数值误差而非标注精度。DINO数值读取为0，原Parquet与episode JSON身份保持不变。保留world数据和全量发布不代表每段、每个窗口都有完整可用的训练标签，训练仍需逐特征mask与任务窗口筛选。

## 历史 v4 EgoSuite 原生旋转 dense22 导出

以下记录v4阶段独立导出与验收证据；当前v5已将同一原生身体表示纳入主归档。`export-native-dense22` 从已转换的 EgoStandard 30Hz 归档生成独立的运动派生文件，保留 EgoSuite 原生关节局部轴。每个关节保存「旋转矩阵前两行6D＋XYZ」，全部22点为每帧198维；不拟合SMPL、不右乘关节轴校正、不从XYZ重新估计旋转。原始归档的相机、头戴轨迹、地面、手部、DINO及时间对应仍通过相同episode/frame索引复用，该历史导出阶段未修改MotionExpert。

| 派生列 | 每帧形状 | 定义 |
|---|---:|---|
| `observation.state.body_world_rot6d_xyz` | 198 → 22×9 | 原生世界旋转6D＋原生世界XYZ |
| `observation.state.body_floor_rot6d_xyz` | 198 → 22×9 | 相机地面参考系中的原生旋转6D＋XYZ |
| `observation.valid.body_{world,floor}_rot6d_xyz_rotation` | 各22 | 完整旋转数值及其参考系可用性 |
| `observation.valid.body_{world,floor}_rot6d_xyz_position` | 各22 | XYZ及其参考系可用性，独立于身体旋转 |
| `observation.interpolated.body_{world,floor}_rot6d_xyz_{rotation,position}` | 各22 | 有效派生量是否依赖身体或地面参考插值 |

派生文件同时保存源运动投影列，包括原生全局wxyz、身体XYZ、地面参考位姿、mask及帧/源时间索引。对有效关节，世界XYZ原值保留，`R_world`来自源四元数；局部表示为 `R_floor.T @ R_world` 和 `R_floor.T @ (p_world - p_floor)`。无效9D块的旋转6维、位置3维分别写零并使用独立mask；数值零不能当作有效性判断。没有地面参考时，世界表示仍可用，floor表示无效。此静态派生文件不使用未来训练窗口的脚点兜底；已有窗口接口仍可在选帧后重建缺失地面参考。

源必须是`egosuite`、`source_global_wxyz`且未配置`body_basis_correction`。`rotation_basis_verified=false`原样保留，表示跨来源完整轴尚未认证；本次原生6D导出不升级该认证，也不支持直接混合HM绝对旋转监督。新增原生表示与下节的标定候选、SMPL-X研究结果分别存放。

```bash
# 小规模验收使用独立输出；全量时移除--limit并选择正式目录。
egomotion-pipeline export-native-dense22 \
  --dataset /gaozt-test1/datasets/MixEgoMotion \
  --output /gaozt-test1/datasets/MixEgoMotion/derived/egosuite-native-dense22-v1 \
  --workers 8
egomotion-pipeline validate-native-dense22 \
  --dataset /gaozt-test1/datasets/MixEgoMotion \
  --output /gaozt-test1/datasets/MixEgoMotion/derived/egosuite-native-dense22-v1 \
  --workers 8
```

新目录的`data/chunk-XXX/file-XXX.parquet`沿用源episode索引；`meta/info.json`定义派生schema，`episodes/`保存逐段来源指纹及验证结果，`manifest.json`固定输入范围，`status.json`和`events.jsonl`记录进度，`validation.json`报告全量验收。恢复时校验源码、源运动数据和已提交派生文件；源或表示变化时拒绝静默复用。`--limit`可扩大已有partial前缀至全量，但不能缩小到排除已提交episode的范围；这种调用在改写进度、元信息或验收报告前报错。数据写入采用逐段原子提交，失败不声明完成。读取不包含DINO列，也不重写源Parquet。

```bash
watch -n 5 cat /gaozt-test1/datasets/MixEgoMotion/derived/egosuite-native-dense22-v1/status.json
tail -f /gaozt-test1/datasets/MixEgoMotion/derived/egosuite-native-dense22-v1/events.jsonl
```

**2026-10-11 全量验收完成：6,614 段 EgoStandard、26,841,893 帧，导出与独立全量磁盘复验均为 0 失败。** 正式任务使用32个CPU进程，导出256.63秒、复验112.39秒；派生Parquet共70,294,065,111字节。有效世界XYZ逐值保留，6D回环矩阵最大元素误差为`5.24e-8`，地面XYZ回环最大误差为`1.20e-7m`，这些是表示数值误差，不是运动标注精度。世界／地面表示的有效关节帧比例分别为87.9575%／87.9379%；全量保存不等于所有关节观测有效，训练需使用逐关节mask。

地面参考无效的11段共9,403帧（全量帧数0.03503%），全部为`no_stationary_feet`、候选数0，DBSCAN未运行。供应方lower_body坏帧标注导致10段脚点全无效；episode894仅一帧双脚有效，无法计算连续脚速。世界系仍保存这些片段的有效关节；地面系保持无效。无有效脚点的窗口无法均值兜底，894中包含该孤立脚帧的窗口可按已有规则得到未认证的启发式地面，不能当作存档地面已恢复。

任务证据位于`/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/native-dense22-20261011T045036Z/`：`production_summary.json`、`validation_result.json`、`integrity.json`、`ground-failures.json`、`real-parity.json`和`pipeline-tests.xml`。299项pipeline测试通过；源episode与配置SHA、源Parquet属性、投影运动SHA及MotionExpert文件SHA检查通过，本次未读取DINO数值或重写源归档。入口为`.pipeline/current_native_dense22.json`，完整review见[review.md](review.md)。

## EgoSuite dense22 关节轴标定与验收

**2026-10-10 验收结论：22 关节的统一旋转轴尚未通过验收，不能将 `rotation_basis_verified` 改为 true。** 已联网下载 26 项官方文档、固定 commit 的 DevKit 源码和数据目录资料，保存于 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/official-downloads/`，逐文件 URL、SHA256 和搜索范围见 `manifest.json`、`findings.json`。在这些公开资料和所检查 MCAP 的 schema／metadata 中，没有找到完整的供应方 22 关节 bind pose 或局部轴定义。此结论限于已检查的资料，不表示供应方内部不存在标定文件。

新增 [joint_basis.py](src/egomotion_pipeline/joint_basis.py)、[audit_egosuite_joint_axes.py](scripts/audit_egosuite_joint_axes.py) 和 [fit_joint_basis.py](scripts/fit_joint_basis.py)。几何候选按 `C_j @ target_rest_direction ≈ R_source[j].T @ source_world_bone` 拟合；正式成对参考按 `R_target[j] ≈ R_source[j] @ C_j` 拟合。两种工具均不会自动认证参考或将候选标为 verified。相机地面参考系中的目标旋转随后应为 `R_floor.T @ R_world[j] @ C_j`；6D 使用 UniEgoMotion 的前两行展平规则。XYZ 则为 `R_floor.T @ (p_world[j] - p_floor)`，不受关节局部轴校正影响。

真实审计使用 24 位不同人物的 EgoStandard MCAP，共 23,622 个身体帧；12 人拟合、12 人留出验证，无人物重叠。按整数 capture timestamp、2ms 容差匹配身体／双手，剔除已有坏帧标注和无效姿态；每个人对每个关节总权重相同。目标使用本机 UniEgoMotion 的 `SMPLX_NEUTRAL.npz`，安全读取数值数组、校验完整 55 点父子结构和前 22 点顺序，零 betas 下以 `J_regressor @ v_template` 得到静止关节点。这个 SMPL-X 资产用于当前 UniEgoMotion 的目标参考，不能代替 HM 解码所需的独立 SMPL-H 资产。

| 关节组 | 可确定的旋转自由度 | 本次结论 |
|---|---:|---|
| pelvis、spine3 | 3，依赖多条不共线骨骼方向及目标骨架对应假设 | 留出骨骼方向 P95 分别 7.26°、7.08°，未通过 5° 数值门槛；骨架形状差异也会产生残差 |
| 左右 wrist | 3，依赖四个手掌 MCP 方向及指序对应假设 | 留出方向 P95 为 3.49°、3.11°；数值通过，但参考语义尚未认证 |
| 其余 15 个单子关节 | 2 | 骨骼方向数值通过，但绕骨轴的 twist 无法从一条骨骼确定；SVD 返回的候选 twist 不作为真值 |
| head、左右 foot | 0 | dense22 无下游方向约束，候选为 null，不使用 identity 冒充已标定 |

以上误差是**骨骼／手掌方向误差**，不是完整 SO(3) 旋转误差。24 种共享正交轴置换中最优假设的留出方向 RMSE 为 9.92°、P95 为 13.75°、最大值 32.35°；这些残差受静止骨架差异影响，不能单独证明原生轴置换错误，也不能认证其正确。手部 wrist 和身体 wrist 的原生局部轴也不能混用：独立 8 人审计中二者旋转存在稳定约 90° 偏移。几何拟合不等同于原生关节轴认证。

完整报告、22 个候选与逐关节自由度、源数据指纹、模型 SHA256、采样配置和安全 NPZ 位于：

```text
/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/
  official-downloads/{manifest,findings}.json
  real-oracle-audit.json
  calibration-24operators-reviewed/
    report.md
    report.json
    manifest.json
    status.json
    anchors-and-candidates.npz
    samples/*.npz
```

复现几何审计时使用新的输出目录；不写入生产归档、不运行 RGB 解码或 GPU 推理：

```bash
cd /root/workspace/wbc/guanzerong/cvpr/pipeline
python scripts/audit_egosuite_joint_axes.py \
  --catalog /gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/source-catalog.json \
  --target-model /gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz \
  --output /gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/calibration-rerun \
  --episodes 24 --max-frames 1024 --seed 20261010
```

正式全 22 关节验收需要供应方可信 bind pose／各关节局部 XYZ 定义，或独立可信、同步且处于同一世界系的完整成对旋转参考；静止姿态的 XYZ 本身不能补齐 twist。成对参考 NPZ 必须包含数值 `source_rotations`、`target_rotations`（`T×22×3×3`），布尔 `fit_mask`、`validation_mask`（`T×22`，互不重叠），Unicode `joint_names`（上述 dense22 顺序）；可提供 Unicode `group_ids`（`T`），脚本将拒绝拟合／验证人物组重叠。默认每关节两组各至少 8 个有效样本，全部拟合和留出 SO(3) 误差不超过 5°。

```bash
python scripts/fit_joint_basis.py --reference /path/to/paired-reference.npz \
  --output /path/to/new-calibration-report.json --max-error-deg 5
```

该命令只检查数值一致性，即使退出码为 0，输出仍保留 `reference_verified=false`、全部 `verified=false`。参考从同一字节快照读取并计算 SHA256，拟合期间文件变化会被拒绝。`load_verified_joint_basis()` 仅接受独立审核完成的 `egomotion-joint-basis-calibration-v1` 文件：固定右乘约定、目标轴 ID、22 点顺序、proper SO(3)、逐关节 verified 和 numerical_pass 全 true、可信参考类型及证据 SHA256 均必须匹配。它检查声明和证据完整性，不能代替供应方真实性审核；当前生产转换未接入该 loader，不能只手改配置或审核标记来补齐缺失证据。

## 方案 B：SMPL-X 拟合 dense22 小规模验证

2026-10-11 已完成 10 位不同人物／任务的离线验证，每人各取两个不重叠的 8 秒、15Hz 窗口，共 2,400 帧。第一个窗口仅估计并冻结 10 维 shape，第二个窗口用观测拟合 pose；两种消融共享 shape、纯 XYZ 初始化和有效性规则。选择的是身体／旋转有效率至少 92%、相机和地面有效的窗口，不代表全量动作覆盖。这是拟合残差验证，不是运动预测或绝对旋转真值精度验证。

新增研究模块 [smpl_fit.py](src/egomotion_pipeline/smpl_fit.py) 和入口 [validate_smpl_fit_pilot.py](scripts/validate_smpl_fit_pilot.py)。拟合采用可微 FK、带 mask 的稳健位置损失、弱姿态／时序正则；第二组增加世界旋转增量约束 `G[t+1] @ G[t].T ≈ R_source[t+1] @ R_source[t].T`，不直接监督未知源关节轴的绝对旋转。丢失／时间间隔异常／疑似跟踪跳变的相邻边不参与该约束。开头双髋位置缺失时，用窗口内首个具有多方向几何约束的骨盆帧向前、向后初始化；这是允许利用未来 XYZ 的离线初始化，不读取源四元数，也不构成 twist 标定。

| 消融 | XYZ 平均残差 | XYZ P95 | 世界旋转增量平均残差 | 增量 P95 | 增量最大值 |
|---|---:|---:|---:|---:|---:|
| 仅 XYZ | 1.500cm | 4.860cm | 0.725° | 2.867° | 33.054° |
| XYZ＋旋转增量 | 1.500cm | 4.866cm | 0.201° | 0.721° | 5.872° |

误差按原归档世界系、有效关节点／相邻边汇总，不做 Procrustes 对齐。每种模式评估 1,200 帧、26,096 个位置和 25,804 条旋转边。拟合输出包括同一模型 FK 的 `fitted_world_xyz`、`fitted_world_rotations`、local rotations、betas 和 translation，以及相机地面参考系的 `body_floor_rot6d_xyz`（T×22×9，前两行 6D＋XYZ）。原始位置／旋转 mask、估计旋转可用性分开保存；模型补出的旋转不当作观测。实际相机轨迹、既有地面、18D 条件与帧索引保留，DINO 不重提取。

**结论：方案 B 的数值拟合与表示导出可行，目前仍不满足全量生产转换条件。** 颈部／锁骨平均 XYZ 残差约 4.5–5.0cm；源 neck→head 平均骨长约 8.81cm，而当前拟合 SMPL-X 为 15.87cm。需要确认关节中心对应及目标骨架策略，不能只靠增加迭代次数。头、脚及单骨段的绝对 twist 没有独立真值：固定 90° twist 可保持 XYZ 和世界旋转增量不变，因此这些旋转必须标为 `model_fitted_estimate`，原生轴认证仍为 false。正式接入时须区分“目标模型轴定义明确”“标签由拟合估计”“源轴经认证”三个状态。

本次仅使用本机许可资产 SMPL-X neutral；未验证 HM 的 SMPL-H 解码／跨来源统一，也未使用学习姿态先验、额外手掌锚点、相机–头安装标定或 RGB 重投影约束。官方 `smplx==0.1.28` 完整模型抽查两模式各 5 帧，XYZ 与小型 FK 最大差 4.44×10⁻⁸m，全局旋转元素误差 0；这证明模型参数解码一致，不证明动作或旋转是真值。脚穿地与静止脚速度是代理指标，不等于人工接触验收。

研究输出完全独立于生产归档：

```text
/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/
  smpl-fit-pilot-10operators-v2-20261011/
    report.md / report.json / comparison.png
    manifest.json / status.json / process.log
    code/                 # 冻结的 8 个源码文件
    inputs/               # 原始观测、相机、mask 与 18D 条件
    results/              # 两组模型拟合结果
    diagnostics/          # 独立审查与官方完整前向复核
```

源 10 段运动列与记录、模型及冻结源码的前后指纹一致。完整 pipeline 测试 239 项通过，含缺失初始双髋、独立 FK、mask／重置和不可辨识 twist 的回归检查。MotionExpert 和生产转换入口未接入本拟合器。

复现时安装研究依赖并使用**新输出目录**；模型资产需自行提供，不随代码发布：

```bash
cd /root/workspace/wbc/guanzerong/cvpr/pipeline
python -m pip install -e '.[smpl-fit,test]'
python scripts/validate_smpl_fit_pilot.py \
  --archive /gaozt-test1/datasets/MixEgoMotion \
  --model /gaozt-test1/guanzerong/UniEgoMotion/body_models/smplx/SMPLX_NEUTRAL.npz \
  --selection /gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/calibration-24operators-reviewed/manifest.json \
  --output /gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smpl-fit-rerun \
  --episodes 10 --frames 120 --iterations 300 --shape-iterations 150
```

## SMPL-H 双来源重定向研究（2026-10-11）

**小样本数值验证完成，尚不满足双来源全量生产转换条件。** 本次读取现有v5运动列研究EgoSuite→共同SMPL-H body22，同时验证真实HM解码和Aria相机同步；现有EgoSuite主Parquet、DINO、split／stats和MotionExpert均未改动。完整记录见 [report.md](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/report.md)、[ego-report.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/ego-report.json) 和独立 [retarget-quality.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/retarget-quality.json)。

共同模型资产为 `/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz`，SHA256 `be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947`：6,890顶点、完整52关节，**实际16个shape basis**，不截成10维、不以零padding替代真实basis。研究入口 [validate_smplh_dual_source_pilot.py](scripts/validate_smplh_dual_source_pilot.py) 使用新 [smplh_retarget.py](src/egomotion_pipeline/smplh_retarget.py)，安全数值NPZ加载、完整父子结构验证及可微body22 FK；可信资产布局与数值解码通过不等于原始旋转轴认证。

关节中心采用显式观测模型 `p_source[j] ≈ p_SMPLH_FK[j] + G_anchor[j] @ d[j]`，仅neck／左右collar／head允许受限固定局部偏移；默认anchor为对应父片段，偏移长度上限12cm、L2系数0.01。canonical输出始终为SMPL-H真实FK中心和全局旋转，带offset的source landmark重建单独保存。拟合约束保留原XYZ和世界旋转增量，betas／offset在校准后冻结。全局旋转转为UniEgoMotion前两行6D，world／floor均为`T×22×9`；相机及18D轨迹、源双手各21点和原mask保持不变。

EgoSuite使用原先10位人物的两个不重叠120帧／15Hz窗口，共2,400独立输入帧。主实验用5位train人物第一窗口估计offset，按人物等权汇总为source共享d；另一5位人物不参与该校准。所有人物第二窗口分别拟合baseline和冻结共享offset的pose，两者使用相同固定零16维beta及纯XYZ初始化。零shape实验只隔离中心策略，不能解决身材差异。另做每人仅以第一窗口肢体骨长估计shape、校准其offset，第二窗口冻结二者的独立消融；这属于同人校准，不是跨人物共享映射。

| 离线重建比较 | 有效XYZ数 | 观测平均／P95残差 | 世界旋转增量平均／P95残差 |
|---|---:|---:|---:|
| 5位留出人物，零shape baseline | 12,896 | 1.061／2.364cm | 0.214／0.770° |
| 同5位人物，冻结source共享offset | 12,896 | 0.794／2.038cm | 0.206／0.718° |
| 10人，各自肢体shape，baseline | 26,096 | 0.830／2.114cm | 0.191／0.689° |
| 同10人，冻结各自shape＋offset | 26,096 | 0.486／1.420cm | 0.182／0.665° |

这些数值是第二窗口观测参与pose拟合后的**source landmark重建残差**，不是未来预测精度，也不是独立canonical关节中心／绝对旋转真值精度。共享neck／左collar／右collar／head偏移长度为4.509／2.601／2.175／2.909cm，5人间scatter RMS为1.352／0.855／0.883／1.077cm；说明单一source偏移仍存在人物／shape／初始化差异。全部15组校准没有触及12cm偏移上限；肢体shape的16维beta绝对最大2.164，没有触及±5边界。没有使用学习人体姿态先验；固定twist不可辨识仍存在，`model_fitted_estimate`、原生轴认证false及全部`canonical_rotation_certified=false`继续保留。

真实H模型上身几何比先前所用X资产更接近此批源骨架，但仍不相同：源neck→head平均8.807cm，H male零beta为10.273cm，旧X neutral零beta为16.410cm；源spine3→neck为22.500cm，H为21.468cm，旧X为16.861cm。此处同时改变模型family与gender／资产，不能把差异归因于family单独变化；历史X拟合beta后的15.87cm也不是同一对比口径。

独立诊断重新解码40个结果NPZ，以NumPy FK核对输出（XYZ float32最大差5.96×10⁻⁸m），确认源XYZ／旋转／mask／帧时间、双手21点和18D条件原dtype逐值保留。5位留出人物共享offset后，身体与原手腕平均偏差0.545cm、P95 0.724cm；同人shape＋offset的10人结果为0.095cm、P95 0.197cm，源身体／手部wrist原先重合。源静止脚代理仅定义为相邻有效脚点速度小于0.15m/s，留出人物水平速度均值0.025679→0.025817m/s；地面穿透最大值源2.606→共享offset 2.719cm，P95均0。它们使用存档地面和关节中心，是接触／滑步代理，不能当作足底与真实接触验收。

HM身体验证覆盖`cmu_bike02_4`、`sfu_cooking026_8`、`uniandes_dance_020_37`，共1,391原30Hz帧，36个实际帧与官方完整SMPL-H前向核对通过，最大body XYZ差5.45×10⁻⁸m；实际16维beta全部保留，源手部21点逐值保留，详见 [hm-body/summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/hm-body/run-v2/reports/hm-body/summary.json)。HM的`valid`仍是chunk有限性，不能冒充逐关节跟踪置信度。

**HM相机生产接口仍被帧对应问题拦截：** 三take的HM长度109／872／410，而timesync为110／873／411，同步序列均多一帧；现有`prepare-aria`严格长度检查拒绝接收。官方`timestamps[:num_frames]`支持“取前HM长度”的研究候选，但发布数据缺少完整source frame映射，不能直接推广成全部take的截尾规则。真实RGB光心、MPS世界／重力、相机基变换和公开2D投影数值检查通过；公开2D本身由3D投影生成；已生成并核对9张叠图中的部分可见手点，仍缺独立图像标注与源帧映射认证。`camera_0`是重复exo相机而非ego相机，CPF／人体head／RGB光心分别保留。详见 [hm-camera/summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/hm-camera/run-v2/summary.json)；其生产ready为false；[身体投影补充报告](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/hm-camera/supplement-summary.json) 的数值通过也不升级帧对应认证。

研究拟合与既有拟合回归34项、body model资产类型回归6项，共40项通过，包含真实SMPL-H非零16beta FK对照、offset冻结、分组隔离、source缺失／gap／reset和90°twist反例。要接入生产，还需扩大动作／缺失／重置／接缝覆盖，确定source共享或subject校准契约、身体与手腕对应，以及估计标签mask／reader；本次未修改生产或训练接口。

## 安装与执行

```bash
cd /root/workspace/wbc/guanzerong/cvpr/pipeline
python -m venv .venv
.venv/bin/pip install -e '.[dino,aria,test]'
```

复制并修改 `configs/unified.example.json` 中路径、版本、坐标配置。配置中的相对路径相对于配置文件目录解析。只需 EgoSuite 时，可直接使用 `configs/egosuite-mcap.example.json`，其输入路径为当前下载目录 `/gaozt-test1/datasets/EgoSuite/EgoStandard-body/mcap`，无需 HM 模型、Aria 资产或事先转换为源 LeRobot 数据。通过 source 的 `episode_ids` 指定 JSON ID 列表，才能明确复现选取的数据范围；目录下所有数据不一定正好是目标小时数。`limit` 可限制每个来源的样本数，用于单独的小规模预览输出。

仅处理当前 MCAP 下载的命令：

```bash
egomotion-pipeline inspect --config configs/egosuite-mcap.example.json
egomotion-pipeline convert --config configs/egosuite-mcap.example.json --output /gaozt-test1/datasets/egomotion_mcap_30hz
egomotion-pipeline validate --dataset /gaozt-test1/datasets/egomotion_mcap_30hz
# 检查相机／地面参考是否可用；坏身体点仍由逐关节 mask 表达
egomotion-pipeline validate --dataset /gaozt-test1/datasets/egomotion_mcap_30hz --require-floor --require-head-reference
# 如需保证所有帧的全部22点可用，可额外使用 --require-body-head-floor
```

首次检查可复制配置并在 source 加 `"limit": 1`，输出到独立预览目录；完整转换用新目录。当前配置和真实样本验收使用 `EgoStandard-body`，`root` 应指向其 `mcap` 子目录，以免混合下载批次。

全量EgoStandard配置为 [configs/egostandard.full.json](configs/egostandard.full.json)，不设置 `limit` 或 episode筛选。本次运动数据输出为 `/gaozt-test1/datasets/MixEgoMotion`；6,614个MCAP的输入清单、实际配置和代码副本固定在输出的 `.pipeline/jobs/<job_id>/`，当前运动任务入口为 `.pipeline/current_job.json`。该任务已冻结0.5.0代码，继续生成v3存档；转换和校验结束后才能升级为本页的v4视觉格式。本次不提取DINO，升级只扩展待提取的占位列。

[scripts/process_egostandard.py](scripts/process_egostandard.py) 使用有界进程队列处理单段数据，由一个主进程提交归档，每25段或60秒更新公共元数据；首段也立即更新。任务目录中的 `status.json` 提供阶段、已处理／失败／跳过段数和提交帧数，`process.log` 记录进度，`events.jsonl`／`failures.jsonl` 保存逐段结果和异常。只含14点的来源会记录跳过，其他输入异常记录失败并继续处理剩余段；存在失败时最终阶段为 `completed_with_errors`，不会声明全部成功。完成转换后自动执行归档校验并写入 `validation.json`。中断后使用该任务的配置、脚本和代码副本重跑即可恢复；已有段仍校验输入指纹。

```bash
egomotion-pipeline inspect --config configs/unified.json
egomotion-pipeline prepare-aria --config configs/unified.json --output /data/aria_aligned
egomotion-pipeline convert --config configs/unified.json --output /data/egomotion_30hz
egomotion-pipeline validate --dataset /data/egomotion_30hz
egomotion-pipeline extract-features --dataset /data/egomotion_30hz --config configs/dino.json
egomotion-pipeline validate --dataset /data/egomotion_30hz --require-dino --require-floor --require-head-reference
```

`prepare-aria` 需要原始 EgoExo4D 的 `takes.json`、take 的 `*noimagestreams.vrs`、MPS closed-loop 轨迹、`captures/<capture>/timesync.csv` 和 frame-aligned RGB MP4。读取 take 的 `[timesync_start_idx, timesync_end_idx)`；timesync 行数必须与 HM 帧数一致。缺失／重复设备时间戳不前向填补，MPS 最近 pose 时间误差超过阈值也标无效。

HM 示例保持 `coordinate_verified=false`：必须先核实身体拟合世界和 MPS 世界是否一致、Z 是否朝上，以及所需 `world_from_source`。未核实时地面和 `body_head_floor` 的 mask 为 false；不能只为产生有效输出就改成 true。`--strict` 还检查身体旋转轴的验证状态，位置输入可使用上面的各项 `--require-*` 检查配合逐关节 mask。`--require-body-head-floor` 要求每帧全部22点有效，厂商坏帧或 HM invalid 帧会使其失败。

HM 的 `cam_R/cam_t` **不自动用作 ego pose**：官方 `read_cameras.py` 默认 view 0 是外部相机副本，1–4 是外部相机，仅可选 Aria 导出另外追加 ego 相机。转换默认要求独立 Aria sidecar；做纯动作预览可设置 `require_camera=false`，头部／相机缺失部分会保留 false mask。

HM 身体使用与发布拟合匹配的 **SMPL-H model asset**，从 `v_template/shapedirs/J_regressor/kintree_table` 求 shape-conditioned rest joints，再用 `root_orient + pose_body` 做 FK，输出身体 22 点和全局旋转。不要用 `joints3d[:22]`：它是 OpenPose25 拓扑。双手分别取 `joints3d[25:46]`、`joints3d[46:67]`，无需解码 hand PCA。`betas_per_frame` 按帧使用，保留原 `chunk_ranges`。官方实现可能将模型缺少的 shape basis 补零；示例显式开启 `pad_shape_coefficients` 以匹配该行为，provenance 记录模型 SHA256 和原 basis 数量。必须自行提供许可允许使用的匹配资产；如果只有带 scipy sparse 的可信 `.pkl` 模型，额外安装 `.[model-pickle]`。

官方 HM 渲染脚本使用 `smplh/male/model.npz`，统一示例按该路径设置。SMPL-X 的前22父子树也相同，但顶点模板和 shape 空间不同；解码器会拦截可识别的 SMPL-X／错误标准模型。骨架检查与 shape 补零都不能代替匹配资产。2026-10-11已取得上述实际16basis的SMPL-H资产，并对3个真实HM take完成身体解码、timesync和MPS数值研究；本次模型不需要为已发布16beta补零。生产相机接受仍受多一帧的同步长度及图像／帧映射验收拦截，不能借用外部相机补齐；详见本页双来源研究记录。

## EgoSuite MCAP 输入

`type="egosuite_mcap"` 递归查找完整的 `.mcap` 文件，每个文件为一个 episode；忽略下载中的 `.mcap.part`。也可将 `root` 指向单个 MCAP。Protobuf 类型由文件内嵌的 schema 动态解码，无需安装厂商 Devkit、提供 `.proto` 或生成 `*_pb2.py`。

| 输入 topic | 读取内容 |
|---|---|
| `/pose/body` | 按官方 SMPL 顺序读取身体 22 点位置和全局 `wxyz` 四元数 |
| `/pose/left_hand`、`/pose/right_hand` | 双手各 21 点世界位置 |
| `/pose/head` | 源头部位姿，单独保留 |
| `/sensor/camera/{camera}/extrinsic` | `world → {camera}_camera` 的 C2W 位姿，按字段名读取四元数，不取逆 |
| `/sensor/camera/{camera}/intrinsic` | 保存首帧相机标定 `K/D/R/P`、尺寸和畸变模型 |
| `/sensor/camera/{camera}/video` | 直接读取 `foxglove.CompressedVideo`，供后续 DINO 提取 |
| `/session/metadata` | 只取内部 episode UUID、scene/environment/operator ID 和批次版本 |
| `/annotation/bad_frame/pose/*` | 将厂商逐帧坏帧标记转为对应位置／旋转 mask |
| `/annotation/bad_frame/camera/{camera}` | 提取视觉特征时屏蔽对应坏图像 |

`camera` 支持 `head_left` 和 `head_right`。默认要求对应的相机外参；纯动作预览可设 `require_camera=false`。缺失手部或 head topic 时保存 false mask；缺失完整身体 topic 报错。只有 14 点 upper-body 的来源跳过，其他不符合 22 点的格式报错，不推测腿部姿态。

EgoSuite-native 消息优先使用 `header.ts`，Foxglove 消息使用 `timestamp.seconds/nanos`，相机外参使用所选 transform 的 timestamp。缺失这些时间字段时才回退到 MCAP `log_time`，并记录使用次数。先用整数纳秒减去首个身体 capture 时间，再转换为相对秒，原 Unix 纳秒原点保存于 provenance 的 `clock.origin_ns`。不把 Unix 纳秒先转换成 float，也不重建假设的 30fps 源时钟。

身体 capture 时间作为源帧轴，其他位姿按最近 capture 时间关联，默认 `max_pose_error_s=0.002`；超阈值或缺帧位置／旋转标无效，不按消息行号配对或前向填补。随后沿用统一 30Hz 位置插值／四元数 SLERP。逐 topic 的匹配数量和误差保存在 `pose_alignment`。重复位姿 capture 时间报错；身体消息允许按 capture 时间排序，视频要求文件内 capture 时间严格递增。

手部坏帧标记屏蔽两只手；upper/lower body 标记分别屏蔽其官方 22 点子集，**upper body 不是身体数组的前 14 点**。只有发布的逐帧 `is_bad` 会参与 mask，聚合 `bad_frame_ratio.json` 不当作逐关节置信度。缺少这些 topic 不表示质量已经确认，有限值／零四元数检查仍保留。

`episode_ids` 使用 **MCAP 文件名去掉扩展名的 UUID**，例如 `["a83cceaa-ab0c-45af-8c09-f9f3a6446677"]`。实测文件名与 `/session/metadata.task_info.episode_uuid` 可能不同：文件名用于选择／恢复，内部 UUID 单独保存为 `source_episode_uuid`。重名文件会报错，避免恢复时碰撞。

旧 `type="egosuite_lerobot"` 仍通过 `meta/episodes` 查找 Parquet shard 和 MP4 offset，兼容多段共用同一文件。

## 地面与 EE4D 头戴参考系

地面高度定义为输出世界坐标中的水平平面 `z = floor_height`，不会直接减去所有位置。`head_reference_height = head_reference_world.z − floor_height`。这样既保留绝对轨迹，也能在后续读取时计算离地位置。**地面高度为零是有效数值；缺失由 mask 表达。**

每个 source 的 `ground` 支持 `estimate`、`manual`、`disabled`。已知场景地面可配置 `{"mode":"manual","height_m":0.0,"verified":true,"source":"scene_calibration"}`，高度必须是已完成 `world_from_source` 转换后的世界坐标值，单位米。未知时默认采用 EE4D 的静止脚部高度聚类思路：

1. 使用身体点 10/11，以实际时间差计算速度，默认低于 0.15m/s 的点进入候选；不跨缺失点或长时间空洞计算。
2. 对候选高度执行 scikit-learn `DBSCAN(eps=0.005, min_samples=3)`，选择非噪声簇中最低的高度中位数。重复高度用 `sample_weight` 保留密度；忽略 `label=-1` 的噪声点。
3. 默认减去 1cm 的 toe joint-to-floor 偏移，与 EE4D 的偏移一致；不同来源的足部点定义应通过真实样本调整 `toe_offset_m`。

不再施加簇内4cm跨度上限或额外0.25秒支持度门槛；`spread_m`／`support_s` 仅作诊断。旧 `min_support_s`／`max_cluster_spread_m` 配置会报错，需移除，避免误以为它们仍参与筛选。与原代码相比，保留实际时间差与有效性检查，并修正将 DBSCAN 噪声标签当作高度簇的行为。参数全部保存。估计结果 `floor_valid=true`、`floor_verified=false`；缺少低速脚点、没有密度簇或世界竖直轴未核对时，30Hz 存档中的地面及依赖字段仍为无效。

地面估计失败后，pipeline 的 `Archive.read_window` 在**训练帧率选帧和窗口截取之后**，用当前窗口的有效脚点10/11计算 `mean(feet_world.z) - 0.02m`。随后从 world 位姿重新生成相机地面投影、头部离地高度、dense22 local 和18维头戴条件。无效脚点的零占位不参与平均；窗口没有有效脚点、地面显式禁用或世界Z轴未核验时不兜底。已经有效的地面，包括真实零地面，保持原值。窗口兜底为 `floor_valid=true`、`floor_verified=false`，不会回写30Hz存档，也不会将整段均值当作所有窗口的地面。

多层脚部高度记录 `multiple_height_levels`，不自动删除台阶／攀爬段；单一水平面无法描述坡面或完整楼梯地形。估计地面不是实测真值，双脚一直踩在高台上也可能把高台估为地面。`--require-floor` 检查存档可用性，不执行窗口兜底，也不把估计升级为实测标定。

`head_world` 继续保留源 head/CPF，身体点 15 继续表示骨架 head。**统一的 EE4D 头戴轨迹使用新增 `head_reference_world`，不能用源 `head_world` 替代。** EE4D 从 world-to-camera 外参取逆，随后右乘固定轴变换；我们的 camera 已是 camera-to-world，所以不再次取逆：

```text
B = [[0,-1,0], [1,0,0], [0,0,1]]
R_world_reference = R_world_camera @ camera_from_ee4d_camera @ B
p_world_reference = p_world_camera
```

默认 `head_reference={"mode":"ee4d"}` 按 adapter 的 `camera_pose_basis` 选择转换：

| 相机来源 | `camera_from_ee4d_camera`（C） | 总右乘 C @ B | `basis_policy` |
|---|---|---|---|
| `aria_rgb_native`：未旋转图像的 Aria SDK RGB 标定 | I | B | `native_aria_rgb` |
| `egosuite_opencv_camera_to_world`：OpenCV X右/Y下/Z前 | B | diag(-1,-1,1) | `egosuite_opencv` |
| 未知来源，包括尚未核验的旧 LeRobot 相机字段 | I 占位 | B 占位，mask未验证 | `unknown` |

Aria 原生 RGB 图像相对自然显示方向旋转了90度；复用 EE4D 的 B 后，统一参考轴为 X左、Y上、Z光学向前。EgoSuite 的相机本身已是 upright OpenCV 定义，因此需要总右乘 diag(-1,-1,1)。这只是相机轴约定转换，不是设备安装角度或相机到解剖头部的标定，不增加固定 1.6m 高度或 headjoint 偏移。参考的 +X 是左向，光学视线是 +Z；不能用 +Z 替换原 EE4D 地面投影所用的 +X。

`head_reference.camera_from_ee4d_camera` 可手动覆盖，须为 proper 3×3 rotation；手动矩阵即使来自 Aria，也默认 `basis_verified=false`，确认后显式设置 true。已知来源自动模式默认验证的是文档中的轴约定，未知来源保持 false；可显式设 false 禁用依赖该轴的地面参考。来源、策略、矩阵和验证含义保存于 episode provenance。图像预处理的旋转不会自动改变相机 pose，使用已转向／裁切的图像做投影检查时还须对应处理标定。

地面投影参考系使用 `head_reference` 局部 +X 在世界 XY 的投影作为朝向，删除 pitch/roll，保留原绝对 XY，并令 Z 为地面高度。轴变换未核对、地面缺失或 +X 几乎竖直时，该投影为无效。投影在重采样完整四元数之后计算，避免先投影再插值引入朝向误差。30Hz 存档只生成绝对参考位姿；窗口规范化及轨迹增量由下面的 pipeline 窗口接口按需计算。

dense22 的参考原点是这个相机光心的地面投影；不是 headjoint、CPF，也不是直接减去相机高度的原始相机坐标：

```text
p_floor = [camera_world.x, camera_world.y, floor_height]
body_head_floor[j] = R_world_floor.T @ (body_world[j] - p_floor)
valid[j] = body_position_valid[j] AND head_reference_floor_valid
interpolated[j] = valid[j] AND (body_position_interpolated[j]
                               OR head_reference_position_interpolated
                               OR head_reference_rotation_interpolated)
```

local Z 始终与世界 Z 一致，仅扣除一次地面高度。pitch/roll 保存在完整头戴轨迹中，不带入身体的地面参考。必须先重采样 world 身体位置／完整头戴四元数，再重新投影和计算 local 位置；不会插值已经变换的 local 位置。坏关节只屏蔽对应点，相机参考缺失则屏蔽该帧全部 local 点，零值须和 mask 一起使用。EgoSuite 默认选择 `head_left`，HM 选择 Aria `camera-rgb`；应固定相机选择并保留相机 ID，左右相机光心的物理差异不会被自动消除。

HM 仍使用 take 的真实 timesync 时间查询 MPS，借用 EE4D 的几何定义，不照搬其近似 30.005Hz SLAM 时钟。

## 视频与 DINO

视频仍保留在源目录或源 MCAP 内；统一数据只存 DINO 数值特征，**不声明虚假的 `observation.images.*` video feature，也不重新编码视频**。`.pipeline/episodes/*.json` 保存 MP4 路径／时间偏移，或 MCAP 路径／video topic／纳秒时钟原点，并保存预处理配置。DINO 提取依赖这些原文件。提取完成后 LeRobot 数值数据读取无需源视频。迁移源目录前应完成提取；这些 provenance 文件是本管线恢复／审核所需，应与数据一同保存。

`timestamp` 每段从 0 开始，30Hz；`source_timestamp` 保留源时钟（MCAP 为相对首个身体 capture 的秒）。DINO 目标 PTS 为 `source_timestamp − timestamp_origin + from_timestamp`。MP4 使用真实视频 PTS；MCAP 使用 `CompressedVideo.timestamp` 减去同一个身体时钟原点，给每个压缩 packet 赋时间，并保持解码器状态处理关键帧／delta 帧和尾帧。支持 `h264`、`h265`、`vp9`、`av1`，具体 codec 仍取决于本机 PyAV/FFmpeg。视频解码按流处理，不加载整段 RGB，也无需导出 MP4。

按真实 PTS 匹配最近帧，超过默认约 17.67ms 的误差标无效，不按视频行号假设 30fps。帧复制、重复视频帧及低帧率来源不会产生新的视觉信息；降低或调整误差阈值会改变可用比例。厂商图像坏帧即使成功解码也不提取特征。`--require-dino` 要求所有目标帧有特征，缺失帧可能使其失败，应通过 mask 或筛选段处理。

两个来源统一使用预训练并冻结的 **`facebook/dinov2-with-registers-large`**（ViT-L/14、hidden size 1024、4个register）。模型输出取 `last_hidden_state[:, :5]`：CLS在前，随后4个register；patch token不保存，也不对这5个token再做平均池化或额外L2归一化。普通 `dinov2-large` 没有这4个register，不能补零伪装成相同特征，提取器会拒绝该模型。[官方模型配置](https://huggingface.co/facebook/dinov2-with-registers-large/blob/main/config.json)

与本地UniEgoMotion提取脚本使用同一预处理规则：先完成来源对应的图像校正／方向处理，再使用RGB uint8输入、bicubic将短边缩放为336，保留宽高比，**不做中心裁剪**；缩放到[0,1]后使用ImageNet mean `(0.485,0.456,0.406)`、std `(0.229,0.224,0.225)`。这会显式覆盖HF processor的256缩放／224裁剪默认值。来源的物理相机、视野、校正和图像质量差异仍保留；模型和张量形状统一不意味着这些差异已经消除。

[configs/dino.example.json](configs/dino.example.json) 已固定2026-10-10核实的官方revision `e4c89a4e05589de9b3e188688a303d0f3c04d0f3`。两个来源须共用checkpoint、revision和预处理；本地权重目录用 `revision="local"`，记录权重／配置SHA256。episode provenance记录实际processor、token顺序、形状、推理dtype、Torch／Transformers版本及视频指纹，防止续跑混用特征。DINO依赖为Torch≥2.2、Transformers≥4.48且<5；测试extra限制datasets<5以兼容该HF Hub依赖。

模型权重和Parquet特征均为float32；本次A100配置使用TF32加速矩阵乘法，输出仍为float32。TF32会产生少量浮点差异，不保证与关闭TF32的FP32推理逐值一致。UniEgoMotion的部分sidecar为float16，两者表示结构一致，但不保证不同库／精度下逐值完全相同。训练如需float16可在读取后转换。新motion模型从头训练时，DINO仍使用冻结的预训练权重；之后如果微调DINO，需要另行设计端到端训练。

0.7.0的本地及示例DINO配置启用以下优化；不填写优化字段的旧配置仍沿用HF预处理、FP32推理及同步读取。TF32、BF16、固定内存和模型编译要求CUDA；CPU验证时应设`inference_precision="fp32"`、`pin_memory=false`、`compile_model=false`。

| 配置 | 本次取值 | 作用 |
|---|---|---|
| `inference_precision` | `tf32` | A100的TF32矩阵乘法；可选`fp32`或`bf16`，保存始终为float32。 |
| `attention_implementation` | `sdpa` | 使用Torch SDPA注意力；可选`auto`或`eager`。 |
| `preprocessor`、`preprocess_workers` | `pil_parallel`、8 | 每卡8个持久线程并行执行PIL缩放和归一化；测试与HF参考输入逐值一致。 |
| `prefetch_batches` | 2 | 在CPU线程预处理下一批，与GPU推理重叠；队列最多保留2个已准备批次，另有1批正在准备。 |
| `decoder_threads` | 4 | 每段视频使用4线程解码，保持原始PTS及坏帧匹配；提前结束MCAP读取时排空解码器。 |
| `cpu_threads` | 2 | 多卡worker限制Torch、BLAS及Arrow线程，避免CPU线程过量竞争。 |
| `pin_memory` | true | 使用固定内存和非阻塞CPU→GPU传输。 |
| `optimized_storage` | true | DINO浮点列使用Parquet byte-stream-split及低级别Zstd，数值和schema保持一致。 |
| `compile_model` | false | 编译模型可选；真实样本前向收益约4%，本次未启用。 |

RGB几何处理、336短边bicubic、归一化、token顺序、视频时间匹配和mask规则保持原定义。每段provenance额外记录优化参数、实际注意力实现、计算精度、Pillow版本及输出dtype；改变这些参数会被恢复签名检查拒绝，避免在同一归档中混用特征。

最终冻结代码的八卡真实测试使用8段共12,288个目标帧，其中11,630帧满足原有视频规则。包含解码、预处理、推理和Parquet提交的速度为**551.85目标帧/秒**，原版为111.47，约**4.95倍**。全部运动列、视频PTS、mask逐值一致；与原版FP32特征比较，段级相对L2误差最大0.1501%，单token余弦相似度最低0.999410。相同样本Parquet由250.28MB降为217.36MB。BF16在端到端测试中未明显快于TF32且偏差更大；延后RGB转换的试验路径较慢，已移除。按26,841,893帧外推，纯提取约13.51小时；从当前等待阶段到最终校验预计15–18小时，实际耗时取决于共享GPU负载、段长度及存储吞吐。

正式预训练权重已下载到 **`/gaozt-test1/datasets/MixEgoMotion/weights/dinov2-with-registers-large/e4c89a4e05589de9b3e188688a303d0f3c04d0f3/`**，包含 `model.safetensors`、`config.json`、`preprocessor_config.json` 和官方模型README。权重文件为1,217,539,376字节，SHA256与官方LFS记录一致：`edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b`；各文件的校验记录保存在权重目录上一级的 `<revision>.manifest.json`。先前 `/gaozt-test1/guanzerong/weights/dinov2-with-registers-large/` 下载目录已按用户要求删除。

本地加载使用 [configs/dino.local.json](configs/dino.local.json)，相同配置另存于 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/dino.local.json`；`checkpoint` 指向上述固定版本目录，`revision="local"`，加载只使用本地文件并记录SHA256。CPU验收中，真实EgoStandard RGB 1456×1920，经短边336预处理成为 `1×3×336×443`，正式权重输出 `1×5×1024`、float32且全部有限。验证报告位于 `.pipeline/jobs/dino-download-20261010T130952Z/verification.json`，下载入口为 `.pipeline/current_dino_download.json`。历史八卡任务入口为 `.pipeline/current_dino_extraction.json`；当前6,614段DINO complete结果见上述v5正式报告。

归档在30Hz运动网格上按实际PTS匹配视频；窗口默认15Hz时只选对应帧，保持视觉、头戴和身体同步。此处统一5×1024 token布局，并未照搬已有UniEgoMotion sidecar的序列封装或低帧率时间索引。历史0.6阶段只修改pipeline，尚未适配MotionExpert；当前MotionExpert已使用完整CLS＋4个register的 `T×5×1024` 输入，不退回仅CLS。

每个 source 的 `video_preprocessing` 可指定：

- `{"mode":"raw","rotate_ccw_quarters":0}`：按原图处理，明确保留源畸变。
- `{"mode":"preundistorted","rotate_ccw_quarters":0}`：已校正的输入；EgoSuite 常规 `/sensor/camera/*/video` 使用该默认标记，原始畸变视频保存在独立 `/sensor/camera_raw/*/video` topic。HM 可设置 `video_root`、`video_template` 指向帧对齐的校正视频。
- `{"mode":"aria","rotate_ccw_quarters":1,"output_size":512,"focal_length":150,"output_rotate_ccw_quarters":-1}`：示例旋转值须按实际视频验证；先转到 native Aria 图像方向，检查尺寸符合 VRS calibration，再执行 Aria 校正。尺寸／裁切不符直接报错，不套用猜测的标定。仍需核对具体下载批次是否符合官方已校正规格。

选择 MCAP 原始相机视频时，显式配置 `video_topic="/sensor/camera_raw/head_left/video"`，并将 `video_preprocessing.mode` 设为 `raw`；没有显式 preprocessing 时此 topic 默认按 raw 处理。普通视频 topic 缺失时不会自动切换到 raw。

## EgoExo4D-HM 视频 DINO sidecar

[process_egoexo_dino.py](scripts/process_egoexo_dino.py) 和 [video_dino.py](src/egomotion_pipeline/video_dino.py) 对已下载的全部 ego RGB MP4 提取独立视觉特征。复用 EgoSuite 正式任务的冻结 DINO encoder、权重、336短边预处理、TF32/SDPA及float32的CLS＋4个register输出；本次任务源代码和配置固定在 `/gaozt-test1/datasets/EgoExo-4D-HM/.pipeline/jobs/dino-videos-20261011T141250Z/`，按用户指定使用GPU 4、5、6、7。

清单包含2,649个Aria和41个ego GoPro视频，共2,690个MP4、11,930,939个30Hz目标帧。各视频header帧数与take的timesync区间长度一致。输出为 `/gaozt-test1/datasets/EgoExo-4D-HM/DINO/takes/<take>/<video_stem>.parquet` 及同名JSON：`observation.features.dino` 每帧5120维float32，按token-major恢复为5×1024；另存frame_index、float64的take时间、实际视频PTS、PTS误差和 `observation.valid.dino`。超出原17.67ms误差门槛的帧保持mask=false和零特征，不插值DINO。

源MP4按HM示例的raw／不旋转策略处理，明确保留Aria源畸变；不能标记为已去畸变。sidecar时钟是frame-aligned take时间，不是发布HM合并NPZ的行号。视觉提取不会建立HM逐帧映射或认证相机／身体坐标，后续关联必须使用明确核对的时间映射。

每视频原子提交，记录输入指纹、encoder和配置、Parquet SHA256；相同命令恢复时只复用校验通过且签名一致的输出。任务记录逐卡进度及逐视频失败，单视频最多重试三次，并在全量结束后独立解码检查全部Parquet的schema、时间、mask、有限性及checksum。缺视频或失败不会报告全量成功。

```bash
JOB=/gaozt-test1/datasets/EgoExo-4D-HM/.pipeline/jobs/dino-videos-20261011T141250Z
PY=/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/dino-torch260-cu118/bin/python
# 查看状态／逐卡已处理帧
watch -n 5 cat "$JOB/status.json"
tail -f "$JOB/process.log"
# 中断后的续跑：使用已冻结的相同任务；GPU选择在launch.json中保存
$PY -u "$JOB/launch.py" full
```

## 目录、恢复与训练边界

```text
egomotion_30hz/
  data/chunk-000/file-000.parquet       # 每段一个文件，所有来源合并到一个索引空间
  meta/info.json
  meta/tasks.parquet                   # LeRobot 必需的恒定结构标签 motion；不处理源文本
  meta/episodes/chunk-000/file-000.parquet
  meta/ontology.json
  weights/dinov2-with-registers-large/<revision>/  # 本次下载的固定版本模型、配置及processor
  .pipeline/config.json
  .pipeline/episodes/00000000.json      # source ID、标定状态、输入指纹、quality、视频/DINO provenance
```

写入使用进程锁、临时 Parquet 原子替换和逐段提交。重新运行同一配置会恢复；输入内容或配置变化拒绝覆盖，选择新输出目录。文件仍在写时不要并发训练；多文件元数据更新不是全数据集事务。未完成 DINO 提取的段有 `dino_ready=false`，对应 feature 零值且 mask=false；不能作为已完成特征训练。

当前新输出schema为 **`egomotion-dense22-hands21-camera-floor-dino5-v5`**；LeRobot容器版本仍是v3.0。历史0.6.0将DINO列1024→5120，0.7.0增加提取优化；v5另新增16个身体／手部表示及mask列。v3（1024维）与v4（5120维）保持按各自规格读取；`upgrade-dino`只升级v3→v4，不顺带生成v5列，之后通过上述v5整合入口升级。v2及更早schema拒绝。视觉格式升级与v5整合都不重算原已有效地面；要应用不同DBSCAN规则，应重新转换运动数据。

运动转换完全结束后，尚未提取DINO的v3存档可执行：

```bash
egomotion-pipeline upgrade-dino --dataset /gaozt-test1/datasets/MixEgoMotion
egomotion-pipeline validate --dataset /gaozt-test1/datasets/MixEgoMotion
# 准备好GPU环境后，使用已下载的本地权重显式启动视觉提取
egomotion-pipeline extract-features --dataset /gaozt-test1/datasets/MixEgoMotion --config configs/dino.local.json
# 使用八张物理GPU，每卡独立处理不同episode
egomotion-pipeline extract-features --dataset /gaozt-test1/datasets/MixEgoMotion --config configs/dino.local.json --gpus 0 1 2 3 4 5 6 7
```

`--gpus` 为可选参数，省略时沿用单进程提取。多卡模式使用spawn进程，每卡独立加载同一份冻结权重，动态领取episode；每个worker最多保留一段待提交结果，全部Parquet、episode provenance及共享元数据由主进程写入。每卡仍使用配置中的batch size（本次为32），归档特征仍为float32、5×1024。GPU编号及卡数不进入特征签名，相同权重、软件版本和配置可更换卡数续跑；已完成段会先检查签名，worker错误会停止任务并保留已提交段。不要同时启动多个归档写入任务。

[scripts/process_dino.py](scripts/process_dino.py) 提供任务状态、逐卡进度、恢复及提取后完整校验，`--wait-for-job` 指向DINO格式升级任务目录。源视频坏帧或无法满足PTS误差阈值的帧保持mask=false，最终报告分别记录完成episode数、有效特征帧数和缺失帧数。

```bash
python scripts/process_dino.py --dataset /gaozt-test1/datasets/MixEgoMotion \
  --config configs/dino.local.json --job-dir /data/dino-job \
  --wait-for-job /gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-format-20261010T130540Z \
  --gpus 0 1 2 3 4 5 6 7
```

以下为历史0.7.0八卡等待／优化阶段的监控说明；当前全量DINO已完成并逐值保留至v5。历史任务为 `dino-extract-opt-8gpu-20261010T151033Z`，目标为6614段、26,841,893帧，独立GPU环境位于 `.pipeline/runtimes/dino-torch260-cu118/`（Torch2.6.0+cu118、Transformers4.49.0）。当时原八卡等待任务停止、尚未提交生产DINO，入口为 `.pipeline/current_dino_extraction.json`。任务目录保留冻结源码、配置、SHA256、日志、状态及验收报告路径。每卡batch size为32，原逐卡正式权重验收峰值CUDA分配约2.48GiB；优化代码的真实前8段×64帧验收通过，512帧特征全部有效，运动字段、视频PTS和mask逐值一致。等待阶段为`waiting_for_format_upgrade`，格式升级及校验完成后自动提取，实际提取为`extracting`，最终成功须为`completed`且`validation_passed=true`。

```bash
watch -n 5 cat /gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-extract-opt-8gpu-20261010T151033Z/status.json
tail -f /gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-extract-opt-8gpu-20261010T151033Z/process.log
```

`completed_episodes`、`completed_frames`为已提交进度，`workers`为逐卡当前段和已编码帧数；`committed_frames_per_second`及`estimated_remaining_seconds`在开始提交后更新。等待升级时生产DINO完成数仍为0，不能把独立验收帧数当作生产进度。

升级持有同一归档写锁，预检全部段必须为pending且feature／mask／视频时间均为零，然后分批原子替换DINO列；其他Arrow列逐值保留，不重新解码MCAP或重算运动。已完成的旧CLS特征拒绝升级，须在新存档重新提取真实5-token特征。升级中断时普通读取被拦截，重跑 `upgrade-dino` 恢复；`.pipeline/dino_upgrade.json` 保留进度和原始转换provenance。升级后的存档继续用于读取／视觉提取，保留旧运动实现来源，不允许用新代码冒充同一运动实现续写。先完成冻结运动任务再升级。

[scripts/process_dino_upgrade.py](scripts/process_dino_upgrade.py) 支持 `--wait-for-job <运动任务目录>`：等待该任务转换及校验完成、释放写锁后，再执行升级和完整校验。后台升级任务入口为 `.pipeline/current_dino_upgrade.json`，其任务目录保存冻结代码、`process.log`、`status.json` 和 `validation.json`；等待阶段不会修改生产数据。运动任务失败时升级任务报错停止，`completed_with_errors` 的源文件失败数会继续保存在升级状态中。

已为本次MixEgoMotion全量归档启动等待任务 `dino-format-20261010T130540Z`，状态文件为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-format-20261010T130540Z/status.json`。运动任务和格式升级分别监控，不能把 `waiting_for_conversion` 当作已升级或已提取DINO。

原转换默认只声明 `all` split；统一参数归档的训练划分见上文 `meta/splits.json`。原转换没有做归一化统计。论文划分应在 **参与者／场景／原始 take** 层面防止泄漏，不能把相邻片段随机分到不同 split。

pipeline 提供独立窗口接口；MotionExpert使用独立的投影Parquet reader读取同一主归档，并重建任务相关窗口。`start_frame` 是30Hz存档帧号，`num_frames` 是选帧后长度，`fps` 可为30的正整数约数，默认15Hz；超出episode报错，不补帧或外推。默认按窗口首帧执行 UniEgoMotion yaw／XY 规范化：相机参考+X的水平投影对齐世界+X，首帧相机XY归零，世界Z不平移。首帧相机参考缺失、轴未验证或投影退化会报错；`canonicalize=False` 可读取保留相应无效mask的未规范化窗口。MotionExpert固定选择零起点偶数帧 `0,2,4,…`，再计算15Hz增量。

窗口内地面解析后重新派生 local 字段，头戴18维为 `完整头戴位姿相对地面参考(6D旋转+3D位置)` 与 `相邻地面参考增量(6D旋转+3D位置)`。6D旋转取矩阵前两行，首帧增量沿用 EE4D 的初始地面位姿，并在编码时令地面Z为0；没有重复扣除地面高度。缺失相机使该帧头戴local无效，相邻增量需两端都有效。增量在15Hz选帧之后计算，不能直接隔帧抽取30Hz增量。

```python
from egomotion_pipeline.storage import Archive

archive = Archive("/data/egomotion_30hz")
window = archive.read_window(0, start_frame=0, num_frames=80, fps=15)
body_dense_local = window.arrays["observation.state.body_head_floor"].reshape(80, 22, 3)
body_dense_valid = window.arrays["observation.valid.body_head_floor_position"]  # (80, 22)
body_dense6d_local = window.arrays["observation.state.body_floor_rot6d_xyz"].reshape(80, 22, 9)
hand_left_local = window.arrays["observation.state.hand_left_floor"].reshape(80, 21, 3)
head_trajectory = window.trajectory       # (80, 18)
head_trajectory_valid = window.trajectory_valid  # (80, 18)
ground = window.metadata["ground"]       # 包含窗口是否兜底及有效脚点数
dino_tokens = window.arrays["observation.features.dino"].reshape(80, 5, 1024)
dino_valid = window.arrays["observation.valid.dino"]  # (80,)，pending时全部false
dino_cls = dino_tokens[:, 0, :]          # (80, 1024)，仅CLS；全部5个token仍保留
```

`read_window` 每次加载一段的Parquet；重复采样同段时，可缓存 `read_arrays(...)` 的结果并调用 `egomotion_pipeline.windows.prepare_window`。若做时间插值增强，应从world位姿和完整参考重新派生，避免复用旧floor缓存。MotionExpert将身体取既有K12顺序：12×(6D＋XYZ)=108维，加地面参考运动9维、双手63＋63维及脚接触2维，合计245维；头戴条件18维、视觉5×1024。归一化仅用训练split统计，接触0/1保持原值；详见[训练说明](../MotionExpert/README.md)。

也可以读取存档中的逐帧数据；这种读取不触发窗口地面兜底：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("local/egomotion", root="/data/egomotion_30hz", download_videos=False)
frame = dataset[0]
body_position = frame["observation.state.body_world"].reshape(22, 3)
body_rotation = frame["observation.state.body_world_rotation"].reshape(22, 4)
body_dense_local = frame["observation.state.body_head_floor"].reshape(22, 3)
body_dense_valid = frame["observation.valid.body_head_floor_position"]
```

## 验证与依据

`pytest` 包含源格式模拟、SMPL-H FK、SLERP、无效／时间空洞处理、真实 MP4 PTS 匹配、Parquet 读写、恢复冲突和 HF datasets／Pandas 元数据读取，以及地面零值、不同帧率的速度阈值、异常脚点、多层高度、EE4D 轴转换、姿态重采样后的地面投影和字段损坏检查。MCAP 测试覆盖内嵌 Protobuf、文件 UUID 选择、Unix 纳秒精度、与 log_time 不同的 capture 时钟、缺帧／错时 mask、upper/lower 坏帧子集、真实 H.264 delta 帧解码、图像坏帧和转换／恢复／特征回写。DBSCAN与窗口测试覆盖宽高度簇、短密度簇、噪声点、重复点密度、15Hz选帧后脚点兜底、缺失相机与增量mask、存档不回写，并与实际MotionExpert数学函数对照18维轨迹及dense22位置。DINO测试覆盖CLS／register顺序、token-major读写、336短边预处理、错误模型／输出拒绝、真实HF小模型CPU前向，以及升级锁、全段预检、中断恢复、运动字段保留和后台任务等待。多卡测试覆盖乱序提交与单卡逐值一致、换卡数续跑、签名冲突预检、worker异常／退出清理、不同encoder拒绝，以及任务进度与前置失败处理。正式大模型已通过八卡真实推理与512帧并行提取验收；全量提取状态以生产任务最终校验为准。匹配SMPL-H资产与Project Aria SDK已完成上述小样本数值研究；生产相机帧对应与跨来源标签仍需独立验收。

- [EgoSuite 官方 MCAP topic 与时间戳](https://docs.lightwheel.net/egocentric_data/mcap_data/overview)
- [EgoSuite 官方身体关节与子集](https://docs.lightwheel.net/egocentric_data/mcap_data/pose/body)
- [EgoSuite 官方头戴相机与 C2W 外参](https://docs.lightwheel.net/egocentric_data/mcap_data/sensor/rgb_camera/head)
- [Foxglove 压缩视频帧规格](https://docs.foxglove.dev/docs/sdk/schemas/compressed-video)
- [EgoSuite 官方 LeRobot 格式](https://docs.lightwheel.net/egocentric_data/lerobot_data/format_specification)
- [EgoExo4D-HM 发布文件说明](https://huggingface.co/datasets/Ego-Exo4D-HM/npz-datasets/blob/main/README.md)
- [HM 官方相机导出代码](https://github.com/Abhiram824/egoexo4d_human_meshes/blob/main/scripts/read_cameras.py)
- [HM 的 Aria 时间同步代码](https://github.com/Abhiram824/egoexo4d_human_meshes/blob/main/slahmr/util/egoexo4d_utils.py)
- [LeRobot 官方数据集代码](https://github.com/huggingface/lerobot/blob/main/src/lerobot/datasets/lerobot_dataset.py)
- [EE4D 官方头戴轴转换](https://github.com/chaitanya100100/UniEgoMotion/blob/main/dataset/egoexo4d_utils.py)
- [Aria 原生 RGB 图像的90度方向](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/samples/undistort_rgb_image)
- [HM 官方模型资产与相机视角说明](https://github.com/Abhiram824/egoexo4d_human_meshes/blob/main/scripts/run_mesh_vis_hands_egoexo.py)
- [EE4D 地面估计](https://github.com/chaitanya100100/UniEgoMotion/blob/main/dataset/egoego_utils.py)
- [EE4D 训练窗口地面兜底与表示](https://github.com/chaitanya100100/UniEgoMotion/blob/main/dataset/representation_utils.py)
- [scikit-learn DBSCAN 与重复点权重](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [DINOv2 官方预训练模型](https://github.com/facebookresearch/dinov2)

MCAP规格于2026-10-09核对；0.5.0运动验收及0.6.0格式升级验收均使用EgoStandard-body样本。验收记录见 [review.md](review.md)。
