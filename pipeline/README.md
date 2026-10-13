# EgoMotion 30Hz 统一数据管线

独立转换 EgoExo4D-HM 发布 NPZ 和 EgoSuite **原生 MCAP** 数据，统一输出 LeRobot v3 数值数据集。仍兼容已有 EgoSuite LeRobot v3 输入。不修改 MotionExpert，不读取源任务描述或语义文本标注。

## 输出与表示

| LeRobot feature | 每帧形状 | 含义 |
|---|---:|---|
| `observation.state.body_world` | 66 → 22×3 | 全部 22 个 SMPL 身体关节，世界坐标位置，米 |
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

数组以一维 feature 存入 Parquet，身体按 joint-major、DINO 按 token-major 展平；恢复时按表中形状 reshape。`meta/ontology.json` 保存关节及 DINO token 顺序，`meta/info.json` 保存 feature dtype、shape 和逐维名称。状态位置、四元数、地面和 DINO 为 float32；源时钟、视频 PTS 为 float64；索引为 int64；mask 为 bool。世界旋转将关节局部列向量映射到世界坐标。无效位置和 DINO 写零，无效旋转写单位四元数，读取时必须使用 mask。

两个来源共用以下训练数据约定：EgoSuite 直接保留完整身体22点；EgoExo4D-HM 用匹配的SMPL-H模型解码为同一22点顺序。身体为 dense22 **位置**，完整头戴参考原点为所选相机光心，身体 local 参考为该相机的地面投影。归档保存30Hz绝对轨迹；窗口接口默认选15Hz并生成EE4D风格18维头戴条件。视觉特征按归档帧对齐，完整保存5个token。

| 窗口返回值 | 形状 | 约定 |
|---|---|---|
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

`body_head_floor` 只使用关节位置和相机参考旋转，**不要求 `rotation_basis_verified=true`**。这为后续 dense22 位置输入提供统一数据；当前 MotionExpert 的 SMPL reader、表示维度、网络和损失还没有切换为 dense22。

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

官方 HM 渲染脚本使用 `smplh/male/model.npz`，统一示例按该路径设置。SMPL-X 的前22父子树也相同，但顶点模板和 shape 空间不同；解码器会拦截可识别的 SMPL-X／错误标准模型。骨架检查与 shape 补零都不能代替匹配资产。本机目前只有 SMPL-X，且所查真实 HM take 缺失 capture timesync 和 MPS 轨迹，所以尚不能完成真实 HM 相机／身体对齐验收；也不能借用外部相机补齐。

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

推理和Parquet均使用float32；UniEgoMotion的部分sidecar为float16，两者表示结构一致，但不保证不同库／精度下逐值完全相同。训练如需float16可在读取后转换。新motion模型从头训练时，DINO仍使用冻结的预训练权重；之后如果微调DINO，需要另行设计端到端训练。

正式预训练权重已下载到 **`/gaozt-test1/datasets/MixEgoMotion/weights/dinov2-with-registers-large/e4c89a4e05589de9b3e188688a303d0f3c04d0f3/`**，包含 `model.safetensors`、`config.json`、`preprocessor_config.json` 和官方模型README。权重文件为1,217,539,376字节，SHA256与官方LFS记录一致：`edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b`；各文件的校验记录保存在权重目录上一级的 `<revision>.manifest.json`。先前 `/gaozt-test1/guanzerong/weights/dinov2-with-registers-large/` 下载目录已按用户要求删除。

本地加载使用 [configs/dino.local.json](configs/dino.local.json)，相同配置另存于 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/dino.local.json`；`checkpoint` 指向上述固定版本目录，`revision="local"`，加载只使用本地文件并记录SHA256。配置的 `device="cuda"` 面向后续GPU提取，本次验证使用CPU：真实EgoStandard RGB 1456×1920，经短边336预处理成为 `1×3×336×443`，正式权重输出 `1×5×1024`、float32且全部有限。验证报告位于 `.pipeline/jobs/dino-download-20261010T130952Z/verification.json`，下载入口为 `.pipeline/current_dino_download.json`。全量DINO特征提取仍未启动。

归档在30Hz运动网格上按实际PTS匹配视频；窗口默认15Hz时只选对应帧，保持视觉、头戴和身体同步。此处统一5×1024 token布局，并未照搬已有UniEgoMotion sidecar的序列封装或低帧率时间索引。后续训练可取 `dino[:,0,:]` 作为T×1024 CLS输入，也可由适配后的网络使用全部token；本次没有修改MotionExpert训练代码。

每个 source 的 `video_preprocessing` 可指定：

- `{"mode":"raw","rotate_ccw_quarters":0}`：按原图处理，明确保留源畸变。
- `{"mode":"preundistorted","rotate_ccw_quarters":0}`：已校正的输入；EgoSuite 常规 `/sensor/camera/*/video` 使用该默认标记，原始畸变视频保存在独立 `/sensor/camera_raw/*/video` topic。HM 可设置 `video_root`、`video_template` 指向帧对齐的校正视频。
- `{"mode":"aria","rotate_ccw_quarters":1,"output_size":512,"focal_length":150,"output_rotate_ccw_quarters":-1}`：示例旋转值须按实际视频验证；先转到 native Aria 图像方向，检查尺寸符合 VRS calibration，再执行 Aria 校正。尺寸／裁切不符直接报错，不套用猜测的标定。仍需核对具体下载批次是否符合官方已校正规格。

选择 MCAP 原始相机视频时，显式配置 `video_topic="/sensor/camera_raw/head_left/video"`，并将 `video_preprocessing.mode` 设为 `raw`；没有显式 preprocessing 时此 topic 默认按 raw 处理。普通视频 topic 缺失时不会自动切换到 raw。

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

当前pipeline为 **0.6.0**，新输出schema为 **`egomotion-body22-camera-floor-dino5-v4`**；LeRobot容器版本仍是v3.0。此次格式变化为DINO列1024→5120，其余运动字段定义和地面处理不变。旧 `egomotion-body22-camera-floor-v3` 可只读，保留其1024维规格；5-token提取须先升级，v2及更早schema仍拒绝。旧0.4的有效地面估计不会因视觉格式升级而重算；要应用0.5的DBSCAN规则，应重新转换运动数据。

运动转换完全结束后，尚未提取DINO的v3存档可执行：

```bash
egomotion-pipeline upgrade-dino --dataset /gaozt-test1/datasets/MixEgoMotion
egomotion-pipeline validate --dataset /gaozt-test1/datasets/MixEgoMotion
# 准备好GPU环境后，使用已下载的本地权重显式启动视觉提取
egomotion-pipeline extract-features --dataset /gaozt-test1/datasets/MixEgoMotion --config configs/dino.local.json
```

升级持有同一归档写锁，预检全部段必须为pending且feature／mask／视频时间均为零，然后分批原子替换DINO列；其他Arrow列逐值保留，不重新解码MCAP或重算运动。已完成的旧CLS特征拒绝升级，须在新存档重新提取真实5-token特征。升级中断时普通读取被拦截，重跑 `upgrade-dino` 恢复；`.pipeline/dino_upgrade.json` 保留进度和原始转换provenance。升级后的存档继续用于读取／视觉提取，保留旧运动实现来源，不允许用新代码冒充同一运动实现续写。先完成冻结运动任务再升级。

[scripts/process_dino_upgrade.py](scripts/process_dino_upgrade.py) 支持 `--wait-for-job <运动任务目录>`：等待该任务转换及校验完成、释放写锁后，再执行升级和完整校验。后台升级任务入口为 `.pipeline/current_dino_upgrade.json`，其任务目录保存冻结代码、`process.log`、`status.json` 和 `validation.json`；等待阶段不会修改生产数据。运动任务失败时升级任务报错停止，`completed_with_errors` 的源文件失败数会继续保存在升级状态中。

已为本次MixEgoMotion全量归档启动等待任务 `dino-format-20261010T130540Z`，状态文件为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-format-20261010T130540Z/status.json`。运动任务和格式升级分别监控，不能把 `waiting_for_conversion` 当作已升级或已提取DINO。

默认只声明 `all` split，没有提前划训练／验证集，没有做归一化统计。论文划分应在 **参与者／场景／原始 take** 层面防止泄漏，不能把相邻片段随机分到不同 split。

pipeline 提供独立窗口接口，当前没有接入或修改 MotionExpert。`start_frame` 是30Hz存档帧号，`num_frames` 是选帧后长度，`fps` 可为30的正整数约数，默认15Hz；超出episode报错，不补帧或外推。默认按窗口首帧执行 UniEgoMotion yaw／XY 规范化：相机参考+X的水平投影对齐世界+X，首帧相机XY归零，世界Z不平移。首帧相机参考缺失、轴未验证或投影退化会报错；`canonicalize=False` 可读取保留相应无效mask的未规范化窗口。

窗口内地面解析后重新派生 local 字段，头戴18维为 `完整头戴位姿相对地面参考(6D旋转+3D位置)` 与 `相邻地面参考增量(6D旋转+3D位置)`。6D旋转取矩阵前两行，首帧增量沿用 EE4D 的初始地面位姿，并在编码时令地面Z为0；没有重复扣除地面高度。缺失相机使该帧头戴local无效，相邻增量需两端都有效。增量在15Hz选帧之后计算，不能直接隔帧抽取30Hz增量。

```python
from egomotion_pipeline.storage import Archive

archive = Archive("/data/egomotion_30hz")
window = archive.read_window(0, start_frame=0, num_frames=80, fps=15)
body_dense_local = window.arrays["observation.state.body_head_floor"].reshape(80, 22, 3)
body_dense_valid = window.arrays["observation.valid.body_head_floor_position"]  # (80, 22)
head_trajectory = window.trajectory       # (80, 18)
head_trajectory_valid = window.trajectory_valid  # (80, 18)
ground = window.metadata["ground"]       # 包含窗口是否兜底及有效脚点数
dino_tokens = window.arrays["observation.features.dino"].reshape(80, 5, 1024)
dino_valid = window.arrays["observation.valid.dino"]  # (80,)，pending时全部false
dino_cls = dino_tokens[:, 0, :]          # (80, 1024)，仅CLS；全部5个token仍保留
```

`read_window` 每次加载一段的Parquet；重复采样同段时，可缓存 `read_arrays(...)` 的结果并调用 `egomotion_pipeline.windows.prepare_window`。若做时间插值增强，应从 world 位置和完整参考重新派生。此接口输出 dense22 位置和头戴条件；现有 MotionExpert 的243维SMPL输入、网络及损失仍需在后续训练任务中适配。

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

`pytest` 包含源格式模拟、SMPL-H FK、SLERP、无效／时间空洞处理、真实 MP4 PTS 匹配、Parquet 读写、恢复冲突和 HF datasets／Pandas 元数据读取，以及地面零值、不同帧率的速度阈值、异常脚点、多层高度、EE4D 轴转换、姿态重采样后的地面投影和字段损坏检查。MCAP 测试覆盖内嵌 Protobuf、文件 UUID 选择、Unix 纳秒精度、与 log_time 不同的 capture 时钟、缺帧／错时 mask、upper/lower 坏帧子集、真实 H.264 delta 帧解码、图像坏帧和转换／恢复／特征回写。DBSCAN与窗口测试覆盖宽高度簇、短密度簇、噪声点、重复点密度、15Hz选帧后脚点兜底、缺失相机与增量mask、存档不回写，并与实际MotionExpert数学函数对照18维轨迹及dense22位置。DINO测试覆盖CLS／register顺序、token-major读写、336短边预处理、错误模型／输出拒绝、真实HF小模型CPU前向，以及升级锁、全段预检、中断恢复、运动字段保留和后台任务等待。匹配SMPL-H资产、Project Aria SDK和正式大模型GPU提取仍需单独验收。

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
