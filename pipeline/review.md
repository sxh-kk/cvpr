# EgoMotion Pipeline Review

- 日期：2026-10-10。
- 范围：仅 `cvpr/pipeline`；MotionExpert 数据读取、表示、模型及损失均未修改。
- 当前版本：pipeline `0.6.0`；schema `egomotion-body22-camera-floor-dino5-v4`。
- 最新结论：DINO格式改动的代码复查、112项测试及EgoStandard旧存档升级验收通过；正式大模型权重SHA256、本地CPU加载及真实EgoStandard单帧推理也已通过。正式GPU全量提取及真实HM配准仍待验收。下文保留0.5.0运动处理的历史证据，最新视觉格式及权重验收见末节。

## 本次处理规则

| 项目 | 实现 |
|---|---|
| 高度聚类 | 实际调用 scikit-learn DBSCAN，默认 `eps=0.005m`、`min_samples=3`；选择最低非噪声簇的中位数，再减去1cm。 |
| 簇内跨度与支持度 | 移除4cm整簇拒绝和额外0.25秒门槛；跨度、时间支持只记录诊断。旧阈值配置显式报错，避免静默失效。 |
| 地面估计失败 | 存档仍保留未知地面及无效local；pipeline窗口接口在选帧和截窗后，用当前窗口有效脚点10/11的平均Z减去2cm兜底。 |
| 地面有效性 | 独立mask；已有效的0米地面保持有效，不触发原代码的零值哨兵兜底。估计和兜底不标作已测量标定。 |

静止脚点仍按实际时间差计算速度，默认0.15m/s；不跨缺失点或时间空洞。重复高度去重后以 `sample_weight` 保留原密度，降低大量完全重复高度造成的邻域内存开销。排除DBSCAN的 `label=-1`，修正原方法可能把孤立低位噪声当作最低簇的行为。这些有效性处理与原代码不同，未声称逐值复现其未加mask的完整预处理。

## 窗口接口与参考系

[windows.py](src/egomotion_pipeline/windows.py) 的 `prepare_window` 和 [storage.py](src/egomotion_pipeline/storage.py) 的 `Archive.read_window` 只依赖pipeline数值模块；不导入训练代码。

1. 以30Hz存档帧号定位起点，按训练帧率选帧，并取得指定窗口；默认15Hz，支持30的正整数约数，不补帧或外推。
2. 已知地面保持原值；失败的自动估计才使用窗口内有效且有限的脚点均值兜底。无脚点、未核验世界Z轴或显式禁用地面时仍无效。
3. 默认按窗口首帧相机参考规范化yaw和XY，Z不平移；首帧缺失／轴未核验／水平投影退化时报错。调用者可用 `canonicalize=False` 读取带无效mask的未规范化窗口。
4. 从窗口world位姿重建相机地面投影、头戴高度、dense22 local及相关mask，不复用地面失败时归档的零local。
5. 在选出的帧率上计算18维头戴条件：完整头戴位姿相对地面参考9维，加地面参考增量9维；6D旋转取矩阵前两行。编码增量前把地面Z置于0，避免重复扣除地面高度。增量需相邻两端参考都有效。

返回数组和metadata均为副本，记录源帧索引、规范化矩阵及窗口地面来源，不修改原Parquet或episode metadata。当前 `read_window` 每次加载完整单段Parquet，重复采样时可缓存数组并调用 `prepare_window`。此接口尚未接入MotionExpert训练reader，不能将现有243维SMPL训练输入视为已经适配dense22。

既有相机参考约定继续使用：Aria native RGB右乘 `B`，EgoSuite OpenCV C2W右乘 `B @ B = diag(-1,-1,1)`，其中 `B=[[0,-1,0],[1,0,0],[0,0,1]]`。完整参考原点为所选相机光心，轴为左／上／光学前向；地面朝向来自参考+X的世界XY投影，local Z为世界上向。

```text
p_floor = [camera_world.x, camera_world.y, floor_height]
body_head_floor[j] = R_world_floor.T @ (body_world[j] - p_floor)
```

EgoSuite保留身体22点；HM仍按匹配SMPL-H资产解码22点。源head/CPF、身体headjoint和相机轨迹分别保留。local有效性由身体位置和相机地面参考共同决定，不依赖身体旋转局部轴验证。

## 审阅文件与检查点

| 文件 | 复查重点 |
|---|---|
| [ground.py](src/egomotion_pipeline/ground.py) | DBSCAN密度、噪声标签、移除跨度拒绝、有效零地面、按窗口有效脚点兜底及失效路径。 |
| [head_reference.py](src/egomotion_pipeline/head_reference.py) | 统一派生函数；从world重建所有依赖字段；标量mask兼容Parquet的(N,)及转换时的(N,1)。 |
| [resampling.py](src/egomotion_pipeline/resampling.py) | 继续先world插值／SLERP再派生，转换和窗口读取共享数学。 |
| [windows.py](src/egomotion_pipeline/windows.py) | 先选帧截窗再均值；首帧规范化；18维条件的旋转顺序、初帧语义和两端mask；原数组不修改。 |
| [storage.py](src/egomotion_pipeline/storage.py) | 窗口只读、索引边界、ontology、版本／实现指纹及新输出目录要求。 |
| [configs](configs)、[pyproject.toml](pyproject.toml)、[README](README.md) | 示例移除旧支持度配置，新增密度参数及scikit-learn依赖，更新0.5.0行为与接口说明。 |

复查中特别确认：窗口兜底不会改写整段地面，未选中的30Hz帧不参与15Hz均值，无效脚点的零占位不参与均值；无效相机不仅屏蔽该帧local，也屏蔽依赖它的相邻增量。原有逐关节坏帧mask保持有效。

## 测试证据

```bash
cd pipeline
/tmp/egomotion-pipeline-venv/bin/python -m pytest -q --tb=short \
  -o junit_family=legacy --junitxml=/tmp/egomotion-dbscan-review-tests.xml
```

结果：**91 passed in 5.65s**，包含既有68项回归。

- [test_dbscan_and_windows.py](tests/test_dbscan_and_windows.py)：宽于4cm的密度簇、短于0.25秒的簇、噪声与重复点密度、移除配置校验、窗口均值、15Hz跳帧、坏关节、真实零地面、缺失脚点／世界轴／禁用地面、相机缺帧与两端增量mask、越界和只读存档。
- [test_motionexpert_reference_parity.py](tests/test_motionexpert_reference_parity.py)：实际调用现有MotionExpert的规范化和表示函数，分别验证Aria／EgoSuite来源。非零地面、变化yaw/pitch/roll下，dense22最大位置误差为 **2.3841858e-7米**；选帧后18维头戴条件最大差异 **1.4156103e-7**；缺失地面均值兜底的18维条件最大差异 **2.3841858e-7**。
- 既有测试继续覆盖MCAP动态Protobuf、capture时钟精度、坏帧子集、SMPL-H FK及资产类型、视频PTS与H.264、Parquet/HF元数据、转换恢复、相机轴变换和字段验证。

数值对照只加载MotionExpert数学函数用于验证。`git diff -- MotionExpert` 为空，并与本次开始时保存的reader／representation副本逐字节核对一致。

## EgoStandard 真实数据验收

本次只使用以下EgoStandard文件，未运行EgoPro样本：

```text
/gaozt-test1/datasets/EgoSuite/EgoStandard-body/mcap/Refill Aroma Diffuser/a83cceaa-ab0c-45af-8c09-f9f3a6446677.mcap
```

| 检查 | 结果 |
|---|---|
| 转换与同配置恢复 | 1段、1,646帧，成功恢复，未新增重复段。 |
| DBSCAN | 3,235个候选，1个簇，噪声0；簇跨度0.0065161m。 |
| 地面 | -1.2169107m，valid=true、verified=false；本段无需窗口兜底。 |
| 归档dense22 local有效比例 | 99.9227%；厂商坏身体点继续由逐关节mask屏蔽。 |
| 归档验证 | `require_floor=True`、`require_head_reference=True`通过；身体旋转局部轴仍标为未核验。 |
| 15Hz窗口 | 起始存档帧0、300、1200，各80帧；头戴条件均为80×18、全部有效，首帧相机XY归零。 |
| local→窗口world重建 | 最大误差5.9604645e-8米。 |
| 窗口读取不回写 | 读取前后Parquet的SHA256一致。 |

输出及完整结果：`/tmp/egomotion-dbscan-egostandard-fv_8h9r1/`，其中 `config.json` 固定单文件输入，`result.json` 保存详细验收结果。该输出只验收运动数值处理，DINO仍为pending；本次地面估计失败分支由合成数据测试覆盖。

## 使用与验收边界

- 要获得新DBSCAN地面估计，使用新输出目录重转；旧0.4的v3存档可读，但有效旧地面不会在读取时重算。旧v2仍拒绝读取／恢复。
- 构建时无法估计地面的段，只有通过pipeline窗口接口才触发按窗口兜底；直接LeRobot逐帧读取或现有MotionExpert reader尚未使用此新接口。
- HM身体世界与MPS世界配准、Z轴、真实timesync仍待验收，示例保持 `coordinate_verified=false`。本机没有匹配SMPL-H资产及完整HM相机同步数据，未以SMPL-X替代。
- 已知相机轴转换不等于设备安装标定或真实图像投影核验。身体局部旋转轴尚未统一，dense22位置不依赖此状态。
- 均值脚点兜底沿用原训练方法，属于窗口水平地面近似；台阶、坡面及长期腾空不由单一平面完整描述。
- dense22训练reader、输入维度、关节mask损失与输出监督属于后续motion任务；本次按用户要求保持训练代码不动。

方法依据与窗口使用示例见 [README](README.md)。

## 全量EgoStandard任务启动

按用户要求，已启动全量运动数据转换到 `/gaozt-test1/datasets/MixEgoMotion`，不提取DINO。固定输入清单为6,614个完整MCAP，无`.part`、无重复文件UUID，仅包含EgoStandard。

- 任务：`egostandard-20261010T123853Z`；后台主进程PID `2430450`，8个CPU处理进程，单进程提交归档。
- 实际配置、代码副本、清单、日志及状态：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/egostandard-20261010T123853Z/`。
- 入口：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/current_job.json`。全量尚在运行，进度以该任务的 `status.json` 为准。
- 首25段的79,762帧已用固定元数据快照通过校验，报告保存为任务目录的 `startup_validation.json`；没有并发修改生产归档元数据。
- 新增 [test_batch_runner.py](tests/test_batch_runner.py) 通过：两个正常MCAP、一个缺相机异常、一个14点输入，确认异常隔离、跳过记录、单个一致归档、已提交段恢复和最终失败状态不被隐藏。
- 任务结束后自动校验全部已提交数据并写入 `validation.json`；此处不把启动成功或前25段校验当作全量完成。

## 0.6.0：统一5×1024 DINO格式

用户决定两个来源统一使用 `facebook/dinov2-with-registers-large`，每帧完整保留CLS和4个register。本次只修改pipeline，MotionExpert训练reader、网络及损失未修改；`git diff -- MotionExpert`为空。

| 项目 | 当前实现 |
|---|---|
| Token布局 | `cls, register_0, register_1, register_2, register_3`；模型输出前5个token，排除patch tokens。 |
| 存储 | 编码器B×5×1024；Parquet为float32、token-major展平5120；窗口恢复T×5×1024。 |
| 模型 | 检查 `dinov2_with_registers`、1024维、4个register、patch14、24层；冻结float32推理。 |
| 预处理 | RGB uint8；bicubic短边336、保留宽高比、无中心裁剪；ImageNet mean/std；无额外特征L2归一化。 |
| 示例revision | 官方仓库于2026-10-10返回 `e4c89a4e05589de9b3e188688a303d0f3c04d0f3`；没有下载正式权重。 |
| provenance | 记录token顺序、逻辑／展平形状、实际processor、推理dtype、库版本、revision／本地SHA256和视频指纹。 |
| 校验 | 检查Parquet物理形状／dtype／null、无效DINO零值、pending mask、valid_count和token格式，防止旧CLS列混用。 |
| 旧存档升级 | 同一写锁、先预检全部pending占位，再流式替换一列；逐段原子提交、journal恢复；保留原运动实现来源。 |

与UniEgoMotion统一的是token布局和图像预处理规则；当前Parquet使用float32，部分参考sidecar为float16，未声称跨库／精度逐值完全一致。视频仍按各来源实际PTS对齐30Hz运动网格；窗口先选帧再计算18维头戴条件，DINO随相同帧索引选择。

复查重点为 [features.py](src/egomotion_pipeline/features.py)、[schema.py](src/egomotion_pipeline/schema.py)、[resampling.py](src/egomotion_pipeline/resampling.py)、[storage.py](src/egomotion_pipeline/storage.py)、[validation.py](src/egomotion_pipeline/validation.py)、[migration.py](src/egomotion_pipeline/migration.py)、[process_dino_upgrade.py](scripts/process_dino_upgrade.py) 和CLI／配置。README整合dense22关节顺序、参考系、头戴18维、DINO布局、dtype／时间／mask、读取示例及新旧版本边界。

完整回归命令：

```bash
cd pipeline
/tmp/egomotion-pipeline-venv/bin/python -m pytest -q --tb=short \
  --junitxml=/tmp/egomotion-dino5-review-tests.xml
```

结果：**112 passed in 11.81s**。修正升级等待阶段的进度显示及补充HF读取5120维断言后，相关16项测试再次通过。新增 [test_dino_registers.py](tests/test_dino_registers.py) 和 [test_dino_migration.py](tests/test_dino_migration.py)：

- 横／竖图的336缩放与PIL bicubic及ImageNet归一化数值对照；CLS/register/patch的区分及token-major回写。
- 真实HF小型register模型CPU前向，验证输出前5个token；正式ViT-L/14由结构检查覆盖，尚未加载实际大模型权重。
- 普通DINO、错误register数量／维度／层数、CLS-only输出和非有限值显式拒绝。
- v3只读、提取前升级要求、所有非DINO列和窗口结果保持一致、升级后特征提取及恢复。
- 全段预检拦截已完成CLS或非零pending数据、与活动writer互斥、中断后的混合列宽恢复、幂等重跑。
- 后台任务在转换期间不改存档；转换失败则停止；`completed_with_errors`保留源输入失败数。

真实验收仅复制先前已验收的EgoStandard单段存档，未改写运行中的全量归档。升级结果：1段、1,646帧，DINO由1646×1024变为1646×5120且仍为pending；**其余58个Arrow列逐值完全一致**；0／300／1200起始的三个15Hz窗口，运动字段、mask和18维头戴条件均与升级前完全一致。地面与相机参考校验通过。证据：[result.json](/tmp/egomotion-dino5-egostandard-dyl3u0l5/result.json)。

已运行的全量任务仍固定0.5.0运动代码，不替换其处理中代码。后续格式升级脚本使用独立冻结副本，等待该任务转换及校验完成后执行，不启动DINO推理；任务入口为目标目录的 `.pipeline/current_dino_upgrade.json`。实际升级是否完成以该任务 `status.json` 为准。

等待任务已启动：`dino-format-20261010T130540Z`，PID `3669269`。启动复核确认状态为 `waiting_for_conversion`，生产schema仍为v3且尚未创建升级journal，冻结包与复查源码逐字节相同；同时运动任务仍在运行、输入失败0。未将排队状态报告为全量格式升级完成。

## 正式预训练权重下载与验收

按用户要求删除先前停止下载的 `/gaozt-test1/guanzerong/weights/dinov2-with-registers-large/`，重新下载到 `/gaozt-test1/datasets/MixEgoMotion/weights/dinov2-with-registers-large/e4c89a4e05589de9b3e188688a303d0f3c04d0f3/`。完整保存官方固定revision的权重、模型配置、图像processor配置及模型README；源版本与pipeline示例一致。

- `model.safetensors`：1,217,539,376字节；SHA256 `edccedab2c4e164e80833096de89a32a6e8d7365870499a066a61dbc8894b42b`，与官方仓库LFS记录一致。
- 本地 [configs/dino.local.json](configs/dino.local.json) 和归档 `.pipeline/dino.local.json` 指向固定版本目录，以 `revision="local"` 离线加载。
- 使用当前真实 `DinoEncoder`，加载正式ViT-L/14预训练权重，确认4个register、1024维、24层和patch14；本次使用4个CPU线程，没有调用GPU。
- 使用先前验收的EgoStandard `a83cceaa-ab0c-45af-8c09-f9f3a6446677` 的一个实际MCAP RGB帧：1456×1920×3，经pipeline来源预处理和DINO processor得到1×3×336×443；真实推理输出 **1×5×1024、float32、全部有限**。此前仅小模型前向的验收范围已扩展至正式大模型CPU前向。
- 各文件SHA256 manifest位于权重版本目录上一级；完整报告、单帧特征及验证脚本位于 `.pipeline/jobs/dino-download-20261010T130952Z/`。下载入口 `.pipeline/current_dino_download.json` 标为complete。

仅下载与单帧验证；没有将归档DINO mask标为有效，没有启动全量视觉提取，也没有修改运行中的冻结运动转换或格式升级代码。
