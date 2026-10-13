# MotionExpert review

## 2026-10-12：训练开放部分历史轨迹，部署仍仅图像

SMPL-H配置factory与默认YAML同步为 `TRAJECTORY_CONDITION_PROB=0.5`：训练每个样本独立抽签，开放或隐藏历史6帧的有效18维相机轨迹；未来5帧轨迹及图像始终隐藏。验证／测试／API推理继续仅使用历史6帧图像，不随训练概率变化。原K12／245维模型和联合11帧监督保持不变，没有新增teacher／student、蒸馏损失或shape分支。旧原生dense配置仍为概率0。

0.5是起始模态dropout实验值。[UniEgoMotion补充表3](https://arxiv.org/html/2508.01126v1#A3) 的纯图像预测消融中，统一模型去掉轨迹为0.322m，单独仅图像训练为0.318m，不能据此宣称混合训练必然提高纯图像部署精度。当前协议、原论文mask概率与时间范围不同；后续按纯图像val未来5帧对比0／0.25／0.5。相关依据及教师—学生的另一条研究路径见[README](README.md)。

- 全套回归 **222 passed，24 subtests passed**，见[tests.xml](exp/smplh_k12_fore_5hz_joint/trajectory_conditioning/tests.xml)。覆盖逐样本混合mask、联合loss、验证纯图像、同表示统计不随概率改变、checkpoint概率不同严格拒绝。
- 独立review **71 passed，4 subtests passed**，无剩余P1／P2；串接独立SMPL-H reader→Dataset→任务mask→实际模型条件组装，历史轨迹只影响抽中的样本。未来条件及有效性改NaN／False不改变模型输出；API的future_motion、one_step_hidden和one_step_global在轨迹及未来输入扰动后差异为0。见[independent-review.json](exp/smplh_k12_fore_5hz_joint/trajectory_conditioning/independent-review.json)。
- 实际 **399,807,365参数**模型读取EgoSuite／HM两条真实11帧窗口，控制抽签分别开放108个历史轨迹通道和0个，未来轨迹均隐藏；前向／反向、AdamW与EMA一步通过。轨迹投影梯度L2为0.150624，历史／未来输出梯度0.045193／0.084148，5组输出投影梯度均有限非零。10步纯图像采样得到未来 `[2,5,245]`，解码为12身体点和左右21手点。见[model-validation.json](exp/smplh_k12_fore_5hz_joint/trajectory_conditioning/model-validation.json)。这是CPU结构诊断，尚未验证训练收敛、精度提升、CUDA或DDP，没有保存checkpoint。

概率只改变条件mask，没有重处理归档或重算同表示统计；checkpoint训练协议严格绑定概率。正式训练尚未启动，读取到的上游状态仍为 `converted_pending_acceptance`；正式开训仍需accepted和完整train的新SMPL-H K12统计。本节替代下方K12基线的纯图像训练条件，下方几何及统计诊断继续作为表示验收证据。源码与本轮产物SHA见[validation-artifacts.json](exp/smplh_k12_fore_5hz_joint/trajectory_conditioning/validation-artifacts.json)。

## 基线：按用户要求保留K12／245维，适配SMPL-H数据（2026-10-12）

当前默认配置为 `configs/smplh_k12_global.yaml`，表示为 `smplh12_rot6d_hands21_contacts_v1`。模型保留原12稀疏身体关节 `[0,4,5,10,11,13,14,15,18,19,20,21]`，全部网络参数tensor的名称／形状与原245维模型一致。归档同模型真实beta16只参与完整body22 FK和misc审计，随后选K12；没有beta预测头、shape loss或完整22关节网络输出。显式双手21点、相机地面参考、来源脚点、5Hz历史6帧图像与联合11帧监督均保留。

新旧245表示的轴和数据来源不同，不能只按宽度识别：统计／checkpoint绑定独立表示、K12顺序、SMPL-H模型SHA／schema和时间协议。normalizer／正式入口按representation检查诊断统计与归档发布状态；统计分片要求明确全覆盖并保留joint_indices。future窗口准入和getitem只检查108维K12 body，不能因未选择的10个身体关节有效而通过。评估按未来5帧的12身体／21手点计算，标注1秒，不含beta指标。

解码只输出12身体点／全局旋转、双手、参考轨迹和接触；不补造完整SMPL-H局部pose、beta或trans。原逐帧camera basis guard继续生效，构建窗口不改变输入数组。新源FK仍保留HM逐帧真实16beta变化和完整FK审计信息，不将它们作为网络条件。

- 全套回归 **220 passed，24 subtests passed**，见 [tests.xml](exp/smplh_k12_fore_5hz_joint/tests.xml)。独立review另跑89项及4个子测试，未发现剩余P1／P2。
- 两来源真实11×245编码／稀疏解码通过，最大身体XYZ往返误差 `1.19e-7m`，旋转矩阵元素误差 `4.77e-7`；仅输出12关节，见 [real-parity.json](exp/smplh_k12_fore_5hz_joint/real-parity.json)。另核验HM跨beta分块窗口，原值跳变1.41937在FK/misc中保留。
- 四段train诊断统计覆盖363个窗口／3993个重叠帧次，向量245／18有限、K12顺序绑定、contacts保留单位尺度，明确标diagnostic。见 [diagnostic-statistics.json](exp/smplh_k12_fore_5hz_joint/diagnostic-statistics.json)。没有写入正式 `stats.pt`。
- 实际 **399,807,365参数**模型用EgoSuite／HM两条真实窗口完成前向／反向、AdamW和EMA一步，全部5组投影梯度有限非零，历史／未来输出梯度L2为0.04032／0.08334。输入 `[2,6,5,1024]` 纯图像，完整生成 `[2,11,245]`、未来 `[2,5,245]`；解码仅12身体点与左右21手点，无shape／fullpose／trans输出。见 [model-validation.json](exp/smplh_k12_fore_5hz_joint/model-validation.json)。CPU结构诊断，未保存checkpoint、未验证CUDA／DDP或预测精度。
- 真实负向门禁验证缺失／反序K12、错误model／schema、diagnostic统计及同宽旧原生245统计／checkpoint全部拒绝，见 [independent-gates.json](exp/smplh_k12_fore_5hz_joint/independent-gates.json)。

本轮上游归档仍为 `converted_pending_acceptance`；正式 `check_data` 正确拒绝，见 [formal-readiness.log](exp/smplh_k12_fore_5hz_joint/formal-readiness.log)。未启动全量训练或修改归档。正式开训仍需上游accepted和完整train的新SMPL-H K12统计；旧原生轴245统计及下方已撤回351统计均不可替代。

## 历史：已撤回的351维方案（2026-10-12）

**以下351维实现和诊断已被上方K12／245维修正取代，不再是当前默认配置，也不作为当前模型的训练／验收证据。** 保留记录便于追溯误改，默认配置及351支持已移除。

默认入口已切换独立SMPL-H参数归档和351维表示：body22各6D＋XYZ198、camera-floor增量9、显式双手126、contacts2、真实beta16。参考官方UniEgoMotion固定commit `580c92c6d70a91672c4106bab30ca82cdd80f379` 的v4_beta身体及相机参考编码；手PCA改显式点，shape10改真实16。旧245配置保留。输入仍为5Hz历史6帧DINO，联合监督历史6＋未来5；shape、身体和真实轨迹均是标签，默认不作为额外条件。

新reader独立完成30→5Hz选帧、同模型FK、原来源脚点的prefix地面兜底和接触，不导入pipeline或读取旧dense归档。使用原mixed-source分组split，隔离47条HM记录，保留原观察／完成／confidence与显式手点mask。新统计及checkpoint绑定351表示、SMPL-H模型SHA和联合时间协议；有限无效参数不升级mask，诊断统计不能用于正式训练／导出。

独立review发现并修复：正式验收split binding为路径→SHA字典；HM记录缺少目标轴marker时应根据已核对参数schema和模型定义临时轴，不改源认证；视觉时钟可能是float32名义时钟提升为float64，join必须保留其精度。另修正beta有效性、逐帧camera basis、reference列与metadata一致性、QC来源绑定、新351统计分片全覆盖和归档移动时split路径。最终独立review未发现剩余阻塞代码问题。

- 最新全套回归 **213 passed，24 subtests passed**，见 [tests.xml](exp/smplh22_fore_5hz_joint/tests.xml)。覆盖新参数reader、独立SciPy FK、真实16维shape、分组／receipt／发布状态、mask／时间join、手部显式监督、351损失与纯图像API，以及旧245兼容。
- 真实两来源编码／解码最大身体XYZ误差约 `1.2e-7 m`、局部旋转矩阵元素误差 `6.6e-7`，beta原值误差0，见 [real-parity.json](exp/smplh22_fore_5hz_joint/real-parity.json)。独立第三个HM窗口跨shape分块边界，保留beta分量跳变1.41937，不冻结HM逐帧shape。
- 363个诊断train窗口统计有限，历史与未来共3993个重叠帧次，重复delta共享统计、contacts保留单位尺度。产物为 [diagnostic-statistics.json](exp/smplh22_fore_5hz_joint/diagnostic-statistics.json)，没有生成正式 `stats.pt`。
- 实际 **399,970,287参数**模型使用EgoSuite／HM两条真实窗口做一次前反向与AdamW＋EMA更新。历史／未来输出梯度L2为0.03688／0.06562，身体、左右手、参考增量、接触和beta投影梯度均有限非零；EMA维护1060个参数tensor。纯输入 `[2,6,5,1024]` 生成完整 `[2,11,351]` 与未来 `[2,5,351]`，解码得到full22、显式hands21、beta16、localAA／root／trans。见 [model-validation.json](exp/smplh22_fore_5hz_joint/model-validation.json)。这是CPU诊断，未保存checkpoint或开始全量训练，未认证CUDA／DDP和预测精度。

正式开训尚需上游参数归档全量验收accepted，以及完整train新351／18统计。核验时归档为 `converted_pending_acceptance`；正式入口正确拒绝，诊断没有更改publisher状态或归档文件。身体旋转和XYZ仍沿用原版的冗余预测，未新增FK一致性损失；显式手腕与模型手腕保持独立并报告差值。模型定义轴和拟合标签不等于绝对旋转准确性认证。

以下保留旧245维迁移与5Hz协议的历史记录，其统计和模型结果不能用于证明新351模型已训练完成。

本次修改范围是 MotionExpert。数据读取和表示、模型/损失、推理/checkpoint 分别实施，并交叉检查；评估和训练入口另做完整集成验证。现有 pipeline 与真实归档作为只读输入。本记录区分代码验证与尚未执行的正式实验。

## 历史：5Hz历史重建与未来预测联合协议（245维）

默认配置已改为历史1秒＋当前的6帧DINO，联合重建历史6帧、预测未来1秒5帧，同一次Flow Matching采样生成11帧。真实人体姿态只作去噪目标，默认不输入真实头轨迹；未来图像不可见。训练损失覆盖完整11帧有效标签，采样整段共同更新，部署返回最后5帧。官方UniEgoMotion同样整段联合生成，20／60指10Hz时间网格上的帧数。

- 源归档、DINO和现有operator／录制split复用；每6个30Hz帧取一帧，在读取时重算5Hz增量及接触，不重新切分主数据文件。
- 失败地面仅使用观测前6帧脚点兜底。统计采用相同历史DINO有效性过滤，仅累计train分组全部11帧有效标签，绑定joint/all_valid_frames协议。
- joint与future_only、旧15Hz的统计和checkpoint严格区分；不会静默改变已存checkpoint的历史生成／监督方式。
- 纯图像API测试覆盖隐藏轨迹／未来图像NaN及有效标记扰动。世界解码独立要求最后观测参考锚点，CLI会明确拒绝无效锚点，普通245维预测保留。
- 最新全套回归182项及24个子测试通过。联合统计32分片已完成706,070个train窗口，覆盖7,766,770个时间帧次；全部有效统计有限，共享delta统计、contacts单位尺度、split及旧资产校验均通过。证据见[statistics-validation.json](exp/dense_k12_fore_5hz_joint/statistics-validation.json)。完整真实模型验收见[training-validation.json](exp/dense_k12_fore_5hz_joint/training-validation.json)：实际历史／未来输出梯度均有限且非零，optimizer及EMA各更新1步，纯6图像→5未来帧接口及10步未来采样通过。train／val／test接受窗口706,070／88,322／88,308，抽查各3个真实窗口。此处仅验证训练链路，正式预测精度仍需全量训练后评估。

以下为dense245迁移阶段的历史验证，15Hz/120帧与三任务测试不代表当前5Hz协议已经训练收敛。

## 已修正的问题

| 问题 | 修正及证据 |
|---|---|
| 旧SMPL/PCA153维不能表达显式手关节和dense旋转 | 使用245维：body108＋delta9＋hand63×2＋contact2；旧153 checkpoint/mask明确拒绝 |
| 文本条件与当前归档不匹配 | 删除文本encoder、预计算、配置与CLI，模型没有文本参数 |
| 只用XYZ会丢掉6D监督 | 12个关节每个6D＋XYZ，6D六通道参与归一化x0 MSE，采样另测角度误差 |
| 15Hz增量可能错误复用30Hz相邻运动 | 固定偶数帧后重算delta/contact；非零yaw的合成与真实窗口验证 |
| 真实缺失mask会取消随机条件dropout | 缺失与随机dropout组合；显式任务mask禁用随机dropout |
| fore未来图像有效标记仍影响attention/router | 只用实际可见图像标记，未来值/NaN/有效性扰动回归通过 |
| 将标签mask当作生成约束会清零未来 | 采样不读取标签mask，前缀输入可直接生成完整未来 |
| 地面窗口兜底使用未来脚点 | fore只用观测前缀，recon使用窗口；未来脚点扰动不影响fore条件 |
| 世界指标可能借用未来GT相机轨迹 | 累積预测delta；fore仅最后观测锚点，未来GT变换扰动不影响结果 |
| EgoSuite与SMPL关节局部轴尚未统一 | 默认仅EgoSuite原生轴，显式native策略、不伪造归档验证标记，跨来源要求有出处的轴映射 |
| 缺失手零值、旧统计影响归一化 | train-only逐维mask统计，delta共享统计，contact单位尺度；严格来源检查 |
| 只有schema升级但DINO全pending仍可启动视觉训练 | 默认有效图像比例0.5，数据预检在分配400M模型前执行 |
| fresh245训练被冻结随机投影30个epoch | fresh训练epoch0全部学习，仅兼容dense warmstart允许显式冻结 |
| EMA/optimizer恢复可能错误匹配旧参数顺序 | 校验格式、phase、参数顺序/数量和状态shape，不兼容在修改状态前拒绝 |
| BF16完整权重采样噪声仍为FP32 | 按模型dtype生成noise，增加BF16回归 |
| 大Parquet重复读取整套图像 | 列投影、窗口row groups、每worker有字节上限LRU；统计不读DINO embedding，当前5Hz另读有效性布尔列 |

## 验证记录

- 合成归档测试覆盖非零世界原点/yaw/地面、身体XYZ和rotation还原、双手显式点、fore历史锚点、独立mask、15Hz、窗口过滤、train统计和operator/recording split。
- 推理、导出、checkpoint、EMA、task/dropout与采样指标回归通过；旧表示和统计来源不一致会失败。
- 全套自动测试54项及4个子测试通过；8个公开CLI入口均可正常加载，`git diff --check -- MotionExpert`通过。
- 完整模型399,807,365参数、12层、latent768：实际CPU B1/T4前向输出motion[1,4,245]、hidden[1,4,768]、global[1,768]；CFG、Flow＋MoE损失和反向梯度均有限，新输入/输出/轨迹/视觉投影梯度非零。
- 完整训练CLI一train一val：global_step=1，motion[1,4,245]、DINO[1,4,5,1024]，实际Adam被EMAOptimizer包装，EMA含1060张量，各监督组有效计数非零，train/val损失有限。
- 周期验证实际执行完整主干的recon/fore/gen三任务10步采样，并产出世界姿态、参考轨迹、旋转、接触指标；将有效计数转换为浮点日志，避免Lightning自动转换提示。
- 真实EgoStandard部分手有效的3个120帧运动窗口只读验证：偶数帧15Hz，native旋转有效，归档跨来源轴验证标记保持false；身体/手位置最大还原误差约6.3e-7m，旋转矩阵元素约1.6e-6。临时train统计245/18通过，未读取DINO列。

执行命令与完整实验计划见 [README.md](README.md)、[validation_plan.md](validation_plan.md)。这些结果验证接口和几何正确性，不代表全量训练收敛。

## 仍需数据或实验解决的事项

1. 当前EgoStandard归档DINO提取已完成；训练仍逐窗口检查历史6帧视觉有效性。
2. 新5Hz联合协议尚无正式训练权重；旧153/text、15Hz和future_only资产不能直接作为其训练恢复状态。全量训练尚未执行。
3. 当前支持EgoSuite原生关节轴；混入EgoExo-HM前需验证SMPL到目标关节轴的映射，手点顺序也须核验。
4. 归档地面是离线参考；纯视频严格在线部署须给定地面来源。缺失参考链不做静默累计，评价报告覆盖率。
5. 脚接触继续原MSE基线，6D继续欧氏MSE基线；实际权重选择、旋转退化、手部一致性与物理表现以held-out采样验证判断。
