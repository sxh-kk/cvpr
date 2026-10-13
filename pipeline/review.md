# EgoMotion Pipeline Review

- 日期：2026-10-11。
- 范围：主归档与既有MotionExpert接口保留；本轮只修改pipeline，新增EgoSuite完整序列SMPL-H候选拟合、联合验证reader及独立审计。此前双来源小样本研究与native sidecar结果作为历史证据保留。
- 当前主schema：`egomotion-dense22-hands21-camera-floor-dino5-v5`；历史 `egosuite-native-dense22-rot6d-xyz-v1` 派生产物继续保留，v4回退快照已在独立复验后删除。
- 源码review通过：主Parquet新增16个dense22身体／双手floor列及独立mask，旧运动／相机／DINO列逐值保留。完整pipeline独立复验 **333 passed，2 warnings，30.72s**，XML已归档至正式任务的 [pipeline-tests.xml](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/pipeline-tests.xml)；其中迁移与DINO回归25项通过，含目录rename及部分metadata发布后的恢复。生产全量完成依据为 `.pipeline/dense_v5_upgrade.json` 的complete与正式验收报告。
- 认证边界：原生轴仍未认证；手腕偏差仅报告，不强制粘合；有效插值保持可监督。真实HM配准与跨来源关节轴／中心统一仍待独立解决。

## EgoSuite完整序列SMPL-H候选（本轮）

正式v5、既有MotionExpert源码／split／stats未切换。独立任务为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/`，与v1失败结果分开保存。模型为与HM匹配的male SMPL-H、真实16beta；统一目标body22的XYZ和全局6D来自同一次FK，native左右手21点、相机／地面／DINO与原帧关联。手指pose没有拟合为SMPL-H手指参数。

共享中心先验只使用20位train operator且人物等权，60位完整序列验证人物与其不相交；train／val／test分别30／15／15位。来源split、operator、源录制标识泄漏检查通过；同operator有10位出现录制间骨长签名差异，以recording-specific校准隔离。每组shape与center offset在序列拟合期间冻结。低覆盖采用显式先验估计状态，模型可用mask不冒充原始观测或认证精度。

v1完整154,473原30Hz帧独立FK／来源／train先验审计通过，但body mean1.192cm和连续local>45°边比例0.642%未通过预登记1cm／0.1%门槛，全量未启动。历史 [expanded-acceptance.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v1-20261011T123135Z/reports/expanded-acceptance.json) 保留。缺失关节重新出现时，逐帧几何初始化会切换旋转分支；接触速度代理又会误将仍在移动的脚点强压为静止。修订仅使用train控制实验，未放宽门槛或用质量mask排除原观测。

v2初始化用源世界旋转增量／父关节运输、弱几何修正及同跟踪段内局部SO(3)桥接；源绝对局部轴不参与初始化。时序项改用旋转对数平方，默认250次迭代；接触项改为观测脚点水平运动匹配，权重1e-4，仍保留源脚点／mask及接触代理语义，不压脚Z为零。shape1e-4与hinge5e-4保持原值：train控制中弱shape虽改善骨长，body XYZ反而恶化。修订依据见 [sequence-v2-controls.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/sequence-v2-controls.json)。

提交锁由非阻塞改为阻塞；真实多进程测试覆盖竞争提交、stage异常与SIGKILL后无半提交／合法续跑。plan复算自身/config/selection签名；冻结19个producer／依赖／审计与接受脚本，worker加载前后、任务前后核对代码和model身份。worker和callback失败立即记录，成功结果继续提交，终态为failed且不宣称complete。原研究module `smplh_retarget.py` 与 `smpl_fit.py` SHA保持原值。

联合reader显式选择拟合标签，逐帧原来源身份校验；15Hz使用30Hz偶数相位，随后构建原18维相机地面轨迹。地面缺失时只用原生脚点，fore只用观察前缀，阻止第二次由拟合脚点兜底。位置／旋转补全与原插值分别标记，嵌套目标basis metadata明确为模型估计。该reader会核完整序列，当前用于验证，未宣称训练吞吐已完成。

完整pipeline回归 **457 passed，2个既有JUnit告警，45.83s**；正式 [pipeline-tests.xml](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/pipeline-tests.xml)。含独立NumPy FK和被重新签名的6D／root／floor／hand／beta／offset损坏反例、多进程提交、缺失／reset、native轴常数右乘不变性、近π时序梯度，以及联合reader／地面／视觉逐帧关联。

独立 [runner-independent-review.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/runner-independent-review.json) 与 [independent-code-review.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/independent-code-review.json) 通过。真实episode29／3909的80帧15Hz联合窗口为198／18／5120，左右手、相机、DINO、18D及mask与source窗口逐值一致，见 [real-reader-smoke.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/real-reader-smoke.json)。

全量入口不直接信任runner的布尔gate；job内lifecycle wrapper启动前重新接受并绑定当前job/policy/audit/registry与精确60段。独立11项过期报告／缺失重复人口／失败不启动／最终全6614接受测试通过，见 [full-entry-review.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/full-entry-review.json)。19个生产冻结文件在expanded执行中保持不变。

v2完整60段／154,473帧独立审计与原门槛**通过**，见 [expanded-acceptance.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/expanded-acceptance.json)、[sidecar-audit.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/sidecar-audit.json)。所有原XYZ观测2,939,012点mean0.714cm／P951.875cm／max18.266cm，未用质量mask筛除失败点；native手腕与body腕306,188点mean0.144cm／P950.372cm／max0.859cm。连续local>45°由21,798降至236／3,395,480（0.00695%），仍87个>90°、max163.51°，主要集中于低有效率和长序列；旋转不认证为真值。独立52关节NumPy FK XYZ最大差3.785e-6m，floor逆世界最大差1.931e-6m；20train prior actual32,274帧、等人物mean一致且无留出人物泄漏。

源v5的13,231文件身份及57个元数据／MotionExpert内容摘要均未变。train-only诊断30段／70,650帧见 [train-diagnosis.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/train-diagnosis.json)：无beta／offset边界饱和，已记录的最后240→250步目标下降小；部分下肢补全和启发式hinge仍有不确定性，不用这些诊断调val/test或宣称独立解剖精度。

2026-10-11 14:07 UTC已启动64进程全量候选：6,614段／26,841,893原30Hz帧；先校准1,829个operator，之后完整序列拟合。产物 `/gaozt-test1/datasets/MixEgoMotion/derived/smplh-body22-sequence-v2/egosuite/`，`logs/full.log` 为进度、`logs/full-lifecycle.log` 为阶段。wrapper启动时重新执行当前expanded接受，job／policy／audit绑定通过；后台进程信息为 `full-process.json`，启动汇总为 [start-summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-sequence-v2-20261011T133348Z/reports/start-summary.json)。全量仍在运行，完成后自动对全部源episode重新独立审计与 `--scope full` 接受。该门槛不直接发布主归档，不认证绝对旋转或运动真值；HM相机帧对应问题仍须独立处理。

## 最新SMPL-H双来源研究验收

2026-10-11已完成EgoSuite共同rig重定向55次拟合及3个真实HM take的body／camera研究。**模型解码与小样本重建数值检查通过，跨来源生产条件尚未满足。** 主结果见 [report.md](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/report.md) 和 [ego-report.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/ego-report.json)。所有输入／结果置于独立`.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/`；没有修改现有v5运动、DINO、split／stats或MotionExpert，没有把研究输出接入生产读者。

新增 [smplh_retarget.py](src/egomotion_pipeline/smplh_retarget.py) 与 [validate_smplh_dual_source_pilot.py](scripts/validate_smplh_dual_source_pilot.py)。研究模块冻结SHA256为`c420adaaf09dffaa55fa079341300b12ad3a9e8af4a4af6c3d8854098c534b28`。真实资产为 `/gaozt-test1/datasets/MixEgoMotion/weights/smplh/male/model.npz`，SHA256`be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947`：6890顶点、52个regressed/tree关节、**真实16个shape basis**。严格校验完整body＋双手父子链，安全数值NPZ快照；测试fixture有显式测试标记，不能通过生产布局认证。匹配资产的官方前向数值一致性与原生关节轴认证分别记录。

中心观测模型为`p_source = p_SMPLH_FK + G_parent @ d`。默认仅neck、左右collar、head的固定局部d可估计，12cm长度限制、L2正则0.01；canonical XYZ始终保持目标FK中心，source landmark重建独立输出。pose由原XYZ＋源世界旋转增量拟合；校准期间beta冻结，第二窗口offset和beta均冻结，不允许用后续观测更新校准。相邻边必须同时满足有效端点、时间间隔和tracking-reset mask；拟合保留模型估计语义，source missing不能变成认证标签。

同原SMPL-X研究使用10人、每人两个互不重叠120帧15Hz窗口，共2400独立输入帧。主实验仅5位train人物第一窗口估计d，再按人物等权汇总；其余5位不参与source映射校准。零16维beta只用于隔离中心策略。两模式以相同纯XYZ初始化拟合第二窗口pose。另以每个人第一窗口的肢体骨长估计16beta、校准subject d，第二窗口冻结二者作同人消融；不与跨人source共享结论混合。

| 比较范围 | 观测重建mean／P95 | 世界旋转增量mean／P95／max |
|---|---:|---:|
| 5人留出，零shape baseline | 1.061／2.364cm | 0.214／0.770／4.284° |
| 同5人，固定source共享d | 0.794／2.038cm | 0.206／0.718／4.180° |
| 10人，同人肢体shape baseline | 0.830／2.114cm | 0.191／0.689／4.312° |
| 同10人，固定subject shape＋d | 0.486／1.420cm | 0.182／0.665／4.154° |

第一比较12896有效XYZ／12714旋转边；同人比较26096有效XYZ／25804旋转边。这里的pose直接使用第二窗口观测拟合，数值是source landmark重建残差，**不是独立canonical中心精度、绝对旋转精度或预测精度**。5人共享d的neck／左collar／右collar／head长度4.509／2.601／2.175／2.909cm；对应跨人scatter RMS 1.352／0.855／0.883／1.077cm，固定source映射仍存在个体／shape／初始化耦合。15组校准均没有12cm饱和；subject beta绝对最大2.164，无±5饱和。目标轴定义明确，但无学习姿态先验或独立绝对twist参考，全部`canonical_rotation_certified=false`，原生轴认证仍false。

独立 [retarget-quality.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/retarget-quality.json) 重新解码40个结果NPZ，用独立NumPy FK重建目标中心／旋转，核对输出SHA、非零rest-root的SMPL transl换算以及13组source／mask／hands21／18D／帧时间的原dtype逐值保留。65个已读输入文件在诊断前后SHA一致；float32 XYZ导出最大差5.96×10⁻⁸m。骨长核对：源neck→head 8.807cm，H male零beta10.273cm，X neutral零beta16.410cm；源spine3→neck22.500cm，H21.468cm，X16.861cm。H与本批源几何更近，但模型family与gender／资产同时变化，且骨长仍不完全相同；历史X拟合shape后的15.87cm不与零beta口径混用。

身体与原hand21 wrist的平均／P95偏差：留出source共享模式0.545／0.724cm，同人shape＋offset模式0.095／0.197cm；源两者原先重合，输出不强制粘合或修改手部观测。源静止脚代理使用有效连续脚点3D速度<0.15m/s；留出水平速度mean 0.025679→0.025817m/s、P95 0.087694→0.088055m/s。脚点相对存档地面穿透最大值源2.606→共享d 2.719cm；同人模式2.611cm，P95均0。原始数据本身存在少量此代理偏差；该诊断没有真实接触标注或足底mesh，不能宣称没有滑步／穿地或已通过物理接触验收。

[HM body oracle](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/hm-body/run-v2/reports/hm-body/summary.json) 完成`cmu_bike02_4`／`sfu_cooking026_8`／`uniandes_dance_020_37`共1391原30Hz帧；36个实际帧与官方完整SMPL-H前向核对通过，最大body XYZ差5.45×10⁻⁸m。真实16beta、全chunk拼接边界和源手部21点保留通过，没有以SMPL-X模板／betas解码HM身体。HM `valid`为chunk有限性，没有额外逐关节置信真值。

[HM camera报告](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/hm-camera/run-v2/summary.json) 的真实RGB光心、MPS世界配准／重力及投影数值候选通过，但生产`prepare-aria`严格长度接受失败：HM长度109／872／410对应timesync110／873／411，同步序列均多一帧。官方源码`timestamps[:num_frames]`支持取HM长度前缀的研究候选，没有完整发布source-frame映射，不能直接扩展为全take截尾规则。公开2D为公开3D的投影；已生成并核对9张叠图中的部分可见手点，仍缺独立图像标注与源帧映射认证；`camera_0`重复exo，不能作ego相机；CPF、body head和RGB光心各有语义。[相机身体投影补充研究](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/hm-camera/supplement-summary.json) 通过也不升级生产接受状态。

研究module新15项、既有拟合19项与body model资产类型6项合计**40 passed，0 failures／errors／skips**；真实SMPL-H非零16beta FK、完整layout/资产替换、offset冻结和分组隔离、mask／gap／reset、90°twist不可辨识反例均覆盖，正式XML见 [research-tests.xml](/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smplh-dual-source-pilot-20261011T113010Z/reports/research-tests.xml)。生产仍需解决HM帧对应政策与图像验证、广泛动作／长期缺失／重置／接缝、中心映射与手腕契约、估计标签置信／mask及训练reader，不能只改`rotation_basis_verified`标记。

## 当前v5全量复验与v4备份清理

最新正式汇总见 [production_summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/production_summary.json)。[validation.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/validation.json) 对当前主归档重新独立解码全部6,614段／26,841,893帧／75列，包括全部DINO值；逐列数值摘要、mask／插值、几何与来源元数据均通过，failed episodes为0，耗时429.16秒。验证物理Parquet 565,873,546,076字节，解码数值661,840,555,701字节；完整／全部live／passed三个标记均true。不读取backup，不复用历史逐段验收结果。11段未知地面及全部原生旋转轴未认证保持原声明，不将允许缺失误写为标签完整。

[motionexpert-validation.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/motionexpert-validation.json) 的当前读取与训练资产验收通过，耗时148.72秒：train/val/test接口245／18／5×1024、forecast屏蔽、统计与split指纹一致，6,614个实际Parquet路径均指向当前主data，无v4备份运行时依赖。该步骤只读，未执行本次训练或模型前向／反向；首次发布的399,807,365参数smoke作为历史证据保留，源码身份与其范围在新报告中明确记录。

独立故障回归 [test_revalidation.py](tests/test_revalidation.py) **23 passed，0 failures／errors／skips**；正式XML为 [revalidation-tests.xml](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/revalidation-tests.xml)。覆盖有限DINO值逐列hash篡改、mask／插值、身体／手部／头戴几何、时间／帧身份、inventory／receipt／源元数据错误、历史验收不复用，以及删除临时fixture备份后Archive／read_window／新validator仍通过。

[cleanup-result.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/full-v5-revalidation-20261011T090604Z/cleanup-result.json) 为passed，实际删除仅 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/migrations/dense-v5-full-20261011T080945Z/backup/`。完整数值验收、23项回归、validator与26个MotionExpert源码SHA检查均通过；删除在独占锁内核对训练资产、备份目录／文件身份及来源派生链后执行。共13,233个文件，删除前逻辑498,734,566,282字节（原data 498,649,658,599字节），分配498,785,428,992字节。删除后v5文件／episode记录身份不变，实际15Hz窗口读取一致。backup已不存在，不能再从该快照回退v4；journal及migration status保持 `state=phase=complete` 并写入 `rollback_available=false`。主数据、native sidecar、DINO权重、split／stats、receipt／inventory及历史验收报告均保留。

## 首次v5全量发布与统计验收

2026-10-11已完成6,614段／26,841,893帧的v4→v5主归档迁移，主路径 `/gaozt-test1/datasets/MixEgoMotion` 不变；任务 `unified-v5-20261011T080945Z` 的status和归档journal均complete。16进程迁移约1,189.67秒，主Parquet为565,873,546,076字节。

首次发布汇总 [production_summary.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/production_summary.json) 包含四项passed验收、报告摘要／SHA、测试、训练资产及当时保留的回退快照。历史报告不改写；快照当前已按上述清理记录删除。

正式报告目录 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/`：

- `publication-validation.json`：passed；所有旧数值列精确保留、DINO来源provenance精确保留、发布后的schema／receipt／文件身份一致，native表示最大差0。各episode在发布前完成独立流式数值摘要与几何检查，发布后校验库存和身份，不将后者误称第二遍全数值复验。
- `statistics-validation.json`：passed；train/val/test为5,323／647／644段，原30Hz帧数21,473,273／2,684,309／2,684,311；operator与原录制分组互不相交。15Hz、120帧训练窗口共360,208个，16个连续分片合并为245＋18维统计，各通道均有有效观测、全部统计有限、非有限观测0，共享增量统计一致、contact identity归一化。通道计数包含重叠窗口。
- `status.json`：reader_checks包含train、val、test；本次数据与统计流程完成。
- [training-smoke.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/training-smoke.json)：passed；真实v5归档及新stats驱动399,807,365参数完整主干，B1／T120运动输入，train/val/test接口均为245／18／5×1024。身体旋转、身体位置、全局旋转／平移、左右手与contact各组loss有限；各组输入／输出通道梯度非零，各条件投影梯度非零且有限，700个有梯度的参数张量全部有限。fore未来相机／视觉屏蔽、prefix排除loss检查通过。未执行optimizer step、未写checkpoint或归档运动数据，不代表已完成模型训练。
- [boundary-validation.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/boundary-validation.json)：passed；真实13段、17个窗口、646项检查，覆盖地面失败、窗口／预测前缀兜底、逐身体／手部mask和零填充、世界逆变换、15Hz增量、相机参考统一及只读性。仅投影运动／mask列，DINO数值读取0；原Parquet及episode JSON身份不变。

6,614段DINO全部complete，有效26,539,780帧，无效302,113帧由原mask保留；地面失败9,403帧仍不在整段兜底。`rotation_basis_verified=true` 的episode数为0，完整编码验收不升级原生轴认证。

真实地面失败共11段，其中10段的双脚始终无效，无法靠当前脚点均值获得floor／local监督。episode894仅原30Hz帧956具有两个有效脚点，该帧保留在15Hz偶数帧相位中：

| 原30Hz起点 | 脚点在15Hz窗口的零起点索引 | recon | fore |
|---|---:|---|---|
| 840 | 58，未来区 | 完整窗口脚点可兜底 | 观测前缀无脚点，保持无效；Dataset拒绝该预测窗口 |
| 900 | 28，观测前缀 | 可兜底 | 仅前缀脚点即可兜底，未来条件仍屏蔽 |

该自然兜底仅依赖一个时刻的两脚观测，地面 `verified=false`；窗口可兜底不代表存档地面已恢复或全段训练标签完整。选中真实样本的身体旋转与XYZ源mask一致，分别传播已检查；自然出现旋转／XYZ有效性不同的情况由独立合成回归覆盖。缺失手点／身体点继续按各自mask排除，不强制与身体wrist合并。DINO与归一化模型路径由独立reader／完整模型smoke覆盖，边界审计不读取DINO值。

边界审计最坏数值误差：身体逆世界 `1.550e-6m`、左／右手 `1.580e-6m`／`1.207e-6m`，身体旋转矩阵元素 `1.803e-6`，相机高度 `1.192e-7m`，15Hz增量 `2.968e-8`。这些衡量表示运算与逆变换，不是运动标注精度。episode3的每手63个缺失joint-label来自3个所选15Hz帧×21个关节点；骨架始终为每手21点，位置特征为63维。

首次发布时，v4快照保存原data/config/meta/episode记录，其中原Parquet为498,649,658,599字节；该快照现已删除。历史native sidecar继续保留。当前模型侧 [splits.json](../MotionExpert/exp/dense_k12/splits.json) 与 [stats.pt](../MotionExpert/exp/dense_k12/stats.pt) 为新v5资产，统计SHA256为 `54457b581a63daf978fee99e70436ace5ad5c634fca0c26ebac59603816279a1`。

验收脚本 [audit-publication.py](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/audit-publication.py) 和 [validate-full-model.py](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/validate-full-model.py) 与pipeline测试XML一起存于正式任务目录。[validation-artifacts.json](/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/unified-v5-20261011T080945Z/validation-artifacts.json) 固定其SHA256及字节数；本次逐文件重算核对全部一致。

## v5 主归档及训练接口审阅

实现 [unified_dense22.py](src/egomotion_pipeline/unified_dense22.py) 将已有global wxyz规范化为矩阵前两行6D，与原XYZ按joint-major拼接为世界／相机地面参考198维。双手各63维floor XYZ与身体共享参考；旋转、位置和插值mask分别保存。世界列为训练读取的权威表示，floor列为缓存：窗口首帧yaw／XY规范化或地面解析后全部重新编码。旧世界XYZ／wxyz列物理保留以供来源核对，编码不新增关节轴校正或SMPL拟合。

[unified_migration.py](src/egomotion_pipeline/unified_migration.py) 在独占写锁内先stage全部文件，独立解码磁盘Parquet，校验每个旧／新列的dtype、shape、逐列数值摘要及几何回环。5120维DINO tokens、有效mask、PTS和误差包含在保留校验中，encoder签名与complete状态原样保留；不重新打开视频或模型。可选历史native输出进行身体数值／mask交叉对照。

提交前锁定source inventory与receipt摘要；提交保留原v4 data/config/meta/episodes，再替换主data目录、逐段记录及全局meta，journal最后标complete。journal非complete时pipeline／MotionExpert读者拒绝访问；旧热对象检查当前schema，需迁移后重新构建。恢复优先检查committing phase，复用验证过的stage与v4备份，不将已重命名的v5文件误当v4输入。普通stage恢复用文件size/mtime/inode与provenance核对已验收receipt，不重复解码已经通过的文件；该快速检查不是对保持这些属性不变的外部故意篡改的重新全量审计。

`motion_representation`逐段记录schema、producer SHA256、motion数值SHA256；新split与统计指纹包含此运动身份，DINO列与其状态不进入motion摘要。MotionExpert优先读world198、按K12取12×9，重新生成任务相关floor与18维头戴条件，保留双手63＋63、全局地面参考9和脚接触2，共245维。`source_native`仅允许EgoSuite；仍不允许将未认证原生绝对旋转与HM直接混训。旧split/stat/checkpoint禁止静默复用。

缺失地面时主归档world标签继续保留，floor及hand-floor无效；迁移不在整段上补地面。固定存档地面作为离线环境标签，未知地面在窗口选成15Hz后可兜底；预测任务只用观测前缀有效脚点。未来条件通过task mask屏蔽，隐藏未来DINO值及可用性不参与router／注意力池化；标签mask用于损失，不清空未来生成状态。

独立迁移回归见 [test_unified_migration.py](tests/test_unified_migration.py)，使用 [unified_v4_fixture.py](tests/unified_v4_fixture.py) 的变化旋转、非零DINO、逐关节缺失／插值、非重合手腕与失败地面，验证：旧Arrow列逐值保留及原v4备份、跨row-group摘要、pilot不改主数据及gate、写锁互斥、worker中断恢复、两个真实data rename后的恢复、config先于meta发布中断恢复、receipt／source／stage变化拒绝、源帧身份拒绝、多进程提交、native交叉对照。迁移15项＋历史DINO10项为 **25 passed**。

下列DBSCAN、窗口接口和真实样本等记录为历史阶段验收；当前reader接口及新主schema按本节与README为准。

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

返回数组和metadata均为副本，记录源帧索引、规范化矩阵及窗口地面来源，不修改原Parquet或episode metadata。`read_window` 每次加载完整单段Parquet，重复采样时可缓存数组并调用 `prepare_window`。该历史窗口验收阶段尚未修改MotionExpert；当前训练reader已适配v5主归档，见上述整合审阅。

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
- 构建时无法估计地面的段，直接LeRobot逐帧读取不兜底；pipeline窗口接口及当前MotionExpert任务窗口reader分别在选帧后按各自窗口／预测前缀解析地面。
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

## 八卡DINO调度与真实GPU验收

增加 `extract-features --gpus 0 1 2 3 4 5 6 7` 和 [process_dino.py](scripts/process_dino.py)。每张GPU对应一个spawn worker，动态处理完整episode；有界pipe返回计算结果，主进程统一提交Parquet、provenance和共享元数据。DINO签名排除GPU分工，保持原模型、预处理、batch配置和软件版本的校验，可更换卡数续跑。异常或停止时回收worker、保留已提交段并刷新元数据；运动训练代码未修改。

- 完整回归：122项通过，16.69秒。随后补充的两项后台runner验收通过，覆盖乱序进度、最终校验及升级失败时停止。多进程测试还覆盖单／多进程逐值一致、运动列保留、更换卡数恢复、签名冲突预检、不同encoder拒绝、worker异常和进程直接退出后的清理。
- 独立GPU环境为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/dino-torch260-cu118/`，Torch2.6.0+cu118、Transformers4.49.0；原CPU环境及冻结运动／升级任务继续使用各自代码和环境。
- 八张A100均使用真实EgoStandard图像和正式固定版本权重完成32帧推理，输出32×5×1024、float32、全部有限，每卡峰值CUDA分配约2.48GiB。GPU2现有服务保留，测试后显存恢复至其原有占用。
- 进一步复制前8段各64帧到独立验收存档，重排全局index，运动列直接复制原数据。真实八卡并行提取并提交512帧，全部DINO mask有效；提交顺序为7、6、0、5、4、1、3、2，读取和完整校验通过，提取前后非视觉字段逐值一致。耗时20.93秒，包含模型初始化；该数值为小样本验收时间。
- 真实验收报告位于 `.pipeline/jobs/dino-extract-8gpu-20261010T141855Z/real-8gpu-smoke/verification.json`，逐卡推理报告为该任务的 `gpu-smoke-0.json` 至 `gpu-smoke-7.json`，冻结源码及其SHA256保存在任务目录。

生产运动转换已完成6614段、26,841,893帧，输入失败0，运动校验通过。单卡等待任务已取消，原八卡等待任务PID149344也已停止，生产DINO提交数为0；新优化任务见下节。512帧验收存档与生产归档分别存放；全量完成状态以生产任务 `completed` 和 `validation_passed=true` 为准。

## 0.7.0：八卡提取优化及生产启动

本次仅修改pipeline视觉提取、调度、视频解码、Parquet写入及配置／文档。模型、dense22、相机／地面参考系、30Hz时间轴、18维头戴条件和MotionExpert均保持原定义。正式配置使用TF32、SDPA、每卡8线程PIL预处理、2批CPU预取、4线程解码、固定内存非阻塞传输、2线程Torch／BLAS／Arrow，以及DINO浮点列byte-stream-split／Zstd编码。所有优化参数和Pillow版本进入encoder provenance和续跑签名。

复查了预取队列的有界缓存、异常传播和提前关闭，多进程单写者与签名预检，以及原PTS／坏帧／时间阈值规则。真实EgoStandard首段在多线程解码后提前关闭时，曾卡在解码器释放；关闭前排空已提交packet后，同一32帧样本0.27秒完成读取。新增回归检查确保该关闭路径只排空一次。PyAV尾帧缺失`time_base`时使用已设置的packet纳秒time base，保持捕获时钟。

完整测试：**148项通过，0失败，0错误，23.18秒**；覆盖并行预处理与HF逐值一致、预取顺序／异常／关闭、真实H.264 MP4／MCAP不同解码线程的像素和PTS一致、Parquet编码及数值无损、单／多进程结果及恢复冲突。报告为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-opt-review-20261010T145320Z/tests-production.xml`。

真实八卡测试使用正式固定权重和EgoStandard，输入与原FP32基线完全相同：

| 配置 | 目标帧/秒 | 对照结果 |
|---|---:|---|
| 原FP32路径 | 111.47 | 基线；12,288个目标帧，11,630帧有效。 |
| TF32、每卡4线程PIL | 523.41 | 最大段级相对L2 0.1501%，最低token余弦0.999410。 |
| BF16、每卡4线程PIL | 519.60 | 最大段级相对L2 1.9229%，最低token余弦0.994119；本次未采用。 |
| 最终TF32、每卡8线程PIL | **551.85** | **4.95倍**基线速度；运动列、视频PTS及mask逐值一致，完整归档校验通过。 |

最终测试的解码／预处理／推理／提交共22.27秒，另有约11秒模型初始化；DINO输出仍为float32、5×1024，特征误差与4线程TF32测试相同。Parquet由250,275,692降至217,364,478字节，约减少13.15%。另以最终代码处理生产首8段各64帧，512帧全部有效，运动列、PTS及mask逐值一致，完整校验通过；该短样本包含启动／尾批开销，用作正确性验收。

编译模型测试：冷编译约22秒，缓存后的单批前向及传输0.2447秒，对照0.2544秒，仅约4%收益；当前主要成本还包括CPU处理及写入，未启用编译。延后RGB转换路径实测496.99帧/秒，比相同4线程TF32路径更慢，已移除。生产选择TF32以保留较低的数值偏差，不将输出dtype为float32误称为FP32逐值等价。

最终冻结代码和配置位于 `.pipeline/jobs/dino-extract-opt-8gpu-20261010T151033Z/`，源码及配置SHA256启动时全部核验。性能与首批验收报告分别为review任务下 `tf32-workers8-decode4-batch32-final/benchmark.json` 和 `tf32-workers8-decode4-batch32-tiny-final/benchmark.json`。

生产任务于2026-10-10 15:13:53 UTC启动，PID **1309285**。已核对实际命令、独占提取等待锁、冻结源码SHA256及目标目录配置；入口为 `.pipeline/current_dino_extraction.json`，当前为`waiting_for_format_upgrade`。格式升级任务继续使用其原冻结代码和CPU环境，生产提取在升级最终校验通过前不写入归档。样本外推纯提取约13.51小时，从当前阶段到最终校验预估15–18小时；共享GPU负载及存储吞吐会影响实际耗时。README提供当前任务监控命令，完整生产验收仍以最终状态及报告为准。

## EgoSuite dense22 关节轴标定工具及真实验收

**本次代码验收通过；全部 22 关节的原生轴认证未通过，0/22 verified。** 新增右乘基变换拟合、SO(3) 数值验收、严格已审核 artifact loader、独立 MCAP 几何审计及测试。只在独立研究目录生成候选和证据，未修改生产 episode 的旋转、basis correction 或审核标记，也未编辑 MotionExpert 训练文件。此结论不把其他工作在共享目录中的修改归入本次标定。

按用户要求联网搜索并直接下载 26 项官方资源，总计 6,049,547 字节，包含固定 commit `c97e420a28ad727c63881c6bb0f04ba038abb3b0` 的 DevKit 源码、身体／头部／双手文档及数据目录证据。逐文件 URL／SHA256 位于 `.pipeline/axis-research/official-downloads/manifest.json`，根任务已重新核对全部下载 SHA。在已检查的公开资料、2,261 个 body task 目录及 MCAP schema／metadata 中未找到完整 22 点 bind pose／关节局部轴定义；搜索范围和访问限制单独记录，未将通用 SMPL 资产当作 EgoSuite 标定。

真实 EgoStandard 审计：固定 seed 20261010，24 位不同人物／任务，12 人拟合、12 人留出，23,622 个身体帧。按 capture timestamp 的整数纳秒做 2ms 匹配，检查姿态有效性及供应方坏帧；不解码 RGB，不调用 GPU。目标采用本机 UniEgoMotion 的完整 `SMPLX_NEUTRAL.npz`，SHA256 `43d8f3a1375d7c5baae207870a5d51def0f7e6b507df709b4937598b5e7d965d`，校验 55 点结构、前 22 点拓扑和四个 MCP 根，明确不用于 HM 的 SMPL-H 解码。

| 检查 | 实际结论 |
|---|---|
| 完整几何自由度 | pelvis／spine3／双 wrist 共 4 点，在锚点与目标静止骨架对应假设下可拟合 3 DOF |
| 单骨骼方向 | 15 点仅 2 DOF，绕骨轴的 twist 未知；多帧／多人物不构成独立 twist 证据 |
| 无方向约束 | head／双 foot 候选为 null，0 DOF |
| 5° 方向数值门槛 | 17 点通过；pelvis／spine3 留出 P95 7.26°／7.08°，双 wrist 3.49°／3.11° |
| 共享轴置换假设 | 留出方向 RMSE 9.92°、P95 13.75°、最大 32.35°；源／目标形态不同，不能仅据此判定真实原生轴对错 |
| 身体与手部 wrist | 独立 8 人审计中稳定约 90° 轴偏移，不直接替换四元数 |
| 正式轴验收 | 22 点均 false；方向误差不冒充 SO(3) 误差，未认证参考不升级为已验证 |

独立 review 找到并修复两项问题：成对 NPZ 原先读取后才按路径算 SHA，拟合期间替换文件可造成数据与证据哈希不一致；现从同一字节快照读取／计算 SHA，拟合后路径变化则拒绝，目标模型和 catalogue 采用同样检查。严格 loader 原先未检查 convention，现强制 `target_world_rotation = source_world_rotation @ correction` 并保留到 provenance，拒绝缺失或左乘声明。MCAP 还检查 inode／device／ctime／size／mtime，模型／catalogue 使用快照及身份核验。

最终加固实现以独立冻结代码重跑，17.636 秒完成，结果与原审计一致。执行脚本 SHA256 `99182784c84a388d82bcf8f661fb2dd4424f06676665ea23b6c8154e9b142617` 和所有依赖模块 SHA 保存在 `.pipeline/axis-research/calibration-24operators-reviewed/manifest.json`；原执行版本另存且不冒充最终版本。24 个 safe NPZ 样本均重新核验 SHA、无 object dtype；所有有值候选均为 proper SO(3)。实际对 `calibration-candidate.json` 调用严格 loader，因 verified 不全 true 被拒绝，结果见 `acceptance.json`。

完整 pipeline 回归：**220 passed，0 failed，0 errors，21.74 秒**，另有两条 pytest JUnit `record_property` 格式警告。报告为 `.pipeline/axis-research/tests-final.xml`；独立 reviewer 复核 65 项 basis／reference 测试全部通过，最终 review 无新增代码阻塞。覆盖未知 twist 的秩检查、已知右乘矩阵恢复、世界坐标变化不影响局部校正、SO(3)／反射、分组留出泄漏、来源文件替换、审核字段／SHA／convention 拒绝，以及完整安全 MCAP 审计和 MCP ring／pinky 命名映射。

报告为 `.pipeline/axis-research/calibration-24operators-reviewed/report.md`／`report.json`；候选为 `calibration-candidate.json` 与 `anchors-and-candidates.npz`，全部标记未认证。README 已补充复现命令、输入格式、5° 数值门槛、严格 loader 的边界与所缺证据。生产冻结 DINO 代码不引入本次研究工具；完整 22 点认证仍需要供应方可信 bind pose／局部 XYZ 定义或独立同步成对旋转参考。

## 方案 B：SMPL-X dense22 小规模拟合及独立 review（2026-10-11）

**小规模数值方法验证通过；当前不批准全量生产转换或原生旋转轴认证。** 本次新增研究模块 `smpl_fit.py`、小规模入口 `validate_smpl_fit_pilot.py` 及 19 项针对性测试，补充可选 `smpl-fit` 依赖与 README。仅生成独立研究数据，未将拟合器接入生产转换器或 MotionExpert；共享目录中其他训练修改不属于本次工作。

最终研究目录：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/axis-research/smpl-fit-pilot-10operators-v2-20261011`。使用同一官方 SMPL-X neutral 资产的前 10 个 shape 系数，10 位不同人物／任务、每人两个不重叠的 8 秒 15Hz 窗口，共 2,400 独立帧。shape 只使用第一个窗口的有效骨长估计并正则化，第二个窗口固定 shape 拟合 pose；不将拟合残差当作留出预测精度。两组消融共享 shape 和原始纯 XYZ 初始化，不从第一组已优化结果 warm-start 第二组。

| 模式 | XYZ mean / P95 | 世界旋转增量 mean / P95 / max |
|---|---:|---:|
| XYZ only | 1.500 / 4.860 cm | 0.725 / 2.867 / 33.054° |
| XYZ＋世界旋转增量 | 1.500 / 4.866 cm | 0.201 / 0.721 / 5.872° |

每模式评估 1,200 帧、26,096 个有效位置、25,804 条有效旋转边；在原世界系计算，不做 Procrustes 对齐。增量使用 `G[t+1] @ G[t].T`，固定右乘局部轴偏移会消掉；不能替换为 `G[t].T @ G[t+1]`。缺失端点、时间异常和疑似跟踪跳变边显式排除。脚接触仅有原始脚速度筛选的代理指标，未伪称人工真值。

独立 review 定位初版中两个窗口开头双髋 XYZ 缺失，恢复几何约束时骨盆初始化 yaw 跳转约 130.7°／66.0°，源对应变化仅约 3.07°／1.51°。现以首个多方向骨盆几何帧为锚向前、向后延续 minimal swing，再初始化下游关节。显式记录 `offline_future_geometry_seed`，只使用位置，保持两组公平。新增回归证明旧实现会失败；相同窗口复测，两恢复边残差从 32.388°／21.531° 降至 3.748°／4.631°，全局最大降至 5.872°。

独立审查重新读取 10 段源 Parquet 投影列／record／stat，核对 8 个冻结代码 SHA 及模型 SHA，全部一致。20 个 shape／eval 输入窗口与 v1 逐数组一致，含位置／四元数／mask／索引／相机／18D／旋转边；所有 shape betas 逐值一致。相机 XYZ／四元数与输入逐值一致。独立复算 18D 最大差为 local9D 8.15×10⁻⁸、floor delta9D 1.46×10⁻⁷；floor `22×9` 最大差 5.96×10⁻⁸，均为 float32 精度。

官方 `smplx==0.1.28` 的 `SMPLXLayer` 完整前向和 `batch_rigid_transform` 抽查两模式各 5 帧：XYZ 最大差 4.44×10⁻⁸m、global rotation 元素误差 0、pelvis／translation 换算误差 6.49×10⁻⁹m。10,475 顶点／20,908 面有效，模型／结果 SHA 保持一致。完整模型 oracle、复现脚本及 mesh NPZ 见 `diagnostics/official_full_forward_oracle.*` 与 `official_forward_meshes.npz`；首帧抽样 mesh／源骨架叠图与全体逐关节残差见 `diagnostics/geometry.png`。该抽查验证解码表示一致，不证明绝对姿态真实或全部网格质量合格。

完整 pipeline 回归：**239 passed，0 failed，0 errors，23.22 秒**，两条既有 JUnit `record_property` 格式警告。报告为 `diagnostics/tests-pipeline.xml`。针对性测试覆盖独立 NumPy FK、非零 pelvis pivot、官方 LBS、零角梯度、有效 mask／缺帧／reset、校准窗口隔离、合成收敛与绝对 twist 不可辨识；无需修改训练代码。

全量生产仍需解决的项目：

- 颈部／锁骨平均位置残差约 4.5–5.0cm。源 neck→head 平均 8.81cm、模型 15.87cm，spine3→collar 差约 5.3–5.4cm，说明同名点不足以保证关节中心／骨架对应；需要定义对应与几何策略。
- 90° 固定 twist 反例保持 FK XYZ 和世界旋转增量不变；绝对旋转与人体合理性未建立独立真值或学习先验。标签保持 `model_fitted_estimate`、`native_axis_certified=false`。
- 本机仅有 SMPL-X，未进行匹配 SMPL-H 的 HM／共同 rig 验证，不能复用 SMPL-X 解码 HM betas。
- 样本筛选偏向有效的短窗；尚未验证全量动作、长期缺失、重置、窗口接缝及跨初始化绝对姿态稳定性。
- 正式训练标签需分别声明目标轴定义、估计标签来源、源轴认证，并建立 mask／reader 契约；不能仅翻转现有 `rotation_basis_verified`。

独立验收结论与逐项证据见最终目录的 `diagnostics/review.md`／`review.json`。README 已同步方法、结果、字段、复现命令和限制，实际相机／地面／18D／DINO 继续保留原定义。

## EgoStandard 原生 dense22 全量导出与 review（2026-10-11）

**全量处理与review通过。** 本次保留EgoSuite原生dense22关节中心和原生局部轴，不运行SMPL-X拟合、不右乘轴校正、不以XYZ重建旋转。输出世界系与相机地面系各`T×198`，每关节顺序为`r00,r01,r02,r10,r11,r12,x,y,z`。旋转为关节局部到相应参考系的全局姿态，不是父子关节之间的局部旋转。输出保持`rotation_basis_verified=false`，数值可用不代表跨来源轴认证。

新增[核心表示](src/egomotion_pipeline/native_dense22.py)、[批量导出与复验](src/egomotion_pipeline/native_export.py)、两组回归测试及CLI的`export-native-dense22`／`validate-native-dense22`。独立派生目录为：

```text
/gaozt-test1/datasets/MixEgoMotion/derived/egosuite-native-dense22-v1/
  data/chunk-XXX/file-XXX.parquet
  episodes/00000000.json
  meta/info.json
  manifest.json
  status.json
  events.jsonl
  validation_status.json
  validation.json
```

21个源运动列逐值复制，加两组198维状态与8组逐关节有效／插值mask，共31列。XYZ和旋转分别写mask，无效3维／6维子块分别写零。相机地面系为`R_floor.T @ R_native`与`R_floor.T @ (p_world - p_floor)`；无地面不能使独立世界系观测失效。帧／episode／全局索引绑定源归档，DINO通过相同索引复用；源Parquet只投影运动列，未读取DINO数值。

| 正式全量检查 | 结果 |
|---|---:|
| 导出／二次磁盘复验 | 6,614段、26,841,893帧；均0失败 |
| 世界有效XYZ／旋转关节帧 | 均519,408,154；87.9575% |
| 地面有效XYZ／旋转关节帧 | 均519,292,268；87.9379% |
| 世界XYZ编码／回环最大差 | 0m，所有有效XYZ逐值保留 |
| 旋转6D编码最大元素差 | 2.9802323e-8 |
| 世界／地面6D还原矩阵最大元素差 | 5.2320558e-8／5.2208672e-8 |
| 地面XYZ编码／回环最大差 | 1.1920929e-7m |
| 地面参考无效 | 11段、9,403帧；占帧数0.03503% |
| 正式派生Parquet | 6,614文件、70,294,065,111字节 |
| 32 CPU进程耗时 | 导出256.63秒；复验112.39秒；完整任务391.22秒 |

有效关节比例以`26,841,893×22`为分母，保留源坏帧mask；全量转换成功不表示所有观测有效。数值回环只证明编码与参考系变换，不代表关节中心、旋转标签或地面的物理精度。

独立真实对照检查episode0、2、892、894，共13,182帧，0／2读取实际试跑Parquet。SciPy独立四元数转换与6D解码：世界XYZ差0，地面XYZ最大差2.996e-8m，6D矩阵还原最大差4.664e-8。现有MotionExpert的15Hz K12窗口在有效地面上mask一致、最大差1.192e-7。该数值对照使用初始冻结核心；随后只有runner的limit范围检查改变，几何核心SHA保持一致。诊断中的`rotation_basis_verified=True`仅用于进入已有reader进行数值对照，未改任何源认证声明。

独立review发现并修复的边界：

- 原始四元数的有限性与范数必须先检查；安全identity占位不能被误认成有效旋转。
- 校验完整派生声明与源证据，并严格区分false、0、null；数值mask和未认证轴声明分别验证。
- 帧身份、30Hz网格、源记录长度和全局索引范围必须一致；已有地面参考必须明确声明已核验相机基。
- 同目录以更小`--limit`重跑不能隐藏已提交段或缩小元信息；现在在改写任何进度／报告前拒绝排除已提交episode。完整／部分范围缩小均有回归，所有输出文件字节不变；partial→full仍可恢复。

最终无未解决代码review问题。源码冻结后用干净正式目录重新导出；旧中断试运行保存在任务目录`pre-P2-output`，不参与正式6614段统计。完整回归为**299 passed，0 failed，0 errors，25.14秒**，含60项原生表示／导出测试；两条既有JUnit `record_property`格式警告不影响测试。测试覆盖独立非零旋转／坐标验算、零／NaN四元数、XYZ／旋转mask独立性、无地面保留world、插值依赖、原生声明拒绝、DINO列未读取、源文件不变、串并行一致、resume与损坏／来源变更拒绝。

地面失败的11段均为`no_stationary_feet`、`candidate_count=0`，不是DBSCAN噪声或旧4cm限制：供应方lower_body坏帧标注屏蔽脚点，10段无有效脚点，894仅一帧双脚有效，全部没有连续有效脚点边。10段无法按训练窗口脚均值兜底；894只有包含孤立脚帧的窗口可按当前reader规则用两个脚点生成未认证启发式地面。派生文件保留存档无地面mask，窗口兜底不回写。供应方坏帧计数来自原capture帧，不能在源帧率不同于30Hz时直接与存档帧数作分母比较。

任务与验收证据：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/native-dense22-20261011T045036Z/`。`production_summary.json`保存结果汇总，`export_result.json`／`validation_result.json`保存全量结果，`ground-failures.json`保存11段诊断，`real-parity.json`保存独立数值对照，`pipeline-tests.xml`保存完整回归。`code_manifest.json`固定23个源码模块SHA，`process.log`与`job_status.json`记录正式任务，入口为`.pipeline/current_native_dense22.json`。

`integrity.json`确认：源episode JSON与config／info／ontology SHA相同；源Parquet的size／mtime／inode相同；每段投影运动SHA在导出和全量复验中一致；MotionExpert与源码快照SHA相同。为避免读取498.65GB主归档中的DINO，未重算源完整Parquet SHA。源码只读源文件、持有源共享读锁，单独写派生目录；本次没有重算视觉特征或改动训练代码。

本结果可作为EgoSuite原生表示数据。它没有统一EgoExo-HM的关节局部轴或关节中心，也没有改变其适配路径；静态floor列与按训练窗口重新估计的floor不能在未明确地面策略时混用。Native 6D导出不能用来将`rotation_basis_verified`改为true。

## 2026-10-12：双来源独立 SMPL-H 参数归档

新任务为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-parameters-v1-20261012T032915Z`，目标归档为 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1`。正式生命周期于北京时间2026-10-12 12:26启动；本节记录代码和小样本验收，不表示全量处理或删除已完成。最新阶段、进度和最终结论分别见任务 `status.json`、来源日志及 `reports/full-parameter-acceptance.json`。

两套来源存同一个真实16维shape／52关节模型的root orientation、21个body local axis-angle、SMPL trans及beta，物理Parquet不保存身体denseXYZ／全局6D／floor缓存。源手21点、相机／头戴／地面、脚点、观测／插值／补全／估计mask与质量单独保留，DINO逐值迁至自有视觉Parquet。HM另保留发布90维手PCA与独立可用mask；EgoSuite没有拟合手指，不能用静态pose代替其手21观测。

库存核对9,263段／38,137,884帧：EgoSuite6,614／26,841,893；HM2,649／11,295,991。EgoSuite保留原split，HM按participant／capture连通分量分组，272个组件；未知participant保守共同分组。HM train/val/test 为2,086／268／295 take，帧比例80.0042%／9.9902%／10.0055%，全部分组跨split计数为0。

审查修复的实际接口问题：

- HM DINO `video_time_error`为float64而旧v5为float32；统一参数字段使用float64，EgoSuite精确升格，避免精度丢失与帧关联失败。视觉字段先按目标dtype转换后计算流式digest。
- 保留无效HM chunk中仍有限的原手坐标，mask保持无效；无效数值不能自动成为观测或监督。
- 对齐依据固定官方代码与SHA，默认`T=N−1`与7个登记的完整100帧chunk截尾规则分别验证；runtime timesync行数单独从原CSV重核。原全视频DINO与未消费尾帧仍保留。
- 拟合receipt绑定原运动projection、模型及fitted Parquet SHA；atomic提交与恢复核对来源身份、当前代码／计划／库存、record与receipt一致性、完整输出SHA。
- 被隔离的来源拒绝canonical训练窗口，原坐标检查仍允许保留其数据和mask。
- 先初始化自有模型／PCA基底再并行两来源；子进程固定pipeline导入路径。运行锁、残留子进程检查及删除阶段恢复避免重复writer与读取已删来源。

122项新回归通过（15.93秒），报告 `reports/parameter-migration-tests.xml`。真实EgoSuite三段16,379帧与HM三take5,219帧全行参数／手／相机／地面／mask／DINO通过独立验算；各自新进程在禁止全部旧来源路径后完成3段／12类读取检查。HM涵盖实际legacy截尾、无效chunk但有限手值和逐帧beta。EgoSuite新旧窗口身体位置最大差1.192e-7m；双来源实际fore/recon编码可生成245维motion、18维轨迹和5×1024视觉。该低层编码检查显式使用模型定义轴，不修改原生轴认证或MotionExpert源入口，不代表训练吞吐验收。

生产冻结54个文件，`code/producer-contract.json` SHA为`3d7fafded3d0a418e160c400e1d74c08e842504b60c66727234b29f3de20cc81`；正在运行的旧拟合19个冻结文件逐个SHA保持一致。独立代码／生命周期／库存review报告保存于任务 `reports/`。本次没有修改MotionExpert。

清理要求原EgoSuite完整6,614段的预登记质量门禁、新归档完整9,263段保真审计与禁止旧路径reader全部通过。精确清单只允许删除旧主归档及登记derived目录中含身体dense列的Parquet；原MCAP、原HM／Aria／全视频DINO、权重、JSON及job研究证据、新参数归档全部保留。删除使用来源stat、验收SHA、protected输出与无writer检查，持久化pending／done记录可恢复；删除后重新逐段读取通过才标记任务complete。当前全量验收仍pending，旧数据尚未删除。

04:51 UTC全量迁移完成，9,263段／38,137,884帧均已提交，自有DINO与来源元数据一并保存；两套来源migration报告均无worker失败。合并train／val／test为7,409／915／939段。HM最终47take／61,743帧隔离，其中44take／54,152帧仅源CSV非单位四元数求逆及float32序列化与SDK规范表示不同，176个view全行原运算复现；1take／5,381帧源optimizer valid全零；2take／2,210帧源MPS实际跨graph，66个有效相机帧落在缺乏匹配exo校准的第二graph。未因事后诊断修改冻结门槛、世界认证或源mask，9,216段满足当前段准入标记。

独立DINO跨来源核查确认同模型权重／配置／5token顺序／最短边336 bicubic及ImageNet归一化；源图像几何与bad-image flags可用性不同，均写入来源语义。EgoSuite的video_frame_index是原30Hz特征行，并非MCAP解码图像ordinal；真实图像按最近PTS关联。HM则保留发布代码row→video帧链。证据为 `reports/independent-cross-source-dino-contract.json`。以上迁移与研究结论不代替全量参数验收；原完整6,614段质量审计与后续新归档独立全量审计继续由后台生命周期执行，所有门禁通过才自动清理旧dense22。

2026-10-12 08:15 UTC状态复核：原完整6,614段独立数值audit已全部通过，body mean0.006875662m／P950.018467287m、hand-wrist P950.003286287m、local>45°比例0.0000456696，四项预登记数值门槛全部通过。原数据13,231项身份检查通过，唯一整体失败是22个MotionExpert源码／config SHA与原冻结快照不同。原full gate为 `full_candidate_quality_failed`；参数生命周期已于06:13 UTC进入 `failed`，PID不再存在。新归档full-parameter-acceptance及cleanup manifest／execution尚未生成，不能表述为继续运行、整体验收通过或已删除旧dense。只更新本说明，未回滚MotionExpert、重签快照、放宽门槛或重启冻结任务。独立观察报告为 `reports/original-full-fit-gate-observation.json`。


## 最新 MotionExpert 整体验收：仅继续未完成的关键检查

任务为 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/smplh-qualified-20261012T084000Z/`。依据用户“数值检查通过就不要重新检查”“只检查没检查过的东西”及“只检查最关键”的要求，已有 native 数值、5Hz 全量索引、18,410 个首尾训练样本及完整模型前反向证据直接复用；不再启动全量数值、FK、索引或模型测试。

已有 9,263 段／38,137,884 帧；最新 MotionExpert 使用245维motion、18维轨迹、5×1024 DINO，train／val／test窗口分别1,001,223／125,262／125,234。47段HM保留隔离，11段EgoSuite保留但无有效脚点地面窗口。首个pipeline独立reader覆盖全部9,263段，完整数值与来源审计执行后仅最终JSON写报告因 `inf` 失败。失败执行收据与日志保留，新execution witness只推断冻结控制流下已完成的断言，不重构未保存的逐样本诊断数值。

剩余关键门禁是恢复报告的必要证据、最新MotionExpert默认正式发布读取，以及条件清理后自有Parquet字段／行数／mask结构可读性。清理后不再次运行FK、数值、窗口索引或模型。正式全量train归一化统计仍未生成，不把归档及接口验收等同于训练就绪。

2026-10-12 10:36 UTC最终状态：`complete`，统一参数归档 `accepted`，关键接口与清理后独立可读性通过。精确删除19,855个旧dense Parquet／835,186,426,720字节，原始输入、权重、JSON及失败任务证据保留。删除前后核对完整清单，每文件使用原身份核验及持久化pending/done，期间仅检查冻结证据身份，避免199轮重复完整保护文件扫描。清理后一次新进程检查覆盖9,263段的自有字段／行数／mask结构：`data_pages_read=false`、`numerical_checks_rerun=false`、`fk_or_model_executed=false`，未重复窗口或模型验收。简明结论为独立任务 `reports/critical-acceptance-summary.json`，完整收据为 `reports/dense-cleanup-execution.json`。正式训练仍待全量train的245＋18维统计。
