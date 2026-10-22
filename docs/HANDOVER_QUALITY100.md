# Handover数据质量、执行差异与100条重新采集

本轮按当前`G1WholebodyHandoverTeleop-v0`推进，保持A100原训练、参数及已用数据不动。
目标是重新物理执行并筛出100条符合本机SONIC v1.1部署协议的TRAIN示例；
100条示例通过验收与训练后策略达到100%成功率分别统计，不能相互替代。
最终应以预先冻结的独立测试集100/100作为一个实测目标，不能保证所有未见场景都成功。

## 已确认的数据来源

原50条包含29条`original_or_repaired`与21条`task_scene_regeneration`。
TRAIN45实际是24条原轨迹修正及21条模板0再生成；VAL5另外属于5个祖先组。
模板0和其21条衍生占TRAIN有效窗口7282/13808，即52.74%；按动作祖先计算只有24组TRAIN。
所以数据集中存在明显的模板权重集中，不能把45条TRAIN理解为45条独立动作示范。

21条生成轨迹有各自的物理执行、图像和完整动作文件SHA；现有证据不支持它们只是逐值复制，
也不支持把所有生成示例都称为坏数据。当前manifest中的TRAIN/VAL祖先无交集，
没有发现已记录划分污染；但来源少和重复权重大可以是泛化受限的重要原因。
持续上升的固定VAL与下降的TRAIN loss也是警讯，单次物理测评不能据此证明唯一根因。

原数据通过的是源端50Hz命令执行。现任务数据按`ceil(k*50/30)`选择后用30Hz保持，
并不保证重现50Hz控制的接触时序。五条VAL中18.5–22.6%的50Hz执行tick会取到不同命令，
最高滞后20ms；手部开闭跳变处最大单通道差约1.49–1.50rad。
这些是命令序列差异的CPU核对，不是它导致失败的因果证明。

另已核实初始动作监督的覆盖差异：旧source0原生650行50Hz只按索引转换为390行30Hz，
没有前置训练观测；历史6帧及完整future30过滤后current为30至360，共331个窗口。
未来偏移全为非负，所以教师30Hz行0至29没有进入任何动作监督窗口。
旧的`controller_priming`列和初始化收据确实存在，但它们没有在训练时前置一段真实观测。
新采集增加30行真实暖机图像，动作不可用；420行中current30至390共有361个窗口，
current30对应教师原始command0，补回最初1秒的30个30Hz动作标签。
新标签的token及具名手targets已逐值核对原生源命令，归一化不变。
本地未重复读取远端旧30Hz整包；该结论由本地真实native parquet、转换器索引、
历史/未来过滤实现及旧实际CPU331窗口收据交叉验证。
source0家族22条的`22×331=7282`也与旧TRAIN计数一致。
这证明存在监督覆盖缺口，尚不能证明它造成了抓取失败，也不表示继承的UnifoLM从未见过其他启动数据。
证据为外部根目录`initial_action_coverage_audit_v1.json`。

## 已定位的失败与执行差异

任务6000步的四条失败均未形成最初右手稳定抓取和抬升，最高仅比初始目标高度高约1.4–3.1mm，
没有进入左手交接。因此优先解决抓取，不能把这四条主要解释为最终释放失败。
source39可以抬起、交接并碰篮，但16秒末仍有左手接触，完整释放没有确认。

在模型首次出动作之前，1秒纯姿态PD已使骨盆下沉约2.1–3.1cm，且倾角明显大于教师同时间状态。
该偏移发生在模型动作之前，不由这一次模型预测造成。
源`collection`手KP每侧为`5,5,5,2.5,2.5,2.5,2.5`，KD全部1，已核实与本机具名分配相同。
源采用的model12配置把非拇指根部手指限扭矩设为0.7Nm；本机默认MJCF上限为1.4Nm，
已有`recording_0p7_nm`选项可独立比较。源实际DDS还保留期望速度和前馈力矩字段，
不能只凭相同KP/KD就宣称完整控制器等价。

本机Python身体KP/KD与现有NVIDIA C++源码转float32后相同；源manifest记录的定制肩roll为±0.2，
与Python相同。源定制二进制对应的完整C++头文件仍未提供，启动时的历史/上一动作状态也不同。
这些核对不支持盲目增加手gain或将全部差异归因模型。

在预先指定的source65/80、相同6000步权重及采样seed上串行做单变量诊断。
将执行窗口30步改为6步，两条均仍未抓起，不能据此采用该改动。
改用`live_pose_sonic_v1_1`暖机后，首次动作骨盆高度约0.757m、倾角约1.2–1.4°，
但两条仍失败，说明改善冷启动姿态不足以独立解决抓取。
这些是小样本诊断，不能作为整体配置优劣或最终成功率结论。
另外完成手指0.7Nm上限及v1.1 FSQ格点投影各两次同场景诊断，也均未抓起。
四种单变量共八次修改配置的模型闭环均未触发成功标志；本轮没有据此改生产默认值或A100训练。
完整分组、seed、视频和结果SHA见仓库外`model_ablation_audit.json`。

## 新采集的准入与当前状态

`plan_handover_quality100.py`绑定原release和selection SHA，排除原VAL以及21条模板再生成，
从24组明确TRAIN祖先按轮均衡安排120个候选，目标接受100，每祖先最多接受5条且需覆盖全部24组。
候选含±1cm以内XY和±0.05rad以内yaw的非零初态扰动及独立随机种子。
它们必须重新执行；改变文件名、seed或图片颜色本身不产生合格示例。
即使100次新执行全部成功，动作祖先仍只有24组，不能称为100条独立人类示范。

新计划在本机仓库外：
`/var/tmp/cvpr-evaluation-zxc-handover-quality100-20261022/collection-plan-v2/candidate_plan.json`。
计划保持`planned_not_collected`，正式队列已独立接受首条，批量采集继续运行；诊断和正式接受数分开记录，不因计划或CPU测试通过而标记100条数据完成。
原五条保持VAL，独立TEST种子空间已预留，但独立祖先绑定待完成，不能把原VAL改称TEST。

实际执行必须使用目标30Hz命令保持、50Hz SONIC解码、200Hz权威物理，重新采集同步实际RGB、状态和下发动作。
只在控制前设置一次物体初态，禁止控制期间复位物体、刚性附着或施加帮助成功的外力。
原成功标志只是接触信号，不足以入库；使用源`handover-physics-v2`的完整必要物理门槛：
右手、双手、左手有序抬升接触，充分抬升、稳定站立，最后1秒50个观测全部无双手/掌根接触，
至少49帧容器接触、位移小于5mm，同时通过容器姿态、碰撞几何范围与实测向上支撑判据。
源门槛为50Hz验收观测和200Hz物理，不是200Hz验收采样。
这些必要条件仍不是篮子精确内腔包含关系的数学证明，视频与RGB/动作时钟核对也需要保留。

新增`handover_contact_quality.py`和`handover_geometry_quality.py`保留源纯函数的数值规则及来源SHA，
从原`mj_step`后、观察刷新`mj_forward`前读取解算接触力；不能事后只重建姿态冒充实际支撑力。
即使物理门槛通过，helper仍返回`dataset_ready=false`，还需要真实RGB/动作、来源和划分验收。

`replay_handover_teacher.py`先对TRAIN source0进行50Hz/30Hz与手限扭矩的受控重放，
直接使用已执行的650行`replay_action78`，核对具名手重排；不用660行参考动作文件中未执行的最后10行。
它不重编码已经正确标识的v1.1 token，不将教师重放算作模型成绩。
`--render-mode none`只做真实物理诊断，不能产出带RGB的已接受训练包；
只有后续实际RTX重采、物理与数据审计全部通过，才能计入100条目标。

已完成TRAIN source0的四个真实CPU物理诊断：50Hz/30Hz各配`mjcf_limits`及`recording_0p7_nm`，
同一1秒measured-PD暖机、650个源执行tick，四次完整`handover-physics-v2`均通过。
这证明该源在当前本机控制器与30Hz时钟下有可执行路径，不能认定官方轨迹一律无法适配v1.1。
它不证明其他23组都能重放，也没有产生4条新的RGB训练数据；none模式四次均保持不入库。
首个旧版本在第一条teacher手指令的NumPy float32 JSON边界失败，已修复并加回归检查，
失败目录保留且不计入成功数据。四次有效诊断使用独立新目录，未执行参考文件的未记录尾段。
四次各701行真实integration state还独立核对了外加力全零，模型均无额外equality约束，
没有靠弹性带、刚性物体附着或外加力产生成功。

## 首轮批量筛查和真实RTX接口验收

按预先登记的variant0对24组TRAIN祖先各做一次30Hz、0.7Nm、1秒measured-PD的真实物理筛查，
13条通过完整门槛、11条失败。该比例是教师候选筛查结果，不是模型成绩，且none模式没有形成24条RGB数据。
source17/29/81的接触阶段虽通过，最终篮倾角分别约73.88°、67.28°、71.06°，真实不满足放置几何门槛；
source31最高仅抬升约1.78mm；source45/67/68已触篮但末窗仍有手接触。这些失败保留，不入库。
因此原120个候选并不能保证足够100条；必要时应登记额外候选或修复教师控制，再重新物理验收，不能降低门槛凑数。

独立的`rtx-interface-pilot-v3`完成source0/variant0同一扰动的真实Isaac RTX重采，
共420个640×360 uint8 RGB事件，其中30个暖机历史事件、390个教师动作事件。
完整物理门槛及新版独立capture audit均通过；实际查看了抓取与最终放篮画面。
该目录仍是接口诊断，未登记进正式100条队列，也未提取DINO或执行训练预检。
旧RTX尝试分别在EULA环境变量缺失和实际ActionCmd缺少`.get`接口时退出，记录保留，均不计成功。
动作接口已用真实ActionCmd对象回归；正式批次前继续绑定渲染桥源码及逐帧相机姿态证据。

独立数据审核v2从保存MJB和`physics_state_integration`通过具名关节/自由体还原权威状态，
逐事件核对状态、物体、根速度和绝对时钟，以SHA绑定源parquet第0行及reset记录核对一次性扰动，
并核对全部50Hz执行tick的30Hz命令保持，不能只相信同文件的状态副数组或已被采样的动作。
恢复计数还要重验已接受capture的所有文件SHA、候选身份、计划绑定及唯一性。
这与原`mj_step`接触力验收互补；离线`mj_forward`只用于几何，不冒充原执行的解算支撑力。

`pack_handover_quality100.py`只接纳上述独立审核通过的正式capture，转换具名43维状态、
64维v1.1 body token和14维手目标，并继承38k归一化统计，不重新拟合bounds。
真实暖机RGB/state可用于历史，但暖机动作标签为NaN、valid和全部action mask为false；
未来监督要求30个真实完整动作，不用末帧重复填充。PNG转视频须逐帧像素无损核对。
真实DINO特征、联合模型数据预检和独立TEST通过前，不得标记完整训练/评测准备完成。

## 暖机与相机坐标诊断

`ancestor_survey_failure_readonly_diagnosis.json`记录了源和本机的对照证据。
source40在measured-PD暖机tick5开始左middle0碰篮，tick25篮倾角约76°，即教师开始前已破坏场景。
source31教师首帧相对源骨盆后移83.3mm、右腕后移117.8mm，1秒后仍后移99.0/97.1mm；
source9也先偏离，但随后大幅恢复，source0有初始低腕却能成功。因此暖机是具体待对照因素，不能解释所有失败。
source45/67/68原末段和本机末段手目标均已精确归零，真实手指也接近开手位，但仍有0.10–0.40mm级碰触；
继续发送open目标不等于撤开手腕。修复须重新物理执行，不能末尾复位物体或删掉碰触帧。

新增`simple/sensors/head_camera_mount.py`从具名实际head mesh组合相机外参，
修正MuJoCo沿用旧`-0.044m`头部原点与本机SONIC v1.1 `-0.054m`的1cm差异。
不移动Isaac真实传感器，不改已采RGB；旧头部模型仍按其实际geometry计算。
USD和SONIC MJCF腰部及肩部joint frame也不同，不能将所有同名link原点直接相减当作渲染滞后。
审核使用来源USD/URDF与实际MJB关节链的明确映射，并分别记录raw与mapped残差，禁止用首帧残差拟合。
14个历史姿态的CPU恢复核对中，完整映射后相机最大残差约0.314µm、torso约0.110µm、腕约0.663µm；
这些只证明该组记录中的坐标映射，正式新RTX仍需逐帧实际检查。
新采集预声明mapped位置容差1mm、旋转0.1°；MuJoCo日志与integration状态/时钟仍做严格独立复核。
源asset实际文件SHA、camera helper与Mujoco引擎源码都要绑定到每次采集。

## 审计位置

本轮仓库外根目录：`/var/tmp/cvpr-evaluation-zxc-handover-quality100-20261022`。
保留`data_lineage_audit.json`、`failure_stage_audit.json`、
`source_control_comparison_audit.json`、`body_static_comparison.json`及源源码/源0输入迁移SHA收据。
模型单变量对照在`/var/tmp/cvpr-evaluation-zxc-task50-eval6000/`的独立`*-ablation`目录，
不覆盖原五条测评结果；新教师重放也与模型测评、原训练分开计数。

## 本机批量入口

当前正式队列为外部根目录下`collection-queue-v3`。
首条source0/variant0通过完整RTX数据审核后登记接受；诊断pilot目录不混入该计数。
`train-warmup-schedule-v1.json`预先冻结23组measured及source31的live暖机选择，
仅用于TRAIN教师采集；source31已有measured失败/live完整成功的真实物理对照。
live暖机先10个真实PD历史采样，再40次固定encoder/decoder控制；暖机30个RGB事件仍全部不监督动作。
采集、筛查和恢复均核对同一schedule，不允许按VAL/TEST场景择优启动后计模型成功率。

打包使用`cvpr-simple-model311` Conda环境。独立MJB审核需要该环境也有CPU MuJoCo读取依赖：

```bash
conda activate cvpr-simple-model311
python -m pip install mujoco==3.3.6
```

本机仅补装该包及`etils 1.14.0`、`glfw 2.11.0`、`PyOpenGL 3.1.10`，沿用现有Torch/CUDA。
打包选定快照时若队列恰好更新，会明确退出并要求重试，不接受前后不一致的登记。
DINO沿用当前A100任务数据的bf16计算、float32保存、`256x320`几何及相同冻结权重，
UnifoLM预训练旧缓存本体为float16，两者的实际NPY头部只读核对已另存，不改A100缓存。

## 首条生产链路实测

`collection-queue-v3/attempt-00000`是真实source0/variant0正式入库示例，
420帧完整物理/动作/状态/渲染审核通过。新采集实际mapped残差最大：link约0.758µm、camera约0.452µm；
raw torso约10.10mm、腕/相机约5.26mm也完整记录，说明模型joint-frame差不能直接当作同步误差。
该审核仅证明实际关节/时钟与已知asset映射一致，不证明两款asset几何完全相同。

首条`packed-pilot1-v3`完成真实LeRobot无损打包，DINO使用本机GPU提取420×5×1024 float32特征。
生产`MotionHistoryIndex`和LeRobot读入真实数据并通过输出校验：首个有效current30使用[0,6,12,18,24,30]，
末有效current390使用[360,366,372,378,384,390]，均为有限[6,5,1024]，当前RGB为[3,256,320]，
每个有效current均有完整30步动作，单条共361个有效训练窗口。
以上是数据读入检查；后续联合模型GPU预检结果见下节，完整100条及独立TEST仍未完成。

### 首条数据的真实联合GPU预检

`joint_gpu_preflight_v1_prepared_final/gpu_result.json`记录本机4090、batch1、一次生产训练路径的
完整前向和反向。加载已导出的task6000三模块联合权重，使用上述真实RGB、冻结DINO和有效动作窗口，
Action、语言骨干、视觉塔、projector和Motion五组梯度均有限且非零；峰值allocated为21.72GiB，
reserved为22.64GiB。原始`TrainingSafetyGuard`通过，优化器更新和checkpoint保存均被明确禁止，
五组参数、来源checkpoint、代码及输入DINO文件的前后SHA保持一致。

该收据SHA为`80069c3f0ec16a299e7f7722b0b4c9c265442aadfd212fe9fe32632c17d2827f`。
单次预检loss为1.0164694786，只说明该随机批次可完成有限反传，不与A100训练loss比较，
也不用于判断模型效果、长程数值重复性或100条数据验收。完整数据包生成后仍须执行其全量审核、
DINO提取和生产读入/GPU预检；当前A100作业未因本检查变动。

其余variant1–4另做96条无RGB筛查，63条通过，四个来源17/40/45/68目前五个初态都未完整通过。
这些诊断不计正式示例；批量队列保持预登记物理门槛与划分，后续修复需单独执行、声明与复验。
仓库外`run_formal_batch_v3.py`先处理120个候选，必要时分轮明确登记额外候选，当前总预算最多480，
每祖先接受上限仍5、目标100且需24组覆盖；预算耗尽不标记完成。
实际进度以`collection-queue-v3/queue.json`和`formal_batch_live_summary.json`为准，勿把候选数当接受数。

## 冷启动、撤手参考与独立批次

原正式队列的代码、recipe和来源SHA保持冻结。后续实现使用独立工作区
`/home/zxc/.codex/worktrees/handover-teacher-repairs/CVPR_Evaluation`，不改正在运行的采集或A100训练。
新增队列为`cvpr-handover-quality100-serial-queue-v3`，审核为
`cvpr-handover-independent-rtx-capture-audit-v3`，authority为
`integration-source-row0-command-trace-renderer-pose-startup-reference-v4`；旧队列继续使用自己的原版审核。

显式暖机模式`live_pose_sonic_v1_1_repeat_first_snapshot`从第一实际状态调用固定v1.1编码器，
初始缺失的解码器历史用同一t0快照填充，前9次填充数为9至1，之后全部是真实50Hz历史。
填充明确标记为synthetic，不捏造过去时间，不覆盖真实RGB、状态或动作标签。
全程记录每次994维解码输入、实际raw action29和PD target29；独立审核从integration重建
关节、速度、局部角速度、重力、上一实际输出与槽位时间，逐值核对输入及SHA。
该模式只替换一个控制器实例的历史输入构造，不修改共享上游模块、力矩规则或别的进程。
暖机的30个真实RGB事件仍只作历史，动作label保持不可用。

部分教师已完成交接和放篮，但末尾仍有左手接触。`prepare_handover_teacher_reference.py`
为此导出明确的TRAIN参考文件，严格保留全部原生50Hz命令前缀，仅追加经固定SONIC v1.1编码器生成的撤手尾段。
原始parquet、初态、对象状态、源祖先和归一化不变；原生参考数组与实际30Hz保持后的执行数组分别保存。
审核还将导出数组绑定到原研究执行的真实命令表，拒绝两份数组同时修改后自洽的伪来源。
记录区分`repair_origin_candidate`和`executed_candidate`；同一来源的新扰动必须重新执行完整物理验收。
任务几何只可用于离线TRAIN教师构造，禁止用于VAL/TEST策略的参考或未来观测。

18次有完整证据的CPU诊断中，source45/variant100以及source17、68的原variant0独立审核通过。
17、68的2秒尾段可以复现完整门槛；source17连同暖机总时长15.68秒。
17、40、68的新variants100至104均失败，不能据原初态通过宣称扰动泛化已解决。
这些none诊断全部不计RGB数据，证据在外部根目录的
`worktree-startup-reference-production-cpu-v2/FAILURE_STAGE_REPORT.json`。
source40的同seed无扰动新旧实现逐值一致，落箱后的篮缘偏心碰撞仍需单独处理。

后续定向source40/variant200已找到完整物理及473次解码历史审核均通过的非零初态，
仅确认该窄范围配置可执行，不外推其它偏移。四个来源17/68/40/45随后均已完成真实RTX独立准入；
末秒篮子承担实际目标重量约100%，双手均无接触，已抽查最终画面。
这些通过说明教师采集路径有效，不是模型对照或100条完成；实时计数以双队列watch为准。

`--warmup-schedule`可增加`teacher_references`，按原始TRAIN source ID指定引用JSON的本地绝对路径和SHA。
引用、代码、原始来源及实际执行参数在队列首次运行前冻结；不得修改旧队列的recipe来接纳新控制。
`--companion-queue`仅读另一个队列的热计数，减少重复采集和超额来源；该计数只是调度提示，
达到100和24组覆盖时也不自动标记数据完成，最终仍须逐条独立审计。

## 不同冻结版本的数据合并

打包器除`--queue-root`外支持互斥参数`--capture-sources selection.json`。
显式选择文件的contract为`cvpr-handover-multi-queue-selection-v1`、scope为`TRAIN_only`，
`sources`中每项声明`queue_root`、`collector_path`、`collector_sha256`和已接受的`candidate_ids`。
加载前要求审核器SHA同时等于实际源码和该队列冻结recipe/代码快照，不把旧批次升级为新版身份。
每个版本仍执行完整的来源、物理、动作、相机与像素审核，再全局检查候选/内容去重和每祖先最多5条。

热队列审核使用最初读取的完整registry快照，只对本次审核函数的那一次`queue.json`读取提供固定输入，
其余文件读取、审核函数和实际路径不变，不修改模块全局或运行进程。
审核前后允许追加候选及进行中条目推进；已接受条目、计划、recipe和旧registration均不得改写。
这样不因漫长像素审核期间正常新增一条而反复失败，最终包仍保存选择时的原始registry字节与SHA。

完整100条打包、全量DINO和联合GPU预检完成前不设置`final_ready`。
已有五条VAL继续保持原划分，独立TEST祖先仍待绑定；教师筛选成功率不作为训练后模型成功率。
