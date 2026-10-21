# 4090：独立举箱保持任务与真实指令数据

本任务依据用户“本地定制、先简单测几个”的要求，作为首个本地适配任务。
它与搬箱放桌是两个任务；原CarryBox成功标准、失败记录和旧派生数据禁训门槛保持不变。
旧数据迁移问题与实验见 [CarryBox记录](4090_SIMPLE_CARRYBOX_FINETUNE.md)。

## 预先固定的任务协议

- Gym：`simple/G1WholebodyLiftBoxHoldSonic-v0`。
- 任务：把箱子举到最低几何点离地至少0.10米，并持续保持离地1.0秒。
- 必须同时有真实手部或融合手掌接触、法向力大于0.01N，且无地面/桌面接触。
- 允许膝盖等其它机器人身体辅助；不将它描述成双手独立支撑或静止悬停。
- 使用同一G1、箱体、scene0、固定地面Z=0及既有物理参数；不修改物体质量或摩擦来通过任务。
- 物理200Hz、decoder50Hz；成功条件在50Hz真实观测上逐次检查，按仿真时间累计，条件中断则清零。
- 同步观测必须为同qpos的`mj_forward`状态与真实新RGB；关闭前必须完成采集封存。

选择这个任务前，只查看了TRAIN源0/1/2的可行性：新v1.1 heading闭环已出现满足上述
条件的连续窗口，分别约5.12、1.00、1.42秒。source1余量很小，必须由新任务实际执行确认；
不能直接把旧CarryBox失败文件改成举箱成功。

## 数据与训练边界

实际新episode采集同一tick的RGB、实际state43和**已执行**的v1.1 token64/hand14。
不拼接旧示范RGB，不以事后编码实测q替换动作标签。RGB无损封存、逐帧核像素；
完整物理状态和MJB用于独立重算接触力和举箱条件，而非仅相信success字段。
不足六帧真实5Hz历史或30个真实未来动作的样本裁掉，不复制历史、填补未来。

TRAIN候选按原训练集次序采集，失败保留并拒绝入包；首个pilot收集3条合格示范。
最初预留VAL源8、TEST源4在协议和控制器冻结后独立运行，不依据其结果调控制参数。
源8实际失败：最长连续保持0.86秒，随后箱子落地；不得将它写成成功示范。
离线BC包改为明确筛选的成功示范：TRAIN源0/1/2、VAL源3、TEST源4，
源3此前未用于控制调试。source8完整失败记录保留，并列为模型前后物理对照的独立初态。
因此原预选五条结果是4/5，而成功示范包是另行筛选的5条；不能把后者称为未筛选成功率。
这只是小规模本地任务适配，不能宣称为官方SIMPLE榜单成绩或广泛泛化能力。

初始化使用独立冻结的A100 `ckpt_28000`，其身份和模型GPU验收见CarryBox记录。
已完成的300步诊断轮只更新ActionExpert，VLM/MotionExpert/DINO冻结；保留A100归一化统计，
不从SIMPLE或TEST重估bounds。优化器、scheduler和step从新任务0开始；
不读取或改变A100训练optimizer/RNG。

按用户后续“最新权重、与A100实际联合训练一致”的要求，正式联合pilot将改用重新核对的
`ckpt_31000`；前述Action-only诊断和28k失败对照保持原始身份。31k已于2026-10-21
14:05:29（北京时间）完成独立模型硬链接保护，确认完整model/optimizer/scheduler/8卡RNG文件和
训练日志保存完成标记，仅迁移模型及构造小文件。联合模型12,066,487,572字节，SHA256
`e43fd2fb07f093dbf7a1e00ee737d5378f02f374c1b520056a65c34f4747daa6`。
这是此次取用时固定的最新完整快照，后续A100继续训练不会改变本轮实验的基线。
本地于2026-10-21 14:27:43（北京时间）完成12,066,487,572字节的完整SHA复核，
随后`prepare_joint_snapshot.py`核对原model/data与统计、严格提取31k Motion并逐tensor序列化比较通过。
本地run为仓库外`resources/latest31000/run`，迁移及Motion准备完成不代表联合微调或物理成功。

本机控制器已确认是NVIDIA官方SONIC **v1.1**，资源revision
`6733128a3d8a523b1418b06bca3cdf61c8b0987f`，encoder SHA256
`fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae`，
decoder SHA256 `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509`。
新数据所有标签均由这份encoder实际产生并执行；原A100训练集逐行token对应的完整二进制身份记录
仍缺失，不能用新teacher记录替原训练记录背书。因此当前物理运行显式保留
`--diagnostic-unverified-sonic` 边界，不报告官方榜单成绩；这不是“本机SONIC版本未知”。

## 正式RTX数据与检查

已完成6条正式RTX采集和独立检查。TRAIN源0/1/2成功，分别437/584/446条命令、
739/985/754个完整物理状态，末端箱底约0.1553/0.2327/0.1626米；
VAL源3成功，371条命令、630个状态，末端箱底约0.1643米；
TEST源4成功，410条命令、695个状态，末端箱底约0.1450米。
五条均实际保持1秒后正常终止，无跌倒。源2有左膝辅助，符合任务条件，不能称为双手独立支撑。
失败源8保存636条命令、1071个状态，未入BC包。

新包在仓库外 `resources/lift_box_hold_v11_rtx_pilot`，共2248帧，
`LIFT_COMMANDED_ROLLOUT_READY.json` SHA256为
`983b13f6b5baa22f8579e1e6744da59c9d29bdf161d16f8fba65951d7f5f35bb`。
完整积分状态重算的接触力/成功条件、逐事件官方encoder输出、原始PNG及打包标签/时间均通过。
Sim Conda的PyAV16.0.1无损编码，Model Conda的PyAV18.1.0解码，与2248张原始RGB逐帧像素一致；
两边都启用 `PYTHONNOUSERSITE=1`，依赖确实位于各自Conda内。
新包的clean DINO缓存覆盖TRAIN1467、VAL371、TEST410帧，未复用旧转换数据缓存。
实际 `--prepare-only` 重新执行CPU物理/标签/RGB检查和全DINO校验，通过后才启动微调。

训练窗口必须有过去1秒真实历史和未来30个真实动作。示范起始1秒会被裁掉，
所以这个小包对任务早期接近阶段的覆盖有限，不能仅凭全帧字节核验宣称训练分布完整。
训练及最终模型闭环结果另行记录；不能用teacher成功代替模型成功。


正式TRAIN动作范围预警：body64约0.23219%的数值低于继承bounds，
最大差0.0625（一个SONIC量化步）；hand14约0.029214%越界，最大约0.003671rad。
未改bounds、未读TEST重估统计。正式RTX包已再测，归一化裁剪影响须单独报告，
离线对照同时保留原始单位误差，不能只比较裁剪后的标签。


正式RTX源0完整物理NPZ与对应EGL资格运行逐值一致；单条含启动的wall96.52秒，
不据此宣称全量吞吐或固定提速倍数。
正式采集使用显式 `--isaac-observation-only --sync-observations`：MuJoCo仍200Hz物理，
Isaac在50Hz观测时更新真实link/相机并零时间渲染，不在每个200Hz子步额外推进Isaac。
2秒资格检查和完整弯腰/举箱均记录真实USD/相机姿态；Isaac时间恒0.004秒，
完整相机实际移动约0.422米，避免以RGB噪声变化冒充相机随动。默认旧分支保持兼容。

接口实测边界：既有USD腰链offset为0.044米，MJCF为0.035+0.019=0.054米，
形成约1厘米的躯干/相机资产FK偏置；肩连接有反向1厘米补偿，弯腰时腕部偏差最大约1.563毫米。
静态关节链公式解释实测偏差，残差低于1微米，未把资产差误报为观测错相。
资产和相机标定未改；训练与前后物理评估使用同一RTX路径并绑定env/bridge/模型身份。
精确证据在仓库外 `reports/simple-usd-mj-torso-attachment-audit.json` 与独立render pose报告。

## 模型闭环中的独立问题

28k标准reset seed10005的第一轮25秒闭环失败，首个模型动作前（1.005秒）已倾斜31.63度，
首个跌倒在1.405秒。该运行等待真实图像历史期间只用固定初姿PD，decoder调用为0；
它没有用source4真实积分初态，不能冒充source4的精确物理复现。
失败结果和视频保存在 `reports/liftbox_before_seed10005`，不以进程退出0或
`bounded_smoke_finished`称为任务成功。预热稳定性和固定初态对照须单独解决再判断微调效果。

新增显式 `--warmup-control live_pose_sonic_v1_1`：仅从首个live状态固定期望身体姿态和世界朝向，
前10帧收集真实proprioception，之后用官方mode0 encoder和既有decoder保持平衡，
等待完整真实RGB历史。参考的重复帧代表恒定期望，不能当作历史测量；不读取teacher未来、
不反复传送机器人、不改控制器增益。CPU完整物理1.5秒对照中，standard/source4/source8
在1.005秒旧PD倾角31.62/50.27/57.60度，新预热0.58/1.32/1.02度；新模式三组均有限且未跌倒。
记录源码、encoder、首个live参考SHA及真实decoder次数。默认旧分支兼容保留。
正式前后对照需相同新预热、相同source4/8首帧完整积分初态、独立fresh服务和固定采样seed，
不能拿旧PD失败作更换控制器后的微调收益。

后续真实状态审计又定位到手形冷启动缺口。source4/8/VAL3在1.005秒首个策略请求时，
保持实测手形的预热使10个屈曲关节平均绝对角达1.6253/1.5822/1.6268rad；
1290个保留TRAIN窗口的相同指标最大只有1.0027rad，最早完整窗约0.164rad。
原示范确有相近的初始折手状态，但因不足真实历史被裁掉；实际抓箱的490个保留窗口也没有覆盖该手形。
因此不能仅凭“teacher成功、文件字节正确”称输入分布已完整覆盖。

新增显式`--warmup-hand-policy open`，仅与live-pose SONIC及同步观测共同使用。
前10帧真实proprio/PD保持原样，之后身体参考保持首个live姿态，手14目标设为通用零位张开；
不读取teacher未来、不写qpos、不注入历史或RGB，不改物理/任务成功标准。默认`measured`保留原行为。
实际CPU重建source4/8/VAL3后，首策略时刻14个实测手关节逐维均进入TRAIN实际min/max，
手部归一化最近邻RMS从0.71917/0.68078/0.71839降至0.06072/0.05585/0.06160，
最近邻均回到TRAIN1第30帧起始阶段。三组首秒最大倾角2.539/3.372/3.043度、未跌倒，
51个50Hz测量的手箱法向力全0，预热没有替模型完成任务。
完整诊断为`resources/carrybox_source_audit_20261021/COLD_START_OPEN_HAND_POLICY_DIAGNOSTIC.json`。
肩yaw/腕等仍有2–4维超出这三条示范的范围，不能由手14结果推断全状态、RGB同分布或模型任务成功。
31k联合pilot的微调前后物理对照统一使用`open`，不与旧`measured`失败运行混算权重收益。

修正后28k基线已在source4、source8各完成25.02秒真实RTX闭环：两条均25个模型chunk，
40次预热decoder调用，首个模型指令在1.005秒；初始倾角分别1.324/1.020度，
全程最大倾角3.377/6.440度，均无跌倒、无非有限状态，但均无举箱成功、无loaded-hand接触，
最长有效保持0秒。对应结果位于 `reports/liftbox_before_source{4,8}_balanced`，
fresh服务RNG记录在同名加`_service`目录。source4/8首状态SHA分别为
`1321ed7ff5d99518b58fa823163c947dcda04d275ea69f67c9652669ae6c8405`、
`14292e019f6949f31774208df41b787c930fce0fee76cd64fe2568bd1e774767`。
每个服务在模型/DINO构造结束后重置采样seed20261021，RTC关闭、10步采样、执行30帧。
基线与本地训练曾共享4090（总显存约30GB，单个RTX实例）；仿真在HTTP等待期间暂停，
这不构成延迟/吞吐的前后性能实验，不能用后续独立运行的延迟计算提速。

## Action-only诊断轮的实际结果

已完成300次更新，单4090、microbatch4、累积4、有效batch16、LR1e-5、warmup20、seed20261021，
实际总wall1677.15秒；1290个完整历史/未来训练窗口。新optimizer/scheduler从step0开始，
未恢复A100状态。峰值本进程CUDA allocated14,900,373,504B、reserved15,460,204,544B。
最终模型在仓库外 `finetune/lift_box_hold_28000_action300_rtx_v1/training/posttrain/` 下的
`lift_box_hold_v11_action_only.us.flow1000.cosine.lr1.0e-05.b16.gpus1.261021125512122991/checkpoints/ckpt_300/model.safetensors`，
SHA256 `e91f0149aed5ea1257304d56c5f674f18ab3be3f6bcf3a3f6ffd4949787c8f09`。

独立TEST4的20个相同窗口、相同私有噪声，原始body64 L1从0.0971483降到0.0390664，
hand14弧度L1从0.196203降到0.0494086；归一化L1分别0.166380→0.0669826、0.204003→0.0512085，
flow loss3.14938→0.591515。保存的预测、targets与继承bounds已独立重算，四类L1均匹配。

ActionExpert确实更新，VLM和Motion在本轮前后冻结哈希一致。另用CPU逐tensor检查：
625个VLM张量等于28k源FP32转换为部署BF16后的值；1060个Motion张量与源FP32逐字节相同。
不能将VLM的BF16文件称为与A100原FP32文件字节相同；两边部署的有效BF16权重一致。

**物理任务仍未通过**：冷加载ckpt300后，source4/8均实际运行25.02秒、无跌倒，但仍无手部承重接触、
箱未离地、有效保持0秒。相对基线，最大躯干倾角分别3.38→25.50度、6.44→25.69度，
视频可见伸手/弯腰后空手抬起。不能用离线误差下降或动作变化替代举箱成功。
完整汇总为仓库外 `reports/liftbox_action300_paired_results.json`，独立审计见
`resources/carrybox_source_audit_20261021/BALANCED_SOURCE{4,8}_BEFORE_AFTER_INDEPENDENT_AUDIT.json`。

四条实际运行的完整初态、五十行预热物理轨迹、四十次预热token、控制/渲染协议和每次采样RNG一致。
RTX重渲像素并非完全确定：首个策略指令前截图MAE约0.122/255、0.095/255。
因此只称同协议/物理初态/采样噪声配对，不能称在线图像逐值相同、或严格只有权重一项数值差异。

## 与A100实际训练及官方v1.1配方对齐

2026-10-21已通过SSH只读核对A100正在运行的同一UnifoLM任务：实际JSON、进程参数和启动日志一致，
`tune_vlm=True`、`tune_motion=True`，DINO冻结，原生DDP/BF16混合精度及gradient checkpointing。
真实optimizer五组为：Action481,577,632参数/LR1e-4；语言骨干1,724,374,016/LR1e-6；
视觉塔306,242,560/LR1e-5；投影层100,714,496/LR1e-4；Motion399,589,008/LR1e-5。
八卡每卡batch16、累积2，有效batch256，40k更新、warmup1000。
只读证据和原配置/argv SHA保存为 `reports/a100_unifolm_actual_training_alignment.json`。
没有修改、停止、重启或使用A100训练GPU执行本地任务。

官方固定commit `cda3fdfafe47cb6fa75400d013de1bd36e3cea05` 不能笼统归为一种冻结策略：

- [SONIC v1.1 UnifoLM后训练](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/scripts/train/psi0/posttrain-psix-unifolm-g1-sonic1.1.sh#L122-L200)解冻整个VLM三组，LR同上，Action从头训练。
- [SONIC v1.1 robust微调](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/scripts/train/psi0/finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh#L85-L208)从UnifoLM后训练的VLM与Action共同初始化，继续更新同三组VLM，LR保持相同。
- [旧SIMPLE脚本](https://github.com/physical-superintelligence-lab/Psi0/blob/cda3fdfafe47cb6fa75400d013de1bd36e3cea05/scripts/train/psi0/finetune-sonic-simple-psi0.sh#L65-L119)显式`no-tune-vlm`，Action78/state43，属于另一配方。

原版没有本项目新增的独立MotionExpert。用户已要求本地与A100实际联合更新保持一致，
所以后续应启用VLM+Motion+Action五组优化，而非复用Action-only诊断轮的冻结设置。
保留本模型6-block双来源结构、256×320视觉协议、真实6帧历史与冻结DINO，不把官方12-block架构
或旧字段直接覆盖进当前checkpoint。300步小包实验也不能称为复现官方40k训练日程。

本机已完成真实的两次联合更新资格检查（单4090、microbatch1、累积1，使用五组峰值LR，
未使用warmup）。两次loss为3.50410、2.21566，五组unscale后、clip前梯度范数均有限且非零；
五组完整参数哈希均变化，原DINO文件哈希未变。实际CUDA峰值allocated23,337,993,728B、
reserved25,578,962,944B。证据为仓库外
`reports/liftbox-joint-gpu-check-v1/joint_check_result.json`。
这只证明小批量联合反传与更新路径，不能代替完整微调、全量吞吐或举箱成功验收。

随后实际microbatch4、累积1的两步扩大检查也通过，五组梯度有限、完整参数哈希变化；
CUDA峰值allocated23,564,754,944B、reserved24,989,663,232B。
只采用`reports/liftbox-joint-b4-gpu-check-v3`的真实b4配置结果；早先stdin启动失败及b1误标尝试
分别保留失败/范围更正记录，未纳入b4验收。正式累积4与完整300更新还须由正式job检查。

为容纳本机显存，独立本地入口使用`CpuMasterAdamW`：模型及梯度保持FP32，沿用生产
安全检查、逐组裁剪与scheduler，把裁剪后的梯度复制到CPU上的原生`torch.optim.AdamW`
参数及状态，再同步更新后的FP32参数到GPU；不修改A100共享trainer。
GPU实测CPU master和moments确实在CPU，未恢复任何A100 optimizer/scheduler/RNG。
小规模四步CUDA原生AdamW对照最大参数差2.33e-10；跨设备浮点计算不能称为逐位相同。

## SONIC量化接口的实际对照

补查发现官方客户端在把预测body64交给控制器前执行FSQ投影，原本地模型闭环缺少这一步。
当前已提供显式`--body-token-projection fsq_v1_1`，只处理反归一化后的learned body64；
手部、padding、训练标签、统计、预热与物理参数不变，默认`none`保留已有实验含义。
范围与步长取自已固定的官方v1.1 encoder计算图：32级、步长1/16、合法区间[-1,15/16]，
不用旧客户端的不同端点硬编码推断v1.1范围。

已在VAL源3用同一Action-only ckpt300、相同初态、预热与采样seed，分别以`none`和
`fsq_v1_1`完成25.02秒真实RTX闭环。两次均无跌倒，但手部承重接触均为0、有效保持均为0秒，
箱未举起。量化补齐没有使这一个验证任务成功，不能将它归为失败的唯一原因。
原始结果分别为`reports/liftbox_val3_continuous`和`reports/liftbox_val3_fsq_v11`。
该例最大躯干倾角25.399→12.857度；只描述观察到的动作差异，不据单例称为稳定性提升。
独立CPU汇总为`reports/liftbox-val3-continuous-vs-fsq-cpu-summary.json`，核对25次成功请求的
时间与前后RNG摘要一致。1201条实际body指令中共比较76864个元素，76794个发生量化变化，
平均绝对变化0.0158995、最大0.03125，无范围裁剪。运行未封存每条body数组，合法网格的
依据是绑定的投影源码与实际调用记录，不能称为逐token数值重放审计。
这组验证不混入前述source4/8微调前后对照，也不据单个验证初态报告泛化成功率。

## 视频归档

用户要求的完整原片及原速左右对比已另存到本机
`/home/zxc/Videos/CVPR_SIMPLE/2026-10-21_action300/`。
五份原片副本SHA与源文件一致并完整解码通过；对比版为source4的28k基线与Action-only300，
1252帧、50fps、25.04秒，标签注明任务未完成。`manifest.json`保存源位置、SHA、权重与结果；
视频和运行记录不进入Git。

接口消融视频独立为`val_source3_continuous_vs_fsq_v1_1_action300_side_by_side.mp4`，
左连续输出、右FSQ v1.1，同一Action-only300权重，均未完成任务。

## 独立联合pilot入口

`finetune_lift_box_hold_joint.py`复用已通过的原始物理/encoder/RGB数值验收，重新绑定实际
25项打包payload、5项DINO特征和源权重/统计/视觉契约的SHA，不重复执行既有数值验收。
真实TRAIN梯度仅来自源0/1/2的1290个完整历史/未来窗口；固定20个TEST窗口只在step0和最终
更新后计算，使用独立RNG，不能用于训练、选步数、估计bounds或提前停止。

正式本地预算为microbatch4、累积4、300更新、warmup20、cosine、seed20261021，
五组峰值LR与A100相同；不恢复A100 optimizer/scheduler/RNG。数据窗口访问预算与原Action300
同为最多4800次，但训练组件和LR不同，不能把结果差异只归因于解冻组件。
仅最终`model.safetensors`以staging目录保存并rename，记录配置、源码SHA和provenance；
不写巨量CPU optimizer状态，也不将模型仅保存称为可续训的完整训练checkpoint。

`--base-run`只允许同源协议的已准备快照：严格核对新旧原训练model/data配置、统计字节、
DINO契约、同一Qwen目录、保存完成记录及实际模型SHA、同step的Motion导出与实际文件SHA。
报告将实际初始化step/SHA与原示范质量的28k权重锚点分别记录；不以改JSON方式伪造新示范验收。
运行命令见[工具说明](../scripts/4090_simple/README.md)。联合训练完成后仍须冷加载最终模型，
用相同31k基线初态、采样seed、真实历史、平衡预热和FSQ协议运行完整物理对照。

31k正式job已启动于仓库外`finetune/lift_box_hold_31000_joint300_rtx_v1`，
阶段状态为`reports/joint31000-pipeline-v2-state.json`，训练日志为
`reports/liftbox-joint31000-training.log`。执行过程依次完成300步、最终模型冷加载、
source4/8各自fresh服务的31k基线与ckpt300物理对照，再汇总真实predicate和视频；
未完成时不填写任务成功结果。更早仅等待资源的协调进程因冷启动手形修正而结束，未启动训练，
其状态及原因保留；独立迁移进程和A100训练未被中断。
