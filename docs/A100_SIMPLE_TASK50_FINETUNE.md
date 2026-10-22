# A100：38k联合权重初始化，Handover任务微调20k步

**当前配方：UnifoLM38k初始化，八卡联合微调20,000步，每4,000步保存。**
2026-10-22该运行在2345次更新后因rank5 CUDA异步断言退出，尚未生成首个完整任务checkpoint。
完整38k初始化权重仍受保护；未保存的2345步不能从磁盘恢复，不能宣称无损续训或已切换融合AdamW。
使用`finetune-handover-task50-joint38k.sh`，完整命令见末尾“当前38k配方”。
下方40k／2k运行是已停止的历史试跑，记录与checkpoint保留，不作为当前初始化来源。

历史试跑曾完整继承UnifoLM 40k checkpoint的ActionExpert、VLM和MotionExpert，实际完成400次任务更新后，
按用户请求停止并切换到38k来源。当前38k配方仍完整继承三个模块，DINO冻结，
优化器、scheduler、RNG及任务步数重新开始，不新建动作头。原运行目录和用户交付数据只读保留。

## 数据和协议

交付目录为`/gaozt-test1/datasets/sigmma_dataset/G1WholebodyHandoverTeleop-v0`。
50条成功示范保留原划分TRAIN45、VAL5：29条原轨迹修正，21条共享模板0的场景再生成仅用于TRAIN。
保留全部视觉备注。示范筛选不是模型成功率，也不是官方benchmark分数；没有独立TEST。
现有物理成功门槛不保证箱体整齐进入篮腔，正式评测仍需独立官方场景。

源50Hz数据的state45为手、手臂、腿/腰、颈占位；40k模型state43为腿、腰、手臂、手。
`prepare_handover_task50.py`按原40k info中的具名关节重排，原状态另存`source_state45`，模型补齐45维。
手14和`body_token_v1_1`64保持原执行值，同步选择`ceil(k*50/30)`的真实RGB、状态、动作。
名义时间为`k/30`，实际源时间另存，最大偏移13.334ms。视频无损保留选中H264解码RGB并逐像素验证。
不重编码token，不插值动作，不宣称30Hz抽样后控制轨迹重新通过了物理验收。
原50Hz成功证据只绑定复用，不重复物理或数值验收。

派生数据在`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz`。
归一化明确继承原40k的`dataset_statistics.json`和bounds，不使用交付包新拟合的bounds。
metadata描述统计仅按具名顺序投影，原统计封存溯源；TRAIN/VAL的原bounds覆盖情况写入检查报告。
冻结DINO使用原权重、256x320几何，每集保存30Hz，读取过去一秒6个5Hz时刻。
首秒只供真实历史，末尾不足30个真实未来动作的窗口排除，不复制首帧或末尾动作。
VAL固定每集8窗，共40窗，八rank等分无补齐；验证使用独立固定RNG且还原训练RNG。

## 历史40k／2k试跑配方（已停止）

- 八张A10080GB，DDP，每卡batch4、累积1，有效batch32。
- FP32 master/AdamW、BF16 autocast、gradient checkpointing，DINO冻结。
- 2000次optimizer更新、100次warmup、cosine，任务步数从0开始。
- 峰值LR：Action1e-5、语言骨干1e-7、视觉塔1e-6、投影1e-5、Motion1e-6。
- 保留生产全rank loss/梯度有限性检查和分组clip1.0。
- 第2、20次更新额外检查每个可训练tensor前4个值跨rank相等，不冒充完整参数逐字节比较。
- 每200步及第20步保存完整checkpoint，最多保留12份，不清理原40k权重。
- 沿用生产验证语义：完成第1、201、401……次及最终更新后验证，验证201不能冒称checkpoint200成绩。

实际窗口数和覆盖遍数见`data_check.json`；这是一轮初始预算，不是最优参数结论。

## 共享环境与历史40k准备命令

独立Conda prefix：`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126`，
不共享原venv可变site-packages。版本和CPU权重核验在仓库外
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_40k_preparation`。

```bash
conda activate /gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126
python scripts/4090_simple/prepare_handover_task50.py \
  --source-root /gaozt-test1/datasets/sigmma_dataset/G1WholebodyHandoverTeleop-v0 \
  --state-contract /gaozt-test1/datasets/UnifoLM/unifolm_sonic_lerobot_train/meta/info.json \
  --output /gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz

# 两个split分别提取DINO；下面为TRAIN。
python VLMExpert/scripts/data/extract_unifolm_dino.py \
  --dataset-root /gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz/Handover-sonic11-train \
  --output /gaozt-test1/datasets/SIMPLE_sonic_v1_1/dino/handover_task50_30hz/Handover-sonic11-train \
  --model-path /gaozt-test1/datasets/MixEgoMotion/weights/dinov2-with-registers-large/e4c89a4e05589de9b3e188688a303d0f3c04d0f3 \
  --device cuda:0 --batch-size 32 --storage-dtype float32 --inference-precision bf16

# 以下40k命令只用于历史复现；当前任务使用末尾的joint38k命令。
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh prepare
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh check
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh run
```

`prepare`生成可复现argv/配置/来源；`check`实际读取DINO和LeRobot样本、检查关节顺序、真实历史及完整未来；
`run`重核文件及GPU空闲后启动，不停止其他训练。已有目录不重复启动，可用`TASK50_RUN_DIR`指定新目录。
旧UnifoLM后训练脚本默认新建动作头，不应用于本任务。
已接受的CPU数据检查仅重绑文件身份，不重复解码或数值验收；父进程和八卡worker使用同一绝对配置路径，
避免工作目录不同使未使用的DeepSpeed配置字段也产生不一致。首次命名失败发生在创建worker之前，0更新记录保留。

## 损失和剩余时间

当前38k配方输出目录为
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_38k_preparation/handover_joint38000_b32_s20000`。
历史40k输出保留在同级`task50_40k_preparation/handover_joint40k_b32_s2000`，两轮文件不混合。

- `run_config.json`、`argv.txt`、`preparation.json`：完整配方与来源。
- `data_check.json`：实际读取、有效窗口、bounds覆盖、文件绑定。
- `training_state.json`、`train.log`：生命周期及控制台。
- `progress/losses.csv`：完成更新数、TRAIN/VAL loss、五组LR。
- `progress/loss.png`、`progress/progress.json`：损失曲线、耗时、ETA及预计完成时刻。
- `progress/monitor_health.json`、`monitor.log`：监控自身状态。

监控只读offline W&B。warmup结束且至少有5个真实更新间隔后，按最近墙钟耗时估计ETA，
包含已观察的验证/保存暂停。未开始、失败及最终收尾时不伪造ETA。训练和监控退出码分别保存。

```bash
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint38k.sh status
```

## 历史40k来源与试跑记录（已停止）

历史40k模型12,066,487,572字节，SHA256为
`ff1b6ea8c63edae671220b443ed08ae11ef8efe07cd83cdcc0f90233f3462dc9`。
真实CPU生产加载通过：1944个stored tensor逐值/dtype/shape一致且有限，两项共享alias按实际共享关系恢复。
该检查0更新、未建optimizer、未用GPU，不替代本任务八卡或物理验收。
训练是否启动、实际loss和任务表现以各自运行记录为准，不因安装或loss下降宣称测评通过。

2026-10-22已从上述40k初始化并实际启动八卡，源代码为`324fea3`；
实际Python3.11.15、Torch2.7.0+cu126、Transformers4.57.1、FlashAttn2.7.4.post1、
W&B0.25.0、Matplotlib3.10.7，驱动535.129.03。原生产包版本保持不变，仅补六项纯CPU绘图库。
TRAIN13808窗、固定VAL40窗实际读取通过。第2、20次更新五组梯度有限且非零，
每rank7700个参数探针一致。旧任务按用户切换请求在400次更新时停止：
`ckpt_20`和`ckpt_200`完整保存，400步保存被停止信号打断，`ckpt_400`不完整，不能加载或续训。
瞬时显存约61.3–62.5GB/卡，不冒充全程峰值或吞吐验收。
早期GPU与监控证据摘要见[a100_task50_joint40k_validation.json](a100_task50_joint40k_validation.json)。

本机查看快照位于`/home/zxc/CVPR_Evaluation_runtime/reports/task50_handover_current`，
CSV、PNG和进度JSON仅绑定历史40k run，原始训练日志和checkpoint保留。两机系统时钟实测约有159秒偏差，
不调整运行中机器时钟，ETA时长始终使用A100自身的墙钟间隔，不把跨机复制时间混入估算。
以上是历史启动和早期检查记录；原2000步预算未跑满，不能写成训练完成或物理任务验收通过。

## 当前38k配方：任务训练20k步

2026-10-22用户将初始化来源改为UnifoLM的38,000步联合checkpoint，并要求任务微调
20,000次optimizer更新、每4,000次保存。此前40k配方及其运行记录保留作为历史；
35k曾被提出，但原训练滚动保存中已不存在该checkpoint，未使用其它权重冒充35k。
当前明确使用同一原UnifoLM运行目录下的`checkpoints/ckpt_38000/model.safetensors`。
38k来源的真实CPU严格加载已通过：1944个stored tensor逐值、dtype、shape一致且有限，
Action259、VLM625、Motion1060，两个共享alias按同storage恢复。模型12,066,487,572字节，
SHA256为`12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7`。
同文件系统下的`protected_source38000/model.safetensors`另有硬链接保护，未复制原optimizer或RNG。
原38k保存完成标记、配置、argv和统计均已绑定；这次严格加载为CPU检查，不代表物理成功。
新八卡任务已于A100时间2026-10-22 08:03:31 UTC启动，代码为`196ec73`，
任务时间戳`261022080240`。生产初始化收据确认38k完整继承、全新优化器和任务步数从0开始。
旧40k试跑的loss、健康检查或checkpoint不作为新38k任务的运行结果。

40k初始化的旧任务因用户切换请求发送SIGINT停止，实际完成400次更新。
`ckpt_20`和`ckpt_200`有生产保存完成标记及完整模型、优化器、调度器和八rank RNG；
停止信号打断了400步checkpoint的模型保存，`ckpt_400`仅保留部分rank RNG文件，
不是可加载模型或可续训checkpoint。全部旧文件原样保留，停止审计为旧准备目录下的
`stop_by_user_audit.json`。这次停止来自用户切换，非loss、梯度或CUDA数值失败。

新任务从38k的ActionExpert、VLM、MotionExpert完整参数重新初始化，任务更新数从0开始；
不继承原UnifoLM或旧40k任务的optimizer、scheduler、RNG。DINO继续冻结。
八卡batch4×累积1、全局batch32和五组峰值LR保持上述配方：Action1e-5、
语言骨干1e-7、视觉塔1e-6、投影1e-5、Motion1e-6。新的cosine预算为20,000更新、
1,000更新warmup；每4,000更新保存完整checkpoint，早期20步保存关闭。
八卡第2、20更新健康探针仍保留。验证仍为固定40个真实窗口，沿用生产触发语义，
在完成第1、1001、2001……次及最终更新后执行，不将验证步号冒认为checkpoint成绩。

沿用已接受的45 TRAIN／5 VAL、30Hz派生数据和18,025帧DINO缓存，不重新采集、
转换或提取特征。新旧模型协议（除权重路径）、完整data配置、归一化和源metadata一致，
重绑1331项已验收文件身份，复用原CPU数据检查，仅重算20k预算对应的覆盖遍数。
证据为新准备目录的`DATA_ACCEPTANCE_REUSE.json`，未重复已通过的数值验收。
初始化与归一化来源明确绑定38k同源运行的配置和统计；
数据metadata描述统计不替代模型原归一化。当前任务预算约为
`20000×32/13808≈46.35`遍有效TRAIN窗口，数据窗口高度相关；
该预算是用户指定，不代表已证明最优，必须结合独立VAL和后续真实物理表现评估。

新准备根目录：
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_38k_preparation`。
38k独立CPU权重审计保存为该目录下的`joint38000_cpu_audit.json`；
新任务准备目录为`handover_joint38000_b32_s20000`，不覆盖旧40k任务。
独立Conda prefix与原数据、DINO路径继续复用。

```bash
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint38k.sh prepare
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint38k.sh check
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint38k.sh run
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint38k.sh status
```

新目录仍记录`train.log`、完整配置及来源、`progress/losses.csv`、`progress/loss.png`、
`progress/progress.json`和监控自身健康状态。ETA须等待新任务warmup结束和足够真实更新间隔，
不把旧任务的耗时或进度接到新任务上。停止的旧任务清空剩余时间估算，并保留用户切换原因。
配置、权重加载、GPU健康、20k训练完成及模型物理成功是分别验收的阶段。

新38k任务第2、20次更新的八卡检查已通过：五组梯度有限且非零，每rank的7700个
参数探针完全相同。探针取每个可训练tensor前4个值，不代表全部参数逐字节比较。
两次TRAIN loss分别为11.513908和6.694449，使用不同batch，不能单独解释为收敛；
首次固定VAL在完成第1次更新后记录loss12.501106。后续NVML快照每卡有一组
62,446–62,532MiB的主要上下文，另有约12,960MiB的上下文，总设备占用75,418–75,504MiB。
容器`/proc`与NVML的host PID不同，额外上下文归属尚未确定，未停止或修改其他进程。
可见每rank的Inductor编译子进程持有继承的GPU设备句柄，这可能相关，但尚无直接PID映射证据。
以上为早期八个训练rank及数值检查通过时的快照；该资源现象单独记录，不把16个NVML条目当作16个训练rank，
不将早期显存视为全程峰值或吞吐验收。完整20k训练及微调后的物理效果尚未验收。
本次启动摘要见[a100_task50_joint38000_validation.json](a100_task50_joint38000_validation.json)。
当前38k的本机查看快照单独存放于
`/home/zxc/CVPR_Evaluation_runtime/reports/task50_handover_joint38000_current`；
它是带复制时间和SHA的早期快照，原始CSV、PNG和最终failed进度保存在A100新准备目录下。

## 融合AdamW：验证后受控切换

用户授权先验证数值和显存，确认通过后保留当前进度切换。原运行在仍使用普通AdamW时故障退出，
不得因为合成小测试通过就直接重启或修改参数。新增工具不会默认停止任何训练。

`benchmark_fused_adamw.py`已经完成单张A100、有限显存的优化器测试：
五组LR、原betas/eps/weight decay、非零动量和逐参数计数、多步含无梯度参数均通过。
两分支合计PyTorch峰值242.21MiB；97步后最大参数差异5.96e-8，计数精确相同。
1.34ms到0.68ms仅为7.28M元素的小样本优化器测量，且与原训练并发，不能当作整网提速。

切换工具使用以下独立路径，不覆盖原运行使用的196ec73源码或旧run：

1. `checkpoint_pause_guard.py`先只读绑定八个worker的PID及启动身份，默认dry-run。
   `watch --execute`等待原任务自然保存4000步；完整文件和保存完成标记通过后短暂停住八rank。
   两次只读栈检查必须证明尚未提交4001步，未知或错过边界会自动恢复原训练。
   独立watchdog租约到期、控制器退出或检查失败都会放行，工具自身不发送终止信号。
2. 只有有效`held_verified`证据再次核验后才能停止该原任务，并硬链接保护完整checkpoint。
   不删除旧记录，不将未完成保存或旧40k试跑的权重用于这次恢复。
3. `validate_task_fused_step.py`在GPU释放后使用真实batch4和完整模型/Adam状态，
   双后端预热后每次重新加载相同checkpoint和RNG，交替执行对照。
   比较全部梯度、参数、Adam两阶状态及逐参数计数，记录整步显存和有限性。
   这是单GPU对照，八卡梯度平均、数据恢复和实际吞吐另行验收。
4. `a100_task_fused_resume.py prepare`只接受绑定相同模型、优化器、scheduler和配置SHA的
   完整GPU验收与小测试报告；`run`恢复到新日志段，任务总预算仍20k，保存仍每4k。
   首先核对实际模型、全部Adam状态、scheduler及八rank主进程RNG，然后才选择fused执行。
   计数不重置，学习率不重启warmup，DINO、三专家反传、batch32及有限性检查保持原配方。

恢复修正了实际发现的两个问题：本任务prepared loader实际每rank431批/轮，原通用估算为432；
普通checkpoint又没有保存独立shuffle生成器、worker增强随机流和预取队列。
`replay_task_data.py`因此从原始seed和第0轮开始CPU重放全部已消费批，保留原四worker、
预取设置和同一个iterator，并在重放后恢复模型侧CPU/CUDA RNG，避免只跳索引改变后续增强。
这段重放耗时单独显示，新段监控以恢复步数为下限，旧CSV不混入新段ETA。

切换后还需八卡第2、20次连续更新检查及实际墙钟耗时对照。有限数值容差通过不代表逐位一致，
也不替代完整20k训练和后续物理验收；原任务未执行融合切换，后续独立验证结果见下文。

### 首次保存前的故障与恢复边界

原运行在A100时间09:12:52 UTC报告`TensorCompare.cu:112 _assert_async_cuda_kernel`断言，
rank5随即SIGABRT，torchrun终止其余worker，监督器09:12:56记录failed。
控制台最后完成2345次更新；checkpoint目录为空，没有完整模型、Adam状态、scheduler及八rank RNG可续接。
CUDA异步栈不能定位最初失败的模型算子，不能凭此认定为数据、优化器或某项并发操作导致。

4000步守卫在等待期间发现任务failed而退出；没有lease或held证据，未进入暂停分支。
最初一次守卫因绑定文件名写错在启动检查阶段退出，日志独立封存；修正后的等待也没有暂停训练。
故障证据保存在A100的`fused_adamw_validation/failure_2345_snapshot`和本机
`/home/zxc/CVPR_Evaluation_runtime/reports/task50_handover_joint38000_failure2345`。

同期独立CPU取数探针50秒内读取15批，确认真实shape和每rank431批。
首批等待27.79秒，后14批平均0.637秒；按该短前缀粗算4000批约43分钟，
不是八rank重放实测或耗时承诺。探针采用独立进程组且禁用CUDA，但没有同期CPU/IO峰值记录；
原训练末段也出现变慢，不能用事后空闲快照证明不存在间接资源影响。

原任务故障后，另外的UnifoLM固定验证作业自然完成并释放GPU，未停止它。
后续整模型AdamW对照使用硬链接保护的完整UnifoLM38k模型和原Adam状态；
这可以检查融合执行的数值和显存，不能替代已经丢失的任务2345步状态或证明无损恢复。
重新开始任务必须明确记录从38k初始化及重算范围，不能借用旧40k任务checkpoint冒充续训。

首次整模型单GPU对照已执行且未通过原严格门槛：两后端loss均为1.647866249，
但更新前的preclip梯度最大绝对差为0.0078125；随后参数最大差1.19209e-7，
Adam状态最大差1.16185e-5。所有值有限、无OOM，首组计时步骤的PyTorch allocated峰值
普通46.58GiB、融合44.26GiB；单GPU结果不代表八卡峰值。
由于梯度差在优化器更新前已经出现，不能将此次差异直接归因于融合AdamW。
原失败报告和容差保留，尚未进行连续三步融合检查，也未因此放行正式切换。
下一步分别检查普通后端重复性，以及使用逐值相同的实际梯度隔离AdamW执行差异；
两类结论分开记录，同梯度后端比较不覆盖原独立反向对照失败。

`validate_task_fused_step.py --mode controlled --microbenchmark-report <已验收小测试报告>`
保留原数值容差，增加standard对standard重复性控制、两组交替同梯度对照和连续三步真实反向。
在更新前分别核验preclip和postclip梯度逐值相同；任一postclip差异都会阻止该优化器更新。
该模式成功状态为`controlled_optimizer_comparison_passed`，旧strict模式及其恢复入口不会将它误认为
`full_step_comparison_passed`。报告仍保留独立反向的重复性结果及未知原因。

用户已接受测试通过后从38k重新初始化、重算未保存的2345步。
拟重跑仍是20k任务更新及原五组学习率、batch32；每4k正式checkpoint保留，新增20步及每1k恢复点，
仅滚动保留新运行最近两次保存并豁免所有4k正式点。启用独立tmpfs中的每rank双槽CPU输入/RNG转储，
诊断模式与存档停顿须计入实际耗时，不能把独立优化器测量冒充正式任务提速。

新入口`a100_task_fused_restart.py prepare`只接受绑定旧准备配置、38k模型及实际微基准SHA的
受控GPU报告；`train`在CUDA模型和全新空Adam状态准备完成后选择fused，保持任务计数从0开始。
`run`使用独立锁拒绝重复启动，在唯一tmpfs目录保留每rank两槽输入/RNG，启用
`CUDA_LAUNCH_BLOCKING=1`以帮助定位上次未知native故障，并记录实际源码SHA和诊断环境。
第2、20次更新既保留原参数探针，也核验八rank真实fused/foreach配置和CUDA显存峰值。
