# 4090：UnifoLM 联合权重的 CarryBox 本地适配

日期：2026-10-21，北京时间。用户在原20k测评准备后，进一步授权使用A100最新完整保存的
UnifoLM联合权重，在本机用SIMPLE训练示范适配，再查看效果。使用训练示范的训练部分更新参数；
独立保留的TEST集只作前后对照，不用于更新、选checkpoint或调参。
原20k资源、A100正在运行的训练及其optimizer/RNG均保留。

## 当前放行状态

**尚未完成本地微调，当前更新次数为0。旧 `carrybox_v11_30hz` 派生包不具备训练放行资格。**
该包的数值、图像和DINO检查通过，仅证明转换一致，不能证明重新生成的动作命令能完成任务。
`finetune_carrybox.py --train` 在入口和训练函数两处拒绝启动，没有强制绕过参数。
下列旧包转换/DINO/prepare命令用于复现准备过程，不构成可直接启动训练的配方。

原始50条示范不能笼统定性为低质量：独立几何/FK/视频阶段审计中，49条有清楚的放桌、
释放和稳定保持；剩余1条靠桌边，接触与稳定性证据须单独保留。记录无逐步接触力和显式
success，因此这不是官方成功率统计。原始具名FK与记录腕位置的全50集RMSE约
3.3–3.8e-7米。当前主要问题是跨版本控制语义迁移，不能归咎于原始示范整体质量。

## 权重与版本身份

本次冻结的来源是2026-10-21 02:25 UTC检查时最新完整保存的 `ckpt_28000`。
先核对训练保存完成日志和完整checkpoint文件，再在训练目录之外保护model-only快照，
经SSH/rsync拉取；未停止、重启或修改A100训练，没有把远端GPU用于本机推理或适配。
`model.safetensors` 为12,066,487,572字节，SHA256：
`14396d45723cf2e1a036487bea7987f6a4ac1dc292ff4da9db601da69c9b2716`。

本机来源为 `/home/zxc/CVPR_Evaluation_runtime/resources/latest28000/source`，
部署/初始化目录为其同级 `run`。`prepare_joint_snapshot.py`检查完整model/data配置及
归一化统计与先前资源协议相同，仅重映射本机路径；Motion参数从28k联合权重重新导出，
没有复用20k Motion参数。Qwen config/processor与DINO使用身份相同的既有本机资源。
28k真实GPU严格加载和在线DINO推理已通过，batch1、单GPU、输出30×80，
这是模型检查，不能替代CarryBox物理任务结果。

实际使用的是官方 **NVIDIA GEAR-SONIC v1.1**，固定发布revision
`6733128a3d8a523b1418b06bca3cdf61c8b0987f`，不是旧版SONIC。

- encoder SHA256：`fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae`。
- decoder SHA256：`34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509`。
- 官方Psi0固定README和 `posttrain-psix-unifolm-g1-sonic1.1.sh` 将该版本与
  `action.body_token_v1_1` 对应；源证据存于外部 `resources/source_contracts`。

原UnifoLM token没有逐值重编码，因此保留数值追溯限制，不能据此说版本未知。
本次CarryBox标签则由上述确定的encoder实际重新生成并核验，未把旧token字段改名。

## 数据与本地任务

下载并核验官方SIMPLE训练示范，固定[psi-data revision](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/e78fb93cc28912a3031a10b8656d32d7f0a2b867)。
Pick/Handover示范缺少可直接追溯的v1.1标签和完整root姿态，不能伪造编码所需状态。
CarryBox含真实43关节状态、root位姿/速度与手部目标，故选择
`G1WholebodyXMoveBendCarryBoxSonic-v0` 作为首个适配任务。

来源50集、57,989帧、50Hz；归档177,653,611字节，SHA256
`c8b31cc1ea61a653210278894a33555b735dd33568c4c5ad97ee48ee4871d1af`。
转换同时绑定归档及实际解压文件SHA；原始匿名关节列以官方记录器、具名状态/动作及
MJCF限位交叉确认，不能使用旧转换脚本中不一致的分组顺序。

`prepare_carrybox_v11.py` 在CPU上执行固定官方C++协议：encoder输入1751、mode0、
原生50Hz未来10点（间隔5帧），具名body29的绝对弧度、按实测时间差分的速度、
当前root heading下的未来旋转。速度是数值派生量，不宣称源数据记录了逐关节速度。
不使用旧 `action.body_token`；不足0.9秒真实未来的末45个原生帧不生成标签。
输出按 `floor(n*50/30)` 同步选择RGB、状态与手动作，保留原episode/frame/timestamp。
双手从thumb/index/middle显式重排为训练的thumb/middle/index。

按整集固定seed20261021划分40 train、5 val、5 test，输出为
`resources/carrybox_v11_30hz/{train,val,test}`：26,969 / 3,097 / 3,398帧，总33,464帧。
训练、验证、测试的源episode无交叉；RGB采用lossless重封装，全帧与源解码RGB逐值相等。
LeRobot的 `info` 和DINO提取器的 `video_info` 描述符保持同值，并留存兼容修订前后SHA。
派生数据是显式30Hz，不把它描述为原始采集30Hz；DINO提取器不再进行采样。
首1秒无完整六帧历史的样本由既有MotionHistoryIndex拒绝，不复制当前帧充当历史。
保留A100统计与bounds，不用SIMPLE或TEST数据重估归一化。

本地任务按固定官方CarryBox代码补入既有SIMPLE快照，不整体更新第三方工程。
物理箱体尺寸、质量、摩擦与源任务一致；Isaac使用本地解析网格，外观不保证与所有原示范一致。
本地成功判据明确采用真实桌面上表面、包括手掌的无接触释放与连续0.9秒仿真时间，
修正了源任务中桌面中心高度与累计时间的问题。因此该指标属于本地任务适配，不能直接称为官方榜单分数。
几何、接触、固定初态与回归说明见 [CarryBox任务说明](../VLMExpert/third_party/SIMPLE/docs/CARRYBOX_LOCAL.md)。

## 执行顺序

使用既有Conda `cvpr-simple-model311` 与 `cvpr-simple-sim310`，所有运行产物在仓库外。
模型环境本次补装ONNX Runtime1.23.2，仿真环境补装PyArrow20；均通过pip check。
Python3.11.17 / 3.10.22、NumPy1.26.4、Torch2.7.0分别使用cu126 / cu128，
不升级宿主570.211.01驱动。本机4090实际报告49,140MiB，不据此推算普通24GB显卡可训练的配置。

```bash
conda activate cvpr-simple-model311
source scripts/4090_simple/local.env.example.sh
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python scripts/4090_simple/prepare_carrybox_v11.py \
  --source "$CVPR_SIMPLE_RUNTIME/resources/simple_finetune_sources/G1WholebodyXMoveBendCarryBoxSonic-v0" \
  --sonic "$CVPR_SONIC_DIR" \
  --contract "$CVPR_SIMPLE_RUNTIME/resources/simple_finetune_sources/CARRYBOX_STATE_ACTION_CONTRACT.json" \
  --output "$CVPR_SIMPLE_RUNTIME/resources/carrybox_v11_30hz"
```

已存在且核验通过的数据不重复转换。DINO对三个split分别生成clean256×320、CLS+4register缓存：

```bash
python VLMExpert/scripts/data/extract_unifolm_dino.py \
  --dataset-root "$CVPR_SIMPLE_RUNTIME/resources/carrybox_v11_30hz/train" \
  --output "$CVPR_SIMPLE_RUNTIME/dino/carrybox_v11_30hz/train" \
  --model-path "$CVPR_DINO_DIR" --video-key observation.images.egocentric \
  --device cuda:0 --batch-size 32 --workers 4 --inference-precision bf16
# val/test各以其同名目录独立执行；GPU工作串行调度。
```

适配入口严格加载三个已有expert，重新建立任务optimizer、scheduler和step0；
与A100继续训练区分。首轮pilot只更新ActionExpert，冻结VLM、MotionExpert和DINO，
默认单卡batch4、累积4、有效batch16、LR1e-5、20步warmup、300次更新。
使用既有训练安全检查与spawn workers，仅保存最终checkpoint以控制磁盘占用。
TEST选五集各四个完整历史/未来窗口；更新前与最终更新后使用同样输入和私有固定噪声，
比较FM loss及body/hand归一化和原单位动作L1。离线误差不等于物理任务成功。

```bash
python scripts/4090_simple/finetune_carrybox.py \
  --base-run "$CVPR_SIMPLE_RUNTIME/resources/latest28000/run" \
  --dataset-root "$CVPR_SIMPLE_RUNTIME/resources/carrybox_v11_30hz" \
  --dino-cache-root "$CVPR_SIMPLE_RUNTIME/dino/carrybox_v11_30hz" \
  --output "$CVPR_SIMPLE_RUNTIME/finetune/carrybox_28000_action300" --prepare-only
# 当前旧派生包物理质量不合格，--train 被明确禁止；不以 prepare 通过放行。
```

数据准备、teacher重放、模型检查及实际适配/闭环分别记录，不能以安装成功、
短时前向或teacher轨迹的成功代替学习策略成功。实际结果与失败边界见下节。

## 当前实际记录

已完成：28k模型严格GPU加载/真实历史推理；全50集CPU转换和逐帧核验；
独立teacher参考控制器物理重放至真实source尾部24.52秒。
teacher为真实MuJoCo物理、50Hz官方decoder/200Hz力矩PD，仅开始时设置一次源状态，
先收集0.2秒真实本体历史；没有逐帧强制机器人或物体姿态。
重放身体跟踪RMSE0.0706rad、未跌倒，能举起箱体但随后掉落，未完成本地放桌判据。
CPU FK/接触审计显示前期末端偏差主要来自根位置漂移，未发现初始具名状态映射错误，
未据此盲目调整增益。原始示范自身手目标与实测状态也有较大差异，不能只看手关节RMSE判错。

28k探索性MuJoCo和RTX各运行25.02秒，均执行1201次decoder，均未成功且触发过倾斜阈值。
这些运行的采样RNG受前一个episode影响，保留为准备阶段记录；不作为等噪声前后配对结果。
后续配对须为每个episode独立启动服务，并在模型构造后设置一致采样seed和记录RNG身份。
全量DINO缓存已完成50集，真实full pack的prepare-only配置与身份检查通过。
本机首次试训在参数哈希审计遇到0维buffer兼容错误，更新次数为0，已退出并保留失败记录；
该实现问题另行修复，不将此运行计为已完成微调。

用户随后明确要求先解决数据质量，因此先停止推进训练，优先验收动作标签的控制语义和物理任务。
目前只能确认派生标签格式、数值、图像和历史对应正确，尚未确认它们在v1.1下可稳定完成任务；
不得把上述CPU/DINO检查作为训练数据最终放行依据。
官方 `teleop_wbc.py` 记录 `observation.eef_state=FK(measured_q)`，
`action.eef=FK(controller_q_target)`；它们是不同字段。
从测得关节轨迹生成mode0 reference，与源示范实际下发的reference command不是同一个量。
source0在同样737个选中时刻，源 `action.body_token` 与新编码token的平均绝对差为0.1863，
但不同模式的latent差异本身不证明原数据错误或原版本身份。
已将原始50Hz command重放、原生50Hz新编码及派生30Hz新编码进行受控对照；
原token保持原字段和未核实的身份标记，仅作诊断，不能改名当作v1.1训练标签。
另对原示范自身的举箱、放置、释放和保持阶段作独立检查，不以 `next.done` 当成功。
在此之前不启动后续微调，A100训练不受影响。


## 控制迁移质量修复记录

所有运行均只在开始时设置一次源初态，随后使用真实200Hz物理、50Hz decoder和力矩PD；
不逐帧写入机器人/物体状态。新参考只使用TRAIN源0/1/2定位问题，未用TEST选择控制参数。

- 原始旧 `action.body_token` 交给v1.1 decoder，仅作明确标注的兼容性诊断，
  source0未能举箱；不得作为v1.1标签训练。
- 单纯将实测q/dq重编码为v1.1 mode0仍失败。引入**实时实际pelvis heading反馈**后，
  source0完整原生50Hz重放成功；在生产30Hz command、50Hz decoder零阶保持下也成功。
  这里生成的是新的闭环参考命令，不声称还原缺失的原始command。
- 发现 `mj_step` 后qpos已更新但派生xpos/contact仍处于上一物理子步，存在5ms错相。
  新增显式 `--sync-observations`：同qpos调用mj_forward刷新，不推进时间，
  Isaac按同一权威MuJoCo位姿同步、零时间render；历史默认行为保持兼容。
- 同步后的source0/1/2各完整跑至真实source尾部：只有source0成功；
  source0真实连续放桌松手保持2.5秒、未跌倒，source1/2举箱后失去抓持。
  单条成功不能说明全部迁移问题已解决。source1/2较大的末期根位置误差发生在掉箱后，
  正在按抓持阶段的腕位置和控制目标定位，不能简单把末期根漂移当成唯一原因。
- 固定官方SONIC默认记录配置的部分手指限矩与当前MJCF不同；该静态证据尚未逐集绑定采集runtime。
  新增显式 `recording_0p7_nm` 配方对照；
  source1仍失败，未把单独改限矩视作解决。默认行为未被偷偷更改。
- 以源实测FK的世界头/双腕目标重新构造mode1、相对实际pelvis反馈，source1仍失败，
  该实验保留为失败诊断，不打包训练。

运行证据位于仓库外 `reports/carrybox-command30-sync-source{0,1,2}`、
`reports/carrybox-command30-sync-hand07-source1`、`reports/carrybox-command30-sync-vr-source1`；
每次包括身份、完整trace、实际指令事件、真实仿真时间、精确可重载MJB、轨迹和视频。
原始source独立审计位于 `resources/carrybox_source_audit_20261021`。

新采集器 `rollout_carrybox_dataset.py` 保存**同一真实tick的RGB、实际state和已执行v1.1命令**；
不能将旧source RGB与新重放状态拼接，也不能事后重编码实测状态充当原动作标签。
command30事件在50Hz物理观测tick上按 `ceil(50*n/30)` 触发，实际间隔40/40/20ms，
保留名义与真实仿真时间；每6个事件覆盖真实0.2秒，可供六帧5Hz历史验证。
RGB使用PNG原始封存，打包lossless x264rgb后逐帧像素核验；预览MP4不作为训练RGB。
packer重新核验50Hz零阶保持、真实encoder输入/命令、MJB接触松手连续0.9秒，
并要求固定完整pilot的train/val/test均合格。当前尚无完整合格pilot，未生成新的已放行数据包。


RTX同步验证：`reports/carrybox-command30-sync-rtx-source0` 全程物理轨迹与对应MuJoCo渲染版
的全部NPZ数组逐值一致，真实RTX视频1237帧，放桌松手2.5秒、未跌倒。
此运行同时发现Isaac关闭会直接结束进程，导致旧外层finally中的raw recorder未封存；
该capture保持incomplete，不手工补写通过。封存已移入环境关闭前的内部finally，
异常和直接退出路径有独立回归；新实际采集仍须补验。


进一步的有界腕IK与源command-FK mode1对照也未修复source1抓持，均保留失败记录，
没有将其作为训练数据。用户允许先定制简单任务后，新增了独立的
[LiftBoxHold任务](4090_SIMPLE_LIFTBOX_FINETUNE.md)，原CarryBox判据和禁训旧包保持不变。
