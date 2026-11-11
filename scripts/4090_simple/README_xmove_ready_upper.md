# XMovePick 通用上肢准备姿态：独立诊断与实测

`ready_upper_v1` 是显式预热协议实验。它保留官方原生零关节重置、桌/目标/干扰物姿态、材质、seed、原生9cm成功判据和800 tick总预算，但将原50 tick预热的后40 tick换为中性mode1参考。它不是原始zero-pose/mode0预热，结果不得混入此前13/30或RTC关闭30例分数。

原生scene身份仍为`native_saved_scene_observed_v1`；新worker、冻结输入、单例结果和最终汇总均明确`warmup_profile=ready_upper_v1`（worker字段`pick_warmup_profile`）、`original_official_initialization=false`、`native_saved_scene_initialization_preserved=true`与`warmup_protocol_modified=true`。这是准备姿态诊断，不是heldout benchmark、训练或策略改进验收。

## 固定控制方案

- 原生零关节重置保持，禁止DEFAULT冷启动、物体移动、qpos注入或关闭碰撞。
- 前10 tick使用原实际测量PD，形成真实SONIC历史；后40 tick使用原SONIC encoder/decoder与官方中性CPU planner。
- 在运行任何case前固定左右肘目标0.6rad、左右肩roll目标+0.3/-0.3rad，其余关节参考保持首次实际测量值。只在独立`MjData`里FK生成双腕/头三点参考；不修改实态。
- 固定平滑系数`smoothstep(clip((tick-10)/39,0,1))`。所有50个真实参考、初始43关节、40个实际encoder body64和14手指目标均保留；不在策略阶段调用准备姿态逻辑。
- 仍只有原生reset的1个物理子步+50×4=201个预热子步。完整相机历史来自实际运动和正常渲染，不复制图像或伪造历史。
- planner seed固定20261027，模型SHA固定`39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea`。训练中位数只用于独立离线误差分析，helper不读取训练episode、教师状态、PICO、目标位置或奖励。

## 已完成CPU证据

独立运行目录`/var/tmp/cvpr-xmove-failure-cpu-20261111/ready_upper_v1/`保存4代表×3对照与固定preset全30场景的201真实物理子步。全30无robot-target/table/distractor/self正力接触，无跌倒，最低pelvis高0.7882629m。目标仅原生自然沉降，最大1.0482mm；同阶段50个post-forward控制帧与原预热目标轨迹最大坐标差4.36e-10m。201未刷新派生量样本与旧private-FK样本存在时序差，不将其约0.308mm的表观差误称为场景改变。

已交付99条训练episode首个packed状态的中位数仅作离线比较：上肢14关节RMSE原mode0约0.3534rad、中性mode1约0.2780rad、固定ready约0.1087rad。真实末端肘约0.584/0.625rad、肩roll约0.395/-0.407rad。分布误差不是成功率指标，GPU结果见下节。

## 2026-11-12 GPU四例实测

冻结源码`6a66b4e8021e7ed4e108775cc5dcb43b2eca1174`的5000步、RTC test_time/all/15试测已正常退出，**4/4成功**。按固定顺序L0c06、L0c00、L2c07、L0c04，实际控制步数312、295、418、333；原相同四例为失败、失败、失败、成功，即1/4。每例50 tick预热、总800 tick上限、原生约9cm成功判据保持，真实RTC与送入控制器的动作均核验通过；启动201物理子步无机器人正力接触，4例技术有效。

证据目录`/var/tmp/cvpr-xmove-failure-fix-20261111/`：

- `pilot_ready_upper_v1_001/FINAL_EVALUATION_ACCEPTANCE.json`，SHA256 `662ad3deeb6d21b3c81b1115ae648fe58d7631f8582ed7478ad55eac5287fedb`。
- `READY_UPPER_PILOT_PAIRED_RESULT.json`与`READY_UPPER_PILOT_EXIT.json`：真实对照、exit0与owned清理证明。
- `full30_rtcoff_001/FINAL_EVALUATION_ACCEPTANCE.json`：另一个独立实验，RTC关闭18/30（60%），三组各6/10；原RTC13/30。11例改善、6例退步，不能与ready四例混为同一因素。

四例是预先固定的诊断样本，**不是完整30例成绩或独立heldout benchmark**；渲染不保证逐位确定，相同种子在此前重复测试中仍有成功翻转。完整ready30尚未启动；扩展门预先固定为四例全部技术有效且成功至少3/4、真实清理退出。另一个既有XMoveBendPick队列仍活跃且无跨campaign显卡锁，须等它真实全退出再启动，不能利用逐例加载间隙抢GPU。训练、原结果与旧队列保持不变。

本文只补充已完成证据，运行树仍冻结在上述6a66提交，后续完整30例不使用文档分支替换源码。

## 四例复现入口（使用全新输出目录）

新worktree在`/dev/shm/cvpr-xmove-ready-upper-repo`，属于易失tmpfs，重启会丢工作文件；源码commit/push后可从分支恢复。权重、部署缓存、原数据、正运行评测均不在此新树中改写。磁盘保留给原XMoveBend权重回传。

使用SIM310，PYTHONPATH必须首先指向此新树的`scripts/4090_simple`、`VLMExpert/third_party/SIMPLE/src`及`VLMExpert/third_party/GR00T-WholeBodyControl/gear_sonic_deploy`。确认当前GPU任务已完整退出且拥有者清理完成后，才执行：

```bash
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python -B \
  /dev/shm/cvpr-xmove-ready-upper-repo/scripts/4090_simple/evaluate_pick30.py \
  --profile xmove_pick --checkpoint-step 5000 \
  --deployment /var/tmp/cvpr-xmove5000-eval-20261111/xmove_pick_ckpt5000/deployment \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --training-manifest /var/tmp/cvpr-xmove5000-eval-20261111/xmove_pick_ckpt5000/transfer/source/release_metadata/dino/manifest.json \
  --runtime-root /home/zxc/CVPR_Evaluation_runtime \
  --expected-model-sha256 d97bcf5be67a5b8065b9b1183af47062e4224a2960325326b70a93205befa6b0 \
  --output /var/tmp/cvpr-xmove-ready-upper-20261112/pilot_ready_upper_001 \
  --port 21155 --sampling-seed 20261105 \
  --diagnostic-cases 0:6,0:0,2:7,0:4 \
  --rtc-mode test_time --rtc-guidance-action-scope all --execute-steps 15 \
  --pick-interaction-diagnostics --pick-warmup-profile ready_upper_v1 \
  --ready-planner-model /home/zxc/CVPR_Evaluation_runtime/handover_random100_20261024/official-planner-v1/planner_sonic.onnx
```

四例模式仅允许原seed的上述4例、5000步同一模型、test_time/all/15。不混入RTC关闭、78维纠偏或10000步。冻结输入合同为`cvpr-xmove-pick-ready-upper-v1-four-case-diagnostic-v1`，固定分母4、role为`fixed_ready_upper_warmup_single_factor_pilot`。每例新server/sim、保留原视频/原生reward/HTTP/每次实际发送指令与全部正力接触。预热若出现任何机器人碰物/桌面/干扰物/自身接触、跌倒或证据缺失，按技术验收失败保存，不改scene或悄悄跳过；策略阶段真实接触仍完整记录并按原生任务判定。

缺省`--pick-warmup-profile native`不导入新helper、不增加planner依赖，旧任务、旧RTC和原场景路径保持不变。无自动重试，不删除旧结果，不启动或停止其他队列。

## 预备完整30例能力（未启动）

缺省`--ready-protocol four`仍仅允许上述固定四例；它的命令、分母和四例合同保持。显式`--ready-protocol full30`必须同时使用`--diagnostic-cases 0:0,0:1,...,2:9`的原始逐级逐case完整顺序，禁止其他子集、重排或静默补充。完整模式独立合同`cvpr-xmove-pick-ready-upper-v1-thirty-case-diagnostic-v1`、role=`fixed_ready_upper_warmup_full30_diagnostic`，分母30而`source_cases_frozen=30`、`formal_evaluation_eligible=false`、`full_official30_claim=false`、`original_official_initialization=false`。原生scene、5k权重、base seed20261105、RTC test_time/all/15、50/800预算与全部实际指令/接触门完全相同。

完整GPU30必须由独立driver检查同save ready四例技术完成、真实clean exit且达到预先声明的效果门，并确认既有GPU队列全退出后才启动。四例门已通过，完整30例结果仍须等待真实运行，不把四例4/4外推为全30成功率。
