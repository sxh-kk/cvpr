# XMovePick 通用上肢准备姿态：独立四例诊断

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

已交付99条训练episode首个packed状态的中位数仅作离线比较：上肢14关节RMSE原mode0约0.3534rad、中性mode1约0.2780rad、固定ready约0.1087rad。真实末端肘约0.584/0.625rad、肩roll约0.395/-0.407rad。该分布改善不证明模型抓取成功率改善；尚未GPU评测。

## 准备好的四例入口（尚未启动）

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

唯一允许的诊断覆盖是原seed的上述4例、5000步同一模型、test_time/all/15。不混入RTC关闭、78维纠偏或10000步。冻结输入合同为`cvpr-xmove-pick-ready-upper-v1-four-case-diagnostic-v1`，固定分母4、role为`fixed_ready_upper_warmup_single_factor_pilot`。每例新server/sim、保留原视频/原生reward/HTTP/每次实际发送指令与全部正力接触。预热若出现任何机器人碰物/桌面/干扰物/自身接触、跌倒或证据缺失，按技术验收失败保存，不改scene或悄悄跳过；策略阶段真实接触仍完整记录并按原生任务判定。

缺省`--pick-warmup-profile native`不导入新helper、不增加planner依赖，旧任务、旧RTC和原场景路径保持不变。无自动重试，不删除旧结果，不启动或停止其他队列。
