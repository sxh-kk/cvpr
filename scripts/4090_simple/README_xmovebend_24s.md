# XMoveBendPick 原失败七例：独立24秒诊断

本模式只回答：同一5000步权重、同一七个场景与seed，在**总时限24秒**下的本次实际结果如何。默认16秒、原始30例、ready_upper_v1及所有已保存结果不改。源码从冻结`6a66b4e8021e7ed4e108775cc5dcb43b2eca1174`独立分支建立，工作树在`/dev/shm`，属于易失存储；提交推送后可按commit恢复。没有复制权重。

仅接受以下显式模式，拒绝其他任务、顺序、任意时长、其他权重步数、RTC关闭/78维mask或ready上肢模式：

- `--pick-budget-profile xmove_bend_failed7_24s`
- `--profile xmove_bend_pick --checkpoint-step 5000 --sampling-seed 20261105`
- `--diagnostic-cases 0:5,1:6,1:7,1:8,2:0,2:4,2:9`
- `--rtc-mode test_time --rtc-guidance-action-scope all --execute-steps 15`

每例原50个真实warmup控制tick保持不变，包含在1200个总控制tick中；50Hz×24秒，单控制tick仍为4个200Hz MuJoCo物理子步。`pick_eval_protocol.make_pick_environment`返回真正的`gym.wrappers.TimeLimit(max_episode_steps=1200)`，worker同时强制`--seconds 24.0`。底层原`SonicLocoManipEnv.step`不自行截断，不能仅增加`--seconds`而保留800步TimeLimit。完整失败必须实际1200tick并`truncated=True`；原生成功可以提前终止。实际环境TimeLimit、已走控制tick、监督停止秒数写入`episode.pick_extended_budget`，case审核独立读取trace、HTTP/RTC、动作与视频帧数，并核两种时限。

官方task源文件、metadata中的800默认值（或原样缺省）、所有actor/相机/机器人初态、reset `dr_level=None`、奖励baseline与原生9cm成功判据均保持；没有教师状态、PICO输入、目标移动或额外预热。`original_official_initialization=True`表示初态未改；同时明确`official_budget_modified=True`、`original_official_time_budget=False`，不把加时称为原官方时限。

## 独立收据与结果解释

合同为`cvpr-xmove-bend-pick-failed7-24s-diagnostic-v1`，角色为`fixed_failed7_extended_time_budget_diagnostic`。冻结全部30条原source身份，但只执行固定七例，分母7，`full_official30_claim=False`、`formal_evaluation_eligible=False`。

`pick_budget_profile`在frozen/state/case中均为`xmove_bend_failed7_24s`。有效预算`maximum_steps_including_warmup=1200`、`maximum_simulated_seconds=24`，原预算另记800/16；原官方metadata字节不改。

每例从本次trace第一个`terminated=True`计算`first_native_success_control_tick`和`first_native_success_elapsed_s`（相对reset后的首观测时钟）。第800tick恰为16秒；分类严格按真实首成功tick≤800或>800，浮点时间独立交叉核验：

- `native_success_within_original_16s`：本次重跑在16秒内成功，不能归因于额外8秒。
- `native_success_during_extra_8s`：本次首成功在16–24秒间，仅描述本次发生时间。
- `no_native_success_by_24s`：本次完整1200tick未成功。

FINAL分别汇总`within_original_16s_successes`、`extra_8s_successes`、`no_native_success_by_24s`。`extra_8s_causal_gain_claim=False`；随机/渲染等重跑差异仍在，不能仅凭重跑声称因果。`can_combine_with_original23_as_measured30=False`，不能将原23个成功加新七例成功称为重新实测完整30例。

## 2026-11-12 GPU实测

北京时间11:13:49，冻结源码`764b395d877f45c98f7545e187a9295abf79b4d0`完成七例实际闭环，**成功3/7**：

- L1 case07：第981控制步、19.62秒成功；本次前800步最大正抬升为0，属于本次16秒后成功。
- L1 case08：第645步、12.90秒成功；L2 case00：第509步、10.18秒成功。两者均在本次原16秒预算内完成，不计为额外8秒收益。
- L0 case05、L1 case06、L2 case04、L2 case09均真实运行1200步/24秒后失败。

四个失败中，L0 case05、L2 case04、L2 case09的目标在policy阶段最大3D位移分别约1.004、0.832、0.852毫米，且最大正抬升为0，属于目标几乎静止；不能据此证明没有触碰。L1 case06本次目标水平最大位移约177毫米但未抬起。L2 case04没有复现原16秒运行末端抬高约58毫米的过程，本次全24秒无正净抬升，不能将旧轨迹直接续算。这些重复差异也说明延长预算与策略稳定性须分别判断。

全部七例技术有效，原50步预热/RTC test_time all80/15/原场景和seed保持；实际258次RTC，七份视频齐全，初始化机器人碰目标计数总和0。campaign与driver真实exit0，owned进程清理通过。原16秒完整结果23/30不改写，不将本次3/7合并成另一次全30实测。

证据目录`/var/tmp/cvpr-xmovebend-24s-20261112/`：

- `diagnostic_001/FINAL_EVALUATION_ACCEPTANCE.json`，SHA256 `4293d0d6650844b34d2e1a37129254ac6ce6d4ab5ae6337e9f80e5265a3b1fc6`。
- `PAIRED_RESULT.json`：逐例原视频/新视频、首成功时间、前800步与全24秒目标高度对照。
- `EXIT.json`：真实退出与清理；`SOURCE_CPU_VALIDATION.json`、`DRIVER_CPU_VALIDATION.json`和两份review分别保留CPU与独立审核范围。

本文在独立结果分支归档，实际运行树保持上述764b395冻结状态。

## 命令入口（复现须选全新output目录）

```bash
python -B scripts/4090_simple/evaluate_pick30.py \
  --profile xmove_bend_pick --checkpoint-step 5000 \
  --deployment /var/tmp/cvpr-xmove5000-eval-20261111/xmove_bend_pick_ckpt5000/deployment \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --training-manifest /var/tmp/cvpr-xmove5000-eval-20261111/xmove_bend_pick_ckpt5000/transfer/source/release_metadata/dino/manifest.json \
  --runtime-root /home/zxc/CVPR_Evaluation_runtime --expected-model-sha256 2de53a97cffb388327a46e2a5fe4661065c49e623e8c04f2e583b49452cd6691 \
  --output /var/tmp/cvpr-xmovebend-24s-reproduction/diagnostic_new \
  --pick-budget-profile xmove_bend_failed7_24s \
  --diagnostic-cases 0:5,1:6,1:7,1:8,2:0,2:4,2:9 \
  --sampling-seed 20261105 --rtc-mode test_time --rtc-guidance-action-scope all \
  --execute-steps 15 --port 21165
```

CPU专属测试为`tests/test_xmove_bend_24s.py`：真实factory所用Gym TimeLimit跨800不截断、1200才截断，严格CLI与mixed-mode拒绝，原场景不变、实际validate_case的800/801成功边界及1200失败、预算篡改、固定七例main调度。测试使用CPU控制计数环境与已审RTC账本夹具，不运行真实模型/renderer/GPU，不冒充物理或策略验收。
