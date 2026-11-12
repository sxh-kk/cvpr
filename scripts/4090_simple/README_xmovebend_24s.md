# XMoveBendPick 原失败七例：独立24秒诊断

本模式只回答：单次指定5000或5500步权重、同一七个场景与seed，在**总时限24秒**下的本次实际结果如何。默认16秒、原始30例、ready_upper_v1及所有已保存结果不改。源码从冻结`6a66b4e8021e7ed4e108775cc5dcb43b2eca1174`独立分支建立，工作树在`/dev/shm`，属于易失存储；提交推送后可按commit恢复。没有复制权重。

仅接受以下显式模式，拒绝其他任务、顺序、任意时长、其他权重步数、RTC关闭/78维mask或ready上肢模式：

- `--pick-budget-profile xmove_bend_failed7_24s`
- `--profile xmove_bend_pick --checkpoint-step 5000`或`5500`，固定`--sampling-seed 20261105`
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

## 命令入口（由独立已审runtime驱动调用，不会自动启动）

```bash
python -B scripts/4090_simple/evaluate_pick30.py \
  --profile xmove_bend_pick --checkpoint-step 5000 \
  --deployment /var/tmp/cvpr-xmove5000-eval-20261111/xmove_bend_pick_ckpt5000/deployment \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --training-manifest /var/tmp/cvpr-xmove5000-eval-20261111/xmove_bend_pick_ckpt5000/transfer/source/release_metadata/dino/manifest.json \
  --runtime-root /home/zxc/CVPR_Evaluation_runtime --expected-model-sha256 "$MODEL_SHA" \
  --output /var/tmp/cvpr-xmovebend-24s-20261112/diagnostic_001 \
  --pick-budget-profile xmove_bend_failed7_24s \
  --diagnostic-cases 0:5,1:6,1:7,1:8,2:0,2:4,2:9 \
  --sampling-seed 20261105 --rtc-mode test_time --rtc-guidance-action-scope all \
  --execute-steps 15 --port 21165
```

CPU专属测试为`tests/test_xmove_bend_24s.py`：真实factory所用Gym TimeLimit跨800不截断、1200才截断，严格CLI与mixed-mode拒绝，原场景不变、实际validate_case的800/801成功边界及1200失败、预算篡改、固定七例main调度。测试使用CPU控制计数环境与已审RTC账本夹具，不运行真实模型/renderer/GPU，不冒充物理或策略验收。

## 5500步权重对照

在独立`codex/xmovebend-5500-20261112`分支中，24秒模式仅允许5000或5500两个整数步数，仍拒绝4500、6000、10000或其他步数；默认16秒的既有步数选择与ready_upper_v1仅5000的门不改。该分支从已冻结24秒提交`764b395d877f45c98f7545e187a9295abf79b4d0`建立，旧24秒源码与结果保留。

5500运行沿用上方七例、24秒/1200tick、50warmup、同seed、test_time/all/15和原生9cm；仅改`--checkpoint-step 5500`，并绑定**5500自己的**严格CPU部署目录、模型SHA和training-manifest收据，使用新的独立output。不能复用5000服务或将旧模型目录改名充当5500。实际server信息、startup权重SHA与每例case结果均绑定所选checkpoint步数。

这次是同七例的权重对照，分母仍为7。`within_original_16s_successes`与`extra_8s_successes`仍仅描述本次5500运行的首原生成功时刻；不是完整30例复测，也不能与原23条成功拼成实测30条。源码准备和CPU测试不会下载模型或启动GPU。
