# SIMPLE 双来源控制与真实历史

`simple.baselines.dual_source_psi0`是20k双来源专用客户端，不沿用旧32-state／36-action路径。
`/observe`和`/act`使用原RequestMessage格式，history明确携带`session_id`、`episode_id`、
`sim_time_s`；`/reset`使用前两项的平坦JSON。425只表示真实历史未满一秒，其他错误中断执行。
客户端仅接受localhost，忽略系统HTTP代理；每个渲染帧都上报，包括动作chunk执行期间。

关节契约依据官方`USC-PSI-Lab/psi-data`固定revision
`e78fb93cc28912a3031a10b8656d32d7f0a2b867`的val ZIP，SHA256为
`3d264d6454d59e83be5dadaa17f122b732a8c6cb86a2429c33c1f4fc912fc4b7`。
其`info.json` SHA256为`3b58f2cb791fc0e82bcc6c4dc08a6ced44cd5729e9531bd741f2e392dff1d467`。
原43维state单位为弧度，body29后两手都是thumb3／middle2／index2；SIMPLE手顺序是
thumb3／index2／middle2，故按关节名映射，再在末尾补2维。动作严格为30×80：
64维body_token_v1_1、14维训练手顺序、2维padding；padding不控制颈部。

环境的`joint_names`和`sim_time_s`已纳入Gymnasium observation_space。
时间直接来自`mjData.time`。高层动作依训练30Hz按仿真时间取样；相机须至少30Hz，
不能按HTTP请求次数或渲染次数加速动作。首任务原配置为physics_dt=0.005、render_hz=50。

## 真实历史采集

以下由仿真Conda环境执行；Isaac与真实评测应由主进程串行调度，避免争抢4090显存。

```bash
source scripts/4090_simple/local.env.example.sh
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python scripts/4090_simple/capture_simple.py \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --sonic-resources /home/zxc/CVPR_Evaluation_runtime/resources/sonic_v1_1 \
  --output /home/zxc/CVPR_Evaluation_runtime/reports/simple-history.npz
```

默认使用Psi0的XMovePick任务、DR0、episode0的官方固定environment_config，
将旧机器SIMPLE/data路径重映射到本机数据根目录。输出真实RGB、sim_time_s、按训练顺序的
state43、episode_id及完整来源metadata，同时保存首末PNG和JSON。
这里采用实测关节姿态PD保持，不运行20k策略或平衡策略；即使采集完成，也不证明任务闭环。

## SONIC v1.1适配与身份边界

`simple.baselines.sonic_v1_1.SonicV11OnnxController`核对固定官方decoder和observation_config
SHA后构建本机ONNX Runtime会话。适配依据仓库固定`GR00T-WholeBodyControl/gear_sonic_deploy`
的`policy_parameters.hpp`及`GatherHis*`／`CreatePolicyCommand`实现：
每个MuJoCo物理tick执行PD，在50Hz控制点采集真实10帧角速度、相对默认角度、关节速度、
上一raw action和重力方向，按term拼接为64＋930维decoder输入。先按IsaacLab顺序解码，
再按关节名输出43个扭矩；增益、default angles、action scales均与固定C++逐值对照。
不重复帧补齐proprioception，不借用旧SONIC standing token，不覆盖第三方快照。
实测姿态PD温启时，上一raw action按实际保持目标反解，保证decoder的last-action历史与
已发指令一致；不把保持姿态的目标误记为零residual。

默认`pd_profile="fixed_cpp_baseline"`保留固定源码增益。显式选择
`official_v1_1_ankle_pitch_1_5`时，核对固定官方README SHA
`0a69d0a418fddc2d96e9a70244876c32d40a44bee2b20c70c21747807c2972fd`，
对body顺序4／10的踝pitch KP和KD额外乘1.5，并将实际增益写入identity。
这是官方文档的v1.1调参配方，是否在本机稳定仍需实测，不能因采用该配方就声明通过。

官方v1.1文件身份与UnifoLM训练body_token_v1_1所用encoder的身份是两项独立证据。
缺少独立兼容证据时，构造器默认拒绝运行；只有显式
`allow_unverified_compatibility=True`才允许诊断，identity记录
`unverified_diagnostic_only`。这类结果不能作为20k正式任务失败率或完整闭环验收。
正式兼容证据JSON需绑定`body_token_version`、`decoder_sha256`、
`compatibility_verified=true`与可复核的`evidence`；调用方必须确认其证据真实成立。

控制器先构造，再显式传给agent：

```python
controller = SonicV11OnnxController(sonic_dir, compatibility_evidence=evidence_json)
agent = DualSourcePsi0Agent(env.unwrapped.task.robot, host="127.0.0.1", port=21074,
                           controller=controller, execute_steps=30)
agent.reset(episode_id="fixed-task-seed-episode")
action = agent.get_action(observation, instruction=env.unwrapped.task.instruction)
observation, reward, terminated, truncated, info = env.step(action)
```

每个episode同时reset环境和agent，清空图像、chunk、RTC和SONIC proprioception历史。
是否站稳、能否操作、任务成功率、GPU峰值与延迟，都必须由真实执行另行验收。

CPU契约检查：

```bash
PYTHONNOUSERSITE=1 /home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  -m unittest discover -s scripts/4090_simple/tests -p test_simple_control.py -v
```

8项覆盖双手state/action顺序、padding排除、逐帧上报与episode reset、HTTP／425、
真实Gymnasium检查器、真实MuJoCo物理tick与50Hz历史，以及编译固定C++常量的数值对照。
其中确定性decoder仅为CPU测试fixture，绝不作为策略闭环结果。

## 有界真实闭环诊断

先在模型Conda环境单独启动真实20k服务，绑定`127.0.0.1:21074`，
`--action-exec-horizon 30`，首轮关闭RTC。确认服务就绪后，在仿真Conda环境运行：

```bash
source scripts/4090_simple/local.env.example.sh
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python scripts/4090_simple/run_closed_loop.py \
  --data-root /home/zxc/CVPR_Evaluation_runtime/resources/simple_local_usd_v1 \
  --sonic-resources /home/zxc/CVPR_Evaluation_runtime/resources/sonic_v1_1 \
  --output /home/zxc/CVPR_Evaluation_runtime/reports/closed-loop-mujoco \
  --port 21074 --seconds 5 --sim-mode mujoco --diagnostic-unverified-sonic
```

默认三个Psi0任务各一个固定episode。可用`--tasks`、`--episodes-per-task`和
`--episode-start`缩小或重复，`--sim-mode isaacsim`使用MuJoCo物理加Isaac真实渲染。
`--pd-profile official_v1_1_ankle_pitch_1_5`是显式可选配方。每个任务独立进程串行运行，
防止关闭Isaac SimulationApp后复用失效进程。`--stop-on-fall`可在首个摔倒诊断处结束；
默认继续到有界时间，以保留服务和decoder调用证据，同时将摔倒阶段明确记录为失败诊断。

输出包括每次HTTP／425／reset、每帧30Hz动作索引、50Hz decoder实际调用、状态有限性、
pelvis初末位置／四元数、最小高度／最大倾斜、首次失衡和首次策略命令时刻，以及真实RGB PNG。
如果温启期间已失衡，报告明确排除策略质量归因。任务predicate、真实loop执行与正式评测资格
分开记录；有界诊断不生成正式成功率。
终止步也重新检查物理有限性和摔倒，避免把最后一步倒地错误归因为任务成功。
由于Isaac关闭可能使出错的Python进程返回0，supervisor同时审核保存的完整状态、episode数量、
实际decoder和同步时钟证据；只有进程与报告都满足条件才标记完成。报告审核的两个CPU回归位于
`test_smoke_reporting.py`。

所有HTTP采用同步等待。推理期间**暂停MuJoCo和Isaac推进**，并逐请求核对前后mjData.time相同；
因此慢于仿真帧周期的GPU预测不会偷用旧状态追赶墙钟。报告分别保留仿真秒、墙钟秒和HTTP延迟，
50Hz控制／30Hz动作只表示仿真时间频率，不宣称实现物理实时吞吐。
