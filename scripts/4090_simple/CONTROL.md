# SIMPLE 双来源控制接口

启动模型服务与评估客户端的方法见[运行入口](README.md)。

## 观测与动作

| 内容 | 格式 |
| --- | --- |
| 机器人状态 | 43 个具名关节，单位为弧度；输入模型前补 2 维为 45 维 |
| 动作序列 | `[30,80]`：64 维 `body_token_v1_1`、14 维手部动作、2 维 padding |
| 模型图像历史 | 6 个真实的 5Hz 时刻，最后一帧与当前观测对齐 |
| 相机采样 | 至少 30Hz，并携带真实仿真时间 |

训练手部顺序为 thumb3／middle2／index2，SIMPLE 为 thumb3／index2／middle2，客户端按关节名映射。
动作末尾的 padding 不用于控制颈部。

## HTTP 接口

客户端 `simple.baselines.dual_source_psi0` 使用 localhost 服务：

- `/observe`：提交图像、状态及 `session_id`、`episode_id`、`sim_time_s`。
- `/act`：请求下一段动作。
- `/reset`：使用 `session_id` 和 `episode_id` 重置一次 episode。
- `/info`：读取模型和控制配置。

每个渲染帧均上报观测，执行动作 chunk 期间也持续更新历史。
HTTP 425 表示历史尚未积累到 1 秒；其他错误应中断本次执行。
每个 episode 同时重置环境和 agent，清空相机历史、动作 chunk、RTC 及 SONIC 历史。

## 控制节奏

高层动作按训练的 30Hz 时间轴取样，使用仿真时间而非 HTTP 请求次数推进。
SONIC v1.1 decoder 使用真实 50Hz 本体感知历史，PD 在 200Hz 物理步上执行。
不通过重复图像或复制本体感知帧填充历史。

`--compatibility-evidence` 指向控制器与训练 `body_token_v1_1` 相匹配的配置记录，
包含 `body_token_version`、`decoder_sha256`、`compatibility_verified` 和 `evidence`。
`--pd-profile` 与 `--hand-torque-profile` 应与所使用的数据及控制器设置一致。

## 任务结果

评估结果同时保留任务进度、跌倒、接触和视频。Handover 示范要求完成右手抬升、双手交接、
左手接管、放置与完全松手；数据筛选规则见[采集说明](README_handover_random100.md)。

`G1WholebodyLiftBoxHoldSonic-v0` 的条件为：箱体几何最低点距地板至少 10cm，真实手部接触力大于 0.01N，
无地板或桌面支撑，并在连续的 50Hz 物理观测中保持 1 秒。该任务不要求放回桌面。
