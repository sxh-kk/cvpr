# PICO 遥操作与示范录制

该入口使用 PICO 全身追踪和双手柄控制 SIMPLE G1，录制 Handover 示范。
路径配置见[项目首页](../README.md#configuration)。仿真需要准备 SIMPLE 场景资产、匹配的 SONIC v1.1 模型与 PICO PC SDK。

## 设备准备

1. 电脑与 PICO 连接同一局域网。
2. 在 PICO 系统中完成两个脚踝定位器的标定。
3. 在 XRoboToolkit 中开启 Head、Controller、Full-body 和 Send，使用双手柄模式。
4. 仅运行一个 PC SDK 数据接收器。

## 启动

从仓库根目录，在仿真环境中执行：

```bash
source scripts/4090_simple/local.env.example.sh
SESSION_OUTPUT="$CVPR_SIMPLE_RUNTIME/recordings/pico-$(date +%Y%m%d-%H%M%S)"

python scripts/4090_simple/pico_simple_teleop.py \
  --data-root "$SIMPLE_DATA_DIR" --sonic-resources "$CVPR_SONIC_DIR" \
  --render-mode mujoco_deferred --output "$SESSION_OUTPUT"
```

默认资源目录为 `~/sigma0/runtime/resources/simple_local_usd_v1` 和 `~/sigma0/runtime/resources/sonic_v1_1`，录制保存到 `~/sigma0/runtime/recordings/` 下的新会话目录。
需要使用其他目录时，可在加载环境示例后调整 `SIMPLE_DATA_DIR`、`CVPR_SONIC_DIR` 和 `SESSION_OUTPUT`。
控制面板为 `http://127.0.0.1:43100/`，
通过 `--port` 修改端口。需要独立的大画面窗口时，另开终端运行：

```bash
python scripts/4090_simple/pico_desktop_preview.py --collector-port 43100
```

打开 `http://127.0.0.1:43101/` 查看预览。

## 录制操作

| 操作 | 功能 |
| --- | --- |
| 左手柄 X | 站直后校准 |
| 左手柄 Y | 开始跟随／暂停 |
| 左右前扳机或侧握键 | 控制同侧手的闭合量，取两者较大值 |
| 右手柄 A | 开始／结束录制 |
| 面板“丢弃” | 丢弃当前录制；未录制时丢弃上一条 |
| 面板“退出” | 结束当前采集会话 |

B 未绑定仿真动作。每条示范按以下顺序操作：

1. 站直、双脚落地，确认全身追踪有效，按 X 校准。
2. 按 Y 开始跟随，确认左右手及抓握方向正确。
3. 按 A 录制，先保持姿势 1 秒，等待起始历史采集完成。
4. 完成右手抬升、双手交接、左手接管、放置与松手，稳定保持至少 1 秒。
5. 再按 A 保存。场景自动重置并暂停；下一条重新按 X、Y。
6. 不需要的记录通过“丢弃”处理，原始记录保留在 `discarded/`。

追踪失效、控制超时或物理失稳会暂停采集。当前模式使用身体姿态和根朝向，不将人在房间中的绝对平移直接映射为机器人位移。

## 图像后处理

默认 `mujoco_deferred` 在操作时显示 MuJoCo 预览，训练用 RTX 图像在采集结束后生成。
退出采集器后，在同一终端执行：

```bash
python scripts/4090_simple/finish_pico_session.py "$SESSION_OUTPUT"
```

该命令等待后台筛选完成，为保留记录生成 RTX 图像，并检查录制内容。中断后可再次执行。
RTX 图像来自记录的同一物理状态；操作者看到的是实时 MuJoCo 预览。

## 输出

单条示范保存在 `episodes/<UUID>/`，完成归类后按编号放入集合目录的 `kept/` 或 `discarded/`。
主要文件包括：

| 文件 | 内容 |
| --- | --- |
| `metadata.json` | 设备、控制和场景信息 |
| `commands.jsonl` | PICO 输入、SONIC token 和手部指令 |
| `quality_samples.jsonl` | 物理状态与接触记录 |
| `applied_force_samples.jsonl` | 外力记录 |
| `episode.json` | 录制结果 |
| `deferred_capture/` | 物理模型、积分状态和图像事件 |
| `capture/` | 后处理生成的 RTX 图像及事件 |
| `session_finalization.json` | 会话后处理结果，位于会话根目录 |

`kept` 表示通过物理筛选；只有后处理结果中的 `eligible_for_packing=true` 才可进入训练数据打包。
人工回放包和 `--diagnostic-record` 不作为新的人类示范。
