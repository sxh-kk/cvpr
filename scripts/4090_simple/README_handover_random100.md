# 随机场景 Handover 示范采集

该流程从原始 Handover 轨迹生成新场景，在物理仿真中重新执行动作，记录 SONIC v1.1 token、手部指令、
状态和 RGB。PICO 人工操作的录制方法见[遥操作说明](../../docs/teleoperation.md)。

## 输入

- 固定来源的 `simple-teleop/G1WholebodyHandoverTeleop-v0/level-0` 轨迹及其资源。
- SIMPLE 场景资产与匹配的 SONIC v1.1 模型。
- 采集要求 JSON 和原有数据包。采集结果写入仓库外的新目录。

来源 39、65、80、92、94 为保留验证来源，38 只有一帧，其余来源可作为训练候选。
目标与容器的位置、朝向、材质和灯光由场景配置采样。

## 操作流程

```bash
python scripts/4090_simple/collect_handover_random100.py prepare \
  --requirements "<REQUIREMENTS_JSON>" --old-pack "<TEMPLATE_PACK>" \
  --raw-root "<RAW_DATA_ROOT>" --data-root "<SIMPLE_DATA_ROOT>" \
  --sonic-resources "<SONIC_RESOURCES>" --python "<SIM_PYTHON>" \
  --output "<QUEUE_ROOT>"

python scripts/4090_simple/collect_handover_random100.py run \
  --queue "<QUEUE_ROOT>" --phase all --gpu-device <GPU_ID>

python scripts/4090_simple/collect_handover_random100.py status --queue "<QUEUE_ROOT>"
```

`--phase screen` 仅运行物理筛选，`--phase capture` 采集图像与状态，`all` 顺序执行两者。
调度器要求相同场景配置的 bootstrap 记录；可通过 `--bootstrap-admission` 指定。

默认 `--placement-encoder-mode g1` 使用关节参考；`teleop` 在放置阶段改用双腕及头部空间目标。
`--placement-locomotion` 可选 `source_future` 或 `actual_hold`。完整参数见各入口的 `--help`。

## 数据筛选与打包

Handover 示例须完成右手抬升、双手交接、左手接管、松手和容器支撑，最后连续 50 个 50Hz 状态保持稳定。
物体的全部碰撞顶点需位于容器外碰撞凸包投影内，最小余量为 5mm。执行期间禁止恢复物体、固定物体或施加外力。

完整数据集要求 100 条有效记录，至少 50 个不同的训练来源，每个来源最多 2 条。
相同来源或场景的重复轨迹不能仅通过更换 seed 计为新示范。

| 工具 | 用途 |
| --- | --- |
| `audit_handover_random100.py` | 单条轨迹的动作、接触、放置和时间对应检查 |
| `diversity_handover_random100.py` | 数据集来源、场景及轨迹多样性筛选 |
| `pack_handover_random100.py` | 将通过筛选的注册表转换为训练数据包 |
| `verify_handover_random100.py` | 检查数据包、DINO 历史特征与模型读取格式 |

数据包沿用原模型的归一化统计。完成图像与 DINO 特征准备后，按项目首页的[训练步骤](../../README.md#training)使用数据。
