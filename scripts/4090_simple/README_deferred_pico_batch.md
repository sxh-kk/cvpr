# 冻结 Grasp 100 条的批量 RTX 补图

`render_deferred_pico_batch.py` 使用一个私有 Grasp 环境和一个 Isaac 实例，
按冻结清单顺序串行执行未修改的 `render_deferred_pico.py`。
每条仍独立执行原始 reset、场景重建绑定、完整归档状态恢复、相机 fresh 检查、
逐帧完整状态不变检查和 RGB / state SHA 桥接；不回放动作、不推进归档物理。
采集环境初始化的既有 reset 行为不属于归档状态积分。

```bash
python scripts/4090_simple/render_deferred_pico_batch.py \
  --manifest /absolute/path/processing/FROZEN100.json \
  --data-root /absolute/path/resources/simple_local_usd_v1 \
  --report /absolute/path/processing/rtx_batch_pilot.json \
  --limit 2
```

先按运行环境要求启动 Python / Isaac 环境。`--limit` 仅用于首批验证，
省略后处理全部 100 条。即使限制数量，也先检查全部冻结文件、100 个唯一 UUID、
官方场景 0–99 覆盖，以及机器人、完整相机配置、灯光拓扑和缓存资源兼容性。
同一房间 UID 的几何配置、同一物体 label 的资产身份发生冲突会停止；
房间、物体的合法位姿和材质变化仍由原 reset / Isaac 更新路径处理。
一个环境仅在整个批次结束或异常时关闭。节省的是初始化开销，尚需实测确认吞吐。

每个新 `rtx_completion.json` 的 `batch_execution` 字段记录包装器和原渲染器 SHA、
冻结清单 SHA、环境租用序号和执行范围，避免把包装执行表示为仅单条入口执行。
总报告绑定最终完成回执 SHA。回执仅表示待独立准入审核，不自动允许打包。

Isaac 的 `SimulationApp.close()` 可能直接退出进程而不返回 Python。
因此在 close 前先落盘完整条目、租用数量、`cleanup_requested=true`、
`cleanup_return_verified=false` 和 `status=render_completed_pending_process_exit`。
外层运行器必须在子进程退出后生成 `<report.stem>_EXIT.json`，绑定
`exit_code`、`finished_beijing`、`report_path`、`report_sha256`、`argv`、
`parentdriver_sha`；独立审核验证 exit 0 和最终报告 SHA 后才能接受该状态。
只有 close 确实返回，Python 才记录 `cleanup_return_verified=true` 和
`completed_pending_independent_audit`。这两个状态均不表示独立准入已通过。

历史首批旧包装器 `0cb9a330f3d520d143222f94227a13cb8428c68d401902414e2799882dfd652f`
曾在 close 前未写完总报告；该特定旧生产者通过精确源字节归档及独立
`cvpr-pico-grasp-batch-close-exit-recovery-v1` 回执绑定原报告、原 exit 0 证据、
冻结清单和所有最终输出。原 running 报告保留；不会普遍放行其他旧版本或不完整报告。

恢复运行必须指定新的 `--report` 路径。已完成的单条输出需要原始四份归档 SHA、
捕获清单 SHA、逐帧桥 SHA、每条解码事件及每张 PNG SHA 全部一致才复用；
复用检查不代替独立审核。部分输出或 rejected 回执一律保留并停止，
重试需要新的 episode 副本；不会删除、覆盖或自动清理失败证据。
原始归档文件和固定采集 source SHA 不改变。
