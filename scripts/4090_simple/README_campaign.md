# 149 条多 checkpoint 串行测评准备

`prepare_handover149_campaign.py` 为用户指定的 `4000 → 6000 → 7000 → 8000 → 9000 → 10000`
生成独立框架。当前 5000 的运行文件、状态、视频和收据均不修改。
本工具只执行 CPU 文件与来源检查，不启动模型、控制器、渲染器或训练。

## 持久串行队列

`queue_handover149_eval.py --campaign <仓库外目录>` 读取该目录的 `config.json`，
按固定顺序执行真实备份、CPU 导出、框架创建、namespace CPU 资格、三例 GPU
预检和正式 30 集。模型备份由 `backup_handover149_checkpoint.py` 完成，只取
模型与推理/来源元数据；传输前后独立查询 A100 全模型 SHA，与本机 SHA 一致后
原子完成备份，不下载优化器。源保存标记必须包含本步。

配置使用 `contract=cvpr-handover149-serial-local-evaluation-v1`，`steps` 必须为
`[4000,6000,7000,8000,9000,10000]`；其余字段为 `predecessor`（当前 5k 输出目录）、
`predecessor_sha256`、`backup_root`、`reference_backup`、`ssh_host`、`model_python`、
`resources`、`repo`、`template_framework`、新 `tag`、实际 `hostname` 与 `gpu_uuid`。
所有路径使用绝对路径。`poll_seconds` 默认 30。实际本轮配置位于仓库外：
`/home/zxc/CVPR_Evaluation_runtime/reports/handover149_serial_20261026/config.json`。

队列的 `state.json` 保存当前阶段、子进程身份、源权重身份和每步最终结果；
`logs/` 保存每次实际命令输出。任务成功率为零也可进入下一模型，技术执行
错误则停止，并保留全部失败产物。队列持有独立互斥锁，namespace 继续使用
原有物理 GPU 锁与空闲检查。CPU 文件准备可与前一模型重叠，GPU 阶段严格串行。
现有训练、测评和其他进程不停止、不重启。

进程意外退出后先检查已保存子进程身份；只在确认旧子进程结束并完成原因审查后
使用 `--resume`。已通过阶段可复用自身真实收据，失败阶段的日志或输出目录存在时
拒绝盲目覆盖，需要人工协调器修复并保留原始证据。不得修改既有配置后 resume。

```bash
python3 /home/zxc/CVPR_Evaluation/scripts/4090_simple/queue_handover149_eval.py \
  --campaign /home/zxc/CVPR_Evaluation_runtime/reports/handover149_serial_20261026
```

每步必须先具备该步实际完整模型备份、A100 原始 SHA256，以及真实独立的
`LOCAL_RUN_READY.json` CPU 组件载入与 Motion 导出收据。不能重命名 5000 的收据。
只有模型从同一 149 条 BS4/global32 正式 run 完成保存，源前后及本机 SHA 全等，
CPU 导出与备份使用同一模型硬链接，才会生成新目录。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 python3 \
  /home/zxc/CVPR_Evaluation/scripts/4090_simple/prepare_handover149_campaign.py \
  --step 4000 --tag campaign001 \
  --source /home/zxc/CVPR_Evaluation_runtime/checkpoints/handover149_bs4_20261026/ckpt_4000 \
  --local-run /绝对路径/该步已通过CPU准备的local_run \
  --expected-model-sha256 该步A100原始64位SHA256
```

可用 `--template-framework` 指定原 5000 `_r2` 框架。输出位于它的同级目录；
存在同名输出时拒绝执行，失败产物保留，重试使用新 tag。模型与 Motion 使用硬链接，
不重复复制大文件。stdout 最后一行是单行 JSON，包含 `framework_host`、
`framework_namespace`、`experiment_host`、`experiment_namespace`、`mirror_home`
及三个阶段的命令。创建证据为新框架的 `CAMPAIGN_GENERATION.json`。

新框架只复制源码和资源策略，不复制旧 `state/results/preflight`、namespace 通过证据
或 `ROOT_GPU_LAUNCH_ADMISSION.json`。源封装从本步真实收据重新生成
`source_pin.json`、`source_bundle/COMPLETE.json` 与所有路径绑定。
`IDENTITY_SOURCE_DIFF.patch` 列出身份适配；`runtime_template_source_receipt.json`
逐文件记录原始和生成字节。runtime/helper 数值源码保持一致，仅允许
`protocol_core.CHECKPOINT_STEPS` 与 DINO 验收的当前部署 step guard 改变。
原 5000 失败案例仍作为 CPU 回归 fixture，绝不当作新模型的 GPU 预检。

## 阶段与串行门槛

每步依次使用返回的 `namespace_launch.sh --phase cpu-check|preflight|run` 命令，
三个阶段使用同一新 `--attempt`。CPU-check 会重新运行环境、六轮实际 DDS、
部署准备与九项 DINO 来源回归；它也使用共享物理 GPU 锁，因此当前 5000 正在
占锁时不要调用。生成器本身不取 GPU 锁，可先准备小型 CPU 文件。

CPU-check 真正成功后，由主协调器依据当前用户授权写本步独立的
`ROOT_GPU_LAUNCH_ADMISSION.json`，字段至少包含：`authorized=true`、本步
`namespace`、真实 `model_sha256`、`allowed_phases=["preflight","run"]`，以及
实际 CPU acceptance 路径/SHA 与授权来源。不能把 CPU 通过填写成 GPU 预检通过。
每步仍须运行三个等级各一个 100-control-tick 真实 GPU 预检；它们不计分。
只有预检实际通过、protocol 进入 ready，才能执行正式 30 集。

队列等待前一轮 `state.status=complete`、`results.status=complete`、30 个不同
场景全部验收、`final_raw_acceptance.status=passed`、checkpoint SHA/step 一致，
再等待前一 namespace 退出、物理 GPU 锁释放与两次 CUDA compute 空闲观察。
不能只看到 `30/30` 或 PID 不存在就判定完成。执行错误停止队列并保留证据，
不计作策略失败；不停止、不重启外部进程，不并行执行两个 checkpoint。

同一官方 Level 0/1/2 各 10 集、场景、种子、重力/接触控制、30 秒预算、完整真实
历史、手控参数、原生和严格两个指标均继承原协议。保持测试时 RTC `.9` 指数
mask、30 行预测/15 行执行、10 步采样和原有精度。仅 checkpoint 来源改变；
同一硬件上的自然 RGB 或随机输出差异不要求与旧 3090 逐字节一致。

CPU 回归：

```bash
python3 -m unittest discover -s /home/zxc/CVPR_Evaluation/scripts/4090_simple/tests \
  -p test_prepare_handover149_campaign.py -v
```
