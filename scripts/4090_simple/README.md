# SIMPLE 仿真与数据采集工具

通用配置、数据准备和训练方法见[项目首页](../../README.md)。本目录提供模型服务、SIMPLE 闭环评估及示范采集入口。

## 路径配置

在仓库根目录执行：

```bash
source scripts/4090_simple/local.env.example.sh
```

默认根目录为 `~/sigma0/runtime`，模型、SONIC 控制器和场景资产放在其 `resources/` 下。需要更换根目录时，在加载环境示例前设置 `CVPR_SIMPLE_RUNTIME`；单独调整场景资产可使用 `--data-root`，PICO 入口也支持 `SIMPLE_DATA_DIR`。

## 模型服务

在模型环境中启动已准备好的运行目录：

```bash
python scripts/4090_simple/serve_evaluation.py \
  --run-dir "<RUN_DIR>" --ckpt-step <CHECKPOINT_STEP> \
  --sampling-seed <SEED> --port 21074 --output "<SERVER_OUTPUT>"
```

`RUN_DIR` 包含模型配置、权重和对应的 DINO 配置。此入口为每个物理 episode 使用独立的服务进程。

## 闭环评估

在仿真环境的另一终端执行：

```bash
python scripts/4090_simple/run_closed_loop.py \
  --data-root "<SIMPLE_DATA_ROOT>" \
  --sonic-resources "<SONIC_RESOURCES>" \
  --compatibility-evidence "<CONTROLLER_COMPATIBILITY_JSON>" \
  --expected-checkpoint-step <CHECKPOINT_STEP> --port 21074 \
  --tasks "<TASK_ID>" --episodes-per-task <NUM_EPISODES> \
  --seconds <DURATION_SECONDS> --seed <SEED> \
  --sim-mode mujoco_isaac --video --output "<EVALUATION_OUTPUT>"
```

服务与评估命令的端口、checkpoint step 保持一致。输出目录使用仓库外的新目录。
`--sim-mode` 可选 `mujoco`、`mujoco_isaac`、`isaacsim`；`--tasks` 的可用任务见脚本参数。
模型服务、动作格式和历史输入要求见[控制接口](CONTROL.md)。

## 示范采集

- [PICO 遥操作与录制](../../docs/teleoperation.md)
- [随机场景 Handover 采集](README_handover_random100.md)

各脚本的参数可通过 `--help` 查看。
