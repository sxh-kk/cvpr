# 三类 PICO Pick 独立训练

`a800_pick_tasks.py` 沿用已完成的 Handover149 BS4 配方。三个任务分别从同一份原始联合预训练 `joint38000` 完整权重初始化，各自训练 10000 次优化器更新，不继承 Handover 微调权重、优化器、scheduler、RNG 或任务步数。

本次用户已确认使用服务器现有的 **8×A800-SXM4-80GB**，顺序为 BendPick、XMovePick、XMoveBendPick。三个任务各占用全部8卡，串行执行，不停止其他任务。每个任务必须先取得其独立服务器 `A100_PICO_DATASET_READY.json`，名称沿用历史接口，实际设备型号以本次计划为准。

- `bend_pick`：100条、15722帧、9822个训练窗口；服务器目录 `pico_bend_pick_20261110_human100_01`。
- `xmove_pick`：99条、19925帧、14084个训练窗口；服务器目录 `pico_xmove_pick_20261110_human99_01`。失败的004不混入训练，后续补采另行处理。
- `xmove_bend_pick`：100条、34569帧、28669个训练窗口；服务器目录 `pico_xmove_bend_pick_20261110_human100_01`。

以上目录均位于 `/gaozt-test1/datasets/SIMPLE_sonic_v1_1/`。训练不修改原始采集、已验收的 packed 数据、clean DINO 缓存或原处理源码。

## 配方与权重保存

每卡 batch4、梯度累积1，全局 batch32；BF16、DDP、融合 AdamW、cosine、warmup1000、梯度裁剪1.0。Action及投影峰值学习率为1e-5，Motion及视觉为1e-6，语言为1e-7。Motion历史使用 clean DINO，仅VLM当前图像进行既定颜色增强，维持256×320几何与原归一化。

初始化权重SHA256为 `12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7`。先以源state dropout0.5严格加载完整联合模型，再使用已验收的helper迁移为0.0，保留所有测量状态和原动作头权重。

永久保留以下13次更新的完整模型：

`1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 10000`

共享trainer使用 `checkpointing_steps=1000`、`early_checkpoint_steps=[4500,5500,6500]`、`preserve_checkpoint_every=500`，无需修改。只有本次新run中过旧的优化器、scheduler和RNG状态会清理，最近两套完整恢复状态及全部模型永久保留。第2、20及10000步检查八卡损失/梯度有限性、融合优化器、每卡实际batch及参数探针一致性；该探针不是完整模型逐参数相等证明。

## 准备、检查和运行

参考目录使用本机 `CVPR_Evaluation_runtime/pick299_training_20261110/reference/` 的24个原样归档文件：保留Handover114溯源链，显式绑定Handover149的BS4配置、准备记录及shape adapter。不能直接使用Pick包内仍记载源state dropout0.5的 `source_run_config.json` 作为新训练配置。

在服务器独立目录内运行，例如：

```bash
CUDA_VISIBLE_DEVICES='' python -B a800_pick_tasks.py prepare \
  --profile bend_pick --reference /path/to/verified/reference \
  --output /path/to/new/bend_pick --training-cpus 0-127 \
  --expected-ready-sha256 <该任务实际服务器验收回执的SHA256>
CUDA_VISIBLE_DEVICES='' python -B a800_pick_tasks.py check --prepared /path/to/new/bend_pick
python -B a800_pick_tasks.py run --prepared /path/to/new/bend_pick
```

`prepare/check`可在隐藏GPU的低优先级CPU子进程中执行。正式`run`必须由普通SSH环境启动，CPU affinity与准备计划一致、nice=0；不得从单核nice19的检查上下文继承训练限制。启动器确认8卡空闲并绑定全部字节、准备配置、参考代码和源权重后才启动8个DDP rank。每个任务输出必须使用全新目录；已启动的run不自动重启或覆盖。

这299条均为 `train_only`，没有独立Pick验证集。保留历史Handover5条40个窗口仅用于回归监控，不能作为Pick独立验证分数或参与训练。

本次持久队列与只读进度镜像位于仓库外 `CVPR_Evaluation_runtime/pick299_training_20261110/`；服务器执行目录为 `/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/pick299_training_20261110_001/`。队列逐任务等上传验收、执行配置检查和正式训练；只有实际完成10000步、八卡最终检查及13个模型头/EOF完整性均通过才进入下一任务。失败保留证据，不自动改变数据或恢复训练。
