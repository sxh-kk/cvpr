# UnifoLM 两次原生 CUDA 断言与恢复边界

本记录整理截至 2026-10-20 的已确认事实与恢复方案。两次故障均未精确复现，尚未定位到失败算子及其输入。恢复措施用于缓解异步报错和补全故障证据，**不能据此宣布根因已修复**；本轮不改 Motion MoE、Qwen 或 PyTorch 的索引数值路径。

## 故障事实

| 运行 | 最后完成的更新 | 断言及时间（UTC） | 已知边界 |
| --- | ---: | --- | --- |
| `261019022505` | 735 | rank5，2026-10-19 03:14:55 | 尚无正式 checkpoint，缺少该时刻权重及完整 batch/RNG。 |
| `261020022152`，提速恢复任务 | 15108 | rank5，2026-10-20 04:06:28 | 从完整 `ckpt_14000` 恢复；故障附近 batch/RNG 未持续落盘。 |

两次原生日志均为：

```text
aten/src/ATen/native/cuda/TensorCompare.cu:112
_assert_async_cuda_kernel: Assertion `` failed.
```

随后 NCCL watchdog 观察到 device-side assert 并终止进程。watchdog 栈不是触发断言的 Python/ATen 调用栈；不能据此认定 NCCL 是根因。这也不是日志中的 CUDA OOM。

15108 是控制台最后完成的更新计数，不是已确认失败算子的精确更新／微批编号。故障时 Python 线程在 backward 附近；该任务使用 `CUDA_LAUNCH_BLOCKING=0`，CUDA 异步执行仍限制了栈的定位能力。日志标记 `global_step=15000` 的验证已结束，之后训练又推进约百次更新；这一先后关系不能证明验证引发故障。该验证对应完成第 15001 次更新后的验证，不能与零基日志编号混用。

原始日志保留在仓库外：

- 首次：`/gaozt-test1/datasets/UnifoLM/.pipeline/training_launches/unifolm-joint-8gpu-20261019T022505Z/train.log`
- 本次：`/gaozt-test1/datasets/UnifoLM/.pipeline/training_launches/unifolm-speedup-resume-20261020T021809Z/train.log`

## 可恢复状态

确认可用的完整恢复基线为：

```text
/gaozt-test1/datasets/UnifoLM/.pipeline/recovery_snapshots/
unifolm-261019053758-speedup-20261020T021809Z/ckpt_14000
```

该快照包含完整模型、optimizer、scheduler 和 8 个 rank 的 RNG 状态，已有结构／CRC 检查与冷恢复验收。失败任务最后完成 15108，但没有将该状态保存为完整 checkpoint；不能声称从 15108 的参数和 optimizer 继续。若仍使用此快照，14001–15108 的 1108 次更新需要重算。

提速恢复链路的早期检查确认了 sampler 跳过 28000 个微批、每卡 batch16、累积2步及原学习率日程。保留的 CPU 诊断则止于 `global_step=14004`、`local_step=28008/28009`。它证明当时的累积位置，**不能重建 15108 附近的故障 batch、增强、随机状态或权重**。旧运行日志和曲线保留追溯；恢复后的有效训练历史必须明确标记 checkpoint 边界与重算区间。

## 原生调用链与 profiler 观察

官方 PyTorch v2.7.0 中，[`TensorCompare.cu`](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/cuda/TensorCompare.cu#L111-L140) 的空消息实现对应无消息 `_assert_async`。一个实际调用来源是 [`Indexing.cu` 的 `wrapIndexOnce`](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/native/cuda/Indexing.cu#L544-L550)，检查索引最大／最小值范围。开启 deterministic 后，部分 `index_add`、`scatter` 及其 backward 经排序索引实现进入这类检查。仅凭 kernel 名称，不能确定具体上层操作。

本轮使用完整模型、单张物理 GPU1、B16，读取 `ckpt_14000` 和保留的 rank5／14004 输入及 RNG，成功完成一次 forward/backward：loss 为 `2.3790488243103027`，没有 optimizer 更新或 checkpoint 写入。它不是 15108 的精确回放。

去掉 `TorchDispatchMode`、仅保留 CPU profiler 和模块标记后，该成功样本记录到 **544 次检查调用**：

| 调用阶段及父节点 | 检查调用数 |
| --- | ---: |
| Motion MoE forward：`index_add_ → index_put_` | 176 |
| Qwen forward：直接 `index_put_` | 32 |
| Qwen forward：`IndexPutFirstAxis → index_put_` | 56 |
| Qwen forward：语言模型内 `index_put_` | 12 |
| backward／重算阶段：`IndexPutFirstAxis → index_put_` | 56 |
| backward：`CopySlices / IndexPutBackward0 → index_put` | 12 |
| Motion backward：`IndexSelectBackward0 → index_select_backward → index_add_` | 176 |
| Motion 路由 backward：`TopkBackward0 → scatter_` | 24 |
| 合计 | 544 |

这些检查在该样本上均通过。调用次数不表示错误次数，更不能按频次认定 Motion 或 Qwen 是根因。上述样本也未覆盖真实 15108 权重、故障输入、8 卡通信或长时间运行。

证据目录：

```text
/gaozt-test1/datasets/UnifoLM/.pipeline/diagnostics/
assert15108-gpu1-profiler-only-20261020T043210Z/result.json
```

### TorchDispatchMode 会改变被测路径

初次探针同时启用了 `TorchDispatchMode`，记录到 1076 次检查，其中多出 532 次 `IndexBackward0 → index_put`。这部分不能计入普通生产路径。

原因有明确源码依据：[`TensorSubclassLikeUtils.h`](https://github.com/pytorch/pytorch/blob/v2.7.0/aten/src/ATen/TensorSubclassLikeUtils.h#L50-L73) 在存在 active dispatch mode 时，将相关判断直接视为 subclass-like；[`FunctionsManual.cpp::index_backward`](https://github.com/pytorch/pytorch/blob/v2.7.0/torch/csrc/autograd/FunctionsManual.cpp#L5154-L5162) 因而从普通 tensor 的内部 `unsafe=true` 分支切换为带范围检查的 `index_put` 分支。即使 Python hook 对 `_assert_async` 的命中数为0，也不说明没有改变 C++ 调用路径。

移除该 mode 后，额外532次检查消失；两次探针 loss 完全相同且 backward 均成功。探针的这项差别已确认，**不构成原训练故障的解释**。首次探针保留于同级 `assert15108-gpu1-parentcalls-20261020T043210Z/`，仅作对照。

## 仍未确定的原因

- Motion 稀疏路由、Qwen 图文索引及 checkpoint 反传均存在能到达原生检查的路径；尚未捕获失败时对应的父操作、索引及形状。
- NaN／Inf 保护和此前有限的 loss／梯度记录，不能代替索引边界或 CUDA 内存正确性检查。watchdog 的进程级终止也不能依赖 Python 异常处理保存 batch。
- 本轮可读取的 GPU ECC 计数为0；内核 Xid 日志因权限限制不可读。不能把“未能读取 Xid”写成“没有 Xid”，也不能由 ECC 为0排除全部硬件／驱动问题。两次均为 rank5 值得保留追踪，但不足以认定物理 GPU5 故障。
- 先前候选数据检查、Motion 索引压力测试、短时完整模型／多卡验收，以及本轮成功的 GPU1 探针，都不是两次失败状态的精确复现。短时通过不能证明长训不再失败。

直接把 `index_select` 换成 `gather`，或把 `index_add_` 换成函数式 `scatter_add`，其 deterministic backward 仍可能回到同一检查族。未获失败调用证据前，不实施这类数值路径替换，不删除边界检查，也不通过放宽检查或容差来宣称修复。

## 本轮恢复方案与验收边界

本轮决定保留模型、稀疏路由、Dropout、optimizer 数值路径及训练目标，采用以下措施增强定位与恢复能力：

1. 恢复 `CUDA_LAUNCH_BLOCKING=1`，使 CUDA 错误尽量在发起算子处同步报告；可能降低吞吐，不把它描述为根因修复。
2. 持续将各 rank 最近累积窗口的 **CPU batch 和 Python／NumPy／CPU Torch／CUDA RNG** 滚动写入 tmpfs，采用临时文件加原子替换，避免只覆盖启动时的14004附近。原生 abort 后从仓库外保存的 CPU 文件归档证据，不读取已损坏的 CUDA 状态。tmpfs 本身不是持久存储，主机重启仍可能丢失。
3. 按用户最新要求每1000次完整更新保存checkpoint，保留模型、optimizer、scheduler 和各 rank RNG，最多保留最近3份；14000步保护快照独立保留，下一次保存为15000。
4. 正式续训前完成真实8卡、每卡16、累积2步的冷恢复与连续更新检查。仍检查各组有限梯度、实际参数更新、跨卡一致性及完整状态恢复；完整保存路径已有此前验收，本次不以单卡 profiler 成功代替多卡检查。

目标日程保持 global batch256、总计40000次更新、warmup1000及既定五组学习率。恢复位置、监控前缀和后续实测吞吐须绑定新运行记录；不能把归档 batch 重播速度当作全量数据训练速度。

本轮真实8卡、每卡16、累积2步的验收已经通过：从14000完整冷恢复，完成4次带来源/参数审计的更新与8次无审计计时更新；五组梯度有限且参数更新，连续审计及最终完整模型的跨卡哈希一致。完整VLM／Motion tokens、pooling和state契约保留。本轮不写checkpoint，保存流程沿用此前验收路径；实际CLI已核对本轮每1000步保存、最多3份。

全程双槽诊断实际保留16个文件，覆盖最近两个update的两个microbatch；每rank每micro写入均值约0.051秒。归档输入重播最后6次更新均值约4.58秒，包含同步CUDA与持续快照；不等同全量数据吞吐，也不证明罕见断言已消除。

CPU已通过56项不同回归用例，含子进程原生abort后快照可读、RNG不推进、I/O失败保持旧快照及实际CLI解析。GPU报告、配置比较及随后正式启动状态见 [恢复验收记录](unifolm_assert_recovery_validation.json)。正式训练是否在跑，以仓库外运行状态为准。
