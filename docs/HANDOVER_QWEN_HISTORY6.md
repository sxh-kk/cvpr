# Handover：Qwen六帧历史消融

本实验检验：在原Handover149任务微调中，只把Qwen的单张当前RGB改为与Motion Expert同步的五张历史RGB加当前RGB，是否改善相同测评条件下的任务成功率。输入应在训练、验证和部署时一致；不把旧单帧权重直接改成六图推理作为正式对照结果。

## 唯一输入开关

`data.transform.model.vlm_history_frames`只允许整数`1`或`6`，缺省为`1`，使原配置与checkpoint继续走单帧路径。

六帧时，相对于当前时刻，Qwen按`[-1.0,-0.8,-0.6,-0.4,-0.2,0.0]`秒、5Hz、从旧到新的顺序读取六张真实RGB。训练读取器直接复用`MotionHistoryIndex.selected_cache_rows`和该episode的video PTS，批量查询同源视频；最后一帧必须与原当前图像查询一致。不能跨episode、读取未来RGB、重复当前帧补齐历史，或通过历史数据项访问改变状态和动作标签。

Qwen使用六个按序的image消息，而非新增视频编码器、时间文字提示、视觉融合网络或学习参数。`model.img_chunk`、`model.n_cams`、`model.observation_horizon`和`data.transform.repack.num_past_frames`均保留原值；机器人仍只输入原有当前状态，动作预测窗口仍是30帧。

训练仍使用原`vlm_color_clean_motion`策略。Qwen的六张图共享一次窗口颜色增强与同一几何处理；Motion继续读取原有clean DINO缓存，不重新提取或改变缓存、样本筛选、归一化、动作mask和train/val划分。验证和推理不做颜色增强。collator检查实际六个图像grid、像素patch数、视觉占位符和完整上下文长度，六帧模式不允许通过截断悄悄丢掉当前图像。

在线服务从checkpoint的数据配置读取同一开关。六帧模型和Motion共用同一次不可变`SimulationHistory`快照；reset、观测和动作commit沿用原规则。startup、`/info`和采样审计记录帧数、时间和RGB身份。闭环客户端可用`--expected-vlm-history-frames 6`检查实际服务，防止测评错误地退回单帧。

## 保持不变的Handover基线

- 数据：原149个训练episode（45个原始、14个random、90个人工演示），59468个完整历史与未来动作窗口；原5个验证episode、固定40个验证窗口。
- 初始化：原UnifoLM联合预训练38000步权重；从任务step0重建优化器、调度器和RNG，不继续训练原Handover5000步checkpoint，也不把GPU检查后的更新当作新实验初始化。
- 随机种子：`20261025`。
- 训练：八卡、每卡batch4、梯度累积1、全局batch32、bf16、任务10000个优化步骤、1000步warmup；保持原五组学习率、优化器、梯度保护与DDP通信配置。
- 模型：原`dual_source_v1`、六层Action Expert、`state_drop_prob=0`、VLM与Motion联合更新、DINO冻结。
- 保存：每1000步保存一次，共10个任务checkpoint；保留全部模型，只保留最近两份优化器恢复状态。原Handover没有Pick三任务的4500/5500/6500额外保存点。

原配置、argv、数据就绪、缓存manifest、训练启动器和窗口选择的只读副本及SHA封存在仓库外的`/home/zxc/CVPR_Evaluation_runtime/handover_qwen_history6_20261113/baseline/`。新实验必须使用独立训练目录；输出路径、实验名和受审源码身份的变化单列为运行命名变更，不混作额外训练超参数。

## 验证与结果解释

新增关键路径需要真实六帧读取和多图Qwen处理、单帧回归、同episode时间对齐、增强一致性、视觉占位符完整性、reset隔离与在线采样审计的CPU检查，并进行真实batch4的完整模型GPU前向/反向、有限梯度检查。正式训练沿用原八卡第2和20步跨rank参数探针与异常保护。

最终需按相同步数、相同场景/seed、成功条件、时限、初始化和RTC条件比较单帧与六帧模型。历史单帧5000步原生结果为28/30（L0/L1/L2为9/9/10），其额外strict指标为19/30，二者不是同一指标。若历史测评原件无法复用，应在同一冻结协议下重测两种输入，不把训练loss下降或历史汇总直接当作因果成功率提升。本文件定义实验；实际GPU检查、训练进度和评测结果以仓库外的真实运行记录为准。

实际训练入口、基线哈希、GPU验收和启动方式见[训练流程](QWEN_HISTORY_HANDOVER_TRAINING.md)。
