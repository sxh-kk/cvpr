# PSI0 `<GLOBAL>` Token 预训练方案

## 目标与范围

在 PSI0 的 Qwen3-VL backbone 中加入一个可学习的特殊 token `<GLOBAL>`，让它汇聚当前图像和 instruction 中与动作任务有关的信息。沿用原工程的 VLM 预训练路线，从 `Qwen/Qwen3-VL-2B-Instruct` 预训练权重初始化，重新完成 EgoDex 约 200K steps，再进行 HE 约 30K steps 的训练。

最终产物是包含新 tokenizer 的 VLM backbone。输入图像和 instruction 后，导出普通 VLM tokens 与 `global_hidden`，供后续自定义模块通过 AdaLN 或其他方式使用。本文所述注册、预训练输入、保存恢复、导出、特征提取和离线评估入口已实现；正式的 200K/30K 训练及表示质量评估尚未执行。

本轮使用原工程的 FAST 动作 token 自回归预测损失，不增加 GLOBAL 人工标签、独立聚合器或额外分类损失，也不要求训练 PSI0 action expert。这里的“重新训练”从已有 Qwen3-VL 权重开始，不是随机初始化整个 VLM。

## 训练路线

| 阶段 | 初始化来源 | 数据与训练量 | 训练目标 | 产物 |
| --- | --- | --- | --- | --- |
| EgoDex 预训练 | 原始 Qwen3-VL 权重，注册 FAST tokens 和 GLOBAL | EgoDex，约 200K steps | FAST 动作 token 的 causal LM 交叉熵 | 带 GLOBAL 的 EgoDex VLM |
| HE 继续预训练 | 上一步导出的模型和 processor | HE，另训练约 30K steps | 相同的 causal LM 交叉熵 | 带 GLOBAL 的最终 VLM backbone |

两阶段都训练 VLM，包括新增 GLOBAL embedding，沿用原预训练脚本的语言模型、视觉编码器和多模态投影器训练范围。不额外设置 GLOBAL warm-up、冻结 action expert 或分阶段解冻流程。

原工程在 VLM 预训练之后另有 action-expert post-training。是否执行取决于最终用途：

| 最终用途 | 是否需要训练 Action Expert |
| --- | --- |
| 导出普通 tokens 和 GLOBAL，连接自己的下游模块 | 不需要；后续训练自己的下游模块 |
| 使用 PSI0 的连续动作预测和机器人执行能力 | 需要训练或适配 action expert，可以从旧权重初始化 |

原 `PosttrainTrainer` 会冻结 VLM，因此该阶段主要让 action expert 适应 backbone 特征，不会继续优化 GLOBAL。不能把它视为本方案 GLOBAL 的主要监督来源。

## 现有代码对应关系

| 文件或入口 | 当前职责及本方案涉及的内容 |
| --- | --- |
| `src/psi/config/transform.py` 的 `Qwen3vlModelTransform` | 构造包含真实 FAST 动作答案的预训练输入和 labels；在答案前插入 GLOBAL |
| `src/psi/trainers/pretrain.py` 的 `PretrainTrainer` | VLM 初始化、词表扩展、FAST tokenizer 创建、自回归训练和 checkpoint 保存 |
| `src/psi/trainers/qwen3vl_mixin.py` | Qwen3-VL 加载、参数冻结设置、优化器分组及 `PaddedCollatorForActionPrediction` |
| `src/psi/tokenizer/fast_action_tokenizer.py` | 注册 FAST 动作 tokens 和动作起止特殊 tokens；需要与 GLOBAL 共享一致的词表初始化流程 |
| `src/psi/config/model_qwen3vl.py` | 保存 GLOBAL 开关、token ID 和输入长度等模型配置 |
| `scripts/train/psi0/pretrain-egodex-psi0-fast.sh` | EgoDex 阶段启动配置 |
| `scripts/train/psi0/pretrain-he-psi0-fast.sh` | HE 阶段启动配置，加载本次 EgoDex 训练产物 |
| `scripts/save_pretrain_qwen3vl_backbone.py` | 导出供下一阶段和下游使用的模型、配置及 processor |
| `src/psi/models/global_vlm.py` | 统一词表初始化、训练与提取输入构建，以及 GLOBAL 特征提取 |
| `src/psi/trainers/pretrain_checkpoint.py` | 保存和校验词表摘要、输入限制及阶段元数据 |
| `scripts/eval_global_vlm.py` | 独立数据上的动作评估、特征导出及线性 probe |
| `tests/test_global_vlm.py` | 小型真实 Qwen3-VL 的梯度、前缀、保存恢复和阶段衔接检查 |

`Psi0ModelTransform`、`Psi0Model.forward` 和 `FinetuneTrainer` 是 diffusion action 路径，不是本轮预训练的主要修改入口。仅在后续接回原 action expert 时，再同步处理这些入口及其推理路径。

## 输入与监督

### 预训练输入

EgoDex 和 HE 使用相同的 token 布局：

```text
[图像、instruction、assistant 前缀、<GLOBAL>、FAST action tokens、结束符]
```

图示省略了 chat template 自带的控制 tokens；实际保留原模板结构。GLOBAL 必须紧邻第一个 FAST 动作 token 之前，并出现在每个样本中恰好一次。

`Qwen3vlModelTransform` 通过共享的 `build_vlm_inputs` 构造 user 消息和包含动作答案的 assistant 消息。完整模板使用 `add_generation_prompt=False`，observation 模板使用 `add_generation_prompt=True`；核对两者的文本 token 边界后，由 processor 展开图像，再在动作答案边界插入 GLOBAL 的单个 token ID。它不会被追加到完整答案之后或插入图像 placeholder 内部。

特征提取时单独使用不含动作答案的模板前缀，并验证它与完整训练输入截至 GLOBAL 的 token IDs 一致。不要通过手工拼接 chat 控制字符串猜测边界。

### Labels 与梯度

主损失保持原预训练的 causal LM 交叉熵：

```text
L = mean CE(logits[:, :-1], labels[:, 1:])，忽略 label == -100 的位置
```

这是损失对齐的示意，实际继续向 Qwen3-VL 传入未移位的 `labels`，由模型内部完成 shift，不要重复移位。

| 输入部分 | Label |
| --- | --- |
| 图像、instruction 和 assistant 前缀 | `-100` |
| GLOBAL | `-100` |
| FAST action tokens | 对应的真实 token ID |
| 原工程监督的答案结束和格式 tokens | 保持原监督规则 |
| Padding | `-100` |

GLOBAL 自身不作为预测目标，但它所在位置的 logits 会预测第一个动作 token。例如：

```text
input_ids: [..., <GLOBAL>, a_1, a_2, ...]
labels:    [...,     -100, a_1, a_2, ...]
                       |
              H_GLOBAL -> lm_head -> 预测 a_1
```

因此第一项动作预测损失可以直接训练 GLOBAL 的表示；后续动作 token 也可以通过 causal attention 读取 GLOBAL 在前面各层的表示。不需要新增 GLOBAL 标签或 action expert 来建立这条监督路径。

labels 根据显式的 observation/answer 边界构造，不再根据尾部答案长度推测。GLOBAL 被忽略，第一个动作仍受监督，模板生成的完整答案结束部分保留监督。

### Causal Attention 与信息泄漏

保持原 VLM 的 causal attention。GLOBAL 能读取图像和 instruction，但不能读取其后的真实动作答案；动作 token 可以读取前面的上下文与 GLOBAL。不要把 VLM 改成双向 attention。

训练时禁止使用 `[图像、instruction、真实动作、GLOBAL]` 作为待导出表示的来源：它会让 GLOBAL 看到真实动作，而且前面的动作预测无法利用这个后置的上下文位置。

## 特殊 Token 与词表初始化

### EgoDex 阶段首次初始化

1. 加载原始 Qwen3-VL 模型与 processor。
2. 沿用原工程注册 FAST 动作 tokens 和动作起止 tokens，再注册 GLOBAL，固定注册顺序。
3. 合并保留已有特殊 tokens；后续 FAST tokenizer 初始化也不能覆盖 GLOBAL 的特殊 token 配置。
4. 确认 GLOBAL 编码为唯一一个 token，记录 `global_token_id`、FAST token 映射和 `action_token_begin_idx`。
5. 根据完整词表和原 embedding 行数确定最终 embedding 大小，统一执行需要的 resize，再创建优化器和分布式模型包装。
6. 用原始普通 token embedding 的均值初始化 GLOBAL 对应的行；确认输入 embedding 与 LM head 的权重共享关系保持正确。
7. 验证 GLOBAL embedding 可训练，且其参数实际进入优化器。

词表注册及 FAST tokenizer 创建均在 `init_models` 中完成，先注册、再按实际需要扩展 embedding，随后创建优化器及数据集。包括 GLOBAL、全部动作 tokens 和动作起止 tokens 的最终词表不会越界，HE 阶段也不会重复增加 `bins` 行。

保留工程现有的 `pad_to_multiple_of=192` 对齐约定，保存实际 embedding 行数。不要因为 tokenizer 长度较小而意外裁掉已有权重。GLOBAL 的 ID 可能落在原有对齐预留行中，必须按 `global_token_id` 初始化，不能假定它是矩阵最后一行。

FAST 动作 token 区间应保持连续，GLOBAL 不得落入动作 token 区间，也不能等于 padding、EOS 或图像控制 token 的 ID。各进程使用同一份映射。

### HE 阶段与断点恢复

加载 EgoDex 阶段保存的完整模型和 processor，复用已有 GLOBAL 和 FAST tokens。初始化流程必须可重复调用：已有 token 不重复注册、不重新编号，GLOBAL embedding 不重新初始化。

恢复时核对 tokenizer 映射、模型配置、embedding 大小与 checkpoint 一致；缺失 GLOBAL 元数据或映射不一致时应报错，不应静默退回原始 tokenizer。

## Padding、截断与多模态位置

先构造每个样本的完整有效序列，再由 `PaddedCollatorForActionPrediction` 做 right padding。GLOBAL 的 `attention_mask` 为 1，padding 为 0，labels 的 padding 为 `-100`。collator 根据样本的真实长度生成 attention mask，同时校验 GLOBAL 与 padding ID 不同。

输入长度预算包括图像展开 tokens、instruction、模板控制 tokens、GLOBAL、完整动作答案和结束符。已移除 collator 的静默序列截断。

长度处理要求：

- instruction 使用固定的 `max_instruction_tokens` 上限，默认 512，完整序列默认上限 `model_max_length=8192`。先单独限制 instruction，再构建模板；训练、评估和提取共用该设置。不能根据真实动作答案长度动态缩短 instruction，否则前缀会不一致。
- 固定限制后仍超长时立即报错并指出样本位置；由使用者调整图像预算、instruction 上限或总长度，不会自动跳过样本或裁断图像块及动作答案。
- 保持 `pixel_values`、`image_grid_thw` 和图像占位 tokens 匹配；图像数量、网格和特征之间的关系不能被截断破坏。
- 初始化时统一模型配置与 `tokenizer.model_max_length`，保存两个长度设置，同阶段恢复时拒绝更改。
- 让 Qwen3-VL 根据最终输入和图像网格计算多模态位置编码，不能复用插入 GLOBAL 之前的 position IDs 或 KV cache。

短样本、不同长度 batch 和接近长度上限的样本都必须验证。

## 训练配置与阶段衔接

### 训练量与优化设置

EgoDex 脚本设置 200,000 次优化器更新，HE 设置另 30,000 次更新。预训练使用 `accelerator.accumulate`，原训练循环只在梯度同步完成后增加 `global_step`。

基于现有预训练启动脚本，第一版沿用以下设置：

| 项目 | 设置 |
| --- | --- |
| 语言模型及其 GLOBAL embedding 的学习率 | `1e-4`，沿用主参数组 |
| Vision tower 学习率 | `1e-5` |
| Multimodal projector 学习率 | `1e-5` |
| 可训练范围 | 开启 `tune_mm_llm`、`tune_mm_vision`、`tune_mm_mlp` |
| 精度与梯度裁剪 | `bf16`，`max_grad_norm=1.0` |
| 学习率调度 | 沿用脚本的 `constant`，`warmup_steps=0` |

GLOBAL 首版随原 embedding 参数训练，不引入单独的 embedding 行学习率、LoRA 或冻结策略。其余优化器设置以复现采用的有效配置为准，保存实际参数组的学习率和 weight decay；当前代码同时存在模型与训练配置中的 weight decay，不能只根据启动命令推断最终值。

保留原数据预处理、FAST tokenizer checkpoint、动作归一化、图像分辨率和动作表示。EgoDex 与 HE 采用顺序训练，不切换成 mixed-data 脚本。记录有效 batch size：

```text
global_batch_size = per_device_batch_size * world_size * gradient_accumulation_steps
```

相同步数在不同有效 batch size 下对应不同的数据曝光量。应对齐复现基线的有效 batch size；checkpoint 通过累加实际 microbatch 样本数并跨进程求和记录 `samples_seen_in_stage`，包含重复采样的数据曝光，不把不足一个累积组的 epoch 尾部按完整 batch 推算。不能仅凭 200K/30K 的名称认定训练量相同。

### EgoDex 到 HE

1. EgoDex 完成约 200K steps 后，导出本次模型、processor 和训练元数据。
2. 将 HE 的 `model_name_or_path` 指向该导出目录，不能继续使用原脚本中不含 GLOBAL 的旧 checkpoint 路径。
3. HE 作为新阶段加载模型权重和 tokenizer，重新建立优化器、scheduler 和阶段内 step，从 0 开始另训练约 30K steps。
4. 同一阶段中断恢复时，才加载该阶段完整的模型、优化器、scheduler、随机状态和进度。

跨阶段加载权重与同阶段 resume 必须区分。不能通过恢复 EgoDex 的 200K step 状态来启动一个 `max_training_steps=30000` 的 HE 任务；不要依赖未经核对的 `resume_from_checkpoint=latest`。

## 保存与导出

EgoDex 和 HE 的 checkpoint 都应保存：

- VLM 权重与模型配置，包括继续 causal LM 训练所需的 LM head。
- 当前训练实际使用的 processor、tokenizer、chat template 及特殊 token 配置。
- GLOBAL 的开关、字符串和 ID，以及 FAST token 映射、词表长度和实际 embedding 行数。
- 输入长度限制、图像处理配置、动作归一化和 FAST tokenizer 来源。
- 当前阶段、阶段内完成步数、初始化 checkpoint、有效 batch size 和有效训练配置。
- 同阶段断点续训所需的 optimizer、scheduler 与随机状态。

`PretrainTrainer.save_checkpoint` 保存模型、实际 processor、配置、`pretraining_metadata.json` 及 Accelerate 完整训练状态。导出脚本从指定 checkpoint 读取这组文件，不会重新加载原始 Qwen processor；输出目录非空时拒绝覆盖。

导出脚本应从 checkpoint 配置与权重恢复实际词表大小，不能再次盲目执行 `len(tokenizer) + bins`。HE 加载和最终下游加载必须使用同目录的模型与 processor。

导出前后用同一输入核对 token IDs、图像处理结果和 GLOBAL hidden state。在相同精度和 attention 后端下设置合理数值容差，确保保存加载没有改变 token 的语义。

## 特征提取接口

特征提取不输入真实动作，也不要求自回归生成动作；一次前向使用：

```text
[图像、instruction、assistant generation prompt、<GLOBAL>]
```

用保存的 processor 构建 `add_generation_prompt=True` 的 observation prompt，追加 GLOBAL ID，然后 padding。验证其有效前缀与训练序列截至 GLOBAL 的部分一致。

VLM 最后一层输出记为 `H`，形状为 `[B, L, D_v]`。接口沿用原 PSI0 的 `Qwen3VLForConditionalGeneration(..., output_hidden_states=True).hidden_states[-1]`。工程固定的 Transformers 4.57.0 在该顶层接口返回 final RMSNorm 之前的 decoder state；内部 `model.model(...).last_hidden_state` 及其最后一个 `hidden_states` 已经过 final RMSNorm，二者不能混用。LM head 实际读取归一化后的表示。提取使用 `logits_to_keep=1` 限制额外 logits 开销，不执行动作生成。GLOBAL 统一按 token ID 提取：

```python
global_mask = input_ids.eq(global_token_id) & attention_mask.bool()
assert torch.all(global_mask.sum(dim=1) == 1)
global_index = global_mask.long().argmax(dim=1)
batch_index = torch.arange(H.shape[0], device=H.device)
global_hidden = H[batch_index, global_index]
```

预训练序列中 GLOBAL 后面有动作答案和结束符，不能使用 `H[:, -1]` 或 `attention_mask.sum(dim=1) - 1` 定位。后一个公式只对以 GLOBAL 结束且 right-padded 的提取输入成立，统一使用 ID 定位更稳妥。

`GlobalVLMFeatures` 接口保留：

| 输出 | 形状或语义 |
| --- | --- |
| `vlm_hidden_states` | `[B, L, D_v]`，与输入对齐 |
| `attention_mask` | `[B, L]`，有效输入位置 |
| `global_index` | `[B]`，GLOBAL 的位置 |
| `global_hidden` | `[B, D_v]`，GLOBAL 表示 |
| `ordinary_mask` | `[B, L]`，有效位置且位于 GLOBAL 之前 |

`ordinary_mask` 可定义为 `attention_mask.bool() & (positions < global_index[:, None])`。实际导出普通 tokens 时按该 mask 取值或保留 padded tensor 与 mask。即使复用完整预训练输入做调试，也不能把动作答案、答案结束符或 padding 当成普通 observation tokens 输出给下游。

`D_v` 从 `model.config.text_config.hidden_size` 读取，不写死。后续若接回原 action expert，还需核对 `views_proj` 的输入维度；现有 PSI0 配置默认值是 1920，而真实训练脚本显式设为 2048。

提取时使用 `eval()` 并关闭梯度与不需要的 cache。保持图像处理和位置编码一致时，causal backbone 的 GLOBAL 表示不应依赖其后的答案。固定权重下，删除末尾 GLOBAL 不会改变前面普通 tokens 的位置和 causal 表示；保留它是为了获得训练过的汇总向量，而不是为了维持前面 token 的位置。

GLOBAL 的范围是当前图像与 instruction，不自动包含未输入 VLM 的 proprioception、历史帧或完整动作轨迹。通过 AdaLN 使用它仍需要训练相应下游模块，不代表新下游可以无需适配直接工作。

## 验证与验收

### 正确性检查

- 两阶段样本均包含且仅包含一个 GLOBAL；它位于第一个 FAST 动作 token 之前，mask 为 1，label 为 `-100`。
- 检查 shift 后 GLOBAL 位置预测第一个动作 token，所有预期受监督动作仍保留 labels，padding 与 prompt 不计入 loss。
- 用短、长、不同长度和多图样本检查 padding、长度限制、图像网格及 position IDs；检查 GLOBAL 和答案均未被误截断。
- 训练小批次检查 loss 有限，GLOBAL 对应的输入表示有梯度，embedding 行实际更新。LM head 与 embedding tied 时，单看该行参数梯度不足以区分输入汇聚路径与输出分类路径。
- 在 `eval()` 下比较完整预训练输入与仅 observation 前缀的 GLOBAL 表示；改变后面的真实动作答案不应改变 GLOBAL，容差考虑后端和精度差异。
- 导出、重新加载及 EgoDex 到 HE 后，检查 GLOBAL/FAST token IDs、embedding 大小和初始化权重保持一致。
- 短程运行检查真实优化器更新次数、checkpoint 恢复进度和 HE 阶段独立计数。

### 表示质量评估

能反向传播是训练成立的条件，不是充分汇总全局信息的保证。动作 token 仍能直接读取原图像和文本上下文，GLOBAL 可能只承担有限的任务信息。因此需要在独立验证数据上评估，不能仅凭训练 loss 下降认定成功。

至少记录：

1. 与匹配训练设置的无 GLOBAL 基线相比，EgoDex 和 HE 验证集的动作 token 交叉熵、准确率与解码动作误差；分别报告 teacher-forcing 结果与无真实答案条件下的生成结果。
2. 冻结 backbone 后，使用相同容量和训练预算的小型下游预测器，比较 GLOBAL、普通有效 tokens 的 masked mean pooling 和原末尾 token 的任务预测能力。无 GLOBAL 基线的原末尾 token 单独报告。
3. 对实际消费 GLOBAL 的下游模块比较正常、GLOBAL 置零、跨样本互换等设置；固定评估随机性，并确保互换样本有不同的观测或任务信息。
4. 分别在 EgoDex 和 HE checkpoint 评估，观察 HE 继续训练后的任务适应与 EgoDex 能力保留。

预训练脚本仍关闭在线验证，`PretrainTrainer.create_dataloaders` 返回的验证 loader 为 `None`。已增加 `scripts/eval_global_vlm.py` 离线入口。现有 HE 数据配置没有实现独立的 `split` 划分，不能直接把它的 `split="val"` 当成独立验证集；评估使用显式准备的独立 JSONL 清单。

普通表示与 GLOBAL 的置零或互换应发生在下游实际消费的位置。只修改 VLM 最终层的 GLOBAL hidden state，不会追溯改变已经算出的其他位置 hidden states，不能据此判断其他动作 token 完全未使用 GLOBAL。

置零或互换后性能不变，只能说明在当前任务和评估设置中未检测到明显贡献，不能直接断定 GLOBAL 不含信息。单看图像增强前后相似度也不足以证明质量，恒定表示同样可能很相似。

第一版完成原两阶段预训练与上述评估，不预设必须加入 global-only dropout。如果发现表示不足，再根据下游需求设计瓶颈或辅助监督实验；若后续 action expert 冻结 VLM，仅在 action expert 侧做 token dropout 不会继续训练 GLOBAL。

## 实施顺序

1. 统一 FAST 与 GLOBAL 的注册、resize、初始化和配置保存，确保在优化器及分布式包装之前完成。
2. 修改预训练 transform 和 collator，完成答案前插入、labels 对齐和长度边界检查。
3. 补齐 checkpoint、backbone 导出和按 ID 提取 GLOBAL 的接口，完成小批次梯度与保存加载验证。
4. 配置本次 EgoDex 约 200K steps，完成训练并导出。
5. HE 加载本次 EgoDex 产物，另训练约 30K steps，再导出并完成表示评估。
6. 当前目标到 backbone 导出结束；以后需要完整 PSI0 动作策略时，再训练或适配 action expert。

## 使用入口

### 环境与两阶段训练

在仓库根目录执行。依赖沿用 `pyproject.toml` 中固定的 Torch 2.7.0、Transformers 4.57.0、Accelerate 1.7.0 和 Qwen VL utils；正式脚本使用 CUDA、bf16、FlashAttention 2 和 DeepSpeed ZeRO-3。原始 Qwen 模型需要预先下载，训练加载禁用自动下载。指定的 `QWEN_CHECKPOINT` 应为原始 Qwen3-VL 权重或其本地缓存标识。

```bash
export PSI_PYTHON=/path/to/psi-environment/bin/python
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export QWEN_CHECKPOINT=/path/to/Qwen3-VL-2B-Instruct
export EGODEX_DATA_ROOT=/path/to/egodex
export HE_DATA_ROOT=/path/to/HE_RAW
export CUDA_VISIBLE_DEVICES=0,1
export TRAIN_BATCH_SIZE=16
export GRADIENT_ACCUMULATION_STEPS=1

bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh

"$PSI_PYTHON" scripts/save_pretrain_qwen3vl_backbone.py \
  /path/to/ego-run/checkpoints/ckpt_200000 --output /path/to/ego-global

bash scripts/train/psi0/pretrain-he-psi0-fast.sh /path/to/ego-global

"$PSI_PYTHON" scripts/save_pretrain_qwen3vl_backbone.py \
  /path/to/he-run/checkpoints/ckpt_30000 --output /path/to/he-global
```

训练输出目录由原工程配置决定，实际路径见启动日志。默认 FAST 来源为仓库的 `src/fast/egodex-rel-50w-1x48-v2048-s100`；两阶段必须使用相同版本，必要时通过 `FAST_CHECKPOINT` 指定路径。`NPROC_PER_NODE` 和 `MASTER_PORT` 可覆盖进程数和端口。默认使用 WandB；仅本地记录时可设置 `WANDB_MODE=offline`。

脚本尾部接受配置覆盖，例如短程检查使用 `--train.max-training-steps=2 --train.checkpointing-steps=1 --train.num-workers=0`。长度限制可通过 `--model.max-instruction-tokens=512 --model.model-max-length=8192` 显式指定。

同阶段恢复使用完整训练 checkpoint，例如：

```bash
bash scripts/train/psi0/pretrain-egodex-psi0-fast.sh \
  --train.resume-from-checkpoint=/path/to/ego-run/checkpoints/ckpt_5000
```

HE 同阶段恢复在 HE 脚本尾部使用同一参数。恢复会检查阶段和词表，加载权重、优化器、scheduler、随机状态及已完成步数；HF 导出目录只用于加载权重，不能替代完整训练状态。数据读取仍沿用原工程逻辑，默认不跳过已经读取的 batches，不承诺逐样本顺序与不中断训练完全一致。不要用 EgoDex 的 resume 状态启动 HE 新阶段。

### 特征提取

```python
from PIL import Image
from psi.models.global_vlm import load_global_backbone, extract_global_features

model, processor = load_global_backbone("/path/to/he-global", device="cuda")
images = [[Image.open("observation.png").convert("RGB")]]
features = extract_global_features(model, processor, images, ["Pick up the object."])
global_hidden = features.global_hidden
ordinary_tokens = features.vlm_hidden_states[0, features.ordinary_mask[0]]
```

每个样本的内层 images 列表包含该样本所有视角。图像须先采用训练时相同的 resize/crop；提取接口负责 processor 的后续处理。默认以 float32、SDPA 加载，也可传入 `dtype=torch.bfloat16` 和 `attn_implementation`，做一致性比较时保持两边设置相同。

### 离线评估

每个 JSONL 记录包含唯一 `id`、图像路径列表 `images`、`instruction` 和二维 `normalized_actions`。图像路径相对清单目录，图像已经过与训练相同的确定性 resize/crop。`normalized_actions` 必须是原 field transform 归一化后、送入 FAST 前的值，当前脚本配置下为 `[1, 48]`；它不是原始世界坐标。不同数据集分别准备清单，并按轨迹或数据来源划分独立集合，不能只更换同一观测的 ID。

```bash
"$PSI_PYTHON" scripts/eval_global_vlm.py /path/to/he-global probe-train.jsonl \
  --device=cuda --dtype=bfloat16 --max-samples=1000 --max-new-tokens=0 \
  --output=eval/probe-train.json --features-output=eval/probe-train.pt

"$PSI_PYTHON" scripts/eval_global_vlm.py /path/to/he-global held-out.jsonl \
  --device=cuda --dtype=bfloat16 --max-samples=1000 \
  --probe-train=eval/probe-train.pt --features-output=eval/held-out.pt \
  --output=eval/held-out.json
```

入口报告 teacher-forcing 交叉熵、FAST token 准确率、归一化动作 MAE、无答案生成 MAE、非法输出和解码失败数量，以及完整输入与 observation 前缀的 GLOBAL 最大绝对误差。MAE 只对成功解码的样本计算，需同时报告有效/失败数量，不能忽略失败样本来比较模型。没有有效解码时 MAE 为 `null`。

`--probe-train` 使用独立训练特征拟合带正则化的线性回归，以相同容量比较 GLOBAL、普通 tokens 均值和普通末尾 token，并在回归器实际输入处做 GLOBAL 置零和跨样本循环互换。脚本拒绝 train/validation ID 重叠；实际观测重复或任务差异需由清单划分保证。该 probe 是首个下游检查，不替代最终 AdaLN 模块的消融实验。

同样入口支持由本次代码保存的无 GLOBAL 基线。训练基线时 EgoDex 使用 `--model.no-use-global-token`，HE 同时使用 `--model.no-use-global-token --model.no-require-global-checkpoint` 并加载对应基线产物；其余数据、步数、长度限制和有效 batch size 对齐。旧 checkpoint 缺少本次元数据时会报错，不会猜测词表或配置。分别对 EgoDex 和 HE 产物运行评估，结果文件由使用者横向比较。

### 本地正确性测试

```bash
PYTHONPATH=src "$PSI_PYTHON" -m unittest discover -s tests -p test_global_vlm.py -v
```

测试使用小型真实 Qwen3-VL 和 CPU Accelerate，覆盖 GLOBAL 注册、labels、长指令前缀一致性、多视角拼接、梯度累积、GLOBAL 输入梯度及 embedding 更新、causal 隔离、导出加载、HE 初始化和离线 probe。它验证实现正确性，不验证完整模型在真实数据上的表示质量，也不替代多 GPU 短程训练。
