# MotionExpert 改进方案：Video-only One-step Denoised Features for Action Expert

状态：设计方案，尚未实施

日期：2026-09-13

本文针对以下目标编写：推理阶段只输入一段历史 ego video，MotionExpert 提取 one-step denoised motion features，交给下游 action expert 与视觉、状态、语言等其他条件共同完成动作去噪；MotionExpert 本身不负责把完整的未来 motion 采样结果作为最终输出。本文只描述分析、实验设计和代码改造边界；写入本文档时不修改模型代码。

## 1. 结论摘要

推荐将部署目标定义为：

```text
历史 ego video -> DINO 特征 -> video-only MotionExpert ->
{H=one_step_hidden [B,T,768], G=one_step_global [B,768]} ->
action adapter -> action expert 在每个动作去噪步融合 H/G 与其他条件
```

这里的“one-step token”是一个成对的特征集合，而不是二选一：`one_step_hidden` 和 `one_step_global` 都必须输入下游。`one_step_hidden` 保留历史帧的时间结构，`one_step_global` 提供窗口级摘要；二者必须来自同一份 MotionExpert 噪声、同一时间步和同一次前向。

第一版建议如下：

1. 保留现有 E7-K12-400M 主干、x0 Flow Matching、10-step Euler 和 153 维 motion 表示。
2. 在已有 action supervision 的适配阶段，将 `video_history` 作为主训练任务（约 80%--100%），使 H/G 学习历史状态、速度和动作阶段。
3. 将 `video_forecast` 仅作为可选辅助任务（约 0%--20%）；若只是无 action 标签的 standalone 预训练，可暂用约 70%/30% 的 history/forecast 比例，之后以 action loss 重新校准。
4. `generation` 默认关闭（0%）；单帧生成与“历史视频条件特征”不是同一个部署分布。
5. 主训练分支中 trajectory 和 text 都使用显式 null/mask，不要依靠低概率随机 dropout 模拟缺失模态。
6. 历史长度按时间跨度采样，而不是固定为 5 张图片；先测试 0.5、1、2、4 秒，默认从 2 秒开始。
7. 同时保留两个语义清晰的候选：`feature_step=0` 是最低延迟的一次前向 baseline，`feature_step=1` 才是完成一次 Euler 更新后的 literal one-step denoised feature；若“denoised”是硬性产品定义，默认应选后者。
8. 第一阶段仍不增加 global 专用预测头；接入 action expert 后，以 action loss 验证 H/G 是否有用，并保留小权重 motion loss 防止 MotionExpert 遗忘。
9. 默认采用“一次提取、动作去噪循环复用”的固定条件模式；只有需要严格同一时间轴时，才启用每个动作去噪步重新计算 MotionExpert 特征的同步模式。

最重要的原则是：训练时的条件、有效帧、噪声时刻和 hidden 读取位置必须与部署时完全一致。只在推理时把 trajectory 和未来视频置空，不能弥补训练分布不一致。

### 与现有 `project.md` 的关系

现有 [`project.md`](project.md) 描述的是 global-token 的原始多模态基线：保留 trajectory/text、沿用原 motion loss，并从第一次 `t=1` 模型调用提取 hidden/global。本文件不是对该基线的回溯，而是针对“部署时只有历史 ego video、H/G 共同作为 action expert 条件”这一新目标提出的后续实验方案。建议先保留原基线 checkpoint 作为 A0，对比 video-only 改进 checkpoint；在完成消融前，不要把两种 checkpoint 的 H/G 混用。

## 2. 目标与假设

### 2.1 本文采用的部署假设

- 输入只有历史 ego video。
- 没有 Aria/SLAM trajectory 或其他外部相机位姿。
- 没有文本条件。
- 下游同时使用 `one_step_hidden [B,T,768]` 和 `one_step_global [B,768]`，二者都是必需输入，不是主/备关系。
- action expert 自己的动作去噪 timestep 与 MotionExpert 的 feature timestep 必须显式定义映射，不能把一次 `t=1` 提取称为每个动作步都同步。
- hidden 提取必须低延迟，优先考虑一次或少量模型前向。

如果实际部署仍会提供 trajectory 或 text，应单独训练一个多模态分支，不能把两种分布混为一个默认实验。

### 2.2 输入只有历史视频，不等于一个确定的任务

任务由输出目标决定：

- 只要历史动作/历史上下文特征：这是 video-only history reconstruction/encoding。
- 要预测未来动作：这是 video-only forecasting。
- 只有一张图仍要生成完整 motion：才是 generation。

因此，原来的三个名字不能直接照搬。对于给 action expert 使用的 H/G，历史重建/编码应当是主任务，因为部署时二者只由历史视频产生；forecasting 只作为小比例的动态先验正则，不能取代历史条件目标。`generation` 与当前接口无关。

### 2.3 MotionExpert 与 action expert 的边界

MotionExpert 输出的是 action expert 的条件，不是 action expert 的中间动作 latent，也不应直接替代 action expert 的 `x_t`。推荐把一次推理写成：

```text
(H, G) = M(history_ego_video; z_M, t_M)
for s in action_schedule:
    action_pred_s = A(action_state_s, s, H, G, other_conditions)
    action_state_{s+1} = solver_step(action_state_s, action_pred_s, s)
```

这里有两种需要严格区分的“同步”：

1. **固定条件（P0，推荐起点）**：MotionExpert 只在开始时准备一对 H/G；低延迟 baseline 直接取 `feature_step=0`，若要求 literal denoised 则先做一次 Euler、取 `feature_step=1`。动作 expert 在自己的每个去噪步复用同一对 H/G。动作 expert 的 `s` 仍然每步变化，因此这是“联合动作去噪”，但不是 MotionExpert 在每步同步积分。
2. **时间步同步（P1）**：动作 expert 的每个 `s` 都映射到 `t_M=phi(s)`，并重新得到 `H_s/G_s`。若要称为严格同步，MotionExpert 还必须维护自己的 noisy motion latent 并按同一 solver 更新；仅仅用同一历史视频、每步重新采样一份独立高斯噪声，不是连续的 denoising trajectory。

固定条件模式的计算量约为一次 MotionExpert 前向；严格同步模式的计算量约为 `N_action_steps` 次，且需要对两个 scheduler 的归一化噪声/时间定义做映射。除非下游实验显示固定 H/G 不足，否则不建议第一版直接采用严格同步。

当前代码还存在一个容易被忽略的实现差异：MotionExpert 的采样时间是归一化的线性序列 `1.0, 0.9, ...`，而 Psi0 action expert 使用 `FlowMatchEulerDiscreteScheduler` 的离散 timestep，并通过 `sigmas` 表示噪声强度（注释中的序列约为 `1.0, 0.889, ...`）。因此 aligned 模式应按噪声强度/归一化 sigma 定义 `t_M=phi(s)`，不能直接把 action scheduler 的整数 timestep 当作 MotionExpert 的 `t`。

MotionExpert 的 `z_M` 与 action expert 的 action noise 还应使用独立 RNG stream。除非专门研究耦合噪声，否则不要因为“同步”就复用同一个随机张量；固定条件模式只需记录 MotionExpert seed，action expert 仍按自己的噪声策略运行。

## 3. 当前实现审计

### 3.1 当前 one-step hidden 的真实含义

当前 [`MotionExpert.extract_features`](motion_expert.py) 的逻辑是：

```text
epsilon ~ N(0, I)
x_1 = epsilon
h_0, g_0 = model(x_1, t=1, condition)
return h_0, g_0
```

`model` 同时输出 `pred_xstart`、逐帧 hidden 和 global hidden。hidden 是 Transformer 最后一层 token 的状态，不是把 `pred_xstart` 再编码一次得到的表示。

因此当前 `one_step_hidden` 更准确的名称是：

```text
hidden_at_initial_noise (t=1, before Euler update)
```

它不是完成一次 Euler 更新后的 hidden。当前 [`FlowMatching.sample_loop_progressive`](mydiffusion/flow_matching.py) 只在第一轮模型调用时保存 hidden；10-step 采样中第一轮的时间为 `t=1`，随后才更新到 `t=0.9`。

### 3.2 `one_step_hidden` 与 `one_step_global`

当前 global 版本返回：

```text
one_step_hidden  : [B, T, 768]  # 每个 motion token
one_step_global  : [B, 768]    # <GLOBAL> token
```

下游同时使用 one_step_hidden [B,T,768] 和 one_step_global [B,768]。前者作为逐帧时序条件，后者作为窗口级全局条件；两者必须来自同一次 feature forward、同一 `feature_t`，而不是分别从不同采样步读取。

### 3.3 当前训练和目标部署的差异

当前 [`utils/task_conditioning.py`](utils/task_conditioning.py) 定义了 `recon/fore/gen` 的显式条件 mask，但根据训练入口，[`dataset/ee4d_motion_dataset.py`](dataset/ee4d_motion_dataset.py) 中的 `process_sample_for_task` 没有被训练流程调用。训练主要通过 [`model/uniegomotion.py`](model/uniegomotion.py) 中的独立随机 mask：

- trajectory 整体 mask 概率为 `0.5`；
- image 整体 mask 概率为 `0.1`；
- 后续帧 suffix mask 概率为 `0.5`；
- text dropout 默认约为 `0.1`。

这种策略不是“连续历史前缀可见、未来图像不可见、trajectory/text 永远缺失”。另外，[`module/uem_module.py`](module/uem_module.py) 当前对完整 motion 计算 Flow Matching loss，虽然 `flow_matching.py` 支持可选 `loss_mask`，但任务级 loss mask 没有成为训练采样的核心控制项。

当前 [`MotionExpert._prepare`](motion_expert.py) 仍要求 trajectory 输入，配置校验也默认 `COND_TRAJ=True`、`COND_TEXT=True`。这说明 video-only 不只是换一个推理参数，后续还需要明确的 null modality 接口。

### 3.4 当前数据时间尺度

根据 [`README.md`](README.md)：

- motion 是 10 FPS；
- DINO sidecar 是 5 FPS；
- DINO 特征会对齐并复制到 10 FPS motion slot；
- 标准窗口是 80 个 motion frame，即 8 秒；
- 当前 forecast prefix 默认是 `window // 4 = 20` 个 motion frame，即 2 秒。

因此“5 帧”必须说明是哪一种帧：

| 计数对象 | 5 帧对应的时间 | 备注 |
| --- | ---: | --- |
| 10 FPS motion slot | 0.5 秒 | 可能只对应约 3 个不同的 DINO 特征 |
| 5 FPS 不重复 DINO 帧 | 1 秒 | 约 5 个视觉观测 |
| 30 FPS 原始视频连续帧 | 0.17 秒 | 对动作趋势通常过短 |

## 4. “一次去噪”必须区分两种定义

### 4.1 方案 A：当前的一次模型前向，推荐作为默认

当前 Flow Matching 使用：


\[
x_t=(1-t)x_0+t\epsilon,
\qquad t=1 \Rightarrow x_t=\epsilon
\]

模型预测 clean estimate：

\[
\hat{x}_0=f_\theta(\epsilon,1,C)
\]

并在同一次前向中读取：

\[
h_0, g_0=\operatorname{hidden}(f_\theta(\epsilon,1,C))
\]

优点：只需一次前向，和当前 `extract_features` 兼容，延迟最低。

缺点：hidden 仍受初始随机噪声影响；它表示的是“高噪声条件下的 denoising state”，不是稳定的 clean-motion encoder。

第一版应保留该语义，新增 metadata：

```text
feature_t = 1.0
feature_step = 0
feature_calls = 1
```

不要在不改字段名的情况下悄悄改变 `one_step_hidden` 的含义。

### 4.2 方案 B：完成一次 Euler 更新后再读取，作为对照实验

当前 10-step Euler 的第一步为：

\[
v_1=\frac{x_1-\hat{x}_0}{1},
\qquad
x_{0.9}=x_1-0.1v_1
\]

然后再次调用模型：

\[
h_1,g_1=f_\theta(x_{0.9},0.9,C)
\]

这个 `h_1/g_1` 才是严格意义上的“完成一步 Euler 去噪后的 hidden”。它需要两次模型前向，不能通过 `num_steps=1` 直接得到，因为 `num_steps=1` 仍然只在 `t=1` 调用一次模型。

建议增加可配置接口，而不是覆盖旧接口：

```text
extract_features(..., feature_step=0)  # 当前 t=1，一次前向
extract_features(..., feature_step=1)  # 一次 Euler 后，第二次前向
```

对于 `feature_step=k`，返回第 `k` 次模型调用的 hidden，并在 metadata 中记录实际 `feature_t`。当前 10-step schedule 下，step 0/1/2 对应约 `t=1.0/0.9/0.8`。

### 4.3 不建议第一版直接改成“one-step motion model”

一个多步 Flow Matching 模型在 `t=1` 的第一次预测，不等价于经过专门训练或蒸馏的单步生成模型。REWIND 的研究表明，若目标是用单步 denoising 直接得到高质量 ego motion，通常需要 diffusion distillation，并且还要使用因果时序结构；简单截取多步模型的第一步不能保证单步 motion 质量。[REWIND](https://arxiv.org/abs/2504.04956)

本项目当前目标是提取 hidden，而不是用一步结果替代完整 motion，因此不应一开始就引入单步蒸馏。只有在后续验证发现 `t=1` hidden 不适合作为特征时，才考虑：

1. 用完整 10-step 模型作为 teacher；
2. 让 `t=1` student hidden 对齐 teacher 的晚期 hidden 或多步聚合 hidden；
3. 同时保留 motion denoising loss，避免 global token 只学到蒸馏目标。

这是二期方案，计算和实验复杂度明显更高。

### 4.4 “one-step denoised token”的命名和验收

目前容易把三个不同概念混在一起：

| 名称 | MotionExpert 调用 | 得到的内容 | 是否可称为 denoised |
| --- | ---: | --- | --- |
| `feature_step=0` | 1 次，`x_1=epsilon,t=1` | 初始高噪声状态的 H/G | 严格说不是，建议标成 `initial-noise feature` |
| `feature_step=1` | 先在 `t=1` 做一次 Euler，再在 `t=0.9` 调用 | 一次更新后的 H/G | 是 literal one-step denoised feature |
| `aligned(s)` | 每个 action step 都推进并读取一次 | 随动作时间变化的 H_s/G_s | 是同步 feature，但成本约随 action 步数增长 |

因此，若下游只需要一份上下文并在 action expert 的循环中复用，推荐接口为 `feature_mode=static, feature_step=1`；`feature_step=0` 作为延迟/质量 baseline。若预算只能允许一次前向，也可以用 step 0，但文档、metadata 和实验名称必须明确写成 initial-noise feature，不能把它与 literal denoised token 混为一谈。

`num_steps=1` 不能替代 `feature_step=1`：在现有采样器中它仍只调用 `t=1` 一次，随后直接结束。验收时应检查模型调用次数、实际 `feature_t` 和 action expert 看到的特征，而不是只检查参数名。

## 5. 研究结论与对本项目的启示

### 5.1 UniEgoMotion 的任务定义不能原样套用

UniEgoMotion 原论文将 reconstruction 定义为逐帧 ego video 加 trajectory 的 motion recovery，将 forecasting 定义为历史视频/trajectory 条件下的未来 motion，将 generation 定义为单帧图像生成。论文的实验设置是 10 FPS、80 帧窗口，forecast 使用前 2 秒输入并预测后续 6 秒；原始训练主要在 reconstruction 和 generation 之间进行随机 masking。[UniEgoMotion 原论文](https://arxiv.org/html/2508.01126)

你的部署没有 trajectory，也没有未来图像，因此应把条件分布改为 video-only prefix，而不是只沿用原论文的 recon/gen 比例。

### 5.2 trajectory 缺失会影响绝对运动，但不一定破坏局部语义

UniEgoMotion 的无 trajectory 消融显示，绝对 head tracking 和 MPJPE 明显变差，但 Procrustes-aligned pose、semantic similarity 和 motion realism 仍相对可用。论文解释这是因为模型被迫隐式解决 visual odometry 问题。[UniEgoMotion trajectory ablation](https://arxiv.org/html/2508.01126)

因此 video-only hidden 应优先被解释为：

- 历史动作状态；
- 局部姿态和动作语义；
- 场景相关的运动倾向。

不能默认它包含可靠的 metric-scale 绝对头部轨迹。如果下游确实依赖绝对位姿，应增加 visual odometry/SLAM/IMU，或重新定义相对 canonical motion 目标。

### 5.3 timestep 会改变表示内容

扩散模型的中间表示不是与 timestep 无关的通用 embedding。已有研究发现，不同 timestep 的 representation 会携带不同尺度或不同程度的语义信息，最佳 timestep 依赖下游任务；因此 `t=1` 不是理论上必然最优的特征读取点，应当做 timestep/step 消融。[Diffusion Model as Representation Learner](https://arxiv.org/abs/2308.10916)、[Exploring Diffusion Time-steps for Unsupervised Representation Learning](https://arxiv.org/abs/2401.11430)

对本项目而言，至少应比较：

```text
step 0: t=1.0, 1 次前向
step 1: t=0.9, 2 次前向
step 2: t=0.8, 3 次前向
step 4: t=0.6, 5 次前向
full : 最后一次 denoising hidden
```

比较指标应包含下游效果、跨 seed 稳定性和延迟，而不能只看 motion MSE。

### 5.4 真正的 online 场景需要因果结构

当前主干对有效 motion token 使用双向 attention。如果特征模式只保留历史 token，这不会产生未来泄漏；但如果保留 `n+H` 个未来 slot，global token 会读取未来随机噪声。对于严格 streaming 的未来预测，还需要 causal attention、rolling window 或 per-token noise schedule。

Diffusion Forcing 将因果 next-token prediction 与序列 diffusion 结合，EgoForce 进一步使用历史/未来不对称噪声和渐进 denoising 来处理在线 ego motion。这些工作说明严格 online 是独立的模型设计问题，不应仅通过把输入帧数改成 5 来解决。[Diffusion Forcing](https://arxiv.org/abs/2407.01392)、[EgoForce](https://arxiv.org/abs/2605.13041)

## 6. 推荐的训练目标

### 6.1 统一的 prefix-conditioned 目标

MotionExpert 的 standalone 训练目标应统一成历史前缀条件目标；它是给 H/G 学习可用运动状态的代理目标，不是下游动作输出目标：

\[
p_\theta(X_{1:n+H}\mid I_{1:n},\varnothing_{traj},\varnothing_{text})
\]

主 feature 样本取 `H=0`（只处理历史窗口）；只有 forecast 辅助样本才取 `H>0`。

其中：

- `n` 是历史长度；
- `H` 是未来长度；
- 只有 `I_1...I_n` 可见；
- 所有 trajectory 条件为 null；
- 所有 text 条件为 invalid/null；
- 输出 token 的有效性与条件可见性分开表示。

`reconstruction`、`forecasting` 和 `generation` 是这个目标的边界情况，不应是三个互相独立且比例固定的分布。真正决定 H/G 是否有用的最终指标是接入 action expert 后的 action loss/控制指标。

### 6.2 任务比例

针对“历史视频 -> H/G -> action expert”的第一版比例：

| 任务 | 条件 | 主要监督 | 初始比例 |
| --- | --- | --- | ---: |
| `video_history` | 历史图像；trajectory/text null；只保留历史输出 token | 历史 motion reconstruction/encoding | 80%--100% |
| `video_forecast` | 历史图像；未来图像 mask；trajectory/text null；导出时只取历史 token | 动态先验正则，不作为部署特征定义 | 0%--20% |
| `single_frame_gen` | 单帧图像；trajectory/text null | 与历史视频条件不匹配的单帧 prior | 0% |

如果下游明确需要 MotionExpert 自己预测未来 motion，才另存一个 forecast-oriented checkpoint；不要把该 checkpoint 的 future-aware H/G 与当前纯历史 action-conditioning checkpoint 混用。

比例要按是否已有 action supervision 区分：如果只是 standalone 预训练、暂时没有 action expert 的标签，可将 `video_history/video_forecast` 设为约 `70%/30%`，用 forecast 补充速度和动作趋势；一旦有真实 `L_action` 参与联合训练，建议把 forecast 降到 `0%--20%`，避免代理的未来 motion 目标压过下游动作目标。两种阶段应在实验记录中明确标注，不能把比例解释成固定真理。

### 6.3 条件 mask、attention mask 和 loss mask

必须保持三种 mask 的区别：

| mask | 作用 | `video_history` | `video_forecast` |
| --- | --- | --- | --- |
| `valid_frames` | motion token 是否参与 attention/输出 | 历史段有效，padding 无效 | 历史+未来目标段有效，padding 无效 |
| `valid_img_embs` | image context 是否存在 | 历史图像有效 | 只有历史图像有效 |
| `img_mask` | image 条件是否被 mask token 替换 | 历史可见 | 未来全部 mask |
| `traj_mask` | trajectory 条件是否可见 | 全部 mask | 全部 mask |
| `text_valid` | text 条件是否有效 | false | false |
| `loss_mask` | 哪些 motion frame 计算监督 | 历史段 | 未来为主，历史为辅 |

不能用 `valid_img_embs` 直接当作 motion attention mask。未来 motion token 在 forecast 中是要生成的输出，不是 padding。

对于导出给 action expert 的 history feature，建议始终使用 `T=n`。如果为了辅助 forecast 使用 `n+H` 个 slot，必须保证导出的 H/G 不读取未来 slot（例如单独的 history 分支/attention mask），否则 global 会把未来初始噪声混入表示，造成训练-部署泄漏。

### 6.4 history/future 加权 loss

对 forecast 样本使用：

\[
L_{motion}=L_{future}+\lambda_hL_{history}
\]

建议初值：

```text
future weight = 1.0
history weight = 0.25--0.5
```

历史段和未来段分别按有效 frame 数归一化，避免较长未来窗口自动压制历史 feature 监督。继续使用现有的 feature-valid mask、153 维表示和 global motion feature 权重。

第一版不建议添加大量几何、速度或对比损失。若出现明显的未来边界跳变，再增加小权重的 boundary velocity/contact loss 做二期实验。

### 6.5 trajectory 和 text 的处理

如果最终部署永远没有 trajectory/text：

- 主训练分支中 100% 使用 null trajectory；
- 主训练分支中 100% 使用 invalid text；
- 使用 learned mask token 或显式 modality-valid mask；
- 不要把未归一化的全零 trajectory 当成真实 trajectory；
- 不要重复最后一帧图像来填未来图像条件。

如果一个 checkpoint 必须同时支持多模态和 video-only，则建议在独立任务比例中加入 70%--80% video-only、20%--30% trajectory-present 样本，并明确记录 modality/task id。不要让默认的 0.5 trajectory dropout 代替这个设计。

### 6.6 接入 action expert 后的联合目标

当 action expert 的训练样本与历史视频在时间上对齐后，action loss 才是检验 H/G 的直接目标。对 action expert 的一个随机 flow step `s`，可写为：

```text
(H, G) = MotionExpert(z_M, t_M, history_ego)
v_hat_A = ActionExpert(a_s, s, P_H(H), P_G(G), other_conditions)
L_action = ||v_hat_A - v_A||^2
```

建议的联合损失为：

\[
L = L_{action} + \lambda_m L_{motion} + \lambda_c L_{cons}
\]

其中 `L_motion` 是 MotionExpert 的历史 denoising 代理损失，`L_cons` 只在 seed 方差确实影响 action 性能时启用。第一版可令 `lambda_m` 为小权重（例如 `0.1--0.5` 的量级，需按 loss 数值归一化后再调），不要让 motion MSE 压过 action objective。

推荐三阶段训练：

1. **Standalone 预训练/适配**：video-only history 为主，获得可复现的 H/G；此阶段不需要 action expert 反向传播。
2. **接口稳定阶段**：冻结 MotionExpert，训练 `P_H/P_G` 和 action expert，验证同时使用 H/G 是否改善 action loss。固定条件模式下每个 action step 复用同一 H/G，必须按这个方式训练和验证。
3. **小范围联合微调**：只解冻 MotionExpert 最后若干层及 global 路径，使用较小学习率，并保留 `lambda_m L_motion` 防止 H/G 退化成只适配单一 action 数据集的 shortcut。

如果训练时把完整 action denoising trajectory 展开，action-step loss 应按步平均后再回传，避免同一份 H/G 因重复使用而得到与步数成比例的梯度放大。若只在随机 action timestep 训练，则不存在这项额外放大。

### 6.7 MotionExpert 样本与 action 样本的时间对齐

每个 action training sample 都应以动作预测时刻 `t_0` 为边界：MotionExpert 只能读取 `t_0` 之前的历史 ego 帧，action target 从 `t_0` 或约定的控制延迟之后开始。不得用包含 action chunk 未来帧的 MotionExpert feature 做训练输入，否则 action loss 会得到未来信息泄漏。

H 的 `T` 个历史位置不应简单复制成 action horizon；应通过 `frame_mask` 和 cross-attention 保留原始时间顺序，让 action expert 自己学习历史到动作 chunk 的对应关系。G 只表示当前 history window，不能被解释成跨窗口记忆。

## 7. 历史长度和采样策略

### 7.1 不固定为 5 帧

历史长度应按秒定义，再转换为 motion slot：

| 历史跨度 | motion frame（10 FPS） | 不重复 DINO frame（5 FPS） | 适用假设 |
| ---: | ---: | ---: | --- |
| 0.5 秒 | 5 | 约 3 | 极低延迟，动作趋势可能不足 |
| 1 秒 | 10 | 5 | 短历史基线 |
| 2 秒 | 20 | 10 | 推荐第一基线 |
| 4 秒 | 40 | 20 | 更强动作/场景上下文 |

原 UniEgoMotion 使用前 2 秒作为 forecasting 观察段，因此 2 秒是与现有窗口和论文设置都兼容的起点；并不意味着 2 秒对 video-only hidden 一定最优。[UniEgoMotion 实验设置](https://arxiv.org/html/2508.01126)

### 7.2 训练中的长度分布

第一轮可从 `{5, 10, 20, 40}` 个 motion frame 中随机选取，概率向 20 帧倾斜，例如：

```text
n=5   : 0.15
n=10  : 0.25
n=20  : 0.45
n=40  : 0.15
```

这只是实验起点。最终分布应根据部署延迟预算和下游验证集性能调整。所有样本满足 `n+H <= 80`，并保留随机起点、随机短缺帧和少量 feature dropout，以模拟真实视频丢帧。

### 7.3 原始视频采样

如果原始视频是 30 FPS，不要只取连续的前 5 帧。应先确定时间跨度，再在跨度内均匀采样到 5 FPS 或 10 FPS 的模型输入率。否则模型看到的是 0.17 秒的微小变化，而不是可用于判断速度和动作阶段的历史。

对 action expert 接口，历史窗口应以动作预测时刻为右端点向过去取样；在线每次更新时滚动窗口，而不是固定取 episode 开头的 5 帧。这样 H 的最后一个 token 才与当前 action denoising context 对齐。

## 8. one-step feature 的稳定性

### 8.1 初始噪声导致的 seed 方差

在当前定义下：

\[
g_0=g_\theta(\epsilon,1,C)
\]

因此相同视频在不同 seed 下不一定得到相同 H 或 G。固定 seed 只能保证复现，不能证明两路表示本身稳定；而且对 H 做多 seed 平均可能抹平逐帧时间差异，不能只平均 global。

必须增加以下诊断：

- 同一条件下 K 个 seed 的 global cosine similarity，以及 hidden 的逐帧 cosine/variance；
- H、G 各自的每个维度和整体范数跨 seed variance；
- 单 seed、固定 seed、H/G 成对 K-seed 平均三种设置的下游性能；
- `feature_step=0/1/2` 的稳定性和延迟。

### 8.2 推理时的处理顺序

建议按以下顺序实验：

1. 固定 seed，建立可复现 baseline；
2. 分别测试 K 个 seed 的平均 G，以及 H/G 成对聚合（不要只聚合其中一路）；
3. 如果下游需要不确定性，保留 seed 维度或同时输出均值/方差；
4. 不要未经训练把初始高斯噪声替换成全零；
5. 只有在 seed 方差明确影响下游时，才加入 noise-consistency loss。

可选的一致性损失为：

\[
L_{cons}=1-\cos(g_\theta(\epsilon_a,1,C),g_\theta(\epsilon_b,1,C))
\]

它必须使用很小权重，并保留 motion denoising loss；否则 global token 可能塌缩成与输入无关的常量。建议把它放在二期，而不是第一版默认方案。

### 8.3 global token 是否需要直接监督

当前 global token 通过对 motion token 的影响获得间接梯度，但没有直接 global label。这在结构上是可行的，却不能保证它自动成为最好的下游摘要。

接入 action expert 后情况不同：如果 `P_G(G)` 真正参与 action prediction，`L_action` 会沿 `G -> P_G -> action expert` 提供直接的下游梯度；H 也会沿 `P_H(H)` 获得直接梯度。因此不要在 standalone 阶段为了“给 global 找标签”过早增加专用 head，先用真实 action loss 验证两路特征的用途。

第一版保持现状，不增加 global head。若下游实验显示：

- motion 指标正常；
- per-frame hidden 有效；
- 但 `one_step_global` 几乎没有下游信息；

再考虑一个小权重的 global 辅助目标，例如预测历史 motion 统计量、速度/接触状态或下游监督标签。新增 head 必须只作为训练工具，不能未经训练直接把随机 readout 当作 feature。

## 9. 训练阶段方案

### 9.1 阶段 0：现有 checkpoint 的 zero-shot 基线

在不改权重的情况下，构造 video-only mask，测量：

- 当前 `t=1` global/hidden 的下游性能；
- full multimodal 条件下的性能；
- trajectory/text 被置空后的性能差距；
- 不同历史长度和 seed 的方差。

这个基线用于判断问题主要来自 modality gap、历史长度，还是 hidden timestep。

### 9.2 阶段 1：部署分布 fine-tune

优先使用已有 motion checkpoint 做低学习率适配：

- 保留 DINO 特征和 projector，先不重训 DINO encoder；
- 使用 video-only task sampler；
- 低学习率可从原学习率的 0.1--0.3 倍开始；
- 先训练若干 epoch 观察 hidden seed variance 和下游指标；
- 再逐步解冻主干。

如果 trajectory 依赖很强，fine-tune 仍无法恢复局部/语义指标，再做从头训练的 video-only checkpoint。若最终永远没有 trajectory/text，独立 checkpoint 通常比强行兼容两种模态更容易解释和验证。

### 9.3 阶段 2：接入 action expert 的接口训练

这一阶段的目标不是让 MotionExpert 单独生成更好的未来 motion，而是让 H/G 在 action expert 的真实去噪过程中提供有效条件。建议分两步：

```text
B1: freeze MotionExpert; train P_H, P_G and action expert
B2: unfreeze MotionExpert last K blocks + global path; keep a small L_motion
```

固定条件模式下，训练和推理都应在 action expert 的每个随机/枚举 `s` 上复用同一 H/G；不要训练时使用 clean-motion hidden、完整采样 hidden，推理时却改回 `t_M=1`。应同时做三组对照：

| 对照 | MotionExpert feature | 目的 |
| --- | --- | --- |
| C0 | 不接 H/G | action expert 原始基线 |
| C1 | 只接 H/G（固定 `t_M=1`） | 测量特征增益 |
| C2 | 接 H/G，并按 `t_M=phi(s)` 更新 | 测量时间步同步是否值得额外计算 |

action loss 应按 action timestep 平均；若 B2 采用完整 unroll，还要对 denoising step 求平均，避免同一 H/G 被重复使用而导致梯度随步数放大。先冻结再解冻也能把“接口是否有效”和“MotionExpert 是否需要联合适配”分开判断。

### 9.4 阶段 3：endpoint/timestep 对齐

保留大部分原有连续 `t` 采样，额外加入少量高噪声端点样本：

```text
80%--90%: 原有连续 t 分布
10%--20%: t in [0.95, 1.0]，或少量精确 t=1
```

这一步只在 `t=1` feature baseline 较弱时启用。不要把全部训练固定到 `t=1`，否则模型会偏向最高噪声端点，可能损害完整 denoising 和中间 timestep 的行为。

如果选择 `feature_mode="aligned"`，不能只做端点 oversampling：应从 action scheduler 的实际 `sigma` 节点采样/分层采样 `t_M=phi(s)`，并在同一 MotionExpert latent trajectory 上读取 H_s/G_s。每个 t 独立重新采样 noise 会让 action expert 学到时间标签而不是连续的 denoising condition。

### 9.5 阶段 4：可选蒸馏/一致性

只有当下游明确需要更低 seed variance 或更强单步语义时，才评估：

- teacher full 10-step hidden/global；
- student t=1 hidden/global；
- noise-consistency；
- feature distillation；
- 或真正的 one-step motion distillation。

这一步应单独保存 checkpoint，不能与基础 video-only checkpoint 混用。

## 10. 文件级修改方案

本节是后续实施顺序，不代表本次已经修改。

### P0：先定义特征合同

涉及：`motion_expert.py`、`mydiffusion/flow_matching.py`、README。

1. 保留现有 `one_step_hidden`/`one_step_global` 字段以兼容旧调用。
2. 增加 `feature_step` 或 `feature_t` 参数。
3. 低延迟 baseline 使用 `feature_step=0`（当前 `t=1` hidden）；若接口的“denoised”语义是硬约束，则默认候选使用 `feature_step=1`。
4. `feature_step=1` 时执行一次 Euler 更新，再进行第二次模型前向。
5. metadata 记录：`feature_t`、`feature_step`、`feature_calls`、seed、task、条件模态和有效 frame mask。
6. 明确 `one_step_global` 是窗口级单 token，`one_step_hidden` 是逐帧 token。
7. 默认关闭 CFG。CFG hidden 是条件/无条件分支的线性外推，不应与普通 hidden 混用。

### P0：定义给 action expert 的联合特征合同

涉及：MotionExpert 输出包装、action adapter、action expert 输入层。

下游接口必须把两路特征作为同一个 `motion_context` 传递，不能只保留 global：

```text
motion_context = {
    hidden:      [B, T, 768],   # 历史逐帧 H
    global:      [B, 768],      # 同一次前向得到的窗口级 G
    frame_mask:  [B, T],        # H 的有效历史帧
    feature_t:   scalar or [B],
    feature_step: int,
    noise_seed:  int or seed metadata,
}
```

建议的第一版融合方式：

- `H` 经过独立 `P_H: 768 -> d_action`，作为 action expert 的 cross-attention K/V 或时序 observation tokens；保留帧顺序和 `frame_mask`。
- `G` 经过独立 `P_G: 768 -> d_action`，作为窗口级 AdaLN/FiLM 条件，或作为一个明确标记的 global condition token。
- 不要未经实验把 G 复制到每个 H 帧后再同时走两条路径；这会重复计权且难以判断 global 的真实贡献。
- action timestep embedding `e_A(s)` 与 G 分开投影/门控；除非已验证尺度兼容，不要直接把 G 加到 timestep embedding 上。

固定条件模式使用 `feature_mode="static"`，只准备一次 MotionExpert 特征（step 0 为 1 次调用，literal step 1 为 2 次调用）；严格同步模式使用 `feature_mode="aligned"`，每个 action step 返回对应的 `H_s/G_s`，并在 metadata 中记录 `t_M=phi(s)`。两种模式的 checkpoint、缓存格式和指标应分开记录。

### P0：针对当前 Psi0 action header 的接入位置

对照当前 `/gaozt-test1/guanzerong/Psi0/src/psi/models/psi0.py` 的结构，action expert 每个 scheduler step 都更新 `action_samples` 和 `timestep`，再由 `ActionTransformerModel` 构造 observation tokens 与 action tokens 做 joint attention。推荐按下面的边界接入：

```text
MotionExpert H [B,T,768] --P_H--> motion_history_tokens [B,T,d_action]
MotionExpert G [B,768]   --P_G--> motion_global [B,1,d_action]
                                      |
                                      +--> observation context / AdaLN condition
VLM、proprio、其他条件 -----------------+
```

1. `P_H` 和 `P_G` 是两个独立的可训练投影/归一化模块；当前 Psi0 action hidden width 为 `d_action`（配置默认 1536），不能直接把 768 维 H/G 拼到 action token 上。
2. H 应作为带 `frame_mask` 的时序 observation context，供 action token 的 cross/joint attention 使用；G 应作为窗口级条件进入 AdaLN/FiLM 或一个明确的 global context token。两路都必须实际参与 forward，不能在实现中只保留其中一路。
3. static 模式应在 `predict_action` 的 scheduler loop 外计算并缓存 H/G；aligned 模式才在 loop 内按 `t_M=phi(s)` 更新。不要把 MotionExpert 的重复调用藏在 action header 内，导致无法统计延迟和梯度。
4. action timestep embedding `temb_A(s)` 仍由 action expert 自己生成；G 的投影和门控独立于 `temb_A`，避免把窗口语义与噪声时间混成同一向量。
5. 如果 action expert 同时有 VLM token、proprio 和 2D trajectory，先把 H/G 作为新增 observation source 做 C0/C1 对照；不要先替换原有条件，否则无法判断 MotionExpert 的增益。

### P0：建立 video-only task conditioning

涉及：`utils/task_conditioning.py`。

建议增加显式任务名或等价 task id：

```text
video_history
video_forecast
single_frame_gen (optional)
```

每个任务显式产生：

- `img_mask`；
- `traj_mask`；
- `valid_img_embs`；
- `valid_frames`；
- `loss_mask`；
- `text_valid`。

不要继续让独立随机 mask 隐式决定任务。对于 history feature，未来位置应是无效 padding 或不存在，而不是“有效但图像缺失的未来 token”。对于 forecast，未来位置有效但未来图像条件无效；但该样本不能直接把包含未来 slot 的 global/H 作为纯历史 action condition 导出。

### P0：数据集按历史长度采样

涉及：`dataset/ee4d_motion_dataset.py`、DataLoader/collate。

1. 以秒为单位采样历史跨度。
2. 转换为 10 FPS motion slot，并正确映射 5 FPS DINO 特征。
3. 在 `n` 个历史 slot 内保留时间顺序。
4. 训练中随机选择 `n`，默认向 20 frame 倾斜。
5. 让 `n+H <= 80`，尾部 padding 只由 `valid_frames` 屏蔽。
6. trajectory/text 生成显式 null 条件，而不是把真实条件留在输入后再依赖随机 mask。

### P0：让 loss mask 真正参与任务训练

涉及：`module/uem_module.py`、`mydiffusion/flow_matching.py`。

1. 对 history/forecast 使用 `loss_mask`。
2. 分别归一化 history 和 future loss。
3. forecast 使用 `future=1.0`、`history=0.25--0.5` 的初始权重。
4. 保持当前 motion feature validity mask 和 153 维 feature weighting。
5. 记录 `history_mse`、`future_mse`、local/global motion MSE，避免只看混合 loss。

### P1：支持 video-only 输入接口

涉及：`motion_expert.py`、`config.py`、`checkpoints.py`、推理 CLI。

1. 允许缺少真实 trajectory 时使用显式 null trajectory/mask。
2. 允许 text 缺失并设置 `text_valid=False`。
3. 不改变旧多模态 checkpoint 的严格加载规则。
4. video-only checkpoint 和原多模态 checkpoint 使用不同架构/配置标识。
5. 在 metadata 中记录实际模态，防止下游误把 video-only feature 当成 trajectory-conditioned feature。
6. 明确 `feature_mode=static/aligned`，并拒绝把 static `t_M=1` 特征标记为 action-step synchronized。

### P1：增加 one-step feature 测试

涉及：`tests/test_global.py`、`tests/test_pipeline.py`。

必须覆盖：

- `feature_step=0` 与现有一次前向结果逐位一致；
- `feature_step=1` 恰好使用两次模型前向，且第二次时间为 `1 - 1/num_steps`；
- history feature 不受未来 image/traj 值变化影响；
- global 不读取无效 future/padding token；
- 相同 seed、条件、dtype、权重时结果可复现；
- 缺少 trajectory/text 时 null mask 行为正确；
- `one_step_global` 是 `[B,768]`，`one_step_hidden` 是 `[B,T,768]`；
- H/G 必须来自同一次 MotionExpert 前向，并在 action adapter 中成对传递；
- static 模式在所有 action timestep 复用同一 H/G；aligned 模式的 `t_M` 与 action scheduler 映射正确；
- `feature_mask` 不与 per-dimension `feature_valid_mask` 混淆。

### P2：可选 feature stability 模块

涉及：训练 loss、配置和评估脚本。

只有 baseline 明确需要时才增加：

- paired-noise consistency；
- teacher hidden distillation；
- global auxiliary prediction；
- K-seed feature aggregation。

## 11. 实验矩阵

建议先进行小规模、可重复的消融，再启动完整 400M 训练。

### 11.1 必做基线

| 编号 | 条件 | hidden 读取 | 目的 |
| --- | --- | --- | --- |
| A0 | 当前多模态条件 | t=1 | 原始能力上限/参考 |
| A1 | video-only | t=1 | 现有 checkpoint 的 modality gap |
| A2 | video-only | 一步后 t=0.9 | literal one-step 对照 |
| A3 | video-only | t=0.8 | 少量额外计算的对照 |
| A4 | video-only fine-tune | t=1 | 推荐第一版 |
| A5 | video-only fine-tune | t=0.9 | feature timestep 对照 |

### 11.2 历史长度

对 A1/A4 至少测试：

```text
n = 5, 10, 20, 40 motion frames
```

记录：

- 单次 feature latency；
- global 下游准确率或 loss；
- per-frame hidden 下游准确率或 loss；
- seed variance；
- 历史长度变化时的性能曲线。

### 11.3 下游/模型指标

不要只用 Flow Matching motion loss 选模型。至少报告：

1. 下游任务性能：分类、检索、预测或控制指标。
2. hidden/global 的线性 probe 和小型 adapter 性能。
3. 同条件跨 seed cosine similarity 和 variance。
4. feature norm、batch std、不同来源数据的分布差异。
5. 如果保留 motion 输出：MPJPE、PA-MPJPE、head/root absolute error、velocity/acceleration、foot contact、foot sliding、semantic similarity 和 FID。
6. K 个随机样本的多样性和校准，不要只用单样本 MPJPE。

### 11.4 数据泄漏检查

- forecast 任务不得读取未来 DINO、未来 trajectory 或未来 text；
- history global 不得读取未来有效 motion token；
- train/validation 使用序列级划分，避免重叠窗口泄漏；
- feature extraction 的 noise、timestep、mask 必须与下游训练一致；
- 不要在下游训练中使用 clean-motion hidden、部署时却使用 `t=1` noise hidden。

### 11.5 action expert 接入消融

这组实验比单独比较 motion MSE 更重要，因为 H/G 的最终用途是动作条件：

| 编号 | H/G 使用方式 | MotionExpert 调用次数 | 训练目的 |
| --- | --- | ---: | --- |
| C0 | 不提供 MotionExpert | 0 | action expert 原始基线 |
| C1 | H/G 同时提供，static `t_M=1` | 1 | 首选低延迟方案 |
| C2 | 只提供 H 或只提供 G | 1 | 验证两路信息是否互补（诊断，不作为最终接口） |
| C3 | H/G 同时提供，aligned `t_M=phi(s)` | 约 `N_action_steps` | 严格时间步同步的收益/成本 |
| C4 | H/G 在 2--3 个时间节点缓存/选择 | 2--3 | hybrid 成本折中 |

每个对照都要使用相同的 action expert、历史帧、随机 action noise 和训练预算，并报告 action loss、控制成功率、端到端延迟、显存以及跨 seed 稳定性。C2 只用于定位贡献，不能据此删掉用户要求的另一条输出。

## 12. 主要风险与应对

### 风险 1：视频无法观测绝对轨迹

表现：局部姿态合理，但 head/root absolute error 很大。

应对：增加 VO/IMU，或把下游目标改为相对 canonical motion；单纯增加 forecast 或 generation 比例无效。

### 风险 2：`t=1` hidden 的 seed 方差太大

表现：同一视频的 global token 跨 seed cosine similarity 低，下游结果不稳定。

应对顺序：固定 seed -> K-seed 平均/保留方差 -> feature_step=1/2 -> 小权重 consistency -> distillation。

### 风险 3：H/G 学成 action 数据集/噪声标识，而不是历史摘要

表现：motion loss 下降，但 action expert 对 H/G 的依赖只在训练数据成立；global linear probe 接近随机，或不同 task/seed 的 H/G 分布明显分裂。

应对：增加 history task 比例；只让 H/G 读取历史有效 token；冻结 action adapter 做独立 probe；最后才考虑 global auxiliary loss。

### 风险 4：把多步模型第一步当成单步生成器

表现：one-step `pred_xstart` 或 motion 质量差，不能替代完整采样。

应对：区分“one-step feature”与“one-step motion”；若要单步 motion，单独做 distillation 和 causal architecture 实验。

### 风险 5：5 帧上下文不足

表现：速度、动作阶段和场景意图无法区分。

应对：按秒采样并测试 0.5/1/2/4 秒；优先从 2 秒 baseline 开始。近期 online 工作使用 5-frame history 只是特定 causal 架构的设置，不能直接迁移为本项目结论。[EgoForce](https://arxiv.org/html/2605.13041)

### 风险 6：非因果注意力造成在线泄漏

表现：使用 `n+H` 窗口时，history hidden 对未来初始噪声敏感。

应对：history feature 使用 `T=n`；真正的在线 forecast 再单独引入 causal/rolling 设计。

### 风险 7：把 static 特征误认为同步特征

表现：action expert 训练时看到的是随 `s` 变化的 MotionExpert H/G，部署时却只缓存一份 `t_M=1` 特征，或反过来；action loss 在离线评估正常，真实 denoising 时明显退化。

应对：把 `feature_mode` 写入数据和 checkpoint metadata；static/aligned 分开训练、缓存和评估，不能只改一个推理开关。

### 风险 8：同步模式的计算和梯度成本不可接受

表现：每个 action step 都重新运行 MotionExpert，端到端延迟约按步数增长，联合 unroll 时 H/G 梯度过大或显存溢出。

应对：先用 static；需要时间变化时采用 2--3 个节点缓存的 hybrid；对完整 unroll 的 action loss 按步平均，并优先冻结 MotionExpert 或只解冻最后 K 层。

## 13. 推荐的第一版配置

在没有更多下游约束时，建议先固定这一组实验：

```text
目标                 : H=one_step_hidden [B,T,768] 与 G=one_step_global [B,768] 同时输入 action expert
默认 feature         : feature_mode=static；低延迟 baseline 为 feature_step=0，literal denoised 候选为 feature_step=1
同步对照             : feature_mode=aligned，按 t_M=phi(s) 每个 action step 更新（高成本）
训练主任务            : video_history + downstream action loss
训练辅助任务          : video_forecast（仅动态先验，按需启用）
任务比例              : video_history 80%--100%，video_forecast 0%--20%，generation 0%
trajectory            : 主分支全部 null/masked
text                  : 主分支全部 invalid
history               : 5/10/20/40 motion frame 随机，20 为最高概率
默认 history          : 20 motion frame = 2 秒
默认 future           : 20 motion frame = 2 秒（仅 forecast 辅助样本）
loss                  : L_action 为主；L_motion 小权重，先不启用 L_cons
Flow Matching         : 保留现有 x0、连续 t、10-step Euler；显式记录两套 scheduler 映射
CFG                   : 默认关闭
noise                 : 保留 Gaussian noise；先固定 seed，再测试多 seed/K-seed
融合                 : H -> cross-attention/KV，G -> AdaLN/FiLM 或独立 global token
训练方式              : standalone video-only -> 冻结专家训 adapter/action -> 小学习率联合微调
```

如果下游只需要历史条件，forecast 可以直接设为 0%，所有导出 H/G 都使用 `T=n`。只有当 action expert 的 action loss 对动态先验仍明显不足时，才增加 forecast，并把它作为独立消融；若把 forecast 提升为主任务，应另存 future-aware checkpoint，不能再把它当成纯历史 action condition。

## 14. 参考资料

以下均为原始论文或作者提供的论文页面：

1. [UniEgoMotion: A Unified Model for Egocentric Motion Reconstruction, Forecasting, and Generation](https://arxiv.org/html/2508.01126)
2. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
3. [Diffusion Model as Representation Learner](https://arxiv.org/abs/2308.10916)
4. [Exploring Diffusion Time-steps for Unsupervised Representation Learning](https://arxiv.org/abs/2401.11430)
5. [Diffusion Forcing: Next-token Prediction Meets Full-Sequence Diffusion](https://arxiv.org/abs/2407.01392)
6. [REWIND: Real-Time Egocentric Whole-Body Motion Diffusion](https://arxiv.org/abs/2504.04956)
7. [EgoForce: Robust Online Egocentric Motion Reconstruction via Diffusion Forcing](https://arxiv.org/abs/2605.13041)

## 15. 最终决策

当前最稳妥的路线不是立刻重写 diffusion 主干，而是先把“one-step hidden 的定义”和“video-only 的训练分布”固定下来：

1. 默认采用 `feature_mode=static` 并提取成对 H/G；先以 `feature_step=0` 建立低延迟 baseline，同时以 `feature_step=1` 作为 literal denoised 主候选，二者 metadata 分开。
2. 以 video-history reconstruction/encoding 为主，forecast 仅作 0%--20% 动态先验辅助，generation 关闭。
3. 只提供历史图像条件，trajectory/text 全部显式 null。
4. 用 `T=n` 保证 H/G 只摘要历史；forecast 辅助分支不得把 future-aware H/G 导出给纯历史 action expert。
5. 先冻结 MotionExpert 训练 H/G adapter 与 action expert，再按 action loss 小学习率解冻最后若干层/global 路径。
6. 对 history length、feature step、static/aligned 模式、scheduler 映射和 seed variance 做系统消融。
7. 只有 baseline 证明必要时，才加入 endpoint oversampling、consistency 或 distillation。

这样可以把三个不同问题拆开：

```text
表示是否有用       -> action loss / 下游 probe / adapter
表示是否稳定       -> 跨 seed 与跨 timestep
motion 是否准确     -> motion metrics
严格 online 是否可行 -> causal architecture / rolling denoising
```

它们不能用一个 `recon/forecast/generation` 混合 loss 或一次 `t=1` 调用同时证明。
