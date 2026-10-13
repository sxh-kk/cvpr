# MotionExpert

本公开快照不包含 checkpoint 或模型权重。环境、数据与外部权重的迁移说明见
[迁移文档](../docs/MIGRATION.md)；`assets/` 内的统计和表示资源继续保留。

从 UniEgoMotion 独立迁移的 **E7-K12-400M 文本条件模型**，完整包含训练、验证、
EMA 断点续训、权重导出和推理。当前正式训练版本在主干加入一个 `<GLOBAL>` token，
序列为 `[time, global, motion...]`，输出全部运动 hidden 和一个全局 hidden。
结构与训练依据见 [project.md](project.md)。

模型为 12 层、768 hidden，每层 1 个 shared expert 和 11 个 routed experts，按 4 帧
chunk 做 Top-2 路由；两个前缀使用有效运动帧的序列级路由，不占用帧 chunk。
global 版本共 403,790,133 参数，保留 CLIP cross-attention、原 x0 Flow Matching
和 MoE 均衡损失，不增加 global 辅助损失。全局运动 9 维损失权重仍为 8，推理用 10 步 Euler。

旧 E7 权重仍可严格加载，参数量为 403,789,365，没有 global 输出。原项目配套的旧 EMA 权重
不包含新增 token；global 版本需要使用本项目训练产生的检查点或导出权重。

已移除其他实验的双输出头、TaskFiLM、自适应任务采样、几何辅助损失、两阶段模型、
DDPM 以及路由消融入口。本目录运行时不依赖 `../UniEgoMotion`。

## 安装

在本目录执行，Python >= 3.10，GPU 训练需要支持 CUDA 的 PyTorch：

```bash
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .
```

已有 `.venv` 时直接激活即可。命令行入口也可写成 `python -m MotionExpert.train`、
`python -m MotionExpert.infer` 等；安装后可从任意工作目录调用。
原来的 `python -m train` 不适用于包内相对导入。

## 输入数据

默认数据根目录是 `../datasets/ee4d_motion_uniegomotion`，可通过 `--data-dir` 或
`MOTIONEXPERT_DATA_DIR` 修改。数据路径相对于数据根目录，模型/输出配置路径相对于
本包目录；命令行传入的路径相对于当前工作目录。绝对路径保持原样。

每个 split（`train`、`val`）需要以下文件，具体路径均可在 YAML 中配置：

| 文件 | 内容 |
| --- | --- |
| `uniegomotion/ee_{split}.pt` | EE4D motion pack，含 SMPL 参数、Aria 轨迹和 3D 关节 |
| `uniegomotion/nymeria_{split}.pt` | NymeriaPlus motion pack，含有效特征 mask 与图像对齐索引 |
| `uniegomotion/egoview_dinov2_{split}.pt` | EE4D 5 FPS DINOv2 特征 |
| `uniegomotion/egoview_dinov2_nymeria_{split}.pt` | NymeriaPlus 5 FPS DINOv2 特征 |
| `uniegomotion/clip_text_{split}_16.pt` | 按序列和窗口起点索引的 16-token CLIP 特征 |
| `uniegomotion/motion_qc_{split}.json` | 通过质量检查的序列名单 |
| `uniegomotion/v4_beta_joint_masked_train_stats_clean.pt` | 训练集联合归一化统计，验证集也使用它 |

沿用原工程的预处理格式：运动 10 FPS、窗口 80 帧、步长 20 帧；DINO sidecar
保存 `[N,5,1024]`，模型消费其中的 CLS token，并对齐到运动帧。
NymeriaPlus 中未观测的手部 PCA 和体型参数不计入损失。DINO 特征不再做额外标准化。
QC 清单可以设为 `null`，但默认的干净统计仍需与所使用的数据一致。

```bash
motionexpert-check-data --config configs/e7_k12_400m_global_text.yaml
```

该命令检查文件、所有选中窗口的文本索引和每个来源的代表样本，并报告文本覆盖率。
没有文字标注的窗口可以使用明确的 empty row；缺少窗口索引会报错。
加载大包需要时间，运动、DINO 和文本张量尽可能使用 mmap。

## 文本和统计

已有 DINO 与 CLIP sidecar 时可直接训练。如果只有文本标注，先生成 CLIP sidecar：

```bash
motionexpert-text \
  --ee-pack /data/uniegomotion/ee_train.pt \
  --nymeria-pack /data/uniegomotion/nymeria_train.pt \
  --ee-text /data/uniegomotion/ee_text_train.pt \
  --takes /data/takes.json \
  --output /data/uniegomotion/clip_text_train_16.pt --device cuda:0
```

验证集同样生成一次。默认编码器为 `openai/clip-vit-large-patch14`，首次使用需下载，
之后可用缓存或 `--model /local/clip`。`--ee-text`、`--takes` 可省略，省略时使用
motion pack 内已有的文本标注。训练本身读取离线 CLIP 特征，不运行文本编码器。

更换数据后，可以用本包内的 `compute_joint_motion_stats`、`build_motion_qc_manifest`
和 `merge_joint_motion_stats` 重建联合统计，具体参数见各模块 `--help`：

```bash
python -m MotionExpert.compute_joint_motion_stats \
  --data-dir /data --ee-pack /data/uniegomotion/ee_train.pt \
  --nymeria-pack /data/uniegomotion/nymeria_train.pt --output /data/pass1.pt
python -m MotionExpert.build_motion_qc_manifest \
  --stats /data/pass1.pt --output /data/uniegomotion/motion_qc_train.json
python -m MotionExpert.compute_joint_motion_stats \
  --data-dir /data --ee-pack /data/uniegomotion/ee_train.pt \
  --nymeria-pack /data/uniegomotion/nymeria_train.pt \
  --sequence-manifest /data/uniegomotion/motion_qc_train.json \
  --max-nonfinite-values 0 --output /data/uniegomotion/v4_beta_joint_masked_train_stats_clean.pt
```

验证集生成自己的 QC 清单（统计脚本加 `--split val`），训练/推理归一化仍使用训练统计。

## 训练与恢复

```bash
# 全部有效参数从随机初始化训练，使用全部通过 QC 的训练窗口
motionexpert-train --config configs/e7_k12_400m_global_text.yaml \
  --from-scratch --devices 8 --ddp-socket

# 运行一次真实数据训练/验证，检查通路，不保存检查点
motionexpert-train --from-scratch --devices 1 --batch-size 2 --num-workers 0 \
  --accumulate-grad-batches 1 --fast-dev-run

# 完整恢复模型、AdamW、LR scheduler、EMA 和训练进度
motionexpert-train --resume exp/e7_k12_400m_global_text/last.ckpt --ddp-socket

# 继续使用旧 E7 架构；需另行准备配套权重
motionexpert-train --config configs/e7_k12_400m_text.yaml --devices 8 --ddp-socket
```

`--from-scratch` 使用随机初始化；`--init-weights path.pt` 只加载权重并重新训练。
省略 `--config` 时使用 global 正式配置，默认不加载初始化权重。恢复默认使用检查点中的
配置，可显式提供 `--config` 调整设置。初始化及恢复均严格检查架构，不能将旧 E7
检查点直接恢复为 global 版本。默认按 YAML 周期保存检查点；正常训练结束时始终
保存 `final.ckpt` 并更新 `last.ckpt` 链接，即使未到周期也保留最后一步。

支持 `--max-epochs`、`--max-steps`、`--output-dir`、`--limit-train-batches`、
`--limit-val-batches` 和 `--devices 0`（CPU）。默认 8 卡、每卡 batch 16、累积 4 步；
global 配置已将 `DATA.SOURCE_SUBSAMPLE.ENABLED` 设为 `false`，学习率为 `1e-4`，
所有有效主干参数从 epoch 0 更新，global 参数也由原 motion loss 训练。
验证会分别记录 EE4D/NymeriaPlus 的运动指标，并额外记录 `t=1` hidden/global 的
范数和 global 的 batch 内标准差；这些诊断不参与训练损失，也不改变训练时间采样。

本机默认 NCCL 通信初始化出现过停滞，双卡使用 `--ddp-socket` 已验证通过。
该选项只适用于单机，禁用 IB/P2P/SHM 并使用 loopback socket；通信环境正常的机器
可省略此选项，以使用更快的默认 GPU 通信。

## 推理与导出

```bash
# 使用新检查点和验证集第 0 个窗口的文本，执行完整 10 步采样
motionexpert-infer --config configs/e7_k12_400m_global_text.yaml \
  --weights exp/e7_k12_400m_global_text/last.ckpt \
  --split val --index 0 --task recon --output exp/prediction.pt

# 指定序列和原始文字，也支持 fore / gen
motionexpert-infer --config configs/e7_k12_400m_global_text.yaml \
  --weights exp/e7_k12_400m_global_text/last.ckpt \
  --sequence SEQUENCE_NAME --start 0 --text "A person walks forward." \
  --task fore --forecast-prefix 20 --output exp/forecast.pt

# 导出 EMA 权重与配套归一化统计
motionexpert-export exp/e7_k12_400m_global_text/last.ckpt weights/e7_k12_400m_global_text.pt

# 直接读取 .ckpt（默认 EMA）或导出的 .pt
motionexpert-infer --input conditioning.pt --weights weights/e7_k12_400m_global_text.pt \
  --output exp/prediction.pt

# 仅提取给下游的特征：无 CFG 时只执行一次模型前向，不运行 Euler
motionexpert-infer --input conditioning.pt --weights weights/e7_k12_400m_global_text.pt \
  --features-only --output exp/features.pt
```

`conditioning.pt` 是包含下列批量张量的字典（也可放在 `y` 键下）：

```python
from MotionExpert import MotionExpert

expert = MotionExpert(weights="weights/e7_k12_400m_global_text.pt", device="cuda:0", dtype="bfloat16")
conditioning = {
    "traj": traj,                 # [B,T,18]，canonical v4_beta 归一化轨迹
    "img_embs": dino,             # [B,T,1024]，对齐后的 DINOv2 CLS 特征
    "valid_frames": valid,        # [B,T]，可省略
    "text_pooled": pooled,        # [B,768]
    "text_tokens": tokens,        # [B,16,768]
    "text_mask": text_mask,       # [B,16]，可省略
}
out = expert.sample(conditioning, seed=62, task="recon", denormalize=True)
features = expert.extract_features(conditioning, seed=62, task="recon")
hidden = features["one_step_hidden"]  # [B,T,768]
global_hidden = features["one_step_global"]  # [B,768]
```

也可省略 CLIP 张量，传入 `text="A person walks forward."` 或长度为 B 的文本列表。
空字符串表示无文本条件；`local_files_only=True` 禁止自动下载编码器。
`valid_img_embs`、`text_valid`、`traj_mask`、`img_mask` 均支持显式传入。
`recon` 使用全部条件，`fore` 使用前缀，`gen` 只保留首帧图像和文本条件。

`traj` 已是 18 维 canonical 表示但未归一化时可用 `normalized=False`，CLI 对应
`--unnormalized-traj`。原始世界坐标轨迹不能直接当作此输入；数据集入口会完成
canonicalization。默认统计随包提供，训练检查点和导出文件会携带对应的统计。

输出 `motion` 为归一化的 `[B,T,153]` K12 表示，`one_step_hidden` 为首次模型调用
的 `[B,T,768]` hidden，global 权重还返回同次调用的 `one_step_global [B,768]`。
两者都在 `t=1` 从最后一层提取，不包含 time 前缀；global 不参与 Euler 积分。
`feature_mask [B,T]` 标记有效输出位置，`metadata` 记录任务、种子、架构、特征时刻和 CFG。
`extract_features` 不返回 `motion`，其特征 Tensor 可以用于训练冻结专家之后的下游模块。
如需同时训练专家，直接使用可微分的 `expert.model(..., return_features=True)`，返回
`(motion_prediction, hidden, global_hidden)`；该选项与 `return_hidden=True` 互斥。

默认 `cond_scale=None`，DINO/文本/轨迹条件仍然有效。启用 CFG 会计算条件与无条件
两个分支，并对 motion、hidden 和 global 使用相同的线性组合。

`denormalize=True` 额外返回 `motion_denormalized`；完整采样的 CLI 默认
包含它，`--all-steps` 另存 10 步的 x0 估计。K12 顺序是
`[0,4,5,10,11,13,14,15,18,19,20,21]`，每关节 9 维，另有 45 维辅助量。
反归一化后的结果仍是 canonical 稀疏运动表示，不是全身 SMPL-X 网格。
T 默认为 80，推理支持 1 至 5000 帧；超出训练窗口的质量未做保证。

导出的 `.pt` 包含 `state_dict`、架构元数据和归一化统计；同时保留 `.stats.pt` 与
`.json` sidecar。包装器自动识别新导出格式和旧裸 state_dict。不要将 global 模型仅保存为
缺少架构元数据的裸 state_dict 后直接用于包装器，使用 `motionexpert-export` 完成导出。

## 验证

```bash
python -m unittest tests.test_pipeline tests.test_global -v
```

回归测试覆盖两种来源的窗口读取、DINO 对齐、文本索引、无效特征损失屏蔽和归一化。
global 版本通过 18 项回归测试、真实 EE4D/NymeriaPlus 单卡 BF16 训练/验证、双卡
socket DDP、短流程 EMA 续训/导出，以及完整采样和单次特征提取。两种特征接口在
相同 seed 下的首次 hidden/global 逐位一致；导出的 EMA 权重也已逐位复核。
这些是通路验证，不代表已完成 300 epoch 训练。旧 E7 权重的 motion 和 hidden 在兼容
路径下仍保持固定输入逐位一致。
原项目配套权重的路径为 `weights/e7_k12_400m_text_ema_fp32.pt`；本公开快照不上传
这个 1.6 GB 权重。迁移时另行准备 `weights/` 或通过 `--weights` 指定外部路径。
