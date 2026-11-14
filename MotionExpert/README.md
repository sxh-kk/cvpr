# MotionExpert

MotionExpert 从历史图像学习人体运动先验，联合重建历史运动并预测未来运动。数据来源、权重、训练参数和训练／评估命令见[项目首页](../README.md)。本页说明输入协议、缓存准备和模型接口。

默认配置为 [`configs/smplh_k12_psi0_pretrain.yaml`](configs/smplh_k12_psi0_pretrain.yaml)，路径可直接在该文件中调整。

## 时间、图像与条件

模型以 5Hz 处理 11 个时刻：`[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]` 秒。前 6 帧为历史 5 帧＋当前帧，后 5 帧为未来。

- 原始运动与离线 DINO 特征保留 30Hz；读取时按零相位、步长 6 取样。
- 视觉条件为 `[B,6,5,1024]`，每帧包含 CLS＋4 个 register；最后一帧与 VLM 当前帧同源、同时间。
- 训练联合监督 11 帧运动，不输入干净历史人体姿态。轨迹模态 dropout 以 0.5 的概率开放历史 6 帧的有效 18 维相机轨迹。
- 验证、测试和 `forward_features()` 隐藏真实轨迹；未来图像和轨迹始终隐藏。身体／手部真值及有效 mask 只参与监督。
- 在线调用须积累约 1 秒真实历史，不复制单帧补齐 6 个时刻。

| 使用方式 | 最终 RGB 高×宽 | 图像策略 |
| --- | --- | --- |
| Motion 预训练 | 256×480 | 确定性几何 |
| UnifoLM 联合训练 | 256×320 | `vlm_color_clean_motion`：VLM 当前帧颜色增强，Motion 使用 clean 历史 DINO |
| 通用在线后训练 | 256×320 | `shared_rgb`：窗口共享颜色增强 |
| 通用 Robust 训练 | 256×480 | `shared_rgb`：窗口共享颜色与随机裁切 |

两路使用相同来源、时间和几何 profile，再分别执行编码器归一化。增强 Motion 图像时，必须在线编码同步增强后的整段历史。详细接口见 [ImageProcess](../ImageProcess/README.md)。

## 运动表示

表示版本为 `smplh12_rot6d_hands21_contacts_v1`，身体轴为 `smplh_body22_axes_v1`。读取器对共同 SMPL-H 模型做 FK，再选择 K12：`[0,4,5,10,11,13,14,15,18,19,20,21]`，依次对应 pelvis、左右 knee、左右 foot、左右 collar、head、左右 elbow、左右 wrist。双手使用左右各 21 点显式 XYZ 监督。

| 通道切片 | 内容 | 维数 |
| --- | --- | ---: |
| `[0:108]` | 身体 12 关节，逐关节旋转 6D＋XYZ | 108 |
| `[108:117]` | 相机地面参考系旋转 6D＋平移 3D 增量 | 9 |
| `[117:180]` | 左手 21 点 XYZ | 63 |
| `[180:243]` | 右手 21 点 XYZ | 63 |
| `[243:245]` | 双脚接触代理 | 2 |
| 总计 | 运动输入／输出 | **245** |

另有 18 维相机轨迹：相机相对地面参考位姿 9 维＋参考运动增量 9 维。重复增量共享统计。6D 旋转取旋转矩阵前两行；世界坐标单位为米，Z-up。窗口首帧相机水平位置与朝向规范化，逐帧使用相机地面参考系，相邻增量在上一帧参考系表达。

地面优先使用归档中的有效值，缺失时由历史 6 帧有效原始脚点平均高度减 2cm 得到。接触代理在 5Hz 上按速度 0.2m/s、高度 0.1m 计算。模型不预测完整 SMPL-H pose 或 shape。

## 数据与缓存

输入必须是包含模型资源、分割、来源记录和质量元数据的 SMPL-H 归档，格式见 [pipeline](../pipeline/README.md#数据格式)。原始 EgoSuite／Ego-Exo4D-HM 下载器不生成该归档；初始原始数据到 SMPL-H 归档的转换流程尚未包含在仓库中。

默认数据根目录为 `~/sigma0/data/MixEgoMotion`：

| 产物 | 相对路径 |
| --- | --- |
| 运动归档 | `smplh-parameters-v4-camera-quality` |
| 数据分割 | `smplh-parameters-v4-camera-quality/meta/splits.json` |
| DINO 特征 | `smplh-parameters-v4-camera-quality/vision_profiles/psi0_256x480_v1_30hz` |
| 窗口缓存 | `training_cache/smplh245_5hz_w11_camera-quality-v4` |
| 训练集统计 | `statistics/smplh_k12_5hz_joint_camera-quality-v4.pt` |

相机点质量和相邻边连续性分开保存：`observation.valid.camera_tracking` 表示原生点质量，`observation.valid.camera_transition[i]` 表示 `i-1→i` 连续性。5Hz 参考增量必须检查间隔内全部 30Hz 点和边，不能跨越跟踪中断。首帧参考失效的窗口会被拒绝；没有有效未来身体监督的窗口不进入训练索引。

Ego-Exo4D-HM 以原始 Aria 质量、时间和跟踪连续性为依据，open-loop 信息不替换 closed-loop 世界位姿。EgoSuite 未提供相机 confidence 时不伪造置信度。相机失败不直接删除独立 world 手／脚观测或 DINO 特征。

### 窗口缓存与统计

预训练使用只读 mmap 缓存，保存 `[11,245]` 运动、`[11,18]` 轨迹、有效 mask 和解码变换。缓存按 episode 构建、按 split 打包；噪声和条件 dropout 在训练时生成，DINO 按需读取。NPY 文件头使用精确长度读取，避免并行文件系统的大块缓冲预读；已有缓存和统计可直接复用。

在已有合规归档和完整 DINO 特征后，依次构建缓存、检查原始窗口、计算完整训练集统计、检查归一化窗口与 GPU 一致性：

```bash
CONFIG=MotionExpert/configs/smplh_k12_psi0_pretrain.yaml
DATA_ROOT="$HOME/sigma0/data/MixEgoMotion"
ARCHIVE_ROOT="$DATA_ROOT/smplh-parameters-v4-camera-quality"
CACHE_ROOT="$DATA_ROOT/training_cache/smplh245_5hz_w11_camera-quality-v4"
STATS_PATH="$DATA_ROOT/statistics/smplh_k12_5hz_joint_camera-quality-v4.pt"
REPORT_ROOT="$DATA_ROOT/.pipeline/window-cache"
mkdir -p "$REPORT_ROOT"

python -m MotionExpert.build_window_cache \
  --config "$CONFIG" --output-root "$CACHE_ROOT" --workers 8

python -m MotionExpert.validate_window_cache \
  --config "$CONFIG" --source-config "$CONFIG" \
  --raw-only --report "$REPORT_ROOT/raw.json"

python -m MotionExpert.compute_cached_motion_stats \
  --config "$CONFIG" --archive-root "$ARCHIVE_ROOT" \
  --cache-root "$CACHE_ROOT" --acceptance "$REPORT_ROOT/raw.json" \
  --output "$STATS_PATH"

python -m MotionExpert.validate_window_cache \
  --config "$CONFIG" --source-config "$CONFIG" \
  --gpu-check --device cuda:0 --report "$REPORT_ROOT/normalized-gpu.json"
```

统计脚本拒绝覆盖已有文件；已有匹配统计时无需重新计算。缓存绑定归档、分割、表示和生成代码，统计绑定产生它的缓存及检查记录；不能混用不同缓存或质量 mask 的统计。

训练还会读取归档 `.pipeline/dataset-state.json`，核对训练就绪状态、统计 SHA、缓存 manifest、模型架构和当前训练代码对应的 GPU 检查记录。上述缓存检查不会发布训练就绪元数据；仓库尚未提供独立的训练就绪发布命令。完整输入要求见 [readiness.py](readiness.py)，不能通过手动改状态或复制旧检查记录替代。

### 可选本机 DINO 读取缓存

并行文件系统的小文件等待较高时，可将已验收窗口所需的零相位 5 Hz DINO 特征打包到本机存储。构建器顺序读取原生帧，在内存中选帧并逐块核对副本的字节，全部完成后发布 manifest；训练按原历史帧索引读取，继续使用原统计、有效 mask 和 split。

```bash
LOCAL_VISION_CACHE=/dev/shm/motionexpert-vision5hz
python -m MotionExpert.dataset.packed_vision_cache \
  --vision-root "$ARCHIVE_ROOT/vision_profiles/psi0_256x480_v1_30hz" \
  --records-path "$CACHE_ROOT/records.json" \
  --output-root "$LOCAL_VISION_CACHE" \
  --report "$REPORT_ROOT/packed-vision-read-cache.json" --workers 8 --block-rows 4096
export MOTIONEXPERT_VISION_READ_CACHE_ROOT="$LOCAL_VISION_CACHE"
```

预先确认本机存储和内存余量；构建器保留 128 GiB 可用内存，拒绝覆盖已有输出。`/dev/shm` 中的缓存随实例退出消失；可重新构建，或取消该环境变量以使用原始读取路径。启用时，缓存缺失、未完成或来源／文件身份不符会拒绝加载。已有源文件校验保留在启动阶段，worker 只映射本机副本。

## 模型与归一化

架构版本为 `e7_smplh12_hands21_meanpool_qkvnorm_v2`。模型不使用可学习 GLOBAL token；flow-time prefix 不参与 pooling，全部 11 个 motion token 参与均值。Self-attention 对 Q/K/V 共用归一化后的 motion；cross-attention 的 Q 与图像／时间 context 分别归一化。

训练直接预测速度场：`x_t = (1-t) x_0 + t ε`，目标为 `v = ε - x_0`。损失仍使用原有有效帧／通道 mask、Global 权重和时间采样分布；完整运动预测仍以十步 Euler 从 `t=1` 积分至 `t=0`。诊断中的干净动作估计为 `x_t - t v_pred`。

归一化使用完整训练集统计，并与 checkpoint 中的表示、分割和数据身份保持一致。不同架构的 checkpoint 不能直接恢复或初始化到本模型。旧 x₀ 预测 checkpoint 也不能当作速度预测模型恢复或推理；请保留旧权重并开始新的速度训练，不能只改 checkpoint 元数据。

训练在反向传播前检查非有限预测、loss 和激活，在优化器更新前检查 unscale 后、clip 前的梯度；DDP 各卡一致停止。异常 batch、mask、来源与 RNG 写入训练目录的 `anomalies/`。保护阈值由默认配置中的 `TRAIN.SAFETY_*` 控制，关闭幅值保护仍拒绝 NaN／Inf。

## 一步hidden与在线共享图像

`forward_features()` 按 [LaMP 的单步特征提取流程](https://arxiv.org/html/2603.25399v1#S3.SS4)，先在完整高斯输入 `t=1` 上预测速度，更新 `x_next = ε - 0.1 v_pred`，再在 `t=0.9` 前向并取第二次 hidden。LaMP 使用相反的时间方向，其 `τ=0→0.1` 与这里的 `t=1→0.9` 等价。这是一次状态更新、两次前向，不执行完整流采样：

| 返回值 | 形状 | 含义 |
| --- | --- | --- |
| `one_step_hidden` | `[B,11,768]` | 一次 Euler 更新后，第二次前向的全部 11 个运动时刻的最后一层 hidden |
| `one_step_global` | `[B,768]` | 严格等于 `one_step_hidden.mean(dim=1)` |

调用方提供 `img_embs [B,6,5,1024]`。该接口不输入干净人体目标或未来标签 mask；缺失标签不能将高斯噪声通道清零。

```python
from pathlib import Path
import torch
from MotionExpert import MotionExpert

weights = Path.home() / "sigma0/runs/motion/smplh_k12_psi0_256x480_5hz_camera_quality_v4_meanpool_from_scratch/final.ckpt"
expert = MotionExpert(weights=weights, device="cuda", dtype=torch.float32,
                      use_ema=False).train()
features = expert.forward_features({"img_embs": img_embs}, seed=0)
prior = features["one_step_hidden"]  # [B,11,768]
```

`seed` 固定初始噪声；训练模式下 dropout 仍由调用方 RNG 决定。保留调用方的 train／eval 模式和现有联合训练 autograd：下游损失经过第二次 hidden、状态更新和第一次速度输出头反传；冻结接口则在两次前向上均使用 `no_grad`。外部 `nn.Module`／DDP 应将 `expert.model` 注册为子模块；MoE 的未使用专家仍需由 DDP 处理，例如设置 `find_unused_parameters=True`。

- 冻结特征：使用 `extract_features()`。
- 完整预测：使用 `predict_future()`，返回完整 `motion [B,11,245]`、未来 `future_motion [B,5,245]`，以及默认启用的对应反归一化结果。
- 在线图像编码：使用 `DinoEncoder.encode_window()`；通用 `shared_rgb` 模式下，将返回的 `rgb[-1]` 同时送入 VLM。UnifoLM 的独立 VLM 颜色策略见 [VLM Expert](../VLMExpert/README.md#inputs-and-image-alignment)。
