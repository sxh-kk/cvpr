# DINO 特征提取

本模块从已准备的 SMPL-H 归档及其原始相机数据提取原生 30Hz DINO 特征，供 MotionExpert 预训练使用。数据与模型下载见[项目首页](../README.md)，默认提取配置为 [`configs/dino.json`](configs/dino.json)。

## 数据格式

输入必须是符合 [SMPL-H 读取器](../MotionExpert/dataset/smplh_archive.py)要求的归档，包含模型资源、分割、来源记录、质量元数据和发布状态。原始 EgoSuite／Ego-Exo4D-HM 下载器只下载来源数据；初始 SMPL-H 转换流程尚未包含在仓库中。

默认目录布局：

```text
~/sigma0/data/MixEgoMotion/
├── smplh-parameters-v4-camera-quality/
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.parquet
│   │   └── splits.json
│   ├── episodes/XXXXXXXX/
│   │   ├── motion.parquet
│   │   └── receipt.json
│   ├── weights/smplh/
│   ├── .pipeline/
│   └── vision_profiles/psi0_256x480_v1_30hz/
├── training_cache/smplh245_5hz_w11_camera-quality-v4/
└── statistics/smplh_k12_5hz_joint_camera-quality-v4.pt
```

`motion.parquet` 保存原生 30Hz 数据：

- `smplh.root_orient [3]`、`pose_body [63]`、`trans [3]`、`betas [16]`，以及来源可用时的手指 PCA。
- 左右各 21 点世界 XYZ、逐点有效 mask、相机／头部／头戴参考位姿及插值标记。
- 相机点质量与相邻边连续性，或归档支持的原生参考全有效证书。
- 地面高度、法向和有效性，原始双脚点及 mask。
- 身体估计与来源观测的位置／旋转 mask、confidence、完成及残差信息。
- 原生帧号、时间、视频帧号、PTS 误差和图像有效性。

共同模型定义 SMPL-H 身体 22 关节轴，世界坐标单位为米、Z-up；模型版本由[运动配置](../MotionExpert/configs/smplh_k12_psi0_pretrain.yaml)指定。MotionExpert 构建窗口时做 FK、选择 K12 并生成相机地面参考系和接触代理，详见[运动表示](../MotionExpert/README.md#运动表示)。

来源记录必须能够定位对应原始 MCAP／MP4、时间同步与相机资源。提取器按来源记录读取图像，不以运动参数渲染图像代替原始相机数据。

## 图像与特征协议

DINO 使用冻结的 `facebook/dinov2-with-registers-large`，编码器版本由[默认配置](configs/dino.json)统一指定。共享 RGB 几何由 [ImageProcess](../ImageProcess/README.md)提供，DINO 禁用自身二次 resize／crop，仅执行编码器的 rescale 和归一化。

| 阶段 | 最终 RGB 高×宽 | 策略 |
| --- | --- | --- |
| Motion 预训练 | 256×480 | 确定性几何，离线提取 |
| UnifoLM 联合训练 | 256×320 | clean 历史 DINO，VLM 当前帧独立颜色增强 |
| 通用在线后训练 | 256×320 | `shared_rgb`，窗口共享颜色增强 |
| 通用 Robust 训练 | 256×480 | `shared_rgb`，窗口共享颜色与随机裁切 |

离线特征保存全部 30Hz 帧，每帧输出 `[5,1024] float32`，顺序为 CLS、register0..3。MotionExpert 以步长 6 取 5Hz 窗口，只有历史 6 帧图像作为条件。原始相机几何沿用来源数据约定，共享处理不新增去畸变。

输出目录包含：

```text
vision_profiles/psi0_256x480_v1_30hz/
├── manifest.json
└── episodes/XXXXXXXX/
    ├── features.npy   # [T30,5,1024]，float32，支持 mmap
    ├── join.npz       # 帧号、时间、来源视频帧号、PTS 和 mask
    └── receipt.json   # 来源、encoder、profile 与输出身份
```

## 提取命令

以下路径与默认 Motion 配置一致：

```bash
ARCHIVE_ROOT="$HOME/sigma0/data/MixEgoMotion/smplh-parameters-v4-camera-quality"
VISION_ROOT="$ARCHIVE_ROOT/vision_profiles/psi0_256x480_v1_30hz"

python -m egomotion_pipeline.cli extract-smplh-features \
  --dataset "$ARCHIVE_ROOT" \
  --output "$VISION_ROOT" \
  --config pipeline/configs/dino.json \
  --gpus 0 --workers-per-gpu 1
```

重复执行同一命令可恢复未完成提取，只复用配置、来源和文件身份匹配的 episode。`--gpus` 指定可见 GPU 编号；`--workers-per-gpu` 控制每卡工作进程数。`--dry-run` 检查来源和提取计划，不加载 DINO 或创建输出。`--episodes` 用于独立的小规模检查目录，其部分缓存不能作为完整训练输入。

完整训练要求 manifest 标记为 `complete/all`，并绑定同一归档、分割和图像 profile。输出图像 mask 必须保持来源有效性，坏帧不能提升为有效。相机质量 mask 改变时须重新计算完整训练集统计；仅改变视觉尺寸不要求重算运动统计，但须显式声明视觉适配并保持 DINO 权重与归一化身份一致。

提取完成后的窗口缓存与统计步骤见 [MotionExpert](../MotionExpert/README.md#数据与缓存)。UnifoLM 使用独立的 [DINO 缓存流程](../VLMExpert/README.md#unifolm-feature-cache)。

## 在线共享图像

通用 `shared_rgb` 模式使用 `DinoEncoder.encode_window(history_rgb, stage=..., seed=...)` 在线编码整段历史，并将返回的 `rgb[-1]` 同时送入 VLM。增强 Motion 图像时不能使用未增强的离线 embedding 代替。

UnifoLM 的 `vlm_color_clean_motion` 策略只增强 VLM 当前帧颜色，Motion 历史使用 clean DINO；两路仍保持来源、时间和几何一致。详细接口见 [ImageProcess 分支策略](../ImageProcess/README.md#unifolm-vlm-color-clean-motion-history)。
