# MixEgoMotion：当前数据与 DINO 提取

当前 pipeline 只负责从已验收的 SMPL-H 运动归档和原始相机视频提取新 DINO 特征。
历史 dense 转换、拟合实验、迁移和全量数值审计脚本已清理。
当前训练目标为 `smplh-parameters-v4-camera-quality`，运动质量修订由MotionExpert负责，
已提取的原生30Hz DINO及join复用相同字节，不需要为运动mask修订重新提取。
原v1/v2/v3目录保留追溯，下方示例使用v4路径。相机原生质量修订继承v3的FK验收，
新增camera point/transition mask，MotionExpert检查5Hz间全部原生边，避免跨跟踪中断生成有效增量。
新缓存、新全train统计与当前模型GPU验收完成前不能启动训练，本次不自动启动训练。

## 保留的数据

根目录：`/gaozt-test1/datasets/MixEgoMotion`。

```text
MixEgoMotion/
├── smplh-parameters-v4-camera-quality/
│   ├── meta/                  # info、episode索引、train/val/test分割
│   ├── episodes/XXXXXXXX/
│   │   ├── motion.parquet     # 原生30Hz运动参数与显式标签
│   │   └── receipt.json       # 原始不可变提交凭证
│   ├── weights/smplh/         # 自有共同人体模型与手部表示资源
│   ├── .pipeline/             # 提交记录、验收证据、旧视觉退休清单
│   └── vision_profiles/psi0_256x480_v1_30hz/  # 原有DINO按相同字节复用
├── statistics/smplh_k12_5hz_joint_camera-quality-v4.pt         # 相机质量mask修订后的全量train统计
├── weights/dinov2-with-registers-large/     # 保留DINO预训练权重
├── .pipeline/runtimes/                     # 现有提取／训练环境
└── training/                               # 新训练产物
```

| 来源 | Episodes | 原生30Hz帧数 |
| --- | ---: | ---: |
| EgoSuite | 6,614 | 26,841,893 |
| EgoExo4D-HM | 2,649 | 11,295,991 |
| 总计 | 9,263 | 38,137,884 |

原分割为 train 7,409、val 915、test 939 episodes，见 `meta/splits.json`。
47段HM隔离记录保留；训练按 `training_eligible` 与窗口mask过滤。
已通过的运动数值验收不重复执行。

`motion.parquet` 保留：

- `smplh.root_orient [3]`、`pose_body [63]`、`trans [3]`、`betas [16]`，以及来源可用时的手指PCA。
- 左右各21点世界XYZ及逐点mask，独立于手指PCA可用性。
- 相机／头部／头戴参考世界位置、四元数、mask及插值标记。
- v4受影响episode额外保存相机点质量与原生incoming-edge连续性mask；其余episode由独立全有效证书继承原mask。
- 地面高度、法向、有效性／认证标记，原始双脚点及mask。
- 身体估计与来源观测的独立位置／旋转mask、confidence、完成及残差信息。
- 原生frame/index、时间、视频帧号、PTS误差及原始图像mask。

共同模型定义SMPL-H身体22关节轴，世界坐标米、Z-up。MotionExpert构建窗口缓存时做FK并选既有K12，训练直接mmap读取，
同时构建相机地面参考坐标和接触代理；不在归档增加dense22缓存。
共同模型SHA256为 `be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947`。

旧 `vision.parquet` 已退休删除；旧receipt和record保留原始来源身份。
`.pipeline/retired-vision.json` 绑定归档metadata、records及逐文件旧身份，允许运动读取独立于已删除的旧特征。
新的视觉缓存拥有独立manifest，训练不能回退到旧视觉版本。

## 图像与特征协议

共享配置位于 [ImageProcess](../ImageProcess/README.md)：

| 阶段 | 初始尺寸 H×W | 最终共享RGB H×W | 增强 |
| --- | --- | --- | --- |
| 预训练 | 270×480 | 256×480 | 确定性处理 |
| 后训练 | 240×320 | 256×320 | 窗口共享颜色增强 |
| Robust | 270×480 | 256×480 | 窗口共享颜色及随机裁切 |
| 验证／真机推理 | 按对应profile | 按对应profile | 确定性处理 |

表中后训练和Robust的窗口共享增强指通用在线 `shared_rgb` 模式。当前UnifoLM配方例外：
`vlm_color_clean_motion` 只增强VLM当前帧颜色，Motion历史六帧复用clean 256×320 DINO缓存；
两路保持同源、同时间、同几何profile，不启用随机裁切，现有缓存无需重提取。
详见 [ImageProcess的分支增强说明](../ImageProcess/README.md#unifolm-vlm-color-clean-motion-history)。

DINO为冻结的 `dinov2-with-registers-large`。共享RGB处理后关闭HF的二次resize/crop，
只执行DINO自身归一化，输出 `[5,1024] float32`，顺序CLS、register0..3。
原始MCAP／MP4保持来源相机既有几何约定；共享模块不新增去畸变。

离线特征保存全部原生30Hz帧。MotionExpert按索引0、6、12…取5Hz，
每窗口11帧，前6帧为历史5帧＋当前帧，后5帧为未来。
特征只允许历史6帧作为条件，未来隐藏。

新缓存格式：

```text
vision_profiles/psi0_256x480_v1_30hz/
├── manifest.json
└── episodes/XXXXXXXX/
    ├── features.npy          # [T30,5,1024]，float32，支持mmap
    ├── join.npz              # 原生帧号／时间／来源视频帧号／PTS／mask
    └── receipt.json          # 来源record、encoder/profile、输出文件身份
```

提取恢复只复用同配置、同来源、同文件身份的已完成episode。
坏帧不提升为有效，输出mask必须与原始来源mask相同，否则正式提取或训练拒绝。
训练要求完整 `complete/all` manifest与同一归档、分割、图像profile绑定。
视觉尺寸变化本身不要求重算运动统计；本次相机质量mask变化，需要新全train统计。视觉版本另行写入checkpoint。

## 全量提取命令

从仓库根目录执行。当前环境和预训练权重已就绪，无需重新下载。
本次运动质量修订复用已有完整视觉缓存，不需要执行以下命令。
以下为另行重提取任务的命令示例，完整数据约781GB float32特征，另加join和凭证；同命令恢复只复用匹配来源和配置的产物。
CLI固定原生30Hz，无需指定fps。当前选定BF16、每进程batch128、每卡2个DINO进程，8卡共16个进程。
每个进程保留8个PIL预处理线程、4个解码线程和2批预取；输出仍为float32。
八卡真实视频对照约1,370～1,419帧/秒，原TF32／batch32／每卡1进程约1,061帧/秒。
此测量包含解码、预处理和写盘，排除模型初始化，使用重复读取的双来源片段。
BF16的特征数值与TF32略有差异；128帧双来源对照中平均token余弦相似度均超过0.9999。
完整配置、实验及选择记录：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/jobs/dino-speed-20261013/summary.json`。

```bash
cd /root/workspace/wbc/guanzerong/cvpr
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/dino-torch260-cu118/bin/python \
  -m egomotion_pipeline.cli extract-smplh-features \
  --dataset /gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v4-camera-quality \
  --output /gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v4-camera-quality/vision_profiles/psi0_256x480_v1_30hz \
  --config pipeline/configs/dino.local.json \
  --gpus 0 1 2 3 4 5 6 7 \
  --workers-per-gpu 2
```

同命令再次执行可以恢复。加 `--dry-run` 只检查原始视频来源与提取计划，不加载DINO、不创建输出。
`--episodes` 只用于独立诊断目录，部分缓存不能用于正式训练。
MotionExpert从头训练命令见 [MotionExpert README](../MotionExpert/README.md)。

后训练／Robust用 `DinoEncoder.encode_window(history_rgb, stage=..., seed=...)` 在线编码，
将返回的 `rgb[-1]` 同时送入Ψ₀，确保当前帧与增强完全相同。
增强变化后不能复用固定离线embedding。运动时间、坐标、标签和mask保持原协议。
