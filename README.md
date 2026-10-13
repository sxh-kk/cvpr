# cvpr

MotionExpert、共享图像处理和 DINO 特征提取代码。VLMExpert 保留其原工程，本轮没有修改。

| 模块 | 当前用途 | 说明 |
| --- | --- | --- |
| MotionExpert | 5Hz 历史5帧＋当前帧，预测未来5帧；支持可反传的一步 hidden 运动先验 | [接口与训练](MotionExpert/README.md) |
| ImageProcess | 与 Ψ₀ 共享当前 RGB、尺寸和窗口增强 | [配置与接口](ImageProcess/README.md) |
| pipeline | 从已验收 SMPL-H 归档和原始 MCAP／MP4，提取原生30Hz DINO特征 | [格式与提取命令](pipeline/README.md) |
| VLMExpert | 原 Ψ₀／Qwen 工程 | [原工程说明](VLMExpert/README.md)、[设计](VLMExpert/project.md) |

正式运动数据位于 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1`。
保留两来源 SMPL-H 参数、双手21点、相机、地面、各自 mask、原分割和自有模型。
旧 dense 派生数据、旧 DINO 特征、旧训练 checkpoint 和历史实验代码已清理。
全量 train 运动统计单独保留在 `/gaozt-test1/datasets/MixEgoMotion/statistics/smplh_k12_5hz_joint.pt`。

新视觉版本采用 `dinov2-with-registers-large`，每帧 CLS＋4 register，即5×1024。
当前DINO提取使用八卡、BF16、每进程batch128、每卡2进程；输出仍存float32，原生30Hz。
预训练与 Robust 最终 RGB 为 H256×W480；后训练为 H256×W320。
MotionExpert 的当前帧与 Ψ₀ 当前帧相同，MotionExpert 额外读取前5帧历史。
两路共享几何和增强结果，各自使用编码器归一化。

全量30Hz DINO提取已启动；完成后再从头训练MotionExpert，MotionExpert训练尚未启动。
训练保留8卡、每卡batch256、每卡24workers、LR1e-4、BF16、200轮、每20轮保存。

环境迁移见 [MIGRATION.md](docs/MIGRATION.md)。数据、预训练权重、训练 checkpoint 和运行环境不随 Git 仓库分发。
第三方代码沿用各目录原有许可与版权。
