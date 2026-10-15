# cvpr

MotionExpert、共享图像处理和 DINO 特征提取代码。VLMExpert 保留其原工程，本轮没有修改。

| 模块 | 当前用途 | 说明 |
| --- | --- | --- |
| MotionExpert | 5Hz 历史5帧＋当前帧，预测未来5帧；支持可反传的一步 hidden 运动先验 | [接口与训练](MotionExpert/README.md) |
| ImageProcess | 与 Ψ₀ 共享当前 RGB、尺寸和窗口增强 | [配置与接口](ImageProcess/README.md) |
| pipeline | 从已验收 SMPL-H 归档和原始 MCAP／MP4，提取原生30Hz DINO特征 | [格式与提取命令](pipeline/README.md) |
| VLMExpert | 原 Ψ₀／Qwen 工程 | [原工程说明](VLMExpert/README.md)、[设计](VLMExpert/project.md) |

训练配置指向双来源最终FK质量修订目标 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v3-fit-quality`，
独立归档验收、缓存与新train统计完成后才能启动；原v1/v2归档保留追溯。
保留两来源 SMPL-H 参数、双手21点、相机、地面、各自 mask、原分割和自有模型。
旧dense派生数据、此前废弃的DINO与早期实验checkpoint已清理；本次训练产物保留。
新全量train统计为 `/gaozt-test1/datasets/MixEgoMotion/statistics/smplh_k12_5hz_joint_fit-quality-v3.pt`；
旧统计与本次已有checkpoint保持保留，修订后不能直接复用旧归一化续训。

新视觉版本采用 `dinov2-with-registers-large`，每帧 CLS＋4 register，即5×1024。
当前DINO提取使用八卡、BF16、每进程batch128、每卡2进程；输出仍存float32，原生30Hz。
预训练与 Robust 最终 RGB 为 H256×W480；后训练为 H256×W320。
MotionExpert 的当前帧与 Ψ₀ 当前帧相同，MotionExpert 额外读取前5帧历史。
两路共享几何和增强结果，各自使用编码器归一化。

重新提取全量30Hz DINO的任务，可用 `python -m MotionExpert.watch_dino --background`
自动启动八卡从头训练；完成计数、mask约定、提取进程退出和缓存锁释放均满足才交接。
守护固定配置快照，记录状态／日志，拒绝重复启动及覆写已有训练目录；
完整命令与监控方式见 [MotionExpert README](MotionExpert/README.md#dino完成后自动启动八卡训练)。
训练保留8卡、每卡batch256、每卡24workers、LR1e-4、BF16、200轮、每20轮保存。
本次双来源最终FK质量修订复用已有DINO，修订处理不会自动启动训练。

MotionExpert支持固定5Hz／11帧的运动窗口mmap缓存，将245维运动、18维轨迹、mask与解码变换
按split存为大型 `.npy`，减少重复Parquet读取、FK和坐标计算；原生30Hz DINO仍使用已有mmap。
缓存构建可与当前训练并行，支持多worker及外部控制文件动态调整并行度；
worker共享父进程已验证的来源快照，已生成窗口可恢复复用。
episode各字段写出后与RAM生成值核对，split打包后核对完整payload及冻结窗口索引，
发现不一致时阻止缓存发布与训练切换；验收报告记录具体episode、起始帧与失败字段。
正式缓存验收与缓存训练启动在各自进程内复用已验证来源Archive，保留完整身份及各split独立检查，减少重复初始化。
同次运行且归一化不变时，原训练可持续运行；缓存全量生成并独立验收通过后，在最近自然20轮保存点完整续训，
保留优化器、EMA、学习率调度及训练参数，不增加保存次数。
关闭超时仅通过安全pidfd接口清理身份仍匹配的旧进程；接口不可用时明确报告人工清理需求，阻止重复启动。
构建、验收、自动切换命令与监控见
[窗口缓存说明](MotionExpert/README.md#固定窗口-mmap-缓存与已有训练切换)。

HM发布的原生 `valid` 和EgoSuite的模型补全mask都不能保证最终身体与头戴相机位置一致。
双来源新增最终SMPL-H FK头部与相机、源头部观测的物理筛选，阈值只用train校准。
EgoSuite保留独立有效源手/脚/相机，屏蔽不可信身体补全；HM拟合关联身体/手/脚同步筛选。
原参数、相机、图像join和DINO字节保留，重建受影响运动窗口并重算全train统计。
本轮在v2基础上新增屏蔽61,311帧异常拟合监督，原生帧与原分割不删除。
归档发布与训练就绪是两阶段；新缓存、统计和GPU验收完成前阻止正式训练。
流程状态为 `.pipeline/current_final_fit_quality.json`，详见
[质量修订说明](MotionExpert/README.md#拟合监督质量修订与统计)。

环境迁移见 [MIGRATION.md](docs/MIGRATION.md)。数据、预训练权重、训练 checkpoint 和运行环境不随 Git 仓库分发。
第三方代码沿用各目录原有许可与版权。
