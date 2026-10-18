# cvpr

MotionExpert、共享图像处理和DINO特征提取代码；VLMExpert同步官方Ψ₀工程。

| 模块 | 当前用途 | 说明 |
| --- | --- | --- |
| MotionExpert | 5Hz历史5帧＋当前帧，预测未来5帧；一步完整11个hidden作为运动先验，可反传 | [接口与训练](MotionExpert/README.md) |
| ImageProcess | 与Ψ₀共享当前RGB、尺寸和窗口增强 | [配置与接口](ImageProcess/README.md) |
| pipeline | 从SMPL-H归档和MCAP／MP4提取原生30Hz DINO | [格式与命令](pipeline/README.md) |
| VLMExpert | Ψ₀／Qwen工程，上游 `cda3fdf`；新增可联合训练 VLM＋Motion 的双来源 action expert | [接口与上游说明](VLMExpert/README.md) |

VLMExpert于2026-10-18替换为官方Psi0 `main` 的固定快照，恢复官方训练、推理、SONIC与基线代码。
此前GLOBAL token／WBT定制保留在Git历史中。
使用独立Python 3.11环境；源码版本、依赖和权重排除记录见 [上游快照](docs/psi0_upstream.json)，
安装与本仓库的展开依赖约定见 [迁移说明](docs/MIGRATION.md#vlmexpert-环境与路径)。

本地 `dual_source_v1` 支持联合训练 VLM、MotionExpert 与新 action expert，读取 VLM 最后一层与完整11个 motion hidden，
各 block 串行执行 action self-attention、VLM cross-attention、Motion cross-attention、FFN；
由 `p_V/p_M/action time` 融合调制。训练与采样需提供真实历史 DINO `[B,6,5,1024]`，
当前帧与 VLM 对齐；不会从单帧输入伪造历史。旧 `psi0_joint` 路径保留，新旧 head checkpoint 不兼容。
UnifoLM 配方解冻两套 expert，DINO 保持冻结；VLM 使用官方分组学习率，Motion 使用独立学习率。
Python接口、数据与服务接线边界见 [双来源说明](VLMExpert/README.md#本仓库扩展双来源-action-expert)。

当前修订目标为 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v4-camera-quality`。
v4继承双来源最终FK拟合验收，新增HM原始Aria质量／跟踪中断、EgoSuite相机物理连续性mask，
并在5Hz增量中检查全部中间30Hz点及边。原SMPL-H参数、双手21点、world观测、地面、RGB、分割均保留。
旧归档、缓存、统计和checkpoint保留追溯；原生30Hz DINO按相同字节复用，无需重新提取。

MotionExpert输出245维运动，另有18维相机轨迹。训练用5Hz／11帧，历史6帧的最后一帧与Ψ₀当前帧一致。
DINO使用冻结的 `dinov2-with-registers-large`，每帧CLS＋4 register，即5×1024，按episode用mmap读取。
预训练与Robust的最终RGB为H256×W480，后训练为H256×W320；共享处理后两路各自做编码器归一化。

模型已去掉可学习GLOBAL token。`forward_features()` 在 `t=1` 完整Gaussian上前向一次，返回
`one_step_hidden [B,11,768]` 和严格的 `one_step_global = one_step_hidden.mean(1) [B,768]`。
下游使用完整 `one_step_hidden`；`one_step_global` 保留为可选摘要及既有冻结读出的接口。
保留autograd，未来标签可用性不影响部署输入。self-attention Q/K/V全归一化，cross context独立归一化。
架构为 `e7_smplh12_hands21_meanpool_qkvnorm_v2`，旧GLOBAL／旧注意力checkpoint拒绝直接恢复；
旧第60轮热启动配置已移除，当前配置从头建立新任务。

固定窗口缓存位于 `training_cache/smplh245_5hz_w11_camera-quality-v4`，
新全train统计为 `statistics/smplh_k12_5hz_joint_camera-quality-v4.pt`，均在MixEgoMotion根目录。
仅独立证明计算完全不变的episode可复用旧shard；受影响episode重建，重新打包验收、统计及GPU验证。
已通过的FK数值验收不重复执行。正式训练就绪绑定新统计、缓存、当前架构和GPU实现身份。

异常保护检查逐样本加权flow loss、每层token RMS及unscale后／clip前梯度；DDP统一判决，更新前停止。
未加权分组MSE单独检查NaN/Inf并记录，不与实际加权loss共用幅值门限。
异常时保存各卡batch、mask、来源、RNG与中间量，不额外保存大型checkpoint，正常步不写诊断文件。
新任务为8卡、每卡256、每卡24workers、LR1e-4、BF16、200轮（10轮warmup）、每20轮保存。
单机正式启动显式启用NCCL P2P／共享内存，不使用 `--ddp-socket`；指标每10步合并同步，
保留完整epoch均值和每步异常保护。普通AdamW与其余训练参数保持不变。
归档与训练路径验收已完成，WD对照在第30轮结束，20／100轮冻结先验验证已完成。
2026-10-17按用户最新安排，从原正式任务第100轮checkpoint完整恢复，目标累计200轮；
保留优化器、EMA和原200轮学习率日程，八卡每卡256／24workers，其余参数不变。
独立GPU诊断入口为 `python -m MotionExpert.probe_prior`，只拟合小读出头，按EgoSuite／HM分别报告；
具体命令、采样与解释边界见 [冻结运动先验验证](MotionExpert/README.md#冻结运动先验验证)。
支持在已验收20／100轮job上只读复用样本和基线，仅新增第160轮提取、读出及配对比较；
冻结先验的train／val指标与全量test动作指标分别报告，原20／100轮结果保留。
完整hidden另用 `MotionExpert.probe_tokens` 和 `MotionExpert.token_probe_fit` 对比历史6／未来5／全部11与DINO，
四组读出头各290,886参数，保留时间与输入有效mask；不在读取前求均值或降维。
协议与限速命令见 [完整token等参数读出对照](MotionExpert/README.md#完整token等参数读出对照)。
第100轮四组、三个读出seed的GPU对照已完成；完整11个hidden相对DINO的腕位移误差在EgoSuite／HM
分别降低约11.8%／7.0%。11与未来5的差异较小，不能据此证明历史重建损失必要或真实下游收益。
配置、手动命令、数据政策和异常诊断见 [MotionExpert README](MotionExpert/README.md)。
修复流程状态位于 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/current_motionexpert_repair.json`。
当前训练PID、日志和启动配置位于 `/gaozt-test1/datasets/MixEgoMotion/.pipeline/current_motionexpert_training.json`。
`MotionExpert/watch_loss.py` 每完成1轮，独立更新训练目录 `plots/loss_curve.png`，
展示从第1轮到当前完整轮次的训练均值与已有验证loss，同时保存CSV。
绘图只读取TensorBoard，不占训练GPU；[运行方式](MotionExpert/README.md#每轮更新loss曲线)。

环境迁移见 [MIGRATION.md](docs/MIGRATION.md)。数据、权重、缓存、checkpoint和环境不随Git分发。
第三方代码沿用各目录原许可与版权。
