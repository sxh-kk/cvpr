# cvpr

MotionExpert、共享图像处理和DINO特征提取代码；VLMExpert同步官方Ψ₀工程。

| 模块 | 当前用途 | 说明 |
| --- | --- | --- |
| MotionExpert | 5Hz历史5帧＋当前帧，预测未来5帧；一步完整11个hidden作为运动先验，可反传 | [接口与训练](MotionExpert/README.md) |
| ImageProcess | 统一RGB几何，支持共享窗口增强与显式分支颜色策略 | [配置与接口](ImageProcess/README.md) |
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
UnifoLM 放在 `/gaozt-test1/datasets/UnifoLM`，原生30Hz DINO 提取与历史6帧读取已接通，
缓存和 Ψ₀ 共用 `shared_image_profile=256x320`；数据准备命令见 [UnifoLM 数据链路](VLMExpert/README.md#unifolm-下载dino-提取与读取)。
当前 UnifoLM 配方显式使用 `motion_image_policy="vlm_color_clean_motion"`：VLM 当前帧单独做 Ψ₀ 随机颜色增强，
Motion 历史6帧保持 clean，复用现有 DINO 缓存；两路仍同源、同时间、同几何 profile，不启用随机裁切。
验证／`no_aug`／推理均保持 clean；通用 `shared_rgb` 在线同步增强模式继续保留。此项调整不启动训练或重做缓存。
Python接口、数据与服务接线边界见 [双来源说明](VLMExpert/README.md#本仓库扩展双来源-action-expert)。

UnifoLM 训练集 DINO 已完成八卡提取及最终完整性检查：7,244个 episode、5,416,386帧，
可读取5,199,066个完整历史窗口。初始化固定本地 Qwen3-VL 实测权重身份与 Motion 第100轮 EMA，
保留完整11个 motion hidden 及 mean pooling；独立 Python3.11／Torch2.7 环境已与当前工程对齐。
真实完整模型已通过单卡 batch1／16、双卡每卡 batch1 的前后向与连续更新、保存恢复及冷加载检查，
结果见 [完整模型验收记录](docs/unifolm_full_model_validation.json)。
新增 [八卡每卡 batch16 验收](docs/unifolm_8gpu_batch16_validation.json) 连续8次更新通过，
累积2步、有效batch256，每步八卡完整模型哈希一致；峰值allocated／reserved为63.15／65.95 GiB。
配置与显存范围见 [UnifoLM 联合训练配方](VLMExpert/README.md#unifolm-联合训练配方)。
Motion后训练LR建议维持 `1e-5`，重点对照WholeBodyWAM、λ₀与WB-WAM的运动先验训练方式；
新head和VLM继续使用Ψ₀分组LR。2026-10-19已按用户授权启动八卡正式UnifoLM后训练：
每卡16、累积2步、全局256、40k更新（约1.97遍窗口）、1000步warmup及cosine。
当前进程、实际配置和日志见 [运行状态](/gaozt-test1/datasets/UnifoLM/.pipeline/current_unifolm_training.json)。
首次20-batch验证和初始连续更新曾通过，见 [历史启动检查记录](docs/unifolm_training_start.json)；
该任务随后在735次更新后由rank5原生CUDA断言退出，尚未生成正式checkpoint。
重启修复包括安全的spawn数据worker、生产一致的确定性验收、全卡更新前数值保护和早期保存；
保留原VLM prefix tokens，按原Ψ₀策略冻结未参与该loss的末端norm。
新验收和实际重启状态见 [重启记录](docs/unifolm_assert_repair_validation.json)，实时PID／日志以外部状态文件为准。
修订任务已于2026-10-19 05:56 UTC重启，首轮验证和第100步完整保存通过，保存后继续更新通过；
新optimizer布局的两卡B16冷恢复与继续更新也已验收，原生断言仍未独立复现，不能视作长跑问题已消除。
W&B使用离线本地记录；此前短时验收不代表长期收敛或正式吞吐验收。

UnifoLM 提速代码已合入当前仓库，续训只启用同步调试关闭、跳过不用的VLM词表输出、
CPU诊断输入／指标同步优化，AdamW保持 `foreach=False`；完整tokens、pooling、state接法和分组LR不变。
第14000步已保留恢复快照；八卡连续更新健康检查及确定性单步完整模型一致性对照通过，
新run已从该步恢复模型／优化器／scheduler／RNG并继续40k日程。
正式恢复后的53次真实数据更新短测约3.89秒／步，loss及五组梯度正常；长跑状态以外部实时记录为准。
map-style恢复在采样前跳过已完成微批，loss曲线以校验过的1–14000步历史接续新run。
[入口与边界](VLMExpert/README.md#unifolm-提速与断点恢复)、
[八卡冷恢复与提速验收记录](docs/unifolm_speedup_resume_validation.json)。

该提速任务在北京时间2026-10-20 12:06、完成15108次更新后，再次由rank5原生CUDA断言退出；
尚未定位失败算子，不能将此前短测通过视作根因修复。可恢复点仍为受保护的14000步，需重算1108次更新。
本轮恢复使用同步CUDA报错、全程tmpfs双窗口CPU batch/RNG快照，并按用户要求每1000步完整保存；
模型、完整tokens／pooling、batch与分组学习率保持原配置。
原因证据、恢复措施和验收边界见 [断言与恢复分析](docs/unifolm_assert_recovery_analysis.md)，
实际启动及进度以外部运行状态为准。
八卡恢复已于北京时间12:47启动，12:58初期检查完成14054次更新，loss与五组梯度有限，
真实双窗口CPU快照及每100步绘图监控均已接回；详见 [恢复验收记录](docs/unifolm_assert_recovery_validation.json)。

训练 loss 可由独立 CPU 观察进程按每100次优化器更新聚合并持续刷新 PNG／SVG／CSV，
入口与状态文件见 [Loss 持续绘图](VLMExpert/README.md#loss-每100步持续绘图)；不改动训练进程或日志写入。

当前修订目标为 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v4-camera-quality`。
v4继承双来源最终FK拟合验收，新增HM原始Aria质量／跟踪中断、EgoSuite相机物理连续性mask，
并在5Hz增量中检查全部中间30Hz点及边。原SMPL-H参数、双手21点、world观测、地面、RGB、分割均保留。
旧归档、缓存、统计和checkpoint保留追溯；原生30Hz DINO按相同字节复用，无需重新提取。

MotionExpert输出245维运动，另有18维相机轨迹。训练用5Hz／11帧，历史6帧的最后一帧与Ψ₀当前帧一致。
DINO使用冻结的 `dinov2-with-registers-large`，每帧CLS＋4 register，即5×1024，按episode用mmap读取。
预训练与Robust的最终RGB为H256×W480，后训练为H256×W320；两路遵循同一几何profile，各自做编码器归一化，增强按显式策略执行。

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
将UnifoLM第20000步联合模型迁移到4090、在该机独立运行模型与SIMPLE的需求、环境配置顺序及验收边界，
见 [4090独立测评交接](docs/4090_SIMPLE_HANDOFF.md)。该文档可直接交给4090端Codex执行环境准备；权重与大仿真资源另行传输。
4090本机的两套Conda环境约束、外部资源路径和无权重冒烟命令见
[环境工具](scripts/4090_simple/README.md)；实际阶段结果见 [4090环境记录](docs/4090_SIMPLE_ENVIRONMENT.md)。
A100等待20k完整保存后的保护／导出、4090 SSH拉取及SHA核验见[资源交接](docs/4090_SIMPLE_TRANSFER.md)。
资源到达后的严格GPU模型、真实历史／DINO、本机HTTP与SIMPLE闭环实际边界见
[4090 GPU验收记录](docs/4090_SIMPLE_GPU_ACCEPTANCE.md)。
用户后续授权的最新完整28k权重迁移、SONIC v1.1重编码与独立CarryBox适配见
[4090本地任务适配](docs/4090_SIMPLE_CARRYBOX_FINETUNE.md)；适配结果与原20k基线分别记录。

[4090独立举箱保持任务与真实v1.1数据](docs/4090_SIMPLE_LIFTBOX_FINETUNE.md)：原CarryBox标准保留，真实成功示范验收后再适配。
第三方代码沿用各目录原许可与版权。
