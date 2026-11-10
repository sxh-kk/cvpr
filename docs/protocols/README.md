# 数据与模型协议

本页汇总跨模块协作所需的稳定约定。具体字段和实现以 [MotionExpert](../../MotionExpert/README.md)、[ImageProcess](../../ImageProcess/README.md)、[pipeline](../../pipeline/README.md) 与 [VLM Expert](../../VLMExpert/README.md) 为准。

## 时间与条件输入

| 项目 | 约定 |
| --- | --- |
| 原始运动与离线 DINO | 原生 30Hz，保留时间与来源绑定 |
| MotionExpert 时间轴 | 5Hz、11 帧；历史 5 帧＋当前帧＋未来 5 帧 |
| 历史视觉条件 | 6 个真实时刻，最后一帧与 VLM 当前帧同源、同时间 |
| DINO 输入张量 | `[B,6,5,1024]`；未来图像不作为条件 |
| 标签与有效 mask | 用于监督；未来标签可用性不得影响部署输入 |

在线输入须积累真实历史，不能复制单帧补齐 6 个时刻。训练中的轨迹模态 dropout、验证／推理的轨迹隐藏规则见 [MotionExpert 时间与条件](../../MotionExpert/README.md#时间图像与条件)。

## 运动表示与模型接口

- 运动表示为 245 维，另有 18 维相机轨迹；保留既定 K12 身体关节和左右各 21 点手部监督。
- 使用相机地面投影参考系，单位为米，世界坐标 Z-up；6D 旋转取旋转矩阵前两行。
- 架构为 `e7_smplh12_hands21_meanpool_qkvnorm_v2`，没有可学习 GLOBAL token；self-attention 的 Q/K/V 全归一化，cross context 独立归一化。
- `forward_features()` 在 `t=1` 完整高斯输入上前向一次，返回 `one_step_hidden [B,11,768]`，并保留 autograd；缺失标签不能将噪声通道清零。
- `one_step_global = one_step_hidden.mean(1)` 为可选摘要。双来源 Action Expert 使用完整 11 个 hidden。
- 旧 GLOBAL／旧注意力 checkpoint 与当前架构不直接兼容；`psi0_joint` 和 `dual_source_v1` 也有独立的动作头 checkpoint 边界。

各通道的切片、归一化和质量规则见 [运动表示](../../MotionExpert/README.md#运动表示)。

## 图像与视觉特征

DINO 使用冻结的 `dinov2-with-registers-large`；每帧输出 CLS＋4 个 register，共 `5×1024`。离线特征按 episode 保存为 `features.npy`，训练通过 mmap 按需读取。

| 阶段 | 最终 RGB 高×宽 |
| --- | --- |
| 运动预训练 | 256×480 |
| 后训练 | 256×320 |
| Robust | 256×480 |

两路共享几何 profile，随后各自执行编码器归一化；DINO 禁用自身二次 resize/crop。原始相机去畸变和几何沿用来源数据约定，ImageProcess 不新增去畸变。

增强策略由配置显式选择：

- `shared_rgb`：每个窗口采样一次增强参数，整段历史共享颜色和随机裁切参数。若增强 Motion 输入，须在线编码同步增强后的全部历史，不能复用 clean DINO 代替。
- `vlm_color_clean_motion`：UnifoLM 配方只增强 VLM 当前帧颜色，Motion 历史 6 帧使用 clean 256×320 DINO 缓存；两路仍同源、同时间、同几何，无随机裁切。
- 验证、`no_aug` 和推理保持 clean。更换视觉尺寸须显式声明适配，DINO 权重身份和归一化仍须匹配。

详细接口见 [共享图像策略](../../ImageProcess/README.md#shared-training-augmentation) 与 [UnifoLM 分支策略](../../ImageProcess/README.md#unifolm-vlm-color-clean-motion-history)。

## 数据质量、来源与迁移

运动归档保留原始分割、手部、相机、地面、world 观测及有效 mask。5Hz 参考增量必须核对间隔内全部 30Hz 点和边，不能跨越跟踪中断。EgoSuite 没有源 confidence 时不伪造该字段；相机质量失败也不直接删除独立 world 观测或 DINO。

数据、模型和输出目录的配置方式见[路径配置](../../README.md#configuration)。
