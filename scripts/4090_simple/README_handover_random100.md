# Handover：重新采集随机化 100 条

这套入口为每条候选重新执行物理仿真，记录实际执行的 SONIC 1.1 token、手部指令、积分状态和真实 RGB。原始动作只提供姿态与协调动作参考；候选来源、场景数量、静态 IK 可达、脚本完成和单步 decoder 误差改善都不代表采集成功。运行数据和失败记录统一保存在仓库外。

## 来源和场景

- 原始来源为固定版本的 `USC-PSI-Lab/psi-data` 的 `simple-teleop/G1WholebodyHandoverTeleop-v0/level-0`。归档 SHA、抽取成员、原始 Parquet 和本地资源均绑定身份。
- 100 条原始记录中，39/65/80/92/94 为保留验证来源，38 只有一帧；其余 94 条是完整 TRAIN 候选。它们不等于 94 条已验收的新示教，也没有可证明独立操作者或采集会话的元数据。
- 正式要求至少 50 个不同的原始 TRAIN 来源，每个最多收录 2 条；先扩充旧 24 个来源之外的候选，不用同一个场景的手部映射 A/B 对照凑数。
- target 与 container 分别连续随机平移及旋转；正式 profile 为每轴 ±5 cm、yaw ±0.25 rad。灯光和桌面 MDL、实际被可见几何使用的机器人及物体材质也随机化并从 USD 回读。相机标定和物体质量保留。当前只使用真实 scene0，不声称多房间。
- `observation.eef_state` 与源动作 EEF 相同，不能当成测量 FK；手腕参考由命名关节和浮动根位姿在私有 MuJoCo 数据中重算。历史手部映射存在歧义，因此显式记录 `declared` / `inferred_effective` 条件。

## 采集与准入

`replay_handover_random100.py` 在新场景中运行物体反馈专家，重新编码官方 SONIC 1.1 动作。实际控制为 200 Hz 物理、50 Hz 官方 decoder、30 Hz 新指令。1 秒真实暖启动不作为动作监督；完整历史/未来窗口之外的数据不作为有效训练 current。

默认 `--placement-encoder-mode g1` 使用关节参考；可显式选择 `teleop`，仅在放置阶段改用官方 mode1 的双腕与头部空间目标，抓取和交接仍沿原路径。`handover_sonic_teleop_reference.py` 按官方 C++ 构造该1751维输入，保留原始腿部参考及躯干风格；空间目标先变换到完整实际根坐标，独立 heading anchor 再按官方规则处理。该实验也只输出新编码的64维 token，不绕过官方 decoder、不直接施加关节位置，最终验收相同。

`--placement-locomotion source_future` 保留接管后的原始腿、腰未来参考；`actual_hold` 使用实际姿态初始化放置阶段的腰与持物臂联合 IK，约束来自实际足底接触几何和重心投影，指令限速。静态保持参考的未来速度为零。该几何约束不是动态平衡证明，两种条件都必须重新通过真实物理验收。

`audit_handover_random100.py` 独立重载来源、配置、场景、积分状态及实际指令，重放专家计算，并核对接触、完整放置、真实 RGB 和时间对应关系。执行期间不得恢复物体、附加物体或施加外力。可选的有界 latent 投影也必须独立重放、绑定派生模型身份；单步投影改善不作为成功证据。

原有 `handover-physics-v2` 不修改：有序的右手抬升、双手交接、左手接管，随后松手、容器承重及末 50 个 50 Hz 状态稳定。新增要求为这 50 个真实状态中，target 的全部实际碰撞顶点均位于 container 外碰撞凸包投影内，最小余量至少 5 mm。外凸包不是精确内腔，报告明确保留这一限制。

`diversity_handover_random100.py` 要求恰好 100 条验收记录，并检查来源上限、唯一场景、实际 XY/yaw 范围与标准差、至少 50 种实际桌面 MDL、关键阶段覆盖及场景相关动作修正。使用去初始全局 SE(3)、时间归一化后的实际关节/FK 轨迹检测近重复；仅改 seed 或拉伸时间不能通过。

## 入口与执行顺序

源码入口均位于本目录：

1. `handover_random100_sources.py`：原始归档核验及候选加载。
2. `handover_random100_scene.py`：冻结场景提案、初始化变换、实际渲染参数应用及回读。
3. `handover_random100_expert.py`：闭环新动作参考及官方在线编码。
4. `replay_handover_random100.py`：单条 CPU 物理筛查或真实 RTX 采集。
5. `audit_handover_random100.py`：独立单条准入。
6. `collect_handover_random100.py prepare/run/status`：可恢复候选队列；完整正式 profile 的独立 RTX bootstrap 通过后，才允许批量运行。
7. `diversity_handover_random100.py`：完整注册表的多样性验收和替换建议。
8. `pack_handover_random100.py`：只打包全部通过的注册表，沿用原 checkpoint 归一化，不重新拟合。
9. `verify_handover_random100.py`：真实 DINO 全量核对及生产数据读取器、RGB/action/state/history/collator 验证。
10. `finalize_handover_random100.py prepare/upload`：完整采集、打包、DINO及本地生产读取均通过后冻结传输清单；上传后核验远端字节、显式重定位DINO metadata，并运行实际A100 CPU读取，全部通过才写独立 `RELEASE_READY.json`。

各入口使用 `--help` 查看参数。默认资源根为 `/home/zxc/CVPR_Evaluation_runtime/resources`；本次运行根为 `/home/zxc/CVPR_Evaluation_runtime/handover_random100_20261024`，冻结标准见其中的 `REBUILD_REQUIREMENTS.json`。正式队列中的实际接纳注册表、独立验收和多样性报告共同决定进度；开发 pilot 不自动计数。

本机 `collect_handover_random100.py run` 与物理/独立动作审核应使用已安装 SIMPLE 的 `cvpr-simple-sim310` 解释器；`cvpr-simple-model311` 可用于准备、打包和模型侧工具，但缺少 SIMPLE 时不能运行调度的 bootstrap 审核。启动验收样本已通过并不等于100条完成，批量调度仍会逐条重新执行和拒绝失败候选。

物理质量、多样性、真实 DINO 和生产读取全部通过后，才将完整包、特征及验收记录同步到 A100 的独立新目录，再核对传输 SHA 和服务器实际读取。验证集、旧数据和现有训练保持原有版本。采集成功只证明这些训练轨迹达到所列标准，不代表微调后的策略成功率或泛化能力已经提高。
