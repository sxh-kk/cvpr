# WBT-native Action Expert Posttraining 改造方案

## 目标与范围

本方案将 WBT 替换 Humanoid Everyday（HE）作为 PSI0 action expert 的 posttraining 数据源。posttraining 直接学习 WBT 的全身 state/action 协议，并在进入 trainer 前把不同来源的手部字段统一为 12D 的源手语义。本文只覆盖 WBT posttraining；后续目标机器人适配和部署不在范围内。

本方案只描述代码和数据接口的改造计划，不包含代码实施、数据转换执行或模型训练。

前置条件是已有一个包含 `<GLOBAL>` 的 VLM checkpoint。本文讨论 action expert 的连续动作 posttraining；WBT 的 48D 是连续 state/action 维度，不是 VLM 预训练 token 维度。posttraining 冻结 VLM，只训练 action expert。

## 当前 HE_RAW 链路

当前 HE posttraining 的职责分布如下：

~~~
scripts/train/psi0/posttrain-he-psi0.sh
    -> src/psi/config/train/posttrain_he_psi0_config.py
    -> src/psi/config/data_he.py
    -> src/psi/data/humanoid/he_raw_dataset.py
    -> src/psi/config/transform.py::HEPosttrainRepackTransform
    -> src/psi/trainers/posttrain.py
~~~

相关入口：

- scripts/train/psi0/posttrain-he-psi0.sh
- src/psi/config/train/posttrain_he_psi0_config.py
- src/psi/config/data_he.py
- src/psi/config/transform.py
- src/psi/trainers/posttrain.py

HE transform 读取 action.joint_angles、observation.hand_joints 和 observation.arm_joints，最终生成 36D action 和 36D padded state。WBT 的字段和动作语义不同，不能只替换 root_dir。

## WBT 对应链路

建议为 WBT 建立与 HE 平行的模块，保留 HE baseline：

~~~
scripts/train/psi0/posttrain-wbt-native-psi0.sh
    -> src/psi/config/train/posttrain_wbt_psi0_config.py
    -> src/psi/config/data_wbt.py
    -> src/psi/data/humanoid/wbt_dataset.py
    -> src/psi/config/transform.py::WBTNativePosttrainRepackTransform
    -> src/psi/trainers/posttrain.py
~~~

文件对应关系：

| HE_RAW 文件 | WBT 文件 | 职责 |
| --- | --- | --- |
| src/psi/config/data_he.py | src/psi/config/data_wbt.py | 数据配置和 dataset 构造 |
| src/psi/data/humanoid/he_raw_dataset.py | src/psi/data/humanoid/wbt_dataset.py | 原始数据读取和时序采样 |
| （无对应文件） | src/psi/data/humanoid/wbt_hand_protocol.py | hand layout 注册、canonicalization 和协议校验 |
| HEPosttrainRepackTransform | WBTNativePosttrainRepackTransform | 字段映射、动作拼接、mask |
| src/psi/config/train/posttrain_he_psi0_config.py | src/psi/config/train/posttrain_wbt_psi0_config.py | 动态配置组合 |
| scripts/train/psi0/posttrain-he-psi0.sh | scripts/train/psi0/posttrain-wbt-native-psi0.sh | 训练入口 |
| assets/stats/he_raw_rel_stats_combined_no_static.json | assets/stats/wbt_native_stats.json | WBT 独立统计量 |

src/psi/config/transform.py 和 src/psi/trainers/posttrain.py 是共享文件，只新增 WBT 分支，不删除或重写 HE 分支。

当前工作树中没有看到 src/psi/data/humanoid/he_raw_dataset.py 的实际文件，但 data_he.py 正在导入它。实施前需要确认它是否来自外部安装包、子模块或未显示的工作树内容；WBT 读取器应保持同样的 import 约定。

`wbt_hand_protocol.py` 建议提供一个显式的 layout registry，而不是在 dataset 中散落索引：

~~~
HAND_LAYOUTS[hand_layout_id] = {
    "left_indices": [...],
    "right_indices": [...],
    "units": [...],
    "signs": [...],
    "offsets": [...],
}
canonicalize_hand(raw_hand, hand_layout_id) -> (left6, right6, metadata)
~~~

registry 缺少 layout、单位或方向信息时必须抛出错误。每个样本只引用已注册的 `hand_layout_id`，不能根据数据维度自动推断。

## WBT-native-v1 协议

posttraining 使用一个与目标手型无关的 WBT 源协议。root、G1 身体关节和左右手各 6 个源语义组成 48D：

~~~
state  = root_pose7 + body_q29 + left_hand_source6 + right_hand_source6 = 48D
action = root_target7 + body_target29 + left_hand_source6 + right_hand_source6 = 48D
~~~

WBT 原始字段的语义是：

~~~
observation.state.robot_q_current: root pose 7 + G1 body joints 29
action.robot_q_desired:             root pose 7 + G1 body joints 29
observation.state.hand_state:       source hand representation, canonicalized to 6D per hand
action.hand_cmd:                    source hand representation, canonicalized to 6D per hand
~~~

协议中应显式保存每一维的名字和顺序，例如：

~~~
root_x, root_y, root_z,
root_qw, root_qx, root_qy, root_qz,
body_0 ... body_28,
left_hand_thumb_flex, left_hand_thumb_lateral,
left_hand_index, left_hand_middle, left_hand_ring, left_hand_little,
right_hand_thumb_flex, right_hand_thumb_lateral,
right_hand_index, right_hand_middle, right_hand_ring, right_hand_little
~~~

上面的手部字段是统一后的源语义名称，不是任何目标机器人的电机编号。每只手固定为
`[thumb_flex, thumb_lateral, index, middle, ring, little]`；原始字段通过显式 permutation、单位换算和方向校正得到这 6 个值。

维度范围必须固定并写入 manifest：

| 维度 | 范围 | 内容 |
| --- | --- | --- |
| 0:7 | 7 | root position + quaternion |
| 7:36 | 29 | G1 body joint state/target |
| 36:42 | 6 | left source hand |
| 42:48 | 6 | right source hand |

这里的 `body_0 ... body_28` 顺序必须直接引用 WBT 的 joint-name 列表，不能按数组位置猜测。root quaternion 的排列、坐标系、零位和是否使用 episode-relative frame 也必须写入协议元数据。

四元数顺序必须依据 WBT 导出代码确认，不能默认 `xyzw` 或 `wxyz`，不能在 transform 中隐式改变。

如果 WBT 数据确实没有 root，或 posttraining 的控制接口明确由外部模块独立处理 root，可以另定义 wbt-native-body-v1：

~~~
state  = body_q29 + hands12 = 41D
action = body_target29 + hands12 = 41D
~~~

41D 版本会丢失 WBT root target，不能作为 48D 版本的无记录替代。默认实现使用 48D；只有在确认数据和控制接口都没有 root 后才使用 41D，并在协议元数据中记录版本。

动作第一版使用绝对目标：

~~~
use_delta_actions = false
~~~

WBT 的 robot_q_desired 是给低层控制器跟踪的目标，不应在数据转换阶段先做差分。绝对和相对动作需要作为不同协议版本处理。

## WBT 原始数据映射

### 观测

~~~
observation.state.robot_q_current
    -> current_root7 + current_body29

observation.state.hand_state
    -> current_left_hand_source6 + current_right_hand_source6

observation.images.head_stereo_left
    -> observation.images.egocentric
~~~

当前使用单头部相机，因此只保留 head_stereo_left。训练时不能使用双头相机而部署时只发送单头相机。

### 动作

~~~
action.robot_q_desired
    -> target_root7 + target_body29

action.hand_cmd
    -> target_left_hand_source6 + target_right_hand_source6
~~~

### 源手部 canonicalization（posttraining 必需）

WBT 的手部字段按数据来源可能是左右手合计 12D，但原始顺序和命名不一定一致。posttraining 只做源协议内的 canonicalization，不做目标机器人 retarget。每个数据仓库必须提供可追溯的 `hand_layout_id`，未知布局直接报错。

至少记录以下信息：

- 原始左右手字段到 6D canonical 顺序的 permutation；
- 每一维的物理含义、单位（角度或弧度）、范围和正负方向；
- 零位偏置、无效值和缺失值处理；
- 数据来源手型及 `hand_layout_id`。

推荐 canonical 顺序为：

~~~
[thumb_flex, thumb_lateral, index, middle, ring, little]
~~~

可先按 WBT 已提供的字段说明注册下表，实际索引仍需用每个仓库的 `info.json` 和样本值复核：

| `hand_layout_id` | 每只手的原始顺序示例 | canonical permutation | 处理 |
| --- | --- | --- | --- |
| `inspire-6d` | index, middle, ring, little, thumb_flex, thumb_lateral | `[4, 5, 0, 1, 2, 3]` | 允许 |
| `brainco-6d` | thumb_flex, thumb_lateral, index, middle, ring, little | `[0, 1, 2, 3, 4, 5]` | 允许 |
| `dex1-1d` | 单个开合量 | 不适用 | 默认排除 |

左右手的拼接顺序也必须单独记录；不能假设 raw vector 总是 `[left6, right6]`。state 和 action 如果使用不同的原始字段，应分别记录 `state_hand_layout_id` 和 `action_hand_layout_id`，不能共用未经核对的索引表。

不能直接截断、补零或把不同语义的 12D 字段按位置拼接。来自 Inspire、BrainCo 等布局不同的数据应先通过各自的 source adapter 转为上述 6D；只有完成统一并通过统计检查后才能混合。仅有 1D 手部信息的数据默认排除，除非另行定义带有效 mask 的协议版本。

canonicalization 的输出应保留 `float32` 原始值和元数据，便于追溯；不要在这一层引入目标机器人关节限位或电机映射。

## wbt_dataset.py 设计

拟新增：

~~~
src/psi/data/humanoid/wbt_dataset.py
~~~

该类可以内部使用 LeRobot v3.0 reader，但对上层暴露与 HE dataset 类似的时序接口。职责包括：

1. 读取 meta/info.json、episode metadata 和 task metadata。
2. 读取 parquet 中的 current state、desired action 和时间戳。
3. 读取 head_stereo_left 视频帧。
4. 以当前帧 t 为 observation，采样 t ... t+H-1 的 action。
5. 在 episode 尾部 padding，并生成有效 action mask。
6. 解析 task 文本，传递给 instruction。
7. 检查 FPS、维度、NaN、Inf 和 episode 边界。

建议配置字段：

~~~
root_dir
train_episodes
val_episodes
train_repo_ids
val_repo_ids
camera_key
hand_protocol
hand_layout_id
state_hand_layout_id
action_hand_layout_id
preserve_raw_hand
action_chunk_size
use_delta_actions
~~~

`hand_protocol` 固定为 `wbt-source-12d-v1`；`hand_layout_id` 可作为 state/action 共用布局的别名，若两者不同则分别填写 `state_hand_layout_id` 和 `action_hand_layout_id`。读取器只负责读取和对齐，source hand adapter 负责 permutation、单位和方向转换，二者都要把协议元数据写入样本或数据集 manifest。

WBT 的 info.json 可能只有 train split，不能让验证集自动复用第一个训练仓库。验证集必须按 episode 划分，或者在转换阶段生成独立的 train/val dataset。

## data_wbt.py 设计

拟新增：

~~~
src/psi/config/data_wbt.py
~~~

结构仿照 HERawDataConfig，但数据源改为 WBTDataset。它负责把配置参数传递给 dataset，并从 transform 中读取：

~~~
num_past_frames
action_chunk_size
use_delta_actions
camera_key
hand_protocol
hand_layout_id
state_hand_layout_id
action_hand_layout_id
~~~

推荐默认值：

~~~
camera_key = observation.images.head_stereo_left
action_chunk_size = 30
use_delta_actions = false
hand_protocol = wbt-source-12d-v1
~~~

如果先将 WBT 离线转换为已经包含 states、action 和 observation.images.egocentric 的标准 LeRobot 数据集，也可以复用 LerobotDataConfig 和 RealRepackTransform，但离线结果必须已经完成 `wbt-source-12d-v1` canonicalization。直接实现 WBTDataset 的方案更接近 HE_RAW 的目录职责，便于保留原始字段检查。

## WBTNativePosttrainRepackTransform 设计

在以下共享文件中新增类：

~~~
src/psi/config/transform.py
~~~

输出结构必须与 PosttrainTrainer 一致：

~~~
{
    "observations": [PIL.Image],
    "states": np.ndarray,        # [num_past_frames + 1, 48]
    "actions": np.ndarray,       # [action_chunk_size, 48]
    "instruction": str,
    "actions_mask": np.ndarray,  # [action_chunk_size, 48]
    "obs_mask": np.ndarray,      # [num_past_frames + 1]
    "dataset": "wbt-native",
    "protocol": "wbt-native-v1",
}
~~~

处理逻辑：

1. 将单头部图像转换为 observations。
2. 将 raw hand fields canonicalize 为左右手各 6D，并拼接 current root、body 和源手状态。
3. 拼接 desired root、body 和源手目标；不调用目标机器人 retarget。
4. 依据时间戳构造动作 chunk。
5. 对 episode 尾部做 padding；padding 只能重复最后一帧作为数值占位，不能读取下一个 episode。
6. 对 padding 时间步的全部 48 个 action 维度设置 `actions_mask=0`，有效时间步为 1。
7. 将 task 转为小写 instruction。
8. 对 state/action、协议版本和 mask 做最终断言，并保留 raw hand 字段的 provenance。

推荐输出尺寸：

~~~
states.shape  == [num_past_frames + 1, 48]  # 默认 [1, 48]
actions.shape == [30, 48]
actions_mask.shape == [30, 48]
~~~

不要在该 transform 中调用 HE 的字段名或 HE 的 hand/arm 排列。也不要在这里生成 FAST token；FAST 属于 VLM 预训练数据通路。若原始 hand layout 未注册，transform 必须失败而不是猜测排列。

## posttraining 配置

拟新增：

~~~
src/psi/config/train/posttrain_wbt_psi0_config.py
~~~

配置组合关系：

~~~
DynamicDataTransform:
    repack = WBTNativePosttrainRepackTransform
    field  = ActionStateTransform
    model  = Psi0ModelTransform

DynamicDataConfig:
    WBTDataConfig

DynamicLaunchConfig:
    LaunchConfig
~~~

模型参数建议：

~~~
train.name = posttrain
model.action_dim = 48
model.odim = 48
model.action_chunk_size = 30
model.action_exec_horizon = 30
model.noise_scheduler = flow
model.train_diffusion_steps = 1000
model.tune_vlm = false
repack.pad_action_dim = null
repack.pad_state_dim = null
field.use_norm_mask = false
~~~

WBT 为 30 FPS 时，30 帧对应约 1 秒预测窗口。若要严格复现 HE action expert 的 16 帧配置，可以将 `action_chunk_size` 和 `action_exec_horizon` 都设为 16；两者必须在数据、模型和部署配置中保持一致。

不要用 `pad_action_dim` 或 `pad_state_dim` 把 48D 静默补成其他维度。若启用 `ActionStateTransform.use_norm_mask`，必须提供长度为 48 的 mask；不能沿用当前面向 7D 动作的默认列表。

## 训练脚本

拟新增：

~~~
scripts/train/psi0/posttrain-wbt-native-psi0.sh
~~~

它与 posttrain-he-psi0.sh 保持同样的启动方式，主要参数改为：

~~~
config:
    posttrain_wbt_psi0_config

data root:
    /hfm/data/WBT

statistics:
    assets/stats/wbt_native_stats.json

repack:
    WBTNativePosttrainRepackTransform

action_dim:
    48

state_dim:
    48

use_delta_actions:
    false
~~~

HE 相关脚本和配置保留不动，用于回归对比。

## 统计量和归一化

拟生成：

~~~
assets/stats/wbt_native_stats.json
~~~

统计文件至少包含：

~~~
states: 48D
action: 48D
~~~

建议第一版使用：

~~~
action_norm_type = bounds_q99
normalize_state = true
~~~

不能复用 HE 的 he_raw_rel_stats_combined_no_static.json。WBT 的 root、body 和手部动作范围不同，且 WBT-native 使用绝对目标。

统计量只能使用训练 episode 计算，不能包含验证 episode。统计脚本应在 source hand canonicalization 之后计算，并同时输出各动作组的范围：

~~~
root
body
left hand source
right hand source
~~~

统计文件还应保存 `protocol`、`hand_protocol`、`hand_layout_ids`、单位和版本号。训练和验证必须加载同一份训练统计量，不能按 episode 临时估计；发现范围异常时先检查 permutation、单位和方向，再决定是否剔除数据。

## trainer 和模型的修改边界

### 必须修改

- 新增 WBT 数据读取器；
- 新增 WBT 数据配置；
- 新增 WBT repack transform；
- 新增 posttraining 配置；
- 新增 WBT posttraining 启动脚本；
- 生成 WBT 独立 statistics；
- 增加数据 shape 和协议检查。

### 第一版不需要修改

- src/psi/trainers/posttrain.py 的主要训练逻辑；
- src/psi/models/psi0.py 的 action header 结构；
- HE transform 和 HE dataset；
- FAST tokenizer；
- VLM 的 GLOBAL 实现。

PosttrainTrainer 从配置读取 action dimension 和 action horizon，因此 48D action 可以通过新 transform 和新配置进入同一 trainer。当前 posttraining 会冻结 VLM，action header 从头训练。WBT checkpoint 的输入输出投影只对应 WBT-native 48D 协议，不能假定能直接复用于其他动作协议。

### 可选修改

后续可以给 Psi0ModelConfig 增加显式的 action loss 权重，让 root、body 和双手分别加权。第一版建议先使用当前均匀维度权重，保证与 HE baseline 的比较简单。

## 前置 VLM checkpoint

posttraining 的 model.model_name_or_path 必须指向已经加入 <GLOBAL> 的 VLM checkpoint：

~~~
EgoDex + GLOBAL checkpoint
~~~

checkpoint 必须同时包含与之匹配的 processor、tokenizer 和 `<GLOBAL>` 配置。posttraining 只加载这个 VLM backbone，不继续更新其参数；WBT action header 从头初始化。旧的 HE 36D action header 不能直接加载到 48D WBT 配置，除非另行实现显式的权重迁移。

本文输出的 checkpoint 只保证 WBT-native-v1 的 48D state/action 接口；目标机器人的手部映射和部署控制由其他阶段处理。

## 验证计划

### 数据验证

- 随机读取至少 100 个 frame，检查图像和 state 时间戳对齐。
- 检查 states.shape[-1] == 48。
- 检查 actions.shape[-1] == 48。
- 检查所有数值为有限值。
- 检查 root quaternion 范数和顺序。
- 检查 `hand_layout_id`、双手 source canonical 顺序、单位和方向。
- 检查 source hand 的范围、无效值和缺失值处理。
- 检查 episode 末尾不会拼接到下一个 episode。
- 检查 train/val episode 没有交叉。

### 运动学验证

- 将 WBT target action 回放到 G1 运动学模型。
- 检查 body joint 顺序是否正确。
- 检查 root target 是否出现跳变。
- 在 source hand 语义层检查每个手指控制量与原始轨迹的单调性和范围。
- 在不涉及目标手型映射的仿真或运动学模型中回放至少一个完整 episode；目标手型 retarget 不属于本阶段。

### 训练验证

- 用单 batch 做 overfit，确认 action loss 能下降。
- 检查 root、body、左右 source hand 的分组 loss。
- 做 episode-level open-loop 评估。
- 对比 HE action expert 的训练曲线和动作误差。
- 先在仿真闭环，再进入真机测试。

## 实施顺序

1. 确认 WBT root pose 的坐标系和 quaternion 顺序。
2. 确认各 WBT 数据集的手型和手部维度。
3. 为每个来源冻结 `hand_layout_id` 到 6D source canonical 顺序的映射。
4. 实现 wbt_dataset.py 和最小样本读取。
5. 实现 source hand canonicalization 和 WBTNativePosttrainRepackTransform。
6. 生成并检查 wbt_native_stats.json。
7. 实现 posttraining config 和启动脚本。
8. 用单 batch 做接口验证，并确认 state/action 维度均为 48。
9. 进行 WBT-native action expert 训练。
10. 保存协议 manifest、统计量和训练元数据，供后续阶段按版本读取。

## 预期产物

本阶段完成后应得到：

~~~
WBT-native action expert checkpoint
protocol = wbt-native-v1
state_dim = 48
action_dim = 48
action_horizon = 30
absolute_action = true
hand_protocol = wbt-source-12d-v1
hand_layout_ids = recorded-per-dataset
~~~

该 checkpoint 代表模型已经学习 WBT 的原生全身动作空间。它的输入输出必须按 `wbt-native-v1` 和对应的训练统计量解释；目标机器人手部映射不属于这个 checkpoint 的 posttraining 产物。
