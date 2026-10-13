# MotionExpert

默认适配独立 SMPL-H 参数归档 `/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1/`，**保留原有12稀疏身体关节及245维模型输入／输出**。读取时用归档自己的同一 SMPL-H 模型和实际16维beta计算完整身体FK，再选择既有K12；左右各21点使用显式手部标签。MotionExpert 独立读取参数与视觉 Parquet，不依赖 pipeline Python 包或旧 dense22 文件，也不修改归档。

时间协议保持 **5Hz，历史1秒＋当前帧的6张图像 → 联合重建历史6帧、预测未来1秒的5帧运动**。11个时刻为 `[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]`，从原30Hz按零相位步长6选帧。真实身体和相机用于监督，训练中一半样本还开放历史相机轨迹；真实shape只在数据端FK与审计中使用，不作为模型条件或预测目标。部署只需要历史图像的 DINO 特征。

## 表示与原 UniEgoMotion 的对应

参考官方 [v4_beta 编码／解码](https://github.com/chaitanya100100/UniEgoMotion/blob/580c92c6d70a91672c4106bab30ca82cdd80f379/dataset/representation_utils.py) 的旋转6D、相机地面参考和联合生成方式。原版包含22身体关节、手PCA与shape；当前模型沿用项目既有K12稀疏身体表示和显式手21点，不增加完整22关节或shape预测分支。

| 输出 | 通道切片（左闭右开） | 维数 |
|---|---|---:|
| 身体12稀疏关节，逐关节前两行6D＋XYZ | `[0:108]` | 108 |
| 相机地面参考系的旋转6D＋平移3D增量 | `[108:117]` | 9 |
| 左手21点XYZ | `[117:180]` | 63 |
| 右手21点XYZ | `[180:243]` | 63 |
| 双脚接触代理 | `[243:245]` | 2 |
| 总计 | | **245** |

身体索引保持 `[0,4,5,10,11,13,14,15,18,19,20,21]`：pelvis、左右knee、左右foot、左右collar、head、左右elbow、左右wrist。表示版本为 `smplh12_rot6d_hands21_contacts_v1`，与旧原生轴dense12版本明确区分。`SPARSE_JOINTS.ENABLED=true`、`MODEL.INPUT_FEATS=245`；配置拒绝切换成全22或351维。HM逐帧beta原值用于FK并保留在misc，不复制首帧或强制平均；没有beta预测头或beta损失。

世界坐标为米、Z-up。窗口先将首帧实际相机的水平位置和朝向规范化，再以每帧相机的地面投影建立参考系；身体全局旋转与手点变换到此参考系。旋转编码取矩阵前两行。相邻参考增量在上一帧参考系中表达，首帧保存初始参考位姿；轨迹标签仍为18维（相机相对地面位姿9＋参考增量9），与运动重复增量共享统计。

解码返回12个身体关节XYZ／全局旋转、双手21点与地面参考轨迹。稀疏12关节无法直接恢复完整SMPL-H局部pose、beta或trans，因此接口不输出这些未预测参数；不把缺失关节补成真实姿态。显式手点不需要手指PCA，也不转换为完整finger pose。身体XYZ和旋转保持独立预测，显式手腕与身体模型手腕的偏差在评估中单独报告。

## 数据准入、mask与来源

新归档目标为 EgoSuite 6,614段／26,841,893帧与 EgoExo4D-HM 2,649段／11,295,991帧。当前记录中9,216段满足 `training_eligible`，47条HM隔离记录保留但不进入训练。正式读取要求 `.pipeline/dataset-state.json` 为 `state=accepted`、`independent_acceptance_pending=false`。这两个条件满足前，转换完成不能视为正式训练数据已发布。

读取使用归档原有 `meta/splits.json`，不重新随机切分：EgoSuite沿用人物／录制分组，HM沿用participant／capture连通分量。reader核对全部record、receipt、文件尺寸与时间身份、全局frame/index、split成员及分组；fingerprint绑定参数schema、模型SHA、全部record与归档索引。已有外置全量验收报告时继续核对其结果和binding；归档搬迁不要求保留旧数据或外置job目录。窗口读取再次检查文件稳定性，不在每个batch重算数百GB哈希。

两来源共用归档的52关节、6890顶点、真实16维beta模型，SHA256为 `be42bff1ff0589cf0467893bcfbbeb3f9b533ba151031c8cdfb64ca1dcb34947`。旋转策略为 `smplh_model_defined / smplh_body22_axes_v1`：模型定义目标轴，原生轴认证和绝对旋转精度认证标记继续保留。EgoSuite身体是拟合估计；HM来自发布的优化参数，数值可用不等于捕获真值。

身体监督使用 `estimated.valid`，原观测、插值、完成标签、confidence与残差留在sample的misc中。不同来源confidence的含义不同，本轮没有直接改为confidence加权损失。双手每点XYZ使用各自mask，独立于身体和可选手指PCA；EgoSuite未拟合finger pose仍可正常参与显式手部监督。HM无效参数被有限零值存储时，mask仍保持无效。

归档地面无效时，只用所选历史前6帧的 `source.feet_world/valid` 估计脚高度均值再减2cm；不使用未来脚点或拟合补全脚点兜底。仍无法确定参考地面的窗口拒绝进入身体训练。脚接触由来源脚点在5Hz上重新计算，使用0.2m/s速度与0.1m高度阈值，保留独立mask。地面和接触均为代理标签，不是独立测量真值。

DINO保持 `dinov2-with-registers-large` 的 `[5,1024]`（CLS＋4 register）。新reader只加载前6帧embedding，后5帧视觉值置零并隐藏；历史有效图像比例默认至少0.5。运动统计只读DINO有效性，不加载embedding。

## 模型、训练与采样

保留12层、hidden768、每层1个shared＋11个routed experts、Top-2路由、4帧chunk和一个GLOBAL token。输入／输出投影保持245维。使用x0 Flow Matching、10步Euler、EMA与MoE均衡损失。

显式 `TRAIN_TASK=fore / MOTION_SUPERVISION=joint`：全部11帧的有效K12身体、手、参考增量和接触参与损失。历史真值属于待生成目标；推理从全部11帧高斯噪声开始，历史和未来共同更新。SMPL-H默认 `TRAJECTORY_CONDITION_PROB=0.5`，训练逐样本独立抽签，决定是否开放前6帧有效的18维相机轨迹；这是长期期望比例，不保证每个batch恰好一半。图像始终只开放前6帧，不再额外随机隐藏整段视频。

| 阶段／样本类型 | 历史6帧DINO | 历史6帧真实相机轨迹 | 未来5帧图像／真实轨迹 |
|---|---|---|---|
| 训练，概率0.5 | 可见有效帧 | 可见有效通道 | 全部隐藏 |
| 训练，概率0.5 | 可见有效帧 | 全部隐藏 | 全部隐藏 |
| 验证、测试、部署 | 可见有效帧 | 全部隐藏 | 全部隐藏 |

两类训练样本都联合监督11帧，不输入干净历史人体姿态。未来参考增量仍是运动预测目标，但从不作为条件输入。独立缺失mask继续生效；开放轨迹不把无效通道变为有效。这里的0.5是模态dropout概率，与采样CFG权重及损失权重分别配置。

这样可以让共享网络在有／无轨迹时都学习，思路与 [ModDrop](https://arxiv.org/abs/1501.00102) 的缺失模态训练一致，但0.5只是起始实验值，没有证据表明最优。最相关的 [UniEgoMotion补充表3](https://arxiv.org/html/2508.01126v1#A3) 中，统一模型去掉轨迹的预测MPJPE为0.322m，单独仅图像训练为0.318m；其主要预测协议在推理时也提供历史轨迹。因此不能据有轨迹分支loss下降推断纯图像部署会改善，也不能将该论文的2–4秒指标直接当成我们的1秒K12结果。固定数据、步数和seed，对比概率0／0.25／0.5，以**仅图像验证的未来5帧**指标选择配置。

若混合训练没有改善纯图像效果，可另行研究“图像＋历史轨迹教师→仅图像学生”。[第一视角多模态蒸馏](https://arxiv.org/abs/2307.07483) 在动作识别中验证了训练额外模态、部署仅RGB的路径；迁移到本运动预测任务仍需设计和消融。当前实现没有教师、学生或蒸馏损失。

损失保留有效通道MSE与 `GLOBAL_WEIGHT=8`，增量权重仍作用于 `[108:117]`。分别报告身体6D／XYZ、左右手、增量旋转／平移、接触的有效计数和MSE。主验证只评价未来5帧：K12身体和手MPJPE、旋转角度、参考漂移、手腕偏差与脚部物理指标。world还原使用预测参考增量；真实最后观察参考位姿仅是坐标输出锚点，不是网络条件。纯图像部署无此锚点时，可输出相对参考坐标的运动。

## 使用

默认 SMPL-H 训练配置为8卡、每卡 batch64、梯度累积1，有效全局 batch512；峰值学习率 `1e-4`。其余参数保持：300 epoch、20 epoch warmup、EMA `0.992028`、每卡4个数据加载 worker、`FUSED_ADAMW=False`，采样仍为5Hz、历史6帧＋未来5帧。

```bash
python -m pip install -e ./MotionExpert
# 默认即此配置
python -m MotionExpert.compute_joint_motion_stats \
  --config MotionExpert/configs/smplh_k12_global.yaml \
  --output MotionExpert/exp/smplh_k12_fore_5hz_joint/stats.pt
python -m MotionExpert.check_data --config MotionExpert/configs/smplh_k12_global.yaml
python -m MotionExpert.train --config MotionExpert/configs/smplh_k12_global.yaml --from-scratch
```

必须在归档正式accepted后计算全部train窗口的SMPL-H K12 245＋18统计。统计逐通道排除无效值，contacts保持mean0/std1，重复增量严格共享统计。维度相同不代表轴和标签分布相同：旧原生轴245统计不能复用，已撤回的351统计也不兼容。`--limit-windows`产生的部分统计标为diagnostic，不可启动正式训练；代码内部 `allow_unpublished=True` 仅供只读诊断，没有正式训练配置开关。

全量统计可并行分片，每个 `I=0..15` 各运行一次（`III`为三位补零索引）：

```bash
python -m MotionExpert.compute_joint_motion_stats \
  --config MotionExpert/configs/smplh_k12_global.yaml \
  --num-shards 16 --shard-index I --shard-mode contiguous \
  --output MotionExpert/exp/smplh_k12_fore_5hz_joint/shards/stats.shard-III-of-016.pt
python -m MotionExpert.merge_joint_motion_stats \
  --input-dir MotionExpert/exp/smplh_k12_fore_5hz_joint/shards --stem stats --num-shards 16 \
  --output MotionExpert/exp/smplh_k12_fore_5hz_joint/stats.pt
```

分片合并检查表示、模型、split、采样协议、区间无重叠及全覆盖。Parquet按列及row group读取，worker缓存受字节上限控制。统计和产物写在归档之外。

```bash
python -m MotionExpert.evaluate \
  --checkpoint MotionExpert/exp/smplh_k12_fore_5hz_joint/last.ckpt \
  --split val --max-samples 16 --output MotionExpert/exp/smplh_k12_fore_5hz_joint/val_metrics.json
python -m MotionExpert.export_weights \
  MotionExpert/exp/smplh_k12_fore_5hz_joint/last.ckpt MotionExpert/weights/smplh_k12_global_ema_fp32.pt
```

推理API `expert.predict_future({"img_embs": img_embs})` 接受 `[B,6,5,1024]`，返回 `future_motion[_denormalized] [B,5,245]`；完整motion为 `[B,11,245]`，hidden为 `[B,11,768]`，GLOBAL hidden为 `[B,768]`。坐标解码输出 `[B,5,12,3]` 身体和左右各 `[B,5,21,3]` 手点；稀疏解码无需加载完整人体模型，数据端FK及checkpoint仍绑定模型SHA。

恢复训练使用 `--resume`；`--from-scratch` 与 `--init-weights` 为明确的新运行入口，三者互斥。原生轴245权重虽投影尺寸相同，表示和轴不同，当前严格入口仍拒绝误恢复／直接初始化；旧153／text和已撤回351权重不兼容。Checkpoint、统计和导出权重绑定表示、K12顺序、模型SHA与5Hz联合协议。单独改变轨迹开放概率不改变物理标签、窗口或统计，因此无需重新处理归档／重算同表示统计；checkpoint的训练协议另绑定该概率，概率不同的checkpoint不能直接严格恢复或初始化。

## 验证及旧实验

```bash
python -m pytest MotionExpert/tests -q
```

本次验证与限制记录在 [review.md](review.md)，训练验收计划见 [validation_plan.md](validation_plan.md)。小样本结构验证不等于已完成新模型训练，也不等于全量上游拟合已验收。

旧 `configs/dense_k12_global.yaml` 保留245维EgoSuite原生轴实验；其历史验证仍在 `exp/dense_k12_fore_5hz_joint/`。`motion_expert_config()` 为旧程序的明确兼容factory，`smplh_motion_expert_config()` 为SMPL-H K12配置factory，`load_config()` 默认选择SMPL-H K12。`exp/smplh22_fore_5hz_joint/` 是已撤回351方案的历史诊断，不能用于当前模型准入。所有结构诊断均不代表已完成全量训练。
