# MotionExpert

当前模型读取统一SMPL-H参数归档，在5Hz上用历史5帧＋当前帧图像预测未来5帧运动。
当前图像与Ψ₀当前图像相同。后训练可取单次 `t=1` 网络调用的hidden作为运动先验，并反传更新MotionExpert。
下游模型不在此模块内。

## 时间、图像与条件

11个时刻为 `[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]` 秒。
原始运动和离线DINO均为30Hz，MotionExpert按零相位步长6采样。
输入DINO为 `[B,6,5,1024]`，CLS＋4 register；未来5帧图像不作为条件。
模型联合监督全部11帧运动，不输入干净历史人体姿态。

| 阶段 | 最终RGB H×W | 增强 |
| --- | --- | --- |
| 预训练 | 256×480 | 确定性共享几何 |
| 后训练 | 256×320 | 窗口共享颜色增强 |
| Robust | 256×480 | 窗口共享颜色＋随机裁切 |

共享处理位于 [ImageProcess](../ImageProcess/README.md)。两路使用同一处理后的当前RGB，各编码器各自归一化。
在线增强需在线DINO编码；固定离线特征只用于当前确定性预训练。
真机建议先攒满约1秒历史后启用运动先验；当前接口要求6个真实时刻，不进行首帧复制补历史。
历史缓冲的时间对齐、就绪判定与调度由调用方负责。

训练保留历史相机轨迹模态dropout：每样本概率0.5开放前6帧有效18维轨迹。
验证、测试和 `forward_features()` 隐藏真实相机轨迹；未来图像和轨迹始终隐藏。
身体和手部真值始终只作为监督。

## 运动表示

读同一自有SMPL-H模型及逐帧16维beta做FK，再选既有K12：
`[0,4,5,10,11,13,14,15,18,19,20,21]`，对应pelvis、左右knee、左右foot、
左右collar、head、左右elbow、左右wrist。双手使用左右各21点显式XYZ监督。
表示版本 `smplh12_rot6d_hands21_contacts_v1`，身体轴为 `smplh_body22_axes_v1`。

| 通道切片 | 内容 | 维数 |
| --- | --- | ---: |
| `[0:108]` | 身体12关节，逐关节旋转6D＋XYZ | 108 |
| `[108:117]` | 相机地面参考系旋转6D＋平移3D增量 | 9 |
| `[117:180]` | 左手21点XYZ | 63 |
| `[180:243]` | 右手21点XYZ | 63 |
| `[243:245]` | 双脚接触代理 | 2 |
| 总计 | 模型运动输入／输出 | **245** |

另有18维相机轨迹：相机相对地面参考位姿9维＋参考运动增量9维。
重复的参考增量共享归一化统计。6D取旋转矩阵前两行。
世界坐标米、Z-up；窗口首帧相机的水平位置／朝向规范化，然后逐帧转到相机地面参考系。
相邻增量在上一帧参考系中表达，首帧保留初始参考约定。
输出包括K12位置／旋转、双手点、参考运动和接触，不预测完整SMPL-H pose或shape。

身体6D、XYZ、双手、增量和接触全部保留独立有效mask与损失。
标签mask只约束监督；缺失标签的运动通道在训练中保留独立高斯噪声。
`t=1` 的全部245维输入与部署的一步hidden入口一致，不由未来标签可用性清零噪声。
EgoSuite身体为拟合估计，HM为发布的优化参数；来源观测、插值、完成和confidence仍留在数据中。
有效地面优先使用归档值；缺失时只用历史6帧原始有效脚点高度均值减2cm兜底。
参考仍无效的窗口不进入训练。接触在5Hz重算，阈值0.2m/s及0.1m，属于代理标签。

## 当前数据与训练

归档：`/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v1`。
两来源共9,263 episodes／38,137,884原生帧，原分割在 `meta/splits.json`。
保留47段HM隔离记录，reader按training eligibility及窗口mask筛选。
旧dense缓存和旧视觉文件已删除；reader只支持新的共享profile视觉缓存。
SMPL-H、双手、相机、地面、mask、模型和物理fingerprint保持原样。

完整train运动统计位于 `/gaozt-test1/datasets/MixEgoMotion/statistics/smplh_k12_5hz_joint.pt`，
原样迁移，无需因新图像尺寸重新计算。新缓存必须保持原mask和训练群体，否则拒绝复用统计。
缓存manifest必须complete/all且与归档、分割、30Hz帧join和profile一致。
统计绑定物理表示，checkpoint另外绑定视觉版本。
归档推理和评测默认要求共享图像profile及DINO权重／归一化／token顺序与checkpoint一致。
显式添加 `--allow-visual-adaptation` 可允许共享RGB尺寸profile变化，DINO身份仍必须匹配；
输出会记录训练视觉版本、实际缓存版本及适配声明。在线后训练的 `forward_features()` 支持两种尺寸。

唯一正式预训练配置：[smplh_k12_psi0_pretrain.yaml](configs/smplh_k12_psi0_pretrain.yaml)。

| 参数 | 设置 |
| --- | --- |
| GPU／batch／workers | 8卡，每卡256，全局2048；每卡24workers |
| 优化器 | AdamW，LR1e-4，weight decay0.01，梯度范数裁剪1.0 |
| 精度 | BF16 mixed |
| 周期 | 200 epochs；20 epochs warmup，然后cosine降至1e-5 |
| 保存／验证 | 每20 epochs保存，结束保存final；每5 epochs验证 |
| 模型 | hidden768、12层、1 shared＋11 routed experts、Top-2、一个GLOBAL token |
| 生成 | x0 Flow Matching，10步Euler采样；EMA0.992028 |
| 提速开关 | fused AdamW和批量指标同步均保持关闭 |

新DINO完整提取后，从仓库根目录执行：

```bash
cd /root/workspace/wbc/guanzerong/cvpr
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/motionexpert-torch260-cu118/bin/python \
  -m MotionExpert.train \
  --config MotionExpert/configs/smplh_k12_psi0_pretrain.yaml \
  --from-scratch
```

Lightning按配置启动8卡DDP，无需另套torchrun。输出为
`/gaozt-test1/datasets/MixEgoMotion/training/smplh_k12_psi0_256x480_5hz_from_scratch`。
不使用旧MotionExpert checkpoint；保留的DINO预训练权重仍用于特征提取。
全量DINO使用BF16、每进程batch128、每卡2进程；可使用下面的守护入口自动转入正式训练。
这些是DINO提取参数，MotionExpert每卡训练batch仍为256。DINO命令见 [pipeline README](../pipeline/README.md)。
续训默认沿用checkpoint学习率；如设置 `TRAIN.USE_CKPT_LR: false`，新学习率会在当前调度阶段生效，
并同步更新scheduler基准，保留已完成的epoch与step。

### DINO完成后自动启动八卡训练

从仓库根目录启动一次守护进程；`--background` 返回后关闭终端不影响守护：

```bash
cd /root/workspace/wbc/guanzerong/cvpr
/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/motionexpert-torch260-cu118/bin/python \
  -m MotionExpert.watch_dino \
  --config MotionExpert/configs/smplh_k12_psi0_pretrain.yaml \
  --gpus 0 1 2 3 4 5 6 7 \
  --background
```

默认每60秒读取DINO manifest；仅在 `complete/all`、全episode／帧数完成、源mask保持一致、
提取PID身份已退出且提取锁释放后，检查视觉profile、数据路径、全量统计及8张可见GPU。
随后使用保存到外部job目录的完整配置快照执行 `MotionExpert.train --from-scratch`，
训练入口继续检查逐episode receipt、split和统计兼容性，然后创建模型。
配置中的batch、workers、学习率、精度、200轮和保存间隔保持原样，不另套torchrun。

状态指针：`/gaozt-test1/datasets/MixEgoMotion/.pipeline/current_dino_to_motion.json`。
其中 `watcher.pid` 为守护PID，`pid` 在训练启动后为训练主PID；
`daemon_log` 记录等待／交接，`log` 是训练日志，`output` 是checkpoint目录。
完整状态、日志和快照保存在 `.pipeline/jobs/dino-to-motion-<时间>/`，
`training/current_run.json` 同步记录本次流程状态。

```bash
cat /gaozt-test1/datasets/MixEgoMotion/.pipeline/current_dino_to_motion.json
JOB_DIR=$(python -c 'import json; print(json.load(open("/gaozt-test1/datasets/MixEgoMotion/.pipeline/current_dino_to_motion.json"))["job_dir"])')
tail -F "$JOB_DIR/daemon.log" "$JOB_DIR/training.log"
```

训练日志在交接时创建，`tail -F` 会持续等待该文件。
DINO失败、提前退出、缓存不完整或启动检查失败时，守护报告错误并退出，不启动训练。
全局锁和输出目录的持久启动标记防止重复启动；已有非空训练目录会拒绝被覆盖。
训练失败不会自动重试或恢复。`SIGTERM`／`SIGINT` 只停止守护：等待时不影响DINO；
交接后已启动的训练保持运行，并继续持有运行锁和缓存读锁。
停止后再次等待可使用新的job目录；已启动过的训练需按实际checkpoint选择续训，不能重复从头覆盖。

## 一步hidden与在线共享图像

`forward_features()` 保留autograd，默认在高斯噪声 `t=1` 上执行一次条件模型调用：

- `one_step_hidden [B,11,768]`：全部11个运动时刻的hidden。
- `one_step_global [B,768]`：GLOBAL hidden。

该入口不进行完整10步采样，不接收干净人体目标，未来条件隐藏。
默认不启用CFG；显式提供 `cond_scale` 会增加无条件网络调用。
调用方可固定seed或提供噪声以控制重复推理中的随机性。

```python
import json
import torch
from MotionExpert import MotionExpert
from egomotion_pipeline.features import DinoEncoder

expert = MotionExpert(weights="new-run/checkpoint.ckpt", device="cuda",
                      dtype=torch.float32, use_ema=False).train()
encoder = DinoEncoder(json.load(open("pipeline/configs/dino.local.json")))

# history_rgb为5Hz六个真实时刻，最后一帧就是Ψ₀当前帧。
window = encoder.encode_window(history_rgb, stage="posttrain", seed=123)
# Robust改为stage="robust"；DINO仍使用同一份冻结预训练权重。
img_embs = torch.from_numpy(window["img_embs"])[None].to("cuda")
features = expert.forward_features({"img_embs": img_embs}, seed=0)
# Ψ₀使用window["rgb"][-1]；下游loss经features反传更新运动网络。
```

MotionExpert是wrapper；外部nn.Module／DDP应注册 `expert.model` 为子模块。
只用hidden时部分输出head没有梯度，外部DDP需处理未使用参数，如 `find_unused_parameters=True`。
`train()/eval()`及`parameters()`控制同一运动网络。
冻结运动模型时可用无梯度 `extract_features()`；完整运动预测用 `predict_future()`，
返回未来 `[B,5,245]` 与完整 `[B,11,245]`。坐标解码得到身体 `[B,5,12,3]` 和双手各 `[B,5,21,3]`。

## 当前检查入口

保留当前时间／mask／SMPL-H坐标解码、训练噪声、采样验证、续训调度、视觉版本及一步hidden梯度的关键回归：

```bash
python -m pytest MotionExpert/tests -q
```

已有运动数值验收不重复执行。新特征完整提取后，训练启动前的读取检查负责验证视觉版本和统计兼容性。
