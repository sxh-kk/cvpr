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

训练目标归档：`/gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v2-hm-quality`，独立验收后可用；
原 `smplh-parameters-v1`、旧统计和已有checkpoint保留追溯。
两来源共9,263 episodes／38,137,884原生帧，原分割在 `meta/splits.json`。
保留47段HM隔离记录，reader按training eligibility及窗口mask筛选。
旧dense缓存和旧视觉文件已删除；reader只支持新的共享profile视觉缓存。
SMPL-H参数、双手XYZ、相机、模型与图像join保留；本次质量mask、地面及物理fingerprint按修订更新。

新完整train运动统计位于 `/gaozt-test1/datasets/MixEgoMotion/statistics/smplh_k12_5hz_joint_hm-quality-v2.pt`。
图像尺寸变化本身不需要重算运动统计；本次监督mask与地面改变，因此重算全train并拒绝复用旧统计。
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

归档、缓存与新统计全部验收完成后，从仓库根目录执行：

```bash
cd /root/workspace/wbc/guanzerong/cvpr
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/motionexpert-torch260-cu118/bin/python \
  -m MotionExpert.train \
  --config MotionExpert/configs/smplh_k12_psi0_pretrain.yaml \
  --from-scratch
```

Lightning按配置启动8卡DDP，无需另套torchrun。输出为
`/gaozt-test1/datasets/MixEgoMotion/training/smplh_k12_psi0_256x480_5hz_hm_quality_v2_from_scratch`。
本次复用已有DINO字节。旧MotionExpert checkpoint保留评估，新统计默认不支持直接续训旧checkpoint。
全量DINO使用BF16、每进程batch128、每卡2进程；可使用下面的守护入口自动转入正式训练。
这些是DINO提取参数，MotionExpert每卡训练batch仍为256。DINO命令见 [pipeline README](../pipeline/README.md)。
续训默认沿用checkpoint学习率；如设置 `TRAIN.USE_CKPT_LR: false`，新学习率会在当前调度阶段生效，
并同步更新scheduler基准，保留已完成的epoch与step。

### DINO完成后自动启动八卡训练

此入口用于重新提取DINO的新任务；本次HM质量修订直接复用既有特征，不调用该训练守护。

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

### 固定窗口 mmap 缓存与已有训练切换

`DATA.WINDOW_CACHE_ROOT` 默认为空，继续使用原参数读取器。设置后，普通训练／验证批次读取
预先计算的5Hz、11帧窗口，省去逐样本Parquet解码、SMPL-H FK、坐标变换和脚接触计算。
缓存保留原窗口顺序、筛选群体与有效mask；归一化仍使用原完整train统计，
噪声、任务条件、轨迹dropout和采样随机性仍在训练时生成。
稀疏采样评测的 `get_sample()` 保留原归档读取与完整世界坐标解码信息。

每个窗口保存原始float32的 `[11,245]` 运动、`[11,18]` 轨迹及对应bool mask，
以及首帧规范化变换、逐帧地面变换、地面兜底标记和历史6帧图像有效性。
历史图像索引由原生起点加 `[0,6,12,18,24,30]` 得到；未来图像仍隐藏。
按episode生成可恢复中间文件，最后按train／val／test压实为每split 12个大型 `.npy`，
训练worker持有这些只读mmap。DINO继续读取原native30Hz `features.npy`，不复制特征。
主SMPL-H归档、DINO和checkpoint都保留。

每个episode的12个数组先在RAM生成，再用普通NPY写入，逐字段读回核对生成值、shape和dtype。
打包前逐episode对照冻结的split索引，同时累计各字段的源payload摘要；
打包写盘后流式读回所有字段，核对源payload摘要及完整起点数组。
文件SHA一致不能替代生成值和窗口身份检查；任何一处不符都不会发布该阶段的完整缓存。
验收失败时原训练继续，切换守护结束；定点恢复后须重新通过缓存验收，并启动新的切换守护。
恢复记录和失败文件备份保存在仓库外的job目录，不能直接把验收状态改为通过。

从当前运行的配置快照复制一份配置，**只增加**：

```yaml
DATA:
  WINDOW_CACHE_ROOT: /gaozt-test1/datasets/MixEgoMotion/training_cache/smplh245_5hz_w11_v1
```

下面的 `SOURCE_CONFIG` 指当前运行的完整配置，`CACHE_CONFIG` 指只增加缓存路径的副本；
二者使用绝对路径。当前任务的副本和验收记录保存在仓库外的 `.pipeline/jobs/window-cache-<时间>/`。

```bash
cd /root/workspace/wbc/guanzerong/cvpr
MOTION_PY=/gaozt-test1/datasets/MixEgoMotion/.pipeline/runtimes/motionexpert-torch260-cu118/bin/python
CACHE_ROOT=/gaozt-test1/datasets/MixEgoMotion/training_cache/smplh245_5hz_w11_v1

# 当前训练继续运行；最多8个CPU构建进程，各自限制为单线程。
# 使用外部控制文件限制同时处理的episode数，先从4开始。
WORKER_CONTROL=/gaozt-test1/datasets/MixEgoMotion/.pipeline/window-cache-workers.json
"$MOTION_PY" -c 'import sys; from MotionExpert.dataset.window_cache import atomic_json; atomic_json(sys.argv[1], {"active_workers": 4})' "$WORKER_CONTROL"
"$MOTION_PY" -u -m MotionExpert.build_window_cache \
  --config "$CACHE_CONFIG" --output-root "$CACHE_ROOT" \
  --workers 8 --worker-control "$WORKER_CONTROL"

# 仅核对新增缓存路径：双来源、各split和地面策略的代表窗口逐值比较，
# 并在真实模型上比较原／缓存输入的BF16损失、有限梯度。
"$MOTION_PY" -u -m MotionExpert.validate_window_cache \
  --config "$CACHE_CONFIG" --source-config "$SOURCE_CONFIG" \
  --report "$ACCEPTANCE_REPORT" --gpu-check --device cuda:0

# 全量缓存验收通过后，等待最近可用的自然20轮保存点，再完整续训。
"$MOTION_PY" -m MotionExpert.switch_to_window_cache \
  --cache-root "$CACHE_ROOT" --acceptance "$ACCEPTANCE_REPORT" \
  --resume-config "$CACHE_CONFIG" --background
```

构建器再次执行同一命令可验证并复用已提交episode；未完成临时目录会重建。
构建worker复用父进程已验证的Archive快照，序列化时清空Arrow和DINO文件缓存，
避免每个worker重复扫描全量元数据。重启仍需父进程来源检查和原窗口规划。
已提交窗口只做文件身份与摘要检查，不重复FK或标签生成。
`--worker-control` 可在运行中调整并行度；比如提高到8：

```bash
"$MOTION_PY" -c 'import sys; from MotionExpert.dataset.window_cache import atomic_json; atomic_json(sys.argv[1], {"active_workers": 8})' "$WORKER_CONTROL"
```

降低并行度时，已有任务自然完成后再收缩。控制值必须为1至 `--workers` 的整数；
不提供控制文件时，保持原有有界任务队列。多worker的实际收益受共享存储限制，
应比较未复用episode的稳定窗口吞吐，而不是把已完成shard的跳过速度当作生成速度。
本次生产任务已按8-worker配置恢复，实际控制文件路径记录在 `.pipeline/current_window_cache_job.json`
的 `worker_control` 中。初始化和重新规划期间，旧 `progress.json` 会保留最后完成计数；
进入恢复检查后该计数可能暂时从本次已处理任务开始累计，已提交的episode文件仍完整保留。
全量生成及三个split打包结束后才发布 `manifest.json` 的 `complete/all`。
`--diagnostic-episodes` 只生成诊断子集，不能用于正式训练或自动切换。
验收支持 `--wait --builder-pid PID`，可与构建并行等待；构建提前退出会报告失败。
恢复指定episode后，可给验收命令增加 `--required-episode-indices 65 7150`，
在保留全量scope和原代表窗口的基础上，额外核对指定episode的首、中、末窗口；
这与仅生成诊断子集的 `--episode-indices` 不同。
验收报告逐窗口更新 `comparisons` 与 `current_sample`；失败时保留 `failed_sample`，
张量差异包含字段、episode和原生起始帧，便于直接定位。
正式验收只为首个split构造一次来源Archive，后续split复用该对象；
复用前核对根目录、发布模式、模型、读取缓存设置及视觉身份，并完整执行 `archive.assert_current()`。
每个split仍独立验证自己的WindowCache、记录、窗口索引、receipt和统计；普通训练默认创建Archive的路径不变。
配置 `WINDOW_CACHE_ROOT` 后，训练DataModule也在每个训练进程内复用首个split已验证的Archive，
覆盖fit及先validate后fit的启动顺序；DataLoader worker继续按原规则建立自己的文件句柄。
已有数据物理数值验收不会重做。短窗口读取测速只代表I/O检查，不代表八卡整体吞吐。

缓存构建和验收期间，原训练持续运行。完整缓存独立验收通过后，
守护等待最近可用的自然20轮保存点，只通过 `last.ckpt` 链接、文件名与stat确认保存完成，
不预读checkpoint内容。随后停止已确认身份的原训练进程组，等待旧DINO守护完成退出，
以 `--resume` 从该保存点启动一次八卡训练。
原训练先收到TERM并最多等待180秒；超时后仅对冻结时已记录、身份与session仍匹配的成员，
通过pidfd发送KILL，再最多等待30秒。Python或系统不支持安全pidfd接口时拒绝强制退出，
明确标记关闭未完成、需要人工清理，不会启动另一份训练，也不会声称原八卡训练仍完整运行。
例如验收通过时训练已经进入第41轮，就等60轮保存；若40轮刚保存、尚未进入下一轮，
可以使用40轮。选定保存点后允许重做刚进入下一轮的少量未保存计算，
不会回退已完成的整轮；错过最近保存时间则等待下一个自然保存点。
保留AdamW、EMA、scheduler、epoch和global_step；batch256、24workers、LR1e-4、BF16、
200轮及每20轮保存均保持原值，不额外保存checkpoint。
保存边界后的少量未保存计算可能需要重做；不会丢失自然checkpoint中的已完成轮次。
失败不自动重试。验收失败、缓存未完成或交接前取消守护，原训练继续运行。
交接后启动失败不会自动重试。训练入口继续执行原有必要的续训兼容检查。

监控文件：

- 缓存构建：`training_cache/smplh245_5hz_w11_v1/progress.json`；初始化与索引阶段查看构建日志。
- 构建job：`.pipeline/current_window_cache_job.json`。
- 切换守护：`.pipeline/current_cache_transition.json`，内含 `daemon_log`、`log` 和切换状态；
  `waiting_cache` 表示原训练继续、等待缓存验收；`waiting_checkpoint` 表示验收通过、等待保存点。
- 训练：`training/current_run.json`；切换后更新为新进程身份及新训练日志。

上述相对路径均位于 `/gaozt-test1/datasets/MixEgoMotion/`。停止切换守护不会停止已有或已续训的模型进程。

## HM质量修订与统计

HM原生 `valid` 主要反映原始拟合是否有效输出，仍可能包含人体偏离头戴相机数米的帧。
质量审计使用发布的OpenPose67拟合鼻点与已注册相机世界位置比较；
`observation.state.head_world` 是Aria CPF，不能代替拟合人体头部用于此项检查。
默认硬筛鼻点与相机距离大于1米的帧，以及相对位置突变大于 `0.25＋3Δt` 米的两端；
0.5米仅作警戒。缺少相机证据单独记录，不将其当作质量已通过的证据，也不直接拒绝有限拟合。
相机自身速度尾部独立审计，不与身体mask混合；保留的拟合不等于已验证的真实人体标签。

`audit_hm_quality` 只写仓库外sidecar，默认覆盖全部可训练HM，保留原47段隔离记录。
生产修订目标为 `smplh-parameters-v2-hm-quality`，先发布candidate，独立验收后才允许训练。
只修改质量相关监督mask、confidence和地面：SMPL-H参数、双手XYZ、相机与视觉帧join保留原值。
被拒绝帧同时失去身体、拟合双手与脚点监督；地面由筛选后的脚点重估，
失败时继续使用既有历史6帧脚点兜底。地面改变时整个episode的窗口重建。
新DINO目录以硬链接复用原特征与join，并重新绑定修订归档，不重新提取。

新缓存为 `training_cache/smplh245_5hz_w11_hm-quality-v2`，
新统计为 `statistics/smplh_k12_5hz_joint_hm-quality-v2.pt`，均位于MixEgoMotion根目录。
旧归档、缓存、统计和checkpoint保持保留。`reuse_window_cache` 仅复用完整record不变、
视觉字节相同且计算实现不变的已验收episode数组；它不发布缓存、不继承旧验收结果。
受影响episode重新构建，三个split重新打包与核验。

处理入口依次为 `audit_hm_quality`、`revise_hm_archive`、`validate_hm_revision --publish`、
`reuse_window_cache` 和 `build_window_cache`。审计和构建始终写新目录；
`validate_hm_revision` 默认只生成独立报告，加 `--publish` 才发布归档。
不可把candidate状态或部分诊断报告改名为已验收数据。

新归档需要新统计，因此缓存验收分为两步：

1. `validate_window_cache --raw-only` 对未归一化窗口与来源读取逐值比较，生成专用raw报告。
   不读取DINO嵌入、不依赖尚未生成的统计；此报告不能用于训练切换。
2. `compute_cached_motion_stats` 从全部train mmap窗口扫描245维运动与18维轨迹及mask，
   保留11帧及重叠窗口权重，参考增量共享统计、脚接触均值0标准差1。
   新统计完成后，执行原归一化窗口与GPU loss/梯度验收。

```bash
"$MOTION_PY" -m MotionExpert.compute_cached_motion_stats \
  --config "$REVISION_CONFIG" \
  --archive-root /gaozt-test1/datasets/MixEgoMotion/smplh-parameters-v2-hm-quality \
  --cache-root /gaozt-test1/datasets/MixEgoMotion/training_cache/smplh245_5hz_w11_hm-quality-v2 \
  --acceptance "$RAW_CACHE_REPORT" \
  --output /gaozt-test1/datasets/MixEgoMotion/statistics/smplh_k12_5hz_joint_hm-quality-v2.pt
```

`MOTION_PY`、`REVISION_CONFIG` 和 `RAW_CACHE_REPORT` 应指向本次修订的运行环境、配置和raw验收报告。
统计全扫拒绝任何NaN／Inf，包含mask无效通道，并定位到episode、窗口、帧和维度。
新统计输出存在时拒绝覆盖。统计修订改变archive fingerprint与数值，旧checkpoint默认不能直接
`--resume` 或 `--init-weights`；不能通过更改标签或跳过检查来宣称兼容。后续训练策略需单独确定。

只停止到自然100轮且不续训，可使用一次性 `stop_at_epoch --completed-epochs 100 --background`。
它只核对 `last.ckpt` 链接、文件名和stat，拒绝错过100轮后改停120轮，不额外保存。
生产运行应使用支持 `os.pidfd_open` 与 `signal.pidfd_send_signal` 的Python；
当前训练runtime缺少这两个接口，使用系统 `/usr/bin/python3` 并通过 `PYTHONPATH` 加载runtime依赖。
守护状态位于 `.pipeline/current_training_stop.json`；质量修订job位于
`.pipeline/current_hm_quality_revision.json`。这两个流程都不会自动启动新训练。

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
