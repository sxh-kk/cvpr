# Handover：按失败阶段增强训练数据

本轮依据用户报告的26场失败改进数据：8场抬升不足、1场交接中断、17场交接后放置未通过。
问题主要发生在4–8秒且仍有22秒以上余量，没有确认过完整三维稳定放置后才退化的场景。
这些分型目前是用户提供的汇总；原始trace、权重身份及开发／最终测试划分尚未绑定。
因此不将26个失败episode直接加入TRAIN，也不声称此次调参仍有完全盲测的资格。

## 已确认的监督覆盖

原quality100有100条真实RTX示范、24个TRAIN动作祖先，32,794个完整历史／未来窗口。
所有示范已通过有序抬升交接、松手、容器承重、必要三维几何和稳定性准入。
粗分中交接后搬运／放置已占约53%，不能把模型放置失败简单归因为整段放置数据太少。
按真实50Hz物理观测的因果标签，稳定放置current也有4,026个；不是完全没有成功保持监督。

本轮保留原RGB、状态、动作、归一化和TRAIN／VAL划分，先增强关键转换附近的训练采样。
同时建立独立的小批量物理重采入口，补充分角色、分方向的新初态；新执行不等于策略恢复示范。

## 关键转换sidecar

`scripts/4090_simple/handover_critical_windows.py`只读原PACK_READY、admission、原始物理记录和事件。
它核对SHA、30/50Hz时钟、祖先身份，并复用生产完整历史／30帧真实未来准入，输出：

- `rows.jsonl`：以`(episode_index,current_row)`标识32,794个窗口，包含训练采样概率。
- `metadata.json`：输入和权重SHA、事件时间、转换范围、祖先分布及用户报告的8/1/17分型。

采样由50%全轨迹与50%关键转换混合；全轨迹按祖先均衡，各转换类内也按祖先均衡。
关键类预算按8/1/17分配给lift、handover、placement。该比例是本轮试验的显式选择，尚未证明最优。
重点是抬升起点、双手／左手接管、三维对齐、容器承重和松手转换，而不是整段放置或静止保持。
`stable_placement`保留为因果观察标签和验收信息，在修订v3中不再触发关键采样提权。

当前观察阶段只取当时及过去的物理测量；未来动作跨度涉及的阶段只用于离线TRAIN采样，
不进入模型输入，不用于部署决策，也不改变任何动作标签。原v2快照保留以追溯review修订，
新训练使用`critical-v3`及其`critical_focus_policy_revision=v3`。

生成命令：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  scripts/4090_simple/handover_critical_windows.py \
  --dataset-root /var/tmp/cvpr-evaluation-zxc-handover-quality100-20261022/packed-quality100-v1 \
  --output /home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024/critical-v3
```

## 新训练的生产索引对接

`handover_critical_sampler.py`严格通过生产`motion_history.original_indices`和
`source_episode_ranges`反解真实窗口身份，再映射到训练Subset；不能把Parquet行号当Dataset索引。
同一seed／epoch生成同一全局抽样流，只由原Accelerator分片，不能再叠加DistributedSampler。
它保留collator、batch、worker、prefetch及验证loader，拒绝对已经准备、运行或恢复中的任务安装。

在新trainer的`create_dataloaders`中，调用`super()`之后、`Accelerator.prepare`之前：

```python
from handover_critical_sampler import install_critical_sampler

receipt = install_critical_sampler(
    self,
    sidecar_dir,
    dataset_root=packed_train_root,
)
return self.train_dataloader, self.val_dataloader
```

父目录`/home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024`保存真实生产预检和上传收据。
A100独立目录为`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_improvement_20261024`。
跨主机只要求sidecar与实际packed元数据SHA匹配，不要求原4090的`/var/tmp`绝对路径在A100存在。
本轮没有修改既有训练的采样器、学习率或checkpoint，没有自动启动新训练。

## 独立补采与准入

`collect_handover_supplement.py`提供`prepare/screen/capture/status`。
新合同允许3厘米／0.10弧度以内的有界初态提案，默认单角色单轴2厘米／0.075弧度；
旧quality100合同仍保持1厘米／0.05弧度及原100条门槛。
补采目标6–12条，至少6个既有TRAIN祖先，每源最多2条，继续排除39/65/80/92/94这五个VAL祖先。

实际流程：先真实CPU物理screen，通过后重新执行并采RTX；CPU通过不计为新增训练条数。
新采集仍必须通过原`handover-physics-v2`和RGB／状态／时钟／渲染pose独立审核。
不改变质量、摩擦、门槛、超时，不在控制中恢复物体，也不添加帮助成功的外力。
目标／容器的扰动仅在暖机之前一次性应用，报告实际执行后的变化，不能把初态2厘米当作全程2厘米。
材质／generation seed若没有实际应用，继续明确记录false。

每条失败保留在queue及失败分母中。收集是否完成只看`accepted_count`和真实独立审核，
不按候选数、screen数或视频生成数冒充训练示范数。补采仍来自已有动作祖先，不能称新增人类动作来源。

示例：

```bash
# 独立计划生成后，先做物理screen，再逐个选择已通过的候选录制。
python scripts/4090_simple/collect_handover_supplement.py screen \
  --queue "$SUPPLEMENT_QUEUE" --max-candidates 32

source /home/zxc/CVPR_Evaluation_runtime/local.env.sh
CUDA_VISIBLE_DEVICES=0 OMNI_KIT_ACCEPT_EULA=YES \
  /home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/collect_handover_supplement.py capture \
  --queue "$SUPPLEMENT_QUEUE" --candidate-id "$ACCEPTED_SCREEN_CANDIDATE"
```

`pack_handover_supplement.py`只接纳已verified_queue登记的admission，检查完整authority、queue及
原包祖先SHA链，逐值核对新视频RGB与Parquet，沿用原38k归一化，不修改旧包。
新包保留1秒真实历史前缀，不对未执行暖机动作监督，不复制末帧或末动作补未来。
DINO与真实生产读取检查通过前，`PACK_READY.final_ready`保持false。

## 真实缓存与两包训练接线

新补采包和原100条包分别保存，不拼改episode编号或复用旧DINO缓存。
新缓存使用原`dinov2-with-registers-large`权重、`256x320`几何、bf16推理和float32存储。
RTX采集全部完成、释放GPU后再执行：

```bash
/home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  VLMExpert/scripts/data/extract_unifolm_dino.py \
  --dataset-root /home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024/supplement-packed-v1/train \
  --output /home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024/supplement-dino-v1 \
  --model-path /home/zxc/CVPR_Evaluation_runtime/resources/dinov2-with-registers-large \
  --shared-image-profile 256x320 --device cuda:0 \
  --storage-dtype float32 --inference-precision bf16 \
  --batch-size 32 --workers 4 --decoder-threads 1 --cpu-threads 4

CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 \
  /home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  scripts/4090_simple/verify_handover_supplement.py \
  --pack-root /home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024/supplement-packed-v1 \
  --dino-root /home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024/supplement-dino-v1 \
  --reference-dino-root /var/tmp/cvpr-evaluation-zxc-handover-quality100-20261022/dino-quality100-v1 \
  --run-config /var/tmp/cvpr-evaluation-zxc-handover-quality100-20261022/joint_gpu_preflight_100_cpu_v2/run_config.json \
  --expected-episodes 6 \
  --output /home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024/supplement_reader_preflight.json
```

验收逐集核对全部DINO值、SHA和连续30Hz帧索引，再核对每个生产窗口的完整历史与未来。
实际查询生产`Dataset`和collator，要求`motion_img_embs=[6,5,1024]`、
`actions/actions_mask=[30,80]`（78有效、2占位）及`states=[1,45]`。
包的完成标志不替代此项验收；不可变`PACK_READY`保留原打包时状态，后续读取就绪记录另存。

新trainer可分别调用`make_production_subset`构造两个真实Subset，
再用`mix_training_subsets(old_subset,new_subset,critical_v3)`获取`ConcatDataset`、
一个全局`EpochWeightedSampler`和混合收据，在`Accelerator.prepare`之前接入训练loader。
collator、batch、worker策略沿用当前训练配置，验证loader沿用原独立VAL。

默认每个补采覆盖祖先将原概率的50%分给其新执行，未覆盖祖先不变。
6个来源时，原100包占87.5%、新6条占12.5%，24个祖先的总概率仍均衡；
旧包内部继续使用v3转换权重，新包在各自祖先内均分完整有效窗口。
这一50%分流是显式实验设置，尚未证明最优。两侧须保持相同归一化、特征和变换合同，
不向已运行或恢复的训练任务热替换数据。

## 2026-10-24 首批实际结果

32个有界初态提案中15个通过CPU物理筛查、17个拒绝，保留全部记录。
选录6个不同TRAIN祖先的候选，6条RTX均正式通过；来源为0/9/64/70/74/95，
其中2条目标扰动、4条容器扰动。它们是新的物理执行，动作祖先仍来自原24个TRAIN来源。
新增包有2,459帧、2,105个完整训练窗；与原100条合计34,899窗。

这6条最大抬升为18.4–32.2厘米；最终50个50Hz观察点内最大位移不超过0.266毫米，
全部通过原门槛的有序交接、松手、承重、必要三维几何和同步渲染验收。
该验收仍不等于精确mesh内腔包含证明，也不等于新策略闭环成功率。

本地已实际完成全部DINO有限性/帧身份校验及36次单包生产查询。
两包混合预检另完成90个真实RGB/state/action样本和跨包batch，
34,899窗全部映射一致，新包概率12.5%、原包87.5%、24祖先总概率保持各1/24。
两个独立sampler在同epoch生成相同全局流，换epoch后流变化。

本次运行根目录为`/home/zxc/CVPR_Evaluation_runtime/handover_improvement_20261024`：

- `critical-v3/`：原100条的关键转换权重。
- `supplement-v1/`：真实筛查、RTX原始采集及逐条物理准入证据。
- `supplement-packed-v1/`、`supplement-dino-v1/`：独立新6条LeRobot包与真实特征。
- `supplement_reader_preflight.json`、`supplement_mix_cpu_preflight.json`：本地实际读取与混合验收。
- `DATA_IMPROVEMENT_READY.json`：聚合本地/A100交付状态及真实检查收据；以其实际状态为准。

本轮没有训练新模型，也没有改变既有训练、VAL或TEST。

## 后续真正的恢复示范

需要用户26场失败的实际trace目录、完整初态和划分后，才能从4–8秒实际偏离状态接管采集。
优先分解为重新抬升／稳定左手接管、容器上方三维对齐、下降承重、松手后撤腕。
这些是成功前的恢复监督；单纯延长末尾等待不能替代它们。恢复采集仍须使用可训练来源与
独立新初态；最终测试场景不因失败就移入TRAIN。

任何数据或采样增强都还需要新训练及同条件开发闭环验证，不能以数据准入替代策略成功率。
