# A100：继承40k三专家的Handover任务微调

完整继承UnifoLM 40k checkpoint的ActionExpert、VLM和MotionExpert，继续联合更新，DINO冻结。
优化器、scheduler、RNG及任务步数重新开始，不新建动作头。原运行目录和用户交付数据只读保留。

## 数据和协议

交付目录为`/gaozt-test1/datasets/sigmma_dataset/G1WholebodyHandoverTeleop-v0`。
50条成功示范保留原划分TRAIN45、VAL5：29条原轨迹修正，21条共享模板0的场景再生成仅用于TRAIN。
保留全部视觉备注。示范筛选不是模型成功率，也不是官方benchmark分数；没有独立TEST。
现有物理成功门槛不保证箱体整齐进入篮腔，正式评测仍需独立官方场景。

源50Hz数据的state45为手、手臂、腿/腰、颈占位；40k模型state43为腿、腰、手臂、手。
`prepare_handover_task50.py`按原40k info中的具名关节重排，原状态另存`source_state45`，模型补齐45维。
手14和`body_token_v1_1`64保持原执行值，同步选择`ceil(k*50/30)`的真实RGB、状态、动作。
名义时间为`k/30`，实际源时间另存，最大偏移13.334ms。视频无损保留选中H264解码RGB并逐像素验证。
不重编码token，不插值动作，不宣称30Hz抽样后控制轨迹重新通过了物理验收。
原50Hz成功证据只绑定复用，不重复物理或数值验收。

派生数据在`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz`。
归一化明确继承原40k的`dataset_statistics.json`和bounds，不使用交付包新拟合的bounds。
metadata描述统计仅按具名顺序投影，原统计封存溯源；TRAIN/VAL的原bounds覆盖情况写入检查报告。
冻结DINO使用原权重、256x320几何，每集保存30Hz，读取过去一秒6个5Hz时刻。
首秒只供真实历史，末尾不足30个真实未来动作的窗口排除，不复制首帧或末尾动作。
VAL固定每集8窗，共40窗，八rank等分无补齐；验证使用独立固定RNG且还原训练RNG。

## 八卡配方

- 八张A10080GB，DDP，每卡batch4、累积1，有效batch32。
- FP32 master/AdamW、BF16 autocast、gradient checkpointing，DINO冻结。
- 2000次optimizer更新、100次warmup、cosine，任务步数从0开始。
- 峰值LR：Action1e-5、语言骨干1e-7、视觉塔1e-6、投影1e-5、Motion1e-6。
- 保留生产全rank loss/梯度有限性检查和分组clip1.0。
- 第2、20次更新额外检查每个可训练tensor前4个值跨rank相等，不冒充完整参数逐字节比较。
- 每200步及第20步保存完整checkpoint，最多保留12份，不清理原40k权重。
- 沿用生产验证语义：完成第1、201、401……次及最终更新后验证，验证201不能冒称checkpoint200成绩。

实际窗口数和覆盖遍数见`data_check.json`；这是一轮初始预算，不是最优参数结论。

## 环境和运行

独立Conda prefix：`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126`，
不共享原venv可变site-packages。版本和CPU权重核验在仓库外
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_40k_preparation`。

```bash
conda activate /gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/envs/task-finetune-py311-cu126
python scripts/4090_simple/prepare_handover_task50.py \
  --source-root /gaozt-test1/datasets/sigmma_dataset/G1WholebodyHandoverTeleop-v0 \
  --state-contract /gaozt-test1/datasets/UnifoLM/unifolm_sonic_lerobot_train/meta/info.json \
  --output /gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz

# 两个split分别提取DINO；下面为TRAIN。
python VLMExpert/scripts/data/extract_unifolm_dino.py \
  --dataset-root /gaozt-test1/datasets/SIMPLE_sonic_v1_1/handover_task50_30hz/Handover-sonic11-train \
  --output /gaozt-test1/datasets/SIMPLE_sonic_v1_1/dino/handover_task50_30hz/Handover-sonic11-train \
  --model-path /gaozt-test1/datasets/MixEgoMotion/weights/dinov2-with-registers-large/e4c89a4e05589de9b3e188688a303d0f3c04d0f3 \
  --device cuda:0 --batch-size 32 --storage-dtype float32 --inference-precision bf16

bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh prepare
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh check
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh run
```

`prepare`生成可复现argv/配置/来源；`check`实际读取DINO和LeRobot样本、检查关节顺序、真实历史及完整未来；
`run`重核文件及GPU空闲后启动，不停止其他训练。已有目录不重复启动，可用`TASK50_RUN_DIR`指定新目录。
旧UnifoLM后训练脚本默认新建动作头，不应用于本任务。
已接受的CPU数据检查仅重绑文件身份，不重复解码或数值验收；父进程和八卡worker使用同一绝对配置路径，
避免工作目录不同使未使用的DeepSpeed配置字段也产生不一致。首次命名失败发生在创建worker之前，0更新记录保留。

## 损失和剩余时间

默认输出目录：
`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/task50_40k_preparation/handover_joint40k_b32_s2000`。

- `run_config.json`、`argv.txt`、`preparation.json`：完整配方与来源。
- `data_check.json`：实际读取、有效窗口、bounds覆盖、文件绑定。
- `training_state.json`、`train.log`：生命周期及控制台。
- `progress/losses.csv`：完成更新数、TRAIN/VAL loss、五组LR。
- `progress/loss.png`、`progress/progress.json`：损失曲线、耗时、ETA及预计完成时刻。
- `progress/monitor_health.json`、`monitor.log`：监控自身状态。

监控只读offline W&B。warmup结束且至少有5个真实更新间隔后，按最近墙钟耗时估计ETA，
包含已观察的验证/保存暂停。未开始、失败及最终收尾时不伪造ETA。训练和监控退出码分别保存。

```bash
bash VLMExpert/scripts/train/psi0/finetune-handover-task50-joint40k.sh status
```

40k模型12,066,487,572字节，SHA256为
`ff1b6ea8c63edae671220b443ed08ae11ef8efe07cd83cdcc0f90233f3462dc9`。
真实CPU生产加载通过：1944个stored tensor逐值/dtype/shape一致且有限，两项共享alias按实际共享关系恢复。
该检查0更新、未建optimizer、未用GPU，不替代本任务八卡或物理验收。
训练是否启动、实际loss和任务表现以各自运行记录为准，不因安装或loss下降宣称测评通过。
