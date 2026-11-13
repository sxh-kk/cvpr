# Handover Qwen六图单因素训练

本实验从原始联合预训练第38000步重新开始，只把Qwen输入从当前1图改成历史5图加当前图。六个时点为当前之前1.0、0.8、0.6、0.4、0.2秒和当前，与Motion原有六帧5Hz真实历史相同。新配置显式为`data.transform.model.vlm_history_frames=6`，不改变`model.img_chunk/n_cams/observation_horizon`或`repack.num_past_frames`。

## 实际基线

服务器原运行位于`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/handover149_training_20261026/bs4_restart_001/formal10000`，实际10000步完成且返回0。149条TRAIN由原45、新随机14、真人90条组成，共68259帧、59468个完整历史及未来30帧窗口。原五条VAL每集固定8窗，共40窗，保持独立。

保持seed20261025、8卡每卡batch4、累积1、全局32；BF16、DDP、融合AdamW、cosine、warmup1000、梯度裁剪1.0；动作头/语言/视觉/投影/Motion峰值LR分别为1e-5/1e-7/1e-6/1e-5/1e-6。保持原256×320几何、VLM颜色增强及clean Motion DINO、不启用view augmentation、六个融合块、状态丢弃0和原归一化。训练`rtc=false`不改。

原checkpoint策略为每1000步保存，实际保留1000至10000共10份模型，不使用后来Pick任务的13节点。仅新运行的optimizer/scheduler/RNG恢复负载保留最近两套；旧数据、训练、模型和特征不清理。

原完整source38000 SHA为`12a71e046e221b83f0bdb0efb50f9ba82442045aca0efabef3ae337f185320b7`。新训练不得加载Handover5000权重、检查更新后的模型或恢复旧优化器/RNG。

本机原字节证据位于`/home/zxc/CVPR_Evaluation_runtime/handover_qwen_history6_20261113/baseline/`。20份小文件的manifest SHA为`cbe9bc8f041f0f16bf6388a84816805aed926c111978d36cfc327121fd83b110`；原prepared config SHA为`0513f440cbeba2c3dfca3db272bfa2effe215d6c5dc50870dcd399bacbe04e09`，实际run config SHA为`63ea48074fd95f6f350a9a3a781ab4f6b409fb567db5d4b7c86fec42037edfb0`。

## 原实现继承与新命名空间

`qwen_history_handover_train.py`按SHA加载原`a100_handover149_bs4.py`，继承完整窗口选择、验证子集、state migration、训练器、融合优化器、DDP、保存及第2/20/10000步健康检查。原质量adapter只在内存中派生一处`CODE`依赖根，其他字节可逆相等；原runner的自身路径路由到新入口，使name/train子进程继续经过新合同。原源码文件不修改。

建议新服务器根为`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/.pipeline/handover_qwen_history6_20261113_001/`，包含`baseline/`、冻结`code/`、独立`single_gpu_probe/`和新`formal10000/`。数据与DINO用新目录内只读引用，仍指向原149TRAIN和原5VAL。

CPU准备器要求独立`CODE_ACCEPTANCE.json`，合同`cvpr-handover-qwen-history6-code-v1`、`status=passed`、`single_factor=data.transform.model.vlm_history_frames`、上述baseline manifest SHA及完整`source_sha256`库存。完整配置递归比较只允许这个1→6语义叶子，以及新实验名/时间戳/输出目录/数据引用目录/deepspeed代码根/W&B项目；其他模型、数据和优化器字段必须相等。

```bash
CUDA_VISIBLE_DEVICES='' <原Python> -B <冻结code>/scripts/4090_simple/qwen_history_handover_train.py prepare \
  --baseline <新root>/baseline --code <冻结code> \
  --code-acceptance <新root>/CODE_ACCEPTANCE.json --output <新root>/formal10000
CUDA_VISIBLE_DEVICES='' <原Python> -B <冻结code>/scripts/4090_simple/qwen_history_handover_train.py check \
  --prepared <新root>/formal10000
```

原Python为服务器`.pipeline/envs/task-finetune-py311-cu126/bin/python`，已实查Torch2.7.0、Transformers4.57.1、Accelerate1.7.0、FlashAttention2.7.4.post1。原正式环境保持NCCL P2P/SHM开启、IB关闭、`CUDA_LAUNCH_BLOCKING=1`；正式启动须普通优先级nice0及全0–127 CPU affinity，不能继承单核低优先级检查环境。

## 真实GPU检查与正式启动门

先在独立probe完成单卡实际batch4、完整Qwen和Motion前反传至少两步，核六图视觉placeholder/token、`image_grid_thw`为24行、历史cache rows/video timestamps与Motion六帧同源同5Hz时点，以及损失和全部梯度有限。probe更新不得转入正式训练。

正式`ROOT_TRAINING_ADMISSION.json`绑定本次配置和preparation SHA、source38000 SHA、真实单卡probe路径/SHA及`fresh_formal_load_from_joint38000=true`。probe收据需明确`actual_batch_size=4`、`qwen_images_per_sample=6`、`motion_history_frames=6`、`forward_backward_completed_steps>=2`、`full_vlm_motion_gradients_finite=true`、`probe_weights_reused_for_formal_training=false`。门通过后才可`run --prepared ...`；正式8rank重新严格加载source38000并fresh seed，实际第2/20步继续验batch、有限梯度、融合optimizer及跨rank参数探针。探针不是完整模型逐参数相等证明。

未部署、未运行真实probe或正式训练时，不将CPU测试称作训练或吞吐验收。任何已有运行状态/输出都拒绝覆盖，失败保留，不自动重启。

## 后续评测边界

历史Handover5000收据为原生28/30（各级9/9/10），strict19/30属于另一指标；原raw评测目录目前不在本机。服务器同一个5000保存文件的路径、dev/inode/size/mtime与旧全SHA备份收据仍完全一致，未为此重读12GB。训练准备不能伪称旧评测原件已恢复，也不能套用Mobile/Pick场景、时限或warmup。

后续应恢复或显式冻结同一Handover30评测协议，对single-frame5000和history6的独立5000权重使用同一RTC/seed/时限/warmup做对照，保留真实原生和strict分母。新history6训练和部署必须使用独立源码合同，不能标成未修改的旧632968d代码。
