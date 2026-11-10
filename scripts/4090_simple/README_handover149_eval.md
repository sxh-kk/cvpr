# Handover149 模型部署准备

`prepare_handover149_eval.py` 将 Handover149 BS4 任务的完整联合模型转换为评估运行目录。
输入包括权重备份、发布元数据和本地构造资源；通用模型评估见[运行入口](README.md)。

`--step` 支持 4000、5000、6000、7000、8000、9000、10000，表示任务微调更新数。
此入口要求模型对应 149 条任务数据、六块双来源 Action Expert 和保留状态的训练配置。

```bash
python scripts/4090_simple/prepare_handover149_eval.py \
  --source "<CHECKPOINT_BACKUP>" \
  --resources "<RESOURCE_ROOT>" \
  --output "<LOCAL_RUN>" --repo "<REPO_ROOT>" \
  --release "<RELEASE_METADATA>" --step <TASK_STEP> \
  --expected-model-sha256 "<MODEL_SHA256>"
```

`CHECKPOINT_BACKUP` 包含实际的 `model.safetensors`、配置及 `backup_receipt.json`。
`RELEASE_METADATA` 包含该模型对应的数据、统计和 DINO 元数据。
`RESOURCE_ROOT` 包含 Qwen、Motion、DINO 和 SIMPLE 场景资源。
`LOCAL_RUN` 使用仓库及源备份目录之外的新目录，并与模型文件位于同一文件系统以创建硬链接。

输出包含重映射后的配置、联合权重、Motion 权重和对应的 DINO 配置。
多保存点评估见[串行评估](README_campaign.md)。
