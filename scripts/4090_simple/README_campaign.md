# Handover149 多保存点评估

这组工具用于 Handover149 联合模型及其配套发布包。通用的单模型评估见[运行入口](README.md)。

## 单个保存点

先按[部署准备](README_handover149_eval.md)生成该保存点的运行目录，再创建评估框架：

```bash
python scripts/4090_simple/prepare_handover149_campaign.py \
  --step <TASK_STEP> --tag <RUN_TAG> \
  --source "<CHECKPOINT_BACKUP>" --local-run "<LOCAL_RUN>" \
  --expected-model-sha256 "<MODEL_SHA256>" \
  --template-framework "<FRAMEWORK_TEMPLATE>"
```

`--step` 支持 4000、6000、7000、8000、9000、10000，表示 Handover 任务微调的更新数。
配套备份、数据和框架模板须属于同一训练任务。

## 串行执行

准备 `CAMPAIGN_DIR/config.json` 后执行：

```bash
python scripts/4090_simple/queue_handover149_eval.py --campaign "<CAMPAIGN_DIR>"
```

配置中的 `contract` 为 `cvpr-handover149-serial-local-evaluation-v1`，`steps` 为
`[4000,6000,7000,8000,9000,10000]`。路径字段包括 `predecessor`、`backup_root`、
`reference_backup`、`model_python`、`resources`、`repo` 和 `template_framework`，
同时填写 `predecessor_sha256`、`ssh_host`、`tag`、`hostname` 和 `gpu_uuid`。
路径使用绝对路径；`poll_seconds` 控制轮询间隔。

队列在前一保存点评估结束后才运行下一项。中断后使用同一配置和 `--resume` 继续；
每个保存点使用独立输出目录及对应权重。
