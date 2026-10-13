# cvpr

MotionExpert 与 VLMExpert 的代码迁移快照，包含配置、测试、文档、必要静态资源和第三方代码。
**本仓库不包含 checkpoint 或模型权重，也不包含训练数据、私有 `.env`、虚拟环境和运行日志。**

| 项目 | 内容 | 入口 |
| --- | --- | --- |
| MotionExpert | E7-K12-400M 文本条件运动专家，支持 GLOBAL 特征、训练、EMA、导出与推理 | [说明](MotionExpert/README.md)、[设计](MotionExpert/project.md) |
| VLMExpert | 基于 PSI0 / Qwen3-VL 的 GLOBAL token 预训练、特征提取，以及 WBT 数据适配 | [设计](VLMExpert/project.md)、[原工程说明](VLMExpert/README.md) |

迁移请从 **[迁移文档](docs/MIGRATION.md)** 开始。两个项目使用不同版本的 Transformers，需要分别建立 Python 环境。

清点结果见 [汇总](docs/inventory/summary.json)、[逐文件清单](docs/inventory/files.csv) 和 [第三方来源](docs/inventory/upstream_dependencies.json)。
清单记录上传与排除状态、大小、SHA-256，以及公开发布时的必要修改。

## 获取代码与资源

大于 1 MiB 的非 checkpoint 资源使用 Git LFS，主要是机器人网格、媒体和第三方 SDK 文件。

```bash
git lfs install
git clone https://github.com/sxh-kk/cvpr.git
cd cvpr
git lfs pull
python scripts/verify_inventory.py
```

只查看代码时可使用 `GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/sxh-kk/cvpr.git`；完整资源校验前仍需执行 `git lfs pull`。

迁移验证通过 MotionExpert 的 18 项测试和 VLMExpert 的 26 项 GLOBAL / WBT 测试。
这些测试使用小型模型和临时数据，未执行正式训练或机器人部署。

第三方代码沿用各目录的原许可、版权和引用；本仓库未对全部内容重新授予统一许可。
