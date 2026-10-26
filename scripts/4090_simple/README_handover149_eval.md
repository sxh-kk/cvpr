# 149 条版本的 ckpt_5000 本机测评准备

`prepare_handover149_eval.py` 只接纳当前 A100 **BS4 重跑**的任务微调第 5,000 步完整联合模型。它不接纳旧 114 条模型、已失败的 149 条 BS8 运行或原始 UnifoLM 第 38,000 步模型。

当前模型从 UnifoLM 38k 完整继承 VLM、六块双来源 ActionExpert 和 MotionExpert，再以 149 条任务数据从任务第 0 步联合微调。此处 `5000` 是新增任务优化器更新数，训练预算为 10,000 步；不能将它称为 UnifoLM 5k 或 38k 原模型。

149 条由原训练集 45 条、新随机场景 14 条、真人 PICO 90 条组成，共 68,259 帧、59,468 个完整历史与未来动作窗口。原独立开发验证集未并入训练。这一训练数据组成及准备检查不代表模型的物理任务成功率。

## 下载完成后执行

使用现有 Conda Python 3.11 模型环境，仅在原子下载完成、`backup_receipt.json` 已生成之后运行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONDONTWRITEBYTECODE=1 \
  /home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  /home/zxc/CVPR_Evaluation/scripts/4090_simple/prepare_handover149_eval.py \
  --source /home/zxc/CVPR_Evaluation_runtime/checkpoints/handover149_bs4_20261026/ckpt_5000 \
  --resources /home/zxc/CVPR_Evaluation_runtime/resources \
  --output /home/zxc/quality100_4090_20261024/mirror/home/weisheng/psi0_runtime/quality100_milestone_eval_20261023/handover149_experiment_5000_official30_4090_trt10133_20261026/local_run \
  --repo /home/zxc/CVPR_Evaluation \
  --release /home/zxc/CVPR_Evaluation_runtime/checkpoints/handover149_bs4_20261026/ckpt_5000/release_metadata
```

输出目录必须尚不存在。失败时保留 `PREPARATION_FAILED.json` 与已有产物；不要覆盖、删除失败证据后把同一目录当成成功重跑。

原始目录需包含 `model.safetensors`、运行目录原样的 `run_config.json`、`argv.txt`、`dataset_statistics.json`、`preparation.json` 和独立备份收据。另需 `evidence/prepared_run_config.json`、`evidence/prepared_argv.txt`，保留准备目录的原始输入及其训练准备 SHA。`release_metadata` 保留训练时原始的四份小型元数据：

- `packed/PACK_READY.json`
- `dino/manifest.json`
- `dino/MERGED149_REINDEX.json`
- `TRAINING_DATASET_READY.json`

## 必须通过的准备检查

- `backup_receipt.json` 使用 `cvpr-handover149-local-checkpoint-backup-v1`，明确不复制优化器，并绑定实际 BS4 运行、完成保存的 5k 里程碑、源文件传输前后与本机 SHA256。
- 准备输入与运行目录的保存配置分别核验 SHA；只准许实际五个叶子字段变化：`train.hf_token` 的确切 Path repr 序列化，以及这次 W&B tracker 的 `entity/group/id/name` 四个确切值。其余完整配置递归相等，准备和运行 argv 经 `shlex.split` 后所有参数 token 完全相同。不能因日志字段变化而放开模型、数据或优化参数检查。
- 完整模型 SHA256 为 `57f689768b0c0edf9b4407eb441f701d1b35be9807d76f2397dda4083fb02cfd`，文件为 12,066,487,572 字节；下载临时文件不是可用模型。
- 训练配置和原始命令、继承的 38k 统计文件、149 条数据组成、DINO 原生 30Hz 清洁历史身份与重索引关系都匹配训练准备收据。不能重拟合统计、改 token 键或用旧 114 条 DINO 合同代替。
- 实际模型源码匹配训练冻结的 `632968d` 关键模块；VLM 和动作头严格 CPU/meta 载入，保存动作头签名必须为六块、`state_drop_prob=0`；Motion 从本次联合权重导出并严格载入，不回退到原始 Motion 权重。
- DINO 的模型字节、处理器和几何身份保持一致。历史参考到本次训练的 Torch / Transformers 版本转换显式记录，不声称跨版本逐位相等。
- 导出合同顶层的 `motion_image_policy` 显式取自本次已核验 `run_config.json` 的 `data.transform.model.motion_image_policy`，并要求为 `vlm_color_clean_motion`。旧参考合同缺字段时也不能依赖默认值或用静态资源合同代替本次部署合同。

准备输出为 `LOCAL_RUN_READY.json`，合同为 `cvpr-handover149-local-deployment-v1`，状态为 `local_handover149_run_prepared`。其模型硬链接、当前 Motion 导出、原始元数据、组件载入记录及 DINO 合同均位于新 `local_run` 中。原始数据和权重不改写。

## 主机与运行命名空间

上述工具生成**主机上的 CPU 准备产物**。评估使用独立 Linux 运行命名空间，其中主机 `/home/zxc/.../mirror/home` 被映射为 `/home`；原始 `/home/zxc/CVPR_Evaluation_runtime/...` 并不自动可见。

独立运行框架需把本次模型及收据硬链接或复制到新评估根的 `source_bundle`，并从原始配置显式映射生成新 `deployment`。VLM 构造资源、当前 Motion、统计文件与 DINO `online_checkpoint` 必须指向命名空间内实际可见的新路径。不能原样使用主机配置路径，也不能改旧 114 条部署冒充本次模型。实际映射及源 SHA 由运行框架另外记录。

CPU 准备不会启动模型服务、DINO 推理、GPU、SONIC 控制或 SIMPLE 物理仿真。只有随后新 149 模型在 4090 完成服务载入、真实在线历史/状态/动作接线和实际物理评估，才可报告对应测评结果。30 场景协议、SONIC v1.1、localhost 通信、RTX/TRT 身份和视频证据由独立运行框架验收；安装成功或 `LOCAL_RUN_READY` 都不等于测评通过。旧测评的 25 个已完成案例须保留，与新模型结果分开。

首次 GPU 预检曾因验收端读取旧静态资源合同、且首版导出缺少上述顶层策略字段而技术失败。该失败不能当作模型输入或策略效果失败。原失败记录、首版 `local_run` 和 `LOCAL_RUN_READY.json` 保留原样；修复后的运行框架使用独立 `_r2` 链，显式绑定补字段的来源与新合同。仓库导出修复仅作用于之后新建的输出，不回写旧收据，也不重复已通过的完整权重数值验收。服务和验收端必须同时读取实际部署合同。

## CPU 回归

```bash
python3 -m unittest discover \
  -s /home/zxc/CVPR_Evaluation/scripts/4090_simple/tests \
  -p 'test_prepare_handover149_eval.py' -v
```

这些测试覆盖来源、保存完成、版本与里程碑、BS4/global32、动作/state 语义、固定归一化、数据组成、DINO 完整性和源码身份拒绝边界，不加载真实模型或使用 CUDA。
