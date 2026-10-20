# A100 → 4090：联合20k资源交接

目标与模型协议见[需求交接](4090_SIMPLE_HANDOFF.md)，4090已完成的环境检查见
[本机报告](4090_SIMPLE_ENVIRONMENT.md)。此步骤只迁移资源，不改训练或启动正式测评。

## 当前A100任务

2026-10-20北京时间21:45:24，已完成本次训练**20,000步完整保存、保护及资源发布**。
发布包含26项带SHA的文件，共14,916,113,138字节（不含manifest与READY小文件）。状态文件：

```text
/gaozt-test1/datasets/UnifoLM/.pipeline/exports/unifolm-20k-261020043942/status.json
```

只有状态为`model_resources_ready`，且下述`resources/MODEL_RESOURCES_READY`存在时才能拉取。
`waiting_for_complete_checkpoint`表示尚在等待；`failed`应查看状态中的error及同级
`unifolm-20k-261020043942-export.log`，不能使用`resources.partial`。

本次联合模型SHA256：

```text
a000c559a0de0a3be477933aa31ac5c681a5a79fe380c344adb89dde399d535a
```

已验收：6项CPU回归、真实argv＋JSON解析和全部bounds不变、20k Motion的1060项权重
（恢复1个共享PE别名后1061项）meta严格加载／有限性／序列化逐值比较、统计与视觉契约、
已发布26项文件的完整SHA复核。导出后训练继续运行，其记录的9个运行源码SHA全部一致。
这些是CPU资源验收，未代替4090 GPU完整模型或仿真闭环验收。

导出器检查模型、optimizer、scheduler、8卡RNG文件及训练日志的精确保存完成标记。
随后在`protected_checkpoint/ckpt_20000`建立独立硬链接目录（跨文件系统时复制），
避免后续保留3份的轮转删除20k。源文件内容／权限不变；不使用GPU，不停止训练。

资源发布目录：

```text
/gaozt-test1/datasets/UnifoLM/.pipeline/exports/unifolm-20k-261020043942/resources/
```

内容约15 GB十进制，最终以manifest字节总数为准：

- 完整联合`model.safetensors`，同时包含本次20k VLM、Motion、ActionExpert。
- Qwen配置／processor／tokenizer小文件，不重复搬初始化VLM大权重。
- `joint20k/motion/joint_motion.ckpt`：20k Motion子权重及构造／架构／统计／视觉metadata。
  原epoch100仅提供metadata，不补回旧独立参数，也不携带EMA或optimizer。
- 原训练统计及重映射后的JSON、完整argv；原件保留在`joint20k/provenance/`。
  只改8项部署路径，模型架构、完整tokens／pooling、state时间条件、bounds不变。
- DINO权重逐文件匹配训练缓存SHA，另带`dino_training_contract.json`。
  当前后训练／推理为256×320，历史6帧无随机增强；Motion预训练metadata的256×480保留，
  导出报告显式校验尺寸适配，并单列NumPy／Pillow版本变化：Motion预训练为2.2.6／12.3.0，
  联合后训练缓存为1.26.4／11.3.0；4090在线处理对齐后者。DINO权重与归一化不变，
  不将这次转换笼统称为仅尺寸变化。在线DINO关闭第二次resize/crop，使用慢版processor。
- `manifest.json`逐文件SHA256及CPU验收结果；不含环境变量文件或认证信息。

**不要传整个exports目录或protected_checkpoint**：4090推理不需要optimizer、训练RNG、
训练集或离线DINO缓存。只传`resources/`，独立保护副本留在A100。

## 4090拉取与核验

先安全同步Git代码；以下在4090执行，`A100_SSH_ALIAS`填写已经连通的SSH别名：

```bash
cd /home/zxc/CVPR_Evaluation
git status --short
git pull --ff-only

export A100_SSH_ALIAS=你的A100连接别名
export CVPR_EXPORT_SOURCE=/gaozt-test1/datasets/UnifoLM/.pipeline/exports/unifolm-20k-261020043942
ssh "$A100_SSH_ALIAS" "cat '$CVPR_EXPORT_SOURCE/status.json'"
ssh "$A100_SSH_ALIAS" "test -f '$CVPR_EXPORT_SOURCE/resources/MODEL_RESOURCES_READY'"
```

确认READY检查成功后继续，目录末尾的`/`需要保留：

```bash
mkdir -p /home/zxc/CVPR_Evaluation_runtime/resources
rsync -avh --partial --info=progress2 \
  "$A100_SSH_ALIAS:$CVPR_EXPORT_SOURCE/resources/" \
  /home/zxc/CVPR_Evaluation_runtime/resources/

/home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  scripts/4090_simple/export_bundle.py --verify \
  /home/zxc/CVPR_Evaluation_runtime/resources
```

校验失败时保留报错，续传后再次核验；不跳过SHA检查。JSON及argv的运行路径已经针对
`/home/zxc/CVPR_Evaluation_runtime/resources`和`/home/zxc/CVPR_Evaluation`生成。
若换目录，需要同时重映射两者；`provenance`和Motion内嵌训练metadata中的旧路径只用于溯源，
不能重新挂载A100来满足运行时读取。

可以将以下任务直接交给4090端代理：

> 阅读AGENTS.md、docs/4090_SIMPLE_HANDOFF.md、docs/4090_SIMPLE_ENVIRONMENT.md及
> docs/4090_SIMPLE_TRANSFER.md。通过已配置SSH从A100拉取已发布资源，核验全部SHA。
> 使用本机双Conda环境补齐真实双来源历史／DINO服务与43-state、64-latent＋14-hand映射，
> 核实配套SONIC v1.1和SIMPLE资产；先做严格模型加载与单episode闭环验收，记录显存和延迟。
> 不使用A100推理，不启动训练或额外仿真微调，不用旧SONIC替代未确认版本。

## 就绪边界

`MODEL_RESOURCES_READY`和`--verify`只证明资源已导出及迁移字节一致。
真实4090 GPU严格加载／DINO推理／历史缓冲／控制闭环仍需验收；本机通信fixture不代表这些已完成。
模型包不会伪造以下尚缺资源：

1. 与训练`body_token_v1_1`配套且有身份依据的SONIC decoder／controller。
   A100旧目录虽有ONNX候选，但不能证明v1.1兼容，未纳入正式资源。
2. SIMPLE任务、USD／纹理及固定初始条件资产；源码data目录的README不算资产。

## 重新运行导出器

仅在独立的新输出目录运行；不覆盖已有快照或已发布包。A100使用现有模型Python环境：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
python scripts/4090_simple/export_bundle.py \
  --run-dir /完整训练run目录 \
  --training-log /对应train.log \
  --output /仓库外新导出目录 \
  --target-resources /home/zxc/CVPR_Evaluation_runtime/resources \
  --target-repo /home/zxc/CVPR_Evaluation \
  --dino-dir /与训练相同的本地DINO目录 --wait
```

CPU回归检查：`python -m unittest discover -s scripts/4090_simple/tests -v`。
导出器还执行真实配置解析、Motion meta严格加载、共享别名恢复、统计／视觉契约校验及
逐tensor序列化比较；这些均不代替4090完整模型GPU验收。
