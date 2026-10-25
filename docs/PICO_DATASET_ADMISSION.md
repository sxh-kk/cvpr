# PICO真人示例：固定90条核定、转换与A100交付

2026-10-25用户指定只整理已冻结的90条候选，后续录制不追加、不补数。
候选清单为本机`pico_data_admission_20261025/FROZEN90_INVENTORY.json`，冻结时间22:15:31（北京时间），
包含唯一UUID、原始编号、已封存目录及episode／metadata／collection摘要。
这90条均已通过原物理预验收；新的CPU格式预检90/90通过，共41,401行、36,091个完整历史及未来窗口。
旧66条数值预检复用已绑定结果，仅新增检查24条；指令均为实际录制的Handover文本。
该预检不代表RTX、完整准入、DINO、训练读取或上传已经完成。

用户继续采集，明确在结束后才补图上传。当前采集、自动补图或其它GPU计算进程存在时，
`process_pico_dataset.py run`返回75及等待状态，不能把资源等待记成数据失败。
程序不结束、重启或修改这些进程，不改A100训练目录、参数或原数据集。
普通暂停界面仍有采集进程，因此不会触发GPU处理；需要本次采集真正退出。
原启动脚本的自动后处理先运行完，本流程才接续，避免同时写同一份审计或争用GPU。

## 阶段与门禁

1. 按路径及SHA读取**已冻结90条**，不重新从持续增长的history选数据；后续主动丢弃不复活。
2. 对尚缺RTX的候选按原`mjSTATE_INTEGRATION`补渲染，不重新执行动作或物理。
   已有完整补图可复用；部分失败产物保留待诊断，不直接覆盖。
3. 完整独立审计只放行真实人类来源、实际v1.1编码身份、完整命令／RGB／状态、零外力、
   物理任务及几何证据全部通过的记录。技术故障与真实拒收分开记录；技术故障未排除时不上传。
   旧审计的输入及检查器完全一致时复用；检查器改变时先保留旧收据再重新核定。
4. 新变长LeRobot包只包含通过的记录，保持独立`train_only`来源，不动旧TRAIN／VAL／TEST。
   人类身份采用命名空间／session／UUID；兼容reader的百万起始整数不是官方数据集episode编号。
5. 提取真实clean DINO原生30Hz特征，固定`256x320`、CLS+4 registers、`[N,5,1024]`、float32存储；
   继承当前已验收编码器／预处理／bf16计算身份，不能用旧视频的特征代替。
6. 真实CPU生产Dataset／collator检查完整未来动作、六帧历史、RGB及归一化数值，生成独立本地就绪收据。
7. 仅在就绪收据与所有当前字节一致后上传全新的A100目录，重读SHA及纳秒mtime。
   远端先明确标为字节已核、路径待适配；保留原cache，独立重绑路径及签名。
8. A100使用独立已提交代码快照及原有Python环境做CPU生产读取，GPU隐藏、nice19且仅绑定一个允许CPU；
   自动解码线程也继承该亲和性，不能宣称总OS线程只有1。通过后才写`A100_PICO_DATASET_READY.json`。
   仅上传完成不等于训练就绪；本流程不启动或切换训练。

跨工作目录的审计按唯一文件角色核对固定摘要，并实际重读原录制路径；不会因checkout前缀不同误拒，
也不接受丢失原文件、假摘要、改字节、重复／缺失角色或身份flag变化。

## 格式保持

- 不重新编码实际执行的64维SONIC v1.1 token；14维手指按38k继承合同的具名顺序保存。
- 保存43维真实关节状态，生产reader按原合同补到45；动作64+14补到80，末两维监督关闭。
- 前30行真实warmup RGB／状态保留作历史；动作标签为NaN、`action_valid=false`、80维mask全false。
- 第30行起只保留具有完整未来30步的current；六帧历史为`[-30,-24,-18,-12,-6,0]`。
- Parquet保留名义`row/30`及真实50Hz ceil网格时钟；其`source_frame_index=ceil(row*50/30)`。
  DINO的`frames.npz source_frame_index`必须为连续视频解码序号row，二者不能混用。
- 视频用无损RGB H.264并逐像素解码比较，Parquet逐值回读；统计与bounds继承原38k，不重新拟合。

## 本机计划及操作

准备好的计划：`/home/zxc/CVPR_Evaluation_runtime/pico_data_admission_20261025/PROCESSING_PLAN.json`。
它绑定固定90条清单的SHA、两个Conda环境、38k模板、DINO参考身份、提交版本和独立远端目录。
从独立的`pico-dataset-admission`工作树运行，不修改正在服务的采集checkout：

```bash
source /home/zxc/CVPR_Evaluation_runtime/local.env.sh
/home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python \
  scripts/4090_simple/process_pico_dataset.py run \
  --plan /home/zxc/CVPR_Evaluation_runtime/pico_data_admission_20261025/PROCESSING_PLAN.json
```

本地发布目录：`/home/zxc/CVPR_Evaluation_runtime/pico_handover90_20261025_human01`。
远端发布目录：`/gaozt-test1/datasets/SIMPLE_sonic_v1_1/pico_handover_20261025_human90_01`。
实际通过数可能小于90；不以新录制填补拒收。
`PROCESSING_STATUS.json`按UUID记录通过、拒收原因和技术异常；`logs/`保留检查、补图、转换与传输日志。
`ADMISSIONS_SELECTION.json`一旦冻结，已有pack必须与其SHA相符，重试不能换清单。
`PACK_READY.json`始终不冒充完整生产就绪；本地与A100分别使用独立就绪收据。

本轮准备验证：44项审计CPU回归、6项打包回归（含真实无损MP4与LeRobot读取）、
11项就绪／路径重绑及5项传输回归、18项流程回归通过。验证包括真实临时DINO cache路径重绑、
远端字节脚本的本地临时执行、缓存就绪后文件变更拒绝，以及子进程亲和性且父进程不变。
真实90条的GPU补图、DINO及A100就绪仍须在采集结束后实测，未完成时不声明上传或完整验收成功。

本机原始记录与预检证据位于`/home/zxc/CVPR_Evaluation_runtime/pico_data_admission_20261025`；
数据、日志、权重和缓存不提交Git。
