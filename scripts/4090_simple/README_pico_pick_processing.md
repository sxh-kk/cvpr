# 三个 Pick 任务的分块补图与交付

本说明适用于隔离处理树 `pick300-processing/CVPR_Evaluation`，处理已录制的真人 PICO 数据，不启动采集器、不重放教师动作，也不继续积分物理状态。三个任务分别保存身份、原生成功判据和独立训练包，不能把它们标记成 Grasp。

## 本次冻结范围

运行根目录：`/home/zxc/CVPR_Evaluation_runtime/pico_latest300_20261109/processing_001`。

- `FROZEN300_INPUTS.json` 保留300条候选及原始文件身份。
- `FROZEN299_PROCESSING.json` 是本次获准生产清单：XMovePick 99、BendPick 100、XMoveBendPick 100。冻结 SHA256 为 `c8991ce63cfcce24e5e2b357ad4b06df03214a339a46bc46551153ae94583f2d`。
- XMovePick 004（UUID 前缀 `84e63cf0`，官方初始化索引1）仅在预热段出现成功，监督段没有成功。本次排除，原件保留，待补采；不能只用预热成功放行。
- XMovePick 117 是明确标注的补充片段，重复官方初始化索引0。`cohort_slot` 是本批条目位置，不是随机种子。不能宣称这批 XMovePick 覆盖100个不同初始化或100个不同 seed。

处理清单通过候选清单路径及SHA、明确排除条目和逐源文件大小、时间戳、SHA绑定原件。补图前检查完整299条冻结身份，再按任务子序列的 `offset`、`limit` 选择本块；报告保存实际 `selected_uuids`，不能用输出目录中的文件数量替代清单核对。

## 补图与空间安排

固定流程为：每块最多20条 RTX 补图 → 逐条独立验收 → 精确无损归档 → 下一块。原始状态、指令、模型和历史控制证据不删除，旧任务产物不覆盖。

`render_deferred_pick_batch.py` 每条单独恢复原始环境并验证绑定，从归档状态做零步恢复及前向更新。XMovePick、XMoveBendPick 保持原生世界坐标；BendPick 的采集环境已经适配世界高度 +0.405m，补图只安装对应帧适配钩子，不能再平移一次。训练图像来自 `head_stereo_left`；其他相机的诊断差异不能被称为相机等价。

既有 `render_deferred_pico.py` 保持原字节，SHA256 为 `c4c4735aaccb111e3ef08d48c69194b02af2f29994d8a6fc4e9e766575664215`。新批量生产者使用独立合同，同时绑定实际批量源码、旧零步辅助源码及任务依赖源码。运行中不要改这些生产者文件。

全批PNG会超过当前磁盘余量。`pico_pick_rgb_storage.compact_episode()` 将本次新增PNG保存为原分辨率、30Hz、RGB无损 master；**全部帧**解码RGB SHA与重新编码PNG整文件SHA都吻合，并持久化 master 和收据后，才移除本次派生PNG。PNG重建使用原Pillow/zlib及压缩级别3，跨环境必须保持兼容；像素相等不能替代PNG文件SHA相等。`restore_episode()` 可按同一门槛恢复PNG。

`/dev/shm` 只用于临时文件，不能作为唯一产物保存位置。训练包可在同文件系统通过显式 `--link-immutable-raw` 复用冻结模型、状态；仅本说明的三个Pick任务还可复用已封存、独立审核SHA绑定的会话控制日志和显式六项JSONL账本（指令、外力、原始质量、deferred事件、RGB事件及RTX逐帧桥），避免重复复制大JSONL。metadata、collection_record、源码及存储JSON收据继续复制。共享日志只在首次建立归档文件时新增硬链接，并严格确认单次链接数量、ctime及文件SHA；同SHA已有归档只复用，不改另一个源文件的链接数。默认复制及既有任务的日志处理保持原行为。无损 master 也可共享底层文件。硬链接的源与归档均不得原地编辑，发布前仍检查文件SHA和修改状态。

## 命令入口

从本隔离仓库根目录执行。先确认采集器和其他GPU作业已经退出，并使用新的报告、日志及输出路径。完整流程由 `process_pick_datasets.py` 串行调度：冻结计划、按余量选择最多20条、RTX、独立审核、压缩，然后每个任务分别打包、DINO和最终读取验收。首次只创建计划，检查后运行同一计划；已有计划直接复用，不能重新覆盖。

```bash
source /home/zxc/CVPR_Evaluation_runtime/local.env.sh
PICK_RUN=/home/zxc/CVPR_Evaluation_runtime/pico_latest300_20261109/processing_001
PICK_SIM_PY=/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python
"$PICK_SIM_PY" scripts/4090_simple/process_pick_datasets.py \
  --init-plan "$PICK_RUN/PICK_PROCESSING_PLAN.json"
"$PICK_SIM_PY" scripts/4090_simple/process_pick_datasets.py \
  --plan "$PICK_RUN/PICK_PROCESSING_PLAN.json"
```

驱动使用SIM310执行审核和压缩，MODEL311执行打包、DINO及生产读取；打包专用的 `dependencies/pillow123_model311` 在计划中逐文件绑定，只注入打包子进程。计划还冻结生产者代码SHA；修改后不能悄悄复用原计划。作业或余量不足时保留已有产物并等待，技术失败则保留现场供诊断，不自动删除后重跑。

底层单块入口如下，适合独立检查。示例接在已完成的 BendPick 前两条之后，不能与完整流程同时运行，也不能覆盖已有试跑结果。

```bash
"$PICK_SIM_PY" scripts/4090_simple/run_pick_rtx.py \
  --python "$PICK_SIM_PY" \
  --manifest "$PICK_RUN/FROZEN299_PROCESSING.json" \
  --profile bend_pick --offset 2 --limit 20 \
  --data-root /home/zxc/CVPR_Evaluation_runtime/pico_bend_pick_official_20261109/preparation_001/resources/data_root \
  --report "$PICK_RUN/rtx_chunks/bend_pick_002_020.json" \
  --log "$PICK_RUN/rtx_chunks/bend_pick_002_020.log"
```

其他任务使用清单 `groups[profile].render_data_root`，并分别维护自己的offset。已压缩块的PNG已经按协议移除，后续块应继续offset，不能将此前压缩片段误当成待恢复补图。失败或部分完成的产物保留，先核对失败原因，以新的运行位置恢复，不覆盖完成收据。

`run_pick_rtx.py` 保存实际驱动源码快照，并在子进程结束后生成 `_EXIT.json`，绑定退出码、argv、报告SHA和驱动SHA。Isaac关闭可能直接退出Python，因此 `render_completed_pending_process_exit` 只有与独立退出证据及该UUID的独立审核一起通过，才可准入；单凭退出码0不代表场景、帧或清理成功。

逐条独立审核入口：

```bash
"$PICK_SIM_PY" scripts/4090_simple/pico_pick_audit.py \
  "$PICK_EPISODE" --historical-sources "$PICK_HISTORICAL_SOURCE_MANIFEST"
```

`PICK_EPISODE` 必须来自本块冻结行；历史源码manifest使用该采集会话已封存且与原metadata相符的文件，不能用当前源码替代。审核通过后再调用 `pico_pick_rgb_storage.compact_episode(PICK_EPISODE)`；审核前不能删除PNG。

每个任务单独建立冻结准入选择清单，随后打包。以下变量需指向实际新输出、已继承统计的模板及安装真实LeRobot reader的Python环境；SIM310渲染环境本身不保证安装了LeRobot。

```bash
"$PICK_PACK_PY" scripts/4090_simple/pack_pico_handover.py \
  --selection "$PICK_ADMISSIONS_SELECTION" \
  --template "$PICK_TEMPLATE" --output "$PICK_PACK_OUTPUT" \
  --link-immutable-raw
```

打包逐帧保持原RGB，检查状态/动作/手部标签及Parquet往返，并实际调用LeRobot reader。`PACK_READY` 不等于最终训练就绪。完成本任务clean DINO后，再独立验证：

```bash
"$PICK_PACK_PY" scripts/4090_simple/verify_pico_handover.py \
  --pack-root "$PICK_PACK_OUTPUT" --dino-root "$PICK_DINO_OUTPUT" \
  --reference-dino-root "$PICK_REFERENCE_DINO" \
  --run-config "$PICK_RUN_CONFIG" --output "$PICK_VERIFICATION_OUTPUT" \
  --expected-episodes "$PICK_EXPECTED_EPISODES"
```

XMovePick 的预期条数为99，另两任务各100。最终验证检查portable包内原始RGB全帧复现、证据依赖、实际DINO和生产读取协议；不会因文件打包完成而自动授权训练、上传或迁移。

## 已观测的先导结果与限制

2026-11-09 本轮先导仅覆盖 BendPick 007、011：分别224、176帧，共400帧，RTX场景绑定、独立准入、全帧无损master及PNG精确重建已通过，原始物理数据未修改。实测输出阶段约9.6–10.7帧/秒，不含启动开销，不能据此宣称全299条或另外两任务已经通过GPU验收。

两条先导随后完成真实LeRobot读取、clean DINO和独立生产读取：400帧、282个训练窗口、6次实际CPU生产查询均通过，见 `PILOT_FULL_CHAIN_RESULT.json` 及 `pilot_release/PICO_DATASET_READY.json`。该结果仅覆盖两条先导，全299条仍以各任务的最终成功收据为准。运行环境缺包或PNG编码版本不同必须解决实际问题，不能放宽SHA门槛。
