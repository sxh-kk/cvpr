# XMove 两任务所选权重本机测评

入口 `run_pick_eval_queue.py` 实现用户确认的范围：XMovePick、XMoveBendPick
各自第 5000 个任务更新的权重，分别进行 L0/L1/L2 各 10 例、合计 30 例的
本机同步 RTC 测评。服务器继续沿用已授权的三个独立任务八卡 A800 训练队列，
每个任务 10000 步；本入口只读已有训练，不启动、重启或变更训练。

## 来源与部署

- 保留原 13 个保存点策略，等待所选任务完整训练结束、实际退出 0、全部模型保留、
  第 10000 步八卡健康检查通过后才回传其 5000 步权重。
- `fetch_pick_checkpoint.py` 固定任务、发布数据与已授权 QUEUE_PLAN 的 SHA；
  单次远端低优先级只读流计算 SHA256，本机落盘重读 SHA256，并检查传输前后
  模型 stat 与全部 24 份元数据一致。中断只回收本次 SSH，保留 partial，不自动重试。
- `prepare_pick_eval.py` 明确接受 `--profile`、`--step`，验证每个任务的发布数据、
  归一化、原始 38000 初始化、state retention 迁移、实际训练与来源；
  严格 CPU 加载所选完整 joint 的 VLM、Action，Motion 参数也只从同一 joint 导出。
  旧 Motion 资源只提供构造元数据。训练 `rtc=False` 保留，服务时显式 test_time RTC。
- 同一卷使用硬链接，避免复制两份 12 GB 模型；两个任务结果、视频、权重均保留。
  启动每项前预留模型、Motion 与 4 GiB 安全余量，空间不足时保留现状并失败。
  CPU 阶段单核、单线程、低优先级；GPU 阶段等待当前 Bend 评测完整退出及 GPU 空闲。

## 物理与评测协议

使用训练时的两个私有 SONIC task adapter，保持官方保存场景的所有真实物体、
机器人、桌面、材质与灯光。仅对未使用空间键、明确为空的相机字段及已有等价证据
的三个 DR 序列化字段做可逆兼容派生，保存原场景与逐项派生凭据。
恢复完整保存场景时 `dr_level=None`，不重新随机采样。

两个任务均保持原生坐标、200 Hz 物理、50 Hz 控制、800 控制步/16 秒预算，
包含原有 50 步预热。XMoveBend 的 L0 源 metadata 缺省预算时，明确记录采用
未修改原任务类的 800 步默认值。模型自己预测移动、弯腰和抓取的 body64/fingers14，
不使用采集时摇杆、导航脚本、距离阈值切换或教师动作。
原生成功仍为目标抬高约 9 cm，无额外移动、弯腰、接触或持有时长成功门。

初始化观察 profile 为 `native_saved_scene_observed_v1`：完整记录隐藏 reset 一步
和 50×4 个预热物理子步，共 201 步；时钟、有限状态、完整账本、零机器人与目标接触、
无实态写入、无额外物理步和策略干预是技术门。沿用的 2 mm/2° 是 Bend 工程参考，
在本原生保存场景评测中仅作为诊断告警，不删除场景或改写成功率。
原始 XMoveBend L1-01 中盒子与玩具枪存在约 9.486 mm 重叠，预热时自然推开，
到策略开始前稳定；按统一规则记录该告警，不为该编号设置例外、不移动物体、
不重置奖励基线、不延长预热。Bend 的独立 `stable_support_v1` 与其严格工程门保持不变。

test_time RTC 固定 30 步预测、执行 15 动作，对应 25 个 50 Hz 控制步；采样种子
仍为 `20261105 + 10*level + case_index`。每例新建推理服务和仿真器；
验证真实权重身份、因果 DINO 历史、双端 HTTP/RTC 动作记录、原生奖励/终止与视频解码。
技术错误单列；原生任务失败保留为实际失败；不自动重试或只选成功例。
本结果是官方固定场景的私有控制适配诊断，源 `split=train` 保留，
不宣称独立 heldout 场景或官方原控制器同协议成绩。

## 启动与结果

源代码通过必要测试、独立 review、提交并推送后，生成独立 acceptance JSON，
绑定当前提交、源码 SHA、授权两任务/5000步/RTC15/每任务30例与初始化观察 profile。
用新的仓库外输出目录启动，已有 ACTIVE/EXIT/结果目录时拒绝覆盖。

```bash
python3 -B scripts/4090_simple/run_pick_eval_queue.py \
  --output /var/tmp/cvpr-xmove5000-eval-20261111 \
  --acceptance /var/tmp/cvpr-xmove5000-eval-20261111/CODE_ACCEPTANCE.json
```

队列 `ACTIVE.json` 指明等待训练、回传、CPU 部署、等待 GPU、测评或完成状态。
每项的 `official30_rtc_001/FINAL_EVALUATION_ACCEPTANCE.json` 保存实际成功数、
分层结果和初始化警告；逐例视频在对应 simulation/episode 目录。
两项完整验收通过后才生成 `RESULTS.json` 和实际退出 0 的 `EXIT.json`。
任何阶段失败保留原始证据，精确回收本次已证明归属的子进程，不影响其他工作。
