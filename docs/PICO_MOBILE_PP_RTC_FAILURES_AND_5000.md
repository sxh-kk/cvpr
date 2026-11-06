# Mobile P&P：RTC失败阶段复核与5000步同配置测评

## 6000步RTC版的15次未成功

本页分析已完成的6000步`test_time` RTC版本，区别于更早RTC关闭的基线。
原运行根为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_eval30_testtimertc_20261105`，
`official30_rtc_001`实际原生成功15/30，技术异常0例、跌倒0例；
15次未成功均完整运行1200个50Hz控制tick，即24秒含预热。
原协议和完整结果见[官方30例测评记录](PICO_MOBILE_PP_OFFICIAL30_EVAL.md#同一6000步模型的test-time-rtc版本)。

结合原视频的抓取、搬运、末端画面及50Hz轨迹，按每例一个主要观察阶段分类：

- 未形成持续抓持／未完成带物搬运：9例。
  L0/case02、03、04、09；L1/case02、03、04、09；L2/case00。
  这里不声称从未短暂触碰或拿起。L1/case02与L2/case00在第一桌停留抓取，
  其余对应案例可见空手向第二桌移动，目标没有完成有效搬运。
- 抓取阶段目标掉落：1例，L2/case06。
  约视频5.2–5.4秒手与盒子分离，目标随后从第一桌边落下。
  现有证据不能确认此前是否建立过稳定抓持，不能直接写成“稳定拿起后搬运中掉落”。
- 搬到篮处后放置失败、物体落地：2例，L2/case04、07。
  case04在约17–18秒可见篮子与目标均落离第二桌；case07释放附近目标落到篮前地面，篮子仍在桌上。
- 24秒结束仍持物，未完成释放入筐：2例，L1/case08、L2/case05。
  这是正常任务预算内末端阶段未完成，不是PICO收包或推理技术超时；不推断延长时间必然成功。
- 预热阶段目标明显变位，后续未恢复抓取：1例，L2/case09。
  首次学习策略发生于绝对仿真1.005秒／视频1.0秒；此前目标已水平移动约13.19厘米、
  倾斜75.24度。完全翻平约视频1.2秒，发生在首次策略动作之后。
  该例单列观察条件，但保留原生未成功及原30例分母。

以上是失败阶段统计，不是已经通过干预对照证明的因果根因。
50Hz接触采样不构成200Hz持续抓持证明；不从本轮固定案例推断过拟合或新场景泛化。
SONIC适配的`formal_evaluation_eligible=false`及RTC非确定性内核边界保持。

### 视频与观察证据

分析目录为上述运行根下的`failure_analysis_001`：

- `failure_levels01.json`与`failure_level2.json`：独立分工查看15例画面后的逐例依据、
  视频时点、轨迹事实、原文件身份、置信与限制。
- `FAILURE_STAGE_SUMMARY.json`及`failure_cases.csv`：统一五类、15例唯一覆盖及逐例原视频路径。
- `Mobile_PP_6000_RTC_15_failure_videos.zip`：15段未剪辑、未转码的原始`rollout.mp4`，
  按`videos/level{L}_case{NN}.mp4`命名；同时包含CSV、观察记录及关键画面。
- `FAILURE_DELIVERY_CHECK_001.json`：新ZIP的实际SHA、大小、完整性检查及15段视频覆盖。
- `FAILURE_CANVAS_TYPECHECK_001.json`：失败结果视图的TypeScript检查；不冒充实际界面渲染验收。

结果视图位于
`/home/zxc/.cursor/projects/home-zxc/canvases/mobile-PP-RTC-failures-20261105.canvas.tsx`，
可按Level查看各失败阶段并打开对应原视频与关键画面。
所有视频、权重、运行日志及图像保留在仓库外。

## 同一停止run的5000步RTC版本

用户授权改用5000步权重，继续采用同一官方Level 0／1／2各10份固定初始化。
新运行根为`/home/zxc/CVPR_Evaluation_runtime/mobile_pp_official100_ckpt5000_testtimertc_20261105`；
`EVALUATION_REQUEST.json`记录请求和已完成6000步RTC参考验收的SHA。
只读取同一已停止训练run保存的`ckpt_5000/model.safetensors`，不启动、重启或修改训练。

部署须使用5000步独立的源前后／本机SHA备份及真实CPU严格加载，
从同一5000步joint重新导出Motion；不能复用6000步Motion参数。
现有`prepare_mobile_pp_official_eval.py --step 5000`和
`evaluate_mobile_pp30.py --checkpoint-step 5000`已支持该保存点，无需改变推理算法或物理实现。

与6000步RTC版本一致的配置为：

- `rtc_mode=test_time`，完整预测30帧，每次执行15帧／0.5秒；
  `d=0`、`s=15`、指数mask、guidance α=0.9、10步采样。
- 环境seed0；策略seed为`20261105 + 10 * level + case`。
  每例新策略服务／新仿真，官方整份初始化恢复，禁止teacher、未来GT或示范状态注入。
- 原生成功阈值0.9，最多1200个50Hz控制tick／24秒含1秒预热。
  HTTP等待期间仿真钟暂停；仍为同步计划衔接，不宣称异步实时延迟补偿。
- 继承同一数据、统计、DINO身份、SONIC资源与控制协议；不追加样本、不改成功标准、不自动重跑。

该5000步legacy／24秒版本已实际完成30例，原生成功8/30，Level 0／1／2为5/10、2/10、1/10；
技术异常、机器人跌倒和自动重试均0，进程返回0。完整记录在上述运行根的`official30_rtc_001`。
之后用户另行授权30秒、`wrist_retreat_v1`及每0.2秒重规划的新组合，实际原生17/30、
同轨迹24秒前缀15/30；该组合与本节配置不同，不能混作本节同配置结果。
预算、逐例配对、原视频与原生评分局限见
[30秒扩展测评](PICO_MOBILE_PP_OFFICIAL30_EVAL.md#5000步30秒与更短执行窗的独立版本2026-11-06)。
