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

另一个与主分类重叠的观察属性是预热位移：已独立复核的补充分析002显示，
L0/case02、03、04、09，L1/case02、03、04、09及L2/case09这9例，
首个学习策略前目标已水平移动约9.44–14.88厘米，9例均未成功；
15例成功案例同期最大位移约0.551厘米。这9例与RTC关闭基线的前50个预热样本
核对字段完全相同，不能归因于高层RTC动作。
上面的五个主类别不重复计数，仅将L2/case09显著倾斜的情况单列；
不能将“单列1例”理解为只有1例出现预热位移，也不能把9例另加到15次失败上。
初始几何、重力／接触与低层预热控制仍需干预实验区分，不据此删除案例或推算修复后分数。
原观察和限制见[补充分析002](PICO_MOBILE_PP_OFFICIAL30_EVAL.md#15例逐例复核与预热关联补充分析002)，
本次结果视图新增`WARMUP_CONTEXT_SUPPLEMENT_001.json`绑定该补充来源，未重做原轨迹数值审计。

### 视频与观察证据

分析目录为上述运行根下的`failure_analysis_001`：

- `failure_levels01.json`与`failure_level2.json`：独立分工查看15例画面后的逐例依据、
  视频时点、轨迹事实、原文件身份、置信与限制。
- `FAILURE_STAGE_SUMMARY.json`及`failure_cases.csv`：统一五类、15例唯一覆盖及逐例原视频路径。
- `Mobile_PP_6000_RTC_15_failure_videos.zip`：15段未剪辑、未转码的原始`rollout.mp4`，
  按`videos/level{L}_case{NN}.mp4`命名；同时包含CSV、观察记录及关键画面。
- `FAILURE_DELIVERY_CHECK_001.json`：新ZIP的实际SHA、大小、完整性检查及15段视频覆盖。
- `FAILURE_CANVAS_TYPECHECK_001.json`及补充后的`FAILURE_CANVAS_TYPECHECK_002.json`：
  失败结果视图的TypeScript检查；不冒充实际界面渲染验收。
- `WARMUP_CONTEXT_SUPPLEMENT_001.json`：复用独立补充复核的9例预热位移属性，
  与主类别重叠，不改变原视频包、CSV或原生结果。
- `WARMUP_SUPPLEMENT_INDEPENDENT_REVIEW_001.json`与`FINAL_FAILURE_DELIVERY_RECORD_002.json`：
  补充18项独立复核通过后的最终交付绑定；保留初始版本和原视频包不变。

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

实际是否启动或完成，以新根的进程收据、逐例`CASE_SUMMARY.json`和最终
`FINAL_EVALUATION_ACCEPTANCE.json`为准；请求文件或启动脚本本身不代表评测已执行。

### 实际5000步备份、CPU部署与启动

5000步模型已于2026-11-05 22:08（北京时间）完成独立备份，
源前后与本机完整SHA均为`50d12beded25fa0dce5be9153bf4ce9807fe89fb952ab21f7733011d04828ba4`，
共12,066,487,572字节。24份metadata及源／本机文件身份通过核准；
`source/backup_receipt.json`的SHA为`6e5798a0adf6bfb02ebeaa349ae3d593c5406672b22d0895da32c47db031cdbf`。
父任务另以`SOURCE_BACKUP_INDEPENDENT_REVIEW_001.json`核对小收据和实际stat，未重复hash大模型。

原传输driver在备份封口之后、CPU部署之前返回143，原因未明；
`DRIVER_TERMINAL_001.json`与原日志保留，不伪造fetch退出码，也未重新下载模型。
首次CPU准备于22:09:42启动，PID3990227、仅自身CPU26／nice19／隐藏GPU，实际返回0。
严格组件加载和同一5000步joint的1060项Motion源张量精确导出通过，
新Motion共1,616,029,181字节，SHA为`f9d61584c2addbb06626bf1b8da51529e81fcd0476071f9154d7e5208ef5bfb6`。
`deployment/LOCAL_RUN_READY.json`的SHA为`89910e54ee905d0673501f169f2896437ed1cf0b07a0b78ab14dbaa7295d766b`，
保留`training_completed=false`及未使用旧Motion参数的证据。

父任务首次预启动检查误将CPU收据状态预期为通用`passed`，在任何子进程启动前拒绝；
`LAUNCH_PRECHECK_FAILURE_001.json`记录该检查器错误。
按实际合同`passed_cpu_deployment_ready`修正检查后，才进行唯一一次真实campaign启动。
此类准备过程问题不混入30例任务失败或技术异常分母。

实际评测于22:11:04启动，独立wrapper PID3992394，监督器PID3992398／start ticks192484207，
端口21077，输出为新根的`official30_rtc_001`。
`LAUNCH_WRAPPER_001.json`及`OFFICIAL30_RTC5000_PROCESS_001.json`记录真实命令、
CPU范围0–31／nice19、源提交`d90d75f756cc0433b77743b165b568eb4d16b7b3`、
新模型SHA及6000步RTC参考验收SHA。
该提交与6000步RTC冻结运行的全部推理／物理源码SHA相同，新增差异仅为文档。
实际服务首例启动记录确认`rtc=true`、`rtc_mode=test_time`、`Ta=15`、10步采样。

首例L0/case00实际原生成功，866个控制tick／17.32秒（含50个预热tick），无跌倒。
`FIRST_CASE_INDEPENDENT_REVIEW_001.json`核准真实5000步权重／新Motion、首次plain加后续32次test-time RTC、
每25tick重规划、因果观察、暂停时钟、参数冻结／梯度为空及867帧视频证据。
首例通过不等同全30例完成；完整结果仍须由最终实际验收及进程退出记录核准。

第一等级10例完成后，`LEVEL0_PREFIX_10_INDEPENDENT_REVIEW_001.json`
仅增量检查其余9例并复用首例既过证据，实际通过，SHA为
`93a3b196f47fe942f41202813fe40c2110babce750135e133a7e806caea82d39`。
Level 0原生成功5/10（case00、01、05、06、07），技术异常0、跌倒0；
五次未成功均已实际查看本次5000步原画面：case02、03、04、09未形成持续带物搬运，
case08已搬到篮边，但约18.6–18.8秒放置时盒子与篮子均掉离第二桌。
逐例依据保留于`failure_observation_001/level0_failures.json`，
不能将该前10例结果当作完整30例成绩。

Level 1随后完成，原生成功2/10（case01、06），累计7/20；
`LEVEL1_PREFIX_10_INDEPENDENT_REVIEW_001.json`实际返回0、SHA为
`04171fc08cf9db852379ca30f94c4fb3e90816ce4070cdc50511e8e7aeff3f8e`。
该回执仅新增核对Level 1十例，前10例复用既过收据、summary SHA及未变证据stat，
20例均无技术异常、跌倒或预策略跌倒；该前缀本身不代表完整30例结果。

### 实际5000步完整30例结果

2026-11-05 22:11:11至23:57:07（北京时间），`official30_rtc_001`实际完成全部30例，
监督器的真实进程收据为`process_returned`、`returncode=0`。
原生成功8/30（26.7%），Level 0／1／2分别5/10、2/10、1/10；
技术异常0例、未尝试0例、跌倒及首次学习策略前跌倒均0例。
`FINAL_EVALUATION_ACCEPTANCE.json`的SHA为
`6b6755750d8216d26f5432fc19d317f4d665cb92ffa82f82098205eaa9a9a3bb`。
每例首次有效策略采样（首个成功的`/act`调用）使用plain初始化，
之前的warmup请求为425且不执行sampler；后续实际1268次`test_time` RTC引导。
保留预测30帧、执行15帧／0.5秒、25个控制tick一窗及HTTP等待暂停仿真钟。
全部22次未成功均正常完成1200个控制tick，不是收包／推理技术超时。
`FINAL_RESULT_INDEPENDENT_REVIEW_001.json`实际有限审计返回0、SHA为
`3bee05581c268bbc97a5c772049cbe262ded820505d8e612676981a3209844e2`；
只新增核对Level 2十例，前20例复用既过收据和未变身份，
全30例实际RTC、参数冻结／梯度为空、因果历史、DINO、时钟、native及原视频证据通过，
最终`state.json`与验收文件字节一致，真实进程收据和原PID退出均通过。

`CKPT5000_VS_6000_RTC_COMPARISON.json`与`ckpt5000_vs_6000_rtc_cases.csv`
核对两轮完整实际验收、真实进程退出、30份官方初始化和逐例策略seed完全相同、
RTC政策相同、不同模型SHA、新5000步Motion及全部60段原视频身份。
配对结果为：两者均成功7例、仅5000步成功1例、仅6000步成功8例、两者均未成功14例。
仅5000步成功的是L2/case07；仅6000步成功的是L0/case08，L1/case00、05、07，
L2/case01、02、03、08。5000步本轮较6000步的15/30少成功7例，即低23.3个百分点。
对照JSON的SHA为`a7fa6e7d4ce531c8ac1591b5d4e195b5f8421229b8952355caa43b76ef4be6fa`。

这次比较保持RTC及执行窗不变，变化是保存点权重和从该joint重新导出的Motion；
不是同一权重的RTC开关实验。两保存点各单次固定30例，CUDA内核不保证逐位确定性，
不能将分数差直接解释为过拟合、RTC因果或统计显著的新场景泛化。
SONIC适配的`formal_evaluation_eligible=false`与Level 0训练同场景限制保持，
原训练提前停止、未完成计划10000步的事实也未改变。

### 5000步的22次未成功与原视频

三组实际新画面的观察保留于新运行根`failure_observation_001/level0_failures.json`、
`level1_failures.json`、`level2_failures.json`，分别覆盖5、8、9次未成功。
每例一个主要阶段，合计22例、不重复计数：

- 未形成持续抓持／未完成带物搬运9例：L0/case02、03、04、09，
  L1/case02、03、04、09，L2/case00。
- 已抓起但搬运停滞、结束仍持物4例：L1/case08，L2/case03、05、08。
  这些案例未到篮处，不能统称“到篮口后不松手”。
- 搬到篮处后放置落地4例：L0/case08，L1/case05、07，L2/case04。
  仅L0/case08的目标与篮子均掉离桌面，其余三例篮子仍在桌上。
- 24秒结束仍持物、未完成释放入筐3例：L1/case00，L2/case01、02。
  L2/case01末段才抵近第二桌，case02伴随篮子倾斜；不声称三例都已入篮。
- 抓取／离桌阶段目标掉落、先前稳定抓持未证实1例：L2/case06。
  本次约7.4–7.8秒从第一桌边掉落，与6000步同例的掉落时点不同。
- 预热明显变位、后续未恢复抓取1例：L2/case09。
  首次策略前净移13.19厘米、倾75.24度；首次接近翻平的采样在elapsed 1.020秒，
  已属`policy_latent`，视频1.2秒翻平也在首次策略之后。该主类不证明根因，
  也不代表其他主类从未出现预热位移。

统一结果为`FAILURE5000_STAGE_SUMMARY.json`和`failure5000_cases.csv`；
保留原观察code以及`observed_failure_stage`细分，CSV只提取说明文字，
结构化观察中的原始时点／元字段仍由组JSON原样随包提供。
汇总SHA为`e3cbe494817d432d205b8dd60629a34e6de38c4066b9f82320a19b2a801d0fbf`。

实际原视频包为`failure_observation_001/Mobile_PP_5000_RTC_22_failure_videos.zip`，
共121,990,460字节（约122 MB），含22段未剪辑、未转码的原始MP4、逐例CSV、
完整观察JSON及关键画面；按`videos/level{L}_case{NN}.mp4`定位。
SHA为`c0f896aba2eb2e365e8dfdf23bfb2039f7b4515328df7b557b222de944b81ccd`。
实际包装进程返回0、ZIP CRC及22段唯一覆盖通过，证据为`FAILURE5000_DELIVERY_CHECK_001.json`。
第一次包装曾被不同观察schema在写CSV／汇总／ZIP之前拦截，
`FAILURE5000_PACKAGING_ATTEMPT_FAILURE_001.json`保留原错误；兼容真实字段后全22例预检通过，
源观察不改，仅重新进行首次成功写出的视频包装，没有重跑任何评测案例。
准备／打包程序问题不计入30例任务失败或评测技术异常。

完整对照视图位于
`/home/zxc/.cursor/projects/home-zxc/canvases/mobile-PP-RTC-5000-vs-6000-20261105.canvas.tsx`，
可按Level查看30个案例、打开两轮全部60段原视频，并取回5000步和6000步各自的完整失败包。
`COMPARISON_CANVAS_GENERATION_001.json`绑定真实完整结果与两个新产物SHA，
`COMPARISON_CANVAS_TYPECHECK_001.json`实际TypeScript 5.9.2检查无诊断，
不声称完成实际界面渲染或浏览器交互验收。所有视频、图片、权重及运行证据均保留在仓库外。
最终实际产物独立复核为`FINAL_ARTIFACT_INDEPENDENT_REVIEW_001.json`，SHA为
`c803dade691136810ad186aacae16f4ba3409fcbcef447fabe863da00df7258a`，
核准配对、CSV、22段ZIP覆盖、原组JSON字节及全部视频身份、Canvas实际内嵌数据和类型检查。
`FINAL_ARTIFACT_IDENTITY_SUPPLEMENT_001.json`另外核准joint／新Motion／READY实际stat未变、
冻结运行绑定及30份服务命令均指向5000步deployment；Motion身份来自该绑定和严格导出，
不冒称服务startup直接报告了Motion SHA。
