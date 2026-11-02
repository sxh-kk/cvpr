# Grasp：官方100个TRAIN初始化的PICO＋SONIC采集

本入口针对六任务中的桌面Grasp，来源是
`simple/G1WholebodyTabletopGraspMP-v0`。它不等于XMovePick。
用户于2026-11-02选择继续PICO＋SONIC，为此新增独立任务身份
`cvpr/G1SonicTabletopGraspPico-v0`，不将其标记为官方AMO控制复现。

## 初始化与成功判据

`pico_grasp_collection_plan.py`固定官方TRAIN元数据SHA
`183474d4439d4d5de2b261f3c6ce0b28328f000af0e1a4f5ae0b9305dfb2cfc7`，
冻结全部100条episode0–99的原始environment_config字符串字节。
不以整数seed猜测初始化，不读取示范动作或未来轨迹。
单桌、目标薯片罐、三个干扰物、scene2、材质、光照及相机均由保存的DR恢复。
原始初始化和控制适配后的配置分别绑定身份，不改官方元数据。

原Grasp使用AMO，保存数据为50Hz，物理步长0.002秒。
本适配使用已有固定SONIC1.1编码器/解码器、200Hz物理、50Hz控制、
具名43关节实测状态及实际64维token＋14手指指令。
PICO双腕与躯干目标从当前实际FK进行X键标定；腿部使用站姿规划参考，
摇杆意图保留在记录中并明确忽略，不锁定或改写实际root/关节状态。
AMO与SONIC的机器人XML、PD、碰撞模型及物理频率有差异，数据不宣称动力学等价。
SONIC冷启动仍使用原官方默认关节角，并在控制历史开始前按实际足部碰撞边界一次落地；
该root高度调整及其后实际站姿预热独立记录，不冒充AMO示范的机器人状态。

成功判定继承原Grasp的`compute_reward`和`check_success`函数。
原函数以reset后首次真实reward时的目标高度为基线，抬升量除以0.02米，
reward≥0.9即成功（对应18毫米）。采集器只记录同一次env.step返回的真实flag、
reward和实际基线，不重算或改变成功标准，不添加容器、接触或持稳条件。
每份初始化通过完整录制与原生成功复核后前进；任务未成功可重试同份，
技术异常保留原件并暂停，不能当成任务质量失败。

## 空间与证据

原AMO场景的桌面约z=0、地面约z=−0.785。
SONIC适配显式将世界场景整体上移0.785米，并补偿机器人XML自身的root偏置；
独立处理世界/机器人局部相机及Isaac场景变换。每条记录绑定frame-adaptation收据。
只有实际模型和相机核对通过，才能称当前场景已准备好。

Grasp没有container，使用独立target-only证据合同
`cvpr-live-pico-grasp-deferred-physical-capture-v1`。
全部干扰物等场景状态仍在float64完整`mjSTATE_INTEGRATION`中；
保留200Hz零外力记录、50Hz原始native样本、30Hz动作/状态、模型和状态SHA。
旧Handover/Mobile两角色合同保持原默认。
EGL画面仅作操作反馈，训练RGB待后续按完整状态零物理步RTX获取。
原始采集通过不代表训练包、DINO或A100上传完成。

## 使用

先为本轮准备独立资源根和冻结计划，再启动：

```bash
bash scripts/4090_simple/start_pico_grasp.sh \
  --round-root /仓库外新目录/grasp_round_001 \
  --data-root /仓库外已核对资源/data_root \
  --collection-plan /仓库外冻结计划/manifest.json
```

`pico_simple_teleop.py --task grasp --startup-check-only`只做真实50步站姿预热、
新鲜EGL画面和参考规划器准备，然后退出；不启动PICO SDK、不录制、不计新数据。
该检查必须使用独立输出目录。实际采集从暂停状态等待操作员。

头显XRoboToolkit选择Controller、连接电脑并开启Send，关闭A键切换Send。
操作顺序：X标定→Y开始跟随→A开始录制→保持初始姿态1秒→抓起薯片罐→A保存。
复核通过后自动进入下一份官方初始化，再X标定。
Y暂停或界面重置属于主动中断；原件仍保留，不算完成。

## 采集过程监测与故障定位

采集器在同一个SDK接收端记录回调序号、距上次回调的时间、最近32个不小于50毫秒的回调间隔，
以及设备断开事件。它们通过`status.json`和原有`result.json`的拒绝记录输出；
诊断读取不消费动作包，不刷新动作包时间，不增加第二个PICO接收端。
50毫秒仅是诊断记录条件，不改变新包等待、数据新鲜度、控制落后或完整性验收阈值。

启动采集器后，可为其实际launcher PID启动一个独立只读监测进程：

```bash
/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python \
  scripts/4090_simple/monitor_pico_collection.py \
  --round-root /仓库外目录/grasp_round_001 \
  --launcher-pid 实际launcher_PID \
  --session-output /仓库外目录/grasp_round_001/collection/实际live目录 \
  --output /仓库外目录/grasp_round_001/monitor_001 \
  --interface wlp18s0
```

监测器每秒GET本机`/status.json`，只读当前会话结果、该launcher及其后代的CPU／RSS／IO／调度信息，
同时记录系统压力、剩余磁盘及指定网卡计数。仅监测器自身使用nice19及单CPU；
不更改采集器或其它任务，不调用SDK、模拟器、GPU或控制POST接口。
`latest.json`提供最新状态，`samples.jsonl`和`events.jsonl`分别最多保留3份8MiB轮转日志；
`snapshot.json`保留最近一次实际故障及此前至多60秒的1Hz现场。
回调间隔仅作诊断，等待头显或历史失败不冒充新故障；录制拒绝及技术复核异常单独记录。
同一轮仅允许一个监测器；本轮完成或采集器退出超过30秒后自动结束，停止监测器不会停止采集。

1Hz监测可能漏过短暂资源尖峰，接收回调间隔也混合了网络、服务、SDK和线程调度延迟。
应结合拒绝时立即保存的接收状态及当前／前一控制循环耗时定位，不能只凭一次收包超时断言Wi-Fi故障，
也不能以短时空闲监测证明连续录制不再中断。日志为本地诊断，不自动发送聊天通知。

## 本轮准备记录

运行记录根：`/home/zxc/CVPR_Evaluation_runtime/pico_grasp_official_20261102`。
`preparation_001/GRASP_SOURCE_AUDIT_001.json`记录官方任务、100条来源和差异；
`preparation_001/resources/RESOURCE_PREPARATION_001.json`及资源闭包收据记录scene2下载与路径适配。
冻结计划SHA为`b63a9c6ea619ce507e1ccd58dfd43ba86bae1dcd97b3e85825c8a85a7effbd06`。
实际启动、物理/画面验收及限制以仓库外真实回执为准，未完成的检查不提前标为通过。

2026-11-02实际准备验收：`CPU_FRAME_GEOMETRY_001.json`对两套真实XML零步编译，
场景锚点平移最大误差0；最终`startup_check_003/result.json`完成50个实际SONIC控制步和新鲜EGL画面，
pelvis从0.7591米到0.7495米，末端倾角3.05度，状态有限。
早期`STARTUP_CHECK_001.log`因诊断入口未重定位原作者绝对资源路径而在构造模型前拒绝，
已修正，仅保留失败日志，不计采集尝试。

独立review另补上target角色不可与干扰物交换、归档头200/50/30Hz严格匹配的技术检查。
最终适配器SHA为`17edaeac6259846730eddf21c4b83ed2cae9e484dbaab47ca05a2dd566e1312f`；
较早check002/RTX001保留其原适配器身份，未冒充最终版本。

`rtx_zero_step_002/result.json`从最终完整状态获取新鲜RTX图像，归档状态无积分且逐字节不变；
目标及三个干扰物USD/MuJoCo位置最大误差2.69e−8米，桌面中心一致。
head camera位置残差约0.637毫米，torso link约10毫米，保留实际SONIC USD/XML差异，
`render_mount_equivalence_verified`仍为false。这些检查不证明两套机器人安装/动力学等价，
也不替代操作员连续采集的吞吐或100条训练包验收。

2026-11-02本轮首次录制001已通过采集阶段原生成功与完整性复核，仍待后续RTX和训练数据终验。
002因等待新PICO动作包约99毫秒而技术中断；两次拒绝前一循环分别约13.79和11.17毫秒，
现有证据不能将它归因为持续控制计算超时，也不能判定网络／SDK中具体哪一层。
按用户明确要求仅重启当前采集界面，保留001全部12个文件的SHA／字节身份、002技术异常原件和历史编号；
清除已封存002对应的阻塞，重试官方episode001（第2份），进度保持1/100，下一次录制编号003。
恢复和重启记录位于`round_001/restart_002/`；新采集会话为`collection/live_20261102_163823`，
`launcher_002.log`与旧日志分开保存。上述新增诊断不改变初始化、SONIC动作、200/50/30Hz或成功标准。

重启后16:40:28和16:41:43两次跟随暂停已捕获同一消费者的即时证据：
最后回调分别距拒绝151.86和153.22毫秒，控制等待99.46和99.25毫秒，
前一循环实际处理10.39和9.62毫秒，规划器无故障，尚未创建新录制。
PC服务日志在两次暂停时同秒仍有TCP读取，之后曾恢复；16:41:58后停止读取，16:42:22明确离线。
这只能证明短时回调断供及后续离线，不能定位150毫秒延迟发生在网络还是服务／SDK调度。
`restart_002/RECEIVER_TIMEOUT_FINDINGS_003.json`及`TRANSPORT_FINDINGS_003.json`保留实际证据。

独立监测器已在`round_001/monitor_001/`启动，显式绑定launcher002和本次live会话，
每秒快照已实际更新；历史002不会被当成本次新故障。
本次接收端、采集入口、控制集成和监测器合计102项CPU回归通过，独立review通过。
控制集成测试同时修复两个在HEAD也存在的fixture／AST定位错误，不改其对应生产逻辑。
未重复已通过的场景数值或RTX审计；监测启动时采集处于暂停等待重新连接状态，
不能据此声称已完成连续采集稳定性验收。

### 18:45故障及同步传输复查

18:45:55再次暂停时，前一循环实际处理10.233毫秒，等待新包99.475毫秒，
最后一次SDK回调已有156.829毫秒；下一包在暂停后38.351毫秒到达，形成195.180毫秒间隔。
故障前后还出现连续约24个回调一批、随后停顿约170毫秒的模式，暂停未录制时同样出现。
本次尝试尚未启动正式capture，成功001及1/100进度保留。现场独立冻结在`round_001/fault_008/`。

只读核查当前安装的`libDeviceConnectionManager.so`及官方实现，确认旧连接清理只清buffer／解绑信号，
没有关闭旧socket。当时29条头显ESTAB中只有1条在区间内收数、28条闲置；
这说明连接生命周期存在独立缺口，不证明它导致本次170毫秒停顿。
临时连接向正式设备交接也复用该解绑路径，不能统一加close而误关新连接；本轮未修改或重启服务。

在用户确认戴着头显、Controller＋Send前台，且不标定／跟随／录制的条件下，
实际完成30秒同步只读TCP_INFO捕获：600个样本，采样间隔中位50.000毫秒，
计数读取平均14.335毫秒，仅诊断进程自身CPU26／nice19；不读取载荷或连接第二个SDK。
同时绑定头显到服务的63901连接、本采集器60061长流两端及既有回调遥测。
本轮复现27个不小于50毫秒的回调间隔。
独立分析中22个具有同一活动socket完整内侧采样的间隔，均出现头显上行TCP接收计数不增长、
接收队列为0；本机服务到采集器只有0或49字节小流量，没有持续大字节输入后仅Python回调停止的证据。
在完整读取窗口之外再增加20毫秒边界余量，仍有19个间隔得到相同结果。
另外5个跨连接替换／所有者发现盲区的间隔不作归因。

上述观测把可覆盖停顿缩小到数据到达PC接收边界之前／其上游，不能进一步等同于Wi-Fi丢包，
仍需头显发送端日志区分生成／发送暂停与无线／TCP延迟；也不将30秒暂停对照当作连续录制验收。
`trace_009/RECEIPT.json`、`TRANSPORT_008.json`及`TRACE_009_ANALYSIS.json`保存具体范围和限制。
用户摘下头显的时段已另行标记，其后停流不直接算作新的自发传输故障。控制和数据验收阈值保持原值。

### 头显日志与USB对照

USB连接后只读确认头显安装`com.xrobotoolkit.client`版本1.1.1，未重装App。
头显时钟比电脑约慢161秒，按ADB读取前后时间校准；单点校时不能排除历史漂移，
不据此声称毫秒级因果顺序。保存的Activity日志显示，前述18:45:55故障及
30秒Wi-Fi对照的27个短间隙均发生在Activity持续恢复的区间内；
18:40:40、18:46:13附近另有与Activity暂停相邻的长停流，二者分开判断。
原文及独立复核位于`round_001/fault_010_usb/USB_LOCAL_EVIDENCE_REVIEW_001.json`。

经用户明确同意，临时建立`adb reverse --no-rebind tcp:63901 tcp:63901`，
再由操作员在XRoboToolkit的PC Service手动地址中连接`127.0.0.1`并开启Controller＋Send。
实际核准服务63901的对端为loopback、原采集器回调持续推进且追踪有效，才开始计时；
仅修改IP或收到操作员确认不视为USB路径已生效。USB重新枚举会使转发失效，
本次曾实际发生，按空映射检查恢复后才测试；不覆盖其它映射、不重启服务或SDK。

20:32:05.404至20:32:35.405完成600个50毫秒间隔的只读样本。
睡眠请求前的首25秒没有达到50毫秒的回调间隙，未复现此前反复约170毫秒的模式。
睡眠过渡阶段有一次58.420毫秒的已结束间隙，随后至捕获结束至少4.434秒没有新回调，
这个尚未结束的停流必须单列，不能以“窗内已结束间隙最大58毫秒”宣称整段稳定通过。

同窗系统日志在约20:32:30.473明确记录`PowerGroup`因`reason=timeout`进入睡眠，
随后`wm_pause_activity`原因为`sleep`，App报告`OnApplicationPause True`并停止XR输入。
同一USB输入socket的接收计数也随之停止，转发映射仍在；
这次末段有系统睡眠的直接证据，不能归为仅PC SDK卡住，也不能推断操作员一定摘下头显。
当前只读标准设置项`screen_off_timeout`为60000毫秒，未修改电源／佩戴／休眠设置；
该读数不独立证明厂商电源服务的实际超时策略。
原文、计数及校时保留在`SYSTEM_PAUSE_WINDOW_011.json`、`SLEEP_SETTINGS_READONLY_011.json`、
`TRACE_011_OPERATOR_START.json`和`fault_008/trace_011_usb/`。

USB在有效发送区间的表现有所改善，但本次对照同时改变了会话及ADB转发调度，
不能据此唯一归因Wi-Fi，也未完成连续30秒或录制负载的稳定性验收。
采集保持暂停，成功001、1/100进度、官方100份初始化和全部控制／验收参数不变。
本次USB映射暂留；若撤销，仅移除本次创建的`tcp:63901`映射，
再将App连接地址恢复为电脑实际IP，不使用`adb reverse --remove-all`。
