# 4090：20k模型与SIMPLE实际验收

日期：2026-10-20，北京时间。基础Conda环境版本与安装问题见
[环境记录](4090_SIMPLE_ENVIRONMENT.md)，A100包协议见[资源交接](4090_SIMPLE_TRANSFER.md)。
本页只记录实际完成的阶段；资源存在、CPU测试和短时GPU检查均不等于正式任务成功率验收。
本页记录20k权重的环境、资源、模型和真实HTTP／物理／RTX诊断链路，没有产出正式成功率。
2026-10-21补证已确认实际使用官方SONIC v1.1；原训练token未逐值复算是数值追溯边界，
不再表述为版本未知。后续28k权重和本地任务适配独立记录，不能并入本页的20k结果。

## 资源身份与运行位置

- 通过已有SSH配置从 `115.191.40.142:56582` 只读拉取已发布的
  `unifolm-20k-261020043942/resources/`，没有读取训练数据、optimizer或训练RNG，
  没有修改A100训练、快照或运行参数。
- 本机资源根：`/home/zxc/CVPR_Evaluation_runtime/resources`；全部26项SHA通过，
  总字节 `14,916,113,138`。联合checkpoint固定step20000，SHA256为
  `a000c559a0de0a3be477933aa31ac5c681a5a79fe380c344adb89dde399d535a`。
  JSON与argv使用本机目录，provenance保留源路径仅作追溯。
- DINO逐文件SHA与训练缓存契约匹配；使用CLS＋4 register、历史6帧5Hz、clean256×320、
  慢版processor、不进行第二次resize/crop，NumPy1.26.4／Pillow11.3.0、BF16。
- SONIC取官方 `nvidia/GEAR-SONIC` 的v1.1发布模型，固定revision
  `6733128a3d8a523b1418b06bca3cdf61c8b0987f`。decoder SHA256为
  `34bae8570d4a4421a5391a5c2befd745d4a02d182ec539e5f9da44c091c67509`。
  本机 `sonic_v1_1/PROVENANCE.json`、`RELEASE_IDENTITY_COMPARISON.json` 保存上游来源与
  首次发布版本的权重同一性；encoder/decoder的CPU ONNX元数据契约为1751→64与994→29。
- 官方SIMPLE资源9个归档全部SHA／ZIP CRC通过，包含G1模型、graspnet、scene0、
  vMaterials与三个任务的固定dr0初始条件。所有运行资源在仓库外；不复制其他机器目录结构。
  任务为 `G1WholebodyXMovePickTeleop-v0`、`G1WholebodyXMoveBendPickTeleop-v0`、
  `G1WholebodyHandoverTeleop-v0`。本机 `simple/RESOURCE_PROVENANCE.json` 记录来源和哈希。
  资产固定[USC-PSI-Lab/SIMPLE](https://huggingface.co/datasets/USC-PSI-Lab/SIMPLE/tree/f5b58ab75210367c71ff7347155a656d403169a8)，
  初始条件固定[psi-data](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/tree/e78fb93cc28912a3031a10b8656d32d7f0a2b867/simple-eval)。

官方Psi0数据/训练脚本把SONIC-v1.1数据与 `action.body_token_v1_1` 联系起来，
记录在本机 `source_contracts/SONIC_VERSION_RELATIONSHIP.json`。
公开材料尚未提供该训练字段生成时的encoder逐文件SHA或可逐token复算样本。
因此当前控制器需要显式 `allow_unverified_compatibility` 诊断选项；不得伪造
`compatibility_verified=true` 或据此报告正式成功率。官方版本身份与训练token精确绑定是两项检查。

## 实际资源与接口修复

- `Simulator.get_data_dir()`原先另行硬编码仓库data，与已支持环境变量的utils实现不一致，
  真实Isaac加载G1 USD时才暴露。现统一委托utils，使用仓库外本机资源；默认／空白回退通过。
- Bend Pick的类uid原为 `g1_sonic_xmove_bend_pick_teleop`，与registry和官方30个固定
  episode的 `g1_wholebody_xmove_bend_pick_teleop` 不同，导致严格reset拒绝。
  已更正类uid，没有放宽Task.reset身份断言。
- 官方scene0主USD的props相对路径越过了资源根，另有black/Black大小写不一致。
  `prepare_simple_assets.py`仅在独立目录修正28个USD的资产路径，其余资源链接原件，
  不改变材质参数／几何，源28个USD SHA保持不变。派生scene CPU组合1758 prim、错误0。
  实际GPU运行目录为 `resources/simple_local_usd_v1`；最终脚本复现到v2，28个派生USD SHA
  与v1完全相同。`DERIVATION_MANIFEST.json`保留源／派生SHA及每次路径和大小写修正。
  这仅覆盖选定的scene0，不宣称其他场景和所有资源已验收。
- Handover原brasket USD的collision父Mesh有CollisionAPI却没有直接points，Isaac的
  GeometryPrim包装触发空碰撞错误。渲染侧改为只访问现有prim：移除空父Mesh的碰撞API，
  有效collider设为禁用，均为当前render stage override，不保存或改变源USD／可视几何。
  MuJoCo仍使用独立23份CoACD碰撞mesh；CPU实际编译通过，4529顶点／8966面，未改其物理。
- supervisor同时审核退出码与完整result.json，避免Kit关闭时exit0将异常计为完成；
  终止步重新检查物理有限性及倾斜。首次失败和中止日志均保留，未混入最终通过记录。

## 新增接线与检查

模型服务支持真实 `/observe`、`/reset` 与 `/act`，绑定127.0.0.1；Python3.10仿真客户端
禁用HTTP代理环境变量。历史按session/episode隔离，以 `mjData.time` 选取过去1秒内6个
不同的因果帧。缺历史返回425，时间倒退、同时间不同RGB、未reset换episode或过期推理返回409。
RGB几何处理一次后供DINO和VLM使用；冻结模型保留测试时RTC需要的局部梯度能力。

SIMPLE状态按关节名称重排43→45（末两项零填充），动作30×80拆为64 latent＋14手关节＋2 padding。
训练手关节顺序为thumb/middle/index，SIMPLE为thumb/index/middle；明确重排，padding不驱动颈部。
SONIC仅在50Hz控制点追加真实本体观测，最近10帧按官方term分组，与body64 latent共同组成decoder的994维输入；
这与encoder的1751维mode0期望参考窗口分开，物理子步执行PD但不额外追加历史帧。
29维输出重排为具名关节目标，再以物理步长执行力矩PD，不能送入旧36维WBC接口。
高层动作按30Hz仿真时间取样；同步HTTP等待期间仿真暂停，墙钟延迟与仿真时长分别记录。

模型环境补充固定 `pyarrow==20.0.0`：原26.0.0虽通过pip metadata检查，实际import要求
NumPy2，与本任务1.26.4冲突。降至20后实际import、训练argv/JSON解析和pip check均通过。

## 已完成的GPU模型检查

使用一张本机RTX4090、batch1、真实SIMPLE episode0相机与具名关节状态；
`capture_simple.py` 采集67帧、1.32秒仿真时间，时间间隔0.02秒。采集用测量姿态PD，
decoder调用0次，不是策略控制或任务成功记录。

`probe_joint.py` 验证26项SHA后，以原argv/JSON和严格子模块加载构造完整联合模型：

- step20000加载通过，参数数 `3,012,717,349`，加载48.61秒。
- 在线DINO输出 `[1,6,5,1024]`；Motion完整11×768 tokens及float32 mean通过；
  VLM完整94×2048 prefix tokens及有效mask mean通过；没有裁掉token代替pooling。
- 原始state `[1,1,45]`，10步采样得到有限 `[1,30,80]`，反归一化后有限。
- 本次DINO0.164秒、模型预测0.728秒；Torch峰值allocated10,005,783,040bytes、
  reserved10,271,850,496bytes。该峰值不包含其他进程或Isaac原生分配，不代表共同运行峰值。
  非独占GPU，保留已有imitator进程；不将单次测量视为稳定延迟或吞吐。

以上是模型检查，尚不能代表SONIC控制、SIMPLE任务成功或闭环稳定性。

## 真实HTTP与有界物理闭环

生产服务绑定 `127.0.0.1:21074`，step20000、batch1、Ta30、10步采样、RTC off，
模型与在线DINO使用本机GPU，SONIC ONNX decoder使用本机CPUExecutionProvider。
没有其他机器参与推理。HTTP replay使用真实67帧采集，两个episode各得到有限30×80动作；
首帧均返回425、重复时间动作和旧episode消息均返回409，reset后不会继承上一episode历史。
两次完整历史HTTP预测约0.335／0.348秒；这是回放验收，不是任务控制。

选定3个任务、dr0、固定seed0，MuJoCo各跑episode0和1，Isaac各跑episode0。
每个短episode均为5.02秒仿真、50帧真实历史预热、5个GPU预测chunk、201次真实SONIC解码。
动作按30Hz仿真时间取样，proprioception／decoder50Hz，物理dt0.005秒，相机50Hz。
同task第二episode复用session并清空图像／chunk／RTC／SONIC历史，检查通过。

- 六个MuJoCo短episode全部有限，墙钟7.29～10.32秒；三个Isaac短episode全部有限，
  控制循环墙钟77.72／80.44／85.27秒（分别Pick／Bend Pick／Handover，不含启动与载入）。
- Pick和Handover未触发本次跌倒诊断；Bend Pick的3个短episode均触发75°倾斜阈值，
  最大倾斜约85.68°～93.60°。高度阈值为0.35米。此为显式诊断条件，
  不把训练token绑定未验或短时动作的问题直接归咎于checkpoint质量。
- 九个短episode均未触发任务成功predicate；窗口仅5秒，不能报告正式0%成功率。
- Isaac三任务的动作HTTP墙钟约0.293～0.351秒；所有请求前后 `mjData.time`不变。
  全流程明显慢于物理实时，50Hz只指仿真时钟，不宣称实现50Hz墙钟控制或测评吞吐。
- 真实RTX RGB已查看，场景／物体／机器人可见。派生资产运行不再出现missing asset引用；
  Pick／Bend Pick日志的Error仅为3项Gymnasium UserWarning被Kit转记stderr：标量Box转换，
  以及现有环境重复返回同一info对象。未关闭checker隐藏问题，诊断脚本不修改info。
  首轮Handover还出现过一次启动纹理descriptor pool错误及空碰撞Mesh错误。
  修复空Mesh的render-stage处理后，Handover完整重跑5.02秒、201次decoder、5个chunk，
  物理有限、未触发倾斜或成功predicate，控制循环墙钟83.26秒；最终日志只有上述3项
  Gymnasium警告，没有空碰撞、missing asset、renderer Error或Traceback。
  最初一次纹理错误在该重测中未复现，其偶发性没有作独立长期稳定性验收。

20项CPU回归通过：模型服务10项（使用expert fixture）、控制契约8项（含真实MuJoCo及
固定C++参数对照）、报告审核2项。它们不替代上述真实GPU与控制执行。

## 显存测量与边界

`gpu-memory.csv`以约0.5秒周期采样整卡显存。三个模型与Isaac共同运行区间的整卡峰值
22,513MiB；停止模型服务后，独立Isaac真实任务／PD采集1.32秒、67帧的整卡峰值12,785MiB。
两个数字都包含原有imitator进程（约5,994MiB）及显示进程，不是各进程独占显存。
模型独立检查的Torch allocated／reserved峰值见上文；不能将Torch allocator统计与整卡采样
混为同一口径，采样亦不能保证捕获每一个瞬时峰值。本机实际报告49,140MiB，不把结果套用到
普通24GB显卡或解释为长时间稳定性保证。

正式测评仍缺训练token与SONIC encoder的精确绑定证据、完整episode／视频统计及足够样本。
当前只验证选定scene0与任务，不能据此声明所有Isaac材质或完全断网复现通过；
模型侧运行设置HF离线标志，所有实际模型与任务路径位于本机，A100仅用于最初的只读传输。
验收结束后已停止本次临时模型服务与显存采样进程，释放GPU；服务重启命令见工具说明。
未停止原有imitator或其他用户任务。

## 本机证据与运行入口

本机仓库外 `reports/joint20k/` 保存source-status、迁移日志、`verify-bundle.json`、
`simple-history.{npz,json,first.png,last.png}`、`joint-gpu.{json,log}` 与真实动作NPZ。
`http-probe/`、`loop-mujoco-three-tasks/`、`loop-isaac-derived/`与
`loop-handover-collision-fix/`分别保存实际HTTP、每episode trace／结果／图像／完整启动日志。
显存区间和口径见 `gpu-memory-summary.json`，仿真独立采集见 `sim-only-history.*`。
CPU接口回归日志为 `reports/dual_source_service_cpu_tests.log`、`simple-control-cpu-tests.log`与
`smoke-reporting-cpu-tests.log`；完整命令见
[工具说明](../scripts/4090_simple/README.md)。后续HTTP、物理与RTX检查须逐项记录，
不得把本页模型通过解释为完整测评通过。
