# BendPick：原地弯腰抓取的PICO全身遥操

独立任务身份为`cvpr/G1SonicBendPickPico-v0`，来源为
`simple/G1WholebodyBendPickMP-v0`。采用现有`PicoSonicReference`全身SMPL mode2，
保留腰、髋、膝及身体俯仰参考；不使用Grasp的站姿腿部mode1规划器，不用摇杆行走。
PICO全身24点和同一包手指输入仍执行原跟踪、连续性与新鲜度检查。

## 弯腰能力与正式采集前检查

mode2会编码人体关节姿态和根部朝向；它不直接编码人体整体绝对位置或高度。
仅把头显整体往下移不能证明机器人会下探，需要真实腰髋膝动作和现场物理反馈。
机器人腰俯仰限位约±30度，更低抓取需要髋、膝和骨盆配合，不能只凭模式名称保证可达。

首次运行使用`--operator-test-only`：可以X标定、Y跟随和暂停，
UI及采集器均拒绝A起录，不产生真人演示或推进100份进度。
操作员先站直标定，跟随站立2–3秒，再小幅屈膝／弯腰并回正，
确认真实腰膝响应、手腕下降、足部稳定与新鲜画面后，再逐步靠近低桌目标。
这段现场测试通过后，重启同一进度并去掉该标志，开放正式录制。

原有站直校准、站立预热和实时保护保留：不为够低而放宽跟踪跳变、
骨盆最低高度或倾角保护。`--startup-check-only`的50步只检验初始化预热，
不代表PICO mode2实际弯腰闭环已经通过。

## 官方初始化与世界适配

官方TRAIN恰有100条episode0–99，元数据SHA为
`bcd7f25c9030287fac5c111810acb3f920dc5ee8c9e0719ac22e662a65b52528`。
冻结各原始environment_config字节，不读取教师动作或未来轨迹。
原始机器人为AMO `g1_wholebody`，物理500Hz；本适配为SONIC、200Hz物理／50Hz控制，
不宣称AMO动力学或控制等价。

原机器人spawn z=0.38米、桌顶z=0、地面z=−0.405米。
独立适配将整个世界上移0.405米，再补偿SONIC XML的0.793米根部挂载偏置；
不能使用Grasp的0.785米平移。实际初始骨盆约0.785米、桌顶0.405米、
饼干盒中心约0.518米，机器人初始水平距离目标约0.29–0.37米。
完整世界相机、场景与各物体一致平移；机器人局部相机遵循实际SONIC挂载。

原生task源码SHA为
`8401203aa20bdc6ede7634642aafc36ed5289e21e414dc584c43f7d91f7bdfe6`。
直接继承原reward/check_success：reward≥0.9，对应原生0.1米高度尺度下抬升约9厘米。
不增加持稳、接触或新几何成功规则；保存同一次env.step返回的实际flag、reward与基线。

## 启动与正式采集操作

```bash
bash scripts/4090_simple/start_pico_bend_pick.sh \
  --round-root /home/zxc/CVPR_Evaluation_runtime/pico_bend_pick_official_20261109/round_001 \
  --data-root /home/zxc/CVPR_Evaluation_runtime/pico_bend_pick_official_20261109/preparation_001/resources/data_root \
  --collection-plan /home/zxc/CVPR_Evaluation_runtime/pico_bend_pick_official_20261109/preparation_001/official100_plan/manifest.json \
  --operator-test-only
```

界面沿用`http://127.0.0.1:43100/`。PICO保持XRoboToolkit前台、Controller与全身跟踪、Send开启。
试跟随通过后去掉`--operator-test-only`，从暂停状态按以下顺序正式采集：

1. 站直X校准、Y跟随。
2. 站直起录，先保持1秒，再原地适度屈膝、弯腰接近低桌目标。
3. 用同侧扳机或侧握抓住饼干盒，向上抬起；建议稳住约1秒再A保存。
4. 后台预复核通过后进入下一份初始化，再次站直X校准。

初始观察1秒是采集建议，不修改原生成功判据。
使用独立target-only证据合同`cvpr-live-pico-bend-pick-deferred-physical-capture-v1`，
保存真实64维mode2 token、14维手指指令、43维关节状态、完整积分状态、
200Hz零外力、50Hz原生质量和30Hz采集及SHA。所有干扰物仍在完整世界状态中。
原始采集不等于RTX、独立训练准入、打包、DINO、上传或训练就绪。

## 本机已验证范围

独立scene0资源overlay静态检查覆盖76个USD顶层、291文件依赖闭包，保存的材质引用无缺失；
MDL运行时编译与RTX仍需后续实际检查。任务单元验证覆盖原生方法不变、100份初始化绑定、
实际双XML的零步进几何与真实SIMPLE重置。
`preparation_001/startup_check_001`完成50个真实控制步、完整场景binding及新鲜EGL预览，
合并的实际元数据通过Bend身份校验；SDK与录制均未在此检查中启动。
现场操作员的全身姿态跟随与低桌可达检查另行记录，未通过前不开始正式采集。
