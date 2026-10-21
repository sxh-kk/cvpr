# CarryBox 本机适配 v1

Gym ID 为 `simple/G1WholebodyXMoveBendCarryBoxSonic-v0`，task UID 为
`g1_wholebody_xmove_bend_carry_box_sonic`。这是固定 SIMPLE 快照上独立增加的兼容任务，
来源为 [SIMPLE 官方任务](https://github.com/physical-superintelligence-lab/SIMPLE/blob/6d10628794d9c7de4596b4f2afb2c054a637c2bc/src/simple/tasks/g1_wholebody_xmove_bend_carry_box_sonic.py)。
保留 G1 SONIC、HSSD scene0、200 Hz 物理、50 Hz 相机与 110° 水平视场；
继续使用具名 43 维观测、v1.1 latent64 + hand14 + padding2 的既有适配器。
新增任务不改变模型归一化、相机历史或 checkpoint 协议。

箱体尺寸为 `[0.23, 0.38, 0.434]` 米、质量 0.1 kg，MuJoCo 直接使用 box 碰撞与解析惯量。
摩擦 `[0.8, 0.05, 0.005]`、`condim=4`、`solref=[0.005,2]`、棕色外观均对应固定官方代码。
Isaac 使用纯本地生成的 USD 网格，路径为
`$CVPR_SIMPLE_RUNTIME/resources/generated/carry_box_v1/<sha256>.usda`，没有外部材质或远端引用。
HSSD、机器人、材料仍从 `SIMPLE_DATA_DIR` 读取；推荐现有 `simple_local_usd_v1` 派生资产根。

机器人初始偏移 `[-1.2,0,0]`，箱体从地面开始，中心高度 0.217 米；
初始 x 均匀取 `[-0.6,-0.5]`、y 取 `[-0.05,0.05]`、yaw=0。
桌面尺寸 `[1.25,0.78999733,0.1]` 米，中心 `[0.3,0,0.4]`，真实上表面为 0.45 米。
房间与桌子固定；灯光和材料沿用官方范围，并由 reset seed 确定。
官方 50 个训练示范的实际箱体初始范围为 x `[-0.598055,-0.500204]`、
y `[-0.047371,0.049018]`、z=0.216905，符合此构造。
单纯末帧位置或 `next.done` 不证明机器人已释放箱体。

独立本地评估可事先固定 `split="val"`、seed `10001,10002,10003,10004,10005`，
并保存每次 `task.state_dict()` 与训练数据来源作对比。这是预注册的本地初态，
seed 数值本身不能证明没有训练泄漏。通过 `env.reset(seed=10001)` 构造；
Gym 初始化需要与既有任务相同的 `sonic_config`、`sim_mode`、`headless=True`。
重放 state 时使用 `dr_state_dict.spatial` 的初态，不能使用已经被仿真更新过的 `layout` 末态。
官方 eval 归档的旧 UID 是 `g1_wholebody_xmove_bend_pick_variant1_teleop`；
必须在单独、留痕的兼容配置中明确转换为上述 canonical UID，不能放松 reset 的 UID 校验。

成功判据版本为 `true_tabletop_continuous_simtime_v1`：箱体与桌面上表面接触（接触点高度容差
0.05 米）、中心高于桌面、双手无接触，按 50 Hz 检查连续保持至少 0.9 秒真实 MuJoCo 时间。
手掌碰撞网格融合在 `wrist_yaw` body 内，故手指和该 body 的接触均视为未释放。
`info["carry_box_progress"]` 返回各子条件、桌面/箱体高度、保持时间和成功状态。
同一时间反复查询不累计奖励；条件中断归零。此处明确修正官方源文件中的桌面中心高度判断，
并把官方累计 `0.02` 改为连续仿真时间，因此结果必须标为本地适配指标，不能直接声称官方基准分数。
模型在线控制仍须先采集完整真实一秒相机历史；teacher replay 的 0.2 秒 proprio warmup 不是模型历史。

只完成 CPU 构造检查不代表通过 GPU 渲染或任务评估。CPU 回归：

```bash
PYTHONNOUSERSITE=1 SIMPLE_DATA_DIR="$CVPR_SIMPLE_RUNTIME/resources/simple_local_usd_v1" \
  conda run -n cvpr-simple-sim310 python -m unittest discover \
  -s VLMExpert/third_party/SIMPLE/tests -p test_carry_box_local.py -v
```

已验证解析质量/惯量/摩擦/颜色、固定种子重现、初态 state roundtrip、地面和桌面高度、
真实 MuJoCo 接触成功判据。生成 USD 已通过 CPU composition 和尺寸检查；Isaac 渲染与实际策略效果须另行验收。
