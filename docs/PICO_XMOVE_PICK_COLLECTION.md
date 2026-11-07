# XMovePick：PICO＋SONIC走近桌面抓取

本入口使用独立身份 `cvpr/G1SonicXMovePickPico-v0`，对应原任务
`simple/G1WholebodyXMovePickTeleop-v0`，原生 `g1_sonic`、200Hz物理、50Hz控制。
使用现有SONIC v1.1 encoder/decoder和连续VR mode1，双臂始终跟随、摇杆规划腿部行走。
这不是Grasp站姿任务，也不宣称重现原档案的遥操作控制器。

## 初始化与成功判据

源TRAIN元数据固定SHA为
`a2bfcde99cb7de9ba32855271ded5a225db3d2a7d230f82c6b0ac3421b2e1afb`，
实际包含99份初始化（episode0–98）。进度显示0/99；不能写成100种官方初始化。
如后续需要100条真人演示，额外重复初始化的演示必须单独标记来源。

冻结计划保留原元数据及原始environment_config字节。原layout中的物体姿态可能已经变化，
因此派生reset按原`dr_state_dict.spatial`的资产UID恢复目标和3个干扰物，
并按`dr_state_dict.lighting`恢复灯光，逐字段记录来源。未读取任何示范动作或未来状态。
世界坐标、桌面、机器人spawn不平移，不采用Grasp的AMO地面偏移。

99份源的材质DR均为None。本地任务只恢复layout已保存的桌面与物体材质，
不重新随机材质；未保存的地面与机器人材质使用后端默认，适配收据明确记录这一限制。
不能宣称这些未保存的视觉字段与原采集完全等价。

原任务源码SHA为
`aae4858775cdbcbddf31a3fe432e80d91519a316b0068e93b7024e1aafad6735`。
继承原`compute_reward`和`check_success`方法：reward≥0.9，
原生高度归一化量0.1米，因此对应抬高约9厘米。保存同一次env.step返回的真实reward、
success flag及原生高度基线，不添加持稳、行走距离或容器条件。

## 启动与操作

```bash
bash scripts/4090_simple/start_pico_xmove_pick.sh \
  --round-root /home/zxc/CVPR_Evaluation_runtime/pico_xmove_pick_official_20261107/round_001 \
  --data-root /home/zxc/CVPR_Evaluation_runtime/pico_xmove_pick_official_20261107/preparation_001/resources/data_root \
  --collection-plan /home/zxc/CVPR_Evaluation_runtime/pico_xmove_pick_official_20261107/preparation_001/official99_plan/manifest.json
```

界面为 `http://127.0.0.1:43100/`，显示“XMovePick 走近桌面抓取数采”。
从暂停状态启动，不自动标定、跟随或录制。PICO保持XRoboToolkit前台及Send开启，
关闭A键切换Send。USB反向连接与既有PC接收服务不变。

1. 双脚站稳、摇杆松开，X校准，然后Y开始跟随。
2. 在起点、走近桌子之前按A起录，保持初始姿态1秒。
3. 左杆前进、右杆转向；双臂持续跟随，无需B切换。松杆站稳，再抓目标饼干盒。
4. 左右扳机或侧握控制同侧手指；抬起后建议观察约1秒，再按A保存。
5. 原生任务成功及录制完整性复核后进入下一份初始化，再次X校准。
   未成功重试本份；技术故障保留原件并暂停。

必须从起点起录，才能覆盖走近过程。建议的观察时间不会变成新的任务成功门槛。
开始采集前可用独立round目录及`--startup-check-only --port 43102`运行50个真实站立控制步
与新鲜EGL预览验收；该模式不启动SDK、不录制、不占用正式采集进度。

## 证据与后处理边界

使用独立target-only合同 `cvpr-live-pico-xmove-pick-deferred-physical-capture-v1`。
保存200Hz零外力、50Hz原生任务样本、30Hz命令/状态、完整float64积分状态及SHA，
所有干扰物仍包含在完整物理状态中。写入队列上限1024，封存仍要求完整FIFO和全量ACK，
没有放宽新鲜度、跟踪跳变或录制完整性门槛。

采集数据、编号和进度与Grasp、Handover、Mobile分开。原生成功只代表原始采集预验收，
EGL画面是操作反馈；RTX补图、独立训练准入、无损打包、DINO和上传另行处理。
本轮不修改公共`render_deferred_pico.py`，保留既有Grasp补图的源码SHA绑定。

## 本机准备验收

`preparation_001/resources/RESOURCE_PREPARATION_001.json`记录独立scene0资源overlay，
99份初始化引用的72个物体、80种直接材质、365个递归依赖文件均存在，
73个USD起点CPU composition无错误。该检查不替代RTX运行时验收。

首次检查因源绝对资产路径尚未重绑而失败，原日志保留。
修复后`startup_check_002`通过50个真实控制步、站立、完整场景binding与新鲜零步EGL预览；
实际合并的episode元数据通过XMovePick质量身份校验。未在这份准备检查中执行行走或人类抓取，
不把站立启动检查报告为任务成功率或正式演示。
