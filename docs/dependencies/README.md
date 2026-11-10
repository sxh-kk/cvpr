# 第三方代码

| 组件 | 来源 | 仓库中的位置 |
| --- | --- | --- |
| Ψ₀ | [Psi0](https://github.com/physical-superintelligence-lab/Psi0) | `VLMExpert/src/psi/` |
| 全身控制与 SONIC | [GR00T-WholeBodyControl](https://github.com/physical-superintelligence-lab/GR00T-WholeBodyControl) | `VLMExpert/third_party/GR00T-WholeBodyControl/` |
| 仿真与机器人任务 | [SIMPLE](https://github.com/physical-superintelligence-lab/SIMPLE) | `VLMExpert/third_party/SIMPLE/` |

本仓库在 Ψ₀ 上加入双来源 Action Expert、MotionExpert 联合训练和 DINO 历史特征读取。
其他基线的使用说明见 [baselines](../../VLMExpert/baselines/README.md)。固定版本与完整来源见 [上游清单](../psi0_upstream.json)。

第三方源码已包含在仓库中；`.gitmodules` 保留上游记录，无需运行 `git submodule update`。

## 许可证

各第三方组件沿用其原许可证：

- [Ψ₀](../../VLMExpert/LICENSE)
- [GR00T-WholeBodyControl](../../VLMExpert/third_party/GR00T-WholeBodyControl/LICENSE)
- [cuRobo 代码](../../VLMExpert/third_party/SIMPLE/third_party/curobo/LICENSE)
- [cuRobo 资产](../../VLMExpert/third_party/SIMPLE/third_party/curobo/LICENSE_ASSETS)

模型、数据和机器人资产遵循各自来源的使用条款。
