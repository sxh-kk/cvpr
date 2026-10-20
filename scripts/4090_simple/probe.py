"""Weight-free probes. A PASS here never means the 20k policy passed evaluation."""
import argparse
import atexit
import importlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone


def cuda():
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        y = x @ x.T
        assert torch.isfinite(y).all()
        torch.cuda.synchronize()
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            "bf16": True, "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI}


def model(output):
    import torch
    from flash_attn import flash_attn_func

    result = cuda()
    sources = {}
    for name in ("ImageProcess", "MotionExpert", "egomotion_pipeline", "psi",
                 "psi.models.psi0", "psi.models.frozen_motion", "psi.models.dual_source_action",
                 "psi.deploy.serve_psi0_simple", "lerobot"):
        module = importlib.import_module(name)
        sources[name] = module.__file__
    with torch.inference_mode():
        torch.manual_seed(42)
        q, k, v = [torch.randn(1, 128, 8, 64, device="cuda", dtype=torch.bfloat16)
                   for _ in range(3)]
        actual = flash_attn_func(q, k, v, dropout_p=0, causal=False)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2).float(), k.transpose(1, 2).float(), v.transpose(1, 2).float()
        ).transpose(1, 2)
        error = (actual.float() - reference).abs().max().item()
        assert torch.isfinite(actual).all() and error < 0.02, error
    result.update(sources=sources, flash_attention_max_abs_error=error,
                  flash_attention_shape=list(actual.shape),
                  boundary="imports and synthetic BF16/FlashAttention only; no pretrained inference")
    return result


def mujoco_probe(output):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from PIL import Image

    result = cuda()
    m = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <light pos="0 0 3"/><geom type="plane" size="3 3 .1" rgba=".3 .3 .3 1"/>
      <camera name="camera" pos="2 -2 2" xyaxes=".707 .707 0 -.408 .408 .816"/>
      <body pos="0 0 1"><freejoint/><geom type="box" size=".1 .1 .1" mass="1" rgba="1 .1 .1 1"/></body>
      </worldbody></mujoco>''')
    d = mujoco.MjData(m)
    initial = float(d.qpos[2])
    for _ in range(100):
        mujoco.mj_step(m, d)
    assert np.isfinite(d.qpos).all() and d.qpos[2] < initial
    with mujoco.Renderer(m, height=240, width=320) as renderer:
        renderer.update_scene(d, camera="camera")
        rgb = renderer.render().copy()
    assert rgb.shape == (240, 320, 3) and rgb.dtype == np.uint8 and rgb.std() > 1
    Image.fromarray(rgb).save(output / "mujoco-rgb.png")
    result.update(mujoco=mujoco.__version__, steps=100, simulated_seconds=d.time,
                  final_height=float(d.qpos[2]), rgb_shape=list(rgb.shape), rgb_std=float(rgb.std()),
                  boundary="primitive scene physics and EGL camera; no SIMPLE task or controller")
    return result


def curobo(output):
    import torch
    import numpy as np
    import pinocchio as pin
    import curobo as package
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig

    result = cuda()
    extensions = {}
    for name in ("lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu"):
        extensions[name] = importlib.import_module("curobo.curobolib." + name).__file__
    urdf = Path(package.__file__).parent / "content/assets/robot/ur_description/ur5e.urdf"
    config = CudaRobotModelConfig.from_basic_urdf(str(urdf), "base_link", "tool0")
    robot = CudaRobotModel(config)
    cpu_model = pin.buildModelFromUrdf(str(urdf))
    cpu_data = cpu_model.createData()
    joint_names = list(robot.joint_names)
    errors = []
    with torch.inference_mode():
        for value in (0., .1):
            q = torch.full((1, robot.get_dof()), value, device="cuda", dtype=torch.float32)
            state = robot.get_state(q)
            assert torch.isfinite(state.ee_position).all() and torch.isfinite(state.ee_quaternion).all()
            cpu_q = np.zeros(cpu_model.nq)
            for column, name in enumerate(joint_names):
                cpu_q[cpu_model.joints[cpu_model.getJointId(name)].idx_q] = value
            pin.framesForwardKinematics(cpu_model, cpu_data, cpu_q)
            reference = cpu_data.oMf[cpu_model.getFrameId("tool0")].translation
            error = float(np.max(np.abs(state.ee_position.cpu().numpy()[0] - reference)))
            assert error < 1e-4, error
            errors.append(error)
    result.update(extensions=extensions, urdf=str(urdf), batch=1, poses=2,
                  max_position_error_m=max(errors),
                  boundary="bundled UR5e CUDA FK against CPU Pinocchio; no G1 policy, IK or controller")
    return result


def isaac(output):
    # Native torch libraries must load before Kit; Omniverse APIs only after SimulationApp.
    import torch
    from curobo.types.state import JointState  # preload before Kit, as SIMPLE requires
    from isaacsim import SimulationApp
    import simple
    # simple.engines.__init__ imports Omni APIs eagerly. Load the standalone helper
    # without executing that package before SimulationApp exists.
    helper_path = Path(simple.__file__).parent / "engines" / "isaac_app.py"
    spec = importlib.util.spec_from_file_location("simple_isaac_app_probe", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    app = helper.create_simulation_app(SimulationApp, headless=True, width=320, height=240)
    # Kit shutdown can terminate Python before returning. Persist results in main
    # before interpreter cleanup; never leave a successful camera probe as 'running'.
    atexit.register(app.close)
    engine = importlib.import_module("simple.engines.isaacsim")
    import numpy as np
    import omni.usd
    from PIL import Image
    from omni.isaac.core import World
    from omni.isaac.core.objects import FixedCuboid
    from omni.isaac.sensor import Camera
    from pxr import UsdLux
    from omni.isaac.core.utils.rotations import euler_angles_to_quat

    world = World(stage_units_in_meters=1.0)
    # Local USD primitives avoid online asset/material downloads.
    light = UsdLux.DomeLight.Define(omni.usd.get_context().get_stage(), "/World/Light")
    light.CreateIntensityAttr(1500)
    world.scene.add(FixedCuboid(prim_path="/World/Box", name="box", size=.4,
                               position=np.array([0., 0., .5]), color=np.array([1., .1, .1])))
    camera = Camera(prim_path="/World/Camera", resolution=(320, 240), frequency=30)
    camera.set_world_pose(position=np.array([2., 0., .5]),
                          orientation=euler_angles_to_quat(np.array([0., 0., 180.]), degrees=True),
                          camera_axes="world")
    world.reset()
    camera.initialize()
    for _ in range(90):
        world.step(render=True)
    rgba = camera.get_rgba()
    assert rgba is not None and rgba.shape == (240, 320, 4), getattr(rgba, "shape", None)
    rgb = rgba[:, :, :3].astype(np.uint8)
    assert rgb.std() > 1, "camera is blank"
    Image.fromarray(rgb).save(output / "isaac-rgb.png")
    return {"isaacsim": importlib.metadata.version("isaacsim"), "rgb_shape": list(rgb.shape),
            "rgb_std": float(rgb.std()), "steps": 90,
            "simple_engine_source": engine.__file__,
            "boundary": "headless RTX primitive camera using SIMPLE app helper; no task assets"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["model", "mujoco", "curobo", "isaac"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"{args.stage}.json"
    base = {"stage": args.stage, "python": sys.version, "executable": sys.executable,
            "started_at": datetime.now(timezone.utc).isoformat()}
    path.write_text(json.dumps(dict(base, status="running"), indent=2) + "\n")
    try:
        result = globals()[{"mujoco": "mujoco_probe"}.get(args.stage, args.stage)](args.output)
    except Exception as exc:
        path.write_text(json.dumps(dict(base, status="failed", error=repr(exc)), indent=2) + "\n")
        raise
    result.update(base, status="passed")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
