"""Capture genuine SIMPLE RGB/state history from a fixed official episode.

This uses measured-pose torque PD for history collection, not policy actions.
Its output supports a standalone 20k model probe; it is not a closed-loop
evaluation or evidence that the robot completes a task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


def remap_episode_paths(value, data_root: Path):
    """Rebase known upstream SIMPLE data paths without recreating old hosts."""
    if isinstance(value, dict):
        return {key: remap_episode_paths(item, data_root) for key, item in value.items()}
    if isinstance(value, list):
        return [remap_episode_paths(item, data_root) for item in value]
    if isinstance(value, str):
        marker = "/SIMPLE/data/"
        if marker in value:
            return str(data_root / value.split(marker, 1)[1])
        if value.endswith("/SIMPLE/data"):
            return str(data_root)
        if value.startswith("data/"):
            return str(data_root / value[5:])
    return value


def load_fixed_episode(path: Path, episode_index: int, data_root: Path):
    with path.open() as stream:
        for line in stream:
            episode = json.loads(line)
            if episode["episode_index"] == episode_index:
                original = json.loads(episode["environment_config"])
                return remap_episode_paths(original, data_root), episode
    raise ValueError(f"Episode {episode_index} missing from {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sonic-resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="G1WholebodyXMovePickTeleop-v0")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dr-level", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=1.3)
    parser.add_argument("--sim-mode", choices=("mujoco", "mujoco_isaac"), default="mujoco_isaac")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 1.1:
        parser.error("Collect at least 1.1 seconds to form six causal 5Hz observations")
    args.data_root = args.data_root.resolve(strict=True)
    if (not args.output.is_absolute() or
            args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2])):
        parser.error("--output must be an absolute path outside the repository")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ["SIMPLE_DATA_DIR"] = str(args.data_root)
    import numpy as np
    from PIL import Image
    import gymnasium as gym
    import simple.envs  # register environments; SimulationApp lifecycle remains in BaseDualSim
    from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
    from simple.baselines.dual_source_psi0 import TRAINING_STATE_JOINT_NAMES, state43_to45
    from simple.baselines.sonic_v1_1 import SonicV11OnnxController

    metadata_path = args.data_root / "eval" / args.task / f"dr-level-{args.dr_level}" / "meta/episodes.jsonl"
    env_config, source_episode = load_fixed_episode(metadata_path, args.episode, args.data_root)
    controller = SonicV11OnnxController(args.sonic_resources, allow_unverified_compatibility=True)
    config = SimLoopConfig().load_wbc_yaml()
    config["ENV_NAME"] = "simple"
    config["SIMULATE_DT"] = float(env_config["metadata"]["physics_dt"])
    env = None
    started = time.perf_counter()
    try:
        env = gym.make("simple/" + args.task, sonic_config=config, sim_mode=args.sim_mode,
                       headless=True, dr_level=args.dr_level)
        observation, info = env.reset(seed=args.seed, options={"state_dict": env_config, "dr_level": args.dr_level})
        controller.reset()
        frames, timestamps, states = [], [], []
        first_time = float(observation["sim_time_s"])
        while True:
            frames.append(np.asarray(observation["head_stereo_left"]).copy())
            timestamps.append(float(observation["sim_time_s"]))
            states.append(state43_to45(observation["joint_qpos"], observation["joint_names"])[0, :43])
            if timestamps[-1] - first_time >= args.seconds:
                break
            observation, reward, terminated, truncated, info = env.step(controller.hold(observation))
            if terminated or truncated:
                raise RuntimeError("Episode ended before the real history capture completed")
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if not np.all(np.diff(timestamps) > 0) or np.max(np.diff(timestamps)) > 1 / 30 + .001:
            raise ValueError("Capture did not produce chronological real camera observations at >=30Hz")
        state = env.unwrapped
        provenance = {
            "purpose": "measured-pose PD history capture; not policy control or completed task evaluation",
            "episode_id": f"{args.task}:dr{args.dr_level}:episode{args.episode}:seed{args.seed}",
            "task": args.task, "episode_index": args.episode, "seed": args.seed, "dr_level": args.dr_level,
            "source": "official SIMPLE Psi0 benchmark fixed episode environment_config",
            "source_metadata_path": str(metadata_path),
            "source_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "source_episode": source_episode, "local_environment_config": env_config,
            "instruction": env.unwrapped.task.instruction,
            "sim_mode": args.sim_mode, "physics_dt_s": float(state.mujoco.mjModel.opt.timestep),
            "requested_render_hz": float(state.task.render_hz),
            "observed_sim_frame_period_min_s": float(np.min(np.diff(timestamps))),
            "observed_sim_frame_period_max_s": float(np.max(np.diff(timestamps))),
            "state43_joint_names": list(TRAINING_STATE_JOINT_NAMES), "state_units": "radians",
            "timestamp_source": "MuJoCo mjData.time", "frame_count": len(frames),
            "controller": controller.identity, "decoder_calls": controller.decoder_calls,
            "wall_seconds": time.perf_counter() - started,
        }
        with args.output.open("wb") as stream:
            np.savez_compressed(stream, rgb=np.stack(frames), sim_time_s=timestamps,
                                state43=np.stack(states),
                                episode_id=np.asarray(provenance["episode_id"]),
                                source_metadata_json=np.asarray(json.dumps(provenance)))
        args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
        Image.fromarray(frames[0]).save(args.output.with_suffix(".first.png"))
        Image.fromarray(frames[-1]).save(args.output.with_suffix(".last.png"))
        print(json.dumps({"status": "real_history_captured", "npz": str(args.output),
                          "frames": len(frames), "sim_seconds": float(timestamps[-1] - timestamps[0]),
                          "decoder_calls": controller.decoder_calls,
                          "boundary": provenance["purpose"]}), flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
