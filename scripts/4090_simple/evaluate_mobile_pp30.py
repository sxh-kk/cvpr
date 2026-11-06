"""Thirty fixed SIMPLE Mobile cases; adapted SONIC diagnostic, never a new seed benchmark.

Every case owns a fresh model server and simulator. Technical failures retain all
evidence and leave the campaign incomplete; native False is a valid task result.
Importing this module uses only the standard library and starts no processes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
from urllib.error import URLError
from urllib.request import build_opener, ProxyHandler

REPO = Path(__file__).resolve().parents[2]
TASK = "G1WholebodyLocomotionPickBetweenTablesTeleop-v0"
TASK_UID = "g1_wholebody_locomotion_pick_between_tables_teleop"
CONTRACT = "cvpr-mobile-pp-fixed-official30-adapted-diagnostic-v1"
READY_CONTRACT = "cvpr-mobile-pp-official100-stopped-local-deployment-v1"
READY_STATUS = "local_mobile_pp_official100_stopped_checkpoint_prepared"
DEPLOYMENT_ARTIFACTS = frozenset(("run_config.json", "argv.txt", "dataset_statistics.json", "dino_training_contract.json",
                                "motion/joint_motion.ckpt", "motion/export_validation.json", "component_load_validation.json"))
OFFICIAL_REVISION = "e78fb93cc28912a3031a10b8656d32d7f0a2b867"
OFFICIAL_METADATA_SHA256 = (
    "ac6a3d7f64a7a3db55321ab3d1a3371ff043cabc9b09b0d57c5a53c542fd9cf2",
    "8ab31d73152ae8577581e30ecae06e1c88bc5416ab73075fac9324b7a4c7c312",
    "60bbc35be1b6a5b380d196b9f1b65ae793efff252fdbad2342d5dcefa3c1397f")
MODEL_PY = Path("/home/zxc/miniconda3/envs/cvpr-simple-model311/bin/python")
SIM_PY = Path("/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python")
BOUNDARY = ("Official Level0/1/2 fixed cases with dual-source SONIC controller adaptation; "
            "not a certified original-controller benchmark or new-scene generalization score")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_identity(path):
    path = Path(path).resolve(strict=True)
    before = path.stat()
    digest = sha(path)
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), f"File changed while reading: {path}")
    return {"path": str(path), "sha256": digest, "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "device": after.st_dev, "inode": after.st_ino}


def unchanged(record):
    value = Path(record["path"]).stat()
    require((value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns) ==
            (record["device"], record["inode"], record["size_bytes"], record["mtime_ns"]),
            "Frozen input changed: " + record["path"])


def evaluation_policy(args):
    """Explicit inference change; the checkpoint and physical protocol stay fixed."""
    mode = getattr(args, "rtc_mode", "off")
    steps = getattr(args, "execute_steps", None)
    if steps is None:
        steps = 15 if mode == "test_time" else 30
    require(mode in ("off", "test_time"), "Only off or test_time RTC evaluation is supported")
    require(type(steps) is int and 1 <= steps <= 30 and steps * 50 % 30 == 0,
            "Execution window must align exactly to both 30Hz actions and 50Hz control ticks")
    require(mode != "test_time" or steps < 30, "Test-time RTC needs a nonempty previous-plan overlap")
    minimum = getattr(args, "rtc_min_exec_horizon", None)
    require(mode == "test_time" or minimum is None, "RTC minimum horizon requires test_time mode")
    horizon = steps if minimum is None else minimum
    require(mode != "test_time" or type(horizon) is int and steps <= horizon <= 30,
            "RTC minimum horizon must satisfy execute_steps <= s <=30")
    return {"rtc_mode": mode, "rtc_enabled": mode == "test_time", "prediction_horizon": 30,
            "execute_steps": steps, "inference_steps": 10, "action_hz": 30, "control_hz": 50,
            "control_ticks_per_window": steps * 50 // 30,
            "rtc_exec_shift": steps if mode == "test_time" else None,
            "rtc_inference_delay": 0 if mode == "test_time" else None,
            "rtc_min_exec_horizon": horizon if mode == "test_time" else None,
            "pig_mask_schedule": "exponential" if mode == "test_time" else None,
            "pig_guidance_alpha": .9 if mode == "test_time" else None,
            "simulation_paused_during_HTTP": True,
            "scope": "synchronous previous-plan continuity guidance; not asynchronous latency compensation"}


def evaluation_budget(args):
    from run_closed_loop import mobile_time_budget
    return mobile_time_budget(getattr(args, "seconds", 24))


def freeze_inputs(args):
    """CPU preparation gate; no model construction or simulator import."""
    from capture_simple import remap_episode_paths
    ready_path = args.deployment / "LOCAL_RUN_READY.json"
    ready = read(ready_path)
    require(ready.get("contract") == READY_CONTRACT and ready.get("status") == READY_STATUS,
            "Deployment does not have the accepted stopped-checkpoint CPU READY contract")
    require(ready.get("task_id") == "simple/" + TASK, "Deployment task differs")
    require(ready.get("task_step") == args.checkpoint_step and ready.get("step") == args.checkpoint_step,
            "Deployment checkpoint step differs")
    require(ready.get("joint_sha256") == args.expected_model_sha256, "Deployment model identity differs")
    require(ready.get("training_completed") is False and ready.get("checkpoint_save_completed") is True
            and ready.get("stopped_by_user") is True, "Stopped run/save completion identity is absent")
    require(ready.get("component_load_validation", {}).get("status") == "passed",
            "Deployment strict CPU component load has not passed")
    manifest = file_identity(args.training_manifest)
    require(ready.get("training_manifest_sha256") == manifest["sha256"], "Training DINO manifest differs from deployment")
    encoder = read(args.training_manifest).get("encoder")
    require(isinstance(encoder, dict) and encoder.get("shape") == [5, 1024], "Training encoder contract missing")
    model = file_identity(args.deployment / "checkpoints" / f"ckpt_{args.checkpoint_step}" / "model.safetensors")
    require(model["sha256"] == args.expected_model_sha256, "Actual checkpoint bytes differ from requested SHA")
    source = {name: file_identity(REPO / "scripts/4090_simple" / name) for name in
              ("evaluate_mobile_pp30.py", "serve_evaluation.py", "run_closed_loop.py", "capture_simple.py", "video_capture.py")}
    for relative in ("deploy/serve_psi0_simple.py", "deploy/dual_source_runtime.py", "models/psi0.py", "models/dual_source_action.py"):
        source["psi/" + relative] = file_identity(REPO / "VLMExpert/src/psi" / relative)
    for relative in ("tasks/g1_wholebody_locomotion_pick_between_tables_teleop.py", "envs/sonic_loco_manip.py",
                     "engines/mujoco.py", "assets/objects.py", "assets/graspnet.py", "assets/collision_meshes.py",
                     "envs/base_dual_env.py", "baselines/dual_source_psi0.py", "baselines/sonic_v1_1.py",
                     "baselines/live_pose_sonic_warmup.py", "baselines/warmup_wrist_reference.py"):
        source["SIMPLE/" + relative] = file_identity(REPO / "VLMExpert/third_party/SIMPLE/src/simple" / relative)
    source["deployment_ready"] = file_identity(ready_path)
    artifacts = ready.get("deployment_artifacts")
    require(isinstance(artifacts, dict) and set(artifacts) == DEPLOYMENT_ARTIFACTS,
            "Deployment READY must bind the seven complete runtime artifacts")
    for name, record in artifacts.items():
        path = args.deployment / name
        require(path.resolve(strict=True).is_relative_to(args.deployment), "Deployment artifact escapes its directory")
        identity = file_identity(path)
        require(identity["sha256"] == record.get("sha256") and identity["size_bytes"] == record.get("bytes"),
                "Deployment artifact differs from CPU preparation: " + name)
        source["deployment/" + name] = identity
    cases, metadata_files = [], []
    for level in range(3):
        path = args.data_root / "eval" / TASK / f"dr-level-{level}" / "meta/episodes.jsonl"
        identity = file_identity(path)
        require(identity["sha256"] == OFFICIAL_METADATA_SHA256[level], f"Level{level} is not the frozen official metadata")
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(len(rows) == 10 and sorted(row["episode_index"] for row in rows) == list(range(10)),
                f"Level{level} must contain exactly the ten official indices 0..9")
        for row in sorted(rows, key=lambda item: item["episode_index"]):
            config = json.loads(row["environment_config"])
            metadata = config["metadata"]
            require(config.get("uid") == TASK_UID and metadata.get("physics_dt") == .005
                    and metadata.get("render_hz") == 50 and metadata.get("max_episode_steps") == 1200
                    and metadata.get("success_criteria") == .4, "Official Mobile metadata protocol differs")
            cases.append({"level": level, "case_index": row["episode_index"],
                          "flat_index": level * 10 + row["episode_index"], "source_row": row,
                          "local_environment_config": remap_episode_paths(config, args.data_root),
                          "metadata_sha256": identity["sha256"], "environment_seed": 0,
                          "sampling_seed": args.sampling_seed + level * 10 + row["episode_index"]})
        unchanged(identity)
        metadata_files.append(identity)
    arm_profile = getattr(args, "warmup_arm_profile", "legacy")
    require(arm_profile in ("legacy", "wrist_retreat_v1"), "Unknown warmup arm profile")
    budget = evaluation_budget(args)
    boundary = BOUNDARY + ("; explicit30s extension, not the original24s budget"
                           if budget["budget_variant"] == "extended30s" else "")
    return {"contract": CONTRACT, "prepared_utc": now(), "task_id": "simple/" + TASK,
            "warmup_arm_profile": arm_profile,
            "checkpoint_step": args.checkpoint_step, "model": model, "deployment": str(args.deployment),
            "training_manifest": manifest, "metadata_files": metadata_files, "sources": source,
            "official_repository": "USC-PSI-Lab/psi-data", "official_revision": OFFICIAL_REVISION,
            "cases": cases, "boundary": boundary, "formal_evaluation_eligible": False,
            "time_budget": budget,
            "maximum_steps_including_warmup": budget["max_episode_steps"],
            "maximum_simulated_seconds": budget["maximum_simulated_seconds"],
            "effective_native_success_criteria": .9, "source_archive_success_criteria": .4,
            "evaluation_policy": evaluation_policy(args)}


def environment(args, python):
    runtime = args.runtime_root
    value = os.environ.copy()
    value.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", OMNI_KIT_ACCEPT_EULA="YES",
                 OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
                 TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                 CUDA_VISIBLE_DEVICES="0", MUJOCO_GL="egl", SIMPLE_DATA_DIR=str(args.data_root),
                 HF_HOME=str(runtime / "cache/huggingface"), TORCH_HOME=str(runtime / "cache/torch"),
                 TORCH_EXTENSIONS_DIR=str(runtime / "cache/torch_extensions"), XDG_CACHE_HOME=str(runtime / "cache/xdg"),
                 SIMPLE_ISAAC_EXPERIENCE=str(runtime / "apps/simple4090.kit"),
                 SIMPLE_ISAAC_PRELOAD_PINOCCHIO="1", CVPR_SIMPLE_RUNTIME=str(runtime),
                 PYTHONPATH=os.pathsep.join(str(REPO / item) for item in
                     (".", "VLMExpert/src", "VLMExpert", "MotionExpert/src", "pipeline/src", "VLMExpert/third_party/SIMPLE/src")),
                 PATH=str(python.parent) + os.pathsep + os.environ.get("PATH", ""))
    return value


def commands(args, case, directory):
    policy = evaluation_policy(args)
    budget = evaluation_budget(args)
    options = ["--rtc-mode", policy["rtc_mode"], "--execute-steps", str(policy["execute_steps"])]
    if policy["rtc_enabled"]:
        options.extend(["--rtc-inference-delay", "0", "--min-exec-horizon", str(policy["rtc_min_exec_horizon"]),
                        "--pig-mask-schedule", "exponential", "--pig-guidance-alpha", "0.9"])
    server = [str(MODEL_PY), str(REPO / "scripts/4090_simple/serve_evaluation.py"),
              "--run-dir", str(args.deployment), "--ckpt-step", str(args.checkpoint_step),
              "--sampling-seed", str(case["sampling_seed"]), "--output", str(directory / "server"),
              "--port", str(args.port), *options]
    simulator = [str(SIM_PY), str(REPO / "scripts/4090_simple/run_closed_loop.py"),
        "--data-root", str(args.data_root), "--sonic-resources", str(args.runtime_root / "resources/sonic_v1_1"),
        "--output", str(directory / "simulation"), "--tasks", TASK, "--episode-start", str(case["case_index"]),
        "--episodes-per-task", "1", "--dr-level", str(case["level"]), "--seed", "0",
        "--seconds", str(budget["maximum_simulated_seconds"]),
        *options, "--expected-checkpoint-step", str(args.checkpoint_step), "--port", str(args.port),
        "--mobile-official-protocol", "--sync-observations", "--isaac-observation-only", "--sim-mode", "mujoco_isaac",
        "--diagnostic-unverified-sonic", "--pd-profile", "official_v1_1_ankle_pitch_1_5",
        "--hand-torque-profile", "mjcf_limits", "--warmup-control", "live_pose_sonic_v1_1",
        "--warmup-hand-policy", "measured", "--body-token-projection", "none", "--video",
        "--worker-wall-timeout", "1800", "--worker-task", TASK]
    arm_profile = getattr(args, "warmup_arm_profile", "legacy")
    if budget["budget_variant"] != "official24s":
        simulator.extend(["--mobile-time-limit-seconds", str(budget["maximum_simulated_seconds"])])
    if arm_profile != "legacy":
        simulator.extend(["--warmup-arm-profile", arm_profile])
    return {"server": server, "simulator": simulator}


def process_identity(pid):
    try:
        values = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return {"group": int(values[2]), "session": int(values[3]), "start": int(values[19]), "state": values[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def mark_owned(process):
    process.owned_identity = process_identity(process.pid)
    require(process.owned_identity is not None and process.owned_identity["group"] == process.pid
            and process.owned_identity["session"] == process.pid, "Cannot establish new child process group ownership")
    return process


def group_members(process):
    original = process.owned_identity
    leader = process_identity(process.pid)
    require(leader is None or leader["start"] == original["start"], "Refuse to signal a reused leader PID")
    members = []
    for item in Path("/proc").iterdir():
        if not item.name.isdigit():
            continue
        identity = process_identity(int(item.name))
        if identity and identity["group"] == identity["session"] == process.pid and identity["state"] != "Z":
            require(identity["start"] >= original["start"], "Process group identity predates owned child")
            members.append(int(item.name))
    return members


def stop_owned(process):
    if process is None:
        return
    require(getattr(process, "owned_identity", None) is not None, "Refuse cleanup without child ownership proof")
    if group_members(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 35
        while group_members(process) and time.monotonic() < deadline:
            time.sleep(.2)
        if group_members(process):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    process.wait(timeout=15)


def runtime_gate(startup, expected, model_sha, seed, pid, policy=None):
    policy = policy or evaluation_policy(argparse.Namespace())
    require(startup.get("weight_sha256") == model_sha and startup.get("pid") == pid
            and startup.get("sampling_seed") == seed and startup.get("Ta") == policy["execute_steps"]
            and startup.get("rtc") is policy["rtc_enabled"] and startup.get("inference_steps") == 10, "Actual server identity differs")
    if policy["rtc_enabled"]:
        for key in ("rtc_mode", "rtc_exec_shift", "rtc_inference_delay", "rtc_min_exec_horizon", "pig_mask_schedule", "pig_guidance_alpha"):
            require(startup.get(key) == policy[key], "Actual RTC startup differs: " + key)
        require(startup.get("Tp") == 30, "RTC must preserve the full30 prediction horizon")
        require(startup.get("parameter_tensor_count", 0) > 0 and startup.get("trainable_parameter_tensor_count") == 0
                and startup.get("parameter_gradient_tensor_count") == 0, "RTC policy parameters must stay frozen and gradient-free")
    actual = startup["online_dino_runtime"]
    keys = ("torch_version", "transformers_version", "local_file_sha256", "preprocessing", "image_process",
            "normalization", "inference_dtype", "storage_dtype", "shape", "token_order")
    for key in keys:
        require(actual[key] == expected[key], "Actual online DINO differs: " + key)
    derived = hashlib.sha256(json.dumps(actual["local_file_sha256"], sort_keys=True,
                                        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(derived == expected["weights_sha256"], "Actual DINO weights differ")
    for key in ("inference_precision", "compute_precision", "attention_resolved", "output_storage_dtype",
                "compile_model", "preprocessor", "pillow_version"):
        require(actual["optimization"][key] == expected["optimization"][key], "Actual DINO optimization differs: " + key)
    return {"status": "passed", "actual": actual, "runtime_fields_compared": list(keys),
            "derived_runtime_weights_sha256": derived, "evaluation_policy": policy,
            "cross_device_bitwise_equivalence_claimed": False}


def json_lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def close_number(actual, expected, message):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=0, abs_tol=1e-8), message)


def validate_rtc_execution(result, ticks, requests, policy):
    """Prove actual RTC calls and exactly consumed windows, rather than an enabled flag."""
    info = result["server_info"]
    require(info.get("rtc_enabled") is True and info.get("rtc_mode") == "test_time"
            and info.get("action", {}).get("action_exec_horizon") == policy["execute_steps"], "RTC response contract differs")
    for key in ("rtc_exec_shift", "rtc_inference_delay", "rtc_min_exec_horizon", "pig_mask_schedule", "pig_guidance_alpha"):
        require(info.get(key) == policy[key], "RTC live server contract differs: " + key)
    successful = [row for row in requests if row["status"] == 200]
    for request in requests:
        counts = request.get("sampler_calls", {})
        require(set(counts) == {"plain", "test_time", "train"}
                and all(type(v) is int and v >= 0 for v in counts.values()), "Missing actual RTC sampler audit counts")
        require(request.get("parameter_gradient_tensor_count") == 0, "RTC accumulated policy parameter gradients")
    policy_ticks = [tick for tick in ticks if tick["phase"] == "policy_latent"]
    window = policy["control_ticks_per_window"]
    require(len(successful) == (len(policy_ticks) + window - 1) // window,
            "RTC must replan exactly after each fully consumed execution window")
    for index, request in enumerate(successful):
        close_number(float(request["sim_time_s"]), policy_ticks[index * window]["sim_time_s"], "RTC request is shifted from the consumed action clock")
        expected_path = "plain" if index == 0 else "test_time"
        require(request.get("sampling_path") == expected_path, "RTC flag lacks the actual expected sampler call")
        require(request.get("response_action_shape") == [policy["execute_steps"], 80], "RTC response is not the exact execution window")
        calls = request.get("sampler_calls", {})
        require(calls.get(expected_path) == 1 and sum(calls.values()) == 1, "RTC sampler call count differs")
    for request in requests:
        if request["status"] == 425:
            require(request.get("sampling_path") is None and sum(request.get("sampler_calls", {}).values()) == 0,
                    "Warmup rejection ran an RTC or plain sampler")
    for index, tick in enumerate(policy_ticks):
        group_start = policy_ticks[index // window * window]["sim_time_s"]
        close_number(tick["chunk_start_sim_time_s"], group_start, "RTC chunk anchoring changed during execution")
        expected_index = math.floor((tick["sim_time_s"] - group_start) * policy["action_hz"] + 1e-7)
        require(tick["action_index_at_30hz"] == expected_index and 0 <= expected_index < policy["execute_steps"],
                "RTC executed a stale, duplicated-window or out-of-window action")
    return {"status": "passed", "plain_requests": min(1, len(successful)),
            "test_time_requests": max(0, len(successful) - 1), "execution_window": policy["execute_steps"],
            "control_ticks_per_window": window, "all_requests_and_action_indices_aligned": True}


def validate_warmup_arms(episode, ticks, expected_profile, expected_helper_sha=None):
    """Bind an explicit preparation variant to actual causal control evidence."""
    warmup = episode.get("warmup_control", {})
    require(warmup.get("arm_profile", "legacy") == expected_profile, "Warmup arm profile differs from frozen campaign")
    if expected_profile == "legacy":
        return
    require(expected_profile == "wrist_retreat_v1", "Unknown frozen arm profile")
    reference, control = warmup.get("arm_reference", {}), warmup.get("arm_controller", {})
    if expected_helper_sha is not None:
        require(reference.get("helper_source_sha256") == expected_helper_sha, "Actual wrist helper differs from frozen source")
    require(reference.get("kind") == "first_live_robot_world_wrist_retreat_v1"
            and reference.get("object_feedback") is False and reference.get("contact_feedback") is False
            and reference.get("future_teacher") is False and reference.get("live_state_writes") is False
            and reference.get("history_injection") is False, "Missing causal robot-only arm reference")
    profile = reference.get("profile", {})
    require(all(profile.get(key) == value for key, value in
                (("backward_m", .05), ("outward_each_m", .03), ("upward_m", .05), ("transition_s", .2))),
            "Fixed wrist preparation trajectory changed")
    count = episode["warmup_frames"]
    events, actual = reference.get("events", []), control.get("events", [])
    require(reference.get("calls") == count == len(events) and len(actual) >= count,
            "Arm profile lacks every real warmup reference/control sample")
    require(control.get("enabled") is True and control.get("arm_gravity_compensation") is True
            and control.get("gravity_physics_steps", 0) >= count * 4,
            "Arm gravity preparation did not run on all warmup physical steps")
    close_number(control["first_gravity_sim_time_s"], ticks[0]["sim_time_s"], "Arm preparation missed its first physical step")
    for index, event in enumerate(events):
        close_number(event["sim_time_s"], ticks[index]["sim_time_s"], "Wrist reference not on the real observation clock")
        close_number(actual[index]["sim_time_s"], ticks[index]["sim_time_s"], "Arm controller clock differs")
        require(event.get("legs_waist_reference_unchanged") is True, "Wrist reference modified non-arm desired joints")
        desired = [value for side in event["arms_left_right"] for value in side["desired_joint_q7"]]
        require(len(desired) == 14 and len(actual[index]["applied_arm14"]) == 14,
                "Missing14-joint actual arm target")
        for a, b in zip(desired, actual[index]["applied_arm14"]):
            close_number(a, b, "Applied arm target differs from the encoded preparation reference")
        close_number(actual[index]["learned_target_weight"], 0., "Warmup arm override ended before policy")
        close_number(actual[index]["gravity_weight"], 1., "Warmup gravity support changed")
    if len(ticks) > count:
        close_number(control["handoff_start_sim_time_s"], ticks[count]["sim_time_s"], "Arm handoff did not start with the first learned action")
        close_number(actual[count]["learned_target_weight"], 0., "First policy arm target jumps at handoff")
        if ticks[-1]["sim_time_s"] >= ticks[count]["sim_time_s"] + .22:
            close_number(control["handoff_completed_sim_time_s"] - control["handoff_start_sim_time_s"], .2,
                         "Preparation did not return to unmodified learned control after0.2s")
            close_number(actual[-1]["learned_target_weight"], 1., "Preparation still overrides the later learned policy")
            close_number(actual[-1]["gravity_weight"], 0., "Preparation gravity support leaked into later policy")


def validate_case(directory, case, frozen):
    """Read actual trace, both HTTP sides and decoded video; never infer success from geometry."""
    from run_closed_loop import mobile_time_budget, worker_result_errors
    budget = mobile_time_budget(frozen.get("maximum_simulated_seconds", 24))
    require(frozen.get("maximum_steps_including_warmup", 1200) == budget["max_episode_steps"],
            "Frozen Mobile step budget differs from its declared seconds")
    require("time_budget" not in frozen or frozen["time_budget"] == budget, "Frozen Mobile time budget differs")
    maximum_steps = budget["max_episode_steps"]
    base = directory / "simulation" / TASK
    result_path = base / "result.json"
    result = read(result_path)
    errors = worker_result_errors(result, 1)
    episodes = result.get("episodes", [])
    # The generic smoke gate requires learned actions. Official native success
    # during causal warmup is still native success, not a technical/task failure.
    if len(episodes) == 1 and episodes[0].get("task_success_observed") is True:
        errors = [error for error in errors if "did not execute the real model/controller loop" not in error]
    require(not errors, "Simulator result is incomplete: " + repr(errors))
    episode = result["episodes"][0]
    require(episode["episode_index"] == case["case_index"] and episode["dr_level"] == case["level"]
            and episode["seed"] == 0 and episode["source_scene_reset_dr_level"] is None, "Wrong case/reset identity")
    require(episode["source_metadata_sha256"] == case["metadata_sha256"]
            and episode["local_environment_config"] == case["local_environment_config"], "Frozen official configuration differs")
    protocol = episode["mobile_protocol"]
    require(protocol["effective_success_criteria"] == .9 and protocol["source_metadata_success_criteria"] == .4
            and protocol["max_episode_steps"] == maximum_steps and protocol["reset_dr_level"] is None, "Native protocol differs")
    for key, expected in budget.items():
        require((key not in protocol and budget["budget_variant"] == "official24s") or protocol.get(key) == expected,
                "Actual Mobile time budget differs: " + key)
    render = result["render_contract"]
    require(render["render_engine"] == "Isaac RTX" and render["authoritative_physics"] == "MuJoCo"
            and render["sync_no_time_advance"] is True and render["isaac_observation_only"] is True, "Render/physics binding differs")
    if "sources" in frozen:
        require(render.get("environment_source_sha256") == frozen["sources"]["SIMPLE/envs/sonic_loco_manip.py"]["sha256"],
                "Actual simulator environment source differs from frozen source")
    episode_dir = base / f"episode-{case['case_index']}"
    ticks = json_lines(episode_dir / "trace.jsonl")
    require(0 < len(ticks) <= maximum_steps and len(ticks) == episode["frames"] == episode["warmup_frames"] + episode["policy_frames"],
            "Incomplete tick ledger")
    require(episode["warmup_frames"] == min(len(ticks), 50), "Causal warmup was removed or extended beyond the official budget")
    validate_warmup_arms(episode, ticks, frozen.get("warmup_arm_profile", "legacy"),
                        frozen.get("sources", {}).get("SIMPLE/baselines/warmup_wrist_reference.py", {}).get("sha256"))
    require(all(type(tick["terminated"]) is bool and type(tick["truncated"]) is bool for tick in ticks), "Native flags are not bool")
    for index, tick in enumerate(ticks):
        require(tick["phase"] == ("warmup_measured_pose_pd" if index < 10 else
                                  "warmup_live_pose_sonic_v1_1" if index < 50 else "policy_latent"),
                "Warmup/policy phase ledger differs")
        close_number(tick["next_sim_time_s"] - tick["sim_time_s"], .02, "Wrong simulation tick")
        if index:
            close_number(tick["sim_time_s"], ticks[index-1]["next_sim_time_s"], "Discontinuous simulation clock")
    require(all(not tick["terminated"] and not tick["truncated"] for tick in ticks[:-1]), "Continued after native terminal")
    native = ticks[-1]["terminated"]
    require(native == episode["task_success_observed"] and (episode["status"] == "task_terminated") == native, "Native success mismatch")
    if not native:
        require(episode["status"] == "env_truncated" and len(ticks) == maximum_steps and ticks[-1]["truncated"],
                "Unsuccessful case lacks full frozen Mobile budget")
    close_number(episode["simulated_seconds"], len(ticks) * .02, "Wrong total time including warmup")
    # Native success even during warmup is retained; policy execution is not an extra success gate.
    requests = json_lines(directory / "server/requests.jsonl")
    http = read(episode_dir / "http.json")
    require([row["ordinal"] for row in requests] == list(range(1, len(requests)+1)), "Server request ordinal gap")
    require(all(row["status"] in (200, 425) for row in requests), "Server HTTP technical failure")
    require(all(row["rng_before"] == row["rng_after"] for row in requests if row["status"] == 425), "Warmup rejection consumed sampling RNG")
    require(sum(row["status"] == 200 for row in requests) == episode["policy_chunks"]
            and sum(row["status"] == 425 for row in requests) == episode["http_425_count"], "Server request count differs")
    identities = {(row["session_id"], row["episode_id"]) for row in requests}
    require(len(identities) == 1 and next(iter(identities))[1] == episode["episode_id"], "Cross-episode server history")
    close_number(float(requests[0]["sim_time_s"]), ticks[0]["sim_time_s"], "Server history does not start at the first observation")
    session_id = next(iter(identities))[0]
    require(dict(Counter(row["endpoint"] for row in http)) == episode["http_counts"]
            and len(http) == result["http_request_count"] and episode["http_counts"].get("/reset") == 1
            and episode["http_counts"].get("/observe") == len(ticks), "Client causal history counts differ")
    require(all(row["history"]["episode_id"] == episode["episode_id"] and row["history"]["session_id"] == session_id
                and row["simulator_before_s"] == row["simulator_after_s"] for row in http), "HTTP identity/paused simulation differs")
    require(all(row["http_status"] == 200 for row in http if row["endpoint"] in ("/reset", "/observe")), "Observe/reset technical error")
    reset = next(row for row in http if row["endpoint"] == "/reset")["reset_response"]
    require(reset.get("history_frames") == 0 and reset.get("rtc_cleared") is True, "Fresh server history reset is unproven")
    acts = [row for row in http if row["endpoint"] == "/act"]
    require(len(acts) == len(requests), "Client/server action count differs")
    for client, server in zip(acts, requests):
        require(client["http_status"] == server["status"], "Client/server action status differs")
        close_number(float(client["history"]["sim_time_s"]), float(server["sim_time_s"]), "Client/server timestamp differs")
    observed = [row["history"]["sim_time_s"] for row in http if row["endpoint"] == "/observe"]
    require(observed == [row["sim_time_s"] for row in ticks], "History is not sourced from each current observation")
    policy = frozen.get("evaluation_policy", evaluation_policy(argparse.Namespace()))
    rtc_execution = (validate_rtc_execution(result, ticks, requests, policy) if policy["rtc_enabled"] else None)
    video = Path(episode["video"]["path"]).resolve(strict=True)
    require(video == (episode_dir / "rollout.mp4").resolve() and episode["video"]["ffmpeg_returncode"] == 0, "Wrong/incomplete case video")
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=width,height,r_frame_rate,nb_read_frames", "-of", "json", str(video)], timeout=120))["streams"]
    require(len(probe) == 1 and probe[0]["r_frame_rate"] == "50/1" and
            (probe[0]["width"], probe[0]["height"]) == (640, 360) and
            int(probe[0]["nb_read_frames"]) == episode["video"]["frames"] == len(ticks) + 1, "Decoded video frame/rate/geometry differs")
    paths = [result_path, episode_dir / "trace.jsonl", episode_dir / "http.json", video,
             directory / "server/startup.json", directory / "server/requests.jsonl", directory / "ONLINE_DINO_RUNTIME_ACCEPTANCE.json"]
    return {"status": "complete", "level": case["level"], "case_index": case["case_index"],
            "native_success": native, "fell": episode["fell"], "fell_before_first_policy": episode["fell_before_first_policy_command"],
            "ticks": len(ticks), "warmup_ticks": episode["warmup_frames"], "policy_ticks": episode["policy_frames"],
            "simulated_seconds": episode["simulated_seconds"], "video_probe": probe[0], "video": str(video),
            "evidence": [file_identity(path) for path in paths], "formal_evaluation_eligible": False,
            "model_sha256": frozen["model"]["sha256"], "evaluation_policy": policy, "rtc_execution": rtc_execution,
            "warmup_arm_profile": frozen.get("warmup_arm_profile", "legacy"), "time_budget": budget,
            "maximum_steps_including_warmup": maximum_steps,
            "maximum_simulated_seconds": budget["maximum_simulated_seconds"]}


def preflight(args):
    for path in (MODEL_PY, SIM_PY, args.runtime_root / "apps/simple4090.kit", args.runtime_root / "resources/sonic_v1_1"):
        require(path.exists(), "Missing runtime resource: " + str(path))
    require(shutil.which("ffprobe") is not None and shutil.which("ffmpeg") is not None, "Video tools unavailable")
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", args.port))
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"], text=True, timeout=15)
    require(not compute.strip(), "Other GPU compute is active; no processes were stopped")


def execute(args, case, frozen, state):
    directory = args.output / f"level{case['level']}" / f"case_{case['case_index']:02}"
    directory.mkdir(parents=True, exist_ok=False)
    row = {"level": case["level"], "case_index": case["case_index"], "status": "starting",
           "directory": str(directory), "started_utc": now(), "sampling_seed": case["sampling_seed"]}
    state["cases"].append(row)
    write(args.output / "state.json", state)
    command = commands(args, case, directory)
    write(directory / "COMMANDS.json", command)
    write(directory / "FROZEN_CASE.json", case)
    processes, interrupted = [], False
    try:
        for identity in [frozen["model"], frozen["training_manifest"], *frozen["sources"].values(), *frozen["metadata_files"]]:
            unchanged(identity)
        preflight(args)
        with (directory / "server.log").open("xb") as log:
            server = subprocess.Popen(command["server"], cwd=REPO, env=environment(args, MODEL_PY), stdout=log,
                                      stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(server)
            mark_owned(server)
            row["server_identity"] = {"pid": server.pid, **server.owned_identity}
            opener = build_opener(ProxyHandler({}))
            deadline = time.monotonic() + 480
            while True:
                require(server.poll() is None, "Model server exited; inspect retained log")
                if (directory / "server/startup.json").is_file():
                    try:
                        with opener.open(f"http://127.0.0.1:{args.port}/health", timeout=2) as response:
                            if response.status == 200:
                                break
                    except (URLError, TimeoutError, OSError):
                        pass
                if time.monotonic() >= deadline:
                    raise TimeoutError("Model server startup exceeded480s")
                time.sleep(1)
            acceptance = runtime_gate(read(directory / "server/startup.json"), read(args.training_manifest)["encoder"],
                args.expected_model_sha256, case["sampling_seed"], server.pid, evaluation_policy(args))
            acceptance["training_manifest_sha256"] = frozen["training_manifest"]["sha256"]
            write(directory / "ONLINE_DINO_RUNTIME_ACCEPTANCE.json", acceptance)
            row["status"] = "simulating"
            write(args.output / "state.json", state)
            with (directory / "simulation.log").open("xb") as simlog:
                simulator = subprocess.Popen(command["simulator"], cwd=REPO, env=environment(args, SIM_PY), stdout=simlog,
                                             stderr=subprocess.STDOUT, start_new_session=True)
                processes.append(simulator)
                mark_owned(simulator)
                row["simulator_identity"] = {"pid": simulator.pid, **simulator.owned_identity}
                require(simulator.wait(timeout=2100) == 0, "Simulator failed; inspect retained log/result")
            for identity in [frozen["model"], frozen["training_manifest"], *frozen["sources"].values(), *frozen["metadata_files"]]:
                unchanged(identity)
            row.update(validate_case(directory, case, frozen))
    except BaseException as error:
        interrupted = not isinstance(error, Exception)
        row.update(status="technical_error", error=repr(error), traceback=traceback.format_exc())
    finally:
        cleanup_errors = []
        for process in reversed(processes):
            try:
                stop_owned(process)
            except Exception as error:
                cleanup_errors.append(repr(error))
        if cleanup_errors:
            row.update(status="technical_error", cleanup_errors=cleanup_errors)
        row["finished_utc"] = now()
        write(directory / ("CASE_SUMMARY.json" if row["status"] == "complete" else "TECHNICAL_FAILURE.json"), row)
        write(args.output / "state.json", state)
    if interrupted:
        raise KeyboardInterrupt("Campaign interrupted; owned children cleaned, evidence retained")


def finish(state):
    complete = [row for row in state["cases"] if row["status"] == "complete"]
    covered = sorted((row["level"], row["case_index"]) for row in complete)
    full = covered == [(level, case) for level in range(3) for case in range(10)] and len(state["cases"]) == 30
    state.update(status="complete" if full else "incomplete", completed_utc=now(),
                 technically_valid_cases=len(complete), technical_errors=sum(row["status"] == "technical_error" for row in state["cases"]),
                 not_attempted_cases=30-len(state["cases"]),
                 native_successes=sum(row["native_success"] for row in complete),
                 levels=[{"level": level, "planned_cases": 10, "technically_valid_cases": sum(row["level"] == level for row in complete),
                          "native_successes": sum(row["native_success"] for row in complete if row["level"] == level),
                          "technical_errors": sum(row["level"] == level and row["status"] == "technical_error" for row in state["cases"])} for level in range(3)])
    return 0 if state["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("deployment", "data-root", "training-manifest", "output", "runtime-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-model-sha256", required=True)
    parser.add_argument("--checkpoint-step", type=int, default=6000)
    parser.add_argument("--port", type=int, default=21075)
    parser.add_argument("--sampling-seed", type=int, default=20261105)
    parser.add_argument("--seconds", type=int, choices=(24, 30), default=24,
                        help="Total simulated budget including1s warmup;30s is an explicit extension of the original24s cases")
    parser.add_argument("--rtc-mode", choices=("off", "test_time"), default="off")
    parser.add_argument("--execute-steps", type=int, help="Default30 with RTC off,15 with test_time; must align to50Hz ticks")
    parser.add_argument("--rtc-min-exec-horizon", type=int,
                        help="Test-time RTC soft-mask horizon s; default execute-steps, with execute-steps <= s <=30 and d=0")
    parser.add_argument("--warmup-arm-profile", choices=("legacy", "wrist_retreat_v1"), default="legacy",
                        help="Explicit separate preparation variant; never changes archived baseline results")
    args = parser.parse_args(argv)
    require(len(args.expected_model_sha256) == 64 and all(char in "0123456789abcdef" for char in args.expected_model_sha256), "Expected SHA must be lowercase SHA256")
    require(args.checkpoint_step > 0 and 1024 <= args.port <= 65535 and 0 <= args.sampling_seed <= 2**32 - 30, "Invalid step/port/seed")
    policy = evaluation_policy(args)
    for name in ("deployment", "data_root", "training_manifest", "runtime_root"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    require(args.output.is_absolute() and not args.output.exists(), "Campaign must be a fresh absolute directory")
    args.output = args.output.resolve()
    require(not args.output.is_relative_to(REPO) and not args.output.is_relative_to(args.deployment)
            and not args.output.is_relative_to(args.data_root), "Output overlaps source/deployment/resources")
    frozen = freeze_inputs(args)
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "FROZEN_CAMPAIGN.json", frozen)
    state = {"contract": CONTRACT, "status": "running", "created_utc": now(), "planned_cases": 30,
             "checkpoint_step": args.checkpoint_step, "model_sha256": args.expected_model_sha256,
             "cases": [], "formal_evaluation_eligible": False, "boundary": frozen["boundary"],
             "technical_errors_are_task_failures": False, "automatic_retries": 0, "evaluation_policy": policy}
    state["warmup_arm_profile"] = args.warmup_arm_profile
    state["time_budget"] = frozen["time_budget"]
    state["maximum_steps_including_warmup"] = frozen["maximum_steps_including_warmup"]
    state["maximum_simulated_seconds"] = frozen["maximum_simulated_seconds"]
    write(args.output / "state.json", state)
    try:
        for case in frozen["cases"]:
            execute(args, case, frozen, state)
    finally:
        code = finish(state)
        write(args.output / "FINAL_EVALUATION_ACCEPTANCE.json", state)
        write(args.output / "state.json", state)
    return code


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Supervisor interrupted; retain evidence and clean only owned children")
    signal.signal(signal.SIGTERM, interrupted)
    raise SystemExit(main())
