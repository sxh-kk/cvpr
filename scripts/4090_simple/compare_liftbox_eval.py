"""Read-only paired local LiftBox pilot summary; this is not a benchmark score.

Run after the fit and both after episodes finish. Each --before/--after takes
SOURCE_EPISODE RUN_DIRECTORY SERVICE_AUDIT_DIRECTORY (repeat for sources 4, 8).
No model, simulator, training library or CUDA runtime is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np

TASK = "G1WholebodyLiftBoxHoldSonic-v0"
INSTRUCTION = "Lift the box so its lowest point is at least 10 cm above the floor, and maintain hand contact without floor or table support for 1 second."
METRIC = "lift_box_airborne_hand_contact_v1"
BASE_SHA = "14396d45723cf2e1a036487bea7987f6a4ac1dc292ff4da9db601da69c9b2716"
V11_ENCODER_SHA = "fb97de22819b2057b41459802128d91723d91a25f0ad73e7bfc41a9cf8365bae"
JOINT_GROUPS = ("action_header", "lang_backbone", "vision_tower", "mm_projector", "motion_expert")
METRICS = ("fm_loss", "body_normalized_action_l1", "hand_normalized_action_l1",
           "body_denormalized_action_l1", "hand_denormalized_action_l1")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    return {"path": str(Path(path).resolve()), "sha256": sha256(path)}


def finite(value, label):
    value = float(value)
    require(math.isfinite(value), f"Non-finite {label}")
    return value


def close(a, b, label, atol=1e-8):
    require(abs(finite(a, label) - finite(b, label)) <= atol, f"Mismatched {label}: {a} vs {b}")


def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_rng(value):
    require(isinstance(value, dict) and set(value) == {"python", "numpy", "torch_cpu", "torch_cuda"}, "Incomplete RNG state")
    require(all(valid_sha(value[k]) for k in ("python", "numpy", "torch_cpu")) and
            isinstance(value["torch_cuda"], list) and len(value["torch_cuda"]) == 1 and
            all(valid_sha(v) for v in value["torch_cuda"]), "Invalid single-GPU RNG digests")


def body_token_projection_contract(result, episode):
    """Match projection recipes, not outcome-dependent change counters.

    The original four continuous runs predate this option and omit both fields.
    They remain a legacy-none pair; an explicit newer recipe cannot silently
    substitute for a missing receipt on just one side.
    """
    if "body_token_projection" not in episode:
        require("body_token_projection" not in result, "Missing episode body-token projection receipt")
        return {"mode":"none", "receipt_version":"legacy_continuous_without_projection_option"}
    receipt = episode["body_token_projection"]
    require(isinstance(receipt, dict) and receipt.get("mode") in ("none", "fsq_v1_1"),
            "Unknown body-token projection contract")
    mode = receipt["mode"]
    expected = ({"step":1/16, "range":[-1.,15/16], "levels":32,
                 "encoder_sha256":V11_ENCODER_SHA, "rounding":"nearest ties-to-even"}
                if mode == "fsq_v1_1" else
                {"step":None, "range":None, "levels":None, "encoder_sha256":None, "rounding":None})
    require(all(key in receipt and receipt[key] == value for key, value in expected.items()) and
            valid_sha(receipt.get("consumer_source_sha256")), "Invalid body-token projection grid or source identity")
    contract = {"mode":mode, "receipt_version":"explicit_v1",
                "consumer_source_sha256":receipt["consumer_source_sha256"], **expected}
    declared = result.get("body_token_projection")
    require(isinstance(declared, dict) and all(declared.get(key) == contract[key]
            for key in ("mode", "consumer_source_sha256", *expected)), "Worker/episode body-token projection differs")
    for key in ("command_samples", "elements_compared", "changed_elements", "range_clipped_elements"):
        require(type(receipt.get(key)) is int and receipt[key] >= 0, "Invalid body-token projection counts")
    require(receipt["command_samples"] == episode["policy_frames"] and
            receipt["elements_compared"] == 64 * receipt["command_samples"] and
            receipt["range_clipped_elements"] <= receipt["changed_elements"] <= receipt["elements_compared"],
            "Body-token projection counts differ from actual learned commands")
    total = finite(receipt["absolute_change_sum"], "projection change sum")
    mean = finite(receipt["mean_absolute_change"], "projection mean change")
    maximum = finite(receipt["maximum_absolute_change"], "projection maximum change")
    require(0 <= mean <= maximum and total >= 0, "Invalid body-token projection changes")
    close(mean, total / max(1, receipt["elements_compared"]), "projection mean/sum")
    if mode == "none":
        require(receipt["changed_elements"] == receipt["range_clipped_elements"] == total == maximum == 0,
                "Continuous body-token projection receipt claims modified commands")
    return contract


def service_audit(startup, rows, episode, expected_weight):
    require(startup["weight_sha256"] == expected_weight, "Service checkpoint SHA differs from its training stage")
    require(startup["host"] == "127.0.0.1" and startup["Ta"] == 30 and startup["rtc"] is False and
            startup["inference_steps"] == 10, "Unexpected localhost sampling contract")
    validate_rng(startup["rng_sha256"])
    previous, successful, sessions = startup["rng_sha256"], [], set()
    require(bool(rows), "Missing service request audit")
    for ordinal, row in enumerate(rows, 1):
        require(row["ordinal"] == ordinal and row["episode_id"] == episode["episode_id"], "Mixed/incomplete service episode audit")
        require(isinstance(row["session_id"], str) and row["session_id"], "Missing service session")
        sessions.add(row["session_id"])
        finite(row["sim_time_s"], "service simulator time")
        require(finite(row["wall_time_s"], "request latency") >= 0, "Negative request latency")
        validate_rng(row["rng_before"])
        validate_rng(row["rng_after"])
        require(row["rng_before"] == previous, "Service RNG chain discontinuity")
        require(row["status"] in (200, 425), "Service request failed")
        if row["status"] == 425:
            require(not successful and row["rng_before"] == row["rng_after"], "Warmup request consumed RNG or followed policy sampling")
        else:
            successful.append(row)
        previous = row["rng_after"]
    require(len(sessions) == 1 and bool(successful), "Require one real session with successful policy requests")
    require(len(rows) == episode["http_counts"]["/act"] and len(successful) == episode["policy_chunks"] and
            len(rows) - len(successful) == episode["http_425_count"], "Service/client request counts differ")
    require(len(episode["successful_action_http_latencies_s"]) == len(successful), "Missing successful client latencies")
    return successful


def physics_metrics(episode, rows, horizon_s):
    """Recheck the logged task predicate/clock at its explicit 50 Hz cadence."""
    require(episode["status"] in {"sim_time_limit", "task_terminated", "env_truncated", "stopped_on_fall"}, "Incomplete physical episode")
    require(episode["finite_states"] is True and episode["diagnostic_loop_executed"] is True and
            episode["simulator_unchanged_during_http"] is True and episode["decoder_calls"] > 0 and
            episode["policy_frames"] > 0, "Incomplete finite model/controller/clock evidence")
    require(bool(rows), "Missing 50 Hz physical trace")
    holds, bottoms, forces, loaded, unsupported = [], [], [], 0, 0
    condition_start, previous_time, fell = None, None, bool(episode["fell"])
    for row in rows:
        now, nxt = finite(row["sim_time_s"], "trace time"), finite(row["next_sim_time_s"], "trace next time")
        close(nxt - now, .02, "50 Hz trace interval")
        if previous_time is not None:
            close(now, previous_time, "trace continuity")
        previous_time = nxt
        finite(row["pelvis_height_m"], "pelvis height")
        finite(row["pelvis_tilt_degrees"], "pelvis tilt")
        fell = fell or bool(row["fell"])
        p = row["task_progress"]
        require(p["metric"] == METRIC and p["measurement_hz"] == 50 and p["continuity_gap"] is False, "Unexpected task measurement contract")
        close(p["sim_time_s"], nxt, "post-step task measurement time")
        bottom, floor = finite(p["box_lowest_z_m"], "box bottom"), finite(p["floor_z_m"], "floor height")
        force, hold = finite(p["hand_normal_force_n"], "hand force"), finite(p["hold_sim_s"], "hold duration")
        # The trace exposes the sum of hand forces, while the actual predicate
        # requires one individual hand contact above .01 N. Several subthreshold
        # contacts can sum above .01: do not manufacture that missing predicate.
        close(floor, 0, "fixed floor height")
        clearance, contact = bottom >= .10, p["loaded_hand_contact"]
        require(isinstance(contact, bool) and force >= 0 and (not contact or force > .01), "Invalid loaded hand contact evidence")
        condition = clearance and contact and not p["floor_contact"] and not p["table_contact"]
        require(p["clearance_ge_10cm"] == clearance and p["condition"] == condition,
                "Logged task predicate differs from its measurements")
        if condition:
            condition_start = nxt if condition_start is None else condition_start
        else:
            condition_start = None
        close(hold, 0 if condition_start is None else nxt - condition_start, "continuous 50 Hz hold")
        require(p["success"] == (condition and hold >= 1.0 - 1e-9), "Task success differs from measured one-second hold")
        holds.append(hold)
        bottoms.append(bottom)
        forces.append(force)
        loaded += int(contact)
        unsupported += int(not p["floor_contact"] and not p["table_contact"])
    elapsed = previous_time - float(rows[0]["sim_time_s"])
    close(elapsed, episode["simulated_seconds"], "episode/trace duration")
    require(episode["frames"] in (len(rows), len(rows) + 1) and
            episode["warmup_frames"] + episode["policy_frames"] == len(rows), "Trace frame count differs from episode")
    require(episode["final_task_progress"] == rows[-1]["task_progress"], "Missing final measured task state")
    if episode["status"] == "sim_time_limit":
        require(horizon_s <= elapsed < horizon_s + .02000001, "Time-limit run did not cover requested horizon")
    measured_success = bool(rows[-1]["task_progress"]["success"])
    require(episode["task_success_observed"] == measured_success and
            (episode["status"] == "task_terminated") == bool(rows[-1]["terminated"]), "Task terminal evidence disagrees")
    success = bool(measured_success and episode["status"] == "task_terminated" and not fell)
    return {"episode_status": episode["status"], "completed_physical_attempt": True, "physical_task_success": success,
            "producer_policy_attribution": episode["task_success_attributed_to_policy"],
            "no_fall": not fell, "finite_states": True, "simulated_seconds": elapsed,
            "max_hold_sim_s": max(holds), "max_box_lowest_z_m": max(bottoms), "max_hand_normal_force_n": max(forces),
            "loaded_hand_contact_measurements": loaded, "unsupported_measurements": unsupported,
            "measurement_count": len(rows), "measurement_hz": 50,
            "physics_hz": 200, "force_scope": "logged task measurements at 50 Hz, including terminal; no claim about all 200 Hz substeps"}


def load_case(case_id, run, service, weight, step, horizon_s):
    run, service = Path(run), Path(service)
    result_path = run / TASK / "result.json"
    result = read_json(result_path)
    require(result["task"] == TASK and result["status"] == "bounded_smoke_finished" and len(result["episodes"]) == 1,
            "Missing complete single-episode worker result")
    episode = result["episodes"][0]
    require(episode["seed"] == 10001 + case_id and episode["dr_level"] == 0 and episode["instruction"] == INSTRUCTION,
            "Unexpected source case seed/instruction")
    require(episode["execute_steps"] == 30 and episode["action_fps"] == 30 and episode["sonic_decoder_hz"] == 50 and
            episode["physics_dt_s"] == .005 and episode["render_hz"] == 50, "Unexpected action/physics clock")
    require(episode["controller_history_length_after_reset"] == 0 and episode["agent_chunk_cleared_after_reset"] is True,
            "Controller/action history was not reset")
    initial = episode["initial_physics"]
    require(initial["source_seed"] == episode["seed"] and initial["source_row"] == 0 and
            initial["source_task_id"] == "simple/" + TASK and initial["initialization_count"] == 1 and
            initial["state_spec"] == 8191 and initial["state_size"] == 548 and
            valid_sha(initial["initial_state_sha256"]) and initial["initial_state_sha256"] == initial["restored_state_sha256"] and
            initial["new_rgb_after_restore"] is True and initial["live_model_data_objects_preserved"] is True and
            initial["task_progress_reset"] is True,
            "Missing complete restored initial integration state")
    compatibility = initial["model_compatibility"]
    require(compatibility["compatible"] is True and compatibility["source_static_sha256"] == compatibility["target_static_sha256"],
            "Initial-state model is incompatible")
    info = result["server_info"]
    require(info["ckpt_step"] == step and info["action"] == {"action_dim": 80, "action_chunk_size": 30, "action_exec_horizon": 30} and
            info["rtc_enabled"] is False and info["rtc_mode"] == "off" and info["num_inference_steps"] == 10 and
            info["online_history"]["frames"] == 6, "Unexpected deployed checkpoint/action/history contract")
    startup_path, requests_path = service / "startup.json", service / "requests.jsonl"
    startup, requests = read_json(startup_path), read_rows(requests_path)
    successful = service_audit(startup, requests, episode, weight)
    projection = body_token_projection_contract(result, episode)
    trace_path = result_path.parent / "episode-0/trace.jsonl"
    metrics = physics_metrics(episode, read_rows(trace_path), horizon_s)
    return {"result": result, "episode": episode, "startup": startup, "successful": successful, "metrics": metrics,
            "body_token_projection":projection,
            "artifacts": {name: artifact(path) for name, path in (("result", result_path), ("trace", trace_path),
                         ("service_startup", startup_path), ("service_requests", requests_path))}}


def pair_cases(case_id, before, after):
    b, a = before["episode"], after["episode"]
    projection = body_token_projection_contract(before["result"], b)
    require(projection == body_token_projection_contract(after["result"], a),
            "Paired body-token projection contract changed; treat it as a separate projection experiment")
    for key in ("instruction", "seed", "local_environment_config", "physics_dt_s", "render_hz", "action_fps",
                "sonic_decoder_hz", "execute_steps", "warmup_frames", "warmup_phase_counts", "initial_pelvis_position_m",
                "initial_pelvis_quaternion_wxyz"):
        require(b[key] == a[key], f"Paired {key} changed")
    initial_keys = ("source_result_sha256", "source_mjb_sha256", "source_trajectory_sha256", "source_seed", "state_spec",
                    "state_size", "source_row", "initial_state_sha256", "restored_state_sha256", "source_sim_time_s")
    initial = {key: b["initial_physics"][key] for key in initial_keys}
    require(initial == {key: a["initial_physics"][key] for key in initial_keys}, "Paired complete initial physics SHA changed")
    # Paths to evidence and process identities can differ; actual numerical and source identities must agree.
    controller = lambda result: {k: v for k, v in result["controller_identity"].items() if k not in ("provenance", "compatibility_evidence")}
    require(controller(before["result"]) == controller(after["result"]), "Paired controller contract changed")
    require(before["result"]["render_contract"] == after["result"]["render_contract"] and
            before["result"]["observation_phase"] == after["result"]["observation_phase"], "Paired renderer/observation phase changed")
    info = lambda result: {k: v for k, v in result["server_info"].items() if k not in ("ckpt_step", "run_dir", "timestamp")}
    require(info(before["result"]) == info(after["result"]), "Paired model serving/DINO/history contract changed")
    require(b["warmup_control"] == a["warmup_control"], "Paired warmup helper/reference/encoder or actual warmup events changed")
    for key in ("sampling_seed", "rng_sha256", "Ta", "rtc", "inference_steps"):
        require(before["startup"][key] == after["startup"][key], f"Paired startup {key} changed")
    left, right = before["successful"], after["successful"]
    common = min(len(left), len(right))
    for index, (old, new) in enumerate(zip(left, right), 1):
        close(old["sim_time_s"], new["sim_time_s"], f"successful request {index} simulator time")
        require(old["rng_before"] == new["rng_before"] and old["rng_after"] == new["rng_after"],
                f"Paired successful request {index} RNG differs")
    def stage(value):
        latencies = [finite(v, "client HTTP latency") for v in value["episode"]["successful_action_http_latencies_s"]]
        return {**value["metrics"], "policy_chunks": len(value["successful"]), "artifacts": value["artifacts"],
                "formal_evaluation_eligible": value["result"]["formal_evaluation_eligible"],
                "wall_seconds": value["episode"]["wall_seconds"],
                "action_http_latency_s": {"mean": statistics.mean(latencies), "median": statistics.median(latencies)}}
    return {"source_episode": case_id, "environment_seed": b["seed"], "initial_physics": initial,
            "sampling_seed": before["startup"]["sampling_seed"], "rng_successful_request_prefix_matched": common,
            "request_counts_may_differ": "episodes may terminate earlier; only their common sampling prefix is paired",
            "controller": controller(before["result"]), "render_contract": before["result"]["render_contract"],
            "body_token_projection":projection,
            "warmup": {k: v for k, v in b["warmup_control"].items() if k != "events"},
            "before": stage(before), "after": stage(after),
            "physical_after_minus_before": {k: after["metrics"][k] - before["metrics"][k]
                                           for k in ("max_hold_sim_s", "max_box_lowest_z_m", "max_hand_normal_force_n")}}


def joint_update_witness(training, preparation, expected_updates):
    """Audit actual five-group updates and CPU Adam tensors, not offload flags."""
    require(training.get("all_five_groups_changed") is True and training.get("dino_files_unchanged") is True,
            "Joint comparison lacks changed-five-groups or unchanged-DINO witness")
    before, after = training.get("before_group_sha256"), training.get("after_group_sha256")
    require(isinstance(before, dict) and isinstance(after, dict) and set(before) == set(after) == set(JOINT_GROUPS) and
            all(valid_sha(before[name]) and valid_sha(after[name]) and before[name] != after[name] for name in JOINT_GROUPS),
            "Joint comparison requires five valid, individually changed group SHA256 values")
    dino = preparation.get("dino_file_sha256")
    require(isinstance(dino, dict) and bool(dino) and all(valid_sha(v) for v in dino.values()),
            "Joint unchanged-DINO claim lacks frozen file identities")
    expected = preparation.get("expected_groups")
    require(isinstance(expected, dict) and set(expected) == set(JOINT_GROUPS), "Joint expected five-group contract missing")
    for value in expected.values():
        require(type(value.get("trainable_parameters")) is int and value["trainable_parameters"] > 0 and
                finite(value["peak_lr"], "joint peak LR") > 0, "Invalid joint expected group count/LR")
    peak_rows = training.get("five_group_peak_contract")
    require(isinstance(peak_rows, list) and len(peak_rows) == 5 and all(isinstance(r, dict) for r in peak_rows) and
            {r.get("name") for r in peak_rows} == set(JOINT_GROUPS), "Missing actual five-group peak contract")
    for row in peak_rows:
        require(row["trainable_parameters"] == expected[row["name"]]["trainable_parameters"], "Joint actual group count differs")
        close(row["lr"], expected[row["name"]]["peak_lr"], "joint actual peak LR", 1e-12)
    total_elements = sum(expected[name]["trainable_parameters"] for name in JOINT_GROUPS)
    for label, steps in (("initial_optimizer", 0), ("final_optimizer", expected_updates)):
        receipt = training.get(label)
        require(isinstance(receipt, dict) and type(receipt.get("completed_steps")) is int and
                receipt["completed_steps"] == steps and receipt.get("master_elements") == total_elements and
                receipt.get("master_device_types") == ["cpu"] and
                receipt.get("actual_state_devices") == (["cpu"] if steps else []),
                f"Joint {label} lacks actual CPU master/state/step evidence")
        wrapper = receipt.get("wrapper_receipt", {})
        require(wrapper.get("schema") == "cvpr_cpu_master_adamw_v1" and wrapper.get("native_optimizer") == "torch.optim.AdamW" and
                type(wrapper.get("completed_steps")) is int and wrapper["completed_steps"] == steps and wrapper.get("model_and_master_dtype") == "float32" and
                wrapper.get("copy_mode") == "synchronous", f"Joint {label} CPU AdamW wrapper identity differs")
        groups = wrapper.get("groups")
        require(isinstance(groups, list) and len(groups) == 5 and all(isinstance(g, dict) for g in groups) and
                {g.get("group_name") for g in groups} == set(JOINT_GROUPS),
                f"Joint {label} CPU AdamW five-group receipts missing")
        initialized = 0
        for group in groups:
            require(group.get("master_devices") == ["cpu"] and group.get("public_devices") == ["cuda:0"] and
                    group.get("state_devices") == (["cpu"] if steps else []) and
                    group.get("numel") == expected[group["group_name"]]["trainable_parameters"],
                    f"Joint {label} group actual devices/count differ")
            tensors, states = group.get("parameter_tensors"), group.get("initialized_states")
            require(type(tensors) is int and tensors > 0 and type(states) is int and
                    (0 < states <= tensors if steps else states == 0), f"Joint {label} group state coverage invalid")
            minimum, maximum = group.get("parameter_step_min"), group.get("parameter_step_max")
            require(type(minimum) is int and type(maximum) is int and 0 <= minimum <= maximum == steps,
                    f"Joint {label} group parameter steps disagree with completed updates")
            initialized += states
        require(receipt.get("native_adamw_state_entries") == initialized, f"Joint {label} actual native AdamW states differ")
    updates = training.get("updates")
    require(isinstance(updates, list) and len(updates) == expected_updates, "Joint synchronized update witnesses missing")
    for step, update in enumerate(updates, 1):
        require(type(update.get("completed_step")) is int and update["completed_step"] == step, "Joint synchronized update order differs")
        finite(update["loss_last_microbatch"], "joint loss")
        norms = update.get("pre_clip_grad_norms")
        require(isinstance(norms, dict) and set(norms) == set(JOINT_GROUPS) and
                all(type(v) in (int, float) and finite(v, "joint pre-clip gradient norm") > 0 for v in norms.values()),
                "Joint five-group gradients must be finite and nonzero at every synchronized update")
        for key in ("used_group_lrs", "next_group_lrs"):
            lrs = update.get(key)
            # Trainer.group_lrs() names metrics lr_<group>; gradient/hash
            # receipts above use the bare optimizer group names.
            require(isinstance(lrs, dict) and set(lrs) == {f"lr_{name}" for name in JOINT_GROUPS} and
                    all(finite(v, "joint applied LR") >= 0 for v in lrs.values()), "Joint applied LR witnesses missing")
    return {"training_mode":"joint_vlm_motion_action", "before_group_sha256":before, "after_group_sha256":after,
            "all_five_groups_changed":True, "dino_files_unchanged":True, "dino_file_sha256":dino,
            "synchronized_updates_audited":len(updates), "actual_cpu_adamw_receipts_verified":True,
            "routed_parameter_scope":"parameter_step_min may be below global updates; each group must reach final step"}


def offline_comparison(fit, expected_updates=300):
    fit = Path(fit)
    training = read_json(fit / "training_result.json")
    require(training["status"] == "completed" and training["comparison_complete"] is True and
            training["completed_updates"] == training["maximum_updates"] == expected_updates and
            training["initial_global_step"] == 0, "Training did not complete the fresh before/after comparison")
    preparation = read_json(fit / "preparation.json")
    mode = training.get("training_mode", "action_only")
    require(mode in ("action_only", "joint_vlm_motion_action"), "Unknown training mode; no comparison witness fallback")
    config = read_json(fit / "run_config.json")
    if mode == "action_only":
        require(training.get("frozen_unchanged") is True and training.get("action_changed") is True and
                training["source_joint_sha256"] == BASE_SHA and preparation["source_step"] == 28000 and
                preparation["source_joint_sha256"] == BASE_SHA, "Training did not complete the fresh Action-only comparison")
        source_step, source_sha, witness = 28000, BASE_SHA, {"training_mode":"action_only", "frozen_unchanged":True, "action_changed":True}
    else:
        source_step, source_sha = preparation.get("source_step"), preparation.get("source_joint_sha256")
        require(type(source_step) is int and source_step > 0 and valid_sha(source_sha) and
                training.get("source_joint_sha256") == source_sha and preparation.get("training_mode") == mode,
                "Joint actual source step/SHA identity missing")
        model = config.get("model", {})
        require(model.get("tune_vlm") is True and model.get("tune_motion") is True,
                "Joint comparison config must update VLM and Motion")
        checkpoint = Path(preparation.get("source_checkpoint", ""))
        require(checkpoint.is_file() and checkpoint.parent.name == f"ckpt_{source_step}" and
                model.get("pretrained_joint_checkpoint") == str(checkpoint), "Joint actual source checkpoint/config differs")
        witness = joint_update_witness(training, preparation, expected_updates)
    require(valid_sha(training["final_checkpoint_sha256"]) and Path(training["final_checkpoint"]).is_file(),
            "Missing final checkpoint identity")
    require(preparation["source_joint_sha256"] == source_sha and
            preparation["initial_global_step"] == 0 and preparation["optimizer_scheduler_rng_restored"] is False,
            "Unexpected joint warmstart provenance")
    field = config["data"]["transform"]["field"]
    normalization = {"stat_path": field["stat_path"], "stat_file_sha256": sha256(field["stat_path"]),
                     "bounds_sha256": {key: hashlib.sha256(json.dumps(field[key], separators=(",", ":"),
                                         allow_nan=False).encode()).hexdigest()
                                       for key in ("action_min", "action_max", "state_min", "state_max")}}
    require(normalization == training["normalization"] == preparation["normalization"], "Inherited normalization identity changed")
    require(field["action_norm_type"] in ("bounds", "bounds_q99"), "Unsupported inherited action normalization")
    low, high = (np.asarray(field[key], dtype=np.float32) for key in ("action_min", "action_max"))
    mask = np.asarray(field["action_norm_masks"] if field["use_norm_mask"] else [True] * 80)
    require(low.shape == high.shape == mask.shape == (80,) and np.isfinite(low).all() and np.isfinite(high).all(),
            "Invalid inherited action bounds")
    manifest = read_json(fit / "test_windows.json")
    windows = manifest["windows"]
    require(manifest["split"] == "test" and manifest["no_aug"] is True and len(windows) == 20 and
            {w["source_episode"] for w in windows} == {4}, "Expected the same 20 clean TEST4 windows")
    require([w["ordinal"] for w in windows] == list(range(len(windows))) and
            all(valid_sha(w["input_sha256"]) and w["fm_seed"] == preparation["launcher_arguments"]["seed"] + 10000 + w["ordinal"] * 2
                and w["inference_seed"] == w["fm_seed"] + 1
                and len(w["history_frames"]) == 6 and w["last_future_frame"] == w["frame"] + 29 for w in windows),
            "Incomplete fixed TEST input/seed/history identities")
    reports, targets = {}, None
    artifacts = {name: artifact(fit / name) for name in ("training_result.json", "test_windows.json", "preparation.json", "run_config.json")}
    for stage, updates in (("before", 0), ("after", expected_updates)):
        report = read_json(fit / f"test_{stage}.json")
        require(report["stage"] == stage and report["optimizer_updates_completed"] == updates and
                report["window_count"] == len(windows) == len(report["windows"]), "TEST stage/update/window mismatch")
        with np.load(fit / f"test_{stage}_actions.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("predictions", "normalized_targets", "raw_targets")}
        require(all(v.shape == (20, 30, 80) and np.isfinite(v).all() for v in arrays.values()), "Invalid fixed TEST action arrays")
        current_targets = (arrays["normalized_targets"], arrays["raw_targets"])
        if targets is not None:
            require(all(np.array_equal(a, b) for a, b in zip(targets, current_targets)), "Before/after TEST targets changed")
        targets = current_targets
        denormalized = np.where(mask, .5 * (arrays["predictions"] + 1) * (high - low) + low, arrays["predictions"])
        for index, (identity, row) in enumerate(zip(windows, report["windows"])):
            require((identity["ordinal"], identity["episode"]) == (row["ordinal"], row["episode"]), "TEST window order changed")
            for key in METRICS:
                require(finite(row[key], key) >= 0, "Negative TEST error")
            for group, sl in (("body", slice(0, 64)), ("hand", slice(64, 78))):
                actual = np.abs(arrays["predictions"][index, :, sl] - arrays["normalized_targets"][index, :, sl]).mean()
                close(actual, row[f"{group}_normalized_action_l1"], "saved TEST normalized L1", 1e-6)
                raw_error = np.abs(denormalized[index, :, sl] - arrays["raw_targets"][index, :, sl]).mean()
                close(raw_error, row[f"{group}_denormalized_action_l1"], "saved TEST raw/denormalized L1", 1e-6)
        for key in METRICS:
            close(report["mean"][key], statistics.mean(row[key] for row in report["windows"]), f"TEST mean {key}", 1e-6)
        reports[stage] = report["mean"]
        for suffix in ("json", "npz"):
            name = f"test_{stage}.json" if suffix == "json" else f"test_{stage}_actions.npz"
            artifacts[name] = artifact(fit / name)
    return training, {"windows": windows, "window_count": len(windows), "source_episode": 4,
        "training_mode":mode, "source_step":source_step, "source_joint_sha256":source_sha, "update_witness":witness,
        "fixed_input_manifest_shared_by_both_stages": True, "normalized_and_raw_targets_bitwise_equal": True,
        "input_identity_scope": "shared immutable window/seed/input-SHA manifest; training producer rehashes each actual batch before both evaluations; summary independently compares saved targets and recalculates action errors",
        "normalization": normalization,
        "before_optimizer_updates": 0, "after_optimizer_updates": expected_updates, **reports,
        "after_minus_before": {k: reports["after"][k] - reports["before"][k] for k in METRICS},
        "interpretation": "offline fixed-window TEST action error; body=64 latent channels, hand=14 joint channels, final 2 padding excluded; raw denormalized error includes inherited-bound clipping",
        "artifacts": artifacts}


def compare(fit, before_specs, after_specs, horizon_s=25):
    training, offline = offline_comparison(fit)
    def specifications(items):
        result = {int(case): (run, service) for case, run, service in items}
        require(len(items) == 2 and set(result) == {4, 8}, "Provide exactly sources 4 and 8 for each stage")
        return result
    before, after = specifications(before_specs), specifications(after_specs)
    cases = []
    for case_id in (4, 8):
        left = load_case(case_id, *before[case_id], offline["source_joint_sha256"], offline["source_step"], horizon_s)
        right = load_case(case_id, *after[case_id], training["final_checkpoint_sha256"], 300, horizon_s)
        cases.append(pair_cases(case_id, left, right))
    return {"status": "paired_local_pilot_complete", "task": TASK, "instruction": INSTRUCTION,
            "training_mode":offline["training_mode"],
            "weights": {"before_step": offline["source_step"], "before_sha256": offline["source_joint_sha256"], "after_fresh_updates": 300,
                        "after_checkpoint": training["final_checkpoint"], "after_sha256": training["final_checkpoint_sha256"],
                        "verification": "each audited server startup hashes its actual model.safetensors file; startup digests must equal the verified preparation source/final training digests, with matching /info checkpoint steps; summary does not reread the model weights"},
            "offline_test": offline, "physical_cases": cases, "physical_case_count": 2,
            "limitations": ["Two prespecified local physical cases, not an official SIMPLE benchmark or success-rate estimate.",
                "Logged task/force measurements are 50 Hz, not evidence of uninterrupted contact at all 200 Hz physics substeps.",
                "Run completion, exit code zero, bounded_smoke_finished and no fall do not establish task success.",
                "Controller compatibility/producer policy attribution boundaries are retained; physical predicate success is reported separately.",
                ("Before inference ran concurrently with training; after runs independently. Latencies are descriptive and cannot establish a speed improvement."
                 if offline["training_mode"] == "action_only" else
                 "Joint pilot latencies are descriptive; this report does not establish a speed improvement or A100 training-budget equivalence."),
                "Offline TEST uses one source episode with twenty windows; original source8 physical failure case remains included."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--before", nargs=3, action="append", required=True, metavar=("SOURCE", "RUN", "SERVICE"))
    parser.add_argument("--after", nargs=3, action="append", required=True, metavar=("SOURCE", "RUN", "SERVICE"))
    parser.add_argument("--horizon-s", type=float, default=25)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.horizon_s > 0 and math.isfinite(args.horizon_s), "Invalid physical horizon")
    repo = Path(__file__).resolve().parents[2]
    require(args.output.is_absolute() and not args.output.resolve().is_relative_to(repo), "Write the report outside the repository")
    report = compare(args.fit_dir, args.before, args.after, args.horizon_s)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "physical_case_count": 2}))


if __name__ == "__main__":
    main()
