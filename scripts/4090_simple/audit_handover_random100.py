"""Independent CPU admission of new online-expert random100 executions.

No dynamics step, renderer, model-policy inference or source-label substitution
occurs here. Official SONIC encoder/expert reproduction uses saved actual
pre-command states. Solved force evidence stays the original live 50 Hz stream.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

CONTRACT = "cvpr-handover-random100-independent-capture-audit-v1"
EXECUTION_CONTRACT = "cvpr-handover-random100-online-execution-v1"
FOOTPRINT_CONTRACT = "cvpr-handover-actual-full-outer-footprint-final50-v1"
ADDITIONAL_DATASET_QUALITY = {"actual_full_outer_footprint_final50": {
    "window_states": 50, "minimum_outer_hull_margin_m": .005,
    "geometry": "actual_MJB_target_collision_vertices_and_container_outer_collision_hull",
    "exact_inner_cavity_claimed": False}}
REPO = Path(__file__).resolve().parents[2]


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def own(root, relative):
    name = Path(relative)
    require(not name.is_absolute() and ".." not in name.parts, "Unsafe run-relative input")
    raw = root / name
    require(raw.is_file() and not raw.is_symlink() and raw.resolve().is_relative_to(root), "Run input escapes its owner")
    return raw


def vector(value, size, label):
    result = np.asarray(value, dtype=np.float64)
    require(result.shape == (size,) and np.isfinite(result).all(), f"Require finite {label}[{size}]")
    return result


def equal32(first, second, width, label):
    require(np.array_equal(vector(first, width, label).astype(np.float32),
                           vector(second, width, label).astype(np.float32)), f"Actual {label} changed")


def same_pose(first, second, label):
    require(np.allclose(vector(first, 7, label), vector(second, 7, label), rtol=0, atol=1e-10),
            f"Actual {label} differs from integration authority")


def _normalize(value):
    import replay_handover_teacher as base
    return json.loads(base.json_text(value))


def stable_projection_diagnostic(value):
    """Wall-clock timings are not deterministic evidence; all other fields are."""
    if isinstance(value, dict):
        return {key: stable_projection_diagnostic(item) for key, item in value.items() if key != "CPU_seconds"}
    if isinstance(value, list):
        return [stable_projection_diagnostic(item) for item in value]
    return value


def projection_phases(args):
    """Missing historical filter means all phases, never the new CLI default."""
    if "latent_projection_phases" not in args:
        return None
    value = args["latent_projection_phases"]
    require(isinstance(value, str), "Recorded latent projection phase filter must be explicit text")
    phases = tuple(value.split(","))
    require(phases and all(phase in ("approach", "right_lift", "handover", "left_transfer", "placement", "release", "hold")
                           for phase in phases), "Unknown recorded latent projection phase filter")
    return phases


def replay_projection_step(args, token, desired, phase, raw, controller, projector):
    phases = projection_phases(args)
    if phases is None or phase in phases:
        return projector.project(token, desired, raw, controller)
    return token, {"state": "skipped_phase_filter", "phase": phase, "enabled_phases": list(phases)}


def projection_bindings(args):
    """Prove the optional artifact differs only in input/output batch metadata."""
    view = args.get("latent_projector_batch_view")
    if view is None:
        return {}
    import onnx
    from handover_sonic_latent_projector import BATCH_CONTRACT, OFFICIAL_DECODER_SHA
    view = Path(view).resolve(strict=True); sidecar = view.with_suffix(view.suffix + ".json")
    official = Path(args["sonic_resources"]) / "model_decoder.onnx"
    receipt = read(sidecar)
    require(receipt.get("contract") == BATCH_CONTRACT and receipt.get("only_input_output_dim0_changed") is True
            and receipt.get("restored_complete_model_equal") is True and receipt.get("official_execution_claim") is False
            and receipt.get("official_decoder_sha256") == sha(official) == OFFICIAL_DECODER_SHA
            and receipt.get("derived_view_sha256") == sha(view), "Derived-view model/evidence SHA differs")
    original = onnx.load_from_string(official.read_bytes()); restored = onnx.load_from_string(view.read_bytes())
    for value in [*restored.graph.input, *restored.graph.output]:
        dims = value.type.tensor_type.shape.dim
        require(len(dims) == 2 and dims[0].dim_param == "jacobian_batch", "Unexpected derived metadata dimensions")
        dims[0].ClearField("dim_param"); dims[0].dim_value = 1
    require(restored.SerializeToString() == original.SerializeToString(), "Derived graph changes more than batch metadata")
    return {str(view): sha(view), str(sidecar): sha(sidecar), str(official): OFFICIAL_DECODER_SHA}


def placement_encoder_mode(args):
    mode = args.get("placement_encoder_mode", "g1")
    require(mode in ("g1", "teleop"), "Unknown placement encoder mode")
    return mode


def placement_locomotion(args):
    value = args.get("placement_locomotion", "source_future")
    require(value in ("source_future", "actual_hold"), "Unknown placement locomotion mode")
    return value


def placement_encoder_bindings(args, capture_code):
    """Mode1 is source-verified pure assembly, independent of optional ONNX FD."""
    if placement_encoder_mode(args) == "g1":
        return {}
    import handover_sonic_teleop_reference as teleop
    evidence = teleop.verify_official_sources()
    helper = str(Path(teleop.__file__).resolve(strict=True))
    bindings = {helper: sha(helper), **evidence["source_sha256"]}
    require(evidence.get("official_sources_rehashed") is True
            and all(capture_code.get(path) == expected for path, expected in bindings.items()),
            "Teleop mode requires captured helper and all four official source identities")
    return bindings


def make_replay_expert(source, args):
    from handover_random100_expert import HandoverObjectFeedbackExpert, ExpertConfig
    return HandoverObjectFeedbackExpert(source["values"], Path(args["sonic_resources"]),
        config=ExpertConfig(placement_encoder_mode=placement_encoder_mode(args),
                            placement_locomotion=placement_locomotion(args)))


def _bindings(run, result, args):
    from handover_random100_sources import load_raw_source
    from handover_random100_scene import randomize_config, transformed_initial, digest
    import replay_handover_teacher as base
    require(result.get("contract") == EXECUTION_CONTRACT and result.get("status") == "execution_complete",
            "Only complete new executions can enter independent admission")
    mode = placement_encoder_mode(args)
    require(result.get("placement_encoder_mode", "g1") == mode, "Execution placement encoder mode differs")
    require(result.get("placement_locomotion", "source_future") == placement_locomotion(args),
            "Execution placement locomotion differs")
    placement_sources = placement_encoder_bindings(args, result["capture_code_sha256"])
    require(Path(args["output"]).resolve() == run and args["render_mode"] == result["render_mode"]
            and args["source_episode"] == result["source_episode_index"]
            and args.get("hand_condition") in ("declared", "inferred_effective")
            and args["hand_condition"] == result.get("hand_command_condition"), "Execution identity/hand condition differs")
    require(result.get("source_split") == "train" and result.get("commands_generated_online") is True
            and result.get("source_tokens_reencoded") is True and result.get("learned_policy_used") is False
            and result.get("object_restores_during_control") is False
            and result.get("accepted_new_dataset") is False
            and result.get("code_unchanged_during_run") is True
            and result.get("calibration_source_assets_unchanged") is True,
            "Online execution isolation or immutable-source evidence is missing")
    from simple.baselines.sonic_v1_1 import (OFFICIAL_DECODER_SHA256, OFFICIAL_OBSERVATION_SHA256, BODY_KP, BODY_KD)
    controller = result["controller_identity"]
    require(result.get("command_rate_hz") == 30 and result.get("physics_hz") == 200 and result.get("decoder_hz") == 50
            and controller.get("body_token_version") == "body_token_v1_1"
            and controller.get("pd_profile") == result.get("pd_profile") == "fixed_cpp_baseline"
            and controller.get("hand_torque_profile") == result.get("hand_torque_profile") == "recording_0p7_nm"
            and controller.get("observation_phase") == "post_step_mj_forward_same_qpos"
            and np.array_equal(controller["body_kp"], BODY_KP) and np.array_equal(controller["body_kd"], BODY_KD),
            "Actual fixed SONIC controller recipe changed")
    decoder = Path(args["sonic_resources"]) / "model_decoder.onnx"
    observation = Path(args["sonic_resources"]) / "observation_config.yaml"
    require(sha(decoder) == controller.get("decoder_sha256") == OFFICIAL_DECODER_SHA256
            and sha(observation) == OFFICIAL_OBSERVATION_SHA256, "Official SONIC decoder/observation bytes changed")
    source = load_raw_source(args["source_episode"], raw_root=Path(args["raw_root"]), asset_root=Path(args["data_root"]))
    source["values"]["hand_condition"] = args["hand_condition"]
    projection_sources = projection_bindings(args)
    dependency_sources = {path: expected for path, expected in projection_sources.items() if path != str(decoder)}
    require(result.get("expert_dependency_sha256", {}) == dependency_sources
            and result["source_sha256"] == source["sources"]
            and read(own(run, "source_evidence.json")) == source["evidence"],
            "Actual independently reloaded TRAIN source differs")
    config = deepcopy(source["config"]); config["uid"] = "g1_wholebody_handover_teleop"
    config = base.remap_paths(config, Path(args["data_root"]), Path(args["data_root"]))
    config, proposal = randomize_config(config, args["seed"], Path(args["data_root"]),
        xy_max_m=args["xy_max_m"], yaw_max_rad=args["yaw_max_rad"])
    initial = transformed_initial(source["initial"], proposal)
    require(read(own(run, "scene_proposal.json")) == proposal and read(own(run, "environment_config.json")) == config
            and read(own(run, "initialization_input.json")) == initial,
            "Frozen scene/config/cold-start cannot be regenerated from its TRAIN source")
    for field, name in (("scene_proposal_sha256", "scene_proposal.json"),
                        ("execution_args_sha256", "execution_args.json"), ("source_evidence_sha256", "source_evidence.json")):
        require(result[field] == sha(own(run, name)), "Registered execution metadata SHA differs")
    bindings = {str(own(run, name)): sha(own(run, name)) for name in (
        "result.json", "execution_args.json", "source_evidence.json", "scene_proposal.json",
        "environment_config.json", "initialization_input.json", "provenance/code_manifest.json")}
    bindings.update({str(decoder): OFFICIAL_DECODER_SHA256, str(observation): OFFICIAL_OBSERVATION_SHA256})
    bindings.update(projection_sources)
    bindings.update(placement_sources)
    entries = result["code_snapshots"]
    require(read(own(run, "provenance/code_manifest.json")) == entries
            and len(entries) == len(result["capture_code_sha256"])
            and {e["original_path"] for e in entries} == set(result["capture_code_sha256"]),
            "Executable source snapshots are incomplete or ambiguous")
    for entry in entries:
        expected = result["capture_code_sha256"][entry["original_path"]]
        require(sha(own(run, entry["path"])) == entry["sha256"] == expected
                and sha(entry["original_path"]) == expected, "Actual executable or archived source changed")
        bindings[str(own(run, entry["path"]))] = expected
    for path, expected in result["source_sha256"].items():
        require(sha(path) == expected, "Bound raw source/resource changed")
    return source, config, proposal, initial, bindings


def _history(run, result, initial):
    import mujoco
    import collect_handover_quality100 as legacy
    from handover_geometry_quality import build_container_geometry_cache
    forces = legacy.audit_applied_forces(run, result)
    model = mujoco.MjModel.from_binary_path(str(own(run, "physics_model.mjb"))); data = mujoco.MjData(model)
    require(np.array_equal(model.opt.gravity, [0, 0, -9.81]) and model.neq == 0, "Gravity or attachment contract changed")
    with np.load(own(run, "trajectory.npz"), allow_pickle=False) as values:
        spec = int(values["physics_state_spec"]); states = values["physics_state_integration"].copy()
        times = values["sim_time_s"].copy(); names = values["joint_names"].tolist()
        auxiliary_q = values["actual_joint_q43"].copy(); auxiliary_root = values["root_pose7"].copy()
        auxiliary_vel = values["actual_qvel"].copy()
    total = result["completed_control_ticks"]
    require(type(total) is int and total >= 100 and result["source_commands"] == total - 50
            and result["expert_control_ticks"] == total - 50 and len(states) == total + 1
            and result["trajectory"]["rows"] == total + 1 and names == initial["joint_names"]
            and len(names) == len(set(names)) == 43, "Complete generated integration/native control count differs")
    require(np.allclose(times, .005 + np.arange(total + 1) / 50, rtol=0, atol=1e-6)
            and math.isclose(result["initialization"]["sim_time_s"], .005, abs_tol=1e-12), "Absolute cold-start clock changed")
    qadr = [int(model.joint(n).qposadr[0]) for n in names]; dadr = [int(model.joint(n).dofadr[0]) for n in names]
    pelvis = model.body("pelvis").id; root_joint = int(model.body_jntadr[pelvis])
    root_qadr = int(model.jnt_qposadr[root_joint]); root_dadr = int(model.jnt_dofadr[root_joint])
    cache = read(own(run, "container_geometry_cache.json"))
    roles = {role: model.body(cache[role + "_root_body_name"]).id for role in ("target", "container")}
    require(len({pelvis, *roles.values()}) == 3, "Robot/target/container bodies are not independent")
    q, dq, root, root_vel, objects = [], [], [], [], {role: [] for role in roles}
    for index, state in enumerate(states):
        mujoco.mj_setState(model, data, state, spec)
        require(math.isclose(data.time, times[index], abs_tol=1e-6), "Integration state time differs from its ledger")
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data); mujoco.mj_camlight(model, data)
        q.append(data.qpos[qadr].copy()); dq.append(data.qvel[dadr].copy())
        root.append(data.qpos[root_qadr:root_qadr + 7].copy()); root_vel.append(data.qvel[root_dadr:root_dadr + 6].copy())
        for role, body in roles.items():
            j = int(model.body_jntadr[body]); require(j >= 0 and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE,
                                                     "Actual task object must own a free joint")
            adr = int(model.jnt_qposadr[j]); objects[role].append(data.qpos[adr:adr + 7].copy())
            require(np.allclose(data.xpos[body], objects[role][-1][:3], rtol=0, atol=1e-10), "Object root/kinematics differ")
        require(np.array_equal(auxiliary_q[index], q[-1]) and np.array_equal(auxiliary_vel[index], data.qvel)
                and np.allclose(auxiliary_root[index], root[-1], rtol=0, atol=1e-12), "Auxiliary data differs from integration authority")
        if index == 0:
            require(np.array_equal(q[-1], initial["joint_qpos"])
                    and np.array_equal(data.qvel[dadr], initial["joint_qvel"]), "Initial named source q/dq changed")
            same_pose(root[-1], initial["root_pose7"], "initial root")
            require(np.array_equal(root_vel[-1], initial["root_qvel6"]), "Initial root qvel changed")
            expected_velocity = np.zeros(model.nv)
            expected_velocity[root_dadr:root_dadr + 6] = initial["root_qvel6"]
            expected_velocity[dadr] = initial["joint_qvel"]
            require(np.array_equal(data.qvel, expected_velocity), "Unregistered object/root/joint initial velocity")
            for role, body in roles.items():
                same_pose(objects[role][-1], initial["objects"][role]["pose7"], "initial " + role)
                require(math.isclose(model.body_mass[body], initial["objects"][role]["mass"], rel_tol=1e-5, abs_tol=1e-7), "Object mass changed")
            rebuilt = build_container_geometry_cache(model, data, container_body=roles["container"], target_body=roles["target"])
            require(set(cache) == set(rebuilt), "Geometry-cache schema differs")
            for key, wanted in rebuilt.items():
                require(np.allclose(cache[key], wanted, rtol=0, atol=1e-10) if isinstance(wanted, list)
                        else cache[key] == wanted, "Geometry cache differs from actual model: " + key)
    guard = rows(own(run, result["force_guard"]["path"]))
    require(np.allclose([r["sim_time_s"] for r in guard], times[0] + np.arange(1, 4 * total + 1) / 200,
                        rtol=0, atol=1e-6), "Actual200Hz force-guard clock differs")
    return {"model": model, "data": data, "states": states, "spec": spec, "times": times, "names": names,
        "q": np.asarray(q), "dq": np.asarray(dq), "root": np.asarray(root), "root_vel": np.asarray(root_vel),
        "objects": {k: np.asarray(v) for k, v in objects.items()}, "roles": roles, "forces": forces}


def validate_command_ledger(commands, trace, history, result):
    """Compare actual generated30Hz labels with every held50Hz controller call."""
    import replay_handover_teacher as base
    total = len(trace); n = total - 50
    require(total == result["completed_control_ticks"] and len(commands) == (n - 1) * 30 // 50 + 1,
            "Generated30Hz command ledger is incomplete")
    hand_names = list(base.HAND_NAMES)
    for ordinal, command in enumerate(commands):
        tick = (ordinal * 50 + 29) // 30
        require(command["command_index30"] == ordinal and command["teacher_tick50"] == tick
                and math.isclose(command["sim_time_s"], history["times"][50 + tick], abs_tol=1e-6)
                and command["hand_joint_names"] == hand_names, "Generated command identity/clock/hand order changed")
        vector(command["body_token64"], 64, "generated body token"); vector(command["hand14"], 14, "generated hands")
    warmup = result["warmup_control"]; live = warmup["events"]
    require(warmup["kind"] == "live_initial_pose_sonic_v1_1" and warmup["encoder_calls"] == len(live) == 40
            and warmup["state_forcing"] is False and warmup["history_injection"] is False,
            "Real live-pose warmup evidence changed")
    from prepare_carrybox_v11 import ENCODER_SHA
    require(warmup.get("encoder_sha256") == ENCODER_SHA and warmup.get("hand_policy") == "measured"
            and warmup.get("initialization_count") == 1
            and math.isclose(warmup["first_sim_time_s"], history["times"][0], abs_tol=1e-6),
            "Warmup encoder/policy/initialization identity differs")
    helper = REPO / "VLMExpert/third_party/SIMPLE/src/simple/baselines/live_pose_sonic_warmup.py"
    require(warmup.get("helper_source_sha256") == sha(helper) and warmup.get("encoder_mode") == 0,
            "Original live warmup helper/mode changed")
    equal32(warmup["reference_body_q29"], history["q"][0, :29], 29, "first actual live body reference")
    require(np.allclose(warmup["reference_root_wxyz"], history["root"][0, 3:], rtol=0, atol=1e-12), "Warmup first root differs")
    for name in hand_names:
        require(np.float32(warmup["reference_hands"][name]) == np.float32(history["q"][0, history["names"].index(name)]),
                "Warmup frozen hands are not initial measured hands")
    decoder = 0
    for tick, row in enumerate(trace):
        require(row["tick50"] == row["physics_state_row_before"] == tick and row["physics_state_row_after"] == tick + 1
                and math.isclose(row["sim_time_s"], history["times"][tick], abs_tol=1e-6)
                and math.isclose(row["next_sim_time_s"], history["times"][tick + 1], abs_tol=1e-6)
                and math.isclose(row["elapsed_s"], tick / 50, abs_tol=1e-6) and row["finite_physics"] is True,
                "Native50Hz trace/state/clock differs")
        for side, index in (("before", tick), ("after", tick + 1)):
            for role in ("target", "container"):
                same_pose(row["measured_" + side][role + "_pose7"], history["objects"][role][index], "trace " + role)
        actual_hands = row["actual_hand_targets"]
        require(set(actual_hands) == set(hand_names), "Actual controller hand names differ")
        if tick < 10:
            require(row["phase"] == "warmup_measured_pose_pd" and row["actual_body_token64"] is None
                    and row["expert_command_index30"] is None, "First ten warmup samples invent labels")
            expected = history["q"][tick, [history["names"].index(name) for name in hand_names]].astype(np.float32)
            increment = 0
        elif tick < 50:
            event = live[tick - 10]
            require(row["phase"] == "warmup_live_pose_sonic_v1_1" and row["expert_command_index30"] is None
                    and math.isclose(event["sim_time_s"], history["times"][tick], abs_tol=1e-6)
                    and event["state_forcing"] is False and event["history_injection"] is False,
                    "Live warmup provenance/clock changed")
            equal32(row["actual_body_token64"], event["token64"], 64, "live warmup token")
            require(np.allclose(event["actual_root_wxyz"], history["root"][tick, 3:], rtol=0, atol=1e-12), "Live warmup actual root changed")
            expected = [warmup["reference_hands"][name] for name in hand_names]; increment = 1
        else:
            ordinal = (tick - 50) * 30 // 50; command = commands[ordinal]
            require(row["phase"] == "online_object_feedback_expert" and row["expert_command_index30"] == ordinal,
                    "Actual50Hz controller call does not hold the deployed30Hz generated command")
            equal32(row["actual_body_token64"], command["body_token64"], 64, "held online body token")
            expected = command["hand14"]; increment = 1
        equal32([actual_hands[name] for name in hand_names], expected, 14, "held named hand command")
        require(row["decoder_calls_before"] == decoder and row["decoder_calls_after"] == decoder + increment,
                "Actual decoder/history count changed")
        decoder += increment
    require(result["decoder_calls"] == decoder, "Final decoder count differs")


def _verify_warmup_encoder(expert, history, result):
    from simple.baselines.live_pose_sonic_warmup import reference_input
    warmup = result["warmup_control"]
    for index, event in enumerate(warmup["events"]):
        value = reference_input(np.asarray(warmup["reference_body_q29"]), np.asarray(warmup["reference_root_wxyz"]),
                                history["root"][10 + index, 3:])
        actual = expert.session.run(None, {expert.session.get_inputs()[0].name: value})[0][0]
        equal32(actual, event["token64"], 64, "independently reencoded live warmup token")


def _expert_feedback(samples, tick):
    feedback = dict(samples[tick - 1]) if tick else {}
    feedback["recent_contact_samples"] = samples[max(0, tick - 30):tick]
    return feedback


def restore_private_expert_state(history, data, row):
    """Refresh private prestate/contact geometry exactly as synchronized obs.

    Forward-computed forces are never quality or support evidence. The expert's
    foot polygon reads contact positions/distances/normals, so kinematics alone
    cannot reproduce this input. No physics integration step is performed.
    """
    import mujoco
    mujoco.mj_setState(history["model"], data, history["states"][row], history["spec"])
    mujoco.mj_forward(history["model"], data)


def full_outer_footprint_final50(history):
    """Rebuild actual full collision extents; outer hull is not an inner cavity."""
    import mujoco
    from handover_geometry_quality import _collision_vertices_world, _hull_xy
    model = history["model"]; data = mujoco.MjData(model)
    require(len(history["states"]) >= 101, "Full-footprint final50 requires complete post-warmup motion")
    measurements = []
    for row in range(len(history["states"]) - 50, len(history["states"])):
        mujoco.mj_setState(model, data, history["states"][row], history["spec"])
        mujoco.mj_kinematics(model, data); mujoco.mj_comPos(model, data)
        container, _, _, _ = _collision_vertices_world(model, data, history["roles"]["container"])
        target, _, _, _ = _collision_vertices_world(model, data, history["roles"]["target"])
        hull = _hull_xy(container)
        edges = np.roll(hull, -1, axis=0) - hull
        offset = target[:, None, :2] - hull[None]
        clearance = ((edges[None, :, 0] * offset[:, :, 1] - edges[None, :, 1] * offset[:, :, 0])
                     / np.linalg.norm(edges, axis=1)[None])
        require(np.isfinite(clearance).all(), "Nonfinite actual full-footprint clearance")
        minimum = float(clearance.min())
        measurements.append({"physics_state_row": row, "sim_time_s": float(data.time),
            "target_collision_vertex_count": len(target), "container_collision_vertex_count": len(container),
            "outer_hull_vertex_count": len(hull), "minimum_outer_hull_margin_m": minimum,
            "all_target_collision_vertices_inside_outer_hull": bool(np.all(clearance >= 0.)),
            "passed": minimum >= .005})
    minimum = min(item["minimum_outer_hull_margin_m"] for item in measurements)
    return {"contract": FOOTPRINT_CONTRACT, "passed": all(item["passed"] for item in measurements),
        "checked_states": len(measurements), "window_states": 50,
        "required_minimum_outer_hull_margin_m": .005, "minimum_outer_hull_margin_m": minimum,
        "all_states_all_vertices_inside_outer_hull": all(item["all_target_collision_vertices_inside_outer_hull"] for item in measurements),
        "measurements": measurements, "physics_steps_performed": 0,
        "geometry_source": "actual_MJB_compiled_collision_vertices_at_each_saved_integration_state",
        "AABB_or_source_bbox_used_as_vertices": False, "exact_inner_cavity_claimed": False,
        "original_handover_physics_v2_modified": False,
        "limitations": "Full target XY vertices within the container outer collision convex hull do not prove exact inner-cavity containment; original solved-force, height, release and stability gates remain required."}


def _replay_commands(commands, samples, history, source, args, result):
    import mujoco
    expert = make_replay_expert(source, args)
    _verify_warmup_encoder(expert, history, result)
    model, data = history["model"], history["data"]
    raw = SimpleNamespace(mujoco=SimpleNamespace(mjModel=model, mjData=data,
        mj_objects={role: data.body(body) for role, body in history["roles"].items()}))
    for command in commands:
        teacher_tick = command["teacher_tick50"]
        restore_private_expert_state(history, data, 50 + teacher_tick)
        token, hands, diagnostic = expert(teacher_tick, raw, feedback=_expert_feedback(samples, teacher_tick))
        equal32(token, command["body_token64"], 64, "independently regenerated expert token")
        equal32(hands, command["hand14"], 14, "independently regenerated expert hands")
        require(_normalize(diagnostic) == command["diagnostics"], "Independent saved-prestate expert diagnostics differ")
    return {"status": "all_generated_commands_reproduced_on_CPU", "commands": len(commands),
            "actual_saved_prestate_used": True, "physics_steps_performed": 0,
            "source_hand_condition": args["hand_condition"], "policy_model_constructed": False}


def replay_native_controls(trace, controller, robot, prepare_state, on_new_command):
    """Exactly one archived50Hz sample per tick, including the full warmup."""
    previous = None
    for tick, row in enumerate(trace):
        prepare_state(tick)
        ordinal = row["expert_command_index30"]
        if ordinal is not None and ordinal != previous:
            on_new_command(tick, ordinal); previous = ordinal
        require(controller.decoder_calls == row["decoder_calls_before"], "Replay controller prefix/decoder count differs")
        controller.apply_physics(robot, row["actual_body_token64"], row["actual_hand_targets"])
        require(controller.decoder_calls == row["decoder_calls_after"], "Replay official50Hz decoder advancement differs")


def _replay_projected_commands(commands, trace, samples, history, source, args, result):
    import mujoco
    from simple.baselines.sonic_v1_1 import SonicV11OnnxController
    from handover_sonic_latent_projector import SonicLatentProjector
    expert = make_replay_expert(source, args)
    _verify_warmup_encoder(expert, history, result)
    model = history["model"]; data = mujoco.MjData(model)
    robot = SimpleNamespace(mjModel=model, mjData=data,
        mj_objects={role: data.body(body) for role, body in history["roles"].items()})
    raw = SimpleNamespace(mujoco=robot)
    controller = SonicV11OnnxController(Path(args["sonic_resources"]), allow_unverified_compatibility=True,
        pd_profile="fixed_cpp_baseline", hand_torque_profile="recording_0p7_nm", sync_mujoco_observations=True)
    projector = SonicLatentProjector(jacobian_batch_view=Path(args["latent_projector_batch_view"]))
    def prepare(tick):
        restore_private_expert_state(history, data, tick)
    def reproduce(tick, ordinal):
        command = commands[ordinal]; teacher_tick = tick - 50
        token, hands, diagnostic = expert(teacher_tick, raw, feedback=_expert_feedback(samples, teacher_tick))
        stored = command["diagnostics"]
        extra = {"latent_projection", "unprojected_body_token64", "applied_body_token_sha256"}
        require(_normalize(diagnostic) == {k: v for k, v in stored.items() if k not in extra},
                "Independent unprojected expert diagnostics differ")
        equal32(token, stored["unprojected_body_token64"], 64, "unprojected expert token")
        actual, projection = replay_projection_step(args, token, diagnostic["desired_body_q29"], diagnostic["phase"],
                                                   raw, controller, projector)
        equal32(actual, command["body_token64"], 64, "projected actual execution token")
        equal32(hands, command["hand14"], 14, "projected execution hands")
        require(stable_projection_diagnostic(_normalize(projection)) == stable_projection_diagnostic(stored["latent_projection"]),
                "Official decoder-history projection diagnostics differ")
        require(hashlib.sha256(np.asarray(actual, np.float32).tobytes()).hexdigest() == stored["applied_body_token_sha256"],
                "Applied projected-token SHA differs")
        require(projection.get("candidate_acceptance_evaluator", "unchanged original official B1") == "unchanged original official B1",
                "Derived Jacobian view cannot accept execution candidates")
    replay_native_controls(trace, controller, robot, prepare, reproduce)
    require(controller.decoder_calls == result["decoder_calls"], "Final projected decoder history differs")
    return {"status": "all_generated_commands_reproduced_on_CPU", "latent_projection_reproduced": True,
        "commands": len(commands), "native50Hz_controller_calls": len(trace), "physics_steps_performed": 0,
        "warmup_included": True, "only_nondeterministic_field_excluded": "CPU_seconds",
        "candidate_acceptance_evaluator": "unchanged original official B1", "policy_model_constructed": False}


def _rgb(run, result, history, commands, bindings):
    import mujoco
    from PIL import Image
    import replay_handover_teacher as base
    import collect_handover_quality100 as legacy
    import pack_handover_quality100 as packer
    capture = run / "capture"; manifest = read(own(capture, "capture.json"))
    require(manifest["contract"] == base.CAPTURE_CONTRACT and manifest["status"] == "raw_capture_completed_pending_independent_audit"
            and manifest["execution_contract"] == EXECUTION_CONTRACT and manifest["commands_generated_online"] is True
            and manifest["source_tokens_reencoded"] is True and manifest["code_unchanged_during_run"] is True
            and manifest["source_episode_index"] == result["source_episode_index"]
            and manifest["source_sha256"] == result["source_sha256"]
            and manifest["capture_code_sha256"] == result["capture_code_sha256"]
            and manifest["scene_proposal_sha256"] == result["scene_proposal_sha256"]
            and manifest["object_restores_during_control"] is False,
            "Complete synchronized real RTX capture identity is missing")
    events_path = own(capture, manifest["events_path"]); events = rows(events_path)
    require(sha(events_path) == manifest["events_sha256"] and manifest["event_count"] == len(events)
            and len(events) == (result["completed_control_ticks"] - 1) * 30 // 50 + 1,
            "Dense real30Hz observation stream is incomplete")
    require(manifest["warmup_event_count"] == 30 and manifest["command_event_count"] == len(events) - 30
            and manifest["command_rate_hz"] == 30 and manifest["physics_hz"] == 200 and manifest["decoder_hz"] == 50,
            "Captured actual warmup/teacher clock counts differ")
    model, data = history["model"], history["data"]
    calibration = legacy.load_frame_calibration(model, result)
    bindings.update(legacy.calibration_asset_bindings(result, calibration))
    bindings[str(own(capture, "capture.json"))] = sha(own(capture, "capture.json")); bindings[str(events_path)] = sha(events_path)
    maxima, fingerprints, seen = {}, [], set()
    for index, event in enumerate(events):
        tick = (index * 50 + 29) // 30; warmup = index < 30
        require(event["event_index"] == event["capture_index30"] == index and event["physical_tick50"] == tick
                and event["action_valid"] is (not warmup) and event["phase"] == ("warmup" if warmup else "teacher_command")
                and math.isclose(event["sim_time_s"], history["times"][tick], abs_tol=1e-6)
                and math.isclose(event["actual_elapsed_s"], tick / 50, abs_tol=1e-6)
                and math.isclose(event["nominal_elapsed_s"], index / 30, abs_tol=1e-9)
                and event["joint_names"] == history["names"], "Captured event/frame/state clock differs")
        require(np.array_equal(vector(event["physical_joint_q43_float64"], 43, "physical q"), history["q"][tick])
                and np.array_equal(vector(event["physical_joint_qvel43"], 43, "physical dq"), history["dq"][tick])
                and np.array_equal(vector(event["state43"], 43, "observed q").astype(np.float32), history["q"][tick].astype(np.float32)),
                "Actual RGB-frame measured state differs from integration")
        same_pose(event["root_pose7"], history["root"][tick], "RGB-frame root")
        require(np.array_equal(vector(event["root_qvel6"], 6, "root velocity"), history["root_vel"][tick]), "RGB-frame root velocity differs")
        for role in ("target", "container"):
            same_pose(event["objects"][role]["pose7"], history["objects"][role][tick], "RGB-frame " + role)
        if warmup:
            require(event["action"] is None and event["source_command_index"] is None, "Warmup invents supervised action labels")
        else:
            command = commands[index - 30]; action = event["action"]
            require(event["source_command_index"] == tick - 50, "New capture generated50Hz index differs")
            equal32(action["body_token64"], command["body_token64"], 64, "captured executed token")
            names = action["hand_joint_names"]
            require(len(names) == len(set(names)) == 14 and set(names) == set(command["hand_joint_names"]), "Captured named hand semantics differ")
            expected = [command["hand14"][command["hand_joint_names"].index(name)] for name in names]
            equal32(action["hand14"], expected, 14, "captured executed hands")
        mujoco.mj_setState(model, data, history["states"][tick], history["spec"])
        errors = legacy.audit_renderer_pose(event, model, data, world_reference=manifest["isaac_world_time_reference_s"],
                                            frame_calibration=calibration)
        for name, value in errors.items(): maxima[name] = max(maxima.get(name, 0.), value)
        rgb = event["rgb"]; image_path = own(capture, rgb["path"])
        require(image_path not in seen and sha(image_path) == rgb["file_sha256"] and image_path.suffix == ".png",
                "Actual raw RTX PNG is reused or changed")
        seen.add(image_path)
        with Image.open(image_path) as image:
            require(image.mode == "RGB" and image.size == (rgb["width"], rgb["height"]), "Actual RTX RGB geometry differs")
            pixels = np.asarray(image)
        pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
        require(rgb["dtype"] == "uint8" and rgb["channels"] == 3 and pixel_sha == rgb["pixel_sha256"], "Raw RTX pixels differ")
        bindings[str(image_path)] = rgb["file_sha256"]
        fingerprints.append({"pixels": pixel_sha, "state": event["state43"], "objects": event["objects"], "action": event["action"]})
    require(all(value <= (legacy.RENDER_POSITION_TOLERANCE_M if key.endswith("_m") else legacy.RENDER_ROTATION_TOLERANCE_RAD)
                for key, value in maxima.items() if not key.startswith("raw_")), "Actual renderer poses exceed original fixed tolerances")
    eligible, histories = packer.eligible_currents(events)
    return {"status": "all_actual_RTX_events_pixels_actions_and_renderer_poses_passed", "event_count": len(events),
            "renderer_maxima": maxima, "eligible_current_rows": eligible, "history_rows": histories,
            "capture_content_sha256": hashlib.sha256(json.dumps(fingerprints, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def audit(run, require_rgb=True, replay_expert=True):
    """Return admission/rejection; incomplete or diagnostic-only checks never admit."""
    run = Path(run).resolve(strict=True)
    report = {"contract": CONTRACT, "status": "independent_capture_rejected", "eligible_dataset": False,
        "run": str(run), "require_rgb": bool(require_rgb), "replay_expert": bool(replay_expert),
        "generated_utc": datetime.now(timezone.utc).isoformat(), "auditor_sha256": sha(__file__),
        "physics_steps_performed": 0, "GPU_used": False, "policy_success_rate_claim": None}
    try:
        import collect_handover_quality100 as legacy
        from handover_random100_scene import audit_scene_snapshot
        from handover_contact_quality import evaluate_capture
        result, args = read(own(run, "result.json")), read(own(run, "execution_args.json"))
        source, config, proposal, initial, bindings = _bindings(run, result, args)
        history = _history(run, result, initial)
        samples = rows(own(run, "quality_samples.jsonl"))
        require(len(samples) == result["source_commands"] and all(row["completed_steps"] == i + 1
                and row["solver_stage"] == "post_mj_step_before_observation_refresh"
                and math.isclose(row["sim_time_s"], history["times"][51 + i], abs_tol=1e-6) for i, row in enumerate(samples)),
                "Original solved50Hz quality stream/clock differs")
        quality = evaluate_capture(samples, initial_target_z_m=float(history["objects"]["target"][0, 2]))
        require(quality == result["physical_quality"] == read(own(run, "physical_quality.json")), "Recomputed originalv2 physical gate differs")
        report["physical_gate"] = quality
        report["actual_full_outer_footprint_final50"] = full_outer_footprint_final50(history)
        require(quality.get("eligible_physics") is True and quality.get("verified_handover_success") is True,
                "Original full handover-physics-v2 quality gate did not pass")
        require(report["actual_full_outer_footprint_final50"]["passed"] is True,
                "Additional actual full outer-footprint final50 dataset gate did not pass")
        commands, trace = rows(own(run, "expert_commands.jsonl")), rows(own(run, "trace.jsonl"))
        validate_command_ledger(commands, trace, history, result)
        if replay_expert:
            report["expert_reproduction"] = (_replay_projected_commands(commands, trace, samples, history, source, args, result)
                if args.get("latent_projector_batch_view") is not None else _replay_commands(commands, samples, history, source, args, result))
        else: report["expert_reproduction"] = {"status": "skipped_diagnostic_only"}
        if require_rgb:
            require(result["render_mode"] == "isaac", "CPU screening cannot certify actual RTX training data")
            scene = read(own(run, "scene_readback.json"))
            require(audit_scene_snapshot(scene["actual_readback"], proposal, initial) == scene,
                    "Actual scene readback receipt no longer passes independent checking")
            application_path = own(run, "scene_visual_application.json")
            application = read(application_path)
            require(sha(application_path) == result["scene_visual_application_sha256"]
                    and application["contract"] == "cvpr-handover-random100-visual-application-v1"
                    and application["proposal_sha256"] == proposal["proposal_sha256"]
                    and application["physics_unchanged"] is True and application["source_assets_edited"] is False
                    and application["observation_refresh_required_before_capture"] is True
                    and application["physics_before_and_after"]["sim_time_s"] <= .005 + 1e-9,
                    "Actual reset-time consuming-shader application receipt differs")
            recorded = {(item["role"], item["path"]): item for item in application["changes"]}
            require(len(recorded) == len(application["changes"]), "Ambiguous visual application paths")
            for role in ("target", "robot"):
                for shader in scene["actual_readback"]["material"][role + "_shaders"]:
                    item = recorded[(role, shader["path"])]
                    require(item["identity"] == shader["identity"]
                            and item["logical_to_shader_input"] == shader["logical_to_shader_input"]
                            and item["bound_consumers"] == shader["bound_consumers"]
                            and item["after"] == shader["inputs"], "Applied shader mapping/consumers/readback changed")
            bindings[str(application_path)] = sha(application_path)
            report["scene_readback_verified"] = True
            report["RGB_state_action_history_audit"] = _rgb(run, result, history, commands, bindings)
            rgb = report["RGB_state_action_history_audit"]
            report.update(event_count=rgb["event_count"], capture_content_sha256=rgb["capture_content_sha256"],
                          eligible_current_rows=rgb["eligible_current_rows"], history_rows=rgb["history_rows"])
        for name in ("physics_model.mjb", "trajectory.npz", "quality_samples.jsonl", "physical_quality.json", "trace.jsonl",
                     "expert_commands.jsonl", "container_geometry_cache.json", result["force_guard"]["path"]):
            bindings[str(own(run, name))] = sha(own(run, name))
        if require_rgb: bindings[str(own(run, "scene_readback.json"))] = sha(own(run, "scene_readback.json"))
        require(all(sha(path) == expected for path, expected in bindings.items()), "Run bytes changed while independently auditing")
        report.update(source_episode_index=result["source_episode_index"], source_split="train", source_sha256=result["source_sha256"],
            bindings=bindings, applied_force_audit=history["forces"], scene_proposal_sha256=result["scene_proposal_sha256"],
            quality_gate_redefined=False, source_fixed_tokens_used_as_new_labels=False,
            placement_encoder_mode=placement_encoder_mode(args), placement_locomotion=placement_locomotion(args))
        report["eligible_dataset"] = bool(require_rgb and replay_expert)
        report["status"] = "independent_capture_audit_passed" if report["eligible_dataset"] else "physics_passed_full_dataset_admission_pending"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    return report
