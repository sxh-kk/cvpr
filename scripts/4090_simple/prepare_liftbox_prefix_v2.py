"""Independent CPU audit and lossless pack of real warmup-prefix Lift data.

The first thirty rows are observations only. Their dense zero action arrays
are explicit unavailable-label placeholders, never commands or supervision.
This entry point cannot accept a v1 pack or grant permission to train.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
from PIL import Image

import collect_liftbox_prefix_v2 as producer
import prepare_lift_box_hold as lift
import rollout_carrybox_dataset as common
from prepare_carrybox_v11 import (TRAIN_STATE_NAMES, TRAIN_HAND_NAMES, VIDEO_KEY,
                                  output_features, vector_stats, numeric_feature)

CONTRACT = "cvpr-simple-lift-real-warmup-prefix-v2"
SPLITS = {"train": [0, 1, 2], "val": [3], "test": [4]}
PREFIX_ROWS = 30
REPO = Path(__file__).resolve().parents[2]
# The first real source0 audit predates retained-evidence support. Its original
# report and exact verifier bytes are preserved, not rewritten as a new audit.
LEGACY_ACCEPTED_REPORTS = {
    "b05560f8fd4296cd1ef397bef7c5cf141a83c35934089eff96577fe7b6de7cf0":
    "b9ef65e15497557a90beca9790cc86d33e6d0f2f6a2d2b9a3c817c5988fdc274",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def equal(left, right, name):
    require(np.array_equal(np.asarray(left), np.asarray(right)), f"Different {name}")


def declared_file(root, path):
    """Canonicalize both sides before containment; reject escaping symlinks."""
    root = Path(root).resolve(strict=True)
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(root), "Declared artifact escapes its physical run directory")
    return common.owned_file(root, path.relative_to(root))


def same_directory(value, root):
    return isinstance(value, (str, Path)) and Path(value).resolve() == Path(root).resolve()


def validate_events(events):
    source = producer.validate_events(events)
    require(type(source) is int and source >= 0, "Invalid source episode")
    require(producer.eligible_window(events, 30), "First supervised current must have real six-frame history and thirty valid future labels")
    return source


def numerical_contract(root, result):
    """Bind every run's declaration bytes; compare cross-run semantics separately."""
    frozen = result["frozen_contract"]
    path = declared_file(root, frozen["path"])
    require(common.sha256(path) == frozen["sha256"], "Frozen declaration SHA differs")
    declaration = read(path)
    require(declaration.get("contract") == CONTRACT, "Old producer declaration is not a v2 capture")
    for key in ("task_contract", "controller_contract", "render_contract"):
        require(declaration.get(key) == result.get(key), f"Frozen {key} differs")
    task = declaration["task_contract"]
    require(task.get("task_id") == lift.TASK_ID and task.get("instruction") == lift.INSTRUCTION and
            task.get("metric") == lift.METRIC and task.get("minimum_oriented_box_clearance_m") == .10 and
            task.get("required_continuous_sim_seconds") == 1. and
            task.get("minimum_hand_contact_normal_force_n_exclusive") == .01 and
            task.get("floor_table_contacts_allowed") is False and task.get("measurement_hz") == 50,
            "Lift task criterion differs")
    control = declaration["controller_contract"]
    expected = {"physics_dt_s": .005, "decoder_hz": 50, "teacher_command_hz": 30,
                "warmup_control_hz": 50, "prefix_observation_hz": 30, "warmup_seconds": 1.,
                "hand_policy": "open", "observation_phase": common.OBSERVATION_PHASE,
                "transition": "continuous; no reset; reference frame0 at row30",
                "pd_profile": "official_v1_1_ankle_pitch_1_5", "hand_torque_profile": "mjcf_limits"}
    require(control == expected, "Unexpected continuous warmup/controller contract")
    source_hashes = {}
    for original, receipt in declaration["source_files"].items():
        snapshot = declared_file(root, receipt["snapshot"])
        require(common.sha256(snapshot) == receipt["sha256"] == common.sha256(Path(original)),
                "Producer/controller/warmup source changed since collection")
        source_hashes[Path(original).name] = receipt["sha256"]
    required = {"collect_liftbox_prefix_v2.py", "sonic_v1_1.py", "live_pose_sonic_warmup.py",
                "g1_wholebody_lift_box_hold_sonic.py", "replay_carrybox_native50.py", "replay_carrybox_v11.py"}
    require(set(source_hashes) == required, "Incomplete frozen producer source identities")
    renderer = result["render_contract"]
    lift.validate_render_contracts([renderer])
    source_root = REPO / "VLMExpert/third_party/SIMPLE/src/simple"
    for key, filename in (("bridge_source_sha256", "engines/isaacsim.py"),
                          ("environment_source_sha256", "envs/sonic_loco_manip.py")):
        require(renderer.get(key) == common.sha256(source_root / filename), "RTX bridge/environment source identity changed")
    return {key: declaration[key] for key in ("task_contract", "controller_contract", "render_contract",
                                             "robot_mjcf_sha256")} | {"source_sha256": source_hashes}


def validate_result(root, result):
    require(result.get("contract") == CONTRACT and result.get("status") == "task_terminated" and
            result.get("physical_outcome_status") == "task_terminated" and
            result.get("task_id") == lift.TASK_ID and result.get("instruction") == lift.INSTRUCTION and
            result.get("task_success_observed") is True and result.get("oracle_success_without_fall") is True and
            result.get("terminated_after_teacher") is True and result.get("fell") is False and
            result.get("finite_states") is True and result.get("run_role") == "pilot_collection",
            "Require a newly executed successful finite non-fallen v2 pilot, not a relabeled v1/failure")
    source = result["teacher"]["source_episode_index"]
    require(result.get("seed") == 10001 + source and result.get("warmup_hand_policy") == "open" and
            result.get("warmup_seconds_requested") == 1. and result.get("prefix_rows") == 30 and
            result.get("controller_reset_at_transition") is False and result.get("warmup_decoder_calls") == 40 and
            result.get("actual_warmup_sim_seconds") is not None and
            abs(result["actual_warmup_sim_seconds"] - 1.) < 1e-7 and
            result.get("observation_phase") == common.OBSERVATION_PHASE,
            "Missing real continuous one-second open warmup")
    require(result["teacher"].get("token_mode") == "feedback-heading-v11" and
            result["teacher"].get("reference_anchor_A") == "identity; source world trajectory stays fixed; no spatial retargeting",
            "Unexpected teacher reference mode/anchor")
    from simple.baselines.sonic_v1_1 import OFFICIAL_DECODER_SHA256
    require(result["controller_identity"].get("decoder_sha256") == OFFICIAL_DECODER_SHA256,
            "Wrong official decoder")
    initialization = result["initialization"]
    require(initialization.get("initialization_count") == 1 and
            initialization.get("subsequent_state_assignment") == "none; physical PD and decoder only",
            "Source initialization must occur once before warmup")
    semantic = numerical_contract(root, result)
    log = common.read_jsonl(Path(result["attempt_log"]["path"]).resolve(strict=True))
    rows = [row for row in log if row.get("attempt_id") == result["attempt_log"]["attempt_id"]]
    require(all(any(row.get("phase") == phase and same_directory(row.get("run_output"), root) and
                    row.get("source_episode") == source and row.get("run_role") == "pilot_collection" and
                    (phase != "finished" or row.get("status") == "task_terminated") and
                    row.get("frozen_contract_sha256") == result["frozen_contract"]["sha256"]
                    for row in rows) for phase in ("started", "finished")),
            "Missing bound append-only attempt facts")
    return semantic


def audit_clock(trajectory, trace, events, result):
    times = np.asarray(trajectory["sim_time_s"], float)
    count = len(trace)
    require(count > 50 and times.shape == (count + 1,) and np.isfinite(times).all() and
            np.allclose(np.diff(times), .02, atol=1e-7, rtol=0), "Incomplete physical50Hz trajectory including terminal")
    equal(trajectory["native_frame_index"], np.arange(count + 1), "global physical native indices")
    expected_phases = ["warmup_observation_prefix"] * 50 + ["oracle_teacher_token"] * (count - 50) + ["task_terminal"]
    equal(trajectory["phase"], expected_phases, "physical phases")
    require(abs(times[0] - result["initial_sim_time_s"]) < 1e-9 and
            abs(times[50] - result["teacher_start_sim_time_s"]) < 1e-9 and
            abs(times[-1] - times[0] - result["actual_sim_seconds"]) < 1e-7,
            "Warmup/teacher/terminal clocks differ")
    require(len(events) == (count - 1) * 30 // 50 + 1, "Missing final real observation events")
    for index, row in enumerate(trace):
        plan = producer.tick_plan(index)
        expected_event = plan["event_index"] if plan["is_event"] else None
        require(row.get("native_frame_index") == index and row.get("phase") == plan["phase"] and
                row.get("source_ref_frame") == plan["source_ref_frame"] and
                row.get("teacher_source_time_s") == plan["teacher_source_time_s"] and
                row.get("observation_event_index") == expected_event and
                row.get("command_updated") is (plan["is_event"] and plan["action_valid"]) and
                row.get("observation_phase") == common.OBSERVATION_PHASE,
                "Trace source/control/observation event clock differs")
        require(abs(row["sim_time_s"] - times[index]) < 1e-9 and
                abs(row["next_sim_time_s"] - times[index+1]) < 1e-9, "Trace time differs from physical trajectory")
        before, after = row["decoder_calls_before_step"], row["decoder_calls_after_step"]
        require(before == max(0, index - 10) and after - before == int(index >= 10),
                "Actual decoder counts must be10PD then uninterrupted50Hz decoder")
        if index >= 50:
            command = events[plan["event_index"]]
            equal(row["applied_body_token"], command["executed_token64"], "actually held teacher body token")
            equal([row["applied_hand_targets"][name] for name in TRAIN_HAND_NAMES],
                  command["hand_targets14"], "actually held teacher hand target")
            require((row.get("runtime_encoder") == command["runtime_encoder"]) if plan["is_event"]
                    else row.get("runtime_encoder") is None, "Held teacher command was regenerated off-clock")
    require(result["decoder_calls"] == count - 10, "Final actual decoder count differs")


def audit_capture(root, result, events, trajectory):
    capture = Path(result["command_capture_root"]).resolve(strict=True)
    manifest = read(common.owned_file(capture, "capture.json"))
    require(manifest.get("contract") == CONTRACT and manifest.get("state") == "capture_closed" and
            manifest.get("instruction") == lift.INSTRUCTION and manifest.get("event_count") == len(events) and
            manifest.get("prefix_rows") == 30 and
            tuple(manifest.get("state43_joint_names", [])) == TRAIN_STATE_NAMES and
            tuple(manifest.get("hand14_joint_names", [])) == TRAIN_HAND_NAMES,
            "Capture is unsealed or lacks the actual v2 task/named schema")
    metadata = manifest["metadata"]
    require(Path(metadata["run_output"]).resolve() == root and metadata["source_episode"] == result["teacher"]["source_episode_index"] and
            metadata["seed"] == result["seed"] and metadata["task_id"] == lift.TASK_ID and
            metadata["run_role"] == "pilot_collection" and metadata["frozen_contract"] == result["frozen_contract"],
            "Lossless capture belongs to another physical run")
    for field, filename in (("events_jsonl_sha256", "events.jsonl"), ("observations_npz_sha256", "observations.npz"),
                            ("collector_sha256", "collector_source.py")):
        require(manifest[field] == common.sha256(common.owned_file(capture, filename)), "Sealed capture artifact SHA differs")
    require(manifest["collector_sha256"] == common.sha256(producer.__file__), "Capture collector identity differs from the audited producer")
    captured = common.read_jsonl(capture / "events.jsonl")
    require(captured == events, "Archived observation events differ from actual collector events")
    with np.load(capture / "observations.npz", allow_pickle=False) as numeric:
        keys = {"event_index", "native_frame_index", "sim_time_s", "collection_elapsed_s",
                "nominal_timestamp_s", "state43", "base_pose7", "box_pose7", "action_valid", "phase"}
        require(set(numeric.files) == keys, "Incomplete raw observation numeric archive")
        for key in keys:
            equal(numeric[key], [row[key] for row in events], f"sealed {key}")
    pngs = []
    for row in events:
        index = row["native_frame_index"]
        for field, key in (("state43", "actual_state43"), ("base_pose7", "actual_base_pose"), ("box_pose7", "actual_box_pose")):
            equal(row[field], trajectory[key][index], f"same-tick {field}")
        require(abs(row["sim_time_s"] - trajectory["sim_time_s"][index]) < 1e-9, "RGB/event clock shifted")
        png = common.owned_file(capture, row["rgb_lossless_png"])
        require(common.sha256(png) == row["rgb_png_file_sha256"], "PNG file bytes changed")
        with Image.open(png) as image:
            rgb = np.asarray(image)
        require(list(rgb.shape) == row["rgb_shape"] and row["rgb_dtype"] == "uint8" and
                common.raw_rgb_sha(rgb) == row["rgb_sha256"], "Actual observation PNG pixels changed")
        pngs.append(png)
    return pngs


def verify_warmup(model_path, trajectory, trace, result):
    """Recompute desired commands from first live pose and actual recorded roots."""
    import mujoco
    from simple.baselines.live_pose_sonic_warmup import LivePoseSonicWarmup
    from simple.core.action import ActionCmd
    model = mujoco.MjModel.from_binary_path(str(model_path))
    data = mujoco.MjData(model)
    resources = Path(result["controller_identity"]["provenance"]).resolve(strict=True).parent
    warmup = LivePoseSonicWarmup(resources, hand_policy="open")
    class CommandAudit:
        def __init__(self):
            self._history = []  # Each append below is a real archived50Hz physical sample.
        def hold(self, obs):
            named = dict(zip(obs["joint_names"], obs["joint_qpos"]))
            return ActionCmd("sonic_v1_1", body_token=None,
                             hand_targets={name: float(named[name]) for name in TRAIN_HAND_NAMES})
        def step(self, token, hands, obs):
            return ActionCmd("sonic_v1_1", body_token=token, hand_targets=hands)
    controller = CommandAudit()
    for index in range(50):
        mujoco.mj_setState(model, data, trajectory["physics_state_integration"][index], 8191)
        mujoco.mj_forward(model, data)
        observation = {"joint_qpos": np.asarray(trajectory["actual_state43"][index], np.float32),
                       "joint_names": TRAIN_STATE_NAMES, "sim_time_s": float(data.time)}
        command = warmup.command(SimpleNamespace(mjData=data), controller, observation)
        row = trace[index]
        require(row["control_phase"] == warmup.phase and row.get("runtime_encoder") is None,
                "Prefix must use actual PD/live warmup, not teacher future")
        if index < 10:
            require(row["applied_body_token"] is None, "Initial PD cannot claim a latent")
        else:
            equal(row["applied_body_token"], command["body_token"], "actual live warmup encoder token")
        equal([row["applied_hand_targets"][name] for name in TRAIN_HAND_NAMES],
              [command["hand_targets"][name] for name in TRAIN_HAND_NAMES], "actual warmup hand target")
        controller._history.append({"sim_time_s": float(data.time), "state43": observation["joint_qpos"].tolist()})
    require(warmup.receipt == result["warmup_control"], "First live reference or actual40 warmup commands changed")
    return {"real_pd_samples": 10, "live_encoder_commands": 40, "tokens_bitwise_equal": True,
            "receipt_recomputed": True, "hand_policy": "open", "helper_source_sha256": warmup.receipt["helper_source_sha256"],
            "encoder_sha256": warmup.receipt["encoder_sha256"],
            "scope": "Auditor uses archived physical samples; it does not invent producer history or render RGB"}


def verify_teacher(result, events, trace, trajectory):
    from replay_carrybox_native50 import load_native
    from prepare_carrybox_v11 import FUTURE_OFFSETS
    teacher = result["teacher"]
    resources = Path(result["controller_identity"]["provenance"]).resolve(strict=True).parent
    values, receipt = load_native(Path(teacher["source_root"]), teacher["source_episode_index"], resources,
                                  Path(teacher["state_action_contract"]), "feedback-heading-v11")
    require(receipt == teacher, "Original native source/reference receipt changed")
    active = [dict(row, source_time_s=row["teacher_source_time_s"]) for row in trace[50:]]
    commands = [dict(row, source_time_s=row["teacher_source_time_s"]) for row in events[30:]]
    reproduced = common.verify_encoder_events(result, commands, active, full_source=False)
    for row in events[30:]:
        frame = row["source_ref_frame"]
        require(frame + max(FUTURE_OFFSETS) < teacher["source_native_frames"], "Incomplete real reference future")
        equal(row["hand_targets14"], values["action"][frame], "teacher original named hand targets")
        equal(row["runtime_encoder"]["reference_frame_indices"], frame + FUTURE_OFFSETS, "teacher reference future indices")
    init = result["initialization"]
    for field, key in (("source_state43", "observation.state"), ("source_base_pose", "observation.base_pose"),
                       ("source_base_vel_qvel_order", "observation.base_vel"), ("source_object_pose", "observation.object_poses")):
        equal(init[field], values[key][0], f"source initialization {field}")
    equal(trajectory["actual_state43"][0], np.asarray(values["observation.state"][0], np.float32),
          "first physical source joint state")
    for field, key in (("actual_base_pose", "observation.base_pose"), ("actual_box_pose", "observation.object_poses")):
        require(np.allclose(trajectory[field][0], values[key][0], atol=1e-10, rtol=0), "First physical source pose differs")
    return reproduced


def audit_run(path):
    root = Path(path).resolve(strict=True)
    result_path = common.owned_file(root, "result.json")
    result = read(result_path)
    semantic = validate_result(root, result)
    capture = Path(result["command_capture_root"]).resolve(strict=True)
    events = common.read_jsonl(common.owned_file(capture, "events.jsonl"))
    source = validate_events(events)
    require(source == result["teacher"]["source_episode_index"] and result["observation_rows"] == len(events) and
            result["teacher_command_events"] == len(events) - 30, "Observation/teacher row counts differ")
    model_path = declared_file(root, result["physics_model"]["path"])
    require(common.sha256(model_path) == result["physics_model"]["sha256"], "Authoritative MJB changed")
    trajectory_path, trace_path = common.owned_file(root, "trajectory.npz"), common.owned_file(root, "trace.jsonl")
    require(common.sha256(trajectory_path) == result["trajectory_sha256"], "Trajectory SHA differs")
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {key: archive[key] for key in archive.files}
    require(trajectory["physics_state_integration"].shape[1:] == (548,) and
            int(trajectory["physics_state_spec"]) == 8191, "Need complete548-element integration states")
    trace = common.read_jsonl(trace_path)
    audit_clock(trajectory, trace, events, result)
    pngs = audit_capture(root, result, events, trajectory)
    warmup = verify_warmup(model_path, trajectory, trace, result)
    reproduced = verify_teacher(result, events, trace, trajectory)
    physical = lift.reconstruct_lift(model_path, trajectory, result, trace)
    # Recheck every logged post-step predicate against its solved integration state.
    import mujoco
    model = mujoco.MjModel.from_binary_path(str(model_path)); data = mujoco.MjData(model)
    hold, previous = 0., False
    for index, row in enumerate(trace):
        mujoco.mj_setState(model, data, trajectory["physics_state_integration"][index+1], 8191)
        mujoco.mj_forward(model, data)
        measured = lift.lift_measurement(model, data); progress = row["task_progress"]
        hold = hold + .02 if measured["condition"] and previous else 0.
        previous = measured["condition"]
        require(abs(progress["sim_time_s"] - data.time) < 1e-9 and
                abs(progress["box_lowest_z_m"] - measured["box_bottom_z_m"]) < 1e-8 and
                progress["loaded_hand_contact"] == (measured["maximum_hand_normal_force_n"] > .01) and
                progress["condition"] == measured["condition"] and
                (progress["floor_contact"] or progress["table_contact"]) == measured["floor_or_table_contact"],
                "Logged task force/geometric predicate differs from exact physical reconstruction")
        require(abs(progress["hold_sim_s"] - hold) < 1e-7 and
                progress["success"] == (hold >= 1.-1e-9), "Logged continuous hold or task success differs")
    require(trace[-1]["terminated"] is True and not any(row["terminated"] for row in trace[:-1]) and
            result["final_task_progress"] == trace[-1]["task_progress"], "Missing unique actual teacher task terminal")
    return events, pngs, {"source_episode": source, "run_root": str(root), "seed": result["seed"],
        "instruction": lift.INSTRUCTION, "task_id": lift.TASK_ID, "status": "task_terminated",
        "prefix_rows": 30, "history_only_rows": 30, "valid_teacher_rows": len(events)-30,
        "result_sha256": common.sha256(result_path), "trace_sha256": common.sha256(trace_path),
        "trajectory_sha256": common.sha256(trajectory_path), "physics_model_sha256": common.sha256(model_path),
        "command_events_sha256": common.sha256(capture/"events.jsonl"),
        "capture_manifest_sha256": common.sha256(capture/"capture.json"),
        "png_file_sha256": [common.sha256(png) for png in pngs], "controller_identity": result["controller_identity"],
        "controller_contract": result["controller_contract"], "render_contract": result["render_contract"],
        "frozen_contract": result["frozen_contract"], "semantic_contract": semantic,
        "attempt_log": result["attempt_log"], "run_role": result["run_role"],
        "source_parquet_sha256": result["teacher"]["source_parquet_sha256"],
        "source_archive_sha256": result["teacher"]["source_archive_sha256"],
        "encoder_sha256": result["teacher"]["encoder_sha256"], "encoder_reproduction": reproduced,
        "warmup_reproduction": warmup, "reconstructed_lift": physical, "measurement_hz": 50,
        "audit_dependency_sha256": audit_dependencies()}


def audit_dependencies():
    from prepare_carrybox_v11 import __file__ as converter
    return {str(Path(path).resolve()):common.sha256(path) for path in (lift.__file__,common.__file__,converter)}


def accepted_identity(root, receipt):
    """Bind unchanged accepted numerics to every raw/code/resource byte.

    The shared attempt log is append-only: bind this attempt's exact facts,
    rather than treating later appended attempts as a mutation of this run.
    No contact solver, reference encoder or PNG pixel decoder runs here.
    """
    root = Path(root).resolve(strict=True)
    result = read(root/"result.json")
    require(validate_result(root,result) == receipt["semantic_contract"], "Accepted protocol semantics changed")
    files = {}
    def bind(path, expected=None):
        path=Path(path).resolve(strict=True);digest=common.sha256(path)
        require(expected is None or digest==expected, f"Accepted bytes changed: {path}")
        files[str(path)]=digest
    for filename,key in (("result.json","result_sha256"),("trace.jsonl","trace_sha256"),
                         ("trajectory.npz","trajectory_sha256")):
        bind(common.owned_file(root,filename),receipt[key])
    bind(declared_file(root,result["physics_model"]["path"]),receipt["physics_model_sha256"])
    if result["physics_model"].get("diagnostic_spec_xml"):
        bind(declared_file(root,result["physics_model"]["diagnostic_spec_xml"]),result["physics_model"]["diagnostic_spec_xml_sha256"])
    evidence=declared_file(root,result["controller_identity"]["compatibility_evidence"])
    bind(evidence)
    require(read(evidence).get("evidence")==result["teacher"],"Actual teacher compatibility evidence changed")
    if result.get("video"):
        bind(declared_file(root,result["video"]["path"]))
    capture=Path(result["command_capture_root"]).resolve(strict=True)
    bind(common.owned_file(capture,"capture.json"),receipt["capture_manifest_sha256"])
    manifest=read(capture/"capture.json")
    bind(common.owned_file(capture,"events.jsonl"),receipt["command_events_sha256"])
    for key,filename in (("events_jsonl_sha256","events.jsonl"),("observations_npz_sha256","observations.npz"),
                         ("collector_sha256","collector_source.py")):
        bind(common.owned_file(capture,filename),manifest[key])
    events=common.read_jsonl(capture/"events.jsonl");source=validate_events(events)
    require(source==receipt["source_episode"] and len(events)==manifest["event_count"] and
            len(receipt["png_file_sha256"])==len(events),"Accepted event count/source changed")
    pngs=[]
    for row,digest in zip(events,receipt["png_file_sha256"],strict=True):
        require(row["rgb_png_file_sha256"]==digest,"Accepted PNG receipt differs")
        png=common.owned_file(capture,row["rgb_lossless_png"]);bind(png,digest);pngs.append(png)
    teacher=result["teacher"];source_root=Path(teacher["source_root"])
    bind(teacher["source_parquet"],receipt["source_parquet_sha256"])
    bind(source_root.with_suffix(".zip"),receipt["source_archive_sha256"])
    bind(source_root/"meta/info.json")
    bind(teacher["state_action_contract"],teacher["state_action_contract_sha256"])
    frozen=declared_file(root,result["frozen_contract"]["path"]);bind(frozen,receipt["frozen_contract"]["sha256"])
    declaration=read(frozen)
    for original,item in declaration["source_files"].items():
        bind(original,item["sha256"]);bind(declared_file(root,item["snapshot"]),item["sha256"])
    simple=REPO/"VLMExpert/third_party/SIMPLE/src/simple"
    for key,filename in (("bridge_source_sha256","engines/isaacsim.py"),("environment_source_sha256","envs/sonic_loco_manip.py")):
        bind(simple/filename,result["render_contract"][key])
    from simple.baselines.sonic_v1_1 import (OFFICIAL_DECODER_SHA256,OFFICIAL_OBSERVATION_SHA256,OFFICIAL_README_SHA256)
    resources=Path(result["controller_identity"]["provenance"]).resolve(strict=True).parent
    for filename,digest in (("model_encoder.onnx",common.ENCODER_SHA),("model_decoder.onnx",OFFICIAL_DECODER_SHA256),
                            ("observation_config.yaml",OFFICIAL_OBSERVATION_SHA256),("README.md",OFFICIAL_README_SHA256),
                            ("PROVENANCE.json",None)):
        bind(resources/filename,digest)
    dependencies=audit_dependencies()
    require(receipt.get("audit_dependency_sha256",dependencies)==dependencies,"Numerical audit dependency source changed")
    for path,digest in dependencies.items():bind(path,digest)
    attempts=common.read_jsonl(Path(result["attempt_log"]["path"]))
    facts=[row for row in attempts if row.get("attempt_id")==result["attempt_log"]["attempt_id"]]
    fact_sha=hashlib.sha256(json.dumps(facts,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
    return events,pngs,{"files":files,"attempt_id":result["attempt_log"]["attempt_id"],"attempt_facts_sha256":fact_sha,
                        "scope":"All accepted raw/state/model/reference/code/resource/PNG bytes and exact per-attempt facts; later appended attempts remain separate"}


def bind_accepted_run(report_path, verifier_snapshot, *, expected_identity=None):
    """Reuse a concrete accepted report; missing/mismatched evidence is fatal."""
    report_path,verifier_snapshot=Path(report_path).resolve(strict=True),Path(verifier_snapshot).resolve(strict=True)
    accepted=read(report_path)
    verifier_sha=common.sha256(verifier_snapshot)
    require(accepted.get("contract")==CONTRACT and accepted.get("status")=="single_run_cpu_audited" and
            accepted.get("training_authorized") is False and
            verifier_sha==accepted.get("packer_sha256") and
            (verifier_sha==common.sha256(__file__) or
             LEGACY_ACCEPTED_REPORTS.get(common.sha256(report_path))==verifier_sha),
            "Accepted numerical report/verifier snapshot identity missing")
    receipt=accepted["receipt"];root=Path(receipt["run_root"]).resolve(strict=True)
    physical,warmup,teacher=(receipt[key] for key in ("reconstructed_lift","warmup_reproduction","encoder_reproduction"))
    require(receipt.get("status")=="task_terminated" and physical.get("measurement_hz")==50 and
            physical.get("integration_state_spec")==8191 and physical.get("frames_reconstructed",0)>100 and
            np.isfinite(physical.get("final_reconstructed_hold_s",np.nan)) and physical["final_reconstructed_hold_s"]>=1.-1e-7 and
            warmup.get("real_pd_samples")==10 and warmup.get("live_encoder_commands")==40 and
            warmup.get("tokens_bitwise_equal") is True and warmup.get("receipt_recomputed") is True and
            teacher.get("input_sha256_matches") is True and teacher.get("tokens_bitwise_equal") is True and
            teacher.get("events_reproduced")==receipt.get("valid_teacher_rows") and
            accepted.get("event_count")==accepted.get("pngs")==receipt.get("valid_teacher_rows",0)+30,
            "Accepted report lacks full completed physical/warmup/teacher numerical evidence")
    events,pngs,identity=accepted_identity(root,receipt)
    require(expected_identity is None or identity==expected_identity,"Accepted raw/code/resource identity changed")
    return events,pngs,receipt,identity


def episode_values(events, episode, offset):
    validate_events(events)
    count = len(events); mask = np.asarray([row["action_valid"] for row in events], bool)
    tokens, hands = np.zeros((count, 64), np.float32), np.zeros((count, 14), np.float32)
    tokens[mask] = [row["executed_token64"] for row in events if row["action_valid"]]
    hands[mask] = [row["hand_targets14"] for row in events if row["action_valid"]]
    return {"observation.state": np.asarray([row["state43"] for row in events], np.float32),
        "action.body_token_v1_1": tokens, "action": hands, "action_valid": mask,
        "phase": [row["phase"] for row in events],
        "observation.base_pose": np.asarray([row["base_pose7"] for row in events], np.float64),
        "observation.object_poses": np.asarray([row["box_pose7"] for row in events], np.float64),
        "timestamp": np.arange(count, dtype=np.float32)/30,
        "source_timestamp": np.asarray([row["sim_time_s"] for row in events], np.float64),
        "source_elapsed_timestamp": np.asarray([row["collection_elapsed_s"] for row in events], np.float64),
        "teacher_source_timestamp": np.asarray([-1. if row["teacher_source_time_s"] is None else row["teacher_source_time_s"] for row in events], np.float64),
        "source_reference_frame_index": np.asarray([-1 if row["source_ref_frame"] is None else row["source_ref_frame"] for row in events], np.int64),
        "source_frame_index": np.asarray([row["native_frame_index"] for row in events], np.int64),
        "source_episode_index": np.full(count, events[0]["source_episode"], np.int64),
        "frame_index": np.arange(count, dtype=np.int64), "episode_index": np.full(count, episode, np.int64),
        "index": np.arange(offset, offset+count, dtype=np.int64), "task_index": np.zeros(count, np.int64),
        "task_description": [lift.INSTRUCTION]*count, "next.done": np.arange(count) == count-1}


def episode_stats(values):
    mask = values["action_valid"]
    return {"observation.state": vector_stats(values["observation.state"]),
            "action.body_token_v1_1": vector_stats(values["action.body_token_v1_1"][mask]),
            "action": vector_stats(values["action"][mask])}


def pack_episode(target, episode, offset, events, pngs, audit):
    import pyarrow as pa
    import pyarrow.parquet as pq
    values = episode_values(events, episode, offset)
    arrays = {key: pa.array(value.tolist(), type=pa.list_(pa.float64() if value.dtype == np.float64 else pa.float32()))
              if isinstance(value, np.ndarray) and value.ndim == 2 else pa.array(value) for key, value in values.items()}
    parquet = target/f"data/chunk-000/episode_{episode:06d}.parquet"; parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(arrays), parquet, compression="zstd")
    video = common.encode_video(pngs, [row["rgb_sha256"] for row in events],
                                target/f"videos/chunk-000/{VIDEO_KEY}/episode_{episode:06d}.mp4")
    return ({"episode_index": episode, "source_episode_index": audit["source_episode"],
             "tasks": [lift.INSTRUCTION], "length": len(events)},
            {"episode_index": episode, "stats": episode_stats(values)},
            {"episode_index": episode, "source_episode_index": audit["source_episode"], "length": len(events),
             "parquet_sha256": common.sha256(parquet), "video": video, "audit": audit})


def train_sources(values):
    values=list(values)
    require(len(values)==3 and len(set(values))==3 and all(type(v) is int and v in (0,1,2,5,6,7) for v in values),
            "Need three unique named TRAIN sources from0,1,2,5,6,7, disjoint from all heldout candidates")
    return values


def test_source(value):
    require(type(value) is int and value in (4,9,10,11), "TEST must use its original4 or prospective candidate9,10,11")
    return value


def validate_pilot(audited, selected_train=(0,1,2), selected_test=4):
    selected_train=train_sources(selected_train)
    selected_test=test_source(selected_test)
    require(set(audited) == set(selected_train+[3,selected_test]), "Require three named successful TRAIN sources, fixed VAL3 and one declared TEST")
    contracts = [item[2]["semantic_contract"] for item in audited.values()]
    require(all(value == contracts[0] for value in contracts), "Five episodes must share frozen task/controller/render/source semantics")
    return {"train":selected_train,"val":[3],"test":[selected_test]}


def curation_facts(audited, *, ledger_paths=None):
    """Retain the append-only ledger and bind each finished pilot to its raw run."""
    paths=sorted({Path(path).resolve(strict=True) for path in ledger_paths}) if ledger_paths is not None else sorted(
        {Path(item[2]["attempt_log"]["path"]).resolve(strict=True) for item in audited.values()})
    require(len(paths)==1,"Ordered curation requires the shared append-only candidate ledger")
    attempts=[row for path in paths for row in common.read_jsonl(Path(path))]
    facts=[];seen=set()
    common_semantic=next(iter(audited.values()))[2]["semantic_contract"]
    for ordinal,attempt in enumerate(attempts):
        if attempt.get("phase")!="finished" or attempt.get("run_role")!="pilot_collection":
            continue
        key=(attempt.get("attempt_id"),attempt.get("run_output"))
        if key in seen:continue
        seen.add(key);run=Path(attempt["run_output"]).resolve(strict=True);result=read(run/"result.json")
        if result.get("contract")!=CONTRACT:continue
        source=result.get("teacher",{}).get("source_episode_index",attempt["source_episode"])
        require(source==attempt["source_episode"] and result.get("run_role")=="pilot_collection" and
                result.get("attempt_log",{}).get("attempt_id")==attempt["attempt_id"] and
                result.get("status")==attempt["status"],"Candidate attempt/source/result identity differs")
        comparable=bool(result.get("frozen_contract"))
        if comparable:
            require(numerical_contract(run,result)==common_semantic and
                    attempt.get("frozen_contract_sha256")==result["frozen_contract"]["sha256"],
                    "Candidate task/controller/render protocol changed")
        independent=run.parent/f"source{source}-failure-independent-physics.json"
        independent_receipt=None
        if independent.is_file() and result.get("task_success_observed") is not True:
            retained=read(independent)
            require(retained.get("status")=="failed_teacher_physics_independently_reconstructed" and
                    retained.get("contract")==CONTRACT and retained.get("source_episode")==source and
                    same_directory(retained.get("run_root"),run) and retained.get("task_success") is False and
                    retained.get("training_eligible") is False,"Wrong independent candidate failure receipt")
            for path,binding in retained["input_identity"].items():
                require(common.sha256(path)==binding["sha256"] and Path(path).stat().st_size==binding["bytes"],
                        "Independent candidate failure input bytes changed")
            independent_receipt={"path":str(independent),"sha256":common.sha256(independent),
                                 "scope":"Retained original numerical failure report and input byte identities; numerics not repeated"}
        facts.append({"source_episode":source,"run_root":str(run),"finished_record_index":ordinal,
            "attempt_id":attempt["attempt_id"],"status":result["status"],
            "physical_outcome_status":result.get("physical_outcome_status"),"task_success_observed":result.get("task_success_observed"),
            "result_sha256":common.sha256(run/"result.json"),
            "trajectory_sha256":common.sha256(run/"trajectory.npz") if (run/"trajectory.npz").is_file() else None,
            "trace_sha256":common.sha256(run/"trace.jsonl") if (run/"trace.jsonl").is_file() else None,
            "protocol_comparable":comparable,"frozen_contract":result.get("frozen_contract"),
            "independent_failure_audit":independent_receipt})
    return attempts,facts


def train_curation(audited, selected_train, *, ledger=None):
    """Keep ordered candidate outcomes as facts; never relabel a source ID."""
    selected_train=train_sources(selected_train)
    attempts,facts=curation_facts(audited) if ledger is None else ledger
    report={"default_train_sources":[0,1,2],"selected_train_sources":selected_train,"candidate_order":[2,5,6,7],
        "selection":"original fixed TRAIN trio" if selected_train==[0,1,2] else "prospective first physically successful replacement demonstration",
        "preserved_attempts":facts,"failed_candidates_retained":[row for row in facts if row["source_episode"] in (2,5,6,7) and row["task_success_observed"] is not True],
        "all_attempt_records":attempts,"excluded_from_training":[3,4,8,9,10,11],"official_benchmark_claim":False,
        "interpretation":"Explicit successful-demonstration curation; failed attempts remain facts, not an unfiltered success-rate estimate"}
    validate_curation(report,selected_train)
    return report


def test_curation(audited, selected_test, *, ledger=None):
    selected_test=test_source(selected_test)
    attempts,facts=curation_facts(audited) if ledger is None else ledger
    report={"default_test_source":4,"selected_test_source":selected_test,"candidate_order":[4,9,10,11],
        "selection":"original fixed TEST4" if selected_test==4 else "prospective first physically successful heldout reference demonstration",
        "preserved_attempts":facts,"failed_candidates_retained":[row for row in facts if row["source_episode"] in (4,9,10,11) and row["task_success_observed"] is not True],
        "all_attempt_records":attempts,"excluded_from_training":[3,4,8,9,10,11],
        "required_additional_physical_before_after_source_episodes":[4,8],"official_benchmark_claim":False,
        "interpretation":"Curated offline successful expert reference; source4/8 physical challenge outcomes remain separate and unchanged"}
    validate_test_curation(report,selected_test)
    return report


def validate_candidate_order(report, order, chosen):
    facts=report["preserved_attempts"]; attempts=report["all_attempt_records"]
    success=[row for row in facts if row["source_episode"]==chosen and row["task_success_observed"] is True and
             row["protocol_comparable"] and row["status"]==row["physical_outcome_status"]=="task_terminated"]
    require(success,"Selected candidate lacks its real successful attempt")
    for source in order[:order.index(chosen)]:
        rows=[row for row in facts if row["source_episode"]==source]
        failures=[row for row in rows if row["protocol_comparable"] and row["physical_outcome_status"] in
                  ("source_end","sim_time_limit","stopped_on_fall","env_truncated","task_terminated") and
                  row["trajectory_sha256"] is not None and row["trace_sha256"] is not None]
        require(rows and all(row["task_success_observed"] is not True for row in rows) and failures,
                "Cannot skip an earlier successful, unattempted or infrastructure-only candidate")
        # Prove the earlier failure existed before the chosen attempt started,
        # rather than retrospectively reordering successful demonstrations.
        require(any(any(index>failure["finished_record_index"] and row.get("phase")=="started" and
                        row.get("attempt_id")==selected["attempt_id"] and row.get("source_episode")==chosen
                        for index,row in enumerate(attempts)) for failure in failures for selected in success),
                "Candidate attempts do not follow the declared prospective order")


def validate_curation(report, selected_train):
    require(report.get("selected_train_sources")==selected_train and
            report.get("candidate_order")==[2,5,6,7] and report.get("excluded_from_training")==[3,4,8,9,10,11] and
            report.get("official_benchmark_claim") is False,"Missing explicit v2 TRAIN curation identities")
    facts=report["preserved_attempts"]
    if selected_train!=[0,1,2]:
        require(selected_train[:2]==[0,1] and selected_train[2] in (5,6,7),
                "Authorized curation keeps TRAIN0/1 and the first successful candidate5,6,7")
        validate_candidate_order(report,[2,5,6,7],selected_train[2])
    require(report.get("failed_candidates_retained")==[row for row in facts if row["source_episode"] in (2,5,6,7) and row["task_success_observed"] is not True],
            "Failed TRAIN candidates were hidden or relabeled")


def validate_test_curation(report, selected_test):
    selected_test=test_source(selected_test)
    require(report.get("selected_test_source")==selected_test and report.get("candidate_order")==[4,9,10,11] and
            report.get("excluded_from_training")==[3,4,8,9,10,11] and
            report.get("required_additional_physical_before_after_source_episodes")==[4,8] and
            report.get("official_benchmark_claim") is False,"Missing separate TEST curation/original physical challenges")
    if selected_test!=4:validate_candidate_order(report,[4,9,10,11],selected_test)
    require(report.get("failed_candidates_retained")==[row for row in report["preserved_attempts"] if
            row["source_episode"] in (4,9,10,11) and row["task_success_observed"] is not True],
            "Failed TEST candidates were hidden or relabeled")


def pack(runs, output, base_run, *, selected_train=(0,1,2), selected_test=4, accepted_audits=()):
    output = Path(output)
    require(output.is_absolute() and not output.exists() and not output.resolve().is_relative_to(REPO),
            "Use a fresh absolute pack outside the repository; preserve all old artifacts")
    # Cheap structural/identity preflight precedes expensive once-only numerics.
    preflight={}; roots=[]
    for run in runs:
        root=Path(run).resolve(strict=True);result=read(root/"result.json")
        source=result["teacher"]["source_episode_index"]
        require(source not in preflight,"Duplicate source variants are not accepted")
        semantic=validate_result(root,result)
        preflight[source]=(None,None,{"semantic_contract":semantic,"attempt_log":result["attempt_log"]})
        roots.append((source,root))
    splits=validate_pilot(preflight,selected_train,selected_test)
    ledger=curation_facts(preflight)
    curation=train_curation(preflight,list(splits["train"]),ledger=ledger)
    test_selection=test_curation(preflight,splits["test"][0],ledger=ledger)
    base_run=Path(base_run).resolve(strict=True)
    base=read(base_run/"LOCAL_RUN_READY.json")
    require(base.get("status")=="local_joint_run_prepared" and base.get("optimizer_rng_restored") is False and
            common.sha256(base_run/"dataset_statistics.json")==common.sha256(Path(base["source_snapshot"])/"dataset_statistics.json"),
            "Source normalization bytes differ from the accepted base-run receipt")
    audited = {}; evidence={}; accepted_files={}
    for report_path,snapshot in accepted_audits:
        item=bind_accepted_run(report_path,snapshot)
        source=item[2]["source_episode"];require(source not in evidence,"Duplicate accepted report")
        evidence[source]=(Path(report_path),Path(snapshot),item)
    for source,root in roots:
        if source in evidence:
            require(Path(evidence[source][2][2]["run_root"]).resolve()==root,"Accepted evidence belongs to another run")
            item=evidence[source][2][:3]
        else:
            item=audit_run(root)
        require(source not in audited, "Duplicate source variants are not accepted")
        audited[source] = item
        provenance=output/"provenance";provenance.mkdir(parents=True,exist_ok=True)
        accepted_report=provenance/f"source{source}_accepted_numerics.json"
        snapshot=provenance/f"source{source}_accepted_verifier.py"
        if source in evidence:
            original,original_snapshot,bound=evidence[source]
            shutil.copy2(original,accepted_report);shutil.copy2(original_snapshot,snapshot);identity=bound[3]
        else:
            # Preserve each accepted run before a later run or video transform can
            # fail. A fresh retry can explicitly bind these reports, without
            # repeating already accepted numerical work.
            common.write_json(accepted_report,{"contract":CONTRACT,"status":"single_run_cpu_audited",
                "event_count":len(item[0]),"pngs":len(item[1]),"receipt":item[2],
                "packer_sha256":common.sha256(__file__),"training_authorized":False})
            shutil.copy2(__file__,snapshot)
            identity=accepted_identity(root,item[2])[2]
        accepted_files[source]=(accepted_report,snapshot,identity)
    require(set(evidence)<=set(audited),"Unused accepted numerical report")
    require(validate_pilot(audited,selected_train,selected_test)==splits,"Preflight/numerical source identities differ")
    receipts = []
    for split, sources in splits.items():
        target = output/split; entries, stats, offset = [], [], 0
        for episode, source in enumerate(sources):
            accepted_report,snapshot,identity=accepted_files[source]
            entry, statistic, receipt = pack_episode(target, episode, offset, *audited[source])
            receipt["accepted_numerical"]={"report":str(accepted_report.relative_to(output)),
                "report_sha256":common.sha256(accepted_report),"verifier_snapshot":str(snapshot.relative_to(output)),
                "verifier_sha256":common.sha256(snapshot),"identity":identity,"reused_prior_numerics":source in evidence,
                "binder_sha256":common.sha256(__file__)}
            entries.append(entry); stats.append(statistic); offset += entry["length"]; receipts.append({"split": split, **receipt})
        features = output_features()
        for key in ("observation.base_vel", "observation.body_joint_velocities"): features.pop(key)
        for key, dtype in (("action_valid", "bool"), ("phase", "string"), ("source_reference_frame_index", "int64"),
                           ("source_elapsed_timestamp", "float64"), ("teacher_source_timestamp", "float64")):
            features[key] = numeric_feature(1, dtype=dtype)
        video = receipts[-1]["video"]; features[VIDEO_KEY]["shape"] = [video["height"],video["width"],3]
        common.write_json(target/"meta/info.json", {"codebase_version":"v2.1", "robot_type":"g1_sonic", "fps":30,
            "total_episodes":len(entries), "total_frames":offset, "total_tasks":1, "total_videos":len(entries),
            "total_chunks":1, "chunks_size":1000, "splits":{split:f"0:{len(entries)}"}, "features":features,
            "data_path":"data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path":"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "action_label_policy":"prefix action_valid=false with dense0 unavailable placeholders; only valid teacher rows supervise"})
        for filename, rows in (("episodes.jsonl",entries), ("episodes_stats.jsonl",stats),
                               ("tasks.jsonl",[{"task_index":0,"task":lift.INSTRUCTION}])):
            (target/"meta"/filename).write_text("".join(json.dumps(row,allow_nan=False)+"\n" for row in rows))
    shutil.copy2(base_run/"dataset_statistics.json",output/"source_dataset_statistics.json")
    first = read(Path(next(iter(audited.values()))[2]["run_root"])/"result.json")
    shutil.copy2(first["teacher"]["state_action_contract"],output/"STATE_ACTION_CONTRACT.json")
    report = {"contract":CONTRACT,"task_id":lift.TASK_ID,"instruction":lift.INSTRUCTION,"metric":lift.METRIC,
        "measurement_hz":50,"prefix_rows":30,"source_joint_sha256":base["joint_sha256"],"splits":splits,
        "render_contract":next(iter(audited.values()))[2]["render_contract"],"validation_source":3,
        "source_statistics_sha256":common.sha256(base_run/"dataset_statistics.json"),"episodes":receipts,
        "label_policy":"First30 observations are history only, action_valid=false/dense0 placeholders; stats and supervision only valid teacher rows",
        "first_eligible_current_frame":30,"first_history_frames":[0,6,12,18,24,30],
        "raw_event_selection":"global ceil(row*50/30), teacher ref=globalnative-50 after row30; no source RGB",
        "termination":"actual new task_terminated success after real warmup, full physical prefix retained",
        "offline_validation_curation":{"selected_validation_source":3,"original_validation_source":8,
             "required_additional_physical_before_after_source_episodes":[8],"official_benchmark_claim":False},
        "train_demo_curation":curation,
        "test_demo_curation":test_selection,
        "evidence_reuse":{"numerical_audit_runs":[source for source in audited if source not in evidence],
            "reused_accepted_numerical_sources":sorted(evidence),
            "verify_root_policy":"Bind accepted raw/model/source/code/resource/PNG SHA identities and verify new packed labels/clock/video; never repeat contact/encoder numerics"},
        "verification_scope":"Raw numerical audit once per new run, explicit accepted reports reused only after complete byte binding; packed transformations independently verified",
        "packer_sha256":common.sha256(__file__),"training_authorized":False,"attempt_log_snapshots":[]}
    for index, path in enumerate(sorted({item[2]["attempt_log"]["path"] for item in audited.values()})):
        target=output/"provenance"/f"attempts_{index}.jsonl";target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target);report["attempt_log_snapshots"].append({"source_path":path,"snapshot":str(target.relative_to(output)),"sha256":common.sha256(target)})
    report["candidate_failure_audits"]=[]
    for fact in curation["preserved_attempts"]:
        independent=fact.get("independent_failure_audit")
        if independent is not None:
            target=output/"provenance"/f"source{fact['source_episode']}_retained_failure_audit.json"
            shutil.copy2(independent["path"],target)
            report["candidate_failure_audits"].append({"source_episode":fact["source_episode"],**independent,
                "snapshot":str(target.relative_to(output))})
    report["packed_files"]={str(path.relative_to(output)):common.sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    common.write_json(output/"LIFT_COMMANDED_ROLLOUT_READY.json",report)
    return report


def verify_packed_episode(root, receipt, events, offset=0):
    import pyarrow.parquet as pq
    import av
    target=root/receipt["split"]; episode=receipt["episode_index"]
    path=target/f"data/chunk-000/episode_{episode:06d}.parquet"
    table=pq.read_table(path); expected=episode_values(events,episode,offset)
    require(set(table.column_names)==set(expected) and table.num_rows==len(events),"Packed column/row schema differs")
    for key,value in expected.items(): equal(table[key].to_pylist(),value,f"packed {key}")
    stats=common.read_jsonl(target/"meta/episodes_stats.jsonl")[episode]
    require(stats=={"episode_index":episode,"stats":episode_stats(expected)},"Prefix placeholders entered action statistics")
    count=0
    with av.open(str(target/f"videos/chunk-000/{VIDEO_KEY}/episode_{episode:06d}.mp4")) as stream:
        for index,frame in enumerate(stream.decode(video=0)):
            require(index<len(events) and common.raw_rgb_sha(frame.to_ndarray(format="rgb24"))==events[index]["rgb_sha256"] and
                    abs(float(frame.pts*frame.time_base)-index/30)<1e-7,"Packed RGB pixels/PTS differ from real observations")
            count+=1
    require(count==len(events),"Packed lossless video was truncated")


def verify_pack(root):
    root=Path(root).resolve(strict=True);report=read(root/"LIFT_COMMANDED_ROLLOUT_READY.json")
    require(report.get("contract")==CONTRACT and report.get("task_id")==lift.TASK_ID and
            report.get("instruction")==lift.INSTRUCTION and report.get("prefix_rows")==30 and
            report.get("training_authorized") is False and report.get("packer_sha256")==common.sha256(__file__),
            "Only the independently audited real-prefix-v2 contract is accepted")
    for relative,digest in report["packed_files"].items():
        require(common.sha256(common.owned_file(root,relative))==digest,f"Packed artifact changed: {relative}")
    selected_train=train_sources(report["splits"]["train"])
    require(len(report["splits"]["test"])==1,"Exactly one named TEST reference is required")
    selected_test=test_source(report["splits"]["test"][0])
    splits={"train":selected_train,"val":[3],"test":[selected_test]}
    audited={}; offsets={key:0 for key in splits}
    for receipt in report["episodes"]:
        acceptance=receipt["accepted_numerical"]
        accepted=common.owned_file(root,acceptance["report"]);snapshot=common.owned_file(root,acceptance["verifier_snapshot"])
        require(common.sha256(accepted)==acceptance["report_sha256"] and common.sha256(snapshot)==acceptance["verifier_sha256"] and
                acceptance["binder_sha256"]==report["packer_sha256"],"Accepted numerical report/code seal changed")
        item=bind_accepted_run(accepted,snapshot,expected_identity=acceptance["identity"])[:3]
        require(item[2]==receipt["audit"],"Retained raw physical/token/pixel evidence differs")
        split=receipt["split"];source=item[2]["source_episode"]
        require(split in splits and source in splits[split] and
                receipt["episode_index"]==splits[split].index(source) and
                receipt["source_episode_index"]==source and receipt["length"]==len(item[0]),
                "Packed episode/split/source identities differ")
        verify_packed_episode(root,receipt,item[0],offsets[split]);offsets[split]+=len(item[0])
        source=item[2]["source_episode"];require(source not in audited,"Duplicate packed source");audited[source]=item
    require(validate_pilot(audited,selected_train,selected_test)==report["splits"],"Changed v2 split identities")
    validate_curation(report["train_demo_curation"],selected_train)
    validate_test_curation(report["test_demo_curation"],selected_test)
    snapshots=[]
    for item in report["attempt_log_snapshots"]:
        path=common.owned_file(root,item["snapshot"])
        require(common.sha256(path)==item["sha256"],"Retained append-only attempt ledger snapshot changed")
        snapshots.append(path)
    ledger=curation_facts(audited,ledger_paths=snapshots)
    require(train_curation(audited,selected_train,ledger=ledger)==report["train_demo_curation"] and
            test_curation(audited,selected_test,ledger=ledger)==report["test_demo_curation"],
            "Retained candidate result/trajectory/trace or ordered attempt evidence changed")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,nargs="+")
    parser.add_argument("--output",type=Path)
    parser.add_argument("--base-run",type=Path)
    parser.add_argument("--train-sources",type=int,nargs=3,default=[0,1,2])
    parser.add_argument("--test-source",type=int,default=4)
    parser.add_argument("--accepted-audit",type=Path,nargs=2,action="append",default=[],
                        metavar=("REPORT","VERIFIER_SNAPSHOT"))
    parser.add_argument("--verify-root",type=Path)
    args=parser.parse_args()
    if args.verify_root is not None:
        if any(value is not None for value in (args.runs,args.output,args.base_run)) or args.accepted_audit or args.test_source!=4 or args.train_sources!=[0,1,2]:parser.error("Verification cannot be mixed with packing overrides")
        report=verify_pack(args.verify_root)
    else:
        if any(value is None for value in (args.runs,args.output,args.base_run)):parser.error("Packing requires --runs --output --base-run")
        report=pack(args.runs,args.output,args.base_run,selected_train=args.train_sources,selected_test=args.test_source,
                    accepted_audits=args.accepted_audit)
    print(json.dumps(report,allow_nan=False))


if __name__=="__main__":
    main()
