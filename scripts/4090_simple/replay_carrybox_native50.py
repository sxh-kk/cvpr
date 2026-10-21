"""Native50Hz recorded-token versus measured-reference re-encoding diagnostic.

The original action.body_token identity is unknown: it is never renamed or
accepted as training data. This script does not read the derived30Hz dataset.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from prepare_carrybox_v11 import (BODY_NAMES, SOURCE_STATE_NAMES, SIMPLE_HAND_NAMES, TRAIN_STATE_NAMES,
    STATE_TO_TRAIN, HAND_TO_TRAIN, ENCODER_SHA, OBSERVATION_SHA, SOURCE_ARCHIVE_SHA, SONIC_REVISION,
    encoder_input, derived_velocity, matrix_wxyz, FUTURE_OFFSETS, sha256)
from replay_carrybox_v11 import run
from run_closed_loop import CARRYBOX_TASK, LIFTBOX_TASK, write_json

FEEDBACK_MODES = ("feedback-heading-v11", "feedback-vr-fk-v11", "feedback-wrist-ik-v11",
                  "feedback-vr-command-fk-v11")


def load_native(source, episode, sonic, contract, mode, *, acknowledge_unverified=False):
    import pyarrow.parquet as pq
    info = json.loads((source / "meta/info.json").read_text())
    if (info.get("fps"), info.get("total_episodes"), info.get("total_frames")) != (50, 50, 57989):
        raise ValueError("Native diagnostic requires the fixed original50Hz CarryBox pack")
    evidence = json.loads(contract.read_text())
    if (evidence.get("mapping_resolved_from_official_sources") is not True or
            tuple(evidence.get("source_joint_names", ())) != SOURCE_STATE_NAMES or
            tuple(evidence.get("source_action_hand_names", ())) != SIMPLE_HAND_NAMES):
        raise ValueError("Native state/action order lacks audited named source evidence")
    archive = source.with_suffix(".zip")
    if sha256(archive) != SOURCE_ARCHIVE_SHA:
        raise ValueError("Native source archive SHA mismatch")
    matches = list(source.glob(f"data/**/episode_{episode:06d}.parquet"))
    if len(matches) != 1:
        raise ValueError("Expected exactly one complete native source episode")
    path = matches[0]
    with zipfile.ZipFile(archive) as bundle:
        original = bundle.read(str(Path(source.name) / path.relative_to(source)))
    import hashlib
    if hashlib.sha256(original).hexdigest() != sha256(path):
        raise ValueError("Native episode differs from the fixed original archive")
    table = pq.read_table(path)
    widths = {"observation.state":43, "action":36, "action.body_token":64,
              "observation.base_pose":7, "observation.base_vel":6, "observation.object_poses":7}
    native = {}
    for key, width in widths.items():
        value = np.asarray(table[key].to_pylist(), dtype=np.float64)
        if value.shape != (table.num_rows, width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid native source column {key}")
        native[key] = value
    timestamps = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
    frames = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    episodes = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    if not np.array_equal(frames, np.arange(len(frames))) or not np.all(episodes == episode):
        raise ValueError("Native source frame/episode indices are not complete chronological recording")
    if not np.allclose(timestamps-timestamps[0], frames/50, atol=1e-5, rtol=0):
        raise ValueError("Native timestamps are not a genuine50Hz recording")
    dq = derived_velocity(native["observation.state"][:, :29], timestamps)
    if mode == "recorded-legacy":
        if not acknowledge_unverified:
            raise ValueError("Original token identity is unknown; explicit --acknowledge-unverified-legacy is required")
        token_key, tokens, count = "action.body_token", native["action.body_token"], len(timestamps)
    elif mode in ("reencoded-v11", *FEEDBACK_MODES):
        import onnxruntime as ort
        if sha256(sonic / "model_encoder.onnx") != ENCODER_SHA or sha256(sonic / "observation_config.yaml") != OBSERVATION_SHA:
            raise ValueError("Official v1.1 encoder identity mismatch")
        if json.loads((sonic / "PROVENANCE.json").read_text()).get("revision") != SONIC_REVISION:
            raise ValueError("Official encoder revision mismatch")
        options = ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        session = ort.InferenceSession(str(sonic / "model_encoder.onnx"), sess_options=options, providers=["CPUExecutionProvider"])
        if list(session.get_inputs()[0].shape) != [1,1751] or list(session.get_outputs()[0].shape) != [1,64]:
            raise ValueError("Official encoder contract mismatch")
        count = len(timestamps)-45
        if mode == "reencoded-v11":
            tokens = np.stack([session.run(None, {session.get_inputs()[0].name: encoder_input(
                native["observation.state"][:, :29], dq, native["observation.base_pose"][:,3:], frame)})[0][0]
                for frame in range(count)])
            token_key = "action.body_token_v1_1"
        else:
            tokens = None
            token_key = {"feedback-heading-v11":"runtime.feedback_heading_body_token_v1_1",
                "feedback-vr-fk-v11":"runtime.feedback_vr_fk_body_token_v1_1",
                "feedback-wrist-ik-v11":"runtime.feedback_wrist_ik_body_token_v1_1",
                "feedback-vr-command-fk-v11":"runtime.feedback_vr_command_fk_body_token_v1_1"}[mode]
    else:
        raise ValueError("Unknown native token mode")
    if count < 2:
        raise ValueError("Native source is too short for the selected diagnostic")
    values = {"observation.state":native["observation.state"][:count,STATE_TO_TRAIN],
              "action":native["action"][:count,:14][:,HAND_TO_TRAIN],
              "observation.base_pose":native["observation.base_pose"][:count],
              "observation.base_vel":native["observation.base_vel"][:count],
              "observation.object_poses":native["observation.object_poses"][:count],
              "observation.body_joint_velocities":dq[:count], "timestamp":frames[:count]/50,
              "source_episode_index":episodes[:count], "source_frame_index":frames[:count],
              "source_timestamp":timestamps[:count]}
    if tokens is not None:
        values[token_key] = tokens
    if mode in FEEDBACK_MODES:
        values["reference_full_body_q"] = native["observation.state"][:,:29]
        values["reference_full_body_dq"] = dq
        values["reference_full_root_wxyz"] = native["observation.base_pose"][:,3:]
        values["reference_full_root_xyz"] = native["observation.base_pose"][:,:3]
        values["reference_full_state43"] = native["observation.state"][:,STATE_TO_TRAIN]
        if mode == "feedback-vr-command-fk-v11":
            for feature, output in (("action.eef","reference_full_action_eef"),
                                    ("observation.eef_state","reference_full_observation_eef")):
                eef = np.asarray(table[feature].to_pylist(),dtype=np.float64)
                if eef.shape != (len(timestamps),14) or not np.isfinite(eef).all():
                    raise ValueError("Recorded wrist reference is not a finite native14-vector")
                values[output] = eef
    receipt = {"source_kind":"original_native50Hz; no30Hz conversion", "source_root":str(source),
        "source_parquet":str(path), "source_parquet_sha256":sha256(path), "source_archive_sha256":SOURCE_ARCHIVE_SHA,
        "source_episode_index":episode, "source_native_frames":len(timestamps), "replay_frames":count,
        "original_source_last_timestamp_s":float(timestamps[-1]), "replay_last_timestamp_s":(count-1)/50,
        "action_hz":50, "action_token_key":token_key, "token_mode":mode,
        "compatibility_verified":mode!="recorded-legacy", "source_token_identity":"unverified recorded action.body_token",
        "encoder_sha256":ENCODER_SHA if mode!="recorded-legacy" else None,
        "excluded_future_tail_frames":len(timestamps)-count, "tail_policy":"no padding or invented future",
        "comparison_valid_prefix_last_timestamp_s":(len(timestamps)-46)/50,
        "state_action_contract":str(contract), "state_action_contract_sha256":sha256(contract),
        "state43_joint_names":list(TRAIN_STATE_NAMES), "training_data_acceptance":False}
    if mode in FEEDBACK_MODES:
        receipt["reference_anchor_A"] = "identity; source world trajectory stays fixed; no spatial retargeting"
        receipt["encoder_reference_mode"] = 0 if mode in ("feedback-heading-v11","feedback-wrist-ik-v11") else 1
    return values, receipt


def feedback_encoder_input(q, dq, world_root_wxyz, frame, actual_yaw):
    """Change only the official heading branch; q/dq/future world roots stay fixed."""
    value = encoder_input(q, dq, world_root_wxyz, frame)
    c, s = np.cos(actual_yaw), np.sin(actual_yaw)
    inverse = np.array([[c,s,0],[-s,c,0],[0,0,1]])
    relative = inverse[None] @ matrix_wxyz(world_root_wxyz[frame+FUTURE_OFFSETS])
    value[0,584:644] = relative[...,:2].reshape(-1)
    return value


class FeedbackHeadingEncoder:
    def __init__(self, values, sonic):
        import onnxruntime as ort
        self.q, self.dq, self.roots = (values[key] for key in
            ("reference_full_body_q","reference_full_body_dq","reference_full_root_wxyz"))
        options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(sonic/"model_encoder.onnx"),sess_options=options,providers=["CPUExecutionProvider"])
        self.input_name=self.session.get_inputs()[0].name

    def __call__(self, frame, raw):
        rotation=raw.mujoco.mjData.xmat[raw.mujoco.mjModel.body("pelvis").id].reshape(3,3)
        actual_yaw=float(np.arctan2(rotation[1,0],rotation[0,0]))
        source_rotation=matrix_wxyz(self.roots[frame])
        source_yaw=float(np.arctan2(source_rotation[1,0],source_rotation[0,0]))
        value=feedback_encoder_input(self.q,self.dq,self.roots,frame,actual_yaw)
        token=self.session.run(None,{self.input_name:value})[0][0]
        if token.shape!=(64,) or not np.isfinite(token).all():
            raise ValueError("Runtime heading encoder did not produce a finite64-vector")
        return token,{"mode":"feedback-heading-v11", "actual_yaw_rad":actual_yaw,"source_current_yaw_rad":source_yaw,
            "source_frame_index":int(frame),"reference_frame_indices":(frame+FUTURE_OFFSETS).tolist(),
            "encoder_input_sha256":__import__('hashlib').sha256(value.tobytes()).hexdigest(),
            "changed_input_slice":[584,644],"q_dq_future_world_reference":"unchanged"}


def summarize_stages(output):
    rows = [json.loads(line) for line in (output / "trace.jsonl").read_text().splitlines()]
    data = np.load(output / "trajectory.npz", allow_pickle=False)
    active = data["phase"] == "oracle_teacher_token"
    times, boxes, refs = data["source_time_s"][active], data["actual_box_pose"][active], data["reference_box_pose"][active]
    initial_z = float(data["reference_box_pose"][0,2])
    events = {}
    if len(times):
        maximum = int(np.argmax(boxes[:,2]));events.update(maximum_actual_box_z_m=float(boxes[maximum,2]),maximum_actual_box_z_source_s=float(times[maximum]))
        for name, mask in (("first_actual_lift_5cm",boxes[:,2]>initial_z+.05),
                           ("first_reference_lift_5cm",refs[:,2]>initial_z+.05),
                           ("first_actual_com_at_table_placement_height",boxes[:,2]>=.45+.217-.03)):
            indices=np.flatnonzero(mask);events[name+"_source_s"]=float(times[indices[0]]) if len(indices) else None
        dropped=np.flatnonzero((np.arange(len(times))>maximum)&(boxes[:,2]<=initial_z+.05))
        events["drop_after_maximum_source_s"]=float(times[dropped[0]]) if len(dropped) and boxes[maximum,2]>initial_z+.05 else None
    contacts=[row for row in rows if row["phase"]=="oracle_teacher_token" and any(c["hand_or_palm"] and c["normal_force_n"]>.01 for c in row.get("box_contacts",[]))]
    events["first_actual_loaded_hand_contact_source_s"]=contacts[0]["source_time_s"] if contacts else None
    events["last_actual_loaded_hand_contact_source_s"]=contacts[-1]["source_time_s"] if contacts else None
    write_json(output / "stage_timing.json", {"purpose":"diagnostic physical stage timing, no source success filtering",
        "definitions":{"lift":"box COM > original initial COM + .05m", "loaded_hand_contact":"actual hand/palm normal force > .01N",
                       "table_height":"COM >= .45+.217-.03m; this alone is not placement success"},"events":events})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("source","contract","data-root","sonic-resources","output"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--token-mode",choices=("recorded-legacy","reencoded-v11",*FEEDBACK_MODES),required=True)
    parser.add_argument("--acknowledge-unverified-legacy",action="store_true")
    parser.add_argument("--episode",type=int,default=0);parser.add_argument("--seconds",type=float,default=60)
    parser.add_argument("--task",choices=(CARRYBOX_TASK,LIFTBOX_TASK),default=CARRYBOX_TASK)
    parser.add_argument("--warmup-seconds",type=float,default=.2);parser.add_argument("--seed",type=int,default=10001)
    parser.add_argument("--sim-mode",choices=("mujoco","mujoco_isaac","isaacsim"),default="mujoco")
    parser.add_argument("--pd-profile",choices=("fixed_cpp_baseline","official_v1_1_ankle_pitch_1_5"),default="official_v1_1_ankle_pitch_1_5")
    parser.add_argument("--hand-torque-profile",choices=("mjcf_limits","recording_0p7_nm"),default="mjcf_limits")
    parser.add_argument("--command-hz",type=int,choices=(30,50),default=50,
                        help="Update token and hands at this rate on real50Hz ticks; decoder stays50Hz")
    parser.add_argument("--record-commands",type=Path,
                        help="Optional fresh external lossless30Hz command capture directory")
    parser.add_argument("--sync-observations",action="store_true",
                        help="Refresh MuJoCo derived state and Isaac render at integrated qpos without time advance")
    parser.add_argument("--isaac-observation-only",action="store_true")
    parser.add_argument("--qualification-only",action="store_true",help="Declare this attempt as an infrastructure/render qualification before execution")
    parser.add_argument("--video",action="store_true");parser.add_argument("--stop-on-fall",action="store_true")
    parser.add_argument("--fall-height",type=float,default=.35);parser.add_argument("--fall-tilt-degrees",type=float,default=75)
    args=parser.parse_args()
    if args.isaac_observation_only and (not args.sync_observations or args.sim_mode == "mujoco"):
        parser.error("Observation-only Isaac requires --sync-observations and Isaac render mode")
    for field in ("source","contract","data_root","sonic_resources"):
        setattr(args,field,getattr(args,field).resolve(strict=True))
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]) or args.output.exists():
        parser.error("Use a fresh absolute output directory outside the repository")
    if args.episode<0 or not math.isfinite(args.seconds) or args.seconds<=0 or not math.isfinite(args.warmup_seconds) or args.warmup_seconds<.2:
        parser.error("Need a nonnegative episode, positive seconds and >=.2s real warmup")
    if args.token_mode=="recorded-legacy" and not args.acknowledge_unverified_legacy:
        parser.error("Recorded token identity is unverified; explicit diagnostic acknowledgement required")
    if args.record_commands is not None:
        if args.command_hz != 30 or args.token_mode not in FEEDBACK_MODES:
            parser.error("Lossless dataset capture requires a feedback v1.1 mode and --command-hz30")
        if not args.sync_observations:
            parser.error("Lossless dataset capture requires --sync-observations")
        if (not args.record_commands.is_absolute() or args.record_commands.exists() or
                args.record_commands.resolve().is_relative_to(Path(__file__).resolve().parents[2])):
            parser.error("Command capture requires a fresh absolute directory outside the repository")
    args.output.mkdir(parents=True);args.dataset_root=args.source;args.conversion_provenance=None;args.state_action_contract=args.contract
    if args.token_mode == "feedback-wrist-ik-v11" and not args.sync_observations:
        parser.error("World-wrist IK requires --sync-observations")
    runtime={}
    def loader(*_):
        values,receipt=load_native(args.source,args.episode,args.sonic_resources,args.contract,args.token_mode,acknowledge_unverified=args.acknowledge_unverified_legacy)
        if args.token_mode=="feedback-heading-v11":
            runtime["encoder"]=FeedbackHeadingEncoder(values,args.sonic_resources)
        elif args.token_mode=="feedback-vr-fk-v11":
            from feedback_vr_fk_v11 import FeedbackVrFkEncoder
            runtime["encoder"]=FeedbackVrFkEncoder(values,args.sonic_resources)
        elif args.token_mode=="feedback-wrist-ik-v11":
            from feedback_wrist_ik_v11 import FeedbackWristIkEncoder
            runtime["encoder"]=FeedbackWristIkEncoder(values,args.sonic_resources)
        elif args.token_mode=="feedback-vr-command-fk-v11":
            from feedback_vr_command_fk_v11 import FeedbackVrCommandFkEncoder
            runtime["encoder"]=FeedbackVrCommandFkEncoder(values,args.sonic_resources)
        return values,receipt
    provider=(lambda frame,raw:runtime["encoder"](frame,raw)) if args.token_mode in FEEDBACK_MODES else None
    recorder=None
    if args.record_commands is not None:
        from rollout_carrybox_dataset import LosslessCommandRecorder
        if args.task == LIFTBOX_TASK:
            from simple.tasks.g1_wholebody_lift_box_hold_sonic import LIFT_BOX_INSTRUCTION
            instruction=LIFT_BOX_INSTRUCTION
        else:
            tasks=[json.loads(line) for line in (args.source/"meta/tasks.jsonl").read_text().splitlines() if line.strip()]
            if len(tasks)!=1 or not isinstance(tasks[0].get("task"),str) or not tasks[0]["task"].strip():
                raise ValueError("Cannot resolve one genuine source task instruction")
            instruction=tasks[0]["task"]
        recorder=LosslessCommandRecorder(args.record_commands,instruction=instruction,metadata={
            "command_hz":args.command_hz,"token_mode":args.token_mode,"seed":args.seed,
            "task_id":"simple/"+args.task,
            "isaac_observation_only":args.isaac_observation_only,
            "run_role":"qualification" if args.qualification_only else "pilot_collection",
            "pd_profile":args.pd_profile,"hand_torque_profile":args.hand_torque_profile,
            "observation_phase":"post_step_mj_forward_same_qpos" if args.sync_observations else "legacy_mj_step_derived_phase",
            "run_output":str(args.output),"source_root":str(args.source),"source_episode":args.episode,
            "reference_anchor_A":"identity; original source world reference remains fixed"})
    try:
        result=run(args,teacher_loader=loader,action_hz=50,continue_after_termination=True,token_provider=provider,
                   command_hz=args.command_hz,command_recorder=recorder)
    finally:
        if recorder is not None:
            recorder.close()
    summarize_stages(args.output)
    return result


if __name__=="__main__":
    raise SystemExit(main())
