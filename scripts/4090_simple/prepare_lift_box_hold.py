"""CPU audit/packing for the independent LiftBoxHold commanded-rollout pilot.

Only newly executed, sealed, synchronized production30Hz heading commands are
eligible. Task termination is verified from complete MuJoCo integration states
at50Hz, including the terminal state. CarryBox's task and old dataset stay separate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np

import rollout_carrybox_dataset as common
from prepare_carrybox_v11 import fixed_split, output_features, VIDEO_KEY, TRAIN_STATE_NAMES

TASK_ID = "simple/G1WholebodyLiftBoxHoldSonic-v0"
INSTRUCTION = "Lift the box so its lowest point is at least 10 cm above the floor, and maintain hand contact without floor or table support for 1 second."
METRIC = "lift_box_airborne_hand_contact_v1"
CONTRACT = "cvpr-simple-lift-box-hold-commanded-rollout-v1"


def lift_measurement(model, data):
    """Independent fixed-box geometric bottom and actual solved normal forces."""
    import mujoco
    cube = model.body("cube").id
    collision = [g for g in range(model.ngeom) if model.geom_bodyid[g] == cube
                 and (model.geom_contype[g] or model.geom_conaffinity[g])]
    if not collision or any(model.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX for g in collision):
        raise ValueError("Lift metric requires the unchanged cube box collision geometry")
    bottom = min(float(data.geom_xpos[g, 2] - np.abs(data.geom_xmat[g].reshape(3, 3)[2]) @ model.geom_size[g])
                 for g in collision)
    hand_force, support = 0., False
    for index, contact in enumerate(data.contact[:data.ncon]):
        a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if cube not in (a, b):
            continue
        other_geom = int(contact.geom2 if a == cube else contact.geom1)
        other_body = b if a == cube else a
        name = model.body(other_body).name or ""
        geom_name = model.geom(other_geom).name or ""
        support |= (model.geom_type[other_geom] == mujoco.mjtGeom.mjGEOM_PLANE
                    or geom_name in ("floor", "ground") or "table" in name.lower())
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, force)
        if not np.isfinite(force).all():
            raise ValueError("Reconstructed contact force is nonfinite")
        if name.startswith(("left_hand_", "right_hand_")) or name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            hand_force = max(hand_force, float(force[0]))
    return {"box_bottom_z_m": bottom, "maximum_hand_normal_force_n": hand_force,
            "floor_or_table_contact": bool(support),
            "condition": bool(bottom >= .10 and hand_force > .01 and not support)}


def reconstruct_lift(model_path, trajectory, result, active):
    import mujoco
    model = mujoco.MjModel.from_binary_path(str(model_path))
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    times = np.asarray(trajectory["sim_time_s"], dtype=np.float64)
    states = np.asarray(trajectory["physics_state_integration"], dtype=np.float64)
    if (int(np.asarray(trajectory["physics_state_spec"]).item()) != spec
            or states.shape != (len(times), mujoco.mj_stateSize(model, spec)) or not np.isfinite(states).all()
            or len(times) < 51 or not np.isfinite(times).all()
            or not np.allclose(np.diff(times), .02, atol=1e-7, rtol=0)):
        raise ValueError("Lift audit requires complete finite50Hz mjSTATE_INTEGRATION states, never qpos-only forces")
    phases = np.asarray(trajectory["phase"])
    if (phases[-1] != "task_terminal" or np.count_nonzero(phases == "task_terminal") != 1
            or abs(times[-1] - active[-1]["next_sim_time_s"]) > 1e-7):
        raise ValueError("Lift prefix must include its actual post-step task terminal state")
    if tuple(trajectory["state43_joint_names"]) != TRAIN_STATE_NAMES:
        raise ValueError("Lift physical state uses a different named joint order")
    data = mujoco.MjData(model)
    hold, previous, rows = 0., False, []
    for index, (time, state) in enumerate(zip(times, states, strict=True)):
        mujoco.mj_setState(model, data, state, spec)
        mujoco.mj_forward(model, data)
        measured = np.array([data.joint(name).qpos[0] for name in TRAIN_STATE_NAMES])
        if (abs(data.time - time) > 1e-9
                or not np.allclose(measured, trajectory["actual_state43"][index], atol=2e-6, rtol=0)
                or not np.allclose(data.qpos[:7], trajectory["actual_base_pose"][index], atol=1e-10, rtol=0)
                or not np.allclose(data.joint("cube_joint").qpos, trajectory["actual_box_pose"][index], atol=1e-10, rtol=0)):
            raise ValueError("Integration state is not the same recorded physical state/RGB clock")
        pelvis = model.body("pelvis").id
        tilt = np.degrees(np.arccos(np.clip(data.xmat[pelvis].reshape(3, 3)[2, 2], -1, 1)))
        if data.xpos[pelvis, 2] < .35 or tilt > 75:
            raise ValueError("Lift physical reconstruction found a fall")
        row = lift_measurement(model, data)
        delta = 0. if index == 0 else float(time - times[index-1])
        hold = hold + delta if row["condition"] and previous else 0.
        previous = row["condition"]
        rows.append({"sim_time_s": float(time), "hold_s": hold, **row})
    if not np.isfinite(hold) or hold < 1. - 1e-7:
        raise ValueError(f"Real LiftBoxHold terminal prefix lacks1s airborne hand-force hold: {hold:.6f}s")
    return {"metric": METRIC, "measurement_hz": 50, "physics_hz": 200,
            "final_reconstructed_hold_s": hold, "frames_reconstructed": len(rows),
            "terminal_measurement": rows[-1], "integration_state_spec": spec,
            "scope": "Solved contact forces from exact MJB and complete integration states at50Hz; no claim about every200Hz substep"}


class LiftAdapter:
    instruction = INSTRUCTION

    def validate_result(self, root, result):
        contract = result.get("task_contract", {})
        if (result.get("status") != "task_terminated" or result.get("task_success_observed") is not True
                or result.get("task_id") != TASK_ID or contract.get("task_id") != TASK_ID
                or contract.get("instruction") != INSTRUCTION or contract.get("metric") != METRIC
                or contract.get("measurement_hz") != 50):
            raise ValueError("Require a newly executed true LiftBoxHold task_terminated success, not a relabeled CarryBox run")
        source = result["teacher"]["source_episode_index"]
        controller = result.get("controller_contract", {})
        if (result.get("seed") != 10001 + source or result.get("warmup_seconds_requested") != .2
                or controller.get("physics_dt_s") != .005 or controller.get("decoder_hz") != 50
                or controller.get("command_hz") != 30 or controller.get("pd_profile") != "official_v1_1_ankle_pitch_1_5"
                or controller.get("hand_torque_profile") != "mjcf_limits"
                or controller.get("observation_phase") != common.OBSERVATION_PHASE):
            raise ValueError("Lift capture differs from the fixed seed rule, warmup, production clocks or controller profiles")
        frozen = result.get("frozen_contract", {})
        path = Path(frozen.get("path", "")).resolve(strict=True)
        if common.sha256(path) != frozen.get("sha256"):
            raise ValueError("Frozen task/controller declaration changed")
        declaration = json.loads(path.read_text())
        if declaration.get("task_contract") != contract or declaration.get("controller_contract") != result.get("controller_contract"):
            raise ValueError("Runtime task/controller differs from the predeclared frozen protocol")
        if declaration.get("render_contract") != result.get("render_contract"):
            raise ValueError("Runtime renderer differs from its frozen declaration")
        attempt = result.get("attempt_log", {})
        records = common.read_jsonl(Path(attempt.get("path", "")).resolve(strict=True))
        matches = [row for row in records if row.get("attempt_id") == attempt.get("attempt_id")]
        if not any(row.get("run_output") == str(root) and row.get("source_episode") == result["teacher"]["source_episode_index"]
                   and row.get("frozen_contract_sha256") == frozen["sha256"] for row in matches):
            raise ValueError("New Lift run lacks its bound append-only attempt record")

    reconstruct = staticmethod(reconstruct_lift)


def audit_run(path):
    events, pngs, receipt = common.audit_run(path, task_adapter=LiftAdapter())
    result = json.loads((path / "result.json").read_text())
    capture = Path(result.get("command_capture_root", "")).resolve(strict=True)
    manifest = json.loads(common.owned_file(capture, "capture.json").read_text())
    if manifest.get("instruction") != INSTRUCTION:
        raise ValueError("New RGB/command capture must carry the new fixed Lift instruction")
    receipt.update(task_id=TASK_ID, metric=METRIC, frozen_contract=result["frozen_contract"],
                   controller_contract=result["controller_contract"], attempt_log=result["attempt_log"],
                   render_contract=result.get("render_contract"), run_role=result.get("run_role"),
                   status="task_terminated", measurement_hz=50)
    receipt["reconstructed_lift"] = receipt.pop("reconstructed_placement")
    return events, pngs, receipt


def validate_render_contracts(renderers):
    if (not renderers or any(value != renderers[0] for value in renderers) or not isinstance(renderers[0], dict)
            or renderers[0].get("sim_mode") != "mujoco_isaac"
            or renderers[0].get("render_engine") != "Isaac RTX"
            or renderers[0].get("isaac_observation_only") is not True
            or renderers[0].get("sync_no_time_advance") is not True
            or renderers[0].get("authoritative_physics") != "MuJoCo"
            or renderers[0].get("measurement_hz") != 50 or renderers[0].get("render_hz") != 50):
        raise ValueError("All training/heldout captures must share the new actual RTX observation-only rendering contract")


def validate_pilot(audited):
    train_order = fixed_split()["train"]
    if 8 not in audited or 4 not in audited or len(audited) != 5:
        raise ValueError("Lift pilot requires3 actual TRAIN successes, locked VAL8 and TEST4")
    train = sorted(set(audited) - {8, 4})
    if len(train) != 3 or not set(train) <= set(train_order):
        raise ValueError("Lift TRAIN sources must remain in the original fixed TRAIN split")
    hashes = {row[2]["frozen_contract"]["sha256"] for row in audited.values()}
    if len(hashes) != 1:
        raise ValueError("All Lift pilot runs must use one frozen task/controller declaration")
    renderers = [row[2].get("render_contract") for row in audited.values()]
    validate_render_contracts(renderers)
    if any(row[2].get("run_role") != "pilot_collection" for row in audited.values()):
        raise ValueError("Qualification captures are reported separately from actual pilot collection")
    return {"train": train, "val": [8], "test": [4]}


def pack(runs, output, base_run):
    if not output.is_absolute() or output.exists() or output.resolve().is_relative_to(common.REPO):
        raise ValueError("Use a fresh absolute Lift dataset directory outside the repository")
    audited = {}
    for run in runs:
        item = audit_run(run.resolve(strict=True))
        source = item[2]["source_episode"]
        if source in audited:
            raise ValueError("Do not select duplicate variants of one source episode")
        audited[source] = item
    splits = validate_pilot(audited)
    base_run = base_run.resolve(strict=True)
    ready = json.loads((base_run / "LOCAL_RUN_READY.json").read_text())
    receipts = []
    for split, sources in splits.items():
        target = output / split
        entries, statistics, offset = [], [], 0
        for episode, source in enumerate(sources):
            entry, stats, receipt = common.pack_episode(target, episode, offset, *audited[source])
            entries.append(entry); statistics.append(stats); offset += entry["length"]
            receipts.append({"split": split, **receipt})
        features = output_features()
        for unused in ("observation.base_vel", "observation.body_joint_velocities"):
            features.pop(unused, None)
        features["source_elapsed_timestamp"] = {"dtype": "float64", "shape": [1], "names": None}
        features["source_reference_frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
        video = receipts[-1]["video"]
        features[VIDEO_KEY]["shape"] = [video["height"], video["width"], 3]
        common.write_json(target / "meta/info.json", {"codebase_version": "v2.1", "robot_type": "g1_sonic", "fps": 30,
            "total_episodes": len(entries), "total_frames": offset, "total_tasks": 1, "total_videos": len(entries),
            "total_chunks": 1, "chunks_size": 1000, "splits": {split: f"0:{len(entries)}"}, "features": features,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"})
        for filename, rows in (("episodes.jsonl", entries), ("episodes_stats.jsonl", statistics),
                               ("tasks.jsonl", [{"task_index": 0, "task": INSTRUCTION}])):
            (target / "meta" / filename).write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    shutil.copy2(base_run / "dataset_statistics.json", output / "source_dataset_statistics.json")
    first_run = Path(next(iter(audited.values()))[2]["run_root"])
    source_contract = Path(json.loads((first_run / "result.json").read_text())["teacher"]["state_action_contract"])
    shutil.copy2(source_contract, output / "STATE_ACTION_CONTRACT.json")
    report = {"contract": CONTRACT, "task_id": TASK_ID, "instruction": INSTRUCTION, "metric": METRIC,
              "measurement_hz": 50, "source_joint_sha256": ready["joint_sha256"], "splits": splits,
              "render_contract": next(iter(audited.values()))[2]["render_contract"],
              "source_statistics_sha256": common.sha256(base_run / "dataset_statistics.json"),
              "episodes": receipts, "raw_event_selection": "actual ceil(event*50/30) command event; no future RGB",
              "termination": "new actual task_terminated success prefix, not CarryBox source_end or relabeled failure",
              "packer_sha256": common.sha256(__file__), "training_authorized": False}
    # Preserve all attempts as facts. Repeats/probes do not create one-shot locks.
    log_paths = sorted({str(row[2]["attempt_log"]["path"]) for row in audited.values()})
    report["attempt_log_snapshots"] = []
    for index, path in enumerate(log_paths):
        destination = output / "provenance" / f"attempts_{index}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        report["attempt_log_snapshots"].append({"source_path": path, "snapshot": str(destination.relative_to(output)),
                                               "sha256": common.sha256(destination)})
    report["packed_files"] = {str(path.relative_to(output)): common.sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}
    common.write_json(output / "LIFT_COMMANDED_ROLLOUT_READY.json", report)
    return report


def verify_packed_episode(root, receipt, events):
    import av
    import pyarrow.parquet as pq
    split, episode = receipt["split"], receipt["episode_index"]
    path = root / split / f"data/chunk-000/episode_{episode:06d}.parquet"
    table = pq.read_table(path)
    expected = {"observation.state": np.asarray([row["state43"] for row in events], dtype=np.float32),
                "action.body_token_v1_1": np.asarray([row["executed_token64"] for row in events], dtype=np.float32),
                "action": np.asarray([row["hand_targets14"] for row in events], dtype=np.float32),
                "observation.base_pose": np.asarray([row["base_pose7"] for row in events]),
                "observation.object_poses": np.asarray([row["box_pose7"] for row in events]),
                "source_timestamp": np.asarray([row["sim_time_s"] for row in events]),
                "source_elapsed_timestamp": np.asarray([row["source_time_s"] for row in events]),
                "source_frame_index": np.asarray([row["native_frame_index"] for row in events]),
                "source_reference_frame_index": np.asarray([row["source_ref_frame"] for row in events]),
                "frame_index": np.arange(len(events)), "timestamp": np.arange(len(events), dtype=np.float32)/30,
                "source_episode_index": np.full(len(events), receipt["source_episode_index"]),
                "episode_index": np.full(len(events), episode), "task_description": [INSTRUCTION]*len(events)}
    if table.num_rows != len(events) or any(not np.array_equal(np.asarray(table[key].to_pylist()), value)
                                           for key, value in expected.items()):
        raise ValueError("Packed labels/state/clock/instruction do not equal their actual executed command events")
    video = root / split / f"videos/chunk-000/{VIDEO_KEY}/episode_{episode:06d}.mp4"
    count = 0
    with av.open(str(video)) as stream:
        for frame in stream.decode(video=0):
            if (count >= len(events) or common.raw_rgb_sha(frame.to_ndarray(format="rgb24")) != events[count]["rgb_sha256"]
                    or abs(float(frame.pts * frame.time_base) - count/30) > 1e-7):
                raise ValueError("Packed RGB/PTS differ from actual lossless command-event observations")
            count += 1
    if count != len(events):
        raise ValueError("Packed video truncates the actual Lift command events")


def verify_pack(root):
    """Recompute raw physical/encoder/RGB evidence and bind every packed byte."""
    root = root.resolve(strict=True)
    report = json.loads((root / "LIFT_COMMANDED_ROLLOUT_READY.json").read_text())
    if report.get("contract") != CONTRACT or report.get("task_id") != TASK_ID or report.get("instruction") != INSTRUCTION:
        raise ValueError("Only the new independent Lift commanded-rollout pack is eligible")
    for relative, digest in report["packed_files"].items():
        if common.sha256(common.owned_file(root, relative)) != digest:
            raise ValueError(f"Packed Lift artifact changed: {relative}")
    audited = {}
    for receipt in report["episodes"]:
        item = audit_run(Path(receipt["audit"]["run_root"]))
        if item[2] != receipt["audit"]:
            raise ValueError("Recomputed new Lift raw evidence differs from the packed provenance")
        verify_packed_episode(root, receipt, item[0])
        audited[item[2]["source_episode"]] = item
    if validate_pilot(audited) != report["splits"]:
        raise ValueError("Lift split identities changed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-run", type=Path)
    parser.add_argument("--verify-root", type=Path, help="Recompute all raw and packed evidence without changing the dataset")
    args = parser.parse_args()
    if args.verify_root is not None:
        if any(value is not None for value in (args.runs, args.output, args.base_run)):
            parser.error("--verify-root cannot be mixed with packing arguments")
        report = verify_pack(args.verify_root)
    else:
        if not all(value is not None for value in (args.runs, args.output, args.base_run)):
            parser.error("Packing requires --runs --output --base-run")
        report = pack(args.runs, args.output, args.base_run)
    print(json.dumps({"contract": report["contract"], "splits": report["splits"], "training_authorized": False}))


if __name__ == "__main__":
    main()
