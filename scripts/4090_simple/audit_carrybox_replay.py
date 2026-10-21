"""CPU FK/contact reconstruction from a physical oracle replay's saved trajectory.

Reconstructed contacts are geometric consistency evidence, not recorded contact
forces or independent FK ground truth. No renderer, ONNX session or GPU is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from replay_carrybox_v11 import sha256
from run_closed_loop import write_json
from simple.assets.carry_box import CarryBoxAsset
from simple.baselines.dual_source_psi0 import TRAINING_STATE_JOINT_NAMES
from simple.core.types import Pose


def build_model(robot_mjcf):
    spec = mujoco.MjSpec.from_file(str(robot_mjcf))
    spec.worldbody.add_body(name="table", pos=[.3, 0, .4]).add_geom(
        name="table_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[.625, .78999733/2, .05],
        condim=6, friction=[2, .04, .0005], priority=10)
    spec.worldbody.add_geom(name="audit_ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 1],
                            friction=[1, .005, .0001])
    CarryBoxAsset().build_mujoco(spec.worldbody, Pose([-.55, 0, .217], [1, 0, 0, 0]))
    return spec.compile()


def set_frame(model, data, q43, base_pose, object_pose):
    data.qpos[:7] = base_pose
    for name, value in zip(TRAINING_STATE_JOINT_NAMES, q43):
        data.joint(name).qpos[0] = value
    data.joint("cube_joint").qpos[:] = object_pose
    mujoco.mj_forward(model, data)


def palm_geoms(model):
    ids = []
    for side in ("left", "right"):
        matches = [index for index in range(model.ngeom) if model.geom_type[index] == mujoco.mjtGeom.mjGEOM_MESH
                   and model.mesh(int(model.geom_dataid[index])).name == f"{side}_hand_palm_link"
                   and model.geom_contype[index] != 0]
        if len(matches) != 1:
            raise ValueError(f"Expected one collision palm geom for {side}, found {matches}")
        ids.append(matches[0])
    return ids


def geometric_hand_contacts(model, data):
    cube = model.body("cube").id
    contacts = []
    for contact in data.contact[:data.ncon]:
        a, b = int(model.geom_bodyid[contact.geom1]), int(model.geom_bodyid[contact.geom2])
        if cube not in (a, b):
            continue
        other = model.body(b if a == cube else a).name
        if "hand" in other or "wrist_yaw" in other:
            contacts.append(other)
    return sorted(set(contacts))


def run(args):
    trajectory = np.load(args.trajectory, allow_pickle=False)
    if tuple(trajectory["state43_joint_names"]) != TRAINING_STATE_JOINT_NAMES:
        raise ValueError("Replay state43 names do not match the training contract")
    model = build_model(args.robot_mjcf)
    actual, reference, root_only, joints_only = [mujoco.MjData(model) for _ in range(4)]
    palms = palm_geoms(model)
    rows = []
    for index in range(len(trajectory["sim_time_s"])):
        qa, qr = trajectory["actual_state43"][index], trajectory["reference_state43"][index]
        ba, br = trajectory["actual_base_pose"][index], trajectory["reference_base_pose"][index]
        oa, orf = trajectory["actual_box_pose"][index], trajectory["reference_box_pose"][index]
        set_frame(model, actual, qa, ba, oa)
        set_frame(model, reference, qr, br, orf)
        set_frame(model, root_only, qr, ba, orf)
        set_frame(model, joints_only, qa, br, orf)
        row = {"sim_time_s": float(trajectory["sim_time_s"][index]),
               "source_time_s": float(trajectory["source_time_s"][index]), "phase": str(trajectory["phase"][index]),
               "source_frame_index": int(trajectory["source_frame_index"][index]),
               "base_position_error_m": float(np.linalg.norm(ba[:3]-br[:3])),
               "box_position_error_m": float(np.linalg.norm(oa[:3]-orf[:3])),
               "reference_hand_contacts": geometric_hand_contacts(model, reference),
               "actual_hand_contacts": geometric_hand_contacts(model, actual)}
        for side, geom in zip(("left", "right"), palms):
            ref, act = reference.geom_xpos[geom], actual.geom_xpos[geom]
            row[f"{side}_palm_error_m"] = float(np.linalg.norm(act-ref))
            row[f"{side}_root_only_palm_error_m"] = float(np.linalg.norm(root_only.geom_xpos[geom]-ref))
            row[f"{side}_joints_only_palm_error_m"] = float(np.linalg.norm(joints_only.geom_xpos[geom]-ref))
            row[f"{side}_reference_palm_xyz"] = ref.tolist()
            row[f"{side}_actual_palm_xyz"] = act.tolist()
            row[f"{side}_actual_palm_to_reference_box_com_m"] = float(np.linalg.norm(act-orf[:3]))
            row[f"{side}_reference_palm_to_box_com_m"] = float(np.linalg.norm(ref-orf[:3]))
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "frames.jsonl").open("w") as stream:
        import json
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    windows = []
    scalar_metrics = ("base_position_error_m", "box_position_error_m", "left_palm_error_m", "right_palm_error_m",
                      "left_root_only_palm_error_m", "right_root_only_palm_error_m",
                      "left_joints_only_palm_error_m", "right_joints_only_palm_error_m")
    for lo, hi in ((0, 3), (3, 6), (6, 9), (9, 12), (12, 15), (15, 20), (20, 30)):
        selected = [row for row in rows if row["phase"] == "oracle_teacher_token" and lo <= row["source_time_s"] < hi]
        if not selected:
            continue
        window = {"source_time_start_s": lo, "source_time_end_s": hi, "frames": len(selected),
                  "reference_hand_contact_frames": sum(bool(row["reference_hand_contacts"]) for row in selected),
                  "actual_hand_contact_frames": sum(bool(row["actual_hand_contacts"]) for row in selected)}
        for metric in scalar_metrics:
            values = [row[metric] for row in selected]
            window[metric + "_mean"] = float(np.mean(values))
            window[metric + "_max"] = float(np.max(values))
        windows.append(window)
    summary = {"purpose": __doc__, "robot_mjcf": str(args.robot_mjcf), "robot_mjcf_sha256": sha256(args.robot_mjcf),
               "trajectory": str(args.trajectory), "trajectory_sha256": sha256(args.trajectory),
               "palm_point": "collision palm mesh frame center, on wrist_yaw body; identical geometry for both FK branches",
               "contact_scope": "CPU mj_forward reconstruction at recorded q/root/box with fixed scene0 table; not recorded force or independent source FK ground truth",
               "counterfactual_scope": "root-only and joints-only errors each replace one input; nonlinear contributions are not additive",
               "initial_frame": rows[0], "windows": windows,
               "first_reference_hand_contact": next((r for r in rows if r["reference_hand_contacts"]), None),
               "first_actual_hand_contact": next((r for r in rows if r["actual_hand_contacts"]), None)}
    write_json(args.output / "summary.json", summary)
    print(str(args.output / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--robot-mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.is_absolute() or args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("--output must be outside the repository")
    run(args)
