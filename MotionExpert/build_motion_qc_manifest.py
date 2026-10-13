"""Report dense12 validity on selected archive windows without requiring all dense22 joints."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from .config import load_config
from .dataset.dense_motion_dataset import DenseMotionDataset, dataset_class_for_representation
from .dataset.motion_layout import motion_layout
from .check_data import inspection_profile, inspect_sample


def build_qc_report(dataset, *, windows_per_sequence=2, require_visual=False,
                    trajectory_condition_prob=None):
    """Report supervised labels and visible conditions on a bounded window subset."""
    if type(windows_per_sequence) is not int or windows_per_sequence < 1:
        raise ValueError("windows_per_sequence must be positive")
    profile=inspection_profile(dataset)
    layout=motion_layout(getattr(dataset,"motion_feature_dim",245))
    joints=len(layout.joint_indices)
    seen=Counter(); status={}
    for index,(sequence,_) in enumerate(dataset.idx_to_sidx_fidx):
        name=dataset.seq_names[sequence]
        if seen[name]>=windows_per_sequence: continue
        seen[name]+=1
        try:
            sample=dataset[index]
            checked=inspect_sample(dataset,sample,require_visual=require_visual,
                                   trajectory_condition_prob=trajectory_condition_prob)
            masks=torch.as_tensor(sample["misc"]["feature_valid_mask"]).bool()
            n=checked["real_frames"]
            future_start=profile["forecast_prefix"] if profile["training_task"]=="fore" else 0
            future_body=masks[future_start:n,layout.body_slice].reshape(-1,joints,9)
            loss=torch.as_tensor(sample["y"].get("loss_mask",sample["y"]["valid_frames"])).bool()
            masks=masks[:n][loss[:n]]
            body=masks[:,layout.body_slice].reshape(len(masks),joints,9)
            row={"accepted":bool(body.any()) if profile["training_task"]=="fore" else bool(body[...,6:].any()),
                 "body_position_valid_fraction":float(body[...,6:].float().mean()),
                 "body_rotation_valid_fraction":float(body[...,:6].float().mean()),
                 "future_body_position_valid_fraction":float(future_body[...,6:].float().mean()),
                 "future_body_rotation_valid_fraction":float(future_body[...,:6].float().mean()),
                 "left_hand_valid_fraction":float(masks[:,layout.left_hand_slice].float().mean()),
                 "right_hand_valid_fraction":float(masks[:,layout.right_hand_slice].float().mean()),**checked}
        except (ValueError,FileNotFoundError) as exc: row={"accepted":False,"reason":str(exc)}
        status.setdefault(name,[]).append(row)
    unindexed=[name for name in dataset.seq_names if name not in status]
    return {"format":"motionexpert-dense-qc-v1","split":dataset.split,
            "split_fingerprint":dataset.split_fingerprint,"profile":profile,
            "representation":getattr(dataset,"repre_type",layout.representation),
            "archive_fingerprint":getattr(dataset,"manifest",{}).get("archive_fingerprint"),
            "model_sha256":getattr(dataset,"model_sha256",None),
            "archive_schema_version":getattr(dataset,"archive_schema_version",None),
            "diagnostic":getattr(dataset,"diagnostic",False),
            "rejected_episodes":getattr(dataset,"rejected_episodes",{}),
            "target_joint_basis":dataset.target_joint_basis,"rotation_basis_mode":dataset.rotation_basis_mode,
            "accepted_sequences":[name for name,rows in status.items() if any(r["accepted"] for r in rows)],
            "sequence_status":status,"unindexed_sequences":unindexed,
            "rejected_windows":getattr(dataset,"rejected_windows",{}),
            "visual_features_checked":require_visual,"windows_per_sequence":windows_per_sequence,
            "policy":"report only; retain partial per-joint validity; no all-dense22 requirement"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=None)
    parser.add_argument("--split",choices=("train","val","test"),default="train")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--windows-per-sequence",type=int,default=2)
    parser.add_argument("--motion-only",action="store_true",help="Skip DINO feature reads and visible-image checks")
    args=parser.parse_args()
    if args.windows_per_sequence<1: parser.error("windows-per-sequence must be positive")
    cfg=load_config(args.config); d=cfg.DATA
    task=getattr(d,"TRAIN_TASK","recon") or "recon"
    check_visual=task=="fore" and bool(d.COND_IMG_FEAT) and not args.motion_only
    representation = getattr(d,"REPRE_TYPE","dense12_rot6d_hands21_contacts_v1")
    dataset_class = dataset_class_for_representation(representation) if representation != "dense12_rot6d_hands21_contacts_v1" else DenseMotionDataset
    dataset=dataset_class(archive_root=getattr(d,"ARCHIVE_ROOT",None) or d.DATA_DIR,
        repre_type=representation,
        split=args.split,split_manifest=d.SPLIT_MANIFEST,do_normalization=False,cond_img_feat=check_visual,
        cond_traj=d.COND_TRAJ,
        window=d.WINDOW,fps=d.FPS,forecast_prefix=d.FORECAST_PREFIX,segment_stride=d.SEGMENT_STRIDE,
        rotation_basis_mode=d.ROTATION_BASIS_MODE,target_joint_basis=d.TARGET_JOINT_BASIS,
        rotation_basis_path=getattr(d,"ROTATION_BASIS_PATH",None),
        train_task=task,require_full_window=getattr(d,"REQUIRE_FULL_WINDOW",False),
        motion_supervision=getattr(d,"MOTION_SUPERVISION","future_only"),
        min_valid_image_fraction=getattr(d,"MIN_VALID_IMAGE_FRACTION",0.) if task=="fore" else 0.,
        contact_speed_m_s=getattr(d,"CONTACT_SPEED_M_S",.2),contact_height_m=getattr(d,"CONTACT_HEIGHT_M",.1))
    output=args.output.resolve(); root=dataset.root.resolve()
    if output==root or root in output.parents: raise ValueError("QC report must be stored outside immutable archive")
    result=build_qc_report(dataset,windows_per_sequence=args.windows_per_sequence,
                           require_visual=check_visual,
                           trajectory_condition_prob=getattr(d,"TRAJECTORY_CONDITION_PROB",None))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"reported {len(result['sequence_status'])} sequences; accepted {len(result['accepted_sequences'])}; "
          f"unindexed {len(result['unindexed_sequences'])}")


if __name__=="__main__": main()
