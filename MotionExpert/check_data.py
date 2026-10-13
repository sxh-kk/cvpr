"""Validate immutable dense archives, grouped splits, statistics and DINO5 readiness."""
import argparse
from collections import Counter
import json
from pathlib import Path
import torch
from .config import load_config, SMPLH_REPRESENTATION_VERSION
from .dataset.ee4d_motion_dataset import EE4D_Motion_DataModule
from .dataset.dense_representation import MOTION_DIM
from .dataset.motion_layout import motion_layout
from .utils.task_conditioning import TASK_TO_ID, apply_task_conditioning


def validate_paths(cfg,splits=("train","val")):
    root=Path(getattr(cfg.DATA,"ARCHIVE_ROOT",None) or cfg.DATA.DATA_DIR)
    if not root.is_dir(): raise FileNotFoundError(f"Missing archive root: {root}")
    if cfg.DATA.REPRE_TYPE == SMPLH_REPRESENTATION_VERSION:
        from .dataset.smplh_archive import assert_smplh_archive_ready
        assert_smplh_archive_ready(root)
    for name in ("SPLIT_MANIFEST","STATS_PATH"):
        path=getattr(cfg.DATA,name,None)
        if not path or not Path(path).is_file(): raise FileNotFoundError(f"Missing DATA.{name}: {path}")
    mapping=getattr(cfg.DATA,"ROTATION_BASIS_PATH",None)
    if mapping and not Path(mapping).is_file(): raise FileNotFoundError(f"Missing joint-basis mapping: {mapping}")
    if cfg.DATA.REPRE_TYPE == SMPLH_REPRESENTATION_VERSION:
        stats = torch.load(cfg.DATA.STATS_PATH,map_location="cpu",weights_only=True)
        if stats.get("diagnostic",False):
            raise ValueError("Diagnostic SMPL-H statistics cannot start formal training; compute the full train split")


def inspection_profile(dataset):
    task = getattr(dataset, "train_task", "recon") or "recon"
    fps = getattr(dataset, "fps", 15)
    prefix = getattr(dataset, "forecast_prefix", 30)
    full = getattr(dataset, "require_full_window", False)
    supervision = getattr(dataset, "motion_supervision", None)
    if task == "fore" and supervision is None:
        supervision = "future_only"  # Legacy forecast datasets had no mode field.
    if supervision not in (None, "joint", "future_only"):
        raise ValueError("Motion supervision must be joint or future_only")
    if fps == 5 and (task != "fore" or dataset.window != 11 or prefix != 6 or not full):
        raise ValueError("5Hz readiness requires fore, WINDOW=11, FORECAST_PREFIX=6 and a full window")
    return {"training_task": task, "fps": fps, "window": dataset.window,
            "forecast_prefix": prefix, "require_full_window": full,
            "motion_supervision": supervision,
            "observed_image_frames": prefix if task == "fore" else dataset.window,
            "future_motion_frames": dataset.window-prefix if task == "fore" else 0}


def inspect_sample(dataset, sample, *, require_visual=False, trajectory_condition_prob=None):
    """Check archive labels and the conditions actually visible to the model."""
    profile = inspection_profile(dataset)
    window, prefix, task = dataset.window, profile["forecast_prefix"], profile["training_task"]
    y, misc = sample["y"], sample["misc"]
    layout = motion_layout(getattr(dataset,"motion_feature_dim",MOTION_DIM))
    for container,key,shape in ((y,"traj",(window,18)), (y,"img_embs",(window,5,1024)),
                                (misc,"motion",(window,layout.motion_dim)),
                                (y,"feature_valid_mask",(window,layout.motion_dim)),
                                (y,"trajectory_valid",(window,18)), (y,"valid_img_embs",(window,))):
        value = torch.as_tensor(container[key])
        if tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid {key} shape/values")
    feature = torch.as_tensor(y["feature_valid_mask"]).bool()
    motion = torch.as_tensor(misc["motion"])
    if bool((motion[~feature] != 0).any()):
        raise ValueError("Invalid labels must be neutral zero")
    n = int(misc.get("num_real_frames", window))
    if not 1 <= n <= window or (profile["require_full_window"] and n != window):
        raise ValueError("Profile requires a complete physical window without tail padding")
    padding = torch.as_tensor(y.get("padding_mask", y["valid_frames"])).bool()
    if padding.shape != (window,) or not torch.equal(padding, torch.arange(window, device=padding.device) < n):
        raise ValueError("Padding mask must match num_real_frames")
    loss = torch.as_tensor(y.get("loss_mask", padding)).bool()
    if loss.shape != (window,):
        raise ValueError("Invalid loss_mask shape")
    model_y = y
    probability = None
    if task == "fore":
        joint = profile["motion_supervision"] == "joint"
        future = torch.arange(window, device=padding.device) >= prefix
        expected = padding if joint else padding & future
        if "loss_mask" not in y or not torch.equal(loss, expected):
            raise ValueError("Joint forecast loss_mask must supervise all real frames" if joint else
                             "Forecast loss_mask must supervise future frames only")
        if not y.get("task_conditioning", False) or int(torch.as_tensor(y.get("task_id", -1))) != TASK_TO_ID["fore"]:
            raise ValueError("Forecast samples must use explicit fore task conditioning")
        if "motion_generation_mask" in y and not torch.equal(
                torch.as_tensor(y["motion_generation_mask"]).bool(),
                torch.ones_like(padding) if joint else future):
            raise ValueError("Joint forecast generation must include history and future" if joint else
                             "Forecast generation mask must keep history as noise-only context")
        for key in ("img_mask", "traj_mask"):
            mask = torch.as_tensor(y.get(key, [])).bool()
            if mask.shape != (window,) or not bool(mask[prefix:].all()):
                raise ValueError(f"Forecast {key} must hide all future conditions")
        probability = (0.0 if profile["fps"] == 5 else float(getattr(dataset,"cond_traj",True))) if trajectory_condition_prob is None else trajectory_condition_prob
        if isinstance(probability, bool) or not isinstance(probability, (int, float)) or not 0 <= probability <= 1:
            raise ValueError("trajectory_condition_prob must lie in [0,1]")
        # Optional camera supervision is a training experiment. Deployment and
        # validation always use the image-only conditions for the 5Hz profile.
        model_y = apply_task_conditioning(y,"fore",forecast_prefix=prefix,
                                         trajectory_condition=profile["fps"] != 5 and probability > 0,
                                         supervise_history=joint,noise_only_history=not joint)
        generation = model_y.get("motion_generation_mask")
        if generation is not None and not torch.equal(torch.as_tensor(generation).bool(),
                                                     torch.ones_like(padding) if joint else future):
            raise ValueError("Forecast generation domain disagrees with motion supervision")
    supervised = feature & loss[:,None]
    body_name = "SMPL-H K12" if getattr(dataset, "repre_type", None) == SMPLH_REPRESENTATION_VERSION else "dense12"
    if not bool(supervised[:,layout.body_slice].any()):
        raise ValueError(f"Sample has no valid supervised {body_name} body labels")
    if task == "fore" and not bool(supervised[prefix:,layout.body_slice].any()):
        raise ValueError(f"Sample has no valid future supervised {body_name} body labels")
    image_mask = torch.as_tensor(model_y.get("img_mask", torch.zeros_like(padding))).bool()
    if image_mask.shape != (window,):
        raise ValueError("Invalid img_mask shape")
    visible = torch.as_tensor(model_y["valid_img_embs"]).bool() & ~image_mask & padding
    observed = min(prefix if task == "fore" else window, n)
    fraction = float(visible[:observed].float().mean())
    if require_visual and (not visible[:observed].any() or fraction < getattr(dataset,"min_valid_image_fraction",0.)):
        raise ValueError("Sample has insufficient real visible DINO5 features in the observed interval")
    traj_mask = torch.as_tensor(model_y.get("traj_mask", torch.zeros_like(padding))).bool()
    visible_traj = torch.as_tensor(model_y["trajectory_valid"]).bool() & ~traj_mask[:,None]
    metadata = json.loads(misc["metadata_json"]) if "metadata_json" in misc else {}
    ground = metadata.get("ground", {})
    if task == "fore" and ground.get("window_fallback_used") and ground.get("estimation_scope") != "observed_prefix":
        raise ValueError("Forecast ground fallback must use the observed prefix only")
    indices = metadata.get("training_window", {}).get("source_frame_indices")
    if indices is not None:
        step = 30//profile["fps"]
        if (len(indices) != n or any(type(i) is not int for i in indices)
                or indices[0] % step or any(b-a != step for a,b in zip(indices,indices[1:]))):
            raise ValueError("Source frame indices disagree with training FPS/phase")
    return {"real_frames": n, "supervised_frames": int(loss.sum()),
            "history_supervised_frames": int(loss[:prefix].sum()) if task == "fore" else 0,
            "future_evaluation_frames": n-prefix if task == "fore" else 0,
            "supervised_valid_channels": int(supervised.sum()),
            "visible_image_frames": int(visible.sum()), "observed_image_valid_fraction": fraction,
            "visible_trajectory_channels": int(visible_traj.sum()),
            "training_auxiliary_trajectory_probability": probability,
            "motion_supervision": profile["motion_supervision"],
            "future_only_loss": task == "fore" and profile["motion_supervision"] == "future_only",
            "history_motion_noise_only": task == "fore" and profile["motion_supervision"] == "future_only",
            "ground_fallback": bool(misc.get("ground_fallback_used",False)),
            "ground_estimation_scope": ground.get("estimation_scope")}


def inspect_dataset(dataset,samples_per_source=2,require_visual=True,trajectory_condition_prob=None):
    counts,selected,valid_features=Counter(),Counter(),Counter()
    sample_checks=[]
    for index,(sequence,start) in enumerate(dataset.idx_to_sidx_fidx):
        record=dataset.records[int(sequence)]; source=record["source"]; counts[source]+=1
        if selected[source]>=samples_per_source: continue
        sample=dataset[index]
        checked = inspect_sample(dataset,sample,require_visual=require_visual and getattr(dataset,"train_task","recon")=="fore",
                                 trajectory_condition_prob=trajectory_condition_prob)
        sample_checks.append({"source_key":record["source_key"],"start_frame":int(start),**checked})
        valid_features[source]+=checked["visible_image_frames"]
        selected[source]+=1
    if require_visual:
        # Metadata distinguishes pending zero placeholders from actual extracted frames.
        states=Counter(r.get("dino",{}).get("state","unknown") for r in dataset.records)
        if not any(valid_features.values()):
            raise ValueError(f"No real valid DINO5 features in inspected windows; archive DINO states={dict(states)}. Extract features before visual training.")
    return {"windows":dict(counts),"checked_samples":dict(selected),"valid_visual_frames":dict(valid_features),
            "rotation_basis_mode":dataset.rotation_basis_mode,"target_joint_basis":dataset.target_joint_basis,
            "profile":inspection_profile(dataset),"rejected_windows":getattr(dataset,"rejected_windows",{}),
            "visual_features_checked":require_visual,"sample_checks":sample_checks}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",default=None)
    parser.add_argument("--data-dir")
    parser.add_argument("--split",choices=("train","val","test","both"),default="both")
    parser.add_argument("--samples-per-source",type=int,default=2)
    parser.add_argument("--motion-only",action="store_true",help="Skip visual columns/readiness checks")
    args=parser.parse_args()
    if args.samples_per_source<1: parser.error("samples-per-source must be positive")
    torch.set_num_threads(4)
    cfg=load_config(args.config,data_dir=args.data_dir)
    if args.motion_only: cfg.DATA.COND_IMG_FEAT=False
    splits=("train","val") if args.split=="both" else (args.split,)
    validate_paths(cfg,splits)
    report={}
    for split in splits:
        dm=EE4D_Motion_DataModule(cfg); dm.setup("train" if split=="train" else ("test" if split=="test" else "validate"))
        report[split]=inspect_dataset(getattr(dm,f"{split}_dataset"),args.samples_per_source,not args.motion_only,
                                     getattr(cfg.DATA,"TRAJECTORY_CONDITION_PROB",None))
    print(json.dumps(report,indent=2))


if __name__=="__main__": main()
