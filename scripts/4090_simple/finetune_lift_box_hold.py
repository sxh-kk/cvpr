"""Action-only LiftBoxHold adaptation after independently recomputing real data quality.

Uses conda's3.10 simulator Python for CPU physics/ORT audits and3.11 model Python
for config/DINO/training. No new task can unlock the deprecated CarryBox CLI.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import numpy as np

import finetune_carrybox as shared
from prepare_lift_box_hold import CONTRACT, TASK_ID

SIM_PYTHON = Path("/home/zxc/miniconda3/envs/cvpr-simple-sim310/bin/python")


def audit_quality(dataset_root, audit_python):
    """An actual verifier execution is mandatory; a success/status JSON is insufficient."""
    script = Path(__file__).with_name("prepare_lift_box_hold.py")
    command = [str(audit_python.resolve(strict=True)), str(script), "--verify-root", str(dataset_root.resolve(strict=True))]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError("New Lift raw/packed quality verification failed:\n" + completed.stderr[-6000:])
    try:
        summary = json.loads(completed.stdout)
    except ValueError as error:
        raise RuntimeError("Lift verifier did not return its actual structured result") from error
    if summary.get("contract") != CONTRACT or len(summary.get("splits", {}).get("train", [])) != 3:
        raise RuntimeError("Lift verifier did not validate the complete independent pilot")
    path = dataset_root / "LIFT_COMMANDED_ROLLOUT_READY.json"
    return {"verifier_script_sha256": shared.sha256(script), "verifier_python": str(audit_python),
            "dataset_contract_sha256": shared.sha256(path), "summary": summary,
            "scope": "Actual CPU recomputation: full integration-state contact forces, new success termination, encoder, lossless RGB, packed labels and timestamps"}


def bounds_report(dataset_root, field):
    """TRAIN labels only: inherited bounds are measured, never estimated or changed."""
    import pyarrow.parquet as pq
    arrays = []
    for path in sorted((dataset_root / "train").glob("data/**/episode_*.parquet")):
        table = pq.read_table(path, columns=["action.body_token_v1_1", "action"])
        arrays.append(np.concatenate((np.asarray(table["action.body_token_v1_1"].to_pylist()),
                                      np.asarray(table["action"].to_pylist())), axis=1))
    return command_bounds(np.concatenate(arrays), field)


def command_bounds(values, field):
    values = np.asarray(values)
    minimum, maximum = np.asarray(field.action_min)[:78], np.asarray(field.action_max)[:78]
    if values.ndim != 2 or values.shape[1] != 78 or not np.isfinite(values).all():
        raise ValueError("Lift TRAIN labels must be finite body64+hand14 commands")
    below, above = np.maximum(minimum - values, 0), np.maximum(values - maximum, 0)
    outside = (below > 0) | (above > 0)
    groups = {}
    for name, sl in (("body64", slice(0, 64)), ("hand14", slice(64, 78))):
        groups[name] = {"outside_fraction": float(outside[:, sl].mean()),
                        "maximum_below_minimum": float(below[:, sl].max()),
                        "maximum_above_maximum": float(above[:, sl].max()),
                        "channels": [{"action_index": i, "outside_fraction": float(outside[:, i].mean()),
                                      "below_fraction": float((below[:, i] > 0).mean()),
                                      "above_fraction": float((above[:, i] > 0).mean()),
                                      "maximum_below_minimum": float(below[:, i].max()),
                                      "maximum_above_maximum": float(above[:, i].max()),
                                      "inherited_minimum": float(minimum[i]), "inherited_maximum": float(maximum[i])}
                                     for i in range(sl.start, sl.stop)]}
    return {"split": "train", "unique_command_events": len(values), "groups": groups,
            "bounds_reestimated": False, "test_used_for_statistics": False,
            "interpretation": "Nonzero outside fractions imply normalization clipping and imperfect command representability; raw heldout errors include clipping error"}


def snapshot_sources(args, pack):
    """Copy explicit actual source, including ignored/untracked simulator modules."""
    relatives = ["scripts/4090_simple/" + name for name in (
        "finetune_lift_box_hold.py", "prepare_lift_box_hold.py", "finetune_carrybox.py", "rollout_carrybox_dataset.py",
        "prepare_carrybox_v11.py", "replay_carrybox_native50.py", "replay_carrybox_v11.py", "run_closed_loop.py", "video_capture.py")]
    relatives += ["VLMExpert/" + name for name in (
        "scripts/train.py", "src/psi/models/psi0.py", "src/psi/config/model_psi0.py", "src/psi/trainers/posttrain.py",
        "src/psi/trainers/trainer.py", "src/psi/trainers/training_safety.py", "src/psi/data/lerobot/dino_cache.py")]
    relatives += ["VLMExpert/third_party/SIMPLE/src/simple/" + name for name in (
        "tasks/g1_wholebody_lift_box_hold_sonic.py", "tasks/g1_wholebody_xmove_bend_carry_box_sonic.py",
        "baselines/sonic_v1_1.py", "envs/sonic_loco_manip.py", "envs/observation_contract.py", "engines/isaacsim.py")]
    target = args.output / "source_at_launch"
    entries = {}
    def copy(path, relative):
        path = path.resolve(strict=True)
        if not path.is_file() or path.stat().st_size > 2 << 20:
            raise ValueError("Source snapshot accepts bounded source/config text only, never data/environments")
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        entries[str(relative)] = {"actual_path": str(path), "sha256": shared.sha256(path), "size_bytes": path.stat().st_size}
    for relative in relatives:
        copy(shared.REPO / relative, Path("tree") / relative)
    for episode in pack["episodes"]:
        audit = episode["audit"]; source = audit["source_episode"]
        copy(Path(audit["frozen_contract"]["path"]), Path("contracts") / f"source_{source}_task_controller_contract.json")
        copy(Path(audit["attempt_log"]["path"]), Path("contracts") / f"source_{source}_attempts.jsonl")
    copy(args.dataset_root / "LIFT_COMMANDED_ROLLOUT_READY.json", Path("contracts/LIFT_COMMANDED_ROLLOUT_READY.json"))
    manifest = {"scope": "Actual execution source/config bytes; includes ignored and untracked files; excludes data, model weights and environments",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=shared.REPO, text=True).strip(),
                "files": entries}
    shared.write_json(target / "manifest.json", manifest)
    return {"path": str(target), "manifest_sha256": shared.sha256(target / "manifest.json"), "files": entries}


def verify_dino(args, source_encoder):
    from psi.data.lerobot.dino_cache import load_cache_manifest, load_episode_cache
    result = {}
    for split, count in (("train", 3), ("val", 1), ("test", 1)):
        root = args.dino_cache_root / split
        manifest = load_cache_manifest(root, args.dataset_root / split, expected_profile="256x320")
        identity_keys = ("model_id", "weights_sha256", "local_file_sha256", "preprocessing", "image_process")
        if (any(manifest["encoder"].get(key) != source_encoder.get(key) for key in identity_keys)
                or manifest["episode_indices"] != list(range(count))):
            raise ValueError("Lift DINO must use the exact frozen source encoder and all3/1/1 episodes")
        rows = []
        for episode in range(count):
            cache = load_episode_cache(root, episode, manifest=manifest, dataset_root=args.dataset_root / split,
                                       verify_checksums=True)
            if not np.isfinite(cache.features).all():
                raise ValueError("Lift DINO features are not finite")
            rows.append({"episode": episode, "frames": len(cache.features),
                         "feature_identity": cache.receipt["outputs"]["features.npy"]})
        result[split] = rows
    return result


def prepare(args):
    args.dataset_root = args.dataset_root.resolve(strict=True)
    args.audit_python = args.audit_python.resolve(strict=True)
    quality = audit_quality(args.dataset_root, args.audit_python)
    contract_path = args.dataset_root / "LIFT_COMMANDED_ROLLOUT_READY.json"
    pack = json.loads(contract_path.read_text())
    config, argv, report = shared.prepare(args, dataset_contract_path=contract_path,
                                          episode_counts={"train": 3, "val": 1, "test": 1},
                                          experiment="lift_box_hold_v11_action_only")
    if (pack["source_joint_sha256"] != report["source_joint_sha256"]
            or pack["source_statistics_sha256"] != report["source_statistics_sha256"]
            or shared.sha256(args.dataset_root / "source_dataset_statistics.json") != report["source_statistics_sha256"]):
        raise ValueError("Lift pack must inherit this exact A100 joint snapshot and normalization statistics")
    source_encoder = json.loads((args.base_run / "dino_training_contract.json").read_text())["encoder"]
    dino = verify_dino(args, source_encoder)
    representation = bounds_report(args.dataset_root, config.data.transform.field)
    snapshot = snapshot_sources(args, pack)
    shared.write_json(args.output / "train_command_representability.json", representation)
    report.update(task_id=TASK_ID, launcher_source_sha256=shared.sha256(__file__),
                  shared_training_mechanics_sha256=shared.sha256(Path(shared.__file__)),
                  dino_checksums=dino, quality_recomputation=quality,
                  train_command_representability=representation,
                  source_at_launch=snapshot,
                  quality_gate={"status": "actual_new_task_evidence_recomputed", "task_id": TASK_ID,
                                "source_frame_rule": "ceil", "split_episode_counts": {"train": 3, "val": 1, "test": 1},
                                "measurement_hz": 50, "test_used_for_control_selection": False})
    shared.write_json(args.output / "preparation.json", report)
    return config, argv, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base-run", "dataset-root", "dino-cache-root", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--audit-python", type=Path, default=SIM_PYTHON)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--test-windows", type=int, default=20)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--train", action="store_true")
    args = parser.parse_args()
    sys.path[:0] = [str(shared.REPO / "VLMExpert/src"), str(shared.REPO)]
    config, argv, preparation = prepare(args)
    if args.train:
        # All five raw episodes, their packed bytes, all DINO checksums and
        # inherited bounds were verified in this execution, before model import.
        shared._train_with_comparison(config, argv, args, preparation, train_episode_count=3,
                                      test_episode_count=1, source_frame_rule="ceil")
    print(json.dumps({"status": "completed" if args.train else "cpu_config_prepared", "output": str(args.output)}))


if __name__ == "__main__":
    main()
