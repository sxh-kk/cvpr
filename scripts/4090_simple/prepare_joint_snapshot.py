"""Prepare a local inference/warmstart run from a save-complete weight snapshot.

Only model weights are transferred from the training machine. Optimizer/RNG are
not restored. The prior local bundle supplies identical immutable Qwen/DINO
construction resources; Motion parameters are extracted from this new joint
checkpoint, never copied from the older step.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

from export_bundle import export_motion, sha256, write_json


def prepare(source: Path, resources: Path, output: Path, repo: Path):
    source, resources, repo = (p.resolve(strict=True) for p in (source, resources, repo))
    output = output.absolute()
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(repo):
        raise ValueError("Use a fresh output directory outside the repository")
    identity = json.loads((source / "SNAPSHOT_READY.json").read_text())
    step = identity["step"]
    if (isinstance(step, bool) or not isinstance(step, int) or step <= 0
            or identity.get("save_completion_marker_verified") is not True
            or identity.get("training_modified") is not False):
        raise ValueError("Snapshot must record a complete saved checkpoint and unchanged training")
    weights = source / "model.safetensors"
    if (weights.stat().st_size != identity["model_bytes"]
            or sha256(weights) != identity["model_sha256"]):
        raise ValueError("Transferred model bytes differ from the protected source snapshot")
    original = json.loads((source / "run_config.json").read_text())
    base = resources / "joint20k"
    previous = json.loads((base / "provenance/run_config.json").read_text())
    if original["model"] != previous["model"]:
        raise ValueError("Source architecture/construction changed; do not reuse prior metadata implicitly")
    statistics = json.loads((source / "dataset_statistics.json").read_text())
    if statistics != json.loads((base / "dataset_statistics.json").read_text()):
        raise ValueError("Source normalization changed; audit its data/visual contracts separately")
    if original["data"] != previous["data"]:
        raise ValueError("Source repacking, normalization or visual data contract changed; audit before reusing metadata")
    config = copy.deepcopy(original)
    replacements = {
        "model.model_name_or_path": str(base / "qwen"),
        "model.motion_checkpoint": str(output / "motion/joint_motion.ckpt"),
        "data.root_dir": str(resources / "simple_local_usd_v1"),
        "data.dino_cache_root": None,
        "data.transform.field.stat_path": str(output / "dataset_statistics.json"),
        "train.output_dir": str(resources.parent / "training"),
        "train.resume_from_checkpoint": None,
        "train.deepspeed_config": str(repo / "VLMExpert/scripts/deepspeed/zero3.json"),
    }
    for key, value in replacements.items():
        node = config
        parts = key.split(".")
        for part in parts[:-1]:
            node = node[part]
        node[parts[-1]] = value
    # Tyro's parser uses the original command only to select the config schema;
    # all runtime values are loaded from run_config.json by the serving code.
    argv = (source / "argv.txt").read_text().split()
    rewritten = []
    found = set()
    i = 0
    while i < len(argv):
        token = argv[i]
        key = token.split("=", 1)[0].removeprefix("--").replace("-", "_")
        if token.startswith("--") and key in replacements:
            value = replacements[key]
            rewritten.append(f"--{key}={'None' if value is None else value}")
            found.add(key)
            if "=" not in token:
                i += 1
        else:
            rewritten.append(token)
        i += 1
    for key, value in replacements.items():
        if key not in found:
            rewritten.append(f"--{key}={'None' if value is None else value}")
    output.mkdir(parents=True)
    provenance = output / "provenance"
    provenance.mkdir()
    for name in ("run_config.json", "argv.txt", "dataset_statistics.json", "SNAPSHOT_READY.json"):
        shutil.copy2(source / name, provenance / name)
    write_json(output / "run_config.json", config)
    (output / "argv.txt").write_text("\n".join(rewritten) + "\n")
    shutil.copy2(source / "dataset_statistics.json", output / "dataset_statistics.json")
    target = output / "checkpoints" / f"ckpt_{step}" / weights.name
    target.parent.mkdir(parents=True)
    # Local sharing avoids a second 12GB copy. Its source is protected and local.
    import os
    os.link(weights, target)
    contract = json.loads((base / "dino_training_contract.json").read_text())
    write_json(output / "dino_training_contract.json", contract)
    motion_report = export_motion(base / "motion/joint_motion.ckpt", target,
                                  output / "motion/joint_motion.ckpt", step, contract["encoder"])
    write_json(output / "motion/export_validation.json", motion_report)
    from psi.utils import parse_args_to_tyro_config, apply_legacy_model_config_defaults
    schema = parse_args_to_tyro_config(output / "argv.txt")
    parsed = schema.model_validate(apply_legacy_model_config_defaults(config))
    assert parsed.model.action_architecture == "dual_source_v1"
    assert (parsed.model.action_dim, parsed.model.odim) == (80, 45)
    report = {"status": "local_joint_run_prepared", "step": step,
              "joint_sha256": identity["model_sha256"], "run_dir": str(output),
              "source_snapshot": str(source), "motion_weights": motion_report["weights"],
              "strict_motion_load": motion_report["strict_meta_load"],
              "optimizer_rng_restored": False, "gpu_validation": "not_run",
              "deployment_path_changes": replacements}
    write_json(output / "LOCAL_RUN_READY.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.resources, args.output, args.repo), indent=2))


if __name__ == "__main__":
    main()
