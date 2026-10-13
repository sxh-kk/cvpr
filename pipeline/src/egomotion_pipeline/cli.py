import argparse
import json
from pathlib import Path
import sys
import numpy as np
from .adapters import read_json, discover_egosuite, read_egosuite, discover_hm, read_hm
from .egosuite_mcap import discover_egosuite_mcap, read_egosuite_mcap, inspect_mcap, UpperBodyOnly
from .body_model import BodyModel, file_sha256
from .resampling import transform_episode, resample
from .storage import Archive, digest


def load_config(path):
    config = read_json(path)
    if config.get("fps", 30) != 30 or not config.get("sources"):
        raise ValueError("Config must select sources and fps=30")
    names = [s.get("name", s["type"]) for s in config["sources"]]
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique")
    for source in config["sources"]:
        if source["type"] not in {"egosuite_mcap", "egosuite_lerobot", "egoexo4d_hm"}:
            raise ValueError(f"Unknown adapter: {source['type']}")
        source.setdefault("name", source["type"])
        # Configuration paths resolve against the configuration directory, not cwd.
        for key in ["root", "model_path", "trajectory_root", "video_root", "egoexo_root", "episode_ids"]:
            if isinstance(source.get(key), str):
                source[key] = str((Path(path).resolve().parent/source[key]).resolve())
    return config


def source_items(source):
    if source["type"] == "egosuite_mcap":
        yield from discover_egosuite_mcap(source)
    elif source["type"] == "egosuite_lerobot":
        yield from discover_egosuite(source)
    else:
        yield from discover_hm(source)


def input_signature(episode, source):
    paths = [episode.metadata["source_data"]]
    if source["type"] == "egosuite_lerobot":
        paths += [str(Path(episode.metadata["source_root"])/"meta/info.json")]
    elif source["type"] == "egoexo4d_hm" and source.get("trajectory_root"):
        paths += [str(Path(source["trajectory_root"])/(episode.uid+ext)) for ext in [".npz", ".json"]]
    fingerprint = []
    for path in paths:
        p = Path(path)
        if p.exists():
            fingerprint.append({"path": str(p), "sha256": file_sha256(p)})
    return digest({"inputs": fingerprint, "metadata": episode.metadata, "video": episode.video})


def convert(config, output):
    archive = Archive(output, config)
    with archive.lock():
        try:
            for source in config["sources"]:
                model = BodyModel(source["model_path"], source.get("pad_shape_coefficients", False)) if source["type"] == "egoexo4d_hm" else None
                for item in source_items(source):
                    if source["type"] == "egosuite_lerobot":
                        source_info = read_json(item[0]/"meta/info.json")
                        shape = source_info["features"].get("observation.state.body_world", {}).get("shape", [])
                        if int(np.prod(shape or [0])) == 42:
                            print(f"Skipped {source['name']}:{item[2]}: upper-body 14 joints", flush=True)
                            continue
                    if source["type"] == "egosuite_mcap":
                        try:
                            episode = read_egosuite_mcap(*item, source)
                        except UpperBodyOnly:
                            print(f"Skipped {source['name']}:{item[1]}: upper-body 14 joints", flush=True)
                            continue
                    elif source["type"] == "egosuite_lerobot":
                        episode = read_egosuite(*item, source)
                    else:
                        episode = read_hm(*item, source, model)
                    episode = transform_episode(episode, source)
                    signature = input_signature(episode, source)
                    arrays = resample(episode, 30, config.get("max_gap_s", .075), config.get("snap_tolerance_s", 1e-5))
                    written = archive.add(episode, arrays, signature)
                    print(f"{'Converted' if written else 'Resumed'} {episode.source}:{episode.uid}: {len(arrays['timestamp'])} frames", flush=True)
            if not archive.records():
                raise ValueError("No selected full-body 22-joint episodes were converted")
        finally:
            # Previously committed episodes remain readable if a later episode fails.
            archive.finalize()


def inspect(config):
    results = []
    for source in config["sources"]:
        count, frames, upper_body = 0, 0, 0
        samples = []
        for item in source_items(source):
            count += 1
            if source["type"] == "egosuite_mcap":
                description = inspect_mcap(*item)
                upper_body += int(description["body_joints"] == 14)
                frames += description["duration_s"]*30
            elif source["type"] == "egosuite_lerobot":
                root, row, uid = item
                info = read_json(root/"meta/info.json")
                shape = info["features"].get("observation.state.body_world", {}).get("shape", [])
                if int(np.prod(shape or [0])) == 42:
                    upper_body += 1
                frames += int(row["length"])*30/float(info["fps"])
                description = {"uid": uid, "fps": info["fps"], "body_shape": shape,
                               "camera_columns": [k for k in info["features"] if "camera_position" in k]}
            else:
                path, uid = item
                with np.load(path, allow_pickle=False) as hm:
                    frames += hm["trans"].shape[1]
                    description = {"uid": uid, "fields": {k: list(hm[k].shape) for k in hm.files},
                                   "camera_warning": "cam_R/cam_t are not accepted as ego-camera poses"}
            if len(samples) < 3:
                samples.append(description)
        results.append({"name": source["name"], "episodes": count, "approx_hours": frames/30/3600,
                        "upper_body_14_episodes": upper_body, "samples": samples})
    print(json.dumps(results, indent=2, ensure_ascii=False))


def main(argv=None):
    parser = argparse.ArgumentParser(description="30 Hz body22 + hands + ego trajectories/DINO to LeRobot v3")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ["inspect", "convert", "prepare-aria"]:
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        if command != "inspect":
            p.add_argument("--output", required=True)
    p = sub.add_parser("extract-features")
    p.add_argument("--dataset", required=True)
    p.add_argument("--config", required=True, help="Standalone DINO JSON config")
    p.add_argument("--gpus", nargs="+", type=int,
                   help="Physical GPU IDs; one worker per GPU and one archive writer")
    p = sub.add_parser("upgrade-dino", help="Upgrade pending CLS-only v3 archives to 5x1024; preserves motion")
    p.add_argument("--dataset", required=True)
    p = sub.add_parser("upgrade-dense-v5", help="Integrate dense22 and explicit hands into one validated archive; preserve existing DINO")
    p.add_argument("--dataset", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--native-root")
    p.add_argument("--pilot-episodes", nargs="+", type=int)
    p.add_argument("--no-commit", action="store_true")
    p.add_argument("--job-dir")
    p = sub.add_parser("export-native-dense22", help="Export EgoSuite native world/floor row-6D + XYZ sidecars without rewriting the archive")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, help="Limit episodes for a separate pilot output")
    p.add_argument("--no-resume", action="store_true", help="Refuse previously exported episodes")
    p = sub.add_parser("validate-native-dense22", help="Recheck every native dense22 sidecar against source motion columns")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("validate")
    p.add_argument("--dataset", required=True)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--require-dino", action="store_true")
    p.add_argument("--require-floor", action="store_true")
    p.add_argument("--require-head-reference", action="store_true")
    p.add_argument("--require-body-head-floor", action="store_true",
                   help="Require every dense22 joint in the projected camera reference to be valid")
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            inspect(load_config(args.config))
        elif args.command == "convert":
            convert(load_config(args.config), args.output)
        elif args.command == "prepare-aria":
            from .aria import prepare_aria
            config = load_config(args.config)
            for source in config["sources"]:
                if source["type"] == "egoexo4d_hm":
                    prepare_aria(source, args.output)
        elif args.command == "extract-features":
            from .features import extract_features
            archive = Archive(args.dataset)
            with archive.lock():
                try:
                    if args.gpus is not None:
                        from .parallel_features import extract_features_parallel
                        extract_features_parallel(archive, read_json(args.config), args.gpus)
                    else:
                        extract_features(archive, read_json(args.config))
                finally:
                    archive.finalize()
        elif args.command == "upgrade-dino":
            from .migration import upgrade_dino
            print(json.dumps(upgrade_dino(args.dataset), indent=2, ensure_ascii=False))
        elif args.command == "upgrade-dense-v5":
            from .unified_migration import upgrade_dense_v5
            result = upgrade_dense_v5(args.dataset, workers=args.workers, native_root=args.native_root,
                                      pilot_indices=args.pilot_episodes, commit=not args.no_commit,
                                      job_dir=args.job_dir)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.command == "export-native-dense22":
            from .native_export import run_export
            result = run_export(args.dataset, args.output, workers=args.workers,
                                limit=args.limit, resume=not args.no_resume)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result.get("failures") or result.get("stage") == "failed":
                return 1
        elif args.command == "validate-native-dense22":
            from .native_export import validate_export
            result = validate_export(args.dataset, args.output, workers=args.workers)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            if result.get("failures") or result.get("stage") == "failed":
                return 1
        else:
            from .validation import validate
            print(json.dumps(validate(Archive(args.dataset), args.require_dino, args.strict, args.require_floor,
                                      args.require_head_reference, args.require_body_head_floor), indent=2, ensure_ascii=False))
    except (ValueError, FileNotFoundError, KeyError, ImportError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
