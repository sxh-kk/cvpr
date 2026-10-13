#!/usr/bin/env python3
"""Download HM, its ego RGB MP4s, MPS trajectories, calibration, and timesync.

Python >= 3.10. Install requirements-download.txt first. AWS credentials use
boto3's normal credential chain; select an existing profile with --aws-profile.
HF_TOKEN or an existing Hugging Face login is used automatically if needed.

Examples:
    python download_egoexo4d_hm.py --aws-profile egoexo --dry-run
    python download_egoexo4d_hm.py --aws-profile egoexo --take cmu_bike02_4
    python download_egoexo4d_hm.py --aws-profile egoexo
    python download_egoexo4d_hm.py --aws-profile egoexo --resume
    python download_egoexo4d_hm.py --aws-profile egoexo --resume --exclude-bucket ego4d-indiana
    python download_egoexo4d_hm.py --aws-profile egoexo --resume --auto-route --probe-interval 120

Outputs under --output-dir (default: /gaozt-test1/datasets/EgoExo-4D-HM):
    EgoExo4D-HM/<take>/points_triangulated_world_results_merged.npz
    EgoExo4D/takes/<take>/frame_aligned_videos/<ego RGB>.mp4
    EgoExo4D/takes/<take>/... (MPS trajectory and noimagestreams VRS)
    EgoExo4D/captures/<capture>/timesync.csv
    EgoExo4D/{takes,captures}.json
    download_plan.json (pinned HF revision, matching IDs, paths, byte sizes)

The published HM NPZ already includes body/hand reconstruction. A matching
licensed SMPL-H asset is supplied separately via --smplh-model, if available.
Re-running skips files with the expected size. Interrupted S3 files restart;
completed files are retained. No dataset data is written by --dry-run.
--resume uses the saved plan and local metadata without requesting catalogs
again; it keeps the previous take selection and pinned HM commit.
--exclude-bucket skips raw downloads from named S3 buckets for this run only.
The complete plan, HM files and metadata are retained. Omit this option when
resuming later to download the deferred files.
--auto-route starts S3 transfers directly and compares each active bucket's
direct/proxy routes periodically. New files and outer retries use the selected
route; in-flight transfers are not cancelled. It requires curl. Probe results
and current choices are saved to route_status.json. HF downloads are unchanged.

Official manifest schema and release location:
https://github.com/facebookresearch/Ego4d/tree/main/ego4d/egoexo/download
https://docs.ego-exo4d-data.org/download/
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, nullcontext
import fcntl
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
from urllib.parse import urlsplit
import zipfile

HM_REPO = "Ego-Exo4D-HM/npz-datasets"
HM_FILE = "points_triangulated_world_results_merged.npz"
DEFAULT_OUTPUT_DIR = Path("/gaozt-test1/datasets/EgoExo-4D-HM")
S3_BASE = "s3://ego4d-consortium-sharing/egoexo-public"
RAW_PARTS = ("takes", "take_trajectory", "take_vrs_noimagestream", "captures")
HM_FIELDS = {"trans", "root_orient", "pose_body", "betas_per_frame", "valid", "joints3d"}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="下载 HM 和对应 ego RGB 视频、头戴轨迹、标定及同步表。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="数据保存目录")
    parser.add_argument("--aws-profile", "--s3-profile", help="已有 AWS profile；省略时使用默认凭证链")
    parser.add_argument("--release", default="v2", help="Ego-Exo4D 发布版本")
    parser.add_argument("--hm-revision", help="HM commit/tag；重跑默认沿用 download_plan.json 的 commit")
    parser.add_argument("--take", nargs="+", action="extend", help="只下载指定 take_name，可传多个")
    parser.add_argument("--limit", type=int, help="只下载排序后的前 N 个 HM take，便于试跑")
    parser.add_argument("--workers", type=int, default=8, help="S3 文件并发数")
    parser.add_argument("--hf-workers", type=int, default=4, help="Hugging Face 文件并发数")
    parser.add_argument("--smplh-model", type=Path, help="已有的匹配 SMPL-H .npz/.pkl 模型资产")
    parser.add_argument("--dry-run", action="store_true", help="只估算体积，不保存数据；--resume 时读取本地清单")
    parser.add_argument("--resume", action="store_true", help="按已有 download_plan.json 继续，跳过远程清单请求")
    parser.add_argument("--exclude-bucket", nargs="+", action="extend", default=[], metavar="BUCKET",
                        help="本次跳过指定 S3 桶的原始数据；可传多个或重复指定，完整清单、HM 和元数据保留")
    parser.add_argument("--auto-route", action="store_true", help="按桶定期比较直连/代理，新文件和外层重试使用优选线路（需 curl）")
    parser.add_argument("--route-proxy", default="http://127.0.0.1:17891", help="自动选路使用的 HTTP(S) 代理")
    parser.add_argument("--probe-interval", type=float, default=120, help="每个活跃桶的探测间隔，秒；至少 30 秒")
    parser.add_argument("--probe-mib", type=float, default=2, help="每条线路每次最多下载的探测数据，MiB；0.0625–16")
    parser.add_argument("--probe-timeout", type=float, default=12, help="单条线路探测总超时，秒；1–30")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.hf_workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("workers、hf-workers 和 limit 必须大于零")
    if args.smplh_model and (not args.smplh_model.is_file() or args.smplh_model.suffix not in {".npz", ".pkl"}):
        parser.error("--smplh-model 必须指向已有的 .npz 或 .pkl 模型文件")
    if args.resume and (args.take or args.limit is not None):
        parser.error("--resume 沿用已有清单的 take 范围，不能同时指定 --take 或 --limit")
    if not 30 <= args.probe_interval < float("inf") or not 0.0625 <= args.probe_mib <= 16 or not 1 <= args.probe_timeout <= 30:
        parser.error("probe-interval 至少 30 秒；probe-mib 应为 0.0625–16；probe-timeout 应为 1–30 秒")
    if args.auto_route:
        from egoexo4d_routes import validate_proxy
        try:
            validate_proxy(args.route_proxy)
        except ValueError as exc:
            parser.error(str(exc))
    return args


def local_path(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"不安全的清单路径：{relative}")
    destination = root.joinpath(*path.parts)
    if not destination.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"清单路径超出数据目录：{relative}")
    return destination


def s3_location(uri):
    parts = urlsplit(uri)
    if parts.scheme != "s3" or not parts.netloc or not parts.path.strip("/"):
        raise ValueError(f"无效的 S3 路径：{uri}")
    return parts.netloc, parts.path.lstrip("/")


def read_s3(s3, uri):
    bucket, key = s3_location(uri)
    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        return response["Body"].read()
    finally:
        response["Body"].close()


def manifest(s3, release, part):
    rows = json.loads(read_s3(s3, f"{S3_BASE}/{release}/{part}/manifest.json"))
    if not isinstance(rows, list):
        raise ValueError(f"{part} 清单格式异常，应为列表")
    return rows


def flatten(rows):
    return [path for row in rows for path in row["paths"]]


def select_hm(info, take_names=None, limit=None):
    files = {item.rfilename: item.size for item in (info.siblings or [])
             if len(PurePosixPath(item.rfilename).parts) == 2
             and PurePosixPath(item.rfilename).name == HM_FILE}
    names = sorted(PurePosixPath(path).parent.name for path in files)
    if not names:
        raise ValueError("Hugging Face 清单中没有 HM 发布 NPZ")
    if take_names:
        missing = sorted(set(take_names) - set(names))
        if missing:
            raise ValueError(f"HM 中不存在这些 take：{missing[:10]}")
        names = [name for name in names if name in set(take_names)]
    if limit is not None:
        names = names[:limit]
    return [{"relative_path": f"{name}/{HM_FILE}", "size": files[f"{name}/{HM_FILE}"]}
            for name in names]


def ego_rgb_paths(take):
    videos = take["frame_aligned_videos"]
    result = set()
    for camera in take["capture"]["cameras"]:
        if not camera["is_ego"]:
            continue
        streams = videos.get(camera["cam_id"], {})
        for stream_name, stream in streams.items():
            is_rgb = stream_name == "rgb" or stream.get("readable_stream_id") == "rgb" or stream.get("stream_id") == "214-1"
            is_gopro = camera.get("device_type") != "aria" and not camera["cam_id"].startswith("aria")
            relative = stream.get("relative_path")
            if relative and (is_rgb or is_gopro) and relative.endswith(".mp4"):
                result.add(str(PurePosixPath(take["root_dir"]) / relative))
    if not result:
        raise ValueError(f"{take['take_name']} 没有可匹配的 ego RGB 视频")
    return result


def select_raw(rows, part, takes):
    ids = {take["capture_uid"] if part == "captures" else take["take_uid"] for take in takes}
    rgb_paths = set().union(*(ego_rgb_paths(take) for take in takes)) if part == "takes" else set()
    paths = []
    covered = set()
    for row in rows:
        # Unlike the general CLI, missing IDs must not include unrelated data.
        if row.get("uid") not in ids:
            continue
        for item in row["paths"]:
            relative = item["relative_path"]
            if part == "takes" and relative not in rgb_paths:
                continue
            if part == "captures" and PurePosixPath(relative).name != "timesync.csv":
                continue
            views = item.get("views")
            if part == "take_trajectory" and views is not None and "ego" not in views:
                continue
            paths.append({**item, "part": part})
            covered.add(row["uid"])
    if part == "takes":
        missing = rgb_paths - {item["relative_path"] for item in paths}
        if missing:
            raise ValueError(f"视频清单缺少匹配的 RGB 文件：{sorted(missing)[:5]}")
    if ids - covered or not paths:
        raise ValueError(f"{part} 清单缺少匹配文件，缺失 ID：{sorted(ids-covered)[:5]}")
    return paths


def load_saved_plan(args):
    root = args.output_dir.resolve()
    path = root / "download_plan.json"
    if not path.is_file():
        raise ValueError("--resume 需要已有 download_plan.json；首次下载请不加 --resume")
    print(f"读取已有下载清单：{path}……", flush=True)
    plan = json.loads(path.read_text())
    if plan["hm_repo"] != HM_REPO or plan["release"] != args.release:
        raise ValueError("已有清单的数据集或 release 不一致")
    if args.hm_revision and args.hm_revision != plan["hm_revision"]:
        raise ValueError("--resume 必须沿用已有清单的 HM commit")
    if not plan["hm_files"] or not plan["s3_files"]:
        raise ValueError("已有下载清单为空，请不加 --resume 重新生成")
    for folder, items in (("EgoExo4D-HM", plan["hm_files"]), ("EgoExo4D", plan["s3_files"])):
        for item in items:
            local_path(root / folder, item["relative_path"])
            if folder == "EgoExo4D":
                s3_location(item["source_path"])
    metadata_bytes = {}
    metadata = {item["relative_path"]: item for item in plan["s3_files"] if item["part"] == "metadata"}
    for name in ("takes.json", "captures.json"):
        item = metadata.get(name)
        source = root / "EgoExo4D" / name
        if item is None or not complete(source, item["size"]):
            raise ValueError(f"--resume 所需的本地 {name} 缺失或不完整；请不加 --resume 重新读取元数据")
        data = source.read_bytes()
        json.loads(data)
        metadata_bytes[name] = data
    plan["output_dir"] = str(root)
    print(f"使用已有下载清单：{path}（跳过远程清单请求）", flush=True)
    return plan, metadata_bytes


def build_plan(args, s3, api):
    if args.resume:
        return load_saved_plan(args)
    root = args.output_dir.resolve()
    previous_path = root / "download_plan.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    if previous and previous["release"] != args.release:
        raise ValueError("该目录已用于其他 release，请指定新的 --output-dir")
    revision = args.hm_revision or previous.get("hm_revision") or "main"
    print("读取 HM 清单与 Ego-Exo4D 元数据……", flush=True)
    print("  请求 Hugging Face HM 清单……", flush=True)
    info = api.repo_info(HM_REPO, repo_type="dataset", revision=revision, files_metadata=True, timeout=20)
    if previous and previous["hm_revision"] != info.sha:
        raise ValueError("该目录的 HM commit 不一致，请指定新的 --output-dir")
    hm_files = select_hm(info, args.take, args.limit)
    print("  请求 S3 metadata 清单……", flush=True)
    metadata_paths = flatten(manifest(s3, args.release, "metadata"))
    metadata, metadata_bytes = [], {}
    for name in ("takes.json", "captures.json"):
        candidates = [item for item in metadata_paths if item["relative_path"] == name]
        if len(candidates) != 1:
            raise ValueError(f"metadata 中应恰好包含一个 {name}")
        item = candidates[0]
        print(f"  读取 S3 {name}……", flush=True)
        metadata_bytes[name] = read_s3(s3, item["source_path"])
        metadata.append({**item, "size": len(metadata_bytes[name]), "part": "metadata"})
    records = json.loads(metadata_bytes["takes.json"])
    lookup = {take["take_name"]: take for take in records}
    names = [PurePosixPath(item["relative_path"]).parent.name for item in hm_files]
    missing = sorted(set(names) - set(lookup))
    if missing:
        raise ValueError(f"HM 和 release={args.release} 无法匹配：{missing[:10]}")
    takes = [lookup[name] for name in names]
    raw = metadata.copy()
    for part in RAW_PARTS:
        print(f"筛选 {part}……", flush=True)
        raw.extend(select_raw(manifest(s3, args.release, part), part, takes))
    unique = {}
    for item in raw:
        local_path(root / "EgoExo4D", item["relative_path"])
        s3_location(item["source_path"])
        existing = unique.get(item["relative_path"])
        if existing and existing["source_path"] != item["source_path"]:
            raise ValueError(f"清单包含相互冲突的目标文件：{item['relative_path']}")
        unique[item["relative_path"]] = item
    raw = list(unique.values())
    # Resolve unknown sizes before downloading; authentication failures are fatal.
    for item in raw:
        if item.get("size") is None:
            bucket, key = s3_location(item["source_path"])
            item["size"] = s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
        if item["size"] <= 0:
            raise ValueError(f"远程文件为空：{item['relative_path']}")
    plan = {"hm_repo": HM_REPO, "hm_revision": info.sha, "release": args.release,
            "output_dir": str(root), "hm_files": hm_files, "s3_files": raw,
            "takes": [{"take_name": take["take_name"], "take_uid": take["take_uid"],
                       "capture_uid": take["capture_uid"]} for take in takes]}
    return plan, metadata_bytes


def complete(path, size):
    return size is not None and path.is_file() and path.stat().st_size == size


def filter_s3_files(plan, excluded_buckets=()):
    """Select this run's files without changing the saved, complete manifest."""
    excluded_buckets = set(excluded_buckets)
    if not excluded_buckets:
        return list(plan["s3_files"]), []
    raw_buckets = {s3_location(item["source_path"])[0] for item in plan["s3_files"]
                   if item["part"] != "metadata"}
    unknown = excluded_buckets - raw_buckets
    if unknown:
        raise ValueError(f"原始数据清单中没有这些桶：{sorted(unknown)}；请使用完整桶名，不加 s3:// 前缀")
    selected, excluded = [], []
    for item in plan["s3_files"]:
        if item["part"] != "metadata" and s3_location(item["source_path"])[0] in excluded_buckets:
            excluded.append(item)
        else:
            selected.append(item)
    return selected, excluded


def print_plan(plan, excluded_buckets=()):
    root = Path(plan["output_dir"])
    selected, excluded = filter_s3_files(plan, excluded_buckets)
    print(f"HM commit: {plan['hm_revision']}\n保存目录：{root}\n匹配 take：{len(plan['takes'])}")
    if excluded:
        skipped_bytes = sum(item["size"] or 0 for item in excluded)
        pending = sum(item["size"] or 0 for item in excluded
                      if not complete(local_path(root / "EgoExo4D", item["relative_path"]), item["size"]))
        print(f"本次排除 S3 桶：{', '.join(sorted(set(excluded_buckets)))}")
        print(f"本次暂缓 {len(excluded)} 个原始数据文件，共 {skipped_bytes/1024**3:.2f} GiB，"
              f"其中未完成 {pending/1024**3:.2f} GiB；完整清单保留。")
        print("以下体积统计为本次筛选范围；HM 和元数据保留。")
    pending_bytes = 0
    unknown_hm = sum(item["size"] is None for item in plan["hm_files"])
    groups = [("HM", plan["hm_files"], root / "EgoExo4D-HM")]
    groups.extend((part, [item for item in selected if item["part"] == part], root / "EgoExo4D")
                  for part in ("metadata", *RAW_PARTS))
    for label, files, directory in groups:
        total = sum(item["size"] or 0 for item in files)
        pending = sum(item["size"] or 0 for item in files
                      if not complete(local_path(directory, item["relative_path"]), item["size"]))
        pending_bytes += pending
        print(f"  {label:26} {len(files):5} 文件，{total/1024**3:9.2f} GiB，待下载 {pending/1024**3:.2f} GiB")
    if unknown_hm:
        print(f"另有 {unknown_hm} 个 HM 文件的体积未在远程清单中提供。")
    print(f"已知待下载合计：{pending_bytes/1024**3:.2f} GiB", flush=True)


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


@contextmanager
def download_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".egoexo4d_hm.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("该数据目录已有下载进程运行") from exc
        yield


def download_s3_file(s3, item, root, transfer_config, router=None):
    destination = local_path(root, item["relative_path"])
    if complete(destination, item["size"]):
        return "跳过"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    bucket, key = s3_location(item["source_path"])
    with router.transfer(bucket) if router is not None else nullcontext():
        for attempt in range(3):
            route, client = router.client_for(bucket) if router is not None else (None, s3)
            try:
                client.download_file(bucket, key, str(temporary), Config=transfer_config)
                if temporary.stat().st_size != item["size"]:
                    raise IOError(f"文件体积不匹配：{item['relative_path']}")
                temporary.replace(destination)
                return "完成" if route is None else f"完成（{'直连' if route == 'direct' else '代理'}）"
            except Exception:
                temporary.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def download_hm_file(hf_hub_download, item, plan, root):
    destination = local_path(root, item["relative_path"])
    if complete(destination, item["size"]):
        return "跳过"
    for attempt in range(3):
        try:
            hf_hub_download(plan["hm_repo"], filename=item["relative_path"],
                            repo_type="dataset", revision=plan["hm_revision"], local_dir=root)
            if item["size"] is not None and not complete(destination, item["size"]):
                raise IOError(f"HM 文件体积不匹配：{item['relative_path']}")
            return "完成"
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def download_hm(plan, root, workers, hf_hub_download):
    # Download the exact manifest paths directly. snapshot_download interprets
    # thousands of literal paths as glob patterns; Python 3.10's 256-entry
    # fnmatch cache then thrashes before any bytes are transferred.
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(download_hm_file, hf_hub_download, item, plan, root): item
                   for item in plan["hm_files"]}
        for index, future in enumerate(as_completed(futures), 1):
            item = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                status = f"失败：{exc}"
                failures.append(item["relative_path"])
            print(f"[HM {index}/{len(futures)}] {status} {item['relative_path']}", flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} 个 HM 文件下载失败；重新运行相同命令可补齐。")


def run(args, s3, api, hf_hub_download, transfer_config, router_factory=None):
    if args.dry_run:
        plan, _ = build_plan(args, s3, api)
        print_plan(plan, args.exclude_bucket)
        print("dry-run 完成，没有保存数据。")
        return
    root = args.output_dir.resolve()
    with download_lock(root):
        # Detect an existing downloader before making any catalog requests.
        plan, metadata_bytes = build_plan(args, s3, api)
        print_plan(plan, args.exclude_bucket)
        selected, excluded = filter_s3_files(plan, args.exclude_bucket)
        write_atomic(root / "download_plan.json", json.dumps(plan, ensure_ascii=False, indent=2).encode())
        if all(complete(local_path(root / "EgoExo4D-HM", item["relative_path"]), item["size"])
               for item in plan["hm_files"]):
            print("HM 文件已全部落盘，检查 NPZ……", flush=True)
        else:
            print("下载 HM NPZ……", flush=True)
            download_hm(plan, root / "EgoExo4D-HM", args.hf_workers, hf_hub_download)
        for item in plan["hm_files"]:
            path = local_path(root / "EgoExo4D-HM", item["relative_path"])
            if item["size"] is not None and not complete(path, item["size"]):
                raise IOError(f"HM 文件体积不匹配：{path}")
            with zipfile.ZipFile(path) as archive:
                fields = {PurePosixPath(name).stem for name in archive.namelist()}
                if not HM_FIELDS.issubset(fields):
                    raise ValueError(f"HM NPZ 缺少字段：{sorted(HM_FIELDS-fields)}")
        for name, data in metadata_bytes.items():
            if not complete(root / "EgoExo4D" / name, len(data)):
                write_atomic(root / "EgoExo4D" / name, data)
        files = [item for item in selected if item["part"] != "metadata"]
        failures = []
        pending = [item for item in files if not complete(local_path(root / "EgoExo4D", item["relative_path"]), item["size"])]
        if args.auto_route and pending:
            print(f"自动选路：初始直连，每 {args.probe_interval:g} 秒按活跃桶比较代理；"
                  "连续两次快至少 30% 才切换，当前线路探测失败时直接切换可用线路。", flush=True)
            if router_factory is None:
                raise ValueError("自动选路未配置路由器")
        route_context = router_factory(pending) if args.auto_route and pending else nullcontext(None)
        with route_context as router:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(download_s3_file, s3, item, root / "EgoExo4D", transfer_config, router): item
                           for item in files}
                for index, future in enumerate(as_completed(futures), 1):
                    item = futures[future]
                    try:
                        status = future.result()
                    except Exception as exc:
                        status = f"失败：{exc}"
                        failures.append(item["relative_path"])
                    print(f"[{index}/{len(files)}] {status} {item['relative_path']}", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} 个 S3 文件下载失败；重新运行相同命令可补齐。")
        if args.smplh_model:
            source = args.smplh_model.resolve()
            target = root / "body_models" / "smplh" / ("model" + source.suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            if source != target.resolve():
                shutil.copy2(source, target)
            print(f"SMPL-H 模型路径：{target}")
        else:
            print("身体 22 点转换还需单独准备匹配的 SMPL-H 模型资产；可用 --smplh-model 指定已有文件。")
        if excluded:
            print(f"本次筛选范围下载完成：{root}；暂缓 {len(excluded)} 个原始数据文件。"
                  "以后使用 --resume 并去掉 --exclude-bucket 可补齐。", flush=True)
        else:
            print(f"下载完成：{root}", flush=True)


def main(argv=None):
    args = parse_args(argv)
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
        from huggingface_hub import HfApi, hf_hub_download
        session = boto3.Session(profile_name=args.aws_profile)
        router_factory = None
        if args.auto_route:
            from egoexo4d_routes import make_router
            router_factory = lambda files: make_router(session, args, files)
        # Construct the client before threading; boto3 clients support concurrent calls.
        s3 = session.client("s3", config=Config(max_pool_connections=args.workers,
                                              connect_timeout=10, read_timeout=30,
                                              retries={"total_max_attempts": 3, "mode": "standard"},
                                              **({"proxies": {}} if args.auto_route else {})))
        run(args, s3, HfApi(), hf_hub_download, TransferConfig(use_threads=False), router_factory)
        return 0
    except ImportError as exc:
        print(f"缺少依赖：{exc}。请安装 requirements-download.txt", file=sys.stderr)
    except KeyboardInterrupt:
        print("下载已中断，重新运行相同命令即可继续。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"下载失败：{exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
