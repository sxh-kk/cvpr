"""Run a fixed EgoStandard manifest with bounded workers and one archive writer."""

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time
import traceback

from egomotion_pipeline.cli import input_signature, load_config
from egomotion_pipeline.egosuite_mcap import read_egosuite_mcap, UpperBodyOnly
from egomotion_pipeline.resampling import resample, transform_episode
from egomotion_pipeline.storage import Archive, atomic_json
from egomotion_pipeline.validation import validate


def now():
    return datetime.now(timezone.utc).isoformat()


def process_record(item, source, config):
    started = time.monotonic()
    result = {"uid": item["uid"], "path": item["path"]}
    try:
        episode = read_egosuite_mcap(Path(item["path"]), item["uid"], source)
        episode = transform_episode(episode, source)
        signature = input_signature(episode, source)
        arrays = resample(episode, 30, config.get("max_gap_s", .075),
                          config.get("snap_tolerance_s", 1e-5))
        result.update(kind="ready", episode=episode, arrays=arrays, signature=signature)
    except UpperBodyOnly as exc:
        result.update(kind="skipped", reason=str(exc))
    except Exception as exc:
        result.update(kind="failed", error_type=type(exc).__name__, reason=str(exc),
                      traceback=traceback.format_exc())
    result["worker_seconds"] = time.monotonic()-started
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    config = load_config(args.config)
    if len(config["sources"]) != 1 or config["sources"][0]["type"] != "egosuite_mcap":
        raise ValueError("This batch runner requires one EgoStandard MCAP source")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    source = config["sources"][0]
    manifest = json.loads(Path(args.manifest).read_text())
    archive = Archive(args.output, config)
    job = Path(args.job_dir)
    state = {"stage": "starting", "pid": os.getpid(), "started_at": now(),
             "output": str(archive.root), "input_root": source["root"],
             "input_files": len(manifest), "workers": args.workers,
             "processed_files": 0, "converted_files": 0, "resumed_files": 0,
             "skipped_files": 0, "failed_files": 0, "dino": "not_requested"}
    started = time.monotonic()

    def update(**fields):
        state.update(fields, updated_at=now(), elapsed_seconds=time.monotonic()-started)
        records = archive.records()
        state["committed_episodes"] = len(records)
        state["committed_frames"] = sum(r["length"] for r in records)
        state["ground_unavailable_episodes"] = sum(
            not r["metadata"]["ground"]["valid"] for r in records)
        atomic_json(job/"status.json", state)

    try:
        with archive.lock():
            update(stage="converting")
            print(f"Started: {len(manifest)} files, {args.workers} workers, output={archive.root}", flush=True)
            try:
                with (job/"events.jsonl").open("a", buffering=1) as events, \
                        (job/"failures.jsonl").open("a", buffering=1) as failures, \
                        ProcessPoolExecutor(max_workers=args.workers,
                                            mp_context=multiprocessing.get_context("spawn")) as pool:
                    items = iter(manifest)
                    pending = deque()
                    for _ in range(args.workers):
                        item = next(items, None)
                        if item is not None:
                            pending.append((item, pool.submit(process_record, item, source, config)))
                    last_finalize = time.monotonic()
                    while pending:
                        item, future = pending.popleft()
                        update(current_uid=item["uid"], current_path=item["path"])
                        result = future.result()
                        kind = result.pop("kind")
                        if kind == "ready":
                            episode, arrays = result.pop("episode"), result.pop("arrays")
                            signature = result.pop("signature")
                            written = archive.add(episode, arrays, signature)
                            kind = "converted" if written else "resumed"
                            result.update(frames=len(arrays["timestamp"]),
                                          floor_valid=episode.metadata["ground"]["valid"])
                            del episode, arrays
                        state[kind+"_files"] += 1
                        state["processed_files"] += 1
                        result.update(kind=kind, finished_at=now(), ordinal=state["processed_files"])
                        events.write(json.dumps(result, ensure_ascii=False)+"\n")
                        if kind == "failed":
                            failures.write(json.dumps(result, ensure_ascii=False)+"\n")
                        print(f"[{state['processed_files']}/{len(manifest)}] {kind}: {result['uid']} "
                              f"frames={result.get('frames', '-')} {result.get('reason', '')}", flush=True)
                        if state["processed_files"] == 1 or state["processed_files"] % 25 == 0 or time.monotonic()-last_finalize >= 60:
                            archive.finalize()
                            last_finalize = time.monotonic()
                        update()
                        item = next(items, None)
                        if item is not None:
                            pending.append((item, pool.submit(process_record, item, source, config)))
            finally:
                archive.finalize()
            update(stage="validating", current_uid=None, current_path=None)
            if not archive.records():
                raise ValueError("No full-body episodes were committed")
            report = validate(archive)
            report.update(input_files=len(manifest), skipped_files=state["skipped_files"],
                          failed_files=state["failed_files"], dino="not_requested")
            atomic_json(job/"validation.json", report)
            stage = "completed_with_errors" if state["failed_files"] else "completed"
            update(stage=stage, finished_at=now(), validation_passed=True)
            print(f"{stage}: {len(archive.records())} episodes, {state['committed_frames']} frames", flush=True)
            return 1 if state["failed_files"] else 0
    except BaseException as exc:
        update(stage="failed", finished_at=now(), fatal_error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
