#!/usr/bin/env python3
"""Download, verify and unpack the official UnifoLM SONIC packs; never train."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
from http.client import HTTPException
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time
import threading
from urllib.request import Request, urlopen
import zipfile
import zlib


REPO = "USC-PSI-Lab/psi-data"
REVISION = "e78fb93cc28912a3031a10b8656d32d7f0a2b867"
PACKS = {
    "unifolm_sonic_lerobot_train": {
        "size": 38855909476,
        "sha256": "76434453893802090c91bd64ef47c258617494efcea76e2cfdb68295b1710cc9",
    },
    "unifolm_sonic_lerobot_val": {
        "size": 33487126,
        "sha256": "3d264d6454d59e83be5dadaa17f122b732a8c6cb86a2429c33c1f4fc912fc4b7",
    },
}


def report(stage, **fields):
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Preparation receipts must not be symlinks")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def verify_archive(path, expected):
    if path.stat().st_size != expected["size"]:
        raise ValueError(f"Archive size mismatch: {path}")
    digest = hashlib.sha256()
    processed = 0
    last_report = time.monotonic()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
            processed += len(block)
            if time.monotonic() - last_report > 30:
                report("archive_hashing", pack=path.name, bytes=processed)
                last_report = time.monotonic()
    if digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Archive SHA256 mismatch: {path}")
    report("archive_verified", pack=path.name, bytes=processed, sha256=digest.hexdigest())


def download_archive(root, pack, expected, *, workers=64, chunk_size=8 * 1024 * 1024,
                     attempts=12):
    """Keep verified local parts across failures; publish only the official SHA.

    Local part hashes detect changed files when resuming. They are not official
    per-part checksums: the pinned whole-archive SHA is the final content check.
    No credentials or signed redirect URLs are written to logs or receipts.
    """
    if min(workers, chunk_size, attempts) < 1:
        raise ValueError("Download workers, chunk size and attempts must be positive")
    root = Path(root).resolve(strict=True)
    destination = owned_output(root, root / "archives/official/sonic" / (pack + ".zip"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    parts = owned_output(root, root / ".pipeline/downloads" / pack / expected["sha256"])
    parts.mkdir(parents=True, exist_ok=True)
    identity = {"repo": REPO, "revision": REVISION, "pack": pack, **expected,
                "chunk_size": chunk_size}
    metadata = owned_output(root, parts / "identity.json")
    lock_path = owned_output(root, parts / "download.lock")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if destination.is_file():
            verify_archive(destination, expected)
            return destination
        if metadata.exists() and json.loads(metadata.read_text()) != identity:
            raise ValueError("Download part identity differs; use a separate parts directory")
        atomic_json(metadata, identity)
        total = expected["size"]
        count = (total + chunk_size - 1) // chunk_size
        stop = threading.Event()

        def transfer(index):
            start = index * chunk_size
            end = min(start + chunk_size, total) - 1
            size = end - start + 1
            path = owned_output(root, parts / f"{index:06d}.bin")
            receipt = owned_output(root, parts / f"{index:06d}.json")
            temporary = owned_output(root, parts / f"{index:06d}.partial")
            base = {"index": index, "start": start, "end": end, "size": size}
            if path.is_file() and receipt.is_file():
                try:
                    saved = json.loads(receipt.read_text())
                except ValueError:
                    saved = {}
                if not isinstance(saved, dict):
                    saved = {}
                if all(saved.get(key) == value for key, value in base.items()) and path.stat().st_size == size:
                    local_digest = hashlib.sha256()
                    with path.open("rb") as stream:
                        for block in iter(lambda: stream.read(256 * 1024), b""):
                            local_digest.update(block)
                    if local_digest.hexdigest() == saved.get("sha256"):
                        return size
            last_error = "unknown"
            for attempt in range(attempts):
                if stop.is_set():
                    raise RuntimeError("Archive download stopped after another part failed")
                # A distinct URL per range prevents intermediary caches from
                # reusing a redirect signed for a different requested range.
                url = (f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/sonic/{pack}.zip"
                       f"?download=true&range_start={start}&range_end={end}&retry={attempt}")
                request = Request(url, headers={"Range": f"bytes={start}-{end}",
                                               "Accept-Encoding": "identity"})
                digest = hashlib.sha256()
                received = 0
                started = time.monotonic()
                try:
                    with urlopen(request, timeout=60) as response, temporary.open("wb") as sink:
                        if (response.status != 206
                                or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}"
                                or response.headers.get("Content-Encoding", "identity") != "identity"):
                            raise ValueError("HTTP response does not match requested archive range")
                        while block := response.read(256 * 1024):
                            received += len(block)
                            if stop.is_set() or received > size or time.monotonic() - started > 240:
                                raise ValueError("HTTP range overflow or transfer deadline exceeded")
                            digest.update(block)
                            sink.write(block)
                        if received != size:
                            raise ValueError("HTTP range ended before the requested byte count")
                        sink.flush()
                        os.fsync(sink.fileno())
                    os.replace(temporary, path)
                    atomic_json(receipt, {**base, "sha256": digest.hexdigest()})
                    return size
                except (OSError, HTTPException, ValueError) as error:
                    last_error = type(error).__name__
                    if attempt + 1 < attempts:
                        code = getattr(error, "code", None)
                        report("download_part_retry", pack=pack, index=index, attempt=attempt + 1,
                               error=last_error, status=code if isinstance(code, int) else None)
                        stop.wait(min(2 ** attempt, 30))
            # Exception strings can contain signed redirect URLs. Report only
            # the class and fixed range identity; preserve every completed part.
            raise RuntimeError(f"Archive part {index} failed after {attempts} attempts ({last_error})") from None

        done_bytes = 0
        last_report = 0.0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(transfer, index) for index in range(count)]
            try:
                for future in as_completed(futures):
                    done_bytes += future.result()
                    if time.monotonic() - last_report >= 15 or done_bytes == total:
                        report("download_parts", pack=pack, bytes=done_bytes, total=total,
                               percent=round(100 * done_bytes / total, 2))
                        last_report = time.monotonic()
            except BaseException:
                stop.set()
                for future in futures:
                    future.cancel()
                raise
        assembled = owned_output(root, destination.with_name(destination.name + ".assembling"))
        with assembled.open("wb") as sink:
            for index in range(count):
                with (parts / f"{index:06d}.bin").open("rb") as source:
                    shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            sink.flush()
            os.fsync(sink.fileno())
        verify_archive(assembled, expected)
        if destination.exists():
            verify_archive(destination, expected)
            assembled.unlink()
        else:
            os.replace(assembled, destination)
        return destination


def safe_members(archive, pack):
    seen = set()
    for info in archive.infolist():
        relative = PurePosixPath(info.filename)
        if (relative.is_absolute() or ".." in relative.parts or "\\" in info.filename
                or not relative.parts or relative.parts[0] != pack
                or stat.S_ISLNK(info.external_attr >> 16)):
            raise ValueError(f"Unsafe archive member: {info.filename!r}")
        if relative in seen:
            raise ValueError(f"Duplicate archive member: {info.filename!r}")
        seen.add(relative)
        yield info, relative


def owned_output(root, path):
    """Reject symlink leftovers at every level before writing/reusing output."""
    root = Path(root).resolve(strict=True)
    path = Path(path)
    if not path.is_relative_to(root):
        raise ValueError(f"Output escapes preparation root: {path}")
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Preparation output must not follow symlinks: {current}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Output escapes preparation root: {path}")
    return path


def verify_unpacked(archive, root, pack):
    """Reuse an existing pack only after checking every original member's CRC."""
    count = 0
    for info, relative in safe_members(archive, pack):
        if info.is_dir():
            continue
        path = owned_output(root, root.joinpath(*relative.parts))
        if not path.is_file() or path.is_symlink() or path.stat().st_size != info.file_size:
            raise ValueError(f"Existing pack is incomplete or different: {path}")
        crc = 0
        with path.open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                crc = zlib.crc32(block, crc)
        if crc != info.CRC:
            raise ValueError(f"Existing pack CRC mismatch: {path}")
        count += 1
    return count


def unpack_pack(path, root, pack):
    destination = owned_output(root, root / pack)
    with zipfile.ZipFile(path) as archive:
        members = list(safe_members(archive, pack))
        if destination.exists():
            return verify_unpacked(archive, root, pack)
        staging = owned_output(root, root / ("." + pack + ".extracting"))
        staging.mkdir(parents=True, exist_ok=True)
        count = 0
        last_report = time.monotonic()
        for info, relative in members:
            target = owned_output(root, staging.joinpath(*relative.parts))
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = owned_output(root, target.with_name(target.name + ".part"))
            # ZipExtFile checks the original CRC while streaming. Only completed
            # files replace previous partial extraction output.
            with archive.open(info) as source, temporary.open("wb") as sink:
                shutil.copyfileobj(source, sink, length=8 * 1024 * 1024)
            os.replace(temporary, target)
            count += 1
            if time.monotonic() - last_report > 30:
                report("unpacking", pack=pack, files=count)
                last_report = time.monotonic()
        os.replace(staging / pack, destination)
        staging.rmdir()
        return count


def prepare(root, *, download=True, download_workers=64):
    root = Path(root).resolve()
    archives = root / "archives" / "official"
    receipts = root / ".pipeline"
    root.mkdir(parents=True, exist_ok=True)
    owned_output(root, receipts)
    records = {}
    for pack, expected in PACKS.items():
        filename = f"sonic/{pack}.zip"
        if download:
            download_archive(root, pack, expected, workers=download_workers)
        path = archives / filename
        verify_archive(path, expected)
        files = unpack_pack(path, root, pack)
        info = json.loads((root / pack / "meta/info.json").read_text())
        if (info["fps"] != 30
                or info["features"]["action.body_token_v1_1"]["shape"] != [64]
                or info["features"]["observation.state"]["shape"] != [43]):
            raise ValueError(f"Unexpected UnifoLM SONIC data protocol: {pack}")
        for name in ("modality.json", "stats_psi0.json", "episodes.jsonl"):
            if not (root / pack / "meta" / name).is_file():
                raise ValueError(f"Missing metadata: {pack}/meta/{name}")
        records[pack] = {
            "archive": str(path), **expected, "files": files,
            "episodes": info["total_episodes"], "frames": info["total_frames"], "fps": info["fps"],
        }
        atomic_json(receipts / f"{pack}.prepared.json", records[pack])
        report("pack_ready", pack=pack, **records[pack])
    stats = [(root / pack / "meta/stats_psi0.json").read_bytes() for pack in PACKS]
    if stats[0] != stats[1]:
        raise ValueError("Official train/val packs must share the same stats_psi0.json")
    receipt = {"repo": REPO, "revision": REVISION, "root": str(root), "packs": records,
               "stats_sha256": hashlib.sha256(stats[0]).hexdigest(), "training_started": False}
    atomic_json(receipts / "unifolm_download_receipt.json", receipt)
    report("dataset_ready", root=str(root), training_started=False)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/gaozt-test1/datasets/UnifoLM"))
    parser.add_argument("--skip-download", action="store_true", help="Verify and unpack existing archives")
    parser.add_argument("--download-workers", type=int, default=64, help="Concurrent persistent HTTP ranges")
    args = parser.parse_args()
    prepare(args.root, download=not args.skip_download, download_workers=args.download_workers)


if __name__ == "__main__":
    main()
