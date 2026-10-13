"""Create/read recording- and operator-disjoint archive splits without archive writes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import numpy as np

SPLIT_FORMAT = "motionexpert-archive-splits-v1"
V5_SCHEMA = "egomotion-dense22-hands21-camera-floor-dino5-v5"


def assert_archive_ready(root):
    journal = Path(root)/".pipeline/dense_v5_upgrade.json"
    if journal.exists() and json.loads(journal.read_text()).get("state") != "complete":
        raise ValueError("Dense v5 archive upgrade unfinished; wait for dense_v5_upgrade.json state=complete")


def motion_representation_identity(record, *, required=False):
    """Stable producer/content identity, independent of mutable DINO status."""
    value = record.get("motion_representation")
    if value is None and not required:
        return None
    if not isinstance(value, dict) or value.get("schema_version") != V5_SCHEMA:
        raise ValueError(f"{record.get('source_key')}: missing or unsupported v5 motion representation")
    identity = {"schema_version": V5_SCHEMA}
    for key in ("producer_signature", "motion_digest"):
        digest = value.get(key)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
            raise ValueError(f"{record.get('source_key')}: motion_representation.{key} must be a SHA256 digest")
        identity[key] = digest.lower()
    return identity


def read_records(root):
    root = Path(root)
    assert_archive_ready(root)
    paths = sorted((root/".pipeline/episodes").glob("*.json"))
    if not paths: raise FileNotFoundError(f"No archive episode records in {root}/.pipeline/episodes")
    records = [json.loads(p.read_text()) for p in paths]
    provenance_path = root/".pipeline/config.json"
    v5 = provenance_path.is_file() and json.loads(provenance_path.read_text()).get("schema_version") == V5_SCHEMA
    for record in records:
        motion_representation_identity(record, required=v5)
    keys = [r["source_key"] for r in records]
    if len(set(keys)) != len(keys): raise ValueError("Duplicate archive source keys")
    indices = [int(r["episode_index"]) for r in records]
    if len(set(indices)) != len(indices): raise ValueError("Duplicate archive episode indices")
    assert_archive_ready(root)
    return records


def recording_keys(record):
    meta = record["metadata"]
    keys = [f"operator:{meta['operator_id']}"] if meta.get("operator_id") else []
    for name in ("source_file_uid", "source_episode_uuid", "source_data"):
        if meta.get(name): keys.append(f"recording:{record['source']}:{name}:{meta[name]}")
    keys.append(f"source_key:{record['source_key']}")
    return keys


def archive_fingerprint(records):
    records = list(records)
    stable = [{"source_key": r["source_key"], "episode_index": int(r["episode_index"]),
               "length": int(r["length"]), "recording_keys": recording_keys(r),
               "input_fingerprint": r.get("input_fingerprint", "")} for r in records]
    # Keep the legacy digest unchanged when no v5 identity is present. A v5
    # migration deliberately invalidates old splits/statistics, while later
    # DINO extraction does not change the motion identity.
    for item, record in zip(stable, records):
        identity = motion_representation_identity(record)
        if identity is not None:
            item["motion_representation"] = identity
    return hashlib.sha256(json.dumps(sorted(stable, key=lambda r:r["source_key"]), sort_keys=True).encode()).hexdigest()


def _components(records):
    parents = list(range(len(records)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]; i = parents[i]
        return i
    owners = {}
    for index, record in enumerate(records):
        if not record["metadata"].get("operator_id"):
            raise ValueError(f"Cannot guarantee operator-disjoint splits: {record['source_key']} has no operator_id")
        for key in recording_keys(record):
            if key in owners: parents[find(index)] = find(owners[key])
            else: owners[key] = index
    result = {}
    for i in range(len(records)): result.setdefault(find(i), []).append(i)
    return list(result.values())


def create_split_manifest(records, *, seed=2026, fractions=(.8, .1, .1)):
    fractions = np.asarray(fractions, float)
    if fractions.shape != (3,) or not np.isfinite(fractions).all() or (fractions <= 0).any() or not np.isclose(fractions.sum(), 1):
        raise ValueError("Train/val/test fractions must be positive and sum to one")
    groups = _components(records)
    if len(groups) < 3: raise ValueError("At least three disjoint operator/recording groups are needed")
    rng = np.random.default_rng(seed)
    # Randomize before sorting to make equal-length group ties reproducible.
    rng.shuffle(groups)
    groups.sort(key=lambda g:sum(int(records[i]["length"]) for i in g), reverse=True)
    total = sum(int(r["length"]) for r in records)
    targets, actual = total*fractions, np.zeros(3)
    names, splits = ("train", "val", "test"), {"train": [], "val": [], "test": []}
    for n, group in enumerate(groups):
        # Reserve one remaining component for each empty split.
        empty = np.flatnonzero(actual == 0)
        split = int(empty[0]) if len(groups)-n == len(empty) else int(np.argmax((targets-actual)/targets))
        for index in group: splits[names[split]].append(records[index]["source_key"])
        actual[split] += sum(int(records[i]["length"]) for i in group)
    for values in splits.values(): values.sort()
    manifest = {"format": SPLIT_FORMAT, "archive_fingerprint": archive_fingerprint(records),
                "seed": int(seed), "group_by": ["operator_id", "original_recording"],
                "requested_fractions": fractions.tolist(), "num_components": len(groups),
                "frame_counts": dict(zip(names, actual.astype(int).tolist())), "splits": splits}
    validate_manifest(manifest, records)
    return manifest


def validate_manifest(manifest, records):
    if manifest.get("format") != SPLIT_FORMAT: raise ValueError("Unsupported split manifest format")
    if manifest.get("archive_fingerprint") != archive_fingerprint(records):
        raise ValueError("Split manifest archive fingerprint mismatch; rebuild deliberately for this archive")
    available = {r["source_key"]: r for r in records}
    owner, group_owner = {}, {}
    for split in ("train", "val", "test"):
        values = manifest.get("splits", {}).get(split, [])
        if not values: raise ValueError(f"Split manifest {split} is empty")
        for key in values:
            if key not in available: raise ValueError(f"Unknown archive sequence {key}")
            if key in owner: raise ValueError(f"Sequence {key} appears in multiple splits")
            owner[key] = split
            record = available[key]
            if not record["metadata"].get("operator_id"): raise ValueError("Operator grouping requires operator_id")
            for group in recording_keys(record):
                if group in group_owner and group_owner[group] != split:
                    raise ValueError(f"Operator/recording leakage across splits: {group}")
                group_owner[group] = split
    if set(owner) != set(available): raise ValueError("Manifest must partition every archive episode exactly once")
    return manifest


def load_split_manifest(path, records):
    if not path or not Path(path).is_file():
        raise FileNotFoundError("A grouped SPLIT_MANIFEST is required; run python -m MotionExpert.dataset.split_manifest")
    return validate_manifest(json.loads(Path(path).read_text()), records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]/"exp/dense_k12/splits.json")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fractions", nargs=3, type=float, default=(.8,.1,.1))
    args = parser.parse_args()
    root, output = args.archive_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Split manifests must be stored outside the immutable archive")
    manifest = create_split_manifest(read_records(root), seed=args.seed, fractions=args.fractions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2)+"\n"); temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "episodes": {s:len(v) for s,v in manifest["splits"].items()},
                      "frame_counts": manifest["frame_counts"]}, indent=2))


if __name__ == "__main__": main()
