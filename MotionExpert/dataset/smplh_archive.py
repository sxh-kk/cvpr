"""Owned SMPL-H parameter files and transient body FK, without pipeline imports.

Only the published archive, its own model and committed episode files are read.
The original dense archive is not used, and hand PCA pose is not a training label.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from ..config import SMPLH_MODEL_SHA256, SMPLH_SCHEMA
from ..utils.rotation_conversions import matrix_to_quaternion
from .parquet_reader import ProjectedParquetReader

MODEL_SHA256 = SMPLH_MODEL_SHA256
PARAMETER_SCHEMA_VERSION = SMPLH_SCHEMA
PARENTS = np.array([-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19])
DINO_COLUMN = "observation.features.dino"
OPTIONAL_COLUMNS = {"smplh.pose_hand_pca": ("float32", 90), "smplh.valid.hand_pose": ("bool", 2),
                    "source.capture_timestamp_ns": ("int64", 1), "source.video_frame_index": ("int64", 1)}


def _specs():
    out = {key: (dtype, 1) for key, dtype in (
        ("timestamp", "float32"), ("index", "int64"), ("frame_index", "int64"),
        ("episode_index", "int64"), ("task_index", "int64"),
        ("observation.source_timestamp", "float64"), ("observation.source_frame_left", "int64"),
        ("observation.source_frame_right", "int64"), ("observation.source_interpolation_weight", "float32"))}
    for name, width in (("hand_left", 63), ("hand_right", 63), ("head", 3), ("camera", 3), ("head_reference", 3)):
        out[f"observation.state.{name}_world"] = ("float32", width)
        for kind in ("valid", "interpolated"):
            out[f"observation.{kind}.{name}_position"] = ("bool", width // 3)
        if name in ("head", "camera", "head_reference"):
            out[f"observation.state.{name}_world_rotation"] = ("float32", 4)
            for kind in ("valid", "interpolated"):
                out[f"observation.{kind}.{name}_rotation"] = ("bool", 1)
    out.update({"observation.valid.dino": ("bool", 1), "observation.video_timestamp": ("float64", 1),
                "observation.video_time_error": ("float64", 1), "observation.environment.floor_height": ("float32", 1),
                "observation.environment.floor_normal": ("float32", 3), "observation.valid.floor": ("bool", 1),
                "observation.valid.floor_verified": ("bool", 1), "observation.valid.head_reference_basis": ("bool", 1)})
    for name, width in (("smplh.root_orient", 3), ("smplh.pose_body", 63), ("smplh.trans", 3),
                        ("smplh.betas", 16), ("source.feet_world", 6)):
        out[name] = ("float32", width)
    for component in ("position", "rotation"):
        for prefix in ("estimated.valid", "source.observed", "source.interpolated", "estimated.completed"):
            out[f"{prefix}.body_{component}"] = ("bool", 22)
        out[f"quality.{component}_confidence"] = ("float32", 22)
    out.update({"quality.observation_residual_valid": ("bool", 22), "quality.observation_residual_m": ("float32", 22),
                "source.feet_valid": ("bool", 2), "source.feet_interpolated": ("bool", 2)})
    return out


MOTION_SPECS = _specs()
MOTION_COLUMNS = tuple(MOTION_SPECS)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _stamp(path):
    s = Path(path).stat()
    return (s.st_size, s.st_mtime_ns)


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a SHA256 digest")
    return value


def _exp_so3(aa):
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1)
    cross = np.zeros((*aa.shape[:-1], 3, 3))
    x, y, z = np.moveaxis(aa, -1, 0)
    cross[..., 0, 1], cross[..., 0, 2] = -z, y
    cross[..., 1, 0], cross[..., 1, 2] = z, -x
    cross[..., 2, 0], cross[..., 2, 1] = -y, x
    return np.eye(3) + np.sinc(theta / np.pi)[..., None, None] * cross + \
        (.5 * np.sinc(theta / (2 * np.pi)) ** 2)[..., None, None] * (cross @ cross)


def assert_smplh_archive_ready(root):
    """Fast publication gate, before opening statistics or training assets."""
    state = json.loads((Path(root) / ".pipeline/dataset-state.json").read_text())
    if (state.get("schema_version") != PARAMETER_SCHEMA_VERSION or state.get("state") != "accepted"
            or state.get("independent_acceptance_pending") is not False):
        raise ValueError("SMPL-H archive is not published: independent full acceptance is pending")


class SMPLHBodyModel:
    """The shared full52 SMPL-H asset's body22 and sixteen actual shape bases."""
    def __init__(self, path, expected_sha256):
        if expected_sha256 != MODEL_SHA256 or file_sha256(path) != MODEL_SHA256:
            raise ValueError("SMPL-H model SHA mismatch; the matching shared16 model is required")
        before = _stamp(path)
        with np.load(path, allow_pickle=False) as asset:
            template = np.asarray(asset["v_template"], dtype=np.float64)
            shape = np.asarray(asset["shapedirs"], dtype=np.float64)
            regressor = np.asarray(asset["J_regressor"], dtype=np.float64)
            tree = np.asarray(asset["kintree_table"])
        if (template.shape != (6890, 3) or shape.shape != (6890, 3, 16)
                or regressor.shape != (52, 6890) or tree.shape != (2, 52)):
            raise ValueError("SMPL-H requires the full6890/52 asset and sixteen real shape bases")
        if not all(np.isfinite(v).all() for v in (template, shape, regressor, tree)):
            raise ValueError("Nonfinite SMPL-H model asset")
        ids = {int(child): i for i, child in enumerate(tree[1])}
        if len(ids) != 52:
            raise ValueError("Nonunique SMPL-H joint IDs")
        try:
            parents = np.array([-1] + [ids[int(parent)] for parent in tree[0, 1:]])
        except KeyError as error:
            raise ValueError("Invalid SMPL-H joint hierarchy") from error
        expected = PARENTS.tolist()
        for wrist in (20, 21):
            for _ in range(5):
                first = len(expected)
                expected.extend((wrist, first, first + 1))
        if not np.array_equal(parents, expected):
            raise ValueError("SMPL-H full52 hierarchy mismatch")
        self.rest = regressor[:22] @ template
        self.shape_basis = np.einsum("jv,vcb->jcb", regressor[:22], shape)
        self.model_sha256 = MODEL_SHA256
        if _stamp(path) != before:
            raise ValueError("SMPL-H model changed while loading")

    def decode(self, arrays):
        count = len(arrays["timestamp"])
        for key, width in (("smplh.root_orient", 3), ("smplh.pose_body", 63), ("smplh.trans", 3), ("smplh.betas", 16)):
            if np.asarray(arrays[key]).shape != (count, width) or not np.isfinite(arrays[key]).all():
                raise ValueError(f"{key} must contain finite T x {width} parameters")
        rest = self.rest[None] + np.einsum("tb,jcb->tjc", arrays["smplh.betas"], self.shape_basis)
        local = _exp_so3(np.concatenate((arrays["smplh.root_orient"][:, None], arrays["smplh.pose_body"].reshape(count, 21, 3)), 1))
        p, r = np.empty((count, 22, 3)), np.empty((count, 22, 3, 3))
        p[:, 0], r[:, 0] = rest[:, 0] + arrays["smplh.trans"], local[:, 0]
        for joint in range(1, 22):
            parent = PARENTS[joint]
            r[:, joint] = r[:, parent] @ local[:, joint]
            p[:, joint] = p[:, parent] + np.einsum("tij,tj->ti", r[:, parent], rest[:, joint] - rest[:, parent])
        if not (np.isfinite(p).all() and np.isfinite(r).all()):
            raise ValueError("SMPL-H FK produced nonfinite values")
        q = matrix_to_quaternion(torch.from_numpy(r)).numpy()
        return p.astype(np.float32), q.astype(np.float32)


class SMPLHParameterArchive:
    def __init__(self, root, *, cache_mb=128, file_cache=8, allow_unpublished=False,
                 vision_profile_root=None, image_profile=None):
        if type(allow_unpublished) is not bool:
            raise ValueError("allow_unpublished must be boolean")
        self.root = Path(root).resolve(strict=True)
        self.allow_unpublished = allow_unpublished
        self.vision_cache = None
        self.reader = ProjectedParquetReader(cache_mb=cache_mb, file_cache=file_cache)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if (self.info.get("schema_version") != PARAMETER_SCHEMA_VERSION or self.info.get("fps") != 30
                or self.info.get("requires_old_dense_archive") is not False):
            raise ValueError("Not an independent native30Hz SMPL-H parameter archive")
        self.state_path = self.root / ".pipeline/dataset-state.json"
        self._assert_publication()
        identity = self.info.get("model", {})
        if identity.get("shape_basis_count") != 16 or identity.get("body_joint_count") != 22 or identity.get("joint_count") != 52:
            raise ValueError("Archive must declare shared SMPL-H body22/full52/shape16")
        self.model_path = self.owned_path(identity["path"])
        self.model = SMPLHBodyModel(self.model_path, identity.get("model_sha256"))
        self.model_sha256 = self.model.model_sha256
        self._asset_stamps = {self.model_path: _stamp(self.model_path)}
        paths = sorted((self.root / ".pipeline/episodes").glob("*.json"))
        if not paths:
            raise ValueError("No committed SMPL-H parameter episodes")
        self._records = [json.loads(p.read_text()) for p in paths]
        self._retirement_path = self.root / ".pipeline/retired-vision.json"
        self._retirement_stamp = None
        self._retired_vision = {}
        if self._retirement_path.exists():
            retirement = json.loads(self._retirement_path.read_text())
            if (retirement.get("format") != "smplh-legacy-vision-retirement-v1"
                    or retirement.get("info_sha256") != file_sha256(self.root / "meta/info.json")
                    or retirement.get("records_manifest_sha256") != hashlib.sha256(canonical_json(self._records)).hexdigest()
                    or not isinstance(retirement.get("episodes"), list)):
                raise ValueError("Legacy vision retirement is not bound to this SMPL-H archive")
            for entry in retirement["episodes"]:
                idx = entry.get("episode_index")
                if type(idx) is not int or not 0 <= idx < len(self._records) or idx in self._retired_vision:
                    raise ValueError("Invalid or duplicate retired vision episode")
                record = self._records[idx]
                if entry.get("path") != record["vision_path"] or entry.get("sha256") != record["vision_sha256"]:
                    raise ValueError("Retired vision identity differs from committed record")
                relative = Path(entry["path"])
                if (relative.is_absolute() or ".." in relative.parts
                        or not (self.root / relative).resolve().is_relative_to(self.root)):
                    raise ValueError("Retired vision path escapes the owned archive")
                self._retired_vision[idx] = entry
            self._retirement_stamp = _stamp(self._retirement_path)
        self._record_paths, self._record_stamps, self._artifact_stamps = {}, {}, {}
        offset = 0
        for i, (path, record) in enumerate(zip(paths, self._records)):
            if type(record.get("episode_index")) is not int or record["episode_index"] != i:
                raise ValueError("SMPL-H episode identities must be contiguous and zero based")
            if type(record.get("length")) is not int or record["length"] < 1:
                raise ValueError("SMPL-H episodes must be nonempty")
            if record.get("dataset_from_index") != offset or record.get("dataset_to_index") != offset + record["length"]:
                raise ValueError("SMPL-H global frame bounds differ from episode lengths")
            offset += record["length"]
            self._record_paths[i], self._record_stamps[i] = path, _stamp(path)
            receipt_path = self.owned_path(f"episodes/{i:08d}/receipt.json")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("schema_version") != PARAMETER_SCHEMA_VERSION or receipt.get("record") != record or not receipt.get("completed_at"):
                raise ValueError("SMPL-H episode lacks its matching committed receipt")
            _sha(receipt.get("conversion_binding"), "conversion_binding")
            self._artifact_stamps[receipt_path] = _stamp(receipt_path)
            for role, key in (("motion", "data_path"), ("vision", "vision_path")):
                digest = _sha(record.get(f"{role}_sha256"), f"{role}_sha256")
                if receipt.get(role, {}).get("sha256") != digest:
                    raise ValueError(f"SMPL-H {role} receipt digest differs from episode record")
                original = receipt.get("output_identities", {}).get(role, {})
                if role == "vision" and i in self._retired_vision:
                    retired = self._retired_vision[i]
                    if (retired.get("bytes"), retired.get("mtime_ns")) != (original.get("bytes"), original.get("mtime_ns")):
                        raise ValueError("Retired vision identity differs from committed receipt")
                    # Historical receipts remain immutable; replacement features
                    # have their own profile manifest and original motion joins.
                    continue
                artifact = self.owned_path(record[key])
                stamp = _stamp(artifact)
                if stamp != (original.get("bytes"), original.get("mtime_ns")):
                    raise ValueError(f"Committed SMPL-H {role} file identity changed")
                self._artifact_stamps[artifact] = stamp
            meta = record.get("metadata", {})
            if (meta.get("target_joint_basis") != "smplh_body22_axes_v1"
                    or meta.get("target_model_axes_defined") is False
                    or meta.get("label_kind") != "model_fitted_estimate"):
                raise ValueError("SMPL-H parameters must declare the common model-defined body axes")
        if len({d.get("source_key") for d in self._records}) != len(self._records):
            raise ValueError("Duplicate SMPL-H source identities")
        self.frames = offset
        self.manifest_path = self.root / "meta/splits.json"
        self.manifest = json.loads(self.manifest_path.read_text())
        self._validate_splits()
        self.split_fingerprint = hashlib.sha256(canonical_json(self.manifest)).hexdigest()
        self.records_manifest_sha256 = hashlib.sha256(canonical_json(self._records)).hexdigest()
        split_paths = sorted((self.root / "meta").glob("splits*.json"))
        self._metadata_stamps = {p: _stamp(p) for p in (self.root / "meta/info.json", *split_paths,
                                                       self.root / "meta/episodes.parquet")}
        rows = pq.read_table(self.root / "meta/episodes.parquet").to_pylist()
        if len(rows) != len(self._records) or any(any(record.get(k) != v for k, v in row.items()) for row, record in zip(rows, self._records)):
            raise ValueError("SMPL-H episode table differs from committed records")
        self.binding = {"info_sha256": file_sha256(self.root / "meta/info.json"),
                        "records_manifest_sha256": self.records_manifest_sha256,
                        "episodes_metadata_sha256": file_sha256(self.root / "meta/episodes.parquet"),
                        "split_metadata_sha256": {str(p.relative_to(self.root)): file_sha256(p) for p in split_paths},
                        "model_sha256": self.model_sha256}
        hand_basis = self.info.get("hand_basis")
        if hand_basis:
            path = self.owned_path(hand_basis["path"])
            if file_sha256(path) != hand_basis.get("sha256"):
                raise ValueError("Owned SMPL-H hand basis digest mismatch")
            self.binding["hand_basis_sha256"] = hand_basis["sha256"]
            self._artifact_stamps[path] = _stamp(path)
            self._asset_stamps[path] = _stamp(path)
        self.archive_fingerprint = hashlib.sha256(canonical_json(self.binding)).hexdigest()
        self.manifest = dict(self.manifest, archive_fingerprint=self.archive_fingerprint)
        self._check_acceptance()
        if vision_profile_root is not None:
            from .vision_cache import VisionProfileCache
            self.vision_cache = VisionProfileCache(vision_profile_root, self, image_profile=image_profile,
                                                   file_cache=file_cache)
        elif image_profile is not None:
            raise ValueError("IMAGE_PROFILE requires VISION_PROFILE_ROOT for archive training")
        self._artifact_stamps[self.model_path] = _stamp(self.model_path)
        self.assert_current()

    def owned_path(self, value):
        relative = Path(value)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("SMPL-H data/model paths must be owned relative paths")
        path = (self.root / relative).resolve(strict=True)
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError("SMPL-H dependency escapes the owned parameter archive")
        return path

    def _assert_publication(self):
        state = json.loads(self.state_path.read_text())
        if state.get("schema_version") != PARAMETER_SCHEMA_VERSION:
            raise ValueError("SMPL-H publication schema mismatch")
        if not self.allow_unpublished and (state.get("state") != "accepted" or state.get("independent_acceptance_pending") is not False):
            raise ValueError("SMPL-H archive is not published: independent full acceptance is pending")
        self.state = state

    def _check_acceptance(self):
        if self.state.get("episodes") != len(self._records) or self.state.get("frames") != self.frames:
            raise ValueError("SMPL-H published population differs from owned records")
        report_path = self.state.get("acceptance_report")
        if self.allow_unpublished or not report_path:
            return
        path = Path(report_path)
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        # Publication state is the durable contract. A moved archive need not
        # retain an optional external job report or any old dense files.
        if not path.is_file():
            if path.exists() or path.is_relative_to(self.root):
                raise ValueError("SMPL-H owned full acceptance report is missing")
            return
        report = json.loads(path.read_text())
        if not isinstance(report, dict):
            raise ValueError("SMPL-H full acceptance report must be a mapping")
        qualified = report.get("format") == "qualified-standalone-smplh-parameter-full-audit-v1"
        required = ("passed", "parameter_and_source_integrity_passed", "old_dense_dependency_absent", "cleanup_allowed")
        required += (("original_fit_numerical_quality_passed", "latest_motionexpert_accepted") if qualified else
                     ("original_egosuite_full_fit_acceptance_passed",))
        if (report.get("format") not in ("independent-standalone-smplh-parameter-full-audit-v1",
                                         "qualified-standalone-smplh-parameter-full-audit-v1")
                or report.get("scope") != "full" or report.get("failures")
                or any(report.get(k) is not True for k in required)):
            raise ValueError("SMPL-H full acceptance report is absent, partial or failed")
        population = report.get("population", {})
        if (not isinstance(population, dict) or population.get("episodes") != len(self._records)
                or population.get("frames") != self.frames):
            raise ValueError("SMPL-H acceptance population mismatch")
        if not isinstance(report.get("binding"), dict):
            raise ValueError("SMPL-H acceptance binding is missing")
        for key, actual in self.binding.items():
            declared = report.get("binding", {}).get(key)
            if declared != actual:
                raise ValueError(f"SMPL-H acceptance {key} mismatch")
        if qualified:
            self._check_qualified_proofs(report, path)

    def _check_qualified_proofs(self, report, report_path):
        """Check requalification evidence once, without reopening old motion data.

        The old registered decision remains failed. Its numerical requalification
        and the latest MotionExpert interface decision are separate, bound proofs.
        External job evidence remains optional after archive relocation, as does
        the external acceptance report; evidence stored inside the archive is owned.
        """
        if (report.get("old_registered_fit_acceptance_passed") is not False
                or report.get("original_egosuite_full_fit_acceptance_passed", False) is not False
                or report.get("failures") != [] or report.get("partial", False) is not False):
            raise ValueError("SMPL-H qualification must preserve the failed original registered decision")
        sources = {}
        for record in self._records:
            value = sources.setdefault(record["source"], {"episodes": 0, "frames": 0})
            value["episodes"] += 1
            value["frames"] += record["length"]
        if report["population"].get("sources") != sources:
            raise ValueError("SMPL-H qualification source population mismatch")
        proofs = report.get("proofs", {})
        if not isinstance(proofs, dict):
            raise ValueError("SMPL-H qualification proofs must be a mapping")
        names = ("parameter_integrity_audit", "original_fit_qualification", "latest_motionexpert_acceptance",
                 "authorization", "latest_me_baseline")
        loaded = {}
        for name in names:
            descriptor = proofs.get(name)
            if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str) or not descriptor["path"]:
                raise ValueError(f"SMPL-H qualification {name} proof is missing")
            digest = _sha(descriptor.get("sha256"), f"SMPL-H qualification {name}")
            path = Path(descriptor["path"])
            owned = not path.is_absolute() or path.is_relative_to(self.root)
            if not path.is_absolute():
                if ".." in path.parts:
                    raise ValueError(f"SMPL-H qualification {name} relative proof escapes the archive")
                path = self.root / path
            path = path.resolve()
            if owned and not path.is_relative_to(self.root):
                raise ValueError(f"SMPL-H qualification owned {name} proof escapes the archive")
            if not path.is_file():
                if path.exists() or path.is_relative_to(self.root):
                    raise ValueError(f"SMPL-H qualification owned {name} proof is missing")
                continue
            if file_sha256(path) != digest:
                raise ValueError(f"SMPL-H qualification {name} proof SHA mismatch")
            loaded[name] = json.loads(path.read_text())
            if not isinstance(loaded[name], dict):
                raise ValueError(f"SMPL-H qualification {name} proof must be a mapping")
        for name in ("parameter_integrity_audit", "original_fit_qualification", "latest_motionexpert_acceptance"):
            proof = loaded.get(name)
            if proof is not None and (proof.get("scope") != "full" or proof.get("partial", False) is not False):
                raise ValueError(f"SMPL-H qualification {name} proof is partial")
        for name in ("parameter_integrity_audit", "latest_motionexpert_acceptance"):
            proof = loaded.get(name)
            if proof is not None:
                if proof.get("population") != report["population"]:
                    raise ValueError(f"SMPL-H qualification {name} proof population mismatch")
                for key, actual in self.binding.items():
                    if proof.get("binding", {}).get(key) != actual:
                        raise ValueError(f"SMPL-H qualification {name} proof {key} mismatch")
        integrity = loaded.get("parameter_integrity_audit")
        if integrity is not None and (integrity.get("format") != "independent-standalone-smplh-parameter-full-audit-v1"
                                      or integrity.get("parameter_and_source_integrity_passed") is not True
                                      or integrity.get("old_dense_dependency_absent") is not True
                                      or integrity.get("original_egosuite_full_fit_acceptance_passed") is not False
                                      or not isinstance(integrity.get("failures"), list)
                                      or any(not isinstance(failure, dict) or failure.get("stage") != "original_egosuite_full_fit_acceptance"
                                             for failure in integrity["failures"])):
            raise ValueError("SMPL-H qualification parameter integrity proof failed")
        fit = loaded.get("original_fit_qualification")
        if fit is not None:
            if (fit.get("format") != "original-smplh-fit-requalification-v1" or fit.get("failures")
                    or any(fit.get(k) is not True for k in ("passed", "numeric_quality_passed", "data_preservation_passed",
                                                         "frozen_producer_passed"))
                    or fit.get("old_registered_acceptance_passed") is not False):
                raise ValueError("SMPL-H qualification original fit proof failed")
            if ({"episodes": fit.get("episodes"), "frames": fit.get("frames")} != sources.get("egosuite")
                    or fit.get("model_sha256") != self.binding["model_sha256"]):
                raise ValueError("SMPL-H qualification original fit proof population/model mismatch")
        me = loaded.get("latest_motionexpert_acceptance")
        if me is not None and (me.get("format") != "latest-motionexpert-acceptance-v1" or me.get("passed") is not True
                               or me.get("failures")):
            raise ValueError("SMPL-H qualification MotionExpert proof failed")
        if me is not None:
            # The qualification gate checks these nested artifacts before committing
            # this proof. Only their identities are consumed here: their original
            # job paths are not dependencies of a moved, owned training archive.
            for name in ("full_population_probe", "regression", "model_smoke"):
                descriptor = me.get(name)
                if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str) or not descriptor["path"]:
                    raise ValueError(f"SMPL-H qualification MotionExpert {name} identity is missing")
                _sha(descriptor.get("sha256"), f"SMPL-H qualification MotionExpert {name}")
        baseline_sha = proofs["latest_me_baseline"]["sha256"]
        for name, proof in (("original_fit_qualification", fit), ("latest_motionexpert_acceptance", me)):
            if proof is not None and proof.get("latest_baseline_sha256") != baseline_sha:
                raise ValueError(f"SMPL-H qualification {name} baseline mismatch")
        baseline = loaded.get("latest_me_baseline")
        if baseline is not None:
            if baseline.get("format") != "latest-motionexpert-baseline-v1" or not isinstance(baseline.get("files"), dict) or not baseline["files"]:
                raise ValueError("SMPL-H qualification MotionExpert baseline is missing")
            for digest in baseline["files"].values():
                _sha(digest, "SMPL-H qualification MotionExpert baseline file")
        authorization = loaded.get("authorization")
        if authorization is not None and (authorization.get("format") != "smplh-requalification-authorization-v1"
                or authorization.get("scope") != "latest_motionexpert_baseline_only"
                or authorization.get("preserve_original_failed_report") is not True
                or authorization.get("retain_original_numeric_policy") is not True):
            raise ValueError("SMPL-H qualification authorization differs from the preserved original policy")

    def _validate_splits(self):
        splits = self.manifest.get("splits", {})
        if set(splits) != {"train", "val", "test"}:
            raise ValueError("SMPL-H archive needs original train/val/test membership")
        membership = {}
        for split, keys in splits.items():
            if not isinstance(keys, list):
                raise ValueError("SMPL-H split membership must be a list")
            for key in keys:
                if key in membership:
                    raise ValueError("Duplicate or overlapping SMPL-H split membership")
                membership[key] = split
        if set(membership) != {d["source_key"] for d in self._records}:
            raise ValueError("SMPL-H split population differs from committed episodes")
        groups = self.manifest.get("split_group_ids", {})
        actual_groups = {}
        for record in self._records:
            if membership[record["source_key"]] != record.get("split"):
                raise ValueError("SMPL-H split differs from original committed membership")
            if not record.get("split_group_ids"):
                raise ValueError("SMPL-H participant/recording split groups are missing")
            for group in record["split_group_ids"]:
                if group in actual_groups and actual_groups[group] != record["split"]:
                    raise ValueError("SMPL-H participant or recording leaks between splits")
                actual_groups[group] = record["split"]
        if actual_groups != groups:
            raise ValueError("SMPL-H split group provenance differs from committed records")

    def records(self):
        return deepcopy(self._records)

    def data_path(self, episode_index):
        return self.owned_path(self._records[episode_index]["data_path"])

    def assert_current(self, record=None):
        self._assert_publication()
        if self.state.get("episodes") != len(self._records) or self.state.get("frames") != self.frames:
            raise ValueError("SMPL-H publication population changed")
        for path, original in self._metadata_stamps.items():
            if _stamp(path) != original:
                raise ValueError("SMPL-H archive metadata changed; rebuild dataset and statistics")
        if self._retirement_stamp is not None and _stamp(self._retirement_path) != self._retirement_stamp:
            raise ValueError("Legacy vision retirement changed while reading the archive")
        if self._retirement_stamp is None and self._retirement_path.exists():
            raise ValueError("Legacy vision retirement was added; reopen the archive")
        indices = range(len(self._records)) if record is None else (record["episode_index"],)
        for idx in indices:
            if _stamp(self._record_paths[idx]) != self._record_stamps[idx]:
                raise ValueError("SMPL-H episode record changed; rebuild dataset and statistics")
            rec = self._records[idx]
            names = [rec["data_path"], f"episodes/{idx:08d}/receipt.json"]
            if idx not in self._retired_vision:
                names.append(rec["vision_path"])
            for name in names:
                path = self.owned_path(name)
                if _stamp(path) != self._artifact_stamps[path]:
                    raise ValueError("Committed SMPL-H episode file changed")
        if any(_stamp(path) != original for path, original in self._asset_stamps.items()):
            raise ValueError("Owned SMPL-H model or hand basis changed")
        if self.vision_cache is not None:
            self.vision_cache.assert_current(None if record is None else record["episode_index"])

    def read_image_validity(self, record, selected, *, expected=None):
        if self.vision_cache is not None:
            return self.vision_cache.read(record, selected, expected=expected)["observation.valid.dino"]
        if expected is not None and "observation.valid.dino" in expected:
            return expected["observation.valid.dino"].copy()
        return self.reader.read(self.data_path(record["episode_index"]), selected,
                                ["observation.valid.dino"])["observation.valid.dino"]

    def _motion(self, record, selected):
        path = self.data_path(record["episode_index"])
        _, (parquet, _) = self.reader._file(path)
        if parquet.metadata.num_rows != record["length"] or parquet.schema_arrow.metadata.get(b"parameter_schema_version") != PARAMETER_SCHEMA_VERSION.encode():
            raise ValueError("SMPL-H physical row count/schema differs from record")
        available = set(parquet.schema_arrow.names)
        if set(MOTION_SPECS) - available or available - set(MOTION_SPECS) - set(OPTIONAL_COLUMNS):
            raise ValueError("SMPL-H parameter columns differ; dense body caches are unsupported")
        specs = {**MOTION_SPECS, **{k: v for k, v in OPTIONAL_COLUMNS.items() if k in available}}
        arrays = self.reader.read(path, selected, specs)
        count = len(selected)
        for key, (dtype, width) in specs.items():
            value = arrays[key]
            if value.shape != ((count,) if width == 1 else (count, width)) or value.dtype != np.dtype(dtype):
                raise ValueError(f"SMPL-H {key} shape/dtype mismatch")
            if not np.isfinite(value).all():
                raise ValueError(f"SMPL-H {key} contains nonfinite stored values")
        if (not np.array_equal(arrays["frame_index"], selected) or not np.all(arrays["episode_index"] == record["episode_index"])
                or not np.array_equal(arrays["index"], record["dataset_from_index"] + selected)
                or not np.array_equal(arrays["timestamp"], (selected / 30).astype(np.float32))):
            raise ValueError("SMPL-H motion frame/time/global-index join differs")
        for component in ("position", "rotation"):
            valid, observed = arrays[f"estimated.valid.body_{component}"], arrays[f"source.observed.body_{component}"]
            if not np.array_equal(arrays[f"estimated.completed.body_{component}"], valid & ~observed):
                raise ValueError("SMPL-H model completion differs from source observation boundary")
            if np.any(arrays[f"source.interpolated.body_{component}"] & ~observed):
                raise ValueError("SMPL-H source interpolation exceeds source availability")
            confidence = arrays[f"quality.{component}_confidence"]
            if np.any((confidence < 0) | (confidence > 1)) or np.any(confidence[~valid] != 0):
                raise ValueError("SMPL-H confidence exceeds explicit estimate availability")
        if (not np.array_equal(arrays["source.feet_valid"], arrays["source.observed.body_position"][:, [10, 11]])
                or not np.array_equal(arrays["source.feet_interpolated"], arrays["source.interpolated.body_position"][:, [10, 11]])):
            raise ValueError("SMPL-H original foot validity differs from source observations")
        if np.any(arrays["source.feet_interpolated"] & ~arrays["source.feet_valid"]):
            raise ValueError("SMPL-H foot interpolation exceeds original validity")
        residual_valid = arrays["quality.observation_residual_valid"]
        if (np.any(arrays["quality.observation_residual_m"] < 0) or np.any(residual_valid & ~(
                arrays["source.observed.body_position"] & arrays["estimated.valid.body_position"]))):
            raise ValueError("SMPL-H residual quality exceeds observed and estimated availability")
        ground = record["metadata"].get("ground", {})
        basis = record["metadata"].get("head_reference", {}).get("basis_verified", False)
        for key, value in (("observation.valid.floor", bool(ground.get("valid", False))),
                           ("observation.valid.floor_verified", bool(ground.get("valid", False) and ground.get("verified", False))),
                           ("observation.valid.head_reference_basis", bool(basis))):
            if not np.all(arrays[key] == value):
                raise ValueError(f"SMPL-H stored reference availability differs from metadata: {key}")
        if not np.array_equal(arrays["observation.environment.floor_normal"], np.tile(np.array([0, 0, 1], np.float32), (count, 1))):
            raise ValueError("SMPL-H floor normal must retain the horizontal output-world plane")
        if (not np.isfinite(ground.get("height_m", np.nan)) or
                not np.all(arrays["observation.environment.floor_height"] == np.float32(ground["height_m"]))):
            raise ValueError("SMPL-H stored floor height differs from episode metadata")
        for key in arrays:
            if key.startswith("observation.interpolated.") and np.any(arrays[key] & ~arrays[key.replace(".interpolated.", ".valid.")]):
                raise ValueError("SMPL-H modality interpolation exceeds validity")
        return arrays

    def read_window(self, episode_index, *, start_frame=0, num_frames=11, fps=5, forecast_prefix=6, load_dino=True):
        if type(episode_index) is not int or not 0 <= episode_index < len(self._records):
            raise ValueError("SMPL-H episode index outside archive")
        if type(fps) is not int or fps < 1 or 30 % fps or type(start_frame) is not int or start_frame < 0 or start_frame % (30 // fps):
            raise ValueError("SMPL-H window starts must match the zero-phase training FPS")
        if type(num_frames) is not int or num_frames < 2 or type(forecast_prefix) is not int or not 0 < forecast_prefix < num_frames:
            raise ValueError("SMPL-H complete windows require a nonempty prefix and future")
        if self.vision_cache is not None and fps != 5:
            raise ValueError("The shared vision cache uses the zero-phase 5Hz protocol")
        record = self._records[episode_index]
        self.assert_current(record)
        selected = start_frame + np.arange(num_frames) * (30 // fps)
        if selected[-1] >= record["length"]:
            raise ValueError("SMPL-H training requires a complete window without tail padding")
        arrays = self._motion(record, selected)
        p, q = self.model.decode(arrays)
        for component, value in (("position", p), ("rotation", q)):
            valid = arrays[f"estimated.valid.body_{component}"]
            key = "observation.state.body_world" + ("_rotation" if component == "rotation" else "")
            arrays[key] = np.where(valid[..., None], value, 0 if component == "position" else [1, 0, 0, 0]).reshape(num_frames, -1).astype(np.float32)
            arrays[f"observation.valid.body_{component}"] = valid.copy()
            arrays[f"observation.source_observed.body_{component}"] = arrays[f"source.observed.body_{component}"].copy()
            arrays[f"observation.estimated.completed.body_{component}"] = arrays[f"estimated.completed.body_{component}"].copy()
        if self.vision_cache is not None:
            expected = {key: value[:forecast_prefix] for key, value in arrays.items()}
            image = self.vision_cache.read(record, selected[:forecast_prefix], expected=expected,
                                            load_features=load_dino)
            valid = image["observation.valid.dino"]
            arrays["observation.valid.dino"] = np.r_[valid, np.zeros(num_frames - forecast_prefix, bool)]
            for key in ("observation.video_timestamp", "observation.video_time_error"):
                arrays[key] = np.r_[image[key], np.zeros(num_frames - forecast_prefix, np.float64)]
            if load_dino:
                arrays[DINO_COLUMN] = np.zeros((num_frames, 5120), np.float32)
                arrays[DINO_COLUMN][:forecast_prefix] = image[DINO_COLUMN]
        elif load_dino:
            raise ValueError("DINO must be re-extracted with the shared image profile; set VISION_PROFILE_ROOT")
        metadata = deepcopy(record["metadata"])
        metadata.update(rotation_supervision_policy="smplh_model_defined", rotation_target_basis="smplh_body22_axes_v1",
                        target_model_axes_defined=True, model_sha256=self.model_sha256, archive_schema_version=PARAMETER_SCHEMA_VERSION)
        if self.vision_cache is not None:
            metadata["vision_profile"] = deepcopy(self.vision_cache.metadata)
        # Original certification flags are retained. Model-defined axes enable
        # estimated-parameter supervision without claiming native axis truth.
        self.assert_current(record)
        return arrays, metadata
