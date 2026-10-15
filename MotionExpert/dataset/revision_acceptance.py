"""Small, strict independent evidence gate for HM-quality archive revisions."""
from __future__ import annotations

import json
from pathlib import Path

from .smplh_archive import file_sha256
from .window_cache import digest


ACCEPTANCE_FORMAT = "independent-smplh-hm-quality-revision-acceptance-v1"
REVISION_FORMAT = "smplh-hm-quality-revision-v1"
AUDIT_FORMAT = "egoexo4d-hm-fit-quality-audit-v1"
INDEPENDENT_AUDIT_FORMAT = "independent-smplh-hm-quality-revision-delta-audit-v1"
ACCEPTANCE_FLAGS = ("parent_acceptance_inherited", "parameter_values_preserved", "camera_values_preserved",
                    "image_join_values_preserved", "quality_masks_accepted", "ground_recomputed_accepted",
                    "latest_motionexpert_accepted")


def _proof(root, descriptor, name):
    if not isinstance(descriptor, dict) or not isinstance(descriptor.get("path"), str):
        raise ValueError(f"HM quality revision {name} proof is missing")
    digest_value = descriptor.get("sha256")
    if (not isinstance(digest_value, str) or len(digest_value) != 64
            or any(c not in "0123456789abcdef" for c in digest_value)):
        raise ValueError(f"HM quality revision {name} proof SHA is invalid")
    relative = Path(descriptor["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("HM quality revision proofs must be owned relative paths")
    path = (root / relative).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("HM quality revision proof escapes its archive")
    if file_sha256(path) != digest_value:
        raise ValueError(f"HM quality revision {name} proof SHA differs")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"HM quality revision {name} proof must be a mapping")
    return value


def check_revision_acceptance(archive, report, report_path):
    """Verify an independent decision and parent proof chain, never auto-accept."""
    if (report.get("format") != ACCEPTANCE_FORMAT or report.get("scope") != "full"
            or report.get("passed") is not True or report.get("failures") != []
            or any(report.get(name) is not True for name in ACCEPTANCE_FLAGS)):
        raise ValueError("HM quality revision independent acceptance failed or is partial")
    population = report.get("population", {})
    if population.get("episodes") != len(archive._records) or population.get("frames") != archive.frames:
        raise ValueError("HM quality revision acceptance population mismatch")
    if report.get("binding") != archive.binding:
        raise ValueError("HM quality revision acceptance binding differs")
    proofs = report.get("proofs", {})
    required = ("revision_manifest", "quality_audit", "independent_revision_audit", "parent_publication")
    loaded = {name: _proof(archive.root, proofs.get(name), name) for name in required}
    revision, quality, independent, parent = (loaded[name] for name in required)
    if (revision.get("format") != REVISION_FORMAT or revision.get("scope") != "full"
            or revision.get("binding") != archive.binding
            or revision.get("archive_fingerprint") != archive.archive_fingerprint
            or not set(revision.get("required_quality_episodes", [])) <= set(revision.get("quality_episodes", []))
            or revision.get("build_binding_sha256") != digest(revision.get("build_binding"))):
        raise ValueError("HM quality revision manifest is partial or not bound to this archive")
    build = revision["build_binding"]
    required_ids = {record["episode_index"] for record in archive._records
                    if record["source"] == "egoexo4d_hm"
                    and record["metadata"].get("training_eligible") is not False
                    and record["metadata"].get("canonical_training_eligible") is not False
                    and not record["metadata"].get("quarantine_reasons", [])}
    quality_ids = [receipt.get("episode_index") for receipt in quality.get("episodes", [])]
    if (build.get("scope") != "full" or required_ids != set(revision["required_quality_episodes"])
            or len(set(quality_ids)) != len(quality_ids) or not required_ids <= set(quality_ids)
            or sorted(quality_ids) != revision["quality_episodes"]
            or quality.get("episodes_completed") != len(quality_ids)
            or quality.get("diagnostic") is not False
            or quality.get("source_hm_population", {}).get("scope", quality.get("scope")) != "all_archive_HM"):
        raise ValueError("HM quality revision evidence does not cover its complete eligible population")
    if (parent.get("state") != "accepted" or parent.get("independent_acceptance_pending") is not False
            or parent.get("episodes") != len(archive._records) or parent.get("frames") != archive.frames
            or build.get("parent_publication_sha256") != proofs["parent_publication"]["sha256"]):
        raise ValueError("HM quality revision parent acceptance is missing or differs")
    if build.get("parent_acceptance_sha256"):
        old = _proof(archive.root, proofs.get("parent_acceptance"), "parent_acceptance")
        if (build["parent_acceptance_sha256"] != proofs["parent_acceptance"]["sha256"]
                or old.get("passed") is not True or old.get("scope") != "full"
                or not isinstance(old.get("binding"), dict)
                or any(old["binding"].get(key) != value for key, value in build.get("parent_archive_binding", {}).items())):
            raise ValueError("HM quality revision inherited acceptance identity differs")
    if (quality.get("format") != AUDIT_FORMAT or quality.get("state") != "complete"
            or quality.get("production_modified") is not False
            or build.get("quality_report_sha256") != proofs["quality_audit"]["sha256"]
            or independent.get("format") != INDEPENDENT_AUDIT_FORMAT
            or any(independent.get(name) is not True for name in ACCEPTANCE_FLAGS)
            or independent.get("passed") is not True or independent.get("scope") != "full"
            or independent.get("failures") != [] or independent.get("binding") != archive.binding
            or independent.get("population") != population
            or independent.get("quality_audit_sha256") != proofs["quality_audit"]["sha256"]
            or independent.get("revision_manifest_sha256") != proofs["revision_manifest"]["sha256"]):
        raise ValueError("HM quality revision independent evidence differs")
