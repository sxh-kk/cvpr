"""Owned independent evidence gate for dual-source final-FK quality revisions."""
from __future__ import annotations

from .revision_acceptance import _proof
from .window_cache import digest

ACCEPTANCE_FORMAT = "independent-smplh-fit-quality-revision-acceptance-v1"
REVISION_FORMAT = "smplh-fit-quality-revision-v1"
AUDIT_FORMAT = "smplh-final-fk-quality-audit-v1"
INDEPENDENT_AUDIT_FORMAT = "independent-smplh-fit-quality-revision-delta-audit-v1"
ACCEPTANCE_FLAGS = ("parent_acceptance_inherited", "parameter_values_preserved", "camera_values_preserved",
                    "image_join_values_preserved", "quality_masks_accepted", "ground_recomputed_accepted",
                    "latest_motionexpert_accepted", "final_fk_quality_independently_checked")


def eligible(record):
    metadata = record["metadata"]
    return (record["source"] in {"egosuite", "egoexo4d_hm"}
            and metadata.get("training_eligible") is not False
            and metadata.get("canonical_training_eligible") is not False
            and not metadata.get("quarantine_reasons", []))


def check_fit_revision_acceptance(archive, report, report_path):
    """Verify the complete owned proof chain; no original payload requalification."""
    if (report.get("format") != ACCEPTANCE_FORMAT or report.get("scope") != "full"
            or report.get("passed") is not True or report.get("failures") != []
            or any(report.get(name) is not True for name in ACCEPTANCE_FLAGS)):
        raise ValueError("Final-FK revision independent acceptance failed or is partial")
    population = report.get("population", {})
    if (population.get("episodes") != len(archive._records) or population.get("frames") != archive.frames
            or report.get("binding") != archive.binding):
        raise ValueError("Final-FK revision acceptance population/binding differs")
    proofs = report.get("proofs", {})
    names = ("revision_manifest", "quality_audit", "independent_revision_audit", "parent_publication")
    revision, quality, independent, parent = [_proof(archive.root, proofs.get(name), name) for name in names]
    build = revision.get("build_binding", {})
    required = {record["episode_index"] for record in archive._records if eligible(record)}
    receipts = quality.get("episodes", [])
    indices = [receipt.get("episode_index") for receipt in receipts]
    source_population = quality.get("source_population", {})
    if (revision.get("format") != REVISION_FORMAT or revision.get("scope") != "full"
            or revision.get("binding") != archive.binding
            or revision.get("archive_fingerprint") != archive.archive_fingerprint
            or revision.get("build_binding_sha256") != digest(build) or build.get("scope") != "full"
            or set(revision.get("required_quality_episodes", [])) != required
            or revision.get("quality_episodes") != sorted(required)
            or len(indices) != len(set(indices)) or set(indices) != required
            or quality.get("episodes_completed") != len(required)
            or quality.get("episodes_planned") != len(required)
            or quality.get("diagnostic") is not False
            or source_population.get("scope") != "all_eligible_archive"
            or source_population.get("total") != len(archive._records)
            or source_population.get("episodes", source_population.get("total")) != source_population.get("total")
            or source_population.get("eligible") != len(required)
            or len(source_population.get("excluded", [])) != len(archive._records) - len(required)
            or {item.get("episode_index") for item in source_population.get("excluded", [])}
                != {record["episode_index"] for record in archive._records} - required):
        raise ValueError("Final-FK revision proof does not cover the complete eligible dual-source population")
    if (parent.get("state") != "accepted" or parent.get("independent_acceptance_pending") is not False
            or parent.get("episodes") != len(archive._records) or parent.get("frames") != archive.frames
            or build.get("parent_publication_sha256") != proofs["parent_publication"]["sha256"]):
        raise ValueError("Final-FK revision inherited parent publication differs")
    if parent.get("quality_revision") and not build.get("parent_acceptance_sha256"):
        raise ValueError("Final-FK revision cannot drop its revised parent independent proof")
    if build.get("parent_acceptance_sha256"):
        old = _proof(archive.root, proofs.get("parent_acceptance"), "parent_acceptance")
        if (build["parent_acceptance_sha256"] != proofs["parent_acceptance"]["sha256"]
                or old.get("passed") is not True or old.get("scope") != "full"
                or not isinstance(old.get("binding"), dict)
                or any(old["binding"].get(key) != value for key, value in build.get("parent_archive_binding", {}).items())):
            raise ValueError("Final-FK revision inherited parent acceptance differs")
    if (quality.get("format") != AUDIT_FORMAT or quality.get("state") != "complete"
            or quality.get("production_modified") is not False or not quality.get("finished_at")
            or build.get("quality_report_sha256") != proofs["quality_audit"]["sha256"]
            or independent.get("format") != INDEPENDENT_AUDIT_FORMAT
            or independent.get("passed") is not True or independent.get("scope") != "full"
            or independent.get("failures") != [] or independent.get("binding") != archive.binding
            or independent.get("population") != population
            or any(independent.get(name) is not True for name in ACCEPTANCE_FLAGS)
            or independent.get("quality_audit_sha256") != proofs["quality_audit"]["sha256"]
            or independent.get("revision_manifest_sha256") != proofs["revision_manifest"]["sha256"]
            or independent.get("new_head_specific_physical_gate", {}).get("scope") != "all_eligible_archive"
            or independent.get("new_head_specific_physical_gate", {}).get("episodes_checked") != len(required)
            or independent.get("new_head_specific_physical_gate", {}).get("episode_indices") != sorted(required)):
        raise ValueError("Final-FK revision independent evidence differs")
