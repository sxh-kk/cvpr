"""Owned independent gate for camera-only immutable archive revisions."""
from __future__ import annotations

from .revision_acceptance import _proof
from .window_cache import digest

ACCEPTANCE_FORMAT = "independent-smplh-camera-quality-revision-acceptance-v1"
REVISION_FORMAT = "smplh-camera-quality-revision-v1"
AUDIT_FORMAT = "smplh-camera-quality-audit-v1"
INDEPENDENT_AUDIT_FORMAT = "independent-smplh-camera-quality-revision-delta-audit-v1"
ACCEPTANCE_FLAGS = ("parent_acceptance_inherited", "parameter_values_preserved", "world_observations_preserved",
                    "camera_values_preserved", "image_join_values_preserved", "camera_masks_accepted",
                    "native_continuity_accepted", "ground_values_preserved", "latest_motionexpert_accepted")


def check_camera_revision_acceptance(archive, report, report_path):
    if (report.get("format") != ACCEPTANCE_FORMAT or report.get("scope") != "full"
            or report.get("passed") is not True or report.get("failures") != []
            or any(report.get(name) is not True for name in ACCEPTANCE_FLAGS)
            or report.get("binding") != archive.binding):
        raise ValueError("Camera quality revision independent acceptance failed or is partial")
    population = report.get("population", {})
    if population.get("episodes") != len(archive._records) or population.get("frames") != archive.frames:
        raise ValueError("Camera quality revision acceptance population differs")
    proofs = report.get("proofs", {})
    names = ("revision_manifest", "quality_audit", "independent_revision_audit", "parent_publication", "parent_acceptance")
    revision, quality, independent, parent, old = [_proof(archive.root, proofs.get(name), name) for name in names]
    build = revision.get("build_binding", {})
    required = list(range(len(archive._records)))
    receipts = quality.get("episodes", [])
    indices = [r.get("episode_index") for r in receipts]
    source_counts = {source: sum(record["source"] == source for record in archive._records)
                     for source in {record["source"] for record in archive._records}}
    if quality.get("source_population", {}).get("sources") != source_counts:
        raise ValueError("Camera source population policy differs")
    for receipt in receipts:
        index = receipt.get("episode_index")
        if type(index) is not int or not 0 <= index < len(archive._records):
            raise ValueError("Camera receipt identity differs")
        record = archive._records[index]
        source_policy = receipt.get("source_policy", {})
        if (receipt.get("source") != record["source"] or receipt.get("source_key") != record["source_key"]
                or receipt.get("length") != record["length"] or receipt.get("policy") != quality.get("policy")
                or source_policy.get("confidence_observed") is not (record["source"] == "egoexo4d_hm")
                or (record["source"] == "egoexo4d_hm" and source_policy.get("open_loop_used_as_pose_or_world") is not False)
                or (record["source"] == "egosuite" and source_policy.get("camera_tracking_score_certified") is not False)):
            raise ValueError("Camera receipt source confidence/world policy differs")
    if (revision.get("format") != REVISION_FORMAT or revision.get("scope") != "full"
            or revision.get("binding") != archive.binding or revision.get("archive_fingerprint") != archive.archive_fingerprint
            or revision.get("build_binding_sha256") != digest(build) or build.get("scope") != "full"
            or revision.get("quality_episodes") != required or revision.get("required_quality_episodes") != required
            or sorted(indices) != required or len(indices) != len(set(indices))
            or quality.get("episodes_completed") != len(required) or quality.get("episodes_planned") != len(required)
            or quality.get("format") != AUDIT_FORMAT or quality.get("state") != "complete"
            or quality.get("diagnostic") is not False or quality.get("production_modified") is not False
            or quality.get("source_population", {}).get("scope") != "all_archive"
            or quality.get("source_population", {}).get("total") != len(required)
            or build.get("quality_report_sha256") != proofs["quality_audit"]["sha256"]
            or build.get("policy") != quality.get("policy")
            or archive.info["quality_revision"].get("policy") != quality.get("policy")):
        raise ValueError("Camera quality proof must cover all native source episodes including quarantines")
    if (parent.get("state") != "accepted" or parent.get("independent_acceptance_pending") is not False
            or parent.get("episodes") != len(required) or parent.get("frames") != archive.frames
            or build.get("parent_publication_sha256") != proofs["parent_publication"]["sha256"]
            or build.get("parent_acceptance_sha256") != proofs["parent_acceptance"]["sha256"]
            or parent.get("acceptance_report_sha256") != proofs["parent_acceptance"]["sha256"]
            or old.get("passed") is not True or old.get("scope") != "full"
            or old.get("binding") != build.get("parent_archive_binding")):
        raise ValueError("Camera revision inherited parent acceptance identity differs")
    if (independent.get("format") != INDEPENDENT_AUDIT_FORMAT or independent.get("passed") is not True
            or independent.get("scope") != "full" or independent.get("failures") != []
            or independent.get("binding") != archive.binding or independent.get("population") != population
            or independent.get("quality_audit_sha256") != proofs["quality_audit"]["sha256"]
            or independent.get("revision_manifest_sha256") != proofs["revision_manifest"]["sha256"]
            or any(independent.get(name) is not True for name in ACCEPTANCE_FLAGS)
            or independent.get("camera_native_population", {}).get("episode_indices") != required):
        raise ValueError("Camera independent native policy/mask evidence differs")
