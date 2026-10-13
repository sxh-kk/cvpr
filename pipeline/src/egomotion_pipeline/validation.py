import numpy as np
from .storage import read_arrays, read_json
from .schema import ROTATION_SHAPES, SCHEMA_VERSION, V4_SCHEMA_VERSION, dino_format


def validate(archive, require_dino=False, strict=False, require_floor=False, require_head_reference=False,
             require_body_head_floor=False):
    info = read_json(archive.root/"meta/info.json")
    records = archive.records()
    if info["fps"] != 30 or info["features"] != archive.specs or info["codebase_version"] != "v3.0":
        raise ValueError("Archive info/schema mismatch")
    if archive.schema_version == SCHEMA_VERSION and info.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Physical v5 archive info does not declare its schema")
    if len(records) != info["total_episodes"] or not records:
        raise ValueError("Empty/inconsistent episode count")
    offset = 0
    warnings = []
    feature_signatures = set()
    for index, record in enumerate(records):
        if record["episode_index"] != index or record["dataset_from_index"] != offset:
            raise ValueError("Episode/global indices are inconsistent")
        arrays = read_arrays(archive.data_path(index), archive.specs)
        count = record["length"]
        if record["dataset_to_index"] != offset+count or any(len(x) != count for x in arrays.values()):
            raise ValueError(f"Episode {index}: length mismatch")
        for key, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"Episode {index}: non-finite {key}")
            if str(value.dtype) != archive.specs[key]["dtype"]:
                raise ValueError(f"Episode {index}: unexpected dtype for {key}")
        if not np.allclose(arrays["timestamp"], (np.arange(count)/30).astype(np.float32), atol=1e-6, rtol=0):
            raise ValueError(f"Episode {index}: timestamps are not 30 Hz")
        if not np.array_equal(arrays["frame_index"], np.arange(count)) or not np.array_equal(arrays["index"], np.arange(offset, offset+count)):
            raise ValueError("Frame index mismatch")
        if not np.all(arrays["episode_index"] == index) or not np.all(arrays["task_index"] == 0):
            raise ValueError("Episode/task index mismatch")
        for modality, shape in ROTATION_SHAPES.items():
            q = arrays[f"observation.state.{modality}_world_rotation"].reshape(count, *shape)
            if not np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-5):
                raise ValueError("Non-unit quaternion")
        for key in arrays:
            if key.startswith("observation.interpolated."):
                mask_key = key.replace("interpolated", "valid")
                if np.any(arrays[key] & ~arrays[mask_key]):
                    raise ValueError("Invalid samples are marked interpolated")
        ground = record["metadata"].get("ground", {})
        floor = arrays["observation.environment.floor_height"]
        floor_valid = arrays["observation.valid.floor"]
        if not np.allclose(floor, ground.get("height_m", 0.), atol=1e-5, rtol=1e-6) or not np.all(floor_valid == ground.get("valid", False)):
            raise ValueError("Floor fields and episode provenance differ")
        if np.any(arrays["observation.valid.floor_verified"] & ~floor_valid):
            raise ValueError("Unavailable floor marked verified")
        if not np.all(arrays["observation.valid.floor_verified"] == bool(ground.get("verified", False) and ground.get("valid", False))):
            raise ValueError("Floor verification differs from provenance")
        if not np.allclose(arrays["observation.environment.floor_normal"], [0, 0, 1], atol=1e-6):
            raise ValueError("The horizontal floor estimator requires output world Z up")
        if require_floor and not floor_valid.all():
            raise ValueError(f"{record['source_key']}: floor height unavailable")
        head_position = arrays["observation.valid.head_reference_position"].reshape(count)
        head_rotation = arrays["observation.valid.head_reference_rotation"].reshape(count)
        basis = arrays["observation.valid.head_reference_basis"]
        if require_head_reference and not (head_position & head_rotation & basis).all():
            raise ValueError(f"{record['source_key']}: head reference unavailable or camera basis unverified")
        height_valid = arrays["observation.valid.head_reference_height"]
        if not np.array_equal(height_valid, head_position & floor_valid):
            raise ValueError("Head height mask is inconsistent")
        expected_height = arrays["observation.state.head_reference_world"][:, 2]-floor
        if not np.allclose(arrays["observation.state.head_reference_height"][height_valid], expected_height[height_valid], atol=1e-5):
            raise ValueError("Head height is inconsistent with the world floor plane")
        from .geometry import quat_to_matrix
        from .head_reference import (reference_rotation, floor_reference, positions_in_floor_reference,
                                     CAMERA_BASIS_POLICIES)
        head_meta = record["metadata"].get("head_reference", {})
        if not np.all(basis == head_meta.get("basis_verified", False)):
            raise ValueError("Head reference basis verification differs from provenance")
        if head_meta.get("mode") == "ee4d":
            source_basis = record["metadata"].get("camera_pose_basis", "unspecified")
            if head_meta.get("source_camera_basis") != source_basis:
                raise ValueError("Head reference source camera basis differs from provenance")
            policy = head_meta.get("basis_policy")
            if policy != "manual":
                expected_policy, correction = CAMERA_BASIS_POLICIES.get(source_basis, ("unknown", np.eye(3)))
                if policy != expected_policy or not np.allclose(head_meta["camera_from_ee4d_camera"], correction):
                    raise ValueError("Head reference source-specific camera axis policy is inconsistent")
            if not np.array_equal(head_position, arrays["observation.valid.camera_position"].reshape(count)) or not np.array_equal(
                    head_rotation, arrays["observation.valid.camera_rotation"].reshape(count)):
                raise ValueError("Head reference masks differ from selected camera masks")
            camera_rotation = quat_to_matrix(arrays["observation.state.camera_world_rotation"])
            expected_rotation = camera_rotation @ reference_rotation(head_meta["camera_from_ee4d_camera"])
            actual_rotation = quat_to_matrix(arrays["observation.state.head_reference_world_rotation"])
            if not np.allclose(actual_rotation[head_rotation], expected_rotation[head_rotation], atol=1e-5):
                raise ValueError("Head reference does not follow the recorded EE4D axis conversion")
            if not np.allclose(arrays["observation.state.head_reference_world"][head_position], arrays["observation.state.camera_world"][head_position], atol=1e-5):
                raise ValueError("EE4D head reference must share the selected camera origin")
        elif head_meta.get("mode") == "disabled":
            if head_position.any() or head_rotation.any() or basis.any():
                raise ValueError("Disabled head reference has valid samples")
        else:
            raise ValueError("Unknown head reference mode")
        floor_pos, floor_quat, floor_mask = floor_reference(
            arrays["observation.state.head_reference_world"].reshape(count, 1, 3),
            arrays["observation.state.head_reference_world_rotation"].reshape(count, 1, 4),
            head_position[:, None], head_rotation[:, None], ground, bool(basis.all()))
        floor_mask = floor_mask[:, 0]
        for kind in ["position", "rotation"]:
            if not np.array_equal(arrays[f"observation.valid.head_reference_floor_{kind}"].reshape(count), floor_mask):
                raise ValueError("Projected head reference mask is inconsistent")
        if not np.allclose(arrays["observation.state.head_reference_floor_world"], floor_pos.reshape(count, 3), atol=1e-5):
            raise ValueError("Projected head reference position is inconsistent")
        if not np.allclose(quat_to_matrix(arrays["observation.state.head_reference_floor_world_rotation"]),
                           quat_to_matrix(floor_quat[:, 0]), atol=1e-5):
            raise ValueError("Projected head reference rotation is inconsistent")
        body_local, body_mask = positions_in_floor_reference(
            arrays["observation.state.body_world"].reshape(count, 22, 3),
            arrays["observation.valid.body_position"], floor_pos, floor_quat, floor_mask[:, None])
        if not np.array_equal(arrays["observation.valid.body_head_floor_position"], body_mask):
            raise ValueError("body_head_floor validity does not follow body/camera/floor masks")
        if not np.allclose(arrays["observation.state.body_head_floor"].reshape(count, 22, 3),
                           body_local, atol=1e-5, rtol=1e-6):
            raise ValueError("body_head_floor positions disagree with the projected camera reference")
        if np.any(arrays["observation.state.body_head_floor"].reshape(count, 22, 3)[~body_mask] != 0):
            raise ValueError("Invalid body_head_floor positions must be zero")
        reference_interpolated = (arrays["observation.interpolated.head_reference_position"].reshape(count, 1)
                                  | arrays["observation.interpolated.head_reference_rotation"].reshape(count, 1))
        for kind in ["position", "rotation"]:
            if not np.array_equal(arrays[f"observation.interpolated.head_reference_floor_{kind}"].reshape(count, 1),
                                  reference_interpolated & floor_mask[:, None]):
                raise ValueError("Projected head reference interpolation mask is inconsistent")
        expected_interpolated = (arrays["observation.interpolated.body_position"] | reference_interpolated) & body_mask
        if not np.array_equal(arrays["observation.interpolated.body_head_floor_position"], expected_interpolated):
            raise ValueError("body_head_floor interpolation mask is inconsistent")
        if require_body_head_floor and not body_mask.all():
            raise ValueError(f"{record['source_key']}: body_head_floor has unavailable joints or camera/floor reference")
        if archive.schema_version == SCHEMA_VERSION:
            from .unified_dense22 import validate_unified_dense22, unified_metadata
            from .numeric_fingerprint import numeric_hashes, motion_digest_from_hashes, NON_MOTION_COLUMNS
            validate_unified_dense22(arrays, arrays, record)
            declared = record["metadata"].get("unified_dense22")
            expected = unified_metadata(record)
            if not isinstance(declared, dict) or any(declared.get(key) != value for key, value in expected.items()):
                raise ValueError("Unified dense22 encoding/axis/cache provenance differs from source")
            representation = record.get("motion_representation", {})
            if representation.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Missing physical v5 motion representation provenance")
            signature = representation.get("producer_signature", "")
            if (not isinstance(signature, str) or len(signature) != 64
                    or any(char not in "0123456789abcdef" for char in signature)):
                raise ValueError("Invalid dense22 producer SHA256 signature")
            motion_arrays = {key: value for key, value in arrays.items() if key not in NON_MOTION_COLUMNS}
            if representation.get("motion_digest") != motion_digest_from_hashes(numeric_hashes(motion_arrays)):
                raise ValueError("Physical v5 motion digest differs from stored columns")
        for requirement in ["coordinate_verified", "rotation_basis_verified"]:
            if not record["metadata"].get(requirement, False):
                warnings.append(f"{record['source_key']}: {requirement}=false")
        if not arrays["observation.valid.camera_position"].any():
            warnings.append(f"{record['source_key']}: camera trajectory unavailable")
        ready = record["dino"]["state"] == "complete"
        dino_valid = arrays["observation.valid.dino"]
        if record["dino"]["state"] not in {"pending", "complete"}:
            raise ValueError("Unknown DINO extraction state")
        if not ready and dino_valid.any():
            raise ValueError("Pending DINO has valid frames")
        if np.any(arrays["observation.features.dino"][~dino_valid] != 0):
            raise ValueError("Invalid DINO features must be zero")
        if require_dino and (not ready or not arrays["observation.valid.dino"].all()):
            raise ValueError(f"{record['source_key']}: DINO is pending or has missing frames")
        if ready:
            signature = record["dino"]
            if archive.schema_version in {V4_SCHEMA_VERSION, SCHEMA_VERSION}:
                expected = dino_format()
                for key in ["shape", "flat_shape", "token_order", "flatten_order", "pooling", "storage_dtype"]:
                    if signature["encoder"].get(key) != expected[key]:
                        raise ValueError(f"DINO token format mismatch: {key}")
            if signature.get("valid_count") != int(dino_valid.sum()):
                raise ValueError("DINO valid_count differs from masks")
            from .storage import digest
            feature_signatures.add(digest(signature["encoder"]))
            mask = arrays["observation.valid.dino"]
            if np.any(np.abs(arrays["observation.video_time_error"][mask]) > signature["max_video_error_s"]+1e-6):
                raise ValueError("Video timestamp error exceeds the recorded tolerance")
        offset += count
    if offset != info["total_frames"]:
        raise ValueError("Global total_frames mismatch")
    if len(feature_signatures) > 1:
        raise ValueError("Archive mixes different DINO checkpoints/processors")
    if strict and warnings:
        raise ValueError("Strict validation failed:\n"+"\n".join(warnings))
    return {"episodes": len(records), "frames": offset, "fps": 30, "warnings": warnings,
            "hours": offset/30/3600}
