"""Frame-joined SMPL-H derivatives with the original v5 camera and vision.

This reader explicitly opts into fitted labels. It does not change v5 or the
MotionExpert source-native reader. Missing-ground fallback uses native feet,
and forecasting limits that fallback to the observed prefix.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .ground import resolve_window_ground
from .smplh_jobs import read_source_motion
from .smplh_sidecar import read_sidecar_window, validate_episode_sidecar
from .storage import Archive
from .unified_dense22 import unified_metadata
from .windows import prepare_window


def native_window_ground(arrays, metadata, selected, *, task="recon", forecast_prefix=30):
    if task not in ("recon", "fore", "gen"):
        raise ValueError("Unknown motion task")
    if task == "fore":
        if isinstance(forecast_prefix, bool) or not isinstance(forecast_prefix, int) or not 1 <= forecast_prefix < len(selected):
            raise ValueError("Forecast prefix must be inside the selected window")
        ground_selected = selected[:forecast_prefix]
    else:
        ground_selected = selected
    ground = resolve_window_ground(
        np.asarray(arrays["observation.state.body_world"])[ground_selected].reshape(-1, 22, 3),
        np.asarray(arrays["observation.valid.body_position"])[ground_selected],
        metadata["ground"], metadata.get("coordinate_verified", False))
    ground["fallback_body_source"] = "native_source_feet"
    ground["fallback_selected_frames"] = len(ground_selected)
    # prepare_window resolves ground again. A failed native/prefix resolution
    # must not get a second chance using fitted or future feet.
    ground["native_window_fallback_policy"] = deepcopy(ground.get("window_fallback", {"mode":"mean_valid_feet", "offset_m":.02}))
    ground["window_fallback"] = {"mode":"disabled_after_native_resolution"}
    return ground


def _vision_rows(path, selected):
    parquet = pq.ParquetFile(path)
    groups, begin = [], 0
    for i in range(parquet.metadata.num_row_groups):
        end = begin + parquet.metadata.row_group(i).num_rows
        if np.any((selected >= begin) & (selected < end)):
            table = parquet.read_row_group(i, columns=["frame_index", "observation.features.dino"])
            frame = table["frame_index"].combine_chunks().to_numpy()
            token = table["observation.features.dino"].combine_chunks().values.to_numpy().reshape(len(table), 5120)
            take = np.isin(frame, selected)
            groups.append((frame[take], token[take].copy()))
        begin = end
    frames = np.concatenate([x[0] for x in groups])
    values = np.concatenate([x[1] for x in groups])
    if not np.array_equal(frames, selected):
        raise ValueError("Visual feature frame join differs from motion")
    if values.dtype != np.float32:
        raise ValueError("Unexpected visual feature storage dtype")
    return values


class SMPLHDerivedArchive:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root/"manifest.json").read_text())
        self.source = Archive(self.manifest["original_archive_root"])
        if self.manifest.get("source_archive_modified") is not False or self.manifest.get("join") != ["episode_index", "frame_index"]:
            raise ValueError("Unsupported derived archive join contract")
        self._source_records = {r["episode_index"]: r for r in self.source.records()}

    def records(self):
        indices = sorted(int(p.parent.name) for p in (self.root/"episodes").glob("*/receipt.json"))
        return [deepcopy(self._source_records[i]) for i in indices]

    def read_window(self, episode_index, *, start_frame=0, num_frames=120, fps=15,
                    canonicalize=True, task="recon", forecast_prefix=30, load_dino=True):
        if fps != 15 or isinstance(start_frame, bool) or not isinstance(start_frame, (int, np.integer)) or start_frame < 0 or start_frame % 2:
            raise ValueError("Training windows use 15Hz and an even 30Hz starting frame")
        if isinstance(num_frames, bool) or not isinstance(num_frames, (int, np.integer)) or num_frames < 2:
            raise ValueError("A training window requires at least two frames")
        record = self._source_records[episode_index]
        path = self.source.data_path(episode_index)
        arrays, _ = read_source_motion(path)
        selected = start_frame + np.arange(num_frames)*2
        if selected[-1] >= record["length"]:
            raise ValueError("Window exceeds the original episode")
        receipt = json.loads((self.root/"episodes"/f"{episode_index:08d}"/"receipt.json").read_text())
        validate_episode_sidecar(self.root, record, arrays, receipt["declaration"]["contract"])
        fitted = read_sidecar_window(self.root, episode_index, start_frame=0,
                                    num_frames=record["length"], fps=30)
        metadata = deepcopy(record["metadata"])
        metadata["ground"] = native_window_ground(arrays, metadata, selected,
            task=task, forecast_prefix=forecast_prefix)
        count = record["length"]
        pvalid = fitted["estimated.valid.body_world_position"]
        rvalid = fitted["estimated.valid.body_world_rotation"]
        arrays["observation.state.body_world"] = fitted["fitted.body_world_rot6d_xyz"].reshape(count, 22, 9)[..., 6:].reshape(count, 66)
        arrays["observation.state.body_world_rotation"] = fitted["fitted.body_world_rotation"]
        arrays["observation.valid.body_position"] = pvalid
        arrays["observation.valid.body_rotation"] = rvalid
        inferred = fitted["estimated.inferred.body_position"]
        rotation_inferred = rvalid & ~fitted["source.observed.body_rotation"]
        for component, completion, availability in (("position", inferred, pvalid),
                                                    ("rotation", rotation_inferred, rvalid)):
            original = arrays[f"observation.interpolated.body_{component}"].copy()
            arrays[f"observation.source_interpolated.body_{component}"] = original
            arrays[f"observation.estimated.completed.body_{component}"] = completion
            arrays[f"observation.interpolated.body_{component}"] = (original | completion) & availability
        for component in ("position", "rotation"):
            arrays[f"observation.quality.body_{component}"] = fitted[f"quality.{component}_confidence"]
        arrays["observation.source_observed.body_position"] = fitted["source.observed.body_position"]
        arrays["observation.source_observed.body_rotation"] = fitted["source.observed.body_rotation"]
        metadata["source_body_representation"] = {
            key: deepcopy(metadata.get(key)) for key in (
                "target_joint_basis", "rotation_semantics", "rotation_basis_verified",
                "body_basis_correction", "unified_dense22")}
        metadata.update(target_joint_basis="smplh_body22_axes_v1", body_position_source="model_fitted_estimate",
            rotation_semantics="model_fitted_global_wxyz", label_kind="model_fitted_estimate",
            rotation_basis_verified=False, source_native_axes_certified=False,
            target_model_axes_defined=True, sidecar_contract_sha256=receipt["contract_sha256"])
        metadata["unified_dense22"] = unified_metadata(metadata)
        metadata["unified_dense22"].update(label_kind="model_fitted_estimate",
            native_axis_certified=False, target_model_axes_defined=True,
            absolute_rotation_accuracy_established=False)
        window = prepare_window(arrays, metadata, start_frame=start_frame,
            num_frames=num_frames, fps=fps, canonicalize=canonicalize)
        if load_dino:
            window.arrays["observation.features.dino"] = _vision_rows(path, selected)
        window.metadata["training_window"].update(task=task, forecast_prefix=forecast_prefix if task=="fore" else None,
            derived_body="smplh_body22", source_vision_frame_join_preserved=True,
            ground_fallback_body="native_source_feet")
        return window
