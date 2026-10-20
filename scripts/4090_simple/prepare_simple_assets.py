"""Audit SIMPLE USD dependencies and make an immutable-source local overlay.

Only USD asset paths are changed. Geometry, material parameters, textures and
the downloaded source files are preserved. This is a CPU USD check, not an
Isaac rendering or offline-MDL acceptance test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def import_usd():
    """Use the installed Isaac USD libraries without starting SimulationApp."""
    try:
        from pxr import Sdf, Tf, Usd, UsdUtils
        return Sdf, Tf, Usd, UsdUtils
    except ImportError:
        if os.environ.get("CVPR_SIMPLE_USD_BOOTSTRAPPED"):
            raise
        cache = Path(sys.prefix) / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages/isaacsim/extscache"
        candidates = sorted(cache.glob("omni.usd.libs-*"))
        candidates = [path for path in candidates if (path / "pxr/Sdf/__init__.py").is_file()]
        if len(candidates) != 1:
            raise RuntimeError("Run with the simulation Conda Python containing Isaac 4.5 USD libraries")
        environment = dict(os.environ, CVPR_SIMPLE_USD_BOOTSTRAPPED="1")
        for key, value in (("PYTHONPATH", str(candidates[0])),
                           ("LD_LIBRARY_PATH", str(candidates[0] / "bin"))):
            environment[key] = value + (os.pathsep + environment[key] if environment.get(key) else "")
        os.execve(sys.executable, [sys.executable, *sys.argv], environment)


def resolve_scene_asset(asset: str, source_layer: Path, scene_root: Path):
    """Resolve existing files and the known upstream misplaced props prefix.

    A unique case-only filename match is permitted, recorded, and still points
    at the original material file. Never synthesize absent textures/materials.
    """
    if not asset or asset == "gltf/pbr.mdl":
        return None, "runtime_mdl_module" if asset else "empty"
    if "://" in asset:
        raise FileNotFoundError(f"Remote dependency requires separate local preparation: {asset}")
    direct = Path(asset) if Path(asset).is_absolute() else source_layer.parent / asset
    if direct.is_file():
        return direct.resolve(), "existing"
    normalized = asset.replace("\\", "/")
    if "/props/" in normalized:
        suffix = normalized.rsplit("/props/", 1)[1]
        if "/" in suffix or suffix in {"", ".", ".."}:
            raise ValueError(f"Unexpected props reference: {asset}")
        target = scene_root / "props" / suffix
        if target.is_file():
            return target.resolve(), "repair_misanchored_props"
        matches = [path for path in target.parent.iterdir()
                   if path.is_file() and path.name.casefold() == target.name.casefold()]
        if len(matches) == 1:
            return matches[0].resolve(), "repair_props_case_unique_match"
    raise FileNotFoundError(f"Missing dependency in {source_layer}: {asset}")


def prepare(source_root: Path, output_root: Path, scene_id: str):
    source_root = source_root.resolve(strict=True)
    output_root = output_root.absolute()
    repo = Path(__file__).resolve().parents[2]
    if (output_root.exists() or output_root.is_symlink() or
            output_root.resolve().is_relative_to(repo) or
            output_root.resolve().is_relative_to(source_root)):
        raise ValueError("Use a new output root outside both the repository and original resources")
    if not scene_id or Path(scene_id).name != scene_id:
        raise ValueError("scene_id must be one directory name")
    source_scene = source_root / "scenes/hssd" / scene_id
    if not source_scene.is_dir():
        raise FileNotFoundError(source_scene)
    Sdf, Tf, Usd, UsdUtils = import_usd()
    layers = sorted(source_scene.rglob("*.usd"))
    source_hashes = {str(path.relative_to(source_root)): sha256(path) for path in layers}
    # Preflight every asset path before publishing any overlay.
    preflight = []
    for source_layer in layers:
        layer = Sdf.Layer.FindOrOpen(str(source_layer))
        if layer is None:
            raise ValueError(f"Unreadable USD layer: {source_layer}")
        sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(source_layer))
        for asset in sorted(set(sublayers + references + payloads)):
            target, reason = resolve_scene_asset(asset, source_layer, source_scene)
            preflight.append({"layer": str(source_layer.relative_to(source_root)),
                              "asset": asset, "reason": reason,
                              "resolved_source": str(target) if target else None})
    output_root.mkdir(parents=True)
    destination_scene = output_root / "scenes/hssd" / scene_id
    # Symlink untouched data; copy and rewrite each scene USD into this overlay.
    for item in source_root.iterdir():
        if item.name == "scenes":
            continue
        (output_root / item.name).symlink_to(item, target_is_directory=item.is_dir())
    for category in (source_root / "scenes").iterdir():
        destination_category = output_root / "scenes" / category.name
        if category.name != "hssd":
            destination_category.parent.mkdir(exist_ok=True)
            destination_category.symlink_to(category, target_is_directory=category.is_dir())
            continue
        destination_category.mkdir(parents=True)
        for scene in category.iterdir():
            if scene.name != scene_id:
                (destination_category / scene.name).symlink_to(scene, target_is_directory=scene.is_dir())
    for item in sorted(source_scene.rglob("*")):
        destination = destination_scene / item.relative_to(source_scene)
        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif item.suffix.lower() != ".usd":
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(item)
    rewritten = []
    for source_layer in layers:
        destination = output_root / source_layer.relative_to(source_root)
        original = Sdf.Layer.FindOrOpen(str(source_layer))
        derived = Sdf.Layer.CreateNew(str(destination))
        derived.TransferContent(original)

        def remap(asset):
            target, _ = resolve_scene_asset(asset, source_layer, source_scene)
            if target is None:
                return asset
            local_target = (destination_scene / target.relative_to(source_scene)
                            if target.is_relative_to(source_scene) else target)
            return os.path.relpath(local_target, destination.parent)

        UsdUtils.ModifyAssetPaths(derived, remap)
        derived.Save()
        rewritten.append({"source": str(source_layer), "source_sha256": sha256(source_layer),
                          "derived": str(destination), "derived_sha256": sha256(destination)})
    audit = []
    for record in rewritten:
        destination = Path(record["derived"])
        sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(destination))
        for asset in sorted(set(sublayers + references + payloads)):
            if asset == "gltf/pbr.mdl":
                audit.append({"layer": str(destination), "asset": asset, "status": "isaac_runtime_mdl_module_unverified"})
                continue
            target = destination.parent / asset
            if not target.is_file():
                raise FileNotFoundError(f"Derived dependency unresolved: {destination}: {asset}")
            audit.append({"layer": str(destination), "asset": asset, "status": "local_file_exists"})
    for relative, digest in source_hashes.items():
        if sha256(source_root / relative) != digest:
            raise RuntimeError(f"Original USD changed during preparation: {relative}")
    # Check the fully composed scene on CPU as well as individual file paths.
    # This does not compile MDL shaders or verify their runtime module search.
    entry_point = destination_scene / f"{scene_id}.usd"
    error_mark = Tf.Error.Mark()
    error_mark.SetMark()
    stage = Usd.Stage.Open(str(entry_point))
    if stage is None:
        raise ValueError(f"Derived scene cannot be composed: {entry_point}")
    prims = list(stage.Traverse())
    composition_errors = [str(error) for prim in prims for error in prim.GetPrimIndex().localErrors]
    tf_errors = [str(error) for error in error_mark.GetErrors()]
    if composition_errors or tf_errors:
        raise ValueError(f"Derived USD composition errors: {composition_errors + tf_errors}")
    result = {"schema_version": 1, "source_root": str(source_root), "derived_root": str(output_root),
              "scene_id": scene_id, "source_unchanged": True,
              "scope": "CPU USD path/asset existence audit; no shader compilation, GPU rendering or task acceptance",
              "preflight": preflight, "rewritten_usd": rewritten, "derived_dependencies": audit,
              "local_file_dependency_audit_passed": True,
              "cpu_composition": {"prim_count": len(prims), "errors": composition_errors + tf_errors},
              "runtime_mdl_modules": ["gltf/pbr.mdl"],
              "runtime_mdl_resolution_verified": False}
    (output_root / "DERIVATION_MANIFEST.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--scene-id", default="107734119_175999932")
    args = parser.parse_args()
    result = prepare(args.source_root, args.output_root, args.scene_id)
    print(json.dumps({key: result[key] for key in (
        "derived_root", "scene_id", "source_unchanged", "local_file_dependency_audit_passed",
        "runtime_mdl_resolution_verified")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
