"""Collision pieces survive renderer-created texture directories."""
from contextlib import ExitStack
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "VLMExpert/third_party/SIMPLE/src"))
from simple.assets.collision_meshes import _collision_meshes_mujoco
from simple.assets import graspnet, objects
from simple.assets.asset_manager import AssetManager


class CollisionMeshDiscoveryTests(unittest.TestCase):
    def test_walk_and_file_order_are_preserved_without_sorting(self):
        walk = [("/asset", ["z", "textures", "a"],
                 ["convex_piece_9.obj", "material.mtl", "convex_piece_2.obj"]),
                ("/asset/z", [], ["convex_piece_7.obj"]),
                ("/asset/textures", [], ["CustomMaterial.png"]),
                ("/asset/a", [], ["convex_piece_1.obj"])]
        with patch("simple.assets.collision_meshes.os.walk", return_value=walk):
            self.assertEqual(_collision_meshes_mujoco("/asset"), [
                "/asset/convex_piece_9.obj", "/asset/convex_piece_2.obj",
                "/asset/z/convex_piece_7.obj", "/asset/a/convex_piece_1.obj"])

    def test_root_23_pieces_survive_nested_textures(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for index in (9, 2, 18, 1, 20, 3, 16, 4, 15, 5, 14, 6, 13, 7, 12,
                          8, 11, 0, 10, 17, 19, 21, 22):
                (root / f"convex_piece_{index}.obj").touch()
            expected = [str(root / filename) for filename in next(os.walk(root))[2]
                        if filename.startswith("convex_piece")]
            (root / "textures").mkdir()
            (root / "textures/CustomMaterial.png").touch()
            self.assertEqual(len(expected), 23)
            self.assertEqual(_collision_meshes_mujoco(str(root)), expected)

    def test_empty_and_missing_collision_asset_fail_clearly(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "textures").mkdir()
            (root / "textures/CustomMaterial.png").touch()
            for path in (root, root / "missing"):
                with self.subTest(path=path), self.assertRaisesRegex(
                        ValueError, "No convex_piece collision meshes found"):
                    _collision_meshes_mujoco(str(path))

    def test_single_and_bulk_managers_keep_all_pieces(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for index in range(23):
                (root / f"convex_piece_{index}.obj").touch()
            (root / "textures").mkdir()
            (root / "textures/CustomMaterial.png").touch()
            expected = _collision_meshes_mujoco(str(root))
            for module, manager_type, bulk_method, names in (
                    (objects, AssetManager._registry["objects"], "_load_objects_assets", "Objects_Names"),
                    (graspnet, AssetManager._registry["graspnet1b"],
                     "_load_graspnet_object_assets", "GraspNet_1B_Object_Names")):
                with self.subTest(manager=manager_type.__name__), ExitStack() as stack:
                    stack.enter_context(patch.object(module, "resolve_data_path",
                        side_effect=lambda path, **kwargs: str(root) if "coacd_0.05" in path else path))
                    stack.enter_context(patch.object(module, names, {0: "object"}))
                    stack.enter_context(patch.object(module.np, "load", return_value=np.empty((0, 7))))
                    manager = manager_type()
                    self.assertEqual(manager.load("0").collision_meshes_mujoco, expected)
                    self.assertEqual(getattr(manager, bulk_method)()[0]["collision_meshes_mujoco"], expected)

    def test_managers_reject_missing_collision_geometry(self):
        with tempfile.TemporaryDirectory() as name:
            for module, manager_type, bulk_method, names in (
                    (objects, AssetManager._registry["objects"], "_load_objects_assets", "Objects_Names"),
                    (graspnet, AssetManager._registry["graspnet1b"],
                     "_load_graspnet_object_assets", "GraspNet_1B_Object_Names")):
                with self.subTest(manager=manager_type.__name__), ExitStack() as stack:
                    stack.enter_context(patch.object(module, "resolve_data_path", return_value=name))
                    stack.enter_context(patch.object(module, names, {0: "object"}))
                    manager = manager_type()
                    for operation in (lambda: manager.load("0"), getattr(manager, bulk_method)):
                        with self.assertRaisesRegex(ValueError, "No convex_piece collision meshes found"):
                            operation()

    def test_real_basket_retains_original_23_root_pieces(self):
        runtime = Path(os.environ.get("CVPR_SIMPLE_RUNTIME", "/home/zxc/CVPR_Evaluation_runtime"))
        root = runtime / "resources/simple/assets/objects/models/000/coacd_0.05"
        if not root.is_dir():
            self.skipTest("Real SIMPLE basket resource is unavailable")
        expected = [str(root / filename) for filename in next(os.walk(root))[2]
                    if filename.startswith("convex_piece")]
        self.assertEqual(len(expected), 23)
        self.assertEqual(_collision_meshes_mujoco(str(root)), expected)


if __name__ == "__main__":
    unittest.main()
