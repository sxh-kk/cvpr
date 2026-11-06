"""Shared collision-piece discovery without changing asset traversal order."""
import os


def _collision_meshes_mujoco(directory: str) -> list[str]:
    """Collect convex pieces, retaining the original os.walk/files order.

    Renderers may create texture subdirectories beside the pieces. Such a
    directory must not replace the pieces already discovered in its parent.
    Empty assets fail here, before a free body with no mass reaches MuJoCo.
    """
    meshes = []
    for current_directory, _, files in os.walk(directory):
        meshes.extend(os.path.join(current_directory, filename)
                      for filename in files if filename.startswith("convex_piece"))
    if not meshes:
        raise ValueError(f"No convex_piece collision meshes found in {directory}")
    return meshes
