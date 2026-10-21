"""Local, mesh-free compatibility for the official SIMPLE CarryBox primitive.

Geometry and MuJoCo dynamics follow SIMPLE commit 6d10628794d9c7de4596b4f2afb2c054a637c2bc.
Only the render USD is generated, outside the checkout; no upstream assets change.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import numpy as np

from simple.core.asset import Asset


class CarryBoxAsset(Asset):
    """A 0.1 kg movable cardboard box, independent of the old static Box type."""

    def __init__(self, size=(0.23, 0.38, 0.434)):
        size = np.asarray(size, dtype=float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError("CarryBox size must contain three positive finite extents")
        self.uid = self.label = "cube"
        self.name = "box"
        self.description = "a cube-shaped box"
        self.size = size.tolist()
        self.stable_poses = [[0.0, 0.0, float(size[2] / 2), 1.0, 0.0, 0.0, 0.0]]
        self.keypoints, self.axes = {}, {}
        self.canonical_grasps, self.functional_grasps = [], {}
        self.usd_path = str(self._render_asset())
        # MuJoCo uses an analytic box, not an OBJ approximation.
        self.collision_mesh_curobo = None
        self.collision_meshes_mujoco = []

    def _render_asset(self) -> Path:
        half = np.asarray(self.size) / 2
        points = [tuple((half * sign).tolist()) for sign in
                  [(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                   (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]]
        point_text = ", ".join("(%s)" % ", ".join(format(v, ".12g") for v in p) for p in points)
        source = '''#usda 1.0
(
    defaultPrim = "CarryBox"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "CarryBox" {
    def Xform "Meshes" {
        def Mesh "collision" {
            point3f[] points = [%s]
            int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
            int[] faceVertexIndices = [0,3,2,1, 4,5,6,7, 0,1,5,4, 1,2,6,5, 2,3,7,6, 3,0,4,7]
            uniform token subdivisionScheme = "none"
            color3f[] primvars:displayColor = [(0.76, 0.6, 0.42)]
        }
    }
}
''' % point_text
        runtime = Path(os.environ.get("CVPR_SIMPLE_RUNTIME", str(Path.home() / "CVPR_Evaluation_runtime")))
        if not runtime.is_absolute():
            raise ValueError("CVPR_SIMPLE_RUNTIME must be an absolute local path")
        digest = hashlib.sha256(source.encode()).hexdigest()
        destination = runtime / "resources/generated/carry_box_v1" / f"{digest}.usda"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if destination.read_text() != source:
                raise RuntimeError(f"Generated CarryBox asset was modified: {destination}")
            return destination
        # Another process may construct the same deterministic asset concurrently.
        with tempfile.NamedTemporaryFile("w", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(source)
        try:
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination

    def build_mujoco(self, world, pose):
        import mujoco

        body = world.add_body(name=self.label, pos=pose.position, quat=pose.quaternion)
        body.add_geom(
            name="cube_box", type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(np.asarray(self.size) / 2).tolist(), condim=4, mass=0.1,
            friction=[0.8, 0.05, 0.005], rgba=[0.76, 0.6, 0.42, 1], solref=[0.005, 2],
        )
        body.add_freejoint(name="cube_joint")

    def to_dict(self):
        return {"res_id": "primitive", "uid": self.uid, "label": self.label,
                "name": self.name, "description": self.description, "size": self.size}
