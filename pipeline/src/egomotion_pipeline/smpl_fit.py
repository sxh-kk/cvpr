"""Small-scale, joints-only SMPL-family fitting research.

This module is deliberately separate from the production converter.  It fits
estimated body22 labels; it does not authenticate EgoSuite's native joint axes.
All matrices act on column vectors.  ``root_world`` is the absolute pelvis
position, whereas the corresponding SMPL translation is ``root_world - J0``.
The world increment G[t+1] @ G[t].T is invariant to a constant unknown right
joint-basis correction.  Absolute twist is still selected by initialization
and regularization, not measured by this objective.

Only this research module imports torch.  No skinning, mesh, camera pose,
contact heuristic, or inferred native joint-basis certificate is produced.
"""

from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import time

import numpy as np
import torch

from .schema import BODY_NAMES, PARENTS


@dataclass(frozen=True)
class FitConfig:
    shape_iterations: int = 100
    pose_iterations: int = 250
    shape_lr: float = .03
    pose_lr: float = .025
    root_lr: float = .01
    lr_final_ratio: float = .05
    shape_regularization: float = 1e-5
    pose_increment_regularization: float = 1e-6
    local_rest_regularization: float = 1e-7
    temporal_regularization: float = 1e-5
    rotation_increment_weight: float = .02
    position_huber_delta: float = .03
    bone_huber_delta: float = .015
    max_dt: float = .1
    beta_bound: float = 5.
    grad_clip: float = 10.
    history_every: int = 10

    def validate(self):
        for name in ("shape_iterations", "pose_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (isinstance(self.history_every, bool)
                or not isinstance(self.history_every, int) or self.history_every < 1):
            raise ValueError("history_every must be a positive integer")
        nonnegative = ("shape_regularization", "pose_increment_regularization",
                       "local_rest_regularization", "temporal_regularization",
                       "rotation_increment_weight")
        for name, value in asdict(self).items():
            if name in ("shape_iterations", "pose_iterations", "history_every"):
                continue
            if not np.isfinite(value) or value < 0 or (value == 0 and name not in nonnegative):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name in nonnegative else 'positive'}")
        if self.lr_final_ratio > 1:
            raise ValueError("lr_final_ratio must be at most one")
        return self


@dataclass
class FitResult:
    positions: np.ndarray
    global_rotations: np.ndarray
    local_rotations: np.ndarray
    root_world: np.ndarray
    smpl_trans: np.ndarray
    betas: np.ndarray
    initialization: dict
    history: list
    metadata: dict


def so3_exp(axis_angle):
    """Differentiable Rodrigues exponential, including a finite gradient at zero."""
    if not isinstance(axis_angle, torch.Tensor) or axis_angle.shape[-1:] != (3,):
        raise ValueError("axis_angle must be a torch tensor with last dimension 3")
    if not axis_angle.is_floating_point():
        raise ValueError("axis_angle must be floating point")
    theta2 = axis_angle.square().sum(-1, keepdim=True)
    # Clamp the inactive trigonometric branch too: 0 * NaN would otherwise
    # contaminate the torch.where backward pass at the zero initialization.
    theta = theta2.clamp_min(1e-12).sqrt()
    a_series = 1 - theta2 / 6 + theta2.square() / 120 - theta2.pow(3) / 5040
    b_series = .5 - theta2 / 24 + theta2.square() / 720 - theta2.pow(3) / 40320
    a = torch.where(theta2 < 1e-6, a_series, torch.sin(theta) / theta)
    b = torch.where(theta2 < 1e-6, b_series, (1 - torch.cos(theta)) / theta2.clamp_min(1e-12))
    x, y, z = axis_angle.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(axis_angle.shape[:-1] + (3, 3))
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device)
    return eye + a[..., None] * skew + b[..., None] * (skew @ skew)


class DifferentiableBody22(torch.nn.Module):
    """The first 22 kinematic joints of a known SMPL-family rest model.

    Constructor accepts already-regressed J(0) and its linear shape basis.
    ``forward`` returns (world positions, global joint rotations); each child
    offset is rotated by its parent's global orientation, as in SMPL FK.
    """

    def __init__(self, rest_joints, shape_basis, *, metadata=None,
                 device="cpu", dtype=torch.float64):
        super().__init__()
        rest = np.asarray(rest_joints, dtype=np.float64)
        basis = np.asarray(shape_basis, dtype=np.float64)
        if (rest.shape != (22, 3) or basis.ndim != 3
                or basis.shape[:2] != (22, 3) or basis.shape[-1] < 1):
            raise ValueError("rest_joints must be (22,3), shape_basis (22,3,K) with K >= 1")
        if not np.isfinite(rest).all() or not np.isfinite(basis).all():
            raise ValueError("model rest joints and shape basis must be finite")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("model dtype must be torch.float32 or torch.float64")
        self.register_buffer("rest_zero", torch.as_tensor(rest.copy(), dtype=dtype, device=device))
        self.register_buffer("shape_basis", torch.as_tensor(basis.copy(), dtype=dtype, device=device))
        self.metadata = dict(metadata or {})
        self.metadata.update({"decoder": "differentiable_smpl_family_body22_fk",
                              "shape_components_used": basis.shape[-1],
                              "joint_names": list(BODY_NAMES),
                              "parents": PARENTS.tolist(),
                              "root_position_convention": "absolute_world_pelvis",
                              "matrix_convention": "column_vectors"})

    @property
    def device(self):
        return self.rest_zero.device

    @property
    def dtype(self):
        return self.rest_zero.dtype

    @property
    def shape_components(self):
        return self.shape_basis.shape[-1]

    def rest_joints(self, betas):
        if not isinstance(betas, torch.Tensor):
            betas = torch.as_tensor(betas, dtype=self.dtype, device=self.device)
        if betas.ndim not in (1, 2) or betas.shape[-1] != self.shape_components:
            raise ValueError("betas must have shape (K,) or (T,K), matching model")
        return self.rest_zero + torch.einsum("...b,jcb->...jc", betas, self.shape_basis)

    def forward(self, local_rotations, root_world, betas):
        if (not isinstance(local_rotations, torch.Tensor) or local_rotations.ndim != 4
                or local_rotations.shape[1:] != (22, 3, 3)):
            raise ValueError("local_rotations must be a torch tensor of shape (T,22,3,3)")
        count = len(local_rotations)
        if not isinstance(root_world, torch.Tensor) or root_world.shape != (count, 3):
            raise ValueError("root_world must be a torch tensor of shape (T,3)")
        rest = self.rest_joints(betas)
        if rest.ndim == 2:
            rest = rest.unsqueeze(0).expand(count, -1, -1)
        elif len(rest) != count:
            raise ValueError("per-frame betas must match frame count")
        global_rotations = [local_rotations[:, 0]]
        positions = [root_world]
        for child in range(1, 22):
            parent = int(PARENTS[child])
            parent_rotation = global_rotations[parent]
            global_rotations.append(parent_rotation @ local_rotations[:, child])
            offset = rest[:, child] - rest[:, parent]
            positions.append(positions[parent] + (parent_rotation @ offset[..., None])[..., 0])
        return torch.stack(positions, 1), torch.stack(global_rotations, 1)


def load_body22_model(path, *, shape_components=10, device="cpu", dtype=torch.float64):
    """Regress rest joints from one immutable, safe NPZ byte snapshot.

    Unlike the HM production decoder this loader intentionally supports
    SMPL-X, but its result must not be used to decode SMPL-H beta parameters.
    Only numerical arrays are accepted; pickle/object loading is disabled.
    """
    if (isinstance(shape_components, bool) or not isinstance(shape_components, int)
            or shape_components < 1):
        raise ValueError("shape_components must be a positive integer")
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("research model loader requires a numerical NPZ asset")
    before = path.stat()
    snapshot = path.read_bytes()
    after = path.stat()
    identity = lambda stat: (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    if identity(before) != identity(after) or len(snapshot) != after.st_size:
        raise ValueError("model asset changed while snapshot was read")
    with np.load(BytesIO(snapshot), allow_pickle=False) as asset:
        template = np.asarray(asset["v_template"], dtype=np.float64)
        shapedirs = np.asarray(asset["shapedirs"], dtype=np.float64)
        regressor = np.asarray(asset["J_regressor"], dtype=np.float64)
        tree = np.asarray(asset["kintree_table"])
    if (template.ndim != 2 or template.shape[1] != 3 or shapedirs.ndim != 3
            or shapedirs.shape[:2] != template.shape or regressor.ndim != 2
            or regressor.shape[0] < 22 or regressor.shape[1] != len(template)
            or tree.ndim != 2 or tree.shape[0] != 2 or tree.shape[1] < 22
            or tree.dtype.kind not in "iu"):
        raise ValueError("invalid SMPL-family rest model arrays")
    if len(set(int(x) for x in tree[1])) != tree.shape[1]:
        raise ValueError("kinematic tree joint IDs must be unique")
    lookup = {int(child): idx for idx, child in enumerate(tree[1])}
    try:
        parents = np.array([-1] + [lookup[int(parent)] for parent in tree[0, 1:22]])
    except KeyError as exc:
        raise ValueError("kinematic tree parent ID missing") from exc
    if not np.array_equal(parents, PARENTS):
        raise ValueError("model first22 do not match the body22 hierarchy")
    if shape_components > shapedirs.shape[-1]:
        raise ValueError("requested shape components exceed model basis count")
    if (not np.isfinite(template).all() or not np.isfinite(regressor[:22]).all()
            or not np.isfinite(shapedirs[..., :shape_components]).all()):
        raise ValueError("model contains nonfinite numerical values")
    rest = regressor[:22] @ template
    basis = np.einsum("jv,vcb->jcb", regressor[:22], shapedirs[..., :shape_components])
    full_joints = len(regressor)
    model_kind = "SMPL-X" if full_joints == 55 else "SMPL-H" if full_joints == 52 else "SMPL" if full_joints == 24 else "body22_fixture"
    metadata = {"model_path": str(path.resolve()),
                "model_sha256": hashlib.sha256(snapshot).hexdigest(),
                "model_kind": model_kind, "vertex_count": len(template),
                "asset_joint_count": full_joints,
                "asset_shape_components": shapedirs.shape[-1],
                "asset_loading": "safe_npz_immutable_byte_snapshot"}
    return DifferentiableBody22(rest, basis, metadata=metadata, device=device, dtype=dtype)


def _boolean_mask(value, shape, name):
    mask = np.asarray(value)
    if mask.dtype != np.bool_ or mask.shape != shape:
        raise ValueError(f"{name} must be a boolean array of shape {shape}")
    return mask.copy()


def _positions_input(positions, position_mask):
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (22, 3) or len(positions) < 1:
        raise ValueError("positions must have shape (T,22,3), T >= 1")
    mask = _boolean_mask(position_mask, positions.shape[:2], "position_mask")
    if not np.isfinite(positions[mask]).all():
        raise ValueError("positions selected by position_mask must be finite")
    if not mask.any():
        raise ValueError("at least one position observation is required")
    return np.where(mask[..., None], positions, 0.), mask


def _timestamps_input(timestamps, count):
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if (timestamps.shape != (count,) or not np.isfinite(timestamps).all()
            or np.any(np.diff(timestamps) <= 0)):
        raise ValueError("timestamps must be finite and strictly increasing with shape (T,)")
    return timestamps


def build_transition_mask(timestamps, valid_mask, transition_mask=None, *, max_dt=.1):
    """Select valid neighboring pairs; never interpolate or join across gaps.

    ``transition_mask`` is an optional explicit (T-1,) or (T-1,J) boolean mask
    for tracking resets or other caller-known discontinuities.  Missing data
    never creates a longer-span edge to the next available frame.
    """
    valid = np.asarray(valid_mask)
    if valid.ndim != 2 or valid.dtype != np.bool_ or len(valid) < 1:
        raise ValueError("valid_mask must be a nonempty boolean array of shape (T,J)")
    timestamps = _timestamps_input(timestamps, len(valid))
    if not np.isfinite(max_dt) or max_dt <= 0:
        raise ValueError("max_dt must be finite and positive")
    mask = valid[:-1] & valid[1:] & (np.diff(timestamps) <= max_dt)[:, None]
    if transition_mask is not None:
        explicit = np.asarray(transition_mask)
        if explicit.dtype != np.bool_ or explicit.shape not in ((len(valid) - 1,), mask.shape):
            raise ValueError("transition_mask must be boolean and have shape (T-1,) or (T-1,J)")
        if explicit.ndim == 1:
            explicit = explicit[:, None]
        mask &= explicit
    return mask


def world_rotation_increments(rotations):
    """Full world increments, NOT body-local R[t].T @ R[t+1]."""
    if isinstance(rotations, torch.Tensor):
        if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
            raise ValueError("rotations must have shape (T,J,3,3)")
        return rotations[1:] @ rotations[:-1].transpose(-1, -2)
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
        raise ValueError("rotations must have shape (T,J,3,3)")
    return rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)


def _validate_rotations(rotations, name):
    if not np.isfinite(rotations).all():
        raise ValueError(f"{name} must be finite")
    if (not np.allclose(np.swapaxes(rotations, -1, -2) @ rotations, np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-5, rtol=0)):
        raise ValueError(f"{name} must be proper SO(3) rotations")


def rotation_increment_errors_deg(fitted_rotations, source_rotations, pair_mask):
    """Selected geodesic increment errors, for reporting rather than optimization."""
    fitted = np.asarray(fitted_rotations, dtype=np.float64)
    source = np.asarray(source_rotations, dtype=np.float64)
    if fitted.ndim != 4 or fitted.shape[-2:] != (3, 3) or source.shape != fitted.shape:
        raise ValueError("rotations must have matching (T,J,3,3) shapes")
    mask = _boolean_mask(pair_mask, (len(fitted) - 1, fitted.shape[1]), "pair_mask")
    # Selecting endpoints before matmul prevents irrelevant masked NaNs from
    # generating invalid floating point intermediates.
    fd = fitted[1:][mask] @ np.swapaxes(fitted[:-1][mask], -1, -2)
    sd = source[1:][mask] @ np.swapaxes(source[:-1][mask], -1, -2)
    _validate_rotations(fd, "fitted selected increments")
    _validate_rotations(sd, "source selected increments")
    relative = np.swapaxes(fd, -1, -2) @ sd
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1) / 2
    return np.rad2deg(np.arccos(np.clip(cosine, -1., 1.)))


def _pseudo_huber_squared(squared_distance, delta):
    return 2 * delta ** 2 * (torch.sqrt(1 + squared_distance / delta ** 2) - 1)


def fit_shape_from_positions(model, positions, position_mask, *, config=None,
                             initial_betas=None, frame_mask=None):
    """Estimate one sequence's shape from masked bone lengths, then freeze it.

    This is only an anthropometric skeleton fit, not a measured surface shape.
    A caller can select independent calibration frames and reuse these betas
    for non-overlapping evaluation windows without beta-label leakage.
    """
    config = (config or FitConfig()).validate()
    observed, mask = _positions_input(positions, position_mask)
    if frame_mask is not None:
        mask &= _boolean_mask(frame_mask, (len(observed),), "frame_mask")[:, None]
    bone_mask = mask[:, 1:] & mask[:, PARENTS[1:]]
    if not bone_mask.any():
        raise ValueError("shape fit requires at least one observed parent-child bone")
    initial = np.zeros(model.shape_components) if initial_betas is None else np.asarray(initial_betas, dtype=np.float64)
    if initial.shape != (model.shape_components,) or not np.isfinite(initial).all():
        raise ValueError("initial_betas must be finite with shape (K,)")
    if np.any(np.abs(initial) > config.beta_bound):
        raise ValueError("initial_betas exceed beta_bound")
    target_lengths = np.linalg.norm(observed[:, 1:] - observed[:, PARENTS[1:]], axis=-1)
    selected_lengths = torch.as_tensor(target_lengths[bone_mask], dtype=model.dtype, device=model.device)
    selected_joints = torch.as_tensor(np.broadcast_to(np.arange(21), bone_mask.shape)[bone_mask], device=model.device)
    beta = torch.nn.Parameter(torch.as_tensor(initial.copy(), dtype=model.dtype, device=model.device))
    optimizer = torch.optim.Adam([beta], lr=config.shape_lr)
    history = []
    start = time.monotonic()

    def objective():
        rest = model.rest_joints(beta)
        lengths = torch.linalg.vector_norm(rest[1:] - rest[PARENTS[1:]], dim=-1)
        residual = lengths[selected_joints] - selected_lengths
        bone = _pseudo_huber_squared(residual.square(), config.bone_huber_delta).mean()
        regularization = beta.square().mean()
        total = bone + config.shape_regularization * regularization
        return total, {"loss": total, "bone_loss": bone,
                       "beta_prior_loss": regularization,
                       "bone_rmse_m": residual.square().mean().sqrt()}

    best_beta = beta.detach().clone()
    best_loss = float("inf")
    for iteration in range(config.shape_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite shape fitting objective")
        value = float(loss.detach())
        if value < best_loss:
            best_beta = beta.detach().clone()
            best_loss = value
        if iteration % config.history_every == 0 or iteration == config.shape_iterations:
            history.append({"iteration": iteration,
                            **{key: float(term.detach()) for key, term in terms.items()}})
        if iteration == config.shape_iterations:
            break
        loss.backward()
        if beta.grad is None or not torch.isfinite(beta.grad).all():
            raise RuntimeError("non-finite shape fitting gradient")
        torch.nn.utils.clip_grad_norm_([beta], config.grad_clip)
        optimizer.param_groups[0]["lr"] = config.shape_lr * (
            config.lr_final_ratio + (1 - config.lr_final_ratio)
            * .5 * (1 + np.cos(np.pi * iteration / max(config.shape_iterations, 1))))
        optimizer.step()
        with torch.no_grad():
            beta.clamp_(-config.beta_bound, config.beta_bound)
    with torch.no_grad():
        beta.copy_(best_beta)
        _, final_terms = objective()
    return {"betas": beta.detach().cpu().numpy().copy(), "history": history,
            "metadata": {"iterations": config.shape_iterations,
                         "elapsed_seconds": time.monotonic() - start,
                         "bone_constraints": int(bone_mask.sum()),
                         "calibration_frames": int(np.any(bone_mask, axis=1).sum()),
                         "shape_kind": "joint_length_fit_with_regularization",
                         "surface_shape_observed": False,
                         "beta_bound": config.beta_bound,
                         "initial_betas": initial.tolist(),
                         "returned_best_objective": best_loss,
                         "final": {key: float(term) for key, term in final_terms.items()}}}


def _minimal_swing(source, target):
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cross = np.cross(source, target)
    cosine = float(np.clip(np.dot(source, target), -1., 1.))
    if cosine < -1 + 1e-8:
        basis = np.eye(3)[int(np.argmin(np.abs(source)))]
        axis = np.cross(source, basis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    skew = np.array([[0., -cross[2], cross[1]],
                     [cross[2], 0., -cross[0]], [-cross[1], cross[0], 0.]])
    return np.eye(3) + skew + skew @ skew / (1 + cosine)


def initialize_local_rotations(model, positions, position_mask, betas=None):
    """Geometry initializer; unconstrained twists/leaves inherit parent choices.

    Multidirection joints use proper Kabsch.  A single direction uses a minimal
    swing relative to the parent frame.  Pelvis orientation is seeded at the
    first frame with multiple non-collinear body directions, then propagated
    backward and forward through underconstrained frames by minimal swing.
    Thus missing hips at the window start do not choose an arbitrary yaw from
    identity before abruptly switching to the measured geometry's yaw.

    This is an offline initializer that can use future position geometry,
    without using source rotations.  The seed selects missing orientation
    degrees of freedom; it is not a twist observation or an axis certificate.
    """
    observed, mask = _positions_input(positions, position_mask)
    if not mask[:, 0].any():
        raise ValueError("initializer requires at least one valid pelvis position")
    beta = np.zeros(model.shape_components) if betas is None else np.asarray(betas, dtype=np.float64)
    if beta.shape != (model.shape_components,) or not np.isfinite(beta).all():
        raise ValueError("initializer betas must have finite shape (K,)")
    with torch.no_grad():
        rest = model.rest_joints(beta).cpu().numpy()
    count = len(observed)
    global_rotations = np.repeat(np.eye(3)[None, None], count * 22, axis=0).reshape(count, 22, 3, 3)
    children = [np.flatnonzero(PARENTS == joint) for joint in range(22)]

    def directions(frame, joint):
        active = [int(child) for child in children[joint] if mask[frame, joint] and mask[frame, child]]
        source = np.array([rest[child] - rest[joint] for child in active]).reshape(-1, 3)
        target = np.array([observed[frame, child] - observed[frame, joint] for child in active]).reshape(-1, 3)
        lengths = np.linalg.norm(source, axis=-1) * np.linalg.norm(target, axis=-1)
        active_mask = lengths > 1e-12
        source, target = source[active_mask], target[active_mask]
        if len(source):
            source /= np.linalg.norm(source, axis=-1)[:, None]
            target /= np.linalg.norm(target, axis=-1)[:, None]
        multidirection = (len(source) >= 2
                          and np.linalg.matrix_rank(source, tol=.02) >= 2
                          and np.linalg.matrix_rank(target, tol=.02) >= 2)
        return source, target, multidirection

    def align(geometry, previous):
        source, target, multidirection = geometry
        if multidirection:
            left, _, right = np.linalg.svd(target.T @ source)
            correction = np.eye(3)
            correction[-1, -1] = np.linalg.det(left @ right)
            return left @ correction @ right
        if len(source):
            return _minimal_swing(previous @ source[0], target[0]) @ previous
        return previous

    root_geometry = [directions(frame, 0) for frame in range(count)]
    anchor = next((frame for frame, geometry in enumerate(root_geometry) if geometry[2]), None)
    if anchor is None:
        # No frame supplies the missing root direction: keep the historical
        # identity-based minimal-swing choice and explicitly report no anchor.
        for frame in range(count):
            previous = global_rotations[frame - 1, 0] if frame else np.eye(3)
            global_rotations[frame, 0] = align(root_geometry[frame], previous)
    else:
        global_rotations[anchor, 0] = align(root_geometry[anchor], np.eye(3))
        for frame in range(anchor - 1, -1, -1):
            global_rotations[frame, 0] = align(root_geometry[frame], global_rotations[frame + 1, 0])
        for frame in range(anchor + 1, count):
            global_rotations[frame, 0] = align(root_geometry[frame], global_rotations[frame - 1, 0])

    for frame in range(count):
        for joint in range(1, 22):
            parent = int(PARENTS[joint])
            global_rotations[frame, joint] = align(directions(frame, joint), global_rotations[frame, parent])
    local = global_rotations.copy()
    for child in range(1, 22):
        local[:, child] = np.swapaxes(global_rotations[:, PARENTS[child]], -1, -2) @ global_rotations[:, child]
    # Only a starting guess for missing pelvis frames; those frames stay masked
    # in the data loss and are not claimed to be observed.
    root = observed[:, 0].copy()
    frames = np.arange(count)
    selected = mask[:, 0]
    for axis in range(3):
        root[:, axis] = np.interp(frames, frames[selected], observed[selected, 0, axis])
    return {"local_rotations": local, "root_world": root, "betas": beta.copy(),
            "metadata": {"method": "geometry_only_root_anchor_bidirectional_minimal_swing_v2",
                         "root_orientation_anchor_frame": anchor,
                         "offline_future_geometry_seed": bool(anchor is not None and anchor > 0),
                         "source_rotations_used": False,
                         "absolute_twist_observed": False,
                         "native_axis_certified": False}}


def fit_body22(model, positions, position_mask, timestamps,
               source_global_rotations=None, rotation_mask=None, *, mode="positions_plus_rotation_increment",
               config=None, initialization=None, fixed_betas=None,
               transition_mask=None, shape_frame_mask=None):
    """Fit estimated body22 world poses; compare ablations using the same start.

    Shape is fitted once before pose optimization, or explicitly supplied and
    frozen.  Reuse ``result.initialization`` and ``result.betas`` for an exactly
    shared ablation start.  Source rotations constrain only their full world
    changes, never their unknown absolute native axes.  Valid timestamps,
    neighboring masks, max_dt and caller-specified reset masks gate all edges.
    """
    config = (config or FitConfig()).validate()
    if mode not in ("positions_only", "positions_plus_rotation_increment"):
        raise ValueError("mode must be positions_only or positions_plus_rotation_increment")
    observed, position_valid = _positions_input(positions, position_mask)
    count = len(observed)
    timestamps = _timestamps_input(timestamps, count)
    # Validate reset masks even when a window has no rotational observations.
    position_pairs = build_transition_mask(timestamps, position_valid, transition_mask, max_dt=config.max_dt)
    source = np.repeat(np.eye(3)[None, None], count * 22, axis=0).reshape(count, 22, 3, 3)
    rotation_valid = np.zeros((count, 22), dtype=bool)
    if source_global_rotations is not None:
        rotations = np.asarray(source_global_rotations, dtype=np.float64)
        if rotations.shape != (count, 22, 3, 3):
            raise ValueError("source_global_rotations must have shape (T,22,3,3)")
        rotation_valid = (np.isfinite(rotations).all(axis=(-2, -1)) if rotation_mask is None
                          else _boolean_mask(rotation_mask, (count, 22), "rotation_mask"))
        rotation_pairs = build_transition_mask(timestamps, rotation_valid, transition_mask, max_dt=config.max_dt)
        endpoint_mask = np.zeros((count, 22), dtype=bool)
        endpoint_mask[:-1] |= rotation_pairs
        endpoint_mask[1:] |= rotation_pairs
        _validate_rotations(rotations[endpoint_mask], "source rotations on selected neighboring pairs")
        source = np.where(endpoint_mask[..., None, None], rotations, source)
    elif rotation_mask is not None:
        raise ValueError("rotation_mask requires source_global_rotations")
    rotation_pairs = build_transition_mask(timestamps, rotation_valid, transition_mask, max_dt=config.max_dt)
    if mode == "positions_plus_rotation_increment" and not rotation_pairs.any():
        raise ValueError("rotation-increment fitting requires at least one valid neighboring pair")

    shape_result = None
    if fixed_betas is not None:
        betas = np.asarray(fixed_betas, dtype=np.float64)
    elif initialization is not None and "betas" in initialization:
        betas = np.asarray(initialization["betas"], dtype=np.float64)
    else:
        shape_result = fit_shape_from_positions(model, observed, position_valid,
                                                config=config, frame_mask=shape_frame_mask)
        betas = shape_result["betas"]
    if betas.shape != (model.shape_components,) or not np.isfinite(betas).all():
        raise ValueError("fixed/initialization betas must be finite with shape (K,)")
    if initialization is None:
        initialization = initialize_local_rotations(model, observed, position_valid, betas)
    if not isinstance(initialization, dict) or not {"local_rotations", "root_world"} <= initialization.keys():
        raise ValueError("initialization must contain local_rotations and root_world")
    initialization_metadata = dict(initialization.get("metadata", {"method": "caller_supplied"}))
    initialization = {key: np.asarray(value, dtype=np.float64).copy()
                      for key, value in initialization.items() if key in ("local_rotations", "root_world", "betas")}
    init_local = np.asarray(initialization["local_rotations"], dtype=np.float64)
    init_root = np.asarray(initialization["root_world"], dtype=np.float64)
    if init_local.shape != (count, 22, 3, 3) or init_root.shape != (count, 3):
        raise ValueError("initialization local_rotations/root_world shape mismatch")
    _validate_rotations(init_local, "initial local_rotations")
    if not np.isfinite(init_root).all():
        raise ValueError("initial root_world must be finite")
    initialization = {"local_rotations": init_local.copy(), "root_world": init_root.copy(),
                      "betas": betas.copy(), "metadata": initialization_metadata}
    tensor = lambda value: torch.as_tensor(value, dtype=model.dtype, device=model.device)
    selected_positions = torch.as_tensor(position_valid, device=model.device)
    selected_position_pairs = torch.as_tensor(position_pairs, device=model.device)
    selected_rotation_pairs = torch.as_tensor(rotation_pairs, device=model.device)
    target = tensor(observed)
    source_deltas = world_rotation_increments(tensor(source))
    local_start = tensor(init_local)
    beta_tensor = tensor(betas)
    pose_delta = torch.nn.Parameter(torch.zeros((count, 22, 3), dtype=model.dtype, device=model.device))
    root = torch.nn.Parameter(tensor(init_root.copy()))
    optimizer = torch.optim.Adam([{"params": [pose_delta], "lr": config.pose_lr},
                                  {"params": [root], "lr": config.root_lr}])
    history = []
    start = time.monotonic()
    eye = torch.eye(3, dtype=model.dtype, device=model.device)
    if count > 1:
        dt = np.diff(timestamps)
        nominal_dt = float(np.median(dt))
        temporal_scale = tensor(np.minimum((nominal_dt / dt) ** 2, 100.))[:, None]
    else:
        temporal_scale = tensor(np.empty((0, 1)))

    def objective():
        local = so3_exp(pose_delta) @ local_start
        fitted, global_rotations = model(local, root, beta_tensor)
        residual = fitted[selected_positions] - target[selected_positions]
        position_loss = _pseudo_huber_squared(residual.square().sum(-1), config.position_huber_delta).mean()
        if mode == "positions_plus_rotation_increment":
            delta_error = (world_rotation_increments(global_rotations)[selected_rotation_pairs]
                           - source_deltas[selected_rotation_pairs])
            increment_loss = .5 * delta_error.square().sum((-2, -1)).mean()
        else:
            increment_loss = fitted.new_zeros(())
        increment_prior = pose_delta.square().mean()
        local_rest_prior = .5 * (local[:, 1:] - eye).square().sum((-2, -1)).mean()
        if selected_position_pairs.any():
            local_change = .5 * (local[1:] - local[:-1]).square().sum((-2, -1))
            temporal_prior = (local_change * temporal_scale)[selected_position_pairs].mean()
        else:
            temporal_prior = fitted.new_zeros(())
        total = (position_loss + config.rotation_increment_weight * increment_loss
                 + config.pose_increment_regularization * increment_prior
                 + config.local_rest_regularization * local_rest_prior
                 + config.temporal_regularization * temporal_prior)
        terms = {"loss": total, "position_loss": position_loss,
                 "rotation_increment_loss": increment_loss,
                 "pose_increment_prior_loss": increment_prior,
                 "local_rest_prior_loss": local_rest_prior,
                 "temporal_prior_loss": temporal_prior,
                 "position_rmse_m": residual.square().sum(-1).mean().sqrt()}
        return total, terms, (fitted, global_rotations, local)

    best_loss = float("inf")
    best_iteration = 0
    best_pose = pose_delta.detach().clone()
    best_root = root.detach().clone()
    for iteration in range(config.pose_iterations + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, terms, _ = objective()
        if not torch.isfinite(loss):
            raise RuntimeError("non-finite body22 fitting objective")
        value = float(loss.detach())
        if value < best_loss:
            best_loss = value
            best_iteration = iteration
            best_pose = pose_delta.detach().clone()
            best_root = root.detach().clone()
        if iteration % config.history_every == 0 or iteration == config.pose_iterations:
            history.append({"iteration": iteration,
                            **{key: float(term.detach()) for key, term in terms.items()}})
        if iteration == config.pose_iterations:
            break
        loss.backward()
        if (pose_delta.grad is None or root.grad is None
                or not torch.isfinite(pose_delta.grad).all() or not torch.isfinite(root.grad).all()):
            raise RuntimeError("non-finite body22 fitting gradient")
        torch.nn.utils.clip_grad_norm_([pose_delta, root], config.grad_clip)
        annealing = config.lr_final_ratio + (1 - config.lr_final_ratio) * .5 * (
            1 + np.cos(np.pi * iteration / max(config.pose_iterations, 1)))
        optimizer.param_groups[0]["lr"] = config.pose_lr * annealing
        optimizer.param_groups[1]["lr"] = config.root_lr * annealing
        optimizer.step()
    with torch.no_grad():
        pose_delta.copy_(best_pose)
        root.copy_(best_root)
        _, final_terms, (fitted, global_rotations, local) = objective()
        rest_root = model.rest_joints(beta_tensor)[0]
        smpl_trans = root - rest_root
    numpy = lambda value: value.detach().cpu().numpy().copy()
    fitted_np, global_np = numpy(fitted), numpy(global_rotations)
    errors = rotation_increment_errors_deg(global_np, source, rotation_pairs)
    metadata = {"mode": mode, "model": dict(model.metadata), "config": asdict(config),
                "initialization": initialization_metadata,
                "iterations": config.pose_iterations, "best_iteration": best_iteration,
                "elapsed_seconds": time.monotonic() - start,
                "returned_best_objective": best_loss,
                "position_constraints": int(position_valid.sum()),
                "rotation_increment_constraints": int(rotation_pairs.sum()),
                "position_neighbor_constraints": int(position_pairs.sum()),
                "shape_frozen_during_pose_fit": True,
                "shape": shape_result["metadata"] if shape_result else {"source": "caller_supplied_fixed_betas"},
                "shape_history": shape_result["history"] if shape_result else [],
                "final": {key: float(term) for key, term in final_terms.items()},
                "increment_errors_deg": {"mean": float(errors.mean()) if len(errors) else None,
                                         "p95": float(np.percentile(errors, 95)) if len(errors) else None,
                                         "max": float(errors.max()) if len(errors) else None},
                "label_kind": "model_fitted_estimate",
                "rotation_basis_verified": False, "native_axis_certified": False,
                "absolute_twist_observed": False, "target_model_axes_defined": True,
                "weak_absolute_orientation_joints": [BODY_NAMES[j] for j in (10, 11, 15, 20, 21)],
                "single_bone_twist": "unobserved_from_xyz; initialization_and_priors_select_absolute_offset",
                "terminal_orientations": "no_body22_xyz_constraints; increments_only_when_available",
                "source_inputs_modified": False,
                "output_position_kind": "model_fk_positions; caller_must_preserve_original_observations",
                "camera_trajectory_modified": False,
                "position_metrics_world_alignment": "none"}
    return FitResult(fitted_np, global_np, numpy(local), numpy(root), numpy(smpl_trans),
                     betas.copy(), initialization, history, metadata)
