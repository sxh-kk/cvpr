"""SO(3) operations, scalar-first quaternions, column-vector transforms."""

import numpy as np


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    valid = np.isfinite(q).all(axis=-1) & np.isfinite(norm[..., 0]) & (norm[..., 0] > 1e-8)
    safe = np.divide(q, norm, out=np.zeros_like(q), where=norm > 1e-8)
    safe[~valid] = [1, 0, 0, 0]
    return safe, valid


def quat_multiply(a, b):
    a, b = np.broadcast_arrays(a, b)
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate([aw * bw - (av * bv).sum(-1, keepdims=True),
                           aw * bv + bw * av + np.cross(av, bv)], axis=-1)


def axis_angle_to_quat(aa):
    aa = np.asarray(aa, dtype=np.float64)
    theta = np.linalg.norm(aa, axis=-1, keepdims=True)
    scale = .5 * np.sinc(theta / (2 * np.pi))
    return np.concatenate([np.cos(theta / 2), aa * scale], axis=-1)


def quat_to_matrix(q):
    q, _ = normalize_quat(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)], -1).reshape(q.shape[:-1]+(3, 3))


def matrix_to_quat(m):
    m = np.asarray(m, dtype=np.float64)
    # Choose the largest component to avoid division by tiny sin(theta).
    candidates = np.stack([1+np.trace(m, axis1=-2, axis2=-1),
                           1+m[..., 0, 0]-m[..., 1, 1]-m[..., 2, 2],
                           1-m[..., 0, 0]+m[..., 1, 1]-m[..., 2, 2],
                           1-m[..., 0, 0]-m[..., 1, 1]+m[..., 2, 2]], -1)
    choice = candidates.argmax(-1)
    q = np.zeros(m.shape[:-2]+(4,))
    for k in range(4):
        mask = choice == k
        r = m[mask]
        s = 2 * np.sqrt(np.maximum(candidates[..., k][mask], 1e-15))
        if k == 0:
            out = np.stack([s/4, (r[..., 2, 1]-r[..., 1, 2])/s,
                            (r[..., 0, 2]-r[..., 2, 0])/s, (r[..., 1, 0]-r[..., 0, 1])/s], -1)
        else:
            i, j, l = k-1, k % 3, (k+1) % 3
            out = np.zeros((len(r), 4))
            out[:, 0] = (r[:, l, j]-r[:, j, l])/s
            out[:, i+1] = s/4
            out[:, j+1] = (r[:, j, i]+r[:, i, j])/s
            out[:, l+1] = (r[:, l, i]+r[:, i, l])/s
        q[mask] = out
    q, _ = normalize_quat(q)
    return np.where(q[..., :1] < 0, -q, q)


def slerp(a, b, weight):
    a, _ = normalize_quat(a)
    b, _ = normalize_quat(b)
    dot = (a*b).sum(-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    w = np.asarray(weight)
    while w.ndim < a.ndim:
        w = w[..., None]
    denom = np.where(sine > 1e-6, sine, 1)
    q = np.sin((1-w)*theta)/denom*a + np.sin(w*theta)/denom*b
    q = np.where(sine > 1e-6, q, (1-w)*a+w*b)
    return normalize_quat(q)[0]


def rigid_transform(value):
    m = np.asarray(value, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("world_from_source must be a finite 4x4 matrix")
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous transform")
    if not np.allclose(m[:3, :3].T@m[:3, :3], np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(m[:3, :3]), 1, atol=1e-5):
        raise ValueError("Transform must be a proper rotation; reflections/scaling are not permitted")
    return m
