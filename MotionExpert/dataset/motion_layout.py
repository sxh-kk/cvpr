"""Versioned channel layouts shared by encoding, losses and inference."""
from dataclasses import dataclass

DENSE_REPRE_TYPE = "dense12_rot6d_hands21_contacts_v1"
SMPLH_REPRE_TYPE = "smplh12_rot6d_hands21_contacts_v1"
K12_JOINT_INDICES = (0, 4, 5, 10, 11, 13, 14, 15, 18, 19, 20, 21)


@dataclass(frozen=True)
class MotionLayout:
    representation: str
    motion_dim: int
    joint_indices: tuple[int, ...]
    body_slice: slice
    global_slice: slice
    left_hand_slice: slice
    right_hand_slice: slice
    contact_slice: slice
    betas_slice: slice | None


_DENSE = MotionLayout(DENSE_REPRE_TYPE, 245, K12_JOINT_INDICES,
                      slice(0, 108), slice(108, 117), slice(117, 180),
                      slice(180, 243), slice(243, 245), None)
_SMPLH = MotionLayout(SMPLH_REPRE_TYPE, 245, K12_JOINT_INDICES,
                      slice(0, 108), slice(108, 117), slice(117, 180),
                      slice(180, 243), slice(243, 245), None)


def motion_layout(representation_or_dim):
    """Reject unknown contracts instead of guessing from a nearby dimension."""
    if type(representation_or_dim) is int:
        if representation_or_dim == 245:
            return _DENSE
    elif isinstance(representation_or_dim, str):
        if representation_or_dim == DENSE_REPRE_TYPE:
            return _DENSE
        if representation_or_dim == SMPLH_REPRE_TYPE:
            return _SMPLH
    raise ValueError(f"Unsupported motion representation/dimension: {representation_or_dim!r}")
