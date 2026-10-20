"""Gymnasium declarations shared by environments exposing named timed state."""
import string

from gymnasium import spaces
import numpy as np


def named_sim_observation_space(original, joint_names):
    if not isinstance(original, spaces.Dict):
        raise TypeError("Named simulation observations require a Dict observation space")
    names = tuple(joint_names)
    if not names or len(names) != len(set(names)):
        raise ValueError("Robot joint names must be unique and nonempty")
    return spaces.Dict({
        **original.spaces,
        "joint_names": spaces.Tuple(tuple(
            spaces.Text(max_length=128, min_length=1, charset=string.ascii_letters + string.digits + "_-/")
            for _ in names)),
        "sim_time_s": spaces.Box(low=0., high=np.inf, shape=(), dtype=np.float64),
    })
