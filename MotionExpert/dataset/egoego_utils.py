# Thanks to https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py
# and
# https://github.com/lijiaman/egoego_release/blob/main/utils/data_utils/process_amass_dataset.py

import math

import numpy as np
import torch

from ..utils import rotation_conversions as transforms


SMPL_JOINTS = {
    "hips": 0,
    "leftUpLeg": 1,
    "rightUpLeg": 2,
    "spine": 3,
    "leftLeg": 4,
    "rightLeg": 5,
    "spine1": 6,
    "leftFoot": 7,
    "rightFoot": 8,
    "spine2": 9,
    "leftToeBase": 10,
    "rightToeBase": 11,
    "neck": 12,
    "leftShoulder": 13,
    "rightShoulder": 14,
    "head": 15,
    "leftArm": 16,
    "rightArm": 17,
    "leftForeArm": 18,
    "rightForeArm": 19,
    "leftHand": 20,
    "rightHand": 21,
}

DISCARD_TERRAIN_SEQUENCES = True  # throw away sequences where the person steps onto objects (determined by a heuristic)

DISCARD_SHORTER_THAN = 1.0  # seconds

# optional viz during processing
VIZ_PLOTS = False
VIZ_SEQ = False

OUT_FPS = 10
# OUT_FPS = 20

# for determining floor height
FLOOR_VEL_THRESH = 0.005 * (30 / OUT_FPS)
FLOOR_HEIGHT_OFFSET = 0.01
# for determining contacts
CONTACT_VEL_THRESH = 0.005 * (30 / OUT_FPS)
CONTACT_TOE_HEIGHT_THRESH = 0.04
CONTACT_ANKLE_HEIGHT_THRESH = 0.08
# for determining terrain interaction
TERRAIN_HEIGHT_THRESH = 0.04  # if static toe is above this height
ROOT_HEIGHT_THRESH = 0.04  # if maximum "static" root height is more than this + root_floor_height
CLUSTER_SIZE_THRESH = 0.25  # if cluster has more than this faction of fps (30 for 120 fps)


def local2global_pose(local_pose, kintree):
    # Thanks to EgoEgo
    # local_pose: ... X J X 3 X 3
    global_pose = local_pose.clone()
    for jId in range(len(kintree)):
        parent_id = kintree[jId]
        if parent_id >= 0:
            global_pose[..., jId, :, :] = torch.matmul(global_pose[..., parent_id, :, :], global_pose[..., jId, :, :])
    return global_pose


















