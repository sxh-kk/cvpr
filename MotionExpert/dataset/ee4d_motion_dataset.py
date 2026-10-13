"""Compatibility import for the dense archive reader; no SMPL assets are loaded."""
from .dense_motion_dataset import (
    DenseMotionDataset, DenseMotionDataModule,
    EE4D_Motion_Dataset, EE4D_Motion_DataModule,
)

__all__ = ["DenseMotionDataset", "DenseMotionDataModule", "EE4D_Motion_Dataset", "EE4D_Motion_DataModule"]
