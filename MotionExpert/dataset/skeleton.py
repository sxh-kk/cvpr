from pathlib import Path

import torch


class RepresentationSkeleton(torch.nn.Module):
    """Only the kinematic tree and hand PCA bases needed by saved motion packs."""

    def __init__(self):
        super().__init__()
        path = Path(__file__).resolve().parents[1] / 'assets/smplx_representation.pt'
        for name, value in torch.load(path, weights_only=True, map_location='cpu').items():
            self.register_buffer(name, value)


def get_smpl():
    return RepresentationSkeleton()
