"""Restore checkpoint names omitted for tensors that truly share a model view."""

from collections import OrderedDict, defaultdict
from collections.abc import Mapping

import torch
from torch import nn


def restore_shared_tensor_aliases(
    module: nn.Module, state_dict: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Return a shallow copy with only proven shared-storage aliases restored.

    Safetensors writers can retain just one name for a shared tensor. Infer the
    names from the constructed target module, never from equal initial values
    or a key-name convention. Independent missing weights remain missing for
    the caller's strict load. Multiple saved aliases must agree before loading
    can overwrite the shared destination storage.
    """
    groups = defaultdict(list)
    for name, value in module.state_dict(keep_vars=True).items():
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            continue
        # data_ptr alone confuses independent empty/meta storages. _cdata is
        # the actual StorageImpl identity; offset and layout prove the view.
        identity = (
            value.untyped_storage()._cdata,
            value.dtype,
            value.device,
            tuple(value.shape),
            tuple(value.stride()),
            value.storage_offset(),
        )
        groups[identity].append(name)

    restored = OrderedDict(state_dict)
    if hasattr(state_dict, "_metadata"):
        restored._metadata = state_dict._metadata
    for names in groups.values():
        if len(names) < 2:
            continue
        present = [name for name in names if name in restored]
        if not present:
            continue
        source = restored[present[0]]
        for name in present[1:]:
            other = restored[name]
            if (not isinstance(source, torch.Tensor) or not isinstance(other, torch.Tensor)
                    or source.dtype != other.dtype or source.shape != other.shape
                    or not torch.equal(source, other)):
                raise RuntimeError(f"Conflicting checkpoint values for shared tensor aliases: {present}")
        for name in names:
            if name not in restored:
                restored[name] = source
    return restored
