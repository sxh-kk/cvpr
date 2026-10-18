"""CPU checks of actual data-path methods, without Qwen/Pydantic construction.

Image operations and tokenization are doubles. The tests validate transport of
provider-supplied history, not its DINO identity or shared-RGB alignment.
"""

from __future__ import annotations

import ast
import copy
from functools import reduce
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence


_ABSENT = object()


def _sample(history=_ABSENT):
    result = dict(input_ids=torch.tensor([1, 2]), pixel_values=torch.ones(1, 4),
                  image_grid_thw=torch.ones(1, 3, dtype=torch.long))
    if history is not _ABSENT:
        result["motion_img_embs"] = history
    return result


def _method_class(tree, name, methods):
    source = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    selected = [copy.deepcopy(node) for node in source.body
                if isinstance(node, ast.FunctionDef) and node.name in methods]
    assert {node.name for node in selected} == set(methods)
    return ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])


@pytest.fixture(scope="module")
def actual_data_methods():
    root = Path(__file__).parents[1] / "src/psi"
    transforms = ast.parse((root / "config/transform.py").read_text())
    collators = ast.parse((root / "trainers/qwen3vl_mixin.py").read_text())
    nodes = [_method_class(transforms, "EgodexRepackTransform", ["__call__"]),
             _method_class(transforms, "Psi0ModelTransform", ["__call__"]),
             _method_class(transforms, "DataTransform", ["__call__"]),
             _method_class(collators, "PaddedCollatorForTogether", ["__init__", "__call__"])]
    nodes += [copy.deepcopy(node) for node in collators.body if isinstance(node, ast.FunctionDef)
              and node.name in {"_stack_images", "_pad_string_tensor_batch"}]
    namespace = dict(torch=torch, np=np, F=F, Image=Image, pad_sequence=pad_sequence,
                     str_to_tensor=lambda value: torch.tensor(list(value.encode()), dtype=torch.uint8),
                     v2=SimpleNamespace(Identity=lambda: lambda value: value,
                                        Compose=lambda functions: lambda value:
                                        reduce(lambda previous, function: function(previous), functions, value)))
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                             *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual-dual-source-data-methods>", "exec"), namespace)
    return SimpleNamespace(**namespace)


class _ImageOperation:
    size = (4, 4)

    def __call__(self):
        return lambda image: image


def _transform_chain(actual):
    repack = actual.EgodexRepackTransform()
    repack.pad_state_dim = repack.pad_action_dim = None
    repack.dataset_name = "egodex"
    model = actual.Psi0ModelTransform()
    model.img_aug = model.view_aug = model.adaptive_resize = False
    model.resize = model.center_crop = _ImageOperation()

    def prefix_inputs(processor, images, instruction):
        count = len(instruction.split()) + 1
        return dict(input_ids=torch.arange(1, count + 1), pixel_values=torch.ones(len(images), 4),
                    image_grid_thw=torch.ones(len(images), 3, dtype=torch.long))

    model.build_qwenvl_inputs = prefix_inputs
    transform = actual.DataTransform()
    transform.repack = repack
    # The field stage also rebuilds its output; the chain must preserve the raw value.
    transform.field = lambda data, **kwargs: {
        **{key: value for key, value in data.items() if key != "motion_img_embs"},
        "raw_actions": data["actions"],
    }
    transform.model = model
    return transform


@pytest.mark.parametrize("source_type", ["numpy", "tensor"])
def test_standard_repack_field_model_collator_chain_preserves_supplied_history(actual_data_methods, source_type):
    transform = _transform_chain(actual_data_methods)
    samples, originals = [], []
    for index, instruction in enumerate(["pick", "pick cup"]):
        history = np.arange(6 * 5 * 1024, dtype=np.float32).reshape(6, 5, 1024) + index
        if source_type == "tensor":
            history = torch.from_numpy(history)
        raw = dict(current_images=[np.zeros((1, 4, 4, 3), dtype=np.uint8)],
                   states=np.zeros((1, 7), dtype=np.float32), actions=np.ones((2, 7), dtype=np.float32),
                   instruction=instruction, motion_img_embs=history)
        # The real EgoDex repacker does discard unrecognized keys.
        assert "motion_img_embs" not in transform.repack(raw)
        sample = transform(raw, no_aug=True)
        assert sample["motion_img_embs"] is history
        samples.append(sample)
        originals.append(torch.as_tensor(history))

    collator = actual_data_methods.PaddedCollatorForTogether(16, 0)
    batch = collator(samples)

    assert batch["motion_img_embs"].shape == (2, 6, 5, 1024)
    torch.testing.assert_close(batch["motion_img_embs"], torch.stack(originals), rtol=0, atol=0)
    torch.testing.assert_close(batch["attention_mask"], torch.tensor([[True, True, False], [True, True, True]]))
    assert batch["actions"].shape == (2, 2, 7)
    assert batch["states"].shape == (2, 1, 7)


def test_collator_stacks_mixed_numpy_and_tensor_without_pixel_dtype_cast(actual_data_methods):
    first = torch.arange(6 * 5 * 1024, dtype=torch.float32).reshape(6, 5, 1024)
    second = np.full((6, 5, 1024), 0.25, dtype=np.float32)
    collator = actual_data_methods.PaddedCollatorForTogether(16, 0, pixel_values_dtype=torch.bfloat16)

    batch = collator([_sample(first), _sample(second)])

    assert batch["pixel_values"].dtype == torch.bfloat16
    assert batch["motion_img_embs"].dtype == torch.float32
    torch.testing.assert_close(batch["motion_img_embs"][0], first, rtol=0, atol=0)
    torch.testing.assert_close(batch["motion_img_embs"][1], torch.from_numpy(second), rtol=0, atol=0)


def _processed_model_input():
    return dict(observations=[Image.new("RGB", (320, 256))], instruction="pick",
                states=np.zeros((1, 7), dtype=np.float32), actions=np.ones((2, 7), dtype=np.float32),
                raw_actions=np.ones((2, 7), dtype=np.float32), motion_img_embs=torch.ones(6, 5, 1024))


def test_history_path_keeps_processed_current_rgb_without_applying_image_ops(actual_data_methods):
    model = _transform_chain(actual_data_methods).model
    data = _processed_model_input()

    def forbidden_image_op():
        raise AssertionError("aligned RGB must not be independently resized or cropped")

    model.resize = model.center_crop = forbidden_image_op
    output = model(data, no_aug=True)

    assert output["observations"] is data["observations"]
    assert output["observations"][0] is data["observations"][0]
    assert output["observations"][0].size == (320, 256)
    assert output["motion_img_embs"] is data["motion_img_embs"]


@pytest.mark.parametrize("option", ["img_aug", "view_aug"])
def test_history_path_rejects_independent_random_augmentation(actual_data_methods, option):
    model = _transform_chain(actual_data_methods).model
    setattr(model, option, True)
    data = _processed_model_input()

    with pytest.raises(ValueError, match="shared upstream"):
        model(data, no_aug=False)
    # Evaluation disables those configured augmentations and preserves upstream RGB.
    output = model(data, no_aug=True)
    assert output["observations"] is data["observations"]


@pytest.mark.parametrize("missing_first", [False, True])
def test_collator_rejects_partially_missing_history_without_filling(actual_data_methods, missing_first):
    supplied = _sample(torch.ones(6, 5, 1024))
    samples = [_sample(), supplied] if missing_first else [supplied, _sample()]
    collator = actual_data_methods.PaddedCollatorForTogether(16, 0)

    with pytest.raises(ValueError, match="every sample"):
        collator(samples)
    assert "motion_img_embs" not in samples[0 if missing_first else 1]


@pytest.mark.parametrize(
    ("history", "message"),
    [(torch.zeros(11, 5, 1024), "shape"),
     (torch.zeros(1, 6, 5, 1024), "shape"),
     (np.zeros((6, 1024), dtype=np.float32), "shape"),
     (torch.zeros(6, 5, 1024, dtype=torch.int64), "floating-point"),
     (np.zeros((6, 5, 1024), dtype=np.uint8), "floating-point"),
     (None, "Tensor or numpy"), ([], "Tensor or numpy")],
)
def test_collator_rejects_invalid_history_instead_of_reshaping(actual_data_methods, history, message):
    collator = actual_data_methods.PaddedCollatorForTogether(16, 0)
    with pytest.raises(ValueError, match=message):
        collator([_sample(history)])


def test_absent_history_remains_absent_in_standard_psi0_path(actual_data_methods):
    transform = _transform_chain(actual_data_methods)
    raw = dict(current_images=[np.zeros((1, 4, 4, 3), dtype=np.uint8)],
               states=np.zeros((1, 7), dtype=np.float32), actions=np.ones((2, 7), dtype=np.float32),
               instruction="pick")
    sample = transform(raw, no_aug=True)
    batch = actual_data_methods.PaddedCollatorForTogether(16, 0)([sample])
    assert "motion_img_embs" not in sample
    assert "motion_img_embs" not in batch


def test_data_chain_does_not_add_passthrough_to_other_model_transforms(actual_data_methods):
    transform = actual_data_methods.DataTransform()
    transform.repack = lambda data, **kwargs: {"kept": data["kept"]}
    transform.field = lambda data, **kwargs: data
    transform.model = lambda data, **kwargs: data

    assert transform({"kept": 1, "motion_img_embs": torch.ones(6, 5, 1024)}) == {"kept": 1}
