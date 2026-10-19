"""Actual length methods, isolated from optional model/LeRobot dependencies."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def _class(path, name, methods, rename=None):
    tree = ast.parse((ROOT / path).read_text())
    node = copy.deepcopy(next(item for item in tree.body if isinstance(item, ast.ClassDef) and item.name == name))
    node.name, node.bases, node.keywords = rename or name, [], []
    node.body = [item for item in node.body if isinstance(item, ast.FunctionDef) and item.name in methods]
    return node


@pytest.fixture(scope="module")
def actual_methods():
    trainer = ROOT / "src/psi/trainers/trainer.py"
    tree = ast.parse(trainer.read_text())
    helper = copy.deepcopy(next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                                and node.name == "_trainer_dataset_length"))
    nodes = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), helper,
             _class("src/psi/trainers/trainer.py", "Trainer", {"len_train_dataset", "len_val_dataset"}),
             _class("src/psi/data/lerobot/lerobot_ext.py", "LeRobotDatasetWrapper", {"__len__"}),
             _class("src/psi/data/dataset.py", "Dataset", {"__init__", "__len__", "dataset_length"})]
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(trainer), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("split", ["train", "val"])
def test_history_wrapper_uses_filtered_map_rows_without_reading_features(actual_methods, split):
    wrapper = actual_methods.LeRobotDatasetWrapper()
    wrapper.motion_history = SimpleNamespace(original_indices=range(5_199_066))
    # Real history wrappers expose episode views, not sharded storage metadata.
    wrapper.datasets = [SimpleNamespace(sharded_trajectories=[[0, 1, 2]])]
    wrapper.base_dataset = range(5_416_386)
    dataset = actual_methods.Dataset(SimpleNamespace(transform=None), wrapper)
    trainer = SimpleNamespace(**{f"{split}_dataset": dataset})
    prop = getattr(actual_methods.Trainer, f"len_{split}_dataset")
    assert prop.fget(trainer) == 5_199_066
    assert not hasattr(wrapper.datasets[0], "shard_lengths")


@pytest.mark.parametrize("split", ["train", "val"])
def test_real_shard_length_protocol_keeps_frame_counts_instead_of_schedule_estimate(actual_methods, split):
    raw = SimpleNamespace(datasets=[SimpleNamespace(shard_lengths=[100, 200]),
                                    SimpleNamespace(shard_lengths=[70, 30])])
    dataset = SimpleNamespace(raw_dataset=raw, dataset_length=10_000_000)
    trainer = SimpleNamespace(**{f"{split}_dataset": dataset})
    assert getattr(actual_methods.Trainer, f"len_{split}_dataset").fget(trainer) == 400


@pytest.mark.parametrize("split", ["train", "val"])
def test_explicit_trainer_length_override_still_wins(actual_methods, split):
    trainer = SimpleNamespace(**{f"{split}_dataset": object()})
    prop = getattr(actual_methods.Trainer, f"len_{split}_dataset")
    prop.fset(trainer, 123)
    assert prop.fget(trainer) == 123


def test_plain_sized_and_explicit_dataset_lengths_remain_supported(actual_methods):
    prop = actual_methods.Trainer.len_train_dataset
    assert prop.fget(SimpleNamespace(train_dataset=range(17))) == 17
    assert prop.fget(SimpleNamespace(train_dataset=SimpleNamespace(dataset_length=23))) == 23


def test_legacy_lerobot_wrapper_without_history_keeps_original_length(actual_methods):
    wrapper = actual_methods.LeRobotDatasetWrapper()
    wrapper.motion_history, wrapper.base_dataset = None, range(41)
    dataset = actual_methods.Dataset(SimpleNamespace(transform=None), wrapper)
    assert actual_methods.Trainer.len_train_dataset.fget(SimpleNamespace(train_dataset=dataset)) == 41


def test_empty_sharded_mixture_keeps_zero_frame_count(actual_methods):
    dataset = SimpleNamespace(raw_dataset=SimpleNamespace(datasets=[]), dataset_length=99)
    assert actual_methods.Trainer.len_val_dataset.fget(SimpleNamespace(val_dataset=dataset)) == 0
