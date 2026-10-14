"""Builder startup checks using tiny owned assets and real CPU spawn workers."""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import pickle
from queue import Empty, Queue
from threading import Event, Lock

import numpy as np
import pytest
import torch

from MotionExpert import build_window_cache as builder
from MotionExpert.config import smplh_motion_expert_config
from MotionExpert.dataset.smplh_archive import SMPLHParameterArchive
from MotionExpert.dataset.window_cache import atomic_json, cache_binding, protocol_from_config
from MotionExpert.tests.fixtures import owned_archive
from MotionExpert.tests.test_vision_profile_cache import write_cache


@pytest.fixture
def validated_builder_source(owned_archive, tmp_path, monkeypatch):
    # Spawn imports NumPy before the initializer sets Arrow/Torch thread limits.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    torch.set_num_threads(1)
    vision = write_cache(owned_archive, tmp_path / "vision")
    archive = SMPLHParameterArchive(owned_archive, vision_profile_root=vision)
    cfg = smplh_motion_expert_config()
    cfg.DATA.ARCHIVE_ROOT = cfg.DATA.DATA_DIR = str(owned_archive)
    cfg.DATA.VISION_PROFILE_ROOT = str(vision)
    return archive, cfg


def test_validated_archive_pickle_discards_warm_arrow_and_vision_handles(validated_builder_source):
    archive, cfg = validated_builder_source
    archive.read_window(0, start_frame=0, num_frames=11, fps=5)
    assert archive.reader.files and archive.reader.groups
    assert archive.reader.cached_bytes > 0
    assert archive.vision_cache._cache
    assert isinstance(archive.vision_cache._cache[0]["features"], np.memmap)

    copied = pickle.loads(pickle.dumps(archive))
    assert not copied.reader.files and not copied.reader.groups
    assert not copied.reader.read_log and copied.reader.cached_bytes == 0
    assert not copied.vision_cache._cache
    assert copied.records() == archive.records()
    np.testing.assert_array_equal(copied.model.rest, archive.model.rest)
    np.testing.assert_array_equal(copied.model.shape_basis, archive.model.shape_basis)
    protocol = protocol_from_config(cfg)
    assert cache_binding(copied, protocol) == cache_binding(archive, protocol)
    copied.assert_current(copied.records()[0])
    # Serialization does not evict the parent's already opened planning cache.
    assert archive.reader.files and archive.vision_cache._cache


def test_spawn_builder_uses_validated_snapshot_without_reconstructing_archive(
        validated_builder_source, tmp_path, monkeypatch):
    archive, cfg = validated_builder_source
    calls = []

    def validated_archive(config, allow_unpublished):
        calls.append((config, allow_unpublished))
        return archive

    monkeypatch.setattr(builder, "_archive", validated_archive)
    manifest = builder.build_window_cache(cfg, tmp_path / "windows", workers=1)
    # The tiny fixture's model SHA is authorized by a parent-only monkeypatch.
    # A fresh archive constructor in the spawned process would reject the asset.
    assert len(calls) == 1
    assert manifest["state"] == "complete"
    assert manifest["completed_episodes"] == 3
    assert manifest["completed_windows"] > 0
    assert set(manifest["packed"]) == {"train", "val", "test"}
    for split, packed in manifest["packed"].items():
        motion = np.load(tmp_path / "windows" / packed["arrays"]["motion"]["path"], mmap_mode="r")
        assert motion.shape == (manifest["indices"][split]["windows"], 11, 245)
        assert np.isfinite(motion).all()


def test_worker_snapshot_rejects_different_source_binding(validated_builder_source, tmp_path):
    archive, cfg = validated_builder_source
    protocol = protocol_from_config(cfg)
    wrong = deepcopy(cache_binding(archive, protocol))
    wrong["archive_fingerprint"] = "different-archive"
    with pytest.raises(ValueError, match="worker source binding differs"):
        builder._init_worker(archive, tmp_path, wrong, protocol, 4096)


def test_worker_control_changes_four_to_eight_and_drains_before_lowering(tmp_path, monkeypatch):
    control = tmp_path / "workers.json"
    atomic_json(control, {"active_workers": 4})
    starts, release = Queue(), [Event() for _ in range(12)]
    lock, lower_requested, lower_observed = Lock(), Event(), Event()
    first_four_finished = Event()
    running = maximum_running = 0
    results = []
    original_control = builder._active_workers

    def observed_control(path, maximum):
        value = original_control(path, maximum)
        if lower_requested.is_set() and value == 4:
            lower_observed.set()
        return value

    def blocked_episode(number):
        nonlocal running, maximum_running
        with lock:
            running += 1
            maximum_running = max(maximum_running, running)
        starts.put(number)
        assert release[number].wait(15), "Test did not release the episode task"
        with lock:
            running -= 1
        return number

    def progress(result, active):
        with lock:
            results.append((result, active))
            if len(results) >= 4:
                first_four_finished.set()

    monkeypatch.setattr(builder, "_active_workers", observed_control)
    monkeypatch.setattr(builder, "_write_episode", blocked_episode)
    with ThreadPoolExecutor(max_workers=8) as pool, ThreadPoolExecutor(max_workers=1) as scheduler:
        future = scheduler.submit(builder._execute_episode_tasks, pool, list(range(12)),
                                  max_workers=8, worker_control=control, progress=progress)
        try:
            assert {starts.get(timeout=3) for _ in range(4)} == set(range(4))
            with pytest.raises(Empty):
                starts.get(timeout=.1)
            atomic_json(control, {"active_workers": 8})
            assert {starts.get(timeout=3) for _ in range(4)} == set(range(4, 8))
            atomic_json(control, {"active_workers": 4})
            lower_requested.set()
            assert lower_observed.wait(3)
            # Lowering permits already submitted work to finish. Four remaining
            # tasks occupy the entire new limit, so no ninth task is admitted.
            for event in release[:4]:
                event.set()
            assert first_four_finished.wait(3)
            with pytest.raises(Empty):
                starts.get(timeout=.1)
            release[4].set()
            assert starts.get(timeout=3) == 8
        finally:
            for event in release:
                event.set()
            future.result(timeout=10)
    assert maximum_running == 8
    assert sorted(result for result, _ in results) == list(range(12))
    assert all(active in (4, 8) for _, active in results)


@pytest.mark.parametrize("value", [
    {"active_workers": 0}, {"active_workers": 9}, {"active_workers": True},
    {"active_workers": 4.0}, {"active_workers": "4"}, {}, [],
    {"active_workers": 4, "unexpected": 1},
])
def test_worker_control_rejects_invalid_limits(tmp_path, value):
    control = tmp_path / "workers.json"
    atomic_json(control, value)
    with pytest.raises(ValueError, match="integer active_workers"):
        builder._active_workers(control, 8)


def test_worker_control_rejects_unreadable_json(tmp_path):
    control = tmp_path / "workers.json"
    with pytest.raises(ValueError, match="Cannot read"):
        builder._active_workers(control, 8)
    control.write_text('{"active_workers":')
    with pytest.raises(ValueError, match="Cannot read"):
        builder._active_workers(control, 8)

