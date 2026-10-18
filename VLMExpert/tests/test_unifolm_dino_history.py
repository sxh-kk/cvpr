"""Native-cache/history integration with tiny committed arrays and pack doubles.

No pretrained weights or real dataset are needed. Cache validation, historical
indexing and shared RGB processing execute the production implementations.
"""

from __future__ import annotations

import importlib
import ast
import copy
import json
from pathlib import Path
import pickle
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from ImageProcess import SharedImageProcessor
from test_dual_source_data import actual_data_methods, _color_history_model, _processed_model_input


IMAGE_KEY = "observation.images.egocentric"


@pytest.fixture(scope="module")
def history_modules():
    # Optional LeRobot/Qwen dependencies live in the package __init__, while
    # these two contract modules need only numpy/Pillow/torch.
    directory = Path(__file__).parents[1] / "src/psi/data/lerobot"
    package_name = "_unifolm_history_contract_tests"
    package = ModuleType(package_name)
    package.__path__ = [str(directory)]
    sys.modules[package_name] = package
    return SimpleNamespace(
        cache=importlib.import_module(f"{package_name}.dino_cache"),
        history=importlib.import_module(f"{package_name}.motion_history"),
    )


class _FrameTable:
    def __init__(self, columns):
        self.columns = columns

    def __getitem__(self, key):
        return self.columns[key]

    def with_format(self, *args, **kwargs):
        return self

    def __len__(self):
        return len(self.columns["episode_index"])


class _Pack(SimpleNamespace):
    def __len__(self):
        return len(self.hf_dataset)


def _cache_pack(tmp_path, api, lengths=(65,), *, mutate_frames=None):
    root = tmp_path / "source"
    cache_root = tmp_path / "cache"
    (root / "meta").mkdir(parents=True)
    cache_root.mkdir()
    info = {"fps": 30, "features": {IMAGE_KEY: {"dtype": "video"}}}
    for name, value in (("info.json", info), ("modality.json", {"video": {}})):
        (root / "meta" / name).write_text(json.dumps(value))
    (root / "meta/episodes.jsonl").write_text("".join(
        json.dumps({"episode_index": episode, "length": length}) + "\n"
        for episode, length in enumerate(lengths)))
    image_process = SharedImageProcessor("256x320").metadata
    local_files = {"model.safetensors": "a" * 64, "config.json": "b" * 64}
    encoder = dict(model_id=api.DINO_MODEL_ID, revision="local", local_file_sha256=local_files,
                   weights_sha256=api.digest(local_files), image_process=image_process,
                   preprocessing=dict(do_resize=False, do_center_crop=False, do_rescale=True,
                                      do_normalize=True, rescale_factor=1 / 255,
                                      image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]))
    manifest = dict(contract=api.CONTRACT, state="complete", shape=[5, 1024], storage_dtype="float16",
                    episode_indices=list(range(len(lengths))), image_process=image_process, encoder=encoder,
                    extraction={"stage": "validation", "training": False, "sampling_step": 1},
                    source=dict(dataset_root=str(root.resolve()), fps=30, video_key=IMAGE_KEY,
                                meta_files={f"meta/{name}": api.file_identity(root / "meta" / name)
                                            for name in ("info.json", "modality.json", "episodes.jsonl")}))
    manifest["signature"] = api.manifest_signature(manifest)
    api.atomic_json(cache_root / "manifest.json", manifest)
    columns = {name: [] for name in ("episode_index", "frame_index", "timestamp", "index")}
    offset = 0
    for episode, length in enumerate(lengths):
        frame = np.arange(length, dtype=np.int64)
        timestamp = (frame / 30).astype(np.float32).astype(np.float64)
        arrays = dict(frame_index=frame, timestamp=timestamp,
                      source_frame_index=np.arange(length, dtype=np.int64),
                      video_timestamp=np.arange(length, dtype=np.float64) / 30,
                      valid=np.ones(length, dtype=bool))
        if mutate_frames:
            mutate_frames(episode, arrays)
        arrays["timestamp"] = arrays["timestamp"].astype(np.float32).astype(np.float64)
        directory = api.episode_directory(cache_root, episode)
        directory.mkdir(parents=True)
        features = np.broadcast_to((np.arange(length) + episode * 1000)[:, None, None],
                                   (length, 5, 1024)).astype(np.float16).copy()
        np.save(directory / "features.npy", features)
        np.savez(directory / "frames.npz", **arrays)
        sources = {}
        for name, relative in (("video", f"videos/episode_{episode:06d}.mp4"),
                               ("parquet", f"data/episode_{episode:06d}.parquet")):
            source_path = root / relative
            source_path.parent.mkdir(exist_ok=True)
            source_path.write_bytes(f"fixture-{name}-{episode}".encode())
            sources[name] = {"path": relative, "identity": api.file_identity(source_path)}
        receipt = dict(contract=api.CONTRACT, state="complete", episode_index=episode,
                       manifest_signature=manifest["signature"], source=sources, frames=length,
                       signature=api.digest({"manifest_signature": manifest["signature"], "source": sources}),
                       outputs={name: api.file_identity(directory / name) for name in ("features.npy", "frames.npz")})
        api.atomic_json(directory / "receipt.json", receipt)
        columns["episode_index"].append(np.full(length, episode, dtype=np.int64))
        columns["frame_index"].append(arrays["frame_index"].copy())
        columns["timestamp"].append(arrays["timestamp"].astype(np.float32))
        columns["index"].append(np.arange(offset, offset + length, dtype=np.int64) + 100000)
        offset += length
    table = _FrameTable({name: np.concatenate(chunks) for name, chunks in columns.items()})
    pack = _Pack(hf_dataset=table, meta=SimpleNamespace(fps=30, info=info,
                           episodes={episode: {"length": length} for episode, length in enumerate(lengths)}))
    return root, cache_root, manifest, pack


def _index(modules, fixture, **kwargs):
    root, cache_root, _, pack = fixture
    return modules.history.MotionHistoryIndex(
        cache_root=cache_root, datasets=[pack], repo_ids=["source"],
        dataset_roots=[root], image_key=IMAGE_KEY, **kwargs)


def test_committed_native_features_remain_readonly_mmap(history_modules, tmp_path):
    root, cache_root, _, _ = _cache_pack(tmp_path, history_modules.cache)
    manifest = history_modules.cache.load_cache_manifest(cache_root, dataset_root=root)
    episode = history_modules.cache.load_episode_cache(cache_root, 0, manifest=manifest, dataset_root=root)
    assert isinstance(episode.features, np.memmap)
    assert episode.features.shape == (65, 5, 1024) and episode.features.dtype == np.float16
    assert not episode.features.flags.writeable
    np.testing.assert_array_equal(episode.frame_index, np.arange(65))
    np.testing.assert_allclose(episode.timestamp, np.arange(65) / 30, atol=1e-7)


def test_history_uses_six_real_5hz_times_and_keeps_current_native_phase(history_modules, tmp_path):
    fixture = _cache_pack(tmp_path, history_modules.cache)
    index = _index(history_modules, fixture)
    np.testing.assert_array_equal(index.original_indices, np.arange(30, 65))
    for filtered, native in ((0, 30), (1, 31), (34, 64)):
        history = index.get_history(filtered)
        assert history.shape == (6, 5, 1024)
        np.testing.assert_array_equal(history[:, 0, 0], native - np.arange(5, -1, -1) * 6)
    history = index.get_history(0)
    history[:] = -1
    assert index.get_history(0)[-1, 0, 0] == 30


def test_history_never_crosses_episode_or_borrows_for_short_episodes(history_modules, tmp_path):
    index = _index(history_modules, _cache_pack(tmp_path, history_modules.cache, lengths=(35, 20, 37)))
    np.testing.assert_array_equal(index.original_indices, np.r_[np.arange(30, 35), np.arange(85, 92)])
    np.testing.assert_array_equal(index.get_history(5)[:, 0, 0], 2000 + np.arange(0, 31, 6))


@pytest.mark.parametrize("break_kind", ["invalid_middle", "missing_index", "time_gap"])
def test_continuity_checks_every_native_frame_between_the_six_selected_times(history_modules, tmp_path, break_kind):
    def mutate(episode, arrays):
        if break_kind == "invalid_middle":
            arrays["valid"][15] = False
        elif break_kind == "missing_index":
            arrays["frame_index"][15:] += 1
            arrays["timestamp"][15:] += 1 / 30
            arrays["video_timestamp"][15:] += 1 / 30
        else:
            arrays["timestamp"][15:] += 0.2
            arrays["video_timestamp"][15:] += 0.2
    index = _index(history_modules, _cache_pack(tmp_path, history_modules.cache, mutate_frames=mutate))
    expected_first = 46 if break_kind == "invalid_middle" else 45
    np.testing.assert_array_equal(index.original_indices, np.arange(expected_first, 65))


@pytest.mark.parametrize("kind", ["profile", "weights", "normalization"])
def test_cache_rejects_different_psi0_profile_or_encoder_identity(history_modules, tmp_path, kind):
    api = history_modules.cache
    root, cache_root, manifest, _ = _cache_pack(tmp_path, api)
    if kind == "profile":
        with pytest.raises(ValueError, match="profile"):
            api.load_cache_manifest(cache_root, dataset_root=root, expected_profile="256x480")
        return
    if kind == "weights":
        manifest["encoder"]["local_file_sha256"]["model.safetensors"] = "c" * 64
    else:
        manifest["encoder"]["preprocessing"]["image_mean"][0] = 0
    manifest["signature"] = api.manifest_signature(manifest)
    api.atomic_json(cache_root / "manifest.json", manifest)
    with pytest.raises(ValueError, match="weights|normalization"):
        api.load_cache_manifest(cache_root, dataset_root=root)


@pytest.mark.parametrize("source", ["meta/info.json", "videos/episode_000000.mp4", "data/episode_000000.parquet"])
def test_cache_rejects_source_changes_before_serving_history(history_modules, tmp_path, source):
    api = history_modules.cache
    root, cache_root, _, _ = _cache_pack(tmp_path, api)
    path = root / source
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Stale|modified"):
        if source.startswith("meta/"):
            api.load_cache_manifest(cache_root, dataset_root=root)
        else:
            api.load_episode_cache(cache_root, 0, dataset_root=root)


def test_current_vlm_rgb_uses_identical_deterministic_shared_processor(history_modules, tmp_path):
    index = _index(history_modules, _cache_pack(tmp_path, history_modules.cache))
    rgb = np.random.default_rng(9).integers(0, 256, size=(73, 101, 3), dtype=np.uint8)
    original = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
    sample = {IMAGE_KEY: original, "unchanged": object()}

    result = index.process_current_rgb(sample)

    assert result is sample and result[IMAGE_KEY].shape == (3, 256, 320)
    actual = np.rint(result[IMAGE_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    expected = SharedImageProcessor("256x320").process_image(rgb)
    np.testing.assert_array_equal(actual, expected)
    torch.testing.assert_close(original, torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255)


def test_color_policy_reader_restores_raw_rgb_bytes_without_preprocessing(history_modules, tmp_path, monkeypatch):
    index = _index(history_modules, _cache_pack(tmp_path, history_modules.cache))
    rgb = np.random.default_rng(9).integers(0, 256, size=(73, 101, 3), dtype=np.uint8)
    original = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255

    def forbidden_preprocessing(*args, **kwargs):
        raise AssertionError("new color policy must defer geometry until the model transform")

    monkeypatch.setattr(index.processor, "process_image", forbidden_preprocessing)
    for current in (original, torch.nextafter(original, torch.zeros_like(original))):
        marker = object()
        sample = {IMAGE_KEY: current, "unchanged": marker}
        result = index.process_current_rgb(sample, apply_geometry=False)

        assert result is sample and result["unchanged"] is marker
        assert result[IMAGE_KEY].shape == (3, 73, 101)
        actual = np.rint(result[IMAGE_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, rgb)


def test_vlm_color_leaves_all_six_history_features_and_cache_bytes_clean(
    history_modules, actual_data_methods, tmp_path,
):
    fixture = _cache_pack(tmp_path, history_modules.cache)
    index = _index(history_modules, fixture)
    path = history_modules.cache.episode_directory(fixture[1], 0) / "features.npy"
    cache_bytes = path.read_bytes()
    history = index.get_history(0)
    clean_history = history.copy()
    np.testing.assert_array_equal(history[:, 0, 0], [0, 6, 12, 18, 24, 30])
    raw = np.random.default_rng(6).integers(0, 256, size=(73, 101, 3), dtype=np.uint8)
    current = torch.from_numpy(raw.copy()).permute(2, 0, 1).float() / 255
    sample = index.process_current_rgb({IMAGE_KEY: current}, apply_geometry=False)
    restored = np.rint(sample[IMAGE_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    model = _color_history_model(actual_data_methods)
    data = {**_processed_model_input(), "motion_img_embs": history,
            "observations": [Image.fromarray(restored)]}

    outputs = []
    for seed in (13, 14):
        torch.manual_seed(seed)
        output = model(data)
        outputs.append(np.asarray(output["observations"][0]))
        assert output["motion_img_embs"] is history
        np.testing.assert_array_equal(output["motion_img_embs"], clean_history)

    assert not np.array_equal(outputs[0], outputs[1])
    np.testing.assert_array_equal(index.get_history(0), clean_history)
    assert path.read_bytes() == cache_bytes


def test_worker_serialization_drops_open_mmaps_and_lru_stays_bounded(history_modules, tmp_path):
    index = _index(history_modules, _cache_pack(tmp_path, history_modules.cache, lengths=(65, 65)),
                   max_open_episodes=1)
    index.get_history(0)
    first = next(iter(index._episodes.values())).features
    assert isinstance(first, np.memmap)
    index.get_history(35)
    assert len(index._episodes) == 1 and first._mmap.closed
    serialized = pickle.dumps(index)
    assert len(serialized) < 100000
    restored = pickle.loads(serialized)
    assert not restored._episodes
    np.testing.assert_array_equal(restored.get_history(35), index.get_history(35))


def test_cache_config_and_psi0_model_transform_must_share_one_profile(history_modules, tmp_path):
    fixture = _cache_pack(tmp_path, history_modules.cache)
    with pytest.raises(ValueError, match="profile"):
        _index(history_modules, fixture, shared_image_profile="256x480")


def test_real_pyav_rgb_roundtrip_matches_offline_geometry_without_uint8_truncation(history_modules, tmp_path):
    import av

    path = tmp_path / "small.mp4"
    pixels = np.random.default_rng(4).integers(0, 256, (72, 96, 3), dtype=np.uint8)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("mpeg4", rate=30)
        stream.width, stream.height, stream.pix_fmt = 96, 72, "yuv420p"
        for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    with av.open(str(path)) as source:
        decoded = next(source.decode(video=0)).to_ndarray(format="rgb24")
    index = _index(history_modules, _cache_pack(tmp_path, history_modules.cache))
    expected = SharedImageProcessor("256x320").process_image(decoded)
    current = torch.from_numpy(decoded).permute(2, 0, 1).float().div(255)
    utility_path = Path(__file__).parents[1] / "src/psi/utils/utils.py"
    utility_tree = ast.parse(utility_path.read_text())
    pt_to_pil = copy.deepcopy(next(node for node in utility_tree.body
                                  if isinstance(node, ast.FunctionDef) and node.name == "pt_to_pil"))
    namespace = dict(Image=Image, np=np)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pt_to_pil], type_ignores=[])),
                 str(utility_path), "exec"), namespace)
    # Include a one-ULP representation change: round restores the decoded bytes,
    # whereas truncation would change many pixels before DINO/VLM branching.
    for image in (current, torch.nextafter(current, torch.zeros_like(current))):
        result = index.process_current_rgb({IMAGE_KEY: image})
        actual = np.rint(result[IMAGE_KEY].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        np.testing.assert_array_equal(actual, expected)
        # This is the real final conversion used by SonicRepackTransform before
        # Psi0 tokenization, not just a second rounding assertion in the test.
        pil = namespace["pt_to_pil"](result[IMAGE_KEY], normalized=False)
        np.testing.assert_array_equal(np.asarray(pil), expected)


@pytest.fixture(scope="module")
def actual_history_config_validator():
    path = Path(__file__).parents[1] / "src/psi/config/data_lerobot.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LerobotDataConfig")
    method = copy.deepcopy(next(node for node in cls.body
                                if isinstance(node, ast.FunctionDef) and node.name == "check_dino_history"))
    method.decorator_list = []
    model_class = type("Psi0ModelTransform", (), {})
    namespace = {"Psi0ModelTransform": model_class}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(path), "exec"), namespace)
    return namespace["check_dino_history"], model_class


def _config(validator):
    _, model_class = validator
    model = model_class()
    model.motion_image_policy = "shared_rgb"
    model.img_aug = model.view_aug = model.adaptive_resize = False
    model.shared_image_profile = None
    model.color_jitter = SimpleNamespace(brightness=0.2, contrast=(0.8, 1.2),
                                         saturation=(0.8, 1.2), hue=0.05)
    model.resize = SimpleNamespace(size=224)
    model.center_crop = SimpleNamespace(size=224)
    return SimpleNamespace(dino_cache_root="/cache", shared_image_profile="256x320", dino_max_open_episodes=2,
                           transform=SimpleNamespace(model=model, repack=SimpleNamespace(
                               num_past_frames=0, image_keys=[IMAGE_KEY])))


def test_actual_config_binds_cached_dino_and_psi0_to_one_shared_profile(actual_history_config_validator):
    validator, _ = actual_history_config_validator
    cfg = _config(actual_history_config_validator)
    assert validator(cfg) is cfg
    assert cfg.transform.model.shared_image_profile == cfg.shared_image_profile == "256x320"
    assert cfg.transform.model.resize.size == cfg.transform.model.center_crop.size == (256, 320)


def test_actual_config_allows_vlm_color_while_binding_the_clean_motion_profile(actual_history_config_validator):
    validator, _ = actual_history_config_validator
    cfg = _config(actual_history_config_validator)
    cfg.transform.model.motion_image_policy = "vlm_color_clean_motion"
    cfg.transform.model.img_aug = True

    assert validator(cfg) is cfg

    assert cfg.transform.model.img_aug
    assert cfg.transform.model.shared_image_profile == cfg.shared_image_profile == "256x320"
    assert cfg.transform.model.resize.size == cfg.transform.model.center_crop.size == (256, 320)
    for option in ("view_aug", "adaptive_resize"):
        setattr(cfg.transform.model, option, True)
        with pytest.raises(ValueError, match="geometry"):
            validator(cfg)
        setattr(cfg.transform.model, option, False)


@pytest.mark.parametrize("option", ["brightness", "contrast", "saturation", "hue"])
def test_actual_config_rejects_color_override_that_differs_from_imageprocess(actual_history_config_validator, option):
    validator, _ = actual_history_config_validator
    cfg = _config(actual_history_config_validator)
    cfg.transform.model.motion_image_policy = "vlm_color_clean_motion"
    cfg.transform.model.img_aug = True
    setattr(cfg.transform.model.color_jitter, option, 0.3)

    with pytest.raises(ValueError, match=rf"color_jitter\.{option}.*ImageProcess profile"):
        validator(cfg)


@pytest.mark.parametrize("setting", ["img_aug", "view_aug", "adaptive_resize", "profile", "past_frames"])
def test_actual_config_rejects_independent_geometry_or_noncurrent_images(actual_history_config_validator, setting):
    validator, _ = actual_history_config_validator
    cfg = _config(actual_history_config_validator)
    if setting == "profile":
        cfg.transform.model.shared_image_profile = "256x480"
    elif setting == "past_frames":
        cfg.transform.repack.num_past_frames = 1
    else:
        setattr(cfg.transform.model, setting, True)
    with pytest.raises(ValueError):
        validator(cfg)


@pytest.fixture(scope="module")
def actual_wrapper(history_modules):
    path = Path(__file__).parents[1] / "src/psi/data/lerobot/lerobot_ext.py"
    tree = ast.parse(path.read_text())
    cls = copy.deepcopy(next(node for node in tree.body
                             if isinstance(node, ast.ClassDef) and node.name == "LeRobotDatasetWrapper"))
    namespace = dict(__name__="_unifolm_history_contract_tests.wrapper",
                     __package__="_unifolm_history_contract_tests", torch=torch, resolve_path=lambda value: value)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls],
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["LeRobotDatasetWrapper"], namespace


@pytest.mark.parametrize("policy", [None, "shared_rgb", "vlm_color_clean_motion"], ids=["legacy", "shared", "color"])
def test_actual_wrapper_keeps_native_action_queries_and_preserves_legacy_path(
    history_modules, actual_wrapper, tmp_path, policy,
):
    cache_enabled = policy is not None
    root, cache_root, _, original_pack = _cache_pack(tmp_path, history_modules.cache, lengths=(35, 37))
    paths = [history_modules.cache.episode_directory(cache_root, episode) / "features.npy" for episode in (0, 1)]
    original_bytes = [path.read_bytes() for path in paths]

    class Repack:
        image_keys = [IMAGE_KEY]

        def delta_timestamps(self, fps):
            assert fps == 30
            return {"action": [0, 1 / fps]}

    class Pack(_Pack):
        def __getitem__(self, index):
            episode = self.hf_dataset["episode_index"][index]
            start, stop = self.episode_data_index["from"][episode], self.episode_data_index["to"][episode]
            return {"native_index": index, "action_query": [max(int(start), min(int(stop) - 1, index + delta))
                                                            for delta in (0, 1)],
                    "state": torch.tensor([index], dtype=torch.float32),
                    IMAGE_KEY: torch.zeros(3, 72, 96)}

    pack = Pack(**original_pack.__dict__)
    pack.episode_data_index = {"from": torch.tensor([5000, 6000]), "to": torch.tensor([5035, 6037])}
    wrapper_class, namespace = actual_wrapper
    calls = []

    def make_dataset(*args, **kwargs):
        calls.append(kwargs)
        return pack

    namespace.update(LeRobotDatasetMetadata=lambda *args: pack.meta, LerobotRepackTransform=Repack,
                     LeRobotDataset=make_dataset, MultiLeRobotDataset=make_dataset)
    cfg = SimpleNamespace(root_dir=str(tmp_path), train_repo_ids=["source"], val_repo_ids=["source"],
                          dino_cache_root=str(cache_root) if cache_enabled else None,
                          shared_image_profile="256x320", dino_max_open_episodes=2,
                          transform=SimpleNamespace(repack=Repack(), model=SimpleNamespace(
                              motion_image_policy=policy or "shared_rgb")))
    wrapper = wrapper_class(cfg)
    assert calls[-1]["delta_timestamps"] == {"action": [0, 1 / 30]}
    assert calls[-1]["image_transforms"] is None
    if cache_enabled:
        assert len(wrapper) == 12
        last_first_episode = wrapper[4]
        assert last_first_episode["native_index"] == 34
        assert last_first_episode["action_query"] == [34, 34]
        assert last_first_episode["state"].tolist() == [34]
        expected_hw = (72, 96) if policy == "vlm_color_clean_motion" else (256, 320)
        assert last_first_episode[IMAGE_KEY].shape == (3, *expected_hw)
        np.testing.assert_array_equal(last_first_episode["motion_img_embs"][:, 0, 0], [4, 10, 16, 22, 28, 34])
        assert wrapper[5]["native_index"] == 65
        assert wrapper[5]["action_query"] == [65, 66]
        assert hasattr(wrapper, "iter_episode")
        assert len(list(wrapper.iter_episode(0, dataset_index=0))) == 5
    else:
        assert len(wrapper) == 72
        assert wrapper[0][IMAGE_KEY].shape == (3, 72, 96)
        assert "motion_img_embs" not in wrapper[0]
        assert not hasattr(wrapper, "iter_episode")
        assert pack.episode_data_index["from"].tolist() == [5000, 6000]
    assert [path.read_bytes() for path in paths] == original_bytes
