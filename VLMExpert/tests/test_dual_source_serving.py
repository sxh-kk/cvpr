"""CPU protocol regressions; expert doubles do not claim GPU/model acceptance."""
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np
import torch
from fastapi.testclient import TestClient

from ImageProcess import SharedImageProcessor
from egomotion_pipeline.contracts import dino_format
from egomotion_pipeline.features import DINO_PREPROCESSING
from egomotion_pipeline.io import file_sha256
from psi.deploy.dual_source_runtime import (
    HistoryError, HistoryWarmingUp, LocalDinoEncoder, SimulationHistory, load_dino_contract,
)
from psi.deploy.helpers import RequestMessage, ResponseMessage
from psi.deploy.serve_psi0_simple import Server


def raises(exception, *, match=None):
    case = unittest.TestCase()
    return case.assertRaisesRegex(exception, match) if match else case.assertRaises(exception)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def metadata(t=0., *, session="first", episode="episode1", **extra):
    return dict(session_id=session, episode_id=episode, sim_time_s=t, **extra)


def image(index=0):
    return np.full((32, 48, 3), index, dtype=np.uint8)


def wire(t=0., *, session="first", episode="episode1", **extra):
    return RequestMessage(
        image={"observation.images.egocentric": image(round(t * 30) % 256)},
        instruction="Pick Up Object", history=metadata(t, session=session, episode=episode, **extra),
        state={"states": np.arange(43, dtype=np.float32)}, condition={}, gt_action=[],
        dataset_name="simple", timestamp="unused_wall_clock",
    ).serialize()


def fill(history, *, session="first", episode="episode1", start=0., count=31):
    for index in range(count):
        history.observe(metadata(start + index / 30, session=session, episode=episode), image(index))


class FakeDino:
    contract = {"shared_image_profile": "256x320", "encoder": {"weights_sha256": "test-double"}}

    def __init__(self):
        self.snapshots = []

    def encode(self, snapshot):
        self.snapshots.append(snapshot)
        return np.stack([np.full((5, 1024), rgb[0, 0, 0], dtype=np.float32) for rgb in snapshot.rgb])


class FakePolicy:
    def __init__(self):
        self.calls = []
        self.entered = None
        self.release = None

    def predict_action(self, **kwargs):
        assert not torch.is_grad_enabled()
        self.calls.append(("plain", kwargs))
        if self.entered is not None:
            self.entered.set()
            assert self.release.wait(5), "test did not release in-flight policy"
        return torch.ones((1, 30, 80))

    def predict_action_with_rtc_flow(self, **kwargs):
        assert not torch.is_grad_enabled()
        # Real PiG RTC must be able to locally enable an action-sample VJP.
        with torch.enable_grad():
            samples = torch.ones(2, requires_grad=True)
            torch.testing.assert_close(torch.autograd.grad(samples.square().sum(), samples)[0],
                                       torch.full((2,), 2.))
        self.calls.append(("rtc", kwargs))
        return torch.ones((1, 30, 80)) * 2


def make_server():
    server = Server.__new__(Server)  # never construct/load a real model in CPU regressions
    server.uses_dual_source = True
    server.motion_history = SimulationHistory()
    server.online_dino = FakeDino()
    server.model = FakePolicy()
    server.device = torch.device("cpu")
    server.maxmin = SimpleNamespace(normalize_state=False, denormalize=lambda value: value + 10)
    server.pad_state_dim, server.To, server.Tp, server.Da, server.Ta = 45, 1, 30, 80, 10
    server.enable_rtc, server.rtc_mode = True, "test_time"
    server.exec_shift, server.min_exec_horizon, server.rtc_inference_delay = 10, 10, 0
    server.pig_mask_schedule, server.pig_guidance_alpha = "exponential", .9
    server.rtc_max_delay, server.num_inference_steps = 8, 1
    server._infer_lock = threading.Lock()
    server.last_serve_time = time.monotonic()
    server.policy, server.run_dir, server.ckpt_step = "dual-source", Path("/local/joint20k"), 20000
    server.model_transform = SimpleNamespace(resize=SimpleNamespace(size=[256, 320]),
                                            center_crop=SimpleNamespace(size=[256, 320]))
    server.image_keys = ["observation.images.egocentric"]
    return server


def observe_episode(client, *, session="first", episode="episode1", start=0.):
    for index in range(31):
        response = client.post("/observe", json=wire(start + index / 30, session=session, episode=episode))
        assert response.status_code == 200, response.text


def contract_fixture(tmp_path):
    checkpoint = tmp_path / "dino"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"test-only-weight-identity")
    (checkpoint / "config.json").write_text("{}")
    hashes = {path.name: file_sha256(path) for path in checkpoint.iterdir()}
    return {
        "online_checkpoint": str(checkpoint), "shared_image_profile": "256x320",
        "encoder": {**dino_format(), "model_id": "facebook/dinov2-with-registers-large",
                    "revision": "local", "local_file_sha256": hashes, "weights_sha256": digest(hashes),
                    "image_process": SharedImageProcessor("256x320").metadata,
                    "preprocessing": DINO_PREPROCESSING},
        "history": {"fps": 5, "frames": 6, "shape": [6, 5, 1024],
                    "augmentation": "none", "timestamps": "simulation_time"},
        "processor_overrides_required": {"use_fast": False, "do_resize": False, "do_center_crop": False},
    }


class TestDualSourceServing(unittest.TestCase):
    def test_history_uses_six_real_causal_simulation_times_and_one_clean_geometry(self):
        history = SimulationHistory()
        fill(history)
        snapshot = history.snapshot(metadata(1.))
        np.testing.assert_allclose([frame.sim_time_s for frame in snapshot.observations],
                                   [0., .2, .4, .6, .8, 1.], atol=1e-8)
        assert [int(rgb[0, 0, 0]) for rgb in snapshot.rgb] == [0, 6, 12, 18, 24, 30]
        assert all(rgb.shape == (256, 320, 3) and not rgb.flags.writeable for rgb in snapshot.rgb)
        duplicate = history.observe(metadata(1.), image(30))
        assert duplicate["duplicate"] and duplicate["history_frames"] == 31
        with raises(HistoryError, match="different RGB"):
            history.observe(metadata(1.), image(31))
        with raises(HistoryError, match="different RGB"):
            history.observe(metadata(1.), image(30).reshape(16, 96, 3))

    def test_missing_history_does_not_repeat_a_frame_or_use_future_observations(self):
        history = SimulationHistory()
        history.observe(metadata(0.), image())
        with raises(HistoryWarmingUp) as error:
            history.snapshot(metadata(0.))
        assert error.exception.status_code == 425
        assert error.exception.detail["status"] == "warming_up"
        # Six requests alone do not supply a second of history.
        for index in range(1, 6):
            history.observe(metadata(index / 30), image(index))
        with raises(HistoryWarmingUp):
            history.snapshot(metadata(5 / 30))
        # A future frame does not fill the .4 s hole.
        for timestamp in [.2, .45, .6, .8, 1.]:
            history.observe(metadata(timestamp), image(round(timestamp * 30)))
        with raises(HistoryWarmingUp) as error:
            history.snapshot(metadata(1.))
        assert .4 in error.exception.detail["missing_sim_times_s"]

    def test_reset_isolates_history_rtc_and_invalidates_inflight_generation(self):
        history = SimulationHistory()
        fill(history)
        old = history.snapshot(metadata(1.))
        history.commit_action(old, np.ones((30, 80), dtype=np.float32))
        history.reset(metadata(episode="episode2"))
        with raises(HistoryError, match="reset during"):
            history.commit_action(old, np.zeros((30, 80), dtype=np.float32))
        with raises(HistoryError, match="Episode changed"):
            history.observe(metadata(1.1), image())
        fill(history, episode="episode2")
        new = history.snapshot(metadata(1., episode="episode2"))
        assert new.previous_action is None and new.generation != old.generation
        fill(history, session="second")
        assert history.snapshot(metadata(1., session="second")).previous_action is None

    def test_history_rejects_bad_identity_and_simulator_times(self):
        for extra in [dict(sim_time_s=True), dict(sim_time_s=float("inf")), dict(sim_time_s=-1),
                      dict(session_id=""), dict(episode_id=None)]:
            values = metadata()
            values.update(extra)
            with raises(HistoryError):
                SimulationHistory().observe(values, image())

    def test_real_http_wire_warms_up_then_routes_shared_current_rgb_dino_state_and_rtc(self):
        server = make_server()
        with TestClient(server.create_app()) as client:
            assert client.get("/info").json()["online_history"]["warming_up_http_status"] == 425
            first = client.post("/act", json=wire())
            assert first.status_code == 425 and first.json()["detail"]["status"] == "warming_up"
            assert not server.model.calls
            observe_episode(client)
            result = client.post("/act", json=wire(1.))
            assert result.status_code == 200, result.text
            action = ResponseMessage.deserialize(result.json()).action
            np.testing.assert_array_equal(action, np.full((10, 80), 11., dtype=np.float32))
            mode, arguments = server.model.calls[-1]
            assert mode == "plain" and arguments["instructions"] == ["pick up object"]
            assert arguments["motion_img_embs"].shape == (1, 6, 5, 1024)
            assert arguments["states"].shape == (1, 1, 45)
            torch.testing.assert_close(arguments["states"][0, 0, :43], torch.arange(43, dtype=torch.float32))
            assert not arguments["states"][0, 0, 43:].any()
            np.testing.assert_array_equal(np.asarray(arguments["observations"][0][0]),
                                          server.online_dino.snapshots[-1].rgb[-1])
            assert client.post("/act", json=wire(1.)).status_code == 409
            # Observe throughout execution; reset=false must preserve this session's RTC.
            for index in range(31, 37):
                assert client.post("/observe", json=wire(index / 30)).status_code == 200
            result = client.post("/act", json=wire(1.2, reset=False))
            assert result.status_code == 200, result.text
            mode, arguments = server.model.calls[-1]
            assert mode == "rtc"
            torch.testing.assert_close(arguments["prev_actions"][0, :20], torch.ones(20, 80))
            assert not arguments["prev_actions"][0, 20:].any()  # raw normalized plan, not +10 output
            assert client.post("/reset", json=metadata(episode="episode2")).json()["rtc_cleared"]
            assert client.post("/act", json=wire(episode="episode2")).status_code == 425
            assert client.post("/observe", json=wire(1.3)).status_code == 409

    def test_observation_and_reset_do_not_wait_for_policy_lock_and_discard_old_episode_action(self):
        server = make_server()
        server.model.entered, server.model.release = threading.Event(), threading.Event()
        with TestClient(server.create_app()) as client, ThreadPoolExecutor(max_workers=1) as pool:
            observe_episode(client)
            pending = pool.submit(client.post, "/act", json=wire(1.))
            try:
                assert server.model.entered.wait(5)
                assert client.post("/observe", json=wire(31 / 30)).status_code == 200
                assert client.post("/reset", json=metadata(episode="episode2")).status_code == 200
            finally:
                server.model.release.set()
            result = pending.result(timeout=5)
            assert result.status_code == 409 and "reset during" in result.json()["detail"]["message"]

    def test_two_http_sessions_never_share_a_previous_action(self):
        server = make_server()
        with TestClient(server.create_app()) as client:
            for session in ("first", "second"):
                observe_episode(client, session=session)
                result = client.post("/act", json=wire(1., session=session))
                assert result.status_code == 200, result.text
                assert server.model.calls[-1][0] == "plain"
            # First client still has its own committed plan after the second call.
            for index in range(31, 37):
                assert client.post("/observe", json=wire(index / 30)).status_code == 200
            result = client.post("/act", json=wire(1.2))
            assert result.status_code == 200, result.text
            assert server.model.calls[-1][0] == "rtc"

    def test_http_invalid_numpy_request_or_state_is_rejected(self):
        server = make_server()
        with TestClient(server.create_app()) as client:
            assert client.post("/observe", json={"image": {}, "history": metadata()}).status_code == 422
            request = wire()
            request["image"]["observation.images.egocentric"]["shape"] = [2, 2, 3]
            assert client.post("/observe", json=request).status_code == 422
            request = wire()
            request["state"] = {"states": np.ones(45, dtype=np.float32)}
            from psi.deploy.helpers import convert_numpy_in_dict, numpy_serialize
            assert client.post("/act", json=convert_numpy_in_dict(request, numpy_serialize)).status_code == 422
            request["state"] = {}
            assert client.post("/act", json=convert_numpy_in_dict(request, numpy_serialize)).status_code == 422

    def test_local_dino_contract_rejects_changed_weights_extra_model_files_and_preprocessing(self):
        tmp_path = Path(self.enterContext(tempfile.TemporaryDirectory()))
        contract = contract_fixture(tmp_path)
        path = tmp_path / "contract.json"
        path.write_text(json.dumps(contract))
        assert load_dino_contract(path)["encoder"]["weights_sha256"] == contract["encoder"]["weights_sha256"]
        changed = copy.deepcopy(contract)
        changed["encoder"]["preprocessing"]["do_resize"] = True
        path.write_text(json.dumps(changed))
        with raises(ValueError, match="preprocessing differs"):
            load_dino_contract(path)
        path.write_text(json.dumps(contract))
        extra = Path(contract["online_checkpoint"]) / "unexpected.bin"
        extra.write_bytes(b"another-model")
        with raises(ValueError, match="outside the pinned identity"):
            load_dino_contract(path)
        extra.unlink()
        (Path(contract["online_checkpoint"]) / "model.safetensors").write_bytes(b"changed")
        with raises(ValueError, match="SHA256 differs"):
            load_dino_contract(path)

    def test_local_dino_encoder_passes_already_prepared_geometry_without_another_resize(self):
        history = SimulationHistory()
        fill(history)
        snapshot = history.snapshot(metadata(1.))
        encoder = LocalDinoEncoder.__new__(LocalDinoEncoder)

        class Backend:
            def prepare(self, images, *, processed):
                assert processed is True
                assert all(left is right for left, right in zip(images, snapshot.rgb))
                return images

            def encode_prepared(self, inputs):
                return np.zeros((6, 5, 1024), dtype=np.float32)

        encoder.encoder = Backend()
        assert encoder.encode(snapshot).shape == (6, 5, 1024)


if __name__ == "__main__":
    unittest.main()
