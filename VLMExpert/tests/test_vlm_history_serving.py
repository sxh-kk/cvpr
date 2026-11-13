"""Actual HTTP history/sampler routes with CPU experts; no pretrained model or GPU."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import hashlib
import importlib.util
import json
import sys
import tempfile
import threading

import numpy as np
from PIL import Image
import unittest
import torch
from fastapi.testclient import TestClient

from psi.deploy.dual_source_runtime import (
    SimulationHistory, HistoryError, HistoryWarmingUp, validate_vlm_history_frames, vlm_history_contract,
)
from test_dual_source_serving import make_server, observe_episode, wire, fill, metadata

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/4090_simple"))
import run_closed_loop as client_protocol

spec = importlib.util.spec_from_file_location("history6_serve_evaluation", REPO / "scripts/4090_simple/serve_evaluation.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def six_server():
    server = make_server()
    server.vlm_history_frames = 6
    server.vlm_history_metadata = vlm_history_contract(6)
    return server


class TestVlmHistoryServing(unittest.TestCase):
    def test_strict_checkpoint_frame_count_defaults_are_one_or_six_only(self):
        assert validate_vlm_history_frames(1) == 1
        assert validate_vlm_history_frames(6) == 6
        for value in (None, True, False, 6., "6", 0, 5, 7):
            with self.assertRaisesRegex(ValueError, "integer1 or6"):
                validate_vlm_history_frames(value)


    def test_snapshot_has_real_ordered_six_and_current_last_with_shared_identity(self):
        history = SimulationHistory()
        fill(history)
        snapshot = history.snapshot(metadata(1.))
        images = snapshot.vlm_rgb(6)
        assert all(left is right for left, right in zip(images, snapshot.rgb))
        assert snapshot.vlm_rgb(1)[0] is images[-1]
        receipt = snapshot.vlm_receipt(6)
        np.testing.assert_allclose(receipt["selected_sim_times_s"], [0., .2, .4, .6, .8, 1.], atol=1e-8)
        np.testing.assert_allclose(receipt["target_sim_times_s"], [0., .2, .4, .6, .8, 1.], atol=1e-8)
        assert [int(rgb[0, 0, 0]) for rgb in images] == [0, 6, 12, 18, 24, 30]
        assert receipt["current_index"] == 5 and receipt["same_snapshot_as_motion"] is True
        assert receipt["source_rgb_sha256"] == [frame.source_sha256 for frame in snapshot.observations]
        assert receipt["processed_rgb_sha256"] == [hashlib.sha256(rgb.tobytes()).hexdigest() for rgb in images]
        history.reset(metadata(episode="next"))
        with self.assertRaises(HistoryWarmingUp):
            history.observe(metadata(0., episode="next"), images[-1])
            history.snapshot(metadata(0., episode="next"))
        with self.assertRaisesRegex(HistoryError, "reset during"):
            history.commit_action(snapshot, np.zeros((30, 80), dtype=np.float32))


    def test_actual_http_six_rgb_plain_then_rtc_share_dino_snapshot_without_padding(self):
        server = six_server()
        with TestClient(server.create_app()) as client:
            assert client.get("/info").json()["vlm_history"] == vlm_history_contract(6)
            assert client.post("/act", json=wire()).status_code == 425
            assert not server.model.calls
            observe_episode(client)
            assert client.post("/act", json=wire(1.)).status_code == 200
            mode, arguments = server.model.calls[-1]
            assert mode == "plain" and len(arguments["observations"][0]) == 6
            for actual, rgb in zip(arguments["observations"][0], server.online_dino.snapshots[-1].rgb):
                np.testing.assert_array_equal(np.asarray(actual), rgb)
            assert arguments["motion_img_embs"][0, :, 0, 0].tolist() == [0, 6, 12, 18, 24, 30]
            assert server.last_policy_history["episode_id"] == "episode1"
            assert client.post("/act", json=wire(1.)).status_code == 409
            for index in range(31, 37):
                assert client.post("/observe", json=wire(index / 30)).status_code == 200
            assert client.post("/act", json=wire(1.2, reset=False)).status_code == 200
            mode, arguments = server.model.calls[-1]
            assert mode == "rtc" and len(arguments["observations"][0]) == 6
            assert [np.asarray(value)[0, 0, 0] for value in arguments["observations"][0]] == [6, 12, 18, 24, 30, 36]
            torch.testing.assert_close(arguments["prev_actions"][0, :20], torch.ones(20, 80))
            assert client.post("/reset", json=metadata(episode="next")).status_code == 200
            assert client.post("/act", json=wire(episode="next")).status_code == 425
            assert client.post("/observe", json=wire(1.3)).status_code == 409


    def test_single_frame_legacy_server_still_sends_only_current_image(self):
        server = make_server()  # Missing field retains historical default1.
        with TestClient(server.create_app()) as client:
            assert client.get("/info").json()["vlm_history"]["frames"] == 1
            observe_episode(client)
            assert client.post("/act", json=wire(1.)).status_code == 200
        images = server.model.calls[-1][1]["observations"][0]
        assert len(images) == 1
        np.testing.assert_array_equal(np.asarray(images[0]), server.online_dino.snapshots[-1].rgb[-1])


    def test_six_frame_reset_during_inference_discards_old_episode_action(self):
        server = six_server()
        server.model.entered, server.model.release = threading.Event(), threading.Event()
        with TestClient(server.create_app()) as client, ThreadPoolExecutor(max_workers=1) as pool:
            observe_episode(client)
            pending = pool.submit(client.post, "/act", json=wire(1.))
            try:
                assert server.model.entered.wait(5)
                assert client.post("/reset", json=metadata(episode="next")).status_code == 200
            finally:
                server.model.release.set()
            result = pending.result(timeout=5)
            assert result.status_code == 409 and "reset during" in result.json()["detail"]["message"]
            assert client.post("/act", json=wire(episode="next")).status_code == 425


    def test_audit_records_actual_six_images_and_rejects_replicated_current_rgb(self):
        def factory():
            server = six_server()
            server.model.parameters = lambda: ()
            server.online_dino.encoder = SimpleNamespace(metadata={"weights_sha256": "CPU-expert"})
            return server
        with tempfile.TemporaryDirectory() as directory:
            server = audit.audited_server(factory, 7, lambda value: None, directory, "CPU-weight-fixture")
            startup = json.loads((Path(directory) / "startup.json").read_text())
            assert startup["vlm_history_frames"] == 6 and startup["vlm_history"] == vlm_history_contract(6)
            with TestClient(server.create_app()) as client:
                observe_episode(client)
                assert client.post("/act", json=wire(1.)).status_code == 200
            rows = [json.loads(line) for line in (Path(directory) / "requests.jsonl").read_text().splitlines()]
            actual = rows[-1]["vlm_observations"]
            assert rows[-1]["vlm_history_frames"] == 6 and actual[0]["image_counts"] == [6]
            np.testing.assert_allclose(actual[0]["history"]["selected_sim_times_s"], [0., .2, .4, .6, .8, 1.], atol=1e-8)
            current = server.online_dino.snapshots[-1].rgb[-1]
            with self.assertRaisesRegex(ValueError, "images differ"):
                server.model.predict_action(observations=[[Image.fromarray(current)] * 6])


    def test_client_requires_explicit_matching_six_frame_contract_and_preserves_old_default(self):
        args = SimpleNamespace(rtc_mode="test_time", execute_steps=15, rtc_inference_delay=0,
            min_exec_horizon=None, pig_mask_schedule="exponential", pig_guidance_alpha=.9,
            expected_checkpoint_step=5000)
        info = dict(client_protocol.requested_rtc_contract(args), ckpt_step=5000,
            action=dict(action_dim=80, action_chunk_size=30, action_exec_horizon=15), online_history={"frames": 6})
        assert client_protocol.validate_server_info(info, args)["status"] == "passed"
        args.expected_vlm_history_frames = 6
        with self.assertRaisesRegex(ValueError, "VLM history differs"):
            client_protocol.validate_server_info(info, args)
        info["vlm_history"] = vlm_history_contract(6)
        assert client_protocol.validate_server_info(info, args)["status"] == "passed"
        for key, value in (("frames", 1), ("order", "current_to_oldest"), ("padding_or_replication", True),
                           ("offsets_s", [0.] * 6), ("source", "previous_episode")):
            bad = {**info, "vlm_history": {**info["vlm_history"], key: value}}
            with self.assertRaises(ValueError):
                client_protocol.validate_server_info(bad, args)


if __name__ == "__main__":
    unittest.main()
