"""CPU stubs for the production-server RNG audit wrapper."""
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from fastapi import HTTPException
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi.responses import JSONResponse

spec = importlib.util.spec_from_file_location("serve_evaluation", Path(__file__).resolve().parents[1] / "serve_evaluation.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


class Stub:
    def __init__(self, draws):
        random.random()
        np.random.rand(draws)
        torch.rand(draws)
        self.noises = []
    def predict_action(self, payload, request):
        self.noises.append((random.random(), float(np.random.rand()), torch.rand(3).tolist()))
        if payload.get("fail"):
            raise HTTPException(425, "warming_up")
        return JSONResponse({"ok": True})
    def observe(self, payload):
        return JSONResponse({"observed": True})
    def reset(self, payload):
        return JSONResponse({"reset": True})


class AuditContracts(unittest.TestCase):
    def test_actual_constructed_dino_runtime_metadata_is_saved(self):
        metadata = {'torch_version': 'test-runtime', 'transformers_version': 'test-transformers',
                    'weights_sha256': 'actual-loaded-weight-identity'}
        def factory():
            server = Stub(1)
            server.online_dino = SimpleNamespace(encoder=SimpleNamespace(metadata=metadata))
            return server
        with tempfile.TemporaryDirectory() as root:
            module.audited_server(factory, 7, seed, Path(root), 'source')
            receipt = json.loads((Path(root) / 'startup.json').read_text())
            self.assertEqual(receipt['online_dino_runtime'], metadata)

    def test_wrapper_registers_production_http_signature(self):
        with tempfile.TemporaryDirectory() as root:
            server = module.audited_server(lambda: Stub(1), 7, seed, Path(root), "source")
            app = FastAPI()
            app.post("/act")(server.predict_action)
            app.post("/reset")(server.reset)
            with TestClient(app) as client:
                identity = {"session_id": "s", "episode_id": "a"}
                self.assertEqual(client.post("/reset", json=identity).status_code, 200)
                self.assertEqual(client.post("/act", json={"history": identity}).status_code, 200)
                self.assertEqual(client.post("/act", json={"history": identity, "fail": True}).status_code, 425)

    def test_post_construction_seed_equalizes_first_noise_and_snapshot_is_pure(self):
        with tempfile.TemporaryDirectory() as root:
            servers = [module.audited_server(lambda: Stub(draws), 20261021, seed, Path(root) / str(draws), "source")
                       for draws in (1, 100)]
            snapshots = [json.loads((Path(root) / str(draws) / "startup.json").read_text())["rng_sha256"] for draws in (1, 100)]
            self.assertEqual(snapshots[0], snapshots[1])
            before = module.rng_sha()
            self.assertEqual(before, module.rng_sha())
            payload = {"history": {"session_id": "s", "episode_id": "a", "sim_time_s": 1.0}}
            seed(20261021)
            servers[0].predict_action(payload, None)
            seed(20261021)
            servers[1].predict_action(payload, None)
            self.assertEqual(servers[0].noises, servers[1].noises)

    def test_exception_ordinal_rng_and_cross_episode_rejection(self):
        with tempfile.TemporaryDirectory() as root:
            server = module.audited_server(lambda: Stub(3), 7, seed, Path(root), "source")
            history = {"session_id": "s", "episode_id": "a", "sim_time_s": 1.0}
            server.reset(history)
            with self.assertRaises(HTTPException) as error:
                server.predict_action({"history": history, "fail": True}, None)
            self.assertEqual(error.exception.status_code, 425)
            before_rejection = module.rng_sha()
            with self.assertRaises(HTTPException) as error:
                server.predict_action({"history": {**history, "episode_id": "b"}}, None)
            self.assertEqual(error.exception.status_code, 409)
            self.assertEqual(before_rejection, module.rng_sha())
            with self.assertRaises(HTTPException):
                server.reset({**history, "episode_id": "b"})
            with self.assertRaises(HTTPException):
                server.observe({"history": {**history, "episode_id": "b"}})
            rows = [json.loads(line) for line in (Path(root) / "requests.jsonl").read_text().splitlines()]
            self.assertEqual([row["ordinal"] for row in rows], [1, 2])
            self.assertEqual([row["status"] for row in rows], [425, 409])
            self.assertNotEqual(rows[0]["rng_before"], rows[0]["rng_after"])
            self.assertEqual(rows[1]["rng_before"], rows[1]["rng_after"])


if __name__ == "__main__":
    unittest.main()
