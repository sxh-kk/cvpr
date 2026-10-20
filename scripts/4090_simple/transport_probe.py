"""Cross-environment HTTP/NumPy transport fixture, NOT a model/history service."""
import argparse
from pathlib import Path
import sys


def server(port):
    import numpy as np
    import uvicorn
    from fastapi import FastAPI
    from psi.deploy.helpers import RequestMessage, ResponseMessage

    app = FastAPI()
    counters = {}

    @app.get("/health")
    def health():
        return {"fixture": True, "model_loaded": False, "python": sys.version}

    @app.post("/act")
    def act(payload: dict):
        req = RequestMessage.deserialize(payload)
        session = req.history["session_id"]
        if req.history.get("reset"):
            counters.pop(session, None)
        counters[session] = counters.get(session, 0) + 1
        assert req.state["states"].shape == (1, 45)
        assert req.image["rgb_head_stereo_left"].shape == (240, 320, 3)
        # Counter markers prove serialization/reset/isolation; these are never robot actions.
        action = np.full((30, 80), counters[session], dtype=np.float32)
        return ResponseMessage(action, 0., req.image["rgb_head_stereo_left"]).serialize()

    uvicorn.run(app, host="127.0.0.1", port=port)


def client(port, image):
    import numpy as np
    from PIL import Image
    from simple.baselines.client import HttpActionClient

    rgb = np.array(Image.open(image).convert("RGB"))
    cli = HttpActionClient("127.0.0.1", port)
    for session, reset, expected in [("a", True, 1), ("a", False, 2),
                                     ("b", True, 1), ("a", False, 3), ("a", True, 1)]:
        history = {"session_id": session}
        if reset:
            history["reset"] = True
        action, err, received_rgb = cli.query_action(
            {"rgb_head_stereo_left": rgb}, "transport fixture only",
            {"states": np.zeros((1, 45), dtype=np.float32)}, {},
            history=history, dataset="simple")
        assert action.shape == (30, 80) and action.dtype == np.float32
        assert (action == expected).all() and err == 0
        assert np.array_equal(rgb, received_rgb)
    print(f"PASS: {sys.executable}: real camera RGB roundtrip; 30x80 fixture; reset/isolation.")
    print("Transport fixture only. Production dual-source service/reset/history remain unvalidated.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["server", "client"])
    p.add_argument("--port", type=int, default=18765)
    p.add_argument("--image", type=Path)
    args = p.parse_args()
    if args.mode == "server":
        server(args.port)
    else:
        if args.image is None:
            p.error("client requires --image from a real camera probe")
        client(args.port, args.image)
