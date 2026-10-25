"""Loopback-only HTTP tests; no SDK, model, or simulation is constructed."""
import http.client
import io
import json
from pathlib import Path
import sys
import threading
import socket
import struct
import unittest
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pico_teleop_ui import COMMANDS, HeadsetVideoBridge, LocalTeleopUI


class LocalUITests(unittest.TestCase):
    def setUp(self):
        self.ui = LocalTeleopUI(port=0).start()
        self.address = urlsplit(self.ui.url)

    def tearDown(self):
        self.ui.close()

    def request(self, method, path, body=None, headers=None):
        client = http.client.HTTPConnection(self.address.hostname, self.address.port, timeout=3)
        try:
            client.request(method, path, body=body, headers=headers or {})
            response = client.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            client.close()

    def post(self, command, **headers):
        return self.request("POST", "/command", json.dumps({"command": command}),
                            {"Content-Type": "application/json", **headers})

    def test_commands_are_only_queued_in_order_and_reset_is_supported(self):
        for command in sorted(COMMANDS):
            code, headers, body = self.post(command, Origin=self.ui.url.rstrip("/"))
            self.assertEqual(code, 202)
            self.assertFalse(json.loads(body)["executed"])
            self.assertFalse(json.loads(body)["accepted_new_dataset"])
        self.assertEqual(self.ui.commands(), sorted(COMMANDS))
        self.assertEqual(self.ui.commands(), [])
        self.assertEqual(self.post("unknown")[0], 400)
        code, _, _ = self.request("POST", "/commands", '{"command":"reset"}', {"Content-Type": "application/json"})
        self.assertEqual(code, 202)
        self.assertEqual(self.ui.commands(), ["reset"])

    def test_cross_origin_host_rebinding_and_non_json_posts_are_rejected(self):
        for origin in ("https://example.org", "null", self.ui.url, "http://localhost:1",
                       f"http://localhost.evil:{self.address.port}"):
            self.assertEqual(self.post("record", Origin=origin)[0], 403)
        self.assertEqual(self.post("record", Host="attacker.example")[0], 403)
        self.assertEqual(self.post("record", **{"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(self.request("POST", "/command", 'command=record', {"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("OPTIONS", "/command")[0], 403)
        self.assertEqual(self.ui.commands(), [])
        self.assertEqual(self.post("follow", Origin=f"http://localhost:{self.address.port}")[0], 202)

    def test_update_copies_status_and_rgb_without_running_jpeg_encoder(self):
        calls = []
        real_encode = self.ui._encode_jpeg
        def observed_encode(pixels):
            calls.append(threading.get_ident())
            return real_encode(pixels)
        self.ui._encode_jpeg = observed_encode
        status = {"state": {"status": "recording"}, "error": ["raw"], "position": np.array([1., 2.])}
        rgb = np.zeros((24, 32, 3), dtype=np.uint8); rgb[..., 0] = 255
        self.ui.update(status, rgb)
        self.assertEqual(calls, [])
        rgb[:] = 0; status["state"]["status"] = "changed"; status["error"].append("changed"); status["position"][:] = 9
        code, _, body = self.request("GET", "/status.json")
        value = json.loads(body)
        self.assertEqual(code, 200)
        self.assertEqual(value["status"]["state"]["status"], "recording")
        self.assertEqual(value["status"]["position"], [1., 2.])
        self.assertEqual(value["status"]["error"], ["raw"])
        self.assertFalse(value["accepted_new_dataset"])
        self.assertTrue(value["image_available"])
        code, headers, body = self.request("GET", "/latest.jpg")
        self.assertEqual(code, 200); self.assertEqual(headers["Content-Type"], "image/jpeg")
        pixels = np.asarray(Image.open(io.BytesIO(body)))
        self.assertGreater(pixels[..., 0].mean(), 245)
        self.assertLess(pixels[..., 1].mean(), 10)
        self.assertEqual(len(calls), 1)
        self.assertNotEqual(calls[0], threading.get_ident())
        self.request("GET", "/latest.jpg?repeat=1")
        self.assertEqual(len(calls), 1)
        self.ui.update({"status": "paused"})
        self.assertEqual(json.loads(self.request("GET", "/status.json")[2])["image_version"], 1)

    def test_no_fake_image_bounded_commands_and_graceful_close(self):
        self.assertEqual(self.request("GET", "/latest.jpg")[0], 404)
        code, headers, body = self.request("GET", "/")
        self.assertEqual(code, 200)
        self.assertIn("待独立验收", body.decode())
        self.assertIn("frame-ancestors 'none'", headers["Content-Security-Policy"])
        for _ in range(64):
            self.assertEqual(self.post("pause")[0], 202)
        self.assertEqual(self.post("record")[0], 503)
        self.assertEqual(len(self.ui.commands()), 64)
        thread = self.ui._thread
        self.ui.close()
        self.assertFalse(thread.is_alive())
        self.ui.close()
        with self.assertRaises(RuntimeError): self.ui.update({})
        with self.assertRaises(RuntimeError): self.ui.start()

    def test_non_loopback_hosts_and_invalid_frames_are_rejected(self):
        for host in ("0.0.0.0", "172.16.20.42", "example.com", "::"):
            with self.assertRaises(ValueError): LocalTeleopUI(host=host)
        with self.assertRaises(ValueError): self.ui.update({}, np.zeros((5, 5), dtype=np.uint8))
        with self.assertRaises(ValueError): self.ui.update({}, np.zeros((5, 5, 3), dtype=np.float32))
        self.ui.update({"loss": float("nan")})
        self.assertIsNone(json.loads(self.request("GET", "/status.json")[2])["status"]["loss"])


def receive_exact(sock, length):
    result = b""
    while len(result) < length:
        block = sock.recv(length - len(result))
        if not block:
            raise RuntimeError("Unexpected test camera disconnection")
        result += block
    return result


class HeadsetVideoTests(unittest.TestCase):
    def test_real_binary_open_ack_and_h264_stereo_frame_with_bounded_sockets(self):
        import av
        sdk_before = "xrobotoolkit_sdk" in sys.modules
        bridge = HeadsetVideoBridge(expected_peer_ip="127.0.0.1", host="127.0.0.1", port=0,
                                    connect_timeout=.2, send_timeout=.15)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(3)
        control = video = None
        try:
            bridge.start()
            control = socket.create_connection(("127.0.0.1", bridge.port), timeout=3)
            camera, ip = b"ZED", b"127.0.0.1"
            request = (b"\xca\xfe\x01" + struct.pack("<iiiiiii", 64, 24, 30, 1_000_000, 0, 0, listener.getsockname()[1])
                       + bytes([len(camera)]) + camera + bytes([len(ip)]) + ip)
            command = b"OPEN_CAMERA"
            body = struct.pack("<I", len(command)) + command + struct.pack("<I", len(request)) + request
            control.sendall(struct.pack(">I", len(body)) + body)
            video, _ = listener.accept(); video.settimeout(3)
            ack_length, = struct.unpack(">I", receive_exact(control, 4))
            self.assertIn(b"OPEN_CAMERA", receive_exact(control, ack_length))
            self.assertTrue(bridge.status["streaming"])
            left = np.zeros((24, 32, 3), dtype=np.uint8); left[..., 0] = 255
            right = np.zeros_like(left); right[..., 1] = 255
            bridge.update(left, right)
            left[:] = 0; right[:] = 0
            size, = struct.unpack(">I", receive_exact(video, 4))
            encoded = receive_exact(video, size)
            decoder = av.CodecContext.create("h264", "r")
            frames = decoder.decode(av.Packet(encoded))
            self.assertTrue(frames)
            rgb = frames[0].to_ndarray(format="rgb24")
            self.assertEqual(rgb.shape, (24, 64, 3))
            self.assertGreater(rgb[8:16, 8:24, 0].mean(), 220)
            self.assertGreater(rgb[8:16, 40:56, 1].mean(), 220)
            sender = bridge._streaming._publishers[0]
            self.assertEqual(sender._sock.gettimeout(), .15)
            self.assertEqual(bridge.status["frames_submitted"], 1)
            self.assertGreater(bridge.status["video_bytes_sent"], 0)
            self.assertFalse(bridge.status["headset_display_verified"])
            self.assertEqual("xrobotoolkit_sdk" in sys.modules, sdk_before)
        finally:
            bridge.close()
            if control is not None: control.close()
            if video is not None: video.close()
            listener.close()
        self.assertFalse(bridge.status["listening"])
        self.assertFalse(bridge.status["streaming"])

    def test_bad_camera_return_targets_and_unverified_codec_do_not_stream(self):
        bridge = HeadsetVideoBridge(expected_peer_ip="127.0.0.1", host="127.0.0.1", port=0).start()
        try:
            request = {"ip": "172.16.20.83", "port": 10000, "width": 64, "height": 24,
                       "fps": 30, "bitrate": 1_000_000, "enableMvHevc": False}
            bridge._open_camera(request)
            self.assertFalse(bridge.status["streaming"])
            self.assertIn("does not match", bridge.status["error"])
            request.update(ip="127.0.0.1", enableMvHevc=True)
            bridge._open_camera(request)
            self.assertIn("MV-HEVC", bridge.status["error"])
            self.assertFalse(bridge.status["streaming"])
            request.update(enableMvHevc=False, width=100_000)
            bridge._open_camera(request)
            self.assertIn("bounded video limits", bridge.status["error"])
            with self.assertRaises(ValueError): bridge.update(np.zeros((5, 5), np.uint8), None)
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main()
