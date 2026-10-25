"""Loopback-only HTTP tests; no SDK, model, or simulation is constructed."""
import http.client
import hashlib
import io
import json
from pathlib import Path
import sys
import threading
import socket
import struct
import time
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

    def test_busy_phase_drops_stale_commands_and_rejects_controls_until_complete(self):
        self.ui.update({"status": "paused"})
        self.assertEqual(self.post("discard")[0], 202)
        self.assertEqual(self.post("calibrate")[0], 202)
        self.ui.update({"status": "saving"})
        self.assertEqual(self.ui.commands(), [])
        for phase in ("saving", "auditing", "resetting"):
            self.ui.update({"status": phase})
            self.assertEqual(self.post("discard")[0], 503)
            self.assertEqual(self.post("record")[0], 503)
        self.assertEqual(self.post("quit")[0], 202)
        self.ui.update({"status": "paused"})
        self.assertEqual(self.ui.commands(), ["quit"])
        self.assertEqual(self.post("discard")[0], 202)
        self.assertEqual(self.ui.commands(), ["discard"])

    def open_stream(self):
        client = http.client.HTTPConnection(self.address.hostname, self.address.port, timeout=3)
        client.request("GET", "/stream.mjpg")
        response = client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "multipart/x-mixed-replace; boundary=frame")
        return client, response

    def read_stream_frame(self, response):
        def read_headers():
            result = {}
            while True:
                line = response.readline(1024)
                if line == b"\r\n":
                    return result
                if not line:
                    raise EOFError("Incomplete MJPEG headers")
                key, value = line.decode("ascii").split(":", 1)
                result[key.lower()] = value.strip()
        headers = getattr(response, "_mjpeg_pending_headers", None)
        if headers is None:
            boundary = response.readline(1024)
            if not boundary:
                raise EOFError("MJPEG connection closed")
            self.assertEqual(boundary, b"--frame\r\n")
            headers = read_headers()
        self.assertEqual(headers["content-type"], "image/jpeg")
        # Multipart image boundaries, rather than per-part Content-Length,
        # determine completion. Include the next complete part header in this
        # read, because Chromium needs it before submitting the previous JPEG.
        body, marker = bytearray(), b"\r\n--frame\r\n"
        while not body.endswith(marker):
            value = response.read(1)
            if not value:
                raise EOFError("Incomplete MJPEG image boundary")
            body.extend(value)
            self.assertLess(len(body), 8 * 1024 * 1024)
        del body[-len(marker):]
        response._mjpeg_pending_headers = read_headers()
        self.assertEqual(response._mjpeg_pending_headers["content-type"], "image/jpeg")
        return hashlib.sha256(body).hexdigest(), np.asarray(Image.open(io.BytesIO(body)))

    def test_real_mjpeg_two_distinct_frames_cache_and_commands_continue_independently(self):
        rgb = np.zeros((24, 32, 3), np.uint8); rgb[..., 0] = 255
        self.ui.update({"status": "following"}, rgb)
        calls = []
        real_encode = self.ui._encode_jpeg
        def encode(frame):
            calls.append(threading.get_ident())
            return real_encode(frame)
        self.ui._encode_jpeg = encode
        client, response = self.open_stream()
        result, failure, received = [], [], threading.Event()
        def read_next():
            try:
                result.append(self.read_stream_frame(response))
            except Exception as error:
                failure.append(error)
            finally:
                received.set()
        reader = None
        try:
            first_hash, pixels = self.read_stream_frame(response)
            self.assertGreater(pixels[..., 0].mean(), 245)
            latest = self.request("GET", "/latest.jpg")[2]
            self.assertEqual(first_hash, hashlib.sha256(latest).hexdigest())
            self.assertEqual(len(calls), 1)  # The stream shares the existing cache.
            reader = threading.Thread(target=read_next)
            reader.start()
            self.assertFalse(received.wait(.08))  # No duplicated frames while idle.
            self.assertEqual(self.post("pause")[0], 202)
            self.assertEqual(self.ui.commands(), ["pause"])
            rgb[:] = 0; rgb[..., 1] = 255
            self.ui.update({"status": "paused"}, rgb)
            rgb[:] = 0
            self.assertTrue(received.wait(2))
            self.assertEqual(failure, [])
            self.assertNotEqual(result[0][0], first_hash)
            self.assertGreater(result[0][1][..., 1].mean(), 245)
            self.assertEqual(len(calls), 2)
            page = self.request("GET", "/")[2].decode()
            self.assertIn('src="/stream.mjpg"', page)
            self.assertNotIn("camera.src='/latest.jpg", page)
            self.assertIn("setInterval(refresh,500)", page)
            self.assertIn("camera.onerror=reconnectStream", page)
            self.assertEqual(self.request("GET", "/stream.mjpg", headers={"Host":"evil.example"})[0], 403)
        finally:
            response.close(); client.close()
            if reader is not None:
                reader.join(timeout=2)
                self.assertFalse(reader.is_alive())

    def test_mjpeg_throttles_fast_producer_to_at_most_thirty_frames_per_second(self):
        self.ui.update({}, np.zeros((12, 16, 3), np.uint8))
        client, response = self.open_stream()
        producer = None
        try:
            self.read_stream_frame(response)
            def publish():
                for index in range(40):
                    self.ui.update({}, np.full((12, 16, 3), index + 1, np.uint8))
                    time.sleep(.004)
            started = time.monotonic()
            producer = threading.Thread(target=publish)
            producer.start()
            second, _ = self.read_stream_frame(response)
            third, _ = self.read_stream_frame(response)
            self.assertNotEqual(third, second)
            self.assertGreaterEqual(time.monotonic() - started, .05)
        finally:
            response.close(); client.close()
            if producer is not None:
                producer.join(timeout=2)
                self.assertFalse(producer.is_alive())

    def test_idle_mjpeg_client_disconnect_and_server_close_release_workers(self):
        client, response = self.open_stream()  # No fake initial image is produced.
        response.close(); client.close()
        deadline = time.monotonic() + 1.
        while time.monotonic() < deadline:
            with self.ui._lock:
                if not self.ui._stream_connections:
                    break
            time.sleep(.01)
        with self.ui._lock:
            self.assertEqual(len(self.ui._stream_connections), 0)
        # A fresh stream may connect after the disconnected browser.
        client, response = self.open_stream()
        initial_part = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        self.assertEqual(response.read(len(initial_part)), initial_part)
        started = time.monotonic()
        self.ui.close()  # Wakes an idle frame wait and joins its request thread.
        self.assertLess(time.monotonic() - started, 1.5)
        self.assertEqual(response.read(), b"")
        response.close(); client.close()
        self.assertFalse(self.ui._thread.is_alive())
        with self.ui._lock:
            self.assertEqual(len(self.ui._stream_connections), 0)

    def test_paused_single_image_immediately_completes_next_part_headers_without_a_second_image(self):
        rgb = np.zeros((24, 32, 3), np.uint8); rgb[..., 2] = 255
        self.ui.update({"status": "paused"}, rgb)
        started = time.monotonic()
        client, response = self.open_stream()
        try:
            digest, pixels = self.read_stream_frame(response)
            self.assertLess(time.monotonic() - started, .75)
            self.assertGreater(pixels[..., 2].mean(), 245)
            self.assertEqual(response._mjpeg_pending_headers, {"content-type": "image/jpeg"})
            status = json.loads(self.request("GET", "/status.json")[2])
            self.assertEqual(status["status"]["status"], "paused")
            self.assertEqual(status["image_version"], 1)  # No second update/image.
            self.assertEqual(digest, hashlib.sha256(self.request("GET", "/latest.jpg")[2]).hexdigest())
        finally:
            response.close(); client.close()


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
