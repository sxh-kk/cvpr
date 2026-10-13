"""ZMQ messaging backend — no rclpy imports anywhere in this module."""

import signal
import threading
import time as _time
from typing import Optional

import msgpack
import msgpack_numpy as mnp
import zmq

from decoupled_wbc.control.utils.messaging import (
    LoopManager,
    MsgPublisher,
    MsgSubscriber,
    Rate,
    ServiceClient,
    ServiceServer,
)

# ---------------------------------------------------------------------------
# Default topic-to-port mapping (overridable via kwargs)
# ---------------------------------------------------------------------------

DEFAULT_TOPIC_PORTS = {
    "ControlPolicy/upper_body_pose": 6001,
    "G1Env/env_state_act": 6002,
    "ControlPolicy/lower_body_policy_status": 6003,
    "ControlPolicy/joint_safety_status": 6004,
    "WBCPolicy/robot_config": 6005,
}

_next_auto_port = 6100  # fallback for unmapped topics


def _port_for_topic(topic_name: str, port: Optional[int] = None) -> int:
    global _next_auto_port
    if port is not None:
        return port
    if topic_name in DEFAULT_TOPIC_PORTS:
        return DEFAULT_TOPIC_PORTS[topic_name]
    # auto-assign
    p = _next_auto_port
    _next_auto_port += 1
    DEFAULT_TOPIC_PORTS[topic_name] = p
    return p


# ---------------------------------------------------------------------------
# Signal handling (mirrors ros_utils.register_keyboard_interrupt_handler)
# ---------------------------------------------------------------------------

_signal_registered = False


def _register_signal_handler():
    global _signal_registered
    if not _signal_registered:

        def handler(signum, frame):
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)
        _signal_registered = True


# ---------------------------------------------------------------------------
# Rate
# ---------------------------------------------------------------------------


class ZMQRate(Rate):
    """time.sleep-based rate with drift compensation."""

    def __init__(self, freq: float):
        self._period = 1.0 / freq
        self._last = _time.monotonic()

    def sleep(self):
        now = _time.monotonic()
        elapsed = now - self._last
        remaining = self._period - elapsed
        if remaining > 0:
            _time.sleep(remaining)
        self._last = _time.monotonic()


# ---------------------------------------------------------------------------
# LoopManager
# ---------------------------------------------------------------------------


class ZMQLoopManager(LoopManager):
    def __init__(self, node_name: str = "default"):
        self._running = threading.Event()
        self._running.set()
        _register_signal_handler()

    def ok(self) -> bool:
        return self._running.is_set()

    def shutdown(self):
        self._running.clear()

    def exceptions(self) -> tuple:
        return (KeyboardInterrupt,)

    def create_rate(self, freq: float) -> Rate:
        return ZMQRate(freq)


# ---------------------------------------------------------------------------
# Publisher
# ---------------------------------------------------------------------------


class ZMQMsgPublisher(MsgPublisher):
    def __init__(self, topic_name: str, port: Optional[int] = None):
        self._port = _port_for_topic(topic_name, port)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 20)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.bind(f"tcp://*:{self._port}")

    def publish(self, msg: dict):
        packed = msgpack.packb(msg, default=mnp.encode)
        try:
            self._socket.send(packed, flags=zmq.NOBLOCK)
        except zmq.Again:
            pass  # drop if HWM reached

    def close(self):
        self._socket.close()
        self._context.term()


# ---------------------------------------------------------------------------
# Subscriber
# ---------------------------------------------------------------------------


class ZMQMsgSubscriber(MsgSubscriber):
    def __init__(self, topic_name: str, host: str = "localhost", port: Optional[int] = None):
        self._port = _port_for_topic(topic_name, port)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.CONFLATE, True)
        self._socket.setsockopt(zmq.RCVHWM, 3)
        self._socket.connect(f"tcp://{host}:{self._port}")
        self._msg = None

    def get_msg(self) -> Optional[dict]:
        try:
            packed = self._socket.recv(flags=zmq.NOBLOCK)
            return msgpack.unpackb(packed, object_hook=mnp.decode)
        except zmq.Again:
            return None

    def close(self):
        self._socket.close()
        self._context.term()


# ---------------------------------------------------------------------------
# ServiceServer (REP socket in a background thread)
# ---------------------------------------------------------------------------


class ZMQServiceServer(ServiceServer):
    def __init__(self, service_name: str, config: dict, port: Optional[int] = None):
        self._port = _port_for_topic(service_name, port)
        self._packed_config = msgpack.packb(config, default=mnp.encode)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REP)
        self._socket.bind(f"tcp://*:{self._port}")
        self._running = True
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while self._running:
            socks = dict(poller.poll(timeout=500))
            if self._socket in socks:
                self._socket.recv()  # consume request
                self._socket.send(self._packed_config)

    def close(self):
        self._running = False
        self._thread.join(timeout=2)
        self._socket.close()
        self._context.term()


# ---------------------------------------------------------------------------
# ServiceClient (REQ socket, connect-and-retry)
# ---------------------------------------------------------------------------


class ZMQServiceClient(ServiceClient):
    def __init__(self, service_name: str, host: str = "localhost", port: Optional[int] = None):
        self._port = _port_for_topic(service_name, port)
        self._host = host
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.connect(f"tcp://{self._host}:{self._port}")

    def get_config(self) -> dict:
        self._socket.send(b"get_config")
        packed = self._socket.recv()
        return msgpack.unpackb(packed, object_hook=mnp.decode)

    def close(self):
        self._socket.close()
        self._context.term()


# ---------------------------------------------------------------------------
# Module-level factory helpers (called by messaging.py)
# ---------------------------------------------------------------------------


def create_loop_manager(node_name: str = "default", **kwargs) -> ZMQLoopManager:
    return ZMQLoopManager(node_name=node_name)


def create_publisher(topic_name: str, port: Optional[int] = None, **kwargs) -> ZMQMsgPublisher:
    return ZMQMsgPublisher(topic_name, port=port)


def create_subscriber(
    topic_name: str, host: str = "localhost", port: Optional[int] = None, **kwargs
) -> ZMQMsgSubscriber:
    return ZMQMsgSubscriber(topic_name, host=host, port=port)


def create_service_server(
    service_name: str, config: dict, port: Optional[int] = None, **kwargs
) -> ZMQServiceServer:
    return ZMQServiceServer(service_name, config, port=port)


def create_service_client(
    service_name: str, host: str = "localhost", port: Optional[int] = None, **kwargs
) -> ZMQServiceClient:
    return ZMQServiceClient(service_name, host=host, port=port)
