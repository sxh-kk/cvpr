"""Abstract messaging interfaces and factory functions for swappable backends (ROS2 / ZMQ)."""

from abc import ABC, abstractmethod
from typing import Optional


class Rate(ABC):
    """Abstracts loop rate control."""

    @abstractmethod
    def sleep(self):
        ...


class LoopManager(ABC):
    """Lifecycle manager for the main loop."""

    @abstractmethod
    def ok(self) -> bool:
        ...

    @abstractmethod
    def shutdown(self):
        ...

    @abstractmethod
    def exceptions(self) -> tuple:
        """Return a tuple of exception types to catch in the main loop."""
        ...

    @abstractmethod
    def create_rate(self, freq: float) -> Rate:
        ...


class MsgPublisher(ABC):
    @abstractmethod
    def publish(self, msg: dict):
        ...


class MsgSubscriber(ABC):
    @abstractmethod
    def get_msg(self) -> Optional[dict]:
        ...


class ServiceServer(ABC):
    """Serves a config dict on request."""

    pass


class ServiceClient(ABC):
    @abstractmethod
    def get_config(self) -> dict:
        ...


# ---------------------------------------------------------------------------
# Factory functions — lazy-import the chosen backend
# ---------------------------------------------------------------------------


def _get_backend_module(backend: str):
    if backend == "ros2":
        from decoupled_wbc.control.utils import messaging_ros2 as mod
    elif backend == "zmq":
        from decoupled_wbc.control.utils import messaging_zmq as mod
    else:
        raise ValueError(f"Unknown messaging backend: {backend!r}. Use 'ros2' or 'zmq'.")
    return mod


def create_loop_manager(backend: str, node_name: str = "default", **kwargs) -> LoopManager:
    mod = _get_backend_module(backend)
    return mod.create_loop_manager(node_name=node_name, **kwargs)


def create_publisher(backend: str, topic_name: str, **kwargs) -> MsgPublisher:
    mod = _get_backend_module(backend)
    return mod.create_publisher(topic_name=topic_name, **kwargs)


def create_subscriber(backend: str, topic_name: str, **kwargs) -> MsgSubscriber:
    mod = _get_backend_module(backend)
    return mod.create_subscriber(topic_name=topic_name, **kwargs)


def create_service_server(backend: str, service_name: str, config: dict, **kwargs) -> ServiceServer:
    mod = _get_backend_module(backend)
    return mod.create_service_server(service_name=service_name, config=config, **kwargs)


def create_service_client(backend: str, service_name: str, **kwargs) -> ServiceClient:
    mod = _get_backend_module(backend)
    return mod.create_service_client(service_name=service_name, **kwargs)
