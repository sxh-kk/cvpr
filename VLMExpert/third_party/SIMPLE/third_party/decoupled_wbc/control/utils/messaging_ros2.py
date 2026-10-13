"""ROS2 messaging backend — thin wrappers around ros_utils.py classes."""

from decoupled_wbc.control.utils.messaging import (
    LoopManager,
    MsgPublisher,
    MsgSubscriber,
    Rate,
    ServiceClient,
    ServiceServer,
)
from decoupled_wbc.control.utils.ros_utils import (
    ROSManager,
    ROSMsgPublisher,
    ROSMsgSubscriber,
    ROSServiceClient,
    ROSServiceServer,
)


class ROS2Rate(Rate):
    def __init__(self, ros_rate):
        self._rate = ros_rate

    def sleep(self):
        self._rate.sleep()


class ROS2LoopManager(LoopManager):
    def __init__(self, node_name: str = "default"):
        self._ros_manager = ROSManager(node_name=node_name)

    def ok(self) -> bool:
        return self._ros_manager.ok()

    def shutdown(self):
        self._ros_manager.shutdown()

    def exceptions(self) -> tuple:
        return self._ros_manager.exceptions()

    def create_rate(self, freq: float) -> Rate:
        return ROS2Rate(self._ros_manager.node.create_rate(freq))


class ROS2MsgPublisher(MsgPublisher):
    def __init__(self, topic_name: str):
        self._pub = ROSMsgPublisher(topic_name)

    def publish(self, msg: dict):
        self._pub.publish(msg)


class ROS2MsgSubscriber(MsgSubscriber):
    def __init__(self, topic_name: str):
        self._sub = ROSMsgSubscriber(topic_name)

    def get_msg(self):
        return self._sub.get_msg()

    @property
    def _msg(self):
        return self._sub._msg

    @_msg.setter
    def _msg(self, value):
        self._sub._msg = value


class ROS2ServiceServer(ServiceServer):
    def __init__(self, service_name: str, config: dict):
        self._srv = ROSServiceServer(service_name, config)


class ROS2ServiceClient(ServiceClient):
    def __init__(self, service_name: str):
        self._cli = ROSServiceClient(service_name)

    def get_config(self) -> dict:
        return self._cli.get_config()


# ---------------------------------------------------------------------------
# Module-level factory helpers (called by messaging.py)
# ---------------------------------------------------------------------------


def create_loop_manager(node_name: str = "default", **kwargs) -> ROS2LoopManager:
    return ROS2LoopManager(node_name=node_name)


def create_publisher(topic_name: str, **kwargs) -> ROS2MsgPublisher:
    return ROS2MsgPublisher(topic_name)


def create_subscriber(topic_name: str, **kwargs) -> ROS2MsgSubscriber:
    return ROS2MsgSubscriber(topic_name)


def create_service_server(service_name: str, config: dict, **kwargs) -> ROS2ServiceServer:
    return ROS2ServiceServer(service_name, config)


def create_service_client(service_name: str, **kwargs) -> ROS2ServiceClient:
    return ROS2ServiceClient(service_name)
