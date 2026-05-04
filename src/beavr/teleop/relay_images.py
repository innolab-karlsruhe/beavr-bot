from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from typing import Optional

import cv2
import numpy as np
import zmq

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import CompressedImage


# Default port from BeaVR-Unity/Assets/Resources/Configurations/Network.json
DEFAULT_PORT = 10505


def make_publisher(port: int) -> tuple[zmq.Context, zmq.Socket]:
    """Create a ZMQ PUB socket bound on all interfaces at ``port``."""
    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)

    # Drop, don't queue. CONFLATE *must* be set before bind/connect.
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.TCP_KEEPALIVE, 1)

    socket.setsockopt(zmq.SNDBUF, 256 * 1024)

    bind_address = f"tcp://*:{port}"
    socket.bind(bind_address)
    print(f"[publisher] PUB socket bound on {bind_address}")
    return context, socket


class LatestFrameSender:
    """Background sender that keeps only the most recent JPEG frame."""

    def __init__(self, socket: zmq.Socket, max_fps: float = 30.0) -> None:
        self._socket = socket
        self._min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._latest: Optional[bytes] = None
        self._dropped_in: int = 0
        self._sent: int = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="zmq-frame-sender", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        self._thread.join(timeout=1.0)

    def submit_encoded(self, jpeg_bytes: bytes) -> None:
        """Hand already-JPEG-encoded bytes to the sender (drop-old)."""
        with self._cond:
            if self._latest is not None:
                # An unsent frame is being overwritten -- count it.
                self._dropped_in += 1
            self._latest = jpeg_bytes
            self._cond.notify()

    def stats_and_reset(self) -> tuple[int, int]:
        """Return (sent, dropped_before_send) since the last call."""
        with self._lock:
            s, d = self._sent, self._dropped_in
            self._sent = 0
            self._dropped_in = 0
        return s, d

    def _run(self) -> None:
        next_send_time = time.monotonic()
        while not self._stop.is_set():
            with self._cond:
                while self._latest is None and not self._stop.is_set():
                    self._cond.wait()
                if self._stop.is_set():
                    return
                payload = self._latest
                self._latest = None

            # Stable wall-clock pacing -- never exceed max_fps. If we
            # are running late (next_send_time already in the past),
            # send immediately and re-anchor the schedule.
            now = time.monotonic()
            if self._min_period > 0:
                if now < next_send_time:
                    time.sleep(next_send_time - now)
                    now = time.monotonic()
                next_send_time = max(now, next_send_time) + self._min_period

            try:
                self._socket.send(payload, flags=zmq.NOBLOCK)
                with self._lock:
                    self._sent += 1
            except zmq.Again:
                pass
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[publisher] sender thread error: {exc}", file=sys.stderr)


def run_ros2(socket: zmq.Socket, topic: str, max_fps: float) -> None:
    """Bridge a ROS 2 ``sensor_msgs/msg/CompressedImage`` topic to VR."""

    sender = LatestFrameSender(socket, max_fps=max_fps)

    class RelayNode(Node):
        def __init__(self) -> None:
            super().__init__("beavr_image_publish_test")
            # Camera drivers commonly publish with BEST_EFFORT reliability;
            # match that so we actually receive frames. Depth=1 with
            # KEEP_LAST means the rclpy executor itself only ever holds
            # the newest message -- another stutter-prevention layer.
            qos = QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.VOLATILE,
                depth=1,
            )
            self._sub = self.create_subscription(
                CompressedImage, topic, self._on_msg, qos
            )
            self._received = 0
            self._last_log = time.monotonic()
            self._first_msg_stamp_ns: Optional[int] = None
            self._last_msg_stamp_ns: Optional[int] = None
            self._stamp_window_count = 0
            self.get_logger().info(
                f"subscribed to {topic} (sensor_msgs/msg/CompressedImage)"
            )

        def _on_msg(self, msg: "CompressedImage") -> None:
            # Track the publisher's own timestamps so we can report the
            # rate the camera *says* it is publishing at, independent of
            # how many messages actually reached this callback.
            stamp_ns = (
                int(msg.header.stamp.sec) * 1_000_000_000
                + int(msg.header.stamp.nanosec)
            )
            if stamp_ns > 0:
                if self._first_msg_stamp_ns is None:
                    self._first_msg_stamp_ns = stamp_ns
                self._last_msg_stamp_ns = stamp_ns
                self._stamp_window_count += 1

            # CompressedImage.data is already a compressed image
            # buffer. Unity's Texture2D.LoadImage accepts only JPEG
            # and PNG, so for those formats we forward the bytes
            # straight through. For any other format we decode and
            # re-encode as JPEG.
            fmt = (msg.format or "").lower()
            raw_bytes = (
                bytes(msg.data)
                if not isinstance(msg.data, bytes)
                else msg.data
            )
            if "jpeg" in fmt or "jpg" in fmt or "png" in fmt or fmt == "":
                payload = raw_bytes
            else:
                arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if decoded is None:
                    return
                ok, buffer = cv2.imencode(
                    ".jpg", decoded, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                if not ok:
                    return
                payload = buffer.tobytes()
            sender.submit_encoded(payload)
            self._received += 1

            now = time.monotonic()
            if now - self._last_log >= 1.0:
                sent, _dropped = sender.stats_and_reset()
                
                self.get_logger().info(
                    f"received={self._received:3d}  sent={sent:3d}  "
                )
                self._received = 0
                self._first_msg_stamp_ns = None
                self._last_msg_stamp_ns = None
                self._stamp_window_count = 0
                self._last_log = now

    rclpy.init()
    node = RelayNode()
    sender.start()
    print(
        f"[publisher] topic={topic}  max_fps={max_fps}\n"
        "[publisher] relaying... (Ctrl+C to stop)"
    )
    try:
        rclpy.spin(node)
    finally:
        sender.stop()
        node.destroy_node()
        rclpy.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relay a ROS 2 sensor_msgs/CompressedImage topic to the "
            "BeaVR VR client (CameraOneStreamer.cs) over ZMQ."
        )
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to bind the PUB socket on (default: {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--topic",
        default="/camera/color/image_raw/compressed",
        help=(
            "ROS 2 sensor_msgs/CompressedImage topic to subscribe to "
            "(default: /camera/color/image_raw/compressed)."
        ),
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=30.0,
        help=(
            "Upper bound on frames sent per second. The newest source "
            "frame is always preferred; intermediate ones are dropped "
            "(default: 30)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Allow Ctrl+C to terminate cleanly even when blocked in send/spin.
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    context, socket = make_publisher(args.port)

    # Tiny grace period so any already-waiting subscribers complete the
    # TCP/ZMQ handshake before the first message is sent (mitigates the
    # classic "slow joiner" issue if the VR app was started first).
    time.sleep(0.2)

    try:
        run_ros2(socket, topic=args.topic, max_fps=args.max_fps)
    finally:
        socket.close(linger=0)
        context.term()
        print("[publisher] shutdown complete")


if __name__ == "__main__":
    main()