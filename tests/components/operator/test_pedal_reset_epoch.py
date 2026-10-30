"""Regression tests for the pedal-press reset epoch in XArmOperator.

Race being pinned (fixed via the _reset_epoch counter): a pedal press that
arrives while a _reset_teleop() is already in progress used to set
`is_first_frame = True` from the ROS2 callback thread, which the in-progress
reset's trailing `is_first_frame = False` then overwrote — silently dropping
the press and leaving the teleop running on a stale baseline. With the epoch
counter, the late press survives the completed reset and triggers one more
reset on the next control cycle.
"""

import sys
import time

import numpy as np

from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.operator.operator_types import CartesianTarget
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator
from beavr.teleop.configs.constants import robots

_HOST = "127.0.0.1"
_KEYPOINTS_PORT = 5555
_ENDEFF_PUBLISH_PORT = 7777
_ENDEFF_SUBSCRIBE_PORT = 6666


class _PedalMsg:
    """Minimal stand-in for std_msgs.msg.Bool."""

    def __init__(self, data: bool) -> None:
        self.data = data


def _make_op() -> XArmOperator:
    return XArmOperator(
        operator_name="xarm7_pedal_test_operator",
        host=_HOST,
        transformed_keypoints_port=_KEYPOINTS_PORT,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=_ENDEFF_PUBLISH_PORT,
        endeff_subscribe_port=_ENDEFF_SUBSCRIBE_PORT,
        moving_average_limit=1,
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
    )


def _publish_robot_pose(bus) -> None:
    bus.publish(
        _HOST,
        _ENDEFF_SUBSCRIBE_PORT,
        "endeff_homo",
        CartesianState(
            timestamp_s=time.time(),
            h_matrix=tuple(map(tuple, np.eye(4).tolist())),
        ),
    )


def _publish_hand_frame(bus) -> None:
    frame_vectors = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    bus.publish(
        _HOST,
        _KEYPOINTS_PORT,
        f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}",
        InputFrame(
            timestamp_s=time.time(),
            hand_side=robots.RIGHT,
            keypoints=[(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS,
            is_relative=False,
            frame_vectors=frame_vectors,
        ),
    )


def test_pedal_press_during_reset_is_not_lost(bus):
    op = _make_op()

    # Simulate the first press having been handled: gate open, armed.
    op._waiting_for_first_pedal = False

    # A reset is "in progress" (is_first_frame set by an earlier trigger)...
    op.is_first_frame = True
    # ...and a new press arrives before that reset finishes.
    op._pedal_pressed_callback(_PedalMsg(True))
    assert op._reset_epoch == 1

    # The in-progress reset completes. With the previous bool handshake this
    # write is exactly what silently dropped the press above.
    op.is_first_frame = False
    assert op._last_reset_epoch == 0  # the press is still pending

    # The next control cycle must serve the pending press with a fresh reset.
    _publish_robot_pose(bus)
    _publish_hand_frame(bus)
    op._apply_retargeted_angles()

    assert op._last_reset_epoch == op._reset_epoch == 1
    assert op.is_first_frame is False
    assert op.hand_init_h is not None
    assert op.robot_init_h is not None


def test_only_rising_edges_request_resets(bus):
    op = _make_op()
    op._waiting_for_first_pedal = False

    op._pedal_pressed_callback(_PedalMsg(True))  # press          -> epoch 1
    op._pedal_pressed_callback(_PedalMsg(True))  # still held (continuous publisher)
    op._pedal_pressed_callback(_PedalMsg(False))  # release
    op._pedal_pressed_callback(_PedalMsg(False))  # still released
    op._pedal_pressed_callback(_PedalMsg(True))  # press          -> epoch 2

    assert op._reset_epoch == 2
    assert op._waiting_for_first_pedal is False


def test_ros2_unavailable_falls_back_to_reset_on_startup(bus, monkeypatch):
    # Force `import rclpy` inside _init_pedal_subscriber to fail regardless of
    # whether ROS2 happens to be installed in the test environment.
    monkeypatch.setitem(sys.modules, "rclpy", None)

    op = _make_op()

    # The gate must be lifted so the operator does not block forever.
    assert op._waiting_for_first_pedal is False

    # The first control cycle captures the baseline at startup and publishes.
    _publish_robot_pose(bus)
    _publish_hand_frame(bus)
    op._apply_retargeted_angles()

    assert op.is_first_frame is False
    assert op.hand_init_h is not None
    assert op.robot_init_h is not None
    cmd = bus.recv_latest(_ENDEFF_PUBLISH_PORT, "endeff_coords")
    assert isinstance(cmd, CartesianTarget)


def test_no_commands_while_waiting_for_first_pedal_press(bus):
    op = _make_op()
    # Simulate the gate still closed (pedal subsystem up, no press yet).
    op._waiting_for_first_pedal = True

    _publish_robot_pose(bus)
    _publish_hand_frame(bus)
    op._apply_retargeted_angles()

    assert op.hand_init_h is None
    assert bus.recv_latest(_ENDEFF_PUBLISH_PORT, "endeff_coords") is None
