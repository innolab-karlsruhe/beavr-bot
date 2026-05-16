import numpy as np

from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.operator.robots.xarm7_operator import XArmOperator
from beavr.teleop.configs.constants import robots


def _make_op(bus):
    return XArmOperator(
        operator_name="xarm7_test_operator",
        host="127.0.0.1",
        transformed_keypoints_port=5555,
        stream_configs={},
        stream_oculus=False,
        endeff_publish_port=7777,
        endeff_subscribe_port=6666,
        moving_average_limit=1,
        h_r_v=np.eye(4),
        h_t_v=np.eye(4),
        final_translation=np.eye(4),
        use_filter=False,
        arm_resolution_port=None,
        teleoperation_state_port=None,
        logging_config={"enabled": False},
        hand_side=robots.RIGHT,
    )


def _publish_hand_coords(bus, host, port, *, gripper_width_m=None, keypoints=None):
    if keypoints is None:
        keypoints = [(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS
    bus.publish(
        host,
        port,
        f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_COORDS}",
        InputFrame(
            timestamp_s=1.0,
            hand_side=robots.RIGHT,
            keypoints=keypoints,
            is_relative=False,
            gripper_width_m=gripper_width_m,
        ),
    )


def test_prefers_explicit_gripper_width_when_set(bus):
    op = _make_op(bus)
    _publish_hand_coords(bus, "127.0.0.1", 5555, gripper_width_m=0.02)
    width = op._extract_gripper_width()
    assert width == 0.02


def test_falls_back_to_keypoints_when_gripper_width_none(bus):
    op = _make_op(bus)
    # Place thumb_tip (idx 5) and index_tip (idx 10) at known distance.
    keypoints = [(0.0, 0.0, 0.0)] * robots.OCULUS_NUM_KEYPOINTS
    # 9cm apart along x => maps to max gripper width.
    keypoints[robots.OCULUS_JOINTS["thumb_tip"]] = (0.0, 0.0, 0.0)
    keypoints[robots.OCULUS_JOINTS["index_tip"]] = (0.09, 0.0, 0.0)
    _publish_hand_coords(bus, "127.0.0.1", 5555,
                         gripper_width_m=None, keypoints=keypoints)
    width = op._extract_gripper_width()
    assert np.isclose(width, robots.OPENARM_GRIPPER_MAX_WIDTH_M, atol=1e-6)


def test_clamps_explicit_gripper_width_above_max(bus):
    op = _make_op(bus)
    _publish_hand_coords(bus, "127.0.0.1", 5555,
                         gripper_width_m=robots.OPENARM_GRIPPER_MAX_WIDTH_M + 1.0)
    width = op._extract_gripper_width()
    assert width == robots.OPENARM_GRIPPER_MAX_WIDTH_M


def test_clamps_explicit_gripper_width_below_min(bus):
    op = _make_op(bus)
    _publish_hand_coords(bus, "127.0.0.1", 5555, gripper_width_m=-0.01)
    width = op._extract_gripper_width()
    assert width == robots.OPENARM_GRIPPER_MIN_WIDTH_M
