from beavr.teleop.components.detector.detector_types import InputFrame


def _base_kwargs():
    return dict(
        timestamp_s=1.0,
        hand_side="right",
        keypoints=[(0.0, 0.0, 0.0)],
        is_relative=True,
    )


def test_input_frame_default_gripper_width_is_none():
    frame = InputFrame(**_base_kwargs())
    assert frame.gripper_width_m is None


def test_input_frame_accepts_gripper_width():
    frame = InputFrame(**_base_kwargs(), gripper_width_m=0.025)
    assert frame.gripper_width_m == 0.025


def test_input_frame_existing_fields_unchanged():
    frame = InputFrame(**_base_kwargs(), frame_vectors=((0.0, 0.0, 0.0),
                                                         (1.0, 0.0, 0.0),
                                                         (0.0, 1.0, 0.0),
                                                         (0.0, 0.0, 1.0)))
    assert frame.timestamp_s == 1.0
    assert frame.hand_side == "right"
    assert frame.is_relative is True
    assert frame.gripper_width_m is None
