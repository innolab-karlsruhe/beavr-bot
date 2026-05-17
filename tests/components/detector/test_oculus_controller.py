import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from beavr.teleop.components.detector.vr.oculus_controller import (
    OculusVRControllerDetector,
)
from beavr.teleop.configs.constants import robots


def test_trigger_to_width_zero_is_max():
    width = OculusVRControllerDetector._trigger_to_width(0.0)
    assert np.isclose(width, robots.OPENARM_GRIPPER_MAX_WIDTH_M, atol=1e-9)


def test_trigger_to_width_one_is_min():
    width = OculusVRControllerDetector._trigger_to_width(1.0)
    assert np.isclose(width, robots.OPENARM_GRIPPER_MIN_WIDTH_M, atol=1e-9)


def test_trigger_to_width_half_is_midway():
    width = OculusVRControllerDetector._trigger_to_width(0.5)
    expected = 0.5 * robots.OPENARM_GRIPPER_MAX_WIDTH_M
    assert np.isclose(width, expected, atol=1e-9)


def test_trigger_to_width_clamps_above_one():
    width = OculusVRControllerDetector._trigger_to_width(1.5)
    assert np.isclose(width, robots.OPENARM_GRIPPER_MIN_WIDTH_M, atol=1e-9)


def test_trigger_to_width_clamps_below_zero():
    width = OculusVRControllerDetector._trigger_to_width(-0.5)
    assert np.isclose(width, robots.OPENARM_GRIPPER_MAX_WIDTH_M, atol=1e-9)


# --- _parse ---

def test_parse_well_formed_relative_message():
    raw = b"relative:0.1,0.2,0.3|0.0,0.707,0.0,0.707|0.42"
    pos, quat, trigger, mode = OculusVRControllerDetector._parse(raw)
    np.testing.assert_allclose(pos, (0.1, 0.2, 0.3), atol=1e-9)
    np.testing.assert_allclose(quat, (0.0, 0.707, 0.0, 0.707), atol=1e-9)
    assert np.isclose(trigger, 0.42)
    assert mode == "relative"


def test_parse_well_formed_absolute_message():
    raw = b"absolute:1.0,2.0,3.0|0,0,0,1|0.0"
    pos, quat, trigger, mode = OculusVRControllerDetector._parse(raw)
    np.testing.assert_allclose(pos, (1.0, 2.0, 3.0))
    np.testing.assert_allclose(quat, (0.0, 0.0, 0.0, 1.0))
    assert trigger == 0.0
    assert mode == "absolute"


def test_parse_strips_trailing_whitespace_and_newline():
    raw = b"relative:0,0,0|0,0,0,1|0.5  \n"
    pos, quat, trigger, mode = OculusVRControllerDetector._parse(raw)
    assert mode == "relative"
    assert trigger == 0.5


def test_parse_malformed_message_returns_none():
    raw = b"this is garbage"
    result = OculusVRControllerDetector._parse(raw)
    assert result == (None, None, None, None)


def test_parse_truncated_quaternion_returns_none():
    raw = b"relative:0,0,0|1,0,0|0.5"  # only 3 quat components
    result = OculusVRControllerDetector._parse(raw)
    assert result == (None, None, None, None)


def test_parse_invalid_float_returns_none():
    raw = b"relative:0,not_a_number,0|0,0,0,1|0.5"
    result = OculusVRControllerDetector._parse(raw)
    assert result == (None, None, None, None)


# --- _frame_from_quat ---

def test_frame_from_quat_identity_quaternion():
    pos = (1.0, 2.0, 3.0)
    # Identity quaternion (xyzw)
    quat = (0.0, 0.0, 0.0, 1.0)
    frame = OculusVRControllerDetector._frame_from_quat(pos, quat)
    assert frame.shape == (4, 3)
    np.testing.assert_allclose(frame[0], pos, atol=1e-9)
    # Columns of identity rotation matrix
    np.testing.assert_allclose(frame[1], (1.0, 0.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(frame[2], (0.0, 1.0, 0.0), atol=1e-9)
    np.testing.assert_allclose(frame[3], (0.0, 0.0, 1.0), atol=1e-9)


def test_frame_from_quat_90deg_around_y():
    # 90° around Y axis: x -> -z, z -> x; quaternion xyzw = (0, sin(45), 0, cos(45))
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    quat = (0.0, s, 0.0, c)
    frame = OculusVRControllerDetector._frame_from_quat((0.0, 0.0, 0.0), quat)
    # After 90° around Y: column 0 (was +x) -> (0, 0, -1); column 2 (was +z) -> (1, 0, 0)
    np.testing.assert_allclose(frame[1], (0.0, 0.0, -1.0), atol=1e-6)
    np.testing.assert_allclose(frame[2], (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(frame[3], (1.0, 0.0, 0.0), atol=1e-6)


def test_frame_from_quat_non_unit_quaternion_is_normalized():
    # 2x identity quaternion is still a valid rotation when normalized.
    pos = (0.0, 0.0, 0.0)
    quat = (0.0, 0.0, 0.0, 2.0)
    frame = OculusVRControllerDetector._frame_from_quat(pos, quat)
    np.testing.assert_allclose(frame[1], (1.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(frame[2], (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(frame[3], (0.0, 0.0, 1.0), atol=1e-6)


# --- _rotate_90_around_x ---

def test_rotate_90_around_x_position_only_consistency():
    # (x, y, z) -> (x, -z, y)
    pos_in = (1.0, 2.0, 3.0)
    quat_in = (0.0, 0.0, 0.0, 1.0)
    pos_out, quat_out = OculusVRControllerDetector._rotate_90_around_x(pos_in, quat_in)
    np.testing.assert_allclose(pos_out, (1.0, -3.0, 2.0), atol=1e-9)


def test_rotate_90_around_x_rotation_composition():
    # Applying the rotation to a vector via the rotated quat must equal
    # applying it directly to the rotated original-quat output.
    rng = np.random.default_rng(seed=0)
    for _ in range(5):
        raw_quat = rng.normal(size=4)
        raw_quat = raw_quat / np.linalg.norm(raw_quat)
        pos = tuple(rng.normal(size=3))
        vec = np.array([0.0, 0.0, 1.0])  # arbitrary probe

        # Reference: apply original rotation to the probe, then rotate result around X.
        R_orig = Rotation.from_quat(raw_quat).as_matrix()
        rotated_vec_ref = R_orig @ vec
        rotated_vec_ref = np.array([
            rotated_vec_ref[0],
            -rotated_vec_ref[2],
            rotated_vec_ref[1],
        ])

        # Helper output: apply rotated quaternion to the probe.
        _, quat_rotated = OculusVRControllerDetector._rotate_90_around_x(pos, tuple(raw_quat))
        rotated_vec_helper = Rotation.from_quat(np.array(quat_rotated)).as_matrix() @ vec

        np.testing.assert_allclose(rotated_vec_helper, rotated_vec_ref, atol=1e-9)


# --- end-to-end: drive stream() with stubbed _receive ---


class _StopStream(Exception):
    """Sentinel raised inside _receive to exit stream() after one iteration."""


def test_stream_publishes_on_both_topics(bus, monkeypatch):
    """Drive stream() through one iteration with a single valid raw message and
    verify both _transformed_hand_frame and _transformed_hand_coords topics
    receive the resulting InputFrame.
    """
    from beavr.teleop.components.detector.vr import oculus_controller as oc_module

    # Stub create_pull_socket so __init__ doesn't try to bind a real ZMQ socket.
    class _StubSocket:
        def close(self):
            pass

    monkeypatch.setattr(
        oc_module,
        "create_pull_socket",
        lambda host, port: _StubSocket(),
    )

    det = OculusVRControllerDetector(
        host="127.0.0.1",
        controller_pub_port=9999,
        hand_config="right",
        right_controller_port=8122,
    )

    # Override _receive so the first call returns a valid message and the
    # second call raises _StopStream — that exits stream() via its finally
    # block without leaving the test in an infinite loop.
    calls = {"n": 0}

    def _fake_receive(side):
        calls["n"] += 1
        if calls["n"] == 1:
            return b"relative:0.1,0.2,0.3|0,0,0,1|0.5"
        raise _StopStream()

    monkeypatch.setattr(det, "_receive", _fake_receive)

    with pytest.raises(_StopStream):
        det.stream()

    frame_topic = bus.recv_latest(9999, f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}")
    coords_topic = bus.recv_latest(9999, f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_COORDS}")
    assert frame_topic is not None
    assert coords_topic is not None

    # trigger=0.5 -> gripper_width_m = 0.5 * MAX
    assert np.isclose(
        frame_topic.gripper_width_m,
        0.5 * robots.OPENARM_GRIPPER_MAX_WIDTH_M,
    )
    assert frame_topic.is_relative is True
    assert frame_topic.hand_side == robots.RIGHT
    # frame_vectors is a 4-tuple of 3-tuples
    assert len(frame_topic.frame_vectors) == 4
    assert all(len(row) == 3 for row in frame_topic.frame_vectors)
