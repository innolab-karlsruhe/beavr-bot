import numpy as np
import pytest

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
