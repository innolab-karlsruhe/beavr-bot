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
