"""Detector for VR-controller-based teleoperation.

Mirror of OculusVRHandDetector for the controller path. Receives compact
pose+trigger messages from beavr-app and publishes InputFrame objects directly
on the existing *_transformed_hand_frame and *_transformed_hand_coords topics
(bypasses keypoint_transform.py).
"""

from __future__ import annotations

import logging

from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class OculusVRControllerDetector:
    """Receives controller pose+trigger and publishes InputFrames directly."""

    @staticmethod
    def _trigger_to_width(trigger: float) -> float:
        """Map analog trigger value [0..1] to OpenArm gripper width in meters.

        trigger=0 (released) -> OPENARM_GRIPPER_MAX_WIDTH_M (open)
        trigger=1 (fully pulled) -> OPENARM_GRIPPER_MIN_WIDTH_M (closed)
        Out-of-range values are clamped.
        """
        clamped = max(0.0, min(1.0, float(trigger)))
        distance = (1.0 - clamped) * robots.OPENARM_GRIPPER_THRESHOLD_M
        gripper_width = (
            (distance / robots.OPENARM_GRIPPER_THRESHOLD_M)
            * robots.OPENARM_GRIPPER_MAX_WIDTH_M
        )
        return max(
            robots.OPENARM_GRIPPER_MIN_WIDTH_M,
            min(gripper_width, robots.OPENARM_GRIPPER_MAX_WIDTH_M),
        )
