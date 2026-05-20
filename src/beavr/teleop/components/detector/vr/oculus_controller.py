"""Detector for VR-controller-based teleoperation.

Mirror of OculusVRHandDetector for the controller path. Receives compact
pose+trigger messages from beavr-app and publishes InputFrame objects directly
on the existing *_transformed_hand_frame and *_transformed_hand_coords topics
(bypasses keypoint_transform.py).
"""

from __future__ import annotations

import logging
import time
from typing import Literal, Optional

import numpy as np
import zmq
from scipy.spatial.transform import Rotation

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.utils import create_pull_socket
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


class OculusVRControllerDetector(Component):
    """Receives controller pose+trigger and publishes InputFrames directly.

    Lifecycle parallels OculusVRHandDetector but the implementation is much
    shorter: we bypass keypoint_transform.py because controller orientation
    is already clean.
    """

    def __init__(
        self,
        host: str,
        controller_pub_port: int,
        hand_config: Literal["right", "left", "bimanual"] = "bimanual",
        right_controller_port: Optional[int] = None,
        left_controller_port: Optional[int] = None,
    ):
        self.notify_component_start("vr controller detector")
        self.host = host
        self.controller_pub_port = controller_pub_port
        self.hand_config = hand_config

        self.sides: list[str] = []
        if hand_config in (robots.RIGHT, robots.BIMANUAL):
            if right_controller_port is None:
                raise ValueError(
                    f"right_controller_port is required when hand_config={hand_config!r}"
                )
            self.sides.append(robots.RIGHT)
        if hand_config in (robots.LEFT, robots.BIMANUAL):
            if left_controller_port is None:
                raise ValueError(
                    f"left_controller_port is required when hand_config={hand_config!r}"
                )
            self.sides.append(robots.LEFT)

        self.sockets: dict[str, zmq.Socket] = {}
        if robots.RIGHT in self.sides:
            self.sockets[robots.RIGHT] = create_pull_socket(host, right_controller_port)
        if robots.LEFT in self.sides:
            self.sockets[robots.LEFT] = create_pull_socket(host, left_controller_port)

        self.publisher_manager = ZMQPublisherManager.get_instance()
        self.timer = FrequencyTimer(robots.VR_FREQ)

    def _receive(self, side: str) -> Optional[bytes]:
        try:
            return self.sockets[side].recv(zmq.NOBLOCK)
        except zmq.Again:
            return None

    def stream(self):
        logger.info(
            f"Starting VR controller detection for sides={self.sides} on port "
            f"{self.controller_pub_port}"
        )
        try:
            while True:
                self.timer.start_loop()
                for side in self.sides:
                    raw = self._receive(side)
                    if raw is None:
                        continue
                    pos, quat, trigger, mode = self._parse(raw)
                    if pos is None:
                        continue
                    pos, quat = self._rotate_90_around_x(pos, quat)
                    frame_vectors = tuple(map(tuple, self._frame_from_quat(pos, quat).tolist()))
                    gripper_width_m = self._trigger_to_width(trigger)

                    logger.debug(
                        f"[{side}] mode={mode} "
                        f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
                        f"quat=({quat[0]:.3f}, {quat[1]:.3f}, {quat[2]:.3f}, {quat[3]:.3f}) "
                        f"trigger={trigger:.3f} gripper_width={gripper_width_m*1000:.1f}mm"
                    )

                    input_frame = InputFrame(
                        timestamp_s=time.time(),
                        hand_side=side,
                        keypoints=[],
                        is_relative=(mode == robots.RELATIVE),
                        frame_vectors=frame_vectors,
                        gripper_width_m=gripper_width_m,
                    )

                    # Operator subscribes to both topics; controller path bypasses
                    # keypoint_transform.py so we publish the same frame to both.
                    for topic_suffix in (
                        robots.TRANSFORMED_HAND_FRAME,
                        robots.TRANSFORMED_HAND_COORDS,
                    ):
                        self.publisher_manager.publish(
                            host=self.host,
                            port=self.controller_pub_port,
                            topic=f"{side}_{topic_suffix}",
                            data=input_frame,
                        )
                self.timer.end_loop()
        finally:
            for side, socket in self.sockets.items():
                socket.close()
                logger.info(f"Closed controller socket for side={side}")
            logger.info("Stopped VR controller detection process.")

    @staticmethod
    def _trigger_to_width(trigger: float) -> float:
        """Map analog trigger value [0..1] to OpenArm gripper width in meters.

        trigger=0 (released) -> OPENARM_GRIPPER_MAX_WIDTH_M (open)
        trigger=1 (fully pulled) -> OPENARM_GRIPPER_MIN_WIDTH_M (closed)
        Out-of-range values are clamped.
        """
        clamped = max(0.0, min(1.0, float(trigger)))
        return (1.0 - clamped) * robots.OPENARM_GRIPPER_MAX_WIDTH_M

    @staticmethod
    def _parse(raw: bytes):
        """Parse a controller wire-format message.

        Format: '<mode>:px,py,pz|qx,qy,qz,qw|trigger'
        Returns (pos_tuple, quat_tuple, trigger_float, mode_str) on success,
        (None, None, None, None) on any parse failure (caller skips frame).
        """
        try:
            text = raw.decode("utf-8").strip()
            mode, payload = text.split(":", 1)
            if mode not in (robots.RELATIVE, robots.ABSOLUTE):
                logger.warning(f"Unknown controller mode: {mode!r}")
                return (None, None, None, None)
            pos_str, quat_str, trigger_str = payload.split("|")
            pos = tuple(float(v) for v in pos_str.split(","))
            quat = tuple(float(v) for v in quat_str.split(","))
            trigger = float(trigger_str)
            if len(pos) != 3 or len(quat) != 4:
                logger.warning(
                    f"Controller message has wrong arity: pos={len(pos)}, quat={len(quat)}"
                )
                return (None, None, None, None)
            return (pos, quat, trigger, mode)
        except (UnicodeDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse controller message: {e!r}; raw={raw!r}")
            return (None, None, None, None)

    @staticmethod
    def _frame_from_quat(pos, quat) -> np.ndarray:
        """Build a 4×3 frame array [origin; Rx; Ry; Rz] from position + xyzw quaternion.

        The 3 basis-vector rows are the columns of the rotation matrix that
        the quaternion represents. This matches the layout that
        xarm7_operator._turn_frame_to_homo_mat() expects.

        Non-unit quaternions are normalized by scipy.spatial.transform.Rotation.
        """
        rot = Rotation.from_quat(np.asarray(quat, dtype=np.float64))
        rotation_matrix = rot.as_matrix()  # shape (3, 3); columns are basis vectors
        frame = np.empty((4, 3), dtype=np.float64)
        frame[0] = np.asarray(pos, dtype=np.float64)
        frame[1] = rotation_matrix[:, 0]
        frame[2] = rotation_matrix[:, 1]
        frame[3] = rotation_matrix[:, 2]
        return frame

    @staticmethod
    def _rotate_90_around_x(pos, quat):
        """Apply OpenArm-convention 90° rotation around the X axis to a pose.

        Position: (x, y, z) -> (x, -z, y). This matches the same operation
        OculusVRHandDetector applies to keypoints.

        Rotation: pre-multiply the original rotation by the 90°-around-X
        rotation, so that applying the resulting rotation to any vector v
        is equivalent to first applying the original rotation to v and then
        rotating the result around X.
        """
        x, y, z = pos
        rotated_pos = (x, -z, y)

        # Quaternion for 90° around +X axis (xyzw): (sin(45), 0, 0, cos(45))
        s = np.sin(np.pi / 4)
        c = np.cos(np.pi / 4)
        r_x = Rotation.from_quat((s, 0.0, 0.0, c))
        r_orig = Rotation.from_quat(np.asarray(quat, dtype=np.float64))

        # Pre-multiply: combined = R_x ∘ R_orig (apply orig first, then R_x).
        combined = r_x * r_orig
        rotated_quat = tuple(combined.as_quat())  # xyzw
        return rotated_pos, rotated_quat
