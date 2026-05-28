import logging
import math
import threading
import time
from copy import deepcopy as copy
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from beavr.teleop.common.logging.logger import PoseLogger
from beavr.teleop.common.network.handshake import HandshakeCoordinator
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import (
    SerializationError,
    cleanup_zmq_resources,
    get_global_context,
)
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components.detector.detector_types import (
    ButtonEvent,
    InputFrame,
    SessionCommand,
)
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.components.operator.operator_base import Operator
from beavr.teleop.components.operator.operator_types import CartesianTarget, GripperCommand
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)
logger.level = logging.WARNING


class XArmOperator(Operator):
    """
    Base class for controlling an XArm robot arm via teleoperation using VR hand tracking.
    Handles communication, coordinate transformations, filtering, and state management.
    Specific arm configurations (e.g., left/right) should inherit from this class
    and provide the appropriate transformation matrices.
    """

    def __init__(
        self,
        operator_name: str,
        host: str,
        transformed_keypoints_port: int,
        stream_configs: Dict[str, Any],
        stream_oculus: bool,
        endeff_publish_port: int,
        endeff_subscribe_port: int,
        moving_average_limit: int,
        use_filter: bool = True,
        arm_resolution_port: Optional[int] = None,
        teleoperation_state_port: Optional[int] = None,
        logging_config: Optional[Dict[str, Any]] = None,
        hand_side: str = robots.RIGHT,
        controller_keypoints_port: Optional[int] = None,
    ):
        """
        Initializes the XArmOperator.

        Args:
            operator_name: Name for this operator instance (e.g., 'xarm7_right_operator').
            host: Network host address for ZMQ communication.
            transformed_keypoints_port: Port for receiving transformed hand keypoints.
            stream_configs: Configuration for streaming data.
            stream_oculus: Flag indicating if Oculus streaming is used.
            endeff_publish_port: Port for publishing end-effector commands.
            endeff_subscribe_port: Port for subscribing to end-effector state.
            moving_average_limit: Number of samples for moving average filter (currently unused).
            use_filter: Whether to enable the complementary state filter.
            arm_resolution_port: Optional port for arm resolution control messages.
            teleoperation_state_port: Optional port for teleoperation reset/pause messages.
            logging_config: Optional configuration dictionary for pose logging.
            hand_side: Hand side ('left' or 'right') to determine the correct topic for keypoint subscription.
        """
        # Basic initialization
        self.operator_name = operator_name
        self.hand_side = hand_side
        self.notify_component_start(self.operator_name)
        self._host, self._port = host, transformed_keypoints_port

        # Initialize ZMQ context and subscribers
        self._context = get_global_context()

        # Determine the correct topic based on hand side
        if hand_side == robots.RIGHT:
            frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
        else:  # LEFT
            frame_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_FRAME}"

        # Receives InputFrame objects containing frame vectors
        self._arm_transformed_keypoint_subscriber = ZMQSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic=frame_topic,
            context=self._context,
            message_type=InputFrame,
        )

        # Optional subscribers
        self._arm_resolution_subscriber = None
        # TODO: Remove the literal in the topic arg use a constant.
        if arm_resolution_port:
            self._arm_resolution_subscriber = ZMQSubscriber(
                host=host,
                port=arm_resolution_port,
                topic="button",
                context=self._context,
                message_type=ButtonEvent,
            )

        # TODO: Remove the literal in the topic arg use a constant.
        self._arm_teleop_state_subscriber = None
        if teleoperation_state_port:
            self._arm_teleop_state_subscriber = ZMQSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic="pause",
                context=self._context,
                message_type=SessionCommand,
            )

        # Receives CartesianState with h_matrix set
        self.endeff_homo_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_homo",
            context=self._context,
            message_type=CartesianState,
        )

        # Subscribe to transformed hand coordinates for gripper calculation
        coords_topic = f"{hand_side}_{robots.TRANSFORMED_HAND_COORDS}"
        self._hand_coords_subscriber = ZMQSubscriber(
            host=host,
            port=transformed_keypoints_port,
            topic=coords_topic,
            context=self._context,
            message_type=InputFrame,
        )
        self._latest_hand_coords = None

        # Controller-tracking path. The controller detector publishes InputFrames
        # with the same topics on a separate port, so we subscribe to both and
        # pick the source with the fresher timestamp in _get_hand_frame and
        # _extract_gripper_width.
        self._controller_frame_subscriber: Optional[ZMQSubscriber] = None
        self._controller_coords_subscriber: Optional[ZMQSubscriber] = None
        if controller_keypoints_port:
            self._controller_frame_subscriber = ZMQSubscriber(
                host=host,
                port=controller_keypoints_port,
                topic=frame_topic,
                context=self._context,
                message_type=InputFrame,
            )
            self._controller_coords_subscriber = ZMQSubscriber(
                host=host,
                port=controller_keypoints_port,
                topic=coords_topic,
                context=self._context,
                message_type=InputFrame,
            )

        self._subscribers = {
            "endeff_homo": self.endeff_homo_subscriber,
            "teleop_state": self._arm_teleop_state_subscriber,
            "resolution_scale": self._arm_resolution_subscriber,
            "hand_coords": self._hand_coords_subscriber,
            "controller_frame": self._controller_frame_subscriber,
            "controller_coords": self._controller_coords_subscriber,
        }

        gripper_publish_port = robots.OPENARM_LEFT_GRIPPER_CMD_PORT if hand_side == robots.LEFT else robots.OPENARM_RIGHT_GRIPPER_CMD_PORT

        # Using the centralized publisher manager
        self._publisher_manager = ZMQPublisherManager.get_instance(self._context)
        self._publisher_host = host
        self._publisher_port = endeff_publish_port
        self._gripper_publish_port = gripper_publish_port
        self._gripper_width = robots.OPENARM_GRIPPER_MIN_WIDTH_M
        self._first_gripper_publish = True

        self._stream_oculus = stream_oculus
        self.stream_configs = stream_configs

        # State initialization
        self.arm_teleop_state = robots.ARM_TELEOP_CONT
        self.resolution_scale = 1.0
        self.is_first_frame = True
        self._timer = FrequencyTimer(robots.VR_FREQ)
        self._robot = None  # Placeholder for potential robot interface object
        self.real = False  # Placeholder, potentially indicating simulation vs real robot

        # Transformation matrices state
        self.robot_init_h: Optional[np.ndarray] = None
        self.robot_moving_h: Optional[np.ndarray] = None
        self.hand_init_h: Optional[np.ndarray] = None
        self.hand_moving_h: Optional[np.ndarray] = None
        self.hand_init_t: Optional[np.ndarray] = None
        self.last_valid_hand_frame: Optional[np.ndarray] = None  # Cache for last received hand frame

        # Filter setup
        self.use_filter = use_filter
        self.comp_filter: Optional[CompStateFilter] = None

        # Moving average setup (Currently unused in _apply_retargeted_angles)
        self.moving_average_queue = []
        self.moving_average_limit = moving_average_limit
        self.hand_frames = []  # Potentially redundant with moving_average_queue

        # Separate moving average limits for position and orientation (Currently unused)
        self.orientation_average_limit = min(10, moving_average_limit * 2)
        self.orientation_queue = []

        # Track previous orientations for stability detection (Currently unused)
        self.prev_orientation: Optional[np.ndarray] = None
        self.last_sent_orientation: Optional[np.ndarray] = None
        self.ori_update_counter: int = 0

        # Initialize pose logger based on config
        self.logging_config = logging_config or {"enabled": False}

        self.logging_config = {"enabled": True, "filename": "openarm"}

        self.logging_enabled = False  # self.logging_config.get("enabled", False)
        self.pose_logger: Optional[PoseLogger] = None

        if self.logging_enabled:
            log_prefix = self.logging_config.get("filename", self.operator_name)
            logger.info(f"Initializing pose logger for {self.operator_name} with config: {self.logging_config}")
            self.pose_logger = PoseLogger(prefix=log_prefix)
        else:
            self.pose_logger = None

        # Pedal-gated initial reset: block _apply_retargeted_angles until the
        # pedal is pressed for the first time so that the baseline (hand_init_h /
        # robot_init_h) is captured at the moment the operator actually wants to
        # start moving, not at system startup.
        self._waiting_for_first_pedal = True
        self._pedal_lock = threading.Lock()

        # Try to subscribe to /pedal_pressed via ROS2.  If ROS2 is not
        # available in this process the flag is cleared immediately so the
        # operator falls back to the original "reset on startup" behaviour.
        self._pedal_ros2_node = None
        self._pedal_executor = None
        self._pedal_spin_thread = None
        self._init_pedal_subscriber()

        # Initialize handshake coordination for this operator
        self._handshake_coordinator = HandshakeCoordinator.get_instance()
        self._handshake_server_id = f"{operator_name}_handshake"

        # Start handshake server for this operator with unique port.
        # Use a deterministic lookup table rather than Python's hash() which is
        # randomised per-process (PYTHONHASHSEED) and can collide modulo 100.
        # Existing robot-interface handshake ports: right=8159, left=8160.
        _OPERATOR_HANDSHAKE_PORT_OFFSETS = {
            "openarm_left_operator":  11,   # → 8161
            "openarm_right_operator": 12,   # → 8162
            "xarm7_left_operator":    21,   # → 8171
            "xarm7_right_operator":   22,   # → 8172
        }
        operator_port_offset = _OPERATOR_HANDSHAKE_PORT_OFFSETS.get(
            operator_name,
            # Deterministic fallback for unknown names: sum of ordinals, offset to
            # avoid the reserved 0-30 range used by known operators/robots.
            (sum(ord(c) for c in operator_name) % 50) + 50,
        )
        handshake_port = robots.TELEOP_HANDSHAKE_PORT + operator_port_offset

        try:
            self._handshake_coordinator.start_server(
                subscriber_id=self._handshake_server_id,
                bind_host="*",
                port=handshake_port,
            )
            logger.info(f"Handshake server started for {operator_name} on port {handshake_port}")
        except Exception as e:
            logger.warning(f"Failed to start handshake server for {operator_name}: {e}")

    @property
    def timer(self) -> FrequencyTimer:
        """Returns the frequency timer instance."""
        return self._timer

    @property
    def robot(self) -> Any:
        """Returns the robot interface object (placeholder)."""
        return self._robot

    @property
    def transformed_arm_keypoint_subscriber(self) -> ZMQSubscriber:
        """Returns the subscriber for transformed hand keypoints."""
        return self._arm_transformed_keypoint_subscriber

    @property
    def transformed_hand_keypoint_subscriber(self) -> None:
        """Required property from the Operator abstract class, returning None."""
        return None

    def return_real(self) -> bool:
        """Returns whether the operator is controlling a real robot (placeholder)."""
        return self.real

    # ------------------------------------------------------------------
    # Pedal-gated initial reset
    # ------------------------------------------------------------------
    def _init_pedal_subscriber(self) -> None:
        """Subscribe to /pedal_pressed via ROS2 in a background thread.

        If ROS2 is unavailable the waiting flag is cleared so the operator
        behaves as before (reset on startup).
        """
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from std_msgs.msg import Bool

            if not rclpy.ok():
                rclpy.init()

            # Use a unique node name to avoid collisions when both left and
            # right operators are running in the same process.
            node_name = f"xarm_pedal_{self.operator_name.replace('-', '_')}"
            self._pedal_ros2_node = Node(node_name)
            self._pedal_ros2_node.create_subscription(
                Bool,
                "/pedal_pressed",
                self._pedal_pressed_callback,
                10,
            )

            self._pedal_executor = SingleThreadedExecutor()
            self._pedal_executor.add_node(self._pedal_ros2_node)
            self._pedal_spin_thread = threading.Thread(
                target=self._pedal_executor.spin, daemon=True
            )
            self._pedal_spin_thread.start()
            logger.info(
                f"[{self.operator_name}] Pedal subscriber initialised "
                f"(node={node_name}); waiting for first pedal press before reset."
            )
        except Exception as e:
            logger.error(
                f"[{self.operator_name}] Could not initialise ROS2 pedal subscriber "
                f"({e})."
            )

    def _pedal_pressed_callback(self, msg) -> None:
        """Called by the ROS2 executor thread on every /pedal_pressed message.

        On the very first True message we clear the waiting flag and mark
        is_first_frame so _reset_teleop() fires on the next control loop
        iteration, aligning the baselines to the current state.
        """
        if not msg.data:
            return
        with self._pedal_lock:
            if not self._waiting_for_first_pedal:
                return  # Already handled
            self._waiting_for_first_pedal = False
            self.is_first_frame = True  # Ensure _reset_teleop() is triggered
        logger.info(
            f"[{self.operator_name}] First pedal press detected — "
            "resetting teleop baseline on next cycle."
        )

    def _contains_nan(self, arr: np.ndarray) -> bool:
        """Check if numpy array contains any NaN values."""
        if arr is None:
            return True
        return bool(np.any(np.isnan(arr)))

    @staticmethod
    def _pick_fresher(a: Optional[InputFrame], b: Optional[InputFrame]) -> Optional[InputFrame]:
        """Return whichever InputFrame has the most recent timestamp_s."""
        if a is None:
            return b
        if b is None:
            return a
        return a if a.timestamp_s >= b.timestamp_s else b

    def _get_hand_frame(self) -> Optional[np.ndarray]:
        """
        Gets the latest hand frame from the ZMQ subscriber.
        Uses a cached value if no new data is available immediately.

        Returns:
            A 4x3 numpy array representing the hand frame ([t; R_col1; R_col2; R_col3]),
            or None if no valid frame is available.
        """

        # Normal mode: Try to get new data without blocking. When a controller
        # subscriber is configured, take whichever side has the fresher frame.
        hand_data = self._arm_transformed_keypoint_subscriber.recv_keypoints()
        ctrl_data = (
            self._controller_frame_subscriber.recv_keypoints()
            if self._controller_frame_subscriber is not None
            else None
        )
        data = self._pick_fresher(hand_data, ctrl_data)

        source = "none"
        if data is hand_data and hand_data is not None:
            source = "hand"
        elif data is ctrl_data and ctrl_data is not None:
            source = "controller"

        logger.debug(
            f"[{self.operator_name}] _get_hand_frame: hand={'yes' if hand_data else 'no'} "
            f"ctrl={'yes' if ctrl_data else 'no'} selected={source} "
            f"frame_vectors={'yes' if data is not None and data.frame_vectors is not None else 'no'}"
        )

        if data is not None:
            # Process new data - expect InputFrame object with frame_vectors
            try:
                if data.frame_vectors is not None:
                    # frame_vectors should be a sequence of 4 tuples (origin + 3 basis vectors)
                    # Convert from Tuple[Tuple[float, float, float], ...] to numpy array (4, 3)
                    frame_data = np.array(data.frame_vectors, dtype=np.float64).reshape(4, 3)
                    self.last_valid_hand_frame = frame_data  # Cache the new valid frame
                    logger.debug(
                        f"[{self.operator_name}] frame accepted from {source}: "
                        f"origin=({frame_data[0,0]:.3f}, {frame_data[0,1]:.3f}, {frame_data[0,2]:.3f})"
                    )
                    return frame_data
                else:
                    logger.debug(f"[{self.operator_name}] data from {source} has no frame_vectors, skipping")

            except Exception as e:
                logger.error(f"Error processing InputFrame data: {e}")
                # Fall through to return cached frame if processing fails

        # If no new data or processing failed, return the cached frame if it exists
        if self.last_valid_hand_frame is not None:
            logger.debug(f"No new data, returning cached frame")
            return self.last_valid_hand_frame

        # If no new data and no cached frame, return None
        logger.debug("No new data and no cached frame, returning None")
        return None

    def _turn_frame_to_homo_mat(self, frame: np.ndarray) -> np.ndarray:
        """
        Converts a 4x3 frame representation to a 4x4 homogeneous transformation matrix.

        Args:
            frame: A 4x3 numpy array ([t; R_col1; R_col2; R_col3]).

        Returns:
            A 4x4 homogeneous transformation matrix.
        """
        if frame is None or frame.shape != (4, 3):
            raise ValueError("Input frame must be a 4x3 numpy array.")
        t = frame[0]
        r_cols = frame[1:]  # Shape (3, 3), columns of rotation matrix

        homo_mat = np.eye(4)
        # The frame stores columns of R, so transpose r_cols to get R
        homo_mat[:3, :3] = r_cols.T
        homo_mat[:3, 3] = t

        # Check if the rotation matrix is valid before returning
        r_mat = homo_mat[:3, :3]
        if self._contains_nan(r_mat):
            logger.warning(f"Hand frame contains NaN in rotation matrix. Frame:\n{frame}")

        return homo_mat

    def _homo2cart(self, homo_mat: np.ndarray) -> np.ndarray:
        """
        Converts a 4x4 homogeneous matrix to a 7D Cartesian pose vector.

        Args:
            homo_mat: A 4x4 homogeneous transformation matrix.

        Returns:
            A 7D numpy array [x, y, z, qx, qy, qz, qw].
        """
        t = homo_mat[:3, 3]
        # Ensure the rotation matrix is valid before converting to quaternion
        r_mat = self.project_to_rotation_matrix(homo_mat[:3, :3])
        r_quat = Rotation.from_matrix(r_mat).as_quat()  # [qx, qy, qz, qw]

        cart = np.concatenate([t, r_quat], axis=0)
        return cart

    def cart2homo(self, cart: np.ndarray) -> np.ndarray:
        """
        Converts a 7D Cartesian pose vector back to a 4x4 homogeneous matrix.

        Args:
            cart: A 7D numpy array [x, y, z, qx, qy, qz, qw].

        Returns:
            A 4x4 homogeneous transformation matrix.
        """
        if cart is None or cart.shape != (7,):
            raise ValueError("Input cart must be a 7D numpy array.")
        homo = np.eye(4)
        t = cart[:3]
        # Normalize quaternion before converting to matrix
        quat = cart[3:]
        norm = np.linalg.norm(quat)
        if norm > 1e-6:  # Avoid division by zero
            quat /= norm
        else:
            # Handle zero quaternion case (e.g., default to identity rotation)
            quat = np.array([0.0, 0.0, 0.0, 1.0])

        r_mat = Rotation.from_quat(quat).as_matrix()
        homo[:3, 3] = t
        homo[:3, :3] = r_mat
        return homo

    def project_to_rotation_matrix(self, r_mat: np.ndarray) -> np.ndarray:
        """
        Adjusts a near-rotation 3x3 matrix to be a valid SO(3) rotation matrix using SVD.
        Ensures the determinant is +1 (removes reflections).

        Args:
            r_mat: A 3x3 numpy array, potentially close to a rotation matrix.

        Returns:
            A valid 3x3 rotation matrix.
        """
        try:
            u, _, vt = np.linalg.svd(r_mat)  # Perform SVD
            r_fixed = u @ vt  # Reconstruct the rotation matrix

            # Ensure determinant is +1 (no reflection)
            if np.linalg.det(r_fixed) < 0:
                vt[-1, :] *= -1  # Flip the sign of the last row of Vt
                # Note: Adjusting Vt is generally preferred over U for fixing determinant
                r_fixed = u @ vt  # Recalculate R
            return r_fixed
        except np.linalg.LinAlgError:
            logger.warning("SVD did not converge. Returning identity matrix.")
            return np.eye(3)  # Fallback

    def _get_resolution_scale_mode(self) -> float:
        # TODO: We may not need this anymore I am not too sure what the use case is.
        # Instead we can default or make this configurable but do we really need it
        # during real time operation?

        """Gets the resolution scale mode from the subscriber."""
        if not self._arm_resolution_subscriber:
            return 1.0  # default if subscriber not configured

        # Use NOBLOCK to avoid waiting if no message is present
        data = self._arm_resolution_subscriber.recv_keypoints()
        if data is None:
            # Keep the current resolution scale if no new message
            return self.resolution_scale
        try:
            # Expect ButtonEvent
            scale_mode = data.value

            # Update internal resolution scale based on mode
            if scale_mode == robots.ARM_HIGH_RESOLUTION:
                self.resolution_scale = 1.0
            elif scale_mode == robots.ARM_LOW_RESOLUTION:
                self.resolution_scale = 0.6
            return self.resolution_scale  # Return the updated scale
        except Exception as e:
            logger.error(f"Error processing resolution scale data: {e}")
            return self.resolution_scale  # Return current scale on error

    def _get_arm_teleop_state(self) -> int:
        """Gets the arm teleoperation state (STOP/CONT) from the subscriber."""
        if not self._arm_teleop_state_subscriber:
            # Default to CONT if no subscriber, assuming continuous operation unless stopped externally
            return robots.ARM_TELEOP_CONT

        # Use NOBLOCK to avoid waiting
        data = self._arm_teleop_state_subscriber.recv_keypoints()
        if data is None:
            return self.arm_teleop_state  # Return current state if no new message
        try:
            # Expect SessionCommand
            if data.command == robots.PAUSE:
                return robots.ARM_TELEOP_STOP
            elif data.command == robots.RESUME:
                return robots.ARM_TELEOP_CONT
            else:
                return self.arm_teleop_state

        except Exception:
            return self.arm_teleop_state  # Return current state on error

    # ------------------------------
    # Teleop reset logic
    # ------------------------------
    def _reset_teleop(self) -> Optional[np.ndarray]:
        """
        Resets the teleoperation baseline by capturing current robot and hand poses.
        Sends a reset signal and waits for the robot's current pose.

        Returns:
            The initial moving hand frame (4x3) captured after reset, or None on failure.
        """

        logger.info(f"****** {self.operator_name}: RESETTING TELEOP ******")
        # Request robot's current pose using a typed contract
        # TODO: Remove the literal in the topic arg use a constant.
        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._publisher_port,
            topic="reset",
            data=SessionCommand(timestamp_s=time.time(), command="reset"),
        )
        robot_frame_homo = self.endeff_homo_subscriber.recv_keypoints()

        logger.info(f"****** {self.operator_name}: RESETTING TELEOP waiting for robot******")
        # Keep trying until we get a response
        while robot_frame_homo is None:
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._publisher_port,
                topic="reset",
                data=SessionCommand(timestamp_s=time.time(), command="reset"),
            )
            robot_frame_homo = self.endeff_homo_subscriber.recv_keypoints()
            time.sleep(0.01)

        logger.info(f"****** {self.operator_name}: RESETTING TELEOP --- robot_frame_homo ******")
        try:
            h = np.array(robot_frame_homo.h_matrix, dtype=np.float64).reshape(4, 4)
            self.robot_init_h = h
            # Validate if it's close to a homogeneous matrix
            if not np.allclose(self.robot_init_h[3, :], [0, 0, 0, 1]):
                logger.warning(
                    f"Warning ({self.operator_name}): Received robot frame is not a valid homogeneous matrix. Resetting bottom row."
                )
                self.robot_init_h[3, :] = [0, 0, 0, 1]
            # Ensure rotation part is valid SO(3)
            self.robot_init_h[:3, :3] = self.project_to_rotation_matrix(self.robot_init_h[:3, :3])

        except Exception:
            # logger.error(f"ERROR ({self.operator_name}): Failed to process received robot frame: {e}")
            self.is_first_frame = True  # Stay in reset state
            return None

        self.robot_moving_h = copy(self.robot_init_h)
        logger.info(f"{self.operator_name} Robot init H:\n{self.robot_init_h}")

        first_hand_frame = None
        while first_hand_frame is None:
            first_hand_frame = self._get_hand_frame()
            time.sleep(0.01)

        try:
            self.hand_init_h = self._turn_frame_to_homo_mat(first_hand_frame)
            self.hand_init_t = copy(self.hand_init_h[:3, 3])  # Store initial hand translation

            # Validate that the rotation matrix is valid (near-orthonormal)
            r_mat = self.hand_init_h[:3, :3]
            # Check if rotation part is valid using SVD
            r_fixed = self.project_to_rotation_matrix(r_mat)
            if np.allclose(r_mat, r_fixed, atol=1e-6):
                logger.info(f"{self.operator_name} Hand init H:\n{self.hand_init_h}")
            else:
                logger.warning(
                    f"WARNING ({self.operator_name}): Initial hand frame rotation matrix was invalid, corrected via SVD"
                )
                self.hand_init_h[:3, :3] = r_fixed
                logger.info(f"{self.operator_name} Hand init H (corrected):\n{self.hand_init_h}")
        except ValueError as e:
            logger.error(f"ERROR ({self.operator_name}): Failed to convert initial hand frame to matrix: {e}")
            self.is_first_frame = True  # Stay in reset state
            return None

        self.is_first_frame = False  # Reset successful
        self.comp_filter = None  # Reset filter, will be initialized on first _apply call
        logger.info(f"{self.operator_name}: TELEOP RESET COMPLETE")
        logger.info(f"[{self.operator_name}] hand_init_h\n{self.hand_init_h}")
        return first_hand_frame  # Return the frame used for initialization

    # ------------------------------
    # Main teleop: transforms
    # ------------------------------
    def _fix_quaternion_flips(self, quats: np.ndarray) -> np.ndarray:
        """
        Ensures consistency in quaternion representation by preventing flips
        across hemispheres relative to the first quaternion in the sequence.

        Args:
            quats: A numpy array of quaternions (Nx4).

        Returns:
            A numpy array of quaternions (Nx4) with flips corrected.
        """
        if quats is None or len(quats) <= 1:
            return quats

        fixed = [quats[0]]  # First quaternion as reference
        for q in quats[1:]:
            # Calculate dot product with the *previous fixed* quaternion
            dot = np.sum(fixed[-1] * q)
            # If negative, the angle is > 90 degrees, meaning it's in the opposite hemisphere. Flip it.
            if dot < 0:
                fixed.append(-q)
            else:
                fixed.append(q)
        return np.array(fixed)

    def _extract_gripper_width(self) -> float:
        """
        Computes continuous gripper width from hand keypoints OR a pre-computed
        gripper_width_m field on the incoming InputFrame.

        When the detector provides gripper_width_m explicitly (controller path),
        we prefer it. Otherwise we fall back to thumb-tip / index-tip distance
        mapping (hand-tracking path).

        Returns:
            Float representing gripper width in meters
        """
        # Get latest hand coordinates. With both hand and controller paths
        # active, prefer whichever source has the fresher InputFrame.
        hand_coords = self._hand_coords_subscriber.recv_keypoints()
        ctrl_coords = (
            self._controller_coords_subscriber.recv_keypoints()
            if self._controller_coords_subscriber is not None
            else None
        )
        coords_data = self._pick_fresher(hand_coords, ctrl_coords)
        if coords_data is None:
            # No new data, return previous width (initialize to 0 if not set)
            return getattr(self, "_gripper_width", robots.OPENARM_GRIPPER_MIN_WIDTH_M)

        # Prefer pre-computed width when the detector provides one (e.g. controller path).
        if coords_data.gripper_width_m is not None:
            width = float(coords_data.gripper_width_m)
            width = max(
                robots.OPENARM_GRIPPER_MIN_WIDTH_M,
                min(width, robots.OPENARM_GRIPPER_MAX_WIDTH_M),
            )
            self._gripper_width = width
            logger.debug(
                f"[{self.operator_name}] Gripper width (explicit): "
                f"{self._gripper_width * 1000:.1f}mm"
            )
            return self._gripper_width

        if coords_data.keypoints is None or len(coords_data.keypoints) == 0:
            return getattr(self, "_gripper_width", robots.OPENARM_GRIPPER_MIN_WIDTH_M)

        # Convert keypoints to numpy array and get thumb and index finger tip positions
        keypoints = np.array(coords_data.keypoints, dtype=np.float64).reshape(-1, 3)
        thumb_tip = keypoints[robots.OCULUS_JOINTS["thumb_tip"]]
        index_tip = keypoints[robots.OCULUS_JOINTS["index_tip"]]

        # Calculate distance between thumb and index fingertips
        distance = np.linalg.norm(thumb_tip - index_tip)
        logger.debug(f"[gripper] current index thumb distance: {distance * 1000:.1f}mm")

        # Clamp distance to valid range [0, 9cm]
        clamped_distance = min(distance, robots.OPENARM_GRIPPER_THRESHOLD_M)

        # Map distance to gripper width: 0-9cm -> 0-4.5cm
        # Linear mapping: width = (distance / max_distance) * max_width
        gripper_width = (clamped_distance / robots.OPENARM_GRIPPER_THRESHOLD_M) * robots.OPENARM_GRIPPER_MAX_WIDTH_M

        # Ensure gripper width is within bounds
        gripper_width = max(robots.OPENARM_GRIPPER_MIN_WIDTH_M, min(gripper_width, robots.OPENARM_GRIPPER_MAX_WIDTH_M))

        self._gripper_width = gripper_width

        logger.debug(
            f"[{self.operator_name}] Gripper width: {self._gripper_width * 1000:.1f}mm "
            f"(distance={clamped_distance * 1000:.1f}mm)"
        )

        return self._gripper_width

    def _apply_retargeted_angles(self):
        """
        Calculates and applies the retargeted end-effector pose based on hand motion.
        Handles state changes (reset, pause/resume), applies transformations,
        filters the result, and publishes the command.
        """
        # Block until the pedal has been pressed for the first time so the
        # baseline is captured at the moment the operator intends to start
        # moving rather than at system startup.
        with self._pedal_lock:
            waiting = self._waiting_for_first_pedal
        if waiting:
            return

        frame_start_time = time.time()

        # 1. Check for state changes (Pause/Resume, Resolution)
        new_arm_teleop_state = self._get_arm_teleop_state()
        self.resolution_scale = self._get_resolution_scale_mode()  # Update resolution scale

        # Determine if a reset is needed
        needs_reset = self.is_first_frame or (
            self.arm_teleop_state == robots.ARM_TELEOP_STOP and new_arm_teleop_state == robots.ARM_TELEOP_CONT
        )

        # Update state *after* checking for transition
        self.arm_teleop_state = new_arm_teleop_state

        # Decide whether we should publish commands this cycle
        publish_commands = self.arm_teleop_state == robots.ARM_TELEOP_CONT

        logger.debug(
            f"[{self.operator_name}] state={self.arm_teleop_state} "
            f"needs_reset={needs_reset} publish={publish_commands}"
        )

        # 2. Handle Reset Condition
        if needs_reset:
            logger.debug(f"Attempting reset for {self.operator_name}")
            moving_hand_frame = self._reset_teleop()
            if moving_hand_frame is None:
                logger.error(f"ERROR ({self.operator_name}): Reset failed, cannot proceed.")
                return  # Exit if reset failed
            logger.debug(f"Reset successful, got hand frame: {moving_hand_frame}")
            # Reset is done, is_first_frame is now False
        else:
            # 3. Get Current Hand Frame (if not resetting)
            moving_hand_frame = self._get_hand_frame()
            logger.debug(f"Got hand frame: {moving_hand_frame}")

        # If no valid hand frame is available (after reset or during normal operation), exit
        if moving_hand_frame is None:
            logger.warning(f"Warning ({self.operator_name}): No valid hand frame received, skipping cycle.")
            return

        # Ensure initial robot/hand poses are set (should be handled by reset)
        if self.robot_init_h is None or self.hand_init_h is None:
            logger.error(f"ERROR ({self.operator_name}): Initial robot or hand poses not set. Triggering reset.")
            self.is_first_frame = True  # Force reset on next cycle
            return

        # 4. Convert current hand frame to Homogeneous Matrix
        try:
            self.hand_moving_h = self._turn_frame_to_homo_mat(moving_hand_frame)
        except ValueError as e:
            logger.error(f"Error ({self.operator_name}): Could not convert moving hand frame: {e}")
            return  # Skip cycle if conversion fails

        # 5. Compute relative hand motion in hand-tracking frame.
        t_init = self.hand_init_h[:3, 3]
        t_cur  = self.hand_moving_h[:3, 3]
        R_init = self.hand_init_h[:3, :3]
        R_cur  = self.hand_moving_h[:3, :3]

        h_ht_hi = np.eye(4)
        h_ht_hi[:3, :3] = R_cur @ R_init.T  # relative rotation in hand-tracking frame
        h_ht_hi[:3, 3]  = t_cur - t_init    # relative translation in hand-tracking frame

        np.set_printoptions(
            precision=3,      # 3 digits after decimal point
            suppress=True,    # avoid scientific notation for small numbers
            linewidth=10_000  # prevent line breaks
        )

        # 6. Map relative motion to robot base frame via a single similarity transform.
        relative_affine_in_robot_frame = h_ht_hi
        relative_affine_in_robot_frame[:3, :3] = self.project_to_rotation_matrix(
            relative_affine_in_robot_frame[:3, :3]
        )
        
        # 7. Calculate target robot pose.
        h_rt_rh = np.eye(4)                                                                                                                
        h_rt_rh[:3, :3] = relative_affine_in_robot_frame[:3, :3] @ self.robot_init_h[:3, :3]                                               
        h_rt_rh[:3, 3]  = self.robot_init_h[:3, 3] + relative_affine_in_robot_frame[:3, 3]

        # Ensure the final target pose has a valid rotation matrix.
        h_rt_rh[:3, :3] = self.project_to_rotation_matrix(h_rt_rh[:3, :3])

        # Store the calculated target pose
        self.robot_moving_h = copy(h_rt_rh)  

        # Log positions for debugging
        logger.debug(
            f"{self.operator_name} - robot_init_h pos: {self.robot_init_h[:3, 3]}, "
            f"robot_moving_h pos: {self.robot_moving_h[:3, 3]}, "
            f"relative translation (robot frame): {relative_affine_in_robot_frame[:3, 3]}"
        )

        # 8. Convert Target Pose to Cartesian [pos, quat]
        cart_target_raw = self._homo2cart(self.robot_moving_h)

        # 9. Apply Filtering
        if self.use_filter:
            # Initialize filter on the first valid frame after reset/start
            if self.comp_filter is None:
                # Use the *raw* target pose from the first frame as the initial filter state
                self.comp_filter = CompStateFilter(
                    init_state=cart_target_raw,
                    pos_ratio=0.7,  # Default values, consider making configurable
                    ori_ratio=0.85,
                    adaptive=True,
                )
                cart_target_filtered = cart_target_raw  # Use raw value for the very first frame
            else:
                cart_target_filtered = self.comp_filter(cart_target_raw)
        else:
            cart_target_filtered = cart_target_raw  # No filtering

        # 10. Prepare filtered pose for publishing (quaternion orientation, positive hemisphere)
        position = cart_target_filtered[0:3]
        orientation_quat = cart_target_filtered[3:7].copy()

        # Normalise and force the quaternion into the *positive* hemisphere (w >= 0)
        norm = np.linalg.norm(orientation_quat)
        if norm < 1e-6:
            orientation_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        else:
            orientation_quat = orientation_quat / norm
            if orientation_quat[3] < 0:  # w component negative → flip sign
                orientation_quat = -orientation_quat

        # 11. Build contract to publish directly
        cartesian_cmd = CartesianTarget(
            timestamp_s=time.time(),
            hand_side=self.hand_side,
            frame_id="world",
            position_m=(float(position[0]), float(position[1]), float(position[2])),
            orientation_xyzw=(
                float(orientation_quat[0]),
                float(orientation_quat[1]),
                float(orientation_quat[2]),
                float(orientation_quat[3]),
            ),
        )

        logger.debug(
            f"[{self.operator_name}] cartesian_cmd: "
            f"pos=({cartesian_cmd.position_m[0]:.4f}, {cartesian_cmd.position_m[1]:.4f}, {cartesian_cmd.position_m[2]:.4f}) "
            f"quat=({cartesian_cmd.orientation_xyzw[0]:.3f}, {cartesian_cmd.orientation_xyzw[1]:.3f}, "
            f"{cartesian_cmd.orientation_xyzw[2]:.3f}, {cartesian_cmd.orientation_xyzw[3]:.3f}) "
            f"publishing={publish_commands}"
        )

        # Publish only if tele-operation is in CONT mode
        if publish_commands:
            try:
                self._publisher_manager.publish(
                    host=self._publisher_host,
                    port=self._publisher_port,
                    topic="endeff_coords",
                    data=cartesian_cmd,
                )
                # logger.info(f"Published end-effector command: {command_data}")

                # Extract and publish gripper width
                gripper_width_m = self._extract_gripper_width()

                gripper_cmd = GripperCommand(
                    timestamp_s=time.time(),
                    hand_side=self.hand_side,
                    width_m=gripper_width_m,
                    speed_mps=robots.OPENARM_GRIPPER_DEFAULT_SPEED_MPS,
                )

                self._publisher_manager.publish(
                    host=self._publisher_host,
                    port=self._gripper_publish_port,
                    topic="gripper_cmd",
                    data=gripper_cmd,
                )
                logger.debug(f"[{self.operator_name}] Published gripper command: width={gripper_width_m:.3f}m")
            except (ConnectionError, SerializationError) as e:
                logger.error(f"Failed to publish end-effector command: {e}")
            except Exception as e:
                logger.error(f"Unexpected error publishing command: {e}")
        else:
            logger.debug(
                f"Skipping command publication: publish_commands={publish_commands}, arm_teleop_state={self.arm_teleop_state}"
            )

        # 12. Logging (Optional)
        if self.logging_enabled and self.pose_logger:
            try:
                frame_processing_time = time.time() - frame_start_time
                # Identify which matrices have NaN values
                nan_matrices = []
                if self.hand_init_h is None or self._contains_nan(self.hand_init_h):
                    nan_matrices.append("hand_init_h")
                if self.robot_init_h is None or self._contains_nan(self.robot_init_h):
                    nan_matrices.append("robot_init_h")
                if self.hand_moving_h is None or self._contains_nan(self.hand_moving_h):
                    nan_matrices.append("hand_moving_h")
                if self.robot_moving_h is None or self._contains_nan(self.robot_moving_h):
                    nan_matrices.append("robot_moving_h")
                if h_ht_hi is None or self._contains_nan(h_ht_hi):
                    nan_matrices.append("h_ht_hi")
                if relative_affine_in_robot_frame is None or self._contains_nan(relative_affine_in_robot_frame):
                    nan_matrices.append("relative_affine_in_robot_frame")
                if cart_target_raw is None or self._contains_nan(cart_target_raw):
                    nan_matrices.append("cart_target_raw")
                if cart_target_filtered is None or self._contains_nan(cart_target_filtered):
                    nan_matrices.append("cart_target_filtered")

                # Only log if all matrices are valid
                if not nan_matrices:
                    self.pose_logger.log_transformation_pipeline(
                        hand_init_h=self.hand_init_h,
                        robot_init_h=self.robot_init_h,
                        hand_moving_h=self.hand_moving_h,
                        h_ht_hi=h_ht_hi,
                        T_coord=self._T_coord,
                        relative_affine=relative_affine_in_robot_frame,
                        robot_moving_h=self.robot_moving_h,
                        cart_target_raw=cart_target_raw,
                        cart_target_filtered=cart_target_filtered,
                        resolution_scale=self.resolution_scale,
                        frame_processing_time=frame_processing_time,
                    )
                    logger.debug(f"Logged transformation pipeline frame {self.pose_logger.frame_count - 1}")
                else:
                    logger.warning(f"Skipping logging due to NaN values in matrices: {', '.join(nan_matrices)}")
                    logger.info(f"hand hand moving frame: {self.hand_moving_h}")
            except Exception as e:
                logger.error(f"Error logging transformation pipeline ({self.operator_name}): {e}")

    def moving_average(self, action: np.ndarray, queue: list, limit: int) -> np.ndarray:
        """
        Applies a simple moving average filter to the input action.
        Note: This is currently not used in the main `_apply_retargeted_angles` loop.

        Args:
            action: The data point (e.g., pose vector) to add.
            queue: The list acting as the moving average queue.
            limit: The maximum size of the queue.

        Returns:
            The averaged action.
        """
        queue.append(action)
        if len(queue) > limit:
            queue.pop(0)
        # Ensure queue is not empty before calculating mean
        if not queue:
            return action  # Or return np.zeros_like(action) or raise error
        return np.mean(queue, axis=0)

    def cleanup(self) -> None:
        """Shut down the ROS2 pedal subscriber, then delegate to the base class."""
        try:
            if self._pedal_executor is not None:
                self._pedal_executor.shutdown()
                self._pedal_executor = None
            if self._pedal_ros2_node is not None:
                self._pedal_ros2_node.destroy_node()
                self._pedal_ros2_node = None
        except Exception as e:
            logger.warning(f"[{self.operator_name}] Error shutting down pedal ROS2 node: {e}")
        super().cleanup()

    def run(self):
        # TODO: Call this method stream to align with rest of the codebase
        """The main execution loop for the operator."""
        try:
            while True:
                with self.timer:  # Ensures loop runs at desired frequency (e.g., VR_FREQ)
                    self._apply_retargeted_angles()
        except KeyboardInterrupt:
            logger.info(f"{self.operator_name} received KeyboardInterrupt. Cleaning up...")
        finally:
            self.cleanup()

    def __del__(self):
        """Destructor ensures cleanup is called."""
        # Safely clean up subscribers if they were initialized
        if hasattr(self, "_subscribers") and self._subscribers:
            for subscriber in self._subscribers.values():
                if subscriber is not None:  # Check if subscriber is not None
                    try:
                        subscriber.stop()
                    except Exception as e:
                        logger.warning(f"Error stopping subscriber in {getattr(self, 'operator_name', 'unknown')}: {e}")

        # Stop handshake server if it exists
        if hasattr(self, "_handshake_coordinator") and hasattr(self, "_handshake_server_id"):
            try:
                self._handshake_coordinator.stop_server(self._handshake_server_id)
            except Exception as e:
                logger.warning(f"Error stopping handshake server: {e}")

        cleanup_zmq_resources()
