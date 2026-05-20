import json
import logging
import time
from copy import deepcopy as copy
from datetime import datetime
from enum import IntEnum
from pathlib import Path

import numpy as np

from beavr.teleop.common.math.vectorops import moving_average, normalize_vector
from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import cleanup_zmq_resources
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import InputFrame
from beavr.teleop.components.detector.vr.log_keypoints import KeypointLogger
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)


def contains_nan(arr: np.ndarray) -> bool:
    """Check if numpy array contains any NaN or Inf values."""
    if arr is None:
        return True
    if not isinstance(arr, np.ndarray):
        return False
    bool_result = bool(np.any(np.isnan(arr)))
    if bool_result:
        pass
    return bool_result or bool(np.any(np.isinf(arr)))


def check_array_validity(arr: np.ndarray, name: str) -> bool:
    """Check if array is valid (contains no NaN or Inf values)."""
    if arr is None:
        return False
    if not isinstance(arr, np.ndarray):
        return False
    return not (bool(np.any(np.isnan(arr))) or bool(np.any(np.isinf(arr))))


class HandMode(IntEnum):
    ABSOLUTE = 1
    RELATIVE = 0


class TransformHandPositionCoords(Component):
    def __init__(
        self,
        host: str,
        keypoint_sub_port: int,
        keypoint_transform_pub_port: int,
        hand_side: str = robots.RIGHT,
        moving_average_limit: int = 5,
        enable_logging: bool = False,
        log_dir: str = "data/keypoint_logs",
        auto_save_interval: int = 100,
    ):
        """
        Initialize the unified keypoint transform component for both left and right hands.

        Args:
            host: Network host address
            keypoint_sub_port: Port to subscribe to keypoints from
            keypoint_transform_pub_port: Port to publish transformed keypoints to
            hand_side: 'right' or 'left' to specify which hand to process
            moving_average_limit: Number of frames for moving average smoothing
            enable_logging: Flag to enable/disable frame logging (default: False)
            log_dir: Directory to save log files (default: "data/keypoint_logs")
            auto_save_interval: Number of frames between auto-saves (default: 100)
        """
        # Validate hand_side parameter
        if hand_side not in [robots.LEFT, robots.RIGHT]:
            raise ValueError(f"hand_side must be {robots.LEFT} or {robots.RIGHT}")

        self.hand_side = hand_side

        # Notify component start with appropriate name
        component_name = f"{hand_side}_hand_keypoint_transform"
        self.notify_component_start(component_name)

        # Store connection info
        self.host = host
        self.keypoint_sub_port = keypoint_sub_port
        self.keypoint_transform_pub_port = keypoint_transform_pub_port

        # Initialize subscriber based on hand side
        if hand_side == robots.RIGHT:
            self.keypoint_subscriber = ZMQSubscriber(self.host, self.keypoint_sub_port, robots.RIGHT)
        else:  # left hand
            self.keypoint_subscriber = ZMQSubscriber(self.host, self.keypoint_sub_port, robots.LEFT)

        # Use publisher manager for both hands consistently
        self.publisher_manager = ZMQPublisherManager.get_instance()

        # Define topic names based on hand side
        if hand_side == robots.RIGHT:
            self.coords_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_COORDS}"
            self.frame_topic = f"{robots.RIGHT}_{robots.TRANSFORMED_HAND_FRAME}"
            self.absolute_mode = robots.ABSOLUTE
            self.relative_mode = robots.RELATIVE
        else:
            self.coords_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_COORDS}"
            self.frame_topic = f"{robots.LEFT}_{robots.TRANSFORMED_HAND_FRAME}"
            self.absolute_mode = robots.ABSOLUTE
            self.relative_mode = robots.RELATIVE

        # Timer
        self.timer = FrequencyTimer(robots.VR_FREQ)

        # Define key landmark indices for stable frame calculation
        self.wrist_idx = 0  # Wrist is typically the first point
        self.index_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][0]  # Index finger knuckle
        self.middle_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][1]  # Middle finger knuckle
        self.pinky_knuckle_idx = robots.OCULUS_JOINTS["knuckles"][-1]  # Pinky finger knuckle

        # Moving average queue
        self.moving_average_limit = moving_average_limit
        # Create a queue for moving average
        self.coord_moving_average_queue, self.frame_moving_average_queue = [], []

        # Initialize keypoint logger if enabled
        self.keypoint_logger = None
        if enable_logging:
            self.keypoint_logger = KeypointLogger(
                hand_side=hand_side,
                log_dir=log_dir,
                auto_save_interval=auto_save_interval,
                moving_average_limit=moving_average_limit,
            )

        # Raw keypoint recording for debugging
        self.raw_keypoint_records = []
        self.raw_keypoint_log_file = None
        if enable_logging:
            raw_log_dir = Path(log_dir)
            raw_log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.raw_keypoint_log_file = raw_log_dir / f"raw_keypoints_{hand_side}_{timestamp}.json"

    def _log_raw_keypoints(self, keypoints: np.ndarray, is_relative: bool):
        """Record raw keypoints from VR device for debugging purposes."""
        if self.raw_keypoint_log_file is None:
            return

        record = {
            "timestamp": time.time(),
            "shape": list(keypoints.shape),
            "size": int(keypoints.size),
            "is_relative": is_relative,
            "keypoints": keypoints.tolist() if keypoints.size < 1000 else f"<large array: {keypoints.size} elements>",
        }
        self.raw_keypoint_records.append(record)

        # Auto-save every 100 records
        if len(self.raw_keypoint_records) % 100 == 0:
            self._save_raw_keypoints()

    def _save_raw_keypoints(self):
        """Save raw keypoint records to JSON file."""
        if self.raw_keypoint_log_file is None or len(self.raw_keypoint_records) == 0:
            return

        try:
            data = {
                "hand_side": self.hand_side,
                "total_records": len(self.raw_keypoint_records),
                "records": self.raw_keypoint_records.copy(),
            }
            with open(self.raw_keypoint_log_file, "w") as f:
                json.dump(data, f, indent=2)
            # logger.debug(f"Saved {len(self.raw_keypoint_records)} raw keypoint records")
            self.raw_keypoint_records.clear()
        except Exception as e:
            logger.error(f"Error saving raw keypoints: {e}")

    def _get_hand_coords(self):
        """Get hand coordinates from the subscriber.

        Returns:
            Tuple of (data_type, coordinates) or (None, None) if no data received
        """
        try:
            input_frame = self.keypoint_subscriber.recv_keypoints()
            if input_frame is None:
                return None, None

            # Extract keypoints from InputFrame object
            keypoints = np.asanyarray(input_frame.keypoints)

            if not check_array_validity(keypoints, "raw_keypoints"):
                self._log_raw_keypoints(keypoints, input_frame.is_relative)
                return None, None

            self._log_raw_keypoints(keypoints, input_frame.is_relative)

            data_type = self.relative_mode if input_frame.is_relative else self.absolute_mode

            expected_size = robots.OCULUS_NUM_KEYPOINTS * 3
            if keypoints.size != expected_size:
                return None, None

            reshaped_keypoints = keypoints.reshape(robots.OCULUS_NUM_KEYPOINTS, 3)
            if not check_array_validity(reshaped_keypoints, "reshaped_keypoints"):
                return None, None

            return data_type, reshaped_keypoints

        except Exception as e:
            logger.error(f"Error in _get_hand_coords: {e}", exc_info=True)
            return None, None

    def _orthogonalize_frame(self, x_vec, y_vec, z_vec):
        """Ensure three vectors form an orthogonal frame using Gram-Schmidt process"""
        if (
            not check_array_validity(x_vec, "x_vec_orthogonalize")
            or not check_array_validity(y_vec, "y_vec_orthogonalize")
            or not check_array_validity(z_vec, "z_vec_orthogonalize")
        ):
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])

        x_vec = x_vec.astype(np.float64)
        y_vec = y_vec.astype(np.float64)
        z_vec = z_vec.astype(np.float64)

        # Ensure x is normalized
        x_vec = normalize_vector(x_vec)

        # Make y orthogonal to x
        y_vec = y_vec - np.dot(y_vec, x_vec) * x_vec
        y_vec = normalize_vector(y_vec)

        # Make z orthogonal to both x and y
        z_vec = np.cross(x_vec, y_vec)
        z_vec = normalize_vector(z_vec)

        return x_vec, y_vec, z_vec

    def _get_stable_coord_frame(self, hand_coords):
        """Create a more stable coordinate frame using multiple hand landmarks"""
        try:
            if not check_array_validity(hand_coords, "hand_coords_stable_frame"):
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

            wrist = hand_coords[self.wrist_idx]
            if contains_nan(wrist):
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            if contains_nan(hand_coords[self.index_knuckle_idx]):
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            if contains_nan(hand_coords[self.pinky_knuckle_idx]):
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            if contains_nan(hand_coords[self.middle_knuckle_idx]):
                return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

            v1 = hand_coords[self.index_knuckle_idx] - wrist
            v2 = hand_coords[self.pinky_knuckle_idx] - wrist
            v3 = hand_coords[self.middle_knuckle_idx] - wrist

            cross_v1_v3 = np.cross(v1, v3)
            palm_normal = normalize_vector(cross_v1_v3)
            if not check_array_validity(palm_normal, "palm_normal"):
                palm_normal = np.array([0.0, 0.0, 1.0])

            avg_direction = (v1 + v2 + v3) / 3
            palm_direction = normalize_vector(avg_direction)
            if not check_array_validity(palm_direction, "palm_direction"):
                palm_direction = np.array([0.0, 1.0, 0.0])

            cross_product = normalize_vector(np.cross(palm_direction, palm_normal))
            if not check_array_validity(cross_product, "cross_product"):
                cross_product = np.array([1.0, 0.0, 0.0])

            x_vec, y_vec, z_vec = self._orthogonalize_frame(cross_product, palm_direction, palm_normal)
            return [x_vec, y_vec, z_vec]

        except Exception as e:
            logger.error(f"Error in _get_stable_coord_frame: {e}", exc_info=True)
            return [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

    def _get_stable_hand_dir_frame(self, hand_coords):
        """Create a more stable frame for hand direction using multiple landmarks"""
        try:
            if not check_array_validity(hand_coords, "hand_coords_dir_frame"):
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

            wrist = hand_coords[self.wrist_idx]
            if contains_nan(wrist):
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            if contains_nan(hand_coords[self.index_knuckle_idx]):
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            if contains_nan(hand_coords[self.pinky_knuckle_idx]):
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
            if contains_nan(hand_coords[self.middle_knuckle_idx]):
                return [np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

            wrist = wrist.copy()
            v1 = hand_coords[self.index_knuckle_idx] - wrist
            v2 = hand_coords[self.pinky_knuckle_idx] - wrist
            v3 = hand_coords[self.middle_knuckle_idx] - wrist

            palm_normal = normalize_vector(np.cross(v1, v3))
            if not check_array_validity(palm_normal, "palm_normal_dir"):
                palm_normal = np.array([0.0, -1.0, 0.0])

            palm_direction = normalize_vector((v1 + v2 + v3) / 3)
            if not check_array_validity(palm_direction, "palm_direction_dir"):
                palm_direction = np.array([0.0, 0.0, 1.0])

            cross_product = normalize_vector(np.cross(palm_direction, palm_normal))
            if not check_array_validity(cross_product, "cross_product_dir"):
                cross_product = np.array([1.0, 0.0, 0.0])

            x_vec, y_vec, z_vec = self._orthogonalize_frame(cross_product, palm_normal, palm_direction)
            if not check_array_validity(x_vec, "x_vec_final"):
                x_vec = np.array([1.0, 0.0, 0.0])
            if not check_array_validity(y_vec, "y_vec_final"):
                y_vec = np.array([0.0, 1.0, 0.0])
            if not check_array_validity(z_vec, "z_vec_final"):
                z_vec = np.array([0.0, 0.0, 1.0])

            return [wrist, x_vec, y_vec, z_vec]

        except Exception as e:
            logger.error(f"Error in _get_stable_hand_dir_frame: {e}", exc_info=True)
            return [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]

    def transform_keypoints(self, hand_coords):
        """
        Input: Hand coordinates from VR detector (N,3)
        Returns:
        - Transformed hand coordinates (N,3) - Hand keypoints translated to "VR frame"
        - Hand direction frame (4,3) - Coordinate frame vectors for "VR frame"
        """
        try:
            if not check_array_validity(hand_coords, "hand_coords_transform"):
                return np.zeros_like(hand_coords), [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]

            translated_coords = copy(hand_coords) - hand_coords[0]
            if not check_array_validity(translated_coords, "translated_coords"):
                return np.zeros_like(hand_coords), [
                    np.array([0.0, 0.0, 0.0]),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]

            original_coord_frame = self._get_stable_coord_frame(translated_coords)
            if not check_array_validity(np.array(original_coord_frame), "original_coord_frame"):
                original_coord_frame = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]

            coord_frame_stack = np.stack(original_coord_frame, axis=1)
            try:
                rotation_matrix = np.linalg.solve(coord_frame_stack, np.eye(3)).T
                if not check_array_validity(rotation_matrix, "rotation_matrix"):
                    rotation_matrix = np.eye(3)
            except (np.linalg.LinAlgError, Exception):
                rotation_matrix = np.eye(3)

            transformed_keypoints = (rotation_matrix @ translated_coords.T).T
            if not check_array_validity(transformed_keypoints, "transformed_keypoints"):
                transformed_keypoints = translated_coords

            coordinate_frame = self._get_stable_hand_dir_frame(hand_coords)
            frame_valid = all(check_array_validity(v, f"coordinate_frame_vec_{i}") for i, v in enumerate(coordinate_frame))
            if not frame_valid:
                coordinate_frame = [
                    hand_coords[0].copy(),
                    np.array([1.0, 0.0, 0.0]),
                    np.array([0.0, 1.0, 0.0]),
                    np.array([0.0, 0.0, 1.0]),
                ]

            return transformed_keypoints, coordinate_frame

        except Exception as e:
            logger.error(f"Error in transform_keypoints: {e}", exc_info=True)
            return np.zeros_like(hand_coords), [
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                np.array([0.0, 0.0, 1.0]),
            ]

    def _log_frame(self, keypoints, coordinate_frame):
        """Log frame data if logging is enabled."""
        if self.keypoint_logger is not None:
            self.keypoint_logger.log_frame(keypoints, coordinate_frame)

    def stream(self):
        """Main streaming loop for processing hand keypoints."""
        frame_count = 0
        error_count = 0
        max_errors = 100

        logger.info(f"Starting keypoint transform stream for {self.hand_side} hand")
        logger.info(f"Moving average limit: {self.moving_average_limit}")
        logger.info(f"Logging enabled: {self.keypoint_logger is not None}")

        while True:
            try:
                self.timer.start_loop()
                frame_count += 1

                data_type, hand_coords = self._get_hand_coords()

                if hand_coords is None or data_type is None:
                    self.timer.end_loop()
                    continue

                # Shift the points to required axes
                (
                    transformed_keypoints,
                    coordinate_frame,
                ) = self.transform_keypoints(hand_coords)

                if not check_array_validity(transformed_keypoints, "transformed_keypoints_after_transform"):
                    error_count += 1
                    self.timer.end_loop()
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                    continue

                self.averaged_keypoints = moving_average(
                    transformed_keypoints,
                    self.coord_moving_average_queue,
                    self.moving_average_limit,
                )
                self.averaged_coordinate_frame = moving_average(
                    coordinate_frame,
                    self.frame_moving_average_queue,
                    self.moving_average_limit,
                )

                origin = self.averaged_coordinate_frame[0]
                if contains_nan(origin):
                    origin = hand_coords[0].copy()

                x_vec = normalize_vector(self.averaged_coordinate_frame[1])
                y_vec = normalize_vector(self.averaged_coordinate_frame[2])
                z_vec = normalize_vector(self.averaged_coordinate_frame[3])

                if contains_nan(x_vec) or contains_nan(y_vec) or contains_nan(z_vec):
                    x_vec = np.array([1.0, 0.0, 0.0])
                    y_vec = np.array([0.0, 1.0, 0.0])
                    z_vec = np.array([0.0, 0.0, 1.0])

                x_vec, y_vec, z_vec = self._orthogonalize_frame(x_vec, y_vec, z_vec)

                if contains_nan(x_vec) or contains_nan(y_vec) or contains_nan(z_vec):
                    x_vec = np.array([1.0, 0.0, 0.0])
                    y_vec = np.array([0.0, 1.0, 0.0])
                    z_vec = np.array([0.0, 0.0, 1.0])

                self.averaged_coordinate_frame = [origin, x_vec, y_vec, z_vec]

                if not check_array_validity(self.averaged_keypoints, "averaged_keypoints"):
                    error_count += 1
                    self.timer.end_loop()
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                    continue

                data = InputFrame(
                    timestamp_s=time.time(),
                    hand_side=self.hand_side,
                    keypoints=self.averaged_keypoints,
                    is_relative=data_type == self.relative_mode,
                    frame_vectors=self.averaged_coordinate_frame,
                )

                if not check_array_validity(data.keypoints, "published_keypoints"):
                    error_count += 1
                    self.timer.end_loop()
                    if error_count >= max_errors:
                        logger.error(f"Too many errors ({error_count}), stopping stream")
                        break
                    continue

                # Publish both transformed coords and frame for consumers
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.keypoint_transform_pub_port,
                    topic=self.coords_topic,
                    data=data,
                )

                # Log and save in a JSON file the hand keypoints and the coordinate frame
                if self.keypoint_logger is not None:
                    self._log_frame(self.averaged_keypoints, self.averaged_coordinate_frame)
                # This is redundant
                # TODO: Remove this and modify LEAPoperator to use coords topic
                self.publisher_manager.publish(
                    host=self.host,
                    port=self.keypoint_transform_pub_port,
                    topic=self.frame_topic,
                    data=data,
                )

                # Log status periodically
                if frame_count % 1000 == 0:
                    logger.info(f"Processed {frame_count} frames successfully. Error count: {error_count}")

                self.timer.end_loop()

            except Exception as e:
                logger.error(f"Error in stream loop at frame {frame_count}: {e}", exc_info=True)
                error_count += 1
                self.timer.end_loop()  # Ensure timer loop is ended even on error

                if error_count >= max_errors:
                    logger.error(f"Too many errors ({error_count}), stopping stream")
                    break

                # Small delay to prevent tight error loops
                time.sleep(0.01)

    # Cleanup
    def cleanup(self):
        """Clean up resources and save any remaining logged data."""
        # Save logged data before cleanup
        if self.keypoint_logger is not None:
            logger.info("Cleanup called. Saving final logged data...")
            self.keypoint_logger.save_data()

        # Save raw keypoint records
        if len(self.raw_keypoint_records) > 0:
            self._save_raw_keypoints()

        self.keypoint_subscriber.stop()
        cleanup_zmq_resources()

    def __del__(self):
        self.cleanup()
