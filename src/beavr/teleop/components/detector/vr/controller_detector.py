import logging
import time
from collections import deque
from copy import deepcopy as copy

import zmq
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from beavr.teleop.common.network.publisher import ZMQPublisherManager
from beavr.teleop.common.network.subscriber import ZMQSubscriber
from beavr.teleop.common.network.utils import (
    SerializationError,
    cleanup_zmq_resources,
    create_pull_socket,
    get_global_context,
)
from beavr.teleop.common.time.timer import FrequencyTimer
from beavr.teleop.components import Component
from beavr.teleop.components.detector.detector_types import SessionCommand
from beavr.teleop.components.operator.operator_types import CartesianTarget, GripperCommand
from beavr.teleop.components.operator.solvers.filters import CompStateFilter
from beavr.teleop.components.interface.interface_types import CartesianState
from beavr.teleop.configs.constants import robots

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class ControllerDetector(Component):
    def __init__(
        self,
        host: str,
        controller_port: int,
        hand_side: str,
        endeff_publish_port: int,
        endeff_subscribe_port: int,
        gripper_publish_port: int,
        h_r_v: np.ndarray,
        h_t_v: np.ndarray,
        final_translation: np.ndarray,
        teleoperation_state_port: Optional[int] = None,
        arm_resolution_port: Optional[int] = None,
        use_filter: bool = True,
    ):
        self.notify_component_start(f"controller_detector_{hand_side}")
        self.host = host
        self.hand_side = hand_side
        self.h_r_v = h_r_v
        self.h_t_v = h_t_v
        self.final_translation = final_translation

        self._context = get_global_context()
        self.controller_socket = create_pull_socket(host, controller_port)

        self._publisher_manager = ZMQPublisherManager.get_instance(self._context)
        self._publisher_host = host
        self._endeff_publish_port = endeff_publish_port
        self._gripper_publish_port = gripper_publish_port

        self._endeff_homo_subscriber = ZMQSubscriber(
            host=host,
            port=endeff_subscribe_port,
            topic="endeff_homo",
            context=self._context,
            message_type=CartesianState,
        )

        self._arm_teleop_state_subscriber = None
        if teleoperation_state_port:
            self._arm_teleop_state_subscriber = ZMQSubscriber(
                host=host,
                port=teleoperation_state_port,
                topic="pause",
                context=self._context,
                message_type=SessionCommand,
            )

        self._arm_resolution_subscriber = None
        if arm_resolution_port:
            self._arm_resolution_subscriber = ZMQSubscriber(
                host=host,
                port=arm_resolution_port,
                topic="button",
                context=self._context,
                message_type=SessionCommand,
            )

        self.arm_teleop_state = robots.ARM_TELEOP_CONT
        self.resolution_scale = 1.0
        self.is_first_frame = True
        self.timer = FrequencyTimer(robots.VR_FREQ)

        self.robot_init_h: Optional[np.ndarray] = None
        self.robot_moving_h: Optional[np.ndarray] = None
        self.controller_init_h: Optional[np.ndarray] = None
        self.controller_moving_h: Optional[np.ndarray] = None
        self.controller_init_t: Optional[np.ndarray] = None
        self.last_valid_controller_h: Optional[np.ndarray] = None

        self.use_filter = use_filter
        self.comp_filter: Optional[CompStateFilter] = None

        self._gripper_width = robots.OPENARM_GRIPPER_MIN_WIDTH_M
        self._first_gripper_publish = True

    @staticmethod
    def _rotate_90_around_x(position: np.ndarray) -> np.ndarray:
        rotated = np.array([position[0], -position[2], position[1]])
        return rotated

    @staticmethod
    def _rotate_quat_90_around_x(quat_xyzw: np.ndarray) -> np.ndarray:
        r = Rotation.from_quat(quat_xyzw)
        rot_x_90 = Rotation.from_rotvec([np.pi / 2, 0, 0])
        r_rotated = rot_x_90 * r
        return r_rotated.as_quat()

    def _process_controller_data(self, data: bytes) -> Optional[dict]:
        try:
            data_str = data.decode().strip()
            if not data_str.startswith("controller:"):
                return None
            payload = data_str.split(":")[1]
            parts = payload.split(",")
            if len(parts) != 8:
                logger.warning(f"Invalid controller data format: expected 8 values, got {len(parts)}")
                return None
            values = [float(v) for v in parts]
            position = np.array(values[0:3])
            orientation_xyzw = np.array(values[3:7])
            trigger_value = values[7]

            position = self._rotate_90_around_x(position)
            orientation_xyzw = self._rotate_quat_90_around_x(orientation_xyzw)

            return {
                "position": position,
                "orientation_xyzw": orientation_xyzw,
                "trigger_value": trigger_value,
            }
        except Exception as e:
            logger.error(f"Error processing controller data: {e}")
            return None

    def _pose_to_homo_mat(self, position: np.ndarray, orientation_xyzw: np.ndarray) -> np.ndarray:
        homo_mat = np.eye(4)
        r_mat = Rotation.from_quat(orientation_xyzw).as_matrix()
        homo_mat[:3, :3] = r_mat
        homo_mat[:3, 3] = position
        return homo_mat

    @staticmethod
    def project_to_rotation_matrix(r_mat: np.ndarray) -> np.ndarray:
        try:
            u, _, vt = np.linalg.svd(r_mat)
            r_fixed = u @ vt
            if np.linalg.det(r_fixed) < 0:
                vt[-1, :] *= -1
                r_fixed = u @ vt
            return r_fixed
        except np.linalg.LinAlgError:
            return np.eye(3)

    def _homo2cart(self, homo_mat: np.ndarray) -> np.ndarray:
        t = homo_mat[:3, 3]
        r_mat = self.project_to_rotation_matrix(homo_mat[:3, :3])
        r_quat = Rotation.from_matrix(r_mat).as_quat()
        return np.concatenate([t, r_quat], axis=0)

    def _get_arm_teleop_state(self) -> int:
        if not self._arm_teleop_state_subscriber:
            return robots.ARM_TELEOP_CONT
        data = self._arm_teleop_state_subscriber.recv_keypoints()
        if data is None:
            return self.arm_teleop_state
        try:
            if data.command == robots.PAUSE:
                return robots.ARM_TELEOP_STOP
            elif data.command == robots.RESUME:
                return robots.ARM_TELEOP_CONT
            else:
                return self.arm_teleop_state
        except Exception:
            return self.arm_teleop_state

    def _reset_teleop(self) -> Optional[np.ndarray]:
        logger.info(f"****** controller_{self.hand_side}: RESETTING TELEOP ******")

        self._publisher_manager.publish(
            host=self._publisher_host,
            port=self._endeff_publish_port,
            topic="reset",
            data=SessionCommand(timestamp_s=time.time(), command="reset"),
        )

        robot_frame_homo = self._endeff_homo_subscriber.recv_keypoints()
        while robot_frame_homo is None:
            self._publisher_manager.publish(
                host=self._publisher_host,
                port=self._endeff_publish_port,
                topic="reset",
                data=SessionCommand(timestamp_s=time.time(), command="reset"),
            )
            robot_frame_homo = self._endeff_homo_subscriber.recv_keypoints()
            time.sleep(0.01)

        try:
            h = np.array(robot_frame_homo.h_matrix, dtype=np.float64).reshape(4, 4)
            self.robot_init_h = h
            if not np.allclose(self.robot_init_h[3, :], [0, 0, 0, 1]):
                self.robot_init_h[3, :] = [0, 0, 0, 1]
            self.robot_init_h[:3, :3] = self.project_to_rotation_matrix(self.robot_init_h[:3, :3])
        except Exception:
            self.is_first_frame = True
            return None

        self.robot_moving_h = copy(self.robot_init_h)

        first_controller_h = self.last_valid_controller_h
        while first_controller_h is None:
            time.sleep(0.01)

        try:
            self.controller_init_h = first_controller_h
            self.controller_init_t = copy(self.controller_init_h[:3, 3])
            r_mat = self.controller_init_h[:3, :3]
            r_fixed = self.project_to_rotation_matrix(r_mat)
            self.controller_init_h[:3, :3] = r_fixed
        except Exception:
            self.is_first_frame = True
            return None

        self.is_first_frame = False
        self.comp_filter = None
        logger.info(f"controller_{self.hand_side}: TELEOP RESET COMPLETE")
        return first_controller_h

    def stream(self):
        logger.info(f"Starting ControllerDetector for {self.hand_side}")
        frame_count = 0
        target_interval = 1.0 / robots.VR_FREQ
        iter_times = deque(maxlen=1000)

        while True:
            start_time_iter = time.perf_counter()

            try:
                data = self.controller_socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
            except Exception:
                time.sleep(0.001)
                continue

            if data is None:
                continue

            controller_data = self._process_controller_data(data)
            if controller_data is None:
                continue

            controller_h = self._pose_to_homo_mat(
                controller_data["position"],
                controller_data["orientation_xyzw"],
            )
            self.last_valid_controller_h = controller_h
            trigger_value = controller_data["trigger_value"]

            new_arm_teleop_state = self._get_arm_teleop_state()
            needs_reset = self.is_first_frame or (
                self.arm_teleop_state == robots.ARM_TELEOP_STOP
                and new_arm_teleop_state == robots.ARM_TELEOP_CONT
            )
            self.arm_teleop_state = new_arm_teleop_state

            publish_commands = self.arm_teleop_state == robots.ARM_TELEOP_CONT

            if needs_reset:
                self._reset_teleop()
                continue

            if self.robot_init_h is None or self.controller_init_h is None:
                self.is_first_frame = True
                continue

            self.controller_moving_h = controller_h

            t_init = self.controller_init_h[:3, 3]
            t_cur = self.controller_moving_h[:3, 3]
            dt_world = t_cur - t_init

            dt_world = np.array([
                dt_world[2],
                dt_world[0],
                -dt_world[1],
            ])

            R_init = self.controller_init_h[:3, :3]
            R_cur = self.controller_moving_h[:3, :3]
            R_rel_world = R_cur @ R_init.T

            h_ht_hi = np.eye(4)
            h_ht_hi[:3, :3] = R_rel_world
            h_ht_hi[:3, 3] = dt_world

            try:
                h_r_v_inv = np.linalg.inv(self.h_r_v)
                h_t_v_inv = np.linalg.inv(self.h_t_v)

                h_ht_hi_r = h_r_v_inv[:3, :3] @ h_ht_hi[:3, :3] @ self.h_r_v[:3, :3]
                h_ht_hi_t = h_t_v_inv[:3, :3] @ h_ht_hi[:3, 3]
            except np.linalg.LinAlgError:
                logger.error("Could not invert H_R_V or H_T_V")
                continue

            h_ht_hi_r = self.project_to_rotation_matrix(h_ht_hi_r)

            relative_affine = np.eye(4)
            relative_affine[:3, :3] = h_ht_hi_r
            relative_affine[:3, 3] = h_ht_hi_t

            h_rt_rh = self.robot_init_h @ relative_affine
            h_rt_rh[:3, :3] = self.project_to_rotation_matrix(h_rt_rh[:3, :3])
            h_rt_rh[:3, 3] = (self.final_translation @ np.r_[h_rt_rh[:3, 3], 1.0])[:3]

            self.robot_moving_h = copy(h_rt_rh)

            cart_target_raw = self._homo2cart(self.robot_moving_h)

            if self.use_filter:
                if self.comp_filter is None:
                    self.comp_filter = CompStateFilter(
                        init_state=cart_target_raw,
                        pos_ratio=0.7,
                        ori_ratio=0.85,
                        adaptive=True,
                    )
                    cart_target_filtered = cart_target_raw
                else:
                    cart_target_filtered = self.comp_filter(cart_target_raw)
            else:
                cart_target_filtered = cart_target_raw

            position = cart_target_filtered[0:3]
            orientation_quat = cart_target_filtered[3:7].copy()

            norm = np.linalg.norm(orientation_quat)
            if norm < 1e-6:
                orientation_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            else:
                orientation_quat = orientation_quat / norm
                if orientation_quat[3] < 0:
                    orientation_quat = -orientation_quat

            if publish_commands:
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

                try:
                    self._publisher_manager.publish(
                        host=self._publisher_host,
                        port=self._endeff_publish_port,
                        topic="endeff_coords",
                        data=cartesian_cmd,
                    )
                except (ConnectionError, SerializationError) as e:
                    logger.error(f"Failed to publish cartesian command: {e}")

                gripper_width_m = (
                    trigger_value / 1.0
                ) * robots.OPENARM_GRIPPER_MAX_WIDTH_M
                gripper_width_m = max(
                    robots.OPENARM_GRIPPER_MIN_WIDTH_M,
                    min(gripper_width_m, robots.OPENARM_GRIPPER_MAX_WIDTH_M),
                )

                gripper_cmd = GripperCommand(
                    timestamp_s=time.time(),
                    hand_side=self.hand_side,
                    width_m=gripper_width_m,
                    speed_mps=robots.OPENARM_GRIPPER_DEFAULT_SPEED_MPS,
                )

                try:
                    self._publisher_manager.publish(
                        host=self._publisher_host,
                        port=self._gripper_publish_port,
                        topic="gripper_cmd",
                        data=gripper_cmd,
                    )
                except (ConnectionError, SerializationError) as e:
                    logger.error(f"Failed to publish gripper command: {e}")

            frame_count += 1

            elapsed_iter = time.perf_counter() - start_time_iter
            iter_times.append(elapsed_iter)

    def cleanup(self):
        logger.info(f"Cleaning up ControllerDetector for {self.hand_side}")
        try:
            self.controller_socket.close()
        except Exception:
            pass
        cleanup_zmq_resources()
