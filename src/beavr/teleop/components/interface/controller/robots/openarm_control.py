import logging
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from beavr.teleop.configs.constants import robots


class OpenArmController:
    def __init__(
        self,
        trajectory_action_name: str = "/left_joint_trajectory_controller/follow_joint_trajectory",
        ik_service_name: str = "/compute_ik",
        joint_names: Optional[list] = None,
        ik_group_name: str = robots.OPENARM_IK_GROUP_NAME,
        ik_frame_id: str = robots.OPENARM_IK_FRAME_ID,
        ik_link_name: str = robots.OPENARM_IK_LINK_NAME,
        trajectory_duration: float = robots.OPENARM_TRAJECTORY_DURATION_SEC,
    ):
        self.joint_names = joint_names or robots.OPENARM_LEFT_JOINT_NAMES
        self.ik_group_name = ik_group_name
        self.ik_frame_id = ik_frame_id
        self.ik_link_name = ik_link_name
        self.trajectory_duration = trajectory_duration
        self.num_joints = len(self.joint_names)

        self._joint_states: Optional[JointState] = None
        self._joint_states_lock = threading.Lock()
        self._current_joint_positions: Optional[np.ndarray] = None
        self._current_joint_velocities: Optional[np.ndarray] = None
        self._current_joint_efforts: Optional[np.ndarray] = None

        self._initialize_ros2()

        self._action_client = ActionClient(
            self._node,
            FollowJointTrajectory,
            trajectory_action_name,
            callback_group=ReentrantCallbackGroup(),
        )

        self._ik_client = self._node.create_client(
            GetPositionIK,
            ik_service_name,
            callback_group=ReentrantCallbackGroup(),
        )

        self._joint_state_subscriber = self._node.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
            callback_group=ReentrantCallbackGroup(),
        )

        logger.info(f"Waiting for trajectory action server: {trajectory_action_name}")
        if not self._action_client.wait_for_server(timeout_sec=5.0):
            logger.error("Trajectory action server not available!")
        else:
            logger.info("Connected to trajectory action server")

        logger.info(f"Waiting for IK service: {ik_service_name}")
        if not self._ik_client.wait_for_service(timeout_sec=5.0):
            logger.error("IK service not available!")
        else:
            logger.info("Connected to IK service")

        self._wait_for_joint_states()

    def _initialize_ros2(self):
        logger.info("Starting ROS2 initialization...")
        if not rclpy.ok():
            logger.info("rclpy not initialized, calling rclpy.init()")
            try:
                rclpy.init()
                logger.info("rclpy.init() successful")
            except Exception as e:
                logger.error(f"Failed to initialize rclpy: {e}")
                raise

        logger.info("Creating ROS2 node: openarm_controller_node")
        try:
            self._node = Node("openarm_controller_node")
        except Exception as e:
            logger.error(f"Failed to create ROS2 node: {e}")
            raise

        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        logger.info("Starting ROS2 executor thread")
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()

        logger.info("ROS2 node initialized: openarm_controller_node")

    def _joint_state_callback(self, msg: JointState):
        with self._joint_states_lock:
            self._joint_states = msg
            positions = []
            velocities = []
            efforts = []
            for joint_name in self.joint_names:
                try:
                    idx = msg.name.index(joint_name)
                    positions.append(msg.position[idx])
                    if msg.velocity:
                        velocities.append(msg.velocity[idx])
                    if msg.effort:
                        efforts.append(msg.effort[idx])
                except ValueError:
                    pass

            if len(positions) == self.num_joints:
                self._current_joint_positions = np.array(positions, dtype=np.float32)
            if len(velocities) == self.num_joints:
                self._current_joint_velocities = np.array(velocities, dtype=np.float32)
            if len(efforts) == self.num_joints:
                self._current_joint_efforts = np.array(efforts, dtype=np.float32)

    def _wait_for_joint_states(self, timeout: float = 10.0):
        logger.info("Waiting for joint states...")
        start_time = time.time()
        while self._current_joint_positions is None:
            if time.time() - start_time > timeout:
                logger.error("Timeout waiting for joint states")
                return False
            time.sleep(0.1)
        logger.info("Received joint states")
        return True

    def get_arm_position(self) -> Optional[np.ndarray]:
        return self._current_joint_positions

    def get_arm_velocity(self) -> Optional[np.ndarray]:
        return self._current_joint_velocities

    def get_arm_torque(self) -> Optional[np.ndarray]:
        return self._current_joint_efforts

    def get_arm_states(self) -> dict:
        return {
            "joint_position": self._current_joint_positions,
            "joint_velocity": self._current_joint_velocities,
            "joint_torque": self._current_joint_efforts,
            "timestamp": time.time(),
        }

    def compute_ik(self, position: np.ndarray, orientation_quat: np.ndarray) -> Optional[np.ndarray]:
        if not self._ik_client.service_is_ready():
            logger.error("IK service is not ready")
            return None
        logger.info(f"position {position}")
        request = GetPositionIK.Request()
        request.ik_request.group_name = self.ik_group_name
        request.ik_request.pose_stamped.header.frame_id = self.ik_frame_id
        request.ik_request.pose_stamped.header.stamp = self._node.get_clock().now().to_msg()
        request.ik_request.ik_link_name = self.ik_link_name

        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(orientation_quat[0])
        pose.orientation.y = float(orientation_quat[1])
        pose.orientation.z = float(orientation_quat[2])
        pose.orientation.w = float(orientation_quat[3])
        request.ik_request.pose_stamped.pose = pose

        request.ik_request.timeout.sec = 1

        future = self._ik_client.call_async(request)

        timeout_sec = 2.0
        start_time = time.time()
        while not future.done():
            if time.time() - start_time > timeout_sec:
                logger.error("IK service call timed out")
                return None
            time.sleep(0.01)

        try:
            response = future.result()
            if response.error_code.val == 1:
                joint_positions = np.array(
                    response.solution.joint_state.position[: self.num_joints],
                    dtype=np.float32,
                )
                return joint_positions
            else:
                logger.error(f"IK failed with error code: {response.error_code.val}")
                return None
        except Exception as e:
            logger.error(f"Exception during IK call: {e}")
            return None

    def move_arm_joint(self, joint_angles: np.ndarray, duration: Optional[float] = None) -> bool:
        if not self._action_client.server_is_ready():
            logger.error("Trajectory action server is not ready")
            return False

        if len(joint_angles) != self.num_joints:
            logger.error(f"Expected {self.num_joints} joint angles, got {len(joint_angles)}")
            return False

        duration = duration or self.trajectory_duration

        goal_msg = FollowJointTrajectory.Goal()
        goal_msg.trajectory = JointTrajectory()
        goal_msg.trajectory.joint_names = self.joint_names

        point = JointTrajectoryPoint()
        point.positions = joint_angles.tolist()
        point.velocities = [0.0] * self.num_joints
        point.time_from_start = Duration(sec=int(duration), nanosec=int((duration % 1) * 1e9))

        goal_msg.trajectory.points = [point]

        logger.info(f"Sending trajectory goal: {joint_angles}")

        send_goal_future = self._action_client.send_goal_async(goal_msg)

        timeout_sec = duration + 1.0
        start_time = time.time()
        while not send_goal_future.done():
            if time.time() - start_time > timeout_sec:
                logger.error("Trajectory goal send timed out")
                return False
            time.sleep(0.01)

        try:
            goal_handle = send_goal_future.result()
            if not goal_handle.accepted:
                logger.error("Trajectory goal rejected")
                return False

            result_future = goal_handle.get_result_async()
            while not result_future.done():
                if time.time() - start_time > timeout_sec * 2:
                    logger.error("Trajectory execution timed out")
                    return False
                time.sleep(0.01)

            result = result_future.result().result
            if result.error_code == 0:
                logger.debug("Trajectory execution succeeded")
                return True
            else:
                logger.error(f"Trajectory execution failed with error code: {result.error_code}")
                return False
        except Exception as e:
            logger.error(f"Exception during trajectory execution: {e}")
            return False

    def move_arm_cartesian(self, cartesian_pose: np.ndarray, duration: Optional[float] = None) -> bool:
        if len(cartesian_pose) != 7:
            logger.error(f"Expected 7D cartesian pose (x,y,z,qx,qy,qz,qw), got {len(cartesian_pose)}")
            return False

        position = cartesian_pose[:3]
        orientation = cartesian_pose[3:7]

        joint_angles = self.compute_ik(position, orientation)
        logger.info(f"joint_angles {joint_angles}")
        if joint_angles is None:
            logger.error("Failed to compute IK solution")
            return False

        return self.move_arm_joint(joint_angles, duration)

    def home_arm(self) -> bool:
        logger.info("Homing arm to zero position")
        return self.move_arm_joint(np.array(robots.OPENARM_HOME_JS))

    def reset_arm(self) -> bool:
        return self.home_arm()

    def get_arm_pose(self) -> Optional[np.ndarray]:
        if self._current_joint_positions is None:
            return None
        # logger.warning("get_arm_pose() returning joint positions. Forward kinematics not implemented.")
        return self._current_joint_positions

    def cleanup(self):
        logger.info("Cleaning up OpenArm controller...")
        if hasattr(self, "_joint_state_subscriber"):
            self._node.destroy_subscription(self._joint_state_subscriber)
        if hasattr(self, "_action_client"):
            self._action_client.destroy()
        if hasattr(self, "_ik_client"):
            self._ik_client.destroy()
        if hasattr(self, "_executor"):
            self._executor.shutdown()
        if hasattr(self, "_node"):
            self._node.destroy_node()


class DexArmControl:
    def __init__(self, **kwargs):
        self._controller = OpenArmController(**kwargs)

    def get_arm_position(self):
        return self._controller.get_arm_position()

    def get_arm_velocity(self):
        return self._controller.get_arm_velocity()

    def get_arm_torque(self):
        return self._controller.get_arm_torque()

    def get_arm_states(self):
        return self._controller.get_arm_states()

    def get_arm_cartesian_coords(self):
        return self._controller.get_arm_pose()

    def get_cartesian_state(self):
        pose = self._controller.get_arm_pose()
        if pose is None:
            return None
        return {
            "cartesian_position": pose,
            "timestamp": time.time(),
        }

    def get_arm_pose(self):
        return self._controller.get_arm_pose()

    def move_arm_joint(self, joint_angles):
        return self._controller.move_arm_joint(joint_angles)

    def move_arm_cartesian(self, cartesian_pos, duration=None):
        return self._controller.move_arm_cartesian(cartesian_pos, duration)

    def arm_control(self, cartesian_pos):
        return self._controller.move_arm_cartesian(cartesian_pos)

    def home_arm(self):
        return self._controller.home_arm()

    def reset_arm(self):
        return self._controller.reset_arm()

    def cleanup(self):
        return self._controller.cleanup()
