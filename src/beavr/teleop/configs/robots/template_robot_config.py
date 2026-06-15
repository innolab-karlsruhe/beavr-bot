"""Auto-generated strongly-typed config for robot `template_robot`."""

from __future__ import annotations

from dataclasses import dataclass, field

from beavr.teleop.components.detector.vr.keypoint_transform import (
    TransformHandPositionCoords,
)
from beavr.teleop.components.interface.interface_base import RobotWrapper
from beavr.teleop.components.operator.robots.template import TemplateArmOperator
from beavr.teleop.components.visualizer.visualizer_2d import Hand2DVisualizer
from beavr.teleop.configs.robots import TeleopRobotConfig


@dataclass
class TransformHandPositionCoordsCfg:
    host: str = "10.31.152.148"
    keypoint_sub_port: str = "${keypoint_port}"
    keypoint_transform_pub_port: str = "${transformed_position_keypoint_port}"
    moving_average_limit: int = 1
    enable_logging: bool = True
    log_dir: str = "data/keypoint_logs"
    auto_save_interval: int = 100

    def build(self):
        return TransformHandPositionCoords(
            host=self.host,
            keypoint_sub_port=self.keypoint_sub_port,
            keypoint_transform_pub_port=self.keypoint_transform_pub_port,
            moving_average_limit=self.moving_average_limit,
            enable_logging=self.enable_logging,
            log_dir=self.log_dir,
            auto_save_interval=self.auto_save_interval,
        )


@dataclass
class Hand2DVisualizerCfg:
    host: str = "10.31.152.148"
    transformed_keypoint_port: str = "${transformed_position_keypoint_port}"
    oculus_feedback_port: str = "15001"
    display_plot: str = "${visualize_right_2d}"

    def build(self):
        return Hand2DVisualizer(
            host=self.host,
            transformed_keypoint_port=self.transformed_keypoint_port,
            oculus_feedback_port=self.oculus_feedback_port,
            display_plot=self.display_plot,
        )


@dataclass
class TemplateArmOperatorCfg:
    host: str = "10.31.152.148"
    transformed_keypoints_port: str = "${transformed_position_keypoint_port}"
    arm_resolution_port: str = "8094"  # ✅ FIX: Updated to match new button publish port
    gripper_port: str = "8108"
    use_filter: bool = True
    cartesian_publisher_port: str = "8118"
    joint_publisher_port: str = "8119"
    cartesian_command_publisher_port: str = "8120"

    def build(self):
        return TemplateArmOperator(
            host=self.host,
            transformed_keypoints_port=self.transformed_keypoints_port,
            arm_resolution_port=self.arm_resolution_port,
            gripper_port=self.gripper_port,
            use_filter=self.use_filter,
            cartesian_publisher_port=self.cartesian_publisher_port,
            joint_publisher_port=self.joint_publisher_port,
            cartesian_command_publisher_port=self.cartesian_command_publisher_port,
        )


@dataclass
class RobotWrapperCfg:
    record: bool = False

    def build(self):
        return RobotWrapper(record=self.record)
