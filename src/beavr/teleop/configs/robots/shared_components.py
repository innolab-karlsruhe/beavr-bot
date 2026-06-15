"""Reusable component configuration dataclasses shared across robot configs.

Each class exposes sensible *defaults* that come from ``configs.constants`` so
individual robot config files no longer need to duplicate IP addresses or port
numbers.  Override any field as usual when you instantiate the dataclass.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from beavr.teleop.components.detector.vr.keypoint_transform import (
    TransformHandPositionCoords,
)
from beavr.teleop.components.detector.vr.oculus_controller import (
    OculusVRControllerDetector,
)
from beavr.teleop.components.visualizer.visualizer_2d import Hand2DVisualizer
from beavr.teleop.configs.constants import network, ports, robots

logger = logging.getLogger(__name__)


class SharedComponentRegistry:
    """
    Registry for shared VR components that should be singleton per hand side.

    This eliminates the need for complex deduplication logic by ensuring
    only one detector/transform/visualizer exists per hand side across all robots.
    """

    _instances: Dict[str, Dict[str, Any]] = {
        "detector": {},  # hand_side -> config instance
        "transform": {},  # hand_side -> config instance
        "visualizer": {},  # hand_side -> config instance
    }

    @classmethod
    def get_transform_config(
        cls,
        hand_side: str,
        host: str = network.HOST_ADDRESS,
        keypoint_sub_port: int = ports.KEYPOINT_STREAM_PORT,
        moving_average_limit: int = 1,
    ) -> "TransformHandPositionCoordsCfg":
        """Get or create transform config for specified hand side."""

        if hand_side not in cls._instances["transform"]:
            # Set hand-side specific ports
            if hand_side == robots.LEFT:
                keypoint_transform_pub_port = ports.LEFT_KEYPOINT_TRANSFORM_PORT
            else:  # RIGHT or default
                keypoint_transform_pub_port = ports.KEYPOINT_TRANSFORM_PORT

            cls._instances["transform"][hand_side] = TransformHandPositionCoordsCfg(
                host=host,
                keypoint_sub_port=keypoint_sub_port,
                keypoint_transform_pub_port=keypoint_transform_pub_port,
                moving_average_limit=moving_average_limit,
                hand_side=hand_side,
            )
            logger.debug(f"🔄 Created shared transform config for {hand_side} hand")

        return cls._instances["transform"][hand_side]

    @classmethod
    def get_visualizer_config(
        cls,
        hand_side: str,
        host: str = network.HOST_ADDRESS,
        oculus_feedback_port: int = ports.OCULUS_GRAPH_PORT,
        display_plot: bool = False,
    ) -> "Hand2DVisualizerCfg":
        """Get or create visualizer config for specified hand side."""

        if hand_side not in cls._instances["visualizer"]:
            # Set hand-side specific ports
            if hand_side == robots.LEFT:
                transformed_keypoint_port = ports.LEFT_KEYPOINT_TRANSFORM_PORT
            else:  # RIGHT or default
                transformed_keypoint_port = ports.KEYPOINT_TRANSFORM_PORT

            cls._instances["visualizer"][hand_side] = Hand2DVisualizerCfg(
                host=host,
                transformed_keypoint_port=transformed_keypoint_port,
                oculus_feedback_port=oculus_feedback_port,
                display_plot=display_plot,
                hand_side=hand_side,
            )
            logger.debug(f"👁️  Created shared visualizer config for {hand_side} hand")

        return cls._instances["visualizer"][hand_side]

    @classmethod
    def clear(cls):
        """Clear all cached instances. Useful for testing."""
        for component_type in cls._instances:
            cls._instances[component_type].clear()
        logger.debug("🧹 Cleared all shared component instances")

    @classmethod
    def get_registered_hands(cls) -> Dict[str, list]:
        """Get list of registered hand sides by component type."""
        return {
            component_type: list(instances.keys()) for component_type, instances in cls._instances.items()
        }


@dataclass
class TransformHandPositionCoordsCfg:
    """Right-hand keypoint transform (VR frame → robot frame)."""

    host: str = network.HOST_ADDRESS
    keypoint_sub_port: int = ports.KEYPOINT_STREAM_PORT
    keypoint_transform_pub_port: int = ports.KEYPOINT_TRANSFORM_PORT
    moving_average_limit: int = 1
    hand_side: str = robots.RIGHT
    enable_logging: bool = True
    log_dir: str = "data/keypoint_logs"
    auto_save_interval: int = 100

    def __post_init__(self):
        """Validate configuration."""
        if not (1 <= self.keypoint_sub_port <= 65535):
            raise ValueError(f"keypoint_sub_port out of range: {self.keypoint_sub_port}")
        if not (1 <= self.keypoint_transform_pub_port <= 65535):
            raise ValueError(f"keypoint_transform_pub_port out of range: {self.keypoint_transform_pub_port}")
        if self.moving_average_limit < 1:
            raise ValueError(f"moving_average_limit must be >= 1: {self.moving_average_limit}")

    def build(self):
        return TransformHandPositionCoords(
            host=self.host,
            keypoint_sub_port=self.keypoint_sub_port,
            keypoint_transform_pub_port=self.keypoint_transform_pub_port,
            moving_average_limit=self.moving_average_limit,
            hand_side=self.hand_side,
            enable_logging=self.enable_logging,
            log_dir=self.log_dir,
            auto_save_interval=self.auto_save_interval,
        )


@dataclass
class Hand2DVisualizerCfg:
    """2-D hand visualizer (Matplotlib / OpenCV)."""

    host: str = network.HOST_ADDRESS
    transformed_keypoint_port: int = ports.KEYPOINT_TRANSFORM_PORT
    oculus_feedback_port: int = ports.OCULUS_GRAPH_PORT
    display_plot: bool = False
    hand_side: str = robots.RIGHT  # Will be overridden based on laterality

    def __post_init__(self):
        """Validate port configuration."""
        for port_name, port_value in [
            ("transformed_keypoint_port", self.transformed_keypoint_port),
            ("oculus_feedback_port", self.oculus_feedback_port),
        ]:
            if not (1 <= port_value <= 65535):
                raise ValueError(f"{port_name} out of valid range: {port_value}")

    def build(self):
        return Hand2DVisualizer(
            host=self.host,
            transformed_keypoint_port=self.transformed_keypoint_port,
            oculus_feedback_port=self.oculus_feedback_port,
            display_plot=self.display_plot,
        )


@dataclass
class OculusVRControllerDetectorCfg:
    """Configuration for the controller-tracking detector path.

    `controller_pub_port` is dedicated to the controller path so the detector
    can run in its own process without ZMQ PUB-bind collisions against
    TransformHandPositionCoords. The operator subscribes to both this port
    and the keypoint-transform port and uses whichever produces a fresher
    frame.
    """

    host: str = network.HOST_ADDRESS
    controller_pub_port: int = ports.RIGHT_CONTROLLER_TRANSFORM_PORT
    hand_config: Literal['right', 'left', 'bimanual'] = robots.BIMANUAL
    right_controller_port: Optional[int] = ports.RIGHT_CONTROLLER_PORT
    left_controller_port: Optional[int] = ports.LEFT_CONTROLLER_PORT

    def __post_init__(self):
        for name, value in [
            ("controller_pub_port", self.controller_pub_port),
            ("right_controller_port", self.right_controller_port),
            ("left_controller_port", self.left_controller_port),
        ]:
            if value is not None and not (1 <= value <= 65535):
                raise ValueError(f"{name} out of valid range (1-65535): {value}")

        # Enforce hand_config / per-side port consistency.
        if self.hand_config == robots.RIGHT and self.left_controller_port is not None:
            raise ValueError(
                f"hand_config=RIGHT requires left_controller_port=None "
                f"(got {self.left_controller_port})"
            )
        if self.hand_config == robots.LEFT and self.right_controller_port is not None:
            raise ValueError(
                f"hand_config=LEFT requires right_controller_port=None "
                f"(got {self.right_controller_port})"
            )

        # Reject duplicate ports across all configured controller-related sockets.
        all_ports = [self.controller_pub_port]
        if self.right_controller_port is not None:
            all_ports.append(self.right_controller_port)
        if self.left_controller_port is not None:
            all_ports.append(self.left_controller_port)
        if len(set(all_ports)) != len(all_ports):
            raise ValueError(
                f"Duplicate ports in OculusVRControllerDetectorCfg: {all_ports}"
            )

    def build(self):
        return OculusVRControllerDetector(
            host=self.host,
            controller_pub_port=self.controller_pub_port,
            hand_config=self.hand_config,
            right_controller_port=self.right_controller_port,
            left_controller_port=self.left_controller_port,
        )
