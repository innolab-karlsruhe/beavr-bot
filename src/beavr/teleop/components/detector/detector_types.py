from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

"""
Detector → Operator contracts: minimal fields required by operators.

Notes:
- Units: positions in meters, orientations as XYZW unit quaternions,
  velocities in m/s and rad/s, time is seconds (monotonic).
- "hand_side" distinguishes left/right streams when bimanual.
"""


HandSide = Literal["left", "right"]


@dataclass(frozen=True)
class ButtonEvent:
    """Discrete button input from VR device.

    - name: logical button identifier (e.g., "trigger", "menu").
    - value: pressed/released boolean or small integer payload.
    """

    timestamp_s: float
    hand_side: HandSide
    name: str
    value: int


@dataclass(frozen=True)
class SessionCommand:
    """Session-level commands emitted by the detector.

    Common commands: "pause", "resume", "reset", "home".
    """

    timestamp_s: float
    command: Literal["pause", "resume", "reset", "home"]


@dataclass(frozen=True)
class InputFrame:
    """Raw per-frame inputs necessary for retargeting.

    - hand_side: stream source side.
    - keypoints: flattened (N,3) landmarks as a sequence of triples.
    - is_relative: whether keypoints are relative to wrist (1) or absolute (0),
      matching current detector output semantics.
    - frame_vectors: optional 3 orthonormal vectors (x,y,z) in absolute mode.
    """

    timestamp_s: float
    hand_side: HandSide
    keypoints: Sequence[Tuple[float, float, float]]
    is_relative: bool
    frame_vectors: Optional[
        Tuple[
            Tuple[float, float, float],
            Tuple[float, float, float],
            Tuple[float, float, float],
        ]
    ] = None


@dataclass(frozen=True)
class ControllerFrame:
    timestamp_s: float
    hand_side: HandSide
    position: Tuple[float, float, float]
    orientation_xyzw: Tuple[float, float, float, float]
    trigger_value: float


__all__ = [
    "HandSide",
    "ButtonEvent",
    "SessionCommand",
    "InputFrame",
    "ControllerFrame",
]
