from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


@dataclass
class Point2D:
    x: float
    y: float


@dataclass
class Keypoints2D:
    nose: Optional[Point2D] = None
    left_wrist: Optional[Point2D] = None
    right_wrist: Optional[Point2D] = None
    left_shoulder: Optional[Point2D] = None
    right_shoulder: Optional[Point2D] = None


@dataclass
class DetectionResult:
    keypoints: Keypoints2D
    metadata: Dict[str, object]


@dataclass
class StabilizerConfig:
    max_tilt_deg: float
    max_jerk: float
    clear_seconds: float


class GameStatus(str, Enum):
    READY = "READY"
    COUNTDOWN = "COUNTDOWN"
    PLAYING = "PLAYING"
    FAIL = "FAIL"
    CLEAR = "CLEAR"


@dataclass
class GameConfig:
    target_fps: int
    difficulty: str
    stabilizer: StabilizerConfig
