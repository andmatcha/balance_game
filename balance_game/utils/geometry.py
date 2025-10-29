from __future__ import annotations

import math
from typing import Optional

from ..types import Point2D


def distance(a: Point2D, b: Point2D) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return math.hypot(dx, dy)


def midpoint(a: Point2D, b: Point2D) -> Point2D:
    return Point2D((a.x + b.x) / 2.0, (a.y + b.y) / 2.0)


def angle_deg_between(vx: float, vy: float, ux: float, uy: float) -> float:
    dot = vx * ux + vy * uy
    norm_v = math.hypot(vx, vy)
    norm_u = math.hypot(ux, uy)
    if norm_v == 0 or norm_u == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (norm_v * norm_u)))
    return math.degrees(math.acos(cos_theta))


def vertical_tilt_deg(origin: Point2D, target: Point2D) -> float:
    # 正上方向 (0, -1) と origin->target ベクトルの角度差を返す
    vx = target.x - origin.x
    vy = target.y - origin.y
    return angle_deg_between(vx, vy, 0.0, -1.0)


def shoulder_angle_deg(left_shoulder: Point2D, right_shoulder: Point2D) -> float:
    # 左→右の水平方向に対する角度。右肩が高ければ負角度、左肩が高ければ正角度
    dx = right_shoulder.x - left_shoulder.x
    dy = right_shoulder.y - left_shoulder.y
    if dx == 0 and dy == 0:
        return 0.0
    return math.degrees(math.atan2(dy, dx))
