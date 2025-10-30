from __future__ import annotations

import math

from ..types import Point2D


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
