from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .types import Keypoints2D, Point2D, StabilizerConfig
from .utils.geometry import vertical_tilt_deg


@dataclass
class MotionState:
    prev_nose: Optional[Point2D] = None
    prev_speed: float = 0.0


@dataclass
class BalanceMetrics:
    tilt_deg: float = 0.0
    jerk: float = 0.0


class BalancePhysics:
    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.motion = MotionState()

    def _compute_tilt(self, kps: Keypoints2D) -> float:
        # 頭頂部→顎先ベクトルの縦方向からの傾き（肩を使わない）
        if kps.head_top and kps.chin:
            return float(vertical_tilt_deg(kps.head_top, kps.chin))
        # フォールバック: 鼻がある場合は頭頂部→鼻
        if kps.head_top and kps.nose:
            return float(vertical_tilt_deg(kps.head_top, kps.nose))
        return 0.0

    def _compute_speed(self, a: Point2D, b: Point2D, dt_s: float) -> float:
        if dt_s <= 0:
            return 0.0
        dx = a.x - b.x
        dy = a.y - b.y
        dist = (dx * dx + dy * dy) ** 0.5
        return dist / dt_s

    def update(
        self, keypoints: Keypoints2D, dt_s: float
    ) -> tuple[bool, BalanceMetrics]:
        metrics = BalanceMetrics()

        # 傾き
        metrics.tilt_deg = self._compute_tilt(keypoints)

        # ジャーク（速度の変化量）: 鼻のみで近似
        if keypoints.nose and self.motion.prev_nose is not None:
            speed = self._compute_speed(keypoints.nose, self.motion.prev_nose, dt_s)
            metrics.jerk = abs(speed - self.motion.prev_speed) / max(dt_s, 1e-6)
            self.motion.prev_speed = speed
        else:
            metrics.jerk = 1e6  # 検出不可時は大きな値にして不安定扱い
            self.motion.prev_speed = 0.0

        self.motion.prev_nose = keypoints.nose

        is_stable = (
            metrics.tilt_deg <= self.config.max_tilt_deg
            and metrics.jerk <= self.config.max_jerk
        )
        return is_stable, metrics


# ------------------------------
# 長方形物体の回転ダイナミクス
# ------------------------------


@dataclass
class RectMotionState:
    prev_nose: Optional[Point2D] = None
    prev_vx: float = 0.0
    theta_rad: float = 0.0
    omega: float = 0.0


@dataclass
class RectBalanceMetrics:
    object_tilt_deg: float = 0.0
    head_tilt_deg: float = 0.0
    head_vx: float = 0.0  # px/s


class RectanglePhysics:
    """頭上の長方形の簡易回転剛体モデル。

    dω/dt = k_tilt * rad(φ) + k_move * (vx / s_norm) - c_damp * ω
    dθ/dt = ω
    安定: abs(deg(θ)) <= config.max_tilt_deg
    """

    def __init__(self, config: StabilizerConfig):
        self.config = config
        self.state = RectMotionState()
        # チューニング係数（経験的）
        self.k_tilt = 3.0  # 頭傾きからのトルク寄与
        self.k_move = 1.5  # 水平速度からのトルク寄与
        self.s_norm = 200.0  # 速度正規化（px/s）
        self.c_damp = 1.2  # 減衰係数

    def reset(self):
        self.state = RectMotionState()

    def _head_tilt(self, kps: Keypoints2D) -> float:
        if kps.head_top and kps.chin:
            return float(vertical_tilt_deg(kps.head_top, kps.chin))
        if kps.head_top and kps.nose:
            return float(vertical_tilt_deg(kps.head_top, kps.nose))
        return 0.0

    def _nose_vx(self, kps: Keypoints2D, dt_s: float) -> float:
        if not kps.nose or dt_s <= 0:
            return 0.0
        vx = 0.0
        if self.state.prev_nose is not None:
            vx = (kps.nose.x - self.state.prev_nose.x) / dt_s
        self.state.prev_nose = kps.nose
        return vx

    def update(
        self, keypoints: Keypoints2D, dt_s: float
    ) -> tuple[bool, RectBalanceMetrics]:
        metrics = RectBalanceMetrics()

        head_tilt_deg = self._head_tilt(keypoints)
        vx = self._nose_vx(keypoints, dt_s)

        metrics.head_tilt_deg = head_tilt_deg
        metrics.head_vx = vx

        # 角加速度
        alpha = (
            self.k_tilt * (head_tilt_deg * 3.1415926535 / 180.0)
            + self.k_move * (vx / self.s_norm)
            - self.c_damp * self.state.omega
        )
        # 積分
        self.state.omega += alpha * dt_s
        self.state.theta_rad += self.state.omega * dt_s

        metrics.object_tilt_deg = self.state.theta_rad * 180.0 / 3.1415926535
        is_stable = abs(metrics.object_tilt_deg) <= self.config.max_tilt_deg
        return is_stable, metrics
