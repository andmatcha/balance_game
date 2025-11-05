import math
from types import SimpleNamespace
from typing import Optional

import cv2
import numpy as np
import pymunk

from .types import Keypoints2D, StabilizerConfig


class FingerTip:
    """指先に追従する当たり判定用の円（ボール）。

    - 中心は引数で渡される指先座標(x,y)
    - 重力などの物理影響は受けず、毎回座標を直接更新
    - 必要に応じて space に追加すれば他形状との当たり判定に利用可能
    """

    def __init__(
        self, space: Optional[pymunk.Space] = None, radius: float = 18.0
    ) -> None:
        self.radius = float(radius)
        # 物理影響を受けず手動で動かすため KINEMATIC を使用
        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.shape = pymunk.Circle(self.body, self.radius)
        # 必要に応じて基本的な摩擦/反発を設定（将来の当たり判定用）
        self.shape.friction = 0.9
        self.shape.elasticity = 0.2
        self._space = space
        if self._space is not None:
            self._space.add(self.body, self.shape)

    def update(self, x: float, y: float) -> tuple[float, float]:
        """中心座標を指先(x,y)に更新して返す。"""
        self.body.position = (float(x), float(y))
        # 明示的に速度をゼロにして“追従のみ”にする
        self.body.velocity = (0.0, 0.0)
        pos = self.body.position
        return float(pos.x), float(pos.y)

    def draw(
        self,
        frame_bgr,
        fill_color: tuple[int, int, int] = (0, 120, 255),
        border_color: tuple[int, int, int] = (0, 0, 0),
        border_thickness: int = 2,
    ) -> None:
        """デバッグ用に現在の円を画面上へ重ね描画する。"""
        cx, cy = int(self.body.position.x), int(self.body.position.y)
        r = int(self.radius)
        cv2.circle(frame_bgr, (cx, cy), r, fill_color, -1)
        if border_thickness > 0:
            cv2.circle(
                frame_bgr, (cx, cy), r, border_color, border_thickness, cv2.LINE_AA
            )


class FingerBalancePhysics:
    """指先の上で横長長方形をバランスさせる最小実装。

    - 指先円は KINEMATIC として毎フレーム座標を更新
    - 横長長方形（Dynamic）はゲーム開始時に指先の直上へ配置
    - 長方形の下端が指先より下に抜けたらゲームオーバー
    """

    def __init__(self, stabilizer: StabilizerConfig):
        self._stabilizer = stabilizer

        # 物理空間（画面座標系に合わせて +Y を下向きとする想定）
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 900.0)

        # 指先当たり判定（小さめ）
        self.finger = FingerTip(space=self.space, radius=12.0)

        # 横長長方形の寸法
        self.rect_half_w = 60.0  # 幅 120px 相当
        self.rect_half_h = 10.0  # 高さ 20px 相当

        self.rect_body: Optional[pymunk.Body] = None
        self.rect_shape: Optional[pymunk.Poly] = None

        # ゲーム開始フラグ（初回に指先が取得できたらスポーン）
        self._spawned = False

    # ---- lifecycle ----
    def reset(self) -> None:
        if self.rect_body is not None and self.rect_shape is not None:
            try:
                self.space.remove(self.rect_body, self.rect_shape)
            except Exception:
                pass
        self.rect_body = None
        self.rect_shape = None
        self._spawned = False

    # ---- core ----
    def _spawn_rect(self, fx: float, fy: float) -> None:
        # 指先直上に重心が来るよう配置
        mass = 2.0
        size = (self.rect_half_w * 2.0, self.rect_half_h * 2.0)
        moment = pymunk.moment_for_box(mass, size)
        body = pymunk.Body(mass, moment)
        body.angle = 0.0
        body.position = (
            float(fx),
            float(fy) - self.finger.radius - self.rect_half_h - 1.0,
        )

        shape = pymunk.Poly.create_box(body, size)
        shape.friction = 0.9
        shape.elasticity = 0.0

        self.space.add(body, shape)
        self.rect_body = body
        self.rect_shape = shape
        self._spawned = True

    def update(self, keypoints: Keypoints2D, dt_s: float) -> tuple[bool, object]:
        # 指先座標：右優先、なければ左
        finger = keypoints.right_index or keypoints.left_index
        if finger is not None:
            self.finger.update(float(finger.x), float(finger.y))

        # スポーンしていなければ、初回に指先が取れた時点で生成
        if not self._spawned and finger is not None:
            self._spawn_rect(float(finger.x), float(finger.y))

        # 物理ステップ（極端な dt はクランプ）
        dt = max(
            1.0 / 240.0,
            min(1.0 / 30.0, float(dt_s) if dt_s and dt_s > 0 else 1.0 / 60.0),
        )
        self.space.step(dt)

        # 状態判定
        is_stable = True
        if self.rect_body is None:
            is_stable = False
        else:
            # 長方形の下端が指先より下に抜けたら「落下」
            rect_cy = float(self.rect_body.position.y)
            bottom_y = rect_cy + self.rect_half_h
            finger_y = float(self.finger.body.position.y)
            fell = bottom_y > (finger_y + self.finger.radius + 6.0)
            is_stable = not fell

        # HUD 用の最小メトリクス
        tilt_deg = 0.0
        if self.rect_body is not None:
            tilt_deg = math.degrees(float(self.rect_body.angle))
        metrics = SimpleNamespace(object_tilt_deg=tilt_deg)
        return is_stable, metrics

    # ---- drawing ----
    def draw(self, frame_bgr) -> None:
        # 指先円の描画（デバッグ）
        cx, cy = int(self.finger.body.position.x), int(self.finger.body.position.y)
        cv2.circle(frame_bgr, (cx, cy), int(self.finger.radius), (0, 120, 255), -1)
        cv2.circle(
            frame_bgr, (cx, cy), int(self.finger.radius), (0, 0, 0), 2, cv2.LINE_AA
        )

        if self.rect_body is None:
            return
        # 長方形の4頂点を算出して描画
        x = float(self.rect_body.position.x)
        y = float(self.rect_body.position.y)
        a = float(self.rect_body.angle)
        ca, sa = math.cos(a), math.sin(a)
        hw, hh = self.rect_half_w, self.rect_half_h
        corners = [
            (+hw, +hh),
            (-hw, +hh),
            (-hw, -hh),
            (+hw, -hh),
        ]
        pts = []
        for px, py in corners:
            rx = x + (px * ca - py * sa)
            ry = y + (px * sa + py * ca)
            pts.append((int(rx), int(ry)))
        cv2.fillPoly(frame_bgr, [np.array(pts, dtype=np.int32)], (0, 200, 255))
        cv2.polylines(
            frame_bgr, [np.array(pts, dtype=np.int32)], True, (0, 0, 0), 2, cv2.LINE_AA
        )
