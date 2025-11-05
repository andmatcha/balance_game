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
        self._last_pos: Optional[tuple[float, float]] = None
        if self._space is not None:
            self._space.add(self.body, self.shape)

    def update(
        self, x: float, y: float, dt: Optional[float] = None
    ) -> tuple[float, float]:
        """中心座標を指先(x,y)に更新して返す。dt があれば速度も設定。"""
        nx, ny = float(x), float(y)
        if dt is not None and dt > 0 and self._last_pos is not None:
            lx, ly = self._last_pos
            vx = (nx - lx) / dt
            vy = (ny - ly) / dt
            self.body.velocity = (float(vx), float(vy))
        else:
            self.body.velocity = (0.0, 0.0)
        self.body.position = (nx, ny)
        self._last_pos = (nx, ny)
        return nx, ny

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
        self.space.iterations = 30
        self._screen_h: Optional[int] = None

        # 指先当たり判定（左右に1つずつ）
        self.left_finger = FingerTip(space=self.space, radius=12.0)
        self.right_finger = FingerTip(space=self.space, radius=12.0)

        # 横長長方形の寸法（長辺×3）
        self.rect_half_w = 180.0  # 幅 360px 相当
        self.rect_half_h = 10.0  # 高さ 20px 相当

        # 左右の長方形
        self.left_rect_body: Optional[pymunk.Body] = None
        self.left_rect_shape: Optional[pymunk.Poly] = None
        self.right_rect_body: Optional[pymunk.Body] = None
        self.right_rect_shape: Optional[pymunk.Poly] = None

        # ゲーム開始フラグ（両手が揃ったら同時にスポーン）
        self._spawned = False

    # ---- lifecycle ----
    def reset(self) -> None:
        if self.left_rect_body is not None and self.left_rect_shape is not None:
            try:
                self.space.remove(self.left_rect_body, self.left_rect_shape)
            except Exception:
                pass
        if self.right_rect_body is not None and self.right_rect_shape is not None:
            try:
                self.space.remove(self.right_rect_body, self.right_rect_shape)
            except Exception:
                pass
        self.left_rect_body = None
        self.left_rect_shape = None
        self.right_rect_body = None
        self.right_rect_shape = None
        self._spawned = False

    # ---- core ----
    def _spawn_rect(self, side: str, fx: float, fy: float) -> None:
        # 指先直上に重心が来るよう配置（side: "left"|"right"）
        mass = 2.0
        size = (self.rect_half_w * 2.0, self.rect_half_h * 2.0)
        moment = pymunk.moment_for_box(mass, size)
        body = pymunk.Body(mass, moment)
        body.angle = 0.0
        # ゲーム開始時の指xに合わせ、画面上端（の少し外）から落下開始
        body.position = (
            float(fx),
            -self.rect_half_h - 1.0,
        )

        shape = pymunk.Poly.create_box(body, size)
        shape.friction = 0.9
        shape.elasticity = 0.0

        self.space.add(body, shape)
        if side == "left":
            self.left_rect_body = body
            self.left_rect_shape = shape
        else:
            self.right_rect_body = body
            self.right_rect_shape = shape

    def update(
        self, keypoints: Keypoints2D, dt_s: float, frame_h: Optional[int] = None
    ) -> tuple[bool, object]:
        if frame_h is not None:
            self._screen_h = int(frame_h)
        # 時間刻み（クランプ）
        dt = max(
            1.0 / 240.0,
            min(1.0 / 30.0, float(dt_s) if dt_s and dt_s > 0 else 1.0 / 60.0),
        )

        # 指先座標を更新（左右）
        lkp = keypoints.left_index
        rkp = keypoints.right_index
        if lkp is not None:
            self.left_finger.update(float(lkp.x), float(lkp.y), dt)
        if rkp is not None:
            self.right_finger.update(float(rkp.x), float(rkp.y), dt)

        # スポーンしていなければ、左右が揃った時点で左右それぞれ生成
        if not self._spawned and (lkp is not None and rkp is not None):
            self._spawn_rect("left", float(lkp.x), float(lkp.y))
            self._spawn_rect("right", float(rkp.x), float(rkp.y))
            self._spawned = True

        # 物理ステップをサブステップ化してトンネリングを軽減
        max_substep = 1.0 / 240.0
        sub_steps = max(1, int(math.ceil(dt / max_substep)))
        sub_dt = dt / sub_steps
        for _ in range(sub_steps):
            self.space.step(sub_dt)

        # 状態判定（どちらか一方でも画面下端に到達でNG）
        is_stable = True
        # 画面高さが未設定なら判定しない（安定扱い）
        if self._screen_h is None:
            is_stable = True
        else:
            # スポーン前は安定扱い（PLAYING への遷移はアプリ側で両手を保証）
            if self._spawned:
                # 左
                if self.left_rect_body is not None:
                    left_bottom = (
                        float(self.left_rect_body.position.y) + self.rect_half_h
                    )
                    if left_bottom >= float(self._screen_h):
                        is_stable = False
                # 右（左で既にNGでも両方見るが結論は同じ）
                if self.right_rect_body is not None:
                    right_bottom = (
                        float(self.right_rect_body.position.y) + self.rect_half_h
                    )
                    if right_bottom >= float(self._screen_h):
                        is_stable = False

        # HUD 用の最小メトリクス
        # HUD 向け：2つの傾きの平均（存在するものだけ）
        tilts: list[float] = []
        if self.left_rect_body is not None:
            tilts.append(math.degrees(float(self.left_rect_body.angle)))
        if self.right_rect_body is not None:
            tilts.append(math.degrees(float(self.right_rect_body.angle)))
        tilt_deg = sum(tilts) / len(tilts) if tilts else 0.0
        metrics = SimpleNamespace(object_tilt_deg=tilt_deg)
        return is_stable, metrics

    # ---- drawing ----
    def draw(self, frame_bgr) -> None:
        # 指先円の描画（左右）
        lcx, lcy = int(self.left_finger.body.position.x), int(
            self.left_finger.body.position.y
        )
        rcx, rcy = int(self.right_finger.body.position.x), int(
            self.right_finger.body.position.y
        )
        cv2.circle(
            frame_bgr, (lcx, lcy), int(self.left_finger.radius), (0, 120, 255), -1
        )
        cv2.circle(
            frame_bgr,
            (lcx, lcy),
            int(self.left_finger.radius),
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            frame_bgr, (rcx, rcy), int(self.right_finger.radius), (0, 120, 255), -1
        )
        cv2.circle(
            frame_bgr,
            (rcx, rcy),
            int(self.right_finger.radius),
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        def _draw_rect(body: pymunk.Body) -> None:
            x = float(body.position.x)
            y = float(body.position.y)
            a = float(body.angle)
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
                frame_bgr,
                [np.array(pts, dtype=np.int32)],
                True,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

        if self.left_rect_body is not None:
            _draw_rect(self.left_rect_body)
        if self.right_rect_body is not None:
            _draw_rect(self.right_rect_body)
