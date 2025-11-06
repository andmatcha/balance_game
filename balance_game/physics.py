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

    def __init__(self, stabilizer: StabilizerConfig, rect_image_path: str = None):
        self._stabilizer = stabilizer

        # 物理空間（画面座標系に合わせて +Y を下向きとする想定）
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 1200.0)
        self.space.iterations = 30
        self._screen_h: Optional[int] = None
        self._screen_w: Optional[int] = None

        # 指先当たり判定（左右に1つずつ）
        self.left_finger = FingerTip(space=self.space, radius=12.0)
        self.right_finger = FingerTip(space=self.space, radius=12.0)

        # 横長長方形の基準寸法（実際の値は画面幅に応じて更新）
        # 横幅は「画面横幅の1/3」に設定される（rect_half_w はその半分）
        self.rect_half_w = 180.0
        # 既定のアスペクト比（W:H = 18:1）に基づく初期値
        self.rect_half_h = 10.0

        # 左右の長方形
        self.left_rect_body: Optional[pymunk.Body] = None
        self.left_rect_shape: Optional[pymunk.Poly] = None
        self.right_rect_body: Optional[pymunk.Body] = None
        self.right_rect_shape: Optional[pymunk.Poly] = None

        # ゲーム開始フラグ（両手が揃ったら同時にスポーン）
        self._spawned = False
        self._left_seen = False
        self._right_seen = False
        self._unstable_time_s: float = 0.0

        # 長方形用の画像を読み込み
        if rect_image_path:
            self._rect_img = cv2.imread(rect_image_path)
            if self._rect_img is None:
                print(f"警告: 画像 {rect_image_path} を読み込めませんでした")
                
        else:
            self._rect_img = None
 
        # 画像読み込み部分
        print(f"[DEBUG] rect_image_path: {rect_image_path}")  # 追加
        if rect_image_path:
            self._rect_img = cv2.imread(rect_image_path)
            if self._rect_img is None:
                print(f"警告: 画像 {rect_image_path} を読み込めませんでした")
                print(f"[DEBUG] 絶対パス: {os.path.abspath(rect_image_path)}")  # 追加
            else:
                print(f"[DEBUG] 画像読み込み成功！サイズ: {self._rect_img.shape}")  # 追加
        else:
            self._rect_img = None
            print("[DEBUG] 画像パスが指定されていません")  # 追加

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
        self._left_seen = False
        self._right_seen = False
        self._unstable_time_s = 0.0

    # ---- core ----
    def _update_rect_size_by_screen(self) -> None:
        """画面横幅に応じて長方形の寸法を更新する（横幅=画面幅の1/3）。"""
        if self._screen_w is None:
            return
        full_w = float(self._screen_w) / 4.0
        # 半幅（rect_half_w）は全幅の半分
        self.rect_half_w = full_w / 2.0
        # 既存のアスペクト比（W:H = 18:1）を維持して高さを算出
        full_h = full_w / 18.0
        self.rect_half_h = full_h / 2.0

    def _spawn_rect(self, side: str, fx: float, fy: float) -> None:
        # 指先直上に重心が来るよう配置（side: "left"|"right"）
        mass = 6.0
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

    # ---- helpers (判定ロジックのカプセル化) ----
    def _aabb(self, body: pymunk.Body) -> tuple[float, float, float, float]:
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
        xs: list[float] = []
        ys: list[float] = []
        for px, py in corners:
            rx = x + (px * ca - py * sa)
            ry = y + (px * sa + py * ca)
            xs.append(rx)
            ys.append(ry)
        return min(xs), max(xs), min(ys), max(ys)

    def _is_visible(self, body: Optional[pymunk.Body]) -> bool:
        if body is None or self._screen_w is None or self._screen_h is None:
            return True
        min_x, max_x, min_y, max_y = self._aabb(body)
        return (
            (max_x >= 0.0)
            and (min_x < float(self._screen_w))
            and (max_y >= 0.0)
            and (min_y < float(self._screen_h))
        )

    def _is_off_finger_horizontally(
        self, body: Optional[pymunk.Body], finger_x: float, finger_radius: float
    ) -> bool:
        if body is None:
            return False
        min_x, max_x, _, _ = self._aabb(body)
        left_band = finger_x - finger_radius
        right_band = finger_x + finger_radius
        return (max_x < left_band) or (min_x > right_band)

    def _is_entirely_below_finger(
        self, body: Optional[pymunk.Body], finger_y: float, finger_radius: float
    ) -> bool:
        if body is None:
            return False
        _, _, min_y, _ = self._aabb(body)
        return min_y > (finger_y + finger_radius + 4.0)

    def _evaluate_unstable_conditions(self) -> bool:
        # スポーン前や画面サイズ未設定時は不安定扱いしない
        if not self._spawned or self._screen_w is None or self._screen_h is None:
            return False

        unstable = False

        # 左
        if self.left_rect_body is not None:
            if self._is_visible(self.left_rect_body):
                self._left_seen = True
            elif self._left_seen:
                unstable = True
            lfx = float(self.left_finger.body.position.x)
            lfy = float(self.left_finger.body.position.y)
            lfr = float(self.left_finger.radius)
            if self._is_off_finger_horizontally(self.left_rect_body, lfx, lfr):
                unstable = True
            if self._is_entirely_below_finger(self.left_rect_body, lfy, lfr):
                unstable = True

        # 右
        if self.right_rect_body is not None:
            if self._is_visible(self.right_rect_body):
                self._right_seen = True
            elif self._right_seen:
                unstable = True
            rfx = float(self.right_finger.body.position.x)
            rfy = float(self.right_finger.body.position.y)
            rfr = float(self.right_finger.radius)
            if self._is_off_finger_horizontally(self.right_rect_body, rfx, rfr):
                unstable = True
            if self._is_entirely_below_finger(self.right_rect_body, rfy, rfr):
                unstable = True

        return unstable

    def update(
        self,
        keypoints: Keypoints2D,
        dt_s: float,
        frame_h: Optional[int] = None,
        frame_w: Optional[int] = None,
    ) -> tuple[bool, object]:
        if frame_h is not None:
            self._screen_h = int(frame_h)
        if frame_w is not None:
            self._screen_w = int(frame_w)
        # 画面幅に応じた長方形サイズ更新（スポーン前に反映される）
        self._update_rect_size_by_screen()
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

        # 状態判定（どちらか一方でもNG条件）を算出 → 0.5秒継続でゲームオーバー
        unstable_now = self._evaluate_unstable_conditions()

        # 連続不安定時間の更新（実時間 dt_s を使用）
        real_dt = float(dt_s) if dt_s and float(dt_s) > 0 else 0.0
        if unstable_now:
            self._unstable_time_s += real_dt
        else:
            self._unstable_time_s = 0.0

        is_stable = self._unstable_time_s < 0.5

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
            
            # 画像がある場合
            if hasattr(self, '_rect_img') and self._rect_img is not None:
                # 画像のサイズを長方形に合わせる
                img = cv2.resize(self._rect_img, (int(hw * 2), int(hh * 2)))
                
                # 回転後の画像サイズを計算
                img_h, img_w = img.shape[:2]
                diagonal = int(math.sqrt(img_h**2 + img_w**2)) + 2
                
                # 大きめのキャンバスを作成して中央に配置
                canvas = np.zeros((diagonal, diagonal, 3), dtype=np.uint8)
                offset_x = (diagonal - img_w) // 2
                offset_y = (diagonal - img_h) // 2
                canvas[offset_y:offset_y + img_h, offset_x:offset_x + img_w] = img
                
                # 回転
                center = (diagonal // 2, diagonal // 2)
                rot_mat = cv2.getRotationMatrix2D(center, -math.degrees(a), 1.0)
                rotated = cv2.warpAffine(canvas, rot_mat, (diagonal, diagonal))
                
                # フレームに合成する位置を計算
                top_left_x = int(x - diagonal // 2)
                top_left_y = int(y - diagonal // 2)
                
                # 貼り付け範囲の計算（クリッピング）
                src_y1 = max(0, -top_left_y)
                src_y2 = min(diagonal, frame_bgr.shape[0] - top_left_y)
                src_x1 = max(0, -top_left_x)
                src_x2 = min(diagonal, frame_bgr.shape[1] - top_left_x)
                
                dst_y1 = max(0, top_left_y)
                dst_y2 = dst_y1 + (src_y2 - src_y1)
                dst_x1 = max(0, top_left_x)
                dst_x2 = dst_x1 + (src_x2 - src_x1)
                
                # 範囲が有効な場合のみ描画
                if src_y2 > src_y1 and src_x2 > src_x1:
                    roi = rotated[src_y1:src_y2, src_x1:src_x2]
                    
                    # マスク作成（黒い部分=0,0,0を透過）
                    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mask = (gray > 10).astype(np.uint8)  # 閾値10で2値化
                    mask_3ch = np.stack([mask, mask, mask], axis=2)  # 3チャンネルに拡張
                    
                    # 背景と合成（NumPy配列演算）
                    bg = frame_bgr[dst_y1:dst_y2, dst_x1:dst_x2]
                    blended = np.where(mask_3ch > 0, roi, bg)
                    frame_bgr[dst_y1:dst_y2, dst_x1:dst_x2] = blended
            else:
                # 画像がない場合は黄色で描画
                corners = [(+hw, +hh), (-hw, +hh), (-hw, -hh), (+hw, -hh)]
                pts = []
                for px, py in corners:
                    rx = x + (px * ca - py * sa)
                    ry = y + (px * sa + py * ca)
                    pts.append((int(rx), int(ry)))
                cv2.fillPoly(frame_bgr, [np.array(pts, dtype=np.int32)], (0, 200, 255))
                cv2.polylines(frame_bgr, [np.array(pts, dtype=np.int32)], True, (0, 0, 0), 2, cv2.LINE_AA)

        # 左右の長方形を描画
        if self.left_rect_body is not None:
            _draw_rect(self.left_rect_body)
        if self.right_rect_body is not None:
            _draw_rect(self.right_rect_body)