from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .config import DIFFICULTY_PRESETS, IMAGES_DIR, default_game_config
from .game_logic import GameLogic
from .overlay import load_rgba, overlay_rgba_center, resize_to_width
from .physics import RectanglePhysics
from .pose_detector import PoseDetector
from .types import GameStatus, Point2D
from .utils.geometry import distance, midpoint, shoulder_angle_deg
from .utils.timing import FrameTimer


def _int_point(p: Point2D) -> tuple[int, int]:
    return int(p.x), int(p.y)


def _draw_debug_points(
    frame: np.ndarray, points: list[tuple[int, int]], color=(0, 255, 0)
):
    for pt in points:
        cv2.circle(frame, pt, 5, color, -1)


def main():
    cfg = default_game_config(difficulty="normal", target_fps=30)
    logic = GameLogic(cfg)
    physics = RectanglePhysics(cfg.stabilizer)

    cam = Camera(index=1)
    detector = PoseDetector(mirrored=True)
    timer = FrameTimer()

    # 長方形描画を使用するため画像アセットは未使用

    window_name = "Balance Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        if frame is None:
            break
        # インカメラの左右反転を補正（水平反転）
        frame = cv2.flip(frame, 1)

        dt_s = timer.tick()
        det = detector.detect(frame)
        is_stable, metrics = physics.update(det.keypoints, dt_s)
        logic.update(is_stable, dt_s)

        # オーバーレイの角度推定（肩がある場合）
        angle_deg = 0.0
        if det.keypoints.left_shoulder and det.keypoints.right_shoulder:
            angle_deg = -shoulder_angle_deg(
                det.keypoints.left_shoulder, det.keypoints.right_shoulder
            )

        # オーバーレイのスケール
        base_width = 120
        if det.keypoints.left_shoulder and det.keypoints.right_shoulder:
            shoulder_w = distance(
                det.keypoints.left_shoulder, det.keypoints.right_shoulder
            )
            base_width = int(max(60, min(240, shoulder_w * 0.8)))

        # 頭上の位置は矩形サイズに依存させる（鼻から矩形高さの約60%上）
        head_center = None

        # 長方形の描画（頭上）
        from .utils.drawing import draw_rotated_rect

        if det.keypoints.head_top is not None or det.keypoints.nose is not None:
            rect_w = max(10, int(base_width * 0.22))
            rect_h = max(20, int(base_width * 1.2))
            obj_angle = (
                metrics.object_tilt_deg if hasattr(metrics, "object_tilt_deg") else 0.0
            )
            if det.keypoints.head_top is not None:
                # 頭頂部の水平線の“上”（矩形高さの半分だけ上）に配置
                head_center = Point2D(
                    det.keypoints.head_top.x, det.keypoints.head_top.y - rect_h * 0.5
                )
            else:
                # フォールバック: 鼻基準
                head_center = Point2D(
                    det.keypoints.nose.x, det.keypoints.nose.y - rect_h * 0.6  # type: ignore
                )
            draw_rotated_rect(
                frame,
                _int_point(head_center),
                rect_w,
                rect_h,
                angle_deg + obj_angle,
                color=(0, 200, 255),
                alpha=0.85,
            )

        # 頭頂部・顎先の点と直線
        if det.keypoints.head_top is not None and det.keypoints.chin is not None:
            ht = _int_point(det.keypoints.head_top)
            ch = _int_point(det.keypoints.chin)
            cv2.circle(frame, ht, 5, (0, 255, 0), -1)
            cv2.circle(frame, ch, 5, (0, 0, 255), -1)
            cv2.line(frame, ht, ch, (255, 255, 0), 2, cv2.LINE_AA)
            # 直交ガイドライン（長さ=顔幅程度）
            vx = float(ch[0] - ht[0])
            vy = float(ch[1] - ht[1])
            norm = (vx * vx + vy * vy) ** 0.5
            if norm > 1e-6:
                nx = -vy / norm
                ny = vx / norm
                face_w = float(base_width)
                if "face_bbox" in det.metadata:
                    x1, y1, x2, y2 = det.metadata["face_bbox"]  # type: ignore
                    face_w = max(10.0, float(x2 - x1))
                half = int(face_w * 0.5)
                p1 = (int(ht[0] + nx * half), int(ht[1] + ny * half))
                p2 = (int(ht[0] - nx * half), int(ht[1] - ny * half))
                cv2.line(frame, p1, p2, (255, 0, 255), 2, cv2.LINE_AA)

        # 腕の点と線（肩-肘-手首-人差し指先）
        def _draw_arm(side: str, color: tuple[int, int, int]):
            shoulder = (
                det.keypoints.left_shoulder
                if side == "left"
                else det.keypoints.right_shoulder
            )
            elbow = (
                det.keypoints.left_elbow
                if side == "left"
                else det.keypoints.right_elbow
            )
            wrist = (
                det.keypoints.left_wrist
                if side == "left"
                else det.keypoints.right_wrist
            )
            index_base = (
                det.keypoints.left_index_mcp
                if side == "left"
                else det.keypoints.right_index_mcp
            )
            pts = []
            for p in [shoulder, elbow, wrist, index_base]:
                if p is not None:
                    pts.append(_int_point(p))
                else:
                    pts.append(None)
            # 点
            for pt in pts:
                if pt is not None:
                    cv2.circle(frame, pt, 4, color, -1)
            # 線（隣接を接続）
            for i in range(3):
                a, b = pts[i], pts[i + 1]
                if a is not None and b is not None:
                    cv2.line(frame, a, b, color, 2, cv2.LINE_AA)

        _draw_arm("left", (0, 255, 255))
        _draw_arm("right", (255, 128, 0))

        # 人差し指各関節（MCP/PIP/DIP/TIP）
        def _draw_index(side: str, color: tuple[int, int, int]):
            if side == "left":
                mcp = det.keypoints.left_index_mcp
                pip = det.keypoints.left_index_pip
                dip = det.keypoints.left_index_dip
                tip = det.keypoints.left_index
            else:
                mcp = det.keypoints.right_index_mcp
                pip = det.keypoints.right_index_pip
                dip = det.keypoints.right_index_dip
                tip = det.keypoints.right_index
            pts = []
            for p in [mcp, pip, dip, tip]:
                pts.append(_int_point(p) if p is not None else None)
            # 点
            for pt in pts:
                if pt is not None:
                    cv2.circle(frame, pt, 3, color, -1)
            # 線（隣接接続）
            for i in range(3):
                a, b = pts[i], pts[i + 1]
                if a is not None and b is not None:
                    cv2.line(frame, a, b, color, 2, cv2.LINE_AA)

        _draw_index("left", (0, 200, 0))
        _draw_index("right", (0, 0, 200))

        # HUD
        from .ui import draw_hud

        draw_hud(
            frame,
            status=logic.runtime.status,
            difficulty=cfg.difficulty,
            elapsed_s=logic.runtime.elapsed_time_s,
            stable_s=logic.runtime.stable_time_s,
            fps=timer.fps(),
            tilt_deg=(
                metrics.object_tilt_deg if hasattr(metrics, "object_tilt_deg") else 0.0
            ),
            head_vx=(metrics.head_vx if hasattr(metrics, "head_vx") else 0.0),
            countdown_s=logic.runtime.countdown_remaining_s,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            physics = RectanglePhysics(cfg.stabilizer)
            logic = GameLogic(cfg)
        elif key == ord("s"):
            # 3秒カウントダウン開始
            logic.start_countdown(3.0)
        elif key in (ord("1"), ord("2"), ord("3")):
            if key == ord("1"):
                cfg.difficulty = "easy"
            elif key == ord("2"):
                cfg.difficulty = "normal"
            elif key == ord("3"):
                cfg.difficulty = "hard"
            cfg.stabilizer = DIFFICULTY_PRESETS[cfg.difficulty]
            physics = RectanglePhysics(cfg.stabilizer)
            logic = GameLogic(cfg)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
