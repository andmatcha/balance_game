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
    detector = PoseDetector()
    timer = FrameTimer()

    # 長方形描画を使用するため画像アセットは未使用

    window_name = "Balance Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = cam.read()
        if frame is None:
            break

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

        # 頭上の位置（鼻があれば少し上へ）
        head_center = None
        if det.keypoints.nose:
            head_center = Point2D(
                det.keypoints.nose.x, det.keypoints.nose.y - base_width * 0.35
            )

        # 手首
        lw = det.keypoints.left_wrist
        rw = det.keypoints.right_wrist

        # 長方形の描画（頭上）
        from .utils.drawing import draw_rotated_rect

        if head_center is not None:
            rect_w = max(10, int(base_width * 0.22))
            rect_h = max(20, int(base_width * 1.2))
            obj_angle = (
                metrics.object_tilt_deg if hasattr(metrics, "object_tilt_deg") else 0.0
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
