from __future__ import annotations

import cv2
import numpy as np

from .types import GameStatus
from .utils.drawing import draw_text_with_bg, draw_countdown_center


def draw_hud(
    frame_bgr: np.ndarray,
    status: GameStatus,
    difficulty: str,
    elapsed_s: float,
    stable_s: float,
    fps: float,
    tilt_deg: float,
    head_vx: float,
    countdown_s: float = 0.0,
):
    y = 24
    draw_text_with_bg(frame_bgr, f"Status: {status}", (10, y))
    y += 22
    draw_text_with_bg(frame_bgr, f"Difficulty: {difficulty}", (10, y))
    y += 22
    draw_text_with_bg(
        frame_bgr, f"Elapsed: {elapsed_s:.1f}s  Stable: {stable_s:.1f}s", (10, y)
    )
    y += 22
    draw_text_with_bg(frame_bgr, f"FPS: {fps:.1f}", (10, y))
    y += 22
    draw_text_with_bg(
        frame_bgr, f"ObjTilt: {tilt_deg:.1f} deg  HeadVx: {head_vx:.1f}", (10, y)
    )
    y += 22

    # キーガイド
    h = frame_bgr.shape[0]
    draw_text_with_bg(
        frame_bgr, "[S] Start  [R] Reset  [1/2/3] Difficulty  [Q] Quit", (10, h - 10)
    )

    # カウントダウン表示（中央）
    if status == GameStatus.COUNTDOWN and countdown_s > 0.0:
        draw_countdown_center(frame_bgr, countdown_s)
