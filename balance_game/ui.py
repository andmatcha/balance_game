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


def draw_title(frame_bgr: np.ndarray, difficulty: str):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (int(w * 0.08), int(h * 0.2)),
        (int(w * 0.92), int(h * 0.8)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)

    # タイトルとガイド
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "BALANCE GAME"
    scale = 1.6
    thickness = 3
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = int((w - tw) / 2)
    ty = int(h * 0.38)
    cv2.putText(
        frame_bgr,
        title,
        (tx, ty),
        font,
        scale,
        (255, 255, 255),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame_bgr, title, (tx, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA
    )

    draw_text_with_bg(
        frame_bgr, f"Difficulty: {difficulty}", (int(w * 0.12), int(h * 0.52))
    )
    draw_text_with_bg(
        frame_bgr, "[1/2/3] Change difficulty", (int(w * 0.12), int(h * 0.58))
    )
    draw_text_with_bg(frame_bgr, "[SPACE] Start", (int(w * 0.12), int(h * 0.64)))
    draw_text_with_bg(frame_bgr, "[Q] Quit", (int(w * 0.12), int(h * 0.70)))


def draw_result(
    frame_bgr: np.ndarray,
    cleared: bool,
    elapsed_s: float,
    stable_s: float,
):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (int(w * 0.08), int(h * 0.3)),
        (int(w * 0.92), int(h * 0.72)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, frame_bgr, 0.4, 0, frame_bgr)

    title = "GAME CLEAR" if cleared else "GAME OVER"
    color = (0, 255, 0) if cleared else (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.8
    thickness = 4
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = int((w - tw) / 2)
    ty = int(h * 0.46)
    cv2.putText(
        frame_bgr, title, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(frame_bgr, title, (tx, ty), font, scale, color, thickness, cv2.LINE_AA)

    draw_text_with_bg(
        frame_bgr,
        f"Elapsed: {elapsed_s:.1f}s  Stable: {stable_s:.1f}s",
        (int(w * 0.12), int(h * 0.56)),
    )
    draw_text_with_bg(
        frame_bgr, "[SPACE] Return to Title", (int(w * 0.12), int(h * 0.62))
    )


def draw_prepare(frame_bgr: np.ndarray, has_head: bool, has_finger: bool):
    h, w = frame_bgr.shape[:2]
    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (int(w * 0.08), int(h * 0.25)),
        (int(w * 0.92), int(h * 0.75)),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.5, frame_bgr, 0.5, 0, frame_bgr)

    title = "Standby: Align for detection"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.2
    thickness = 3
    (tw, th), _ = cv2.getTextSize(title, font, scale, thickness)
    tx = int((w - tw) / 2)
    ty = int(h * 0.38)
    cv2.putText(
        frame_bgr, title, (tx, ty), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
    )
    cv2.putText(
        frame_bgr, title, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA
    )

    status_head = "OK" if has_head else "--"
    status_finger = "OK" if has_finger else "--"
    draw_text_with_bg(frame_bgr, f"Head: {status_head}", (int(w * 0.12), int(h * 0.52)))
    draw_text_with_bg(
        frame_bgr, f"Index finger: {status_finger}", (int(w * 0.12), int(h * 0.58))
    )
    draw_text_with_bg(
        frame_bgr,
        "Hold position. Countdown starts when both are detected",
        (int(w * 0.12), int(h * 0.66)),
    )
