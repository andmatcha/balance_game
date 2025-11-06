from __future__ import annotations

import cv2
import numpy as np

from .types import GameStatus
from .utils.drawing import draw_countdown_center
from .retro import (
    render_pyxel_title_overlay,
    render_pyxel_hud,
    render_pyxel_prepare_overlay,
    render_pyxel_result_overlay,
)


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
    # レトロ調HUDへ置換
    render_pyxel_hud(
        frame_bgr,
        status=str(status),
        difficulty=difficulty,
        elapsed_s=elapsed_s,
        stable_s=stable_s,
        fps=fps,
        tilt_deg=tilt_deg,
        head_vx=head_vx,
        countdown_s=countdown_s,
    )

    # カウントダウンは中央に残す（数値のみ）
    if status == GameStatus.COUNTDOWN and countdown_s > 0.0:
        draw_countdown_center(frame_bgr, countdown_s)


def draw_title(frame_bgr: np.ndarray, difficulty: str):
    # Pyxel 風のレトロタイトルを現在フレームに直接オーバーレイ
    render_pyxel_title_overlay(frame_bgr, difficulty)


def draw_result(
    frame_bgr: np.ndarray,
    cleared: bool,
    elapsed_s: float,
    stable_s: float,
):
    render_pyxel_result_overlay(
        frame_bgr,
        cleared=cleared,
        elapsed_s=elapsed_s,
        stable_s=stable_s,
    )


def draw_prepare(frame_bgr: np.ndarray, has_head: bool, has_finger: bool):
    render_pyxel_prepare_overlay(frame_bgr, has_head=has_head, has_finger=has_finger)
