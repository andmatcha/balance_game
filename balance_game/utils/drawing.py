from __future__ import annotations

import cv2
import numpy as np


def draw_text_with_bg(
    frame: np.ndarray,
    text: str,
    org: tuple[int, int],
    font_scale: float = 0.6,
    text_color: tuple[int, int, int] = (255, 255, 255),
    bg_color: tuple[int, int, int] = (0, 0, 0),
    thickness: int = 1,
    alpha: float = 0.6,
):
    x, y = org
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    pad = 4
    cv2.rectangle(
        frame, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg_color, -1
    )
    cv2.putText(
        frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA
    )


def draw_countdown_center(
    frame: np.ndarray,
    seconds_remaining: float,
    color: tuple[int, int, int] = (0, 255, 0),
):
    if seconds_remaining <= 0:
        return
    h, w = frame.shape[:2]
    text = f"{int(np.ceil(seconds_remaining))}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 3.0
    thickness = 6
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = int((w - tw) / 2)
    y = int((h + th) / 3)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
