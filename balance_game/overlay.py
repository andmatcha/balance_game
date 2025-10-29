from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def load_rgba(path: str) -> Optional[np.ndarray]:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.shape[2] == 3:
        b, g, r = cv2.split(img)
        a = np.full_like(b, 255)
        img = cv2.merge((b, g, r, a))
    return img


def rotate_and_scale(
    img_rgba: np.ndarray, angle_deg: float, scale: float
) -> np.ndarray:
    h, w = img_rgba.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, angle_deg, scale)
    rotated = cv2.warpAffine(
        img_rgba,
        m,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return rotated


def resize_to_width(img_rgba: np.ndarray, width: int) -> np.ndarray:
    h, w = img_rgba.shape[:2]
    if w == 0:
        return img_rgba
    scale = width / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img_rgba, new_size, interpolation=cv2.INTER_AREA)


def overlay_rgba_center(
    frame_bgr: np.ndarray, img_rgba: np.ndarray, center_xy: Tuple[int, int]
) -> np.ndarray:
    fh, fw = frame_bgr.shape[:2]
    ih, iw = img_rgba.shape[:2]
    cx, cy = center_xy
    x1 = int(cx - iw / 2)
    y1 = int(cy - ih / 2)
    x2 = x1 + iw
    y2 = y1 + ih

    # クリッピング
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(fw, x2)
    y2_clip = min(fh, y2)
    if x1_clip >= x2_clip or y1_clip >= y2_clip:
        return frame_bgr

    roi = frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip]
    img_roi = img_rgba[(y1_clip - y1) : (y2_clip - y1), (x1_clip - x1) : (x2_clip - x1)]

    b, g, r, a = cv2.split(img_roi)
    alpha = a.astype(float) / 255.0
    alpha_3 = cv2.merge([alpha, alpha, alpha])

    bg = roi.astype(float)
    fg = cv2.merge([b, g, r]).astype(float)
    blended = cv2.add(cv2.multiply(alpha_3, fg), cv2.multiply(1.0 - alpha_3, bg))
    frame_bgr[y1_clip:y2_clip, x1_clip:x2_clip] = blended.astype(np.uint8)
    return frame_bgr
