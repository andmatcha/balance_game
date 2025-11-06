from __future__ import annotations

import cv2
import numpy as np


# Pyxel (PICO-8 系) デフォルト16色パレット（RGB）を OpenCV 用に BGR へ変換
_PYXEL_PALETTE_RGB: list[tuple[int, int, int]] = [
    (0, 0, 0),
    (29, 43, 83),
    (126, 37, 83),
    (0, 135, 81),
    (171, 82, 54),
    (95, 87, 79),
    (194, 195, 199),
    (255, 241, 232),
    (255, 0, 77),
    (255, 163, 0),
    (255, 236, 39),
    (0, 228, 54),
    (41, 173, 255),
    (131, 118, 156),
    (255, 119, 168),
    (255, 204, 170),
]

PYXEL_PALETTE_BGR: np.ndarray = np.array(
    [(b, g, r) for (r, g, b) in _PYXEL_PALETTE_RGB], dtype=np.uint8
)


def _quantize_to_palette(img_bgr: np.ndarray, palette_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    flat = img_bgr.reshape(-1, 3).astype(np.int16)
    pal = palette_bgr.astype(np.int16)  # (16,3)
    # 距離計算（ベクトル化）
    diff = flat[:, None, :] - pal[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)  # (N,16)
    idx = np.argmin(dist2, axis=1).astype(np.int32)
    quant = pal[idx].astype(np.uint8).reshape(h, w, 3)
    return quant


def _add_scanlines(img_bgr: np.ndarray, strength: float = 0.25) -> np.ndarray:
    out = img_bgr.copy()
    out[1::2] = (out[1::2].astype(np.float32) * (1.0 - strength)).astype(np.uint8)
    return out


def render_pyxel_title_overlay(frame_bgr: np.ndarray, difficulty: str) -> None:
    """現在のフレーム上に Pyxel 風タイトルパネルをオーバーレイ描画する（破壊的）。

    - 背景をうっすら暗く
    - パネルは Pyxel パレットへ量子化
    - スキャンライン適用
    """
    h, w = frame_bgr.shape[:2]

    # フルスクリーンのパネル領域（カメラ映像は表示しない）
    pw, ph = w, h
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)

    # 背景色（暗紺）と枠線（サイアン/紫）
    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[1])  # (29,43,83)
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[12])  # (41,173,255)
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[13])  # (131,118,156)
    panel[:, :] = bg
    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), border_outer, 6)
    cv2.rectangle(panel, (12, 12), (pw - 13, ph - 13), border_inner, 3)

    # タイトル文字（拡大ネアレストでドット感を強調）
    title = "BALANCE GAME"
    small_h = 18
    small_w = min(pw - 40, 520)
    small = np.full((small_h, small_w, 3), bg, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thick = 1
    (tw, th), _ = cv2.getTextSize(title, font, scale, thick)
    tx = max(0, (small_w - tw) // 2)
    ty = (small_h + th) // 2 - 2
    cv2.putText(small, title, (tx + 1, ty + 1), font, scale, (0, 0, 0), 2, cv2.LINE_8)
    cv2.putText(
        small,
        title,
        (tx, ty),
        font,
        scale,
        tuple(int(c) for c in PYXEL_PALETTE_BGR[7]),  # off-white in Pyxel palette
        1,
        cv2.LINE_8,
    )

    # パネル内に収まる整数スケール（横幅/高さの両方を満たす）
    max_w = pw - 40
    max_h = int(ph * 0.28)
    scale_w = max(1, max_w // small_w)
    scale_h = max(1, max_h // small_h)
    scale_factor = int(max(1, min(6, min(scale_w, scale_h))))
    big = cv2.resize(
        small, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST
    )

    # big をパネル中央上部へ貼り付け
    bh, bw = big.shape[:2]
    bx = max(0, (pw - bw) // 2)
    by = int(ph * 0.18) - bh // 2
    by = max(20, min(ph - bh - 20, by))
    panel[by : by + bh, bx : bx + bw] = big

    # 情報テキスト
    def put_line(text: str, y_ratio: float, col_idx: int = 7):
        color = tuple(int(c) for c in PYXEL_PALETTE_BGR[col_idx])
        (tw2, th2), _ = cv2.getTextSize(text, font, 0.55, 1)
        tx2 = max(12, (pw - tw2) // 2)
        ty2 = int(ph * y_ratio)
        cv2.putText(
            panel, text, (tx2 + 1, ty2 + 1), font, 0.55, (0, 0, 0), 2, cv2.LINE_8
        )
        cv2.putText(panel, text, (tx2, ty2), font, 0.55, color, 1, cv2.LINE_8)

    put_line(f"Difficulty: {difficulty}", 0.55, 10)
    put_line("[1/2/3] Change difficulty", 0.62, 6)
    put_line("[SPACE] Start", 0.69, 11)
    put_line("[Q] Quit", 0.76, 8)

    # スキャンライン適用（量子化は重い/色破綻につながるため廃止）
    panel_sl = _add_scanlines(panel, strength=0.16)

    # フレームへ合成（全画面置換）
    frame_bgr[:, :] = panel_sl
