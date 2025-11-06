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


def _put_retro_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color_idx: int = 7,
    font_scale: float = 0.55,
):
    font = cv2.FONT_HERSHEY_SIMPLEX
    line = cv2.LINE_8
    color = tuple(int(c) for c in PYXEL_PALETTE_BGR[color_idx])
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), font, font_scale, (0, 0, 0), 2, line)
    cv2.putText(img, text, (x, y), font, font_scale, color, 1, line)


def render_pyxel_hud(
    frame_bgr: np.ndarray,
    status: str,
    difficulty: str,
    elapsed_s: float,
    stable_s: float,
    fps: float,
    tilt_deg: float,
    head_vx: float,
    countdown_s: float = 0.0,
):
    h, w = frame_bgr.shape[:2]

    # パネルサイズ・位置
    x0, y0 = 12, 12
    pw = min(440, max(300, int(w * 0.32)))
    ph = min(210, max(150, int(h * 0.28)))

    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[1])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[12])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[13])

    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[:, :] = bg
    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), border_outer, 4)
    cv2.rectangle(panel, (8, 8), (pw - 9, ph - 9), border_inner, 2)

    # テキスト
    y = 28
    _put_retro_text(panel, "STATUS", (14, y), color_idx=10, font_scale=0.6)
    y += 24
    _put_retro_text(panel, f"{status}", (14, y), color_idx=7, font_scale=0.75)
    y += 24
    _put_retro_text(panel, f"Difficulty: {difficulty}", (14, y), color_idx=7)
    y += 22
    _put_retro_text(
        panel, f"Elapsed: {elapsed_s:.1f}s  Stable: {stable_s:.1f}s", (14, y)
    )
    y += 22
    _put_retro_text(panel, f"FPS: {fps:.1f}", (14, y))
    y += 22
    _put_retro_text(panel, f"Tilt: {tilt_deg:.1f} deg  HeadVx: {head_vx:.1f}", (14, y))

    panel = _add_scanlines(panel, strength=0.12)
    # 合成
    y1, x1 = y0 + ph, x0 + pw
    frame_bgr[y0:y1, x0:x1] = panel

    # 画面下のキーガイド（軽量描画）
    _put_retro_text(
        frame_bgr,
        "[S] Start  [R] Reset  [1/2/3] Difficulty  [Q] Quit",
        (12, h - 12),
        color_idx=7,
    )


def render_pyxel_prepare_overlay(
    frame_bgr: np.ndarray, has_head: bool, has_finger: bool
):
    h, w = frame_bgr.shape[:2]

    # パネル
    pw = int(w * 0.7)
    ph = int(h * 0.5)
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[1])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[12])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[13])
    panel[:, :] = bg
    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), border_outer, 6)
    cv2.rectangle(panel, (12, 12), (pw - 13, ph - 13), border_inner, 3)

    # 文言
    font = cv2.FONT_HERSHEY_SIMPLEX
    title = "Standby: Align for detection"
    (tw, th), _ = cv2.getTextSize(title, font, 0.8, 2)
    tx = max(16, (pw - tw) // 2)
    ty = int(ph * 0.35)
    cv2.putText(panel, title, (tx + 1, ty + 1), font, 0.8, (0, 0, 0), 3, cv2.LINE_8)
    cv2.putText(
        panel,
        title,
        (tx, ty),
        font,
        0.8,
        tuple(int(c) for c in PYXEL_PALETTE_BGR[7]),
        2,
        cv2.LINE_8,
    )

    status_head = "OK" if has_head else "--"
    status_finger = "OK" if has_finger else "--"
    _put_retro_text(panel, f"Head: {status_head}", (int(pw * 0.12), int(ph * 0.55)))
    _put_retro_text(
        panel, f"Index finger: {status_finger}", (int(pw * 0.12), int(ph * 0.62))
    )
    _put_retro_text(
        panel,
        "Hold position. Countdown starts when both are detected",
        (int(pw * 0.12), int(ph * 0.70)),
    )

    panel = _add_scanlines(panel, strength=0.16)

    x0 = (w - pw) // 2
    y0 = (h - ph) // 2
    frame_bgr[y0 : y0 + ph, x0 : x0 + pw] = panel


def render_pyxel_result_overlay(
    frame_bgr: np.ndarray,
    cleared: bool,
    elapsed_s: float,
    stable_s: float,
):
    h, w = frame_bgr.shape[:2]

    pw = int(w * 0.7)
    ph = int(h * 0.42)
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[1])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[12])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[13])
    panel[:, :] = bg
    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), border_outer, 6)
    cv2.rectangle(panel, (12, 12), (pw - 13, ph - 13), border_inner, 3)

    title = "GAME CLEAR" if cleared else "GAME OVER"
    color_idx = 11 if cleared else 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(title, font, 1.2, 3)
    tx = max(16, (pw - tw) // 2)
    ty = int(ph * 0.42)
    cv2.putText(panel, title, (tx + 1, ty + 1), font, 1.2, (0, 0, 0), 5, cv2.LINE_8)
    cv2.putText(
        panel,
        title,
        (tx, ty),
        font,
        1.2,
        tuple(int(c) for c in PYXEL_PALETTE_BGR[color_idx]),
        3,
        cv2.LINE_8,
    )

    _put_retro_text(
        panel,
        f"Elapsed: {elapsed_s:.1f}s  Stable: {stable_s:.1f}s",
        (int(pw * 0.12), int(ph * 0.62)),
    )
    _put_retro_text(panel, "[SPACE] Return to Title", (int(pw * 0.12), int(ph * 0.72)))

    panel = _add_scanlines(panel, strength=0.16)

    x0 = (w - pw) // 2
    y0 = (h - ph) // 2
    frame_bgr[y0 : y0 + ph, x0 : x0 + pw] = panel


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
