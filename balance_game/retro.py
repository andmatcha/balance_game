from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path


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


# -------------------------
# タイトル背景用 ピザ・パーティクル（等間隔・等サイズ）
_pizza_particles: list[dict] = []  # [{'x','y','r','vy','tops'}]
_pizza_screen_size: tuple[int, int] = (0, 0)
_pizza_r: int = 10
_pizza_dx: int = 40
_pizza_dy: int = 40
_pizza_cols: int = 0
_pizza_rows: int = 0
_pizza_sprite_base_rgba: np.ndarray | None = None
_pizza_sprite_cache_size: int = 0
_pizza_sprite_cache_rgba: np.ndarray | None = None


def _clear_pizzas():
    global _pizza_particles, _pizza_screen_size
    _pizza_particles = []
    _pizza_screen_size = (0, 0)


def _gen_toppings(r: int) -> list[tuple[int, int, int]]:
    toppings: list[tuple[int, int, int]] = []
    num_top = int(np.random.randint(2, 5))
    for _ in range(num_top):
        ang = float(np.random.uniform(0, 2 * np.pi))
        rad = float(np.random.uniform(0.2 * r, 0.7 * r))
        dx = int(rad * np.cos(ang))
        dy = int(rad * np.sin(ang))
        col_idx = 8 if np.random.rand() < 0.65 else 11
        toppings.append((dx, dy, col_idx))
    return toppings


def _load_pizza_sprite_base() -> np.ndarray:
    global _pizza_sprite_base_rgba
    if _pizza_sprite_base_rgba is not None:
        return _pizza_sprite_base_rgba
    img_path = Path(__file__).resolve().parent / "images" / "pizza.png"
    sprite = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if sprite is None:
        # フォールバック：透明な1x1
        _pizza_sprite_base_rgba = np.zeros((1, 1, 4), dtype=np.uint8)
        return _pizza_sprite_base_rgba
    if sprite.ndim == 3 and sprite.shape[2] == 3:
        # アルファ無し → 黒(0,0,0)を透過に近い扱いへ（白抜きより安全）
        alpha = (np.any(sprite != 0, axis=2).astype(np.uint8) * 255)[:, :, None]
        sprite = np.concatenate([sprite, alpha], axis=2)
    _pizza_sprite_base_rgba = sprite
    return _pizza_sprite_base_rgba


def _get_pizza_sprite_rgba(size_px: int) -> np.ndarray:
    global _pizza_sprite_cache_size, _pizza_sprite_cache_rgba
    if (
        _pizza_sprite_cache_rgba is not None
        and _pizza_sprite_cache_size == size_px
        and _pizza_sprite_cache_rgba.shape[0] == size_px
    ):
        return _pizza_sprite_cache_rgba
    base = _load_pizza_sprite_base()
    interp = cv2.INTER_AREA if size_px < max(base.shape[:2]) else cv2.INTER_LINEAR
    scaled = cv2.resize(base, (size_px, size_px), interpolation=interp)
    _pizza_sprite_cache_rgba = scaled
    _pizza_sprite_cache_size = size_px
    return scaled


def _ensure_pizzas(w: int, h: int):
    global _pizza_particles, _pizza_screen_size, _pizza_r, _pizza_dx, _pizza_dy, _pizza_cols, _pizza_rows
    if _pizza_screen_size != (w, h) or len(_pizza_particles) == 0:
        # 画面サイズに応じて等サイズ・等間隔を決定
        _pizza_r = int(max(6, min(12, int(min(w, h) * 0.012))))
        # 要望: 現在の5倍サイズ
        _pizza_r = max(1, _pizza_r * 5)
        _pizza_dx = int(_pizza_r * 4)
        _pizza_dy = int(_pizza_r * 4)
        _pizza_cols = max(1, (w - 2 * _pizza_r) // _pizza_dx)
        if _pizza_cols <= 0:
            _pizza_cols = 1
        # 左右マージンを均等にするためのオフセット
        x0 = int((w - (_pizza_cols - 1) * _pizza_dx) / 2)
        # 画面上から下まで等間隔に埋まる行数（上に余白を持たせる）
        _pizza_rows = int(np.ceil((h + 2 * _pizza_dy) / _pizza_dy)) + 1
        vy = max(1.0, _pizza_dy / 40.0)
        parts: list[dict] = []
        for r_idx in range(_pizza_rows):
            y = float(-_pizza_dy + r_idx * _pizza_dy)
            for c_idx in range(_pizza_cols):
                x = float(x0 + c_idx * _pizza_dx)
                parts.append(
                    {
                        "x": x,
                        "y": y,
                        "r": float(_pizza_r),
                        "vy": float(vy),
                        "tops": _gen_toppings(_pizza_r),
                    }
                )
        _pizza_particles = parts
        _pizza_screen_size = (w, h)


def _draw_pizza(
    img: np.ndarray, cx: int, cy: int, r: int, toppings: list[tuple[int, int, int]]
):
    # 画像スプライトで描画（toppings は保持のみ）
    size = int(max(2, r * 2))
    sprite = _get_pizza_sprite_rgba(size)
    h, w = img.shape[:2]
    sh, sw = sprite.shape[:2]
    x0 = int(cx - sw // 2)
    y0 = int(cy - sh // 2)
    x1 = x0 + sw
    y1 = y0 + sh
    ix0 = max(0, x0)
    iy0 = max(0, y0)
    ix1 = min(w, x1)
    iy1 = min(h, y1)
    if ix0 >= ix1 or iy0 >= iy1:
        return
    sx0 = ix0 - x0
    sy0 = iy0 - y0
    sx1 = sx0 + (ix1 - ix0)
    sy1 = sy0 + (iy1 - iy0)
    roi = img[iy0:iy1, ix0:ix1]
    sub = sprite[sy0:sy1, sx0:sx1]
    if sub.shape[2] == 4:
        alpha = sub[:, :, 3:4].astype(np.float32) / 255.0
        src_rgb = sub[:, :, :3].astype(np.float32)
        dst_rgb = roi.astype(np.float32)
        out = (alpha * src_rgb + (1.0 - alpha) * dst_rgb).astype(np.uint8)
        roi[:] = out
    else:
        roi[:] = sub


def _update_and_draw_pizzas(img: np.ndarray):
    h, w = img.shape[:2]
    _ensure_pizzas(w, h)
    # 等速度で下方向へ移動、等間隔維持のため行単位で巻き戻す
    for p in _pizza_particles:
        p["y"] += p["vy"]
        # 描画
        _draw_pizza(img, int(p["x"]), int(p["y"]), int(p["r"]), p["tops"])
        # 画面外へ出たら列・間隔を保ったまま上へ巻き戻し
        if p["y"] - p["r"] > h + 4:
            p["y"] -= float(_pizza_rows * _pizza_dy)


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

    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[0])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[8])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[10])

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
    # タイトル以外ではピザを無効化
    _clear_pizzas()
    h, w = frame_bgr.shape[:2]

    # パネル
    pw = int(w * 0.7)
    ph = int(h * 0.5)
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[0])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[8])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[10])
    panel[:, :] = bg
    # 枠線
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
    # タイトル以外ではピザを無効化
    _clear_pizzas()
    h, w = frame_bgr.shape[:2]

    pw = int(w * 0.7)
    ph = int(h * 0.42)
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[0])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[8])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[10])
    panel[:, :] = bg
    # 枠線のみ（ピザは表示しない）
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

    # 背景色（黒）と枠線（赤/黄）
    bg = tuple(int(c) for c in PYXEL_PALETTE_BGR[0])
    border_outer = tuple(int(c) for c in PYXEL_PALETTE_BGR[8])
    border_inner = tuple(int(c) for c in PYXEL_PALETTE_BGR[10])
    panel[:, :] = bg
    # まずピザ（テキストより後ろ）
    _update_and_draw_pizzas(panel)
    # 次に枠線（ピザより前）
    cv2.rectangle(panel, (0, 0), (pw - 1, ph - 1), border_outer, 6)
    cv2.rectangle(panel, (12, 12), (pw - 13, ph - 13), border_inner, 3)

    # タイトル文字（拡大ネアレストでドット感を強調）
    title = "PIZZA ACROBAT"
    # 小さく描いてから整数スケールで拡大する（より大きく見せるため幅を小さめに）
    small_h = 24
    small_w = min(pw - 40, 360)
    small = np.full((small_h, small_w, 3), bg, dtype=np.uint8)
    # 情報テキスト用は従来通り SIMPLEX を使う
    font = cv2.FONT_HERSHEY_SIMPLEX
    # タイトルは太めで見栄えの良い TRIPLEX を使用
    title_font = cv2.FONT_HERSHEY_TRIPLEX
    title_scale = 0.8
    # タイトルが枠からはみ出す場合はスケールを自動調整
    (tw, th), _ = cv2.getTextSize(title, title_font, title_scale, 2)
    if tw > small_w - 8:
        title_scale = title_scale * (small_w - 8) / max(1, tw)
        (tw, th), _ = cv2.getTextSize(title, title_font, title_scale, 2)
    tx = max(0, (small_w - tw) // 2)
    ty = (small_h + th) // 2 - 2
    # 多重ストロークでインパクトを出す（影→枠→縁→本体）
    shadow_thick = max(2, int(round(4 * title_scale)))
    accent_thick = max(2, int(round(3 * title_scale)))
    fill_thick = max(1, int(round(2 * title_scale)))
    stroke_thick = max(accent_thick + 1, int(round(5 * title_scale)))
    main_color = tuple(int(c) for c in PYXEL_PALETTE_BGR[7])  # off-white
    accent_color = tuple(int(c) for c in PYXEL_PALETTE_BGR[9])  # orange
    shadow_color = tuple(int(c) for c in PYXEL_PALETTE_BGR[1])  # dark navy (非黒)
    stroke_color = tuple(int(c) for c in PYXEL_PALETTE_BGR[5])  # dark gray (非黒)
    # 影（ドロップシャドウ）
    cv2.putText(
        small,
        title,
        (tx + 2, ty + 2),
        title_font,
        title_scale,
        shadow_color,
        shadow_thick,
        cv2.LINE_8,
    )
    # 外枠（非黒で透過回避）
    cv2.putText(
        small,
        title,
        (tx, ty),
        title_font,
        title_scale,
        stroke_color,
        stroke_thick,
        cv2.LINE_8,
    )
    # 縁（アクセントカラー）
    cv2.putText(
        small,
        title,
        (tx, ty),
        title_font,
        title_scale,
        accent_color,
        accent_thick,
        cv2.LINE_8,
    )
    # 本体（白系）
    cv2.putText(
        small,
        title,
        (tx, ty),
        title_font,
        title_scale,
        main_color,
        fill_thick,
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

    # big をパネル中央上部へ貼り付け（背景色は透過扱いで合成）
    bh, bw = big.shape[:2]
    bx = max(0, (pw - bw) // 2)
    by = int(ph * 0.18) - bh // 2
    by = max(20, min(ph - bh - 20, by))
    panel_roi = panel[by : by + bh, bx : bx + bw]
    bg_bgr = np.array(bg, dtype=np.uint8)
    mask = np.any(big != bg_bgr, axis=2)
    panel_roi[mask] = big[mask]

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
