import numpy as np

from balance_game.overlay import overlay_rgba_center


def test_overlay_changes_region():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    # 赤い四角 + アルファ
    overlay = np.zeros((20, 20, 4), dtype=np.uint8)
    overlay[..., 2] = 255
    overlay[..., 3] = 200

    out = overlay_rgba_center(frame.copy(), overlay, (50, 50))
    assert out.sum() > frame.sum()
