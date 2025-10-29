from __future__ import annotations

import time
from collections import deque
from typing import Deque


class FrameTimer:
    def __init__(self, window_size: int = 30):
        self.prev_time = time.perf_counter()
        self.deltas: Deque[float] = deque(maxlen=window_size)

    def tick(self) -> float:
        now = time.perf_counter()
        dt = now - self.prev_time
        self.prev_time = now
        # dt が極端に大きい/小さい場合をクランプ
        if dt < 1e-6:
            dt = 1e-6
        if dt > 1.0:
            dt = 1.0
        self.deltas.append(dt)
        return dt

    def fps(self) -> float:
        if not self.deltas:
            return 0.0
        avg_dt = sum(self.deltas) / len(self.deltas)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0
