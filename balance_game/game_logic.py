from __future__ import annotations

from dataclasses import dataclass

from .types import GameConfig, GameStatus


@dataclass
class GameRuntime:
    status: GameStatus = GameStatus.READY
    elapsed_time_s: float = 0.0
    stable_time_s: float = 0.0
    countdown_remaining_s: float = 0.0


class GameLogic:
    def __init__(self, config: GameConfig):
        self.config = config
        self.runtime = GameRuntime()

    def reset(self):
        self.runtime = GameRuntime(status=GameStatus.READY)

    def start_countdown(self, seconds: float):
        if self.runtime.status in (GameStatus.READY, GameStatus.FAIL, GameStatus.CLEAR):
            self.runtime = GameRuntime(
                status=GameStatus.COUNTDOWN, countdown_remaining_s=seconds
            )

    def update(self, is_stable: bool, dt_s: float):
        # COUNTDOWN 中はカウントダウンを進めるのみ
        if self.runtime.status == GameStatus.COUNTDOWN:
            self.runtime.countdown_remaining_s = max(
                0.0, self.runtime.countdown_remaining_s - dt_s
            )
            if self.runtime.countdown_remaining_s <= 0.0:
                self.runtime.status = GameStatus.PLAYING
            return
        if self.runtime.status not in (GameStatus.PLAYING, GameStatus.READY):
            return
        if self.runtime.status == GameStatus.READY:
            # READY 中は経過時間のみ進める（安定/不安定はカウントしない）
            self.runtime.elapsed_time_s += dt_s
            return

        self.runtime.elapsed_time_s += dt_s
        if is_stable:
            self.runtime.stable_time_s += dt_s
        else:
            self.runtime.status = GameStatus.FAIL

        if self.runtime.stable_time_s >= self.config.stabilizer.clear_seconds:
            self.runtime.status = GameStatus.CLEAR
