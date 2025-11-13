from __future__ import annotations

from typing import Dict

from .types import GameConfig, StabilizerConfig


EASY = StabilizerConfig(max_tilt_deg=18.0, max_jerk=0.8, clear_seconds=5.0)
NORMAL = StabilizerConfig(max_tilt_deg=12.0, max_jerk=0.5, clear_seconds=10.0)
HARD = StabilizerConfig(max_tilt_deg=8.0, max_jerk=0.3, clear_seconds=15.0)


DIFFICULTY_PRESETS: Dict[str, StabilizerConfig] = {
    "easy": EASY,
    "normal": NORMAL,
    "hard": HARD,
}


def default_game_config(difficulty: str = "normal", target_fps: int = 30) -> GameConfig:
    difficulty_key = difficulty.lower()
    stabilizer = DIFFICULTY_PRESETS.get(difficulty_key, NORMAL)
    return GameConfig(
        target_fps=target_fps, difficulty=difficulty_key, stabilizer=stabilizer
    )
