from __future__ import annotations

from typing import Optional, Any

import numpy as np

try:
    import pygame  # type: ignore

    _PYGAME_AVAILABLE = True
except Exception:  # pragma: no cover - 環境依存のため安全に無効化
    pygame = None  # type: ignore
    _PYGAME_AVAILABLE = False


class SoundEffects:
    """短い効果音を再生するための極小ユーティリティ。

    - 追加の音声アセットに依存せず、NumPy で波形を合成
    - pygame.mixer が使えない環境では自動的に無効化（ノーオペ）
    """

    def __init__(self) -> None:
        self._enabled: bool = False
        self._sample_rate: int = 44100
        self._select_sound: Optional[Any] = None  # pygame.mixer.Sound | None
        self._diff_sound: Optional[Any] = None  # pygame.mixer.Sound | None

        self._init_mixer()
        if self._enabled:
            self._build_sounds()

    def _init_mixer(self) -> None:
        if not _PYGAME_AVAILABLE:  # pygame が無い場合は無効化
            self._enabled = False
            return
        try:
            if not pygame.mixer.get_init():  # type: ignore[attr-defined]
                pygame.mixer.pre_init(
                    frequency=self._sample_rate, size=-16, channels=1, buffer=512
                )
                pygame.mixer.init()
                pygame.mixer.set_num_channels(8)
            self._enabled = True
        except Exception:
            # オーディオデバイスが無い等の理由で初期化に失敗した場合は無効化
            self._enabled = False

    def _build_sounds(self) -> None:
        """UI で使用する短い効果音を合成して準備する。"""
        # Select: 短い上昇2音 + 微小な無音
        select_wave = np.concatenate(
            [
                self._tone(660.0, 0.06, 0.5),
                self._silence(0.01),
                self._tone(880.0, 0.07, 0.5),
            ],
            axis=0,
        )
        # Difficulty: 一音の軽いクリック風
        diff_wave = self._tone(740.0, 0.05, 0.45)

        self._select_sound = self._make_sound(select_wave)
        self._diff_sound = self._make_sound(diff_wave)

    def _tone(
        self, freq_hz: float, duration_s: float, volume: float = 0.5
    ) -> np.ndarray:
        n_samples = max(1, int(self._sample_rate * duration_s))
        t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
        wave = np.sin(2.0 * np.pi * float(freq_hz) * t).astype(np.float32)
        wave *= float(max(0.0, min(1.0, volume)))
        # クリックノイズ軽減のためフェードイン/アウト
        fade_len = max(1, int(min(0.01, duration_s * 0.15) * self._sample_rate))
        if fade_len > 0 and wave.size > fade_len * 2:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            wave[:fade_len] *= fade_in
            wave[-fade_len:] *= fade_out
        # int16 へ
        wave_i16 = np.clip(wave * 32767.0, -32768.0, 32767.0).astype(np.int16)
        return wave_i16

    def _silence(self, duration_s: float) -> np.ndarray:
        n_samples = max(1, int(self._sample_rate * duration_s))
        return np.zeros(n_samples, dtype=np.int16)

    def _make_sound(self, wave_i16: np.ndarray):  # -> pygame.mixer.Sound | None
        if not self._enabled:
            return None
        try:
            # pygame.sndarray は (samples,) もしくは (samples, channels)
            return pygame.sndarray.make_sound(wave_i16)  # type: ignore[attr-defined]
        except Exception:
            return None

    def play_select(self) -> None:
        if not self._enabled or self._select_sound is None:
            return
        try:
            self._select_sound.play()  # type: ignore[union-attr]
        except Exception:
            pass

    def play_difficulty_change(self) -> None:
        if not self._enabled or self._diff_sound is None:
            return
        try:
            self._diff_sound.play()  # type: ignore[union-attr]
        except Exception:
            pass


_singleton: Optional[SoundEffects] = None


def get_sound_effects() -> SoundEffects:
    global _singleton
    if _singleton is None:
        _singleton = SoundEffects()
    return _singleton
