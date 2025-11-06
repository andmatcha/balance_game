from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import cv2

from .camera import Camera
from .config import DIFFICULTY_PRESETS, default_game_config
from .game_logic import GameLogic
from .physics import FingerBalancePhysics
from .pose_detector import PoseDetector
from .types import GameStatus, GameConfig
from .ui import draw_hud, draw_title, draw_result, draw_prepare
from .utils.timing import FrameTimer


class Screen(Enum):
    TITLE = auto()
    PREPARE = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    RESULT = auto()


@dataclass
class AppState:
    cfg: GameConfig
    logic: GameLogic
    physics: FingerBalancePhysics


class ScreenManager:
    def __init__(self) -> None:
        self.screen: Screen = Screen.TITLE
        self._prepare_ok_frames: int = 0

    def reset(self) -> None:
        self.screen = Screen.TITLE
        self._prepare_ok_frames = 0

    def update(self, has_both_fingers: bool, logic: GameLogic) -> None:
        if self.screen == Screen.PREPARE:
            if has_both_fingers:
                self._prepare_ok_frames += 1
            else:
                self._prepare_ok_frames = 0
            if self._prepare_ok_frames >= 9:
                logic.start_countdown(3.0)
                self.screen = Screen.COUNTDOWN
                self._prepare_ok_frames = 0

        if self.screen == Screen.COUNTDOWN:
            if not has_both_fingers:
                logic.reset()
                self.screen = Screen.PREPARE
                self._prepare_ok_frames = 0
            elif logic.runtime.status == GameStatus.PLAYING:
                self.screen = Screen.PLAYING

        if self.screen == Screen.PLAYING and logic.runtime.status in (
            GameStatus.CLEAR,
            GameStatus.FAIL,
        ):
            self.screen = Screen.RESULT

    def draw(self, frame, state: AppState, timer: FrameTimer, metrics) -> None:
        if self.screen == Screen.TITLE:
            draw_title(frame, difficulty=state.cfg.difficulty)
            return
        if self.screen == Screen.PREPARE:
            draw_prepare(frame, has_head=False, has_finger=True)
            return
        if self.screen == Screen.RESULT:
            draw_result(
                frame,
                cleared=(state.logic.runtime.status == GameStatus.CLEAR),
                elapsed_s=state.logic.runtime.elapsed_time_s,
                stable_s=state.logic.runtime.stable_time_s,
            )
            return
        draw_hud(
            frame,
            status=state.logic.runtime.status,
            difficulty=state.cfg.difficulty,
            elapsed_s=state.logic.runtime.elapsed_time_s,
            stable_s=state.logic.runtime.stable_time_s,
            fps=timer.fps(),
            tilt_deg=(
                metrics.object_tilt_deg if hasattr(metrics, "object_tilt_deg") else 0.0
            ),
            head_vx=(metrics.head_vx if hasattr(metrics, "head_vx") else 0.0),
            countdown_s=state.logic.runtime.countdown_remaining_s,
        )


def handle_key_input(
    key: int, state: AppState, sm: ScreenManager
) -> Tuple[bool, AppState]:
    quit_game = False
    cfg = state.cfg
    logic = state.logic
    physics = state.physics

    if key == ord("q"):
        quit_game = True
    elif key == ord("r"):
        physics = FingerBalancePhysics(cfg.stabilizer, rect_image_path="pizza-64.png")
        logic = GameLogic(cfg)
        sm.reset()
    elif key == ord("s"):
        logic.start_countdown(3.0)
        sm.screen = Screen.COUNTDOWN
    elif key == ord(" "):
        if sm.screen == Screen.TITLE:
            physics = FingerBalancePhysics(cfg.stabilizer, rect_image_path="pizza-64.png")
            logic = GameLogic(cfg)
            sm.screen = Screen.PREPARE
        elif sm.screen == Screen.RESULT:
            physics = FingerBalancePhysics(cfg.stabilizer, rect_image_path="pizza-64.png")
            logic = GameLogic(cfg)
            sm.screen = Screen.TITLE
    elif key in (ord("1"), ord("2"), ord("3")):
        if key == ord("1"):
            cfg.difficulty = "easy"
        elif key == ord("2"):
            cfg.difficulty = "normal"
        elif key == ord("3"):
            cfg.difficulty = "hard"
        cfg.stabilizer = DIFFICULTY_PRESETS[cfg.difficulty]
        physics = FingerBalancePhysics(cfg.stabilizer, rect_image_path="pizza-64.png")
        logic = GameLogic(cfg)

    return quit_game, AppState(cfg=cfg, logic=logic, physics=physics)


class GameApp:
    def __init__(self, difficulty: str = "normal", target_fps: int = 30) -> None:
        cfg = default_game_config(difficulty=difficulty, target_fps=target_fps)
        self.state = AppState(
            cfg=cfg, logic=GameLogic(cfg), physics=FingerBalancePhysics(cfg.stabilizer, rect_image_path="pizza-64.png")
        )

        self.cam = Camera(width=1280, height=720)
        self.detector = PoseDetector(mirrored=True)
        self.timer = FrameTimer()
        self.sm = ScreenManager()

        self.window_name = "Balance Game"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

    def run(self) -> None:
        while True:
            frame = self.cam.read()
            if frame is None:
                break
            frame = cv2.flip(frame, 1)
            # 表示・物理計算を 1280x720 ベースに統一
            frame = cv2.resize(frame, (1280, 720))

            dt_s = self.timer.tick()
            det = self.detector.detect(frame)

            # 検出状態
            has_left_finger = det.keypoints.left_index is not None
            has_right_finger = det.keypoints.right_index is not None
            has_both_fingers = has_left_finger and has_right_finger

            # PLAYING までは常にリセットして開始時にスポーンさせる
            if self.state.logic.runtime.status != GameStatus.PLAYING:
                self.state.physics.reset()

            # 物理更新
            is_stable, metrics = self.state.physics.update(
                det.keypoints, dt_s, frame.shape[0], frame.shape[1]
            )
            self.state.logic.update(is_stable, dt_s)

            # 描画
            self.state.physics.draw(frame)
            self.sm.draw(frame, self.state, self.timer, metrics)

            # 画面遷移
            self.sm.update(has_both_fingers=has_both_fingers, logic=self.state.logic)

            cv2.imshow(self.window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            quit_game, self.state = handle_key_input(key, self.state, self.sm)
            if quit_game:
                break

        self.cam.release()
        cv2.destroyAllWindows()
