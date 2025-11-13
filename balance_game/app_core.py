from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Tuple

import cv2
import numpy as np
import os
import pkgutil 


import mediapipe as mp
mp_selfie_segmentation = mp.solutions.selfie_segmentation 


from .camera import Camera
from .config import DIFFICULTY_PRESETS, default_game_config
from .game_logic import GameLogic
from .physics import FingerBalancePhysics 
from .pose_detector import PoseDetector
from .types import GameStatus, GameConfig
from .ui import draw_hud, draw_title, draw_result, draw_prepare
from .utils.timing import FrameTimer
from .sound import get_sound_effects


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
    sfx = get_sound_effects()

    # 難易度処理のためのヘルパー関数
    def _create_new_physics(config: GameConfig) -> FingerBalancePhysics:
        return FingerBalancePhysics(
            config.stabilizer, 
            rect_image_path="pizza-64.png",
            difficulty=config.difficulty
        )

    # ★ 終了キーの修正: q, Q, ESC のいずれかで終了
    if key in (ord("q"), ord("Q"), 27): 
        quit_game = True
    elif key == ord("r"):
        physics = _create_new_physics(cfg) 
        logic = GameLogic(cfg)
        sm.reset()
    elif key == ord("s"):
        logic.start_countdown(3.0)
        sm.screen = Screen.COUNTDOWN
    elif key == ord(" "):
        sfx.play_select()
        if sm.screen == Screen.TITLE:
            physics = _create_new_physics(cfg) 
            logic = GameLogic(cfg)
            sm.screen = Screen.PREPARE
        elif sm.screen == Screen.RESULT:
            physics = _create_new_physics(cfg) 
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
        physics = _create_new_physics(cfg) 
        logic = GameLogic(cfg)
        sfx.play_difficulty_change()

    return quit_game, AppState(cfg=cfg, logic=logic, physics=physics)


class GameApp:
    def __init__(self, difficulty: str = "normal", target_fps: int = 30) -> None:
        cfg = default_game_config(difficulty=difficulty, target_fps=target_fps)
        
        
        self.state = AppState(
            cfg=cfg, 
            logic=GameLogic(cfg), 
            physics=FingerBalancePhysics(
                cfg.stabilizer, 
                rect_image_path="pizza-64.png",
                difficulty=cfg.difficulty
            )
        )


        self.cam = Camera(width=1280, height=720)
        self.detector = PoseDetector(mirrored=True)
        self.timer = FrameTimer()
        self.sm = ScreenManager()
        
        # ✅ MediaPipeのSelfieSegmentationを初期化
        self.segmenter = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
        
        self.window_name = "Balance Game"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)
        
        #  背景画像の読み込み 
        self.background_img = None
        FILE_NAME = "haikei1.png"
        
    
        fallback_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", FILE_NAME)
        self.background_img = cv2.imread(fallback_path, cv2.IMREAD_COLOR)

      
        if self.background_img is not None:
            self.background_img = cv2.resize(self.background_img, (1280, 720))
        # --- ここまで背景画像読み込み ---

    # 背景合成（セグメンテーション）処理
 
    def _apply_segmentation_effect(self, frame, mask_results):
        if self.background_img is None:
            return frame

        # MediaPipeのマスク（mask.segmentation_mask）
        mask_resized = cv2.resize(
            mask_results.segmentation_mask, # MediaPipeの結果からマスクデータを取り出す
            (frame.shape[1], frame.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )
        
        # マスクを3チャンネルに拡張 (1280x720x3)
        alpha_3ch = np.stack((mask_resized,) * 3, axis=-1)

      
        binary_mask = (alpha_3ch > 0.25).astype(np.float32)

        # 1. 前景 (人物) の抽出: 
        foreground = np.multiply(frame, binary_mask).astype('uint8')
        
        # 2. 背景 (haikei1.png) の抽出: 
        inverse_mask = np.ones(binary_mask.shape, dtype=np.float32) - binary_mask
        background = np.multiply(self.background_img, inverse_mask).astype('uint8')
        
        # 3. 合成
        return cv2.add(foreground, background)

    def run(self) -> None:
        while True:
            frame = self.cam.read()
            if frame is None:
                break
            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (1280, 720))

            dt_s = self.timer.tick()
            det = self.detector.detect(frame)

            # 検出状態
            has_left_finger = det.keypoints.left_index is not None
            has_right_finger = det.keypoints.right_index is not None
            has_both_fingers = has_left_finger and has_right_finger

            # セグメンテーション処理の実行
            # OpenCVはBGR、MediaPipeはRGBを想定するため色空間を変換
            segmentation_results = self.segmenter.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # 背景画像が読み込まれており、セグメンテーション結果がある場合に合成
            perform_segmentation_and_blend = (
                self.background_img is not None and 
                segmentation_results.segmentation_mask is not None
            )

            if perform_segmentation_and_blend:
                # 合成処理
                frame = self._apply_segmentation_effect(frame, segmentation_results)
            else:
                pass 

            # PLAYING までは常にリセットして開始時にスポーンさせる
            if self.state.logic.runtime.status != GameStatus.PLAYING:
                self.state.physics.reset()

            # 物理更新
            is_stable, metrics = self.state.physics.update(
                det.keypoints, dt_s, frame.shape[0], frame.shape[1]
            )
            self.state.logic.update(is_stable, dt_s)

            # 描画 (セグメンテーション後のフレームに、物理オブジェクトを描画する)
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