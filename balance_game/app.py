from __future__ import annotations

import sys
from enum import Enum, auto

import cv2

from .camera import Camera
from .config import DIFFICULTY_PRESETS, default_game_config
from .game_logic import GameLogic
from .pose_detector import PoseDetector
from .physics import FingerBalancePhysics
from .types import GameStatus
from .utils.timing import FrameTimer
from .ui import draw_hud, draw_title, draw_result, draw_prepare


class Screen(Enum):
    TITLE = auto()
    PREPARE = auto()
    COUNTDOWN = auto()
    PLAYING = auto()
    RESULT = auto()


def main():
    cfg = default_game_config(difficulty="normal", target_fps=30)
    logic = GameLogic(cfg)
    physics = FingerBalancePhysics(cfg.stabilizer)

    cam = Camera(index=1)
    detector = PoseDetector(mirrored=True)
    timer = FrameTimer()

    window_name = "Balance Game"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    screen = Screen.TITLE
    prepare_ok_frames = 0

    while True:
        frame = cam.read()
        if frame is None:
            break
        # インカメラの左右反転を補正（水平反転）
        frame = cv2.flip(frame, 1)

        dt_s = timer.tick()
        det = detector.detect(frame)
        # PLAYING までは常にリセットして開始時にスポーンさせる
        if logic.runtime.status != GameStatus.PLAYING:
            physics.reset()
        # 物理更新（指先は内部で追従）
        is_stable, metrics = physics.update(
            det.keypoints, dt_s, frame.shape[0], frame.shape[1]
        )
        logic.update(is_stable, dt_s)

        # 物理オブジェクトの描画（指先・長方形）
        physics.draw(frame)
        # 人差し指の骨格描画は簡略化のため省略

        # 検出の有無（PREPARE/COUNTDOWN のゲート用）
        has_left_finger = det.keypoints.left_index is not None
        has_right_finger = det.keypoints.right_index is not None
        has_both_fingers = has_left_finger and has_right_finger

        # HUD / タイトル / PREPARE / リザルト描画
        if screen == Screen.TITLE:
            draw_title(frame, difficulty=cfg.difficulty)
        elif screen == Screen.PREPARE:
            draw_prepare(frame, has_head=False, has_finger=has_both_fingers)
        elif screen == Screen.RESULT:
            draw_result(
                frame,
                cleared=(logic.runtime.status == GameStatus.CLEAR),
                elapsed_s=logic.runtime.elapsed_time_s,
                stable_s=logic.runtime.stable_time_s,
            )
        else:
            draw_hud(
                frame,
                status=logic.runtime.status,
                difficulty=cfg.difficulty,
                elapsed_s=logic.runtime.elapsed_time_s,
                stable_s=logic.runtime.stable_time_s,
                fps=timer.fps(),
                tilt_deg=(
                    metrics.object_tilt_deg
                    if hasattr(metrics, "object_tilt_deg")
                    else 0.0
                ),
                head_vx=(metrics.head_vx if hasattr(metrics, "head_vx") else 0.0),
                countdown_s=logic.runtime.countdown_remaining_s,
            )

        # スクリーン遷移（GameStatus と同期）
        if screen == Screen.PREPARE:
            if has_both_fingers:
                prepare_ok_frames += 1
            else:
                prepare_ok_frames = 0
            if prepare_ok_frames >= 9:
                logic.start_countdown(3.0)
                screen = Screen.COUNTDOWN
                prepare_ok_frames = 0
        if screen == Screen.COUNTDOWN:
            if not has_both_fingers:
                # 検出が外れたらカウントダウンを中止し PREPARE に戻す
                logic.reset()
                screen = Screen.PREPARE
                prepare_ok_frames = 0
            elif logic.runtime.status == GameStatus.PLAYING:
                screen = Screen.PLAYING
        if screen == Screen.PLAYING and logic.runtime.status in (
            GameStatus.CLEAR,
            GameStatus.FAIL,
        ):
            screen = Screen.RESULT

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            physics = FingerBalancePhysics(cfg.stabilizer)
            logic = GameLogic(cfg)
            screen = Screen.TITLE
        elif key == ord("s"):
            # 3秒カウントダウン開始
            logic.start_countdown(3.0)
            screen = Screen.COUNTDOWN
        elif key == ord(" "):
            # スペースキー: Titleで開始、Resultでタイトルに戻る
            if screen == Screen.TITLE:
                physics = FingerBalancePhysics(cfg.stabilizer)
                logic = GameLogic(cfg)
                screen = Screen.PREPARE
            elif screen == Screen.RESULT:
                physics = FingerBalancePhysics(cfg.stabilizer)
                logic = GameLogic(cfg)
                screen = Screen.TITLE
        elif key in (ord("1"), ord("2"), ord("3")):
            if key == ord("1"):
                cfg.difficulty = "easy"
            elif key == ord("2"):
                cfg.difficulty = "normal"
            elif key == ord("3"):
                cfg.difficulty = "hard"
            cfg.stabilizer = DIFFICULTY_PRESETS[cfg.difficulty]
            physics = FingerBalancePhysics(cfg.stabilizer)
            logic = GameLogic(cfg)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
