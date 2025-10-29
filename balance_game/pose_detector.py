from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import DetectionResult, Keypoints2D, Point2D


class PoseDetector:
    def __init__(self):
        self._mp_pose = None
        self._pose = None
        try:
            import mediapipe as mp  # type: ignore

            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            raise ImportError(
                "MediaPipeがインストールされていません。`uv sync` を実行して依存をインストールしてください。"
            ) from e

    def _to_point(self, lm, w: int, h: int) -> Point2D:
        return Point2D(x=float(lm.x * w), y=float(lm.y * h))

    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        h, w = frame_bgr.shape[:2]
        keypoints = Keypoints2D()
        meta = {"method": "mediapipe"}

        if self._pose is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)
            if result.pose_landmarks:
                lm = result.pose_landmarks.landmark
                pm = self._mp_pose.PoseLandmark  # type: ignore

                def safe_get(idx: int) -> Optional[Point2D]:
                    try:
                        p = self._to_point(lm[idx], w, h)
                        if 0 <= p.x < w and 0 <= p.y < h:
                            return p
                    except Exception:
                        return None
                    return None

                keypoints.nose = safe_get(pm.NOSE.value)
                keypoints.left_wrist = safe_get(pm.LEFT_WRIST.value)
                keypoints.right_wrist = safe_get(pm.RIGHT_WRIST.value)
                keypoints.left_shoulder = safe_get(pm.LEFT_SHOULDER.value)
                keypoints.right_shoulder = safe_get(pm.RIGHT_SHOULDER.value)
                return DetectionResult(keypoints=keypoints, metadata=meta)
        return DetectionResult(keypoints=keypoints, metadata=meta)
