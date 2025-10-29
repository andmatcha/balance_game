from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .types import DetectionResult, Keypoints2D, Point2D


class PoseDetector:
    def __init__(self):
        self._mp_pose = None
        self._pose = None
        self._mp_face_mesh = None
        self._face_mesh = None
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
            # FaceMesh（頭頂部/顎先用）
            self._mp_face_mesh = mp.solutions.face_mesh
            self._face_mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
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
                # 続けて FaceMesh を処理（頭頂部/顎先）
                if self._face_mesh is not None:
                    fresult = self._face_mesh.process(rgb)
                    if fresult.multi_face_landmarks:
                        flm = fresult.multi_face_landmarks[0].landmark

                        def to_xy(idx: int) -> Optional[tuple[float, float]]:
                            try:
                                x = float(flm[idx].x * w)
                                y = float(flm[idx].y * h)
                                if 0 <= x < w and 0 <= y < h:
                                    return (x, y)
                            except Exception:
                                return None
                            return None

                        # 顎先(152)
                        chin_xy = to_xy(152)
                        if chin_xy is not None:
                            keypoints.chin = Point2D(*chin_xy)

                        # 頭頂部: 顔メッシュの上端を基準に少し上へオフセットして近似
                        xs = [float(lm.x * w) for lm in flm]
                        ys = [float(lm.y * h) for lm in flm]
                        if xs and ys:
                            min_y = min(ys)
                            max_y = max(ys)
                            min_x = min(xs)
                            max_x = max(xs)
                            face_h = max(1.0, max_y - min_y)
                            offset = 0.15  # 顔高の15%分、上方向へ
                            top_y = max(0.0, min_y - offset * face_h)
                            top10 = to_xy(10)
                            if top10 is not None:
                                top_x = top10[0]
                            else:
                                top_x = sum(xs) / len(xs)
                            keypoints.head_top = Point2D(float(top_x), float(top_y))
                            # 顔バウンディングをmetaへ
                            meta["face_bbox"] = (
                                int(min_x),
                                int(min_y),
                                int(max_x),
                                int(max_y),
                            )
                            meta["face_mesh"] = True
                return DetectionResult(keypoints=keypoints, metadata=meta)
        return DetectionResult(keypoints=keypoints, metadata=meta)
