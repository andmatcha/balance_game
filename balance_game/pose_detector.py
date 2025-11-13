from __future__ import annotations

from typing import Optional
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    import numpy as np

from .types import DetectionResult, Keypoints2D, Point2D


class PoseDetector:
    # MediaPipe の各検出器（Pose/FaceMesh/Hands）を初期化
    def __init__(self, mirrored: bool = False):
        self._mp_pose = None
        self._pose = None
        self._mp_face_mesh = None
        self._face_mesh = None
        self._mp_hands = None
        self._hands = None
        self._mirrored = mirrored
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
            # Hands（人差し指各関節の高精度取得）
            self._mp_hands = mp.solutions.hands
            self._hands = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            raise ImportError(
                "MediaPipeがインストールされていません。`uv sync` を実行して依存をインストールしてください。"
            ) from e

    # MediaPipe の正規化座標(lm.x,lm.y)を画像のピクセル座標に変換し、画面内にある場合のみ Point2D を返す。
    def _to_point(self, lm, w: int, h: int) -> Optional[Point2D]:
        x = float(lm.x * w)
        y = float(lm.y * h)
        if 0 <= x < w and 0 <= y < h:
            return Point2D(x=x, y=y)
        return None

    # 1フレームから必要なキーポイントを検出して返す。
    # Pose: 鼻のみ
    # FaceMesh: 顎先、頭頂部（および顔バウンディング）
    # Hands: 人差し指 TIP:指先, DIP:第一関節, PIP:第二関節, MCP:第三関節（左右）
    def detect(self, frame_bgr: np.ndarray) -> DetectionResult:
        h, w = frame_bgr.shape[:2]
        keypoints = Keypoints2D()
        meta = {"method": "mediapipe"}

        if self._pose is None:
            return DetectionResult(keypoints=keypoints, metadata=meta)

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Pose: 鼻のみ
        result = self._pose.process(rgb)
        if result and result.pose_landmarks:
            lms = result.pose_landmarks.landmark
            pm = self._mp_pose.PoseLandmark  # type: ignore
            keypoints.nose = self._to_point(lms[pm.NOSE.value], w, h)

        # FaceMesh: 顎先と頭頂部、顔バウンディング
        if self._face_mesh is not None:
            fresult = self._face_mesh.process(rgb)
            if fresult and fresult.multi_face_landmarks:
                flm = fresult.multi_face_landmarks[0].landmark
                chin = self._to_point(flm[152], w, h)
                if chin is not None:
                    keypoints.chin = chin
                xs = [float(p.x * w) for p in flm]
                ys = [float(p.y * h) for p in flm]
                if xs and ys:
                    min_y, max_y = min(ys), max(ys)
                    min_x, max_x = min(xs), max(xs)
                    face_h = max(1.0, max_y - min_y)
                    top_y = max(0.0, min_y - 0.15 * face_h)
                    top10 = self._to_point(flm[10], w, h)
                    top_x = (
                        float(top10.x)
                        if top10 is not None
                        else float(sum(xs) / len(xs))
                    )
                    keypoints.head_top = Point2D(float(top_x), float(top_y))
                    meta["face_bbox"] = (int(min_x), int(min_y), int(max_x), int(max_y))
                    meta["face_mesh"] = True

        # Hands: 人差し指の各関節（左右）
        if self._hands is not None:
            hresult = self._hands.process(rgb)
            if hresult and hresult.multi_hand_landmarks:
                for i, hls in enumerate(hresult.multi_hand_landmarks):
                    label = None
                    if getattr(hresult, "multi_handedness", None) and i < len(
                        hresult.multi_handedness
                    ):
                        classes = hresult.multi_handedness[i].classification
                        if classes:
                            label = classes[0].label.lower()
                    mapped = label
                    if self._mirrored and label in ("left", "right"):
                        mapped = "right" if label == "left" else "left"
                    ids = {"mcp": 5, "pip": 6, "dip": 7, "tip": 8}
                    coords: dict[str, Point2D] = {}
                    for name, idx in ids.items():
                        pt = self._to_point(hls.landmark[idx], w, h)
                        if pt is not None:
                            coords[name] = pt
                    if mapped == "left":
                        if "tip" in coords:
                            keypoints.left_index = coords["tip"]
                        if "mcp" in coords:
                            keypoints.left_index_mcp = coords["mcp"]
                        if "pip" in coords:
                            keypoints.left_index_pip = coords["pip"]
                        if "dip" in coords:
                            keypoints.left_index_dip = coords["dip"]
                    elif mapped == "right":
                        if "tip" in coords:
                            keypoints.right_index = coords["tip"]
                        if "mcp" in coords:
                            keypoints.right_index_mcp = coords["mcp"]
                        if "pip" in coords:
                            keypoints.right_index_pip = coords["pip"]
                        if "dip" in coords:
                            keypoints.right_index_dip = coords["dip"]
                meta["hands"] = True

        return DetectionResult(keypoints=keypoints, metadata=meta)
