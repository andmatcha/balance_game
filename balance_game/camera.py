from __future__ import annotations

import cv2


class Camera:
    def __init__(
        self, index: int = 0, width: int | None = None, height: int | None = None
    ):
        self.index = index
        self.cap = cv2.VideoCapture(self.index)
        if width is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        return frame

    def release(self):
        if self.cap:
            self.cap.release()
