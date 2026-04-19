import cv2
import numpy as np


class TrackVisualizer:
    def __init__(
        self,
        colors,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 2,
    ):
        self.colors = colors
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def draw_tracks(self, frame: np.ndarray, tracks) -> np.ndarray:
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, ind = track
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])

            color = self.colors[track_id % len(self.colors)]

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                self.box_thickness,
            )

            cv2.putText(
                frame,
                f"ID:{track_id}",
                (x1, max(0, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                color,
                self.font_thickness,
            )

        return frame