from pathlib import Path

import cv2


class VideoReader:
    def __init__(self, input_path: str):
        self.input_path = input_path
        self.cap = cv2.VideoCapture(input_path)

        if not self.cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {input_path}")

        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.width <= 0 or self.height <= 0:
            raise ValueError("Invalid video resolution.")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class VideoWriter:
    def __init__(self, output_path: str, fps: int, width: int, height: int):
        self.output_path = output_path
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        self.writer = cv2.VideoWriter(
            str(output_file),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()