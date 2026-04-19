import argparse

from app.config import AppConfig
from app.detector import PersonDetector
from app.tracker import PersonTracker
from app.utils import get_device, make_color_table
from app.video_io import VideoReader, VideoWriter
from app.visualizer import TrackVisualizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="input.mp4", help="input path")
    parser.add_argument("--output", type=str, default="output.mp4", help="output path")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO weights path")
    parser.add_argument("--reid", type=str, default="osnet_ain_x1_0_msmt17.pt", help="ReID weights path")
    parser.add_argument("--conf", type=float, default=0.4, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference image size")
    return parser.parse_args()


def build_config(args) -> AppConfig:
    return AppConfig(
        input_path=args.input,
        output_path=args.output,
        yolo_weights=args.model,
        reid_weights=args.reid,
        conf=args.conf,
        imgsz=args.imgsz,
    )


def main():
    print("Program Start")
    args = parse_args()
    cfg = build_config(args)

    device = get_device()
    print(f"Using device: {device}")

    video_reader = VideoReader(cfg.input_path)
    video_writer = VideoWriter(
        cfg.output_path,
        video_reader.fps,
        video_reader.width,
        video_reader.height,
    )

    detector = PersonDetector(
        weights=cfg.yolo_weights,
        conf=cfg.conf,
        imgsz=cfg.imgsz,
        person_class=cfg.person_class,
    )

    print("Loading OSNet ReID...")
    tracker = PersonTracker(
        reid_weights=cfg.reid_weights,
        device=device,
        fps=video_reader.fps,
        track_secs=cfg.strongsort_track_secs,
        max_obs_extra=cfg.max_obs_extra,
        n_init=cfg.n_init,
        nn_budget=cfg.nn_budget,
        max_cos_dist=cfg.max_cos_dist,
        mc_lambda=cfg.mc_lambda,
        ema_alpha=cfg.ema_alpha,
        max_iou_dist=cfg.max_iou_dist,
    )

    colors = make_color_table(cfg.color_seed, cfg.color_table_size)
    visualizer = TrackVisualizer(
        colors=colors,
        box_thickness=cfg.box_thickness,
        font_scale=cfg.font_scale,
        font_thickness=cfg.font_thickness,
    )

    print("Starting ReID...")
    frame_count = 0

    try:
        while True:
            ret, frame = video_reader.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            frame = visualizer.draw_tracks(frame, tracks)

            video_writer.write(frame)
            frame_count += 1

    finally:
        video_reader.release()
        video_writer.release()

    print(f"Done! Processed {frame_count} frames.")


if __name__ == "__main__":
    main()