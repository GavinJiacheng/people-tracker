from dataclasses import dataclass


@dataclass
class AppConfig:
    input_path: str = "input.mp4"
    output_path: str = "output.mp4"

    yolo_weights: str = "yolov8n.pt"
    reid_weights: str = "osnet_ain_x1_0_msmt17.pt"

    conf: float = 0.4
    imgsz: int = 960
    person_class: int = 0

    strongsort_track_secs: int = 8
    max_obs_extra: int = 10
    n_init: int = 1
    nn_budget: int = 200
    max_cos_dist: float = 0.47
    mc_lambda: float = 0.9
    ema_alpha: float = 0.98
    max_iou_dist: float = 0.75

    color_seed: int = 42
    color_table_size: int = 10000
    box_thickness: int = 2
    font_scale: float = 0.6
    font_thickness: int = 2