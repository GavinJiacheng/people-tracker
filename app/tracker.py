from pathlib import Path

from boxmot import StrongSort


class PersonTracker:
    def __init__(
        self,
        reid_weights: str,
        device: str,
        fps: int,
        track_secs: int,
        max_obs_extra: int,
        n_init: int,
        nn_budget: int,
        max_cos_dist: float,
        mc_lambda: float,
        ema_alpha: float,
        max_iou_dist: float,
    ):
        self.tracker = StrongSort(
            reid_weights=Path(reid_weights),
            device=device,
            half=False,
            max_age=int(fps * track_secs),
            max_obs=int(fps * track_secs) + max_obs_extra,
            n_init=n_init,
            nn_budget=nn_budget,
            max_cos_dist=max_cos_dist,
            mc_lambda=mc_lambda,
            ema_alpha=ema_alpha,
            max_iou_dist=max_iou_dist,
        )

    def update(self, detections, frame):
        return self.tracker.update(detections, frame)