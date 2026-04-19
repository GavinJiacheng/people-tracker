import numpy as np
import torch


def get_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def make_color_table(seed: int, size: int = 10000):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, 3), dtype=np.uint8).tolist()