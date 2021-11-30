"""
Initialization functions for a Task

Eg: Drawing mask, getting downsampling function setup
"""

from utils.bicubic import BicubicDownSample
from functools import partial
from loss import forward_loss_registry
from torch.nn import functional as F
from loguru import logger
from matplotlib import pyplot as plt
from pathlib import Path
from roipoly import RoiPoly
from einops import rearrange
import numpy as np

task_registry = {}


def _register(func):
    task_registry[func.__name__] = func


@_register
def super_resolution(img, task_cfg, **kwargs):
    downsample_func = eval(task_cfg.get("downsample_func", BicubicDownSample))(
        task_cfg.factor
    )
    metric = eval(task_cfg.get("metric", F.mse_loss))

    forward_func = partial(
        forward_loss_registry[task_cfg.name],
        downsample_func=downsample_func,
        metric=metric,
    )
    return img, forward_func


@_register
def inpainting(img, task_cfg, **kwargs):
    # Rearrange, normalize to 0...1
    img_draw = rearrange(img.clone(), "1 c h w -> h w c")
    img_draw = (img_draw - img_draw.min()) / (img_draw.max() - img_draw.min())

    # Choose RoI
    if not Path(task_cfg.mask.path).exists():
        logger.info(f"No mask found at {task_cfg.mask.path}")

        plt.imshow(img_draw)
        plt.title("Choose region to mask out")
        my_roi = RoiPoly(color="r")
        my_roi.display_roi()

        # Get mask
        mask = my_roi.get_mask(img_draw[:, :, 0])

        np.save(task_cfg.mask.path, mask)
    else:
        logger.info(f"Loaded mask from {task_cfg.mask.path}")
        mask = np.load(task_cfg.mask.path)

    # Mask out
    img[:, :, mask] = 0

    # Forward func
    metric = eval(task_cfg.get("metric", F.mse_loss))

    forward_func = partial(
        forward_loss_registry[task_cfg.name], mask=mask, metric=metric
    )

    return img, forward_func
