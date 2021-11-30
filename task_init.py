"""
Initialization functions for a Task

Eg: Drawing mask, getting downsampling function setup
"""

from functools import partial
from pathlib import Path

import numpy as np
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from roipoly import RoiPoly
from torch.nn import functional as F

from utils.bicubic import BicubicDownSample

task_registry = {}


def _register(func):
    task_registry[func.__name__] = func


@_register
def super_resolution(img, task_cfg, **kwargs):
    downsample_func = eval(task_cfg.get("downsample_func", BicubicDownSample))(
        task_cfg.factor
    )
    metric = eval(task_cfg.get("metric", F.mse_loss))

    forward_func = downsample_func

    return img, forward_func, metric


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

    def _mask_image(image, mask):
        assert image.ndim == 4, "Expected NCHW format"
        assert mask.ndim == 2, "Expected HW format"
        return image[:, :, mask]

    forward_func = partial(_mask_image, mask=mask)

    return img, forward_func, metric
