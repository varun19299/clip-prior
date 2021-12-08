from typing import Union

import cv2
from kornia.geometry import resize, center_crop
import torch
from matplotlib import pyplot as plt
from pathlib import Path

from omegaconf import DictConfig

from models.stylegan import Generator

from loguru import logger

from einops import rearrange


def load_img(
    path: str,
    height: int = 256,
    width: int = 256,
    bits: int = 8,
    plot: bool = False,
    crop_mode: str = "centre-crop",
    save_gt: bool = False,
    **kwargs,
) -> torch.Tensor:
    img = cv2.imread(str(path), -1)[:, :, ::-1] / (2 ** bits - 1)
    img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)

    if crop_mode == "resize-crop":
        # Resize such that shorter side matches corresponding target side
        smaller_side = min(height, width)
        img = resize(img.unsqueeze(0), smaller_side, align_corners=False).squeeze(0)

    img = center_crop(img.unsqueeze(0), (height, width), align_corners=False)
    img = img.squeeze(0).permute(1, 2, 0)

    if plot:
        plt.imshow(img)
        plt.show()

    if save_gt:
        cv2.imwrite("gt.png", img.numpy()[:, :, ::-1] * 255.0)

    # H x W x 3 to NHWC
    img = rearrange(img, "h w c -> 1 c h w")
    return img


def load_latent(
    stylegan_gen: Generator,
    path: str,
    plot: bool = False,
    save_gt: bool = False,
    **kwargs,
):

    # Pick mean latent if no image supplied
    if not path.exists():
        logger.info(f"No latent file at {path}. Loading mean latent instead.")
        latent = stylegan_gen.mean_latent(n_latent=4096).detach().clone()
    else:
        logger.info(f"Found latent file at {path}")
        latent = torch.load(path, map_location="cpu")

    # Either d or 18 x d needed.
    if latent.ndim == 2:
        assert latent.shape[0] == 18, f"Required 18 x d, found {latent.shape}"
        latent = latent.unsqueeze(0)
    elif latent.ndim == 1:
        latent = latent.unsqueeze(0).repeat(1, 18, 1)
    else:
        raise AssertionError

    # Generate img from StyleGAN
    with torch.no_grad():
        img, _ = stylegan_gen(
            [latent],
            input_is_latent=True,
            randomize_noise=False,
        )

    if plot or save_gt:
        img_draw = rearrange(img.cpu().clone(), "1 c h w -> h w c")
        img_draw = (img_draw - img_draw.min()) / (img_draw.max() - img_draw.min())

    if plot:
        plt.imshow(img_draw)
        plt.show()

    if save_gt:
        cv2.imwrite("gt.png", img_draw.numpy()[:, :, ::-1] * 255.0)

    return img


def load_latent_or_img(path, **kwargs):
    """
    Load either image (.png, .jpg, .jpeg) or latent (.zip, .pt, .pth) based on type

    :param path: string or Path object
    :return: image as NCHW
    """
    path = Path(path)

    if path.suffix in [".pt", ".zip", ".pth"] or path.name == "mean_latent":
        img = load_latent(path=path, **kwargs)

    elif path.suffix in [".png", ".jpg", ".jpeg"]:
        img = load_img(path=path, **kwargs)

    return img


def load_caption(caption_cfg: Union[str, DictConfig]) -> str:
    """
    Load captions from caption_cfg
    :param caption_cfg: DictConfig
        (in which case we look for a file to load from)
         or the caption itself
    :return: image caption
    """
    if isinstance(caption_cfg, str):
        return caption_cfg
    elif isinstance(caption_cfg, DictConfig):
        if caption_cfg.get("path"):
            with open(caption_cfg.path) as f:
                lines = f.readlines()[: caption_cfg.get("lines", None)]

            logger.info(f"Loaded {len(lines)} lines from {caption_cfg.path}")

            return " ".join(lines)
    else:
        raise AssertionError