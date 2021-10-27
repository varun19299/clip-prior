import os
from pathlib import Path

import clip
import hydra
import numpy as np
import torch
import wandb
from einops import rearrange
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from roipoly import RoiPoly
from torch.nn import functional as F
from tqdm import tqdm

from data import load_img
from models import registry
from utils.catch_error import catch_error_decorator
from utils.train_helper import get_optimizer_lr_scheduler, get_device


def preprocess_for_CLIP(image):
    """
    pytorch-based preprocessing for CLIP

    Source: https://github.com/codestella/putting-nerf-on-a-diet/blob/90427f6fd828abb2f0b7966fc82753ff43edb338/nerf/clip_utils.py#L153
    Args:
        image [B, 3, H, W]: batch image
    Return
        image [B, 3, 224, 224]: pre-processed image for CLIP
    """
    device = image.device
    dtype = image.dtype

    mean = torch.tensor(
        [0.48145466, 0.4578275, 0.40821073], device=device, dtype=dtype
    ).reshape(1, 3, 1, 1)
    std = torch.tensor(
        [0.26862954, 0.26130258, 0.27577711], device=device, dtype=dtype
    ).reshape(1, 3, 1, 1)
    image = F.interpolate(
        image, (224, 224), mode="bicubic"
    )  # assume that images have rectangle shape.
    image = (image - mean) / std
    return image


@catch_error_decorator
@hydra.main(config_name="config", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    device = get_device(cfg.device)

    # Image (H x W x 3)
    img = load_img(**cfg.img)

    # CLIP model
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    # Prior model (UNet, stylegan etc)
    prior_model = registry[cfg.model.name](**cfg.model.kwargs).to(device)

    # Preprocess
    text = clip.tokenize([cfg.img.caption]).to(device)

    # Choose RoI
    if not Path(cfg.mask.path).exists():
        plt.imshow(img)
        plt.title("Choose region to mask out")
        my_roi = RoiPoly(color="r")
        my_roi.display_roi()

        # Get mask
        mask = my_roi.get_mask(img[:, :, 0])

        np.save(cfg.mask.path, mask)
    else:
        mask = np.load(cfg.mask.path)

    img[mask] = 0

    # Noise tensor
    img = rearrange(img, "h w c -> 1 c h w").to(device)
    noise_tensor = torch.rand(size=img.shape).to(device)

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(prior_model, cfg.optim)

    # wandb
    if cfg.wandb.use:
        with open(cfg.wandb.api_key) as f:
            os.environ["WANDB_API_KEY"] = f.read()

        wandb.init(
            entity=cfg.wandb.entity,
            config=OmegaConf.to_container(cfg, resolve=True),
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            reinit=True,
            save_code=True,
        )

    # Train
    pbar = tqdm(total=cfg.train.num_steps)
    for step in range(cfg.train.num_steps):
        pbar.update(1)
        optim.zero_grad()

        out = prior_model(noise_tensor)

        # CLIP similarity
        loss_clip = 1 - clip_model(preprocess_for_CLIP(out), text)[0] / 100
        loss_clip *= cfg.loss.clip

        # MSE
        loss_forward = ((img[:, :, ~mask] - out[:, :, ~mask]) ** 2).mean()
        loss_forward *= cfg.loss.forward

        loss = loss_forward + loss_clip
        loss.backward()

        optim.step()

        # pbar
        log_dict = {
            "loss_forward": loss_forward,
            "loss_clip": loss_clip.item(),
            "loss": loss.item(),
        }
        pbar.set_description(" | ".join([f"{k}:{v:.3f}" for k, v in log_dict.items()]))

        if step % cfg.train.log_steps == 0:
            log_dict.update(
                {
                    "input_image": [
                        wandb.Image(
                            img.detach(),
                            caption=cfg.img.name,
                        )
                    ],
                    "output_image": [
                        wandb.Image(
                            out.detach(),
                            caption=cfg.img.name,
                        )
                    ],
                }
            )
            wandb.log(log_dict, step=step)


if __name__ == "__main__":
    main()
