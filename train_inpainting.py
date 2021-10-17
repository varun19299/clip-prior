import numpy as np
import torch
from torch import nn
from PIL import Image
import clip
from einops import rearrange
from tqdm import tqdm
import wandb
import os
from roipoly import RoiPoly
from matplotlib import pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.catch_error import catch_error_decorator
from utils.train_helper import get_optimizer_lr_scheduler, get_device
from data import load_img


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
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Preprocess
    img = preprocess(Image.fromarray(np.uint8(img * 255))).unsqueeze(0).to(device)
    text = clip.tokenize([cfg.img.caption]).to(device)

    # Noise tensor
    out = nn.Parameter(torch.rand(size=img.shape), requires_grad=True)

    # Choose RoI
    plt.imshow(rearrange(img, "1 c h w -> h w c"))
    plt.title("Choose region to mask out")
    my_roi = RoiPoly(color="r")
    my_roi.display_roi()

    mask = my_roi.get_mask(img[0, 0, :, :])
    img[:, :, mask] = 0

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler([out], cfg.optim)

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

        img_features = model.encode_image(img)

        out_features = model.encode_image(out)
        text_features = model.encode_text(text)

        # Cosine sim
        loss_clip = (
            1
            - (out_features @ text_features.T)
            / (out_features.norm() * text_features.norm()).squeeze()
        )
        loss_clip *= cfg.loss.clip

        # MSE
        loss_forward = ((img[:, :, ~mask] - out[:, :, ~mask]) ** 2).mean()
        loss_forward *= cfg.loss.forward

        # Feature space
        loss_feature = ((out_features - img_features) ** 2).mean()
        loss_feature *= cfg.loss.feature

        loss = loss_forward + loss_feature + loss_clip
        loss.backward()

        optim.step()

        # pbar
        log_dict = {
            "loss_forward": loss_forward,
            "loss_feature": loss_feature.item(),
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
