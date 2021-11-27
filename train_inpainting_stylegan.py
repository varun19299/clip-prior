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
from tqdm import tqdm

from data import load_img
from models.stylegan import Generator
from utils.catch_error import catch_error_decorator
from utils.train_helper import (
    get_optimizer_lr_scheduler,
    get_device,
    CLIPLoss,
    LossGeocross,
)


@catch_error_decorator
@hydra.main(config_name="inpainting_stylegan", config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    torch.manual_seed(cfg.seed)

    # Set device
    device = get_device(cfg.device)

    # StyleGANv2
    g_ema = Generator(cfg.stylegan.size, 512, 8)
    stylegan_ckpt = torch.load(cfg.stylegan.ckpt, map_location="cpu")["g_ema"]
    g_ema.load_state_dict(stylegan_ckpt, strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    ## TODO: Why 4096?
    ## TODO: Mean will ensure same output always
    latent_code_init = g_ema.mean_latent(n_latent=4096).detach().clone().repeat(1, 18, 1)
    random_latent = g_ema.random_latent().clone().detach().repeat(1, 18, 1)

    # Image (H x W x 3)
    if cfg.img.get("from_stylegan"):
        with torch.no_grad():
            img, _ = g_ema(
                [latent_code_init],
                input_is_latent=True,
                randomize_noise=False,
            )
            img = rearrange(img.clone(), "1 c h w -> h w c")

    else:
        img = load_img(**cfg.img)

    # sample_z = torch.randn(1, 512, device=device).detach().clone().repeat(1, 18, 1)
    # sample, _ = g_ema([sample_z], truncation_latent=mean_latent)

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

    # CLIP model
    clip_loss = CLIPLoss(cfg.stylegan, device=device)

    # Preprocess
    text = clip.tokenize([cfg.img.caption]).to(device)

    img[mask] = 0

    # Noise tensor
    img = rearrange(img, "h w c -> 1 c h w").to(device)

    # Latent init
    random_latent.requires_grad = True

    # Trainable noise (adds to style)
    num_layers = g_ema.num_layers
    num_trainable_noise_layers = cfg.stylegan.num_trainable_noise_layers
    noise_type = cfg.stylegan.noise_type
    bad_noise_layers = cfg.stylegan.bad_noise_layers

    noise = []
    noise_vars = []

    for i in range(18):
        # dimension of the ith noise tensor
        res = (1, 1, 2 ** (i // 2 + 2), 2 ** (i // 2 + 2))

        if (noise_type == "zero") or (i in bad_noise_layers):
            new_noise = torch.zeros(res, dtype=torch.float, device=device)
            new_noise.requires_grad = False
        elif noise_type == "fixed":
            new_noise = torch.randn(res, dtype=torch.float, device=device)
            new_noise.requires_grad = False
        elif noise_type == "trainable":
            new_noise = torch.randn(res, dtype=torch.float, device=device)
            if i < num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False
        else:
            raise Exception(f"unknown noise type {noise_type}")

    var_list = [random_latent] + noise_vars

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(var_list, cfg.optim)

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
        optim.opt.zero_grad()

        out, _ = g_ema(
            [random_latent], input_is_latent=True, noise_tensor_ll=noise_vars
        )

        # CLIP similarity
        loss_clip = clip_loss(out, text) * cfg.loss.clip

        # MSE
        loss_forward = ((img[:, :, ~mask] - out[:, :, ~mask]) ** 2).mean()
        loss_forward *= cfg.loss.forward

        # Geocross
        loss_geocross = LossGeocross(random_latent) * cfg.loss.geocross

        loss = loss_forward + loss_clip + loss_geocross
        loss.backward()

        optim.step()
        lr_scheduler.step()

        # pbar
        log_dict = {
            "loss_forward": loss_forward,
            "loss_clip": loss_clip.item(),
            "loss_geocross": loss_geocross.item(),
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
