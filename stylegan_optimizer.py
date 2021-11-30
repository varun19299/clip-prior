import os
from pathlib import Path

import clip
import cv2
import hydra
import torch
import wandb
from einops import rearrange
from loguru import logger
from matplotlib import pyplot as plt
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from loss import CLIPLoss
from models.stylegan import Generator
from task_init import task_registry
from utils.catch_error import catch_error_decorator
from utils.train_helper import (
    get_optimizer_lr_scheduler,
    get_device,
    LossGeocross,
)


@catch_error_decorator
@hydra.main(config_name=Path(__file__).stem, config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    if cfg.get("seed"):
        torch.manual_seed(cfg.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)
            torch.backends.cudnn.deterministic = True

    # Set device
    device = get_device(cfg.device)

    # StyleGANv2
    g_ema = Generator(cfg.stylegan.size, 512, 8)
    logger.info(f"Loading StyleGANv2 ckpt from {cfg.stylegan.ckpt}")
    stylegan_ckpt = torch.load(cfg.stylegan.ckpt, map_location="cpu")["g_ema"]
    g_ema.load_state_dict(stylegan_ckpt, strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    # Pick mean latent if no image supplied
    ## TODO: Why 4096?
    latent_path = Path(cfg.img.get("latent_path", "mean_latent.pt"))
    if not latent_path.exists():
        logger.info(f"No latent file at {latent_path}")
        latent_init = g_ema.mean_latent(n_latent=4096).detach().clone().repeat(1, 18, 1)
    else:
        logger.info(f"Found latent file at {latent_path}")
        latent_init = torch.load(cfg.img.latent_path).unsqueeze(0).repeat(1, 18, 1)

    # Random init, from where optimization begins
    random_latent = g_ema.random_latent().clone().detach().repeat(1, 18, 1)
    random_latent.requires_grad = True

    # Generate img from StyleGAN
    with torch.no_grad():
        img, _ = g_ema(
            [latent_init],
            input_is_latent=True,
            randomize_noise=False,
        )

        img_plot = rearrange(img.numpy(), "1 c h w -> h w c")
        img_plot = (img_plot - img_plot.min()) / (img_plot.max() - img_plot.min())

        # Plot image
        if cfg.plot:
            logger.info("Plotting Groundtruth")
            plt.imshow(img_plot)
            plt.show()

        # Save groundtruth
        if cfg.save_gt:
            logger.info("Saving Groundtruth")
            cv2.imwrite("groundtruth.png", img_plot[:, :, ::-1] * 255)

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

        wandb.log(
            {
                "groundtruth": [
                    wandb.Image(
                        img.detach(),
                        caption=cfg.img.name,
                    )
                ],
            },
            step=0,
        )

    # Setup forward func, modify img
    img, forward_func, metric = task_registry[cfg.task.name](img, cfg.task)

    # CLIP model
    clip_loss = CLIPLoss(cfg.stylegan, device=device)

    # Preprocess
    text = clip.tokenize([cfg.img.caption]).to(device)

    img = img.to(device)

    # Trainable noise (adds to style)
    num_layers = g_ema.num_layers
    num_trainable_noise_layers = cfg.stylegan.num_trainable_noise_layers
    noise_type = cfg.stylegan.noise_type
    bad_noise_layers = cfg.stylegan.bad_noise_layers

    noise_ll = []
    noise_vars = []

    for i in range(num_layers):
        noise_tensor = getattr(g_ema.noises, f"noise_{i}")

        if (noise_type == "zero") or (i in bad_noise_layers):
            new_noise = torch.zeros_like(noise_tensor)
            new_noise.requires_grad = False
        elif noise_type == "fixed":
            new_noise = noise_tensor
        elif noise_type == "trainable":
            new_noise = noise_tensor.detach().clone()
            if i < num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False
        else:
            raise Exception(f"unknown noise type {noise_type}")

        noise_ll.append(new_noise)

    var_list = [random_latent] + noise_vars
    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(var_list, cfg.optim)

    # Train
    pbar = tqdm(total=cfg.train.num_steps)
    for step in range(cfg.train.num_steps):
        pbar.update(1)
        optim.opt.zero_grad()

        out, _ = g_ema(
            [random_latent],
            input_is_latent=True,
            noise_tensor_ll=noise_ll,
            randomize_noise=False,
        )

        # CLIP similarity
        loss_clip = clip_loss(out, text) * cfg.loss.clip

        # MSE
        loss_forward = metric(forward_func(img), forward_func(out))
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
                            forward_func(img).detach(),
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
