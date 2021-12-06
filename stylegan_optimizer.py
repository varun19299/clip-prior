from pathlib import Path

import clip
import hydra
import torch
import wandb
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from loss import CLIPLoss, LossGeocross
from models.stylegan import Generator
from task_init import task_registry
from utils.catch_error import catch_error_decorator
from utils.train_helper import (
    get_optimizer_lr_scheduler,
    train_setup,
    wandb_image,
    setup_wandb,
)
from utils.data import load_latent_or_img


def set_noise_vars(g_ema, stylegan_cfg):
    """
    Setup trainable and non-trainable noise tensors for stylegan

    :param g_ema: stylegan generator
    :param stylegan_cfg: stylegan config
    :return:
        noise_ll: Set of all noise tensors
        noise_var_ll: Set of trainable (requires_grad=True) noise tensors
    """
    # Trainable noise (adds to style)
    num_layers = g_ema.num_layers
    num_trainable_noise_layers = stylegan_cfg.num_trainable_noise_layers
    noise_type = stylegan_cfg.noise_type
    bad_noise_layers = stylegan_cfg.bad_noise_layers

    noise_ll = []
    noise_var_ll = []

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
                noise_var_ll.append(new_noise)
            else:
                new_noise.requires_grad = False
        else:
            raise Exception(f"unknown noise type {noise_type}")

        noise_ll.append(new_noise)

    return noise_ll, noise_var_ll


@catch_error_decorator
@hydra.main(config_name=Path(__file__).stem, config_path="conf")
def main(cfg: DictConfig):
    device = train_setup(cfg)

    # StyleGANv2
    g_ema = Generator(cfg.stylegan.size, 512, 8)
    logger.info(f"Loading StyleGANv2 ckpt from {cfg.stylegan.ckpt}")
    stylegan_ckpt = torch.load(cfg.stylegan.ckpt, map_location="cpu")["g_ema"]
    g_ema.load_state_dict(stylegan_ckpt, strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    img = load_latent_or_img(stylegan_gen=g_ema, **cfg.img)

    # Random init, from where optimization begins
    random_latent = g_ema.random_latent().clone().detach()
    random_latent = random_latent.unsqueeze(0).repeat(1, 18, 1)
    random_latent.requires_grad = True

    # Setup forward func, modify img
    img, forward_func, metric = task_registry[cfg.task.name](img, cfg.task)

    # wandb
    setup_wandb(cfg, img, forward_func)

    # CLIP model
    clip_loss = CLIPLoss(image_size=cfg.stylegan.size, device=device)

    # Preprocess
    text = clip.tokenize([cfg.img.caption]).to(device)

    img = img.to(device)

    # Noise tensors, including those that require gradients
    noise_ll, noise_var_ll = set_noise_vars(g_ema, cfg.stylegan)
    var_list = [random_latent] + noise_var_ll

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(var_list, cfg.optim)

    # Train
    pbar = tqdm(total=cfg.train.num_steps)
    for step in range(cfg.train.num_steps):
        pbar.update(1)

        if cfg.optim.get("use_spherical", False):
            optim.opt.zero_grad()
        else:
            optim.zero_grad()

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
            "loss_forward": loss_forward.item(),
            "loss_clip": loss_clip.item(),
            "loss_geocross": loss_geocross.item(),
            "loss": loss.item(),
        }
        pbar.set_description(" | ".join([f"{k}:{v:.3f}" for k, v in log_dict.items()]))

        if step % cfg.train.log_steps == 0:
            log_dict.update(
                {
                    "output": wandb_image(out, cfg.img.name),
                    "output_forward": wandb_image(forward_func(out), cfg.img.name),
                }
            )
            wandb.log(log_dict, step=step)


if __name__ == "__main__":
    main()
