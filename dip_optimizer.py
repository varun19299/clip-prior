import os
from pathlib import Path

import clip
import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from data import load_img
from loss import CLIPLoss
from models import registry
from task_init import task_registry
from utils.catch_error import catch_error_decorator
from utils.train_helper import (
    manual_seed,
    get_optimizer_lr_scheduler,
    get_device,
)


@catch_error_decorator
@hydra.main(config_name=Path(__file__).stem, config_path="conf")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # Manual seeds
    manual_seed(cfg.get("seed"))

    # Set device
    device = get_device(cfg.device)

    # Log more frequently on CPU
    # Assuming you will be debugging then
    if device == torch.device("cpu"):
        cfg.train.log_steps = 5

    # Image (H x W x 3)
    img = load_img(**cfg.img)

    # CLIP model
    clip_loss = CLIPLoss(image_size=cfg.stylegan.size, device=device)

    # Prior model (UNet, stylegan etc)
    prior_model = registry[cfg.model.name](**cfg.model.kwargs).to(device)

    # Preprocess
    text = clip.tokenize([cfg.img.caption]).to(device)

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

    if cfg.wandb.use:
        wandb.log(
            {
                "input_image": [
                    wandb.Image(
                        forward_func(img).detach(),
                        caption=cfg.img.name,
                    )
                ],
            },
            step=0,
        )

    # Noise tensor
    noise_tensor = torch.rand(size=img.shape).to(device)

    # Optimizer
    optim, lr_scheduler = get_optimizer_lr_scheduler(
        prior_model.parameters(), cfg.optim
    )

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
        loss_clip = clip_loss(out, text) * cfg.loss.clip

        # MSE
        loss_forward = metric(forward_func(img), forward_func(out))
        loss_forward *= cfg.loss.forward

        loss = loss_forward + loss_clip
        loss.backward()

        optim.step()
        lr_scheduler.step()

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
                    "output": [
                        wandb.Image(
                            out.detach(),
                            caption=cfg.img.name,
                        )
                    ],
                    "output_forward": [
                        wandb.Image(
                            forward_func(out).detach(),
                            caption=cfg.img.name,
                        )
                    ],
                }
            )
            wandb.log(log_dict, step=step)


if __name__ == "__main__":
    main()
