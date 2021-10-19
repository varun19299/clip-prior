# README

## Install

Pre-requisities:
* conda

`make install.cpu` or `make install.gpu` as required.
Environment `clip` will be created.

## Makefile

`make help` to list all available commands.

## W&B Configuration

Copy your WandB API key to wandb_api.key. Will be used to login to your dashboard for visualisation. Alternatively, you can skip W&B visualisation, and set wandb.use=False while running the python code or USE_WANDB=False while running make commands.

## View all configs

python train_inpainting.py --cfg job

We use [hydra](https://github.com/facebookresearch/hydra) for configs. YAML files present under conf/.