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

## Note on StyleGAN ckpts

Avaliable at `outputs/ckpt`:

* [`rosalinity-stylegan2-ffhq-config-f.pt`](https://github.com/rosinality/stylegan2-pytorch#pretrained-checkpoints): from here, suitable ONLY for 256px.
* [`stylegan2-ffhq-config-f.pt`](https://drive.google.com/file/d/1EM87UquaoQmk17Q8d5kYIAHqu0dkYqdT/view?usp=sharing): used in the e4e paper, looks like the converted NVLab ckpt. Suitable for 1MPixel.

## Data

* [FFHQ](): each image has original, estimated latent vector and inversion.

## View all configs

python dip_optimizer.py --cfg job

We use [hydra](https://github.com/facebookresearch/hydra) for configs. YAML files present under conf/.