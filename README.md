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

* [FFHQ](https://github.com/NVlabs/ffhq-dataset) | `data/ffhq`: each image has original, estimated latent vector and inversion.
* [Celeba HQ](https://github.com/IIGROUP/Multi-Modal-CelebA-HQ-Dataset) | `data/celeba-hq`: has original, captions, latent (for a few) and inversion (for a few).

## Running metrics

### Check for these folders first...

```
data/celeba-hq
|-- caption (30K txt files)
|-- latents (if you want to set GT to the inversion)
`-- rgb (30K jpg files)
```

### Multi-run with Hydra

```
python stylegan_optimizer.py img=celeba-hq exp_name=stylegan_metrics img.index='range(0,100)' -m
```

For `range` syntax, see [here](https://hydra.cc/docs/advanced/override_grammar/extended/) under section `Range Sweep`.

If you wish to initialize the groundtruth explicitly from a StyleGAN latent file:

```
python stylegan_optimizer.py img=celeba-hq-latent exp_name=stylegan_metrics img.index='range(0,100)' -m
```

### The Sharp bits

* **Choosing between latent and image**

    We do this by looking at the extension of the path in `img.path`.
    If it is a `.pth, .zip`, we load using `torch.load`, treat it as the latent to StyleGANv2.
    Else if it one of `.png, .jpg, .jpeg`, we load the image using OpenCV2.
    See `utils.data.load_latent_or_img`.
    
* **StyleGAN weights**
    
    We use the weights from the [e4e](https://github.com/omertov/encoder4editing) repository, which seem to be the weights ported from NVLabs (tensorflow) to pytorch using [rosinality](https://github.com/rosinality/stylegan2-pytorch)
    The rosinality trained (not ported) weights however, are poor fidelity.
   
* **Caption: string vs text file**
   
    If string present, load it.
    Else load full text file or (if provided) limited number of lines.
    See `utils.data.load_caption`.
    
## View all configs

python dip_optimizer.py --cfg job

We use [hydra](https://github.com/facebookresearch/hydra) for configs. YAML files present under conf/.