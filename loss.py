import clip
import torch
from torch.nn import functional as F

forward_loss_registry = {}


def _register(func):
    forward_loss_registry[func.__name__] = func


class CLIPLoss(torch.nn.Module):
    def __init__(self, stylegan_cfg, device=torch.device("cpu")):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=stylegan_cfg.size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


@_register
def inpainting(src, dest, mask, metric=F.mse_loss):
    """
    Inpainting loss function

    :param src: NCHW format
    :param dest: NCHW format
    :param mask: 1 corresponds to masked out (regions to ignore)
    :param metric: distance metric, eg: L2/ L1
    :return:
    """
    assert src.ndim == dest.ndim == 4, "Expected NCHW format"
    assert mask.ndim == 2, "Expected HW format"
    return metric(src[:, :, ~mask], dest[:, :, ~mask])


@_register
def super_resolution(src, dest, downsample_func, metric=F.mse_loss):
    """
    Super-resolution loss func

    :param src: NCHW format
    :param dest: NCHW format
    :param downsample_func: Downsampling func, includes the kernel too (eg: bicubic)
    :param metric: distance metric, eg: L2/ L1
    :return:
    """
    return metric(downsample_func(src), downsample_func(dest))
