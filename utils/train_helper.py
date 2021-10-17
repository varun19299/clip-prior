from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
from torch.nn import Parameter


@contextmanager
def _blank_context():
    yield


def get_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device(device_str)
    else:
        return torch.device("cpu")


def get_optimizer_lr_scheduler(
    param: List[Parameter], optim_cfg: Dict, quantize_mode: bool = False
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    optim_dict = {
        "adam": torch.optim.Adam,
    }

    kwargs = {k: v for k, v in optim_cfg.items() if k != "name"}
    optim = optim_dict[optim_cfg.name](param, **kwargs)

    if quantize_mode:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 1000, gamma=0.5)
    else:
        # empirically, step lr (cut by half after 1k steps) should be better
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optim, 2000, gamma=0.5)

    return optim, lr_scheduler
