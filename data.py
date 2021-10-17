import cv2
import kornia
import torch
from matplotlib import pyplot as plt


def load_img(
    path: str,
    height: int = 256,
    width: int = 256,
    bits: int = 8,
    plot: bool = False,
    crop_mode: str = "centre-crop",
    save_gt: bool = False,
    **kwargs
) -> torch.Tensor:
    img = cv2.imread(path, -1)[:, :, ::-1] / (2 ** bits - 1)
    img = torch.from_numpy(img.copy()).float().permute(2, 0, 1)

    if crop_mode == "resize-crop":
        # Resize such that shorter side matches corresponding target side
        smaller_side = min(height, width)
        img = kornia.resize(
            img.unsqueeze(0), smaller_side, align_corners=False
        ).squeeze(0)

    img = kornia.center_crop(img.unsqueeze(0), (height, width), align_corners=False)
    img = img.squeeze(0).permute(1, 2, 0)

    if plot:
        plt.imshow(img)
        plt.show()

    if save_gt:
        cv2.imwrite("gt.png", img.numpy()[:, :, ::-1] * 255.0)

    # H x W x 3
    return img
