import clip
import torch


class CLIPLoss(torch.nn.Module):
    def __init__(self, image_size, device=torch.device("cpu")):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=image_size // 32)

    def forward(self, image, text):
        image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity


def LossGeocross(latent):
    """
    Uses geodesic distance on sphere to sum pairwise distances of the 18 vectors
    """
    if latent.shape[1] == 1:
        return 0
    else:
        X = latent.view(-1, 1, 18, 512)
        Y = latent.view(-1, 18, 1, 512)
        A = ((X - Y).pow(2).sum(-1) + 1e-9).sqrt()
        B = ((X + Y).pow(2).sum(-1) + 1e-9).sqrt()
        D = 2 * torch.atan2(A, B)
        D = ((D.pow(2) * 512).mean((1, 2)) / 8.0).sum()
        return D