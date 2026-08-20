from math import exp

import torch
import torch.nn.functional as F


def gaussian(window_size, sigma):
    values = [exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel).to(device=img1.device, dtype=img1.dtype)
    return float(_ssim(img1, img2, window, window_size, channel, size_average).detach().cpu().item())
