import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gaussian(window_size, sigma):
    values = [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


def _ssim_tensor(img1, img2, window, window_size, channel):
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
    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    return ssim_map.mean(), cs_map.mean()


def calculate_msssim(img1, img2, window_size=11):
    if img1.device != img2.device:
        img2 = img2.to(img1.device)
    _, channel, _, _ = img1.size()
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], device=img1.device, dtype=img1.dtype)
    mcs = []
    current_img1 = img1
    current_img2 = img2
    ssim_val = None
    for index in range(5):
        ssim_val, cs_val = _ssim_tensor(current_img1, current_img2, window, window_size, channel)
        mcs.append(cs_val)
        if index < 4:
            current_img1 = F.avg_pool2d(current_img1, 2)
            current_img2 = F.avg_pool2d(current_img2, 2)
    mcs = torch.stack(mcs)
    mcs[-1] = ssim_val
    return torch.prod(mcs ** weights).item()


SOBEL_KERNEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_KERNEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)


def _uiconm(x, window_size):
    k1 = x.shape[2] // window_size
    k2 = x.shape[1] // window_size
    x = x[:, : k2 * window_size, : k1 * window_size]
    weight = -1.0 / (k1 * k2)
    patches = x.unfold(1, window_size, window_size).unfold(2, window_size, window_size)
    patches = patches.reshape(-1, k2, k1, window_size, window_size)
    min_value = torch.min(torch.min(torch.min(patches, dim=-1).values, dim=-1).values, dim=0).values
    max_value = torch.max(torch.max(torch.max(patches, dim=-1).values, dim=-1).values, dim=0).values
    numerator = max_value - min_value
    denominator = max_value + min_value
    contrast = (numerator / denominator) * torch.log(numerator / denominator)
    contrast = torch.where(
        torch.isnan(contrast) | (denominator == 0.0) | (numerator == 0.0),
        torch.zeros_like(contrast),
        contrast,
    )
    return weight * contrast.sum()


def _trimmed_mean(x, alpha_left=0.1, alpha_right=0.1):
    x, _ = torch.sort(x)
    count = len(x)
    left = math.ceil(alpha_left * count)
    right = math.floor(alpha_right * count)
    return torch.sum(x[left + 1 : count - right]) / (count - left - right)


def _variance(x, mean):
    return torch.sum(torch.pow(x - mean, 2)) / len(x)


def _uicm(x):
    red = x[0].flatten()
    green = x[1].flatten()
    blue = x[2].flatten()
    red_green = red - green
    yellow_blue = ((red + green) / 2) - blue
    mean_rg = _trimmed_mean(red_green)
    mean_yb = _trimmed_mean(yellow_blue)
    variance_rg = _variance(red_green, mean_rg)
    variance_yb = _variance(yellow_blue, mean_yb)
    colorfulness = torch.sqrt(torch.pow(mean_rg, 2) + torch.pow(mean_yb, 2))
    chroma_spread = torch.sqrt(variance_rg + variance_yb)
    return (-0.0268 * colorfulness) + (0.1586 * chroma_spread)


def _eme(x, window_size):
    k1 = x.shape[1] // window_size
    k2 = x.shape[0] // window_size
    x = x[: k2 * window_size, : k1 * window_size]
    patches = x.view(k2, window_size, k1, window_size)
    patches = patches.permute(0, 2, 1, 3).contiguous().view(-1, window_size * window_size)
    max_values, _ = torch.max(patches, dim=1)
    min_values, _ = torch.min(patches, dim=1)
    valid = (min_values != 0) & (max_values != 0)
    ratios = torch.zeros_like(max_values)
    ratios[valid] = torch.log(max_values[valid] / min_values[valid])
    return (2.0 / (k1 * k2)) * ratios.sum()


def _sobel(x):
    kernel_x = SOBEL_KERNEL_X.to(x.device)
    kernel_y = SOBEL_KERNEL_Y.to(x.device)
    inp = x[None, None, :, :] if x.dim() == 2 else x
    dx = F.conv2d(inp, kernel_x, padding=1)
    dy = F.conv2d(inp, kernel_y, padding=1)
    magnitude = torch.hypot(dx, dy)
    max_value = torch.max(magnitude)
    if max_value > 0:
        magnitude *= 255.0 / max_value
    return magnitude.squeeze()


def _uism(x):
    red_edge = _sobel(x[0]) * x[0]
    green_edge = _sobel(x[1]) * x[1]
    blue_edge = _sobel(x[2]) * x[2]
    return 0.299 * _eme(red_edge, 10) + 0.587 * _eme(green_edge, 10) + 0.144 * _eme(blue_edge, 10)


def calculate_uiqm_single(img):
    x = torch.clamp(img.detach(), 0.0, 1.0) * 255.0
    return 0.0282 * _uicm(x) + 0.2953 * _uism(x) + 3.5753 * _uiconm(x, 10)


def calculate_uiqm(img_tensor):
    if img_tensor.dim() == 4:
        total = sum(calculate_uiqm_single(img_tensor[i]) for i in range(img_tensor.shape[0]))
        return (total / img_tensor.shape[0]).item()
    return calculate_uiqm_single(img_tensor).item()
