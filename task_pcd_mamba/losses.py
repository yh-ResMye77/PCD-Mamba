import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def gaussian(window_size, sigma):
    values = [math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)]
    kernel = torch.tensor(values, dtype=torch.float32)
    return kernel / kernel.sum()


def create_window(window_size, channel):
    window_1d = gaussian(window_size, 1.5).unsqueeze(1)
    window_2d = window_1d.mm(window_1d.t()).float().unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channel, 1, window_size, window_size).contiguous()


class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        if channel == self.channel and self.window.dtype == img1.dtype and self.window.device == img1.device:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(device=img1.device, dtype=img1.dtype)
            self.window = window
            self.channel = channel
        return 1.0 - self._ssim(img1, img2, window, channel)

    def _ssim(self, img1, img2, window, channel):
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        if self.size_average:
            return ssim_map.mean()
        return ssim_map.mean(1).mean(1).mean(1)


def get_frequency_amplitude_and_phase(x):
    with torch.amp.autocast("cuda", enabled=False):
        x_float = x.float()
        fft_x = torch.fft.rfft2(x_float, norm="ortho")
        amplitude = torch.abs(fft_x)
        phase = torch.angle(fft_x)
    return amplitude, phase


class FocalFrequencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha

    def forward(self, pred, target):
        pred_amplitude, _ = get_frequency_amplitude_and_phase(pred)
        target_amplitude, _ = get_frequency_amplitude_and_phase(target)
        pred_amplitude = torch.log1p(pred_amplitude)
        target_amplitude = torch.log1p(target_amplitude)
        diff = torch.abs(pred_amplitude - target_amplitude) ** 2
        weight = (diff / (diff.max() + 1e-8)) ** self.alpha
        return (weight * diff).mean() * self.loss_weight


class PhaseConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        _, pred_phase = get_frequency_amplitude_and_phase(pred)
        _, target_phase = get_frequency_amplitude_and_phase(target)
        return (1.0 - torch.cos(pred_phase - target_phase)).mean() * self.loss_weight


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer("kernel_y", kernel_y.unsqueeze(0).unsqueeze(0))

    def forward(self, pred, target):
        _, channels, _, _ = pred.shape
        kernel_x = self.kernel_x.to(pred.device).expand(channels, 1, 3, 3)
        kernel_y = self.kernel_y.to(pred.device).expand(channels, 1, 3, 3)
        pred_grad_x = F.conv2d(pred, kernel_x, groups=channels, padding=1)
        pred_grad_y = F.conv2d(pred, kernel_y, groups=channels, padding=1)
        target_grad_x = F.conv2d(target, kernel_x, groups=channels, padding=1)
        target_grad_y = F.conv2d(target, kernel_y, groups=channels, padding=1)
        return F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)


class ProgressiveFrequencyGuidedCurriculumLoss(nn.Module):
    """Progressive Frequency-Guided Scheduling Curriculum (PFSC)."""

    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.focal_frequency = FocalFrequencyLoss(loss_weight=0.1)
        self.phase_consistency = PhaseConsistencyLoss(loss_weight=0.1)
        self.gradient = GradientLoss()
        self.ssim = SSIMLoss()

    def forward(self, pred, target, stage_name="refine"):
        if stage_name == "structure":
            return 1.0 * self.l1(pred, target) + 0.5 * self.phase_consistency(pred, target) + 0.2 * self.ssim(pred, target)
        if stage_name == "detail":
            return (
                0.8 * self.l1(pred, target)
                + 1.0 * self.focal_frequency(pred, target)
                + 1.0 * self.gradient(pred, target)
                + 0.1 * self.phase_consistency(pred, target)
            )
        return (
            1.0 * self.l1(pred, target)
            + 0.1 * self.focal_frequency(pred, target)
            + 0.1 * self.phase_consistency(pred, target)
            + 0.1 * self.gradient(pred, target)
            + 0.1 * self.ssim(pred, target)
        )
