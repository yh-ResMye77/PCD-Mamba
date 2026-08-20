import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class AdaptiveDualPriorPerceiver(nn.Module):
    """Adaptive Dual-Prior Perceiver (ADPP)."""

    def __init__(self, in_channels=3, prior_channels=32):
        super().__init__()
        self.attenuation_estimator = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.color_prior_embedding = nn.Sequential(
            nn.Conv2d(in_channels + 1, prior_channels, kernel_size=1),
            nn.SiLU(),
        )

    def extract_dark_channel(self, x, window_size=15):
        attenuation = self.attenuation_estimator(x)
        weighted = x * attenuation
        min_pool = -F.max_pool2d(-weighted, kernel_size=window_size, stride=1, padding=window_size // 2)
        return torch.min(min_pool, dim=1, keepdim=True)[0]

    def extract_structural_mask(self, x):
        x_fp32 = x.float()
        if x_fp32.shape[1] == 3:
            gray = 0.299 * x_fp32[:, 0:1] + 0.587 * x_fp32[:, 1:2] + 0.114 * x_fp32[:, 2:3]
        else:
            gray = x_fp32

        fft_x = torch.fft.rfft2(gray, norm="ortho")
        phase = torch.angle(fft_x)
        phase_only = torch.fft.irfft2(torch.exp(1j * phase), s=gray.shape[-2:], norm="ortho")
        response = torch.abs(phase_only)

        batch, channel, _, _ = response.shape
        response_flat = response.view(batch, channel, -1)
        response_min = response_flat.min(dim=-1, keepdim=True)[0].unsqueeze(-1)
        response_max = response_flat.max(dim=-1, keepdim=True)[0].unsqueeze(-1)
        structural_mask = (response - response_min) / (response_max - response_min + 1e-8)
        return structural_mask.to(dtype=x.dtype)

    def forward(self, x):
        dark_channel = self.extract_dark_channel(x)
        structural_mask = self.extract_structural_mask(x)
        color_prior = self.color_prior_embedding(torch.cat([x, dark_channel * structural_mask], dim=1))
        return color_prior, structural_mask


class PhaseAlignedDiscretizationModulation(nn.Module):
    """Phase-Aligned Discretization Modulation (PADM)."""

    def __init__(self, dim, prior_channels=32):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.prior_projection = nn.Linear(prior_channels, dim)
        self.scale_head = nn.Linear(dim, dim)
        self.shift_head = nn.Linear(dim, dim)

    def forward(self, feature_map, color_prior):
        batch, height, width, channels = feature_map.shape
        feature_sequence = feature_map.view(batch, -1, channels)
        resized_prior = F.interpolate(color_prior, size=(height, width), mode="bilinear", align_corners=False)
        prior_sequence = resized_prior.permute(0, 2, 3, 1).contiguous().view(batch, -1, resized_prior.shape[1])
        prior_embedding = self.prior_projection(prior_sequence)

        normalized_features = self.norm(feature_sequence)
        scale = torch.sigmoid(self.scale_head(prior_embedding))
        shift = self.shift_head(prior_embedding)
        modulated_features = normalized_features * (1.0 + scale) + shift
        return feature_sequence, modulated_features


class PhaseGuidedMambaBlock(nn.Module):
    """Phase-Guided Mamba Block (PGMB)."""

    def __init__(self, dim, d_state=16, prior_channels=32):
        super().__init__()
        self.padm = PhaseAlignedDiscretizationModulation(dim=dim, prior_channels=prior_channels)
        self.mamba = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)

    def forward(self, feature_map, color_prior):
        batch, height, width, channels = feature_map.shape
        residual, modulated = self.padm(feature_map, color_prior)
        propagated = self.mamba(modulated)
        return (propagated + residual).view(batch, height, width, channels)


class PhaseMaskedSpatialCompensation(nn.Module):
    """Phase-Masked Spatial Compensation (PMSC)."""

    def __init__(self, dim):
        super().__init__()
        self.residual_projection = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1)
        self.gate = nn.Sequential(nn.Conv2d(dim * 2, 1, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def extract_high_frequency_residual(x):
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
            dtype=torch.float32,
            device=x.device,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(x.shape[1], 1, 1, 1)
        return F.conv2d(x, kernel, padding=1, groups=x.shape[1])

    def forward(self, decoder_feature, encoder_feature, structural_mask):
        high_frequency = self.extract_high_frequency_residual(encoder_feature)
        high_frequency = self.residual_projection(high_frequency)
        mask = F.interpolate(structural_mask, size=high_frequency.shape[2:], mode="bilinear", align_corners=False)
        high_frequency = high_frequency * mask
        compensation_gate = self.gate(torch.cat([decoder_feature, high_frequency], dim=1))
        return decoder_feature + compensation_gate * high_frequency


class PCDMamba(nn.Module):
    """Phase-Conditioned Decoupled Mamba for underwater image enhancement."""

    def __init__(self, in_channels=3, out_channels=3, dim=64):
        super().__init__()
        self.adpp = AdaptiveDualPriorPerceiver(in_channels=in_channels, prior_channels=32)

        self.stem = nn.Conv2d(in_channels, dim, kernel_size=3, stride=1, padding=1)
        self.encoder_level1 = PhaseGuidedMambaBlock(dim)
        self.downsample_level1 = nn.Conv2d(dim, dim * 2, kernel_size=2, stride=2)
        self.encoder_level2 = PhaseGuidedMambaBlock(dim * 2)
        self.downsample_level2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=2, stride=2)
        self.encoder_level3 = PhaseGuidedMambaBlock(dim * 4)
        self.bottleneck = PhaseGuidedMambaBlock(dim * 4)

        self.upsample_level3 = nn.PixelShuffle(2)
        self.reduce_level3 = nn.Conv2d(dim, dim * 2, kernel_size=1)
        self.decoder_level2 = PhaseGuidedMambaBlock(dim * 2)
        self.upsample_level2 = nn.PixelShuffle(2)
        self.reduce_level2 = nn.Conv2d(dim // 2, dim, kernel_size=1)
        self.decoder_level1 = PhaseGuidedMambaBlock(dim)

        self.pmsc_level2 = PhaseMaskedSpatialCompensation(dim * 2)
        self.pmsc_level1 = PhaseMaskedSpatialCompensation(dim)
        self.reconstruction_head = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        color_prior, structural_mask = self.adpp(x)

        level0 = self.stem(x)
        level1 = self.encoder_level1(level0.permute(0, 2, 3, 1), color_prior).permute(0, 3, 1, 2)
        level2_input = self.downsample_level1(level1)
        level2 = self.encoder_level2(level2_input.permute(0, 2, 3, 1), color_prior).permute(0, 3, 1, 2)
        level3_input = self.downsample_level2(level2)
        level3 = self.encoder_level3(level3_input.permute(0, 2, 3, 1), color_prior).permute(0, 3, 1, 2)
        bottleneck = self.bottleneck(level3.permute(0, 2, 3, 1), color_prior).permute(0, 3, 1, 2)

        up_level2 = self.reduce_level3(self.upsample_level3(bottleneck))
        fused_level2 = self.pmsc_level2(up_level2, level2, structural_mask)
        decoded_level2 = self.decoder_level2(fused_level2.permute(0, 2, 3, 1), color_prior).permute(0, 3, 1, 2)

        up_level1 = self.reduce_level2(self.upsample_level2(decoded_level2))
        fused_level1 = self.pmsc_level1(up_level1, level1, structural_mask)
        decoded_level1 = self.decoder_level1(fused_level1.permute(0, 2, 3, 1), color_prior).permute(0, 3, 1, 2)

        residual = self.reconstruction_head(decoded_level1)
        return residual + x
