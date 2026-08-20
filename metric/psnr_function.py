import torch


def calculate_psnr(img1, img2, eps=1e-10):
    img1 = torch.clamp(img1.detach().float(), 0.0, 1.0)
    img2 = torch.clamp(img2.detach().float(), 0.0, 1.0)
    mse = torch.mean((img1 - img2) ** 2)
    if mse <= eps:
        return 100.0
    return float(10.0 * torch.log10(1.0 / mse).detach().cpu().item())
