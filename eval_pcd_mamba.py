import argparse
import csv
import os
import sys

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader
from tqdm import tqdm


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import PairedUnderwaterDataset
from methods.pcd_mamba import PCDMamba
from metric.psnr_function import calculate_psnr
from metric.ssim_function import calculate_ssim
from metric.underwater_metrics import calculate_msssim, calculate_uiqm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate PCD-Mamba.")
    parser.add_argument("--private_dir", type=str, required=True, help="Dataset root containing val/ or test/.")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test", "train"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results/PCD_Mamba_eval")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def load_checkpoint(model, checkpoint_path, device, strict=False):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    result = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        print(f"[Load] missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}")


def main():
    args = parse_args()
    args.private_dir = resolve_path(args.private_dir)
    args.checkpoint = resolve_path(args.checkpoint)
    args.results_dir = resolve_path(args.results_dir)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    model = PCDMamba().to(device)
    load_checkpoint(model, args.checkpoint, device, strict=args.strict)
    model.eval()

    dataset = PairedUnderwaterDataset(os.path.join(args.private_dir, args.split), args.image_size, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics_dir = os.path.join(args.results_dir, "metrics")
    cat_dir = os.path.join(args.results_dir, "cat_images")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(cat_dir, exist_ok=True)

    sums = {"psnr": 0.0, "ssim": 0.0, "ms_ssim": 0.0, "uiqm": 0.0}
    count = 0

    with torch.no_grad():
        for degraded, target, names in tqdm(loader, desc=f"Eval {args.split}", ncols=120):
            degraded = degraded.to(device)
            target = target.to(device)
            restored = torch.clamp(model(degraded), 0.0, 1.0)

            sums["psnr"] += float(calculate_psnr(restored, target))
            sums["ssim"] += float(calculate_ssim(restored, target))
            sums["ms_ssim"] += float(calculate_msssim(restored, target))
            sums["uiqm"] += float(calculate_uiqm(restored))
            count += 1

            base = os.path.splitext(os.path.basename(names[0]))[0] + ".png"
            vutils.save_image(torch.cat([degraded, restored, target], dim=3), os.path.join(cat_dir, base))

    avg = {key: value / max(count, 1) for key, value in sums.items()}
    print(
        f"[Eval {args.split}] PSNR:{avg['psnr']:.4f} SSIM:{avg['ssim']:.4f} "
        f"MS-SSIM:{avg['ms_ssim']:.4f} UIQM:{avg['uiqm']:.4f}"
    )

    csv_path = os.path.join(metrics_dir, "eval_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "checkpoint", "psnr", "ssim", "ms_ssim", "uiqm"])
        writer.writerow([args.split, args.checkpoint, avg["psnr"], avg["ssim"], avg["ms_ssim"], avg["uiqm"]])
    print(f"[Eval] saved metrics to {csv_path}")


if __name__ == "__main__":
    main()
