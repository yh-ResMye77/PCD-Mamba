import argparse
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torchvision.utils as vutils
from torch.cuda.amp import GradScaler, autocast
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
from task_pcd_mamba import ProgressiveFrequencyGuidedCurriculumLoss


def parse_args():
    parser = argparse.ArgumentParser(description="Train PCD-Mamba.")
    parser.add_argument("--private_dir", type=str, required=True, help="Dataset root containing train/ and val/.")
    parser.add_argument("--results_dir", type=str, default="results/PCD_Mamba")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--total_epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_every", type=int, default=50)
    parser.add_argument("--val_every", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    return parser.parse_args()


def resolve_path(path):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(PROJECT_ROOT, path))


def get_pfsc_stage(epoch, total_epochs):
    progress = epoch / max(total_epochs, 1)
    if progress <= 0.25:
        return "structure"
    if progress <= 0.75:
        return "detail"
    return "refine"


def save_loss_plot(loss_history, save_path):
    if not loss_history:
        return
    plt.figure()
    plt.plot(range(1, len(loss_history) + 1), loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.yscale("log")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    if not checkpoint_path:
        return 1, 0.0
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    model.load_state_dict(state_dict, strict=False)
    if isinstance(checkpoint, dict):
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        return int(checkpoint.get("epoch", 0)) + 1, float(checkpoint.get("best_psnr", 0.0))
    return 1, 0.0


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_psnr, args):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_psnr": best_psnr,
            "config": vars(args),
        },
        path,
    )


def main():
    args = parse_args()
    args.private_dir = resolve_path(args.private_dir)
    args.results_dir = resolve_path(args.results_dir)
    args.resume = resolve_path(args.resume) if args.resume else ""

    dirs = {name: os.path.join(args.results_dir, name) for name in [
        "configs", "losses", "metrics", "models", "best_images", "cat_images"
    ]}
    for folder in dirs.values():
        os.makedirs(folder, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    print(f"[PCD-Mamba] device={device}, amp={use_amp}")

    model = PCDMamba().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.total_epochs, eta_min=1e-6)
    criterion = ProgressiveFrequencyGuidedCurriculumLoss().to(device)
    scaler = GradScaler(enabled=use_amp)

    start_epoch, best_psnr = load_checkpoint(model, optimizer, scheduler, args.resume, device)
    train_dataset = PairedUnderwaterDataset(os.path.join(args.private_dir, "train"), args.image_size, augment=True)
    val_dataset = PairedUnderwaterDataset(os.path.join(args.private_dir, "val"), args.image_size, augment=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    metrics_path = os.path.join(dirs["metrics"], "metrics_log.csv")
    if start_epoch == 1:
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write("epoch,psnr,ssim,ms_ssim,uiqm\n")
        with open(os.path.join(dirs["configs"], "config.txt"), "w", encoding="utf-8") as f:
            for key, value in vars(args).items():
                f.write(f"{key}: {value}\n")

    loss_history = []
    for epoch in range(start_epoch, args.total_epochs + 1):
        stage = get_pfsc_stage(epoch, args.total_epochs)
        model.train()
        epoch_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.total_epochs} [{stage}]", ncols=120)
        for degraded, target, _ in loop:
            degraded = degraded.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                restored = model(degraded)
                loss = criterion(restored.float(), target.float(), stage_name=stage)

            if not torch.isfinite(loss):
                print(f"[Warn] skipping non-finite loss at epoch={epoch}")
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.detach().cpu().item())
            loop.set_postfix(loss=epoch_loss / max(loop.n + 1, 1), lr=optimizer.param_groups[0]["lr"])

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        loss_history.append(avg_loss)
        with open(os.path.join(dirs["losses"], "loss.csv"), "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.6f},{optimizer.param_groups[0]['lr']:.8f}\n")

        save_checkpoint(os.path.join(dirs["models"], "latest.pth"), model, optimizer, scheduler, epoch, best_psnr, args)
        if epoch % args.save_every == 0:
            save_checkpoint(os.path.join(dirs["models"], f"epoch_{epoch}.pth"), model, optimizer, scheduler, epoch, best_psnr, args)

        if epoch % args.val_every != 0:
            save_loss_plot(loss_history, os.path.join(dirs["losses"], "loss_curve.png"))
            continue

        for name in os.listdir(dirs["cat_images"]):
            if name.lower().endswith((".png", ".jpg", ".jpeg")):
                os.remove(os.path.join(dirs["cat_images"], name))

        model.eval()
        psnr_sum = ssim_sum = ms_ssim_sum = uiqm_sum = 0.0
        with torch.no_grad():
            for degraded, target, names in tqdm(val_loader, desc=f"Val {epoch}", ncols=120):
                degraded = degraded.to(device)
                target = target.to(device)
                restored = torch.clamp(model(degraded), 0.0, 1.0)

                psnr_sum += calculate_psnr(restored, target)
                ssim_sum += calculate_ssim(restored, target)
                ms_ssim_sum += calculate_msssim(restored, target)
                uiqm_sum += calculate_uiqm(restored)

                base = os.path.splitext(os.path.basename(names[0]))[0] + ".png"
                vutils.save_image(torch.cat([degraded, restored, target], dim=3), os.path.join(dirs["cat_images"], base))

        count = max(len(val_loader), 1)
        avg_psnr = psnr_sum / count
        avg_ssim = ssim_sum / count
        avg_ms_ssim = ms_ssim_sum / count
        avg_uiqm = uiqm_sum / count
        print(
            f"[Val {epoch}] PSNR:{avg_psnr:.4f} SSIM:{avg_ssim:.4f} "
            f"MS-SSIM:{avg_ms_ssim:.4f} UIQM:{avg_uiqm:.4f}"
        )

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_psnr:.4f},{avg_ssim:.4f},{avg_ms_ssim:.4f},{avg_uiqm:.4f}\n")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            save_checkpoint(os.path.join(dirs["models"], "best_psnr.pth"), model, optimizer, scheduler, epoch, best_psnr, args)
            for name in os.listdir(dirs["best_images"]):
                os.remove(os.path.join(dirs["best_images"], name))
            for name in os.listdir(dirs["cat_images"]):
                shutil.copy2(os.path.join(dirs["cat_images"], name), os.path.join(dirs["best_images"], name))
            print(f"[Best] epoch={epoch}, PSNR={best_psnr:.4f}")

        save_loss_plot(loss_history, os.path.join(dirs["losses"], "loss_curve.png"))


if __name__ == "__main__":
    main()
