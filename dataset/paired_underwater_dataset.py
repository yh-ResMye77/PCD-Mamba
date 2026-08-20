import random
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
INPUT_DIR_NAMES = ("hazy", "input", "inputs", "raw", "underwater", "source")
TARGET_DIR_NAMES = ("clear", "groundtruth", "gt", "target", "reference", "clean")


def underwater_color_cast(img_tensor):
    img_aug = img_tensor.clone()
    if random.random() < 0.5:
        img_aug[0] *= random.uniform(0.5, 0.8)
    if random.random() < 0.5:
        img_aug[random.choice([1, 2])] *= random.uniform(1.0, 1.2)
    return torch.clamp(img_aug, 0.0, 1.0)


def is_image_file(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def find_first_existing_dir(parent, names):
    for name in names:
        candidate = Path(parent) / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find paired image folders under {parent}")


def build_target_index(target_dir):
    index = {}
    for path in sorted(target_dir.iterdir()):
        if path.is_file() and is_image_file(path):
            index.setdefault(path.stem, path)
    return index


class PairedUnderwaterDataset(Dataset):
    def __init__(self, split_dir, image_size=256, augment=False, use_color_attenuation=True):
        super().__init__()
        self.split_dir = Path(split_dir)
        self.input_dir = find_first_existing_dir(self.split_dir, INPUT_DIR_NAMES)
        self.target_dir = find_first_existing_dir(self.split_dir, TARGET_DIR_NAMES)
        self.augment = augment
        self.use_color_attenuation = use_color_attenuation
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        target_index = build_target_index(self.target_dir)
        self.samples = []
        for input_path in sorted(self.input_dir.iterdir()):
            if not input_path.is_file() or not is_image_file(input_path):
                continue
            target_path = self.target_dir / input_path.name
            if not target_path.is_file():
                target_path = target_index.get(input_path.stem)
            if target_path is not None and target_path.is_file():
                self.samples.append((input_path, target_path))

        if not self.samples:
            raise RuntimeError(f"No paired images found under {self.split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, target_path = self.samples[index]
        degraded = self.transform(Image.open(input_path).convert("RGB"))
        target = self.transform(Image.open(target_path).convert("RGB"))

        if self.augment:
            if random.random() < 0.5:
                degraded = torch.flip(degraded, dims=[2])
                target = torch.flip(target, dims=[2])
            if self.use_color_attenuation:
                degraded = underwater_color_cast(degraded)

        return degraded, target, input_path.name
