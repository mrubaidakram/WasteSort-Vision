"""Dataset discovery, validation, deterministic stratified splits and loaders."""

from pathlib import Path
import random

import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from . import CLASSES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = 224


class WasteImages(Dataset):
    """ImageFolder-like dataset with explicit class order and decode checks."""

    def __init__(self, root: str | Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")
        self.classes = list(CLASSES)
        self.samples: list[tuple[Path, int]] = []
        missing = []
        for label, name in enumerate(CLASSES):
            folders = [p for p in self.root.rglob(name) if p.is_dir() and p.name == name]
            files = sorted(file for folder in folders for file in folder.rglob("*")
                           if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES)
            if not files:
                missing.append(name)
            self.samples.extend((file, label) for file in files)
        if missing:
            raise ValueError(f"No supported images found for class folders: {', '.join(missing)}")
        if not self.samples:
            raise ValueError(f"No supported images found under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError(f"Could not decode dataset image {path}: {exc}") from exc
        if self.transform:
            image = self.transform(image)
        return image, label


def _stratified_split(dataset: WasteImages, seed: int):
    grouped = [[] for _ in CLASSES]
    for index, (_, label) in enumerate(dataset.samples):
        grouped[label].append(index)
    rng = random.Random(seed)
    train, validation, test = [], [], []
    for label, indices in enumerate(grouped):
        if len(indices) < 3:
            raise ValueError(f"Class {CLASSES[label]!r} needs at least 3 images; found {len(indices)}")
        rng.shuffle(indices)
        test_count = max(1, round(len(indices) * 0.15))
        validation_count = max(1, round(len(indices) * 0.15))
        if test_count + validation_count >= len(indices):
            raise ValueError(f"Not enough images in class {CLASSES[label]!r} for train/validation/test splits")
        test.extend(indices[:test_count])
        validation.extend(indices[test_count:test_count + validation_count])
        train.extend(indices[test_count + validation_count:])
    return train, validation, test


def make_loaders(root: str | Path, batch_size: int, workers: int, seed: int):
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers cannot be negative")
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    base = WasteImages(root)
    train_indices, validation_indices, test_indices = _stratified_split(base, seed)
    train_data = WasteImages(root, train_transform)
    eval_data = WasteImages(root, evaluation_transform)
    generator = torch.Generator().manual_seed(seed)
    options = {"batch_size": batch_size, "num_workers": workers,
               "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(Subset(train_data, train_indices), shuffle=True, generator=generator, **options),
        DataLoader(Subset(eval_data, validation_indices), shuffle=False, **options),
        DataLoader(Subset(eval_data, test_indices), shuffle=False, **options),
        {"classes": list(CLASSES), "train": train_indices, "validation": validation_indices,
         "test": test_indices, "samples": [str(path) for path, _ in base.samples],
         "labels": [label for _, label in base.samples]},
    )


def make_test_loader(root: str | Path, batch_size: int, workers: int, seed: int):
    """Build only the held-out loader using the same split and evaluation transform."""
    if batch_size < 1 or workers < 0:
        raise ValueError("batch_size must be positive and workers cannot be negative")
    evaluation_transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    base = WasteImages(root)
    _, _, test_indices = _stratified_split(base, seed)
    dataset = WasteImages(root, evaluation_transform)
    loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False,
                        num_workers=workers, pin_memory=torch.cuda.is_available())
    return loader
