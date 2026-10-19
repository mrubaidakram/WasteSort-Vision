"""Predict a category for one user-supplied image."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms

from . import CLASSES
from .data import IMAGE_SIZE
from .model import build_model

MAX_IMAGE_PIXELS = 40_000_000


def load_image(path: str | Path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image does not exist: {path}")
    try:
        with Image.open(path) as source:
            width, height = source.size
            if width < 1 or height < 1 or width * height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image dimensions must be positive and no more than {MAX_IMAGE_PIXELS} pixels")
            image = source.convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError(f"Input is not a supported, readable image: {path}") from exc
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(image).unsqueeze(0)


def predict(image_path: str | Path, checkpoint_path: str | Path):
    image = load_image(image_path)
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    classes = checkpoint.get("classes")
    if classes != list(CLASSES):
        raise ValueError("Checkpoint classes do not match this WasteSort Vision version")
    model = build_model(len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.inference_mode():
        scores = torch.softmax(model(image.to(device))[0], dim=0).cpu().tolist()
    ranked = sorted(zip(classes, scores), key=lambda item: item[1], reverse=True)
    return {"predicted_category": ranked[0][0], "confidence": ranked[0][1],
            "scores": {name: score for name, score in ranked}}


def main():
    parser = argparse.ArgumentParser(description="Classify a single waste image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="artifacts/best.pt")
    args = parser.parse_args()
    try:
        print(json.dumps(predict(args.image, args.checkpoint), indent=2))
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
