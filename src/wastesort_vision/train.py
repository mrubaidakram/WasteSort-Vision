"""Reproducible training, validation-based model selection, and final test evaluation."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from . import CLASSES
from .data import make_loaders
from .metrics import classification_report
from .model import build_model


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collect_predictions(model, loader, device):
    model.eval()
    truths, predictions = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            truths.extend(labels.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return truths, predictions


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate WasteSort Vision on a class-folder dataset.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.learning_rate <= 0:
        parser.error("epochs/batch-size must be positive, workers non-negative, learning-rate positive")

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, validation_loader, test_loader, split = make_loaders(
        args.data_dir, args.batch_size, args.workers, args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

    model = build_model(len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    best_f1 = -1.0
    with (output / "training_metrics.jsonl").open("w", encoding="utf-8") as history:
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = count = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(images), labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * labels.size(0)
                count += labels.size(0)
            truth, prediction = collect_predictions(model, validation_loader, device)
            report = classification_report(truth, prediction, CLASSES)
            record = {"epoch": epoch, "train_loss": total_loss / count,
                      "validation_accuracy": report["accuracy"],
                      "validation_macro_f1": report["macro_avg"]["f1"]}
            print(json.dumps(record), flush=True)
            history.write(json.dumps(record) + "\n")
            history.flush()
            if report["macro_avg"]["f1"] > best_f1:
                best_f1 = report["macro_avg"]["f1"]
                torch.save({"model_state": model.cpu().state_dict(), "classes": list(CLASSES),
                            "seed": args.seed, "epoch": epoch, "image_size": 224,
                            "normalization": "imagenet"}, output / "best.pt")
                model.to(device)

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    truth, prediction = collect_predictions(model, test_loader, device)
    evaluation = {"selected_epoch": checkpoint["epoch"], "split": "test", "seed": args.seed,
                  "report": classification_report(truth, prediction, CLASSES)}
    (output / "evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
