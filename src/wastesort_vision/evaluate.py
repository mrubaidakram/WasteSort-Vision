"""Evaluate a saved checkpoint on the held-out split using its recorded manifest."""

import argparse
import json
from pathlib import Path

import torch

from . import CLASSES
from .data import make_test_loader
from .metrics import classification_report
from .model import build_model
from .train import collect_predictions


def main():
    parser = argparse.ArgumentParser(description="Re-run held-out test evaluation for a training run.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="artifacts/best.pt")
    parser.add_argument("--output", default="artifacts/evaluation.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("batch-size must be positive and workers cannot be negative")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("classes") != list(CLASSES):
        parser.error("Checkpoint classes do not match this WasteSort Vision version")
    test_loader = make_test_loader(args.data_dir, args.batch_size, args.workers, args.seed)
    model = build_model(len(CLASSES))
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    truth, prediction = collect_predictions(model, test_loader, device)
    result = {"selected_epoch": checkpoint.get("epoch"), "split": "test", "seed": args.seed,
              "report": classification_report(truth, prediction, CLASSES)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
