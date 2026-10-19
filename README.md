# WasteSort Vision

WasteSort Vision is a local computer vision project for classifying a single waste item in an image into one of six TrashNet material categories. It provides reproducible dataset preparation, model training, held-out evaluation, and an image prediction CLI.

## Problem and supported categories

Given a photo containing one reasonably visible item, predict its material category:

| Category | Examples in dataset |
| --- | --- |
| cardboard | Cardboard packaging |
| glass | Glass bottles and containers |
| metal | Metal cans and objects |
| paper | Loose paper and paper products |
| plastic | Plastic containers and objects |
| trash | Other waste represented by TrashNet's residual class |

These are dataset labels, not authoritative local recycling instructions. Local rules vary, composite/contaminated materials may need special handling, and the model can be uncertain. Verify disposal decisions with local guidance.

## Dataset

The project uses **TrashNet**, a publicly accessible dataset of 2,527 images across the six categories above. Images were captured against a relatively simple background and the class counts are imbalanced. This is a small research/educational dataset and does not represent the full range of real-world waste imagery.

- Dataset repository: [garythung/trashnet](https://github.com/garythung/trashnet)
- Paper: Gary Thung and Mindy Yang, *Classification of Trash for Recyclability Status* (2016), [project paper](https://cs229.stanford.edu/proj2016/report/ThungYang-ClassificationOfTrashForRecyclabilityStatus-report.pdf)
- The dataset repository describes the images and provides the download. Check its license and any source terms before use or redistribution; this project does not bundle or redistribute dataset images.

Download the dataset archive linked from [the TrashNet repository](https://github.com/garythung/trashnet) (also available directly at [dataset-resized.zip](https://github.com/garythung/trashnet/blob/master/data/dataset-resized.zip)), then extract it so images are arranged in class folders under `data/`, for example:

```text
data/                # root passed via --data-dir
└── dataset-resized/ # or another directory; loader detects class folders recursively
    ├── cardboard/
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/
```

You can clone the dataset repository and point the commands at its `dataset-resized` directory. Do not rename class folders. The loader checks for all six categories and supported image formats.

## Setup

Python 3.10 or newer is required. Create an environment and install the package:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install a PyTorch build appropriate for your CPU/GPU if the default package index does not provide a suitable build; see [PyTorch's official install selector](https://pytorch.org/get-started/locally/).

## Train and evaluate

```bash
wastesort-train --data-dir data/dataset-resized --output-dir artifacts \
  --epochs 20 --batch-size 32 --seed 42
```

The pipeline creates deterministic, stratified 70% train / 15% validation / 15% test splits. Training-only augmentation applies random horizontal flips and small rotations; validation and test use resize, center crop, and ImageNet normalization. A pretrained ResNet-18 is not used: training starts from torchvision's standard ResNet-18 architecture with random weights, avoiding implicit downloads and keeping the procedure explicit. The final checkpoint is selected by validation macro-F1, then evaluated once on the held-out test set.

Outputs include `artifacts/best.pt`, epoch history in `artifacts/training_metrics.jsonl`, split metadata in `artifacts/splits.json`, and `artifacts/evaluation.json` with overall accuracy, per-class precision/recall/F1/support, macro averages, and the confusion matrix. Metrics are produced only by running the pipeline; no performance result is claimed in this repository.

Training options:

```bash
wastesort-train --help
```

## Predict an image

After training, run:

```bash
wastesort-predict --checkpoint artifacts/best.pt --image path/to/item.jpg
```

The JSON response includes the top category, its confidence, and softmax scores for every category. Confidence scores are model scores, not calibrated probabilities. The CLI validates that the input exists, decodes as an image, and is within configured size limits; it returns a useful error for invalid images. Batch inference is intentionally out of scope.

## Reproducibility and limitations

- Pass the same seed and dataset snapshot to reproduce split assignment and training initialization. GPU kernels and hardware/library versions can still produce small run-to-run differences.
- Dataset class imbalance and studio-like backgrounds can cause poor performance on cluttered, outdoor, or unfamiliar images.
- One image may contain multiple items, labels may not match local recycling rules, and the model cannot inspect contamination or material composition.
- Do not use this model as the sole basis for safety, compliance, or operational waste decisions. Ask a person when uncertain.
- No training or evaluation was run while preparing this repository; reported metrics will appear in the generated evaluation file after you run training.

## Repository layout

```text
src/wastesort_vision/  dataset, model, training, evaluation, prediction
data/                  local dataset (ignored by Git)
artifacts/             checkpoints and metrics (generated; only README tracked)
```

## License and attribution

Project code is provided under the MIT License. The dataset remains subject to the terms of its source; consult the TrashNet repository before downloading, using, or redistributing it. See [LICENSE](LICENSE).
