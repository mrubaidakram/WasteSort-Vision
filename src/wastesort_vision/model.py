"""Model factory for the WasteSort classifier."""

from torch import nn
from torchvision.models import resnet18


def build_model(num_classes: int):
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
