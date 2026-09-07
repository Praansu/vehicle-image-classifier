#!/usr/bin/env python
"""Generate confusion matrix visualization for the trained model."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src.data_loader import VehicleDataset, get_val_transform
from src.model import create_resnet_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test dataset
    test_dataset = VehicleDataset(root_dir="data/test", transform=get_val_transform())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    classes = test_dataset.classes
    print(f"Classes: {classes}")
    print(f"Test samples: {len(test_dataset)}")

    # Load model
    model = create_resnet_model(num_classes=len(classes))
    model.load_state_dict(torch.load("api/vehicle_model.pth", map_location=device))
    model = model.to(device)
    model.eval()

    # Predict
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax1,
    )
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Actual")
    ax1.set_title("Confusion Matrix (Counts)")

    # Normalized
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax2,
    )
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix (Normalized %)")

    plt.tight_layout()
    output_path = "confusion_matrix.png"
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved confusion matrix to {output_path}")

    # Per-class accuracy
    print("\nPer-class accuracy:")
    for i, cls in enumerate(classes):
        acc = cm_normalized[i, i] * 100
        print(f"  {cls}: {acc:.1f}%")

    # Overall accuracy
    overall = np.trace(cm) / np.sum(cm) * 100
    print(f"\nOverall accuracy: {overall:.1f}%")


if __name__ == "__main__":
    main()
