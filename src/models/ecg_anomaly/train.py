"""
models/ecg_anomaly/train.py

SageMaker Training Job entry-point script.

Invoked by the SageMaker TrainingJob with:
  - Input channel "train" → /opt/ml/input/data/train/
  - Input channel "val"   → /opt/ml/input/data/val/
  - Hyperparameters       → /opt/ml/input/config/hyperparameters.json
  - Model artifacts saved → /opt/ml/model/

The trained model is exported as TorchScript (.pt) so the inference container
can load it without the training-time class definition on the Python path.

Usage (local, for testing):
    python train.py \
        --train-dir data/train \
        --val-dir data/val \
        --epochs 30 \
        --batch-size 64 \
        --lr 1e-3 \
        --model-dir /tmp/model_output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.models.ecg_anomaly.model import ECGAnomalyDetector, FocalLoss
from src.features.signal_processing import bandpass_filter, normalise_window

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ECGWindowDataset(Dataset):
    """
    Loads pre-segmented ECG windows from a directory of .npz files.

    Each .npz contains:
      "window"  : float32 array of shape (WINDOW_SAMPLES,)
      "label"   : int  (index into ECGAnomalyDetector.CLASS_LABELS)
      "fs"      : float (sampling frequency, Hz)
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.files = sorted(Path(data_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")
        logger.info("Loaded dataset: %d windows from %s", len(self.files), data_dir)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        data = np.load(self.files[idx])
        window = data["window"].astype(np.float32)
        fs = float(data["fs"])
        label = int(data["label"])

        # Apply the same preprocessing as the Lambda inference path
        filtered = bandpass_filter(window, lowcut=0.5, highcut=40.0, fs=fs)
        normalised = normalise_window(filtered)

        x = torch.from_numpy(normalised).unsqueeze(0)  # (1, WINDOW_SAMPLES)
        y = torch.tensor(label, dtype=torch.long)
        return x, y


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def compute_class_weights(dataset: ECGWindowDataset, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights to initialise FocalLoss weight tensor."""
    counts = torch.zeros(num_classes)
    for _, y in DataLoader(dataset, batch_size=256, num_workers=4):
        for label in y:
            counts[label.item()] += 1
    counts = counts.clamp(min=1)
    weights = counts.sum() / (num_classes * counts)
    return weights / weights.sum()


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training on device: %s", device)

    train_dataset = ECGWindowDataset(args.train_dir)
    val_dataset = ECGWindowDataset(args.val_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=device.type == "cuda",
    )

    model = ECGAnomalyDetector(
        window_samples=args.window_samples,
        num_classes=args.num_classes,
        lstm_hidden=args.lstm_hidden,
        dropout=args.dropout,
    ).to(device)

    class_weights = compute_class_weights(train_dataset, args.num_classes).to(device)
    criterion = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Training ---
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_dataset)
        scheduler.step()

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += criterion(logits, y).item() * len(x)
                correct += (logits.argmax(dim=1) == y).sum().item()

        val_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)

        logger.info(
            "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.4f",
            epoch, args.epochs, train_loss, val_loss, val_acc,
        )

        # SageMaker metric regex: "val_loss: 0.1234"
        print(f"val_loss: {val_loss:.6f}")
        print(f"val_accuracy: {val_acc:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_model(model, args.model_dir, args.model_version)

    logger.info("Training complete. Best val_loss: %.6f", best_val_loss)


def _save_model(model: ECGAnomalyDetector, model_dir: str, version: str) -> None:
    """
    Export model as TorchScript for framework-agnostic serving.
    SageMaker inference container loads model.pt without requiring this module.
    """
    os.makedirs(model_dir, exist_ok=True)
    model.eval()
    scripted = torch.jit.script(model)
    model_path = os.path.join(model_dir, "model.pt")
    scripted.save(model_path)

    metadata = {
        "model_version": version,
        "class_labels": ECGAnomalyDetector.CLASS_LABELS,
        "framework": "pytorch",
        "export_format": "torchscript",
    }
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Model saved to %s (version %s)", model_path, version)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ECG anomaly detector training script")

    # SageMaker injects these as /opt/ml/... paths in a TrainingJob
    p.add_argument("--train-dir",      default=os.environ.get("SM_CHANNEL_TRAIN", "data/train"))
    p.add_argument("--val-dir",        default=os.environ.get("SM_CHANNEL_VAL",   "data/val"))
    p.add_argument("--model-dir",      default=os.environ.get("SM_MODEL_DIR",     "/tmp/model"))
    p.add_argument("--model-version",  default="0.0.1")

    # Hyperparameters (tunable via SageMaker HyperParameter Tuning Jobs)
    p.add_argument("--epochs",         type=int,   default=30)
    p.add_argument("--batch-size",     type=int,   default=64)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--dropout",        type=float, default=0.3)
    p.add_argument("--lstm-hidden",    type=int,   default=128)
    p.add_argument("--focal-gamma",    type=float, default=2.0)
    p.add_argument("--window-samples", type=int,   default=2500)
    p.add_argument("--num-classes",    type=int,   default=5)

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
