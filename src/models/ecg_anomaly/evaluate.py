"""
models/ecg_anomaly/evaluate.py

SageMaker Processing Job entry-point for post-training model evaluation.

This script is run by the SageMaker Pipeline after the TrainingJob completes.
It computes clinical-grade metrics and writes an evaluation_report.json that
the Pipeline's ConditionStep uses to gate model registration.

Clinical metric gates (must ALL pass to register the model):
  precision_at_recall_90 >= 0.88   — high precision when recall is forced to 90%
  false_negative_rate    <= 0.05   — miss-rate ceiling (patient safety)
  inference_latency_p99  <= 80ms   — on ml.c5.2xlarge (contractual SLA)

Why precision_at_recall_90 and not AUC-ROC:
  AUC-ROC is misleading on imbalanced datasets. A model that detects 99% of
  NORMAL windows but misses 30% of AFIB events achieves high AUC-ROC.
  Precision@Recall=0.90 directly expresses the clinical trade-off: "if we
  want to catch 90% of true anomalies, how often do we false-alarm?"
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from torch.utils.data import DataLoader

from src.models.ecg_anomaly.model import ECGAnomalyDetector
from src.models.ecg_anomaly.train import ECGWindowDataset
from src.features.signal_processing import bandpass_filter, normalise_window

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

REPORT_PATH = os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/processing/output")
MODEL_DIR   = os.environ.get("SM_CHANNEL_MODEL",   "/opt/ml/processing/model")
TEST_DIR    = os.environ.get("SM_CHANNEL_TEST",    "/opt/ml/processing/test")

LATENCY_WARMUP_RUNS = 20
LATENCY_TIMED_RUNS  = 200


def measure_inference_latency_ms(model: torch.nn.Module, window_samples: int) -> dict[str, float]:
    """
    Measure single-sample inference latency on CPU (ml.c5 profile).
    Warmup runs excluded from measurement to account for JIT optimisation.
    """
    device = torch.device("cpu")
    dummy = torch.randn(1, 1, window_samples, dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP_RUNS):
            _ = model(dummy)

        latencies = []
        for _ in range(LATENCY_TIMED_RUNS):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)
    return {
        "p50_ms": float(np.percentile(latencies_arr, 50)),
        "p95_ms": float(np.percentile(latencies_arr, 95)),
        "p99_ms": float(np.percentile(latencies_arr, 99)),
        "mean_ms": float(latencies_arr.mean()),
    }


def evaluate() -> None:
    model_path = os.path.join(MODEL_DIR, "model.pt")
    metadata_path = os.path.join(MODEL_DIR, "metadata.json")

    with open(metadata_path) as f:
        metadata = json.load(f)
    model_version = metadata["model_version"]
    class_labels = metadata["class_labels"]
    num_classes = len(class_labels)

    model = torch.jit.load(model_path, map_location="cpu")
    model.eval()

    dataset = ECGWindowDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            all_logits.append(logits)
            all_labels.append(y)

    logits_tensor = torch.cat(all_logits, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    probs = torch.softmax(logits_tensor, dim=1).numpy()
    preds = probs.argmax(axis=1)
    labels_np = labels_tensor.numpy()

    # Anomaly score = P(not NORMAL) = 1 - P(NORMAL=0)
    anomaly_scores = 1.0 - probs[:, 0]
    # Binary ground truth: NORMAL=0 → 0, anything else → 1
    binary_labels = (labels_np != 0).astype(int)

    # Precision-recall curve for anomaly detection (binary)
    precision_arr, recall_arr, _ = precision_recall_curve(binary_labels, anomaly_scores)

    # Precision at recall >= 0.90
    valid_mask = recall_arr >= 0.90
    precision_at_recall_90 = float(precision_arr[valid_mask].max()) if valid_mask.any() else 0.0

    # False negative rate (class-averaged across all anomaly classes)
    cm = confusion_matrix(labels_np, preds, labels=list(range(num_classes)))
    fnr_per_class = []
    for i in range(1, num_classes):  # skip NORMAL (index 0)
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fnr_per_class.append(fn / (tp + fn + 1e-12))
    false_negative_rate = float(np.mean(fnr_per_class))

    # Per-class classification report
    clf_report = classification_report(
        labels_np, preds, target_names=class_labels, output_dict=True
    )

    # Latency measurement
    window_samples = logits_tensor.shape[0]  # wrong — use actual dim
    # Derive window_samples from first test file
    sample_npz = np.load(sorted(Path(TEST_DIR).glob("*.npz"))[0])
    window_samples = len(sample_npz["window"])
    latency_stats = measure_inference_latency_ms(model, window_samples)

    # Gate evaluation
    gates_passed = (
        precision_at_recall_90 >= 0.88
        and false_negative_rate <= 0.05
        and latency_stats["p99_ms"] <= 80.0
    )

    report = {
        "model_version":          model_version,
        "gates_passed":           gates_passed,
        "precision_at_recall_90": round(precision_at_recall_90, 4),
        "false_negative_rate":    round(false_negative_rate, 4),
        "inference_latency":      latency_stats,
        "classification_report":  clf_report,
        "gate_thresholds": {
            "precision_at_recall_90_min": 0.88,
            "false_negative_rate_max":    0.05,
            "latency_p99_ms_max":         80.0,
        },
    }

    os.makedirs(REPORT_PATH, exist_ok=True)
    report_file = os.path.join(REPORT_PATH, "evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation report written to %s", report_file)
    logger.info(
        "Gates: precision@recall90=%.4f | FNR=%.4f | latency_p99=%.1fms | passed=%s",
        precision_at_recall_90,
        false_negative_rate,
        latency_stats["p99_ms"],
        gates_passed,
    )

    if not gates_passed:
        # Non-zero exit code causes SageMaker Pipeline ConditionStep to fail
        raise SystemExit(1)


if __name__ == "__main__":
    evaluate()
