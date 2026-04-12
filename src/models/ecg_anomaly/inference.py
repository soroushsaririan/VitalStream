"""
models/ecg_anomaly/inference.py

SageMaker inference container entry-point.

SageMaker calls these four functions in sequence:
  1. model_fn(model_dir)          — load model from /opt/ml/model/
  2. input_fn(body, content_type) — deserialise the HTTP request body
  3. predict_fn(input, model)     — run inference
  4. output_fn(prediction, accept)— serialise the response

The container is started once per endpoint instance. model_fn is called once
at container start; input_fn/predict_fn/output_fn are called per request.

This module is packaged alongside model.pt in the SageMaker model artifact
(model.tar.gz) so it is available on the inference container's Python path.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import torch

# These modules are present in the artifact alongside inference.py
from features.signal_processing import normalise_window  # type: ignore[import]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

_MODEL_VERSION: str = "unknown"
_CLASS_LABELS: list[str] = ["NORMAL", "AFIB", "VT", "STEMI", "NOISE"]


def model_fn(model_dir: str) -> torch.jit.ScriptModule:
    """
    Load the TorchScript model from the model directory.
    Called once when the inference container starts.
    """
    global _MODEL_VERSION, _CLASS_LABELS

    model_path = os.path.join(model_dir, "model.pt")
    metadata_path = os.path.join(model_dir, "metadata.json")

    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            meta = json.load(f)
        _MODEL_VERSION = meta.get("model_version", "unknown")
        _CLASS_LABELS = meta.get("class_labels", _CLASS_LABELS)

    device = torch.device("cpu")  # ml.c5 instances are CPU-only
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Request deserialisation
# ---------------------------------------------------------------------------

def input_fn(body: str | bytes, content_type: str) -> dict[str, Any]:
    """
    Deserialise the inference request payload.

    Expected JSON schema:
    {
        "window_data":      [<float>, ...],  // flattened float32 array
        "window_shape":     [1, 2500],       // original shape for reshape
        "rms_amplitude":    <float>,
        "peak_to_peak":     <float>,
        "spectral_entropy": <float>
    }
    """
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}. Expected application/json")

    if isinstance(body, bytes):
        body = body.decode("utf-8")

    return json.loads(body)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_fn(
    input_data: dict[str, Any],
    model: torch.jit.ScriptModule,
) -> dict[str, Any]:
    """
    Run anomaly detection inference.

    The window is re-normalised here (z-score) as a safeguard against
    upstream pre-processing drift. The model was trained on z-score
    normalised windows; double-normalising an already-normalised input
    is a no-op (mean≈0, std≈1 → output unchanged).
    """
    window_data = np.array(input_data["window_data"], dtype=np.float32)
    window_shape = input_data["window_shape"]
    window = window_data.reshape(window_shape)  # (1, WINDOW_SAMPLES)

    # Re-normalise the single-channel window
    normalised = normalise_window(window[0])
    x = torch.from_numpy(normalised).unsqueeze(0).unsqueeze(0)  # (1, 1, W)

    with torch.no_grad():
        logits = model(x)                                # (1, num_classes)
        probs = torch.softmax(logits, dim=1)[0]          # (num_classes,)

    predicted_idx = int(probs.argmax().item())
    anomaly_score = float(1.0 - probs[0].item())        # P(not NORMAL)

    return {
        "anomaly_score":  round(anomaly_score, 6),
        "anomaly_class":  _CLASS_LABELS[predicted_idx],
        "class_probs":    {
            label: round(float(prob.item()), 6)
            for label, prob in zip(_CLASS_LABELS, probs)
        },
        "model_version":  _MODEL_VERSION,
    }


# ---------------------------------------------------------------------------
# Response serialisation
# ---------------------------------------------------------------------------

def output_fn(prediction: dict[str, Any], accept: str) -> tuple[str, str]:
    """
    Serialise prediction dict to the HTTP response body.

    Returns (body_string, content_type) as expected by the SageMaker SDK.
    """
    if accept not in ("application/json", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}. Expected application/json")

    return json.dumps(prediction), "application/json"
