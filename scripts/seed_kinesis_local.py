"""
scripts/seed_kinesis_local.py

Generates synthetic physiological signal data and puts records into a
Kinesis stream (local localstack or real AWS) for development and integration
testing purposes.

Usage:
    # Against localstack (default)
    python scripts/seed_kinesis_local.py \
        --signal-type ECG_LEAD_II \
        --duration-sec 30 \
        --patient-id pt-dev-001

    # Against real AWS (uses default profile)
    python scripts/seed_kinesis_local.py \
        --signal-type ECG_LEAD_II \
        --duration-sec 10 \
        --endpoint-url ""

WARNING: This script generates synthetic data only. It must never be used
to seed a stream carrying real patient data.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import uuid

import boto3
import numpy as np


# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------

def generate_ecg_window(n_samples: int, fs: float, heart_rate_bpm: float = 72.0) -> list[float]:
    """
    Generate a realistic synthetic ECG-like waveform using a sum of Gaussians
    to approximate the P-QRS-T complex morphology.

    Not a clinical-grade ECG simulator; suitable only for pipeline integration testing.
    """
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    rr_interval = 60.0 / heart_rate_bpm  # seconds

    signal = np.zeros(n_samples, dtype=np.float32)

    # Add noise baseline
    rng = np.random.default_rng()
    signal += rng.normal(0, 5, n_samples).astype(np.float32)

    # Place PQRS-T complexes at each heartbeat position
    beat_times = np.arange(0, n_samples / fs, rr_interval)
    for bt in beat_times:
        # P wave: small positive deflection
        signal += 50  * np.exp(-((t - (bt + 0.08)) ** 2) / (2 * 0.012 ** 2))
        # Q wave: small negative deflection
        signal -= 20  * np.exp(-((t - (bt + 0.14)) ** 2) / (2 * 0.005 ** 2))
        # R wave: large positive spike
        signal += 800 * np.exp(-((t - (bt + 0.16)) ** 2) / (2 * 0.004 ** 2))
        # S wave: negative deflection after R
        signal -= 150 * np.exp(-((t - (bt + 0.18)) ** 2) / (2 * 0.005 ** 2))
        # T wave: broad positive deflection
        signal += 100 * np.exp(-((t - (bt + 0.30)) ** 2) / (2 * 0.030 ** 2))

    return signal.tolist()


def generate_emg_window(n_samples: int, fs: float) -> list[float]:
    """Bandlimited Gaussian noise approximating surface EMG during light contraction."""
    rng = np.random.default_rng()
    raw = rng.normal(0, 200, n_samples).astype(np.float32)
    return raw.tolist()


def generate_bp_window(n_samples: int, fs: float) -> list[float]:
    """Synthetic arterial blood pressure waveform (systolic ~120 mmHg, diastolic ~80 mmHg)."""
    t = np.linspace(0, n_samples / fs, n_samples, endpoint=False)
    hr = 72.0
    rr = 60.0 / hr
    signal = np.full(n_samples, 80.0, dtype=np.float32)  # diastolic baseline

    for bt in np.arange(0, n_samples / fs, rr):
        # Systolic peak
        signal += 40 * np.exp(-((t - (bt + 0.12)) ** 2) / (2 * 0.04 ** 2))
        # Dicrotic notch
        signal += 5  * np.exp(-((t - (bt + 0.30)) ** 2) / (2 * 0.01 ** 2))

    rng = np.random.default_rng()
    signal += rng.normal(0, 1, n_samples).astype(np.float32)
    return signal.tolist()


_GENERATORS = {
    "ECG_LEAD_II":   generate_ecg_window,
    "ECG_V1":        generate_ecg_window,
    "EMG_CHANNEL_0": generate_emg_window,
    "BP_CONTINUOUS": generate_bp_window,
}

_SAMPLE_RATES = {
    "ECG_LEAD_II":   500,
    "ECG_V1":        500,
    "EMG_CHANNEL_0": 2000,
    "BP_CONTINUOUS": 125,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def seed(args: argparse.Namespace) -> None:
    client_kwargs: dict = {"region_name": args.region}
    if args.endpoint_url:
        client_kwargs["endpoint_url"] = args.endpoint_url

    kinesis = boto3.client("kinesis", **client_kwargs)

    fs = _SAMPLE_RATES.get(args.signal_type, 500)
    window_samples = fs * args.window_seconds
    generator = _GENERATORS.get(args.signal_type, generate_ecg_window)

    total_windows = math.ceil(args.duration_sec / args.window_seconds)
    print(f"Seeding {total_windows} windows of {args.signal_type} "
          f"({args.window_seconds}s @ {fs} Hz) for patient {args.patient_id}")

    for seq in range(total_windows):
        samples = generator(window_samples, float(fs))
        timestamp_ms = int(time.time() * 1000) + seq * args.window_seconds * 1000

        payload = json.dumps({
            "patient_id":      args.patient_id,
            "device_id":       f"dev-seed-{uuid.uuid4().hex[:8]}",
            "signal_type":     args.signal_type,
            "timestamp_ms":    timestamp_ms,
            "sequence_number": seq,
            "samples":         samples,
        })

        kinesis.put_record(
            StreamName=args.stream_name,
            Data=payload.encode(),
            PartitionKey=args.patient_id,
        )

        print(f"  [{seq + 1}/{total_windows}] window timestamp_ms={timestamp_ms}")
        time.sleep(args.window_seconds)  # real-time pacing

    print("Done.")


def main() -> None:
    p = argparse.ArgumentParser(description="Seed synthetic telemetry into Kinesis")
    p.add_argument("--signal-type",    default="ECG_LEAD_II", choices=list(_GENERATORS))
    p.add_argument("--duration-sec",   type=int,   default=30)
    p.add_argument("--window-seconds", type=int,   default=5)
    p.add_argument("--patient-id",     default=f"pt-dev-{uuid.uuid4().hex[:8]}")
    p.add_argument("--stream-name",    default="vitalstream-staging-patient-telemetry")
    p.add_argument("--region",         default="us-east-1")
    p.add_argument("--endpoint-url",   default="http://localhost:4566",
                   help="localstack endpoint; set to '' for real AWS")
    seed(p.parse_args())


if __name__ == "__main__":
    main()
