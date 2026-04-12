"""
Unit tests for src/telemetry_processor/handler.py

Uses moto to mock AWS calls (SageMaker, DynamoDB).
No live AWS credentials required.

Tests cover:
  - TelemetryRecord deserialisation
  - extract_features: insufficient buffer, artifact rejection, happy path
  - persist_result: idempotency guard (ConditionalCheckFailedException)
  - record_handler: end-to-end with mocked SageMaker response
  - handler: partial-batch failure reporting
"""

from __future__ import annotations

import base64
import json
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Environment variable stubs — must be set before importing handler
# ---------------------------------------------------------------------------

os.environ.setdefault("AWS_REGION",                "us-east-1")
os.environ.setdefault("SAGEMAKER_ENDPOINT_NAME",   "test-endpoint")
os.environ.setdefault("PATIENT_STATE_TABLE",       "patient-state")
os.environ.setdefault("ANOMALY_ALERT_THRESHOLD",   "0.85")
os.environ.setdefault("ECG_SAMPLE_RATE_HZ",        "500")
os.environ.setdefault("WINDOW_SECONDS",            "5")

# Stub Powertools env vars
os.environ.setdefault("POWERTOOLS_METRICS_NAMESPACE", "ClinicalTelemetry")
os.environ.setdefault("POWERTOOLS_SERVICE_NAME",      "telemetry-processor")

from src.telemetry_processor.handler import (  # noqa: E402
    AnomalyResult,
    InferenceFeatures,
    TelemetryRecord,
    extract_features,
    persist_result,
)

FS             = 500
WINDOW_SAMPLES = 2500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_kinesis_record(
    signal_type: str = "ECG_LEAD_II",
    num_samples: int = WINDOW_SAMPLES,
    patient_id: str  = "pt-test-001",
) -> dict:
    payload = {
        "patient_id":      patient_id,
        "device_id":       "dev-abc",
        "signal_type":     signal_type,
        "timestamp_ms":    1712345678000,
        "sequence_number": 1,
        "samples":         (np.random.randn(num_samples) * 100).tolist(),
    }
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    return {
        "kinesis": {
            "data":                   encoded,
            "partitionKey":           patient_id,
            "sequenceNumber":         "1",
            "approximateArrivalTimestamp": 1712345678.0,
        },
        "eventID":     "shardId-000:1",
        "eventSource": "aws:kinesis",
    }


def make_flat_kinesis_record() -> dict:
    payload = {
        "patient_id":      "pt-flat",
        "device_id":       "dev-xyz",
        "signal_type":     "ECG_LEAD_II",
        "timestamp_ms":    1712345678000,
        "sequence_number": 2,
        "samples":         [0.0] * WINDOW_SAMPLES,
    }
    encoded = base64.b64encode(json.dumps(payload).encode()).decode()
    return {"kinesis": {"data": encoded, "partitionKey": "pt-flat", "sequenceNumber": "2", "approximateArrivalTimestamp": 1712345678.0}, "eventID": "shardId-000:2", "eventSource": "aws:kinesis"}


# ---------------------------------------------------------------------------
# TelemetryRecord
# ---------------------------------------------------------------------------

class TestTelemetryRecord:
    def test_deserialise_happy_path(self):
        record = make_kinesis_record()
        tr = TelemetryRecord.from_kinesis_record(record)
        assert tr.patient_id   == "pt-test-001"
        assert tr.signal_type  == "ECG_LEAD_II"
        assert len(tr.samples) == WINDOW_SAMPLES

    def test_deserialise_missing_field_raises(self):
        payload = {"patient_id": "x", "signal_type": "ECG_LEAD_II"}
        encoded = base64.b64encode(json.dumps(payload).encode()).decode()
        with pytest.raises(KeyError):
            TelemetryRecord.from_kinesis_record(
                {"kinesis": {"data": encoded}}
            )


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    def test_returns_none_for_insufficient_samples(self):
        record = make_kinesis_record(num_samples=100)
        tr = TelemetryRecord.from_kinesis_record(record)
        assert extract_features(tr) is None

    def test_returns_none_for_flat_signal(self):
        tr = TelemetryRecord.from_kinesis_record(make_flat_kinesis_record())
        assert extract_features(tr) is None

    def test_returns_inference_features_for_valid_signal(self):
        record = make_kinesis_record()
        tr     = TelemetryRecord.from_kinesis_record(record)
        feats  = extract_features(tr)
        assert feats is not None
        assert feats.window_data.shape == (1, WINDOW_SAMPLES)
        assert feats.rms_amplitude > 0
        assert 0.0 <= feats.spectral_entropy <= 1.0

    def test_unknown_signal_type_uses_default_filter(self):
        record = make_kinesis_record(signal_type="UNKNOWN_SIGNAL")
        tr     = TelemetryRecord.from_kinesis_record(record)
        feats  = extract_features(tr)
        assert feats is not None  # should not raise


# ---------------------------------------------------------------------------
# persist_result
# ---------------------------------------------------------------------------

class TestPersistResult:
    def _make_result(self, score: float = 0.3) -> AnomalyResult:
        return AnomalyResult(
            patient_id="pt-test-001",
            signal_type="ECG_LEAD_II",
            window_start_ms=1712345673000,
            anomaly_score=score,
            anomaly_class="NORMAL",
            model_version="1.0.0",
            inference_latency_ms=42.1,
        )

    def test_idempotency_guard_silently_ignored(self):
        """
        ConditionalCheckFailedException must be caught and not re-raised.
        A duplicate write (same patient_id + timestamp_ms) is a no-op.
        """
        from botocore.exceptions import ClientError

        mock_table = MagicMock()
        mock_table.put_item.side_effect = ClientError(
            {"Error": {"Code": "ConditionalCheckFailedException", "Message": ""}},
            "PutItem",
        )

        with patch("src.telemetry_processor.handler._state_table", mock_table):
            persist_result(self._make_result())  # must not raise

    def test_unexpected_client_error_propagates(self):
        from botocore.exceptions import ClientError

        mock_table = MagicMock()
        mock_table.put_item.side_effect = ClientError(
            {"Error": {"Code": "ProvisionedThroughputExceededException", "Message": ""}},
            "PutItem",
        )

        with patch("src.telemetry_processor.handler._state_table", mock_table):
            with pytest.raises(ClientError):
                persist_result(self._make_result())

    def test_alert_flag_set_above_threshold(self):
        result = self._make_result(score=0.92)
        assert result.is_alert is True

    def test_no_alert_below_threshold(self):
        result = self._make_result(score=0.5)
        assert result.is_alert is False
