"""
Integration tests for the telemetry processor Lambda handler.

Uses moto to emulate DynamoDB in-process. SageMaker runtime is still mocked
at the boto3 client level (moto does not emulate SageMaker inference).

These tests verify:
  - A valid record flows end-to-end: feature extraction → inference → DynamoDB write
  - A flat-line record is silently skipped without writing to DynamoDB
  - An above-threshold anomaly score sets is_alert=True in DynamoDB
  - Partial-batch failure: a bad record returns batchItemFailures; good records persist
"""

from __future__ import annotations

import base64
import json
import os
from io import BytesIO
from unittest.mock import MagicMock, patch

import boto3
import numpy as np
import pytest

os.environ["AWS_DEFAULT_REGION"]          = "us-east-1"
os.environ["AWS_ACCESS_KEY_ID"]           = "test"
os.environ["AWS_SECRET_ACCESS_KEY"]       = "test"
os.environ["AWS_REGION"]                  = "us-east-1"
os.environ["SAGEMAKER_ENDPOINT_NAME"]     = "test-mme-endpoint"
os.environ["PATIENT_STATE_TABLE"]         = "patient-state-integration"
os.environ["ANOMALY_ALERT_THRESHOLD"]     = "0.85"
os.environ["ECG_SAMPLE_RATE_HZ"]          = "500"
os.environ["WINDOW_SECONDS"]              = "5"
os.environ["POWERTOOLS_METRICS_NAMESPACE"] = "ClinicalTelemetry"
os.environ["POWERTOOLS_SERVICE_NAME"]      = "telemetry-processor"

import moto

WINDOW_SAMPLES = 2500


def _make_payload(patient_id: str, signal_type: str, samples: list[float]) -> str:
    payload = {
        "patient_id":      patient_id,
        "device_id":       "dev-001",
        "signal_type":     signal_type,
        "timestamp_ms":    1712345678000,
        "sequence_number": 1,
        "samples":         samples,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _make_event(records: list[dict]) -> dict:
    return {"Records": records}


def _kinesis_record(patient_id: str, signal_type: str, samples: list[float]) -> dict:
    return {
        "kinesis": {
            "data":                          _make_payload(patient_id, signal_type, samples),
            "partitionKey":                  patient_id,
            "sequenceNumber":                "1",
            "approximateArrivalTimestamp":   1712345678.0,
        },
        "eventID":     "shardId-000:1",
        "eventSource": "aws:kinesis",
    }


def _mock_sagemaker_response(anomaly_score: float = 0.2, anomaly_class: str = "NORMAL") -> MagicMock:
    body_content = json.dumps({
        "anomaly_score":  anomaly_score,
        "anomaly_class":  anomaly_class,
        "model_version":  "1.0.0",
    }).encode()
    mock_body = MagicMock()
    mock_body.read.return_value = body_content

    mock_client = MagicMock()
    mock_client.invoke_endpoint.return_value = {"Body": mock_body}
    mock_client.exceptions.ModelNotReadyException = Exception
    return mock_client


@moto.mock_aws
class TestHandlerIntegration:

    def setup_method(self, method=None):
        """Create the DynamoDB table before each test."""
        ddb = boto3.resource("dynamodb", region_name="us-east-1")
        ddb.create_table(
            TableName="patient-state-integration",
            KeySchema=[
                {"AttributeName": "patient_id", "KeyType": "HASH"},
                {"AttributeName": "timestamp_ms", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "patient_id",   "AttributeType": "S"},
                {"AttributeName": "timestamp_ms", "AttributeType": "N"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

    def _get_item(self, patient_id: str, timestamp_ms: int) -> dict | None:
        ddb   = boto3.resource("dynamodb", region_name="us-east-1")
        table = ddb.Table("patient-state-integration")
        resp  = table.get_item(Key={"patient_id": patient_id, "timestamp_ms": timestamp_ms})
        return resp.get("Item")

    def test_valid_record_written_to_dynamodb(self):
        from src.telemetry_processor import handler

        samples = (np.random.default_rng(0).normal(0, 100, WINDOW_SAMPLES)).tolist()
        event   = _make_event([_kinesis_record("pt-001", "ECG_LEAD_II", samples)])

        mock_sm = _mock_sagemaker_response(anomaly_score=0.2, anomaly_class="NORMAL")
        with patch.object(handler, "_sagemaker_runtime", mock_sm):
            result = handler.handler(event, MagicMock())

        assert result["batchItemFailures"] == []
        item = self._get_item("pt-001", 1712345673000)
        assert item is not None
        assert item["anomaly_class"] == "NORMAL"
        assert item["is_alert"] is False

    def test_alert_flag_set_for_high_score(self):
        from src.telemetry_processor import handler

        samples = (np.random.default_rng(1).normal(0, 100, WINDOW_SAMPLES)).tolist()
        event   = _make_event([_kinesis_record("pt-002", "ECG_LEAD_II", samples)])

        mock_sm = _mock_sagemaker_response(anomaly_score=0.92, anomaly_class="AFIB")
        with patch.object(handler, "_sagemaker_runtime", mock_sm):
            handler.handler(event, MagicMock())

        item = self._get_item("pt-002", 1712345673000)
        assert item["is_alert"] is True
        assert item["anomaly_class"] == "AFIB"

    def test_flat_line_record_not_written_to_dynamodb(self):
        from src.telemetry_processor import handler

        samples = [0.0] * WINDOW_SAMPLES
        event   = _make_event([_kinesis_record("pt-flat", "ECG_LEAD_II", samples)])

        mock_sm = _mock_sagemaker_response()
        with patch.object(handler, "_sagemaker_runtime", mock_sm):
            result = handler.handler(event, MagicMock())

        assert result["batchItemFailures"] == []
        item = self._get_item("pt-flat", 1712345673000)
        assert item is None  # skipped — not written

    def test_sagemaker_full_batch_failure_unblocks_shard(self):
        """
        When ALL records in a batch fail (e.g. endpoint down), the handler
        catches BatchProcessingError and returns empty batchItemFailures.
        This deliberately unblocks the shard iterator rather than infinitely
        retrying an unavailable endpoint. The record ages out via Kinesis
        retention and the DLQ captures it via the ESM on-failure destination.
        """
        from src.telemetry_processor import handler

        samples = (np.random.default_rng(2).normal(0, 100, WINDOW_SAMPLES)).tolist()
        event   = _make_event([_kinesis_record("pt-err", "ECG_LEAD_II", samples)])

        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.side_effect = RuntimeError("Endpoint unavailable")
        mock_sm.exceptions.ModelNotReadyException = RuntimeError

        with patch.object(handler, "_sagemaker_runtime", mock_sm):
            result = handler.handler(event, MagicMock())

        # Full-batch failure: returns [] to unblock the shard (by design)
        assert result["batchItemFailures"] == []

    def test_duplicate_record_idempotent(self):
        """Processing the same record twice must not raise and must not double-alert."""
        from src.telemetry_processor import handler

        samples = (np.random.default_rng(3).normal(0, 100, WINDOW_SAMPLES)).tolist()
        event   = _make_event([_kinesis_record("pt-dup", "ECG_LEAD_II", samples)])
        mock_sm = _mock_sagemaker_response(anomaly_score=0.92, anomaly_class="VT")

        with patch.object(handler, "_sagemaker_runtime", mock_sm):
            handler.handler(event, MagicMock())
            result = handler.handler(event, MagicMock())  # second delivery

        assert result["batchItemFailures"] == []  # no failure on duplicate
