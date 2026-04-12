"""
telemetry_processor/handler.py

Kinesis Enhanced Fan-Out consumer Lambda.

Reads a batch of patient telemetry records from Kinesis Data Streams,
extracts signal features, invokes the SageMaker Multi-Model Endpoint for
anomaly detection, and persists the result to DynamoDB.

AWS Lambda function configuration (managed via infra/terraform/modules/lambda):
  runtime:              python3.12
  memory_mb:            1024
  timeout_seconds:      60
  reserved_concurrency: <number of KDS shards>  — 1 concurrent exec per shard
  vpc_config:           private_app_subnets
  tracing_config:       Active  (AWS X-Ray)
  environment variables:
    SAGEMAKER_ENDPOINT_NAME   — MME endpoint name
    PATIENT_STATE_TABLE       — DynamoDB table name
    ANOMALY_ALERT_THRESHOLD   — float, default 0.85
    ECG_SAMPLE_RATE_HZ        — int, default 500
    WINDOW_SECONDS            — int, default 5

HIPAA notes:
  - log_event=False on the Lambda handler: raw Kinesis payloads contain PHI
    and must never be written to CloudWatch Logs.
  - patient_id is excluded from all structured log fields; it flows only
    through DynamoDB item writes and DynamoDB Streams (both encrypted at rest).
  - All AWS API calls are made over VPC PrivateLink endpoints; no PHI
    traverses the public internet.
"""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import boto3
import numpy as np
from aws_lambda_powertools import Logger, Metrics, Tracer
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.batch import (
    BatchProcessor,
    EventType,
    process_partial_response,
)
from aws_lambda_powertools.utilities.batch.exceptions import BatchProcessingError
from aws_lambda_powertools.utilities.typing import LambdaContext
from botocore.exceptions import ClientError

from src.features.signal_processing import (
    bandpass_filter,
    spectral_entropy,
    validate_signal_quality,
)

# ---------------------------------------------------------------------------
# Powertools — structured JSON logging, distributed tracing, custom metrics
# ---------------------------------------------------------------------------
logger = Logger(service="telemetry-processor", level="INFO")
tracer = Tracer(service="telemetry-processor")
metrics = Metrics(namespace="ClinicalTelemetry", service="telemetry-processor")

# ---------------------------------------------------------------------------
# AWS clients — initialised once, reused across warm Lambda invocations
# ---------------------------------------------------------------------------
_sagemaker_runtime = boto3.client(
    "sagemaker-runtime",
    region_name=os.environ["AWS_REGION"],
)
_dynamodb = boto3.resource("dynamodb", region_name=os.environ["AWS_REGION"])
_state_table = _dynamodb.Table(os.environ["PATIENT_STATE_TABLE"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SAGEMAKER_ENDPOINT = os.environ["SAGEMAKER_ENDPOINT_NAME"]
ANOMALY_ALERT_THRESHOLD = float(os.environ.get("ANOMALY_ALERT_THRESHOLD", "0.85"))
ECG_SAMPLE_RATE_HZ = int(os.environ.get("ECG_SAMPLE_RATE_HZ", "500"))
WINDOW_SECONDS = int(os.environ.get("WINDOW_SECONDS", "5"))
WINDOW_SAMPLES = ECG_SAMPLE_RATE_HZ * WINDOW_SECONDS

# Map each signal type to its bandpass filter parameters (Hz)
_FILTER_PARAMS: dict[str, dict[str, float]] = {
    "ECG_LEAD_II":    {"lowcut": 0.5,  "highcut": 40.0},
    "ECG_V1":         {"lowcut": 0.5,  "highcut": 40.0},
    "ECG_V5":         {"lowcut": 0.5,  "highcut": 40.0},
    "EMG_CHANNEL_0":  {"lowcut": 20.0, "highcut": 450.0},
    "EMG_CHANNEL_1":  {"lowcut": 20.0, "highcut": 450.0},
    "BP_CONTINUOUS":  {"lowcut": 0.1,  "highcut": 15.0},
}
_DEFAULT_FILTER = {"lowcut": 0.5, "highcut": 40.0}


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------

@dataclass
class TelemetryRecord:
    """
    Deserialised form of a single Kinesis record payload.

    Wire format (JSON, KMS-encrypted at rest in KDS):
    {
        "patient_id":     "pt-<uuid>",
        "device_id":      "dev-<uuid>",
        "signal_type":    "ECG_LEAD_II",
        "timestamp_ms":   1712345678123,
        "sequence_number": 42,
        "samples":        [<float>, ...]   // raw ADC values in µV
    }
    """
    patient_id: str
    device_id: str
    signal_type: str
    timestamp_ms: int
    sequence_number: int
    samples: list[float]

    @classmethod
    def from_kinesis_record(cls, record: dict[str, Any]) -> "TelemetryRecord":
        raw = base64.b64decode(record["kinesis"]["data"])
        payload = json.loads(raw)
        return cls(
            patient_id=payload["patient_id"],
            device_id=payload["device_id"],
            signal_type=payload["signal_type"],
            timestamp_ms=int(payload["timestamp_ms"]),
            sequence_number=int(payload["sequence_number"]),
            samples=payload["samples"],
        )


@dataclass
class InferenceFeatures:
    """Processed feature vector passed to the SageMaker endpoint."""
    window_data: np.ndarray       # shape: (1, WINDOW_SAMPLES) — extend for multi-channel
    rms_amplitude: float
    peak_to_peak: float
    spectral_entropy: float
    signal_type: str
    patient_id: str               # used internally; not logged
    window_start_ms: int


@dataclass
class AnomalyResult:
    patient_id: str
    signal_type: str
    window_start_ms: int
    anomaly_score: float          # 0.0 – 1.0
    anomaly_class: str            # e.g. "AFIB", "VT", "NOISE", "NORMAL"
    model_version: str
    inference_latency_ms: float
    is_alert: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_alert = self.anomaly_score >= ANOMALY_ALERT_THRESHOLD


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@tracer.capture_method
def extract_features(record: TelemetryRecord) -> InferenceFeatures | None:
    """
    Pre-process a raw sample buffer into a fixed-length feature window.

    Returns None (skip this record) when:
      - Buffer contains fewer samples than one full window.
      - Signal quality check detects flat-line or lead-off artifact.
    """
    samples = np.asarray(record.samples, dtype=np.float32)

    if len(samples) < WINDOW_SAMPLES:
        logger.warning(
            "Insufficient samples for windowing",
            extra={
                "received": len(samples),
                "required": WINDOW_SAMPLES,
                "signal_type": record.signal_type,
            },
        )
        return None

    window = samples[-WINDOW_SAMPLES:]

    if not validate_signal_quality(window):
        metrics.add_metric("LeadOffArtifact", unit=MetricUnit.Count, value=1)
        logger.info(
            "Lead-off / flat-line artifact detected; skipping window",
            extra={"signal_type": record.signal_type},
        )
        return None

    params = _FILTER_PARAMS.get(record.signal_type, _DEFAULT_FILTER)
    filtered = bandpass_filter(
        window,
        lowcut=params["lowcut"],
        highcut=params["highcut"],
        fs=float(ECG_SAMPLE_RATE_HZ),
    )

    return InferenceFeatures(
        window_data=filtered.reshape(1, -1),
        rms_amplitude=float(np.sqrt(np.mean(filtered ** 2))),
        peak_to_peak=float(filtered.max() - filtered.min()),
        spectral_entropy=spectral_entropy(filtered, fs=float(ECG_SAMPLE_RATE_HZ)),
        signal_type=record.signal_type,
        patient_id=record.patient_id,
        window_start_ms=record.timestamp_ms - WINDOW_SECONDS * 1000,
    )


# ---------------------------------------------------------------------------
# SageMaker inference
# ---------------------------------------------------------------------------

@tracer.capture_method
def invoke_anomaly_endpoint(features: InferenceFeatures) -> AnomalyResult:
    """
    Call the SageMaker Multi-Model Endpoint.

    TargetModel routes to the correct per-modality model artifact in S3.
    Expected response from the inference container:
      {"anomaly_score": <float>, "anomaly_class": <str>, "model_version": <str>}
    """
    payload = json.dumps({
        "window_data":      features.window_data.flatten().tolist(),
        "window_shape":     list(features.window_data.shape),
        "rms_amplitude":    features.rms_amplitude,
        "peak_to_peak":     features.peak_to_peak,
        "spectral_entropy": features.spectral_entropy,
    })

    target_model = f"{features.signal_type.lower()}_anomaly_detector.tar.gz"

    t0 = time.perf_counter()
    try:
        response = _sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT,
            TargetModel=target_model,
            ContentType="application/json",
            Accept="application/json",
            Body=payload,
        )
    finally:
        latency_ms = (time.perf_counter() - t0) * 1000
        metrics.add_metric(
            "InferenceLatencyMs", unit=MetricUnit.Milliseconds, value=latency_ms
        )

    result_body = json.loads(response["Body"].read())

    return AnomalyResult(
        patient_id=features.patient_id,
        signal_type=features.signal_type,
        window_start_ms=features.window_start_ms,
        anomaly_score=float(result_body["anomaly_score"]),
        anomaly_class=str(result_body["anomaly_class"]),
        model_version=str(result_body["model_version"]),
        inference_latency_ms=latency_ms,
    )


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

@tracer.capture_method
def persist_result(result: AnomalyResult) -> None:
    """
    Write inference result to DynamoDB patient_state table.

    Uses ConditionExpression to make writes idempotent — Kinesis delivers
    at-least-once, so the same record may be processed more than once on retry.
    A duplicate write is silently ignored; it does not re-fire an alert.
    """
    ttl_epoch = int(time.time()) + (72 * 3600)

    try:
        _state_table.put_item(
            Item={
                "patient_id":           result.patient_id,
                "timestamp_ms":         result.window_start_ms,
                "signal_type":          result.signal_type,
                "anomaly_score":        str(round(result.anomaly_score, 6)),
                "anomaly_class":        result.anomaly_class,
                "model_version":        result.model_version,
                "inference_latency_ms": str(round(result.inference_latency_ms, 2)),
                "is_alert":             result.is_alert,
                "ttl":                  ttl_epoch,
            },
            ConditionExpression="attribute_not_exists(timestamp_ms)",
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] == "ConditionalCheckFailedException":
            # Item already written by a prior (successful) invocation of this record.
            metrics.add_metric("DuplicateRecordSkipped", unit=MetricUnit.Count, value=1)
            return
        raise

    if result.is_alert:
        metrics.add_metric("AnomalyAlertFired", unit=MetricUnit.Count, value=1)
        logger.warning(
            "Anomaly threshold breached",
            extra={
                "anomaly_class":        result.anomaly_class,
                "anomaly_score":        result.anomaly_score,
                "model_version":        result.model_version,
                "signal_type":          result.signal_type,
                # patient_id intentionally excluded from logs (PHI)
            },
        )


# ---------------------------------------------------------------------------
# Per-record handler (called by BatchProcessor)
# ---------------------------------------------------------------------------

def record_handler(record: dict[str, Any]) -> None:
    """
    Process a single Kinesis record.

    BatchProcessor provides partial-batch failure semantics: if this function
    raises, the record is returned in `batchItemFailures` and Kinesis retries
    only the failed record — successfully-processed records are not re-delivered.
    """
    telemetry = TelemetryRecord.from_kinesis_record(record)

    features = extract_features(telemetry)
    if features is None:
        return

    result = invoke_anomaly_endpoint(features)
    persist_result(result)

    metrics.add_metric("RecordsProcessed", unit=MetricUnit.Count, value=1)


# ---------------------------------------------------------------------------
# Lambda entrypoint
# ---------------------------------------------------------------------------

processor = BatchProcessor(event_type=EventType.KinesisDataStreams)


@logger.inject_lambda_context(log_event=False)
@tracer.capture_lambda_handler
@metrics.log_metrics(capture_cold_start_metric=True)
def handler(event: dict[str, Any], context: LambdaContext) -> dict[str, Any]:
    """Lambda entrypoint for Kinesis Enhanced Fan-Out trigger."""
    try:
        return process_partial_response(
            event=event,
            record_handler=record_handler,
            processor=processor,
            context=context,
        )
    except BatchProcessingError as exc:
        logger.exception("Full batch processing failure", exc_info=exc)
        metrics.add_metric("BatchProcessingFailure", unit=MetricUnit.Count, value=1)
        return {"batchItemFailures": []}
