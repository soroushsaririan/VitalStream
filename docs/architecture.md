# System Architecture

## Ingestion Layer

Devices publish via MQTT over TLS 1.3 with mutual TLS (X.509 per-device certificates).
AWS IoT Core validates certificates against the device registry, applies per-device
IoT Policies, and routes topic-matched messages via Rule Engine to Kinesis Data Streams.

Topic schema: `telemetry/{patient_id}/{signal_type}`

Kinesis Data Streams is partitioned by `patient_id` — all windows for a given patient
land on the same shard, preserving ordering for windowed feature extraction.

## Processing Layer

**Hot path (Lambda):**
- Enhanced Fan-Out consumer (dedicated 2 MB/s throughput per shard)
- 5-second tumbling window of raw samples
- Bandpass filter appropriate to signal modality
- Feature extraction (RMS, peak-to-peak, spectral entropy)
- SageMaker Multi-Model Endpoint invocation
- Result written to DynamoDB `patient_state` table

**Warm path (Kinesis Data Analytics / Apache Flink):**
- 60-second tumbling window aggregates (HRV, signal power spectra)
- Output to S3 Parquet, registered in Glue Data Catalog
- Used for retrospective analysis and model retraining dataset construction

## Inference Layer

SageMaker Multi-Model Endpoint hosts one TorchScript model per signal modality.
The Lambda processor routes to the correct model via the `TargetModel` parameter.
Inference response includes `anomaly_score` (0–1), `anomaly_class`, and `model_version`.

## State and Alert Layer

DynamoDB `patient_state` table:
- PK: `patient_id`, SK: `timestamp_ms`
- GSI on `is_alert` for fast alert queries
- TTL: 72 hours (hot operational state)
- Point-in-time recovery enabled
- DynamoDB Streams feeds alert-routing Lambda → SNS → PagerDuty / clinical dashboard

## Data Lake

S3 buckets with SSE-KMS and Object Lock (WORM):
- `/raw/` — Kinesis Firehose delivery of raw records
- `/processed/` — Flink 60s aggregate Parquet
- `/inference/` — SageMaker Data Capture
- `/audit/` — CloudTrail logs

Retention: 7 years (HIPAA §164.530(j))

## Network Architecture

```
VPC: 10.0.0.0/16
├── AZ-A
│   ├── Public:       10.0.1.0/24   (NAT Gateway only)
│   ├── Private App:  10.0.11.0/24  (Lambda ENIs, ECS)
│   └── Private Data: 10.0.21.0/24  (DDB/S3 endpoints)
├── AZ-B
│   ├── Public:       10.0.2.0/24
│   ├── Private App:  10.0.12.0/24
│   └── Private Data: 10.0.22.0/24
└── AZ-C
    ├── Public:       10.0.3.0/24
    ├── Private App:  10.0.13.0/24
    └── Private Data: 10.0.23.0/24
```

VPC Interface Endpoints (PrivateLink):
- `kinesis-streams`, `sagemaker.runtime`, `secretsmanager`, `kms`, `logs`, `monitoring`, `iot.data`

VPC Gateway Endpoints:
- `s3`, `dynamodb`
