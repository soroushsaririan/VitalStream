# VitalStream

**Serverless real-time streaming pipeline for remote patient telemetry on AWS.**

Ingests continuous physiological time-series data (12-lead ECG, continuous blood
pressure, multi-channel sEMG), processes it in a serverless stream-processing
layer, and invokes a deployed ML model to detect physiological anomalies in
near real-time. Designed to operate within HIPAA-eligible AWS services with
end-to-end encryption and strict least-privilege IAM.

---

## Architecture Overview

```
IoT Device (mTLS)
    └─► AWS IoT Core (device registry, rule engine)
            └─► Kinesis Data Streams (SSE-KMS, 24h retention)
                    ├─► Lambda (hot path — feature extraction + SageMaker inference)
                    │       └─► DynamoDB (patient state, 72h TTL, DDB Streams → alerts)
                    └─► Kinesis Data Analytics / Flink (warm path — 60s aggregates → S3)

SageMaker Multi-Model Endpoint
    └─► 1D-CNN + LSTM ensemble, one model per signal modality
    └─► Model Registry → Blue/Green deployment, no pipeline downtime

All traffic routed through VPC PrivateLink — no PHI crosses the public internet.
```

Full architecture specification: [`docs/architecture.md`](docs/architecture.md)

---

## Repository Layout

```
VitalStream/
├── docs/                          # Architecture, ADRs, compliance notes
├── infra/
│   ├── terraform/                 # All AWS infrastructure as code
│   │   ├── modules/               # vpc, kinesis, lambda, sagemaker, dynamodb
│   │   └── environments/          # staging, prod variable overrides
│   └── sagemaker_pipelines/       # Training + evaluation pipeline definition
├── src/
│   ├── telemetry_processor/       # Lambda stream consumer (core handler)
│   ├── models/
│   │   ├── ecg_anomaly/           # PyTorch model: train, inference, evaluate
│   │   └── emg_anomaly/           # sEMG model: train, inference, evaluate
│   └── features/                  # Shared DSP feature extraction utilities
├── tests/
│   ├── unit/                      # Pure logic tests, no AWS calls
│   └── integration/               # localstack (KDS, DDB) + moto (SageMaker)
├── .github/workflows/             # CI (lint/test/scan) + CD (SageMaker Pipeline)
└── scripts/                       # Local dev helpers (seed Kinesis, invoke endpoint)
```

---

## Quickstart (Local Development)

### Prerequisites

- Python 3.12+
- Docker (for localstack)
- AWS CLI v2 configured with a profile that has no production access
- Terraform >= 1.7

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

### 2. Start localstack

```bash
docker compose -f docker-compose.localstack.yml up -d
```

### 3. Run the test suite

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 4. Seed a simulated ECG stream (local)

```bash
python scripts/seed_kinesis_local.py --signal-type ECG_LEAD_II --duration-sec 30
```

---

## Infrastructure Deployment

```bash
cd infra/terraform/environments/staging
terraform init
terraform plan -out=tfplan
terraform apply tfplan
```

See [`infra/terraform/README.md`](infra/terraform/README.md) for full variable
reference and required IAM permissions.

---

## MLOps Pipeline

Model training and deployment is managed by a SageMaker Pipeline defined in
`infra/sagemaker_pipelines/training_pipeline.py`.

Trigger manually:
```bash
python infra/sagemaker_pipelines/training_pipeline.py --run
```

Or via CI on any merge to `main` that touches `src/models/` or `src/features/`.

Deployment is Blue/Green with automatic rollback on CloudWatch alarm breach.
Manual approval gate (clinical engineer + MLOps on-call) required before
traffic shifts to a new model version.

---

## Compliance Notes

| Control | Implementation |
|---|---|
| Encryption at rest | SSE-KMS (CMK) on KDS, S3, DynamoDB, SageMaker EBS |
| Encryption in transit | TLS 1.3 (IoT Core), TLS 1.2 minimum (all VPC endpoints) |
| Network isolation | All PHI workloads in private subnets; VPC PrivateLink for AWS APIs |
| Access control | IAM least-privilege per service; DynamoDB attribute-level RBAC |
| Audit logging | CloudTrail (all regions, S3 Object Lock), VPC Flow Logs |
| Data retention | S3 Object Lock WORM, 7-year retention per HIPAA §164.530(j) |
| PHI in logs | `log_event=False` on all Lambda handlers; no PHI in structured log fields |
| Idempotency | DynamoDB `ConditionExpression` guards against duplicate alert firing |

---

## Key Engineering Decisions

See [`docs/adr/`](docs/adr/) for full Architecture Decision Records.

- **Kinesis over MSK:** HIPAA-eligible, fully serverless, zero broker ops at portfolio scale.
- **Partial-batch failure reporting:** Prevents one malformed record from blocking an entire shard iterator.
- **Multi-Model Endpoint:** Single `ml.c5.2xlarge` serves all signal modalities; independent versioning per model.
- **`ConditionExpression` on DDB writes:** Kinesis at-least-once delivery requires idempotency guards to prevent duplicate clinical alerts.

---

## License

MIT
