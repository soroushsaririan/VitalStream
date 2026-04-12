# ADR-002: SageMaker Multi-Model Endpoint over Separate Endpoints

## Status
Accepted

## Context
The pipeline serves inference for multiple signal modalities: ECG (multiple leads),
continuous blood pressure, and sEMG. Each modality has a separately trained and
versioned model.

## Decision
Use a single SageMaker Multi-Model Endpoint (MME) with one TorchScript `.tar.gz`
model artifact per signal type, routed via the `TargetModel` parameter.

## Rationale

- **Cost:** One `ml.c5.2xlarge` instance hosts all models. Separate endpoints
  would require one instance per modality, multiplying baseline cost.
- **Independent versioning:** Each model artifact in S3 is independently versioned.
  Updating the ECG model does not require touching the sEMG model artifact.
- **Cold-load latency:** MME loads models into memory on first invocation and
  caches them. For a portfolio demo with continuous traffic per modality, all
  models remain warm. The `ModelNotReadyException` retry path in the Lambda
  handler covers the initial cold-load case.
- **Operational simplicity:** One endpoint ARN to monitor, one CloudWatch
  dashboard, one IAM `sagemaker:InvokeEndpoint` permission.

## Consequences
- All models share the same instance memory. If model artifacts grow large
  (>1 GB each), an `ml.c5.4xlarge` or GPU instance may be required.
- MME does not support real-time A/B traffic splitting between model versions
  within a single target model. Shadow testing uses a separate endpoint
  configuration with a production variant and a shadow variant.
