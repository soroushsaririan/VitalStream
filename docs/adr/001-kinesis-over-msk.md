# ADR-001: Kinesis Data Streams over Amazon MSK

## Status
Accepted

## Context
The pipeline requires a HIPAA-eligible, managed streaming broker. The two
primary options on AWS are Kinesis Data Streams (KDS) and Amazon MSK (Kafka).

## Decision
Use Kinesis Data Streams.

## Rationale

| Factor | KDS | MSK |
|---|---|---|
| HIPAA BAA eligibility | Yes | Yes |
| Broker management | Zero (fully serverless) | Requires broker sizing, patching, ZK/KRaft management |
| Scaling model | Shard split/merge (can be automated) | Partition reassignment, broker rebalancing |
| At-rest encryption | SSE-KMS CMK, native | Requires explicit broker config |
| Consumer model | Enhanced Fan-Out (push, 2 MB/s/shard dedicated) | Pull, consumer group lag management |
| Cost model | Per-shard-hour + PUT units | Per broker-hour (minimum 2 brokers) |
| Portfolio scale | <100 concurrent devices fits comfortably in 2–4 shards | MSK minimum viable cluster is over-provisioned for this scale |

## Consequences
- Ordering is guaranteed per shard (partition key = `patient_id`).
- Retention is capped at 365 days on KDS (sufficient; raw data is also archived to S3 via Firehose).
- If the project scales to >10k concurrent devices, revisit MSK for finer-grained partition control and Kafka ecosystem tooling (ksqlDB, Kafka Connect).
