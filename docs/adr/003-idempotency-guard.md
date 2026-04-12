# ADR-003: DynamoDB ConditionExpression for Idempotency

## Status
Accepted

## Context
Kinesis Data Streams provides at-least-once delivery semantics. Under retry
conditions (Lambda timeout, throttling, shard rebalance), the same record
can be delivered to the processor more than once.

## Decision
Every `PutItem` call to the `patient_state` DynamoDB table uses a
`ConditionExpression` of `attribute_not_exists(timestamp_ms)` to make writes
idempotent.

## Rationale
Without this guard, a retry of a previously-processed record would:
1. Re-invoke the SageMaker endpoint (wasteful, adds latency cost).
2. Overwrite the existing DynamoDB item (benign in most cases, but races are possible).
3. Trigger a second DynamoDB Streams event → second alert notification for the
   same clinical anomaly event. **This is a patient-safety concern.**

The `ConditionExpression` makes the write a no-op if the item already exists.
The `ConditionalCheckFailedException` is caught and silently ignored — it
indicates successful prior processing, not an error.

## Consequences
- Adds one conditional write unit to each DynamoDB call (negligible cost).
- Requires that `(patient_id, timestamp_ms)` is a stable, deterministic key
  for a given window. The processor derives `window_start_ms` deterministically
  from the record timestamp and window size, satisfying this requirement.
