variable "name_prefix" {
  type = string
}

variable "shard_count" {
  description = "Number of Kinesis shards. Rule of thumb: 1 shard per 10 concurrent devices at 500 Hz ECG."
  type        = number
  default     = 4
}

variable "lambda_role_arn" {
  description = "IAM role ARN of the telemetry processor Lambda (granted KMS decrypt)"
  type        = string
}

variable "data_lake_bucket_arn" {
  description = "ARN of the S3 data lake bucket for Firehose raw archival"
  type        = string
}

variable "tags" {
  type    = map(string)
  default = {}
}
