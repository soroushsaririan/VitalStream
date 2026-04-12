variable "name_prefix"               { type = string }
variable "kinesis_stream_arn"         { type = string }
variable "kinesis_kms_key_arn"        { type = string }
variable "kinesis_shard_count"        { type = number }
variable "sagemaker_endpoint_name"    { type = string }
variable "dynamodb_table_arn"         { type = string }
variable "dynamodb_table_name"        { type = string }
variable "dynamodb_kms_key_arn"       { type = string }
variable "private_app_subnet_ids"     { type = list(string) }
variable "sg_lambda_processor_id"     { type = string }
variable "deployment_package_path"    { type = string }

variable "anomaly_alert_threshold" {
  type    = number
  default = 0.85
}

variable "ecg_sample_rate_hz" {
  type    = number
  default = 500
}

variable "window_seconds" {
  type    = number
  default = 5
}

variable "log_level" {
  type    = string
  default = "INFO"
}

variable "alarm_sns_arn" {
  type    = string
  default = ""
}

variable "tags" {
  type    = map(string)
  default = {}
}
