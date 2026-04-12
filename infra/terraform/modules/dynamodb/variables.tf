variable "name_prefix" {
  type = string
}

variable "lambda_role_arn" {
  description = "IAM role ARN of the telemetry processor Lambda"
  type        = string
}

variable "tags" {
  type    = map(string)
  default = {}
}
