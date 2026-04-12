variable "name_prefix" {
  description = "Prefix applied to all resource names"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "kms_key_arn" {
  description = "ARN of the KMS CMK used to encrypt CloudWatch Log Groups"
  type        = string
}

variable "tags" {
  description = "Tags applied to all resources"
  type        = map(string)
  default     = {}
}
