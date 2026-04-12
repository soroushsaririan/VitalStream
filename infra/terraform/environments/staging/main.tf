###############################################################################
# Staging environment
#
# Deploys the core data infrastructure:
#   VPC → Kinesis Data Streams → DynamoDB → S3 Data Lake → Lambda processor
#
# SageMaker endpoint is excluded from this deployment (requires GPU instance,
# not Free Tier eligible). The Lambda runs with SAGEMAKER_ENDPOINT_NAME set
# to a placeholder; inference calls gracefully fail and are routed to the DLQ.
###############################################################################

terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.40"
    }
    time = {
      source  = "hashicorp/time"
      version = "~> 0.11"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }

  backend "s3" {
    bucket         = "vitalstream-tfstate-567768661071"
    key            = "vitalstream/staging/terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "vitalstream-tfstate-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "VitalStream"
      Environment = "staging"
      ManagedBy   = "Terraform"
      Compliance  = "HIPAA"
    }
  }
}

locals {
  name_prefix = "vitalstream-staging"
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# ---------------------------------------------------------------------------
# KMS CMK — shared for this staging deployment
# ---------------------------------------------------------------------------

resource "aws_kms_key" "main" {
  description             = "VitalStream staging CMK"
  deletion_window_in_days = 7
  enable_key_rotation     = true
  tags                    = { Name = "${local.name_prefix}-cmk" }
}

resource "aws_kms_alias" "main" {
  name          = "alias/${local.name_prefix}"
  target_key_id = aws_kms_key.main.key_id
}

# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------

module "vpc" {
  source      = "../../modules/vpc"
  name_prefix = local.name_prefix
  vpc_cidr    = "10.0.0.0/16"
  kms_key_arn = aws_kms_key.main.arn
  tags        = {}
}

# ---------------------------------------------------------------------------
# S3 Data Lake
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "data_lake" {
  bucket        = "${local.name_prefix}-data-lake-${data.aws_caller_identity.current.account_id}"
  force_destroy = true
  tags          = { Name = "${local.name_prefix}-data-lake" }
}

resource "aws_s3_bucket_versioning" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data_lake" {
  bucket = aws_s3_bucket.data_lake.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.main.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "data_lake" {
  bucket                  = aws_s3_bucket.data_lake.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "data_lake" {
  bucket     = aws_s3_bucket.data_lake.id
  depends_on = [aws_s3_bucket_public_access_block.data_lake]
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "DenyUnencryptedPuts"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:PutObject"
        Resource  = "${aws_s3_bucket.data_lake.arn}/*"
        Condition = {
          StringNotEquals = { "s3:x-amz-server-side-encryption" = "aws:kms" }
        }
      },
      {
        Sid       = "DenyNonTLS"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource  = [aws_s3_bucket.data_lake.arn, "${aws_s3_bucket.data_lake.arn}/*"]
        Condition = { Bool = { "aws:SecureTransport" = "false" } }
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Kinesis Data Streams
# ---------------------------------------------------------------------------

module "kinesis" {
  source               = "../../modules/kinesis"
  name_prefix          = local.name_prefix
  shard_count          = 1
  lambda_role_arn      = module.lambda.role_arn
  data_lake_bucket_arn = aws_s3_bucket.data_lake.arn
  tags                 = {}
}

# ---------------------------------------------------------------------------
# DynamoDB
# ---------------------------------------------------------------------------

module "dynamodb" {
  source          = "../../modules/dynamodb"
  name_prefix     = local.name_prefix
  lambda_role_arn = module.lambda.role_arn
  tags            = {}
}

# ---------------------------------------------------------------------------
# Lambda deployment package — built inline from source
# ---------------------------------------------------------------------------

data "archive_file" "lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../../../../"
  output_path = "${path.module}/lambda_deployment.zip"
  excludes = [
    ".venv", ".git", "__pycache__", "*.pyc",
    "infra", "docs", "tests", "scripts",
    "dashboard", "*.md", "*.toml", "*.txt",
    "docker-compose*", "*.zip"
  ]
}

# ---------------------------------------------------------------------------
# Lambda
# ---------------------------------------------------------------------------

module "lambda" {
  source                  = "../../modules/lambda"
  name_prefix             = local.name_prefix
  kinesis_stream_arn      = module.kinesis.stream_arn
  kinesis_kms_key_arn     = module.kinesis.kms_key_arn
  kinesis_shard_count     = 1
  sagemaker_endpoint_name = "vitalstream-staging-placeholder"
  dynamodb_table_arn      = module.dynamodb.table_arn
  dynamodb_table_name     = module.dynamodb.table_name
  dynamodb_kms_key_arn    = module.dynamodb.kms_key_arn
  private_app_subnet_ids  = module.vpc.private_app_subnet_ids
  sg_lambda_processor_id  = module.vpc.sg_lambda_processor_id
  deployment_package_path = data.archive_file.lambda_zip.output_path
  anomaly_alert_threshold = 0.85
  tags                    = {}
}

# ---------------------------------------------------------------------------
# Outputs — printed after terraform apply, used for screenshots
# ---------------------------------------------------------------------------

output "vpc_id" {
  value = module.vpc.vpc_id
}

output "kinesis_stream_name" {
  value = module.kinesis.stream_name
}

output "dynamodb_table_name" {
  value = module.dynamodb.table_name
}

output "lambda_function_name" {
  value = module.lambda.function_name
}

output "data_lake_bucket" {
  value = aws_s3_bucket.data_lake.bucket
}

output "kms_key_arn" {
  value = aws_kms_key.main.arn
}
