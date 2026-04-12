###############################################################################
# Kinesis Data Streams — patient telemetry ingestion
#
# One stream per environment. Partition key = patient_id ensures all windows
# for a given patient land on the same shard, preserving ordering for the
# Lambda windowed feature extractor.
#
# KMS CMK encryption: all shard data encrypted at rest. The CMK is separate
# from the VPC/logging CMK to allow independent key rotation and audit.
###############################################################################

resource "aws_kinesis_stream" "telemetry" {
  name             = "${var.name_prefix}-patient-telemetry"
  shard_count      = var.shard_count
  retention_period = 24  # hours; raw data also archived to S3 via Firehose

  encryption_type = "KMS"
  kms_key_id      = aws_kms_key.kinesis.arn

  stream_mode_details {
    stream_mode = "PROVISIONED"  # switch to ON_DEMAND if shard count is unpredictable
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-patient-telemetry" })
}

# ---------------------------------------------------------------------------
# KMS CMK for Kinesis stream encryption
# ---------------------------------------------------------------------------

resource "aws_kms_key" "kinesis" {
  description             = "CMK for ${var.name_prefix} Kinesis patient telemetry stream"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowRootAccountFullAccess"
        Effect = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "AllowKinesisEncryption"
        Effect = "Allow"
        Principal = { Service = "kinesis.amazonaws.com" }
        Action   = ["kms:GenerateDataKey", "kms:Decrypt"]
        Resource = "*"
      },
      {
        Sid    = "AllowLambdaDecrypt"
        Effect = "Allow"
        Principal = { AWS = var.lambda_role_arn }
        Action   = ["kms:Decrypt", "kms:DescribeKey"]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "kinesis.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      },
      {
        Sid    = "AllowIoTCoreEncrypt"
        Effect = "Allow"
        Principal = { Service = "iot.amazonaws.com" }
        Action   = ["kms:GenerateDataKey", "kms:Decrypt"]
        Resource = "*"
      }
    ]
  })

  tags = merge(var.tags, { Name = "${var.name_prefix}-kinesis-cmk" })
}

resource "aws_kms_alias" "kinesis" {
  name          = "alias/${var.name_prefix}-kinesis"
  target_key_id = aws_kms_key.kinesis.key_id
}

# ---------------------------------------------------------------------------
# Kinesis Firehose — raw record archival to S3 data lake
# ---------------------------------------------------------------------------

resource "aws_kinesis_firehose_delivery_stream" "raw_archive" {
  name        = "${var.name_prefix}-raw-archive"
  destination = "extended_s3"

  kinesis_source_configuration {
    kinesis_stream_arn = aws_kinesis_stream.telemetry.arn
    role_arn           = aws_iam_role.firehose.arn
  }

  extended_s3_configuration {
    role_arn            = aws_iam_role.firehose.arn
    bucket_arn          = var.data_lake_bucket_arn
    prefix              = "raw/year=!{timestamp:yyyy}/month=!{timestamp:MM}/day=!{timestamp:dd}/"
    error_output_prefix = "raw-errors/!{firehose:error-output-type}/year=!{timestamp:yyyy}/"
    buffering_size      = 128   # MB
    buffering_interval  = 300   # seconds

    compression_format = "GZIP"

    s3_backup_mode = "Disabled"

    cloudwatch_logging_options {
      enabled         = true
      log_group_name  = "/aws/kinesisfirehose/${var.name_prefix}-raw-archive"
      log_stream_name = "S3Delivery"
    }
  }

  tags = var.tags
}

resource "aws_iam_role" "firehose" {
  name = "${var.name_prefix}-firehose-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "firehose.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "firehose" {
  role = aws_iam_role.firehose.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["kinesis:GetRecords", "kinesis:GetShardIterator", "kinesis:DescribeStream", "kinesis:ListShards"]
        Resource = aws_kinesis_stream.telemetry.arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${var.data_lake_bucket_arn}/raw/*"
      },
      {
        Effect   = "Allow"
        Action   = ["kms:Decrypt", "kms:GenerateDataKey"]
        Resource = aws_kms_key.kinesis.arn
      }
    ]
  })
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
