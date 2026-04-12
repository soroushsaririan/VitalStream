###############################################################################
# DynamoDB — patient state table
#
# Stores inference results keyed by (patient_id, timestamp_ms).
# DynamoDB Streams feeds the alert-routing Lambda.
# TTL = 72h for hot operational state; S3 is the long-term record of truth.
###############################################################################

resource "aws_dynamodb_table" "patient_state" {
  name         = "${var.name_prefix}-patient-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "patient_id"
  range_key    = "timestamp_ms"

  attribute {
    name = "patient_id"
    type = "S"
  }

  attribute {
    name = "timestamp_ms"
    type = "N"
  }

  attribute {
    name = "is_alert"
    type = "S"  # "true" / "false" — DDB does not natively index booleans in GSI
  }

  # GSI: query all active alerts across all patients
  global_secondary_index {
    name            = "AlertStatusIndex"
    hash_key        = "is_alert"
    range_key       = "timestamp_ms"
    projection_type = "ALL"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled     = true
    kms_key_arn = aws_kms_key.dynamodb.arn
  }

  stream_enabled   = true
  stream_view_type = "NEW_IMAGE"  # alert router only needs the new state

  tags = merge(var.tags, { Name = "${var.name_prefix}-patient-state" })
}

# ---------------------------------------------------------------------------
# KMS CMK for DynamoDB
# ---------------------------------------------------------------------------

resource "aws_kms_key" "dynamodb" {
  description             = "CMK for ${var.name_prefix} DynamoDB patient state table"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "AllowRootFullAccess"
        Effect = "Allow"
        Principal = { AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root" }
        Action   = "kms:*"
        Resource = "*"
      },
      {
        Sid    = "AllowDynamoDB"
        Effect = "Allow"
        Principal = { Service = "dynamodb.amazonaws.com" }
        Action   = ["kms:Encrypt", "kms:Decrypt", "kms:ReEncrypt*", "kms:GenerateDataKey*", "kms:DescribeKey"]
        Resource = "*"
      },
      {
        Sid    = "AllowLambdaWriteRead"
        Effect = "Allow"
        Principal = { AWS = var.lambda_role_arn }
        Action   = ["kms:Decrypt", "kms:GenerateDataKey", "kms:DescribeKey"]
        Resource = "*"
        Condition = {
          StringEquals = {
            "kms:ViaService" = "dynamodb.${data.aws_region.current.name}.amazonaws.com"
          }
        }
      }
    ]
  })

  tags = merge(var.tags, { Name = "${var.name_prefix}-dynamodb-cmk" })
}

resource "aws_kms_alias" "dynamodb" {
  name          = "alias/${var.name_prefix}-dynamodb"
  target_key_id = aws_kms_key.dynamodb.key_id
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
