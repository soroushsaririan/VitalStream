###############################################################################
# Lambda — telemetry stream processor
#
# One function, one Kinesis Enhanced Fan-Out (EFH) event source mapping.
# Reserved concurrency = shard count: prevents one hot patient from consuming
# all available Lambda concurrency and starving other shards.
###############################################################################

data "aws_region" "current" {}
data "aws_caller_identity" "current" {}

terraform {
  required_providers {
    time = {
      source  = "hashicorp/time"
      version = "~> 0.11"
    }
  }
}

# ---------------------------------------------------------------------------
# IAM Role
# ---------------------------------------------------------------------------

resource "aws_iam_role" "processor" {
  name = "${var.name_prefix}-telemetry-processor"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "lambda.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "processor" {
  role = aws_iam_role.processor.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "KinesisConsume"
        Effect = "Allow"
        Action = [
          "kinesis:GetRecords",
          "kinesis:GetShardIterator",
          "kinesis:DescribeStream",
          "kinesis:DescribeStreamSummary",
          "kinesis:ListShards",
          "kinesis:ListStreams",
          "kinesis:SubscribeToShard"
        ]
        Resource = var.kinesis_stream_arn
      },
      {
        Sid    = "SageMakerInvoke"
        Effect = "Allow"
        Action = ["sagemaker:InvokeEndpoint"]
        Resource = "arn:aws:sagemaker:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:endpoint/${var.sagemaker_endpoint_name}"
      },
      {
        Sid    = "DynamoDBWrite"
        Effect = "Allow"
        Action = ["dynamodb:PutItem", "dynamodb:UpdateItem"]
        Resource = var.dynamodb_table_arn
      },
      {
        Sid    = "KMSDecrypt"
        Effect = "Allow"
        Action = ["kms:Decrypt", "kms:GenerateDataKey", "kms:DescribeKey"]
        Resource = [var.kinesis_kms_key_arn, var.dynamodb_kms_key_arn]
      },
      {
        Sid    = "VPCNetworking"
        Effect = "Allow"
        Action = [
          "ec2:CreateNetworkInterface",
          "ec2:DescribeNetworkInterfaces",
          "ec2:DeleteNetworkInterface"
        ]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents"]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.name_prefix}-*"
      },
      {
        Sid    = "XRayTrace"
        Effect = "Allow"
        Action = ["xray:PutTraceSegments", "xray:PutTelemetryRecords"]
        Resource = "*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Lambda function
# ---------------------------------------------------------------------------

# IAM role policies take ~10s to propagate globally before Lambda can use them.
# Without this wait, VPC-attached Lambda creation fails with
# "does not have permissions to call CreateNetworkInterface on EC2".
resource "time_sleep" "iam_propagation" {
  depends_on      = [aws_iam_role_policy.processor]
  create_duration = "15s"
}

resource "aws_lambda_function" "processor" {
  function_name = "${var.name_prefix}-telemetry-processor"
  role          = aws_iam_role.processor.arn
  depends_on    = [time_sleep.iam_propagation]
  runtime       = "python3.12"
  handler       = "src.telemetry_processor.handler.handler"
  filename      = var.deployment_package_path
  timeout       = 60
  memory_size   = 1024

  reserved_concurrent_executions = var.kinesis_shard_count

  environment {
    variables = {
      SAGEMAKER_ENDPOINT_NAME  = var.sagemaker_endpoint_name
      PATIENT_STATE_TABLE      = var.dynamodb_table_name
      ANOMALY_ALERT_THRESHOLD  = tostring(var.anomaly_alert_threshold)
      ECG_SAMPLE_RATE_HZ       = tostring(var.ecg_sample_rate_hz)
      WINDOW_SECONDS           = tostring(var.window_seconds)
      POWERTOOLS_LOG_LEVEL     = var.log_level
      AWS_LAMBDA_LOG_LEVEL     = var.log_level
    }
  }

  vpc_config {
    subnet_ids         = var.private_app_subnet_ids
    security_group_ids = [var.sg_lambda_processor_id]
  }

  tracing_config {
    mode = "Active"
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-telemetry-processor" })
}

# ---------------------------------------------------------------------------
# Kinesis Enhanced Fan-Out event source mapping
# ---------------------------------------------------------------------------

resource "aws_lambda_event_source_mapping" "kinesis_efo" {
  event_source_arn                   = var.kinesis_stream_arn
  function_name                      = aws_lambda_function.processor.arn
  starting_position                  = "LATEST"
  batch_size                         = 100
  parallelization_factor             = 1   # 1 concurrent Lambda per shard
  bisect_batch_on_function_error     = true
  maximum_retry_attempts             = 3
  maximum_record_age_in_seconds      = 3600

  destination_config {
    on_failure {
      destination_arn = aws_sqs_queue.dlq.arn
    }
  }

  filter_criteria {
    filter {
      # Only process records with a known signal_type to drop malformed payloads early
      pattern = jsonencode({
        data = {
          signal_type = ["ECG_LEAD_II", "ECG_V1", "ECG_V5", "EMG_CHANNEL_0", "EMG_CHANNEL_1", "BP_CONTINUOUS"]
        }
      })
    }
  }
}

# ---------------------------------------------------------------------------
# Dead Letter Queue for failed batches
# ---------------------------------------------------------------------------

resource "aws_sqs_queue" "dlq" {
  name                       = "${var.name_prefix}-processor-dlq"
  message_retention_seconds  = 1209600  # 14 days
  kms_master_key_id          = "alias/aws/sqs"

  tags = merge(var.tags, { Name = "${var.name_prefix}-processor-dlq" })
}

# ---------------------------------------------------------------------------
# CloudWatch alarms
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_metric_alarm" "errors" {
  alarm_name          = "${var.name_prefix}-processor-errors"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "Errors"
  namespace           = "AWS/Lambda"
  period              = 60
  statistic           = "Sum"
  threshold           = 0
  alarm_description   = "Telemetry processor Lambda errors"
  alarm_actions       = var.alarm_sns_arn != "" ? [var.alarm_sns_arn] : []

  dimensions = {
    FunctionName = aws_lambda_function.processor.function_name
  }

  tags = var.tags
}

resource "aws_cloudwatch_metric_alarm" "duration_p99" {
  alarm_name          = "${var.name_prefix}-processor-duration-p99"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 3
  extended_statistic  = "p99"
  metric_name         = "Duration"
  namespace           = "AWS/Lambda"
  period              = 60
  threshold           = 45000  # 45s — warn at 75% of 60s timeout
  alarm_description   = "Processor p99 duration approaching timeout"
  alarm_actions       = var.alarm_sns_arn != "" ? [var.alarm_sns_arn] : []

  dimensions = {
    FunctionName = aws_lambda_function.processor.function_name
  }

  tags = var.tags
}
