output "stream_arn" {
  value = aws_kinesis_stream.telemetry.arn
}

output "stream_name" {
  value = aws_kinesis_stream.telemetry.name
}

output "kms_key_arn" {
  value = aws_kms_key.kinesis.arn
}
