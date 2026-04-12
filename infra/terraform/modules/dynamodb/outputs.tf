output "table_name" {
  value = aws_dynamodb_table.patient_state.name
}

output "table_arn" {
  value = aws_dynamodb_table.patient_state.arn
}

output "stream_arn" {
  value = aws_dynamodb_table.patient_state.stream_arn
}

output "kms_key_arn" {
  value = aws_kms_key.dynamodb.arn
}
