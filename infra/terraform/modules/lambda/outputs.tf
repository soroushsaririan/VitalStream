output "function_arn"  { value = aws_lambda_function.processor.arn }
output "function_name" { value = aws_lambda_function.processor.function_name }
output "role_arn"      { value = aws_iam_role.processor.arn }
output "dlq_arn"       { value = aws_sqs_queue.dlq.arn }
