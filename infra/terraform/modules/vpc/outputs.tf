output "vpc_id" {
  value = aws_vpc.this.id
}

output "private_app_subnet_ids" {
  value = aws_subnet.private_app[*].id
}

output "private_data_subnet_ids" {
  value = aws_subnet.private_data[*].id
}

output "sg_lambda_processor_id" {
  value = aws_security_group.lambda_processor.id
}

output "sg_vpc_endpoints_id" {
  value = aws_security_group.vpc_endpoints.id
}
