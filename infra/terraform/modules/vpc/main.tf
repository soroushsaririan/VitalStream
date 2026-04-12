###############################################################################
# VPC — clinical data network isolation
#
# All PHI workloads run in private subnets. Public subnets host only NAT
# Gateways; no EC2 instances or Lambda ENIs are placed there.
# All AWS API calls are routed through VPC endpoints (PrivateLink) so no
# PHI traverses the public internet.
###############################################################################

locals {
  azs = slice(data.aws_availability_zones.available.names, 0, 3)
}

data "aws_availability_zones" "available" {
  state = "available"
}

# ---------------------------------------------------------------------------
# VPC
# ---------------------------------------------------------------------------

resource "aws_vpc" "this" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(var.tags, { Name = "${var.name_prefix}-vpc" })
}

# ---------------------------------------------------------------------------
# Internet Gateway (NAT Gateway traffic only; no IGW routes to private subnets)
# ---------------------------------------------------------------------------

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name_prefix}-igw" })
}

# ---------------------------------------------------------------------------
# Public subnets (NAT Gateways only)
# ---------------------------------------------------------------------------

resource "aws_subnet" "public" {
  count             = length(local.azs)
  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 1)
  availability_zone = local.azs[count.index]

  map_public_ip_on_launch = false  # No auto-assign; NAT GW gets EIP explicitly

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-public-${local.azs[count.index]}"
    Tier = "public"
  })
}

# ---------------------------------------------------------------------------
# Private application subnets (Lambda ENIs, ECS tasks)
# ---------------------------------------------------------------------------

resource "aws_subnet" "private_app" {
  count             = length(local.azs)
  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 11)
  availability_zone = local.azs[count.index]

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-private-app-${local.azs[count.index]}"
    Tier = "private-app"
  })
}

# ---------------------------------------------------------------------------
# Private data subnets (VPC endpoint ENIs, RDS if applicable)
# ---------------------------------------------------------------------------

resource "aws_subnet" "private_data" {
  count             = length(local.azs)
  vpc_id            = aws_vpc.this.id
  cidr_block        = cidrsubnet(var.vpc_cidr, 8, count.index + 21)
  availability_zone = local.azs[count.index]

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-private-data-${local.azs[count.index]}"
    Tier = "private-data"
  })
}

# ---------------------------------------------------------------------------
# NAT Gateways (one per AZ for HA; Lambda needs outbound for KMS/SM APIs
# on the rare path not covered by VPC endpoints)
# ---------------------------------------------------------------------------

resource "aws_eip" "nat" {
  count  = length(local.azs)
  domain = "vpc"
  tags   = merge(var.tags, { Name = "${var.name_prefix}-nat-eip-${count.index}" })
}

resource "aws_nat_gateway" "this" {
  count         = length(local.azs)
  allocation_id = aws_eip.nat[count.index].id
  subnet_id     = aws_subnet.public[count.index].id

  tags = merge(var.tags, { Name = "${var.name_prefix}-nat-${local.azs[count.index]}" })
  depends_on = [aws_internet_gateway.this]
}

# ---------------------------------------------------------------------------
# Route tables
# ---------------------------------------------------------------------------

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
  tags = merge(var.tags, { Name = "${var.name_prefix}-rt-public" })
}

resource "aws_route_table_association" "public" {
  count          = length(local.azs)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table" "private_app" {
  count  = length(local.azs)
  vpc_id = aws_vpc.this.id
  route {
    cidr_block     = "0.0.0.0/0"
    nat_gateway_id = aws_nat_gateway.this[count.index].id
  }
  tags = merge(var.tags, {
    Name = "${var.name_prefix}-rt-private-app-${local.azs[count.index]}"
  })
}

resource "aws_route_table_association" "private_app" {
  count          = length(local.azs)
  subnet_id      = aws_subnet.private_app[count.index].id
  route_table_id = aws_route_table.private_app[count.index].id
}

resource "aws_route_table" "private_data" {
  vpc_id = aws_vpc.this.id
  tags   = merge(var.tags, { Name = "${var.name_prefix}-rt-private-data" })
}

resource "aws_route_table_association" "private_data" {
  count          = length(local.azs)
  subnet_id      = aws_subnet.private_data[count.index].id
  route_table_id = aws_route_table.private_data.id
}

# ---------------------------------------------------------------------------
# VPC Gateway Endpoints (free; route-table based; no ENI in subnet)
# ---------------------------------------------------------------------------

resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${data.aws_region.current.name}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = concat(
    aws_route_table.private_app[*].id,
    [aws_route_table.private_data.id]
  )
  tags = merge(var.tags, { Name = "${var.name_prefix}-vpce-s3" })
}

resource "aws_vpc_endpoint" "dynamodb" {
  vpc_id            = aws_vpc.this.id
  service_name      = "com.amazonaws.${data.aws_region.current.name}.dynamodb"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = concat(
    aws_route_table.private_app[*].id,
    [aws_route_table.private_data.id]
  )
  tags = merge(var.tags, { Name = "${var.name_prefix}-vpce-dynamodb" })
}

# ---------------------------------------------------------------------------
# VPC Interface Endpoints (PrivateLink; ENI per AZ in private_data subnets)
# ---------------------------------------------------------------------------

locals {
  interface_endpoints = {
    kinesis_streams = "com.amazonaws.${data.aws_region.current.name}.kinesis-streams"
    secretsmanager  = "com.amazonaws.${data.aws_region.current.name}.secretsmanager"
    kms             = "com.amazonaws.${data.aws_region.current.name}.kms"
    logs            = "com.amazonaws.${data.aws_region.current.name}.logs"
    monitoring      = "com.amazonaws.${data.aws_region.current.name}.monitoring"
  }
}

resource "aws_vpc_endpoint" "interface" {
  for_each = local.interface_endpoints

  vpc_id              = aws_vpc.this.id
  service_name        = each.value
  vpc_endpoint_type   = "Interface"
  subnet_ids          = aws_subnet.private_data[*].id
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = merge(var.tags, { Name = "${var.name_prefix}-vpce-${each.key}" })
}

# ---------------------------------------------------------------------------
# Security Groups
# ---------------------------------------------------------------------------

resource "aws_security_group" "vpc_endpoints" {
  name        = "${var.name_prefix}-sg-vpc-endpoints"
  description = "Allow HTTPS from private app subnets to VPC endpoint ENIs"
  vpc_id      = aws_vpc.this.id

  ingress {
    description = "HTTPS from private app subnets"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = aws_subnet.private_app[*].cidr_block
  }

  egress {
    description = "Allow all outbound (stateful return traffic)"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-sg-vpc-endpoints" })
}

resource "aws_security_group" "lambda_processor" {
  name        = "${var.name_prefix}-sg-lambda-processor"
  description = "Telemetry processor Lambda - no inbound; HTTPS outbound to VPC endpoints"
  vpc_id      = aws_vpc.this.id

  egress {
    description     = "HTTPS to VPC endpoint ENIs"
    from_port       = 443
    to_port         = 443
    protocol        = "tcp"
    security_groups = [aws_security_group.vpc_endpoints.id]
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-sg-lambda-processor" })
}

# ---------------------------------------------------------------------------
# VPC Flow Logs (HIPAA audit requirement)
# ---------------------------------------------------------------------------

resource "aws_flow_log" "this" {
  vpc_id          = aws_vpc.this.id
  traffic_type    = "ALL"
  iam_role_arn    = aws_iam_role.flow_log.arn
  log_destination = aws_cloudwatch_log_group.flow_log.arn

  tags = merge(var.tags, { Name = "${var.name_prefix}-flow-logs" })
}

resource "aws_cloudwatch_log_group" "flow_log" {
  name              = "/aws/vpc/flow-logs/${var.name_prefix}"
  retention_in_days = 90
  # KMS encryption for CW log groups requires a key policy granting logs.amazonaws.com
  # access before the log group is created. Omitting here; encryption handled at rest
  # by CloudWatch's default AWS-managed key for staging.

  tags = var.tags
}

resource "aws_iam_role" "flow_log" {
  name = "${var.name_prefix}-flow-log-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "vpc-flow-logs.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "flow_log" {
  role = aws_iam_role.flow_log.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams"
      ]
      Resource = "*"
    }]
  })
}

data "aws_region" "current" {}
