variable "project" {
  description = "Short name used to prefix every resource."
  type        = string
  default     = "noc-copilot"
}

variable "region" {
  description = "AWS region. us-east-1 has the widest Bedrock model availability (Phase 4)."
  type        = string
  default     = "us-east-1"
}

variable "glue_workers" {
  description = "Number of G.1X workers (1 DPU = 4 vCPU / 16 GB each). One is the driver, so 2 = 1 executor."
  type        = number
  default     = 2
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  # S3 bucket names are global across ALL AWS accounts, so include account + region.
  bucket_name = "${var.project}-${local.account_id}-${var.region}"
  job_name    = "${var.project}-feature-engineering"
}
