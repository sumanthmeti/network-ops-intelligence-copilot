# One bucket, separated by prefix. The prefixes are the security boundary: the IAM
# policy in iam.tf grants different access to each one.
#
#   raw/       input telemetry           job: read only
#   scripts/   PySpark code Glue runs    job: read only
#   features/  engineered output         job: read + write
resource "aws_s3_bucket" "data" {
  bucket = local.bucket_name

  # Lets `terraform destroy` delete the bucket even if it still holds objects.
  # Right for a demo project you tear down between sessions; in production you
  # would leave this false so a destroy can never wipe real data.
  force_destroy = true
}

# Block every form of public access, at the bucket level. Nothing in this project
# should ever be readable from the internet.
resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = true
  restrict_public_buckets = true
}

# Disable ACLs entirely: access is controlled only by IAM policies, one model to reason about.
resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Encryption at rest with S3-managed keys (SSE-S3). S3 does this by default now;
# stating it makes the choice explicit. SSE-KMS would add key-level audit logs
# and access control, at a per-request cost.
resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Failed multipart uploads leave invisible, billable parts behind. Clean them up.
resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    id     = "abort-incomplete-multipart"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# The Spark script is code, so Terraform deploys it. `etag` makes Terraform
# re-upload whenever the local file changes. The telemetry DATA is not uploaded
# here: data is not infrastructure, and it changes every time we regenerate it.
resource "aws_s3_object" "feature_script" {
  bucket = aws_s3_bucket.data.id
  key    = "scripts/feature_engineering.py"
  source = "${path.module}/../spark_jobs/feature_engineering.py"
  etag   = filemd5("${path.module}/../spark_jobs/feature_engineering.py")
}
