output "bucket" {
  value = aws_s3_bucket.data.id
}

output "glue_job_name" {
  value = aws_glue_job.feature_engineering.name
}

output "glue_role_arn" {
  value = aws_iam_role.glue_job.arn
}

output "region" {
  value = var.region
}
