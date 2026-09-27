# IAM role the Glue job runs as.
#
# A role has two separate policies, and interview questions usually probe the difference:
#   1. Trust policy       -> WHO may assume the role (here: only the Glue service, only for this account)
#   2. Permission policy  -> WHAT the role may do once assumed (here: specific S3 prefixes + its own logs)
#
# Deliberately NOT using the AWS-managed AWSGlueServiceRole policy: it grants access to
# any bucket named aws-glue-*, EC2 network describe calls, IAM ListRolePolicies and more.
# Everything below is the minimum this one job needs.

# --- 1. Trust policy ---------------------------------------------------------
data "aws_iam_policy_document" "glue_trust" {
  statement {
    sid     = "GlueServiceAssumesRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["glue.amazonaws.com"]
    }

    # Confused-deputy protection: Glue may assume this role only when acting on
    # behalf of THIS account. Without it, in principle another account's Glue job
    # that learned this role's ARN could try to use it.
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "glue_job" {
  name               = "${var.project}-glue-job"
  description        = "Runtime role for the feature-engineering Glue job"
  assume_role_policy = data.aws_iam_policy_document.glue_trust.json
}

# --- 2. Permission policy ----------------------------------------------------
data "aws_iam_policy_document" "glue_job" {
  # Listing is a BUCKET-level action (resource = the bucket ARN, no /*), so it can't
  # be scoped by object ARN. The s3:prefix condition scopes it instead: the job can
  # list inside raw/, scripts/ and features/, and nothing else in the bucket.
  statement {
    sid       = "ListOnlyProjectPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["raw/*", "scripts/*", "features", "features/*"]
    }
  }

  # Read the input data and the script. No Put/Delete here: a bug in the job
  # cannot overwrite or delete its own source data.
  statement {
    sid     = "ReadInputsAndCode"
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/raw/*",
      "${aws_s3_bucket.data.arn}/scripts/*",
    ]
  }

  # Write output. Why each action is needed:
  #   PutObject    - write the Parquet files
  #   DeleteObject - mode("overwrite") deletes the previous run's output, and Spark's
  #                  commit protocol writes to _temporary/ then "renames" (S3 has no
  #                  rename: it is copy + delete)
  #   GetObject    - the job reads its own output back for the sanity-check summary
  statement {
    sid       = "ReadWriteFeatures"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.data.arn}/features/*"]
  }

  # Glue streams driver/executor stdout+stderr to CloudWatch Logs under /aws-glue/.
  # Scoped to this account, this region and that log-group prefix only.
  statement {
    sid    = "WriteGlueLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = [
      "arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws-glue/*",
      "arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws-glue/*:log-stream:*",
    ]
  }
}

# Inline policy (lives and dies with the role) instead of a standalone managed
# policy: nothing else should ever attach these exact permissions.
resource "aws_iam_role_policy" "glue_job" {
  name   = "feature-engineering-least-privilege"
  role   = aws_iam_role.glue_job.id
  policy = data.aws_iam_policy_document.glue_job.json
}
