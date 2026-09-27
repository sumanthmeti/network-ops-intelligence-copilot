# Execution role for SageMaker training jobs and the inference endpoint.
#
# SageMaker assumes this role to: pull the scikit-learn container image from AWS's ECR,
# read training features + code from S3, write the model artifact back, write logs.
# The endpoint uses the same role to download model.tar.gz at startup.
#
# Scoping mirrors the Glue role: separate prefixes, read vs write kept apart.
#   features/            read  (Glue output = training input)
#   sagemaker/code/      read  (sourcedir.tar.gz: train.py, inference.py, features.py)
#   sagemaker/models/    read + write (training writes model.tar.gz; endpoint reads it)

data "aws_iam_policy_document" "sagemaker_trust" {
  statement {
    sid     = "SageMakerServiceAssumesRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [local.account_id]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  name               = "${var.project}-sagemaker"
  description        = "Execution role for anomaly-model training jobs and endpoint"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_trust.json
}

data "aws_iam_policy_document" "sagemaker" {
  statement {
    sid       = "ListOnlyMLPrefixes"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["features/*", "sagemaker/*"]
    }
  }

  statement {
    sid     = "ReadTrainingDataAndCode"
    effect  = "Allow"
    actions = ["s3:GetObject"]
    resources = [
      "${aws_s3_bucket.data.arn}/features/*",
      "${aws_s3_bucket.data.arn}/sagemaker/code/*",
    ]
  }

  statement {
    sid       = "ReadWriteModelArtifacts"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/sagemaker/models/*"]
  }

  # GetAuthorizationToken is account-wide by design (it returns a docker login
  # token), so it can't be scoped to a repository. The actual image pulls can.
  statement {
    sid       = "EcrLogin"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  # AWS publishes the scikit-learn containers from its own account (683313688378
  # in us-east-1). Pull access to that one repository only.
  statement {
    sid    = "PullSklearnImage"
    effect = "Allow"
    actions = [
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchCheckLayerAvailability",
    ]
    resources = ["arn:aws:ecr:${var.region}:683313688378:repository/sagemaker-scikit-learn"]
  }

  statement {
    sid    = "WriteSageMakerLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = [
      "arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/sagemaker/*",
      "arn:aws:logs:${var.region}:${local.account_id}:log-group:/aws/sagemaker/*:log-stream:*",
    ]
  }

  # PutMetricData has no resource ARNs; the namespace condition is the only way to scope it.
  statement {
    sid       = "PublishSageMakerMetrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
    condition {
      test     = "StringLike"
      variable = "cloudwatch:namespace"
      values   = ["/aws/sagemaker/*"]
    }
  }
}

resource "aws_iam_role_policy" "sagemaker" {
  name   = "anomaly-model-least-privilege"
  role   = aws_iam_role.sagemaker.id
  policy = data.aws_iam_policy_document.sagemaker.json
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker.arn
}
