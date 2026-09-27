# Serverless Spark: this resource is only a job DEFINITION. Nothing runs and nothing
# is billed until a job run is started. Each run gets fresh workers, runs the script,
# and releases them. There is no cluster to keep alive or patch.
#
# Cost: G.1X = 1 DPU at $0.44/DPU-hour, billed per second with a 1-minute minimum.
# 2 workers for ~3 minutes is about $0.04 per run.
resource "aws_glue_job" "feature_engineering" {
  name     = local.job_name
  role_arn = aws_iam_role.glue_job.arn

  # Glue 5.0 = Spark 3.5 + Python 3.11: the same Spark major/minor we test locally,
  # so window-function semantics match.
  glue_version = "5.0"

  command {
    name            = "glueetl" # Spark job type (vs "pythonshell" for plain Python)
    script_location = "s3://${aws_s3_bucket.data.id}/${aws_s3_object.feature_script.key}"
    python_version  = "3"
  }

  worker_type       = "G.1X"
  number_of_workers = var.glue_workers

  # Guard rails on cost: kill a hung run after 20 minutes, and don't auto-retry a
  # broken script (a retry would just double the bill for the same failure).
  timeout     = 20
  max_retries = 0

  # Passed to the script as normal CLI flags. It's the same script that runs
  # locally; only the paths change.
  default_arguments = {
    "--input"              = "s3://${aws_s3_bucket.data.id}/raw/telemetry.parquet"
    "--output"             = "s3://${aws_s3_bucket.data.id}/features"
    "--shuffle-partitions" = "8" # 1 executor x 4 cores; 2 tasks per core
    "--job-language"       = "python"
  }

  execution_property {
    max_concurrent_runs = 1 # two runs overwriting features/ at once would corrupt it
  }
}
