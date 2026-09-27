"""Upload telemetry to S3, run the Glue feature-engineering job, and print its output.

Reads the bucket / job name from `terraform output`, so infra and runner can't drift.

Usage:
    python spark_jobs/run_glue_job.py              # upload data + run
    python spark_jobs/run_glue_job.py --skip-upload

AWS CLI equivalent of what this does:
    aws s3 cp data/raw/telemetry.parquet s3://<bucket>/raw/telemetry.parquet
    aws glue start-job-run --job-name noc-copilot-feature-engineering
    aws glue get-job-run --job-name noc-copilot-feature-engineering --run-id <id>
    aws logs get-log-events --log-group-name /aws-glue/jobs/output --log-stream-name <id>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {"SUCCEEDED", "FAILED", "STOPPED", "TIMEOUT", "ERROR"}


def terraform_outputs() -> dict:
    out = subprocess.run(
        ["terraform", f"-chdir={ROOT / 'infra'}", "output", "-json"],
        check=True, capture_output=True, text=True,
    ).stdout
    return {k: v["value"] for k, v in json.loads(out).items()}


def print_driver_output(logs, run_id: str, failed: bool) -> None:
    """Glue writes the driver's stdout to /aws-glue/jobs/output, one stream per run.
    The error group is mostly Spark INFO logging, so only show it on failure."""
    groups = ["/aws-glue/jobs/output"] + (["/aws-glue/jobs/error"] if failed else [])
    for group in groups:
        try:
            events = logs.get_log_events(logGroupName=group, logStreamName=run_id, startFromHead=True)["events"]
        except logs.exceptions.ResourceNotFoundException:
            continue
        lines = [e["message"].rstrip() for e in events]
        if group.endswith("error"):
            lines = [l for l in lines if "ERROR" in l or "Traceback" in l or "Error:" in l][-30:]
        if lines:
            print(f"\n--- {group} ---")
            print("\n".join(lines))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data/raw/telemetry.parquet")
    p.add_argument("--skip-upload", action="store_true")
    a = p.parse_args()

    tf = terraform_outputs()
    session = boto3.Session(region_name=tf["region"])
    s3, glue, logs = session.client("s3"), session.client("glue"), session.client("logs")
    bucket, job = tf["bucket"], tf["glue_job_name"]

    if not a.skip_upload:
        key = "raw/telemetry.parquet"
        print(f"Uploading {a.data} ({a.data.stat().st_size / 1e6:.1f} MB) -> s3://{bucket}/{key}")
        s3.upload_file(str(a.data), bucket, key)

    run_id = glue.start_job_run(JobName=job)["JobRunId"]
    print(f"Started {job} run {run_id}")

    t0 = time.time()
    state = None
    while state not in TERMINAL:
        time.sleep(15)
        run = glue.get_job_run(JobName=job, RunId=run_id)["JobRun"]
        if run["JobRunState"] != state:
            state = run["JobRunState"]
            print(f"  [{time.time() - t0:5.0f}s] {state}")

    print(f"\nFinal state: {state}")
    print(f"Execution time: {run.get('ExecutionTime', 0)}s, "
          f"DPU-seconds: {run.get('DPUSeconds', 'n/a')}, "
          f"~cost: ${run.get('ExecutionTime', 0) / 3600 * run.get('MaxCapacity', 2) * 0.44:.3f}")
    if run.get("ErrorMessage"):
        print(f"Error: {run['ErrorMessage']}")
    print_driver_output(logs, run_id, failed=state != "SUCCEEDED")
    return 0 if state == "SUCCEEDED" else 1


if __name__ == "__main__":
    sys.exit(main())
