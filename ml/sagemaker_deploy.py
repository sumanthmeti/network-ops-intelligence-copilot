"""Train and deploy the anomaly model on SageMaker using the raw boto3 API.

Deliberately not the high-level `sagemaker` SDK: every step below is one SageMaker
API call, so each piece can be explained (what a training job is, what a Model vs
EndpointConfig vs Endpoint is, where the container comes from).

    python ml/sagemaker_deploy.py train     # CreateTrainingJob on the Glue features in S3
    python ml/sagemaker_deploy.py deploy    # CreateModel -> CreateEndpointConfig (serverless) -> CreateEndpoint
    python ml/sagemaker_deploy.py invoke    # InvokeEndpoint with real test rows
    python ml/sagemaker_deploy.py delete    # delete endpoint + config + model (stops all cost)

Serverless inference: the endpoint scales to zero between requests and bills per
request-duration, so an idle demo endpoint costs nothing (unlike an ml.m5 instance
endpoint, which bills every hour it exists). Trade-off: cold starts of several seconds.

Account requirements: SageMaker quotas for "ml.m5.large for training job usage" and
"Maximum number of serverless endpoints" must be >= 1 (new accounts start at 0), and
the AWS Free plan blocks SageMaker compute entirely.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tarfile
import time
from pathlib import Path

import boto3
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "spark_jobs"))
from run_glue_job import terraform_outputs  # noqa: E402

# AWS-published scikit-learn container. Same image for training and serving.
SKLEARN_IMAGE = "683313688378.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3"
NAME = "noc-anomaly-detector"
STATE = ROOT / "artifacts" / "sagemaker_state.json"  # last training job name, for deploy


def sourcedir_tar() -> bytes:
    """Code SageMaker runs: train.py (training), inference.py (serving), shared features.py."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in ["train.py", "inference.py", "features.py"]:
            tar.add(HERE / f, arcname=f)
        tar.add(HERE / "sagemaker_requirements.txt", arcname="requirements.txt")
    return buf.getvalue()


def train(sm, s3, tf, a) -> None:
    bucket, region = tf["bucket"], tf["region"]
    code_key = "sagemaker/code/sourcedir.tar.gz"
    s3.put_object(Bucket=bucket, Key=code_key, Body=sourcedir_tar())
    job = f"{NAME}-{time.strftime('%Y%m%d-%H%M%S')}"

    # Script mode: the SDK convention is JSON-encoded hyperparameter values.
    hp = {
        "sagemaker_program": "train.py",
        "sagemaker_submit_directory": f"s3://{bucket}/{code_key}",
        "sagemaker_region": region,
        "class-weight-power": 1.0,  # champion config from ml/experiment.py
    }
    req = dict(
        TrainingJobName=job,
        RoleArn=tf["sagemaker_role_arn"],
        AlgorithmSpecification={
            "TrainingImage": SKLEARN_IMAGE.format(region=region),
            "TrainingInputMode": "File",  # copy the channel to local disk before start
            # Scrape the "metric name=value;" lines train.py prints into CloudWatch.
            "MetricDefinitions": [{"Name": m, "Regex": f"metric {m}=([0-9.eE+-]+);"}
                                  for m in ["f1", "precision", "recall", "pr_auc", "val_pr_auc"]],
        },
        HyperParameters={k: json.dumps(v) for k, v in hp.items()},
        # The Glue job's output IS the training input: no copy step in between.
        InputDataConfig=[{
            "ChannelName": "features",  # -> SM_CHANNEL_FEATURES=/opt/ml/input/data/features
            "DataSource": {"S3DataSource": {"S3DataType": "S3Prefix",
                                            "S3Uri": f"s3://{bucket}/features/",
                                            "S3DataDistributionType": "FullyReplicated"}},
        }],
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/sagemaker/models/"},
        ResourceConfig={"InstanceType": "ml.m5.large", "InstanceCount": 1, "VolumeSizeInGB": 5},
        StoppingCondition={"MaxRuntimeInSeconds": 1800},
    )
    if a.spot:
        # Managed spot: up to ~70% cheaper; the job may wait for capacity or be interrupted.
        req["EnableManagedSpotTraining"] = True
        req["StoppingCondition"]["MaxWaitTimeInSeconds"] = 3600

    sm.create_training_job(**req)
    print(f"Started training job {job}")
    sm.get_waiter("training_job_completed_or_stopped").wait(TrainingJobName=job)
    d = sm.describe_training_job(TrainingJobName=job)
    print(f"Status: {d['TrainingJobStatus']}  billable seconds: {d.get('BillableTimeInSeconds')}")
    for m in d.get("FinalMetricDataList", []):
        print(f"  {m['MetricName']} = {m['Value']:.4f}")
    if d["TrainingJobStatus"] != "Completed":
        sys.exit(d.get("FailureReason", "training failed"))
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"training_job": job, "model_data": d["ModelArtifacts"]["S3ModelArtifacts"],
                                 "code": f"s3://{bucket}/{code_key}"}))


def deploy(sm, tf, a) -> None:
    st = json.loads(STATE.read_text())
    suffix = time.strftime("%Y%m%d-%H%M%S")
    model, config = f"{NAME}-{suffix}", f"{NAME}-serverless-{suffix}"

    # Model = image + artifact + role. Just a pointer; nothing runs yet.
    sm.create_model(
        ModelName=model,
        ExecutionRoleArn=tf["sagemaker_role_arn"],
        PrimaryContainer={
            "Image": SKLEARN_IMAGE.format(region=tf["region"]),
            "ModelDataUrl": st["model_data"],
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py",
                            "SAGEMAKER_SUBMIT_DIRECTORY": st["code"]},
        },
    )
    # EndpointConfig = how to host it. Serverless: memory size + max concurrency, no instances.
    sm.create_endpoint_config(
        EndpointConfigName=config,
        ProductionVariants=[{"VariantName": "AllTraffic", "ModelName": model,
                             "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 1}}],
    )
    # Endpoint = the live HTTPS resource. Updating it to a new config later is a
    # zero-downtime deploy: SageMaker brings up the new variant before switching.
    try:
        sm.describe_endpoint(EndpointName=NAME)
        sm.update_endpoint(EndpointName=NAME, EndpointConfigName=config)
        print(f"Updating endpoint {NAME} -> {config}")
    except sm.exceptions.ClientError:
        sm.create_endpoint(EndpointName=NAME, EndpointConfigName=config)
        print(f"Creating endpoint {NAME}")
    sm.get_waiter("endpoint_in_service").wait(EndpointName=NAME)
    print("Endpoint InService")


def invoke(runtime, a) -> None:
    preds = pd.read_parquet(ROOT / "artifacts/output/test_predictions.parquet")
    rows = pd.concat([preds[preds.is_anomaly].groupby("anomaly_type").nth(3), preds[~preds.is_anomaly].head(2)])
    body = json.dumps({"instances": json.loads(
        rows.drop(columns=["anomaly_score", "alert", "predicted_type"]).to_json(orient="records", date_format="iso"))})
    t0 = time.time()
    r = runtime.invoke_endpoint(EndpointName=NAME, ContentType="application/json", Body=body)
    out = json.loads(r["Body"].read())["predictions"]
    print(f"Latency {time.time() - t0:.2f}s (first call after idle includes a serverless cold start)")
    for truth, p in zip(rows["anomaly_type"].fillna("normal"), out):
        print(f"  true={truth:18s} -> {p}")


def delete(sm, a) -> None:
    try:
        cfg = sm.describe_endpoint(EndpointName=NAME)["EndpointConfigName"]
        model = sm.describe_endpoint_config(EndpointConfigName=cfg)["ProductionVariants"][0]["ModelName"]
    except sm.exceptions.ClientError:
        print("No endpoint to delete"); return
    sm.delete_endpoint(EndpointName=NAME)
    sm.delete_endpoint_config(EndpointConfigName=cfg)
    sm.delete_model(ModelName=model)
    print(f"Deleted endpoint {NAME}, config {cfg}, model {model}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["train", "deploy", "invoke", "delete"])
    p.add_argument("--spot", action="store_true", help="managed spot training")
    a = p.parse_args()
    tf = terraform_outputs()
    session = boto3.Session(region_name=tf["region"])
    sm = session.client("sagemaker")
    {"train": lambda: train(sm, session.client("s3"), tf, a),
     "deploy": lambda: deploy(sm, tf, a),
     "invoke": lambda: invoke(session.client("sagemaker-runtime"), a),
     "delete": lambda: delete(sm, a)}[a.action]()


if __name__ == "__main__":
    main()
