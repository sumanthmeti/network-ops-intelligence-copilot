"""Anomaly detector backends: the trained model, loaded locally or called on SageMaker.

    LocalDetector       loads artifacts/model/model.joblib (the MLflow champion export)
    SageMakerDetector   calls the serverless endpoint (ml/sagemaker_deploy.py) over HTTPS
Both return the same shape, so the agent is indifferent to where the model runs.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class LocalDetector:
    def __init__(self, model_path: str | Path | None = None):
        path = Path(model_path or os.environ.get("MODEL_PATH", ROOT / "artifacts/model/model.joblib"))
        sys.path.insert(0, str(path.parent))        # features.py ships next to the model
        sys.path.insert(0, str(ROOT / "ml"))
        self.bundle = joblib.load(path)
        from train import score
        self._score = score

    def predict(self, rows: pd.DataFrame) -> list[dict]:
        s, ptype = self._score(self.bundle["model"], rows)
        alert = s >= self.bundle["threshold"]
        return [{"anomaly_score": float(x), "alert": bool(a), "predicted_type": str(t) if a else "normal"}
                for x, a, t in zip(s, alert, ptype)]


class SageMakerDetector:
    def __init__(self, endpoint: str = "noc-anomaly-detector", region: str | None = None):
        import boto3
        self.endpoint = endpoint
        self.runtime = boto3.client("sagemaker-runtime", region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def predict(self, rows: pd.DataFrame) -> list[dict]:
        body = json.dumps({"instances": json.loads(rows.to_json(orient="records", date_format="iso"))})
        r = self.runtime.invoke_endpoint(EndpointName=self.endpoint, ContentType="application/json", Body=body)
        return json.loads(r["Body"].read())["predictions"]


def make_detector():
    return SageMakerDetector() if os.environ.get("DETECTOR_BACKEND") == "sagemaker" else LocalDetector()
