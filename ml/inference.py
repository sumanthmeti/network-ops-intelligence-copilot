"""SageMaker scikit-learn container serving hooks.

The container's model server calls these four functions per request:
    model_fn(model_dir)                   once at startup: load model.tar.gz contents
    input_fn(body, content_type)          request bytes -> DataFrame
    predict_fn(data, model)               DataFrame -> predictions
    output_fn(prediction, accept)         predictions -> response bytes

Request:  {"instances": [{<feature columns from the Spark job>}, ...]}
Response: {"predictions": [{"anomaly_score": .., "alert": .., "predicted_type": ..}, ...]}
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def model_fn(model_dir):
    sys.path.insert(0, model_dir)  # features.py ships inside model.tar.gz with the model
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(body, content_type="application/json"):
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type {content_type}; send application/json")
    payload = json.loads(body)
    instances = payload["instances"] if isinstance(payload, dict) else payload
    return pd.DataFrame(instances)


def predict_fn(data, bundle):
    from train import score  # same scoring code as training/evaluation
    s, ptype = score(bundle["model"], data)
    alert = s >= bundle["threshold"]
    return [
        {"anomaly_score": float(sc), "alert": bool(a), "predicted_type": str(t) if a else "normal"}
        for sc, a, t in zip(s, alert, ptype)
    ]


def output_fn(prediction, accept="application/json"):
    return json.dumps({"predictions": prediction}), "application/json"
