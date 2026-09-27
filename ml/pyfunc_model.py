"""MLflow pyfunc wrapper: packages model + threshold + feature derivation as ONE artifact.

Anything that loads the model from the registry (the FastAPI service, a batch job)
calls predict(raw_feature_rows) and gets decisions back. It never re-implements the
threshold or the feature derivation, so it can't get them subtly wrong.
"""

from __future__ import annotations

import joblib
import mlflow.pyfunc
import pandas as pd


class AnomalyDetector(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.bundle = joblib.load(context.artifacts["bundle"])

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        from train import score  # shipped via code_paths
        s, ptype = score(self.bundle["model"], model_input)
        alert = s >= self.bundle["threshold"]
        return pd.DataFrame({
            "anomaly_score": s,
            "alert": alert,
            "predicted_type": [t if a else "normal" for t, a in zip(ptype, alert)],
        })
