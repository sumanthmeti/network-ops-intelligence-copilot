"""Run the model comparison, track every run in MLflow, register the champion.

Candidates:
    rules    static thresholds (today's NOC practice): the baseline to beat
    iforest  unsupervised Isolation Forest: no labels needed, can't name the anomaly type
    hgb-*    supervised gradient boosting, varying class-imbalance handling and capacity

Selection: highest validation PR-AUC. The test days are only reported, never used to
pick, so the reported test numbers are an honest estimate.

MLflow layout:
    tracking store  sqlite:///mlflow.db      (runs, params, metrics, model registry)
    artifacts       ./mlartifacts/           (plots, pyfunc model packages)
    view with:      mlflow ui --backend-store-uri sqlite:///mlflow.db

Usage:
    python ml/experiment.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import joblib
import matplotlib
import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from features import CLASSES  # noqa: E402
from pyfunc_model import AnomalyDetector  # noqa: E402
from rules import THRESHOLDS, rule_alerts  # noqa: E402
from train import evaluate, load_features, parse_args, run, split_by_time  # noqa: E402

EXPERIMENT = "router-anomaly-detection"
MODEL_NAME = "noc-anomaly-detector"
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"

CANDIDATES = {
    "iforest":            dict(model_type="iforest"),
    "hgb-unweighted":     dict(class_weight_power=0.0),
    "hgb-sqrt-weighted":  dict(class_weight_power=0.5),
    "hgb-balanced":       dict(class_weight_power=1.0),
    "hgb-small-slow":     dict(class_weight_power=0.5, max_leaf_nodes=15, learning_rate=0.05, max_iter=400),
}


def plots(preds: pd.DataFrame, out: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5, 4))
    PrecisionRecallDisplay.from_predictions(preds["is_anomaly"], preds["anomaly_score"], ax=ax)
    ax.set_title("Precision-recall (test days)")
    pr = out / "pr_curve.png"; fig.savefig(pr, dpi=110, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        preds["anomaly_type"].fillna("normal"), preds["predicted_type"], labels=CLASSES,
        ax=ax, xticks_rotation=35, colorbar=False)
    ax.set_title("True vs predicted type (test days)")
    cm = out / "confusion_matrix.png"; fig.savefig(cm, dpi=110, bbox_inches="tight"); plt.close(fig)
    return [pr, cm]


def false_alert_causes(preds: pd.DataFrame) -> dict:
    """Which benign event caused each false alert: the question on-call actually asks."""
    fp = preds[preds["alert"] & ~preds["is_anomaly"]]
    return fp["benign_event"].fillna("none (noise)").value_counts().to_dict()


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    tmp = ROOT / "artifacts" / "tmp"; tmp.mkdir(parents=True, exist_ok=True)

    df = load_features(ROOT / "data/features")
    results = []

    # --- baseline: static rules (no training, fixed thresholds) ----------------
    _, _, test = split_by_time(df)
    s, alert, ptype = rule_alerts(test)
    with mlflow.start_run(run_name="rules-static-thresholds") as r:
        m = evaluate(test, s, alert, ptype)
        mlflow.log_params({f"threshold_{k}": v for k, v in THRESHOLDS.items()} | {"model_type": "rules"})
        mlflow.log_metrics(m)
        preds = test.assign(anomaly_score=s, alert=alert, predicted_type=ptype)
        for f in plots(preds, tmp):
            mlflow.log_artifact(str(f))
        mlflow.log_dict(false_alert_causes(preds), "false_alert_causes.json")
        results.append(("rules-static-thresholds", r.info.run_id, None, m))

    # --- ML candidates -----------------------------------------------------------
    for name, overrides in CANDIDATES.items():
        p = parse_args([])
        for k, v in overrides.items():
            setattr(p, k, v)
        with mlflow.start_run(run_name=name) as r:
            bundle, m, preds = run(df, p)
            params = {k: v for k, v in vars(p).items() if not k.endswith("_dir")}
            mlflow.log_params(params)
            mlflow.log_metrics({k: v for k, v in m.items() if isinstance(v, (int, float))})
            for f in plots(preds, tmp):
                mlflow.log_artifact(str(f))
            mlflow.log_dict(false_alert_causes(preds), "false_alert_causes.json")

            bundle_path = tmp / f"{name}.joblib"
            joblib.dump(bundle, bundle_path)
            example = preds.drop(columns=["anomaly_score", "alert", "predicted_type"]).head(5)
            detector = AnomalyDetector()
            detector.bundle = bundle  # what load_context does at serving time
            mlflow.pyfunc.log_model(
                name="model",
                python_model=AnomalyDetector(),
                artifacts={"bundle": str(bundle_path)},
                code_paths=[str(HERE / "features.py"), str(HERE / "train.py")],
                # Records the input/output schema, so the registry documents what the model expects.
                signature=infer_signature(example, detector.predict(None, example)),
                input_example=example,
                pip_requirements=["scikit-learn", "pandas", "numpy", "joblib"],
            )
            results.append((name, r.info.run_id, preds, m))
            print(f"{name:22s} val_pr_auc={m['val_pr_auc']:.4f}  test f1={m['f1']:.3f}  "
                  f"precision={m['precision']:.3f} recall={m['recall']:.3f}")

    # --- champion: best validation PR-AUC among ML candidates ----------------------
    ml = [x for x in results if x[2] is not None]
    name, run_id, preds, m = max(ml, key=lambda x: x[3]["val_pr_auc"])
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    client = MlflowClient()
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    client.set_model_version_tag(MODEL_NAME, version, "selected_by", "val_pr_auc")

    # Export champion for the non-MLflow consumers: SageMaker container, Docker image, A/B test.
    out = ROOT / "artifacts"; (out / "model").mkdir(parents=True, exist_ok=True); (out / "output").mkdir(exist_ok=True)
    shutil.copy(tmp / f"{name}.joblib", out / "model" / "model.joblib")
    shutil.copy(HERE / "features.py", out / "model" / "features.py")
    preds.to_parquet(out / "output" / "test_predictions.parquet", index=False)
    (out / "output" / "metrics.json").write_text(json.dumps(m | {"champion": name, "registry_version": version}, indent=2))

    print("\n" + pd.DataFrame(
        [{"run": n, **{k: mm.get(k) for k in ["val_pr_auc", "pr_auc", "f1", "precision", "recall",
                                              "false_alerts_per_router_day", "episode_recall", "recall_latency_spike"]}}
         for n, _, _, mm in results]).to_string(index=False))
    print(f"\nChampion: {name} -> registered {MODEL_NAME} v{version} @champion")


if __name__ == "__main__":
    main()
