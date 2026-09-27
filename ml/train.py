"""Train the router-anomaly model. SageMaker script-mode compatible.

SageMaker contract (what the training container provides):
    SM_CHANNEL_FEATURES   dir with the S3 "features" channel copied in (Glue output, date=... partitions)
    SM_MODEL_DIR          dir whose contents SageMaker tars into model.tar.gz -> S3
    SM_OUTPUT_DATA_DIR    dir whose contents become output.tar.gz (metrics, test predictions)
    hyperparameters       passed as --key value CLI args
Locally the same env vars default to project folders, so this exact file runs in both.

Model: HistGradientBoostingClassifier, multi-class (normal + 4 anomaly types).
    - handles NaN natively (early rows have no z-score history; we don't impute)
    - the anomaly score is 1 - P(normal); the predicted class feeds the agent's routing
    - class imbalance (~1% anomalies) via sample weights, tunable strength

Evaluation is split BY TIME (train on early days, pick the threshold on the next day,
test on the last two). A random split would leak: rolling-window features of
neighbouring rows overlap, and rows of one anomaly episode would land on both sides.

Usage (local):
    python ml/train.py
    python ml/train.py --model-type iforest
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import CLASSES, FEATURES, NORMAL, labels, to_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- data
def load_features(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)  # reads the date=... partition dirs Spark/Glue wrote
    df["date"] = df["date"].astype(str)
    return df.sort_values(["router_id", "timestamp"]).reset_index(drop=True)


def split_by_time(df: pd.DataFrame, test_days: int = 2, val_days: int = 1):
    dates = sorted(df["date"].unique())
    test_dates = dates[-test_days:]
    val_dates = dates[-(test_days + val_days):-test_days]
    train = df[~df["date"].isin(test_dates + val_dates)]
    return train, df[df["date"].isin(val_dates)], df[df["date"].isin(test_dates)]


# --------------------------------------------------------------------------- model
def class_weights(y: np.ndarray, power: float) -> np.ndarray:
    """power=0: no reweighting, 1: fully balanced, 0.5: sqrt, a middle ground.
    Fully balanced maximises recall but floods on-call with false alerts."""
    classes, counts = np.unique(y, return_counts=True)
    w = {c: (len(y) / (len(classes) * n)) ** power for c, n in zip(classes, counts)}
    return np.array([w[c] for c in y])


def fit(train: pd.DataFrame, p: argparse.Namespace):
    X = to_matrix(train)
    if p.model_type == "hgb":
        y = labels(train)
        model = HistGradientBoostingClassifier(
            learning_rate=p.learning_rate,
            max_iter=p.max_iter,
            max_leaf_nodes=p.max_leaf_nodes,
            l2_regularization=p.l2_regularization,
            early_stopping=False,
            random_state=42,
        )
        model.fit(X, y, sample_weight=class_weights(y, p.class_weight_power))
    elif p.model_type == "iforest":
        # Unsupervised: never sees labels. Realistic when incidents aren't labeled,
        # but it can only say "unusual", not which kind of anomaly.
        model = IsolationForest(n_estimators=200, random_state=42, n_jobs=-1)
        model.fit(np.nan_to_num(X, nan=0.0))
    else:
        raise ValueError(p.model_type)
    return model


def score(model, df: pd.DataFrame):
    """Returns (anomaly_score in [0,1]-ish, predicted_type)."""
    X = to_matrix(df)
    if isinstance(model, HistGradientBoostingClassifier):
        proba = model.predict_proba(X)
        normal_idx = list(model.classes_).index(NORMAL)
        # Sum the anomaly-class probabilities directly rather than 1 - P(normal):
        # when P(normal) rounds to 1.0 the subtraction gives exactly 0 and ranking is lost.
        anomaly_score = np.delete(proba, normal_idx, axis=1).sum(axis=1)
        # Most likely NON-normal class, used when the score crosses the threshold.
        masked = proba.copy()
        masked[:, normal_idx] = -1
        ptype = np.asarray(model.classes_)[masked.argmax(axis=1)]
        return anomaly_score, ptype
    raw = -model.score_samples(np.nan_to_num(X, nan=0.0))  # higher = more anomalous
    return raw, np.full(len(df), "unknown", dtype=object)


def pick_threshold(y_true: np.ndarray, s: np.ndarray) -> float:
    """Threshold that maximises F1 on the validation day (never on test)."""
    prec, rec, thr = precision_recall_curve(y_true, s)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])])


# --------------------------------------------------------------------------- metrics
def evaluate(df: pd.DataFrame, s: np.ndarray, pred: np.ndarray, ptype: np.ndarray) -> dict:
    y = df["is_anomaly"].to_numpy()
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum())
    router_days = df["router_id"].nunique() * df["date"].nunique()

    # Episode-level: an incident counts as caught if ANY of its samples alerts.
    # This is what on-call actually experiences.
    # Detection delay: samples from episode start to the first alert (df is time-sorted per router).
    caught, delays = [], []
    for _, flags in df.assign(pred=pred)[df["episode_id"] >= 0].groupby("episode_id")["pred"]:
        f = flags.to_numpy()
        caught.append(f.any())
        if f.any():
            delays.append(int(np.argmax(f)))
    caught = np.array(caught)

    m = {
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "f1": float(f1_score(y, pred)),
        "pr_auc": float(average_precision_score(y, s)),
        "false_alert_ratio": fp / max(tp + fp, 1),  # share of fired alerts that were noise
        "false_alerts_per_router_day": fp / router_days,
        "episode_recall": float(caught.mean()),
        "median_detect_delay_intervals": float(np.median(delays)) if delays else float("nan"),
        "alerts": tp + fp,
    }
    # Type accuracy among correctly-alerted anomalous rows (routing quality for the agent).
    hit = pred & y
    if hit.any() and not (ptype[hit] == "unknown").all():
        m["type_accuracy"] = float((ptype[hit] == df.loc[hit, "anomaly_type"].to_numpy()).mean())
    for c in CLASSES[1:]:
        mask = (df["anomaly_type"] == c).to_numpy()
        m[f"recall_{c}"] = float(pred[mask].mean()) if mask.any() else float("nan")
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in m.items()}


def run(df: pd.DataFrame, p: argparse.Namespace):
    """Full train/threshold/test cycle. Returns (bundle, metrics, test_predictions)."""
    train, val, test = split_by_time(df)
    t0 = time.time()
    model = fit(train, p)
    train_seconds = time.time() - t0

    val_s, _ = score(model, val)
    threshold = pick_threshold(val["is_anomaly"].to_numpy(), val_s)

    test_s, test_type = score(model, test)
    test_pred = test_s >= threshold
    metrics = evaluate(test, test_s, test_pred, test_type)
    # Model SELECTION uses validation PR-AUC (threshold-free, comparable across model
    # types). Test metrics are only reported, never used to choose.
    metrics["val_pr_auc"] = round(float(average_precision_score(val["is_anomaly"], val_s)), 5)
    metrics.update(threshold=round(threshold, 6), train_seconds=round(train_seconds, 2),
                   train_rows=len(train), test_rows=len(test))

    bundle = {
        "model": model,
        "threshold": threshold,
        "features": FEATURES,
        "classes": CLASSES,
        "model_type": p.model_type,
        "sklearn_version": sklearn.__version__,
    }
    preds = test.assign(anomaly_score=test_s, alert=test_pred,
                        predicted_type=np.where(test_pred, test_type, NORMAL))
    return bundle, metrics, preds


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-type", choices=["hgb", "iforest"], default="hgb")
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--max-leaf-nodes", type=int, default=31)
    # Must be > 0. Boosting leaf values are Newton steps: gradient / (hessian + l2).
    # With ~99% "normal" rows, near-pure leaves have hessian p(1-p) ~ 0, and with l2=0
    # the steps explode (we observed raw scores of 1.4 million and a useless model).
    p.add_argument("--l2-regularization", type=float, default=1.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--features-dir", default=os.environ.get("SM_CHANNEL_FEATURES", str(ROOT / "data/features")))
    p.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", str(ROOT / "artifacts/model")))
    p.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", str(ROOT / "artifacts/output")))
    args, _ = p.parse_known_args(argv)  # SageMaker may pass extra args
    return args


def main() -> None:
    p = parse_args()
    df = load_features(p.features_dir)
    bundle, metrics, preds = run(df, p)

    model_dir, output_dir = Path(p.model_dir), Path(p.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "model.joblib")
    # features.py travels with the model so the serving container uses the exact same code.
    (model_dir / "features.py").write_text((Path(__file__).parent / "features.py").read_text())
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    preds.to_parquet(output_dir / "test_predictions.parquet", index=False)

    # "name=value;" lines: SageMaker metric_definitions regexes scrape these from
    # the logs into CloudWatch metrics for the training job.
    for k, v in metrics.items():
        print(f"metric {k}={v};")


if __name__ == "__main__":
    main()
