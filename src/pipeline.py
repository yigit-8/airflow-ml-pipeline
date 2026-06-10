"""
Core pipeline steps called by the Airflow DAG.

Each function is a standalone step that reads from and writes to
the shared data directory. This keeps the DAG clean and the logic testable.
"""

import json
import os
import time

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH = os.path.join(DATA_DIR, "raw.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model.joblib")
REPORT_PATH = os.path.join(DATA_DIR, "report.json")
RUN_LOG_PATH = os.path.join(DATA_DIR, "run_log.json")
REFERENCE_PATH = os.path.join(DATA_DIR, "reference.csv")
DRIFT_REPORT_PATH = os.path.join(DATA_DIR, "drift_report.json")

DEFAULT_MIN_ROC_AUC = 0.70


def _append_run_log(step: str, status: str, details: dict = None):
    logs = []
    if os.path.exists(RUN_LOG_PATH):
        with open(RUN_LOG_PATH) as f:
            logs = json.load(f)
    logs.append({
        "step": step,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **(details or {}),
    })
    with open(RUN_LOG_PATH, "w") as f:
        json.dump(logs, f, indent=2)


def fetch_data(n_samples: int = 2000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame({
        "tenure": rng.integers(1, 72, n_samples),
        "monthly_charges": rng.uniform(20, 120, n_samples).round(2),
        "num_products": rng.integers(1, 6, n_samples),
        "has_internet": rng.integers(0, 2, n_samples),
        "support_calls": rng.integers(0, 10, n_samples),
        "contract_months": rng.choice([1, 12, 24], n_samples),
    })

    churn_prob = (
        0.05
        + 0.30 * (df["contract_months"] == 1)
        + 0.15 * (df["monthly_charges"] > 80)
        - 0.10 * (df["tenure"] > 24)
        + 0.05 * (df["support_calls"] > 5)
    ).clip(0.02, 0.95)

    df["churn"] = rng.binomial(1, churn_prob).astype(int)

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(RAW_PATH, index=False)
    _append_run_log("fetch_data", "success", {"rows": n_samples})
    print(f"Fetched {len(df)} rows -> {RAW_PATH}")


def preprocess_data() -> None:
    df = pd.read_csv(RAW_PATH)

    features = ["tenure", "monthly_charges", "num_products",
                "has_internet", "support_calls", "contract_months"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    df.to_csv(PROCESSED_PATH, index=False)
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.joblib"))
    _append_run_log("preprocess_data", "success", {"features": features})
    print(f"Preprocessed data -> {PROCESSED_PATH}")


def train_model(
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> None:
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("airflow-churn-pipeline")
    with mlflow.start_run():
        params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
        }
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "f1": round(f1_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        }
        mlflow.log_metrics(metrics)

        joblib.dump({"model": model, "metrics": metrics, "params": params}, MODEL_PATH)

        # Snapshot the raw training features as the drift reference
        pd.read_csv(RAW_PATH).drop(columns=["churn"]).to_csv(REFERENCE_PATH, index=False)

        _append_run_log("train_model", "success", {"metrics": metrics})
        print(f"Model trained. Metrics: {metrics}")


def evaluate_model(min_roc_auc: float = DEFAULT_MIN_ROC_AUC) -> dict:
    bundle = joblib.load(MODEL_PATH)
    metrics = bundle["metrics"]

    passed = metrics["roc_auc"] >= min_roc_auc
    report = {
        "status": "pass" if passed else "fail",
        "metrics": metrics,
        "quality_gate": {"min_roc_auc": min_roc_auc},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    _append_run_log("evaluate_model", report["status"], {"roc_auc": metrics["roc_auc"]})
    print(f"Evaluation: {report['status']} (ROC-AUC: {metrics['roc_auc']})")

    if not passed:
        raise ValueError(
            f"Model failed quality gate. ROC-AUC: {metrics['roc_auc']:.4f} < {min_roc_auc}"
        )
    return report


def check_drift(
    current_path: str = RAW_PATH,
    p_threshold: float = 0.05,
    drift_share: float = 0.5,
) -> dict:
    """Compare the latest data batch against the training reference.

    Runs a Kolmogorov-Smirnov test per feature. The dataset is considered
    drifted when at least `drift_share` of the features reject the null
    hypothesis at `p_threshold`.
    """
    from scipy.stats import ks_2samp

    if not os.path.exists(REFERENCE_PATH):
        report = {"drift_detected": False, "reason": "no_reference_data"}
        _append_run_log("check_drift", "skipped", report)
        return report

    reference = pd.read_csv(REFERENCE_PATH)
    current = pd.read_csv(current_path)
    features = list(reference.columns)

    drifted = []
    p_values = {}
    for feature in features:
        _, p_value = ks_2samp(reference[feature], current[feature])
        p_values[feature] = round(float(p_value), 6)
        if p_value < p_threshold:
            drifted.append(feature)

    drift_detected = len(drifted) / len(features) >= drift_share
    report = {
        "drift_detected": drift_detected,
        "drifted_features": drifted,
        "p_values": p_values,
        "thresholds": {"p_threshold": p_threshold, "drift_share": drift_share},
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    with open(DRIFT_REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    _append_run_log("check_drift", "drift" if drift_detected else "no_drift",
                    {"drifted_features": drifted})
    print(f"Drift detected: {drift_detected} (drifted: {drifted})")
    return report


def run_pipeline(
    n_samples: int = 2000,
    min_roc_auc: float = DEFAULT_MIN_ROC_AUC,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
) -> dict:
    if os.path.exists(RUN_LOG_PATH):
        os.remove(RUN_LOG_PATH)

    fetch_data(n_samples=n_samples)
    preprocess_data()
    train_model(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate)
    return evaluate_model(min_roc_auc=min_roc_auc)
