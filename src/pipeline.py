"""
Core pipeline steps called by the Airflow DAG.

Each function is a standalone step that reads from and writes to
the shared data directory. This keeps the DAG clean and the logic testable.
"""

import json
import os

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


def fetch_data(n_samples: int = 2000, seed: int = 42) -> None:
    """Generate synthetic customer churn data and save to data/raw.csv."""
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
    print(f"Fetched {len(df)} rows -> {RAW_PATH}")


def preprocess_data() -> None:
    """Scale numeric features and save to data/processed.csv."""
    df = pd.read_csv(RAW_PATH)

    features = ["tenure", "monthly_charges", "num_products",
                "has_internet", "support_calls", "contract_months"]
    scaler = StandardScaler()
    df[features] = scaler.fit_transform(df[features])

    df.to_csv(PROCESSED_PATH, index=False)
    joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.joblib"))
    print(f"Preprocessed data -> {PROCESSED_PATH}")


def train_model() -> None:
    """Train GradientBoosting classifier and log run to MLflow."""
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["churn"])
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("airflow-churn-pipeline")
    with mlflow.start_run():
        params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}
        mlflow.log_params(params)

        model = GradientBoostingClassifier(**params, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }
        mlflow.log_metrics(metrics)

        joblib.dump({"model": model, "metrics": metrics}, MODEL_PATH)
        print(f"Model trained. Metrics: {metrics}")


def evaluate_model() -> None:
    """Load the trained model, run evaluation, and save a JSON report."""
    bundle = joblib.load(MODEL_PATH)
    metrics = bundle["metrics"]

    report = {
        "status": "pass" if metrics["roc_auc"] >= 0.70 else "fail",
        "metrics": metrics,
    }

    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Evaluation report: {report}")
    if report["status"] == "fail":
        raise ValueError(f"Model failed quality gate. ROC-AUC: {metrics['roc_auc']:.4f} < 0.70")
