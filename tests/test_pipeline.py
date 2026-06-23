import json
import os

import joblib
import pandas as pd
import pytest

from src.pipeline import (
    DATA_DIR,
    DRIFT_REPORT_PATH,
    MODEL_PATH,
    PROCESSED_PATH,
    RAW_PATH,
    REFERENCE_PATH,
    REPORT_PATH,
    RUN_LOG_PATH,
    check_drift,
    evaluate_model,
    fetch_data,
    preprocess_data,
    run_pipeline,
    train_model,
)


def test_fetch_data_creates_csv():
    fetch_data(n_samples=200)
    assert os.path.exists(RAW_PATH)
    df = pd.read_csv(RAW_PATH)
    assert len(df) == 200
    assert "churn" in df.columns


def test_preprocess_creates_csv():
    fetch_data(n_samples=200)
    preprocess_data()
    assert os.path.exists(PROCESSED_PATH)


def test_train_creates_model():
    fetch_data(n_samples=200)
    preprocess_data()
    train_model()
    assert os.path.exists(MODEL_PATH)


def test_train_custom_hyperparams():
    fetch_data(n_samples=200)
    preprocess_data()
    train_model(n_estimators=50, max_depth=3, learning_rate=0.05)
    bundle = joblib.load(MODEL_PATH)
    assert bundle["params"]["n_estimators"] == 50
    assert bundle["params"]["max_depth"] == 3


def test_evaluate_creates_report():
    fetch_data(n_samples=500)
    preprocess_data()
    train_model()
    report = evaluate_model()
    assert os.path.exists(REPORT_PATH)
    assert "status" in report
    assert "metrics" in report
    assert "quality_gate" in report


def test_evaluate_fails_strict_threshold():
    fetch_data(n_samples=200)
    preprocess_data()
    train_model()
    with pytest.raises(ValueError, match="quality gate"):
        evaluate_model(min_roc_auc=1.0)


def test_run_log_is_written():
    if os.path.exists(RUN_LOG_PATH):
        os.remove(RUN_LOG_PATH)
    fetch_data(n_samples=200)
    assert os.path.exists(RUN_LOG_PATH)
    with open(RUN_LOG_PATH) as f:
        log = json.load(f)
    assert any(entry["step"] == "fetch_data" for entry in log)


def test_run_log_tracks_all_steps():
    if os.path.exists(RUN_LOG_PATH):
        os.remove(RUN_LOG_PATH)
    fetch_data(n_samples=500)
    preprocess_data()
    train_model()
    evaluate_model()
    with open(RUN_LOG_PATH) as f:
        log = json.load(f)
    steps = [entry["step"] for entry in log]
    assert "fetch_data" in steps
    assert "preprocess_data" in steps
    assert "train_model" in steps
    assert "evaluate_model" in steps


def test_train_saves_drift_reference():
    fetch_data(n_samples=300)
    preprocess_data()
    train_model()
    assert os.path.exists(REFERENCE_PATH)
    ref = pd.read_csv(REFERENCE_PATH)
    assert "churn" not in ref.columns


def test_check_drift_same_distribution_no_drift():
    fetch_data(n_samples=1000, seed=1)
    preprocess_data()
    train_model()
    fetch_data(n_samples=1000, seed=2)  # same distribution, new sample
    report = check_drift()
    assert report["drift_detected"] is False
    assert os.path.exists(DRIFT_REPORT_PATH)


def test_check_drift_detects_shifted_data():
    fetch_data(n_samples=1000, seed=1)
    preprocess_data()
    train_model()

    shifted = pd.read_csv(RAW_PATH)
    shifted["monthly_charges"] += 100
    shifted["tenure"] += 36
    shifted["support_calls"] += 5
    shifted["num_products"] += 2
    shifted_path = os.path.join(DATA_DIR, "shifted.csv")
    shifted.to_csv(shifted_path, index=False)

    report = check_drift(current_path=shifted_path)
    assert report["drift_detected"] is True
    assert "monthly_charges" in report["drifted_features"]


def test_check_drift_without_reference_skips():
    if os.path.exists(REFERENCE_PATH):
        os.remove(REFERENCE_PATH)
    fetch_data(n_samples=200)
    report = check_drift()
    assert report["drift_detected"] is False
    assert report["reason"] == "no_reference_data"


def test_run_pipeline():
    report = run_pipeline(n_samples=300)
    assert report["status"] == "pass"
    assert report["metrics"]["roc_auc"] >= 0.70


def test_run_pipeline_custom_params():
    report = run_pipeline(
        n_samples=300,
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
    )
    assert "metrics" in report
    assert "roc_auc" in report["metrics"]


def test_run_pipeline_clears_log():
    fetch_data(n_samples=100)
    run_pipeline(n_samples=500)
    with open(RUN_LOG_PATH) as f:
        log = json.load(f)
    steps = [e["step"] for e in log]
    assert steps.count("fetch_data") == 1
