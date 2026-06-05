import json
import os

import pytest

from src.pipeline import (
    DATA_DIR,
    MODEL_PATH,
    PROCESSED_PATH,
    RAW_PATH,
    REPORT_PATH,
    RUN_LOG_PATH,
    evaluate_model,
    fetch_data,
    preprocess_data,
    run_pipeline,
    train_model,
)


def test_fetch_data_creates_csv():
    fetch_data(n_samples=200)
    assert os.path.exists(RAW_PATH)
    import pandas as pd
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
    import joblib
    bundle = joblib.load(MODEL_PATH)
    assert bundle["params"]["n_estimators"] == 50
    assert bundle["params"]["max_depth"] == 3


def test_evaluate_creates_report():
    fetch_data(n_samples=200)
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
    fetch_data(n_samples=200)
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
    run_pipeline(n_samples=200)
    with open(RUN_LOG_PATH) as f:
        log = json.load(f)
    steps = [e["step"] for e in log]
    assert steps.count("fetch_data") == 1
