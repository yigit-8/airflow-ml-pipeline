import json
import os

import pytest

from src.pipeline import (
    DATA_DIR,
    MODEL_PATH,
    PROCESSED_PATH,
    RAW_PATH,
    REPORT_PATH,
    evaluate_model,
    fetch_data,
    preprocess_data,
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


def test_evaluate_creates_report():
    fetch_data(n_samples=200)
    preprocess_data()
    train_model()
    evaluate_model()
    assert os.path.exists(REPORT_PATH)
    with open(REPORT_PATH) as f:
        report = json.load(f)
    assert "status" in report
    assert "metrics" in report


def test_full_pipeline_runs():
    fetch_data(n_samples=500)
    preprocess_data()
    train_model()
    evaluate_model()
    with open(REPORT_PATH) as f:
        report = json.load(f)
    assert report["metrics"]["roc_auc"] > 0.5
