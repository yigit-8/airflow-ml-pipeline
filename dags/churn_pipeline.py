"""
Churn prediction ML pipeline DAG.

Runs daily at midnight. Steps:
    1. fetch_data    - generate/pull raw data
    2. preprocess    - scale features
    3. train         - train GradientBoosting, log to MLflow
    4. evaluate      - check quality gate (ROC-AUC >= 0.72)

If evaluate fails, the DAG is marked as failed and the on-call gets notified.
"""

import os
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import evaluate_model, fetch_data, preprocess_data, train_model

default_args = {
    "owner": "yigit",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="churn_prediction_pipeline",
    description="Daily churn prediction training pipeline",
    schedule="0 0 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "churn", "production"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data,
    )

    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data,
    )

    t3 = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        op_kwargs={
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.05,
        },
    )

    t4 = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        op_kwargs={"min_roc_auc": 0.72},
    )

    t1 >> t2 >> t3 >> t4
