"""
Drift monitoring DAG.

Runs every 6 hours. Fetches a fresh batch of data, compares it against
the reference snapshot saved at training time, and triggers the training
DAG when drift is detected. This closes the monitor -> decide -> retrain loop.

    fetch_fresh_batch -> check_drift -+-> trigger_retraining (drift)
                                      +-> no_drift           (no drift)
"""

import os
import random
import sys
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import check_drift, fetch_data

default_args = {
    "owner": "yigit",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}


def fetch_fresh_batch():
    # New seed each run simulates a fresh batch of incoming data
    fetch_data(n_samples=2000, seed=random.randint(0, 100_000))


def decide_on_drift() -> str:
    report = check_drift()
    return "trigger_retraining" if report["drift_detected"] else "no_drift"


with DAG(
    dag_id="drift_monitoring",
    description="Checks for data drift and triggers retraining when needed",
    schedule="0 */6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ml", "monitoring", "drift"],
) as dag:

    t1 = PythonOperator(
        task_id="fetch_fresh_batch",
        python_callable=fetch_fresh_batch,
    )

    t2 = BranchPythonOperator(
        task_id="check_drift",
        python_callable=decide_on_drift,
    )

    trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="churn_prediction_pipeline",
        wait_for_completion=False,
    )

    no_drift = EmptyOperator(task_id="no_drift")

    t1 >> t2 >> [trigger, no_drift]
