from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.log.logging_mixin import LoggingMixin

from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification

logger = logging.getLogger("airflow.task")
RANDOM_STATE = 57


def generate_features_and_target(raw_file_location):
    X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, random_state=RANDOM_STATE,
                               n_clusters_per_class=1)
    rng = np.random.RandomState(RANDOM_STATE)
    X += 2 * rng.uniform(size=X.shape)

    pd.DataFrame(X).to_csv(raw_file_location + 'data.csv', index=False)
    pd.DataFrame(y).to_csv(raw_file_location + 'target.csv', index=False)
    LoggingMixin().log.info(f"Samples are stored in {raw_file_location}")


with DAG(
        'prepare_data',
        schedule_interval='@daily',
        catchup=False,
        max_active_runs=1,
        default_args={
            'owner': 'airflow',
            'depends_on_past': False,
            'start_date': datetime(2022, 6, 20),
            'retries': 0,
            'retry_delay': timedelta(seconds=15)
        }
) as dag:
    mkdir_raw = BashOperator(
        task_id='mkdir_raw',
        bash_command='mkdir -p /opt/airflow/data/raw/{{ ds }}',
        dag=dag,
    )

    task_generate_features_and_target = PythonOperator(
        task_id='build_project',
        python_callable=generate_features_and_target,
        dag=dag,
        op_kwargs={'raw_file_location': '/opt/airflow/data/raw/{{ ds }}/'},
    )

    mkdir_raw >> task_generate_features_and_target
