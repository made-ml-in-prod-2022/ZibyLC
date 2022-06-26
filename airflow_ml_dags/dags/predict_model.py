from airflow.models import Variable
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator

from datetime import datetime, timedelta
import logging

import pandas as pd
from joblib import load

logger = logging.getLogger("airflow.task")


def predict(raw_file_location, predictions_file_location, model_file_location):
    clf = load(model_file_location + 'model.joblib')
    X = pd.read_csv(raw_file_location + 'data.csv')
    pd.DataFrame(clf.predict(X)).to_csv(predictions_file_location + 'predictions.csv', index=False)


with DAG(
        'prediction_inference',
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
    mkdir_pred = BashOperator(
        task_id='mkdir_pred',
        bash_command='mkdir -p /opt/airflow/data/predictions/{{ ds }}',
        dag=dag,
    )

    task_pred = PythonOperator(
        task_id='predict',
        python_callable=predict,
        dag=dag,
        op_kwargs={
            'raw_file_location': '/opt/airflow/data/raw/{{ ds }}/',
            'predictions_file_location': '/opt/airflow/data/predictions/{{ ds }}/',
            'model_file_location': '/opt/airflow/data/models/{{ ds }}/',
        },
    )

    mkdir_pred >> task_pred
