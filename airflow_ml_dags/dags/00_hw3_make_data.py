
from datetime import timedelta, datetime

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

AIRFLOW_DATA_PATH = "/home/sokolov/prod/sokoloid/airflow_ml_dags/data"

default_args = {'owner': 'airflow',
                'start_date': datetime(2018, 1, 30),
                'email': ['aesokolov1975@gmail.com'],
                'email_on_failure': True,
                'email_on_retry': True,
                'retry_exponential_backoff': True,
                'retry_delay': timedelta(seconds=300),
                'retries': 3
                }

with DAG(
        dag_id="00_HW3_01_make_dataset",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:
    start_notify = BashOperator(task_id="start", bash_command="echo 'task started'")

    make_data = DockerOperator(
        task_id="Generate_data",
        image="hw03-make-data",
        command="--output_dir /data/raw/{{ ds }}",
        network_mode="bridge",
        do_xcom_push=False,
        volumes=[f"{AIRFLOW_DATA_PATH}:/data"],
    )

    finish_notify = BashOperator(task_id="finish", bash_command="echo 'task finished'")

    start_notify >> make_data >> finish_notify
