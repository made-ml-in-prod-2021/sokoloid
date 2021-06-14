from datetime import timedelta, datetime
import os

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

AIRFLOW_DATA_PATH = "/home/sokolov/prod/sokoloid/airflow_ml_dags/data/"

default_args = {'owner': 'airflow',
                'start_date': datetime(2018, 1, 30),
                'email': ['aesokolov1975@gmail.com'],
                'email_on_failure': True,
                'email_on_retry': True,
                'retry_exponential_backoff': True,
                'retry_delay': timedelta(seconds=300),
                'retries': 3
                }

try:
    model_path = Variable.get("MODEL_PATH")
except KeyError:
    model_path = "data/model/{{ ds }}/"

with DAG(
        dag_id="00_HW3_03_predict",
        default_args=default_args,
        schedule_interval="@daily",
        start_date=days_ago(14),
) as dag:
    wait_data = FileSensor(
        task_id="wait-for-train-data",
        filepath="data/raw/{{ ds }}/data.csv",
        poke_interval=30,
    )

    predict = DockerOperator(
        task_id="predict",
        image="hw03-predict",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/predictions/{{ds}} --model_dir " + model_path,
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{AIRFLOW_DATA_PATH}:/data"],
    )

    wait_data >> predict
