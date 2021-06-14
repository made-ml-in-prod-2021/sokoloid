from datetime import timedelta, datetime
import os

from airflow import DAG
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
with DAG(
        dag_id="00_HW3_02_train_validate",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(14),
) as dag:
    wait_train_data = FileSensor(
        task_id="wait-for-train-data",
        filepath="data/raw/{{ ds }}/data.csv",
        poke_interval=30,
    )

    wait_train_target = FileSensor(
        task_id="wait-for-train-target",
        filepath="data/raw/{{ ds }}/target.csv",
        poke_interval=30,
    )

    convert_data = DockerOperator(
        task_id="convert-data",
        image="hw03-convert-data",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/processed/{{ds}}",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{AIRFLOW_DATA_PATH}:/data"],
    )

    split_data = DockerOperator(
        task_id="split-data",
        image="hw03-split-data",
        command="--input_dir /data/processed/{{ ds }} --output_dir /data/processed/{{ds}}",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{AIRFLOW_DATA_PATH}:/data"],
    )

    train = DockerOperator(
        task_id="train",
        image="hw03-train",
        command="--input_dir /data/processed/{{ ds }} --output_dir /data/model/{{ds}} ",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{AIRFLOW_DATA_PATH}:/data"],
    )

    validate = DockerOperator(
        task_id="validate",
        image="hw03-validate",
        command="--input_dir /data/processed/{{ ds }} --model_dir /data/model/{{ds}}, ",
        network_mode="bridge",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{AIRFLOW_DATA_PATH}:/data"],
    )

    [wait_train_data, wait_train_target] >> convert_data >> split_data >> train >> validate
