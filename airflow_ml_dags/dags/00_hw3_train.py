from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

LOCAL_DATA_DIR = "/tmp/data/"
LOCAL_DATA_PATH = LOCAL_DATA_DIR + "{{ ds }}"

# AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"
# HOST_RAW_DATA_PATH = "/data/raw/{{ ds }}"
# HOST_PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
# HOST_SPLITTED_DATA_PATH = "/data/splitted/{{ ds }}"
# HOST_MODELS_PATH = "/data/models/{{ ds }}"

HOST_DATA_DIR = Variable.get("HOST_DATA_DIR")

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
        task_id="Train model",
        default_args=default_args,
        schedule_interval="@weekly",
        start_date=days_ago(14),
) as dag:
    wait_train_data = FileSensor(
        task_id="wait-for-train-data",
        filepath=f"{LOCAL_DATA_PATH}/data.csv",
        poke_interval=30,
    )

    wait_train_target = FileSensor(
        task_id="wait-for-train-target",
        filepath=f"{LOCAL_DATA_PATH}/target.csv",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command=f"--input_dir {HOST_RAW_DATA_PATH} --output_dir {HOST_PROCESSED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-preprocess-train-data",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command=f"--input_dir {HOST_PROCESSED_DATA_PATH} --output_dir {HOST_SPLITTED_DATA_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-split-train-data",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command=f"--input_dir {HOST_SPLITTED_DATA_PATH} --models_dir {HOST_MODELS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-train-model",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command=f"--input_dir {HOST_SPLITTED_DATA_PATH} --models_dir {HOST_MODELS_PATH}",
        network_mode="bridge",
        task_id="docker-airflow-validate-model",
        do_xcom_push=False,
        auto_remove=True,
        volumes=[f"{HOST_DATA_DIR}:/data"],
    )

    [wait_train_data, wait_train_target] >> preprocess >> split >> train >> validate
