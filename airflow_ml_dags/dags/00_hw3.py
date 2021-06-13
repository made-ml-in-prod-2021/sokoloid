from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago
VOLUME = '/tmp/data/:/data'

with DAG(
        dag_id="00_HW3_01_make_dataset",
        schedule_interval="@daily",
        start_date=days_ago(5),
) as dag:


    start_notify = BashOperator(task_id="start", bash_command="echo 'task started'")

    make_data = DockerOperator(
        task_id = "Generate_data",
        image = "hw03-make-data",
        command = "python make_data.py /data/raw/{{ ds }}",
        network_mode = "bridge",
        do_xcom_push = False,
        volumes = [VOLUME],
    )

    finish_notify = BashOperator(task_id="finish", bash_command="echo 'task finished'")

    start_notify >> make_data >> finish_notify

