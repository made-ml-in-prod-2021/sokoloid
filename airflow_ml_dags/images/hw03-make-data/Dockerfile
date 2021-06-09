FROM apache/airflow:2.0.0-python3.6
RUN pip install --user --no-cache-dir sklearn

COPY make_data.py /make_data.py

ENTRYPOINT ["python", "make_data.py"]
