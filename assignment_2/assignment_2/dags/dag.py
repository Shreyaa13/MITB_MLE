from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
) as dag:


    # --- model inference ---
    model_inference_start = DummyOperator(task_id="model_inference_start")

    model_xgb_inference = BashOperator(
        task_id='model_xgb_inference',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_inference.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_2024_09_01.pkl"'
        ),
    )

    model_inference_completed = DummyOperator(task_id="model_inference_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_start >> model_xgb_inference >> model_inference_completed


    # --- model monitoring ---
    model_monitor_start = DummyOperator(task_id="model_monitor_start")

    model_xgb_monitor = BashOperator(
        task_id='model_xgb_monitor',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_monitoring.py '
            '--snapshotdate "{{ ds }}" '
            '--modelname "credit_model_2024_09_01.pkl"'
        ),
    )
    
    model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    model_inference_completed >> model_monitor_start
    model_monitor_start >> model_xgb_monitor >> model_monitor_completed

    # --- model auto training ---

    model_automl_start = DummyOperator(task_id="model_automl_start")
    
    model_xgb_automl = BashOperator(
        task_id='model_xgb_automl',
        bash_command=(
            'cd /opt/airflow/scripts && '
            'python3 model_auto_train.py '
            '--snapshotdate "{{ ds }}"'
        ),
    )
    
    model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    model_automl_start >> model_xgb_automl >> model_automl_completed
