import subprocess
import sys
from pathlib import Path

from prefect import flow, task

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@task(retries=2, retry_delay_seconds=30, log_prints=True)
def train():
    subprocess.run([sys.executable, "src/train.py"], check=True, cwd=PROJECT_ROOT)


@flow(name="clv_training_pipeline")
def training_pipeline():
    train()


if __name__ == "__main__":
    training_pipeline()
