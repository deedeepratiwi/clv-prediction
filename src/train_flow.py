import subprocess

from prefect import flow, task


@task(retries=2, retry_delay_seconds=30)
def train():
    subprocess.run(["python", "src/train.py"], check=True)


@flow(name="clv_training_pipeline")
def training_pipeline():
    train()


if __name__ == "__main__":
    training_pipeline()
