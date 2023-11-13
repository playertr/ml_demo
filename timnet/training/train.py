import lightning as L
from omegaconf import DictConfig

import mlflow.pytorch
from mlflow import MlflowClient

from timnet.dataset.dataset import get_dataloader
from timnet.models.models import MNISTModel

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def train(cfg: DictConfig) -> None:

    train_loader = get_dataloader(cfg.dataloader)
    mnist_model = MNISTModel(cfg.model)

    # Initialize a trainer.
    trainer = L.Trainer(max_epochs=3)

    # Auto log all MLflow entities
    mlflow.pytorch.autolog()

    # Train the model.
    with mlflow.start_run() as run:
        trainer.fit(mnist_model, train_loader)

    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))