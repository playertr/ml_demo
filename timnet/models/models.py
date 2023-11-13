import lightning as L
import torch
from torch.nn import functional as F
from torchmetrics import Accuracy
from omegaconf import DictConfig

class MNISTModel(L.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)
        self.accuracy = Accuracy("multiclass", num_classes=10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        pred = logits.argmax(dim=1)
        acc = self.accuracy(pred, y)

        # PyTorch `self.log` will be automatically captured by MLflow.
        self.log("train_loss", loss, on_epoch=True)
        self.log("acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
