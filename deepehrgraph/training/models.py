"""Define the model for training."""

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import Accuracy, F1Score, Precision, Recall

from deepehrgraph.logger import get_logger

logger = get_logger(__name__)


class BinaryClassificationModel(nn.Module):
    """BinaryClassificationModel network model."""

    def __init__(self, input_size, hidden_size, output_size):
        """Init."""
        super(BinaryClassificationModel, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass."""
        x = self.layer1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x


class EHROutcomeClassifier(L.LightningModule):
    """EHROutcomeClassifier network model."""

    def __init__(self, nb_input_features=59, nb_output_features=1):
        super().__init__()
        self.model = BinaryClassificationModel(
            nb_input_features, 10, nb_output_features
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        y_pred = self(x)
        loss = nn.BCELoss()(y_pred, y.unsqueeze(-1))

        self.log("train_loss", loss)
        self.log_dict(self._compute_metrics("train", y_pred, y.unsqueeze(-1)))
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        y_pred = self(x)
        loss = nn.BCELoss()(y_pred, y.unsqueeze(-1))

        self.log("val_loss", loss)
        self.log_dict(self._compute_metrics("val", y_pred, y.unsqueeze(-1)))

    def configure_optimizers(self):
        """Configure the optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def forward(self, inputs):
        """Forward pass."""
        return self.model(inputs)

    @staticmethod
    def _compute_metrics(
        stage: str, y_pred: torch.Tensor, y: torch.Tensor
    ) -> dict[str, float]:
        """Compute metrics."""
        metrics = {}
        metrics[f"{stage}_accuracy"] = Accuracy(task="binary")(y_pred, y)
        metrics[f"{stage}_precision"] = Precision(task="binary")(y_pred, y)
        metrics[f"{stage}_recall"] = Recall(task="binary")(y_pred, y)
        metrics[f"{stage}_f1"] = F1Score(task="binary")(y_pred, y)
        return metrics
