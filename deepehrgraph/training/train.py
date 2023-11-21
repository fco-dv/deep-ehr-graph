""" This file contains the training loop for the model."""

import argparse
import os

import lightning as L
from imblearn.over_sampling import RandomOverSampler
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from deepehrgraph.dataset.dataset import EHDRDataset
from deepehrgraph.dataset.features_selection import reduce_colinear_features
from deepehrgraph.logger import get_logger
from deepehrgraph.training.dataset import TorchEHRDataset
from deepehrgraph.training.models import EHROutcomeClassifier


def _save_for_production(
    trainer: L.Trainer, production_dir: str, outcome_type: str
) -> None:
    """Save the model for production."""
    best_model = EHROutcomeClassifier.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )
    scripted_model = best_model.to_torchscript()
    if not os.path.exists(production_dir):
        os.makedirs(production_dir)
    scripted_model.save(
        os.path.join(production_dir, f"best_model_scripted_{outcome_type}.pt")
    )


def train(namespace: argparse.Namespace) -> None:
    """Train a model on a dataset."""
    logger = get_logger(__name__)

    ehr_master_dataset = EHDRDataset(download=False)
    outcome = namespace.outcome
    max_epochs = namespace.max_epochs
    oversample_outcome = namespace.oversample_outcome

    logger.info(f"Training model for outcome {outcome.value} and {max_epochs} epochs")

    logger.info("Features selection step")
    features = reduce_colinear_features(
        features=ehr_master_dataset.features,
        desired_explained_variance=0.95,
        display_plot=False,
    )

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        ehr_master_dataset.outcomes[outcome.value],
        test_size=0.3,
        random_state=42,
    )

    if oversample_outcome:
        # Oversampling the minority class using RandomOverSampler
        ros = RandomOverSampler(random_state=42)
        X_train, y_train = ros.fit_resample(X_train, y_train)

        logger.info(
            "Check that target outcome is balanced after resampling  training dataset"
        )
        logger.info(y_train.describe())

    # Torch Dataset
    ehr_train_dataset = TorchEHRDataset(X_train, y_train)
    ehr_valid_dataset = TorchEHRDataset(X_test, y_test)

    ehr_train_dataloader = DataLoader(ehr_train_dataset)
    ehr_valid_dataloader = DataLoader(ehr_valid_dataset)

    model = EHROutcomeClassifier()

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="models/checkpoints/",
        filename="ehr-model-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        mode="min",
    )

    tb_logger = TensorBoardLogger("tb_logs", name="ehr_outcome_classifier")

    trainer = L.Trainer(
        max_epochs=max_epochs,
        enable_progress_bar=True,
        logger=tb_logger,
        log_every_n_steps=5,
        callbacks=[ModelSummary(), checkpoint_callback],
    )

    trainer.fit(
        model=model,
        train_dataloaders=ehr_train_dataloader,
        val_dataloaders=ehr_valid_dataloader,
    )

    logger.info(f"Saving torschscript model weights for outcome {outcome.value}")

    _save_for_production(
        trainer=trainer, production_dir="models/production", outcome_type=outcome.value
    )
