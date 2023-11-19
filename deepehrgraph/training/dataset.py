"""Pytorch dataset for the EHR dataset."""
import pandas as pd
import torch
from torch.utils.data import Dataset


class TorchEHRDataset(Dataset):
    """Pytorch dataset for the EHR dataset."""

    def __init__(self, features: pd.DataFrame, outcomes: pd.DataFrame):
        self.features = features
        self.outcome = outcomes

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        features = torch.tensor(self.features.values[idx]).float()
        outcome = torch.tensor(self.outcome.values[idx]).float()

        return features, outcome
