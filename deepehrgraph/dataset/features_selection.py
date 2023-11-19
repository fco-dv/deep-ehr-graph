"""Feature selection for the dataset."""
import argparse

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split

from deepehrgraph.dataset.dataset import EHDRDataset
from deepehrgraph.logger import get_logger

logger = get_logger(__name__)


def select_kbest_features(
    dataset: EHDRDataset, k: int, label_type: str
) -> pd.DataFrame:
    """Select k best features."""

    # splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.features, dataset.outcomes[label_type], test_size=0.2, random_state=42
    )

    # Init Kbest
    k_best = SelectKBest(score_func=chi2, k=k)

    k_best.fit_transform(X_train, y_train)

    # Transform the testing data using the selected features
    k_best.transform(X_test)

    # Display the selected features
    selected_feature_names = dataset.features.columns[k_best.get_support(indices=True)]
    logger.info(f"Selected features: {selected_feature_names}")
    return selected_feature_names


def features_selection(namespace: argparse.Namespace) -> None:
    """Exploratory Data Analysis (EDA) for the MIMIC-IV demo dataset."""
    ehr_dataset = EHDRDataset(download=False)

    select_kbest_features(ehr_dataset, 10, "outcome_inhospital_mortality")
