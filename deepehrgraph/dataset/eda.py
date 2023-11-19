""" Exploratory Data Analysis (EDA) for the MIMIC-III dataset. """
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from deepehrgraph.dataset.dataset import EHDRDataset
from deepehrgraph.logger import get_logger

logger = get_logger(__name__)


def _heatmap(correlation_matrix: pd.DataFrame):
    """Plot heatmap of correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def _print_info(dataframe: pd.DataFrame):
    """Print info about the dataframe."""
    logger.info(dataframe.describe())
    logger.info(dataframe.dtypes)


def _display_correlation_matrix(dataframe: pd.DataFrame):
    """Display correlation matrix."""
    correlation_matrix = dataframe.select_dtypes(include=["float64", "int64"]).corr()
    _heatmap(correlation_matrix)


def _display_linear_dependency(dataframe: pd.DataFrame, feat1: str, feat2: str) -> None:
    """Display linear dependence between two features."""
    plt.figure(figsize=(10, 8))
    sns.regplot(x=feat1, y=feat2, data=dataframe)
    plt.ylim(
        0,
    )
    plt.show()


def eda(namespace: argparse.Namespace) -> None:
    """Exploratory Data Analysis (EDA) for the MIMIC-IV demo dataset."""
    logger.info(f"Input arguments: {namespace}")
    ehr_dataset = EHDRDataset(download=False)

    logger.info("Features info:")
    _print_info(ehr_dataset.features)
    _display_correlation_matrix(ehr_dataset.features)

    logger.info("Outcomes info:")
    _print_info(ehr_dataset.outcomes)

    _display_linear_dependency(ehr_dataset.features, "cci_Liver2", "cci_Liver1")
