""" Exploratory Data Analysis (EDA) for the MIMIC-III dataset. """
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from deepehrgraph.dataset.dataset import EHDRDataset
from deepehrgraph.logger import get_logger
from deepehrgraph.training.enums import OutcomeType

logger = get_logger(__name__)


def _heatmap(correlation_matrix: pd.DataFrame):
    """Plot heatmap of correlation matrix."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()


def _print_info(dataframe: pd.DataFrame):
    """Print info about the dataframe."""
    logger.info(dataframe.info())


def _display_correlation_matrix(dataframe: pd.DataFrame):
    """Display correlation matrix."""
    correlation_matrix = dataframe.select_dtypes(include=["float64", "int64"]).corr()
    _heatmap(correlation_matrix)


def _get_redundant_pairs(dataframe):
    """Get diagonal and lower triangular pairs of correlation matrix"""
    pairs_to_drop = set()
    cols = dataframe.columns
    for i in range(0, dataframe.shape[1]):
        for j in range(0, i + 1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def _get_top_abs_correlations(dataframe, n=5):
    """Get top absolute correlations"""
    au_corr = dataframe.corr().abs().unstack()
    labels_to_drop = _get_redundant_pairs(dataframe)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def display_outcomes_class_repartition(outcomes: pd.DataFrame):
    """Display outcomes class repartition."""
    fig, axes = plt.subplots(1, len(list(OutcomeType)), figsize=(15, 4))
    for i, outcome in enumerate(list(OutcomeType)):
        axes[i].hist(outcomes.astype(int)[outcome.value], bins=[0, 0.5, 1.5])
        axes[i].set_title(f"{outcome.value}")
    plt.tight_layout()
    plt.show()


def eda(namespace: argparse.Namespace) -> None:
    """Exploratory Data Analysis (EDA) for the MIMIC-IV demo dataset."""
    logger.info(f"Input arguments: {namespace}")
    ehr_dataset = EHDRDataset(download=False)

    logger.info("Features info:")
    _print_info(ehr_dataset.features)
    logger.info("Outcomes info:")
    _print_info(ehr_dataset.outcomes)

    logger.info("Features Missing values count:")
    logger.info(ehr_dataset.features.isnull().sum().sum())
    logger.info("Outcomes Missing values count:")
    logger.info(ehr_dataset.outcomes.isnull().sum().sum())

    logger.info("Features Correlation matrix:")
    _display_correlation_matrix(ehr_dataset.features)
    logger.info("Compute Top of Correlation matrix:")

    n = 5
    top_abs_correlated = _get_top_abs_correlations(ehr_dataset.features, 25)
    logger.info(f"Top {n} Correlated features : \n {top_abs_correlated}")

    logger.info("Display outcomes class repartition:")
    display_outcomes_class_repartition(ehr_dataset.outcomes)
