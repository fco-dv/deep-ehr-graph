"""Feature selection for the dataset."""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deepehrgraph.dataset.dataset import EHDRDataset
from deepehrgraph.dataset.eda import _display_correlation_matrix
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


def reduce_colinear_features(
    features, desired_explained_variance=0.95, display_plot=True
):
    """
    Reduce colinear features using PCA.

    Parameters:
    - features (pd.DataFrame): Input DataFrame containing
    features and target variable.
    - desired_explained_variance (float): Desired cumulative
      explained variance threshold (default is 0.95).
    - display_plot (bool): Whether to display the cumulative
      explained variance plot (default is True).

    Returns:
    - X_pca_retained (pd.DataFrame): Transformed DataFrame with retained components.
    """

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    # Perform PCA
    pca = PCA()
    pca.fit_transform(X_scaled)

    # Find the number of components that meet or exceed the desired explained variance
    cumulative_explained_variance = pca.explained_variance_ratio_.cumsum()
    num_components_to_retain = sum(
        cumulative_explained_variance >= desired_explained_variance
    )

    # Retain components
    pca = PCA(n_components=num_components_to_retain)
    X_pca_retained = pca.fit_transform(X_scaled)

    # Display the result
    logger.info(
        f"Number of components to retain for \
        {desired_explained_variance * 100}% explained \
        variance: {num_components_to_retain}"
    )

    # Plotting explained variance ratio if display_plot is True
    if display_plot:
        plt.plot(
            range(1, len(cumulative_explained_variance) + 1),
            cumulative_explained_variance,
            marker="o",
            linestyle="--",
        )
        plt.title("Cumulative Explained Variance")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.show()

    return pd.DataFrame(
        X_pca_retained, columns=[f"PC{i+1}" for i in range(num_components_to_retain)]
    )


def features_selection(namespace: argparse.Namespace) -> None:
    """Exploratory Data Analysis (EDA) for the MIMIC-IV demo dataset."""

    logger.info(f"Features selection phase : {namespace}")

    logger.info("Load EHRDataset:")
    ehr_dataset = EHDRDataset(download=False)

    reduced_features = reduce_colinear_features(
        features=ehr_dataset.features,
        desired_explained_variance=0.95,
        display_plot=True,
    )

    logger.info("Display correlation matrix on selected features:")
    _display_correlation_matrix(reduced_features)
