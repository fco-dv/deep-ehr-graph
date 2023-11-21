from __future__ import annotations

import hashlib
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pytest
import torch

from deepehrgraph import __version__
from deepehrgraph.dataset.dataset import (
    EHDRDataset,
    create_master_dataset,
    download_mimiciv_compressed_dataset,
)
from deepehrgraph.dataset.eda import _display_correlation_matrix, _heatmap, _print_info
from deepehrgraph.logger import get_logger
from deepehrgraph.training.enums import OutcomeType
from deepehrgraph.training.models import EHROutcomeClassifier


def test_version():
    assert __version__ == "0.3.4"


def test_download_mimiciv_compressed_dataset():
    """
    Test download_mimiciv_compressed_dataset function.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = download_mimiciv_compressed_dataset(tmp_dir)
        print(path)
        assert os.path.exists(path)


def test_master_dataset():
    """
    Test master_dataset function.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        mimic_iv_path = download_mimiciv_compressed_dataset(tmp_dir)
        create_master_dataset(mimic_iv_path, tmp_dir)
        assert os.path.exists(os.path.join(tmp_dir, "mimic_iv_demo_master_dataset.csv"))

        # check hash of master.csv
        expected_hash = b"~c\t$\xdaM\x85Z\xaaP\x02\xf9\xde\xbb\x14\xf2"

        with open(os.path.join(tmp_dir, "mimic_iv_demo_master_dataset.csv"), "rb") as f:
            computed_hash = hashlib.file_digest(f, "md5")

        assert expected_hash == computed_hash.digest()


def test_no_missing_valuesin_dataset():
    """
    Test no missing values in dataset.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = EHDRDataset(
            dir_name=tmp_dir, download=True, filename="test_ehrdataset.csv"
        )
        assert dataset.features.isnull().sum().sum() == 0
        assert dataset.outcomes.isnull().sum().sum() == 0


def test_ehrdataset_class():
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = EHDRDataset(
            dir_name=tmp_dir, download=True, filename="test_ehrdataset.csv"
        )
        assert dataset.features is not None
        assert dataset.outcomes is not None
        assert dataset.filename == "test_ehrdataset.csv"
        assert dataset.dir_name == tmp_dir

        # check hash of master.csv
        expected_hash = b"~c\t$\xdaM\x85Z\xaaP\x02\xf9\xde\xbb\x14\xf2"
        with open(os.path.join(tmp_dir, dataset.filename), "rb") as f:
            computed_hash = hashlib.file_digest(f, "md5")

        assert expected_hash == computed_hash.digest()

        categorical_features = dataset.features.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()
        assert categorical_features == []


def test_get_logger():
    """
    Test get_logger function.
    """
    logger = get_logger()
    assert logger is not None
    assert logger.name == "deepehrgraph.logger"


def test_outcome_enum_values():
    """
    Test OutcomeType.
    """
    assert OutcomeType.INHOSPITAL_MORTALITY.value, "outcome_inhospital_mortality"
    assert OutcomeType.ICU_TRANSFER_12H.value, "outcome_icu_transfer_12h"
    assert OutcomeType.HOSPITALIZATION.value, "outcome_hospitalization"
    assert OutcomeType.CRITICAL.value, "outcome_critical"
    assert OutcomeType.ED_REVISIT_3D.value, "outcome_ed_revisit_3d"


class TestEDA(unittest.TestCase):
    @patch("deepehrgraph.dataset.eda.plt.show")
    @patch("deepehrgraph.dataset.eda.sns.heatmap")
    def test_heatmap(self, mock_heatmap, mock_show):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        _heatmap(df)
        mock_heatmap.assert_called_once()
        mock_show.assert_called_once()

    @patch("deepehrgraph.dataset.eda.logger.info")
    def test_print_info(self, mock_info):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        _print_info(df)
        mock_info.assert_called_once()

    @patch("deepehrgraph.dataset.eda._heatmap")
    def test_display_correlation_matrix(self, mock_heatmap):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        _display_correlation_matrix(df)
        mock_heatmap.assert_called_once()


class TestEHROutcomeClassifier(unittest.TestCase):
    def setUp(self):
        self.model = EHROutcomeClassifier()

    def test_training_step(self):
        x = torch.randn(1, 37).float()
        y = torch.tensor([1]).float()
        loss = self.model.training_step((x, y), 0)
        self.assertIsNotNone(loss)

    def test_validation_step(self):
        x = torch.randn(1, 37).float()
        y = torch.tensor([1]).float()
        try:
            self.model.validation_step((x, y), 0)
        except Exception:
            pytest.fail("Validation step failed")

    def test_configure_optimizers(self):
        optimizer = self.model.configure_optimizers()
        self.assertIsNotNone(optimizer)
