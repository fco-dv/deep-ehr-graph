from __future__ import annotations

import hashlib
import os
import tempfile

from deepehrgraph import __version__
from deepehrgraph.dataset.dataset import (
    create_master_dataset,
    download_mimiciv_compressed_dataset,
)
from deepehrgraph.logger import get_logger


def test_version():
    assert __version__ == "0.3.0"


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


def test_get_logger():
    """
    Test get_logger function.
    """
    logger = get_logger()
    assert logger is not None
    assert logger.name == "deepehrgraph.logger"
