""" Generate mimic iv demo dataset as csv file. """
import argparse
import io
import os  # pylint: disable=E1101
import zipfile
from dataclasses import dataclass, field

import pandas as pd  # pylint: disable=E0401
import requests
from sklearn.preprocessing import LabelEncoder

from deepehrgraph.dataset.legacy.helpers import (
    add_age,
    add_ed_los,
    add_inhospital_mortality,
    add_outcome_icu_transfer,
    fill_na_ethnicity,
    generate_future_ed_visits,
    generate_numeric_timedelta,
    generate_past_admissions,
    generate_past_ed_visits,
    generate_past_icu_visits,
    merge_icustays_admissions_on_subject,
    merge_icustays_patients_on_subject,
    read_admissions_table,
    read_diagnoses_table,
    read_icustays_table,
    read_patients_table,
)
from deepehrgraph.dataset.legacy.medcode_utils import commorbidity
from deepehrgraph.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EHDRDataset:
    """Class for EHDR dataset."""

    features: pd.DataFrame = field(default="false", init=False, repr=False)
    outcomes: pd.DataFrame = field(default="false", init=False, repr=False)
    filename: str = field(
        default="mimic_iv_demo_master_dataset.csv", init=True, repr=True
    )
    dir_name: str = field(default="data", init=True, repr=True)
    download: bool = field(default=False, init=True, repr=True)
    encode_categorical_features: bool = field(default=True, init=True, repr=True)

    def __hash__(self) -> int:
        return hash(self.filename)

    def __post_init__(self):
        """Initialize EHDR dataset."""
        if self.download:
            data = create_master_dataset(
                download_mimiciv_compressed_dataset(self.dir_name),
                self.dir_name,
                self.filename,
            )
        else:
            data = pd.read_csv(os.path.join(self.dir_name, self.filename))

        self.features = self._features(data)
        self.outcomes = self._outcomes(data)

        if self.encode_categorical_features:
            label_encoder = LabelEncoder()
            categorical_features = self.features.select_dtypes(
                include=["object", "category"]
            ).columns
            self.features[categorical_features] = self.features[
                categorical_features
            ].apply(lambda x: label_encoder.fit_transform(x))

    @staticmethod
    def _outcomes(data) -> pd.DataFrame:
        """Get outcomes."""
        labels_prefix = ["outcome_"]
        labels_cols = [
            col
            for col in data.columns
            if any(col.startswith(prefix) for prefix in labels_prefix)
        ]
        return data[labels_cols]

    @staticmethod
    def _features(data) -> pd.DataFrame:
        """Get features."""
        features_prefix = ["cci_", "eci_", "n_", "age", "gender"]
        features_cols = [
            col
            for col in data.columns
            if any(col.startswith(prefix) for prefix in features_prefix)
        ]
        return data[features_cols]


def download_mimiciv_compressed_dataset(dir_name: str = "data") -> str:
    """
    Download mimic iv demo dataset from physionet.org

    :param str dir_name: directory name to store the dataset
    :return: str path to the mimic iv demo dataset
    :rtype str
    """
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    url = (
        "https://physionet.org/static/published-projects/mimic-iv-demo/"
        "mimic-iv-clinical-database-demo-2.2.zip"
    )

    # Send a HTTP request to the URL of the file, stream = True prevents the
    # content from being loaded into the memory
    response = requests.get(url, stream=True)

    # Check if the download was successful
    logger.info("Downloading mimic iv demo dataset...")
    if response.status_code == 200:
        # Get the file from the HTTP response
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            zip_file.extractall(dir_name)
    else:
        raise response.raise_for_status()

    return os.path.join(dir_name, "mimic-iv-clinical-database-demo-2.2")


def create_master_dataset(
    mimic_iv_path: str,
    output_path: str = "data",
    filename: str = "mimic_iv_demo_master_dataset.csv",
):
    """
    Generate master mimic iv demo dataset as csv file.

    :param str mimic_iv_path: path to the mimic iv demo dataset
    :param str output_path: path to the output directory
    :return: None
    """
    logger.info("Creating mimic iv demo master dataset")

    icu_transfer_timerange = 12
    next_ed_visit_timerange = 3

    mimic_iv_hosp_path = os.path.join(mimic_iv_path, "hosp")
    mimic_iv_icu_path = os.path.join(mimic_iv_path, "icu")

    icu_filename_dict = {
        "chartevents": "chartevents.csv.gz",
        "datetimeevents": "datetimeevents.csv.gz",
        "d_items": "d_items.csv.gz",
        "icustays": "icustays.csv.gz",
        "inputevents": "inputevents.csv.gz",
        "outputevents": "outputevents.csv.gz",
        "procedureevents": "procedureevents.csv.gz",
    }
    core_filename_dict = {
        "patients": "patients.csv.gz",
        "admissions": "admissions.csv.gz",
        "transfers": "transfers.csv.gz",
    }
    hosp_filename_dict = {
        "d_hcpcs": "d_hcpcs.csv.gz",
        "d_icd_diagnoses": "d_icd_diagnoses.csv.gz",
        "d_labitems": "d_labitems.csv.gz",
        "emar": "emar.csv.gz",
        "hcpcsevents": "hcpcsevents.csv.gz",
        "microbiologyevents": "microbiologyevents.csv.gz",
        "poe": "poe.csv.gz",
        "prescriptions": "prescriptions.csv.gz",
        "services": "services.csv.gz",
        "diagnoses_icd": "diagnoses_icd.csv.gz",
        "d_icd_procedures": "d_icd_procedures.csv.gz",
        "drgcodes": "drgcodes.csv.gz",
        "emar_detail": "emar_detail.csv.gz",
        "labevents": "labevents.csv.gz",
        "pharmacy": "pharmacy.csv.gz",
        "poe_detail": "poe_detail.csv.gz",
        "procedures_icd": "procedures_icd.csv.gz",
    }

    df_patients = read_patients_table(
        os.path.join(mimic_iv_hosp_path, core_filename_dict["patients"])
    )
    df_admissions = read_admissions_table(
        os.path.join(mimic_iv_hosp_path, core_filename_dict["admissions"])
    )
    df_icustays = read_icustays_table(
        os.path.join(mimic_iv_icu_path, icu_filename_dict["icustays"])
    )

    df_diagnoses = read_diagnoses_table(
        os.path.join(mimic_iv_hosp_path, hosp_filename_dict["diagnoses_icd"])
    )

    # Merging patients -> merging admissions -> master
    df_master = merge_icustays_patients_on_subject(df_icustays, df_patients)
    df_master = merge_icustays_admissions_on_subject(df_master, df_admissions)

    # Adding age, mortality and ICU transfer outcome
    df_master = add_age(df_master)
    df_master = add_inhospital_mortality(df_master)
    df_master = add_ed_los(df_master)
    df_master = add_outcome_icu_transfer(df_master, df_icustays, icu_transfer_timerange)
    df_master["outcome_hospitalization"] = ~pd.isnull(df_master["hadm_id"])
    df_master["outcome_critical"] = (
        df_master["outcome_inhospital_mortality"]
        | df_master[
            "".join(["outcome_icu_transfer_", str(icu_transfer_timerange), "h"])
        ]
    )

    # Sort Master table for further process
    df_master = df_master.sort_values(["subject_id", "intime"]).reset_index()

    # Filling subjects NA ethnicity, takes ~17s
    df_master = fill_na_ethnicity(df_master)

    # Generate past ED visits
    df_master = generate_past_ed_visits(df_master, timerange=30)
    df_master = generate_past_ed_visits(df_master, timerange=90)
    df_master = generate_past_ed_visits(df_master, timerange=365)

    # Oucome:  future ED revisit variables
    df_master = generate_future_ed_visits(df_master, next_ed_visit_timerange)

    # Generate past admissions
    df_master = generate_past_admissions(df_master, df_admissions, timerange=30)
    df_master = generate_past_admissions(df_master, df_admissions, timerange=90)
    df_master = generate_past_admissions(df_master, df_admissions, timerange=365)

    # Generate past icu visits
    df_master = generate_past_icu_visits(df_master, df_icustays, timerange=30)
    df_master = generate_past_icu_visits(df_master, df_icustays, timerange=90)
    df_master = generate_past_icu_visits(df_master, df_icustays, timerange=365)

    # Generate numeric timedelta variables
    df_master = generate_numeric_timedelta(df_master)

    # This function takes about 10 min
    df_master = commorbidity(df_master, df_diagnoses, df_admissions, timerange=356 * 5)

    # Reset dataframe index
    df_master.reset_index(drop=True, inplace=True)

    # Store master_dataset as a single csv
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df_master.to_csv(os.path.join(output_path, filename), index=False)

    return df_master


def generate_dataset(namespace: argparse.Namespace) -> None:
    """Generate mimic iv demo dataset as csv file."""
    logger.info(f"Input arguments : {namespace}")
    EHDRDataset(
        dir_name="data", filename="mimic_iv_demo_master_dataset.csv", download=True
    )
