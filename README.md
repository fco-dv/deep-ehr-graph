# deep-ehr-graph


[![PyPI Version][pypi-image]][pypi-url]
[![Build Status][build-image]][build-url]
[![Code Coverage][coverage-image]][coverage-url]


<!-- Badges: -->

[pypi-image]: https://img.shields.io/pypi/v/deepehrgraph
[pypi-url]: https://pypi.org/project/deepehrgraph/
[build-image]: https://github.com/fco-dv/deep-ehr-graph/actions/workflows/build.yaml/badge.svg
[build-url]: https://github.com/fco-dv/deep-ehr-graph/actions/workflows/build.yaml
[coverage-image]: https://codecov.io/gh/fco-dv/deep-ehr-graph/branch/main/graph/badge.svg
[coverage-url]: https://codecov.io/gh/fco-dv/deep-ehr-graph/

## Description
This project aims at demonstring deep learning methodologies for EHR data. 
The use case is to predict different outcomes for patients in the ICU. The dataset is from (MIMIC-IV demo) 

## Installation
### With pip
```bash
pip3 install -U deepehrgraph
```


## Dataset
(mimic iv demo dataset)[https://physionet.org/content/mimic-iv-demo/2.2/]
### Generate main dataset from compressed files
```bash
python3 -m deepehrgraph.dataset.dataset_generator
```
This step will download the archive files from physionet and generate the master dataset in the `data` folder.
CCI and ECI indexes are calculated and added to the dataset.

### Features 
In the context of medical studies, CCI (Charlson Comorbidity Index) and ECI (Elixhauser Comorbidity Index)
are tools used to assess the burden of comorbidities in individuals.
Comorbidities refer to the presence of additional health conditions in a patient alongside the primary
condition under investigation. Both CCI and ECI are designed to quantify and summarize the impact of comorbidities on patient health.

Charlson Comorbidity Index (CCI):

Purpose: Developed by Dr. Mary Charlson, the CCI is a widely used tool to predict the 10-year mortality for patients with multiple comorbidities. It assigns weights to various comorbid conditions based on their impact on mortality.
Calculation: Each comorbid condition is assigned a score, and the total CCI score is the sum of these individual scores. The higher the CCI score, the greater the burden of comorbidities.
Conditions: The CCI includes conditions such as myocardial infarction, heart failure, dementia, diabetes, liver disease, and others.

Elixhauser Comorbidity Index (ECI):

Purpose: The ECI, developed by Dr. Claudia Elixhauser, is another comorbidity index used to assess the impact of comorbid conditions on healthcare outcomes. It is often employed in administrative databases and research studies.
Calculation: Similar to the CCI, the ECI assigns weights to comorbid conditions. However, the ECI covers a broader range of conditions and is often used for risk adjustment in research studies.
Conditions: The ECI includes a comprehensive list of conditions such as hypertension, obesity, renal failure, coagulopathy, and others.


Selected features: 
    
    ['gender', 'age', 'n_ed_30d', 'n_ed_90d', 'n_ed_365d', 'n_hosp_30d',
       'n_hosp_90d', 'n_hosp_365d', 'n_icu_30d', 'n_icu_90d', 'n_icu_365d',
       'cci_MI', 'cci_CHF', 'cci_PVD', 'cci_Stroke', 'cci_Dementia',
       'cci_Pulmonary', 'cci_Rheumatic', 'cci_PUD', 'cci_Liver1', 'cci_DM1',
       'cci_DM2', 'cci_Paralysis', 'cci_Renal', 'cci_Cancer1', 'cci_Liver2',
       'cci_Cancer2', 'cci_HIV', 'eci_CHF', 'eci_Arrhythmia', 'eci_Valvular',
       'eci_PHTN', 'eci_PVD', 'eci_HTN1', 'eci_HTN2', 'eci_Paralysis',
       'eci_NeuroOther', 'eci_Pulmonary', 'eci_DM1', 'eci_DM2',
       'eci_Hypothyroid', 'eci_Renal', 'eci_Liver', 'eci_PUD', 'eci_HIV',
       'eci_Lymphoma', 'eci_Tumor2', 'eci_Tumor1', 'eci_Rheumatic',
       'eci_Coagulopathy', 'eci_Obesity', 'eci_WeightLoss', 'eci_FluidsLytes',
       'eci_BloodLoss', 'eci_Anemia', 'eci_Alcohol', 'eci_Drugs',
       'eci_Psychoses', 'eci_Depression']


### Outcomes

### Data preprocessing

### Feature selection


## Use Case

## Models


## Resources
https://mimic.mit.edu/docs/iv/modules/hosp/
Xie F, Zhou J, Lee JW, Tan M, Li SQ, Rajnthern L, Chee ML, Chakraborty B, Wong AKI, Dagan A, Ong MEH, Gao F, Liu N. Benchmarking emergency department prediction models with machine learning and public electronic health records. Scientific Data 2022 Oct; 9: 658. <https://doi.org/10.1038/s41597-022-01782-9>
https://www.sciencedirect.com/science/article/pii/S2352914823001089
https://github.com/healthylaife/MIMIC-IV-Data-Pipeline#How-to-use-the-pipeline

###ML
https://scikit-learn.org/stable/common_pitfalls.html

## Contributing
### Install dependencies
```bash
pip3 install poetry
poetry install
```
### Pre-commit hooks
```bash
poetry run pre-commit install
```
Run pre-commit hooks on all files
```bash
poetry run pre-commit run --all-files
```
### Run tests
Tox is using pre-commit hooks to run tests and linting.
```bash
cd deep-ehr-graph
tox .
```
