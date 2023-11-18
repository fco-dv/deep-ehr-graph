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
This project aims at demonstring deep learning methodolgies for EHR data. The use case is to predict the mortality of patients in the ICU. The dataset is from MIMIC-IV demo.

## Installation
### With pip
```bash
pip3 install -U deepehrgraph
```


## Dataset
(mimic iv demo dataset) https://physionet.org/content/mimic-iv-demo/2.2/
### Generate main dataset from compressed files
```bash
python3 -m deepehrgraph.dataset.dataset_generator
```
#### Outcomes
### Data preprocessing
### Feature selection


## Use Case

## Models


## Resources
https://mimic.mit.edu/docs/iv/modules/hosp/
Xie F, Zhou J, Lee JW, Tan M, Li SQ, Rajnthern L, Chee ML, Chakraborty B, Wong AKI, Dagan A, Ong MEH, Gao F, Liu N. Benchmarking emergency department prediction models with machine learning and public electronic health records. Scientific Data 2022 Oct; 9: 658. <https://doi.org/10.1038/s41597-022-01782-9>
https://www.sciencedirect.com/science/article/pii/S2352914823001089
https://github.com/healthylaife/MIMIC-IV-Data-Pipeline#How-to-use-the-pipeline

## Contributing
### Install dependencies
```bash
pip3 install poetry
poetry install
```
### Run tests
```bash
cd deep-ehr-graph
tox .
```
