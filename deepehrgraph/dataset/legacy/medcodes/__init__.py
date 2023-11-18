"""
MedCodes
========
MedCodes is a tool for interpreting medical text.
"""
from .diagnoses.comorbidities import charlson, comorbidities, elixhauser
from .drugs.classification import atc_classification
