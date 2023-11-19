"""Enum class for outcome types."""
from enum import Enum


class OutcomeType(Enum):
    """Enum class for outcome types."""

    INHOSPITAL_MORTALITY = "outcome_inhospital_mortality"
    ICU_TRANSFER_12H = "outcome_icu_transfer_12h"
    HOSPITALIZATION = "outcome_hospitalization"
    CRITICAL = "outcome_critical"
    ED_REVISIT_3D = "outcome_ed_revisit_3d"
