"""Hardcoded employee dataset for PTC demo."""

from typing import Any, Optional, Set

import pandas as pd
from mloda.provider import BaseInputData, DataCreator, FeatureGroup, FeatureSet

EMPLOYEE_FEATURES: Set[str] = {"employee_id", "department", "salary", "years_experience", "performance_score"}


class EmployeeDataFeatures(FeatureGroup):
    """Hardcoded employee dataset for PTC demo. 10 employees across 3 departments."""

    @classmethod
    def input_data(cls) -> Optional[BaseInputData]:
        return DataCreator(EMPLOYEE_FEATURES)

    @classmethod
    def feature_names_supported(cls) -> Set[str]:
        return EMPLOYEE_FEATURES

    @classmethod
    def calculate_feature(cls, data: Any, features: FeatureSet) -> Any:
        return pd.DataFrame(
            {
                "employee_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "department": [
                    "Engineering",
                    "Engineering",
                    "Engineering",
                    "Sales",
                    "Sales",
                    "Sales",
                    "HR",
                    "HR",
                    "HR",
                    "Engineering",
                ],
                "salary": [95000, 88000, 102000, 72000, 68000, 75000, 61000, 58000, 64000, 110000],
                "years_experience": [5, 3, 8, 4, 2, 6, 7, 3, 5, 10],
                "performance_score": [87, 72, 95, 81, 65, 78, 90, 55, 83, 98],
            }
        )
